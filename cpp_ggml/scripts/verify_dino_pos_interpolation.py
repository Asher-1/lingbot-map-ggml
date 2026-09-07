#!/usr/bin/env python3
"""Verify the native DINO positional-table interpolation against PyTorch.

The exported GGUF stores a fixed 37x37 DINOv2 positional table. The native
graph resamples it on the host with a faithful reimplementation of PyTorch's
antialias bicubic resampler (``upsample_gen2d_aa_out_frame`` with
``BicubicFilterFunctor``), which is what ``interpolate_pos_encoding``
executes for ``mode="bicubic", antialias=True, align_corners=False``.

This script has three modes:

1. Algorithm check: a numpy replica of the C++ resampler is compared against
   ``torch.nn.functional.interpolate`` on random tables.
2. Table check: the numpy replica is compared against the real GGUF table.
3. Native check: an optional C++ ``dino.pos`` debug dump (LINGBOT_DUMP_STAGES)
   is compared against the PyTorch reference.

Examples:
    python3 cpp_ggml/scripts/verify_dino_pos_interpolation.py \
        --gguf cpp_ggml/models/gguf/lingbot-map-q8.gguf --height 392 --width 392
    python3 cpp_ggml/scripts/verify_dino_pos_interpolation.py \
        --gguf ... --height 392 --width 392 --cpp-dump /tmp/dino.pos.dump
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = 0x4C424F31


# ---------------------------------------------------------------------------
# numpy replica of the C++ implementation in cpp_ggml/src/model.cpp
# ---------------------------------------------------------------------------

def bicubic_aa_filter(x):
    """Pillow bicubic (a = -0.5) used by PyTorch's antialias resampler."""
    a = np.float32(-0.5)
    x = np.float32(abs(x))
    if x < np.float32(1.0):
        return np.float32((np.float32(a + 2.0) * x - np.float32(a + 3.0)) * x * x + np.float32(1.0))
    if x < np.float32(2.0):
        return np.float32((((x - np.float32(5.0)) * x + np.float32(8.0)) * x - np.float32(4.0)) * a)
    return np.float32(0.0)


def aa_axis_weights(dst, in_size, scale):
    """One axis of upsample_antialias::_compute_weights_span/_compute_weights."""
    scale = np.float32(scale)
    support = np.float32(2.0 * scale) if scale >= np.float32(1.0) else np.float32(2.0)
    center = np.float32(scale * (dst + 0.5))
    start = max(int(center - support + np.float32(0.5)), 0)
    count = min(int(center + support + np.float32(0.5)), in_size) - start
    invscale = np.float32(1.0) / scale if scale >= np.float32(1.0) else np.float32(1.0)
    weights = np.empty(count, dtype=np.float32)
    total = np.float32(0.0)
    for j in range(count):
        w = bicubic_aa_filter(np.float32((j + start) - center + np.float32(0.5)) * invscale)
        weights[j] = w
        total = np.float32(total + w)
    if total != np.float32(0.0):
        weights /= total
    return start, weights


def interpolate_table(src, dim, m, out_w, out_h):
    """Resample patch rows of the table; replicate DINOv2's interpolation order.

    src: [1 + m*m, dim] float32 token-major. Returns [1 + out_w*out_h, dim].
    DINOv2 swaps w/h when calling interpolate_pos_encoding (w = x.shape[-2]),
    so F.interpolate receives size=(H_p, W_p) and torch flattens the
    resampled (out_h, out_w) grid row-major over the width: token t reads
    grid row t // out_w and column t % out_w. The row axis resamples
    m -> out_h and the column axis m -> out_w.
    """
    scale_y = np.float32(np.float32(m) / np.float32(out_h))
    scale_x = np.float32(np.float32(m) / np.float32(out_w))
    dst = np.zeros((1 + out_w * out_h, dim), dtype=np.float32)
    dst[0] = src[0]
    wy_cache, wx_cache = {}, {}
    for t in range(out_w * out_h):
        i, j = t // out_w, t % out_w
        if i not in wy_cache:
            wy_cache[i] = aa_axis_weights(i, m, scale_y)
        if j not in wx_cache:
            wx_cache[j] = aa_axis_weights(j, m, scale_x)
        (ymin, wy), (xmin, wx) = wy_cache[i], wx_cache[j]
        acc = np.zeros(dim, dtype=np.float32)
        for y, wyv in enumerate(wy):
            # token index of the x-window start for input row ymin+y
            base = 1 + (ymin + y) * m + xmin
            rows = src[base:base + len(wx)]
            horiz = np.zeros(dim, dtype=np.float32)
            for x, wxv in enumerate(wx):
                horiz = np.float32(horiz + np.float32(wxv) * rows[x])
            acc = np.float32(acc + np.float32(wyv) * horiz)
        dst[1 + t] = acc
    return dst


# ---------------------------------------------------------------------------
# PyTorch reference (interpolate_pos_encoding semantics)
# ---------------------------------------------------------------------------

def torch_reference(table, dim, m, width, height, device):
    import torch
    import torch.nn.functional as F
    w0, h0 = width // 14, height // 14
    npatch = w0 * h0
    x = torch.from_numpy(table).float().unsqueeze(0).to(device)
    if npatch == m * m and width == height:
        return x[0].cpu().numpy()
    cls, patch = x[:, :1], x[:, 1:]
    patch = patch.reshape(1, m, m, dim).permute(0, 3, 1, 2)
    # DINOv2 swaps w/h at the call site (w = x.shape[-2]), so the effective
    # interpolate size is (height_p, width_p) and the flattened grid is
    # row-major over the width.
    patch = F.interpolate(patch, size=(h0, w0), mode="bicubic", antialias=True)
    patch = patch.permute(0, 2, 3, 1).reshape(1, -1, dim)
    return torch.cat((cls, patch), dim=1)[0].cpu().numpy()


def load_gguf_pos_embed(path):
    """Return (table [tokens, dim] float32, M) from the GGUF pos_embed tensor."""
    from gguf import GGUFReader, GGMLQuantizationType
    from gguf.quants import dequantize
    reader = GGUFReader(str(path))
    for tensor in reader.tensors:
        if tensor.name != "aggregator.patch_embed.pos_embed":
            continue
        qtype = GGMLQuantizationType(tensor.tensor_type)
        dim = int(tensor.shape[0])            # ggml ne order: (dim, tokens, ...)
        tokens = int(np.prod(tensor.shape[1:]))
        values = dequantize(tensor.data, qtype)
        # tensor.data is a C-order view whose last axis is dim, i.e. token-major.
        arr = np.asarray(values, dtype=np.float32).reshape(tokens, dim)
        return arr.copy(), int(round((tokens - 1) ** 0.5))
    raise SystemExit(f"pos_embed tensor not found in {path}")


def load_cpp_dump(path, dim, tokens):
    raw = Path(path).read_bytes()
    values = np.frombuffer(raw, dtype=np.float32)
    if values.size != dim * tokens:
        raise SystemExit(f"cpp dump has {values.size} values, expected {dim * tokens}")
    # A contiguous ggml [dim, tokens] tensor streams token-major (ne[0] varies
    # fastest), matching the GGUF data view.
    return values.reshape(tokens, dim).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, help="GGUF containing the pos_embed tensor; "
                    "omit together with --torture for a random-table sweep")
    ap.add_argument("--height", type=int, default=392)
    ap.add_argument("--width", type=int, default=392)
    ap.add_argument("--cpp-dump", type=Path, help="dino.pos dump from LINGBOT_DUMP_STAGES")
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--torture", action="store_true", help="random-table algorithm sweep")
    ap.add_argument("--dim", type=int, default=1024)
    args = ap.parse_args()

    if args.gguf:
        table, _ = load_gguf_pos_embed(args.gguf)
    elif args.torture:
        table = None
    else:
        raise SystemExit("--gguf is required unless --torture is given")
    tokens, dim = (1 + 37 * 37, args.dim) if table is None else table.shape
    m = int(round((tokens - 1) ** 0.5))
    if m * m + 1 != tokens:
        raise SystemExit(f"pos_embed is not a square patch table: {tokens} tokens")
    print(f"gguf table: tokens={tokens} dim={dim} M={m}")

    if args.torture:
        rng = np.random.default_rng(0)
        worst = 0.0
        for (w0, h0) in [(24, 24), (26, 26), (28, 28), (32, 32), (37, 37), (40, 40), (28, 26)]:
            rand = rng.standard_normal((1 + m * m, 16)).astype(np.float32)
            ref = torch_reference(rand, 16, m, w0 * 14, h0 * 14, args.device)
            mine = interpolate_table(rand, 16, m, w0, h0)
            err = float(np.max(np.abs(ref - mine)))
            worst = max(worst, err)
            print(f"  random grid {w0}x{h0}: max_abs={err:.3e}")
        print(f"TORTURE {'PASS' if worst < 1e-4 else 'FAIL'} worst={worst:.3e}")

    if table is None:
        return

    ref = torch_reference(table, dim, m, args.width, args.height, args.device)
    w0, h0 = args.width // 14, args.height // 14
    if w0 * h0 + 1 == tokens and args.width == args.height:
        print("identity grid: PyTorch short-circuits, interpolation not exercised")
        return
    mine = interpolate_table(table, dim, m, w0, h0)
    err = float(np.max(np.abs(ref - mine)))
    print(f"algorithm vs torch ({w0}x{h0} grid): max_abs={err:.3e}")

    if args.cpp_dump:
        native = load_cpp_dump(args.cpp_dump, dim, ref.shape[0])
        err = float(np.max(np.abs(ref - native)))
        rel = err / float(np.max(np.abs(ref)))
        print(f"cpp dump vs torch: max_abs={err:.3e} rel={rel:.3e}")
        print("NATIVE " + ("PASS" if err < 1e-4 else "FAIL"))
    elif err < 1e-4:
        print("ALGORITHM PASS")


if __name__ == "__main__":
    main()
