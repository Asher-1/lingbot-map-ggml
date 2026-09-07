#!/usr/bin/env python3
"""Export a LingBot-Map checkpoint to GGUF.

The exporter preserves every checkpoint tensor under its original key and
stores architecture metadata consumed by the C++ loader. Quantization is
restricted to 2-D weight matrices; biases, norms and convolution kernels stay
floating point so that layout and normalization semantics remain explicit.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import gguf
from gguf.quants import quantize

ARCH = "lingbot-map"
_q4 = getattr(gguf.GGMLQuantizationType, "Q4_K_M", None)
if _q4 is None:
    _q4 = gguf.GGMLQuantizationType.Q4_0
# The Python GGUF quantizer shipped with some environments exposes the K
# enum but not its implementation. Q4_0 is the portable GGML v0.21 fallback.
try:
    quantize(np.zeros((1, 256), dtype=np.float32), _q4)
except Exception:
    _q4 = gguf.GGMLQuantizationType.Q4_0
QUANT = {"q8": gguf.GGMLQuantizationType.Q8_0, "q4": _q4}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--dino-checkpoint", type=Path, help="optional DINOv2 checkpoint used by the patch embed")
    ap.add_argument("--outtype", choices=("f32", "f16", "q8", "q4"), default="f16")
    ap.add_argument("--image-size", type=int, default=518)
    ap.add_argument("--patch-size", type=int, default=14)
    ap.add_argument("--embed-dim", type=int, default=1024)
    ap.add_argument("--block-count", type=int, default=24)
    ap.add_argument("--num-heads", type=int, default=16)
    ap.add_argument("--quantize-prefix", action="append", default=[], metavar="PREFIX",
                    help="for q8/q4, quantize only matrices whose names start with PREFIX; repeatable")
    ap.add_argument("--f32-prefix", action="append", default=[], metavar="PREFIX",
                    help="keep matching tensors in f32 (for bounded mixed-precision parity experiments)")
    args = ap.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = raw.get("model", raw.get("state_dict", raw))
    if not isinstance(state, dict) or not state:
        raise SystemExit("checkpoint does not contain a non-empty model/state_dict")
    if args.dino_checkpoint:
        if not args.dino_checkpoint.is_file():
            raise SystemExit(f"DINO checkpoint not found: {args.dino_checkpoint}")
        dino_raw = torch.load(args.dino_checkpoint, map_location="cpu", weights_only=False)
        dino_state = dino_raw.get("model", dino_raw.get("state_dict", dino_raw))
        if not isinstance(dino_state, dict) or not dino_state:
            raise SystemExit("DINO checkpoint does not contain a non-empty state_dict")
        # GCT loads DINO weights into aggregator.patch_embed; preserve that
        # ownership in GGUF so C++ never silently falls back to random weights.
        state = dict(state)
        for key, value in dino_state.items():
            state.setdefault("aggregator.patch_embed." + key, value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(args.output), ARCH)
    writer.add_uint32(f"{ARCH}.image_size", args.image_size)
    writer.add_uint32(f"{ARCH}.patch_size", args.patch_size)
    writer.add_uint32(f"{ARCH}.embedding_length", args.embed_dim)
    writer.add_uint32(f"{ARCH}.block_count", args.block_count)
    writer.add_uint32(f"{ARCH}.num_heads", args.num_heads)
    writer.add_uint32(f"{ARCH}.output_dim", 9)
    writer.add_string(f"{ARCH}.weight_type", args.outtype)
    writer.add_string(f"{ARCH}.graph_version", "gct-v1-contract")
    qtype = QUANT.get(args.outtype)
    for name, value in state.items():
        if not torch.is_tensor(value):
            continue
        arr = value.detach().to(torch.float32).cpu().numpy()
        # Quantizers require rows divisible by their block size. Leave such
        # tensors in f16 when the source shape cannot be represented exactly.
        selected = not args.quantize_prefix or any(name.startswith(p) for p in args.quantize_prefix)
        keep_f32 = any(name.startswith(p) for p in args.f32_prefix)
        quantizable = qtype is not None and selected and not keep_f32 and arr.ndim == 2 and arr.shape[-1] % 32 == 0
        if quantizable:
            writer.add_tensor(name, quantize(arr, qtype), raw_dtype=qtype)
        else:
            dtype = np.float32 if args.outtype == "f32" or keep_f32 else np.float16
            writer.add_tensor(name, arr.astype(dtype))
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file()
    print(f"wrote {args.output} ({args.outtype})")

if __name__ == "__main__":
    main()
