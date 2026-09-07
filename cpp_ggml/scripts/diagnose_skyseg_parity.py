#!/usr/bin/env python3
"""Locate the source of the skyseg ggml-vs-onnx mask disagreement.

Protocol (frame 7 of the courthouse stream by default):

1. Official chain   : cv2.imread -> cv2.resize(u8) 320x320 -> norm -> onnxruntime
2. GGML input chain : u8 round -> resize_rgb_u8 (half_pixel + lround, the C++
                      resize_rgb_u8 semantics) -> norm -> the SAME onnxruntime
                      network. Isolates the preprocessing difference.
3. Native ggml      : skyseg-cli on the same PNG (f32 GGUF, CPU), with
                      SKYSEG_DUMP_LAYER for every simplified-graph node.
                      Layer names equal the onnx node output names, so each
                      ggml dump {W,H,C,N} is compared against the onnx
                      intermediate of the same name run on input chain 2.

Outputs per node: max abs diff / RMSE, printed in graph order. The first
node whose diff jumps marks the layer where the engines diverge.
"""
import os
import subprocess
import sys

import cv2
import numpy as np
import onnx
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILD = os.path.join(ROOT, "cpp_ggml", "build-cuda")  # cli binary location
SIM = "/tmp/skyseg_sim.onnx"
GGUF = "/tmp/skyseg-f32.gguf"
CLI = os.path.join(BUILD, "skyseg-cli")
WORK = "/tmp/skyseg_diag"

K_IN = 320
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_rgb_u8(src, dw, dh):
    """Mirror of src/skyseg.cpp resize_rgb_u8 (interleaved RGB u8 in/out)."""
    sh, sw, _ = src.shape
    dst = np.empty((dh, dw, 3), np.uint8)
    ys = (np.arange(dh) + 0.5) * (sh / dh) - 0.5
    xs = (np.arange(dw) + 0.5) * (sw / dw) - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, sh - 1)
    y1 = np.clip(y0 + 1, 0, sh - 1)
    fy = np.clip(ys - np.floor(ys), 0, 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, sw - 1)
    x1 = np.clip(x0 + 1, 0, sw - 1)
    fx = np.clip(xs - np.floor(xs), 0, 1)
    # gather then weight (float, like the C++ path)
    v00 = src[np.ix_(y0, x0)].astype(np.float32)
    v01 = src[np.ix_(y0, x1)].astype(np.float32)
    v10 = src[np.ix_(y1, x0)].astype(np.float32)
    v11 = src[np.ix_(y1, x1)].astype(np.float32)
    fxg = fx[None, :, None]
    fyg = fy[:, None, None]
    v = (v00 * (1 - fxg) * (1 - fyg) + v01 * fxg * (1 - fyg)
         + v10 * (1 - fxg) * fyg + v11 * fxg * fyg)
    np.rint(v, out=v)
    dst[...] = np.clip(v, 0, 255).astype(np.uint8)
    return dst


def norm_chw(rgb320):
    x = rgb320.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def official_input(bgr):
    small = cv2.resize(bgr, (K_IN, K_IN))          # cv2 u8 bilinear
    return norm_chw(small[:, :, ::-1])             # BGR->RGB


def ggml_input(bgr):
    small = resize_rgb_u8(bgr[:, :, ::-1], K_IN, K_IN)  # rgb u8 in, C++ mirror
    return norm_chw(small)


def build_extended_model():
    ext_path = os.path.join(WORK, "skyseg_sim_ext.onnx")
    if os.path.exists(ext_path):
        return ext_path
    m = onnx.load(SIM)
    names = [n.output[0] for n in m.graph.node if n.op_type != "Constant"]
    shapes = {}
    for v in m.graph.value_info:
        shapes[v.name] = v
    have = {o.name for o in m.graph.output}
    for nm in names:
        if nm in have:
            continue
        vi = shapes.get(nm)
        if vi is None:
            vi = onnx.helper.make_empty_tensor_value_info(nm)
        out = m.graph.output.add()
        out.CopyFrom(vi)
    onnx.save(m, ext_path)
    return ext_path


def ggml_layout_to_nchw(raw, c, h, w):
    """ggml dump order: W fastest, then H, then C, then N(=1)."""
    a = np.fromfile(raw, np.float32)
    return a.reshape(1, c, h, w)


def main():
    frame = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "example/courthouse/000007.png")
    os.makedirs(WORK, exist_ok=True)
    bgr = cv2.imread(frame)
    print(f"frame {frame}: {bgr.shape[1]}x{bgr.shape[0]}")

    x_official = official_input(bgr)
    x_ggml = ggml_input(bgr)
    print(f"preprocess input diff (official vs ggml-mirror): "
          f"max={np.abs(x_official - x_ggml).max():.6f} "
          f"rmse={np.sqrt(((x_official - x_ggml) ** 2).mean()):.6f}")

    ext = build_extended_model()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(ext, so, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    res_off = sess.run(out_names, {"input.1": x_official})
    res_ggml = sess.run(out_names, {"input.1": x_ggml})
    ort_out = dict(zip(out_names, res_ggml))
    ort_out_off = dict(zip(out_names, res_off))

    raw_off = ort_out_off["1959"][0, 0]
    raw_gml = ort_out["1959"][0, 0]
    print(f"onnx raw '1959': official-input vs ggml-input "
          f"max={np.abs(raw_off - raw_gml).max():.6g} "
          f"rmse={np.sqrt(((raw_off - raw_gml) ** 2).mean()):.6g}")

    # native ggml run with per-node dumps
    m = onnx.load(SIM)
    node_names = [n.output[0] for n in m.graph.node if n.op_type != "Constant"]
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in m.graph.value_info}
    shapes["input.1"] = [1, 3, K_IN, K_IN]
    env = {**os.environ,
           "SKYSEG_DUMP_INPUT": os.path.join(WORK, "ggml_input.f32"),
           "SKYSEG_DUMP_LAYER": ",".join(node_names),
           "SKYSEG_DUMP_LAYER_OUT": os.path.join(WORK, "ggml")}
    prefix = os.path.join(WORK, "ggml_f32")
    cmd = [CLI, GGUF, frame, "CPU", prefix, "1"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit("skyseg-cli failed")
    print("".join(l for l in r.stdout.splitlines(keepends=True) if l.startswith("map ") or l.startswith("RESULT")))

    dumped_input = np.fromfile(os.path.join(WORK, "ggml_input.f32"), np.float32)
    dumped_input = dumped_input.reshape(3, K_IN, K_IN)[None]
    print(f"ggml preprocess (dumped) vs python mirror: "
          f"max={np.abs(dumped_input - x_ggml).max():.6g}")

    # where does the input-chain difference live? save a diff heat map
    dv = np.abs(x_official - x_ggml).max(axis=(0, 1))  # HxW, max over C
    print(f"input diff pixels >1 u8-level: {(dv > 1 / (255 * STD.min())).sum()}")

    raw_cli = np.fromfile(prefix + ".f32", np.float32).reshape(1, 1, K_IN, K_IN)[0, 0]
    raw_diff = raw_cli - raw_gml
    print(f"native ggml raw vs onnx raw (ggml input): "
          f"max={np.abs(raw_diff).max():.6g} "
          f"rmse={np.sqrt((raw_diff ** 2).mean()):.6g}")
    print(f"native ggml raw vs onnx raw (official input): "
          f"max={np.abs(raw_cli - raw_off).max():.6g} "
          f"rmse={np.sqrt(((raw_cli - raw_off) ** 2).mean()):.6g}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(raw_gml, cmap="gray"); axes[0].set_title("onnx raw (ggml input)")
    axes[1].imshow(raw_cli, cmap="gray"); axes[1].set_title("ggml f32 raw")
    axes[2].imshow(np.abs(raw_diff), cmap="inferno"); axes[2].set_title("|diff|")
    for ax in axes: ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(WORK, "raw_diff.png"), dpi=90)
    print(f"saved {WORK}/raw_diff.png")

    # per-node comparison, graph order
    rows = []
    for nm in node_names:
        p = os.path.join(WORK, f"ggml.{nm}.f32")
        if not os.path.exists(p) or nm not in ort_out:
            continue
        c, h, w = shapes[nm][1:]
        g = ggml_layout_to_nchw(p, c, h, w)[0]
        o = ort_out[nm][0]
        d = np.abs(g - o)
        rows.append((nm, c, h, w, d.max(), np.sqrt((d ** 2).mean())))
    print(f"\n{'node':>12} {'C':>4} {'H':>4} {'W':>4} {'max':>10} {'rmse':>10}")
    for nm, c, h, w, mx, rm in rows:
        print(f"{nm:>12} {c:>4} {h:>4} {w:>4} {mx:>10.6g} {rm:>10.6g}")
    np.save(os.path.join(WORK, "rows.npy"), np.array(
        [(nm, mx, rm) for nm, _, _, _, mx, rm in rows], dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()
