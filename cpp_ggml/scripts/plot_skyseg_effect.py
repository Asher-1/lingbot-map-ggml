#!/usr/bin/env python3
"""Recreate the sky-segmentation effect sheet: official onnxruntime vs native
GGML GGUF on the 8-frame courthouse stream (rows: frames 0/6/7 by default,
columns: input | official overlay | ggml overlay | pixel disagreement).

Overlays use the official _save_sky_mask_visualization tint (sky pixels get
[255,64,64] @ 0.65). The ggml column uses the f16 GGUF raw map postprocessed
with the same min-max u8 + bit-exact cv2 resize chain the CLI runs.
"""
import os
import subprocess
import tempfile

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLI = os.path.join(ROOT, "cpp_ggml", "build-cuda", "skyseg-cli")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def exact_u8_resize(src, dw, dh):
    sh, sw = src.shape
    fx = np.float32((np.arange(dw) + 0.5) * (sw / dw) - 0.5)
    sx = np.floor(fx).astype(np.int64)
    fx = (fx - sx.astype(np.float32)).astype(np.float32)
    neg = sx < 0; right = sx >= sw - 1
    fx = np.where(neg | right, np.float32(0), fx)
    sx = np.where(neg, 0, sx); sx = np.where(right, sw - 1, sx)
    a0 = np.rint((np.float32(1) - fx) * np.float32(2048)).astype(np.int64)
    a1 = np.rint(fx * np.float32(2048)).astype(np.int64)
    fy = np.float32((np.arange(dh) + 0.5) * (sh / dh) - 0.5)
    sy = np.floor(fy).astype(np.int64)
    fy = (fy - sy.astype(np.float32)).astype(np.float32)
    b0 = np.rint((np.float32(1) - fy) * np.float32(2048)).astype(np.int64)
    b1 = np.rint(fy * np.float32(2048)).astype(np.int64)
    r0 = np.clip(sy, 0, sh - 1); r1 = np.clip(sy + 1, 0, sh - 1)
    si = src.astype(np.int64)
    h0 = a0[None, :] * si[:, sx] + a1[None, :] * si[:, np.minimum(sx + 1, sw - 1)]
    t = (b0[:, None] * (h0[r0] >> 4) >> 16) + (b1[:, None] * (h0[r1] >> 4) >> 16) + 2
    return (t >> 2).astype(np.uint8)


def keep_mask_official(bgr, session, iname, oname):
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (320, 320))
    x = small[:, :, ::-1].astype(np.float32) / 255.0
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    raw = session.run([oname], {iname: x})[0].squeeze()
    lo, hi = float(raw.min()), float(raw.max())
    u8 = ((raw - lo) * 255.0 / max(hi - lo, 1e-8)).astype(np.uint8)
    return cv2.resize(u8, (w, h), interpolation=cv2.INTER_LINEAR) == 0


def keep_mask_ggml(gguf, png, backend, h, w):
    with tempfile.TemporaryDirectory(prefix="skyseg_eff_") as tmp:
        prefix = os.path.join(tmp, "map")
        r = subprocess.run([CLI, gguf, png, backend, prefix, "1"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(r.stdout + r.stderr)
        raw = np.fromfile(prefix + ".f32", np.float32).reshape(320, 320)
    lo, hi = float(raw.min()), float(raw.max())
    u8 = ((raw - lo) * 255.0 / max(hi - lo, 1e-8)).astype(np.uint8)
    return exact_u8_resize(u8, w, h) == 0


def overlay(rgb, keep):
    out = rgb.astype(np.float32).copy()
    sky = ~keep
    out[sky] = out[sky] * 0.35 + np.array([255, 64, 64], np.float32) * 0.65
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    frames_dir = os.path.join(ROOT, "example/courthouse")
    rows = [0, 6, 7]
    gguf = os.path.join(ROOT, "cpp_ggml/models/gguf/lingbot-map-skyseg-f16.gguf")
    onnx = os.path.join(ROOT, "skyseg.onnx")
    if not os.path.isfile(onnx):
        onnx = "/tmp/skyseg.onnx"
    session = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    iname = session.get_inputs()[0].name
    oname = session.get_outputs()[0].name

    fig, axes = plt.subplots(len(rows), 4, figsize=(19, 3.6 * len(rows)))
    for ri, idx in enumerate(rows):
        name = f"{idx:06d}.png"
        png = os.path.join(frames_dir, name)
        bgr = cv2.imread(png)
        h, w = bgr.shape[:2]
        rgb = bgr[:, :, ::-1]
        keep_off = keep_mask_official(bgr, session, iname, oname)
        keep_gg = keep_mask_ggml(gguf, png, "CUDA0", h, w)
        agree = (keep_off == keep_gg).mean() * 100
        dis = (~keep_off & keep_gg) | (keep_off & ~keep_gg)
        axes[ri][0].imshow(rgb); axes[ri][0].set_ylabel(f"frame {idx}", fontsize=11)
        axes[ri][1].imshow(overlay(rgb, keep_off))
        axes[ri][1].set_title("official skyseg.onnx" if ri == 0 else None)
        axes[ri][2].imshow(overlay(rgb, keep_gg))
        axes[ri][2].set_title("GGML skyseg GGUF (f16)" if ri == 0 else None)
        axes[ri][3].imshow(dis, cmap="gray")
        axes[ri][3].set_title("pixel disagreement" if ri == 0 else None)
        axes[ri][3].text(w / 2, h + h * 0.12, f"agreement {agree:.2f}%",
                         ha="center", fontsize=11)
        if ri == 0:
            axes[ri][0].set_title("input frame")
        for ax in axes[ri]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("sky segmentation: official onnxruntime vs native GGML GGUF — "
                 "sky tinted red, 8-frame courthouse stream", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(ROOT, "cpp_ggml/benchmarks/skyseg_effect_comparison_20260912.png")
    fig.savefig(out, dpi=100)
    print("wrote", out)


if __name__ == "__main__":
    main()
