#!/usr/bin/env python3
"""End-to-end sky-mask alignment: official onnxruntime chain vs native ggml.

Official chain per frame (lingbot_map/vis/sky_segmentation.py):
    cv2.imread -> cv2.resize(u8) 320x320 -> ImageNet norm -> onnxruntime
    -> min-max normalize *255, astype(u8) -> cv2.resize(u8, (W,H), LINEAR)
    -> keep = (resized u8 == 0)   [the uint8 clip collapses every non-zero]

ggml chain: skyseg-cli raw sigmoid map -> same min-max u8 -> the bit-exact
cv2 u8 INTER_LINEAR replica inside skyseg.cpp (skyseg_resize_mask_u8)
-> same keep rule.

Prints per-quantization pixel agreement and keep-mask IoU over the requested
frames, plus (optionally) latency for every gguf x backend pair via
skyseg-cli's internal timing loop.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLI = os.path.join(ROOT, "cpp_ggml", "build-cuda", "skyseg-cli")

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def official_keep(bgr, session, input_name, output_name):
    """Official sky_mask semantics -> bool array, True where the pixel is kept."""
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (320, 320))
    x = small[:, :, ::-1].astype(np.float32) / 255.0
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    raw = session.run([output_name], {input_name: x})[0].squeeze()
    lo, hi = float(raw.min()), float(raw.max())
    u8 = ((raw - lo) * 255.0 / max(hi - lo, 1e-8)).astype(np.uint8)
    up = cv2.resize(u8, (w, h), interpolation=cv2.INTER_LINEAR)
    return up == 0


def ggml_keep_python(raw_map, h, w):
    """Python mirror of skyseg_resize_mask_u8 (verified bit-exact vs cv2)."""
    lo, hi = float(raw_map.min()), float(raw_map.max())
    u8 = ((raw_map - lo) * 255.0 / max(hi - lo, 1e-8)).astype(np.uint8)

    def exact(src, dw, dh):
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

    return exact(u8, w, h) == 0


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default=os.path.join(ROOT, "example/courthouse"))
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--onnx", default=os.path.join(ROOT, "skyseg.onnx"))
    ap.add_argument("--gguf-dir", default=os.path.join(ROOT, "cpp_ggml/models/gguf"))
    ap.add_argument("--quants", nargs="+", default=["f32", "f16", "q8_0"])
    ap.add_argument("--backend", default="CPU")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.frames_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))[:args.num_frames]
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    results = {}
    for q in args.quants:
        gguf = os.path.join(args.gguf_dir, f"lingbot-map-skyseg-{q}.gguf")
        if not os.path.isfile(gguf):
            print(f"skip {q}: missing {gguf}")
            continue
        agree_all, ious, per_frame = [], [], []
        with tempfile.TemporaryDirectory(prefix="skyseg_align_") as tmp:
            for name in files:
                bgr = cv2.imread(os.path.join(args.frames_dir, name))
                h, w = bgr.shape[:2]
                keep_off = official_keep(bgr, session, input_name, output_name)
                prefix = os.path.join(tmp, name)
                cmd = [CLI, gguf, os.path.join(args.frames_dir, name),
                       args.backend, prefix, "1"]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    print(r.stdout, r.stderr); raise SystemExit(f"cli failed on {name}")
                raw = np.fromfile(prefix + ".f32", np.float32)
                keep_gg = ggml_keep_python(raw.reshape(320, 320), h, w)
                agree = float((keep_off == keep_gg).mean())
                agree_all.append(agree); ious.append(iou(keep_off, keep_gg))
                per_frame.append({"frame": name, "agreement": agree, "keep_iou": iou(keep_off, keep_gg)})
        results[q] = {
            "mean_agreement": float(np.mean(agree_all)),
            "min_agreement": float(np.min(agree_all)),
            "keep_iou_min": float(np.min(ious)),
            "keep_iou_max": float(np.max(ious)),
            "frames": per_frame,
        }
        print(f"{q:>5}: agreement mean={np.mean(agree_all)*100:.2f}% "
              f"min={np.min(agree_all)*100:.2f}%  keep-IoU {np.min(ious):.4f}-{np.max(ious):.4f}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=1)
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()
