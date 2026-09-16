#!/usr/bin/env python3
"""Prepare the official demo long sequences for the real-reconstruction
comparison campaign (cpp_ggml/scripts/long_real/).

For each dataset (lingbo_world | drive | indoor):
  1. extract frames from the official demo video exactly like demo.py's
     load_images video branch (cv2, target fps=10, interval=round(src/fps),
     default JPEG quality), stopping early once --max-frames are extracted;
  2. preprocess with the official crop rule (load_and_preprocess_images
     mode="crop", image_size=518, patch_size=14) — the byte-identical input
     both engines consume;
  3. write <ds>.bin [N,3,H,W] float32 (CLI), <ds>.npy [1,N,3,H,W] (mirror)
     and a meta.json (fps, interval, frame count, resolution, auto
     keyframe_interval = ceil(N/320) per demo.py's streaming rule).

Usage:  prepare_data.py --dataset lingbo_world [--max-frames 2000]
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lingbot_map.utils.load_fn import load_and_preprocess_images  # noqa: E402

VIDEOS = Path("/tmp/long_real/videos")
OUT = Path("/tmp/long_real")
DATASETS = {
    "lingbo_world": "lingbo_world_frames.mp4",
    "drive": "drive_frames.mp4",
    "indoor": "indoor_travel.MP4",
}


def extract_frames(video: Path, out_dir: Path, fps: float, max_frames: int):
    """demo.py load_images video branch, with an early stop at max_frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, round(src_fps / fps))
    saved = []
    idx = 0
    while True:
        if len(saved) >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            path = out_dir / f"{len(saved):06d}.jpg"
            cv2.imwrite(str(path), frame)
            saved.append(str(path))
        idx += 1
    cap.release()
    return saved, src_fps, interval, total_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--fps", type=float, default=10.0, help="official demo.py default")
    ap.add_argument("--max-frames", type=int, default=2000)
    args = ap.parse_args()

    video = VIDEOS / DATASETS[args.dataset]
    frame_dir = OUT / "frames" / args.dataset
    meta_path = OUT / f"{args.dataset}.meta.json"

    # Idempotent: reuse an extracted frame folder when the count matches.
    existing = sorted(frame_dir.glob("*.jpg")) if frame_dir.exists() else []
    if len(existing) >= args.max_frames or (existing and meta_path.exists()):
        paths = existing[: args.max_frames]
        meta = json.loads(meta_path.read_text())
        print(f"[{args.dataset}] reusing {len(paths)} extracted frames")
    else:
        paths, src_fps, interval, total = extract_frames(video, frame_dir, args.fps, args.max_frames)
        meta = {"video": str(video), "src_fps": src_fps, "fps": args.fps,
                "interval": interval, "src_total_frames": total}
        print(f"[{args.dataset}] extracted {len(paths)} frames "
              f"(src {total} @ {src_fps:.0f}fps, interval={interval})")

    print(f"[{args.dataset}] preprocessing {len(paths)} images (official crop rule)...")
    images = load_and_preprocess_images(paths, mode="crop", image_size=518, patch_size=14)
    _, _, h, w = images.shape
    arr = images.numpy().astype(np.float32)
    bin_path = OUT / f"{args.dataset}.bin"
    npy_path = OUT / f"{args.dataset}.npy"
    arr.tofile(bin_path)
    np.save(npy_path, arr[None])

    n = arr.shape[0]
    meta.update({"n_frames": n, "height": int(h), "width": int(w),
                 # demo.py auto keyframe policy for streaming runs > 320 frames
                 "auto_keyframe_interval": max(1, math.ceil(n / 320)),
                 "bin": str(bin_path), "npy": str(npy_path),
                 "bytes": bin_path.stat().st_size})
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[{args.dataset}] {n} frames {w}x{h} -> {bin_path} ({meta['bytes']/1e9:.2f} GB) "
          f"auto_kf_interval={meta['auto_keyframe_interval']}")


if __name__ == "__main__":
    main()
