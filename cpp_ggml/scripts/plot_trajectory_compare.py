#!/usr/bin/env python3
"""Plot the localization trajectory (camera centers in world frame) of the
PyTorch reference against one or more GGML full-stream LBO outputs.

This checkpoint's pose encoding is camera-to-world (verified by
`check_view_consistency.py`, and relied upon by `demo.py` + the viewer), so the
camera center is the translation component itself. Treating it as an OpenCV w2c
[R|t] and plotting -R^T t distorted every trajectory.
"""
import argparse
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_lbo(path: Path):
    raw = path.read_bytes()
    magic, n_pose, n_depth = struct.unpack("<III", raw[:12])
    if magic != 0x4C424F31:
        raise RuntimeError(f"invalid LBO1 output: {path}")
    values = np.frombuffer(raw, dtype=np.float32, offset=12)
    return values[:n_pose], values[n_pose:n_pose + n_depth]


def camera_centers(pose):
    return np.asarray(pose, dtype=np.float64).reshape(-1, 9)[:, :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pytorch", type=Path, required=True)
    ap.add_argument("--lbo", type=Path, action="append", required=True,
                    help="GGML LBO; repeatable")
    ap.add_argument("--label", action="append", default=[],
                    help="label for each --lbo, in order")
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="Localization trajectory (286 frames)")
    args = ap.parse_args()

    if args.label and len(args.label) != len(args.lbo):
        raise SystemExit("one --label per --lbo (or none)")

    ref = np.load(args.pytorch)
    ref_c = camera_centers(ref["pose_enc"])[:args.frames]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    views = [("top-down (x-z)", 0, 2), ("side (z-y)", 2, 1)]
    labels = args.label or [p.stem for p in args.lbo]

    axes[0].plot(ref_c[:, 0], ref_c[:, 2], "-", color="black", lw=2.0,
                 label="PyTorch reference", zorder=1)
    axes[1].plot(ref_c[:, 2], ref_c[:, 1], "-", color="black", lw=2.0,
                 label="PyTorch reference", zorder=1)
    colors = plt.cm.tab10(np.arange(len(labels)) % 10)
    for lab, path, col in zip(labels, args.lbo, colors):
        pose, _ = read_lbo(path)
        c = camera_centers(pose)[:args.frames]
        # per-frame center deviation from the reference, in world units
        dev = np.linalg.norm(c - ref_c, axis=1)
        axes[0].plot(c[:, 0], c[:, 2], "-", color=col, lw=1.2, alpha=0.9,
                     label=f"{lab} (center dev max {dev.max():.2e} m)")
        axes[1].plot(c[:, 2], c[:, 1], "-", color=col, lw=1.2, alpha=0.9, label=lab)

    for ax, (name, xi, yi) in zip(axes, views):
        ax.set_xlabel(f"world {'xyz'[xi]}")
        ax.set_ylabel(f"world {'xyz'[yi]}")
        ax.set_title(f"{args.title} — {name}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", linewidth=0.4)
        ax.legend(fontsize=8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
