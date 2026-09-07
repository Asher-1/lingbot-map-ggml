#!/usr/bin/env python3
"""Create a visual PyTorch/GGML reconstruction comparison for one scene."""
import argparse
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def read_lbo(path: Path):
    raw = path.read_bytes()
    magic, n_pose, n_depth = struct.unpack("<III", raw[:12])
    if magic != 0x4C424F31:
        raise RuntimeError(f"invalid LBO1 output: {path}")
    values = np.frombuffer(raw, dtype=np.float32, offset=12)
    return values[:n_pose], values[n_pose:n_pose + n_depth]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--pytorch", type=Path, required=True)
    ap.add_argument("--ggml-lbo", type=Path, required=True)
    ap.add_argument("--ggml-label", default="ggml")
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    args = ap.parse_args()

    image_files = sorted(args.scene.glob("*.png"))[:args.frames]
    if len(image_files) != args.frames:
        raise RuntimeError(f"expected {args.frames} PNG frames, found {len(image_files)}")
    rgb = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in image_files])
    ref = np.load(args.pytorch)
    pt_depth = np.asarray(ref["depth"], dtype=np.float32)
    pt_pose = np.asarray(ref["pose_enc"], dtype=np.float32).reshape(args.frames, 9)
    ggml_pose, ggml_depth = read_lbo(args.ggml_lbo)
    # Depth grid follows the run's actual resolution: --height/--width when
    # given (non-square profiles like 518x378), else inferred for square runs.
    if args.height and args.width:
        height, width = args.height, args.width
    else:
        height = width = int(round((pt_depth.size // args.frames) ** 0.5))
    pt_depth = pt_depth.reshape(args.frames, height, width)
    ggml_depth = ggml_depth.reshape(args.frames, height, width)
    ggml_pose = ggml_pose.reshape(args.frames, 9)

    finite = np.concatenate((pt_depth[np.isfinite(pt_depth)], ggml_depth[np.isfinite(ggml_depth)]))
    finite = finite[finite > 0]
    if finite.size == 0:
        raise RuntimeError("no positive depth values to visualize")
    vmin, vmax = np.percentile(finite, (1.0, 99.0))
    diff = np.abs(ggml_depth - pt_depth)
    dmax = max(float(np.percentile(diff, 99.5)), 1e-6)
    picks = np.linspace(0, args.frames - 1, min(4, args.frames), dtype=int)

    fig = plt.figure(figsize=(4.2 * len(picks), 14.0))
    grid = fig.add_gridspec(5, len(picks), height_ratios=(1.0, 1.0, 1.0, 1.0, 2.4),
                            hspace=0.62, top=0.92, bottom=0.05, left=0.04, right=0.99)
    rows = [(rgb, "Official RGB"), (pt_depth, "PyTorch depth"),
            (ggml_depth, f"GGML {args.ggml_label} depth"), (diff, f"|{args.ggml_label} - PyTorch|")]
    for row, (data, title) in enumerate(rows):
        for col, frame_idx in enumerate(picks):
            ax = fig.add_subplot(grid[row, col])
            if row == 0:
                ax.imshow(data[frame_idx])
            else:
                cmap = "magma" if row == 3 else "viridis"
                lo, hi = (0.0, dmax) if row == 3 else (vmin, vmax)
                ax.imshow(data[frame_idx], cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(f"{title}\nframe {frame_idx:04d}", fontsize=10, pad=4)
            ax.axis("off")

    ax = fig.add_subplot(grid[4, :])
    # Translation is directly encoded in the first three pose values. Centering
    # both curves makes the comparison independent of the global origin.
    for pose, name, color in ((pt_pose, "PyTorch", "#1f77b4"),
                              (ggml_pose, f"GGML {args.ggml_label}", "#d62728")):
        xyz = pose[:, :3] - pose[0, :3]
        ax.plot(xyz[:, 0], xyz[:, 2], "o-", ms=3, lw=1.5, label=name, color=color)
    ax.set_xlabel("Camera trajectory x (z, origin at frame 0)")
    ax.set_ylabel("z"); ax.grid(True, linestyle=":")
    # adjustable="box" shrinks the axes to the data instead of padding the x
    # limits out to the full figure width, which flattened the loop to a dot.
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    fig.suptitle(
        f"{args.scene.name}: PyTorch vs GGML {args.ggml_label} reconstruction | "
        f"depth RMSE={np.sqrt(np.mean((ggml_depth - pt_depth) ** 2)):.5f}, "
        f"max abs={np.max(diff):.5f}", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"wrote reconstruction effect image: {args.output}")


if __name__ == "__main__":
    main()
