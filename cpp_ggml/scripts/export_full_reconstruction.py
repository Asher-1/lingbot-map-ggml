#!/usr/bin/env python3
"""Export all-frame PyTorch/GGML depth reconstructions as colored point clouds.

The unprojection, pose convention and confidence filtering all come from
`export_reconstruction_cloud.py`, so the two exporters cannot drift apart. This
script only pairs a PyTorch reference with its GGML counterpart and renders the
two clouds side by side.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_reconstruction_cloud import build_cloud, read_lbo, read_lbo_post, write_ply


def draw_cloud(ax, cloud, title, max_points=180000):
    ids = np.linspace(0, len(cloud) - 1, min(max_points, len(cloud)), dtype=np.int64)
    colors = np.stack((cloud["red"][ids], cloud["green"][ids], cloud["blue"][ids]), axis=1) / 255.0
    x, z = cloud["x"][ids], cloud["z"][ids]
    ax.scatter(x, z, c=colors, s=0.12, linewidths=0, rasterized=True)
    ax.set_title(title)
    ax.set_xlabel("world x")
    ax.set_ylabel("world z")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.4)
    # Percentile limits keep a few far-field outliers from collapsing the view.
    for setter, values in ((ax.set_xlim, x), (ax.set_ylim, z)):
        lo, hi = np.percentile(values, (1, 99))
        pad = 0.05 * max(hi - lo, 1e-3)
        setter(lo - pad, hi + pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--pytorch", type=Path, required=True)
    ap.add_argument("--ggml-lbo", type=Path, required=True)
    ap.add_argument("--ggml-lbo-post", type=Path,
                    help="LBP2 sidecar carrying depth_conf (default: <ggml-lbo>.post)")
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--ggml-label", default="q8")
    # --height/--width are the model input grid (the CLI's H W arguments); depth
    # and RGB are reshaped/resized to it.
    ap.add_argument("--height", type=int, default=518)
    ap.add_argument("--width", type=int, default=518)
    ap.add_argument("--conf-threshold", type=float, default=1.5,
                    help="drop pixels with depth_conf below this (demo.py default)")
    ap.add_argument("--downsample", type=int, default=4, help="keep every Nth pixel")
    args = ap.parse_args()

    frames, height, width = args.frames, args.height, args.width
    images = sorted(args.scene.glob("*.png"))[:frames]
    if len(images) != frames:
        raise RuntimeError(f"expected {frames} PNG frames, found {len(images)}")
    rgb = np.stack([np.asarray(Image.open(p).convert("RGB").resize((width, height)))
                    for p in images])

    ref = np.load(args.pytorch)
    pt_pose = np.asarray(ref["pose_enc"], dtype=np.float32).reshape(frames, 9)
    pt_depth = np.asarray(ref["depth"], dtype=np.float32).reshape(frames, height, width)
    pt_conf = np.asarray(ref["depth_conf"], dtype=np.float32).reshape(frames, height, width)

    ggml_pose, ggml_depth = read_lbo(args.ggml_lbo)
    ggml_pose = ggml_pose.reshape(frames, 9)
    ggml_depth = ggml_depth.reshape(frames, height, width)
    post = args.ggml_lbo_post or args.ggml_lbo.with_suffix(args.ggml_lbo.suffix + ".post")
    ggml_conf = read_lbo_post(post, frames, height * width).reshape(frames, height, width)

    pt_cloud, pt_kept = build_cloud(pt_pose, pt_depth, pt_conf, rgb,
                                    args.conf_threshold, args.downsample)
    ggml_cloud, ggml_kept = build_cloud(ggml_pose, ggml_depth, ggml_conf, rgb,
                                        args.conf_threshold, args.downsample)
    pt_ply = args.out_prefix.with_name(args.out_prefix.name + "_pytorch.ply")
    ggml_ply = args.out_prefix.with_name(args.out_prefix.name + f"_ggml_{args.ggml_label}.ply")
    write_ply(pt_ply, pt_cloud)
    write_ply(ggml_ply, ggml_cloud)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    draw_cloud(axes[0], pt_cloud,
               f"PyTorch reference — {frames} frames {width}x{height} (kept {100 * pt_kept:.1f}%)")
    draw_cloud(axes[1], ggml_cloud,
               f"GGML {args.ggml_label} — {frames} frames {width}x{height} (kept {100 * ggml_kept:.1f}%)")
    effect = args.out_prefix.with_name(args.out_prefix.name + "_comparison.png")
    effect.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(effect, dpi=180)
    plt.close(fig)
    print(f"wrote {pt_ply}: {len(pt_cloud)} points")
    print(f"wrote {ggml_ply}: {len(ggml_cloud)} points")
    print(f"wrote {effect}")


if __name__ == "__main__":
    main()
