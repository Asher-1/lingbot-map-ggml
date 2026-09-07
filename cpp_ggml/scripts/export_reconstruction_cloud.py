#!/usr/bin/env python3
"""Export one reconstruction run as a confidence-filtered colored point cloud.

Unlike the older `export_full_reconstruction.py`, this exporter reuses the
official visualization contract instead of re-deriving it:

  * intrinsics/extrinsics come from `pose_encoding_to_extri_intri`, so the FoV
    -> focal mapping is the checkpoint's own;
  * points come from `unproject_depth_map_to_point_map`, fed the same transform
    `demo.py:postprocess` feeds the viewer, i.e. the *inverse* of the decoded
    pose. The docstrings label the decoded pose "cam from world", but this
    checkpoint's pose encoding is camera-to-world: `check_view_consistency.py`
    measures 3% median cross-frame depth error this way versus 43% without the
    inversion, and the shipped demo relies on the same cancellation;
  * low-confidence pixels are dropped with the same `depth_conf > threshold`
    rule (and default 1.5) that `demo.py --conf_threshold` feeds into
    `PointCloudViewer`. Without this filter the fused cloud is dominated by
    sky / texture-less pixels the model itself marks as unreliable.

Inputs may be a PyTorch reference `.npz` (pose_enc/depth/depth_conf) or the
C++ CLI's `.lbo` plus its `.lbo.post` sidecar (which carries depth_conf).
"""
import argparse
import struct
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map


def invert_se3(extrinsic):
    """Invert a batch of 3x4 transforms, mirroring `demo.py:postprocess`."""
    square = np.zeros((len(extrinsic), 4, 4), dtype=np.float64)
    square[:, :3, :4] = extrinsic
    square[:, 3, 3] = 1.0
    return closed_form_inverse_se3(square)[:, :3, :4]


def read_lbo(path: Path):
    raw = path.read_bytes()
    magic, n_pose, n_depth = struct.unpack("<III", raw[:12])
    if magic != 0x4C424F31:
        raise RuntimeError(f"invalid LBO1 output: {path}")
    values = np.frombuffer(raw, dtype=np.float32, offset=12)
    return values[:n_pose], values[n_pose:n_pose + n_depth]


def read_lbo_post(path: Path, frames: int, pixels: int):
    """Read the LBP2 sidecar and return its depth-confidence block."""
    raw = path.read_bytes()
    magic, n_c2w, n_intr, n_conf = struct.unpack("<IIII", raw[:16])
    if magic != 0x4C425032:
        raise RuntimeError(f"invalid LBP2 sidecar: {path}")
    values = np.frombuffer(raw, dtype=np.float32, offset=16)
    conf = values[n_c2w + n_intr:n_c2w + n_intr + n_conf]
    if conf.size != frames * pixels:
        raise RuntimeError(f"{path}: confidence has {conf.size} values, expected {frames * pixels}")
    return conf


def load_run(args):
    """Return (pose[S,9], depth[S,H,W], conf[S,H,W])."""
    if args.npz:
        data = np.load(args.npz)
        pose = np.asarray(data["pose_enc"], dtype=np.float32).reshape(-1, 9)
        depth = np.asarray(data["depth"], dtype=np.float32).reshape(len(pose), -1)
        conf = np.asarray(data["depth_conf"], dtype=np.float32).reshape(len(pose), -1)
    else:
        pose, depth = read_lbo(args.lbo)
        pose = pose.reshape(-1, 9)
        depth = depth.reshape(len(pose), -1)
        conf = read_lbo_post(
            args.lbo_post or args.lbo.with_suffix(args.lbo.suffix + ".post"),
            len(pose), depth.shape[1],
        ).reshape(len(pose), -1)
    height, width = args.height, args.width
    if depth.shape[1] != height * width:
        raise RuntimeError(f"depth has {depth.shape[1]} pixels/frame, not {height}x{width}")
    return pose, depth.reshape(-1, height, width), conf.reshape(-1, height, width)


def build_cloud(pose, depth, conf, rgb, conf_threshold, downsample):
    """Unproject every frame with the official utilities and fuse."""
    frames, height, width = depth.shape
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        torch.from_numpy(pose)[None].float(), (height, width)
    )
    world = unproject_depth_map_to_point_map(
        depth[..., None], invert_se3(extrinsic[0].numpy()), intrinsic[0].numpy()
    )

    chunks, kept, total = [], 0, 0
    for i in range(frames):
        mask = (
            np.isfinite(world[i]).all(axis=-1)
            & (depth[i] > 1e-6)
            & (conf[i] > conf_threshold)
        )
        if downsample > 1:
            keep = np.zeros(mask.size, dtype=bool)
            keep[::downsample] = True
            mask &= keep.reshape(mask.shape)
        total += mask.size
        kept += int(mask.sum())
        if not mask.any():
            continue
        points = world[i][mask]
        colors = rgb[i][mask]
        out = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                          ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        out["x"], out["y"], out["z"] = points[:, 0], points[:, 1], points[:, 2]
        out["red"], out["green"], out["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
        chunks.append(out)
    if not chunks:
        raise RuntimeError("every pixel was filtered out; lower --conf-threshold")
    return np.concatenate(chunks), kept / total


def write_ply(path: Path, cloud):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(cloud)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        cloud.tofile(f)


def render(path: Path, cloud, title, max_points=300000):
    ids = np.linspace(0, len(cloud) - 1, min(max_points, len(cloud)), dtype=np.int64)
    colors = np.stack((cloud["red"][ids], cloud["green"][ids], cloud["blue"][ids]), axis=1) / 255.0
    xyz = np.stack((cloud["x"][ids], cloud["y"][ids], cloud["z"][ids]), axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
    for ax, (i, j, xl, yl) in zip(axes, ((0, 2, "x", "z"), (0, 1, "x", "y"))):
        ax.scatter(xyz[:, i], xyz[:, j], c=colors, s=0.15, linewidths=0, rasterized=True)
        ax.set_xlabel(f"world {xl}")
        ax.set_ylabel(f"world {yl}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", linewidth=0.4)
    # Percentile limits keep a few far-field outliers from collapsing the view.
    for ax, (i, j) in zip(axes, ((0, 2), (0, 1))):
        for setter, axis in ((ax.set_xlim, i), (ax.set_ylim, j)):
            lo, hi = np.percentile(xyz[:, axis], (1, 99))
            pad = 0.05 * max(hi - lo, 1e-3)
            setter(lo - pad, hi + pad)
    axes[1].invert_yaxis()
    axes[0].set_title(f"{title} — top-down (x-z)")
    axes[1].set_title("front (x-y)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz", type=Path, help="PyTorch reference npz")
    src.add_argument("--lbo", type=Path, help="C++ CLI LBO1 output")
    ap.add_argument("--lbo-post", type=Path, help="LBP2 sidecar (default: <lbo>.post)")
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--title", default="reconstruction")
    ap.add_argument("--conf-threshold", type=float, default=1.5,
                    help="demo.py --conf_threshold default")
    ap.add_argument("--downsample", type=int, default=4,
                    help="keep every Nth pixel (demo.py uses 10 for interactive display)")
    args = ap.parse_args()

    pose, depth, conf = load_run(args)
    frames = len(pose)
    files = sorted(args.scene.glob("*.png"))[:frames]
    if len(files) != frames:
        raise RuntimeError(f"scene has {len(files)} frames, run has {frames}")
    rgb = np.stack([
        np.asarray(Image.open(p).convert("RGB").resize((args.width, args.height)))
        for p in files
    ])

    cloud, ratio = build_cloud(pose, depth, conf, rgb, args.conf_threshold, args.downsample)
    ply = args.out_prefix.with_name(args.out_prefix.name + ".ply")
    png = args.out_prefix.with_name(args.out_prefix.name + ".png")
    write_ply(ply, cloud)
    render(png, cloud, args.title)
    print(f"{args.title}: {frames}x{args.height}x{args.width} "
          f"kept={100 * ratio:.2f}% conf_mean={conf.mean():.3f} "
          f"points={len(cloud)} -> {ply.name}, {png.name}")


if __name__ == "__main__":
    main()
