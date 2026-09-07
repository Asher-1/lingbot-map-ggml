#!/usr/bin/env python3
"""Measure multi-view geometric self-consistency of a reconstruction run.

Depth + pose are self-consistent when frame j's unprojected points, projected
back into frame i's camera, land on frame i's own depth values. This gives an
objective quality number that does not depend on how a point cloud is rendered,
so it can separate "the reconstruction is wrong" from "the plot looks noisy".

`--extrinsic-convention` selects which transform is fed to the unprojection:

  w2c  — `pose_encoding_to_extri_intri` output used directly, i.e. what
         `unproject_depth_map_to_point_map` documents as its input.
  c2w  — the transform after `demo.py:postprocess` inverts it. The viewer then
         inverts a second time, so this reproduces the shipped demo's geometry.
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3


def read_lbo(path: Path):
    raw = path.read_bytes()
    magic, n_pose, n_depth = struct.unpack("<III", raw[:12])
    if magic != 0x4C424F31:
        raise RuntimeError(f"invalid LBO1 output: {path}")
    values = np.frombuffer(raw, dtype=np.float32, offset=12)
    return values[:n_pose].reshape(-1, 9), values[n_pose:n_pose + n_depth]


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz", type=Path)
    src.add_argument("--lbo", type=Path)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--extrinsic-convention", choices=("w2c", "c2w"), default="w2c")
    ap.add_argument("--gap", type=int, default=5, help="frame offset for each pair")
    ap.add_argument("--stride", type=int, default=8, help="pixel stride when sampling frame j")
    ap.add_argument("--tolerance", type=float, default=0.02, help="relative depth agreement")
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    if args.npz:
        data = np.load(args.npz)
        pose = np.asarray(data["pose_enc"], dtype=np.float32).reshape(-1, 9)
        depth = np.asarray(data["depth"], dtype=np.float32).reshape(len(pose), args.height, args.width)
    else:
        pose, flat = read_lbo(args.lbo)
        depth = flat.reshape(len(pose), args.height, args.width)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        torch.from_numpy(pose)[None].float(), (args.height, args.width)
    )
    w2c = extrinsic[0].numpy().astype(np.float64)
    intrinsic = intrinsic[0].numpy().astype(np.float64)
    if args.extrinsic_convention == "c2w":
        square = np.zeros((len(w2c), 4, 4)); square[:, :3, :4] = w2c; square[:, 3, 3] = 1.0
        w2c = closed_form_inverse_se3(square)[:, :3, :4]

    frames = len(pose)
    ys, xs = np.mgrid[0:args.height:args.stride, 0:args.width:args.stride]
    ys, xs = ys.ravel(), xs.ravel()
    agree, rel_errors = [], []
    for j in range(args.gap, frames):
        i = j - args.gap
        d = depth[j, ys, xs].astype(np.float64)
        keep = d > 1e-6
        if not keep.any():
            continue
        # frame j: pixel -> camera -> world (inverse of its own w2c)
        kj = intrinsic[j]
        cam = np.stack(((xs[keep] - kj[0, 2]) * d[keep] / kj[0, 0],
                        (ys[keep] - kj[1, 2]) * d[keep] / kj[1, 1], d[keep]), axis=1)
        rj, tj = w2c[j, :3, :3], w2c[j, :3, 3]
        world = (cam - tj) @ rj
        # frame i: world -> camera -> pixel
        ri, ti = w2c[i, :3, :3], w2c[i, :3, 3]
        cam_i = world @ ri.T + ti
        z = cam_i[:, 2]
        front = z > 1e-6
        ki = intrinsic[i]
        u = np.rint(cam_i[front, 0] / z[front] * ki[0, 0] + ki[0, 2]).astype(np.int64)
        v = np.rint(cam_i[front, 1] / z[front] * ki[1, 1] + ki[1, 2]).astype(np.int64)
        inside = (u >= 0) & (u < args.width) & (v >= 0) & (v < args.height)
        if not inside.any():
            continue
        z_pred = z[front][inside]
        z_ref = depth[i, v[inside], u[inside]].astype(np.float64)
        valid = z_ref > 1e-6
        if not valid.any():
            continue
        rel = np.abs(z_pred[valid] - z_ref[valid]) / z_ref[valid]
        agree.append(float(np.mean(rel < args.tolerance)))
        rel_errors.append(float(np.median(rel)))

    print(f"{args.label}: {args.extrinsic_convention} {frames}x{args.height}x{args.width} "
          f"gap={args.gap} agree<{args.tolerance:.0%}={100 * np.mean(agree):.1f}% "
          f"median_rel_err={np.mean(rel_errors):.4f}")


if __name__ == "__main__":
    main()
