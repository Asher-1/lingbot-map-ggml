#!/usr/bin/env python3
"""Verify the C++ LBP pose sidecar against the official utility."""
import argparse
import struct
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri


def read_lbo(path: Path):
    raw = path.read_bytes()
    magic, pose_count, depth_count = struct.unpack("<III", raw[:12])
    if magic != 0x4C424F31:
        raise ValueError(f"{path} is not LBO1")
    values = np.frombuffer(raw, dtype=np.float32, offset=12)
    if values.size != pose_count + depth_count or pose_count % 9:
        raise ValueError(f"{path} has an invalid LBO1 payload")
    return values[:pose_count].reshape(1, pose_count // 9, 9)


def read_post(path: Path, frames: int):
    raw = path.read_bytes()
    magic, c2w_count, intr_count = struct.unpack("<III", raw[:12])
    if magic == 0x4C425031:
        values = np.frombuffer(raw, dtype=np.float32, offset=12)
        if c2w_count != frames * 16 or intr_count != frames * 4 or values.size != c2w_count + intr_count:
            raise ValueError(f"{path} has an invalid LBP1 payload")
        return values[:c2w_count].reshape(frames, 4, 4), values[c2w_count:].reshape(frames, 4)
    if magic != 0x4C425032 or len(raw) < 16:
        raise ValueError(f"{path} is not LBP1 or LBP2")
    confidence_count, = struct.unpack("<I", raw[12:16])
    values = np.frombuffer(raw, dtype=np.float32, offset=16)
    if c2w_count != frames * 16 or intr_count != frames * 4 or values.size != c2w_count + intr_count + confidence_count:
        raise ValueError(f"{path} has an invalid LBP2 payload")
    return values[:c2w_count].reshape(frames, 4, 4), values[c2w_count:c2w_count + intr_count].reshape(frames, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lbo", type=Path)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--atol", type=float, default=2e-6)
    args = parser.parse_args()

    pose = read_lbo(args.lbo).copy()
    c2w, intrinsics = read_post(Path(str(args.lbo) + ".post"), pose.shape[1])
    official_w2c, official_k = pose_encoding_to_extri_intri(
        torch.from_numpy(pose), (args.height, args.width)
    )
    # pose_encoding_to_extri_intri returns the camera-from-world [R|t]; the
    # C++ sidecar (and demo.py's postprocess) stores camera-to-world, so take
    # the SE(3) inverse before comparing.
    w2c_4x4 = np.tile(np.eye(4, dtype=np.float32), (pose.shape[1], 1, 1))
    w2c_4x4[:, :3, :] = official_w2c.numpy()[0]
    expected_c2w = np.linalg.inv(w2c_4x4)
    expected_intrinsics = np.stack((
        official_k.numpy()[0, :, 0, 0],
        official_k.numpy()[0, :, 1, 1],
        official_k.numpy()[0, :, 0, 2],
        official_k.numpy()[0, :, 1, 2],
    ), axis=1)
    c2w_error = float(np.max(np.abs(c2w - expected_c2w)))
    intr_error = float(np.max(np.abs(intrinsics - expected_intrinsics)))
    print(f"c2w: max_abs={c2w_error:.6g}")
    print(f"intrinsics: max_abs={intr_error:.6g}")
    if not np.allclose(c2w, expected_c2w, atol=args.atol, rtol=args.atol):
        raise SystemExit("C++ C2W postprocess parity failed")
    if not np.allclose(intrinsics, expected_intrinsics, atol=args.atol, rtol=args.atol):
        raise SystemExit("C++ intrinsics postprocess parity failed")
    print("POSTPROCESS PARITY PASS")


if __name__ == "__main__":
    main()
