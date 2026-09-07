#!/usr/bin/env python3
"""Compare a C++ LBO1 dump with a PyTorch reference dump saved as .npz."""
import argparse, struct
import numpy as np

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cpp_bin"); ap.add_argument("torch_npz"); ap.add_argument("--atol", type=float, default=1e-4); ap.add_argument("--rtol", type=float, default=1e-3); args = ap.parse_args()
    raw = open(args.cpp_bin, "rb").read(12); magic, n_pose, n_depth = struct.unpack("<III", raw)
    if magic != 0x4c424f31: raise SystemExit("invalid C++ dump magic")
    vals = np.fromfile(args.cpp_bin, dtype=np.float32, offset=12)
    if vals.size != n_pose + n_depth: raise SystemExit("truncated C++ dump")
    ref = np.load(args.torch_npz)
    for name, n, off in (("pose_enc", n_pose, 0), ("depth", n_depth, n_pose)):
        if name not in ref: raise SystemExit(f"reference missing {name}")
        a, b = vals[off:off+n].reshape(-1), np.asarray(ref[name], dtype=np.float32).reshape(-1)
        if a.shape != b.shape: raise SystemExit(f"{name} shape mismatch: cpp={a.shape} torch={b.shape}")
        err = float(np.max(np.abs(a-b))) if a.size else 0.0
        if not np.allclose(a, b, atol=args.atol, rtol=args.rtol): raise SystemExit(f"{name} parity failed: max_abs={err}")
        print(f"{name}: max_abs={err:.6g}")
    print("PARITY PASS")

if __name__ == "__main__": main()
