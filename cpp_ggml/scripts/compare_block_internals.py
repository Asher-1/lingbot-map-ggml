#!/usr/bin/env python3
"""Compare one transformer block's residual boundaries between GGML and PyTorch."""
import argparse
from pathlib import Path

import numpy as np


BOUNDARIES = ("norm1", "qkv", "attn", "ls1", "attn_residual", "norm2", "fc1", "gelu", "fc2", "ls2")
PREFIXES = {
    "dino": "aggregator.patch_embed.blocks",
    "frame": "aggregator.frame_blocks",
    "global": "aggregator.global_blocks",
    "camera": "camera_head.trunk",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cpp_prefix", type=Path,
                        help="LINGBOT_DUMP_STAGES prefix")
    parser.add_argument("torch_npz", type=Path)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--family", choices=sorted(PREFIXES), default="dino")
    args = parser.parse_args()
    ref = np.load(args.torch_npz)
    prefix = f"{PREFIXES[args.family]}.{args.block}"
    print("boundary\tshape\tmae\trmse\tp99_abs\tmax_abs\tcorr")
    for boundary in BOUNDARIES:
        cpp_boundary = "attn.proj" if boundary == "attn" else boundary
        if boundary == "qkv":
            cpp_boundary = "attn.qkv"
        if boundary in ("fc1", "gelu", "fc2"):
            cpp_boundary = f"mlp.{boundary}"
        cpp = args.cpp_prefix.parent / (
            f"{args.cpp_prefix.name}.{prefix}.{cpp_boundary}.{args.frame}")
        if not cpp.is_file():
            # CPU single-frame dumps predate indexed filenames.
            cpp = args.cpp_prefix.parent / f"{args.cpp_prefix.name}.{prefix}.{cpp_boundary}"
        torch_key = f"internal_{args.family}_{args.block}_{boundary}"
        if not cpp.is_file() or torch_key not in ref:
            raise SystemExit(f"missing {boundary}: cpp={cpp} torch={torch_key}")
        a = np.fromfile(cpp, dtype=np.float32)
        b = np.asarray(ref[torch_key], dtype=np.float32).reshape(-1)
        if a.size != b.size:
            raise SystemExit(f"{boundary}: count mismatch cpp={a.size} torch={b.size}")
        d = np.abs(a - b)
        corr = np.corrcoef(a, b)[0, 1] if np.std(a) and np.std(b) else float("nan")
        print(f"{boundary}\t{a.size}\t{d.mean():.8g}\t{np.sqrt(np.mean((a-b)**2)):.8g}\t"
              f"{np.quantile(d, .99):.8g}\t{d.max():.8g}\t{corr:.8g}")


if __name__ == "__main__":
    main()
