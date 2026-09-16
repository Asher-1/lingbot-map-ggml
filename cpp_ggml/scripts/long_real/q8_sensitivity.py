#!/usr/bin/env python3
"""Per-tensor q8 quantization-error ranking for the long checkpoint.

Reads the f32 and q8 GGUFs, dequantizes, and ranks every tensor by relative
weight error ||w_q8 - w_f32|| / ||w_f32||. The top tensors name the mixed-
quantization candidates (--f32-prefix / --quantize-prefix in
convert_lingbot.py): q8 everywhere except the sensitive prefixes.

Usage: q8_sensitivity.py [--top 30]
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("gguf python package required (pip install gguf)")

ROOT = Path(__file__).resolve().parents[3]
F32 = ROOT / "cpp_ggml/models/gguf/lingbot-map-long-f32.gguf"
Q8 = ROOT / "cpp_ggml/models/gguf/lingbot-map-long-q8.gguf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    rf, rq = gguf.GGUFReader(F32), gguf.GGUFReader(Q8)
    f32 = {}
    for t in rf.tensors:
        if t.tensor_type == gguf.GGMLQuantizationType.F32:
            f32[t.name] = t.data.view(np.float32).reshape(tuple(int(x) for x in reversed(t.shape)))
        else:
            f32[t.name] = None

    rows = []
    for t in rq.tensors:
        if t.name not in f32 or f32[t.name] is None:
            continue
        w8 = gguf.dequantize(t.data, t.tensor_type)
        if w8 is None:
            w8 = t.data.view(np.float32)
        wf = f32[t.name]
        if wf.shape != w8.shape:
            continue
        denom = float(np.linalg.norm(wf))
        if denom == 0:
            continue
        err = float(np.linalg.norm(w8.reshape(wf.shape).astype(np.float32) - wf)) / denom
        rows.append((err, t.name, str(t.tensor_type)))

    rows.sort(reverse=True)
    print(f"{'rel_err':>9}  type  tensor")
    for err, name, tt in rows[: args.top]:
        print(f"{err:9.5f}  {tt:14s} {name}")

    # pattern rollup: mean error by name prefix family
    fams = {}
    for err, name, _ in rows:
        parts = name.split(".")
        fam = ".".join(parts[:2]) if name.startswith("aggregator") else ".".join(parts[:2])
        fams.setdefault(fam, []).append(err)
    print("\nBy family (mean / max rel err, n):")
    for fam, errs in sorted(fams.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  {fam:44s} mean {np.mean(errs):.5f}  max {np.max(errs):.5f}  n={len(errs)}")


if __name__ == "__main__":
    main()
