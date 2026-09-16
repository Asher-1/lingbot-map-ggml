#!/usr/bin/env python3
"""P3 attribution: F32-cache (`--kv-f16 none`) long-stream rows @ indoor 2000f.

Question: is the strict-mode pose accumulation (5.90e-04 vs the F32-cache
mirror) caused by the F16 cache scatter, or is it the engine's own noise?
The none rows run the same f16 GGUF with an exact F32 cache, so:

  none vs PT mf16   -> pure engine parity under the F32-cache contract
                       (decisive: compare against strict-vs-mf16 5.90e-04)
  none vs PT fp32   -> total deviation (f16 weight cost + engine, F32 cache)
  PT mf16 vs fp32   -> the mirror's own f16 weight cost @ indoor (baseline)
  none cuda vs vulkan -> cross-backend engine consistency (no PT involved)

Depth metrics use the relative view; sampled with stride 2 to match the
campaign's evaluate.py --frames-sample 2 convention for the 2000f sets.

Usage: attribute_p3.py [--csv PATH]
"""
import argparse
import gc
import struct
from pathlib import Path

import numpy as np

RUNS = Path("/tmp/long_real/runs/indoor")
H, W = 294, 518
TAIL_ABS = 1.5e-3
STRIDE = 2  # matches evaluate.py --frames-sample 2 for the 2000f datasets


def read_lbo(path):
    raw = Path(path).read_bytes()
    _, n_pose, n_depth = struct.unpack("<III", raw[:12])
    v = np.frombuffer(raw, dtype=np.float32, offset=12)
    return (v[:n_pose].reshape(-1, 9).astype(np.float32),
            v[n_pose:n_pose + n_depth].reshape(-1, H, W).astype(np.float32))


def read_npz(path):
    z = np.load(path)
    return (z["pose_enc"].reshape(-1, 9).astype(np.float32),
            z["depth"].reshape(-1, H, W).astype(np.float32))


def stats(err_p, err_d, ref_depth):
    pfr = np.sqrt((err_p ** 2).mean(axis=1))
    return {
        "pose_rmse": float(np.sqrt((err_p ** 2).mean())),
        "pose_first200": float(pfr[:200].mean()),
        "pose_last500": float(pfr[-500:].mean()),
        "depth_rmse": float(np.sqrt((err_d ** 2).mean())),
        "depth_rel": float(np.sqrt((err_d ** 2).mean()) / max(float(np.abs(ref_depth).mean()), 1e-9)),
        "depth_tail_pct": float(100.0 * (err_d > TAIL_ABS).mean()),
    }


def compare(name, a, b, rows):
    pa, da = a
    pb, db = b
    n = min(len(pa), len(pb))
    err_p = np.abs(pa[:n] - pb[:n])
    err_d = np.abs(da[:n] - db[:n])
    if STRIDE > 1:  # memory guard, mirrors evaluate.py sampling
        idx = np.arange(0, n, STRIDE)
        err_p, err_d, rd = err_p[idx], err_d[idx], db[:n][idx]
    else:
        rd = db[:n]
    rows.append({"comparison": name, "frames": n, **stats(err_p, err_d, rd)})
    del err_p, err_d
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    poses, depths = {}, {}

    def load(tag):
        if tag not in depths:
            src = RUNS / f"{tag}.npz"
            if src.exists():
                poses[tag], depths[tag] = read_npz(src)
            else:
                poses[tag], depths[tag] = read_lbo(RUNS / f"{tag}.lbo")
        return poses[tag], depths[tag]

    rows = []
    # decisive: engine parity under the exact-F32-cache contract
    compare("P3 GGML none cuda vs PT mf16 (engine, F32-cache contract)",
            load("ggml_long_f16_none_cuda"), load("pt_long_mf16"), rows)
    compare("P3 GGML none vulkan vs PT mf16 (engine, F32-cache contract)",
            load("ggml_long_f16_none_vulkan"), load("pt_long_mf16"), rows)
    del depths["pt_long_mf16"]
    gc.collect()
    # totals vs the pristine checkpoint
    compare("P3 GGML none cuda vs checkpoint fp32 (total)",
            load("ggml_long_f16_none_cuda"), load("pt_long_fp32"), rows)
    del depths["ggml_long_f16_none_cuda"]
    gc.collect()
    compare("P3 GGML none vulkan vs checkpoint fp32 (total)",
            load("ggml_long_f16_none_vulkan"), load("pt_long_fp32"), rows)
    del depths["ggml_long_f16_none_vulkan"]
    gc.collect()
    # mirror's own f16 weight cost @ indoor (baseline for the totals)
    compare("P3 PT mf16 vs checkpoint fp32 (f16 weight cost, mirror)",
            load("pt_long_mf16"), load("pt_long_fp32"), rows)
    del depths["pt_long_mf16"], depths["pt_long_fp32"]
    gc.collect()
    # cross-backend engine consistency (no PT involved)
    compare("P3 GGML none cuda vs none vulkan (cross-backend)",
            load("ggml_long_f16_none_cuda"), load("ggml_long_f16_none_vulkan"), rows)

    hdr = (f"{'comparison':52s} {'n':>5s} {'pose':>9s} {'f200':>9s} {'last500':>9s} "
           f"{'d_rmse':>9s} {'d_rel':>9s} {'tail%':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['comparison']:52s} {r['frames']:5d} {r['pose_rmse']:9.2e} "
              f"{r['pose_first200']:9.2e} {r['pose_last500']:9.2e} "
              f"{r['depth_rmse']:9.2e} {r['depth_rel']:9.2e} {r['depth_tail_pct']:7.2f}")

    if args.csv:
        import csv
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
