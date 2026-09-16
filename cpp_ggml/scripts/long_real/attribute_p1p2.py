#!/usr/bin/env python3
"""P1/P2 attribution on the drive 1050f campaign rows.

P1 — flash pose attribution: isolate the F16 K/V cache-contract cost inside
PyTorch (flashcache vs mf16, f16 weights constant on both sides) and set the
GGML flash totals against it. Answers: is the flash pose excess on long
streams the cache contract itself (reproduced by PyTorch, upstream weight/
contract property) or engine-added noise?

P2 — q8 mixed-quant attribution: the q8mix GGUF (q8 aggregator+depth_head,
f16 camera_head) weight cost vs checkpoint fp32 and vs the all-f16 mirror,
plus C++ engine parity vs its own mirror. Answers: does keeping the camera
head in f16 remove the AdaLN pose amplification of q8, and does the engine
add anything on top?

All references: /tmp/long_real/runs/drive/ (npz mirrors, lbo engine rows).
Depth metrics use the relative view (RMSE / mean|depth| of the reference);
absolute gates are depth-scale-weighted (see validation_report.md pitfall).

Usage: attribute_p1p2.py [--csv PATH]
"""
import argparse
import gc
import struct
from pathlib import Path

import numpy as np

RUNS = Path("/tmp/long_real/runs/drive")
H, W = 294, 518
TAIL_ABS = 1.5e-3


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
    n = len(err_p)
    pfr = np.sqrt((err_p ** 2).mean(axis=1))
    return {
        "pose_rmse": float(np.sqrt((err_p ** 2).mean())),
        "pose_first50": float(pfr[:50].mean()),
        "pose_lastQ": float(pfr[-n // 4:].mean()),
        "depth_rmse": float(np.sqrt((err_d ** 2).mean())),
        "depth_rel": float(np.sqrt((err_d ** 2).mean()) / max(float(np.abs(ref_depth).mean()), 1e-9)),
        "depth_tail_pct": float(100.0 * (err_d > TAIL_ABS).mean()),
    }


def compare(name, a, b, rows):
    """a, b: (pose, depth); deviation of a relative to b."""
    pa, da = a
    pb, db = b
    n = min(len(pa), len(pb))
    err_p = np.abs(pa[:n] - pb[:n])
    err_d = np.abs(da[:n] - db[:n])
    rows.append({"comparison": name, "frames": n, **stats(err_p, err_d, db[:n])})
    del err_p, err_d
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    poses = {}   # tiny, keep resident
    depths = {}  # 632MB each, load on demand

    def load(tag):
        if tag not in depths:
            src = RUNS / f"{tag}.npz"
            if src.exists():
                poses[tag], depths[tag] = read_npz(src)
            else:
                poses[tag], depths[tag] = read_lbo(RUNS / f"{tag}.lbo")
        return poses[tag], depths[tag]

    rows = []

    # ---- P1: cache-contract isolation inside PyTorch -----------------------
    # weights constant (f16) on both sides; only the K/V cache precision
    # entering attention differs (F32 cache vs every-K/V-F16).
    compare("P1 PT flashcache vs PT mf16 (F16-KV contract, weights const)",
            load("pt_long_mf16_flashcache"), load("pt_long_mf16"), rows)
    del depths["pt_long_mf16_flashcache"], depths["pt_long_mf16"]
    gc.collect()

    # cross-checks vs the pristine fp32 checkpoint (mirror the p1ref log)
    compare("P1 PT mf16 vs checkpoint fp32 (f16 weight cost)",
            load("pt_long_mf16"), load("pt_long_fp32"), rows)
    del depths["pt_long_mf16"]
    gc.collect()
    compare("P1 PT flashcache vs checkpoint fp32 (f16w + F16-KV total)",
            load("pt_long_mf16_flashcache"), load("pt_long_fp32"), rows)
    del depths["pt_long_mf16_flashcache"]
    gc.collect()

    # ---- P2: q8mix weight cost + engine parity -----------------------------
    # weight-level: q8mix mirror vs all-f16 mirror isolates the q8 rounding
    # of aggregator+depth_head (camera_head is f16 in both).
    compare("P2 q8mix mirror vs f16 mirror (agg+depth q8 rounding)",
            load("pt_long_q8mix_mf"), load("pt_long_mf16"), rows)
    del depths["pt_long_mf16"]
    gc.collect()
    compare("P2 q8mix mirror vs checkpoint fp32 (q8mix weight cost)",
            load("pt_long_q8mix_mf"), load("pt_long_fp32"), rows)
    del depths["pt_long_fp32"]
    gc.collect()
    # engine parity under the q8mix weights
    compare("P2 GGML q8mix strict vs q8mix mirror (engine parity)",
            load("ggml_long_q8mix_strict_cuda"), load("pt_long_q8mix_mf"), rows)
    compare("P2 GGML q8mix flash vs q8mix mirror (engine parity)",
            load("ggml_long_q8mix_flash_cuda"), load("pt_long_q8mix_mf"), rows)
    del depths["pt_long_q8mix_mf"]
    gc.collect()
    # total rows for the report matrix (vs pristine checkpoint)
    compare("P2 GGML q8mix strict vs checkpoint fp32 (total)",
            load("ggml_long_q8mix_strict_cuda"), load("pt_long_fp32"), rows)
    compare("P2 GGML q8mix flash vs checkpoint fp32 (total)",
            load("ggml_long_q8mix_flash_cuda"), load("pt_long_fp32"), rows)

    # ---- report ------------------------------------------------------------
    hdr = (f"{'comparison':58s} {'n':>5s} {'pose':>9s} {'f50':>9s} {'lastQ':>9s} "
           f"{'d_rmse':>9s} {'d_rel':>9s} {'tail%':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['comparison']:58s} {r['frames']:5d} {r['pose_rmse']:9.2e} "
              f"{r['pose_first50']:9.2e} {r['pose_lastQ']:9.2e} "
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
