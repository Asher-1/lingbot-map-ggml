#!/usr/bin/env python3
"""Evaluation + charts for the long_real comparison campaign (Stage 4).

Reads /tmp/long_real/runs/<ds>/ rows (metrics.json + LBO/.post + mirror npz)
and produces, per dataset, in cpp_ggml/benchmarks/long_real/<ds>/:
  alignment.csv   — per GGML row vs same-format mirror (engine parity) and vs
                    checkpoint fp32 (total deviation): pose rmse/max, depth
                    rmse/REL/max/tail% (relative view always paired, the
                    absolute gates are depth-scale-weighted)
  resources.csv   — wall, FPS, GPU peak, per-process GPU peak, RSS peaks, util
  parity_curves.png — per-frame pose/depth RMSE vs frame index (long-stream
                    stability evidence), keyframe ticks at the auto interval
  speed.png / memory.png — wall and VRAM bars across all rows
  behavior.csv    — long-vs-balanced behavior on the dataset (mean|depth|,
                    depth p95, trajectory length; no-GT honesty: consistency,
                    speed and memory dimensions only)
  cloud_eval.csv + cloud_effect.png — unprojected cloud agreement (conf>1.5
                    official rule, sampled frames) + 2-view renders

Global: long_real_report.md across the three datasets.

Usage: evaluate.py DATASET [--frames-sample 100] [--skip-cloud]
"""
import argparse
import json
import math
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path("/tmp/long_real/runs")
OUTROOT = Path(__file__).resolve().parents[2] / "benchmarks" / "long_real"
TAIL_ABS = 1.5e-3


def read_lbo(path: Path, h: int, w: int):
    raw = path.read_bytes()
    _, n_pose, n_depth = struct.unpack("<III", raw[:12])
    v = np.frombuffer(raw, dtype=np.float32, offset=12)
    pose = v[:n_pose].reshape(-1, 9)
    depth = v[n_pose:n_pose + n_depth].reshape(-1, h, w)
    post = Path(str(path) + ".post")
    conf = c2w = intr = None
    if post.exists():
        raw = post.read_bytes()
        _, n_c2w, n_intr, n_conf = struct.unpack("<IIII", raw[:16])
        p = np.frombuffer(raw, dtype=np.float32, offset=16)
        c2w = p[:n_c2w].reshape(-1, 4, 4)
        intr = p[n_c2w:n_c2w + n_intr].reshape(-1, 4)
        conf = p[n_c2w + n_intr:n_c2w + n_intr + n_conf].reshape(-1, h, w)
    return {"pose": pose, "depth": depth, "conf": conf, "c2w": c2w, "intr": intr}


def read_npz(path: Path, h: int, w: int):
    z = np.load(path)
    return {"pose": z["pose_enc"].reshape(-1, 9).astype(np.float32),
            "depth": z["depth"].reshape(-1, h, w).astype(np.float32),
            "conf": z["depth_conf"].reshape(-1, h, w).astype(np.float32),
            "c2w": None, "intr": None}


def depth_stats(err, ref_depth):
    rel = math.sqrt(float((err ** 2).mean())) / max(float(np.abs(ref_depth).mean()), 1e-9)
    return {
        "depth_rmse": float(np.sqrt((err ** 2).mean())),
        "depth_rel": rel,
        "depth_max": float(err.max()),
        "depth_tail_pct": float(100.0 * (err > TAIL_ABS).mean()),
    }


def compare(row, ref, sample_stride=1):
    """Elementwise alignment stats; slice to the shorter length."""
    n = min(len(row["pose"]), len(ref["pose"]))
    perr = np.abs(row["pose"][:n] - ref["pose"][:n])
    derr = np.abs(row["depth"][:n] - ref["depth"][:n])
    if sample_stride > 1:  # memory guard for the 2000-frame datasets
        idx = np.arange(0, n, sample_stride)
        perr, derr = perr[idx], derr[idx]
        rd = ref["depth"][:n][idx]
    else:
        rd = ref["depth"][:n]
    stats = {
        "frames": n,
        "pose_rmse": float(np.sqrt((perr ** 2).mean())),
        "pose_max": float(perr.max()),
    }
    stats.update(depth_stats(derr, rd))
    return stats


def per_frame_rmse(row, ref):
    n = min(len(row["pose"]), len(ref["pose"]))
    perr = np.sqrt(((row["pose"][:n] - ref["pose"][:n]) ** 2).mean(axis=1))
    derr = np.sqrt(((row["depth"][:n] - ref["depth"][:n]) ** 2).reshape(n, -1).mean(axis=1))
    return perr, derr


def camera_centers(pose):
    """Official decoder image: w2c = [quat_mat | t] -> center = -R^T t."""
    n = len(pose)
    q = pose[:, 3:7]  # scalar-last xyzw
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    t = pose[:, :3]
    return np.einsum("nij,nj->ni", R.transpose(0, 2, 1), -t)


def load_metrics(path: Path):
    if not path.exists():
        return None
    m = json.loads(path.read_text())
    if m.get("exit_code") != 0:
        return None
    return m


def quat_to_mat(q):
    """scalar-last xyzw -> R, numpy port of the official quat_to_mat."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def decode_c2w_intr(pose, h, w):
    """Official decoder image in numpy: w2c=[R|t] (verified elementwise),
    fx=(W/2)/tan(fov_w/2), fy=(H/2)/tan(fov_h/2); returns c2w + intrinsics."""
    R = quat_to_mat(pose[:, 3:7])
    t = pose[:, :3]
    w2c = np.zeros((len(pose), 4, 4))
    w2c[:, :3, :3] = R
    w2c[:, :3, 3] = t
    w2c[:, 3, 3] = 1.0
    inv = np.zeros_like(w2c)
    RT = R.transpose(0, 2, 1)
    inv[:, :3, :3] = RT
    inv[:, :3, 3] = np.einsum("nij,nj->ni", RT, -t)
    inv[:, 3, 3] = 1.0
    fy = (h / 2.0) / np.tan(pose[:, 7] / 2.0)
    fx = (w / 2.0) / np.tan(pose[:, 8] / 2.0)
    intr = np.stack([fx, fy, np.full(len(pose), w / 2.0), np.full(len(pose), h / 2.0)], axis=1)
    return inv, intr


def unproject_cloud(c2w, intr, depth, conf, conf_thresh=1.5, px_stride=4):
    """Official viewer rule (conf > 1.5) with pinhole unprojection."""
    fx, fy, cx, cy = intr
    hh, ww = depth.shape
    v, u = np.mgrid[0:hh:px_stride, 0:ww:px_stride]
    d = depth[::px_stride, ::px_stride]
    c = conf[::px_stride, ::px_stride]
    keep = c > conf_thresh
    if not keep.any():
        return np.zeros((0, 3))
    pc = np.stack([(u[keep] - cx) / fx * d[keep],
                   (v[keep] - cy) / fy * d[keep],
                   d[keep]], axis=1)
    return pc @ c2w[:3, :3].T + c2w[:3, 3]


def cloud_nn_rmse(a, b, max_points=150_000, seed=0):
    """Symmetric nearest-neighbour RMSE between two clouds (both directions)."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    if len(a) > max_points:
        a = a[rng.choice(len(a), max_points, replace=False)]
    if len(b) > max_points:
        b = b[rng.choice(len(b), max_points, replace=False)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    da, _ = cKDTree(b).query(a, k=1)
    db, _ = cKDTree(a).query(b, k=1)
    return float(np.sqrt(0.5 * ((da ** 2).mean() + (db ** 2).mean())))


def render_cloud(ax, pts, title, view=(0, 2), max_pts=60_000, seed=0):
    rng = np.random.default_rng(seed)
    if len(pts) > max_pts:
        pts = pts[rng.choice(len(pts), max_pts, replace=False)]
    if len(pts) == 0:
        ax.set_title(title + " (empty)")
        return
    ax.scatter(pts[:, view[0]], pts[:, view[1]], s=0.3, c=-pts[:, 1], cmap="viridis_r", linewidths=0)
    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)


def cloud_eval(ds, meta, run_dir, out_dir, frame_stride=10, px_stride=4):
    """Cloud agreement (NN RMSE) + 2-view renders for the headline rows."""
    n, h, w = meta["n_frames"], meta["height"], meta["width"]
    frames = list(range(0, n, frame_stride))
    targets = ["pt_long_fp32", "pt_bal_fp32", "ggml_long_f16_flash_cuda", "ggml_bal_f16_flash_cuda"]
    clouds = {}
    for tag in targets:
        if (run_dir / f"{tag}.npz").exists():
            row = read_npz(run_dir / f"{tag}.npz", h, w)
            c2w_all, intr_all = decode_c2w_intr(row["pose"], h, w)
            src = "npz"
        elif (run_dir / f"{tag}.lbo").exists():
            row = read_lbo(run_dir / f"{tag}.lbo", h, w)
            if row["c2w"] is None:
                print(f"cloud: {tag} has no .post sidecar, skipped")
                continue
            c2w_all, intr_all = row["c2w"], row["intr"]
            src = "lbo"
        else:
            continue
        pts = []
        for fi in frames:
            pts.append(unproject_cloud(c2w_all[fi], intr_all[fi], row["depth"][fi],
                                       row["conf"][fi], px_stride=px_stride))
        clouds[tag] = np.concatenate(pts, axis=0)
        print(f"cloud[{tag}] src={src}: {len(clouds[tag])} points")

    rows = []
    names = list(clouds)
    if names:
        import csv
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                rmse = cloud_nn_rmse(clouds[a], clouds[b])
                rows.append({"cloud_a": a, "cloud_b": b, "nn_rmse": rmse})
                print(f"cloud NN rmse {a} <-> {b}: {rmse:.4f}")
        with open(out_dir / "cloud_eval.csv", "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=["cloud_a", "cloud_b", "nn_rmse"])
            wtr.writeheader()
            wtr.writerows(rows)
        # 2-view render panel
        k = len(names)
        fig, axes = plt.subplots(k, 2, figsize=(13, 4.6 * k), squeeze=False)
        for i, tag in enumerate(names):
            pts = clouds[tag]
            render_cloud(axes[i][0], pts, f"{tag} (top view x-z, {len(pts)} pts)", view=(0, 2))
            render_cloud(axes[i][1], pts, f"{tag} (front view z-y)", view=(2, 1))
        fig.suptitle(f"{ds}: unprojected clouds (conf>1.5, every {frame_stride} frames)")
        fig.savefig(out_dir / "cloud_effect.png", dpi=120)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--frames-sample", type=int, default=1,
                    help="elementwise sampling stride for the 2000-frame sets")
    ap.add_argument("--skip-cloud", action="store_true")
    args = ap.parse_args()
    ds = args.dataset
    meta = json.loads(Path(f"/tmp/long_real/{ds}.meta.json").read_text())
    n, h, w, kf = meta["n_frames"], meta["height"], meta["width"], meta["auto_keyframe_interval"]
    run_dir = RUNS / ds
    out_dir = OUTROOT / ds
    out_dir.mkdir(parents=True, exist_ok=True)

    refs = {}
    for ckpt in ("long", "bal"):
        for kind in ("fp32", "bf16", "mf16", "mq8"):
            p = run_dir / f"pt_{ckpt}_{kind}.npz"
            if p.exists():
                refs[f"{ckpt}_{kind}"] = read_npz(p, h, w)

    rows = sorted(run_dir.glob("ggml_*.lbo"))
    align_rows, res_rows = [], []
    curves = {}
    for lbo in rows:
        tag = lbo.stem
        m = load_metrics(lbo.with_suffix(".metrics.json"))
        if m is None:
            print(f"skip {tag} (no/incomplete metrics)")
            continue
        row = read_lbo(lbo, h, w)
        _, fmt, mode, backend = tag.replace("ggml_", "").split("_")
        ckpt = tag.split("_")[1]
        fps = n / m["wall_s"]
        pk = m["peaks"]
        res_rows.append({
            "row": tag, "engine": "ggml", "checkpoint": ckpt, "format": fmt,
            "mode": mode, "backend": backend, "wall_s": m["wall_s"], "fps": round(fps, 3),
            "gpu_peak_mb": pk.get("gpu_used_mb"), "proc_gpu_peak_mb": pk.get("proc_gpu_mb"),
            "rss_peak_mb": pk.get("rss_mb"), "vmhwm_mb": pk.get("vmhwm_mb"),
            "gpu_util_mean_pct": pk.get("gpu_util_mean_pct"),
        })
        # engine parity: same-format mirror; f32 GGUF is a bit-exact conversion
        # of the checkpoint, so the fp32 row references the checkpoint run.
        mirror_kind = {"f16": "mf16", "q8": "mq8", "f32": "fp32"}[fmt]
        ref = refs.get(f"{ckpt}_{mirror_kind}")
        if ref is not None:
            st = compare(row, ref, args.frames_sample)
            align_rows.append({"row": tag, "reference": f"{ckpt}_{mirror_kind}", **st})
            curves.setdefault(f"{ckpt}_{fmt}_{mode}_{backend}", per_frame_rmse(row, ref))
        ckpt_ref = refs.get(f"{ckpt}_fp32")
        if ckpt_ref is not None:
            st = compare(row, ckpt_ref, args.frames_sample)
            align_rows.append({"row": tag, "reference": f"{ckpt}_fp32", **st})

    # Deployment-noise reference rows: the official bf16 deployment vs its own
    # fp32 checkpoint (same implementation) — the noise floor any engine is
    # expected to sit below for a "fully aligned" verdict.
    for ckpt in ("long", "bal"):
        a, b = refs.get(f"{ckpt}_bf16"), refs.get(f"{ckpt}_fp32")
        if a is None or b is None:
            continue
        st = compare(a, b, args.frames_sample)
        align_rows.append({"row": f"pt_{ckpt}_bf16", "reference": f"{ckpt}_fp32 (PT self-noise)", **st})

    for ckpt in ("long", "bal"):
        for kind in ("fp32", "bf16", "mf16", "mq8"):
            m = load_metrics(run_dir / f"pt_{ckpt}_{kind}.metrics.json")
            if m is None:
                continue
            pk = m["peaks"]
            res_rows.append({
                "row": f"pt_{ckpt}_{kind}", "engine": "pytorch", "checkpoint": ckpt,
                "format": kind, "mode": "-", "backend": "cuda",
                "wall_s": m["wall_s"], "fps": round(n / m["wall_s"], 3),
                "gpu_peak_mb": pk.get("gpu_used_mb"), "proc_gpu_peak_mb": pk.get("proc_gpu_mb"),
                "rss_peak_mb": pk.get("rss_mb"), "vmhwm_mb": pk.get("vmhwm_mb"),
                "gpu_util_mean_pct": pk.get("gpu_util_mean_pct"),
            })

    # ---- CSVs -------------------------------------------------------------
    import csv
    for name, recs in (("alignment.csv", align_rows), ("resources.csv", res_rows)):
        if not recs:
            continue
        with open(out_dir / name, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
            wtr.writeheader()
            wtr.writerows(recs)

    # ---- parity curves ----------------------------------------------------
    if curves:
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, (tag, (perr, derr)) in enumerate(curves.items()):
            axes[0].plot(perr, lw=1.0, alpha=0.85, label=tag, color=colors[i % 10])
            axes[1].plot(derr, lw=1.0, alpha=0.85, label=tag, color=colors[i % 10])
        for ax, k in zip(axes, (0, 8)):
            ax.axvspan(0, k, color="grey", alpha=0.15, label="scale pass")
            for x in range(k, n, kf):
                ax.axvline(x, color="grey", lw=0.3, alpha=0.4)
        axes[0].set_ylabel("pose RMSE/frame")
        axes[1].set_ylabel("depth RMSE/frame")
        axes[1].set_xlabel("frame index (vertical ticks: keyframes, kf=%d)" % kf)
        axes[0].legend(fontsize=6, ncol=2)
        fig.suptitle(f"{ds}: per-frame parity vs same-format mirror (N={n}, {w}x{h}, kf={kf})")
        fig.savefig(out_dir / "parity_curves.png", dpi=130)
        plt.close(fig)

    # ---- speed / memory bars ----------------------------------------------
    if res_rows:
        for name, key, label in (("speed.png", "wall_s", "wall time (s)"),
                                 ("memory.png", "proc_gpu_peak_mb", "process GPU peak (MB)")):
            recs = sorted(res_rows, key=lambda r: r["row"])
            fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(recs)), 5.5), constrained_layout=True)
            cols = ["#3b6db5" if r["engine"] == "ggml" else "#c2554d" for r in recs]
            vals = [r[key] or 0 for r in recs]
            ax.bar(range(len(recs)), vals, color=cols)
            ax.set_xticks(range(len(recs)))
            ax.set_xticklabels([r["row"].replace("ggml_", "").replace("pt_", "pt:") for r in recs],
                               rotation=75, fontsize=6)
            ax.set_ylabel(label)
            ax.set_title(f"{ds}: {label} (blue=GGML, red=PyTorch; N={n}, kf={kf})")
            fig.savefig(out_dir / name, dpi=130)
            plt.close(fig)

    # ---- behavior (no-GT honesty table) ------------------------------------
    beh = []
    for ckpt in ("long", "bal"):
        src = refs.get(f"{ckpt}_fp32")
        if src is None:
            continue
        d = src["depth"]
        c = camera_centers(src["pose"])
        seg = np.linalg.norm(np.diff(c, axis=0), axis=1).sum() if len(c) > 1 else 0.0
        beh.append({
            "checkpoint": ckpt, "mean_abs_depth": float(np.abs(d).mean()),
            "depth_p95": float(np.percentile(np.abs(d), 95)),
            "trajectory_length": float(seg),
        })
    if beh:
        with open(out_dir / "behavior.csv", "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(beh[0].keys()))
            wtr.writeheader()
            wtr.writerows(beh)

    # ---- trajectory overlay (decoded camera centers C = -R^T t) ------------
    traj_targets = [("pt_long_fp32", refs.get("long_fp32")), ("pt_bal_fp32", refs.get("bal_fp32")),
                    ("ggml_long_f16_flash_cuda", None), ("ggml_bal_f16_flash_cuda", None)]
    traj = []
    for tag, ref in traj_targets:
        if ref is not None:
            traj.append((tag, camera_centers(ref["pose"])))
        else:
            lbo = run_dir / f"{tag}.lbo"
            if lbo.exists():
                traj.append((tag, camera_centers(read_lbo(lbo, h, w)["pose"])))
    if len(traj) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True)
        styles = {"pt_long_fp32": ("black", 2.0), "pt_bal_fp32": ("dimgray", 2.0)}
        for i, (tag, c) in enumerate(traj):
            st = styles.get(tag, (plt.cm.tab10(i % 10), 1.0))
            axes[0].plot(c[:, 0], c[:, 2], color=st[0], lw=st[1], alpha=0.9, label=tag)
            axes[1].plot(c[:, 2], c[:, 1], color=st[0], lw=st[1], alpha=0.9, label=tag)
        axes[0].set_xlabel("x"); axes[0].set_ylabel("z"); axes[0].set_title("top-down (x-z)")
        axes[1].set_xlabel("z"); axes[1].set_ylabel("y"); axes[1].set_title("side (z-y)")
        for ax in axes:
            ax.set_aspect("equal"); ax.legend(fontsize=7); ax.grid(alpha=0.2)
        fig.suptitle(f"{ds}: decoded camera trajectories (C = -R\u1d40t of w2c pose_enc)")
        fig.savefig(out_dir / "trajectory.png", dpi=130)
        plt.close(fig)

    print(f"[{ds}] alignment rows: {len(align_rows)}, resource rows: {len(res_rows)}")
    for r in align_rows:
        print(f"  {r['row']:38s} vs {r['reference']:14s} pose {r['pose_rmse']:.2e} "
              f"depth {r['depth_rmse']:.2e} rel {r['depth_rel']:.2e} tail {r['depth_tail_pct']:.2f}%")
    for r in res_rows:
        print(f"  {r['row']:38s} wall {r['wall_s']:8.1f}s fps {r['fps']:6.2f} "
              f"gpu {r['proc_gpu_peak_mb'] or '-':>7}MB util {r['gpu_util_mean_pct'] or '-'}%")

    if not args.skip_cloud:
        try:
            cloud_eval(ds, meta, run_dir, out_dir)
        except Exception as exc:  # keep the CSV deliverables alive on cloud failure
            print(f"cloud eval failed: {exc}")


if __name__ == "__main__":
    main()
