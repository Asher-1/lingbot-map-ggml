#!/usr/bin/env python3
"""Latency-matrix charts for the long_real campaign.

Three views over resources.csv (all rows, all datasets):
  latency_matrix.png  — per-frame latency heatmap, {checkpoint x format x mode}
                        rows x {CUDA,Vulkan} columns per dataset, wall-seconds
                        annotated; the "full latency matrix" view.
  fps_scaling.png     — fps vs stream length (667/1050/2000) for the headline
                        configs; shows how each engine scales with N.
  latency_breakdown.png — per-frame latency decomposition per dataset:
                        PT bf16 baseline -> +F32 GEMM (GGML flash residual)
                        -> +F32 attention (strict delta); Vulkan delta noted.

Usage: plot_latency.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "long_real"
DS = ["lingbo_world", "drive", "indoor"]
NS = {"lingbo_world": 667, "drive": 1050, "indoor": 2000}


def load(ds):
    return list(csv.DictReader(open(BENCH / ds / "resources.csv")))


def fmt_row(rows, row_name, ds):
    """Seconds per frame for one row name; None if absent."""
    for r in rows:
        if r["row"] == row_name and r["wall_s"]:
            return float(r["wall_s"]) / NS[ds]
    return None


def main():
    data = {ds: load(ds) for ds in DS}

    # ---- latency matrix heatmaps ------------------------------------------
    ckpts = ["long", "bal"]
    fmts = ["f32", "f16", "q8"]
    modes = ["strict", "flash"]
    row_labels = [f"{c}/{f}/{m}" for c in ckpts for f in fmts for m in modes]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    for ax, backend in zip(axes, ["cuda", "vulkan"]):
        mat = np.full((len(row_labels), len(DS)), np.nan)
        for j, ds in enumerate(DS):
            for r in data[ds]:
                if r["engine"] != "ggml" or r["backend"] != backend:
                    continue
                label = f"{r['checkpoint']}/{r['format']}/{r['mode']}"
                if label in row_labels:
                    mat[row_labels.index(label), j] = float(r["wall_s"])
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(DS)))
        ax.set_xticklabels([f"{d}\n(N={NS[d]}, kf={3 if d=='lingbo_world' else 4 if d=='drive' else 7})" for d in DS])
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.0f}s", ha="center", va="center", fontsize=7)
        ax.set_title(f"GGML {backend.upper()} — wall seconds (per full stream)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Latency matrix: GGML configs x datasets (official profile, auto keyframe)")
    fig.savefig(BENCH / "latency_matrix.png", dpi=130)
    plt.close(fig)

    # ---- fps scaling --------------------------------------------------------
    headline = [
        ("ggml_long_f16_flash_cuda", "GGML long f16 flash CUDA", "tab:blue", "o"),
        ("ggml_long_f16_flash_vulkan", "GGML long f16 flash Vulkan", "tab:cyan", "o"),
        ("ggml_long_f16_strict_cuda", "GGML long f16 strict CUDA", "tab:orange", "s"),
        ("ggml_long_q8_flash_cuda", "GGML long q8 flash CUDA", "tab:green", "o"),
        ("pt_long_fp32", "PT fp32 streaming", "tab:red", "^"),
        ("pt_long_bf16", "PT bf16 (deployment)", "tab:purple", "v"),
    ]
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ns = [NS[d] for d in DS]
    for row, label, color, marker in headline:
        ys = []
        for ds in DS:
            v = fmt_row(data[ds], row, ds)
            ys.append(NS[ds] / v if v else np.nan)
        ax.plot(ns, ys, marker + "-", color=color, label=label, lw=1.8, ms=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("stream length N (frames, log)")
    ax.set_ylabel("throughput (fps, log)")
    ax.set_title("Throughput vs stream length (long checkpoint, official profile)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.savefig(BENCH / "fps_scaling.png", dpi=130)
    plt.close(fig)

    # ---- latency decomposition ---------------------------------------------
    fig, axes = plt.subplots(1, len(DS), figsize=(4.6 * len(DS), 5), sharey=False, constrained_layout=True)
    for ax, ds in zip(np.atleast_1d(axes), DS):
        spf = {}
        for name, row in [
            ("PT bf16", "pt_long_bf16"), ("GGML flash", "ggml_long_f16_flash_cuda"),
            ("GGML strict", "ggml_long_f16_strict_cuda"), ("PT fp32", "pt_long_fp32"),
        ]:
            v = fmt_row(data[ds], row, ds)
            if v:
                spf[name] = v
        base = spf.get("PT bf16")
        flash = spf.get("GGML flash")
        strict = spf.get("GGML strict")
        bars = [
            ("PT bf16\n(deploy base)", base, "#8c6bb1"),
            ("GGML flash\n= base + F32 GEMM", flash, "#3b6db5"),
            ("GGML strict\n= flash + F32 attn", strict, "#e08214"),
        ]
        xs = np.arange(len(bars))
        vals = [b[1] for b in bars]
        ax.bar(xs, vals, color=[b[2] for b in bars], width=0.62)
        for x, v in zip(xs, vals):
            if v:
                ax.text(x, v * 1.02, f"{v*1000:.0f} ms/f", ha="center", fontsize=8)
        # delta annotations
        if base and flash:
            ax.annotate(f"GEMM headroom\n+{(flash-base)*1000:.0f} ms/f", xy=(1, flash), xytext=(1, flash * 0.55),
                        ha="center", fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8))
        if flash and strict:
            ax.annotate(f"F32 attention\n+{(strict-flash)*1000:.0f} ms/f", xy=(2, strict), xytext=(2, strict * 0.75),
                        ha="center", fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8))
        ax.set_xticks(xs)
        ax.set_xticklabels([b[0] for b in bars], fontsize=7)
        ax.set_ylabel("seconds per frame")
        vk = fmt_row(data[ds], "ggml_long_f16_flash_vulkan", ds)
        title = f"{ds} (N={NS[ds]})"
        if vk and flash:
            title += f"\nVulkan flash +{(vk-flash)*1000:.0f} ms/f vs CUDA"
        ax.set_title(title, fontsize=9)
    fig.suptitle("Per-frame latency decomposition (long f16, CUDA)")
    fig.savefig(BENCH / "latency_breakdown.png", dpi=130)
    plt.close(fig)
    print(f"wrote latency_matrix.png / fps_scaling.png / latency_breakdown.png in {BENCH}")


if __name__ == "__main__":
    main()
