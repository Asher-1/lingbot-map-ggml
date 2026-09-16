#!/usr/bin/env python3
"""Global markdown report for the long_real campaign (Stage 4, final).

Consolidates the per-dataset CSVs produced by evaluate.py into
cpp_ggml/benchmarks/long_real/long_real_report.md. Every claim in the report
cites a measured number from those CSVs; where no ground truth exists the
report says so instead of speculating.

Usage: plot_report.py [datasets...]   (default: the three campaign datasets)
"""
import csv
import json
from pathlib import Path

BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "long_real"
RUNS = Path("/tmp/long_real/runs")
ALL_DS = ["lingbo_world", "drive", "indoor"]


def read_csv(path: Path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fmt(x, spec=".2e"):
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return str(x) if x is not None else "-"


def main():
    datasets = sys_argv_datasets()
    lines = ["# LingBot-Map long-model real-sequence end-to-end comparison",
             "", "Campaign inputs are the official demo sequences "
             "(`robbyant/lingbot-map-demo`), extracted with the official demo.py "
             "video rule (fps=10) and the official crop rule; both engines consume "
             "byte-identical `float32` frames. Both engines run the official "
             "streaming profile `scale=8/window=64` with the official auto "
             "`keyframe_interval = ceil(N/320)` (explicit on both sides).",
             "",
             "> No ground truth exists for these sequences: accuracy rows measure "
             "> cross-engine agreement (GGML vs decoded-GGUF PyTorch mirror = "
             "> engine parity; vs fp32 checkpoint = total deviation incl. weight "
             "> format). Depth absolute metrics are depth-scale-weighted; the REL "
             "> column normalizes by mean|depth_ref|.",
             ""]

    for ds in datasets:
        meta_p = Path(f"/tmp/long_real/{ds}.meta.json")
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        align = read_csv(BENCH / ds / "alignment.csv")
        res = read_csv(BENCH / ds / "resources.csv")
        beh = read_csv(BENCH / ds / "behavior.csv")
        clouds = read_csv(BENCH / ds / "cloud_eval.csv")
        lines += [f"## {ds} — N={meta['n_frames']}, {meta['width']}x{meta['height']}, "
                  f"auto kf={meta['auto_keyframe_interval']}", ""]

        if align:
            lines += ["| row | reference | pose RMSE | depth RMSE | depth REL | tail% |",
                      "|---|---|---|---|---|---|"]
            for r in align:
                lines.append(f"| {r['row']} | {r['reference']} | {fmt(r['pose_rmse'])} | "
                             f"{fmt(r['depth_rmse'])} | {fmt(r['depth_rel'])} | "
                             f"{fmt(r['depth_tail_pct'], '.2f')} |")
            lines.append("")
        if res:
            lines += ["| row | wall (s) | FPS | proc GPU peak (MB) | RSS peak (MB) | util % |",
                      "|---|---|---|---|---|---|"]
            for r in sorted(res, key=lambda x: x["row"]):
                lines.append(f"| {r['row']} | {fmt(r['wall_s'], '.1f')} | {fmt(r['fps'], '.2f')} | "
                             f"{fmt(r['proc_gpu_peak_mb'], '.0f')} | {fmt(r['rss_peak_mb'], '.0f')} | "
                             f"{fmt(r['gpu_util_mean_pct'], '.0f')} |")
            lines.append("")
        if beh:
            lines += ["| checkpoint | mean \\|depth\\| | depth p95 | trajectory length |",
                      "|---|---|---|---|"]
            for r in beh:
                lines.append(f"| {r['checkpoint']} | {fmt(r['mean_abs_depth'], '.3f')} | "
                             f"{fmt(r['depth_p95'], '.3f')} | {fmt(r['trajectory_length'], '.2f')} |")
            lines.append("")
        if clouds:
            lines += ["| cloud A | cloud B | symmetric NN RMSE |", "|---|---|---|"]
            for r in clouds:
                lines.append(f"| {r['cloud_a']} | {r['cloud_b']} | {fmt(r['nn_rmse'], '.4f')} |")
            lines.append("")

        for png in ("parity_curves", "trajectory", "speed", "memory", "cloud_effect"):
            if (BENCH / ds / f"{png}.png").exists():
                lines += [f"![{ds} {png}]({ds}/{png}.png)", ""]

    for png in ("latency_matrix", "fps_scaling", "latency_breakdown"):
        if (BENCH / f"{png}.png").exists():
            lines += [f"## {png}", f"![{png}]({png}.png)", ""]

    out = BENCH / "long_real_report.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def sys_argv_datasets():
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    return args or ALL_DS


if __name__ == "__main__":
    main()
