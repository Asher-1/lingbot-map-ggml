#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", type=Path, required=True); ap.add_argument("--allow-incomplete", action="store_true"); args = ap.parse_args()
    rows = list(csv.DictReader(args.csv.open()))
    required = {(b,p) for b in ("cpu","cuda","vulkan") for p in ("f32","f16","q8","q4")}
    got = {(r["backend"],r["precision"]) for r in rows}
    missing = sorted(required - got)
    if missing and not args.allow_incomplete: raise SystemExit("latency matrix incomplete; missing measured rows: " + ", ".join(f"{b}/{p}" for b,p in missing))
    backends = ["cpu","cuda","vulkan"]; precisions = ["f32","f16","q8","q4"]
    matrix = [[next((float(r["median_ms"]) for r in rows if r["backend"]==b and r["precision"]==p), float("nan")) for p in precisions] for b in backends]
    fig, ax = plt.subplots(figsize=(8,4)); im=ax.imshow(matrix, cmap="viridis"); ax.set_xticks(range(4), precisions); ax.set_yticks(range(3), backends)
    for i,row in enumerate(matrix):
        for j,v in enumerate(row): ax.text(j,i,f"{v:.1f} ms",ha="center",va="center",color="white")
    ax.set_title("LingBot-Map GGML latency matrix" + (" (missing rows unmeasured/OOM)" if missing else "")); fig.colorbar(im, ax=ax, label="ms"); fig.tight_layout(); fig.savefig(args.csv.parent/"latency_matrix.png", dpi=160)

if __name__ == "__main__": main()
