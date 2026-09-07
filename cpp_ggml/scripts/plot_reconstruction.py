#!/usr/bin/env python3
"""Plot measured reconstruction quality; never invents missing values."""
import argparse, csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csv", type=Path); args = ap.parse_args()
    rows = list(csv.DictReader(args.csv.open()))
    required = {"engine", "scene", "ate_rmse", "depth_rmse"}
    if not rows or not required.issubset(rows[0]): raise SystemExit("CSV must contain engine,scene,ate_rmse,depth_rmse")
    scenes = sorted({r["scene"] for r in rows}); engines = sorted({r["engine"] for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, key in enumerate(("ate_rmse", "depth_rmse")):
        for engine in engines:
            vals = [float(next(r[key] for r in rows if r["engine"] == engine and r["scene"] == s)) for s in scenes]
            ax[i].plot(scenes, vals, marker="o", label=engine)
        ax[i].set_title(key); ax[i].grid(True, linestyle=":"); ax[i].tick_params(axis="x", rotation=35)
    ax[0].legend(); fig.tight_layout(); fig.savefig(args.csv.parent / "reconstruction_compare.png", dpi=160)

if __name__ == "__main__": main()
