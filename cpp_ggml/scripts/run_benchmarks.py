#!/usr/bin/env python3
"""Run available GGML/PyTorch measurements and generate comparison figures.

No synthetic measurements are accepted. Missing model/backend rows remain
missing and make the matrix plot fail with an actionable message.
"""
import argparse, csv, subprocess
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, default=Path("build")); ap.add_argument("--models", type=Path, default=None)
    ap.add_argument("--pytorch", type=Path, default=None); ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--precisions", nargs="+", default=["f32", "f16", "q8", "q4"])
    ap.add_argument("--iters", type=int, default=5, help="measured iterations per row")
    ap.add_argument("--backends", nargs="+", default=["cpu", "cuda", "vulkan"])
    ap.add_argument("--allow-incomplete", action="store_true"); args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    build = args.build if args.build.is_absolute() else Path.cwd() / args.build
    models = args.models or (root / "models/gguf"); models = models if models.is_absolute() else Path.cwd() / models
    pytorch = args.pytorch or (root / "models/pytorch"); pytorch = pytorch if pytorch.is_absolute() else Path.cwd() / pytorch
    out = args.out or (root / "benchmarks/inference_speed.csv"); out = out if out.is_absolute() else Path.cwd() / out
    exe = build / "lingbot-map-bench"
    if not exe.is_file(): raise SystemExit(f"benchmark executable not found: {exe}; configure cpp_ggml first")
    rows = []
    backend_args = {"cpu": "cpu", "cuda": "CUDA0", "vulkan": "Vulkan0"}
    for backend in ((name, backend_args[name]) for name in args.backends):
        for precision in args.precisions:
            model = models / f"lingbot-map-{precision}.gguf"
            if not model.is_file(): continue
            proc = subprocess.run([str(exe), str(model), backend[1], str(args.iters)], text=True, capture_output=True)
            for line in proc.stdout.splitlines():
                if line.startswith("RESULT"):
                    fields = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
                    rows.append({"engine":"ggml", "backend":backend[0], "precision":precision, "mode":"forward", "median_ms":fields["median_ms"], "frames_per_s":fields["frames_per_s"]})
    checkpoint = pytorch / "lingbot-map.pt"
    input_npy = pytorch / "parity_input.npy"
    if checkpoint.is_file() and input_npy.is_file():
        proc = subprocess.run(["python3", str(Path(__file__).with_name("benchmark_pytorch.py")), str(checkpoint), str(input_npy)], text=True, capture_output=True)
        if proc.returncode == 0:
            fields = dict(x.split("=", 1) for x in proc.stdout.strip().split() if "=" in x)
            rows.append({"engine":"pytorch", "backend":fields.get("backend", "cuda"), "precision":"f32", "mode":"forward", "median_ms":fields["median_ms"], "frames_per_s":fields["frames_per_s"]})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["engine","backend","precision","mode","median_ms","frames_per_s"]); writer.writeheader(); writer.writerows(rows)
    plot = ["python3", str(Path(__file__).with_name("plot_benchmarks.py")), "--csv", str(out)]
    if args.allow_incomplete: plot.append("--allow-incomplete")
    subprocess.run(plot, check=True)

if __name__ == "__main__": main()
