#!/usr/bin/env python3
"""Measure GCTStream VRAM on the official vs the parity reference profile.

Answers "does the official PyTorch model really need >24 GiB": the KV cache
residency is (kv_cache_scale_frames + kv_cache_sliding_window) frames x 24
layers x tokens x dim x 2(K,V) x dtype_bytes, and its dtype follows the run
precision. The official path (bf16 autocast) therefore holds ~half the bytes
of the F32 parity reference, which exists only to match the C++ F32 graph.

Two profiles:
  official  bf16 autocast, cuDNN/TF32 defaults, SDPA backend
  parity    F32, cuDNN+TF32 disabled, SDPA backend (run_pytorch_reference contract)

Usage:
  python3 cpp_ggml/scripts/measure_pytorch_vram.py --checkpoint \
      cpp_ggml/models/pytorch/lingbot-map.pt --profile official --frames 100
"""
import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import torch


def fmt(gbytes: float) -> str:
    return f"{gbytes:6.2f} GB"


def gpu_gib() -> tuple[float, float]:
    return (torch.cuda.memory_allocated() / 2**30, torch.cuda.max_memory_allocated() / 2**30)


def theoretical_cache_gib(tokens: int, layers: int, dim: int, frames: int, dtype_bytes: int) -> float:
    return tokens * dim * 2 * layers * frames * dtype_bytes / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("cpp_ggml/models/pytorch/lingbot-map.pt"))
    ap.add_argument("--profile", choices=("official", "parity"), default="official")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--height", type=int, default=518)
    ap.add_argument("--width", type=int, default=518)
    ap.add_argument("--scale-frames", type=int, default=8)
    ap.add_argument("--kv-window", type=int, default=64)
    ap.add_argument("--report-every", type=int, default=25)
    args = ap.parse_args()

    device = torch.device("cuda")
    if args.profile == "parity":
        # run_pytorch_reference.py contract: cuBLAS-only F32, no TF32/cuDNN.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.enabled = False
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from lingbot_map.models.gct_stream import GCTStream

    torch.cuda.reset_peak_memory_stats()
    model = GCTStream(img_size=518, patch_size=14, use_sdpa=True,
                      enable_3d_rope=False, camera_num_iterations=4,
                      kv_cache_scale_frames=args.scale_frames,
                      kv_cache_sliding_window=args.kv_window)
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = raw.get("model", raw)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    del raw, state
    model.eval().to(device)
    w_alloc, w_peak = gpu_gib()
    print(f"[{args.profile}] weights loaded: alloc={fmt(w_alloc)} peak={fmt(w_peak)}")

    rng = np.random.default_rng(0)
    frames = rng.random((1, args.frames, 3, args.height, args.width), dtype=np.float32)
    inp = torch.from_numpy(frames)
    tokens = (args.height // 14) * (args.width // 14)
    dtype_bytes = 2 if args.profile == "official" else 4

    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if args.profile == "official" else contextlib.nullcontext())
    scale = min(args.scale_frames, args.frames)
    with torch.no_grad(), autocast:
        torch.compiler.cudagraph_mark_step_begin()
        out = model.forward(inp[:, :scale].to(device), num_frame_for_scale=scale,
                            num_frame_per_block=scale, causal_inference=True)
        a, p = gpu_gib()
        th = theoretical_cache_gib(tokens, 24, 1024, scale, dtype_bytes)
        print(f"[{args.profile}] after scale pass ({scale}f): alloc={fmt(a)} peak={fmt(p)} "
              f"cache_theory<={fmt(th)}")
        del out
        for i in range(scale, args.frames):
            torch.compiler.cudagraph_mark_step_begin()
            out = model.forward(inp[:, i:i + 1].to(device), num_frame_for_scale=scale,
                                num_frame_per_block=1, causal_inference=True)
            del out
            if (i + 1 - scale) % args.report_every == 0 or i == args.frames - 1:
                a, p = gpu_gib()
                cached = min(i + 1, args.scale_frames + args.kv_window)
                th = theoretical_cache_gib(tokens, 24, 1024, cached, dtype_bytes)
                print(f"[{args.profile}] frame {i + 1:4d}: alloc={fmt(a)} peak={fmt(p)} "
                      f"cache_frames={cached} cache_theory<={fmt(th)}")
    torch.cuda.synchronize()
    a, p = gpu_gib()
    print(f"[{args.profile}] FINAL alloc={fmt(a)} peak={fmt(p)}")
    if p > 20.0:
        print("VERDICT: >20 GiB peak — expected only for the F32 parity profile at 518")
    else:
        print("VERDICT: within a 24 GiB card with headroom")


if __name__ == "__main__":
    main()
