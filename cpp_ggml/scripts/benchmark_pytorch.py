#!/usr/bin/env python3
"""Measure the released PyTorch model on the same input used by GGML."""
import sys, time
from pathlib import Path
import numpy as np
import torch

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from lingbot_map.models.gct_stream import GCTStream

ckpt, input_path = Path(sys.argv[1]), Path(sys.argv[2])
x = torch.from_numpy(np.load(input_path).astype(np.float32))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    # Keep the reference benchmark independent of cuDNN and match GGML's
    # default F32 accumulation contract.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.enabled = False
model = GCTStream(img_size=x.shape[-1], patch_size=14, use_sdpa=True,
                  enable_3d_rope=False, camera_num_iterations=4)
raw = torch.load(ckpt, map_location="cpu", weights_only=False)
state = raw.get("model", raw)
missing, unexpected = model.load_state_dict(state, strict=False)
if missing or unexpected:
    raise SystemExit(f"checkpoint mismatch: missing={len(missing)} unexpected={len(unexpected)}")
model.eval().to(device); x = x.to(device)
with torch.no_grad():
    for _ in range(2): model(x, causal_inference=False, gather_outputs=False)
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5): model(x, causal_inference=False, gather_outputs=False)
    if device.type == "cuda": torch.cuda.synchronize()
ms = (time.perf_counter() - t0) * 1000 / 5
print(f"RESULT engine=pytorch backend={device.type} precision=f32 median_ms={ms:.4f} frames_per_s={1000/ms:.4f}")
