#!/usr/bin/env bash
# P1 attribution references: pt_long_fp32 + pt_long_mf16 @ drive (deleted in cleanup).
set -uo pipefail
cd /home/asher/develop/code/dl/lingbot-map-ggml
PY="$HOME/anaconda3/envs/python3.12/bin/python"
while pgrep -f "gpu_p2.sh" > /dev/null 2>&1; do sleep 60; done
OUT=/tmp/long_real/runs/drive
for kind in fp32 mf16; do
  [ -f "$OUT/pt_long_${kind}.npz" ] && continue
  case $kind in fp32) ARGS=();; mf16) ARGS=(--gguf cpp_ggml/models/gguf/lingbot-map-long-f16.gguf);; esac
  echo "=== pt_long_$kind at $(date +%H:%M)"
  "$PY" cpp_ggml/scripts/run_pytorch_reference.py cpp_ggml/models/pytorch/lingbot-map-long.pt \
    /tmp/long_real/drive.npy "$OUT/pt_long_${kind}.npz" \
    --device cuda --streaming --scale-frames 8 --kv-cache-scale 8 --kv-cache-window 64 \
    --keyframe-interval 4 "${ARGS[@]}" 2>&1 | tail -1
done
echo "=== P1 REFS DONE"
"$PY" - <<'PYEOF'
import numpy as np
h, w = 294, 518
z32 = np.load('/tmp/long_real/runs/drive/pt_long_fp32.npz')
zfc = np.load('/tmp/long_real/runs/drive/pt_long_mf16_flashcache.npz')
zmf = np.load('/tmp/long_real/runs/drive/pt_long_mf16.npz')
p32 = z32['pose_enc'].reshape(-1,9); d32 = z32['depth'].reshape(-1,h,w)
p32m = np.abs(d32).mean()
for name, z in (('PT flash-cache vs fp32-cache', zfc), ('PT fp32-cache(mf16) vs checkpoint fp32', zmf)):
    p = z['pose_enc'].reshape(-1,9); d = z['depth'].reshape(-1,h,w)
    perr = np.abs(p-p32); derr = np.abs(d-d32)
    pfr = np.sqrt((perr**2).mean(axis=1))
    print(f"{name}: pose {np.sqrt((perr**2).mean()):.2e} depth REL {np.sqrt((derr**2).mean())/p32m:.2e} | pose first50={pfr[:50].mean():.2e} lastQ={pfr[-len(pfr)//4:].mean():.2e}")
PYEOF
