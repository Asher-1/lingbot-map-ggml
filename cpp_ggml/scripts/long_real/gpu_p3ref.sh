#!/usr/bin/env bash
# P3 attribution references: pt_long_fp32 + pt_long_mf16 @ indoor.
set -uo pipefail
cd /home/asher/develop/code/dl/lingbot-map-ggml
PY="$HOME/anaconda3/envs/python3.12/bin/python"
while pgrep -f "gpu_p1ref.sh" > /dev/null 2>&1; do sleep 60; done
OUT=/tmp/long_real/runs/indoor
for kind in fp32 mf16; do
  [ -f "$OUT/pt_long_${kind}.npz" ] && continue
  case $kind in fp32) ARGS=();; mf16) ARGS=(--gguf cpp_ggml/models/gguf/lingbot-map-long-f16.gguf);; esac
  echo "=== indoor pt_long_$kind at $(date +%H:%M)"
  "$PY" cpp_ggml/scripts/run_pytorch_reference.py cpp_ggml/models/pytorch/lingbot-map-long.pt \
    /tmp/long_real/indoor.npy "$OUT/pt_long_${kind}.npz" \
    --device cuda --streaming --scale-frames 8 --kv-cache-scale 8 --kv-cache-window 64 \
    --keyframe-interval 7 "${ARGS[@]}" 2>&1 | tail -1
done
echo "=== P3 REFS DONE at $(date +%H:%M)"
"$PY" - <<'PYEOF'
import numpy as np, struct
def read_lbo(p, h=294, w=518):
    raw = open(p,'rb').read()
    _, n_pose, n_depth = struct.unpack('<III', raw[:12])
    v = np.frombuffer(raw, dtype=np.float32, offset=12)
    return v[:n_pose].reshape(-1,9), v[n_pose:n_pose+n_depth].reshape(-1,h,w)
z = np.load('/tmp/long_real/runs/indoor/pt_long_fp32.npz')
none_ = read_lbo('/tmp/long_real/runs/indoor/ggml_long_f16_none_cuda.lbo')
p32, d32 = z['pose_enc'].reshape(-1,9), z['depth'].reshape(-1,294,518)
n = min(len(none_[0]), len(p32))
perr = np.abs(none_[0][:n]-p32[:n]); derr = np.abs(none_[1][:n]-d32[:n])
pfr = np.sqrt((perr**2).mean(axis=1))
print(f"P3: GGML none(F32 cache) vs PT fp32-cache @ indoor 2000f:")
print(f"  pose {np.sqrt((perr**2).mean()):.2e} depth REL {np.sqrt((derr**2).mean())/np.abs(d32).mean():.2e}")
print(f"  pose per-frame: first200={pfr[:200].mean():.2e} -> last500={pfr[-500:].mean():.2e}")
print("  (strict(F16 cache) vs same mirror was pose 5.90e-04 / REL 1.19e-04)")
PYEOF
