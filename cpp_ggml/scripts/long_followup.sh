#!/usr/bin/env bash
# Follow-up attribution for the long-checkpoint depth tail:
#  A. regenerate per-dtype decoded-GGUF references at persistent paths
#  B. re-run the strict CLI per (backend, dtype) and record elementwise stats
#  C. re-run the same weights with --kv-f16 none (exact F32 cache): if the
#     elementwise tail collapses, it is the F16 cache rounding, not the engine
#  D. balanced f16 baseline at the same profile: long-specific or not
#  E. flash LBO stats vs the matching references
set -uo pipefail
ROOT=/home/asher/develop/code/dl/lingbot-map-ggml
cd "$ROOT"
PY="$HOME/anaconda3/envs/python3.12/bin/python"

# input .npy for run_pytorch_reference (the matrix only wrote the .bin)
"$PY" - <<'EOF'
import numpy as np
vals = np.fromfile('/tmp/frames286.bin', dtype=np.float32).reshape(286, 3, 294, 518)
np.save('/tmp/frames286.npy', vals[None])
print('npy:', vals[None].shape)
EOF

stats () { # LBO REF label
  "$PY" - "$1" "$2" "$3" <<'EOF'
import sys, numpy as np, struct
lbo, ref_path, label = sys.argv[1], sys.argv[2], sys.argv[3]
raw = open(lbo, 'rb').read()
_, n_pose, n_depth = struct.unpack('<III', raw[:12])
vals = np.frombuffer(raw, dtype=np.float32, offset=12)
d = vals[n_pose:n_pose+n_depth]; p = vals[:n_pose].reshape(-1, 9)
ref = np.load(ref_path)
rd = np.asarray(ref['depth'], np.float32).reshape(-1)
rp = np.asarray(ref['pose_enc'], np.float32).reshape(-1, 9)
if d.size != rd.size:
    print(f'{label}: SIZE MISMATCH d={d.size} ref={rd.size}'); sys.exit(0)
err = np.abs(d - rd); perr = np.abs(p - rp)
print(f'{label}: pose_rmse={np.sqrt((perr**2).mean()):.3e} pose_max={perr.max():.3e} '
      f'depth_rmse={np.sqrt((err**2).mean()):.3e} depth_max={err.max():.3e} '
      f'tail>1.5e-3={100*(err>1.5e-3).mean():.2f}%')
EOF
}

# A. persistent references
for dtype in f16 q8 f32; do
  echo "===== REF $dtype"
  "$PY" cpp_ggml/scripts/run_pytorch_reference.py cpp_ggml/models/pytorch/lingbot-map.pt \
    /tmp/frames286.npy "/tmp/ref_long_${dtype}.npz" --device cuda --streaming \
    --scale-frames 8 --kv-cache-scale 8 --kv-cache-window 64 \
    --gguf "cpp_ggml/models/gguf/lingbot-map-long-${dtype}.gguf" 2>&1 | tail -2
done

# B. strict elementwise stats per (backend, dtype)
declare -A BUILD=( [cuda]=cpp_ggml/build-cuda [vulkan]=cpp_ggml/build-vulkan )
declare -A DEV=( [cuda]=CUDA0 [vulkan]=Vulkan0 )
for backend in cuda vulkan; do
  for dtype in f16 q8 f32; do
    echo "===== STRICT-STATS $backend $dtype"
    /usr/bin/time -f "STRICT_WALL_%es" "${BUILD[$backend]}/lingbot-map-cli" \
      "cpp_ggml/models/gguf/lingbot-map-long-${dtype}.gguf" /tmp/frames286.bin "${DEV[$backend]}" 294 518 \
      "/tmp/strict_long_${backend}_${dtype}.lbo" 286 \
      --kv-f16 strict --kv-scale 8 --kv-window 64 --scale-frames 8 2>&1 | grep -E "STRICT_WALL|error|failed"
    stats "/tmp/strict_long_${backend}_${dtype}.lbo" "/tmp/ref_long_${dtype}.npz" "strict $backend $dtype"
  done
done

# C. F32-cache attribution on cuda f16
echo "===== F32-CACHE ATTRIBUTION cuda f16"
/usr/bin/time -f "STRICT32_WALL_%es" cpp_ggml/build-cuda/lingbot-map-cli \
  cpp_ggml/models/gguf/lingbot-map-long-f16.gguf /tmp/frames286.bin CUDA0 294 518 \
  /tmp/strict32_long_cuda.lbo 286 --kv-f16 none --kv-scale 8 --kv-window 64 --scale-frames 8 2>&1 | grep -E "STRICT32_WALL|error|failed"
stats /tmp/strict32_long_cuda.lbo /tmp/ref_long_f16.npz "f32cache cuda f16"

# D. balanced f16 baseline at the same profile (long-specific check)
echo "===== BALANCED f16 BASELINE cuda"
/usr/bin/time -f "BAL_WALL_%es" cpp_ggml/build-cuda/lingbot-map-cli \
  cpp_ggml/models/gguf/lingbot-map-f16.gguf /tmp/frames286.bin CUDA0 294 518 \
  /tmp/bal_f16_cuda.lbo 286 --kv-f16 strict --kv-scale 8 --kv-window 64 --scale-frames 8 2>&1 | grep -E "BAL_WALL|error|failed"
"$PY" cpp_ggml/scripts/run_pytorch_reference.py cpp_ggml/models/pytorch/lingbot-map.pt \
  /tmp/frames286.npy /tmp/ref_bal_f16.npz --device cuda --streaming \
  --scale-frames 8 --kv-cache-scale 8 --kv-cache-window 64 \
  --gguf cpp_ggml/models/gguf/lingbot-map-f16.gguf 2>&1 | tail -1
stats /tmp/bal_f16_cuda.lbo /tmp/ref_bal_f16.npz "balanced f16 cuda"

# E. flash stats vs matching references
for backend in cuda vulkan; do
  for dtype in f16 q8 f32; do
    stats "/tmp/flash_long_${backend}_${dtype}.lbo" "/tmp/ref_long_${dtype}.npz" "flash $backend $dtype"
  done
done
echo "===== FOLLOWUP DONE"
