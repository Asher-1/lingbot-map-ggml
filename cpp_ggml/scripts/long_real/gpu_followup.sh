#!/usr/bin/env bash
# Follow-up GPU chain: P1 flash-cache attribution mirror -> P3 f32-cache rows.
set -uo pipefail
cd /home/asher/develop/code/dl/lingbot-map-ggml
PY="$HOME/anaconda3/envs/python3.12/bin/python"
MON="$PY cpp_ggml/scripts/long_real/monitor.py"
GGUF=cpp_ggml/models/gguf
OUT=/tmp/long_real/runs/drive
wait_gpu() { while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-0}" -lt 6000 ] && break; echo "[gpu busy $u] wait"; sleep 60; done; }

echo "===== P1: mirror with FULL F16 K/V (flash cache contract) @ drive 1050f"
wait_gpu
LINGBOT_KV_CACHE_F16=flash "$PY" cpp_ggml/scripts/run_pytorch_reference.py \
  cpp_ggml/models/pytorch/lingbot-map-long.pt /tmp/long_real/drive.npy /tmp/long_real/runs/drive/pt_long_mf16_flashcache.npz \
  --device cuda --streaming --scale-frames 8 --kv-cache-scale 8 --kv-cache-window 64 \
  --keyframe-interval 4 --gguf "$GGUF/lingbot-map-long-f16.gguf" 2>&1 | tail -1

echo "===== P3: GGML --kv-f16 none @ indoor 2000f (F32 cache escape hatch)"
for backend in cuda vulkan; do
  case "$backend" in cuda) BUILD=cpp_ggml/build-cuda; DEV=CUDA0;; vulkan) BUILD=cpp_ggml/build-vulkan; DEV=Vulkan0;; esac
  tag="ggml_long_f16_none_${backend}"
  m="/tmp/long_real/runs/indoor/$tag.metrics.json"
  [ -f "$m" ] && grep -q '"exit_code": 0' "$m" && { echo "skip $tag"; continue; }
  wait_gpu
  $MON --out "$m" --label "indoor/$tag" -- \
    "$BUILD/lingbot-map-cli" "$GGUF/lingbot-map-long-f16.gguf" /tmp/long_real/indoor.bin "$DEV" 294 518 \
    "/tmp/long_real/runs/indoor/$tag.lbo" 2000 \
    --kv-f16 none --kv-scale 8 --kv-window 64 --scale-frames 8 --kv-total 2000 --keyframe-interval 7 \
    2>&1 | grep -E "MONITOR|RESULT|error|failed" || echo "P3 FAILED: $tag"
done
echo "===== GPU FOLLOWUP DONE"
