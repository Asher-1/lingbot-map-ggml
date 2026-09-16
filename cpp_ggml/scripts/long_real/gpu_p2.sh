#!/usr/bin/env bash
# P2 verification: q8mix (q8 aggregator+depth_head, f16 camera_head) @ drive.
set -uo pipefail
cd /home/asher/develop/code/dl/lingbot-map-ggml
PY="$HOME/anaconda3/envs/python3.12/bin/python"
MON="$PY cpp_ggml/scripts/long_real/monitor.py"
GGUF=cpp_ggml/models/gguf/lingbot-map-long-q8mix.gguf
wait_gpu() { while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-0}" -lt 6000 ] && break; echo "[gpu busy $u] wait"; sleep 60; done; }
# wait for the followup chain to finish
while pgrep -f "gpu_followup.sh" > /dev/null 2>&1; do sleep 60; done

OUT=/tmp/long_real/runs/drive
echo "===== P2 mirror reference for q8mix"
wait_gpu
"$PY" cpp_ggml/scripts/run_pytorch_reference.py cpp_ggml/models/pytorch/lingbot-map-long.pt \
  /tmp/long_real/drive.npy "$OUT/pt_long_q8mix_mf.npz" \
  --device cuda --streaming --scale-frames 8 --kv-cache-scale 8 --kv-cache-window 64 \
  --keyframe-interval 4 --gguf "$GGUF" 2>&1 | tail -1

echo "===== P2 C++ rows"
for mode in strict flash; do
  tag="ggml_long_q8mix_${mode}_cuda"
  m="$OUT/$tag.metrics.json"
  [ -f "$m" ] && grep -q '"exit_code": 0' "$m" && { echo "skip $tag"; continue; }
  wait_gpu
  $MON --out "$m" --label "drive/$tag" -- \
    cpp_ggml/build-cuda/lingbot-map-cli "$GGUF" /tmp/long_real/drive.bin CUDA0 294 518 \
    "$OUT/$tag.lbo" 1050 --kv-f16 "$mode" --kv-scale 8 --kv-window 64 --scale-frames 8 \
    --kv-total 1050 --keyframe-interval 4 2>&1 | grep -E "MONITOR|RESULT|error|failed" || echo "P2 FAILED: $tag"
done
echo "===== P2 GPU DONE"
