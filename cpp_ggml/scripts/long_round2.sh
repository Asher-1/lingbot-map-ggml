#!/usr/bin/env bash
# Root-cause round 2 for the long depth tail + speed/accuracy alignment matrix.
# GPU sequence:
#   E1a/E1b  PyTorch mirror WITH the F16 cache (LINGBOT_KV_CACHE_F16=1):
#            does the tail reproduce outside the C++ engine? (long + balanced)
#   S1-S3    balanced f16 GGML walls (strict vulkan, flash cuda, flash vulkan)
#            for the long-vs-balanced speed alignment
#   P1/P2    PyTorch long streaming walls (fp32 + bf16 deployment autocast)
set -uo pipefail
ROOT=/home/asher/develop/code/dl/lingbot-map-ggml
cd "$ROOT"
PY="$HOME/anaconda3/envs/python3.12/bin/python"
export LINGBOT_KV_CACHE_F16=1

echo "===== E1a mirror long-f16 GGUF + F16 cache"
/usr/bin/time -f "E1A_WALL_%es" "$PY" cpp_ggml/scripts/run_pytorch_reference.py \
  cpp_ggml/models/pytorch/lingbot-map-long.pt /tmp/frames286.npy /tmp/ref_long_f16_f16cache.npz \
  --device cuda --streaming --scale-frames 8 --kv-cache-scale 8 --kv-cache-window 64 \
  --gguf cpp_ggml/models/gguf/lingbot-map-long-f16.gguf 2>&1 | grep -E "E1A_WALL|wrote|Error"

echo "===== E1b mirror balanced-f16 GGUF + F16 cache"
/usr/bin/time -f "E1B_WALL_%es" "$PY" cpp_ggml/scripts/run_pytorch_reference.py \
  cpp_ggml/models/pytorch/lingbot-map.pt /tmp/frames286.npy /tmp/ref_bal_f16_f16cache.npz \
  --device cuda --streaming --scale-frames 8 --kv-cache-scale 8 --kv-cache-window 64 \
  --gguf cpp_ggml/models/gguf/lingbot-map-f16.gguf 2>&1 | grep -E "E1B_WALL|wrote|Error"
unset LINGBOT_KV_CACHE_F16

echo "===== S1 balanced strict vulkan f16"
/usr/bin/time -f "S1_WALL_%es" cpp_ggml/build-vulkan/lingbot-map-cli \
  cpp_ggml/models/gguf/lingbot-map-f16.gguf /tmp/frames286.bin Vulkan0 294 518 /tmp/bal_f16_vk.lbo 286 \
  --kv-f16 strict --kv-scale 8 --kv-window 64 --scale-frames 8 2>&1 | grep -E "S1_WALL|RESULT|error"

echo "===== S2 balanced flash cuda f16"
/usr/bin/time -f "S2_WALL_%es" cpp_ggml/build-cuda/lingbot-map-cli \
  cpp_ggml/models/gguf/lingbot-map-f16.gguf /tmp/frames286.bin CUDA0 294 518 /tmp/bal_f16_flash_cuda.lbo 286 \
  --kv-f16 flash --kv-scale 8 --kv-window 64 --scale-frames 8 2>&1 | grep -E "S2_WALL|RESULT|error"

echo "===== S3 balanced flash vulkan f16"
/usr/bin/time -f "S3_WALL_%es" cpp_ggml/build-vulkan/lingbot-map-cli \
  cpp_ggml/models/gguf/lingbot-map-f16.gguf /tmp/frames286.bin Vulkan0 294 518 /tmp/bal_f16_flash_vk.lbo 286 \
  --kv-f16 flash --kv-scale 8 --kv-window 64 --scale-frames 8 2>&1 | grep -E "S3_WALL|RESULT|error"

echo "===== P1 PyTorch long fp32 streaming wall"
/usr/bin/time -f "P1_WALL_%es" "$PY" cpp_ggml/scripts/run_pytorch_reference.py \
  cpp_ggml/models/pytorch/lingbot-map-long.pt /tmp/frames286.npy /tmp/ref_ckpt_long_p1.npz \
  --device cuda --streaming --scale-frames 8 --kv-cache-scale 8 --kv-cache-window 64 2>&1 | \
  grep -E "P1_WALL|wrote|it/s\]$"

echo "===== P2 PyTorch long bf16 deployment wall + reference"
/usr/bin/time -f "P2_WALL_%es" "$PY" cpp_ggml/scripts/run_pytorch_reference.py \
  cpp_ggml/models/pytorch/lingbot-map-long.pt /tmp/frames286.npy /tmp/ref_long_bf16.npz \
  --device cuda --streaming --scale-frames 8 --kv-cache-scale 8 --kv-cache-window 64 \
  --autocast-bf16 2>&1 | grep -E "P2_WALL|wrote|it/s\]$"

echo "===== ROUND2 GPU DONE"
