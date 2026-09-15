#!/usr/bin/env bash
# Build and validate one GGUF/backend pair against an independent PyTorch GGUF
# reference, at the native-aspect profile (518x294, scale=8/window=64) — the
# same profile the one-click GUI and every parity row in
# cpp_ggml/benchmarks/validation_report.md use.
#
# Usage:
#   bash cpp_ggml/scripts/run_e2e.sh cuda q8              # 286 frames, default
#   bash cpp_ggml/scripts/run_e2e.sh vulkan f16 40        # quick 40-frame gate
#   bash cpp_ggml/scripts/run_e2e.sh vulkan f16 40 long   # lingbot-map-long variant
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BACKEND=${1:-cuda}
DTYPE=${2:-q8}
FRAMES=${3:-286}
# VARIANT: empty = balanced lingbot-map checkpoint; "long" = lingbot-map-long.
VARIANT=${4:-}
TAG="${VARIANT:+${VARIANT}_}"

case "$BACKEND" in
  cuda) BUILD="$ROOT/cpp_ggml/build-cuda"; DEVICE=CUDA0; CMAKE_ARGS=(-DLINGBOT_GGML_CUDA=ON -DLINGBOT_GGML_CUDA_ARCHITECTURES=86-real -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc) ;;
  vulkan) BUILD="$ROOT/cpp_ggml/build-vulkan"; DEVICE=Vulkan0; CMAKE_ARGS=(-DLINGBOT_GGML_VULKAN=ON) ;;
  *) echo "usage: $0 {cuda|vulkan} {q8|f16} [frames] [variant]" >&2; exit 2 ;;
esac

MODEL="$ROOT/cpp_ggml/models/gguf/lingbot-map-${VARIANT:+${VARIANT}-}${DTYPE}.gguf"
SCENE="$ROOT/example/courthouse"
OUT="$ROOT/cpp_ggml/benchmarks/current_${TAG}${DTYPE}_${BACKEND}_courthouse.csv"
EFFECT="$ROOT/cpp_ggml/benchmarks/current_${TAG}${DTYPE}_${BACKEND}_courthouse.png"
EXPORT="$ROOT/cpp_ggml/benchmarks/current_${TAG}${DTYPE}_${BACKEND}_courthouse"

test -f "$MODEL" || { echo "missing $MODEL; see cpp_ggml/models/MODEL_CARD.md" >&2; exit 2; }
test -d "$SCENE" || { echo "missing official scene $SCENE" >&2; exit 2; }

# Probe for an interpreter with the full stack (numpy/PIL for the GGML side,
# torch/einops for the PyTorch reference). The bare `python3` usually lacks
# torch, so scan conda environments the same way run_gui.sh does.
PYTHON=${PYTHON:-}
if [ -z "$PYTHON" ]; then
  for cand in python3 python "$HOME"/anaconda3/envs/*/bin/python "$HOME"/miniconda3/envs/*/bin/python; do
    if command -v "$cand" > /dev/null 2>&1 || [ -x "$cand" ]; then
      if "$cand" -c "import numpy, PIL, torch, einops" > /dev/null 2>&1; then PYTHON="$cand"; break; fi
    fi
  done
fi
[ -n "$PYTHON" ] || { echo "error: no python with numpy/PIL/torch/einops found; set PYTHON=/path/to/python" >&2; exit 2; }
echo "Using Python: $PYTHON"

cmake -S "$ROOT/cpp_ggml" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release "${CMAKE_ARGS[@]}"
cmake --build "$BUILD" --parallel 6

# run_reconstruction.py now derives the height from the official crop rule
# (aspect-preserving, snapped to the patch grid), so we do not pass --height.
# It also runs the persistent F16 KV cache by default (LINGBOT_KV_CACHE_F16=1),
# matching the upstream scale=8/window=64 profile.
python3_in_use="$PYTHON"
"$python3_in_use" "$ROOT/cpp_ggml/scripts/run_reconstruction.py" \
  --build "$BUILD" --backend "$DEVICE" --model "$MODEL" --reference-gguf "$MODEL" \
  --scene "$SCENE" --frames "$FRAMES" --out "$OUT" --effect-out "$EFFECT" \
  --full-export-prefix "$EXPORT" --pose-tol 0.001 --depth-tol 0.003

# The elementwise gate reads the same LBO the run just wrote; the native
# courtyard aspect is 294x518 (width x height), which is the identity output of
# the official crop rule for these frames.
"$python3_in_use" "$ROOT/cpp_ggml/scripts/compare_parity.py" /tmp/lingbot_scene.lbo /tmp/lingbot_scene_ref.npz \
  --atol 0.001 --rtol 0.001
"$python3_in_use" "$ROOT/cpp_ggml/scripts/compare_postprocess.py" /tmp/lingbot_scene.lbo \
  --height 294 --width 518
