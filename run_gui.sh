#!/usr/bin/env bash
# One-command 3D reconstruction GUI, for either inference engine.
#
# Both engines open the same official viser viewer (lingbot_map.vis.PointCloudViewer)
# with the same preprocessing (official crop to --image_size, 518 by default) and
# the same streaming profile (scale=8 / window=64), so the two reconstructions are
# directly comparable — only the inference engine differs.
#
# Engines:
#   --engine ggml     native C++ GGML runtime (no PyTorch checkpoint needed),
#                     builds cpp_ggml on first run; a live viser view grows the
#                     reconstruction frame-by-frame during inference
#   --engine pytorch  official PyTorch pipeline (demo.py), needs the .pt checkpoint
#   --engine auto     pick per what this machine has: the official checkpoint
#                     plus a torch environment -> pytorch; otherwise ggml
#                     [default]
#
# The PyTorch engine auto-discovers the checkpoint (LINGBOT_CKPT env, then
# cpp_ggml/models/pytorch/*.pt, then *.pt next to this script) and falls back
# to the SDPA attention backend automatically when flashinfer is not installed.
#
# Usage:
#   bash run_gui.sh                                             # auto engine, courthouse
#   bash run_gui.sh --engine ggml --image_folder example/loop --frames 40
#   bash run_gui.sh --engine pytorch --image_folder example/courthouse
#   bash run_gui.sh --engine ggml --backend CUDA0               # picks build-cuda
#   bash run_gui.sh --engine ggml --gguf cpp_ggml/models/gguf/lingbot-map-f16.gguf
#   PYTHON=/path/to/python bash run_gui.sh --engine pytorch --mask_sky
#
# Outdoor scenes (sky in view) add --mask_sky: the ggml engine runs the
# NATIVE skyseg GGUF inside the C++ CLI (no onnxruntime), zeroing sky-pixel
# confidence; --sky_mask_dir caches the masks as PNGs and
# --sky_mask_visualization_dir writes official-style original|mask|overlay
# panels. The skyseg GGUF auto-resolves to
# cpp_ggml/models/gguf/lingbot-map-skyseg-f16.gguf.
#
# GGML defaults: --backend CUDA0 when a build-cuda exists (the accuracy/speed
# reference backend), else Vulkan0; the f16 GGUF (full end-to-end alignment
# with the official PyTorch pipeline) when it exists, else q8. Override via
# --backend / --gguf or the GGML_MODEL / GGML_BUILD env vars.
#
# Env overrides: PYTHON, GGML_MODEL, GGML_BUILD, LINGBOT_CKPT.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"

ENGINE=auto
GGUF_FLAG=""
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --engine) ENGINE="${2:-}"; shift 2 ;;
    --engine=*) ENGINE="${1#*=}"; shift ;;
    --gguf) GGUF_FLAG="${2:-}"; shift 2 ;;
    --gguf=*) GGUF_FLAG="${1#*=}"; shift ;;
    -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
case "$ENGINE" in
  ggml|pytorch|auto) ;;
  *) echo "error: --engine must be 'ggml', 'pytorch' or 'auto' (got '$ENGINE')" >&2; exit 2 ;;
esac

# ----------------------------------------------------------------------------
# Interpreter discovery: explicit PYTHON wins, then the project venv, then the
# system interpreters, then every conda-flavour env on this machine. No paths
# are hardcoded beyond the standard conda install locations.
# ----------------------------------------------------------------------------
PYTHON_CANDIDATES=(${PYTHON:+"$PYTHON"})
[ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ] && PYTHON_CANDIDATES+=("$VIRTUAL_ENV/bin/python")
[ -x "$ROOT/.venv/bin/python" ] && PYTHON_CANDIDATES+=("$ROOT/.venv/bin/python")
PYTHON_CANDIDATES+=(python3 python)
for base in "$HOME"/*conda*/envs/* "$HOME"/envs/*; do
  [ -x "$base/bin/python" ] && PYTHON_CANDIDATES+=("$base/bin/python")
done

# engine-specific module requirements (checked on the candidate interpreters)
GGML_MODULES="numpy, PIL, viser"
PT_MODULES="numpy, PIL, viser, torch, torch.nn.attention.flex_attention"

probe_python () { # $1 = modules expression; sets PYTHON_BIN
  PYTHON_BIN=""
  for cand in "${PYTHON_CANDIDATES[@]}"; do
    if command -v "$cand" > /dev/null 2>&1 || [ -x "$cand" ]; then
      if "$cand" -c "import $1" > /dev/null 2>&1; then PYTHON_BIN="$cand"; return 0; fi
    fi
  done
  return 1
}

# ----------------------------------------------------------------------------
# PyTorch checkpoint discovery (official engine). Order: explicit env, a
# --model_path already present in ARGS, the bundled models directory, then the
# repository root / a checkpoints/ folder.
# ----------------------------------------------------------------------------
find_ckpt () {
  [ -n "${LINGBOT_CKPT:-}" ] && [ -f "$LINGBOT_CKPT" ] && { CKPT="$LINGBOT_CKPT"; return 0; }
  local a; for a in ${ARGS[@]+"${ARGS[@]}"}; do
    if [ "${a%%=*}" = "--model_path" ]; then
      local v
      if [ "$a" = "--model_path" ]; then continue; fi # value form handled below
    fi
  done
  local i=-1
  for a in ${ARGS[@]+"${ARGS[@]}"}; do
    i=$((i+1))
    if [ "$a" = "--model_path" ] && [ $((i+1)) -lt ${#ARGS[@]} ]; then
      [ -f "${ARGS[$((i+1))]}" ] && { CKPT="${ARGS[$((i+1))]}"; return 0; }
    fi
    case "$a" in --model_path=*) [ -f "${a#*=}" ] && { CKPT="${a#*=}"; return 0; } ;; esac
  done
  local f
  for f in "$ROOT"/cpp_ggml/models/pytorch/lingbot-map*.pt "$ROOT"/cpp_ggml/models/pytorch/*.pt \
           "$ROOT"/lingbot-map*.pt "$ROOT"/checkpoints/*.pt; do
    [ -f "$f" ] && { CKPT="$f"; return 0; }
  done
  return 1
}

resolve_engine () {
  case "$ENGINE" in
    ggml|pytorch) RESOLVED="$ENGINE"; return 0 ;;
  esac
  # auto: prefer the official PyTorch pipeline when this machine can run it
  if find_ckpt && probe_python "$PT_MODULES"; then
    RESOLVED=pytorch
  else
    RESOLVED=ggml
  fi
}

resolve_engine
if [ "$ENGINE" = auto ]; then
  echo "Engine: $RESOLVED (auto-detected; force one with --engine ggml|pytorch)"
else
  RESOLVED="$ENGINE"
  echo "Engine: $RESOLVED"
fi

# ----------------------------------------------------------------------------
# Interpreter resolution for the chosen engine. The viewer extras are small
# enough to auto-install when only those are missing; torch is NOT
# auto-installed (multi-GB) — print the official pointer instead.
# ----------------------------------------------------------------------------
if [ "$RESOLVED" = pytorch ]; then
  if ! probe_python "$PT_MODULES"; then
    # maybe only viser is missing from an otherwise-torch env
    for cand in "${PYTHON_CANDIDATES[@]}"; do
      if command -v "$cand" > /dev/null 2>&1 || [ -x "$cand" ]; then
        if "$cand" -c "import numpy, PIL, torch, torch.nn.attention.flex_attention" > /dev/null 2>&1; then
          echo "Installing viewer dependencies (viser, trimesh) into $cand..."
          "$cand" -m pip install --quiet viser trimesh && probe_python "$PT_MODULES" && break
        fi
      fi
    done
  fi
  if [ -z "$PYTHON_BIN" ]; then
    echo "error: no python satisfying '$PT_MODULES' found for the PyTorch engine." >&2
    echo "  Install the official environment first, e.g.:" >&2
    echo "    pip install -e .            # torch >= 2.5 required (flex_attention)" >&2
    echo "    pip install viser trimesh    # viewer extras" >&2
    echo "  Then rerun with: PYTHON=/path/to/python bash run_gui.sh --engine pytorch" >&2
    echo "  (or drop the checkpoint and use --engine ggml, which needs no torch)" >&2
    exit 1
  fi
else
  if ! probe_python "$GGML_MODULES"; then
    for cand in "${PYTHON_CANDIDATES[@]}"; do
      if command -v "$cand" > /dev/null 2>&1 || [ -x "$cand" ]; then
        if "$cand" -c "import numpy, PIL" > /dev/null 2>&1; then
          echo "Installing viewer dependencies (viser, trimesh) into $cand..."
          "$cand" -m pip install --quiet viser trimesh && probe_python "$GGML_MODULES" && break
        fi
      fi
    done
  fi
  if [ -z "$PYTHON_BIN" ]; then
    echo "error: no python satisfying '$GGML_MODULES' found. Install first:" >&2
    echo "  pip install viser trimesh    # viewer only, into any python" >&2
    echo "Then rerun with: PYTHON=/path/to/python bash run_gui.sh --engine ggml" >&2
    exit 1
  fi
fi
echo "Using Python: $PYTHON_BIN"

# Default to the bundled official scene when the caller did not pick one.
case " ${ARGS[*]-} " in *" --image_folder "*|*" --video_path "*) ;;
  *) ARGS+=(--image_folder example/courthouse) ;;
esac

# Translate the shared convenience flags between the two engines so callers
# can use the same command line for either: `--frames N` is the ggml_demo
# spelling, `--first_k N` is the demo.py spelling.
translate_arg () { # $1 = from-flag, $2 = to-flag (rewrites ARGS in place)
  local out=() a i=0 n=${#ARGS[@]}
  while [ $i -lt $n ]; do
    a="${ARGS[$i]}"
    if [ "$a" = "$1" ] && [ $((i+1)) -lt $n ]; then
      out+=("$2" "${ARGS[$((i+1))]}"); i=$((i+2)); continue
    fi
    case "$a" in
      "$1="*) out+=("$2=${a#*=}"); i=$((i+1)); continue ;;
    esac
    out+=("$a"); i=$((i+1))
  done
  ARGS=("${out[@]}")
}

if [ "$RESOLVED" = pytorch ]; then
  if ! find_ckpt; then
    echo "error: PyTorch checkpoint not found (looked in \$LINGBOT_CKPT," >&2
    echo "  cpp_ggml/models/pytorch/*.pt, ./lingbot-map*.pt, ./checkpoints/*.pt)." >&2
    echo "  Download it and retry:" >&2
    echo "    huggingface-cli download robbyant/lingbot-map --local-dir cpp_ggml/models/pytorch" >&2
    echo "  or point at an existing one:" >&2
    echo "    LINGBOT_CKPT=/path/to/lingbot-map.pt bash run_gui.sh --engine pytorch" >&2
    exit 2
  fi
  echo "Checkpoint: $CKPT"
  # flashinfer is the demo default; fall back to SDPA automatically when it
  # is not installed so the engine still runs everywhere.
  if ! "$PYTHON_BIN" -c "import flashinfer" > /dev/null 2>&1; then
    case " ${ARGS[*]-} " in *" --use_sdpa "*|*" --no-use_sdpa "*) ;;
      *) echo "flashinfer not installed - using the SDPA attention backend (--use_sdpa)" ;;
    esac
    ARGS+=(--use_sdpa)
  fi
  if ! "$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available()" > /dev/null 2>&1; then
    echo "warning: CUDA is not available to this python - the PyTorch engine will run on CPU (very slow)." >&2
  fi
  translate_arg --frames --first_k
  exec "$PYTHON_BIN" "$ROOT/demo.py" --model_path "$CKPT" "${ARGS[@]}"
fi

# --mask_sky with the ggml engine needs a skyseg GGUF; fail loudly here
# rather than silently skipping the masking deep inside the demo.
case " ${ARGS[*]-} " in *" --mask_sky "*)
  SKYSEG_GGUF="$ROOT/cpp_ggml/models/gguf/lingbot-map-skyseg-f16.gguf"
  if [ "$ENGINE" != "pytorch" ] && [ ! -f "$SKYSEG_GGUF" ]; then
    echo "error: --mask_sky requested but the native skyseg GGUF is missing: $SKYSEG_GGUF" >&2
    echo "  build it from the official onnx (see cpp_ggml/scripts/convert_skyseg.py)," >&2
    echo "  or download lingbot-map-skyseg-f16.gguf next to the other GGUFs." >&2
    exit 2
  fi
  # default the mask cache/visualization dirs next to the image folder,
  # mirroring the official <folder>_sky_masks convention
  HAS_MDIR=0; HAS_VDIR=0
  for a in ${ARGS[@]+"${ARGS[@]}"}; do
    case "$a" in --sky_mask_dir) HAS_MDIR=1 ;; --sky_mask_visualization_dir) HAS_VDIR=1 ;; esac
  done
  [ $HAS_MDIR -eq 1 ] || ARGS+=(--sky_mask_dir "$(pwd)/ggml_sky_masks")
  [ $HAS_VDIR -eq 1 ] || ARGS+=(--sky_mask_visualization_dir "$(pwd)/ggml_sky_masks_vis")
  ;;
esac

# ----------------------------------------------------------------------------
# GGML engine: resolve the model and the build directory for the requested
# backend (build-<backend> when it exists, otherwise the default and a
# first-run build).
# ----------------------------------------------------------------------------
# Model priority: --gguf flag > GGML_MODEL env > f16 (full end-to-end
# alignment with the official PyTorch pipeline, pose 1.72e-04 / depth
# 4.72e-04 over 286 frames, see cpp_ggml/benchmarks/validation_report.md)
# > q8 (memory-saving default, present on fresh clones).
MODEL="${GGML_MODEL:-}"
[ -n "$GGUF_FLAG" ] && MODEL="$GGUF_FLAG"
if [ -z "$MODEL" ]; then
  for f in "$ROOT"/cpp_ggml/models/gguf/lingbot-map-f16.gguf \
           "$ROOT"/cpp_ggml/models/gguf/lingbot-map-q8.gguf; do
    [ -f "$f" ] && { MODEL="$f"; break; }
  done
fi
[ -n "$MODEL" ] || MODEL="$ROOT/cpp_ggml/models/gguf/lingbot-map-q8.gguf"
if [ ! -f "$MODEL" ]; then
  echo "error: GGUF model not found: $MODEL" >&2
  echo "  curl -L -o $MODEL \\" >&2
  echo "    https://huggingface.co/Asher-1/lingbot-map-gguf/resolve/main/lingbot-map-q8.gguf" >&2
  echo "  (f16 matches the official PyTorch accuracy end to end: --gguf .../lingbot-map-f16.gguf)" >&2
  exit 2
fi

# Backend default: CUDA0 when a CUDA build exists (the accuracy/speed
# reference backend), else Vulkan0 - which is also the first-run build
# default (fewer toolchain requirements). An explicit --backend always wins.
BACKEND="Vulkan0"
[ -f "$ROOT/cpp_ggml/build-cuda/lingbot-map-cli" ] && BACKEND="CUDA0"
i=-1
for a in ${ARGS[@]+"${ARGS[@]}"}; do
  i=$((i+1))
  if [ "$a" = "--backend" ] && [ $((i+1)) -lt ${#ARGS[@]} ]; then BACKEND="${ARGS[$((i+1))]}"; fi
  case "$a" in --backend=*) BACKEND="${a#*=}" ;; esac
done
BUILD="${GGML_BUILD:-}"
if [ -z "$BUILD" ]; then
  case "$BACKEND" in
    CUDA*|cuda*) [ -f "$ROOT/cpp_ggml/build-cuda/lingbot-map-cli" ] && BUILD="$ROOT/cpp_ggml/build-cuda" ;;
    *)           [ -f "$ROOT/cpp_ggml/build-vulkan/lingbot-map-cli" ] && BUILD="$ROOT/cpp_ggml/build-vulkan" ;;
  esac
  [ -n "$BUILD" ] || BUILD="$ROOT/cpp_ggml/build-vulkan"
fi
if [ ! -f "$BUILD/lingbot-map-cli" ]; then
  echo "Building the native GGML runtime (first run)..."
  if [ -n "${VULKAN_SDK:-}" ]; then export PATH="$VULKAN_SDK/bin:$PATH"; fi
  CMAKE_FLAGS=(-DCMAKE_BUILD_TYPE=Release)
  case "$BACKEND" in
    CUDA*|cuda*) CMAKE_FLAGS+=(-DLINGBOT_GGML_CUDA=ON) ;;
    *)           CMAKE_FLAGS+=(-DLINGBOT_GGML_VULKAN=ON) ;;
  esac
  cmake -S "$ROOT/cpp_ggml" -B "$BUILD" "${CMAKE_FLAGS[@]}"
  cmake --build "$BUILD" --parallel 8
fi
echo "GGUF: $MODEL"
echo "Build: $BUILD"

translate_arg --first_k --frames
# Pass the resolved backend down explicitly: ggml_demo's own default is
# Vulkan0, which would mismatch a CUDA build directory chosen here.
exec "$PYTHON_BIN" "$ROOT/ggml_demo.py" --model "$MODEL" --build "$BUILD" --backend "$BACKEND" "${ARGS[@]}"
