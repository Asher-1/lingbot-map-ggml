#!/usr/bin/env bash
# Long-real comparison campaign matrix runner (Stage 3).
#
#   run_matrix.sh DATASET [PHASE]
#     DATASET: lingbo_world | drive | indoor
#     PHASE:   refs | flash | strict | all   (default all)
#
# Rows (per dataset), official streaming profile scale=8/window=64 and the
# official auto keyframe_interval = ceil(N/320) passed explicitly to BOTH
# engines (deterministic):
#   refs  : 8 PyTorch rows — {long,bal} x {fp32 checkpoint, bf16 deployment,
#           f16-mirror, q8-mirror} (run_pytorch_reference.py)
#   flash : 12 GGML rows — {long,bal} x {f32,f16,q8} x {cuda,vulkan} (--kv-f16 flash)
#   strict: the same 12 GGML cells with --kv-f16 strict
#
# Resumable: a row whose metrics JSON exists with exit_code 0 is skipped.
# Every row is wrapped in monitor.py (wall + GPU/RSS peaks -> .metrics.json).
set -uo pipefail
ROOT=/home/asher/develop/code/dl/lingbot-map-ggml
cd "$ROOT"
PY="$HOME/anaconda3/envs/python3.12/bin/python"
MON="$PY cpp_ggml/scripts/long_real/monitor.py"
DATA=/tmp/long_real
RUNS=$DATA/runs
GGUF=cpp_ggml/models/gguf
PT=cpp_ggml/models/pytorch

DATASET="${1:?dataset: lingbo_world|drive|indoor}"
PHASE="${2:-all}"
META="$DATA/$DATASET.meta.json"
[ -f "$META" ] || { echo "missing $META (run prepare_data.py first)"; exit 1; }
read -r N H W KF < <("$PY" - "$META" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
print(m["n_frames"], m["height"], m["width"], m["auto_keyframe_interval"])
EOF
)
BIN="$DATA/$DATASET.bin"
NPY="$DATA/$DATASET.npy"
OUT="$RUNS/$DATASET"
mkdir -p "$OUT"
echo "== dataset=$DATASET N=$N ${W}x${H} auto_kf_interval=$KF phase=$PHASE"

declare -A PTW=( [long]="$PT/lingbot-map-long.pt" [bal]="$PT/lingbot-map.pt" )
declare -A GGW=( [long]="$GGUF/lingbot-map-long" [bal]="$GGUF/lingbot-map" )
declare -A BUILD=( [cuda]=cpp_ggml/build-cuda [vulkan]=cpp_ggml/build-vulkan )
declare -A DEV=( [cuda]=CUDA0 [vulkan]=Vulkan0 )

wait_gpu() { # block while another process holds the GPU (e.g. api_server)
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-0}" -lt 6000 ] && break
    echo "   [gpu busy: ${used} MiB] waiting 60s..."
    sleep 60
  done
}

done_ok() { [ -f "$1" ] && grep -q '"exit_code": 0' "$1" 2>/dev/null; }

run_pt() { # CKPT KIND EXTRACTION_FLAGS...
  local ckpt=$1 kind=$2; shift 2
  local tag="pt_${ckpt}_${kind}"
  local m="$OUT/$tag.metrics.json"
  if done_ok "$m"; then echo "   skip $tag"; return 0; fi
  wait_gpu
  echo "== PT $tag (N=$N kf=$KF)"
  $MON --out "$m" --label "$DATASET/$tag" -- \
    "$PY" cpp_ggml/scripts/run_pytorch_reference.py "${PTW[$ckpt]}" "$NPY" \
    "$OUT/$tag.npz" --device cuda --streaming --scale-frames 8 \
    --kv-cache-scale 8 --kv-cache-window 64 --keyframe-interval "$KF" "$@" \
    2>&1 | grep -E "MONITOR|wrote|Error|error" || echo "PT FAILED: $tag"
  rm -f "$OUT/$tag.npz.tmp"
}

run_ggml() { # CKPT FMT MODE BACKEND
  local ckpt=$1 fmt=$2 mode=$3 backend=$4
  local tag="ggml_${ckpt}_${fmt}_${mode}_${backend}"
  local m="$OUT/$tag.metrics.json"
  if done_ok "$m"; then echo "   skip $tag"; return 0; fi
  wait_gpu
  echo "== GGML $tag (N=$N kf=$KF)"
  $MON --out "$m" --label "$DATASET/$tag" -- \
    "${BUILD[$backend]}/lingbot-map-cli" "${GGW[$ckpt]}-${fmt}.gguf" "$BIN" \
    "${DEV[$backend]}" "$H" "$W" "$OUT/$tag.lbo" "$N" \
    --kv-f16 "$mode" --kv-scale 8 --kv-window 64 --scale-frames 8 \
    --kv-total "$N" --keyframe-interval "$KF" \
    2>&1 | grep -E "MONITOR|RESULT|error|failed" || echo "GGML FAILED: $tag"
}

# ---- refs ---------------------------------------------------------------
if [ "$PHASE" = all ] || [ "$PHASE" = refs ]; then
  run_pt long fp32
  run_pt long bf16 --autocast-bf16
  run_pt long mf16 --gguf "$GGUF/lingbot-map-long-f16.gguf"
  run_pt long mq8 --gguf "$GGUF/lingbot-map-long-q8.gguf"
  run_pt bal fp32
  run_pt bal bf16 --autocast-bf16
  run_pt bal mf16 --gguf "$GGUF/lingbot-map-f16.gguf"
  run_pt bal mq8 --gguf "$GGUF/lingbot-map-q8.gguf"
fi

# ---- flash (deployment matrix) ------------------------------------------
if [ "$PHASE" = all ] || [ "$PHASE" = flash ]; then
  for ckpt in long bal; do
    for fmt in f16 q8 f32; do
      for backend in cuda vulkan; do
        run_ggml "$ckpt" "$fmt" flash "$backend"
      done
    done
  done
fi

# ---- strict (bit-level parity route) -------------------------------------
if [ "$PHASE" = all ] || [ "$PHASE" = strict ]; then
  for ckpt in long bal; do
    for fmt in f16 q8 f32; do
      for backend in cuda vulkan; do
        run_ggml "$ckpt" "$fmt" strict "$backend"
      done
    done
  done
fi
echo "== MATRIX PHASE $PHASE DONE: $DATASET"
