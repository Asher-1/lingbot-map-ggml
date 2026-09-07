#!/usr/bin/env bash
# Regenerate the F16-cache full-stream reconstruction evidence from the final
# validation LBOs (see validation_report.md "Persistent F16 KV Cache"):
#   - per-run colored point clouds as PLY (PyTorch reference + GGML)
#   - side-by-side cloud comparison PNGs
#   - localization trajectory PNGs (reference vs CUDA vs Vulkan camera centers)
#   - per-frame depth effect PNGs (CUDA rows)
# Inputs default to the /tmp artifacts of the 2026-09-09 pass; override via env.
# The REF* npz files are the cache-precision-aware references produced by
# `run_pytorch_reference.py --gguf <q8 gguf> --streaming` (GGUF weights decoded
# back into PyTorch), so the comparisons isolate graph parity from the q8
# weight format's own loss.
#
# /tmp is not durable. To rebuild the headline native-aspect inputs (518x294 =
# the identity output of the official crop for these frames, ~25 min per
# backend plus ~5 min for the reference):
#
#   python - <<'PY'
#   import numpy as np; from pathlib import Path; from PIL import Image
#   f = sorted(Path('example/courthouse').glob('*.png'))[:286]
#   a = np.stack([np.asarray(Image.open(p).convert('RGB').resize((518, 294),
#           Image.Resampling.BICUBIC), np.float32).transpose(2,0,1)/255 for p in f])
#   np.save('/tmp/lingbot_scene_native286.npy', a[None]); a.tofile('/tmp/native286.bin')
#   PY
#   for b in cuda vulkan; do
#     LINGBOT_KV_CACHE_F16=1 LINGBOT_KV_CACHE_SCALE=8 LINGBOT_KV_CACHE_WINDOW=64 \
#     LINGBOT_NUM_SCALE_FRAMES=8 cpp_ggml/build-$b/lingbot-map-cli \
#       cpp_ggml/models/gguf/lingbot-map-q8.gguf /tmp/native286.bin \
#       "$([ $b = cuda ] && echo CUDA0 || echo Vulkan0)" 294 518 \
#       "/tmp/full2_native_${b/vulkan/vk}.lbo" 286
#   done
#   LINGBOT_KV_CACHE_F16=1 python cpp_ggml/scripts/run_pytorch_reference.py \
#     cpp_ggml/models/pytorch/lingbot-map.pt /tmp/lingbot_scene_native286.npy \
#     /tmp/ref16man_native.npz --device cuda --streaming --scale-frames 8 \
#     --kv-cache-scale 8 --kv-cache-window 64 \
#     --gguf cpp_ggml/models/gguf/lingbot-map-q8.gguf
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${PY:-/home/asher/anaconda3/envs/python3.12/bin/python}
SC=${SC:-example/courthouse}
OUT=${OUT:-cpp_ggml/benchmarks}
F=${F:-286}
REFNAT=${REFNAT:-/tmp/ref16man_native.npz}
REF518=${REF518:-/tmp/ref16man_518.npz}
REF392=${REF392:-/tmp/ref16man_392.npz}
REF378=${REF378:-/tmp/ref16man_378.npz}

export_run () { # profile_suffix ref lbo label height width
    local sfx=$1 ref=$2 lbo=$3 label=$4 h=$5 w=$6
    $PY cpp_ggml/scripts/export_full_reconstruction.py \
        --scene "$SC" --pytorch "$ref" --ggml-lbo "$lbo" --frames "$F" \
        --height "$h" --width "$w" \
        --out-prefix "$OUT/reconstruction_full_f16cache_${label}_courthouse${sfx}_20260909" \
        --ggml-label "f16cache-${label%-*}"
}

effect_run () { # sfx ref lbo label height width
    local sfx=$1 ref=$2 lbo=$3 label=$4 h=$5 w=$6
    $PY cpp_ggml/scripts/plot_reconstruction_effect.py \
        --scene "$SC" --pytorch "$ref" --ggml-lbo "$lbo" --frames "$F" \
        --height "$h" --width "$w" \
        --ggml-label "f16cache-${label%-*}" \
        --output "$OUT/reconstruction_effect_f16cache_${label%-*}_courthouse${sfx}_full_20260909.png"
}

traj_run () { # sfx ref lbo_cuda lbo_vk title
    local sfx=$1 ref=$2 lcuda=$3 lvk=$4 title=$5
    $PY cpp_ggml/scripts/plot_trajectory_compare.py \
        --pytorch "$ref" --frames "$F" \
        --lbo "$lcuda" --lbo "$lvk" --label CUDA0 --label Vulkan0 \
        --title "$title" \
        --out "$OUT/reconstruction_trajectory_f16cache_courthouse${sfx}_20260909.png"
}

# 518x294 = the native courthouse aspect: the identity output of the official
# load_and_preprocess_images(mode="crop", image_size=518) for these frames, so
# no resampling at all. This is the headline reconstruction row; the square /
# portrait profiles below stretch the frames and are backend-parity evidence.
# The cloud prefixes carry the profile in the backend label (cuda-native), so
# they take an empty suffix; the trajectory/effect names key off the suffix.
export_run ""            "$REFNAT" /tmp/full2_native_cuda.lbo cuda-native   294 518
export_run ""            "$REFNAT" /tmp/full2_native_vk.lbo   vulkan-native 294 518
traj_run  _native        "$REFNAT" /tmp/full2_native_cuda.lbo /tmp/full2_native_vk.lbo "courthouse 518x294 (native aspect, official crop) scale=8 window=64"
effect_run _native       "$REFNAT" /tmp/full2_native_cuda.lbo cuda-native   294 518

# 518x518, scale=1 / window=4
export_run ""            "$REF518" /tmp/full2_518_cuda.lbo cuda-518    518 518
export_run ""            "$REF518" /tmp/full2_518_vk.lbo  vulkan-518  518 518
traj_run  ""             "$REF518" /tmp/full2_518_cuda.lbo /tmp/full2_518_vk.lbo "courthouse 518x518 scale=1 window=4"
effect_run ""            "$REF518" /tmp/full2_518_cuda.lbo cuda-518    518 518

# 392x392, scale=8 / window=64
export_run 392           "$REF392" /tmp/full2_392_cuda.lbo cuda-392    392 392
export_run 392           "$REF392" /tmp/full2_392_vk.lbo  vulkan-392  392 392
traj_run  392            "$REF392" /tmp/full2_392_cuda.lbo /tmp/full2_392_vk.lbo "courthouse 392x392 scale=8 window=64"
effect_run 392           "$REF392" /tmp/full2_392_cuda.lbo cuda-392    392 392

# 378 wide x 518 high (portrait), scale=8 / window=64. Kept as a parity row
# only: it stretches the 518x294 source frames by ~2.4x, so the fused cloud is
# geometrically distorted no matter which backend produces it.
export_run 378           "$REF378" /tmp/full2_378_cuda.lbo cuda-378    518 378
export_run 378           "$REF378" /tmp/full2_378_vk.lbo  vulkan-378  518 378
traj_run  378            "$REF378" /tmp/full2_378_cuda.lbo /tmp/full2_378_vk.lbo "courthouse 378x518 (WxH) scale=8 window=64"
effect_run 378           "$REF378" /tmp/full2_378_cuda.lbo cuda-378    518 378

echo ALL_RECON_EVIDENCE_DONE
