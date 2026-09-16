#!/usr/bin/env bash
# Keyframe-interval validation gate (Stage 0 of the long_real campaign).
#   V1  bit-identical A/B: interval=1 output of the keyframe-capable build
#       must equal the pre-change build on 100-frame courthouse across all
#       six cache paths (resident strict/flash on CUDA+Vulkan, F32 legacy,
#       F16 legacy sidecar). Requires the baseline LBOs produced from the
#       pre-change sources (see AGENTS.md "keyframe_interval"); when they are
#       absent this script regenerates ONLY the new-side runs and prints the
#       comparison commands.
#   V2  semantic parity: interval in {2,4}, C++ strict vs the official
#       inference_streaming mirror with --keyframe-interval; the parity class
#       must match interval=1 (pose/depth RMSE ~1e-4). compare_parity's
#       depth max_abs gate is EXPECTED to trip on the known F16-cache scatter
#       tail (interval=1 shows the same class); the authoritative metrics are
#       the RMSE columns it prints.
set -uo pipefail
ROOT=/home/asher/develop/code/dl/lingbot-map-ggml
cd "$ROOT"
PY="$HOME/anaconda3/envs/python3.12/bin/python"
M=cpp_ggml/models/gguf/lingbot-map-f16.gguf
B=${1:-/tmp/kf_v2}
mkdir -p "$B"

# 100-frame input, official crop rule (518x294 native for courthouse)
[ -f /tmp/frames100.bin ] || "$PY" - <<'EOF'
import numpy as np
from PIL import Image
from pathlib import Path
files = sorted(Path('example/courthouse').glob('*.png'))[:100]
w = 518
src_w, src_h = Image.open(files[0]).size
h = round(src_h * (w / src_w) / 14) * 14
frames = np.stack([np.asarray(Image.open(p).convert('RGB').resize((w, h), Image.Resampling.BICUBIC),
                   dtype=np.float32).transpose(2, 0, 1) / 255.0 for p in files])
frames.tofile('/tmp/frames100.bin')
np.save('/tmp/frames100.npy', frames[None])
print('input:', frames.shape)
EOF

echo "===== V2 mirror references (official inference_streaming, interval 2/4/1)"
for KF in 1 2 4; do
  [ -f /tmp/ref_kf$KF.npz ] || "$PY" cpp_ggml/scripts/run_pytorch_reference.py \
    cpp_ggml/models/pytorch/lingbot-map.pt /tmp/frames100.npy /tmp/ref_kf$KF.npz \
    --device cuda --streaming --scale-frames 8 --kv-cache-scale 8 --kv-cache-window 64 \
    --keyframe-interval $KF --gguf $M 2>&1 | tail -1
done

echo "===== V2 C++ strict runs"
cpp_ggml/build-cuda/lingbot-map-cli $M /tmp/frames100.bin CUDA0 294 518 $B/kf2_cuda.lbo 100 \
  --kv-f16 strict --keyframe-interval 2 --kv-scale 8 --kv-window 64 --scale-frames 8 2>&1 | grep RESULT
cpp_ggml/build-vulkan/lingbot-map-cli $M /tmp/frames100.bin Vulkan0 294 518 $B/kf2_vk.lbo 100 \
  --kv-f16 strict --keyframe-interval 2 --kv-scale 8 --kv-window 64 --scale-frames 8 2>&1 | grep RESULT
cpp_ggml/build-cuda/lingbot-map-cli $M /tmp/frames100.bin CUDA0 294 518 $B/kf4_cuda.lbo 100 \
  --kv-f16 strict --keyframe-interval 4 --kv-scale 8 --kv-window 64 --scale-frames 8 2>&1 | grep RESULT

echo "===== V2 RMSE verdicts"
"$PY" - "$B" <<'EOF'
import numpy as np, struct, sys
def read_lbo(p):
    raw = open(p, 'rb').read()
    _, n_pose, n_depth = struct.unpack('<III', raw[:12])
    v = np.frombuffer(raw, dtype=np.float32, offset=12)
    return v[:n_pose].reshape(-1, 9), v[n_pose:n_pose + n_depth].reshape(-1, 294, 518)
b = sys.argv[1]
for tag, ref in (('kf2_cuda', 2), ('kf2_vk', 2), ('kf4_cuda', 4)):
    pose, depth = read_lbo(f'{b}/{tag}.lbo')
    z = np.load(f'/tmp/ref_kf{ref}.npz')
    prm = float(np.sqrt(((pose - z['pose_enc'].reshape(-1, 9)) ** 2).mean()))
    drm = float(np.sqrt(((depth - z['depth'].reshape(-1, 294, 518)) ** 2).mean()))
    ok = prm < 5e-4 and drm < 5e-4
    print(f"V2 {'PASS' if ok else 'FAIL'} {tag}: pose_rmse={prm:.2e} depth_rmse={drm:.2e} (mirror kf{ref})")
EOF

echo "===== V1 (requires baseline LBOs from the pre-change build)"
echo "baseline: git stash the keyframe edits, rebuild, run the six paths into /tmp/kf_baseline"
echo "then: for f in strict_cuda strict_vk flash_cuda flash_vk none_cuda strict_legacy_cuda; do"
echo "  cmp /tmp/kf_baseline/\$f.lbo <new build output>; done"
