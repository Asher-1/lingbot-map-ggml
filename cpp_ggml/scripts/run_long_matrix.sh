#!/usr/bin/env bash
# Full long-checkpoint validation matrix on the 286-frame courthouse stream:
#   per (backend, dtype): strict e2e gate (run_e2e.sh: parity + postprocess +
#   effect PNG + full export) followed by a timed flash run whose output is
#   parity-checked against the same decoded-GGUF reference the strict run
#   just produced (reference stays in /tmp between the two).
set -uo pipefail
ROOT=/home/asher/develop/code/dl/lingbot-map-ggml
cd "$ROOT"
export PYTHON="$HOME/anaconda3/envs/python3.12/bin/python"
FRAMES=286

# 286-frame input, generated once with the official crop rule (518x294).
"$PYTHON" - <<'EOF'
import numpy as np
from PIL import Image
from pathlib import Path
files = sorted(Path('example/courthouse').glob('*.png'))[:286]
w = 518
src_w, src_h = Image.open(files[0]).size
h = round(src_h * (w / src_w) / 14) * 14
frames = np.stack([np.asarray(Image.open(p).convert('RGB').resize((w, h), Image.Resampling.BICUBIC),
                   dtype=np.float32).transpose(2, 0, 1) / 255.0 for p in files])
frames.tofile('/tmp/frames286.bin')
print(f'input: {frames.shape} -> /tmp/frames286.bin')
EOF

for backend in cuda vulkan; do
  case "$backend" in
    cuda)   BUILD="$ROOT/cpp_ggml/build-cuda";   DEVICE=CUDA0 ;;
    vulkan) BUILD="$ROOT/cpp_ggml/build-vulkan"; DEVICE=Vulkan0 ;;
  esac
  for dtype in f16 q8 f32; do
    echo "############ STRICT $backend $dtype long $FRAMES ############"
    bash cpp_ggml/scripts/run_e2e.sh "$backend" "$dtype" "$FRAMES" long 2>&1 | \
      grep -E "RECONSTRUCTION|PARITY|POSTPROCESS|FAIL|Error|error:" || echo "E2E FAILED: $backend $dtype"

    echo "############ FLASH $backend $dtype long $FRAMES ############"
    LBO="/tmp/flash_long_${backend}_${dtype}.lbo"
    /usr/bin/time -f "FLASH_WALL_%es" "$BUILD/lingbot-map-cli" \
      "cpp_ggml/models/gguf/lingbot-map-long-${dtype}.gguf" /tmp/frames286.bin "$DEVICE" 294 518 "$LBO" $FRAMES \
      --kv-f16 flash --kv-scale 8 --kv-window 64 --scale-frames 8 2>&1 | \
      grep -E "RESULT|FLASH_WALL|error|failed" || echo "FLASH FAILED: $backend $dtype"
    "$PYTHON" cpp_ggml/scripts/compare_parity.py "$LBO" /tmp/lingbot_scene_ref.npz \
      --atol 0.001 --rtol 0.001 2>&1 | tail -5 || echo "FLASH PARITY FAILED: $backend $dtype"
  done
done
echo "############ MATRIX DONE ############"
