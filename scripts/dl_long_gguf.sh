#!/usr/bin/env bash
# Download the lingbot-map-long GGUF series into cpp_ggml/models/gguf/.
# curl -C - resumes partial transfers; sizes are verified afterwards.
set -uo pipefail
GGUF_DIR=/home/asher/develop/code/dl/lingbot-map-ggml/cpp_ggml/models/gguf
BASE=https://huggingface.co/Asher-1/lingbot-map-gguf/resolve/main
for f in lingbot-map-long-q8.gguf lingbot-map-long-f16.gguf lingbot-map-long-f32.gguf; do
  echo "=== $f"
  curl -L --fail --retry 5 --retry-delay 5 -C - -o "$GGUF_DIR/$f.part" "$BASE/$f" \
    && mv "$GGUF_DIR/$f.part" "$GGUF_DIR/$f" \
    || { echo "DOWNLOAD FAILED: $f"; exit 1; }
done
echo "=== all downloads complete"
ls -la "$GGUF_DIR"/lingbot-map-long-*.gguf
