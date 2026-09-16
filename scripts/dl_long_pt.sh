#!/usr/bin/env bash
# Download the upstream lingbot-map-long.pt checkpoint for checkpoint-level
# validation; verify the documented sha256 before renaming into place.
set -uo pipefail
DIR=/home/asher/develop/code/dl/lingbot-map-ggml/cpp_ggml/models/pytorch
URL=https://huggingface.co/robbyant/lingbot-map/resolve/main/lingbot-map-long.pt
EXPECT_SHA=832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409
EXPECT_BYTES=4632303465

curl -L --fail --retry 5 --retry-delay 5 -C - -o "$DIR/lingbot-map-long.pt.part" "$URL" || { echo "DOWNLOAD FAILED"; exit 1; }
BYTES=$(stat -c%s "$DIR/lingbot-map-long.pt.part")
SHA=$(sha256sum "$DIR/lingbot-map-long.pt.part" | cut -d' ' -f1)
echo "bytes=$BYTES (expect $EXPECT_BYTES)"
echo "sha256=$SHA"
if [ "$SHA" != "$EXPECT_SHA" ] || [ "$BYTES" != "$EXPECT_BYTES" ]; then
  echo "SHA256/SIZE MISMATCH — keeping the .part file for inspection"; exit 1
fi
mv "$DIR/lingbot-map-long.pt.part" "$DIR/lingbot-map-long.pt"
echo "OK: $DIR/lingbot-map-long.pt"
