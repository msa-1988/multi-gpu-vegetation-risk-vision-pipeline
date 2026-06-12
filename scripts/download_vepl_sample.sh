#!/usr/bin/env bash
set -euo pipefail

ARCHIVE="data/raw/vepl/TESELLATED_WITHOUT_AUGMENTATION.zip"
TARGET_DIR="data/vepl"
URL="https://zenodo.org/api/records/7800234/files/TESELLATED_WITHOUT_AUGMENTATION.zip/content"

mkdir -p "$(dirname "$ARCHIVE")" "$TARGET_DIR"

if [[ ! -f "$ARCHIVE" ]]; then
  curl -L --fail --continue-at - --output "$ARCHIVE" "$URL"
else
  echo "archive already exists: $ARCHIVE"
fi

unzip -oq "$ARCHIVE" -d "$TARGET_DIR"
find "$TARGET_DIR" -maxdepth 3 -type f | sed -n '1,20p'
