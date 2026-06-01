#!/usr/bin/env bash
# Download nvidia/Fixer weights from HuggingFace into the user's HF cache.
# Used by threedgrut/correction/difix.py (DifixPostProcessor).
#
# Output layout:
#   $HF_HOME/nvidia-Fixer/
#     pretrained_fixer.pkl
#     models/base/model_fast_tokenizer.pt
#     models/base/tokenizer_fast.pth
#
# License: weights are under the NVIDIA Open Model License (commercial use
# permitted, redistribution restricted). Do not commit them to git.

set -euo pipefail

: "${HF_HOME:=$HOME/.cache/huggingface}"
TARGET_DIR="$HF_HOME/nvidia-Fixer"

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: 'hf' CLI not found. Install with: pip install -U huggingface_hub" >&2
  exit 1
fi

mkdir -p "$TARGET_DIR"
echo "==> Downloading nvidia/Fixer -> $TARGET_DIR"

hf download nvidia/Fixer --local-dir "$TARGET_DIR"

echo "==> Downloaded files:"
find "$TARGET_DIR" -maxdepth 3 -type f -exec ls -lh {} \;

echo "==> Total size:"
du -sh "$TARGET_DIR"

echo "==> Done. Point DifixPostProcessor at:"
echo "      ckpt_path=$TARGET_DIR/pretrained_fixer.pkl"
