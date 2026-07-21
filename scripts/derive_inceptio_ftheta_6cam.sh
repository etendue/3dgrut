#!/usr/bin/env bash
# Derive the matched six-camera native-FTheta NCore dataset from source data.

set -Eeuo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_MANIFEST="${SOURCE_MANIFEST:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
LINK_MODE="${LINK_MODE:-hardlink}"

[ -n "$SOURCE_MANIFEST" ] || { echo "ERROR: SOURCE_MANIFEST is required" >&2; exit 1; }
[ -n "$OUTPUT_DIR" ] || { echo "ERROR: OUTPUT_DIR is required" >&2; exit 1; }

if [ -n "${PYTHON_BIN:-}" ]; then
  :
elif [ -x "/home/inceptio/miniforge3/envs/3dgrut2/bin/python" ]; then
  PYTHON_BIN="/home/inceptio/miniforge3/envs/3dgrut2/bin/python"
elif [ -x "$REPO_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$REPO_DIR/.venv/bin/python"
else
  PYTHON_BIN=python3
fi

cd "$REPO_DIR"
"$PYTHON_BIN" scripts/derive_inceptio_ftheta_ncore.py \
  --source-manifest "$SOURCE_MANIFEST" \
  --camera-id camera_front_wide_120fov \
  --camera-id camera_cross_left_120fov \
  --camera-id camera_cross_right_120fov \
  --camera-id camera_rear_left_70fov \
  --camera-id camera_rear_right_70fov \
  --camera-id camera_back_rear_wide_90fov \
  --output-dir "$OUTPUT_DIR" \
  --link-mode "$LINK_MODE"
