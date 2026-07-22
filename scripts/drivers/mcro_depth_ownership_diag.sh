#!/usr/bin/env bash
# Read-only depth-aware ownership diagnosis for an existing checkpoint.

set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CHECKPOINT OUT_DIR" >&2
  exit 2
fi

CKPT=$1
OUT_ROOT=$2
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-/home/inceptio/miniforge3/envs/3dgrut2/bin/python}"
CUDA_HOME="${CUDA_HOME:-/home/inceptio/miniforge3/envs/3dgrut2}"
FRONT_CAMERA="camera_front_wide_120fov"

test -f "$CKPT"
test -x "$PYTHON_BIN"
mkdir -p "$OUT_ROOT/logs"
cd "$REPO_DIR"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for layer in background road sky_envmap; do
  "$PYTHON_BIN" render.py \
    --checkpoint "$CKPT" \
    --out-dir "$OUT_ROOT/$layer" \
    --eval-cameras "$FRONT_CAMERA" \
    --ownership-dump \
    --enabled-layers "$layer" \
    >"$OUT_ROOT/logs/${layer}.log" 2>&1
done

BG_OWNERSHIP=$(find "$OUT_ROOT/background" -type d -name ownership -print -quit)
ROAD_OWNERSHIP=$(find "$OUT_ROOT/road" -type d -name ownership -print -quit)
SKY_OWNERSHIP=$(find "$OUT_ROOT/sky_envmap" -type d -name ownership -print -quit)
test -d "$BG_OWNERSHIP"
test -d "$ROAD_OWNERSHIP"
test -d "$SKY_OWNERSHIP"

"$PYTHON_BIN" -m scripts.drivers.mcro_layer_ownership_eval \
  --bg-ownership "$BG_OWNERSHIP" \
  --road-ownership "$ROAD_OWNERSHIP" \
  --sky-ownership "$SKY_OWNERSHIP" \
  --out "$OUT_ROOT/ownership.json" \
  --erosion-px 1

echo "MCRO_DEPTH_OWNERSHIP_DONE checkpoint=$CKPT report=$OUT_ROOT/ownership.json"
