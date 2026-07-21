#!/usr/bin/env bash
# Native-FTheta six-camera sanity run: 5-second window, 5k iterations.

set -Eeuo pipefail

MODE="${1:-run}"
[ "$MODE" = "run" ] || [ "$MODE" = "--preflight" ] || {
  echo "Usage: $0 [--preflight]" >&2
  exit 2
}

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PIN_DATA_PATH="${PIN_DATA_PATH:-}"
FTHETA_DATA_PATH="${FTHETA_DATA_PATH:-}"
PIN_BASELINE_PARSED="${PIN_BASELINE_PARSED:-}"
RUN_BASE="${RUN_BASE:-$HOME/work/output/pin_ftheta_native_6cam_smoke}"
ARMS="${ARMS:-F}"
CONFIG_NAME="apps/ncore_3dgut_mcmc_multilayer_inceptio_6cam_native_ab"

[ -n "$PIN_DATA_PATH" ] || { echo "ERROR: PIN_DATA_PATH is required" >&2; exit 1; }
[ -n "$FTHETA_DATA_PATH" ] || { echo "ERROR: FTHETA_DATA_PATH is required" >&2; exit 1; }
[ -f "$PIN_BASELINE_PARSED" ] || { echo "ERROR: PIN_BASELINE_PARSED must reference the recorded P parsed.yaml" >&2; exit 1; }
[ "$ARMS" = "F" ] || {
  echo "ERROR: this native-FTheta launcher only runs arm F; the Pinhole 30k baseline is reused" >&2
  exit 1
}

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
"$PYTHON_BIN" -m scripts.pin_ftheta_native_6cam_validation \
  --mode smoke --pin-manifest "$PIN_DATA_PATH" --ftheta-manifest "$FTHETA_DATA_PATH" \
  --baseline-parsed "$PIN_BASELINE_PARSED"

if [ "$MODE" = "--preflight" ]; then
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$RUN_BASE"

COMMON_OVERRIDES=(
  n_iterations=5000
  seed_initialization=42
  test_last=true
  out_dir="$RUN_BASE"
  dataset.train.seek_offset_sec=0.0
  dataset.train.duration_sec=5.0
  dataset.val.seek_offset_sec=0.0
  dataset.val.duration_sec=5.0
  dataset.downsample=1.0
  dataset.n_val_image_subsample=1
  trainer.sky_backend=mlp
  trainer.use_lidar_depth=false
  trainer.use_depth_prior=false
  dataset.load_lidar_depth_map=false
  dataset.load_depth_prior=false
  num_workers=10
)

run_arm() {
  local arm="$1"
  local data_path="$2"
  local name="pin_ftheta_native_6cam_smoke_arm${arm}"
  "$PYTHON_BIN" train.py --config-name "$CONFIG_NAME" \
    experiment_name="$name" path="$data_path" "${COMMON_OVERRIDES[@]}"
}

run_arm F "$FTHETA_DATA_PATH"
