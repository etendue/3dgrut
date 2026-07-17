#!/bin/bash
# Full-window 30k matched PIN-FTHETA P/F A/B on the approved seven cameras.
# Arm P and Arm F differ only in dataset.ftheta_params_path. Run from an
# isolated committed inceptio worktree with the nested-driver launch pattern.

set -Eeuo pipefail

export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODE="${1:-run}"
[ "$MODE" = "run" ] || [ "$MODE" = "--preflight" ] || {
  echo "Usage: $0 [--preflight]"
  exit 2
}

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_PATH="${DATA_PATH:-$HOME/work/data/inc_b6a9ed61_20s/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9.json}"
RUN_BASE="${RUN_BASE:-$HOME/work/output/pin_ftheta_7cam_full_ab_runs}"
FTHETA_PARAMS="scripts/pin_ftheta_b6a9_7cam_params.json"
CONFIG_FILE="configs/apps/ncore_3dgut_mcmc_multilayer_inceptio_7cam.yaml"
DRIVER_FILE="scripts/pin_ftheta_7cam_full_ab.sh"
VALIDATOR_FILE="scripts/pin_ftheta_full_ab_validation.py"
VALIDATOR_MODULE="scripts.pin_ftheta_full_ab_validation"
RUN_ID="$(date -u '+%Y%m%dT%H%M%SZ')_$(date +%s%N)_pid$$_r${RANDOM}"
RUN_ROOT="$RUN_BASE/$RUN_ID"
TRAIN_OUTPUT_ROOT="$RUN_ROOT/train_outputs"
EVAL_OUTPUT_ROOT="$RUN_ROOT/eval_outputs"
RUN_MANIFEST="$RUN_ROOT/run_manifest.json"
CURRENT_STAGE="pre-manifest"

cd "$REPO_DIR"
if command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
elif [ -x "$REPO_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$REPO_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
else
  echo "ERROR: no Python interpreter found"
  exit 1
fi
[ -f "$DATA_PATH" ] || { echo "ERROR: dataset manifest missing: $DATA_PATH"; exit 1; }
[ -f "$FTHETA_PARAMS" ] || { echo "ERROR: FTheta artifact missing: $FTHETA_PARAMS"; exit 1; }
[ -f "$CONFIG_FILE" ] || { echo "ERROR: experiment config missing: $CONFIG_FILE"; exit 1; }
[ -f "$VALIDATOR_FILE" ] || { echo "ERROR: full A/B validator missing: $VALIDATOR_FILE"; exit 1; }

if [ "$MODE" = "--preflight" ]; then
  "$PYTHON_BIN" -m "$VALIDATOR_MODULE" preflight \
    --repo-root "$REPO_DIR" \
    --config-name apps/ncore_3dgut_mcmc_multilayer_inceptio_7cam \
    --artifact "$FTHETA_PARAMS" --input-manifest "$DATA_PATH"
  echo "=== PIN-FTHETA full A/B preflight passed (no GPU work, no run output) ==="
  exit 0
fi

mkdir -p "$RUN_BASE"
mkdir "$RUN_ROOT"
mkdir "$RUN_ROOT/arms" "$TRAIN_OUTPUT_ROOT" "$EVAL_OUTPUT_ROOT"
"$PYTHON_BIN" -m "$VALIDATOR_MODULE" manifest-create \
  --path "$RUN_MANIFEST" --run-id "$RUN_ID" --repo-root "$REPO_DIR" \
  --dataset-manifest "$DATA_PATH" --config "$CONFIG_FILE" \
  --artifact "$FTHETA_PARAMS" --driver "$DRIVER_FILE" --validator "$VALIDATOR_FILE"

on_error() {
  local rc=$?
  trap - ERR
  if [ -f "$RUN_MANIFEST" ]; then
    "$PYTHON_BIN" -m "$VALIDATOR_MODULE" manifest-fail \
      --manifest "$RUN_MANIFEST" --stage "$CURRENT_STAGE" --exit-code "$rc" || true
  fi
  echo "ERROR: PIN-FTHETA full A/B failed at stage=$CURRENT_STAGE rc=$rc manifest=$RUN_MANIFEST" >&2
  exit "$rc"
}
trap on_error ERR

exec > >(tee "$RUN_ROOT/driver.log") 2>&1
echo "=== PIN-FTHETA immutable full-run root: $RUN_ROOT ==="

COMMON_OVERRIDES=(
  n_iterations=30000
  seed_initialization=42
  test_last=true
  path="$DATA_PATH"
  out_dir="$TRAIN_OUTPUT_ROOT"
  dataset.train.seek_offset_sec=0.0
  dataset.train.duration_sec=-1
  dataset.val.seek_offset_sec=0.0
  dataset.val.duration_sec=-1
  dataset.downsample=1.0
  dataset.n_val_image_subsample=1
  dataset.mask_forward_invalid_pixels=true
  dataset.opencv_pinhole_use_validity_domain=false
  trainer.sky_backend=mlp
  trainer.use_lidar_depth=false
  trainer.use_depth_prior=false
  dataset.load_lidar_depth_map=false
  dataset.load_depth_prior=false
  num_workers=10
)

run_arm() {
  local arm="$1"
  local ftheta_params_path="$2"
  local name="pin_ftheta_7cam_arm${arm}_full_30k"
  local ARM_ROOT="$RUN_ROOT/arms/$arm"
  local train_log="$ARM_ROOT/train.log"
  local eval_log="$ARM_ROOT/eval.log"
  local inventory="$ARM_ROOT/native_render_inventory.json"
  mkdir "$ARM_ROOT"
  CURRENT_STAGE="arm${arm}-source-verify"
  "$PYTHON_BIN" -m "$VALIDATOR_MODULE" manifest-verify --path "$RUN_MANIFEST" --repo-root "$REPO_DIR"

  CURRENT_STAGE="arm${arm}-train-test_last"
  echo "=== PIN-FTHETA full Arm $arm train start: $name $(date '+%F %T') ==="
  python train.py --config-name apps/ncore_3dgut_mcmc_multilayer_inceptio_7cam \
    experiment_name="$name" \
    "${COMMON_OVERRIDES[@]}" \
    dataset.ftheta_params_path="$ftheta_params_path" \
    > "$train_log" 2>&1

  CURRENT_STAGE="arm${arm}-train-log-validation"
  "$PYTHON_BIN" -m "$VALIDATOR_MODULE" log \
    --path "$train_log" --arm "$arm" --artifact "$FTHETA_PARAMS"

  local checkpoint parsed_yaml
  checkpoint=$(find "$TRAIN_OUTPUT_ROOT/$name" -name ckpt_last.pt -print -quit)
  [ -n "$checkpoint" ] || { echo "ERROR: Arm $arm checkpoint missing"; return 1; }
  parsed_yaml="$(dirname "$checkpoint")/parsed.yaml"
  [ -f "$parsed_yaml" ] || { echo "ERROR: Arm $arm parsed.yaml missing"; return 1; }
  CURRENT_STAGE="arm${arm}-checkpoint-validation"
  "$PYTHON_BIN" -m "$VALIDATOR_MODULE" checkpoint \
    --path "$checkpoint" --arm "$arm" --artifact "$FTHETA_PARAMS" \
    --input-manifest "$DATA_PATH"

  CURRENT_STAGE="arm${arm}-native-render"
  echo "=== PIN-FTHETA full Arm $arm native render: $checkpoint $(date '+%F %T') ==="
  python render.py --checkpoint "$checkpoint" --out-dir "$EVAL_OUTPUT_ROOT/$name" \
    > "$eval_log" 2>&1

  local metrics_path
  metrics_path=$(find "$EVAL_OUTPUT_ROOT/$name" -name metrics.json -print -quit)
  [ -n "$metrics_path" ] || { echo "ERROR: Arm $arm metrics.json missing"; return 1; }
  CURRENT_STAGE="arm${arm}-native-evidence-validation"
  "$PYTHON_BIN" -m "$VALIDATOR_MODULE" metrics --path "$metrics_path" --artifact "$FTHETA_PARAMS"
  "$PYTHON_BIN" -m "$VALIDATOR_MODULE" render-tree \
    --metrics "$metrics_path" --artifact "$FTHETA_PARAMS" --inventory "$inventory"
  "$PYTHON_BIN" -m "$VALIDATOR_MODULE" record-arm \
    --manifest "$RUN_MANIFEST" --arm "$arm" --parsed-yaml "$parsed_yaml" \
    --checkpoint "$checkpoint" --metrics "$metrics_path" \
    --train-log "$train_log" --eval-log "$eval_log" \
    --native-render-inventory "$inventory" \
    --artifact "$FTHETA_PARAMS" --input-manifest "$DATA_PATH" --repo-root "$REPO_DIR"
  echo "Arm $arm native metrics: $metrics_path"
  echo "=== PIN-FTHETA full Arm $arm done $(date '+%F %T') ==="
}

run_arm "P" "null"
run_arm "F" "scripts/pin_ftheta_b6a9_7cam_params.json"
CURRENT_STAGE="final-matched-evidence-gate"
"$PYTHON_BIN" -m "$VALIDATOR_MODULE" finalize --manifest "$RUN_MANIFEST" --repo-root "$REPO_DIR"
trap - ERR

echo "=== PIN-FTHETA matched seven-camera full A/B complete $(date '+%F %T') manifest=$RUN_MANIFEST ==="
