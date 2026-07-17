#!/bin/bash
# Matched PIN-FTHETA 5-second smoke on the user-approved seven-camera subset.
#
# The historical 9cam filename is retained for plan/artifact traceability. Both
# arms use the same seven-camera config; the only arm-specific override is the
# explicit FTheta parameter artifact (null for P, strict artifact for F).
#
# Launch on inceptio from an isolated worktree (AGENTS.md nested-driver mode):
#   ssh -n inceptio 'cd ~/repo/3dgrut2-wt/pin_ftheta_smoke && \
#     setsid bash scripts/pin_ftheta_9cam_smoke.sh \
#     > /tmp/pin_ftheta_9cam_smoke.log 2>&1 < /dev/null & echo PID_$!'

set -euo pipefail

export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_PATH="${DATA_PATH:-$HOME/work/data/inc_b6a9ed61_20s/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9.json}"
RUN_BASE="${RUN_BASE:-$HOME/work/output/pin_ftheta_smoke_runs}"
FTHETA_PARAMS="scripts/pin_ftheta_b6a9_7cam_params.json"
CONFIG_FILE="configs/apps/ncore_3dgut_mcmc_multilayer_inceptio_7cam.yaml"
DRIVER_FILE="scripts/pin_ftheta_9cam_smoke.sh"
VALIDATOR_FILE="scripts/pin_ftheta_smoke_validation.py"
VALIDATOR_MODULE="scripts.pin_ftheta_smoke_validation"
RUN_ID="$(date -u '+%Y%m%dT%H%M%SZ')_$(date +%s%N)_pid$$_r${RANDOM}"
RUN_ROOT="$RUN_BASE/$RUN_ID"
TRAIN_OUTPUT_ROOT="$RUN_ROOT/train_outputs"
EVAL_OUTPUT_ROOT="$RUN_ROOT/eval_outputs"
RUN_MANIFEST="$RUN_ROOT/run_manifest.json"

cd "$REPO_DIR"
[ -f "$DATA_PATH" ] || { echo "ERROR: dataset manifest missing: $DATA_PATH"; exit 1; }
[ -f "$FTHETA_PARAMS" ] || { echo "ERROR: FTheta artifact missing: $FTHETA_PARAMS"; exit 1; }
[ -f "$CONFIG_FILE" ] || { echo "ERROR: experiment config missing: $CONFIG_FILE"; exit 1; }
[ -f "$VALIDATOR_FILE" ] || { echo "ERROR: smoke validator missing: $VALIDATOR_FILE"; exit 1; }
mkdir -p "$RUN_BASE"
mkdir "$RUN_ROOT"
mkdir "$RUN_ROOT/arms" "$TRAIN_OUTPUT_ROOT" "$EVAL_OUTPUT_ROOT"
python -m "$VALIDATOR_MODULE" manifest-create \
  --path "$RUN_MANIFEST" --run-id "$RUN_ID" --repo-root "$REPO_DIR" \
  --dataset-manifest "$DATA_PATH" --config "$CONFIG_FILE" \
  --artifact "$FTHETA_PARAMS" --driver "$DRIVER_FILE" --validator "$VALIDATOR_FILE"
exec > >(tee "$RUN_ROOT/driver.log") 2>&1
echo "=== PIN-FTHETA immutable run root: $RUN_ROOT ==="

COMMON_OVERRIDES=(
  n_iterations=5000
  seed_initialization=42
  test_last=true
  path="$DATA_PATH"
  out_dir="$TRAIN_OUTPUT_ROOT"
  dataset.train.seek_offset_sec=0.0
  dataset.train.duration_sec=5.0
  dataset.val.seek_offset_sec=0.0
  dataset.val.duration_sec=5.0
  dataset.downsample=1.0
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
  local name="pin_ftheta_7cam_arm${arm}_5s_5k"
  local ARM_ROOT="$RUN_ROOT/arms/$arm"
  local train_log="$ARM_ROOT/train.log"
  local eval_log="$ARM_ROOT/eval.log"
  mkdir "$ARM_ROOT"
  python -m "$VALIDATOR_MODULE" manifest-verify --path "$RUN_MANIFEST" --repo-root "$REPO_DIR"

  echo "=== PIN-FTHETA Arm $arm train start: $name $(date '+%F %T') ==="
  python train.py --config-name apps/ncore_3dgut_mcmc_multilayer_inceptio_7cam \
    experiment_name="$name" \
    "${COMMON_OVERRIDES[@]}" \
    dataset.ftheta_params_path="$ftheta_params_path" \
    > "$train_log" 2>&1

  python -m "$VALIDATOR_MODULE" log \
    --path "$train_log" --arm "$arm" --artifact "$FTHETA_PARAMS"

  local checkpoint parsed_yaml
  checkpoint=$(find "$TRAIN_OUTPUT_ROOT/$name" -name ckpt_last.pt -print -quit)
  [ -n "$checkpoint" ] || { echo "ERROR: Arm $arm checkpoint missing"; exit 1; }
  parsed_yaml="$(dirname "$checkpoint")/parsed.yaml"
  [ -f "$parsed_yaml" ] || { echo "ERROR: Arm $arm parsed.yaml missing"; exit 1; }
  python -m "$VALIDATOR_MODULE" checkpoint \
    --path "$checkpoint" --arm "$arm" --artifact "$FTHETA_PARAMS" \
    --input-manifest "$DATA_PATH"

  echo "=== PIN-FTHETA Arm $arm native eval: $checkpoint $(date '+%F %T') ==="
  python render.py --checkpoint "$checkpoint" --out-dir "$EVAL_OUTPUT_ROOT/$name" \
    > "$eval_log" 2>&1

  local metrics_path
  metrics_path=$(find "$EVAL_OUTPUT_ROOT/$name" -name metrics.json -print -quit)
  [ -n "$metrics_path" ] || { echo "ERROR: Arm $arm metrics.json missing"; exit 1; }
  python -m "$VALIDATOR_MODULE" metrics \
    --path "$metrics_path" --artifact "$FTHETA_PARAMS"
  python -m "$VALIDATOR_MODULE" record-arm \
    --manifest "$RUN_MANIFEST" --arm "$arm" --parsed-yaml "$parsed_yaml" \
    --checkpoint "$checkpoint" --metrics "$metrics_path" \
    --train-log "$train_log" --eval-log "$eval_log" \
    --artifact "$FTHETA_PARAMS" --input-manifest "$DATA_PATH" --repo-root "$REPO_DIR"
  echo "Arm $arm metrics: $metrics_path"
  echo "=== PIN-FTHETA Arm $arm done $(date '+%F %T') ==="
}

run_arm "P" "null"
run_arm "F" "scripts/pin_ftheta_b6a9_7cam_params.json"
python -m "$VALIDATOR_MODULE" finalize \
  --manifest "$RUN_MANIFEST" --repo-root "$REPO_DIR"

echo "=== PIN-FTHETA matched seven-camera smoke complete $(date '+%F %T') manifest=$RUN_MANIFEST ==="
