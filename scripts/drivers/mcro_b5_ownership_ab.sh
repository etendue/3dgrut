#!/usr/bin/env bash
# MCRO B5: launch exactly one confirmed 5-second ownership arm per invocation.

set -Eeuo pipefail

MODE="${1:---preflight}"
ARM="${MCRO_ARM:-R1}"
case "$MODE" in --preflight|--run) ;; *) echo "usage: $0 [--preflight|--run]" >&2; exit 2;; esac
case "$ARM" in R0|R1|R2|R3|R3O|R3C) ;; *) echo "MCRO_ARM must be R0/R1/R2/R3/R3O/R3C" >&2; exit 2;; esac

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-/home/inceptio/miniforge3/envs/3dgrut2/bin/python}"
MANIFEST="${MCRO_MANIFEST:-/home/inceptio/work/data/inc_b6a9ed61_20s_6cam_directional/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9.json}"
RUN_BASE="${MCRO_RUN_BASE:-/home/inceptio/work/output/mcro_b5_ownership}"
CONFIG_NAME="apps/ncore_3dgut_mcmc_multilayer_inceptio_6cam_native_ab"
FRONT_CAMERA="camera_front_wide_120fov"
GUARDS="$REPO_DIR/configs/eval/mcro_ownership_guards.json"
NAME="mcro_b5_${ARM,,}_5s_5k"

test -x "$PYTHON_BIN"
test -f "$MANIFEST"
test -f "$GUARDS"
cd "$REPO_DIR"

COMMON_OVERRIDES=(
  n_iterations=5000
  seed_initialization=42
  test_last=true
  path="$MANIFEST"
  out_dir="$RUN_BASE"
  experiment_name="$NAME"
  dataset.train.seek_offset_sec=0.0
  dataset.val.seek_offset_sec=0.0
  dataset.train.duration_sec=5.0
  dataset.val.duration_sec=5.0
  trainer.sky_backend=mlp
  trainer.use_lidar_depth=false
  trainer.use_depth_prior=false
  dataset.load_lidar_depth_map=false
  dataset.load_depth_prior=false
  num_workers=10
  trainer.per_camera_telemetry=true
)

# Strict cumulative arms: each row adds exactly one config change.
ARM_OVERRIDES=()
if [[ "$ARM" != R0 ]]; then
  ARM_OVERRIDES+=(layers.semantic_disjoint_init=true)
fi
if [[ "$ARM" == R2 ]] || [[ "$ARM" =~ ^R3 ]]; then
  ARM_OVERRIDES+=(layers.bg_road_exclusion.enabled=true)
fi
if [[ "$ARM" =~ ^R3 ]]; then
  ARM_OVERRIDES+=(loss.road_responsibility.enabled=true)
fi
if [[ "$ARM" == R3O ]] || [[ "$ARM" == R3C ]]; then
  ARM_OVERRIDES+=(++layers.overrides.road.road_init_initial_opacity=0.7)
fi
if [[ "$ARM" == R3C ]]; then
  ARM_OVERRIDES+=(++layers.overrides.road.road_init_use_lidar_color=true)
fi

echo "MCRO_PREFLIGHT arm=$ARM commit=$(git rev-parse --short HEAD)"
printf 'OVERRIDE %s\n' "${ARM_OVERRIDES[@]}"
if [[ "$MODE" == "--preflight" ]]; then
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$RUN_BASE" "$RUN_BASE/logs"
TRAIN_LOG="$RUN_BASE/logs/${NAME}_train.log"

CKPT=$(find "$RUN_BASE/$NAME" -name ckpt_last.pt -print -quit 2>/dev/null || true)
if [[ -z "$CKPT" ]]; then
  "$PYTHON_BIN" train.py --config-name "$CONFIG_NAME" \
    "${COMMON_OVERRIDES[@]}" "${ARM_OVERRIDES[@]}" >"$TRAIN_LOG" 2>&1
  CKPT=$(find "$RUN_BASE/$NAME" -name ckpt_last.pt -print -quit)
fi
test -f "$CKPT"

EVAL_ROOT="$RUN_BASE/${NAME}_eval"
render_layer() {
  local layer="$1"
  local out="$EVAL_ROOT/$layer"
  local log="$RUN_BASE/logs/${NAME}_${layer}.log"
  local args=(--checkpoint "$CKPT" --out-dir "$out" --eval-cameras "$FRONT_CAMERA" --ownership-dump)
  if [[ "$layer" == full ]]; then
    args+=(--novel-view)
  else
    args+=(--enabled-layers "$layer")
  fi
  "$PYTHON_BIN" render.py "${args[@]}" >"$log" 2>&1
}

for LAYER in full background road sky_envmap; do
  render_layer "$LAYER"
done

FULL_METRICS=$(find "$EVAL_ROOT/full" -name metrics.json -print -quit)
BG_OWNERSHIP=$(find "$EVAL_ROOT/background" -type d -name ownership -print -quit)
ROAD_OWNERSHIP=$(find "$EVAL_ROOT/road" -type d -name ownership -print -quit)
SKY_OWNERSHIP=$(find "$EVAL_ROOT/sky_envmap" -type d -name ownership -print -quit)
test -f "$FULL_METRICS"
test -d "$BG_OWNERSHIP"
test -d "$ROAD_OWNERSHIP"
test -d "$SKY_OWNERSHIP"

OWNERSHIP_JSON="$EVAL_ROOT/ownership.json"
"$PYTHON_BIN" -m scripts.drivers.mcro_layer_ownership_eval \
  --bg-ownership "$BG_OWNERSHIP" --road-ownership "$ROAD_OWNERSHIP" \
  --sky-ownership "$SKY_OWNERSHIP" --out "$OWNERSHIP_JSON" --erosion-px 1

if [[ "$ARM" == R0 ]]; then
  echo "MCRO_5S_BASELINE_DONE ckpt=$CKPT metrics=$FULL_METRICS ownership=$OWNERSHIP_JSON"
  exit 0
fi

QUALITY_BASELINE=$(find "$RUN_BASE/mcro_b5_r0_5s_5k_eval/full" -name metrics.json -print -quit)
test -f "$QUALITY_BASELINE"

set +e
"$PYTHON_BIN" -m scripts.drivers.mcro_ownership_guard \
  --ownership "$OWNERSHIP_JSON" --metrics "$FULL_METRICS" \
  --quality-baseline "$QUALITY_BASELINE" --guards "$GUARDS" --out "$EVAL_ROOT"
GUARD_EXIT=$?
set -e
echo "MCRO_ARM_DONE arm=$ARM ckpt=$CKPT metrics=$FULL_METRICS guard_exit=$GUARD_EXIT"
exit "$GUARD_EXIT"
