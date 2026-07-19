#!/bin/bash
# PIN-CAM-1: controlled 2x2 ablation of the calibrated OpenCV validity domain
# and valid-only unscented-transform accumulation.

set -eo pipefail

export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$HOME/repo/3dgrut2-wt/pinhole-ftheta-remap"
export PYTHONPATH="$PWD"

MANIFEST="$HOME/work/data/inc_b6a9ed61_20s/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9.json"
OUT="$HOME/work/output"
CAMERA="camera_front_wide_120fov"

COMMON=(
  --config-name apps/ncore_3dgut_mcmc_multilayer_inceptio
  n_iterations=5000
  path="$MANIFEST"
  out_dir="$OUT"
  trainer.sky_backend=mlp
  trainer.use_lidar_depth=false
  trainer.use_depth_prior=false
  dataset.load_lidar_depth_map=false
  dataset.load_depth_prior=false
  num_workers=10
  dataset.train.duration_sec=5.0
  dataset.val.duration_sec=5.0
  dataset.train.seek_offset_sec=0.0
  dataset.val.seek_offset_sec=0.0
  "dataset.camera_ids=[$CAMERA]"
  dataset.opencv_pinhole_inverse_iterations=30
)

run_arm() {
  local name="$1"
  local domain="$2"
  local valid_only="$3"
  local train_log="/tmp/${name}_train.log"
  local eval_log="/tmp/${name}_eval.log"
  local ckpt metrics

  echo "=== ARM $name domain=$domain valid_only=$valid_only ==="
  metrics=$(find "$OUT/${name}_eval" -name metrics.json -print -quit 2>/dev/null || true)
  if [ -n "$metrics" ] && [ -f "$metrics" ]; then
    echo "ARM_ALREADY_DONE $name"
    echo "METRICS_$metrics"
    return 0
  fi

  ckpt=$(find "$OUT/$name" -name ckpt_last.pt -print -quit 2>/dev/null || true)
  if [ -z "$ckpt" ] || [ ! -f "$ckpt" ]; then
    python train.py "${COMMON[@]}" \
      experiment_name="$name" \
      dataset.opencv_pinhole_use_validity_domain="$domain" \
      render.splat.ut_valid_only="$valid_only" \
      >"$train_log" 2>&1
    ckpt=$(find "$OUT/$name" -name ckpt_last.pt -print -quit)
  else
    echo "TRAIN_ALREADY_DONE $name"
  fi
  test -f "$ckpt"

  python render.py \
    --checkpoint "$ckpt" \
    --out-dir "$OUT/${name}_eval" \
    --dataset-cameras "$CAMERA" \
    >"$eval_log" 2>&1
  metrics=$(find "$OUT/${name}_eval" -name metrics.json -print -quit)
  test -f "$metrics"

  echo "ARM_DONE $name"
  echo "CKPT_$ckpt"
  echo "METRICS_$metrics"
  python - "$metrics" <<'PY'
import json
import sys

metrics = json.load(open(sys.argv[1]))
keys = ("mean_psnr", "mean_psnr_masked", "mean_lpips_masked", "mean_cc_psnr_masked")
print(json.dumps({key: metrics.get(key) for key in keys}, sort_keys=True))
PY
}

# Run all four arms from the same source revision.  Names encode
# domain (legacy/calibrated) and UT accumulation (legacy/valid-only).
run_arm pin_cam_2x2_legacy_gate_legacy_ut false false
run_arm pin_cam_2x2_legacy_gate_valid_ut false true
run_arm pin_cam_2x2_calibrated_gate_legacy_ut true false
run_arm pin_cam_2x2_calibrated_gate_valid_ut true true

echo ALL_2X2_ARMS_DONE
