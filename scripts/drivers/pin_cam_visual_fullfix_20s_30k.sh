#!/bin/bash
# PIN-CAM visual full-fix probe: long single-camera full-fix training.
# This is intentionally separate from the 5 s / 5k mechanism A/B.

set -euo pipefail

export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT=${PIN_CAM_REPO_ROOT:-/home/inceptio/repo/3dgrut2-wt/pin-cam-contract-clean}
cd "$REPO_ROOT"
export PYTHONPATH="$PWD"

MANIFEST=${PIN_CAM_MANIFEST:-/home/inceptio/work/data/inc_b6a9ed61_20s/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9.json}
OUT=${PIN_CAM_OUT:-/home/inceptio/work/output}
NAME=${PIN_CAM_NAME:-pin_cam_visual_fullfix_frontwide_20s_30k}
CAMERA=camera_front_wide_120fov

if find "$OUT/${NAME}_eval" -name metrics.json -print -quit 2>/dev/null | grep -q .; then
  echo "ALREADY_DONE $NAME"
  exit 0
fi

CKPT=$(find "$OUT/$NAME" -name ckpt_last.pt -print -quit 2>/dev/null || true)
if [ -z "$CKPT" ]; then
  python train.py \
    --config-name apps/ncore_3dgut_mcmc_multilayer_inceptio \
    n_iterations=30000 \
    path="$MANIFEST" \
    out_dir="$OUT" \
    experiment_name="$NAME" \
    trainer.sky_backend=mlp \
    trainer.use_lidar_depth=false \
    trainer.use_depth_prior=false \
    dataset.load_lidar_depth_map=false \
    dataset.load_depth_prior=false \
    num_workers=10 \
    dataset.train.duration_sec=20.0 \
    dataset.val.duration_sec=20.0 \
    dataset.train.seek_offset_sec=0.0 \
    dataset.val.seek_offset_sec=0.0 \
    "dataset.camera_ids=[$CAMERA]" \
    dataset.opencv_pinhole_inverse_iterations=30 \
    dataset.opencv_pinhole_use_validity_domain=true \
    render.splat.ut_valid_only=true
  CKPT=$(find "$OUT/$NAME" -name ckpt_last.pt -print -quit)
fi

test -f "$CKPT"
python render.py \
  --checkpoint "$CKPT" \
  --out-dir "$OUT/${NAME}_eval" \
  --dataset-cameras "$CAMERA"

METRICS=$(find "$OUT/${NAME}_eval" -name metrics.json -print -quit)
test -f "$METRICS"
echo "CKPT=$CKPT"
echo "METRICS=$METRICS"
echo "VISUAL_FULLFIX_DONE=$NAME"
