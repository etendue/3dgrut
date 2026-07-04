#!/bin/bash
# lattice v2 续跑（inceptio 重启后）：R0c 已完成，只跑 R1p/R2p/R3p（修正配方 use_opacity=false）
set -eo pipefail
export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."

MANIFEST=$HOME/work/data/inc_b6a9ed61_20s/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9.json
OUT=$HOME/work/output
CAMS6='[camera_front_wide_120fov,camera_cross_left_120fov,camera_cross_right_120fov,camera_left_wide_90fov,camera_right_wide_90fov,camera_back_rear_wide_90fov]'
COMMON="path=$MANIFEST out_dir=$OUT trainer.sky_backend=mlp trainer.use_lidar_depth=false trainer.use_depth_prior=false dataset.load_lidar_depth_map=false dataset.load_depth_prior=false num_workers=10 loss.use_opacity=false"

run_one () {
  local name=$1 cams=$2 iters=$3; shift 3
  echo "=== RUN $name train start $(date '+%F %T') ==="
  python train.py --config-name apps/ncore_3dgut_mcmc_multilayer \
    n_iterations="$iters" "dataset.camera_ids=$cams" experiment_name="$name" \
    $COMMON "$@" > "/tmp/v5a_${name}_train.log" 2>&1
  echo "--- 告警计数 dead/nonfinite:"
  grep -ac "layer fully dead" "/tmp/v5a_${name}_train.log" || true
  grep -ac "non-finite" "/tmp/v5a_${name}_train.log" || true
  local ckpt
  ckpt=$(ls -dt "$OUT/$name"/*/ckpt_last.pt | head -1)
  echo "=== RUN $name eval ckpt=$ckpt $(date '+%F %T') ==="
  python render.py --checkpoint "$ckpt" --out-dir "$OUT/${name}_eval" \
    > "/tmp/v5a_${name}_eval.log" 2>&1
  find "$OUT/${name}_eval" -name metrics.json | head -1
  echo "=== RUN $name done $(date '+%F %T') ==="
}

run_one R1p_6cam_maskoff "$CAMS6" 30000 '++trainer.bg_dyn_cuboid_penalty.use_cuboid_mask=false'
run_one R2p_6cam_maskon  "$CAMS6" 30000
run_one R3p_6cam_interp  "$CAMS6" 30000 dataset.cuboid_ts_mode=per_camera_interp

echo "ALL LATTICE V2B RUNS COMPLETE $(date '+%F %T')"
