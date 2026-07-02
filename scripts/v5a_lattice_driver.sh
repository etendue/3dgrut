#!/bin/bash
# v5 任务A — 单变量重训 lattice 驱动（inceptio 4090 串行，setsid 后台启动）
#
#   sanity  6-cam 500 iter（默认 mask on → 验证 [A5] pinhole 日志 + road init 点数）
#   R0b     3-cam 30k, cuboid-mask off  → vs R0 锚 21.04   单变量 = aux 修复
#   R1      6-cam 30k, cuboid-mask off  → vs R0b           单变量 = 相机集
#   R2      6-cam 30k, cuboid-mask on   → vs R1            单变量 = A5 mask
#   R3      6-cam 30k, + per_camera_interp → vs R2         单变量 = A2 ts
#
# 每个 run: train → 找最新 ckpt → render.py eval 出 metrics.json。
# 日志: /tmp/v5a_<name>_{train,eval}.log；本脚本自身输出走启动方的重定向。
set -eo pipefail
export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."

MANIFEST=$HOME/work/data/inc_b6a9ed61_20s/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9.json
OUT=$HOME/work/output
CAMS6='[camera_front_wide_120fov,camera_cross_left_120fov,camera_cross_right_120fov,camera_left_wide_90fov,camera_right_wide_90fov,camera_back_rear_wide_90fov]'
CAMS3='[camera_front_wide_120fov,camera_cross_left_120fov,camera_cross_right_120fov]'
# inceptio 铁律: depth-off + num_workers=10
COMMON="path=$MANIFEST out_dir=$OUT trainer.sky_backend=mlp trainer.use_lidar_depth=false trainer.use_depth_prior=false dataset.load_lidar_depth_map=false dataset.load_depth_prior=false num_workers=10"

run_one () {  # <name> <cams> <iters> <do_eval:0|1> [extra overrides...]
  local name=$1 cams=$2 iters=$3 do_eval=$4; shift 4
  echo "=== RUN $name train start $(date '+%F %T') ==="
  python train.py --config-name apps/ncore_3dgut_mcmc_multilayer \
    n_iterations="$iters" "dataset.camera_ids=$cams" experiment_name="$name" \
    $COMMON "$@" > "/tmp/v5a_${name}_train.log" 2>&1
  echo "--- train tail:"
  tail -12 "/tmp/v5a_${name}_train.log"
  if [ "$do_eval" = "1" ]; then
    local ckpt
    ckpt=$(ls -dt "$OUT/$name"/*/ckpt_last.pt | head -1)
    echo "=== RUN $name eval ckpt=$ckpt $(date '+%F %T') ==="
    python render.py --checkpoint "$ckpt" --out-dir "$OUT/${name}_eval" \
      > "/tmp/v5a_${name}_eval.log" 2>&1
    find "$OUT/${name}_eval" -name metrics.json | head -1
  fi
  echo "=== RUN $name done $(date '+%F %T') ==="
}

run_one sanity_6cam "$CAMS6" 500 0
echo "--- sanity A5 wiring check:"
grep -m1 "\[A5\] dyn_mask_cuboid" /tmp/v5a_sanity_6cam_train.log \
  || echo "WARN: [A5] pinhole mask log NOT found"
grep -m1 "road layer initialized" /tmp/v5a_sanity_6cam_train.log \
  || echo "WARN: road init log NOT found"

run_one R0b_3cam_auxfix "$CAMS3" 30000 1 '++trainer.bg_dyn_cuboid_penalty.use_cuboid_mask=false'
run_one R1_6cam_maskoff "$CAMS6" 30000 1 '++trainer.bg_dyn_cuboid_penalty.use_cuboid_mask=false'
run_one R2_6cam_maskon  "$CAMS6" 30000 1
run_one R3_6cam_interp  "$CAMS6" 30000 1 dataset.cuboid_ts_mode=per_camera_interp

echo "ALL LATTICE RUNS COMPLETE $(date '+%F %T')"
