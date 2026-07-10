#!/bin/bash
# C2 阶梯 6→8 cam proxy（Task 9 Step 1）
#
# 双臂 6k proxy，同 R4e 配方，唯一差异 = camera_ids 6-cam vs 8-cam：
#   Arm A：c2_armA_6cam_6k —— R4e yaml 内置 6-cam（front_wide/cross_L/R/left/right/back_rear）
#   Arm B：c2_armB_8cam_6k —— 同 yaml + CLI 覆盖 dataset.camera_ids=[8-cam]
#          （+camera_rear_left_70fov +camera_front_standard_55fov；
#            camera_rear_right_70fov 永久 held-out 不入训）
#
# kill-criterion（Task 9 Step 1）：
#   run 名                 c2_armA_6cam_6k / c2_armB_8cam_6k
#   观察点 iter 2k         无 NaN、无死层告警、loss 曲线正常；任一破线 → 停 run 排障
#   判定 metric            Arm A vs Arm B 6 共有相机 psnr 差 ≤ 0.3 dB（守护线）
#                          新相机（rear_left / front_standard）弱 >2 dB → telew 调权
#                          （0.5 档步进重跑 Arm B，命名 c2_armBw_<telew>_8cam_6k）
#
# 数据 / 环境铁律（CLAUDE.md）：
#   depth-off + num_workers=10 + PYTORCH_CUDA_ALLOC_CONF=expandable_segments
#   config yaml 内置：cuboid_ts_mode=per_camera_interp + loss.use_opacity=false
#
# 启动模式（外部 ssh 侧，CLAUDE.md 嵌套 driver 铁律）：
#   ssh -n inceptio 'cd ~/repo/3dgrut2-wt/c2_8cam && \
#     setsid bash scripts/drivers/c2_8cam_stepladder_proxy.sh \
#     > /tmp/c2_8cam_proxy.log 2>&1 < /dev/null & echo PID_$!'
#
# driver-owned stdout（不依赖 launcher redirect）
exec > >(tee -a "$HOME/work/output/c2_8cam_proxy_driver.log") 2>&1

set -eo pipefail
export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MANIFEST="$HOME/work/data/inc_b6a9ed61_20s/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9.json"
OUT="$HOME/work/output"

COMMON=(
  path="$MANIFEST"
  out_dir="$OUT"
  trainer.sky_backend=mlp
  trainer.use_lidar_depth=false
  trainer.use_depth_prior=false
  dataset.load_lidar_depth_map=false
  dataset.load_depth_prior=false
  num_workers=10
)

CAM_8=(
  camera_front_wide_120fov
  camera_cross_left_120fov
  camera_cross_right_120fov
  camera_left_wide_90fov
  camera_right_wide_90fov
  camera_back_rear_wide_90fov
  camera_rear_left_70fov
  camera_front_standard_55fov
)
# Hydra list literal
CAM_8_LIST="[$(IFS=,; echo "${CAM_8[*]}")]"

run_arm() {
  local name="$1"; shift
  echo "=== C2 $name train start $(date '+%F %T') ==="
  python train.py --config-name apps/ncore_3dgut_mcmc_multilayer_inceptio \
    n_iterations=6000 experiment_name="$name" \
    "${COMMON[@]}" "$@" > "/tmp/${name}_train.log" 2>&1
  echo "--- $name 告警计数 (dead / non-finite / [P0.2] fallback / [A5] cuboid_mask):"
  grep -ac "layer fully dead"                       "/tmp/${name}_train.log" || true
  grep -ac "non-finite"                              "/tmp/${name}_train.log" || true
  grep -ac "\[P0.2\] ego mask via aux itar fallback" "/tmp/${name}_train.log" || true
  grep -ac "\[A5\] dyn_mask_cuboid filled"           "/tmp/${name}_train.log" || true

  local CKPT
  CKPT=$(ls -dt "$OUT/$name"/*/ckpt_last.pt | head -1)
  echo "=== C2 $name eval ckpt=$CKPT $(date '+%F %T') ==="
  python render.py --checkpoint "$CKPT" --out-dir "$OUT/${name}_eval" \
    > "/tmp/${name}_eval.log" 2>&1
  find "$OUT/${name}_eval" -name metrics.json | head -1
  echo "=== C2 $name done $(date '+%F %T') ==="
}

# Arm A: 6-cam baseline @ 6k (identical to R4e yaml, only n_iterations shortened)
run_arm c2_armA_6cam_6k

# Arm B: 8-cam single-variable @ 6k (dataset.camera_ids override; unquoted list literal)
run_arm c2_armB_8cam_6k "dataset.camera_ids=$CAM_8_LIST"

echo "=== C2 proxy both arms done $(date '+%F %T') ==="
echo "all done"
