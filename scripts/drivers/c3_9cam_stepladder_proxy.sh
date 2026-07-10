#!/bin/bash
# C3 阶梯 8→9 cam proxy（Task 10 Step 1）
#
# 双臂 6k proxy，同 R5c 配方（yaml 内置 8-cam camera_ids），唯一差异 = camera_ids 8→9 + telew：
#   Arm A：c3_armA_8cam_6k  —— R5c yaml 内置 8-cam（无 CLI 覆盖）
#   Arm B：c3_armB_9cam_tw2p0_6k —— 同 yaml + CLI 覆盖 dataset.camera_ids=[9-cam]
#          + ++loss.camera_loss_weights.camera_front_tele_30fov=2.0
#          （加入 camera_front_tele_30fov；4cab 证据：tele 无权重 18.04 → 加权 26.24；
#            初始 weight=2.0 起步，视 proxy per-cam psnr 调 0.5 档步进）
#
# kill-criterion（Task 10 Step 1）：
#   run 名                 c3_armA_8cam_6k / c3_armB_9cam_tw2p0_6k
#   观察点 iter 2k         无 NaN、无死层告警、loss 曲线正常；任一破线 → 停 run 排障
#   判定 metric            Arm A vs Arm B 8 共有相机 cc_psnr_masked 差 ≤ 0.3 dB（守护线）
#                          front_tele 弱 (<19) → telew 调权（0.5 档步进重跑 Arm B）
#                          调权 run 单独命名 c3_armBw_<telew>_9cam_6k
#
# 数据 / 环境铁律（CLAUDE.md）：
#   depth-off + num_workers=10 + PYTORCH_CUDA_ALLOC_CONF=expandable_segments
#   config yaml 内置：cuboid_ts_mode=per_camera_interp + loss.use_opacity=false
#
# 启动模式（外部 ssh 侧，CLAUDE.md 嵌套 driver 铁律）：
#   ssh -n inceptio 'cd ~/repo/3dgrut2-wt/c3 && \
#     setsid bash scripts/drivers/c3_9cam_stepladder_proxy.sh \
#     > /tmp/c3_9cam_proxy.log 2>&1 < /dev/null & echo PID_$!'
#
# driver-owned stdout（不依赖 launcher redirect）
exec > >(tee -a "$HOME/work/output/c3_9cam_proxy_driver.log") 2>&1

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

CAM_9=(
  camera_front_wide_120fov
  camera_cross_left_120fov
  camera_cross_right_120fov
  camera_left_wide_90fov
  camera_right_wide_90fov
  camera_back_rear_wide_90fov
  camera_rear_left_70fov
  camera_front_standard_55fov
  camera_front_tele_30fov
)
CAM_9_LIST="[$(IFS=,; echo "${CAM_9[*]}")]"

run_arm() {
  local name="$1"; shift
  echo "=== C3 $name train start $(date '+%F %T') ==="
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
  echo "=== C3 $name eval ckpt=$CKPT $(date '+%F %T') ==="
  python render.py --checkpoint "$CKPT" --out-dir "$OUT/${name}_eval" \
    > "/tmp/${name}_eval.log" 2>&1
  find "$OUT/${name}_eval" -name metrics.json | head -1
  echo "=== C3 $name done $(date '+%F %T') ==="
}

# Arm A: 8-cam baseline @ 6k (identical to R5c yaml, only n_iterations shortened; no CLI override)
run_arm c3_armA_8cam_6k

# Arm B: 9-cam single-variable @ 6k + telew=2.0 on front_tele
run_arm c3_armB_9cam_tw2p0_6k \
  "dataset.camera_ids=$CAM_9_LIST" \
  "++loss.camera_loss_weights.camera_front_tele_30fov=2.0"

echo "=== C3 proxy both arms done $(date '+%F %T') ==="
echo "all done"
