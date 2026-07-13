#!/bin/bash
# C4 阶梯 9→11 cam proxy（Task 11 Step 2；可弃项）
#
# 双臂 6k proxy，同 R6t 配方（yaml 内置 9-cam camera_ids + telew=2.0 CLI），
# 唯一差异 = camera_ids 9→11 + 两台 FTheta 鱼眼：
#   Arm A：c4_armA_9cam_tw2p0_6k  —— R6t yaml + CLI ++loss.camera_loss_weights.
#                                    camera_front_tele_30fov=2.0（复现 C3 R6t 6k proxy）
#   Arm B：c4_armB_11cam_tw2p0_6k —— 同 + dataset.camera_ids=[11-cam]
#                                    （加 camera_front_fisheye + camera_back_rear_fisheye；
#                                     鱼眼**无先例数据**——初始 telew=1.0 默认无加权，
#                                     proxy 后 per-cam psnr 弱 → 试 telew 0.5 档步进）
#
# kill-criterion（Task 11 明确可弃项）：
#   run 名                     c4_armA_9cam_tw2p0_6k / c4_armB_11cam_tw2p0_6k
#   观察点 iter 2k              无 NaN、无死层告警、loss 曲线正常
#   判定 metric 1：shared 9-cam mean cc_psnr_masked 退 >0.5 dB（严于 C1-C3 的 ≤0.3）
#                              telew 调不回 → 弃 9-cam 收口
#   判定 metric 2：鱼眼 per-cam cc_psnr_masked <10（训练崩）→ 弃
#   判定 metric 3：proxy 后目检 novel view render 出现明显尖刺伪影
#                              （upstream issue #238 鱼眼尖刺风险）→ 弃
#   弃因入档 v5_plan.md §4 Done Log Task 11 条目，Phase C 3/4 定案
#
# 数据 / 环境铁律（CLAUDE.md）：
#   depth-off + num_workers=10（12GB 数据管线仍可）+ expandable_segments
#   config yaml 内置 R6t：cuboid_ts_mode=per_camera_interp + loss.use_opacity=false
#                        + camera_ids 9-cam（yaml 头注释锚 R6t）
#
# 启动模式（外部 ssh 侧，CLAUDE.md 嵌套 driver 铁律）：
#   ssh -n inceptio 'cd ~/repo/3dgrut2-wt/c4 && \
#     setsid bash scripts/drivers/c4_11cam_stepladder_proxy.sh \
#     > /tmp/c4_11cam_proxy.log 2>&1 < /dev/null & echo PID_$!'
#
exec > >(tee -a "$HOME/work/output/c4_11cam_proxy_driver.log") 2>&1

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

CAM_11=(
  camera_front_wide_120fov
  camera_cross_left_120fov
  camera_cross_right_120fov
  camera_left_wide_90fov
  camera_right_wide_90fov
  camera_back_rear_wide_90fov
  camera_rear_left_70fov
  camera_front_standard_55fov
  camera_front_tele_30fov
  camera_front_fisheye
  camera_back_rear_fisheye
)
CAM_11_LIST="[$(IFS=,; echo "${CAM_11[*]}")]"

run_arm() {
  local name="$1"; shift
  echo "=== C4 $name train start $(date '+%F %T') ==="
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
  echo "=== C4 $name eval ckpt=$CKPT $(date '+%F %T') ==="
  python render.py --checkpoint "$CKPT" --out-dir "$OUT/${name}_eval" \
    > "/tmp/${name}_eval.log" 2>&1
  find "$OUT/${name}_eval" -name metrics.json | head -1
  echo "=== C4 $name done $(date '+%F %T') ==="
}

# Arm A: 9-cam baseline (R6t 复现 @ 6k) —— yaml 9-cam + telew=2.0
run_arm c4_armA_9cam_tw2p0_6k \
  "++loss.camera_loss_weights.camera_front_tele_30fov=2.0"

# Arm B: 11-cam single-variable @ 6k —— +2 fisheye, telew=2.0 (tele unchanged), 鱼眼无加权
run_arm c4_armB_11cam_tw2p0_6k \
  "dataset.camera_ids=$CAM_11_LIST" \
  "++loss.camera_loss_weights.camera_front_tele_30fov=2.0"

echo "=== C4 proxy both arms done $(date '+%F %T') ==="
echo "all done"
