#!/bin/bash
# C2 阶梯 8-cam 30k 全量（Task 9 Step 2 晋级 KPI run）
#
# 6k proxy 判定通过（scripts/drivers/c2_8cam_stepladder_proxy.sh 出数 2026-07-10）：
#   Arm B 8-cam vs Arm A 6-cam @6k：
#     - shared 6 mean cc_psnr_masked −0.28 dB（守护线 ≤ 0.3 dB，过）
#     - 8-cam mean_cc_psnr_masked 17.88 vs A 17.05（+0.83 dB，读数改善）
#     - 新相机 rear_left 18.48 / front_standard 23.72（均 >2 dB 阈值，无需 telew）
#     - Arm A/B alarm counts：dead 0 / non-finite 3（A1 极点预期）
#
# 单变量差异 = 仅 camera_ids 6→8（+rear_left_70fov +front_standard_55fov；
#   rear_right_70fov 永久 held-out 保 P0.6 eval 用）；其他 = R4e/Arm A/Arm B 完全同配方
#
# kill-criterion 登记：
#   run 名                 c2_8cam_30k
#   观察点 iter 5k / 15k    无 NaN、无死层、shared 6 台 val cc 不退 R4e 30k 参照 >0.5 dB
#   砍单动作               停 run 排障，若为容量摊薄（train loss 平台过早）
#                          → 触发 v5_plan Issue I4 分层正则精调路线
#
# 数据 / 环境铁律（CLAUDE.md）：
#   depth-off + num_workers=10 + PYTORCH_CUDA_ALLOC_CONF=expandable_segments
#
# 启动模式（外部 ssh 侧）：
#   ssh -n inceptio 'cd ~/repo/3dgrut2-wt/c2_8cam && \
#     setsid bash scripts/drivers/c2_8cam_30k.sh \
#     > /tmp/c2_8cam_30k.log 2>&1 < /dev/null & echo PID_$!'
#
# driver-owned tee（不依赖 launcher redirect）
exec > >(tee -a "$HOME/work/output/c2_8cam_30k_driver.log") 2>&1

set -eo pipefail
export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MANIFEST="$HOME/work/data/inc_b6a9ed61_20s/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9.json"
OUT="$HOME/work/output"
NAME=c2_8cam_30k

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
CAM_8_LIST="[$(IFS=,; echo "${CAM_8[*]}")]"

echo "=== C2 $NAME train start $(date '+%F %T') ==="
python train.py --config-name apps/ncore_3dgut_mcmc_multilayer_inceptio \
  n_iterations=30000 experiment_name="$NAME" \
  path="$MANIFEST" out_dir="$OUT" \
  trainer.sky_backend=mlp \
  trainer.use_lidar_depth=false \
  trainer.use_depth_prior=false \
  dataset.load_lidar_depth_map=false \
  dataset.load_depth_prior=false \
  num_workers=10 \
  "dataset.camera_ids=$CAM_8_LIST" \
  > "/tmp/${NAME}_train.log" 2>&1

echo "--- $NAME 告警计数 (dead / non-finite / [P0.2] fallback / [A5] cuboid_mask):"
grep -ac "layer fully dead"                       "/tmp/${NAME}_train.log" || true
grep -ac "non-finite"                              "/tmp/${NAME}_train.log" || true
grep -ac "\[P0.2\] ego mask via aux itar fallback" "/tmp/${NAME}_train.log" || true
grep -ac "\[A5\] dyn_mask_cuboid filled"           "/tmp/${NAME}_train.log" || true

CKPT=$(ls -dt "$OUT/$NAME"/*/ckpt_last.pt | head -1)
echo "=== C2 $NAME eval ckpt=$CKPT $(date '+%F %T') ==="
python render.py --checkpoint "$CKPT" --out-dir "$OUT/${NAME}_eval" \
  > "/tmp/${NAME}_eval.log" 2>&1
find "$OUT/${NAME}_eval" -name metrics.json | head -1
echo "=== C2 $NAME done $(date '+%F %T') ==="
echo "all done"
