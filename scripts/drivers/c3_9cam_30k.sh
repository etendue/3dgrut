#!/bin/bash
# C3 阶梯 9-cam 30k 全量（Task 10 Step 2 晋级 KPI run）
#
# 6k proxy 判定通过后启用（scripts/drivers/c3_9cam_stepladder_proxy.sh 出数记入 Done Log）
# telew 权重 = proxy 采纳值（默认 2.0，若调整则在 CLI 覆盖）
#
# 单变量差异 = R5c 8-cam yaml + camera_ids 8→9（+camera_front_tele_30fov）
#            + camera_loss_weights.front_tele=<TELEW>
#   其他 = R5c 完全同配方；rear_right_70fov 永久 held-out 保 P0.6 eval 唯一外推 anchor
#
# kill-criterion 登记：
#   run 名                 c3_9cam_tw<TELEW>_30k
#   观察点 iter 5k / 15k    无 NaN、无死层、shared 8 台 val cc 不退 R5c 30k 参照 >0.5 dB
#   砍单动作               停 run 排障；若为容量摊薄（train loss 平台过早）
#                          → 触发 v5_plan Issue I4 分层正则精调
#
# 数据 / 环境铁律（CLAUDE.md）：depth-off + num_workers=10 + expandable_segments
#
# 启动模式（外部 ssh 侧）：
#   TELEW=2.0 ssh -n inceptio 'cd ~/repo/3dgrut2-wt/c3 && \
#     setsid env TELEW=2.0 bash scripts/drivers/c3_9cam_30k.sh \
#     > /tmp/c3_9cam_30k.log 2>&1 < /dev/null & echo PID_$!'
#
# driver-owned tee（不依赖 launcher redirect）
TELEW="${TELEW:-2.0}"
NAME=c3_9cam_tw${TELEW//./p}_30k
exec > >(tee -a "$HOME/work/output/${NAME}_driver.log") 2>&1

set -eo pipefail
export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MANIFEST="$HOME/work/data/inc_b6a9ed61_20s/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9.json"
OUT="$HOME/work/output"

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

echo "=== C3 $NAME train start $(date '+%F %T') TELEW=$TELEW ==="
python train.py --config-name apps/ncore_3dgut_mcmc_multilayer_inceptio \
  n_iterations=30000 experiment_name="$NAME" \
  path="$MANIFEST" out_dir="$OUT" \
  trainer.sky_backend=mlp \
  trainer.use_lidar_depth=false \
  trainer.use_depth_prior=false \
  dataset.load_lidar_depth_map=false \
  dataset.load_depth_prior=false \
  num_workers=10 \
  "dataset.camera_ids=$CAM_9_LIST" \
  "++loss.camera_loss_weights.camera_front_tele_30fov=$TELEW" \
  > "/tmp/${NAME}_train.log" 2>&1

echo "--- $NAME 告警计数 (dead / non-finite / [P0.2] fallback / [A5] cuboid_mask):"
grep -ac "layer fully dead"                       "/tmp/${NAME}_train.log" || true
grep -ac "non-finite"                              "/tmp/${NAME}_train.log" || true
grep -ac "\[P0.2\] ego mask via aux itar fallback" "/tmp/${NAME}_train.log" || true
grep -ac "\[A5\] dyn_mask_cuboid filled"           "/tmp/${NAME}_train.log" || true

CKPT=$(ls -dt "$OUT/$NAME"/*/ckpt_last.pt | head -1)
echo "=== C3 $NAME eval ckpt=$CKPT $(date '+%F %T') ==="
python render.py --checkpoint "$CKPT" --out-dir "$OUT/${NAME}_eval" \
  > "/tmp/${NAME}_eval.log" 2>&1
find "$OUT/${NAME}_eval" -name metrics.json | head -1
echo "=== C3 $NAME done $(date '+%F %T') ==="
echo "all done"
