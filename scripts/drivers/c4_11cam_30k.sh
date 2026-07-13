#!/bin/bash
# C4 阶梯 11-cam 30k 全量（Task 11 Step 3 晋级 KPI run；仅在 proxy 判定过后启用）
#
# 6k proxy 判定通过后启用（scripts/drivers/c4_11cam_stepladder_proxy.sh 出数记入 Done Log）
# telew 权重：tele=2.0（C3 R6t 继承）+ 鱼眼 telew（可选，proxy 若定 weight != 1 则 CLI 覆盖）
#
# 单变量差异 = R6t 9-cam yaml + camera_ids 9→11（+camera_front_fisheye
#            + camera_back_rear_fisheye）
#   其他 = R6t 完全同配方；rear_right_70fov 永久 held-out 保 P0.6 eval 唯一外推 anchor
#
# kill-criterion 登记：
#   run 名                    c4_11cam_tw${TELEW//./p}_30k
#   观察点 iter 5k / 15k       无 NaN、无死层、shared 9 台 val cc 不退 R6t 30k 参照 >0.5 dB
#                              鱼眼 per-cam cc_psnr_masked > 10（训练可收敛）
#   砍单动作                   停 run 排障；触发 kill-criterion 且不可修 → 弃 9-cam 收口
#
# 数据 / 环境铁律（CLAUDE.md）：depth-off + num_workers=10 + expandable_segments
#
# 启动模式（外部 ssh 侧）：
#   TELEW=2.0 ssh -n inceptio 'cd ~/repo/3dgrut2-wt/c4 && \
#     setsid env TELEW=2.0 bash scripts/drivers/c4_11cam_30k.sh \
#     > /tmp/c4_11cam_30k.log 2>&1 < /dev/null & echo PID_$!'
#
# 若鱼眼需要 telew，用附加 CLI: 环境变量 FISHEW=X（默认不设 = 无鱼眼加权）
#
TELEW="${TELEW:-2.0}"
FISHEW="${FISHEW:-}"
if [ -n "$FISHEW" ]; then
  NAME=c4_11cam_tw${TELEW//./p}_fw${FISHEW//./p}_30k
else
  NAME=c4_11cam_tw${TELEW//./p}_30k
fi
exec > >(tee -a "$HOME/work/output/${NAME}_driver.log") 2>&1

set -eo pipefail
export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MANIFEST="$HOME/work/data/inc_b6a9ed61_20s/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9.json"
OUT="$HOME/work/output"

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

EXTRA_OVERRIDES=()
EXTRA_OVERRIDES+=("++loss.camera_loss_weights.camera_front_tele_30fov=$TELEW")
if [ -n "$FISHEW" ]; then
  EXTRA_OVERRIDES+=("++loss.camera_loss_weights.camera_front_fisheye=$FISHEW")
  EXTRA_OVERRIDES+=("++loss.camera_loss_weights.camera_back_rear_fisheye=$FISHEW")
fi

echo "=== C4 $NAME train start $(date '+%F %T') TELEW=$TELEW FISHEW=${FISHEW:-none} ==="
python train.py --config-name apps/ncore_3dgut_mcmc_multilayer_inceptio \
  n_iterations=30000 experiment_name="$NAME" \
  path="$MANIFEST" out_dir="$OUT" \
  trainer.sky_backend=mlp \
  trainer.use_lidar_depth=false \
  trainer.use_depth_prior=false \
  dataset.load_lidar_depth_map=false \
  dataset.load_depth_prior=false \
  num_workers=10 \
  "dataset.camera_ids=$CAM_11_LIST" \
  "${EXTRA_OVERRIDES[@]}" \
  > "/tmp/${NAME}_train.log" 2>&1

echo "--- $NAME 告警计数 (dead / non-finite / [P0.2] fallback / [A5] cuboid_mask):"
grep -ac "layer fully dead"                       "/tmp/${NAME}_train.log" || true
grep -ac "non-finite"                              "/tmp/${NAME}_train.log" || true
grep -ac "\[P0.2\] ego mask via aux itar fallback" "/tmp/${NAME}_train.log" || true
grep -ac "\[A5\] dyn_mask_cuboid filled"           "/tmp/${NAME}_train.log" || true

CKPT=$(ls -dt "$OUT/$NAME"/*/ckpt_last.pt | head -1)
echo "=== C4 $NAME eval ckpt=$CKPT $(date '+%F %T') ==="
python render.py --checkpoint "$CKPT" --out-dir "$OUT/${NAME}_eval" \
  > "/tmp/${NAME}_eval.log" 2>&1
find "$OUT/${NAME}_eval" -name metrics.json | head -1
echo "=== C4 $NAME done $(date '+%F %T') ==="
echo "all done"
