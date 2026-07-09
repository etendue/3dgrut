#!/bin/bash
# R4e 重锚 30k（扩相机作战 Phase C P0.4）
#
# R3p 同配方（configs/apps/ncore_3dgut_mcmc_multilayer_inceptio.yaml 零改动）
#   + P0.2 EgomaskAuxReader/datasetNcore fallback 接线（cecb6b0，合 main cc26b08）
#   + P0.3 b6a9 视觉多边形静态 ego mask（40277d2，10 台已入 clip itar）
# → 单变量差异 = 仅 ego-mask 生效。
#
# kill-criterion 登记：
#   run 名                 r4e_30k
#   观察点 iter 2k        无 NaN、无死层告警、loss 曲线正常
#   砍单动作              停 run 回 P0.2/P0.3 查（不改配方就地重跑）
#
# 数据 / 环境铁律（CLAUDE.md）：
#   depth-off + num_workers=10 + PYTORCH_CUDA_ALLOC_CONF=expandable_segments
#   config yaml 内置：6-cam camera_ids + cuboid_ts_mode=per_camera_interp
#                       + loss.use_opacity=false
#
# 启动模式（外部 ssh 侧）参照 CLAUDE.md「嵌套 driver 场景」：
#   ssh -n inceptio 'cd ~/repo/3dgrut2-wt/r4e_rebase && \
#     setsid bash scripts/drivers/r4e_rebaseline.sh > /tmp/r4e_30k.log 2>&1 \
#     < /dev/null & echo PID_$!'

set -eo pipefail
export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MANIFEST="$HOME/work/data/inc_b6a9ed61_20s/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9.json"
OUT="$HOME/work/output"
NAME=r4e_30k

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

echo "=== R4e $NAME train start $(date '+%F %T') ==="
python train.py --config-name apps/ncore_3dgut_mcmc_multilayer_inceptio \
  n_iterations=30000 experiment_name="$NAME" \
  "${COMMON[@]}" > "/tmp/${NAME}_train.log" 2>&1

echo "--- 告警计数 (dead / non-finite / [P0.2] fallback / [A5] cuboid_mask):"
grep -ac "layer fully dead"                       "/tmp/${NAME}_train.log" || true
grep -ac "non-finite"                              "/tmp/${NAME}_train.log" || true
grep -ac "\[P0.2\] ego mask via aux itar fallback" "/tmp/${NAME}_train.log" || true
grep -ac "\[A5\] dyn_mask_cuboid filled"           "/tmp/${NAME}_train.log" || true

CKPT=$(ls -dt "$OUT/$NAME"/*/ckpt_last.pt | head -1)
echo "=== R4e $NAME eval ckpt=$CKPT $(date '+%F %T') ==="
python render.py --checkpoint "$CKPT" --out-dir "$OUT/${NAME}_eval" \
  > "/tmp/${NAME}_eval.log" 2>&1
find "$OUT/${NAME}_eval" -name metrics.json | head -1
echo "=== R4e $NAME done $(date '+%F %T') ==="
