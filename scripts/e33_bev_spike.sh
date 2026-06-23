#!/bin/bash
# ===========================================================================
# E3.3 BEV 纹理平面化 spike — G1 micro (500 step + grid-trains verify) → G2 6k A/B
# ===========================================================================
# on=bevroad preset (road 颜色 ← learnable BEV grid，pre-bake) vs off=multilayer.
# 单变量：唯一差异 = road BEV override（两边同 depth-off + nw=10 + sky_mlp +
# test_last=false → eval 全交 render.py）。前置认知（E3.2.6）：takeover 拉不动
# 白条、road 层仍弱；本 spike 测 BEV 颜色参数化在 road 弱现状下能否独立改善 lane。
#
# 启动（inceptio worktree，setsid 嵌套 driver 模式，CLAUDE.md 铁律）：
#   ssh -n inceptio 'cd ~/repo/3dgrut2-wt/e33 && setsid bash scripts/e33_bev_spike.sh \
#     > /tmp/e33_spike.log 2>&1 < /dev/null & echo PID_$!'
# ===========================================================================
export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/repo/3dgrut2-wt/e33 || exit 1

CLIP=/home/inceptio/work/data/9ae151dc/pai_9ae151dc-e87b-41a7-8e85-71772f9603d7.json
OUT=/home/inceptio/work/output
DEPTHOFF="dataset.load_lidar_depth_map=false trainer.use_lidar_depth=false trainer.use_depth_prior=false"
COMMON="num_workers=10 test_last=false path=$CLIP out_dir=$OUT $DEPTHOFF"

echo "########## [G1] 500-step micro-spike: numerical + grid-trains $(date) ##########"
python train.py --config-name apps/ncore_3dgut_mcmc_multilayer_bevroad $COMMON \
  n_iterations=500 experiment_name=e33_g1 || { echo "!!! G1 TRAIN FAILED"; exit 2; }
G1_CKPT=$(ls -t $OUT/e33_g1/*/ckpt_last.pt 2>/dev/null | head -1)
echo "G1_CKPT=$G1_CKPT"
python scripts/e33_verify_grid.py "$G1_CKPT" || { echo "!!! G1 GRID-TRAINS GATE FAILED — aborting before 6k"; exit 3; }
echo "########## [G1] PASS — grid trains + numerically stable, proceeding to G2 ##########"

echo "########## [G2 1/3] TRAIN ON bevroad 6k $(date) ##########"
python train.py --config-name apps/ncore_3dgut_mcmc_multilayer_bevroad $COMMON \
  n_iterations=6000 experiment_name=e33_g2_on || { echo "!!! ON TRAIN FAILED"; exit 4; }

echo "########## [G2 2/3] TRAIN OFF multilayer+sky_mlp 6k $(date) ##########"
python train.py --config-name apps/ncore_3dgut_mcmc_multilayer $COMMON \
  trainer.sky_backend=mlp n_iterations=6000 experiment_name=e33_g2_off || { echo "!!! OFF TRAIN FAILED"; exit 5; }

echo "########## [G2 3/3] EVAL lateral 3m/6m lane $(date) ##########"
ON_CKPT=$(ls -t $OUT/e33_g2_on/*/ckpt_last.pt 2>/dev/null | head -1)
OFF_CKPT=$(ls -t $OUT/e33_g2_off/*/ckpt_last.pt 2>/dev/null | head -1)
echo "ON_CKPT=$ON_CKPT"
echo "OFF_CKPT=$OFF_CKPT"
python render.py --checkpoint "$ON_CKPT" --out-dir $OUT/e33_g2_on/novel_eval \
  --novel-view --novel-only --load-lane-masks || echo "!!! ON EVAL FAILED"
python render.py --checkpoint "$OFF_CKPT" --out-dir $OUT/e33_g2_off/novel_eval \
  --novel-view --novel-only --load-lane-masks || echo "!!! OFF EVAL FAILED"

echo "########## E3.3 SPIKE DONE $(date) ##########"
echo "===== ON (bevroad) metrics ====="
find $OUT/e33_g2_on/novel_eval -name "metrics.json" -exec cat {} \;
echo ""
echo "===== OFF (multilayer) metrics ====="
find $OUT/e33_g2_off/novel_eval -name "metrics.json" -exec cat {} \;
echo "########## END ##########"
