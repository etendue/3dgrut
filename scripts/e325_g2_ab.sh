#!/bin/bash
# ===========================================================================
# E3.2.5 G2 A/B spike — roaddisk(on) vs multilayer(off), 6k step, inceptio.
# ===========================================================================
# 单变量：唯一差异 = roaddisk preset 的 road 几何硬退化 bundle（KNN中值 init /
# 1mm薄盘 / position软冻+rotation硬冻+MCMC豁免）。两边同 depth-off + nw=10 +
# sky_backend=mlp（off 走 CLI 对齐 preset pin）+ test_last=false（eval 全交
# render.py）。lateral 3m/6m lane 判据 via render.py --novel-view --novel-only
# --load-lane-masks（出 lane_grad_corr / band_lpips + interpolated 守护线）。
#
# 启动（inceptio worktree，setsid 嵌套 driver 模式）：
#   ssh -n inceptio 'cd ~/repo/3dgrut2-wt/e325 && setsid bash scripts/e325_g2_ab.sh \
#     > /tmp/e325_g2.log 2>&1 < /dev/null & echo PID_$!'
# ===========================================================================
export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/repo/3dgrut2-wt/e325 || exit 1

CLIP=/home/inceptio/work/data/9ae151dc/pai_9ae151dc-e87b-41a7-8e85-71772f9603d7.json
OUT=/home/inceptio/work/output
COMMON="n_iterations=6000 num_workers=10 dataset.load_lidar_depth_map=false trainer.use_lidar_depth=false trainer.use_depth_prior=false test_last=false path=$CLIP out_dir=$OUT"

echo "########## [1/4] TRAIN ON roaddisk $(date) ##########"
python train.py --config-name apps/ncore_3dgut_mcmc_multilayer_roaddisk $COMMON \
  experiment_name=e325_g2_on || { echo "!!! ON TRAIN FAILED"; exit 2; }

echo "########## [2/4] TRAIN OFF multilayer+sky_mlp $(date) ##########"
python train.py --config-name apps/ncore_3dgut_mcmc_multilayer $COMMON \
  trainer.sky_backend=mlp experiment_name=e325_g2_off || { echo "!!! OFF TRAIN FAILED"; exit 3; }

echo "########## [3/4] EVAL ON lateral 3m/6m lane $(date) ##########"
ON_CKPT=$(ls -t $OUT/e325_g2_on/*/ckpt_last.pt 2>/dev/null | head -1)
echo "ON_CKPT=$ON_CKPT"
python render.py --checkpoint "$ON_CKPT" --out-dir $OUT/e325_g2_on/novel_eval \
  --novel-view --novel-only --load-lane-masks || echo "!!! ON EVAL FAILED"

echo "########## [4/4] EVAL OFF lateral 3m/6m lane $(date) ##########"
OFF_CKPT=$(ls -t $OUT/e325_g2_off/*/ckpt_last.pt 2>/dev/null | head -1)
echo "OFF_CKPT=$OFF_CKPT"
python render.py --checkpoint "$OFF_CKPT" --out-dir $OUT/e325_g2_off/novel_eval \
  --novel-view --novel-only --load-lane-masks || echo "!!! OFF EVAL FAILED"

echo "########## G2 DONE $(date) ##########"
echo "===== ON metrics ====="
find $OUT/e325_g2_on/novel_eval -name "metrics.json" -exec cat {} \;
echo ""
echo "===== OFF metrics ====="
find $OUT/e325_g2_off/novel_eval -name "metrics.json" -exec cat {} \;
echo "########## END ##########"
