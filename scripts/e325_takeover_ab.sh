#!/bin/bash
# ===========================================================================
# E3.2.5 takeover spike — roaddisk + 调强 bg_road_penalty（λ0.1→0.4 / z_band0.4→1.5）
# ===========================================================================
# 验证「road-takeover 够强能否把车道线白条从 bg 收回 road 层」。
# 对照 A = e325_g2_on（已有 ckpt，takeover λ=0.1 z_band=0.4）— 白条部分被 bg 抢。
# 新 B = 本 run（takeover λ=0.4 z_band=1.5）— 激进压 bg。
# 验收：① 视觉 road-only 白条是否收回 road；② lateral lane vs A（0.422/0.309）
#       + 守护线 cc 别退太多（调强压 bg 有过压 bg 质量风险，cc 退多→回调 λ/z_band）。
# 启动：ssh -n inceptio 'cd ~/repo/3dgrut2-wt/e325 && setsid bash \
#   scripts/e325_takeover_ab.sh > /tmp/e325_tk.log 2>&1 < /dev/null & echo PID_$!'
# ===========================================================================
export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/repo/3dgrut2-wt/e325 || exit 1

CLIP=/home/inceptio/work/data/9ae151dc/pai_9ae151dc-e87b-41a7-8e85-71772f9603d7.json
OUT=/home/inceptio/work/output
COMMON="n_iterations=6000 num_workers=10 dataset.load_lidar_depth_map=false trainer.use_lidar_depth=false trainer.use_depth_prior=false test_last=false path=$CLIP out_dir=$OUT"

echo "########## TRAIN takeover-strong (roaddisk + bg_road_penalty λ0.4 z_band1.5) $(date) ##########"
python train.py --config-name apps/ncore_3dgut_mcmc_multilayer_roaddisk $COMMON \
  trainer.bg_road_penalty.lambda=0.4 trainer.bg_road_penalty.z_band=1.5 \
  experiment_name=e325_takeover || { echo "!!! TRAIN FAILED"; exit 2; }

echo "########## EVAL takeover lateral 3m/6m lane $(date) ##########"
CK=$(ls -t $OUT/e325_takeover/*/ckpt_last.pt 2>/dev/null | head -1)
echo "CK=$CK"
python render.py --checkpoint "$CK" --out-dir $OUT/e325_takeover/novel_eval \
  --novel-view --novel-only --load-lane-masks || echo "!!! EVAL FAILED"

echo "########## DONE $(date) ##########"
echo "===== takeover metrics (vs A: cc 24.01 / lane3m 0.422 / lane6m 0.309) ====="
find $OUT/e325_takeover/novel_eval -name metrics.json -exec python -m json.tool {} \; 2>/dev/null | grep -iE "cc_psnr_masked|grad_corr_lateral|band_lpips_lateral" | grep -v per_frame
echo "########## END ##########"
