#!/bin/bash
# ===========================================================================
# E3.2.6 takeover 温和版 — roaddisk + bg_road_penalty λ0.2 z_band1.5
# ===========================================================================
# 大g 决策（2026-06-23）：takeover λ0.4 过压（cc 24.01→23.78 退回 multilayer +
# 白条仍没收回 road）→ 试温和 λ0.2（对 B λ0.4 单变量只降 λ，z_band1.5 保持宽
# 范围覆盖白条所在 bg 区，λ 减半减轻过压）。
# 验收：cc 接近/不低于 roaddisk λ0.1 锚 24.01（不过压）+ lateral lane 不退 + 白条
#       视觉改善。兼顾则 roaddisk+λ0.2 进 baseline，否则回退纯 roaddisk。
# 对照：A roaddisk λ0.1 (cc24.01/lane0.422/0.309) · B λ0.4 (cc23.78/0.413/0.297)
#       · multilayer baseline (cc23.74/0.380/0.307)
# 启动：ssh -n inceptio 'cd ~/repo/3dgrut2-wt/e33 && setsid bash \
#   scripts/e326_takeover_l02_ab.sh > /tmp/e326_l02.log 2>&1 < /dev/null & echo PID_$!'
# ===========================================================================
export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/repo/3dgrut2-wt/e33 || exit 1

CLIP=/home/inceptio/work/data/9ae151dc/pai_9ae151dc-e87b-41a7-8e85-71772f9603d7.json
OUT=/home/inceptio/work/output
COMMON="n_iterations=6000 num_workers=10 dataset.load_lidar_depth_map=false trainer.use_lidar_depth=false trainer.use_depth_prior=false test_last=false path=$CLIP out_dir=$OUT"

echo "########## TRAIN roaddisk + takeover λ0.2 z_band1.5 $(date) ##########"
python train.py --config-name apps/ncore_3dgut_mcmc_multilayer_roaddisk $COMMON \
  trainer.bg_road_penalty.lambda=0.2 trainer.bg_road_penalty.z_band=1.5 \
  experiment_name=e326_tk_l02 || { echo "!!! TRAIN FAILED"; exit 2; }

echo "########## EVAL lateral 3m/6m lane $(date) ##########"
CK=$(ls -t $OUT/e326_tk_l02/*/ckpt_last.pt 2>/dev/null | head -1)
echo "CK=$CK"
python render.py --checkpoint "$CK" --out-dir $OUT/e326_tk_l02/novel_eval \
  --novel-view --novel-only --load-lane-masks || echo "!!! EVAL FAILED"

echo "########## E3.2.6 λ0.2 DONE $(date) ##########"
echo "===== metrics (对照 A λ0.1 cc24.01/0.422/0.309 · B λ0.4 cc23.78/0.413/0.297) ====="
find $OUT/e326_tk_l02/novel_eval -name metrics.json -exec python -m json.tool {} \; 2>/dev/null | grep -iE "cc_psnr_masked|grad_corr_lateral|band_lpips_lateral" | grep -v per_frame
echo "########## END ##########"
