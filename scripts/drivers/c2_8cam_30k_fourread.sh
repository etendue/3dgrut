#!/bin/bash
# C2 8-cam 30k 四读数（Task 9 Step 3）
#
# Step 3 四读数（阶梯每步公用格式）：
#   ① held-out cc_psnr / lpips（rear_right_70fov 真外推，扩相机收益 vs R4e −2.35 dB gap）
#   ② novel FID 各档（lateral 1/2/3/6m + yaw 5/10/30/60deg，B4 off-track 天花板同口径）
#   ③ per-cam 守护线（已从 c2_8cam_30k render.py metrics.json 读出）
#   ④ automobile class_psnr（同上，18.74 vs R4e 18.71 = +0.03）
#
# 本 driver 只跑 ① + ②（③ ④ 已在 30k render.py 直接落盘 metrics.json）
#
# 启动模式（外部 ssh 侧）：
#   ssh -n inceptio 'cd ~/repo/3dgrut2-wt/c2_8cam && \
#     nohup setsid bash scripts/drivers/c2_8cam_30k_fourread.sh \
#     > /tmp/c2_8cam_fourread.log 2>&1 < /dev/null & disown; echo PID_$!'
#
exec > >(tee -a "$HOME/work/output/c2_8cam_fourread_driver.log") 2>&1
set -eo pipefail
export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT=$(ls -dt $HOME/work/output/c2_8cam_30k/*/ckpt_last.pt | head -1)
test -f "$CKPT" || { echo "ERROR: c2 8-cam 30k ckpt missing"; exit 1; }
echo "=== C2 four-read start $(date '+%F %T') CKPT=$CKPT ==="

# ① held-out via existing parametric driver
echo "--- ① held-out driver ---"
bash scripts/drivers/eval_heldout_b6a9.sh "$CKPT" c2_8cam_30k

# ② novel FID via render.py (inline, since b5 driver is R4e-hardcoded)
NAME=c2_8cam_30k_novelfid
OUT="$HOME/work/output/$NAME"
echo "--- ② novel FID render ---"
echo "=== $NAME start $(date '+%F %T') ==="
python render.py \
  --checkpoint "$CKPT" \
  --out-dir   "$OUT" \
  --novel-view --novel-fid --render-only \
  > "/tmp/${NAME}_render.log" 2>&1
echo "=== $NAME render done $(date '+%F %T') ==="

echo "--- $NAME 告警计数 (Traceback / OOM / novel-fid lines):"
grep -ac "Traceback"                        "/tmp/${NAME}_render.log" || true
grep -ac "OutOfMemory\|CUDA out of memory"  "/tmp/${NAME}_render.log" || true
grep -ac "mean_render_fid\|mean_novel_fid_" "/tmp/${NAME}_render.log" || true

MJSON=$(find "$OUT" -name metrics.json | head -1)
echo "--- metrics.json path: $MJSON"
if [ -n "$MJSON" ]; then
  python3 -c "
import json
m = json.load(open('$MJSON'))
keys = sorted([k for k in m if 'fid' in k or 'kid' in k])
for k in keys:
    v = m[k]
    if isinstance(v, float):
        print(f'  {k:40s} = {v:.4f}')
    else:
        print(f'  {k:40s} = {v}')
"
fi

echo "=== C2 four-read all done $(date '+%F %T') ==="
echo "all done"
