#!/bin/bash
# C3 9-cam 30k 四读数（Task 10 Step 2 判定读数）
#
# Step 3 四读数（阶梯每步公用格式，全部 masked 口径）：
#   ① held-out cc_psnr_masked（rear_right_70fov 真外推，vs R5c 15.51）
#   ② novel FID 各档（vs R5c render 167.81 / lateral 172.98-182.01 / yaw 171.22-213.19）
#   ③ per-cam 守护线（已从 c3_9cam_*_30k render.py metrics.json 读出，shared 8 台 ≤ 0.3 dB）
#   ④ automobile class_psnr（vs R5c 18.74）
#
# 本 driver 只跑 ① + ②（③ ④ 已在 30k render.py 直接落盘 metrics.json）
#
# 启动模式（外部 ssh 侧）：
#   TELEW=2.0 ssh -n inceptio 'cd ~/repo/3dgrut2-wt/c3 && \
#     nohup setsid env TELEW=2.0 bash scripts/drivers/c3_9cam_30k_fourread.sh \
#     > /tmp/c3_9cam_fourread.log 2>&1 < /dev/null & disown; echo PID_$!'
#
TELEW="${TELEW:-2.0}"
NAME=c3_9cam_tw${TELEW//./p}_30k
exec > >(tee -a "$HOME/work/output/${NAME}_fourread_driver.log") 2>&1
set -eo pipefail
export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT=$(ls -dt $HOME/work/output/$NAME/*/ckpt_last.pt | head -1)
test -f "$CKPT" || { echo "ERROR: C3 9-cam 30k ckpt missing (NAME=$NAME)"; exit 1; }
echo "=== C3 four-read start $(date '+%F %T') CKPT=$CKPT NAME=$NAME ==="

# ① held-out via existing parametric driver
echo "--- ① held-out driver ---"
bash scripts/drivers/eval_heldout_b6a9.sh "$CKPT" "$NAME"

# ② novel FID via render.py
NAME_FID=${NAME}_novelfid
OUT="$HOME/work/output/$NAME_FID"
echo "--- ② novel FID render ---"
echo "=== $NAME_FID start $(date '+%F %T') ==="
python render.py \
  --checkpoint "$CKPT" \
  --out-dir   "$OUT" \
  --novel-view --novel-fid --render-only \
  > "/tmp/${NAME_FID}_render.log" 2>&1
echo "=== $NAME_FID render done $(date '+%F %T') ==="

echo "--- $NAME_FID 告警计数 (Traceback / OOM / novel-fid lines):"
grep -ac "Traceback"                        "/tmp/${NAME_FID}_render.log" || true
grep -ac "OutOfMemory\|CUDA out of memory"  "/tmp/${NAME_FID}_render.log" || true
grep -ac "mean_render_fid\|mean_novel_fid_" "/tmp/${NAME_FID}_render.log" || true

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

echo "=== C3 four-read all done $(date '+%F %T') ==="
echo "all done"
