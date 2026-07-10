#!/bin/bash
# B5 novel FID 链路移植 b6a9（扩相机作战 Phase C P0.5）—— render-only 无悔棋
#
# 对 R4e ckpt 跑 --novel-view --novel-fid → 出 mean_render_fid / mean_novel_fid_<mode>
# / mean_novel_kid_<mode> 全 8 novel mode（lateral_1m/2m/3m/6m + yaw_5/10/30/60）。
#
# 参照 v4 E1.4 3-cam baseline（`b20ff48` + `5e61064`，v4_plan §5 Done Log 2026-06-12）：
#   FID render 75.3 → 1m 124 → 2m 152 → 3m 168 → 6m 193（**全程单调 = 健康信号**）
#   KID render 0.021 → 0.102 @6m（subset 自适应 37）
#
# 验收：
#   ① mean_novel_fid_lateral_{1m<2m<3m<6m} 单调递增（离轴越远越差）
#   ② mean_novel_kid_lateral_* 单调递增
#   ③ mean_render_fid 与 novel 各档 FID 数量级正确
#   ④ 与 B4 held-out gap 方向一致（held-out cc_psnr 崩 −10~−12 dB → novel FID 显著高
#      = 两条独立度量方向一致）
#
# kill-criterion 登记：
#   run 名                 b5_r4e_novel_fid
#   观察点                 (a) 首次输出 "novel-view mode ON — 8 extra renders" header;
#                          (b) inception 权重可用（cache 已有 pt_inception-2015-12-05-*.pth）;
#                          (c) 出 lateral 单调 sanity
#   砍单动作               config 差异 / OOM → 停 run 最小适配，先写回归测试再改代码
#
# 环境铁律（CLAUDE.md）：
#   PATH conda env 3dgrut2 + CUDA_VISIBLE_DEVICES=0 + PYTORCH_CUDA_ALLOC_CONF
#
# 启动（外部 ssh 侧，参照 CLAUDE.md 嵌套 driver）：
#   ssh -n inceptio 'cd ~/repo/3dgrut2-wt/r4e_rebase && \
#     setsid bash scripts/drivers/b5_novel_fid_b6a9.sh > /tmp/b5_r4e_novel_fid.log 2>&1 \
#     < /dev/null & echo PID_$!'

set -eo pipefail
export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="$HOME/work/output/r4e_30k/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-0907_104500/ckpt_last.pt"
OUT="$HOME/work/output/b5_novel_fid_r4e"
NAME=b5_r4e_novel_fid

test -f "$CKPT" || { echo "ERROR: R4e ckpt missing at $CKPT"; exit 1; }

echo "=== B5 $NAME start $(date '+%F %T') ==="
echo "ckpt: $CKPT"
echo "out : $OUT"

python render.py \
  --checkpoint "$CKPT" \
  --out-dir   "$OUT" \
  --novel-view --novel-fid --render-only \
  > "/tmp/${NAME}_render.log" 2>&1

echo "=== B5 $NAME render done $(date '+%F %T') ==="

echo "--- 告警计数 (Traceback / OOM / novel FID lines):"
grep -ac "Traceback"                        "/tmp/${NAME}_render.log" || true
grep -ac "OutOfMemory\|CUDA out of memory"  "/tmp/${NAME}_render.log" || true
grep -ac "mean_render_fid\|mean_novel_fid_" "/tmp/${NAME}_render.log" || true

echo "--- metrics.json path:"
MJSON=$(find "$OUT" -name metrics.json | head -1)
echo "$MJSON"

echo "--- FID/KID key excerpt (sanity):"
if [ -n "$MJSON" ]; then
  python3 -c "
import json, sys
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

echo "=== B5 $NAME all done $(date '+%F %T') ==="
