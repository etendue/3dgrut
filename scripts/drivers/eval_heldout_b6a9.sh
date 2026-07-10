#!/bin/bash
# P0.6 held-out 一键评估驱动 b6a9（扩相机作战 Phase C 收官前置项）
#
# 用法：bash scripts/drivers/eval_heldout_b6a9.sh <ckpt_path> <out_tag>
#
# 两组 render（B4 协议的 4-组简化版，rear_right 是 R4e 6-cam 训练之外的真 held-out）：
#   ① train-cam 组：R4e 6 训练相机（front_wide/cross_L/cross_R/left_wide/right_wide/back_rear_wide）
#   ② held-out 组：camera_rear_right_70fov（P0.3 已手工 mask 但 R4e 未参训）
#
# 两组都用 --dataset-cameras → BilateralGrid exposure 自动禁用 → **cc 口径统一可比**
# （render.py L413 "📷 --dataset-cameras active → BilateralGrid exposure model disabled"）；
# --render-only 关 aux/lane/lidar/depth/NTA 提速，保留 extra metrics（psnr/ssim/lpips）+ FID/KID。
#
# 汇总输出（结尾 SUMMARY 段）：
#   train  cc_psnr / lpips / mean_render_fid
#   heldout cc_psnr / lpips / mean_render_fid
#   gap (held-out − train) 三口径
#
# 阶梯每步（C1/C2/C3/C4）复用：bash eval_heldout_b6a9.sh <ckpt> <run_tag>
# → 与 novel FID / per-cam psnr / automobile 组成 C 阶梯每步四读数中的第一读数。
#
# v4 E1.3 协议出处：B4 底稿（`.superpowers/sdd/b4_summary.md`，尚未落盘的实测数字见
# `docs/superpowers/plans/2026-07-06-b4-b1-offtrack-anchors.md`），本 driver 是其
# 4-组流程的 2-组封装版（train + rear_right held-out）。
#
# kill-criterion 登记：
#   run 名                 p06_heldout_<tag>
#   观察点                 (a) train 组 6 相机 [P0.2] fallback 命中；held-out 组 1 相机 [P0.2] 命中
#                          (b) log 中出现 "📷 --dataset-cameras active" 双次（两组各一次）
#                          (c) 两组 metrics.json 都出 mean_cc_psnr + mean_render_fid + per_camera
#   砍单动作               缺 exposure 禁用告警 / 缺 --dataset-cameras 生效 log
#                          → 停 run 查 conf struct 或 dataset init 路径；改码先写回归测试

set -eo pipefail
export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="${1:?usage: $0 <ckpt_path> <out_tag>}"
TAG="${2:?usage: $0 <ckpt_path> <out_tag>}"
OUT="$HOME/work/output/heldout_${TAG}"

TRAIN_CAMS='camera_front_wide_120fov,camera_cross_left_120fov,camera_cross_right_120fov,camera_left_wide_90fov,camera_right_wide_90fov,camera_back_rear_wide_90fov'
HELDOUT_CAMS='camera_rear_right_70fov'

test -f "$CKPT" || { echo "ERROR: ckpt missing at $CKPT"; exit 1; }

mkdir -p "$OUT/train" "$OUT/heldout"

# driver-owned log — 不依赖 launcher 侧 shell 重定向（P0.6 首跑教训：
# setsid + `ssh -n "... > /tmp/xxx 2>&1 &"` 时 detach 后 launcher redirect fd 关闭，
# driver 侧 echo/SUMMARY 输出被丢弃，Monitor 挂 /tmp 关键字永远等不到）。
exec > >(tee -a "$OUT/driver.log") 2>&1

echo "=== P0.6 heldout $TAG start $(date '+%F %T') ==="
echo "ckpt   : $CKPT"
echo "out    : $OUT"
echo "train  : $TRAIN_CAMS"
echo "heldout: $HELDOUT_CAMS"
echo

echo "--- run 1: train-cam 组（R4e 6 训练相机同口径重跑，exposure OFF → cc） ---"
python render.py \
  --checkpoint "$CKPT" \
  --out-dir "$OUT/train" \
  --dataset-cameras "$TRAIN_CAMS" \
  --novel-fid --render-only \
  > "$OUT/train_render.log" 2>&1

echo "--- run 2: held-out 组（camera_rear_right_70fov 未参训 = 真外推） ---"
python render.py \
  --checkpoint "$CKPT" \
  --out-dir "$OUT/heldout" \
  --dataset-cameras "$HELDOUT_CAMS" \
  --novel-fid --render-only \
  > "$OUT/heldout_render.log" 2>&1

echo "=== P0.6 heldout $TAG render done $(date '+%F %T') ==="

# —— 一次性 sanity + 汇总 ——
TJSON=$(find "$OUT/train"   -name metrics.json 2>/dev/null | head -1)
HJSON=$(find "$OUT/heldout" -name metrics.json 2>/dev/null | head -1)

echo
echo "--- 告警计数（两 log 合计）:"
echo -n "Traceback           : "; grep -acE "Traceback"                                            "$OUT/train_render.log" "$OUT/heldout_render.log" | awk -F: '{s+=$2}END{print s}'
echo -n "OOM/CUDA err        : "; grep -acE "OutOfMemory|CUDA out of memory|CUDA error"           "$OUT/train_render.log" "$OUT/heldout_render.log" | awk -F: '{s+=$2}END{print s}'
echo -n "exposure disable告警: "; grep -acE "dataset-cameras active|BilateralGrid.*disabled"      "$OUT/train_render.log" "$OUT/heldout_render.log" | awk -F: '{s+=$2}END{print s}'
echo -n "[P0.2] fallback     : "; grep -acE "\[P0.2\] ego mask via aux itar fallback"             "$OUT/train_render.log" "$OUT/heldout_render.log" | awk -F: '{s+=$2}END{print s}'

echo
echo "--- metrics.json paths:"
echo "train  : $TJSON"
echo "heldout: $HJSON"

if [ -n "$TJSON" ] && [ -n "$HJSON" ]; then
python3 - "$TJSON" "$HJSON" "$TAG" <<'PY'
import json, sys
tjson, hjson, tag = sys.argv[1], sys.argv[2], sys.argv[3]
t = json.load(open(tjson))
h = json.load(open(hjson))

def g(m, k, default=None):
    return m.get(k, default)

tc, hc = g(t, "mean_cc_psnr"), g(h, "mean_cc_psnr")
tp, hp = g(t, "mean_psnr"),    g(h, "mean_psnr")
tl, hl = g(t, "mean_lpips"),   g(h, "mean_lpips")
tf, hf = g(t, "mean_render_fid"), g(h, "mean_render_fid")
tk, hk = g(t, "mean_render_kid"), g(h, "mean_render_kid")
tn, hn = len(g(t, "per_camera", {}) or {}), len(g(h, "per_camera", {}) or {})

def fmt(v, prec=4):
    return "n/a" if v is None else f"{v:.{prec}f}"

def diff(a, b, prec=4):
    if a is None or b is None: return "n/a"
    return f"{a-b:+.{prec}f}"

print()
print(f"===================== P0.6 SUMMARY [{tag}] =====================")
print(f"                     train-cam ({tn})     held-out ({hn})     Δ (held − train)")
print(f"  cc_psnr           {fmt(tc):>15s}  {fmt(hc):>15s}  {diff(hc, tc):>16s}")
print(f"  psnr              {fmt(tp):>15s}  {fmt(hp):>15s}  {diff(hp, tp):>16s}")
print(f"  lpips             {fmt(tl):>15s}  {fmt(hl):>15s}  {diff(hl, tl):>16s}")
print(f"  render FID        {fmt(tf, 2):>15s}  {fmt(hf, 2):>15s}  {diff(hf, tf, 2):>16s}")
print(f"  render KID        {fmt(tk):>15s}  {fmt(hk):>15s}  {diff(hk, tk):>16s}")
print()

pc_h = g(h, "per_camera", {}) or {}
if pc_h:
    print("  --- held-out per-cam ---")
    for cam in sorted(pc_h):
        r = pc_h[cam]
        n  = r.get("n_frames", "?")
        cp = r.get("mean_cc_psnr")
        lp = r.get("mean_lpips")
        print(f"    {cam}: n={n}  cc_psnr={fmt(cp)}  lpips={fmt(lp)}")
print()
PY
fi

echo "=== P0.6 heldout $TAG all done $(date '+%F %T') ==="
