#!/usr/bin/env bash
# V3-L5/L8/L9 vast.ai 5k smoke A/B run.
#
# Prerequisites (must be ready before invoking):
#   - vast instance with /root/3dgrut/.venv installed (scripts/v3_l589_vast_setup.sh)
#   - /root/data/ncore/clips/9ae151dc-.../pai_9ae151dc-....json present (~7.2 GB)
#   - GPU free (`nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader` < 5%)
#
# Run from Mac:
#   ssh vast-rtx4090 'bash -s' < scripts/v3_l589_vast_smoke_ab.sh 2>&1 | tee /tmp/vast_smoke.log
#
# Outputs (on remote):
#   /root/out/v3_L589_baseline_5k_<ts>/  — OFF run (sym_axis=null, albedo=false, scale=false)
#   /root/out/v3_L589_on_5k_<ts>/        — ON  run (sym_axis=Y, albedo=true, scale=true)
#   Each contains the trained ckpt + metrics.json with V3 diagnostic fields.
#
# Expected duration: 2 × 20-30 min on RTX 4090 = 45-60 min total.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/3dgrut}"
DATA_PATH="${DATA_PATH:-/root/data/ncore/clips/9ae151dc-e87b-41a7-8e85-71772f9603d7/pai_9ae151dc-e87b-41a7-8e85-71772f9603d7.json}"
OUT_DIR="${OUT_DIR:-/root/out}"
N_ITER="${N_ITER:-5000}"
TS=$(date +%y%m_%d_%H%M%S)

cd "$REPO_DIR"
source .venv/bin/activate

[ -f "$DATA_PATH" ] || { echo "ERR: dataset manifest missing at $DATA_PATH"; exit 1; }
mkdir -p "$OUT_DIR"

# Common training flags (multilayer config = NuRec-like 4-layer + dynfix penalties).
# sky_backend=mlp is REQUIRED on vast (nvdiffrast not built in PyTorch container).
COMMON_FLAGS=(
  --config-name apps/ncore_3dgut_mcmc_multilayer
  n_iterations=$N_ITER
  path="$DATA_PATH"
  trainer.sky_backend=mlp
  out_dir="$OUT_DIR"
)

# --- Run 1: BASELINE (all OFF; matches v2 dynfix baseline schema) -----------
EXP_OFF="v3_L589_baseline_5k_${TS}"
echo ""
echo "===================== RUN 1: BASELINE (OFF) ====================="
echo "experiment: $EXP_OFF"
echo "config: symmetric_axis=null  albedo=false  scale=false"
echo "starting at $(date +%H:%M:%S)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
python train.py \
  "${COMMON_FLAGS[@]}" \
  experiment_name="$EXP_OFF" \
  ++layers.overrides.dynamic_rigids.symmetric_axis=null \
  ++layers.overrides.dynamic_rigids.optimize_track_albedo=false \
  ++layers.overrides.dynamic_rigids.optimize_track_scale=false \
  2>&1 | tee "/tmp/${EXP_OFF}.log"
echo "BASELINE done at $(date +%H:%M:%S)"

# --- Run 2: EXPERIMENT (V3-L5 + V3-L8 + V3-L9 all ON) -----------------------
EXP_ON="v3_L589_on_5k_${TS}"
echo ""
echo "===================== RUN 2: EXPERIMENT (ON) ===================="
echo "experiment: $EXP_ON"
echo "config: symmetric_axis=Y  albedo=true  scale=true  warmup=500"
echo "starting at $(date +%H:%M:%S)"
python train.py \
  "${COMMON_FLAGS[@]}" \
  experiment_name="$EXP_ON" \
  ++layers.overrides.dynamic_rigids.symmetric_axis=Y \
  ++layers.overrides.dynamic_rigids.optimize_track_albedo=true \
  ++layers.overrides.dynamic_rigids.optimize_track_scale=true \
  2>&1 | tee "/tmp/${EXP_ON}.log"
echo "EXPERIMENT done at $(date +%H:%M:%S)"

# --- Summary: print metrics.json delta --------------------------------------
echo ""
echo "===================== SUMMARY ====================="
for run in "$EXP_OFF" "$EXP_ON"; do
  echo ""
  echo "=== $run ==="
  # metrics.json lives under <out_dir>/<exp>/pai_<clip>/ours_<N>/metrics.json
  find "$OUT_DIR/$run" -name metrics.json 2>/dev/null | while read mp; do
    echo "  $mp"
    python - <<PY
import json
with open("$mp") as f: m = json.load(f)
keep = ['mean_psnr', 'mean_ssim', 'mean_psnr_masked', 'mean_class_psnr',
        'class_psnr_by_class', 'symmetric_axis',
        'track_albedo_l2_mean', 'track_log_scale_mean', 'track_log_scale_std']
for k in keep:
    if k in m:
        print(f"    {k}: {m[k]}")
PY
  done
done

echo ""
echo "=== DONE. Run rsync from Mac to pull metrics.json + ckpt: ==="
echo "rsync -avz vast-rtx4090:$OUT_DIR/ /Users/etendue/repo/report/v3_L589_5k_results/"
