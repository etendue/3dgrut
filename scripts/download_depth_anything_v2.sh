#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Stage 11 T11.D1 — Download the DepthAnythingV2 metric outdoor model.
#
# Pulls the transformers-compatible (-hf) variant of the DepthAnythingV2
# metric-outdoor-large checkpoint from HuggingFace into models/depth_anything_v2/
# (gitignored). scripts/dump_depth_priors.py then loads the local snapshot via
# AutoModelForDepthEstimation + AutoImageProcessor.
#
# The metric-outdoor model is public — no HF token required. If a token is
# present (~/.cache/huggingface/token or $HF_TOKEN) it is used automatically.
#
# Usage:
#   bash scripts/download_depth_anything_v2.sh
#   REPO_ID=depth-anything/Depth-Anything-V2-Metric-Outdoor-Large \
#     bash scripts/download_depth_anything_v2.sh   # non -hf fallback
set -euo pipefail

# Default: the transformers-compatible (-hf) repo. Override with REPO_ID env.
REPO_ID="${REPO_ID:-depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf}"

# A800 cannot reach huggingface.co directly ("Network is unreachable"); the
# hf-mirror.com mirror works. Default to it but allow override (set HF_ENDPOINT=
# https://huggingface.co on a host with direct HF access). Verified A800
# 2026-05-29: the non -hf repo 404s on the mirror, the -hf repo resolves.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "[download_depth_anything_v2] HF_ENDPOINT=${HF_ENDPOINT}"

# Resolve repo root from this script's location so it works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/models/depth_anything_v2}"

mkdir -p "${OUT_DIR}"

echo "[download_depth_anything_v2] repo_id=${REPO_ID}"
echo "[download_depth_anything_v2] out_dir=${OUT_DIR}"

python - "$REPO_ID" "$OUT_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, out_dir = sys.argv[1], sys.argv[2]
# Skip the heavy non-safetensors mirrors; keep config + processor + weights.
local = snapshot_download(
    repo_id=repo_id,
    local_dir=out_dir,
    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.model"],
)
print(f"[download_depth_anything_v2] snapshot at: {local}")
PY

echo "[download_depth_anything_v2] done. Contents:"
ls -la "${OUT_DIR}"
