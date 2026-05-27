#!/usr/bin/env bash
# V3-L5/L8/L9 vast.ai bootstrap on a fresh PyTorch 2.4 / CUDA 12.1 container.
#
# Run from Mac:
#   ssh vast-rtx4090 'bash -s' < scripts/v3_l589_vast_setup.sh 2>&1 | tee /tmp/vast_setup.log
#
# What this does (idempotent — safe to rerun on partial failures):
#   1. apt: git, python3.11-venv (uv needs >=3.10), rsync, build-essential
#   2. git clone etendue/3dgrut + checkout V3 branch + submodule init/update
#   3. install_env_uv.sh → creates /root/3dgrut/.venv with PyTorch+CUDA+kaolin
#   4. install_slangc.sh → slangc compiler for 3dgrut CUDA kernels
#   5. Sanity: torch.cuda.is_available() + pytest collect-only on V3 tests
#
# Cost guard: this script is ~10-20 min on RTX 4090; do NOT run nvdiffrast
# rebuild — multilayer.yaml uses sky_backend=mlp on vast (nvdiffrast unavailable
# without GLFW which isn't in this image).
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/3dgrut}"
BRANCH="${BRANCH:-worktree-feat-v3-l589-symmetric}"

echo "=== [1/5] apt prereqs ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git python3.11-venv rsync build-essential curl ca-certificates 2>&1 | tail -5

echo "=== [2/5] git clone etendue/3dgrut @ $BRANCH ==="
if [ -d "$REPO_DIR/.git" ]; then
  echo "  repo exists, fetching latest"
  cd "$REPO_DIR"
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git reset --hard "origin/$BRANCH"
else
  rm -rf "$REPO_DIR"
  git clone https://github.com/etendue/3dgrut.git "$REPO_DIR"
  cd "$REPO_DIR"
  git checkout "$BRANCH"
fi
echo "  HEAD: $(git log --oneline -1)"

echo "=== [3/5] git submodules ==="
git submodule update --init --recursive 2>&1 | tail -5

echo "=== [4/5] install_env_uv.sh ==="
# install_env_uv.sh expects nvcc on PATH or CUDA_HOME set. PyTorch container has
# /usr/local/cuda → ln to canonical path expected by uv pip's CUDA detection.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
[ -x "$CUDA_HOME/bin/nvcc" ] || { echo "ERR: nvcc not at $CUDA_HOME/bin/nvcc"; exit 1; }
# uv comes pre-installed in some PyTorch images; install if missing.
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
bash install_env_uv.sh 2>&1 | tail -30

echo "=== [5/5] sanity ==="
source .venv/bin/activate
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python -c "import slangtorch; print('slangtorch', slangtorch.__version__)" 2>&1 | head -1
# Collect-only on V3 tests to verify imports work (no GPU runtime).
python -m pytest threedgrut/tests/test_dynamic_rigid_init_symmetric.py \
                  threedgrut/tests/test_track_albedo_scale_params.py \
                  threedgrut/tests/test_track_warmup_optim.py \
                  threedgrut/tests/test_v3_metrics_diagnostics.py \
                  --collect-only -q 2>&1 | tail -10

echo "=== DONE ==="
echo "Next step: rsync NCore 9ae151dc clip from A800 → /root/data/ncore/clips/"
echo "Then run scripts/v3_l589_vast_smoke_ab.sh"
