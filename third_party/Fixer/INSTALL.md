# Installing NVIDIA Fixer dependencies (for `threedgrut/correction/difix.py`)

Source: https://github.com/nv-tlabs/Fixer
Weights: https://huggingface.co/nvidia/Fixer

The DiFix post-processor depends on a heavy NVIDIA stack (`cosmos_predict2`,
`imaginaire`, `transformer_engine`, `diffusers`) that we deliberately keep
**out of the 3dgrut2 `pyproject.toml`**. This file documents how to set up the
DiFix runtime on a GPU host (recommended: Vast.ai H100 / A100 / RTX 4090).

The wrapper `threedgrut/correction/difix.py` uses **lazy import** — these
dependencies are only required when `render.use_difix=true` is set at runtime,
not at module-import time.

## Path A — Docker (recommended, mirrors NVIDIA's tested environment)

```bash
docker pull nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2
cd ~/3dgrut2
docker run --gpus all \
  -v "$PWD":/work \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  --rm -it nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2

# inside container
cd /work
pip install -e .                            # install 3dgrut2 itself
bash scripts/download_difix.sh              # fetch weights into HF cache
python -c "from threedgrut.correction.difix import DifixPostProcessor; print('import OK')"
```

## Path B — Native pip (lighter but fragile)

Only attempt this on a fresh Vast.ai instance with PyTorch ≥ 2.4. Known
failure modes: `transformer_engine` build needs CUDA toolkit + matching torch;
`cosmos_predict2` is not on PyPI as of 2026-05.

```bash
# Inspect Dockerfile.cosmos on https://github.com/nv-tlabs/Fixer for the
# exact pip / git clone steps used inside the official container.
pip install diffusers
pip install transformer_engine
pip install git+https://github.com/NVIDIA/cosmos-predict2.git    # check actual URL
pip install git+https://github.com/NVlabs/imaginaire.git          # check actual URL
```

If pip-only fails, fall back to Path A.

## Weight layout (downloaded by `scripts/download_difix.sh`)

```
$HF_HOME/nvidia-Fixer/
├── pretrained_fixer.pkl              # Pix2Pix_Turbo state_dict (~1.2 GB)
└── models/base/
    ├── model_fast_tokenizer.pt       # DC-AE tokenizer
    └── tokenizer_fast.pth
```

The vendored `pix2pix_turbo_nocond_cosmos_base_faster_tokenizer.py` hardcodes:

```python
config.dit_path = '/work/models/base/model_fast_tokenizer.pt'
config.tokenizer["vae_pth"] = '/work/models/base/tokenizer_fast.pth'
```

When running outside the Docker `/work` layout, you must override these paths
via `DifixPostProcessor.__init__` arguments — see `threedgrut/correction/difix.py`
for the parameter list.

## License

- `LICENSE.txt` — Apache-2.0 (code, vendored from nv-tlabs/Fixer)
- `THIRD_PARTY_LICENSE.txt` — third-party deps' licenses
- Model weights — NVIDIA Open Model License (commercial use permitted; do not
  commit weights to git)
