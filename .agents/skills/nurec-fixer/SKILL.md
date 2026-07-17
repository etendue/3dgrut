---
name: nurec-fixer
description: >-
  Use to run NVIDIA DiffusionHarmonizer (public successor to the
  older Fixer recipes) to enhance, harmonize, evaluate, or
  fine-tune novel-view frames from NRE / NuRec / 3DGS / NeRF
  reconstructions. Do NOT use for training the 3D reconstruction
  itself (use `nre`) or for sensor-to-NCore conversion (use
  `ncore`).
version: "0.5.0"
tools:
  - Shell
  - Read
  - Write
license: CC-BY-4.0 AND Apache-2.0
compatibility: >-
  Linux + NVIDIA GPU Ampere+ (compute capability >= 8.0; A100,
  A10, L40, H100, RTX 30/40/PRO, B200, GB200), Docker + NVIDIA
  Container Toolkit, ~120 GB free disk. HF_TOKEN required for the
  `nvidia/Harmonizer` checkpoints and the base
  `nvidia/Cosmos-Predict2-0.6B-Text2Image` model; NGC_API_KEY
  needed when pulling the `nvcr.io/nvidia/pytorch:25.10-py3` base
  image from nvcr.io. Code at github.com/NVIDIA/harmonizer;
  runtime is the `harmonizer-cosmos-env` image built from the
  project `Dockerfile.cosmos`, which layers cosmos-predict2 and
  the required patches on top of `nvcr.io/nvidia/pytorch:25.10-py3`.
dependencies:
  - bash
  - docker
  - nvidia-container-toolkit
  - python3
  - git
  - huggingface_hub
metadata:
  author: NVIDIA NRS <nurec-skills@nvidia.com>
  tags:
    - nurec
    - fixer
    - harmonizer
    - diffusionharmonizer
    - reconstruction
    - autonomous-vehicles
    - post-processing
    - evaluation
  upstream: https://github.com/NVIDIA/harmonizer
  hf_model: https://huggingface.co/nvidia/Harmonizer
  hf_dataset: https://huggingface.co/datasets/nvidia/Harmonizer-Dataset
  container: nvcr.io/nvidia/pytorch:25.10-py3
  paper: https://arxiv.org/abs/2602.24096
  product_page: https://research.nvidia.com/labs/sil/projects/diffusion-harmonizer/
  model_license: NVIDIA Open Model License Agreement
---

# NVIDIA DiffusionHarmonizer (NuRec post-processing)

## Purpose

Run NVIDIA DiffusionHarmonizer on rendered images from neural
reconstructions. DiffusionHarmonizer is a single-step,
temporally-aware image diffusion enhancer for NeRF / 3DGS /
NuRec-style renderings. It improves realism, reduces
reconstruction artifacts, and harmonizes inserted dynamic
objects with the surrounding scene.

## When to Use / When NOT to Use

**Use this skill when** the user has rendered frames from NRE,
NuRec, 3DGS, NeRF, or a similar reconstruction pipeline and
wants to enhance, harmonize, evaluate, or optionally fine-tune
the DiffusionHarmonizer model.

**Do NOT use this skill when:**

- The user wants to train or render the 3D reconstruction itself
  (use `nre`).
- The user wants to convert raw sensor data to NCore V4 (use
  `ncore`).
- The user wants a generic photo enhancer. DiffusionHarmonizer
  is tuned for neural-reconstruction artifacts and
  object-insertion failures.
- The user only wants NRE inline rendering with
  `--enable-difix`. That remains an NRE runtime feature; use the
  `nre` skill for the complete `serve-grpc` / `render-grpc`
  command shape.

## What changed from the older Fixer skill

This skill follows the public `NVIDIA/harmonizer` release, not
the older NGC JIT `.pt` artifact recipe. Use these public
release artifacts:

- Code: <https://github.com/NVIDIA/harmonizer>
- Model: `nvidia/Harmonizer` on Hugging Face (the paper checkpoint
  `models/diffusion_harmonizer.pkl`), plus the base
  `nvidia/Cosmos-Predict2-0.6B-Text2Image` model that inference
  also requires.
- Checkpoint download: `./download_checkpoints.sh` from the repo
  root. It fetches the Harmonizer checkpoints into `models/`
  (`diffusion_harmonizer.pkl`, `harmonizer_nontemporal.pt`) and
  the base Cosmos DiT + tokenizer into
  `src/checkpoints/nvidia/Cosmos-Predict2-0.6B-Text2Image/`.
- Runtime: the `harmonizer-cosmos-env` image built from
  `Dockerfile.cosmos` (base `nvcr.io/nvidia/pytorch:25.10-py3`).
- Inference entry: `src/inference_pix2pix_turbo_harmonizer.py`,
  run from inside `/work/src` so it can import its sibling
  modules.
- Evaluation entry: `src/evaluate_test_dataset.py`
- Training entry: `src/train_pix2pix_turbo_harmonizer.py`

Do not use the obsolete standalone recipe that downloads
`nvidia/nre/nurec-fixer:cosmos_3dgut_fixer_harmonizer`, mounts
`harmonizer_temporal.pt`, or runs `inference_jit_harmonizer.py`
inside `nvcr.io/nvidia/pytorch:24.10-py3` unless the user
explicitly asks for that older beta artifact. Do not run
`inference_pretrained_model.py`; the current README documents
`inference_pix2pix_turbo_harmonizer.py` as the inference entry
point.

## Background

DiffusionHarmonizer is described in *DiffusionHarmonizer:
Bridging Neural Reconstruction and Photorealistic Simulation
with Online Diffusion Enhancer*
([arXiv 2602.24096](https://arxiv.org/abs/2602.24096), CVPR 2026).
It distills a pretrained multi-step diffusion model into a
single-step enhancer designed for online simulation and offline
data cleanup.

Two operating modes:

- **Offline:** clean pseudo-training views rendered from a
  reconstruction, then distill the improved views back into the
  3D representation.
- **Online:** enhance frames during simulation/inference by
  harmonizing color and lighting, reconstructing missing or
  inconsistent shadows for inserted actors, and reducing residual
  reconstruction artifacts.

The public model card describes `DiffusionHarmonizer-cosmos-0.6B`,
a Cosmos Predict2 Diffusion Transformer post-trained at
`576x1024` input and output resolution.

## Inputs

- **code_dir** — checkout of
  <https://github.com/NVIDIA/harmonizer>.
- **model_dir** — checkpoints fetched by `./download_checkpoints.sh`
  from the repo root. It places the paper checkpoint at
  `models/diffusion_harmonizer.pkl` and the required base Cosmos
  model under
  `src/checkpoints/nvidia/Cosmos-Predict2-0.6B-Text2Image/`.
- **input_dir** — directory of rendered RGB frames (`.png`,
  `.jpg`, `.jpeg`) to enhance.
- **output_dir** — `inference_pix2pix_turbo_harmonizer.py` does
  not take an output flag; it writes to a sibling folder named
  `<input_dir>_<model_identifier>` next to the input directory.
- **HF_TOKEN** — Hugging Face token with access to
  `nvidia/Harmonizer`, `nvidia/Cosmos-Predict2-0.6B-Text2Image`,
  and (if used) `nvidia/Harmonizer-Dataset`. Accept the
  model/dataset license terms first.
- **NGC_API_KEY** — often needed to authenticate `docker pull`
  from `nvcr.io`. Use only for container pulls, not model
  download.

## Instructions

1. **Validate the host.** Have the agent execute
   `scripts/validate_setup.py` via its standard script runner —
   e.g. `run_script("scripts/validate_setup.py")` or
   `python scripts/validate_setup.py`. It checks Docker, the
   NVIDIA Container Toolkit, GPU architecture, `git`, the
   Hugging Face CLI, token presence, and free disk space — and
   exits non-zero on any missing prerequisite.
2. **Clone the code and build (or pull) the runtime image.**
   Full commands and the Blackwell patch caveat live in
   [`references/inference.md`](references/inference.md).

   ```bash
   git clone https://github.com/NVIDIA/harmonizer.git
   cd harmonizer
   docker build -t harmonizer-cosmos-env -f Dockerfile.cosmos .
   ```

3. **Download the checkpoints.** From the repo root run the
   helper, which fetches both the Harmonizer checkpoints and the
   base Cosmos model into the paths the code expects.

   ```bash
   export HF_TOKEN=<your-hugging-face-token>
   hf auth login --token "$HF_TOKEN"
   ./download_checkpoints.sh
   ```

   Verify `models/diffusion_harmonizer.pkl` and
   `src/checkpoints/nvidia/Cosmos-Predict2-0.6B-Text2Image/`
   exist.
4. **Confirm `input_dir` exists and filenames sort into frame
   order.** The temporal inference script sorts frames with
   natural sort and uses previous outputs as references; prefer
   zero-padded names such as `frame_000001.png`.
5. **Run inference inside the container** with the repo mounted
   at `/work`, then `cd /work/src` and run
   `inference_pix2pix_turbo_harmonizer.py`. Pin
   `-u $(id -u):$(id -g)` so outputs are owned by the host user.
   Output frames land in
   `<input_dir>_<model_identifier>`. Full `docker run` recipe and
   flag matrix in
   [`references/inference.md`](references/inference.md).
6. **Validate that the output frame count matches the input
   frame count,** spot-check frames, and (if ground truth is
   available) run paired evaluation with PSNR/LPIPS — see
   [`references/evaluation.md`](references/evaluation.md).
7. **(Optional) Train or fine-tune.** Download the dataset (or
   prepare JSON manifests in the documented format), then run
   `src/train_pix2pix_turbo_harmonizer.py` with the recommended
   hyperparameters. For fine-tuning, initialize from the released
   checkpoint with `--pretrained_path
   /path/to/diffusion_harmonizer.pkl`. Full recipe + NuRec
   data-pair recipes in
   [`references/training.md`](references/training.md).
8. **(Optional) Teardown.** Follow
   [`references/teardown.md`](references/teardown.md) to remove
   images, code clones, model weights, datasets, and outputs.

## Examples

### Example 1 — Enhance a folder of rendered frames

Run `scripts/validate_setup.py`, build the image once (see
[`references/wrapper-image.md`](references/wrapper-image.md)), then
invoke `inference_pix2pix_turbo_harmonizer.py` inside the
`harmonizer-cosmos-env` container with the repo checkout mounted at
`/work`. From `/work/src`, point `--input_image` at the rendered
frames, `--model_path` at `/work/models/diffusion_harmonizer.pkl`,
set `--model_identifier`, and pass typical flags
`--timestep 250 --resolution 1024 --use_sched`. Enhanced frames are
written to `<input_dir>_<model_identifier>`. The canonical
`docker run` command and the full flag matrix live in
[`references/inference.md`](references/inference.md).

### Example 2 — Quantitative PSNR / LPIPS evaluation

Prepare the paired `test_dataset/{scene}/render` +
`test_dataset/{scene}/gt` layout, then run
`src/evaluate_test_dataset.py` inside the container. See
[`references/evaluation.md`](references/evaluation.md) for the
exact directory shape and the `docker run` command.

### Example 3 — Fine-tune from the public checkpoint

Download `nvidia/Harmonizer-Dataset`, prepare the
training JSON, then run
`src/train_pix2pix_turbo_harmonizer.py` with the multi-GPU
`accelerate launch` recipe in
[`references/training.md`](references/training.md). For
fine-tuning add `--pretrained_path
/path/to/diffusion_harmonizer.pkl` and use
`--fixing_data_weight 3` on the released dataset.

### Example 4 — Non-temporal (frame-by-frame) enhancement

`inference_pix2pix_turbo_harmonizer.py` is temporal by default
and uses previous enhanced frames as references
(`--offset_list -1 -2 -3 -4`). When the user wants each frame
enhanced independently (e.g. unordered images), add
`--nontemporal` to disable temporal conditioning. Full command +
`--offset_list` defaults in
[`references/inference.md`](references/inference.md).

## Consuming the inline Fixer via NRE

`nre --enable-difix` is still the right answer when the user
wants NRE to enhance frames as part of rendering without a
separate harmonizer checkout. That path is owned by the sibling
`nre` skill. Do not mix the standalone DiffusionHarmonizer
HF/Cosmos workflow with NRE's internal cache flags unless the
NRE documentation for the user's tag explicitly says they share
weights.

Use this standalone skill when the user wants the public
DiffusionHarmonizer code, model card, training/evaluation
scripts, or post-processing of frames that already exist on
disk.

## Prerequisites

- **OS:** Linux host.
- **GPU / driver:** NVIDIA GPU Ampere or newer (compute
  capability `>= 8.0`; A100, A10, L40, H100, RTX 30/40/PRO,
  B200, GB200).
- **Container runtime:** Docker with the NVIDIA Container
  Toolkit.
- **Tools:** `git`, `python3`, Hugging Face CLI (`hf` or
  `huggingface-cli`).
- **Secrets:**
  - `HF_TOKEN` with the
    [`nvidia/Harmonizer`](https://huggingface.co/nvidia/Harmonizer)
    and
    [`nvidia/Cosmos-Predict2-0.6B-Text2Image`](https://huggingface.co/nvidia/Cosmos-Predict2-0.6B-Text2Image)
    licenses accepted (required to download model weights and
    the optional dataset).
  - `NGC_API_KEY` (often required for `docker login nvcr.io`
    before pulling `nvcr.io/nvidia/pytorch:25.10-py3`).
- **Disk:** at least ~120 GB free for the runtime image, build
  cache, model weights, optional dataset, and outputs combined.
- **Source / model / dataset:**
  - Code: <https://github.com/NVIDIA/harmonizer>.
  - Model: <https://huggingface.co/nvidia/Harmonizer>.
  - Base model:
    <https://huggingface.co/nvidia/Cosmos-Predict2-0.6B-Text2Image>.
  - Optional dataset:
    <https://huggingface.co/datasets/nvidia/Harmonizer-Dataset>.

The fail-fast check that enforces all of the above is
`scripts/validate_setup.py`.

## Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/validate_setup.py` | Verify Docker, NVIDIA Container Toolkit, GPU architecture, `git`, Hugging Face CLI, token presence, and disk space. No network calls. | `run_script("scripts/validate_setup.py")` or `python scripts/validate_setup.py` |
| `scripts/.env.example` | Template for `HF_TOKEN` and optional `NGC_API_KEY`. | `cp scripts/.env.example .env && set -a && . ./.env && set +a` |

## References

- [`references/inference.md`](references/inference.md) — container
  build, raw-base fallback, Blackwell patches, checkpoint
  download, `inference_pix2pix_turbo_harmonizer.py` flag matrix,
  non-temporal mode.
- [`references/evaluation.md`](references/evaluation.md) — paired
  `test_dataset/` layout and `evaluate_test_dataset.py` command.
- [`references/training.md`](references/training.md) — dataset
  download, training JSON format, multi-GPU `accelerate launch`
  recipe, fine-tuning flags, NuRec data-pair recipes.
- [`references/wrapper-image.md`](references/wrapper-image.md) —
  build and run the project image for repeat inference.
- [`references/troubleshooting.md`](references/troubleshooting.md)
  — extended diagnostic notes.
- [`references/teardown.md`](references/teardown.md) — cleanup
  inventory for images, code, Hugging Face caches, datasets,
  and outputs.
- Public code: <https://github.com/NVIDIA/harmonizer>
- Model card: <https://huggingface.co/nvidia/Harmonizer>
- Dataset: <https://huggingface.co/datasets/nvidia/Harmonizer-Dataset>
- Paper: <https://arxiv.org/abs/2602.24096>
- Project page: <https://research.nvidia.com/labs/sil/projects/diffusion-harmonizer/>

## Limitations

- **Rendered inputs only.** The model is tuned for
  neural-reconstruction renderings and object-insertion
  artifacts, not arbitrary real photos.
- **Primary model resolution is 576x1024.** The inference script
  maps resolution key `1024` to `1024x576`. Only `1024`, `960`,
  and `1360` are supported keys; `1024` matches the model-card
  operating point.
- **Temporal references and filename order matter.** The
  inference script enhances frames in natural-sorted order and
  feeds previous outputs back as temporal references. Use
  zero-padded frame numbers, or pass `--nontemporal` for
  unordered images.
- **Container builds can be large.** The runtime image, build
  cache, model weights, and optional dataset can exceed 100 GB.
- **Training is multi-GPU by default.** The README command
  assumes 8 GPUs with bf16 mixed precision.
- **Public code evolves.** The current README documents
  `inference_pix2pix_turbo_harmonizer.py` as the inference entry
  point; if a future checkout renames it, prefer the script and
  flags present in that checkout.

## Troubleshooting (top 5)

| Error / symptom | Most common cause |
|-----------------|-------------------|
| `docker: could not select device driver ... gpu` | NVIDIA Container Toolkit missing or Docker is not configured for the NVIDIA runtime. |
| `docker pull` 401 / 403 from `nvcr.io` | Docker is not authenticated to NGC, or the API key lacks container access. |
| `hf download ... 401 / 403` | `HF_TOKEN` is missing/expired/lacks read scope, or the model/dataset license has not been accepted. |
| `diffusion_harmonizer.pkl` missing | Checkpoint download path is wrong or incomplete. Re-run `./download_checkpoints.sh` from the repo root. |
| Output files owned by `root` | The `docker run` omitted `-u $(id -u):$(id -g)`. |

Full matrix in
[`references/troubleshooting.md`](references/troubleshooting.md).

## Teardown

A full workflow can leave large artifacts on disk: the Cosmos
image, project image, build cache, `harmonizer` code checkout,
Hugging Face model weights, optional dataset, evaluation
outputs, and enhanced frames. Reclaim them with the inventory
in [`references/teardown.md`](references/teardown.md). Do not
revoke `HF_TOKEN` or `NGC_API_KEY` as normal cleanup. Rotate a
token only if you suspect it was leaked.
