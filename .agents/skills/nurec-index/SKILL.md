---
name: nurec-index
description: >-
  Router for NVIDIA NuRec / NRE / 3DGUT / USDZ / NCore V4 / asset
  harvest / frame cleanup tasks — picks the right sibling (nre,
  ncore, asset-harvester, nurec-fixer, physical-ai-datasets). Use
  when the sub-skill is unclear or a multi-stage pipeline is needed;
  do NOT use for non-NuRec tasks or to run any pipeline itself.
version: "0.2.4"
tools:
  - Read
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: NVIDIA NRS <nurec-skills@nvidia.com>
  tags:
    - nurec
    - index
    - router
    - table-of-contents
  canonical_repo: https://github.com/NVIDIA/nurec-skills
  canonical_skills_dir: .agents/skills/
  agentskills_io_compatible: true
  trigger_keywords:
    - nurec
    - nurec index
    - nurec router
    - neural reconstruction engine
    - NRE
    - 3DGUT
    - 3DGRT
    - USDZ
    - sensorsim
    - novel view synthesis
    - PhysicalAI-Autonomous-Vehicles-NuRec
    - NuRec pipeline
    - NuRec workflow
    - warm serve-grpc
    - nre thin client
    - batch_render_rgb
    - nurec teardown
---

# NuRec Skills Index

A routing skill. It decides *which* sibling skill to read next for a
NuRec / Neural Reconstruction task. Five siblings cover the full
pipeline:

- `physical-ai-datasets` — find an existing NVIDIA dataset.
- `ncore` — convert raw sensor data into NCore V4.
- `nre` — train a 3DGUT reconstruction and render novel views from
  NCore V4 (or a pre-trained USDZ).
- `asset-harvester` — extract per-object 3D Gaussian Splat assets
  from sparse AV-clip views.
- `nurec-fixer` — clean up artifacts in already-rendered frames.

**Use this index when** the user mentions NuRec / NRE / 3DGUT /
USDZ / "render this clip" / "convert this bag" / "extract objects"
but the right sub-skill is not yet obvious. Always read this first.

**Do NOT use this index when:**

- The right sub-skill is already obvious (open it directly).
- The task is not a NuRec task (this skill will not help).
- Hands-on implementation steps are needed — defer to the sub-skill
  this index points at.

## Purpose

This skill exists so an agent never has to *guess* which NuRec-family
skill to read next. It is a hand-curated router for the
five-skill NuRec family and nothing else.

**Use cases this skill is built for:**

- **Disambiguate a NuRec request.** The user says "render this
  clip", "convert this bag", "fix these frames", or "extract this
  car" and you need to pick the one sibling skill that owns that
  verb.
- **Bootstrap a multi-stage pipeline.** The user's goal needs two or
  three siblings in a fixed order (convert → train → render, dataset
  → render, harvest → insert, render → harmonize). This index points
  you at the right starting sibling and the matching workflow A–G in
  [`references/workflows.md`](references/workflows.md).
- **Translate NuRec jargon for a beginner.** NuRec vs NRE, USDZ vs
  NCore V4, 3DGUT vs 3DGRT, NRE's built-in Fixer vs the standalone
  DiffusionHarmonizer — see [Easy mix-ups](#easy-mix-ups).
- **Locate a sibling skill that is not on disk.** Defer to
  [`references/discovery.md`](references/discovery.md) instead of
  guessing a path.
- **Plan disk cleanup across the family.** A full NuRec workflow
  can leave 150 GB+ behind; defer to
  [`references/teardown.md`](references/teardown.md) for the
  documented order.

**Use cases this skill is explicitly NOT built for:**

- Running any container, training job, rendering job, conversion,
  dataset download, server, or teardown — always handed off to the
  sibling.
- Routing tasks outside the NuRec family (Omniverse, Isaac Sim,
  CARLA, Cosmos-* training, generic Hugging Face downloads).
- Discovering newly-added sibling skills automatically — the
  catalogue is hand-curated and must be edited by hand (see
  [Keeping this index up to date](#keeping-this-index-up-to-date)).

## Instructions

Follow these steps when answering a NuRec-shaped question:

1. **Classify the request.** Read the user's goal and match it to a
   row in [Pick a skill](#pick-a-skill). Multi-stage tasks
   (convert + train, dataset + render) usually start with `ncore`
   or `physical-ai-datasets` and then hand off to `nre`.
2. **Open exactly one sibling skill next.** Refer to it by `name:`
   (e.g. `nre`), not by file path — names are portable across
   runtimes. If the sibling is not on disk locally, follow
   [`references/discovery.md`](references/discovery.md).
3. **Defer execution to the sibling.** This index is read-only and
   describes routing only; it never runs containers, training,
   rendering, conversion, or downloads itself.
4. **For multi-step pipelines, walk a workflow in order.** Pick a
   workflow ID (A–G) from
   [`references/workflows.md`](references/workflows.md) and open
   the named siblings one at a time, in the listed order. Do not
   collapse steps from that file into this index.
5. **On disk cleanup**, follow
   [`references/teardown.md`](references/teardown.md). A complete
   NuRec workflow can leave 150 GB+ on disk; each sibling owns its
   own teardown.
6. **Never echo secrets** (`NGC_API_KEY`, `HF_TOKEN`). Use each
   sibling's `scripts/validate_setup.py` when present, or
   `hf auth whoami`. See the "Secrets" block in
   [`references/teardown.md`](references/teardown.md).

## What is NuRec?

**NuRec** (NVIDIA Omniverse Neural Reconstruction) takes a
recording from cameras and LiDAR — usually from a self-driving car
or a robot — and turns it into a 3D scene that can be re-rendered
from any angle.

Names that appear often:

- **NRE** — "Neural Reconstruction Engine", the program that does
  the actual training and rendering. NuRec is the product name;
  NRE is the engine inside it.
- **USDZ** — the file format the trained scene is saved in. A zip
  archive that Omniverse, Isaac Sim, and CARLA know how to open.
- **NCore V4** — the input format. Raw recordings must be
  converted into NCore V4 before NRE can consume them.
- **3DGUT / 3DGRT** — two flavours of 3D Gaussian Splatting that
  NRE uses internally. Most users never pick between them; the
  default Hydra recipe handles it.

A typical NuRec project has three stages:

1. **Get the input** — convert a recording into NCore V4, or
   download a pre-converted dataset from Hugging Face.
2. **Train the reconstruction** — feed NCore V4 to NRE; out comes
   a USDZ file.
3. **Render new views** — point NRE at the USDZ to render images,
   videos, or LiDAR sweeps from any camera angle.

Some projects skip step 2 entirely by downloading a USDZ that
NVIDIA has already trained.

## Pick a skill

Match the user's goal in the left column, then open the skill on
the right. Arrows mean "do these in order".

| Goal | Skill to read |
|------|---------------|
| Find or download a NuRec dataset NVIDIA has published | `physical-ai-datasets` |
| Convert camera / LiDAR / radar / depth / stereo into NCore V4 | `ncore` |
| Write a new converter for a sensor setup not yet supported (drone, RGB-D, ROS 2 bag, COLMAP, …) | `ncore` |
| Train a 3D reconstruction from an NCore clip | `ncore` → `nre` |
| Generate the extra inputs NRE needs (segmentation, depth, ego mask) | `nre` (via `nre-tools` container) |
| Render a USDZ along the **original** camera positions | `nre` |
| Render at full resolution / highest quality | `nre` ("Quality presets" inside that skill) |
| Render along a **shifted** trajectory | `nre` |
| Render through a server so a simulator can ask for frames | `nre` (`serve-grpc`) |
| Render the same USDZ many times back-to-back from Python with minimal per-call latency | `nre` (warm `serve-grpc` + thin Python client / `batch_render_rgb`) |
| Render LiDAR sweeps (point clouds) from a USDZ | `nre` (`render-grpc --lidar`) |
| Skip training and just render an NVIDIA-built driving scene | `physical-ai-datasets` → `nre` |
| Skip training and use a pre-built indoor robotics scene | `physical-ai-datasets` → `nre` (then Isaac Sim 5.1) |
| Extract individual 3D objects (cars, pedestrians) from a driving clip | `asset-harvester` |
| Add, remove, or replace cars / pedestrians in a NuRec scene | `asset-harvester` → `nre` |
| Clean up rendered frames (ghosting, floaters, flickering, inserted-object lighting) | `nurec-fixer`, **or** `--enable-difix` inside `nre` |
| Export the scene as PLY / mesh / depth maps / ego mask | `nre` |
| Upgrade an old USDZ so newer NRE versions load it faster | `nre` (`upgrade-artifact`) |
| Open a USDZ or PLY in a browser viewer | `nre` (`viewer` / `ply_viewer`) |
| Measure rendering quality (PSNR / SSIM / LPIPS) | `nre` (`eval-rendering-metrics`) |
| Benchmark different reconstruction methods on the same scenes | `physical-ai-datasets` (`PhysicalAI-NuRec-PPISP`) → `nre` |
| Train on multiple GPUs or on SLURM | `nre` (Workflow D) |

For multi-step pipelines, see
[`references/workflows.md`](references/workflows.md) (workflows
A–G).

## The skills in this folder

Open siblings by their **Name** — that is the canonical
identifier. The **Folder** column is just where the skill lives in
this repo if it has been cloned locally.

| Name | Folder | What it does |
|------|--------|--------------|
| `physical-ai-datasets` | `physical-ai-datasets/` | Catalog and download recipes for every NVIDIA Physical AI dataset on Hugging Face (driving, robotics, manipulation, NuRec scenes, benchmarks). |
| `ncore` | `ncore/` | Converts any sensor recording into NCore V4. Also covers writing a new converter. |
| `nre` | `nre/` | The Neural Reconstruction Engine itself. Trains reconstructions, renders frames, exports meshes / point clouds / depth, edits actors, runs the gRPC server, browses results, evaluates quality. |
| `asset-harvester` | `asset-harvester/` | Apache-2.0 pipeline that extracts individual 3D objects from sparse driving-clip views as `.ply` Gaussian splats plus metadata. |
| `nurec-fixer` | `nurec-fixer/` | Standalone DiffusionHarmonizer workflow that cleans up rendered frames, harmonizes inserted actors, evaluates PSNR/LPIPS, and optionally fine-tunes the model. |

## Easy mix-ups

These pairs sound similar but are different things. When in doubt,
come back here.

- **NuRec vs NRE.** NuRec is the product name; NRE is the engine
  inside it. Both map to the same skill: `nre`.
- **NRE's built-in Fixer vs standalone DiffusionHarmonizer.**
  `--enable-difix` inside `nre` is an inline NRE rendering
  feature. The `nurec-fixer` skill covers the standalone public
  DiffusionHarmonizer release (code at `NVIDIA/harmonizer`, model
  at `nvidia/DiffusionHarmonizer`) for frames already on disk,
  paired evaluation, and fine-tuning. Do not assume these two
  paths share cache layout or weights unless the NRE tag's own
  docs say so.
- **`ncore` vs `nre`.** They run **in order**, never as
  alternatives. `ncore` produces the input format; `nre` reads it.
- **`asset-harvester` vs `nre`'s `export-external-assets`.** Asset
  Harvester **produces** the per-object `.ply` files. `nre`'s
  `export-external-assets` **packages** them into a USDZ. Always
  Asset Harvester first.
- **Cosmos-Drive-Dreams vs PhysicalAI-Autonomous-Vehicles-NuRec.**
  Both are AV datasets on Hugging Face; both are managed by
  `physical-ai-datasets`. Cosmos-Drive-Dreams is **synthetic**
  weather-augmented video (CC-BY-4.0). The NuRec dataset is
  **real** driving scenes turned into renderable USDZs (gated AV
  License).

## Prerequisites

This index is read-only and needs no tooling. The *real*
prerequisites live in each sibling skill:

| Sibling | Hard requirements |
|---------|-------------------|
| `ncore` | Python 3.10+, `pip install nvidia-ncore`; source data on disk |
| `nre` | Linux x86_64, NVIDIA GPU (Ampere+, ≥24 GB VRAM), Docker 23+, NVIDIA Container Toolkit, `NGC_API_KEY` |
| `asset-harvester` | Linux + conda, NVIDIA driver ≥570, ~16 GB VRAM, `HF_TOKEN` |
| `nurec-fixer` | Linux, NVIDIA GPU (Ampere+), Docker, NVIDIA Container Toolkit, `HF_TOKEN`; `NGC_API_KEY` may be needed for `nvcr.io` pulls |
| `physical-ai-datasets` | Python + `huggingface_hub`, `HF_TOKEN` (gated datasets need license acceptance) |

Each sibling skill ships `scripts/validate_setup.py` (where
applicable) — run it before invoking the workflow. Never echo
secret env vars; see
[`references/teardown.md`](references/teardown.md).

## Examples

Concrete routing examples. The user prompt is on the left; the
correct action this index should take is on the right.

### Example 1 — single-skill routing

> **User:** "I have a Waymo Open recording. How do I get it into a
> format NRE accepts?"

- Match the row "Convert camera / LiDAR / radar / depth / stereo
  into NCore V4" in [Pick a skill](#pick-a-skill).
- Open the `ncore` skill next; do not run any commands from this
  index.

### Example 2 — multi-stage pipeline

> **User:** "I have a driving clip. I want to train NuRec and then
> render a new camera trajectory through it."

- Match "Train a 3D reconstruction from an NCore clip" → `ncore`
  → `nre`.
- Cross-check workflow A in
  [`references/workflows.md`](references/workflows.md).
- Open `ncore` first, then `nre`, in that order.

### Example 3 — skip training, just render

> **User:** "Can I just see NuRec working on a scene NVIDIA already
> built?"

- Match "Skip training and just render an NVIDIA-built driving
  scene" → `physical-ai-datasets` → `nre`.
- Cross-check workflow B.
- Open `physical-ai-datasets` first to download **one** scene
  (~1.5–2 GB), then `nre` to render.

### Example 4 — actor editing

> **User:** "I want to add a pedestrian to this NuRec scene."

- Match "Add, remove, or replace cars / pedestrians" →
  `asset-harvester` → `nre`.
- Cross-check workflow D.
- Confirm the original NCore clip is on disk first (Asset
  Harvester needs it), then open `asset-harvester`, then `nre`
  with `serve-grpc --enable-editing-actors`.

### Example 5 — ambiguous "fix it" request

> **User:** "My rendered frames look fuzzy with weird floaters.
> Can you clean them up?"

- Match the "Clean up rendered frames" row. Two valid paths:
  - **Quick path:** `--enable-difix` inside `nre` if the user is
    already rendering through NRE.
  - **Standalone path:** `nurec-fixer` for already-rendered
    frames on disk or for paired evaluation / fine-tuning.
- Ask which interface the user wants, then open the matching
  skill. See workflow E for details.

### Example 6 — sibling skill not on disk

> **User:** "Where do I find the `nre` skill? It is not in my
> repo."

- Do not guess a path. Follow
  [`references/discovery.md`](references/discovery.md): look under
  `.agents/skills/nre/SKILL.md`, then `.Codex/skills/nre/`, then
  `.cursor/skills/nre/`, then `~/.cursor/skills/nre/`, and as a
  last resort clone
  `https://github.com/NVIDIA/nurec-skills`.

## Limitations

- **Routes only**; never runs pipelines. For hands-on steps, open
  the sibling skill this index points to.
- **Hand-curated catalogue.** A newly-added sibling skill is not
  discoverable here until someone updates
  [Pick a skill](#pick-a-skill).
- **Names are portable, paths are not.** Cross-skill links assume
  the canonical layout (`.agents/skills/<name>/SKILL.md`). In a
  different runtime layout, prefer name-based skill resolution
  over file paths.
- **No Omniverse / Isaac Sim integration steps** — those live in
  upstream Omniverse / Isaac docs, not in the NuRec skill family.

## Troubleshooting

| Symptom | Likely cause | Resolution |
|---------|--------------|------------|
| Agent picked the wrong sibling skill | The user's task spans multiple stages (e.g. convert + train) | Re-read [Pick a skill](#pick-a-skill) and follow the arrows; multi-stage tasks usually start with `ncore` or `physical-ai-datasets`, then hand off to `nre`. |
| Sibling skill not found on disk | The host repo only has the index | Follow [`references/discovery.md`](references/discovery.md). |
| Stale link to `ncore-data-conversion` | Older snapshots used that name; the skill is now `ncore` | Update the link to `ncore`. |
| User wants to delete disk artifacts | NuRec workflow caches grow large | Walk [`references/teardown.md`](references/teardown.md) in the documented order. |
| User asks "should I retrain or just clean up frames?" | Conflating reconstruction vs post-processing | Retrain → `nre`; clean already-rendered frames → `nurec-fixer`. |

## References

Detailed material that this index intentionally keeps out of the
hot path. Read only when the matching section above points there.

- [`references/workflows.md`](references/workflows.md) — full
  step-by-step multi-skill workflows A–G.
- [`references/teardown.md`](references/teardown.md) — disk
  cleanup order across all five siblings plus secrets-handling
  policy.
- [`references/discovery.md`](references/discovery.md) — how to
  locate or fetch a sibling skill that is not already on disk;
  thin-local-skill policy and upstream links.

## Keeping this index up to date

This index is **hand-curated** — it groups skills by what users
want to do, not alphabetically. There is no generator script; edit
this `SKILL.md` by hand whenever the sibling set changes.

When a new sibling skill is added or a use case shifts:

1. Add a row to [Pick a skill](#pick-a-skill) for the new use
   case.
2. Add a row to
   [The skills in this folder](#the-skills-in-this-folder).
3. If the new skill changes a multi-step pipeline, update
   [`references/workflows.md`](references/workflows.md).
4. Confirm the sibling's `metadata.upstream` field still points at
   the canonical upstream repo or container.

Otherwise the index will quietly drift and beginners will end up
reading the wrong skill.
