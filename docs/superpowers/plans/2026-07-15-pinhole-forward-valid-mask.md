# OpenCV Pinhole Forward-Valid Supervision Mask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development or executing-plans task-by-task. Follow RED→GREEN TDD.

**Goal:** Add an opt-in NCore dataset mask that removes OpenCV rational pixels whose finite inverse rays are rejected by the same camera model's forward projection, while preserving default behavior and making FTheta/PAI a strict no-op.

**Architecture:** Add a pure helper that converts `camera_rays_to_pixels(...).valid_flag` into an image mask and merges it into the existing static camera valid mask after non-finite-ray repair. Expose `dataset.mask_forward_invalid_pixels` with default `false`, pass it through train/val/test dataset factories, and add an inceptio probe that reports per-camera coverage and confirms FTheta no-op. Training and evaluation already consume the shared per-frame `valid` mask, so no trainer/render loss fork is introduced.

**Tech Stack:** Python 3.11+, NumPy, PyTorch, NCore SDK, pytest, Hydra/OmegaConf, inceptio RTX 4090 environment for real-manifest probes.

## Global Constraints

- Work only on branch `fix/pinhole-forward-valid-mask` in `.claude/worktrees/pinhole-forward-valid-mask`; do not edit or merge main.
- Default `dataset.mask_forward_invalid_pixels=false` must preserve existing behavior.
- When enabled, apply only to `ncore.sensors.OpenCVPinholeCameraModel`; FTheta and OpenCVFisheye must be byte-equivalent/no-op.
- Reuse the already precomputed, repaired camera rays. Do not recompute inverse rays solely for the mask.
- Compute forward validity after `repair_nonfinite_rays()` and before static mask copies are expanded per frame.
- Merge by logical AND with the existing ego/static valid mask. Never overwrite ego masking.
- Train, val, and `make_test()` must receive the same config flag.
- Log per-camera kept/removed pixel counts and percentage only when the flag is enabled.
- Do not modify intrinsic coefficients, 3DGUT kernels, renderer projection formulas, photometric loss definitions, or the Viser overlay projector.
- Do not claim visual improvement from code/probe alone. PIN-MASK-1 ends at code + real coverage validation; PIN-AB-1 owns GPU quality evidence.
- Preserve `repair_nonfinite_rays()` behavior and tests.
- Update `docs/pinhole_camera_kanban.md`: PIN-MASK-1 moves to Review after code/probe gates, not Done; PIN-AB-1 remains Ready until review approval.

---

## Task 1: Pure Forward-Validity Mask Helper

**Files:**
- Modify: `threedgrut/datasets/utils.py`
- Create: `threedgrut/tests/test_forward_valid_camera_mask.py`

**Produces:**

```python
def compute_forward_valid_pixel_mask(camera_model, rays: np.ndarray) -> np.ndarray:
    """Return bool mask shaped rays.shape[:-1] from camera_rays_to_pixels().valid_flag."""
```

- [ ] Write RED tests with a fake model returning a `valid_flag` tensor/array: preserves H×W shape, bool dtype, exact values.
- [ ] Add RED tests for flat N×3 rays and invalid output length/shape.
- [ ] Run focused test and verify expected failure because helper is absent.
- [ ] Implement minimal helper: flatten rays, call `camera_rays_to_pixels`, normalize tensor/NumPy valid flag, validate element count, reshape to `rays.shape[:-1]`.
- [ ] Run focused test GREEN and commit:

```bash
git commit -m "test(ncore): add forward-valid camera mask helper"
```

## Task 2: Dataset Opt-In Wiring and Model Guard

**Files:**
- Modify: `threedgrut/datasets/datasetNcore.py`
- Modify: `threedgrut/tests/test_forward_valid_camera_mask.py`

**Interface:** new constructor kwarg:

```python
mask_forward_invalid_pixels: bool = False
```

- [ ] Write RED tests around a small extracted/patchable application helper or constructor seam proving:
  - disabled flag leaves mask unchanged and never calls forward projection;
  - enabled OpenCVPinhole mask ANDs with existing ego mask;
  - existing false ego pixels remain false;
  - FTheta/OpenCVFisheye path is no-op;
  - non-finite repaired pixels stay invalid.
- [ ] Verify RED.
- [ ] Store constructor flag on `self`.
- [ ] After both full and subsampled rays are generated/repaired, for OpenCVPinhole only:
  - compute full forward-valid mask from repaired full rays;
  - `camera_valid_pixels_ego_mask &= forward_valid_mask`;
  - compute/log kept and removed coverage;
  - do not alter cached rays;
  - subsampled forward validity does not need a separate persistent mask because validation RGB validity derives from the full static mask and existing resize/subsample path; verify actual val code before implementation and adjust only if the current indexing requires it.
- [ ] Run RED tests GREEN plus `test_repair_nonfinite_rays.py`.
- [ ] Commit:

```bash
git commit -m "feat(ncore): mask forward-invalid rational pixels"
```

## Task 3: Train/Val/Test Configuration Parity

**Files:**
- Modify: `threedgrut/datasets/__init__.py`
- Modify: `configs/dataset/ncore.yaml`
- Create or modify: focused dataset-factory config test under `threedgrut/tests/`.

- [ ] Add RED test that monkeypatches `NCoreDataset` and asserts all three construction paths receive the exact flag: train, val, `make_test`.
- [ ] Verify RED.
- [ ] Add to `configs/dataset/ncore.yaml`:

```yaml
mask_forward_invalid_pixels: false
```

with comments explaining opt-in OpenCV rational trust-domain masking and FTheta no-op.
- [ ] Pass `config.dataset.get("mask_forward_invalid_pixels", False)` in train, val, and test constructors.
- [ ] Run factory tests GREEN and compose/config smoke showing default false and CLI override true.
- [ ] Commit:

```bash
git commit -m "config(ncore): expose forward-valid pixel masking"
```

## Task 4: Real b6a9 and PAI Coverage Probe

**Files:**
- Create: `scripts/probe_forward_valid_camera_masks.py`
- Create: `docs/T8_artifacts/pinhole_forward_valid_mask_validation.md`

**CLI:** manifest, camera ids, optional downsample. Output model type, total/kept/removed pixels and kept percentage.

- [ ] Implement probe using the production helper and NCore camera models.
- [ ] Push branch to inceptio and create dedicated worktree; copy submodules if needed.
- [ ] Run on b6a9 9-camera set. Expected qualitative gate:
  - standard ≈100%; tele ≈100%;
  - wide/cross/side/rear rational cameras ≈63–65% kept;
  - no non-finite output.
- [ ] Run on PAI 9ae FTheta camera set with the feature application path. Expected: model guard no-op, kept 100%, no mask change.
- [ ] Record exact output, commands, commit, and manifest paths in validation doc.
- [ ] Commit:

```bash
git commit -m "test(ncore): validate forward-valid mask coverage"
```

## Task 5: Regression, Review State, and Handoff to A/B

**Files:**
- Modify: `docs/pinhole_camera_kanban.md`
- Modify: `docs/inceptio_opencv_rational_peripheral_blur_analysis_2026-07-14.md` only if implementation details changed.

- [ ] Run focused tests for new helper/wiring/config plus `test_repair_nonfinite_rays.py`.
- [ ] Run full Mac suite `pytest threedgrut/tests/ -q`.
- [ ] Run `python -m py_compile` on modified/new Python files and `git diff --check`.
- [ ] Review default-off diff to ensure no existing configuration silently enables the flag.
- [ ] Update Kanban PIN-MASK-1 to Review with branch, commits, test count, b6a9 coverage and PAI no-op evidence. Keep PIN-AB-1 Ready.
- [ ] Commit docs:

```bash
git commit -m "docs(pinhole): move forward-valid mask to review"
```

- [ ] Return branch/commits, RED evidence, full tests, real coverage, PAI result, changed files, and confirmation not merged.

---

## PIN-AB-1 Execution Handoff

After independent review approves PIN-MASK-1, run two 5-second experiments on inceptio from the same reviewed commit/worktree:

- Arm A: `dataset.mask_forward_invalid_pixels=false`
- Arm B: `dataset.mask_forward_invalid_pixels=true`

Fixed variables:

- config `apps/ncore_3dgut_mcmc_multilayer_inceptio`;
- the standard 9 cameras;
- `++loss.camera_loss_weights.camera_front_tele_30fov=2.0`;
- depth-off and `num_workers=10`;
- same seed, iterations, seek offset, 5-second train/val windows;
- no overlay changes.

Required report:

- fixed radial bins `r<0.5`, `0.5–0.7`, `0.7–0.9`, `r≥0.9` using identical geometric pixels;
- forward-valid-only metrics;
- center `r<0.7` guard;
- standard/tele no-op guard;
- wide/cross/side camera visual samples;
- full-image metrics labeled as different mask denominators and not compared naively;
- logs, output directories, metrics JSON and commit hash.

PIN-AB-1 completion decides whether to promote the flag, revise the valid ROI strategy, or reopen projection/UT analysis.
