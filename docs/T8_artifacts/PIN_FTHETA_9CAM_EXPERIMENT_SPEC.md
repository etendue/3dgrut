# PIN-FTHETA 9-Camera Experiment Specification

> **Scope revision (user-approved, 2026-07-17):** The nine-camera survey remains frozen historical evidence. The matched training experiment now uses the seven passing cameras in both arms, excluding `camera_front_standard_55fov` and `camera_front_tele_30fov`; the tele-camera loss weight is removed. The filename is retained for traceability.
>
> **v4 contract (user-approved, 2026-07-18):**
> [`2026-07-18-ftheta-v4-full-domain-retrain-fix.md`](../superpowers/plans/2026-07-18-ftheta-v4-full-domain-retrain-fix.md)
> supersedes the July 17 hard-STOP calibration policy and old GPU execution
> results. Residual thresholds are quality warnings for the selected seven
> cameras; runtime-domain, monotonicity, coverage, provenance, data completeness,
> and matched native-render invariants remain hard gates.
>
> Results made with `pin-ftheta-numpy-v3-physical-domain`, front-wide
> `max_angle=41.84°` (`0.730310... rad`), or artifact SHA-256 prefix
> `73965c6d...` are invalid for the v4 decision. Preserve their historical
> paths/manifests/checkpoints; do not rewrite or metadata-swap them.
> Their exact read-only recovered paths, hashes, steps, and native inventories
> are frozen in
> [`PIN_FTHETA_V3_INVALIDATED_EVIDENCE.md`](PIN_FTHETA_V3_INVALIDATED_EVIDENCE.md).

## Frozen Inputs

- Clip/manifest: `inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9`; manifest SHA-256 `df2021203cfe318cfa8da3462e38c5b7fbf6bf3963d3a8149d145f98f6036e31`.
- Base config: `configs/apps/ncore_3dgut_mcmc_multilayer_inceptio_7cam.yaml`. Final source/config/artifact hashes must be frozen from the clean v4 implementation commit before GPU launch.
- Native resolution: `1920×1080` for every camera. No image-domain resampling is allowed in the primary A/B.
- Camera order: `camera_front_wide_120fov`, `camera_cross_left_120fov`, `camera_cross_right_120fov`, `camera_left_wide_90fov`, `camera_right_wide_90fov`, `camera_back_rear_wide_90fov`, `camera_rear_left_70fov`.
- Initialization seed: `seed_initialization=42` (the frozen `configs/base_gs.yaml` default).
- Smoke window: train and validation `seek_offset_sec=0`, `duration_sec=5.0`, 5,000 iterations. It is mechanism evidence only.
- Full run: the complete 20-second clip (`duration_sec=-1`), 30,000 iterations, with the default empty `camera_loss_weights`; no tele-camera weight is present because tele is excluded.
- GPU environment after hard gates pass: `inceptio_2` RTX 4090, `CUDA_VISIBLE_DEVICES=0`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, Python `/home/inceptio/miniforge3/envs/3dgrut2/bin/python`, depth and depth-prior loading/supervision off, `num_workers=10`, `trainer.sky_backend=mlp`.
- Canonical data target: `/home/inceptio/work/data/inc_b6a9ed61_20s/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9/`. It is currently incomplete on `inceptio_2`; all 14 raw stores and seven-camera aux coverage must be verified first.

The implementation commit is intentionally not invented while the worktree is uncommitted. It must be recorded with the resolved configs before any GPU launch.

## Matched Arms

- Arm P: native OpenCV rational rays and 3DGUT OpenCV projection, with the approved forward-valid supervision mask.
- Arm F: rays and 3DGUT projection both use the frozen per-camera eight-field FTheta artifact.
- Before any GPU launch, both resolved configs **must** freeze `dataset.mask_forward_invalid_pixels=true`, `dataset.opencv_pinhole_use_validity_domain=false`, and `dataset.camera_max_fov_deg=190.0`. This is a future hard requirement for the Phase 2 implementation and resolved-config preflight; the current worktree/configs are not evidence that it is implemented. Arm P uses the approved OpenCV forward-valid mask. Arm F must use its own `theta < max_angle` supervision mask and never call the OpenCV `icD` validity path.
- Camera IDs, frames, timestamps, poses, resolution, split windows, seed, losses, layer configuration, depth policy, iteration count, machine, and evaluation are identical. Camera-model representation is the only variable.
- A viewer-side projector swap or a pinhole-trained checkpoint rendered as FTheta is not Arm F.

## Predeclared Regions and Decision Contract

- Spatial radius is measured from each calibrated principal point and normalized by the image half-diagonal.
- Center: `r<0.5`. Periphery: `r>=0.9`. The middle rings remain available for the existing PIN-AB radial-bin analysis.
- Calibration evaluation samples every native-resolution integer pixel at all azimuths. Later folded/non-invertible rational branches are invalid and are never replaced by ideal-pinhole rays.
- Hard gates: finite exact eight-field parameters; preserved resolution/principal point; positive dense forward/inverse derivatives; first-branch/Jacobian protection; accepted v4 coverage; immutable provenance; no runtime camera-model/Pinhole/ideal-Pinhole/stale-artifact fallback; FTheta supervision only where `theta < max_angle`; complete data and matched P/F evidence.
- A deterministic numerical fallback inside the fitter is allowed only when it is recorded in provenance and passes the same structural, branch, dense-monotonicity, coverage, and warning-reporting checks as the primary numerical path. It must never become a runtime camera-model fallback.
- Warning-only residual limits: `nonradial_floor_mean_deg<0.01`, `forward_poly_max_px<1.5`, angular `mean<0.02°`, `P95<0.04°`, `P99<0.08°`, `max<0.15°`, and `outer_P99<0.10°`. Exceedances remain reported and correlated with final radial KPI but do not alone stop training.

Accepted warnings: front-wide none; cross-left non-radial mean/p95/p99/max/outer p99; cross-right none; left-wide forward max/p95/p99/max/outer p99; right-wide forward max/mean/p95/p99/max/outer p99; back-rear-wide p95/max; rear-left max.

| Camera | FTheta domain | OpenCV calibration domain | Pixels outside FTheta `max_angle` |
|---|---:|---:|---:|
| front-wide | 99.9929% | 100.0000% | 148 |
| cross-left | 99.9933% | 100.0000% | 138 |
| cross-right | 99.9936% | 100.0000% | 133 |
| left-wide | 98.7290% | 98.4722% | 26,355 |
| right-wide | 97.8640% | 97.5162% | 44,292 |
| back-rear-wide | 99.9942% | 100.0000% | 120 |
| rear-left | 99.9951% | 100.0000% | 101 |

## Training and Native-Evaluation Acceptance

- Smoke must show samples from all seven selected cameras, finite rays/loss, no runtime camera-model fallback, all seven metadata entries, both final training and test tables, and complete per-camera metric keys.
- Full Arm P/F runs must each produce a resolved config, code/config/parameter hashes, log, checkpoint, native renders, and `metrics.json`.
- Native comparison uses identical camera/frame/timestamp/pose/resolution tuples and reports per camera before aggregates: center/periphery PSNR, SSIM, LPIPS, gradient correlation, edge sharpness, gap, invalid rays, and invalid projections.
- Viser is permitted only after native parity, using the same fixed frames and active-camera parameter fingerprints.

## Current Gate Outcome

**PROCEED WITH WARNINGS for the selected seven-camera v4 experiment**, subject
to the hard gates above. Front-standard/front-tele remain excluded; this does
not approve nine-camera training. The old roughly 63% coverage was caused by
importing the Pinhole `icD` runtime gate and is invalid. In v4 the legacy metric
name `physical_domain_retention` means calibration-domain comparison, never the
Pinhole runtime domain.

Historical locations remain traceable but are not v4 evidence:
`scripts/pin_ftheta_b6a9_{7cam,9cam}_params.json`,
`~/work/output/pin_ftheta_smoke_runs`, and
`~/work/output/pin_ftheta_7cam_full_ab_runs`. Immutable v4 artifacts and output
roots must use distinct versioned names.
