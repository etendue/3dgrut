# PIN-FTHETA 9-Camera Experiment Specification

## Frozen Inputs

- Clip/manifest: `inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9`; manifest SHA-256 `df2021203cfe318cfa8da3462e38c5b7fbf6bf3963d3a8149d145f98f6036e31`.
- Baseline config: `configs/apps/ncore_3dgut_mcmc_multilayer_inceptio.yaml`, SHA-256 `21c325c1d88924095b6583295532bc7c226d8c4798e749742d4b4bf2ccde5e59`, baseline commit `73d3deaf8580fdf1507865fa467815b7f0309214`.
- Native resolution: `1920×1080` for every camera. No image-domain resampling is allowed in the primary A/B.
- Camera order: `camera_front_wide_120fov`, `camera_cross_left_120fov`, `camera_cross_right_120fov`, `camera_left_wide_90fov`, `camera_right_wide_90fov`, `camera_back_rear_wide_90fov`, `camera_rear_left_70fov`, `camera_front_standard_55fov`, `camera_front_tele_30fov`.
- Initialization seed: `seed_initialization=42` (the frozen `configs/base_gs.yaml` default).
- Smoke window: train and validation `seek_offset_sec=0`, `duration_sec=5.0`, 5,000 iterations. It is mechanism evidence only.
- Full run: the complete 20-second clip (`duration_sec=-1`), 30,000 iterations, with `camera_front_tele_30fov=2.0` loss weight as required by R6t.
- GPU environment, if the fit gate passes: inceptio RTX 4090, `CUDA_VISIBLE_DEVICES=0`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, conda env `3dgrut2`, depth and depth-prior loading/supervision off, `num_workers=10`, `trainer.sky_backend=mlp`.

The implementation commit is intentionally not invented while the worktree is uncommitted. It must be recorded with the resolved configs before any GPU launch.

## Matched Arms

- Arm P: native OpenCV rational rays and 3DGUT OpenCV projection, with the approved forward-valid supervision mask.
- Arm F: rays and 3DGUT projection both use the frozen per-camera eight-field FTheta artifact.
- Camera IDs, frames, timestamps, poses, resolution, split windows, seed, losses, layer configuration, depth policy, iteration count, machine, and evaluation are identical. Camera-model representation is the only variable.
- A viewer-side projector swap or a pinhole-trained checkpoint rendered as FTheta is not Arm F.

## Predeclared Regions and Fit Gate

- Spatial radius is measured from each calibrated principal point and normalized by the image half-diagonal.
- Center: `r<0.5`. Periphery: `r>=0.9`. The middle rings remain available for the existing PIN-AB radial-bin analysis.
- Calibration evaluation samples every native-resolution integer pixel at all azimuths. Pixels on a non-invertible rational branch are counted as invalid coverage, never replaced by an ideal-pinhole ray.
- Every camera must satisfy strict limits: `nonradial_floor_mean_deg<0.01`, `forward_poly_max_px<1.5`, angular `mean<0.02°`, `P95<0.04°`, `P99<0.08°`, `max<0.15°`, and `outer_P99<0.10°` where the camera reaches 55°.
- Any camera failure yields STOP: no smoke or full training may start until the representation decision is explicitly revised.

## Training and Native-Evaluation Acceptance

- Smoke must show samples from all nine cameras, finite rays/loss, no model fallback, all nine metadata entries, both final training and test tables, and complete per-camera metric keys.
- Full Arm P/F runs must each produce a resolved config, code/config/parameter hashes, log, checkpoint, native renders, and `metrics.json`.
- Native comparison uses identical camera/frame/timestamp/pose/resolution tuples and reports per camera before aggregates: center/periphery PSNR, SSIM, LPIPS, gradient correlation, edge sharpness, gap, invalid rays, and invalid projections.
- Viser is permitted only after native parity, using the same fixed frames and active-camera parameter fingerprints.

## Current Gate Outcome

The corrected 2026-07-17 native-resolution survey is **STOP**: seven cameras pass and two fail. `camera_front_standard_55fov` fails the non-radial and full-image angular gates; `camera_front_tele_30fov` fails `forward_poly_max_px` at `6.5084 px`. The seven wide/rear cameras have expected OpenCV physical-domain coverage of `62.8745%–64.5493%` under the production `0.8<icD<1.2` contract; later low-residual rational roots are explicitly invalid. Phase 1 therefore does not launch GPU training. See `PIN_FTHETA_9CAM_PARAMETER_SURVEY.md` and `scripts/pin_ftheta_b6a9_9cam_params.json`.
