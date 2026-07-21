# MCRO baseline checkpoints and front-wide evaluation protocol

**Frozen:** 2026-07-21 (Task 2).  This document is the sole comparison
contract for the multi-camera KPI diagnosis.  All paths below are on
`inceptio`; checkpoints and source data are read-only inputs.

## Checkpoint inventory

| role | run directory | checkpoint MD5 / size | camera count | resolved temporal window | front-wide held-out KPI |
| --- | --- | --- | ---: | --- | --- |
| 1-camera baseline | `/home/inceptio/work/output/pin_cam_visual_fullfix_frontwide_20s_30k/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1707_103927` | `dc1271e91edfb7670929e06f003a93a0` / 1,069,875,467 B | 1 | 0.0–20.0 s | CC-PSNR-masked 24.4050 dB; road PSNR 29.5483; road LPIPS 0.22964 |
| 6-camera directional comparison | `/home/inceptio/work/output/pin_cam_fullfix_6cam_directional_20s_30k/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1907_122746` | `b56a6e37e8c5256e0e968c5913d12671` / 1,072,495,883 B | 6 | 0.0–19.999081 s | CC-PSNR-masked 22.7074 dB; road PSNR 27.1640; road LPIPS 0.26451 |
| 9-camera reference only | `/home/inceptio/work/output/pin_cam_fullfix_9cam_20s_30k_10d0391/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1807_162507` | `f688a271e100868dc0506cc5e03e9c5c` / 1,073,875,339 B | 9 | 0.0–19.999081 s | CC-PSNR-masked 22.4131 dB; road PSNR 26.5090; road LPIPS 0.30035 |

The KPI column is always the `per_camera.camera_front_wide_120fov`
entry from that run's `metrics.json`; its held-out count is 24 frames.
It is not a top-level mean.

## Resolved training contract

All three runs have `seed_initialization=42`, `n_iterations=30000`,
`num_workers=10`, four layers (`background`, `road`, `dynamic_rigids`,
`sky_envmap`), MLP sky, depth supervision disabled
(`trainer.use_lidar_depth=false`, `trainer.use_depth_prior=false`,
`dataset.load_lidar_depth_map=false`, `dataset.load_depth_prior=false`),
and a 0.0-second seek offset.  The camera sets are:

| role | `dataset.camera_ids` | configured train / val duration |
| --- | --- | --- |
| 1-camera | `camera_front_wide_120fov` | 20.0 s / 20.0 s |
| 6-camera | front-wide, cross-left, cross-right, rear-left, rear-right, back-rear-wide | -1 / -1 |
| 9-camera | front-wide, cross-left, cross-right, left-wide, right-wide, back-rear-wide, rear-left, front-standard, front-tele | -1 / -1 |

`-1` means the entire sequence.  Both manifests advertise the same
sequence interval `[0, 19,999,081)` microseconds, so it resolves to the
same 20-second clip and is not a temporal-window confound here.

The 9-camera run is reference-only: it enables
`model.bg_road_slab_exclude` plus projection/footprint exclusion, whereas
the 1- and 6-camera checkpoints have that mechanism disabled.  Do not use
it as a single-variable capacity or camera-count comparison.

## Manifest and frame-set validation

Compared manifests:

- 1-camera / 9-camera: `/home/inceptio/work/data/inc_b6a9ed61_20s/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9.json`
- 6-camera: `/home/inceptio/work/data/inc_b6a9ed61_20s_6cam_directional/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9.json`

For `camera_front_wide_120fov`, the full timestamp tuple is identical:

| set | 1-camera | 6-camera | equality |
| --- | ---: | ---: | --- |
| all frames | 186 | 186 | true |
| train frames (every eighth frame held out) | 162 | 162 | true |
| held-out frames | 24 | 24 | true |
| all-frame SHA-256 | `4fb71a578b63eea56f424f883cebd3cb3f7eaa07cfae456fb10573f01cfea718` | same | true |

The common all-frame range is 60,658–19,960,640 microseconds.  This
validates the front-wide frame population; future A/B renders must use the
frozen frames below rather than whichever split an ad-hoc command chooses.

## Frozen diagnostic frames and crops

Frame coordinates are full-resolution front-wide pixels (`1920×1080`) in
`[x0, y0, x1, y1]` order.  Train and held-out groups remain separate in all
reports.  The selected samples span the clip and include a road-text frame,
a foliage detail frame, and a parked-vehicle frame.

| split | timestamp (us) | purpose | fixed crop(s) |
| --- | ---: | --- | --- |
| train | 160,660 | early training view | context only |
| train | 10,060,651 | mid training view | context only |
| train | 19,960,640 | late training view | context only |
| held-out | 60,658 | early road-text view | `road_text=[850,650,1550,1060]` |
| held-out | 10,360,657 | mid foliage view | `foliage_right=[1360,170,1880,630]` |
| held-out | 19,860,638 | late parked-vehicle view | `parked_vehicles_right=[1390,660,1880,940]` |

The crop definitions are tied to the listed timestamps, not blindly applied
to unrelated frames.  Task 4 must render these same frames for each arm and
report the associated crop metric on its matching frame.

## Evaluation rules

1. Use `render.py` output and read only
   `metrics.json.per_camera.camera_front_wide_120fov` for the main
   1-camera versus 6-camera conclusion.  Top-level `mean_*` fields aggregate
   different camera populations and are prohibited for this comparison.
2. Keep train-view, held-out, and lateral/yaw novel-view results in separate
   tables.  A train-view improvement alone is not geometry evidence.
3. Record CC-masked PSNR, masked PSNR/SSIM/LPIPS, road-crop PSNR/LPIPS,
   sharpness, radial buckets, and the fixed-crop metrics.  Preserve the
   reference JSON alongside each report.
4. The 9-camera checkpoint may be displayed as historical context only; do
   not attribute its difference to camera count because the road-exclusion
   mechanism differs.
