# C4 Mixed-Camera Viser Fix Validation

Date: 2026-07-14
Implementation branch: `fix/viser-mixed-camera`（merge target: `main`）
Validated implementation range: `0159698..66f668a` plus C4 decision/docs follow-up
Host: `inceptio` / RTX 4090 / renderer `3dgrt`

## Checkpoint

- Experiment: `c4_11cam_tw2p0_30k`
- Checkpoint: `/home/inceptio/work/output/c4_11cam_tw2p0_30k/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1307_170739/ours_30000/ckpt_30000.pt`
- Manifest: `/home/inceptio/work/data/inc_b6a9ed61_20s/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9.json`
- Viewer: `http://10.8.28.130:8090`
- Server log: `/tmp/viser_c4_mixed_camera_fix.log`

## Automated gates

- Mac focused viewer/projector/parity/rig-trajectory suite: `55 passed`（latest focused gate）
- Mac full suite after rig-origin fix: `1008 passed, 2 skipped`
- inceptio focused suite: `39 passed`
- Python compile and `git diff --check`: passed
- Runtime log: no traceback/error after camera switching and playback
- Runtime GPU memory: approximately 7.6 GiB

## Startup contract

Launch used `--initial_cam_id camera_front_fisheye`.

Observed startup state:

```text
active camera='camera_front_fisheye' model=FTheta render=(1920, 1080)
```

The GUI dropdown and Camera status field both reported the same camera/model/resolution. The old metadata-primary overwrite was not reproduced.

## Camera-switch matrix

| Step | GUI/status result | Projection state result |
|---|---|---|
| front fisheye | `camera_front_fisheye`, `FTheta`, `1920×1080` | FTheta image-space overlay; visible radial/wide geometry |
| front wide | `camera_front_wide_120fov`, `OpenCVPinhole`, `1920×1080` | OpenCVPinhole image-space overlay; FTheta state cleared |
| front tele | `camera_front_tele_30fov`, `OpenCVPinhole`, `1920×1080` | narrow camera rays selected; interpolated pose reported |
| rear fisheye | `camera_back_rear_fisheye`, `FTheta`, `1920×1080` | FTheta compositor rebuilt after OpenCV camera |
| cross left | `camera_cross_left_120fov`, `OpenCVPinhole`, `1920×1080` | FTheta state cleared again; side-camera pose selected |
| front wide return | `camera_front_wide_120fov`, `OpenCVPinhole`, `1920×1080` | same state as first front-wide visit; no stale projection |

## Follow Camera

During playback with Follow Camera enabled, the status field reported examples such as:

```text
pose: interpolated 45→46 α=0.662 | nearest Δt=33.8 ms
pose: interpolated 41→42 α=0.836 | nearest Δt=32.8 ms
```

The camera and image-space trajectory remained registered to the road. No nearest-frame 10 Hz step or projection-state jump was observed in the inspected playback interval.

## Visual interpretation

Confirmed viewer fixes:

- front fisheye is no longer displayed as a flattened metadata-primary/pinhole mismatch;
- FTheta and OpenCVPinhole switching does not retain stale fields;
- trajectories remain attached to the roadway through calibrated image-space projection;
- selected-camera pose, frustum source, backdrop projection and status identify the same camera;
- Follow Camera consumes interpolated poses.

Remaining visible defects are checkpoint/model-quality phenomena rather than the previously confirmed viewer-state bugs:

- blocky/noisy near-road splats;
- floaters along road shoulders and vegetation boundaries;
- soft or smeared trees, poles, sky and distant structures;
- foreground road texture fragmentation.

This validation restores the viewer as evidence for camera-selection and projection-state correctness. It does not by itself promote C4 or prove model-quality parity at the image periphery.

### Fisheye display limitation（accepted, no frontend change for now）

The renderer produces the FTheta raster with calibrated polynomial rays, and the Python-side overlays share that image-space projection. However, viser 1.0.29 displays `set_background_image()` as a texture on a plane attached to a Three.js `PerspectiveCamera`; plane scale and orientation come from perspective focal length/film size/quaternion. Viser therefore has no native FTheta camera model and cannot faithfully represent a near-180° polynomial image as an interactive perspective viewport. Peripheral cropping, rotated-plane white corners, or a center-dominant “rectified-looking” presentation can remain even when the underlying FTheta raster is correct.

Decision on 2026-07-14: keep the current implementation. A future exact solution would require a screen-space calibrated 2D view/fullscreen quad or a nonlinear fisheye shader, not further scalar-FOV tuning.

## Native/viewer radial parity status

Implemented and unit-tested:

- `scripts/validate_viser_render_parity.py`
- full/center/peripheral MAE and PSNR;
- radial masks and absolute-difference heatmaps;
- matching `<camera>/<frame>.png` tree comparison.

Not yet produced:

- a legal same-checkpoint, same-camera, same-timestamp native/viewer image pair.

Reason: C4 has native eval renders, but `viser_gui_4d.py` currently has no exact, UI-free frame-dump endpoint. Browser screenshots include GUI layout and browser scaling, so comparing them against native renders would fabricate a meaningless PSNR. The remaining parity task is to add a deterministic viewer-contract PNG dump using the exact `CameraRenderState`, then feed those pairs to the implemented comparator.

Until that dump exists, no center/periphery parity number is claimed.
