---
name: ncore
description: >-
  Use when converting any sensor dataset into NVIDIA NCore V4 format
  (and feeding it to NuRec or a robotics-to-sim "r2s" pipeline).
  Covers ingesting raw cameras, LiDARs, radars, IMUs, depth or stereo
  into V4 sequences; authoring a new converter from the template;
  adapting PAI / Waymo / PandaSet / NuScenes to V4; handling non-AV
  rigs (mono+depth, mono+lidar, stereo, multi-stereo, RGB-D, COLMAP /
  SfM, ROS2 bag); and diagnosing a broken converter against
  `validate.py`. Do NOT use to train reconstructions (use `nre`) or
  to extract per-object 3D assets (use `asset-harvester`). Trigger
  keywords: ncore, ncore v4, convert, ingest, zarr, itar, nurec,
  waymo, pandaset, nuscenes, pai, hyperion, colmap, scannetpp,
  stereo, multi-stereo, mono+depth, mono+lidar, kitti, sfm, camera,
  lidar, radar, imu, cuboid, poses, intrinsics, ego mask, ros2,
  rosbag, mcap, realsense, zed, rgb-d, r2s, robotics, sam2.
version: "0.1.0"
tools:
  - Shell
  - Read
  - Write
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: NVIDIA NCore
  tags:
    - ncore
    - data-conversion
    - autonomous-vehicles
    - robotics
    - sensors
  upstream: https://github.com/NVIDIA/ncore
  spec_docs: https://nvidia.github.io/ncore/data/conventions.html
  release_tag: "2026.04"
---

# NCore V4 Data Conversion

## Purpose

Convert any sensor recording (cameras, LiDAR, radar, IMU, depth,
stereo, COLMAP/SfM, ROS 2 bag) into a valid NVIDIA **NCore V4** store
so it can be consumed by NuRec / Asset Harvester / `ncore_vis`, or
wired into a robotics-to-sim ("r2s") pipeline. Drive the existing
in-tree converters (PAI, Waymo, COLMAP/ScanNet++) or author a new
converter from `ncore_template/`.

**Use this skill when:** the user has raw sensor data (any rig) that
NuRec or Asset Harvester needs to ingest, or when an existing
converter is failing `validate.py` / producing NuRec data-quality
complaints.

**Do NOT use this skill when:**

- The user is **already on V4** and only wants to train or render
  (use the `nre` skill).
- The user wants per-object 3D assets from sparse views (use
  `asset-harvester`).
- The user only needs to browse / pick an existing NVIDIA dataset
  (use `physical-ai-datasets`).

This skill teaches an agent to take **any** sensor dataset and produce a valid
NCore V4 store that NuRec / Asset Harvester / `ncore_vis` will accept. It covers
both **driving the existing in-tree converters** (PAI, Waymo, COLMAP/ScanNet++)
and **writing a new one** for unsupported formats (PandaSet, NuScenes, KITTI,
stereo, mono+depth, mono+lidar, custom robotics rigs).

## Table of Contents

1. [When to use which path](#when-to-use-which-path)
2. [Install & references](#install--references)
3. [Mental model — the V4 store](#mental-model--the-v4-store)
4. [Path A — drive an existing in-tree converter](#path-a--drive-an-existing-in-tree-converter)
5. [Path B — author a new converter from the template](#path-b--author-a-new-converter-from-the-template)
6. [V4 conventions you must obey](#v4-conventions-you-must-obey)
7. [Format recipes (AV)](#format-recipes-av)
   - [PAI (NVIDIA Physical AI Autonomous Vehicles, HuggingFace)](#pai-nvidia-physical-ai-autonomous-vehicles-huggingface)
   - [Waymo Open](#waymo-open)
   - [PandaSet](#pandaset)
   - [NuScenes](#nuscenes)
8. [Format recipes (non-AV / sensor-only)](#format-recipes-non-av--sensor-only)
   - [Mono camera (no depth, no LiDAR) → COLMAP track](#mono-camera-no-depth-no-lidar--colmap-track)
   - [Stereo cameras](#stereo-cameras)
   - [Multi-stereo rig (surround stereo)](#multi-stereo-rig-surround-stereo)
   - [Mono + depth (RGB-D / learned depth)](#mono--depth-rgb-d--learned-depth)
   - [Mono + LiDAR (handheld / robot)](#mono--lidar-handheld--robot)
   - [Solid-state / non-repetitive LiDAR (Livox)](#solid-state--non-repetitive-lidar-livox)
   - [IMU + camera (visual-inertial)](#imu--camera-visual-inertial)
   - [ROS2 bag (MCAP / SQLite3)](#ros2-bag-mcap--sqlite3)
   - [Aerial / drone](#aerial--drone)
9. [Robotics pipeline shards (r2s)](#robotics-pipeline-shards-r2s)
10. [Validation & end-to-end NuRec](#validation--end-to-end-nurec)
11. [Common failure modes (and the fix file)](#common-failure-modes-and-the-fix-file)
12. [Additional resources](#additional-resources)

---

## When to use which path

| You have… | Use |
|-----------|-----|
| PAI clip on HuggingFace, or a local PAI clip directory | **Path A** — `tools/data_converter/pai:convert` (`pai-stream-v4` / `pai-v4`) |
| Waymo `.tfrecord` files | **Path A** — `tools/data_converter/waymo:convert` (`waymo-v4`) |
| COLMAP scene (or ScanNet++ DSLR) | **Path A** — `tools/data_converter/colmap:convert` (`colmap-v4` / `scannetpp-v4`) |
| Mono RGB images, no poses | **Path A.5** — run COLMAP first, then `colmap-v4` |
| PandaSet, NuScenes, KITTI, Argoverse, custom AV rig | **Path B** — author from `ncore_template/impl/data_converter/example_converter.py` |
| Stereo / mono+depth / mono+lidar / robotics | **Path B** |
| Already have parsed numpy/torch arrays in memory | **Path B** but skip Bazel — use `ncore.data.v4` API directly |

If a candidate path exists in upstream, **prefer it**. Hand-rolling a Waymo or
PAI converter on top of the template is wasted work and almost certainly wrong
(rolling-shutter timing, FTheta intrinsics, Waymo camera-frame rotation, etc).

---

## Prerequisites

- Linux host with Python ≥ 3.10 and `pip`.
- `git` for the upstream converter sources.
- `bazel` only if you run the in-tree converters (Path A); pure-Python
  in-process writes (`ncore.data.v4`) need no Bazel.
- Disk: budget tens of GB per converted clip; pre-zarr scratch can be
  larger than the final `.zarr.itar`.
- HuggingFace token (`HF_TOKEN`) only if pulling a gated PAI clip.

### Verifying secrets safely

**Always verify prerequisites with the upstream `validate.py` or by
running the converter against a tiny test slice; never write ad-hoc
bash that interpolates `HF_TOKEN` values.** The common one-liner

```bash
# BAD — leaks the secret to the terminal when the variable is set
echo "HF_TOKEN: ${HF_TOKEN:+yes}${HF_TOKEN:-no}"
```

prints `yes<token-value>` whenever `HF_TOKEN` is set, because
`${VAR:-no}` only falls back to "no" when the variable is empty. Use
a length-only check, which never echoes the value:

```bash
# OK — prints "set (N chars)" or "missing", never the value
test -n "$HF_TOKEN" && echo "HF_TOKEN: set (${#HF_TOKEN} chars)" || echo "HF_TOKEN: missing"
```

Rotate any token you suspect was echoed at
<https://huggingface.co/settings/tokens>.

## Install & references

```bash
pip install nvidia-ncore        # pure-Python API for in-process writes
git clone --depth 1 https://github.com/NVIDIA/ncore.git   # for upstream converters (Bazel)
```

- Source + upstream converters: <https://github.com/NVIDIA/ncore>
- V4 spec / conventions: <https://nvidia.github.io/ncore/data/conventions.html>
- API reference: <https://nvidia.github.io/ncore/apis/data.v4.html>
- Conversion guides: <https://nvidia.github.io/ncore/conversions/index.html>
- Sensor models (camera, LiDAR, windshield): <https://nvidia.github.io/ncore/data/sensor_models.html>
- The template scaffold (every method documented inline):
  [`ncore_template/impl/data_converter/example_converter.py`](ncore_template/impl/data_converter/example_converter.py)
- End-to-end NCore → NRE training and rendering: see the sibling
  [`nre`](../nre/SKILL.md) skill (Workflow A). NVIDIA's reference
  OSMO recipe that wires PAI → NCore → NuRec → USDZ training lives
  in the upstream NCore repo at
  <https://github.com/NVIDIA/ncore/tree/main/tools/data_converter/pai>
  and the NRE container docs at
  <https://www.nvidia.com/en-us/omniverse/nurec/>.

---

## Mental model — the V4 store

Every V4 sequence is one **store** (`itar` archive or plain directory) holding
**component groups**. The required components for NuRec are:

| Component | What it carries | API class |
|-----------|-----------------|-----------|
| `Poses` | Dynamic `T_rig_world` (per timestamp); static `T_sensor_rig` per camera/lidar/radar; static `T_world_world_global` | `PosesComponent` |
| `Intrinsics` | Per-camera model (Pinhole / Fisheye / FTheta) + per-LiDAR spinning model | `IntrinsicsComponent` |
| `CameraSensor` | Encoded image bytes + per-frame `[exposure_start, exposure_end]` µs | `CameraSensorComponent` |
| `LidarSensor` | Per-ray unit direction + per-ray µs timestamp + distance(s) + intensity + `model_element=(row,col)` | `LidarSensorComponent` |
| `RadarSensor` *(optional)* | Same shape as LiDAR minus intensity / model_element | `RadarSensorComponent` |
| `Cuboids` *(optional, recommended)* | `CuboidTrackObservation` list referencing `rig` / `world` / sensor frames | `CuboidsComponent` |
| `Masks` | Per-camera dict of `{name: PIL.Image}` (NuRec **requires** ego masks) | `MasksComponent` |
| `PointClouds` *(optional)* | Pre-computed dense or SfM points (e.g. COLMAP `sfm_points`, depth-derived) | `PointCloudsComponent` |

Frames of reference (these are non-negotiable — wrong frames = silent NuRec failure):

- **Rig**: `+X` forward, `+Y` left, `+Z` up. Origin at the middle of the rear axle on nominal ground for AV, or any natural body-fixed point for non-AV. All extrinsics are `T_sensor → rig`.
- **Camera sensor**: `+X` right, `+Y` down, `+Z` forward (optical axis). NCore's convention. Waymo (X-fwd) and similar must be **rotated** before storing extrinsics.
- **LiDAR model frame**: azimuth 0° = `+X`, 90° = `+Y`, `+Z` up. Independent of the raw sensor's native azimuth — you choose how `column_azimuths_rad` maps physical columns.
- **World**: sequence-local. Re-reference all `T_rig_world` to the **first** ego pose so origin is near the vehicle start (raw UTM/ECEF at 10+ km loses precision in float32). Carry the original first pose into `T_world_world_global` (float64) if you need a global anchor.
- **Image pixels**: `u` right, `v` down, origin at the top-left **corner** of the top-left pixel (so pixel centers are at `0.5, 0.5`).
- **Units**: timestamps in µs everywhere, distances in metres, angles in radians.

The single source of truth is the spec — when in doubt, open it:
<https://nvidia.github.io/ncore/data/conventions.html>.

### Sequence-level metadata

Some downstream pipelines need metadata beyond the per-component data — carry
it on the sequence's `generic_meta_data` (passed to
`SequenceComponentGroupsWriter(...)`):

- **Stereo pairs** — required by stereo-depth modules (Foundation Stereo) and
  multi-camera training configs to discover left/right pairings:

  ```json
  {"stereo_pairs": [{"left": "camera_front_left", "right": "camera_front_right"}]}
  ```

  Multiple pairs are allowed for surround-stereo rigs.

- **Source tag** — distinguishes real from synthetic data. Renderer-output
  shards (NuRec sim) set `{"source": "simulation", "model_checkpoint": "..."}`;
  real-sensor shards omit the key or set `{"source": "real"}`. Downstream
  validators key off this to skip "expected vs measured" checks on sim data.

- **Calibration / egomotion provenance** — the example template writes
  `calibration_type` and `egomotion_type` on the `PosesComponent`'s
  `generic_meta_data`. Use this to track the upstream tool (e.g.
  `egomotion_type: "cuvslam-stereo"` or `"kiss-icp"` or `"vio:orbslam3"`) so
  later modules can pick refinement strategies that match the input quality.

---

## Instructions

Pick one of the two paths below.

- **Path A** if the user's dataset format is already supported in
  `ncore/tools/data_converter/` (PAI, Waymo, COLMAP/ScanNet++) —
  bootstrap the upstream repo and drive the existing binary.
- **Path B** if the format is unsupported (PandaSet, NuScenes, KITTI,
  custom rig, robotics bag, …) — copy `ncore_template/` next to the
  dataset and fill in the four hand-written hooks. The V4 conventions
  in the **Mental model** section above are mandatory; the recipes
  further down show typical configurations per rig.

After conversion, always run the **Validation & end-to-end NuRec**
section below before handing the store to NRE.

## Path A — drive an existing in-tree converter

The upstream `tools/data_converter/<format>` modules are Bazel targets. Build
once, run per dataset.

### Bootstrap

```bash
git clone --depth 1 https://github.com/NVIDIA/ncore.git
cd ncore
bazel build //tools/data_converter/pai:convert     # or waymo, or colmap
```

Each `convert` binary takes shared **base** flags (`--root-dir`, `--output-dir`,
`--no-cameras`, `--camera-id`, `--no-lidars`, `--lidar-id`, `--verbose`) followed
by a **subcommand** (`pai-v4`, `pai-stream-v4`, `waymo-v4`, `colmap-v4`,
`scannetpp-v4`) with format-specific flags.

### Standard sub-flags worth knowing

| Flag | Default | Meaning |
|------|---------|---------|
| `--store-type {itar,directory}` | `itar` | `itar` is fastest for NuRec; `directory` is debuggable |
| `--profile {default,separate-sensors,separate-all}` | varies | NuRec wants `separate-sensors` |
| `--sequence-meta` / `--no-sequence-meta` | enabled | Writes `<sequence>.json` next to the store — NuRec/`ncore_vis` need it |
| `--world-global-mode {none,identity,localized}` | varies | For NuRec releases that require the `world→world_global` edge, use `identity` (or `localized` to keep a real global anchor) |

### When to script vs run interactively

For a **one-off conversion**, the bare `bazel run` form in each format recipe
below is enough. For repeatable cluster runs, wrap the same two steps
(`bazel build` then `bazel run`) inside an OSMO / Slurm / Kubernetes task that
clones NCore at a pinned ref, runs the convert step, and chains the result
into the [`nre`](../nre/SKILL.md) training and aux-data containers (see
`nre`'s Workflow A). The upstream
[`NVIDIA/ncore`](https://github.com/NVIDIA/ncore) repo ships reference
converter targets that you can pin by Git commit for reproducibility.

---

## Path B — author a new converter from the template

The scaffold is intentionally minimal but writes **every** required component
type with placeholder data. Treat it as a checklist: every `# FILL IN` is a
correctness gate — none can be skipped.

### Scaffold

```bash
# Copy the scaffold next to your dataset (run from this skill's folder)
cp -r ncore_template /path/to/ncore-myformat
```

Then rename the package and class (`ExampleConverter` → `MyFormatConverter`) and
implement the contract:

| Method | Contract |
|--------|----------|
| `get_sequence_ids(config) -> list[str]` | Discover sequence IDs from `config.root_dir` (or wherever your dataset lives — manifest CSV, HF clip index, ROS bag glob) |
| `from_config(config) -> Converter` | One-time setup (load calibration, open dataset index, init shared interpolators). Heavy lifting that all sequences share goes here |
| `convert_sequence(sequence_id) -> None` | Per-sequence work: open `SequenceComponentGroupsWriter`, register component writers, write data, `finalize()`, write `<sequence_id>.json` |

Inside `convert_sequence` the canonical order is **Poses → Intrinsics → Masks →
Camera → LiDAR → Radar → Cuboids → finalize**. This order is not required by the
writer but it surfaces calibration / pose / timing bugs **before** you've spent
minutes encoding image bytes.

The skeleton walks each step explicitly and lists every silent-correctness
trap inline — read these before filling them in:

- Spinning-LiDAR pitfalls (`spinning_direction`, non-uniform `row_elevations_rad`,
  `column_azimuths_rad` ordering, Ouster `row_azimuth_offsets_rad`):
  [`example_converter.py:61-120`](ncore_template/impl/data_converter/example_converter.py#L61-L120)
- Pose trajectory density + float64 → float32 + re-referencing rules:
  [`example_converter.py:336-492`](ncore_template/impl/data_converter/example_converter.py#L336-L492)
- Camera intrinsics for Pinhole / Fisheye / FTheta + shutter type:
  [`example_converter.py:524-578`](ncore_template/impl/data_converter/example_converter.py#L524-L578)
- Per-ray LiDAR timestamps and the three data shapes (range image / sensor-frame
  XYZ / world-frame XYZ requiring decompensation):
  [`example_converter.py:746-829`](ncore_template/impl/data_converter/example_converter.py#L746-L829)
- Cuboid centroid convention (geometric center, not bottom-center):
  [`example_converter.py:876-980`](ncore_template/impl/data_converter/example_converter.py#L876-L980)

### In-process API (no Bazel)

If you already have parsed arrays in Python and don't need a CLI, skip
`FileBasedDataConverter` entirely and call the V4 writer directly:

```python
from ncore.data.v4 import SequenceComponentGroupsWriter, PosesComponent, ...
writer = SequenceComponentGroupsWriter(
    output_dir_path=out / seq_id,
    store_base_name=seq_id,
    sequence_id=seq_id,
    sequence_timestamp_interval_us=interval,
    store_type="itar",
)
poses_writer = writer.register_component_writer(PosesComponent.Writer, ...)
# … write each component …
paths = writer.finalize()
```

The contract (component order, dtype rules, timestamp constraints) is identical.

---

## V4 conventions you must obey

These are the rules that turn into runtime asserts (or worse: silent NuRec
artefacts). Cross-reference the spec before relaxing any of them.

### Time

- **Microseconds, uint64, everywhere.** `np.uint64`, not `np.int64`.
- The sequence interval is half-closed `[start, stop)`. Build it with
  `HalfClosedInterval.from_start_end(start, end_inclusive)` — do **not**
  pre-add 1 to `end`.
- Dynamic poses **must exactly span** the interval: `timestamps[0] == start` and
  `timestamps[-1] == stop - 1`. Sensors with timestamps slightly outside this
  range are clamped at write time.
- Per-sensor frame timestamps are `[exposure_start, exposure_end]` (cameras) or
  `[sweep_start, sweep_end]` (LiDAR/radar). They must lie within the sequence
  interval and the **end** must be unique within that sensor's writer.
- Rolling-shutter cameras: `start = trigger + half_shutter`,
  `end = readout_done - half_shutter`. Global shutter: `start == end` is OK.
- LiDAR `frame_timestamps_us[0]` is the **sweep start** (not midpoint). Per-ray
  `timestamp_us` must lie in `[sweep_start, sweep_end]`. Treating the dataset's
  frame timestamp as the midpoint and subtracting half a sweep duration
  introduces a ~50 ms shift on a 10 Hz LiDAR and produces motion-comp blur.

### Pose graph

- All intermediate pose math is **float64**. Cast to float32 only as the very
  last step before `store_dynamic_pose` / `store_static_pose`.
  Exception: `world → world_global` stays float64 (NuRec's
  `RigTrajectories.T_world_base` is float64).
- Re-reference dynamic poses to the first ego pose:
  `poses_f64 = inv(poses_f64[0]) @ poses_f64`. Without this, GPS/UTM/ENU at
  10+ km loses sub-cm precision once cast to float32.
- The pose trajectory must be **dense** (waypoint spacing < LiDAR sweep
  duration, e.g. <50 ms for 10 Hz). Combine every available source: per-camera
  per-frame poses, per-LiDAR per-sweep poses, IMU/GPS/odometry. Concatenate,
  `np.unique` by timestamp, sort.
- `T_camera_rig` and `T_lidar_rig` are float32 (NuRec's `transform_poses` leaks
  input dtype through an internal matmul; float64 here collides with the
  float32 trajectory and crashes "Get Lidar Point Clouds" with
  `RuntimeError: double != float`).

### LiDAR

- Direction vectors are **unit-norm**, in **sensor coordinates**, at each ray's
  **measurement time** (not the sweep-start time). Three input shapes:
  1. **Polar range image** with beam geometry → derive directly from
     `column_azimuths_rad[col]`, `row_elevations_rad[row]`,
     `row_azimuth_offsets_rad[row]`. Direction:
     `(cos(elev)*cos(azi), cos(elev)*sin(azi), sin(elev))`.
  2. **Sensor-frame XYZ** → `direction = xyz / |xyz|`.
  3. **World-frame or ego-compensated XYZ** → **decompensate** with
     `MotionCompensator.motion_decompensate_points`, then normalise.
- Spinning LiDAR parameter traps (silent — wrong values pass write but fail
  NuRec):
  - `spinning_direction`: nearly all automotive spinning LiDARs (Velodyne,
    Hesai, Ouster, Robosense) are **`"cw"`**. The template defaults to
    `"ccw"` to force the question. Wrong value mirrors Y → Z-flip in NuRec.
  - `row_elevations_rad`: **non-uniform on every common sensor**. Read real
    per-beam angles from the sensor calibration. **Strictly descending**
    (highest beam first); reverse if your source is ascending. NCore asserts
    `np.diff > 0` after projecting through `relative_angle`.
  - `column_azimuths_rad`: must reflect the **actual** per-column azimuth at
    frame start (not a synthetic `linspace`). NCore validates strict ordering
    via `relative_angle(azim[0], azim, spinning_direction)`. If raw
    per-column heading is available, use it; otherwise `linspace(0, ±2π, N,
    endpoint=False)` is acceptable for `"ccw"` / `"cw"` respectively.
  - `row_azimuth_offsets_rad`: zero for most sensors; **non-zero on Ouster**
    (`beam_azimuth_angles` from the HTTP API). Skipping this on Ouster
    misaligns LiDAR-to-camera projection.
- Per-ray `timestamp_us` is **required** for motion compensation; supply real
  per-column firing times (typically column-linear across the sweep).

### Cuboids

- `BBox3.centroid` is the **geometric center**. Many AV datasets use
  bottom-center → add `dim_z / 2` to z. Verify empirically: `mean(z) - mean(h)/2`
  near 0 = geometric center; near `mean(h)/2` above ground = bottom-center.
- `reference_frame_id` ∈ `{"rig", "world", <sensor_id>}`.
  `reference_frame_timestamp_us` and `timestamp_us` must lie within the
  sequence interval.
- **`LabelSource`**: tag the **origin pipeline**, not quality. Third-party
  dataset labels (Waymo, NuScenes, PandaSet) → `EXTERNAL`, even if the upstream
  is human GT. Use `GT_ANNOTATION` only when this converter's team owns the
  annotation.

### Camera

- Image bytes stored verbatim (JPEG/PNG); NCore does not re-encode.
- Use the model that matches your sensor: `OpenCVPinholeCameraModelParameters`
  (most AV cameras), `OpenCVFisheyeCameraModelParameters` (Kannala-Brandt
  fisheye, e.g. ScanNet++ / GoPro), `FThetaCameraModelParameters` (NVIDIA
  Hyperion / equidistant-radial). Resolution must match the actual image bytes
  bit-for-bit.
- Map shutter direction by **enum name**, not integer cast. The five
  `ShutterType` values are `ROLLING_TOP_TO_BOTTOM=1`, `ROLLING_LEFT_TO_RIGHT=2`,
  `ROLLING_BOTTOM_TO_TOP=3`, `ROLLING_RIGHT_TO_LEFT=4`, `GLOBAL=5`. Build a
  source-enum-to-name dict.
- Provide ego masks. NuRec's data-quality guide **requires** binary
  ego-vehicle masks per camera. Without them the hood/roof rack leaks into the
  reconstruction.

### NuRec data-quality minimums

Per the NuRec "Ensure Data Quality" guide, hitting these is the difference
between a clean reconstruction and visible artefacts:

- Camera extrinsics: < 0.5° rotation, < 2 cm translation (relative to rig).
- Camera intrinsics: < 1 px reprojection error.
- Egomotion: < 0.5° / < 2.5 cm consecutive-frame error; trajectory must cover
  every sensor frame's start and end timestamp.
- Cuboids: < 1° / < 5 cm position / < 5 cm dimension; per-track stable
  `track_id`s.
- Original sensor resolution at full FPS, **no undistortion / rectification**.

---

## Format recipes (AV)

### PAI (NVIDIA Physical AI Autonomous Vehicles, HuggingFace)

**Built-in.** Streaming or local. Uses Hyperion 8 / 8.1 sensor IDs.

```bash
# Streaming (no download — recommended for cloud / OSMO)
bazel run //tools/data_converter/pai:convert -- \
    --output-dir <OUT> \
    --camera-id camera_front_wide_120fov \
    --camera-id camera_cross_left_120fov \
    --camera-id camera_cross_right_120fov \
    --camera-id camera_front_tele_30fov \
    pai-stream-v4 \
        --clip-id <CLIP_ID> \
        --hf-token "$HF_TOKEN" \
        --store-type itar \
        --profile separate-sensors \
        --sequence-meta
```

Notes:

- HF dataset: `nvidia/PhysicalAI-Autonomous-Vehicles`. License must be accepted
  on HF before `HF_TOKEN` works.
- 7 cameras (FTheta + Fisheye + Pinhole depending on FOV), 1 top LiDAR
  (`lidar_top_360fov`, spinning, see datasheet for elevations).
- Camera intrinsics include `shutter_delay_us` for per-row rolling-shutter
  timestamping, plus optional `BivariateWindshieldModelParameters` for
  windshield refraction.
- Output path: `<output-dir>/pai_<clip-id>/pai_<clip-id>.ncore4.zarr.itar`.
- Discover clip IDs via the `clip_index.parquet` blob in the HF dataset
  (`nvidia/PhysicalAI-Autonomous-Vehicles`) — the
  [`physical-ai-datasets`](../physical-ai-datasets/SKILL.md) skill has the
  download recipe and toolkit pointers.
- Subset by time with `--seek-sec` / `--duration-sec`; subset sensors with
  `--no-lidars` / `--camera-id` (repeatable).
- **car2sim_6cam** sim configs target CARLA sensor names; PAI exports use real
  Hyperion IDs — when feeding NRE training, override `dataset.camera_ids` /
  `dataset.lidar_ids` on the NRE Hydra command line (see
  [`nre`](../nre/SKILL.md) Workflow A and `references/configuration.md`).

### Waymo Open

**Built-in.** Reads `.tfrecord` segment files.

```bash
bazel run //tools/data_converter/waymo:convert -- \
    --root-dir <DIR_OF_TFRECORDS> \
    --output-dir <OUT> \
    waymo-v4 \
        --store-type itar \
        --profile separate-sensors \
        --world-global-mode localized   # or 'identity' / 'none'
```

Notes:

- 5 cameras (`camera_front_50fov`, `camera_front_left_50fov`,
  `camera_front_right_50fov`, `camera_side_left_50fov`,
  `camera_side_right_50fov`) + 1 top LiDAR (`lidar_top`).
- Camera intrinsics → `OpenCVPinholeCameraModelParameters`. Waymo's local
  camera frame is **`+X` principal axis**; the converter rotates to NCore's
  `+Z` principal axis. If you re-derive extrinsics manually, apply the same
  rotation to `T_camera_rig`.
- Cuboid classes: `unknown, vehicle, pedestrian, sign, cyclist`. Map → NCore
  `class_id` strings; tag `LabelSource.EXTERNAL`.
- Multi-return LiDAR (primary + secondary). Stack into the `[R, N]` distance /
  intensity arrays.

### PandaSet

**No upstream converter — author with Path B.** PandaSet (Hesai +
Scale-AI-labelled) ships JSON metadata with each sequence.

Sensor inventory:

- 6 cameras (`front_camera`, `front_left_camera`, `front_right_camera`,
  `back_camera`, `left_camera`, `right_camera`) — pinhole, global shutter.
  Intrinsics in `meta/intrinsics/<id>.json`.
- 2 LiDARs:
  - `front_lidar`: Hesai PandarGT (mechanical, 60° HFOV, 150° VFOV
    forward-facing). Treat as a partial spinning sensor (`column_azimuths_rad`
    spans the 60° wedge).
  - `top_lidar` (key sensor for AV reconstruction): Hesai Pandar64,
    **`spinning_direction="cw"`**, 64 beams with **non-uniform** elevations
    (Pandar64 datasheet table — copy the 64 angles, sort descending),
    `row_azimuth_offsets_rad = zeros(64)`, 1800 columns.

Data layout per sequence (extract):

```text
<seq_id>/
├── meta/
│   ├── intrinsics/<camera_id>.json     # focal, principal point, distortion
│   ├── timestamps.json                 # per-camera-frame µs
│   └── poses.json                      # per-frame ego pose (UTM)
├── camera/<camera_id>/<frame>.jpg
├── camera/<camera_id>/poses.json       # per-frame camera pose in world
├── lidar/<frame>.pkl.gz                # XYZ + intensity + timestamp + ring
└── cuboids/<frame>.pkl.gz              # bottom-center xyz + dim + yaw
```

Conversion checklist:

- Pose trajectory: union of **per-camera per-frame** poses (6 × ~80 = ~480
  waypoints) and per-LiDAR-frame poses, deduped + sorted, re-referenced to
  the first ego pose.
- LiDAR points are **world-frame XYZ already ego-compensated** → run
  `MotionCompensator.motion_decompensate_points` to recover sensor-frame
  per-ray XYZ, then normalise to direction. Per-ray µs from the `timestamp`
  column.
- Map raw Pandar64 `ring` → NCore `model_element[:, 0]` via a
  `ring_id → row_index` permutation that sorts elevations **descending** (raw
  ring IDs are firing order, not beam index).
- Cuboids: **bottom-center → geometric center** (`centroid_z += dim_z/2`).
  Yaw-only rotation → `rot=(0, 0, yaw_rad)`. `LabelSource.EXTERNAL`.
- Cameras: global shutter (`ShutterType.GLOBAL`). `frame_timestamps_us = [t, t]`.

### NuScenes

**No upstream converter — author with Path B.** NuScenes ships nested JSON
tables (`sample`, `sample_data`, `ego_pose`, `calibrated_sensor`, `sensor`,
`sample_annotation`, `instance`).

Sensor inventory:

- 6 cameras (`CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`,
  `CAM_BACK_LEFT`, `CAM_BACK_RIGHT`) — pinhole. Rolling shutter; consult the
  device datasheet for shutter direction (rename to lowercase NCore IDs).
- 1 LiDAR (`LIDAR_TOP`): Velodyne HDL-32E, **`spinning_direction="cw"`**,
  32 beams, 1800 columns at 0.2°. Use VLP/HDL-32 datasheet elevations
  (descending). `row_azimuth_offsets_rad = zeros(32)`.
- 5 radars (`RADAR_FRONT`, `RADAR_FRONT_LEFT`, `RADAR_FRONT_RIGHT`,
  `RADAR_BACK_LEFT`, `RADAR_BACK_RIGHT`). Continental ARS 408. Optional —
  NuRec ignores radar.

Conversion checklist:

- A NuScenes "scene" → one V4 sequence. Iterate `sample`s in the scene to
  enumerate `sample_data` per sensor.
- Timestamps in `sample_data.timestamp` are **µs** already.
- `ego_pose` is rig→world (translation + quaternion). Build the dense
  trajectory from the union of ego_pose entries across all sample_data
  (cameras at 12 Hz, LiDAR at 20 Hz, radars at 13 Hz → ~50 ms spacing).
- `calibrated_sensor.translation` + `.rotation` is `T_sensor_rig` (named
  `T_calib` in their docs — verify direction by transforming a sensor-frame
  test point and checking it lands in the expected rig position).
- LiDAR `.pcd.bin` files are already-ego-compensated sensor-frame XYZ + ring
  index + intensity. Per-ray timestamps are not stored — synthesise from
  `column_index` (azimuth bin) and the sweep duration (50 ms at 20 Hz):
  `t_ray = sweep_start + (azim_bin / 1800) * sweep_duration`.
- `sample_annotation`: bottom-of-box origin → **add `size[2]/2` to z**.
  `rotation` quaternion → XYZ-Euler radians. `instance_token` → `track_id`,
  `category_name` → `class_id`. `LabelSource.EXTERNAL`.
- Ego masks: NuScenes does not ship them. Generate via SAM2 / off-the-shelf
  ego segmentation per camera — strongly recommended for NuRec quality.

---

## Format recipes (non-AV / sensor-only)

These do not have a vehicle rig; pick a body-fixed origin and apply the same
rig conventions (`+X` forward, `+Y` left, `+Z` up).

### Mono camera (no depth, no LiDAR) → COLMAP track

You only have RGB images. Run COLMAP first to produce poses + sparse points,
then feed COLMAP into the upstream converter:

```bash
colmap automatic_reconstructor \
    --workspace_path <SCENE> --image_path <SCENE>/images \
    --camera_model OPENCV --single_camera 1
bazel run //tools/data_converter/colmap:convert -- \
    --root-dir <SCENE> --output-dir <OUT> \
    colmap-v4 --include-3d-points --start-time-sec 0
```

Notes:

- COLMAP timestamps are synthetic (1 FPS by default; tune with
  `--start-time-sec` and the FPS embedded in your image filenames).
- The COLMAP camera frame already matches NCore (`+Z` optical) — no rotation
  required.
- SfM points become a `PointCloudsComponent` named `sfm_points`. Use this when
  no LiDAR is available so NuRec has a sparse geometric prior.
- Ego masks (`<image_basename>_mask.png` next to images, or `--masks-dir`).

### Stereo cameras

Two synchronised cameras at known baseline. No LiDAR.

- **Rig origin**: midpoint between the two camera optical centres (or the left
  camera — pick one and stay consistent).
- **Static extrinsics**: `T_left_rig`, `T_right_rig` from your stereo calibration
  (OpenCV `stereoCalibrate` outputs `R, T` from right-to-left → invert/compose
  to get rig-relative).
- **Trajectory**: stereo-VIO (ORB-SLAM3, OpenVSLAM) or COLMAP run on left
  images → propagate to rig with `T_left_rig`.
- **No LiDAR component.** Two options:
  1. Skip LiDAR entirely (set `lidar_ids = []`). NuRec falls back to
     image-only reconstruction (lower quality, more views needed).
  2. Compute disparity per stereo pair → 3-D point cloud per frame in the
     left-camera frame → write as a `PointCloudsComponent` per frame
     (analogous to COLMAP `sfm_points` but dense). This gives NuRec a
     geometric prior without faking a spinning LiDAR.
- Ego masks: render or hand-paint a static mask of the rig body if visible in
  the FOV.

### Multi-stereo rig (surround stereo)

Multiple synchronised stereo pairs on one rig (e.g. NVIDIA Hyperion 8.1
surround stereo, AV1, custom inspection robots). Each pair feeds Foundation
Stereo independently to produce dense depth around the platform.

- Encode **each camera as its own** `CameraSensorComponent` with a standard
  `T_camera_rig` extrinsic — do **not** pre-rectify or fuse a pair into a
  single virtual sensor. NCore does not have a "stereo pair" component
  type; pairing is metadata, not structure.
- Declare the pairings on the **sequence-level** `generic_meta_data` so
  downstream tools (Foundation Stereo, NuRec aux-data) can discover them:

  ```json
  {"stereo_pairs": [
      {"left": "camera_front_left",  "right": "camera_front_right"},
      {"left": "camera_rear_left",   "right": "camera_rear_right"},
      {"left": "camera_side_left_a", "right": "camera_side_left_b"}
  ]}
  ```

- Pose trajectory must include every **(camera × frame timestamp)** sample —
  surround stereo at 8 cameras × 100 frames yields 800 trajectory waypoints,
  more than dense enough for per-ray motion compensation. Do not subsample.
- Calibration: stereo intrinsics + extrinsics from `cv2.stereoCalibrate` give
  you `R, T` from right-to-left. Compose with your chosen rig origin to get
  `T_left_rig` and `T_right_rig` separately; do **not** store only the
  baseline.
- Scale is metric by construction (calibrated baseline). Skip scale refinement
  (or run it as a sanity check only).

### Mono + depth (RGB-D / learned depth)

Single RGB camera + per-frame depth (sensor: RealSense, Kinect, ZED depth, or
learned mono-depth like Marigold / DepthAnythingV2 / MoGe-2).

**RGB-D specifics:**

- **RealSense D4xx / L515**: active IR stereo (D4xx) or LiDAR-class TOF (L515).
  Depth and RGB are co-triggered but **not** pixel-aligned out of the box —
  use the `rs2_align` filter (or pre-aligned topics) before treating depth
  as RGB-frame metric. Intrinsics: read the colour stream's intrinsics for
  the RGB component; ignore the depth-stream intrinsics (depth is aligned
  into the colour frame).
- **Microsoft Kinect Azure / Kinect v2**: TOF depth with non-trivial
  invalidation near object edges. Mask invalidated pixels (depth == 0) before
  encoding as a point cloud.
- **Stereolabs ZED 2 / X**: stereo with a built-in disparity engine. Either
  store both raw left/right images and treat as a stereo rig (preferred —
  Foundation Stereo can re-derive depth at higher quality), or store the
  ZED-native depth as a per-frame `PointCloudsComponent`.
- **Learned mono-depth** (Marigold, DepthAnythingV2, MoGe-2) is **scale
  ambiguous**. Either anchor with one absolute reference (a known object
  size, ground-plane height, IMU + visual-inertial scale) before storing,
  or accept that scale refinement (r2s module 6) will run downstream.

- **Rig origin**: camera optical centre (or device body if you have a static
  IMU offset).
- **Trajectory**: ARKit/ARCore pose stream, IMU+camera VIO, or depth-aided
  RGB-D SLAM (Open3D, Spectacular AI).
- **No LiDAR** — depth is *not* a LiDAR. Two valid encodings:
  1. **Per-frame `PointCloudsComponent`** (preferred for learned/stereo-quality
     depth, where reliability is uneven). Convert depth + intrinsics →
     camera-frame XYZ, transform to world via `T_camera_world(t)`, store as
     a point cloud per frame. Carry the dense depth into `generic_data` if
     downstream consumers want it.
  2. **Synthetic spinning LiDAR** (only if your depth is dense and reliable).
     Sample a fixed grid of azimuths/elevations, ray-cast against the depth
     map at each frame, and write a `LidarSensorComponent`. This is more
     work and less honest than option 1 — prefer point clouds unless NuRec
     specifically needs a LiDAR component.
- Camera intrinsics: pinhole or fisheye depending on the lens. Depth-camera
  manufacturers ship calibration JSON — copy `fx, fy, cx, cy` and distortion
  coefficients verbatim.
- Frame timestamps: depth and RGB are usually co-triggered. Use the RGB
  exposure timestamp; depth has no separate component.

### Mono + LiDAR (handheld / robot)

Single camera + spinning or solid-state LiDAR, e.g. handheld scanner, ground
robot, drone.

- **Rig origin**: choice driven by mechanical mounting. If the LiDAR is the
  reference for ego-motion (LiDAR-inertial SLAM), set rig = LiDAR (so
  `T_lidar_rig = I`). Otherwise pick the IMU body frame or camera centre.
- **Trajectory**: LIO-SAM, FAST-LIO, or any LiDAR-inertial pipeline.
  Re-reference to the first frame; cast last (float64 → float32).
- **Static extrinsics**: `T_camera_rig` from camera-LiDAR calibration
  (kalibr, lidar_align). Apply the NCore camera-frame rotation if your
  calibration target uses a different convention.
- **LiDAR**:
  - Spinning (Velodyne / Ouster / Hesai / Robosense / Livox Mid-360 in
    repetitive mode): use `RowOffsetStructuredSpinningLidarModelParameters`
    with the sensor's real elevations (datasheet) and `"cw"`. For Ouster
    populate `row_azimuth_offsets_rad` from `beam_azimuth_angles`.
  - Solid-state non-repetitive (Livox Avia, Mid-40, Mid-70 in
    non-repetitive mode): the spinning model does not fit. Either replay
    each scan as a fake "spinning" sweep (stash directions in
    `column_azimuths_rad` per-scan — fragile) or, **preferred**, write the
    points as a `PointCloudsComponent` per frame in the sensor frame and
    skip `LidarSensorComponent`. NuRec can consume point clouds.
- Cuboids: usually unavailable for non-AV — leave the component empty or
  unregistered.
- Ego masks: render the rig body if it intrudes on the FOV (drone arm,
  robot chassis); empty `{}` if not.

### Solid-state / non-repetitive LiDAR (Livox)

Livox Avia / Mid-40 / Mid-70 — and Mid-360 in non-repetitive mode — produce
a point cloud per scan that does **not** lay out on a row-major spinning
grid. The `RowOffsetStructuredSpinningLidarModel` does not fit; forcing it
(synthetic columns, fake row bins) breaks NCore validation and
motion-compensation alignment.

Encoding rules:

- **Preferred** — write each scan as a per-frame `PointCloudsComponent`
  instance in the **sensor frame**, with per-point µs timestamps. Skip
  `LidarSensorComponent` entirely. NuRec consumes point clouds and r2s depth
  refinement (module 7) treats them as it would LiDAR sweeps.
- **Fallback for repetitive Mid-360** — the Mid-360 in repetitive mode does
  produce a structured grid; treat it as a spinning LiDAR with the
  datasheet's beam table and `spinning_direction="cw"`. This is the only
  Livox variant the spinning model fits.

Per-point timestamps: Livox custom messages carry `offset_time` (ns from
sweep start). Convert to absolute µs:
`timestamp_us = sweep_start_us + offset_time_ns // 1000`.

### IMU + camera (visual-inertial)

IMU is **not** stored as its own NCore component — it densifies the pose
trajectory (and, optionally, anchors metric scale on monocular setups). Two
paths:

1. **VIO trajectory (preferred)**: run a visual-inertial pipeline (cuVSLAM
   stereo-inertial, ORB-SLAM3, OpenVSLAM, OpenVINS, Spectacular AI) →
   IMU-rate (100–200 Hz) `T_world_rig` poses → store as the dynamic pose.
   The trajectory is already dense enough that per-ray motion compensation
   works without further densification.
2. **IMU integration only (bootstrap / fallback)**: pre-integrate IMU
   (linear accel + angular vel) over short windows, anchor each window with
   the next available camera- or LiDAR-rate pose. Useful for filling gaps or
   extrapolating to per-LiDAR sweep timestamps when the SLAM stack only
   emits poses at frame rate.

Either way, IMU sample timestamps go into the pose trajectory, **not** into
a separate component. If a downstream consumer wants raw IMU, stash the
samples in the sequence-level `generic_meta_data["imu_samples"]` (compact)
or in a sidecar file referenced from there.

Calibration: IMU-to-camera and IMU-to-LiDAR extrinsics live in the static
pose graph as `T_imu_rig = I` if you take the IMU body frame as the rig
(common for legged robots and drones), with `T_camera_rig` /
`T_lidar_rig` from kalibr / `lidar_align` outputs. Pick **rig = IMU** when
the IMU is the trajectory reference; otherwise pick the mechanical body
frame and store `T_imu_rig` for downstream tools that want IMU-frame data.

### ROS2 bag (MCAP / SQLite3)

Most robotics datasets ship as ROS2 bags. The standard path is the
**rosbags** Python library — no `rclpy` and no native ROS install required:

```bash
pip install rosbags pyav   # pyav decodes H.264 video chunks; rosbags handles .mcap and .db3
```

Convert in **two passes**: first enumerate sensors and collect calibration /
the static TF tree, then stream frames into the V4 writer. Common topic →
component mapping:

| ROS2 message type | NCore mapping |
|-------------------|---------------|
| `sensor_msgs/Image` (raw) | `CameraSensorComponent.store_frame` — re-encode to JPEG with PIL before storing (NCore stores bytes verbatim; raw bitmaps balloon the store) |
| `sensor_msgs/CompressedImage` | `CameraSensorComponent.store_frame` — pass `data` directly with `image_format="jpeg"` or `"png"` |
| `foxglove_msgs/CompressedVideo` (H.264 chunks) | Decode with PyAV (`av.open(BytesIO(...))`) → re-encode each frame to JPEG → store |
| `sensor_msgs/CameraInfo` | Source for `OpenCVPinholeCameraModelParameters` (or fisheye) — `K`, `D`, `width/height` |
| `tf2_msgs/TFMessage` | Read once, build the static TF tree (rosbags' built-in TF helper), derive `T_sensor_rig` per camera/lidar |
| `nav_msgs/Odometry`, `geometry_msgs/PoseStamped` | Dynamic `T_world_rig` waypoints — dense (IMU rate) if available, sparse otherwise |
| `sensor_msgs/Imu` | Trajectory densifier only — integrate or feed into VIO; never stored as its own NCore component |
| `sensor_msgs/PointCloud2` | LiDAR sweep — per-point fields `(x, y, z, intensity, ring, t)` map to `direction = xyz / norm`, `intensity`, `model_element[:, 0] = ring`. If `t` is absent, synthesise from sweep duration and azimuth bin (see Mono + LiDAR notes). |
| `livox_ros_driver2/CustomMsg` | Livox custom format — same as PointCloud2 plus per-point `offset_time` (ns from sweep start). Solid-state non-repetitive scans go into `PointCloudsComponent`, **not** `LidarSensorComponent` (see solid-state recipe). |

Notes:

- Rolling-shutter ROS2 cameras: `Image.header.stamp` is the **frame trigger**.
  Compute `[exposure_start, exposure_end]` from `CameraInfo` exposure metadata
  if present, otherwise read the device datasheet. Setting
  `frame_timestamps_us = [trigger, trigger]` is acceptable for
  well-synchronised global-shutter rigs, **wrong** for rolling-shutter
  unless the readout time is small enough to ignore.
- TF tree → rig frame: pick `base_link` (or whichever frame is configured
  as the robot body) as **rig**. ROS REP 105 (`x` forward, `y` left,
  `z` up) matches NCore's rig convention exactly — no rotation needed.
- The `rosbags` API opens both MCAP and SQLite3 with the same code path;
  only the file extension differs.
- For multi-bag datasets (one bag per take), one bag = one V4 sequence.
- Keep the bag's recording timestamps (not wall-clock playback) — bag
  timestamps are µs-quantised already, perfect for NCore.

### Aerial / drone

Drones add three failure modes on top of the standard mono+lidar / stereo
recipes:

- **Fast rotation**: pose trajectory must densify enough to keep angular
  drift per waypoint below ~0.5°. At 100 °/s yaw rate, that means
  waypoints every 5 ms — much tighter than a typical 10 Hz LiDAR. Use IMU
  samples (200 Hz+) as the trajectory backbone; do not rely on per-LiDAR
  poses alone.
- **Rolling-shutter cameras** are common on consumer drones. Set
  `ShutterType.ROLLING_TOP_TO_BOTTOM`, supply real `[exposure_start,
  exposure_end]` per frame, and carry the per-row shutter delay into the
  intrinsics if your model supports it (`shutter_delay_us` on
  `OpenCVPinholeCameraModelParameters` for FTheta-like models).
- **Wide altitude / range**: a 5+ km loop in raw UTM at the end of the
  flight loses the same precision that the AV first-pose re-referencing
  fixes. For continuous takes under ~2 km, re-reference once to the
  takeoff pose; for longer flights, segment the bag and emit one V4
  sequence per segment.

Rig origin on a drone: **IMU body frame** is the canonical choice. All
extrinsics are then `T_sensor_imu`. Rotor masks are usually unnecessary —
rotors blur out of the FOV at flight RPM — but include a static mask of any
fixed gimbal arm visible in the FOV.

---

## Robotics pipeline shards (r2s)

The r2s ("robotics-to-sim") pipeline chains NCore through pose, depth, mask,
and refinement steps before NuRec. Each step writes its **own** NCore shard
so that any module can be swapped, re-run, or skipped without rebuilding the
preceding stages. If you're authoring a converter that feeds r2s — or a
downstream module that reads/writes intermediate shards — follow this
naming + content contract.

### Shard naming and contents

| Shard | Producer | Components carried | Notes |
|-------|----------|--------------------|-------|
| `{name}.zarr.itar` | Ingestion (this skill) | `CameraSensor`, `Intrinsics`, `Poses` (static `T_sensor_rig` + identity-or-GPS `T_world_rig`), optional `LidarSensor`, sequence `generic_meta_data.stereo_pairs` | Base shard. All-frame images, no derived data. |
| `{name}.poses.zarr.itar` | Pose Estimation (cuVSLAM / KISS-ICP) | `Poses` (per-frame dynamic `T_world_rig`) | Full trajectory at every input-frame timestamp; **not** subsampled. |
| `{name}.kf.zarr.itar` | Pose Refinement (cuSFM / COLMAP) | `CameraSensor` (keyframe subset), `Intrinsics` (refined), `Poses` (BA-optimised), `PointClouds` (sparse SfM map, instance `sfm_points`) | Keyframe timestamps must be a strict subset of the base shard's frame timestamps. |
| `{name}.depth.zarr.itar` | Depth Generation (Foundation Stereo / MoGe-2 / LiDAR projection) | per-frame `PointClouds` named `depth_<camera_id>` (one entry per keyframe) | Until a first-class `DepthComponent` lands in V4, encode dense depth as a per-frame point cloud (camera-frame XYZ from depth + intrinsics). Stash uint16 mm depth in `generic_data["depth_mm"]` for tools that want pixel-aligned access. |
| `{name}.masks.zarr.itar` | Semantic Mask Generation (SAM2 / Grounded-SAM) | `Masks` (per-camera, per-keyframe) | Masks here are **semantic** (sky, ground, dynamic-object), not just ego-vehicle. Ego masks may live alongside in the same component instance. |
| `{name}.poses.scaled.zarr.itar` | Scale Refinement | `Poses` with `generic_meta_data.scale_factor` and per-frame bias | Same trajectory schema as `{name}.poses.zarr.itar`; recovered scale + per-frame bias recorded in metadata. |
| `{name}.depth.refined.zarr.itar` | Depth Refinement | per-frame `PointClouds` (refined) | Same component layout as `{name}.depth.zarr.itar`, with cross-frame scale/bias and occlusion removal applied. |
| `{name}.sim.zarr.itar` | Sensor-Data Simulation (NuRec renderer) | `CameraSensor`, `Poses`, `Intrinsics` matching the **target** trajectory | Same schema as a real base shard — directly consumable by any module. **Always** tag `generic_meta_data.source = "simulation"`. |

### Component encoding until V4 ships first-class depth / mappoints

The r2s spec references two components that are not yet in upstream V4:

- `DepthComponent` (per-pixel uint16 depth, mm, per camera per keyframe).
- `MapPointsComponent` (sparse SfM points + per-frame observations).

Until they ship, the safe encodings are:

- **Dense depth** → per-frame `PointCloudsComponent`, instance name
  `depth_<camera_id>`, points in the camera frame at the keyframe
  timestamp. Carry the raw uint16 depth array in
  `generic_data["depth_mm"]` if downstream tools need pixel-aligned access.
- **SfM map** → a single `PointCloudsComponent` instance named
  `sfm_points` (already produced by the upstream COLMAP converter).
  Per-frame observation tracks, when needed, go in
  `generic_meta_data["track_observations"]`.

Treat these as forward-compatible encodings: when V4 adds the first-class
components, the migration path is mechanical (copy the same arrays into
the new writer) and the data shape is already correct.

### Validator hooks

Each module's validator runs after its shard is written. The spec
assertions that map onto NCore writer-time checks are already enforced
(timestamp ordering, valid intrinsics, pose-trajectory completeness,
strict-increasing LiDAR azimuths). The remaining checks (depth value range,
mask temporal consistency, scale factor bounds, etc.) are the **module's**
responsibility — see the r2s `modules.md` for the canonical list of per-step
validations.

---

## Validation & end-to-end NuRec

After every conversion, before assuming success:

1. **`ncore_vis`** — visualise the store. Wrong sensor extrinsics, mirrored
   LiDAR (`spinning_direction` flip), or rotated cameras are obvious here.
2. **`ncore_project_pc_to_img`** — projects LiDAR onto camera frames. Crisp
   alignment confirms `T_lidar_rig`, `T_camera_rig`, intrinsics, per-ray
   timestamps, and pose-trajectory density are all correct simultaneously.
   Smearing or doubling = motion-comp error (per-ray timestamps wrong, or
   pose trajectory too sparse).
3. **NuRec end-to-end**. Hand the converted store to the
   [`nre`](../nre/SKILL.md) sibling skill — Workflow A wires together
   aux-data generation (via `nre-tools`), 3DGUT training, USDZ export,
   and novel-view rendering, and `references/configuration.md` documents
   how to override `dataset.camera_ids` / `dataset.lidar_ids` so the
   training Hydra recipe matches your sensor IDs. For repeatable cluster
   runs, wrap the convert + train + export sequence in an OSMO / Slurm /
   Kubernetes pipeline that pins both the
   [`NVIDIA/ncore`](https://github.com/NVIDIA/ncore) and NRE container
   versions.

NuRec's "Ensure Data Quality" doc (in the NuRec image) also lists
`check_lidar_camera_sweep_alignment` and other validators you can run
post-conversion.

---

## Troubleshooting (common failure modes and the fix file)

The canonical fixes live in code (`example_converter.py` inline comments) and
in the spec; most "NuRec produced garbage" complaints reduce to one of:

| Symptom (NuRec / `ncore_vis`) | Almost-always cause | Fix |
|--------|----------------|-----|
| Z-flipped surfaces | `spinning_direction` wrong | Set `"cw"` for all common automotive spinning LiDARs |
| Whole point cloud rotated horizontally | `column_azimuths_rad` starting at 0 instead of real heading | Derive azimuths from per-column pose data, or rotate `T_lidar_rig` to compensate (consistently) |
| Strict-increasing assert in NCore writer | LiDAR elevations ascending or contain duplicates | Sort descending, nudge duplicates by 1e-6 rad |
| `"Dynamic poses must cover the full sequence time range"` | Sequence interval built with `from_start_end(start, end + 1)` | Pass real inclusive end; the helper internally adds 1 |
| `RuntimeError: double != float` at "Get Lidar Point Clouds" | `T_camera_rig` / `T_lidar_rig` written as float64 | Cast extrinsics to float32 (only `world_world_global` stays float64) |
| Reconstruction loses sub-cm detail at scene scale | Poses not re-referenced; raw UTM/ECEF cast to float32 | `poses = inv(poses[0]) @ poses` in float64, then cast |
| Motion-comp blur / rowing artefacts | Pose trajectory too sparse | Combine all available pose sources (every camera × every frame, plus IMU/GPS), unique + sort |
| Ghosting / motion blur on rolling-shutter cameras | Single global timestamp used for all cameras, or `ShutterType.GLOBAL` set on rolling sensor | Per-camera per-frame `[exposure_start, exposure_end]`; map shutter direction by enum **name** |
| Cuboids floating above ground | Bottom-center origin not converted | `centroid_z += dim_z / 2`; verify with `mean(z) - mean(h)/2 ≈ 0` |
| Hood / roof rack baked into reconstruction | Missing ego mask | Generate per-camera binary ego mask; pass via `MasksComponent.store_camera_masks` |
| LiDAR projects 50 ms ahead of camera | Frame-start used as sweep midpoint (or vice versa) | `frame_timestamps_us[0]` = real sweep start; per-ray `timestamp_us` linear in column |
| Foundation Stereo (or aux-data) cannot find its pair | Missing `stereo_pairs` on sequence `generic_meta_data` | Set `generic_meta_data={"stereo_pairs": [{"left": ..., "right": ...}]}` on `SequenceComponentGroupsWriter` |
| Livox scan stored as a `LidarSensorComponent` but most rays read range 0 | Solid-state non-repetitive doesn't have a column grid | Use per-frame `PointCloudsComponent` (sensor-frame XYZ + per-point µs); skip `LidarSensorComponent` |
| Drone reconstruction shows streaks / motion smear | Trajectory densified at LiDAR rate (10 Hz) on a 200 °/s yaw motion | Use IMU samples (200 Hz+) as trajectory backbone; merge per-frame poses on top |
| ROS2 raw-Image topic blows up store size 30× | Stored bitmap bytes verbatim instead of re-encoding to JPEG | PIL-encode each `sensor_msgs/Image` to JPEG before `store_frame`; use `image_format="jpeg"` |
| Pipeline downstream module sees real data instead of sim | Renderer-output shard missing `source` tag | Set `generic_meta_data["source"] = "simulation"` (and `"model_checkpoint"`) on the simulated sequence shard |
| IMU samples written into a "sensor" component and rejected by NuRec | IMU is not a first-class V4 component | Drop IMU samples into the pose trajectory (or `generic_meta_data["imu_samples"]`); never register an IMU writer |

---

## Limitations

- **No first-class IMU component.** V4 has no IMU sensor type — IMU
  samples must ride in the pose trajectory or in `generic_meta_data`
  (never as a sensor writer).
- **Per-ray LiDAR is the only motion-comp-safe option** for spinning
  LiDARs. If the dataset only gives frame-level timestamps, expect
  motion-blur smearing in NuRec; reconstruct per-ray timestamps from
  azimuth or accept the artefact.
- **Sub-cm scene detail at world scale** requires float64 pose math
  followed by re-referencing (`poses = inv(poses[0]) @ poses`)
  *before* writing as float32. Raw UTM/ECEF casts to float32 will
  silently lose centimetre-scale geometry.
- **Rolling-shutter cameras** need per-camera per-frame
  `[exposure_start, exposure_end]` *and* the correct `ShutterType`
  by enum **name** — a single global timestamp will produce ghosting.
- **Solid-state non-repetitive LiDAR (Livox-style)** must use
  `PointCloudsComponent`, not `LidarSensorComponent` (the latter
  assumes a column grid).
- **The converter is one-way.** There is no `v4 → original` tool;
  always keep the source dataset alongside.
- This skill is **convert + validate only**. Training, rendering, or
  USDZ packaging is the `nre` skill's job.

## Additional resources

- Install: `pip install nvidia-ncore`
- Source, docs, examples: <https://github.com/NVIDIA/ncore>
- V4 spec: <https://nvidia.github.io/ncore/data/conventions.html>
- Sensor models: <https://nvidia.github.io/ncore/data/sensor_models.html>
- API reference: <https://nvidia.github.io/ncore/apis/data.v4.html>
- Conversion guides (PAI, Waymo, COLMAP, ScanNet++):
  <https://nvidia.github.io/ncore/conversions/index.html>
- Skeleton converter (FILL IN walkthrough):
  [`ncore_template/impl/data_converter/example_converter.py`](ncore_template/impl/data_converter/example_converter.py)
- NRE training / rendering / USDZ export from a converted store:
  [`../nre/SKILL.md`](../nre/SKILL.md)
- HuggingFace dataset catalog and download recipes:
  [`../physical-ai-datasets/SKILL.md`](../physical-ai-datasets/SKILL.md)
- Upstream NCore PAI / Waymo / COLMAP converter targets:
  <https://github.com/NVIDIA/ncore/tree/main/tools/data_converter>
