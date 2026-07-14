# Viser Mixed-Camera Projection and Follow-Camera Bugfix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `viser_gui_4d` reliably switch among FTheta and OpenCVPinhole cameras, keep Gaussian backdrop and overlays on the same projection, interpolate Follow Camera poses smoothly, and provide render/viewer parity evidence for center-versus-periphery quality.

**Architecture:** Introduce a pure `CameraRenderState` resolver as the single source of truth for camera id, pose sampling, projection model, calibrated rays, resolution, FOV, and overlay projector. `Viser4DViewer` applies that state atomically; a generic calibrated overlay compositor handles both FTheta and OpenCVPinhole projectors. Follow Camera uses timestamped SE(3) interpolation, while an offline parity script compares native `render.py` output against the same viewer render contract.

**Tech Stack:** Python 3.11/3.14, NumPy, PyTorch, Kaolin Camera, viser 1.0, NCore SDK, Pillow, pytest, 3DGUT/3DGRT.

## Global Constraints

- Follow TDD strictly: each production change must be preceded by a test that fails for the intended reason.
- Do not change checkpoint schema or retrain C4.
- Preserve legacy behavior for cameras without NCore calibration: ideal-pinhole browser primitives remain the fallback.
- FTheta and OpenCVPinhole state are mutually exclusive; every camera switch must explicitly set or clear every projection field.
- The selected Camera dropdown value, active camera state, client pose source, engine intrinsics, render resolution, FOV, and overlay projector must always refer to the same camera id.
- Use NCore camera convention consistently: camera `+Z` forward, `+Y` down, c2w in world-global frame; image-space projectors receive `world_to_camera_flip=np.eye(4)`.
- Do not claim the center/periphery blur is fixed until the render/viewer parity task identifies whether the residual belongs to the viewer or checkpoint.
- Mac tests must remain CPU-runnable by stubbing viser/kaolin/engine as existing viewer tests do.
- Final GPU validation runs on inceptio RTX 4090 with the C4 11-camera 30k checkpoint and manifest listed in Task 8.
- Update `docs/viser_mixed_camera_buglist.md` statuses and evidence after each closed task; do not mark the overall buglist closed before Task 8 passes.

---

## File Structure

### New files

- `threedgrut_playground/utils/camera_render_state.py` — immutable camera projection/pose state, model enum, timestamp interpolation, resolver.
- `threedgrut/tests/test_camera_render_state.py` — pure NumPy unit tests for state resolution, stale-field clearing, timestamp interpolation, and gap reporting.
- `threedgrut/tests/test_viser_gui_4d_camera_switch.py` — viewer integration tests for GUI initial camera, atomic state application, mixed transition sequences, frustum source, and overlay lifecycle.
- `scripts/validate_viser_render_parity.py` — headless native-camera parity and center/periphery metric tool.
- `threedgrut/tests/test_validate_viser_render_parity.py` — CPU tests for radial masks and parity metric computation.
- `docs/T8_artifacts/mixed_camera/README.md` — validation protocol and artifact index.

### Modified files

- `threedgrut_playground/viser_gui_4d.py` — consume/apply `CameraRenderState`, preserve initial camera, update diagnostics, use interpolated Follow Camera pose, route all calibrated overlays image-space.
- `threedgrut_playground/utils/viser_overlay_compositor.py` — generalize compositor from FTheta-only to any projector implementing `project_points` / `project_polylines`.
- `threedgrut_playground/utils/viser_math.py` — add matrix/quaternion conversion and short-arc SLERP helpers used by pose interpolation.
- `threedgrut/tests/test_viser_math.py` — quaternion roundtrip and SLERP tests.
- `threedgrut/tests/test_viser_gui_4d_cuboid_overlay.py` — add OpenCVPinhole overlay path and dynamic projector transition coverage.
- `threedgrut/tests/test_viser_4d_ftheta_overlay_integration.py` — keep FTheta behavior pinned after compositor generalization.
- `.claude/skills/viser-gui-4d/SKILL.md` — remove temporary “not trustworthy” warning only after Task 8, replace with fixed mixed-camera workflow and diagnostics.
- `docs/viser_mixed_camera_buglist.md` — status/evidence updates.
- `v5_plan.md` — append viewer-fix Done Log evidence only if project convention requires this work under Phase C/C4; do not change C4 promotion decision in this plan.
- `v2_architecture.md` — register `CameraRenderState` and the projection-state invariant after implementation.

---

### Task 1: Pure CameraRenderState and Projection Model Resolver

**Files:**
- Create: `threedgrut_playground/utils/camera_render_state.py`
- Create: `threedgrut/tests/test_camera_render_state.py`

**Interfaces:**
- Consumes: one `_load_multi_cam_poses()` entry with keys `c2w`, `timestamps_us`, `ftheta_dict`, `opencv_pinhole_dict`, `opencv_pinhole_rays`, `resolution`, `fov_y_rad`.
- Produces:
  - `CameraModelKind(str, Enum)` with values `FTHETA`, `OPENCV_PINHOLE`, `IDEAL_PINHOLE`.
  - `PoseSample` dataclass with `c2w: np.ndarray`, `left_idx: int`, `right_idx: int`, `alpha: float`, `nearest_dt_us: int`, `source_gap_us: int`, `interpolated: bool`.
  - `CameraRenderState` dataclass with `camera_id`, `model_kind`, `pose_sample`, `resolution`, `fov_y_rad`, `ftheta_dict`, `opencv_pinhole_dict`, `opencv_pinhole_rays`.
  - `interpolate_c2w(poses, timestamps_us, t_us) -> PoseSample`.
  - `resolve_camera_render_state(camera_id, entry, t_us) -> CameraRenderState`.

- [ ] **Step 1: Add quaternion helper RED tests**

Extend `threedgrut/tests/test_viser_math.py` with exact contracts:

```python
def test_wxyz_to_mat_roundtrip():
    c2w = np.eye(4, dtype=np.float64)
    angle = np.deg2rad(70.0)
    c2w[:3, :3] = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle),  np.cos(angle), 0.0],
        [0.0,            0.0,           1.0],
    ])
    rebuilt = wxyz_to_mat(mat_to_wxyz(c2w))
    np.testing.assert_allclose(rebuilt[:3, :3], c2w[:3, :3], atol=1e-6)


def test_slerp_wxyz_midpoint_short_arc():
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array([0.0, 0.0, 0.0, 1.0])
    q = slerp_wxyz(q0, q1, 0.5)
    R = wxyz_to_mat(q)[:3, :3]
    np.testing.assert_allclose(R[:2, :2], [[0.0, -1.0], [1.0, 0.0]], atol=1e-5)
```

- [ ] **Step 2: Run quaternion tests and verify RED**

Run:

```bash
python -m pytest threedgrut/tests/test_viser_math.py -k 'wxyz_to_mat or slerp' -v
```

Expected: FAIL because `wxyz_to_mat` and `slerp_wxyz` are not defined.

- [ ] **Step 3: Implement minimal quaternion helpers**

Add to `threedgrut_playground/utils/viser_math.py`:

```python
def wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    """Unit wxyz quaternion to homogeneous 4x4 rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    return out


def slerp_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Shortest-arc SLERP; normalized lerp for nearly identical quaternions."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 /= max(np.linalg.norm(q0), 1e-12)
    q1 /= max(np.linalg.norm(q1), 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        out = (1.0 - alpha) * q0 + alpha * q1
        return out / max(np.linalg.norm(out), 1e-12)
    theta = np.arccos(dot)
    out = (np.sin((1.0-alpha)*theta) * q0 + np.sin(alpha*theta) * q1) / np.sin(theta)
    return out / max(np.linalg.norm(out), 1e-12)
```

- [ ] **Step 4: Run quaternion tests and verify GREEN**

Run the Step 2 command. Expected: both new tests PASS and existing `test_viser_math.py` remains green.

- [ ] **Step 5: Write CameraRenderState RED tests**

Create `threedgrut/tests/test_camera_render_state.py` with these cases:

```python
def test_resolve_ftheta_sets_only_ftheta_fields():
    state = resolve_camera_render_state("fish", _entry_ftheta(), 500_000)
    assert state.model_kind is CameraModelKind.FTHETA
    assert state.ftheta_dict is not None
    assert state.opencv_pinhole_dict is None
    assert state.opencv_pinhole_rays is None


def test_resolve_opencv_sets_only_opencv_fields():
    state = resolve_camera_render_state("wide", _entry_opencv(), 500_000)
    assert state.model_kind is CameraModelKind.OPENCV_PINHOLE
    assert state.ftheta_dict is None
    assert state.opencv_pinhole_dict is not None
    assert state.opencv_pinhole_rays is not None


def test_interpolate_c2w_midpoint_lerps_translation_and_slerps_rotation():
    sample = interpolate_c2w(_poses_identity_to_180deg(), np.array([0, 1_000_000]), 500_000)
    np.testing.assert_allclose(sample.c2w[:3, 3], [5.0, 0.0, 0.0], atol=1e-6)
    forward = sample.c2w[:3, :3] @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(forward[:2], [0.0, 1.0], atol=1e-5)
    assert sample.left_idx == 0 and sample.right_idx == 1
    assert sample.alpha == pytest.approx(0.5)
    assert sample.interpolated is True


def test_interpolate_c2w_reports_large_source_gap():
    sample = interpolate_c2w(_two_identity_poses(), np.array([0, 600_000]), 300_000)
    assert sample.source_gap_us == 600_000
    assert sample.nearest_dt_us == 300_000


def test_resolve_rejects_ftheta_and_opencv_both_set():
    entry = _entry_ftheta()
    entry["opencv_pinhole_dict"] = _opencv_dict()
    entry["opencv_pinhole_rays"] = np.zeros((4, 6, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_camera_render_state("bad", entry, 0)
```

Use synthetic 6×4 resolution entries to keep tests small. The OpenCV entry must use distinct sentinel arrays so later tests can check identity and stale-state clearing.

- [ ] **Step 6: Run CameraRenderState tests and verify RED**

Run:

```bash
python -m pytest threedgrut/tests/test_camera_render_state.py -v
```

Expected: collection FAIL because `camera_render_state.py` does not exist.

- [ ] **Step 7: Implement the pure resolver**

Implement dataclasses and functions in `camera_render_state.py`. Required rules:

```python
if ftheta_dict is not None:
    model_kind = CameraModelKind.FTHETA
    assert opencv_dict is None and opencv_rays is None
elif opencv_dict is not None and opencv_rays is not None:
    model_kind = CameraModelKind.OPENCV_PINHOLE
else:
    model_kind = CameraModelKind.IDEAL_PINHOLE
```

`interpolate_c2w` must:

- validate `(N,4,4)` poses and matching sorted timestamps;
- clamp before first/after last timestamp;
- use `np.searchsorted(..., side="right")` to select bracketing frames;
- lerp translation;
- convert both rotations with `mat_to_wxyz`, SLERP, then rebuild with `wxyz_to_mat`;
- return source indices, alpha, nearest timestamp delta, and bracket gap.

- [ ] **Step 8: Run pure-state tests and full math tests**

Run:

```bash
python -m pytest \
  threedgrut/tests/test_camera_render_state.py \
  threedgrut/tests/test_viser_math.py -v
```

Expected: PASS.

- [ ] **Step 9: Commit Task 1**

```bash
git add threedgrut_playground/utils/camera_render_state.py \
        threedgrut_playground/utils/viser_math.py \
        threedgrut/tests/test_camera_render_state.py \
        threedgrut/tests/test_viser_math.py
git commit -m "feat(viser): add atomic camera render state resolver"
```

---

### Task 2: Preserve Initial Camera and Apply Camera State Atomically

**Files:**
- Create: `threedgrut/tests/test_viser_gui_4d_camera_switch.py`
- Modify: `threedgrut_playground/viser_gui_4d.py:140-200, 320-371, 728-750, 1108-1160`

**Interfaces:**
- Consumes: `resolve_camera_render_state(camera_id, entry, t_us)` from Task 1.
- Produces:
  - `Viser4DViewer._active_camera_state: Optional[CameraRenderState]`.
  - `Viser4DViewer._active_render_wh: Optional[tuple[int, int]]`; calibrated FTheta and OpenCVPinhole cameras both set it to their native resolution, legacy ideal pinhole leaves it `None`.
  - `Viser4DViewer._apply_camera_state(state: CameraRenderState) -> None`.
  - `Viser4DViewer._set_active_camera(cam_id: str, t_us: int, *, snap_clients: bool = True) -> CameraRenderState`.

- [ ] **Step 1: Write initial-camera RED test**

In the new test file, reuse the existing heavy-dependency stubbing pattern. Build a fake GUI object that records `add_dropdown(initial_value=...)` and assert:

```python
def test_initial_cam_id_wins_over_metadata_primary(viewer_module):
    viewer = _construct_viewer_with_gui(
        viewer_module,
        metadata_primary="camera_front_wide_120fov",
        initial_cam_id="camera_front_fisheye",
        multi_cam_poses=_mixed_camera_entries(),
    )
    assert viewer._cam_dropdown.value == "camera_front_fisheye"
    assert viewer._current_dropdown_cam == "camera_front_fisheye"
    assert viewer._active_camera_state.camera_id == "camera_front_fisheye"
```

- [ ] **Step 2: Write mixed transition RED test**

```python
def test_mixed_switch_sequence_has_no_stale_projection_fields(viewer):
    sequence = ["wide", "fish_front", "tele", "fish_rear", "wide"]
    expected = [
        (CameraModelKind.OPENCV_PINHOLE, False, True),
        (CameraModelKind.FTHETA, True, False),
        (CameraModelKind.OPENCV_PINHOLE, False, True),
        (CameraModelKind.FTHETA, True, False),
        (CameraModelKind.OPENCV_PINHOLE, False, True),
    ]
    for cam_id, (kind, has_ft, has_cv) in zip(sequence, expected):
        state = viewer._set_active_camera(cam_id, 500_000)
        assert state.model_kind is kind
        assert (viewer.ftheta_intrinsics is not None) is has_ft
        assert (viewer.opencv_pinhole_intrinsics is not None) is has_cv
        assert (viewer.opencv_pinhole_rays is not None) is has_cv
        assert viewer._current_dropdown_cam == cam_id
```

Also assert OpenCV arrays are the selected entry's sentinel arrays, not the previous camera's arrays.

Add a resolution contract:

```python
def test_calibrated_camera_uses_native_render_resolution(viewer):
    viewer._set_active_camera("wide", 500_000)
    assert viewer._active_render_wh == (6, 4)
    viewer._set_active_camera("fish_front", 500_000)
    assert viewer._active_render_wh == (6, 4)
    viewer._set_active_camera("legacy", 500_000)
    assert viewer._active_render_wh is None
```

- [ ] **Step 3: Run camera-switch tests and verify RED**

Run:

```bash
python -m pytest threedgrut/tests/test_viser_gui_4d_camera_switch.py -v
```

Expected: FAIL because `_active_camera_state`, `_apply_camera_state`, and `_set_active_camera` do not exist; initial dropdown still selects metadata primary.

- [ ] **Step 4: Implement atomic state application**

In `viser_gui_4d.py`:

1. Initialize `_active_camera_state = None` before GUI creation.
2. Replace ad-hoc constructor FTheta/OpenCV initialization with `_set_active_camera(initial_cam_id, t_us, snap_clients=False)` when valid.
3. In `_build_visibility_gui`, compute:

```python
initial_cam = (
    self._initial_cam_id
    if self._initial_cam_id in cam_options
    else self.meta.ego_primary_camera_id
    if self.meta.ego_primary_camera_id in cam_options
    else cam_options[0]
)
```

4. `_apply_camera_state` must assign all fields every time:

```python
self.ftheta_intrinsics = state.ftheta_dict
self.ftheta_render_wh = state.resolution if state.model_kind is FTHETA else None
self.opencv_pinhole_intrinsics = state.opencv_pinhole_dict
self.opencv_pinhole_rays = state.opencv_pinhole_rays
self.opencv_pinhole_render_wh = state.resolution if state.model_kind is OPENCV_PINHOLE else None
self._active_render_wh = state.resolution if state.model_kind is not IDEAL_PINHOLE else None
```

5. `_set_active_camera` resolves, applies, updates `_current_dropdown_cam`, synchronizes dropdown only when its value differs, then optionally writes client pose/FOV.
6. Keep `_snap_clients_to_camera` as a backward-compatible thin wrapper calling `_set_active_camera`.
7. In `update()`, replace the FTheta-only resolution branch with `_active_render_wh`; when non-`None`, use its exact `W,H`. The resolution slider controls actual render shape only for `IDEAL_PINHOLE` fallback.
8. Update the GUI explanation so both FTheta and OpenCVPinhole calibrated modes report native-resolution lock; camera switches refresh visibility/text rather than leaving the initial mode's message stale.

- [ ] **Step 5: Run transition tests and existing viewer tests**

Run:

```bash
python -m pytest \
  threedgrut/tests/test_viser_gui_4d_camera_switch.py \
  threedgrut/tests/test_viser_gui_4d_fov.py \
  threedgrut/tests/test_viser_gui_4d_follow_ego.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add threedgrut_playground/viser_gui_4d.py \
        threedgrut/tests/test_viser_gui_4d_camera_switch.py
git commit -m "fix(viser): make camera selection and projection state atomic"
```

---

### Task 3: Generalize Image-Space Overlay for FTheta and OpenCVPinhole

**Files:**
- Modify: `threedgrut_playground/utils/viser_overlay_compositor.py`
- Modify: `threedgrut_playground/viser_gui_4d.py:199-242, 1002-1039, 1131-1153, 1262-1292, 1370-1462, 1597-1608`
- Modify: `threedgrut/tests/test_viser_gui_4d_cuboid_overlay.py`
- Modify: `threedgrut/tests/test_viser_4d_ftheta_overlay_integration.py`

**Interfaces:**
- Produces `Viser4DOverlayCompositor(projector, height, width, subdivide_n=...)`, where projector is either `FthetaForwardProjector` or `PinholeForwardProjector`.
- Produces `Viser4DViewer._build_overlay_compositor(state) -> Optional[Viser4DOverlayCompositor]`.
- Invariant: calibrated camera state means image-space overlay is active; browser primitives are only for `IDEAL_PINHOLE` fallback.

- [ ] **Step 1: Write generic compositor RED test**

Update integration tests to construct both projectors explicitly:

```python
@pytest.mark.parametrize("projector", [
    FthetaForwardProjector(_ftheta_dict(), world_to_camera_flip=np.eye(4)),
    PinholeForwardProjector(_opencv_dict(), world_to_camera_flip=np.eye(4)),
])
def test_compositor_accepts_calibrated_projector(projector):
    cmp = Viser4DOverlayCompositor(projector, height=1080, width=1920)
    out = cmp.composite(_backdrop(), [_on_axis_polyline()], np.eye(4))
    assert out.shape == (1080, 1920, 3)
    assert np.any(out != _backdrop())
```

- [ ] **Step 2: Write dynamic overlay lifecycle RED tests**

Add viewer tests:

```python
def test_opencv_to_ftheta_creates_compositor_from_none(viewer):
    viewer._set_active_camera("wide", 0)
    assert viewer._overlay_compositor is not None
    assert isinstance(viewer._overlay_compositor.projector, PinholeForwardProjector)
    viewer._set_active_camera("fish", 0)
    assert isinstance(viewer._overlay_compositor.projector, FthetaForwardProjector)


def test_calibrated_opencv_skips_browser_cuboid_primitives(viewer):
    viewer._set_active_camera("wide", 0)
    viewer._update_active_cuboids(0)
    viewer.server.scene.add_line_segments.assert_not_called()


def test_legacy_ideal_pinhole_keeps_browser_primitives(viewer):
    viewer._set_active_camera("legacy", 0)
    viewer._update_active_cuboids(0)
    viewer.server.scene.add_line_segments.assert_called()
```

- [ ] **Step 3: Run overlay tests and verify RED**

Run:

```bash
python -m pytest \
  threedgrut/tests/test_viser_4d_ftheta_overlay_integration.py \
  threedgrut/tests/test_viser_gui_4d_cuboid_overlay.py \
  threedgrut/tests/test_viser_gui_4d_camera_switch.py -v
```

Expected: FAIL because compositor constructor is FTheta-specific, OpenCVPinhole does not create an image-space overlay, and browser primitives remain active.

- [ ] **Step 4: Generalize compositor**

Change constructor from `ftheta_dict` to `projector`:

```python
class Viser4DOverlayCompositor:
    def __init__(self, projector, height: int, width: int, subdivide_n: int = 20):
        required = ("project_points", "project_polylines")
        if not all(hasattr(projector, name) for name in required):
            raise TypeError("projector must implement project_points and project_polylines")
        self.projector = projector
        self.renderer = OverlayRenderer(height=height, width=width)
        ...
```

Update all existing call sites/tests to instantiate `FthetaForwardProjector` first. Preserve output behavior and FTheta flip identity.

- [ ] **Step 5: Build projector by state**

Add `_build_overlay_compositor(state)`:

```python
if state.model_kind is CameraModelKind.FTHETA:
    projector = FthetaForwardProjector(state.ftheta_dict, world_to_camera_flip=np.eye(4))
    subdivide_n = 20
elif state.model_kind is CameraModelKind.OPENCV_PINHOLE:
    projector = PinholeForwardProjector(state.opencv_pinhole_dict, world_to_camera_flip=np.eye(4))
    subdivide_n = 4
else:
    return None
return Viser4DOverlayCompositor(projector, height=H, width=W, subdivide_n=subdivide_n)
```

Rebuild compositor on every camera-id/model/resolution change. `_apply_camera_state` assigns it even when previous value was `None`.

Rename comments and helpers from “FTheta overlay” to “calibrated image-space overlay”. `_collect_overlay_layer_specs` remains model-agnostic.

Build `_overlay_static_ego_polylines` and `_overlay_static_track_polylines`
unconditionally whenever metadata exists. Their content is world-space and independent of the initial
camera model; only projection is state-dependent. Add a regression test that constructs the viewer with
an initial OpenCVPinhole camera, switches to FTheta, and still receives ego + track trajectory specs.

- [ ] **Step 6: Ensure primitive lifecycle transitions both directions**

When switching from `IDEAL_PINHOLE` to calibrated overlay:

- remove stale cuboid line segments and 3D labels;
- remove stale ego/track trajectory line segments;
- use cached world polylines for the image overlay.

When switching from calibrated overlay to `IDEAL_PINHOLE`:

- clear compositor;
- recreate browser trajectory primitives once;
- next `_update_active_cuboids` recreates cuboid primitives.

Do not duplicate overlays or labels.

- [ ] **Step 7: Run overlay suite and projector suites**

Run:

```bash
python -m pytest \
  threedgrut/tests/test_ftheta_projector.py \
  threedgrut/tests/test_pinhole_projector.py \
  threedgrut/tests/test_viser_4d_ftheta_overlay_integration.py \
  threedgrut/tests/test_viser_gui_4d_cuboid_overlay.py \
  threedgrut/tests/test_viser_gui_4d_camera_switch.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

```bash
git add threedgrut_playground/utils/viser_overlay_compositor.py \
        threedgrut_playground/viser_gui_4d.py \
        threedgrut/tests/test_viser_4d_ftheta_overlay_integration.py \
        threedgrut/tests/test_viser_gui_4d_cuboid_overlay.py \
        threedgrut/tests/test_viser_gui_4d_camera_switch.py
git commit -m "fix(viser): align overlays for FTheta and OpenCVPinhole cameras"
```

---

### Task 4: Current-Camera Frustum and Diagnostics UI

**Files:**
- Modify: `threedgrut_playground/viser_gui_4d.py:718-758, 1083-1106`
- Modify: `threedgrut/tests/test_viser_gui_4d_camera_switch.py`

**Interfaces:**
- Produces `_active_camera_status_text(state) -> str`.
- Uses `_active_camera_state.pose_sample.c2w` for ego frustum when a camera is selected.

- [ ] **Step 1: Write frustum RED test**

```python
def test_ego_frustum_uses_current_camera_not_initial_camera(viewer):
    viewer._initial_cam_id = "front"
    viewer._set_active_camera("rear", 500_000)
    viewer._update_ego_frustum(500_000)
    np.testing.assert_allclose(viewer.h_ego_frustum.position, [20.0, 0.0, 0.0])
```

Use different translations for front and rear synthetic entries.

- [ ] **Step 2: Write diagnostics RED tests**

```python
def test_status_text_identifies_projection_and_pose_source(viewer):
    state = viewer._set_active_camera("fish", 500_000)
    text = viewer._active_camera_status_text(state)
    assert "camera: fish" in text
    assert "model: FTheta" in text
    assert "render: 6×4" in text
    assert "interpolated" in text
    assert "overlay: FTheta image-space" in text


def test_status_text_warns_on_large_pose_gap(viewer):
    state = viewer._set_active_camera("gappy", 300_000)
    assert "WARNING" in viewer._active_camera_status_text(state)
    assert "600.0 ms" in viewer._active_camera_status_text(state)
```

Set large-gap warning threshold to `250_000 us` as a named constant.

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
python -m pytest threedgrut/tests/test_viser_gui_4d_camera_switch.py \
  -k 'frustum or status' -v
```

Expected: FAIL because frustum still uses `_initial_cam_id` and status helper does not exist.

- [ ] **Step 4: Implement current-state frustum and read-only status**

- `_update_ego_frustum` first uses `_active_camera_state.pose_sample.c2w`; only fallback to metadata ego pose when no active camera state exists.
- Add a GUI markdown/status field below Camera/Follow Camera.
- Refresh status whenever `_set_active_camera` or `_on_time_change` resolves a new pose sample.
- Status must include camera id, model, resolution, frame bracket, alpha, nearest Δt, source gap, overlay model, and warning state.

- [ ] **Step 5: Run camera-switch suite**

Run:

```bash
python -m pytest threedgrut/tests/test_viser_gui_4d_camera_switch.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add threedgrut_playground/viser_gui_4d.py \
        threedgrut/tests/test_viser_gui_4d_camera_switch.py
git commit -m "fix(viser): bind frustum and diagnostics to active camera state"
```

---

### Task 5: Smooth Follow Camera with SE(3) Interpolation

**Files:**
- Modify: `threedgrut_playground/viser_gui_4d.py:1042-1070, 1108-1160`
- Modify: `threedgrut/tests/test_viser_gui_4d_camera_switch.py`

**Interfaces:**
- Consumes interpolated `PoseSample` from Task 1.
- `_set_active_camera(cam_id, t_us)` resolves a fresh interpolated state at every timeline tick.

- [ ] **Step 1: Write continuous-follow RED test**

```python
def test_follow_camera_midpoint_uses_interpolated_pose(viewer):
    viewer._current_dropdown_cam = "moving"
    viewer._follow_camera_enabled = True
    viewer._on_time_change(500_000, source="play")
    np.testing.assert_allclose(viewer.fake_client.camera.position, [5.0, 0.0, 0.0], atol=1e-6)
    assert viewer._active_camera_state.pose_sample.interpolated is True
```

- [ ] **Step 2: Write endpoint and no-follow RED tests**

```python
def test_follow_camera_clamps_before_first_pose(viewer):
    viewer._set_active_camera("moving", -100)
    np.testing.assert_allclose(viewer.fake_client.camera.position, [0.0, 0.0, 0.0])


def test_timeline_change_does_not_move_client_when_follow_camera_off(viewer):
    before = viewer.fake_client.camera.position.copy()
    viewer._follow_camera_enabled = False
    viewer._on_time_change(500_000, source="play")
    np.testing.assert_array_equal(viewer.fake_client.camera.position, before)
```

- [ ] **Step 3: Run follow-camera tests and verify RED**

Run:

```bash
python -m pytest threedgrut/tests/test_viser_gui_4d_camera_switch.py \
  -k 'follow_camera' -v
```

Expected: midpoint test FAIL with nearest-frame position 0 or 10 instead of 5.

- [ ] **Step 4: Route every follow tick through resolver**

In `_on_time_change`:

```python
if self._follow_camera_enabled and self._current_dropdown_cam:
    self._set_active_camera(self._current_dropdown_cam, t_us, snap_clients=True)
```

Do not independently search timestamps in `_snap_clients_to_camera`. Keep camera state/projection/pose synchronized even while playing.

- [ ] **Step 5: Run follow-camera and Follow Ego regression suites**

Run:

```bash
python -m pytest \
  threedgrut/tests/test_viser_gui_4d_camera_switch.py \
  threedgrut/tests/test_viser_gui_4d_follow_ego.py -v
```

Expected: PASS; Follow Ego and Follow Camera remain mutually exclusive.

- [ ] **Step 6: Commit Task 5**

```bash
git add threedgrut_playground/viser_gui_4d.py \
        threedgrut/tests/test_viser_gui_4d_camera_switch.py
git commit -m "fix(viser): interpolate follow-camera poses over timestamp"
```

---

### Task 6: Native Render vs Viewer Projection Parity Tool

**Files:**
- Create: `scripts/validate_viser_render_parity.py`
- Create: `threedgrut/tests/test_validate_viser_render_parity.py`

**Interfaces:**
- Produces `radial_region_masks(height, width, center_radius=0.35, peripheral_inner=0.65, peripheral_outer=0.95)`.
- Produces `compute_region_metrics(reference_rgb, candidate_rgb, masks) -> dict` with MAE and PSNR for full, center, peripheral.
- CLI consumes checkpoint, manifest, config name, camera ids, frame indices/timestamps, output dir, renderer.
- CLI outputs per-camera PNGs, absolute-difference heatmaps, and `parity_metrics.json`.

- [ ] **Step 1: Write radial-mask RED tests**

```python
def test_radial_masks_are_disjoint_and_nonempty():
    masks = radial_region_masks(100, 200)
    assert masks["center"].any()
    assert masks["peripheral"].any()
    assert not np.any(masks["center"] & masks["peripheral"])


def test_center_mask_contains_principal_point():
    masks = radial_region_masks(101, 201)
    assert masks["center"][50, 100]
```

- [ ] **Step 2: Write metric RED test**

```python
def test_region_metrics_detect_peripheral_only_error():
    ref = np.zeros((100, 200, 3), dtype=np.uint8)
    cand = ref.copy()
    masks = radial_region_masks(100, 200)
    cand[masks["peripheral"]] = 32
    m = compute_region_metrics(ref, cand, masks)
    assert m["center_mae"] == 0.0
    assert m["peripheral_mae"] == pytest.approx(32.0)
    assert m["peripheral_psnr"] < m["center_psnr"]
```

Use `float("inf")` for zero-error PSNR.

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
python -m pytest threedgrut/tests/test_validate_viser_render_parity.py -v
```

Expected: collection FAIL because script/functions do not exist.

- [ ] **Step 4: Implement metric helpers and CLI skeleton**

The CLI must:

1. Load the same dataset/config and checkpoint once.
2. For each requested native camera frame, obtain exact camera id, frame END timestamp, c2w, camera model parameters, and GT-sized resolution.
3. Produce `native.png` using the same render path used by standalone eval (`Batch` with native NCore intrinsics/rays and timestamp).
4. Produce `viewer_contract.png` through `Engine3DGRUT.render_pass` using the resolved `CameraRenderState` and exact same c2w/timestamp/resolution.
5. Assert both outputs have identical shape.
6. Save `absdiff_x4.png` and metrics.

CLI arguments:

```text
--checkpoint PATH
--dataset_path PATH
--config_name apps/ncore_3dgut_mcmc_multilayer_inceptio
--camera_id CAM [repeatable]
--frame_index N [repeatable, default 0,mid,last]
--renderer 3dgrt|3dgut
--output_dir PATH
```

`parity_metrics.json` schema:

```json
{
  "camera_front_fisheye": {
    "0": {
      "timestamp_us": 123,
      "model": "FTheta",
      "full_mae": 0.0,
      "center_mae": 0.0,
      "peripheral_mae": 0.0,
      "full_psnr": 99.0,
      "center_psnr": 99.0,
      "peripheral_psnr": 99.0
    }
  }
}
```

Represent infinite PSNR as `99.0` in JSON for stable serialization, while helper tests may retain `inf` internally.

- [ ] **Step 5: Run CPU tests**

Run Step 3 command. Expected: PASS.

- [ ] **Step 6: Add CLI help smoke test**

Run:

```bash
python scripts/validate_viser_render_parity.py --help
```

Expected: exit 0 and all listed arguments present.

- [ ] **Step 7: Commit Task 6**

```bash
git add scripts/validate_viser_render_parity.py \
        threedgrut/tests/test_validate_viser_render_parity.py
git commit -m "feat(viser): add native-camera render parity validator"
```

---

### Task 7: Mac Regression Suite and Static Verification

**Files:**
- Modify only if failures expose regressions in Tasks 1–6.

**Interfaces:** None; this is the merge gate before GPU validation.

- [ ] **Step 1: Run focused viewer/projector suite**

```bash
python -m pytest \
  threedgrut/tests/test_camera_render_state.py \
  threedgrut/tests/test_viser_math.py \
  threedgrut/tests/test_ftheta_intrinsics.py \
  threedgrut/tests/test_ftheta_projector.py \
  threedgrut/tests/test_pinhole_projector.py \
  threedgrut/tests/test_viser_gui_4d_fov.py \
  threedgrut/tests/test_viser_gui_4d_follow_ego.py \
  threedgrut/tests/test_viser_gui_4d_camera_switch.py \
  threedgrut/tests/test_viser_gui_4d_cuboid_overlay.py \
  threedgrut/tests/test_viser_4d_ftheta_overlay_integration.py \
  threedgrut/tests/test_overlay_renderer.py \
  threedgrut/tests/test_validate_viser_render_parity.py -v
```

Expected: all PASS, no collection skips caused by accidental heavy imports.

- [ ] **Step 2: Run full project unit suite**

```bash
python -m pytest threedgrut/tests/ -q
```

Expected: no new failures versus pre-task baseline.

- [ ] **Step 3: Run syntax/import checks**

```bash
python -m py_compile \
  threedgrut_playground/viser_gui_4d.py \
  threedgrut_playground/utils/camera_render_state.py \
  threedgrut_playground/utils/viser_overlay_compositor.py \
  scripts/validate_viser_render_parity.py
```

Expected: exit 0.

- [ ] **Step 4: Review diff for forbidden split state**

Run:

```bash
git diff --check
git diff -- threedgrut_playground/viser_gui_4d.py
```

Reviewer must verify no remaining direct writes to projection fields outside constructor defaults and `_apply_camera_state`:

```text
ftheta_intrinsics =
opencv_pinhole_intrinsics =
opencv_pinhole_rays =
ftheta_render_wh =
opencv_pinhole_render_wh =
_overlay_compositor =
_current_dropdown_cam =
```

Exceptions must be documented and justified.

- [ ] **Step 5: Commit any test-only cleanup**

If no changes are required, do not create an empty commit. Otherwise:

```bash
git add <only-the-cleanup-files>
git commit -m "test(viser): complete mixed-camera regression gate"
```

---

### Task 8: Inceptio C4 GPU Parity and Interactive Visual Acceptance

**Files:**
- Create: `docs/T8_artifacts/mixed_camera/README.md`
- Generate under: `docs/T8_artifacts/mixed_camera/c4_11cam/`
- Modify: `docs/viser_mixed_camera_buglist.md`

**Inputs:**

```text
Host: inceptio
Repo/worktree: ~/repo/3dgrut2-wt/viser-mixed-camera
Conda env: /home/inceptio/miniforge3/envs/3dgrut2
Checkpoint:
/home/inceptio/work/output/c4_11cam_tw2p0_30k/
inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1307_170739/
ours_30000/ckpt_30000.pt
Manifest:
/home/inceptio/work/data/inc_b6a9ed61_20s/
inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9/
inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9.json
Renderer: 3dgrt
```

- [ ] **Step 1: Create isolated branch/worktree and sync submodules**

From Mac:

```bash
git push inceptio <branch>:<branch>
ssh inceptio 'cd ~/repo/3dgrut2 && git worktree add ~/repo/3dgrut2-wt/viser-mixed-camera <branch>'
ssh inceptio 'cd ~/repo/3dgrut2; WT=~/repo/3dgrut2-wt/viser-mixed-camera; for p in $(git config --file .gitmodules --get-regexp path | cut -d" " -f2); do rsync -a ~/repo/3dgrut2/$p/ $WT/$p/; done'
ssh inceptio 'cd ~/repo/3dgrut2-wt/viser-mixed-camera && git log --oneline -1'
```

Expected: remote worktree HEAD equals Mac branch HEAD.

- [ ] **Step 2: Run focused tests on inceptio**

```bash
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH && cd ~/repo/3dgrut2-wt/viser-mixed-camera && python -m pytest threedgrut/tests/test_camera_render_state.py threedgrut/tests/test_viser_gui_4d_camera_switch.py threedgrut/tests/test_viser_gui_4d_cuboid_overlay.py -q'
```

Expected: PASS.

- [ ] **Step 3: Run parity validator**

Use cameras covering both models and wide/narrow behavior:

```bash
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH && export CUDA_VISIBLE_DEVICES=0 && cd ~/repo/3dgrut2-wt/viser-mixed-camera && python scripts/validate_viser_render_parity.py \
  --checkpoint /home/inceptio/work/output/c4_11cam_tw2p0_30k/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1307_170739/ours_30000/ckpt_30000.pt \
  --dataset_path /home/inceptio/work/data/inc_b6a9ed61_20s/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9.json \
  --config_name apps/ncore_3dgut_mcmc_multilayer_inceptio \
  --camera_id camera_front_fisheye \
  --camera_id camera_back_rear_fisheye \
  --camera_id camera_front_wide_120fov \
  --camera_id camera_cross_left_120fov \
  --camera_id camera_front_standard_55fov \
  --camera_id camera_front_tele_30fov \
  --renderer 3dgrt \
  --output_dir /tmp/c4_viser_parity'
```

Acceptance:

- no shape mismatch or unsupported-model fallback;
- every camera reports the expected model;
- native and viewer-contract images use the same radial geometry;
- target numerical gate after confirming deterministic renderer noise:
  - center MAE ≤ 1.0/255 × 255 = 1.0 intensity level;
  - peripheral MAE ≤ 2.0 intensity levels;
  - viewer peripheral error minus center error ≤ 1.5 intensity levels.

If 3DGRT stochastic/progressive behavior violates these strict gates, rerun both paths with `3dgut` as the deterministic diagnostic backend. Do not loosen gates without recording both runs.

- [ ] **Step 4: Launch fixed viewer**

```bash
ssh -n inceptio "export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:\$PATH && export CUDA_VISIBLE_DEVICES=0 && cd /home/inceptio/repo/3dgrut2-wt/viser-mixed-camera && rm -f /tmp/viser_c4_mixed_camera_fix.log && nohup timeout 7200 python threedgrut_playground/viser_gui_4d.py \
  --gs_object /home/inceptio/work/output/c4_11cam_tw2p0_30k/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1307_170739/ours_30000/ckpt_30000.pt \
  --dataset_path /home/inceptio/work/data/inc_b6a9ed61_20s/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9.json \
  --initial_cam_id camera_front_fisheye --port 8090 --renderer 3dgrt \
  > /tmp/viser_c4_mixed_camera_fix.log 2>&1 & echo PID \$!"
```

Verify:

```bash
ssh inceptio 'grep -iE "active camera|model:|listening|Traceback|Error" /tmp/viser_c4_mixed_camera_fix.log | tail -80; ss -ltn | grep :8090'
```

Expected: `*:8090` listening and active camera diagnostic says front fisheye/FTheta/1920×1080.

- [ ] **Step 5: Execute mandatory camera-switch matrix**

In browser, use this exact order:

```text
front_wide
→ front_fisheye
→ front_tele
→ back_rear_fisheye
→ cross_left
→ front_wide
```

At every step record a screenshot containing canvas and camera status. Acceptance per step:

- dropdown camera id equals status camera id;
- model type is correct;
- FTheta views visibly retain fisheye radial geometry, not flattened pinhole geometry;
- front standard/tele FOV differs from front wide and does not inherit fisheye distortion;
- cuboids and trajectories stay attached in center and periphery;
- no duplicate browser-side labels/lines over image-space overlays;
- frustum points in the selected camera direction.

- [ ] **Step 6: Validate smooth Follow Camera**

For front fisheye, rear fisheye, and front wide:

1. enable Follow Camera;
2. play at 1× for at least 10 seconds;
3. inspect the status `left/right frame`, `alpha`, and `source gap`;
4. record one screen capture.

Acceptance:

- no 10 Hz stepwise 1 m snapping;
- motion is continuous between frames;
- large source gaps are reported as warnings;
- no projection model changes during playback;
- overlay remains attached while camera moves.

- [ ] **Step 7: Archive artifacts**

Copy parity output and selected screenshots to:

```text
docs/T8_artifacts/mixed_camera/c4_11cam/
```

Write `README.md` with:

- commit hash;
- ckpt and manifest paths;
- renderer/GPU;
- parity metric table by camera/frame;
- screenshot index;
- each MC bug pass/fail result;
- any residual center/periphery gap classified as viewer or checkpoint.

- [ ] **Step 8: Update buglist statuses**

In `docs/viser_mixed_camera_buglist.md`:

- MC-1 through MC-9 and MC-11 may become ✅ only with tests plus C4 visual evidence;
- MC-10 becomes either ✅ viewer parity restored, residual attributed to checkpoint, or remains 🟡 with explicit blocker;
- update “mixed-camera viser not trustworthy” warning only when all P0 bugs pass.

- [ ] **Step 9: Commit validation evidence**

```bash
git add docs/T8_artifacts/mixed_camera docs/viser_mixed_camera_buglist.md
git commit -m "test(viser): validate mixed-camera projection on C4 11-cam"
```

---

### Task 9: Documentation, Architecture Invariant, and Skill Repair

**Files:**
- Modify: `.claude/skills/viser-gui-4d/SKILL.md`
- Modify: `v2_architecture.md`
- Modify: `v5_plan.md` if this viewer work is tracked under C4 evidence
- Modify: `docs/viser_mixed_camera_buglist.md`

**Interfaces:** Documentation only; consumes Task 8 evidence.

- [ ] **Step 1: Update skill with the fixed workflow**

Replace the temporary warning with these durable rules:

- mixed-camera dropdown applies one atomic camera state;
- Camera status must be inspected before visual judgment;
- FTheta and OpenCVPinhole both use calibrated image-space overlays;
- Follow Camera is interpolated and displays source gaps;
- parity validator command for projection disputes;
- native `render.py` remains the final quality reference, while viewer is validated for parity.

Do not remove historical gotchas that remain valid.

- [ ] **Step 2: Add architecture invariant**

Add to `v2_architecture.md` §7:

```text
Viewer camera-state invariant: selected camera id, c2w pose sample, projection model,
per-pixel rays/intrinsics, render resolution/FOV, and overlay projector are resolved
from one CameraRenderState and applied atomically. FTheta/OpenCVPinhole fields are
mutually exclusive; calibrated overlays never use browser ideal-pinhole projection.
```

Use full-width Chinese parentheses inside Mermaid labels if a graph is changed.

- [ ] **Step 3: Add Done Log evidence if required**

If C4 evidence tracking belongs in `v5_plan.md`, append the viewer fix separately from the C4 model result. Do not rewrite C4 metrics or declare model promotion. Record:

- fix commits;
- tests count;
- parity numbers;
- camera-switch matrix result;
- residual model-quality findings after viewer parity.

- [ ] **Step 4: Verify docs**

Run:

```bash
python - <<'PY'
from pathlib import Path
for p in [
    Path('docs/viser_mixed_camera_buglist.md'),
    Path('docs/superpowers/plans/2026-07-14-viser-mixed-camera-bugfix.md'),
    Path('.claude/skills/viser-gui-4d/SKILL.md'),
]:
    assert p.exists() and p.stat().st_size > 1000, p
print('docs ok')
PY

git diff --check
```

If `v5_plan.md` Mermaid changes, run:

```bash
awk '/```mermaid/{i=1;next} /```/&&i{i=0;next} i&&/\(/{print FILENAME":"NR": "$0}' v5_plan.md
```

Expected: no newly introduced invalid half-width parentheses in Mermaid blocks.

- [ ] **Step 5: Commit documentation**

```bash
git add .claude/skills/viser-gui-4d/SKILL.md \
        v2_architecture.md \
        docs/viser_mixed_camera_buglist.md \
        v5_plan.md
git commit -m "docs(viser): close mixed-camera projection buglist"
```

Omit `v5_plan.md` from `git add` when Step 3 determined it was not required.

---

## Final Completion Gate

The implementation is complete only when all conditions hold:

- [x] Camera dropdown, active state, client pose, engine projection, resolution/FOV, and overlay projector identify the same camera after every switch.
- [x] FTheta → OpenCVPinhole clears all FTheta state.
- [x] OpenCVPinhole → FTheta creates a compositor even when initial compositor was `None`.
- [x] OpenCVPinhole camera switches update rational intrinsics and per-pixel rays.
- [x] FTheta and OpenCVPinhole cuboids/trajectories use calibrated image-space projection.
- [x] Ego frustum follows the selected camera, not startup camera.
- [x] Follow Camera uses translation lerp + quaternion SLERP and reports large source gaps.
- [ ] Native/viewer parity metrics distinguish viewer error from checkpoint peripheral blur. Comparator is implemented/tested, but exact UI-free viewer PNG dump is still missing; no fabricated browser-screenshot PSNR is accepted.
- [x] Focused and full Mac suites pass（latest focused `55 passed`; full after rig-origin fix `1008 passed, 2 skipped`）.
- [x] C4 inceptio switch matrix and playback pass; evidence is archived in `docs/T8_artifacts/C4_mixed_camera_viewer_fix_validation.md`.
- [x] Buglist, architecture invariant, skill, and validation report are updated.

## Self-Review Record

- **Spec coverage:** All user observations map to MC-1 through MC-10; missing mixed-transition coverage is MC-11. Tasks 1–5 fix state, overlay, frustum, and motion; Task 6 isolates center/periphery quality; Task 8 validates the exact C4 checkpoint.
- **Scope decomposition:** State resolution, overlay projection, pose interpolation, parity diagnosis, and GPU validation are independently reviewable deliverables. No training/model changes are included.
- **Type consistency:** `CameraRenderState` and `PoseSample` are defined in Task 1 and consumed unchanged by Tasks 2–6. Overlay compositor accepts a projector object consistently in Task 3 onward.
- **Placeholder scan:** No TBD/TODO/fill-later steps. GPU paths, camera ids, commands, expected outcomes, and artifact locations are explicit.
- **Risk control:** C4 promotion remains out of scope; viewer correctness is restored before model-quality interpretation resumes.
