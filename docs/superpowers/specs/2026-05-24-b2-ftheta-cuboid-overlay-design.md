# B2 — FTheta Cuboid Overlay (Design Spec)

**Date:** 2026-05-24
**Status:** Approved (pending user spec review)
**Parent bug:** B2 in [T8_buglists.md](../../../T8_buglists.md) — cuboid wireframe vs Gaussian backdrop projection mismatch
**Related code version:** main @ `46e643f` (B6/B7/B8 fixes) + `b1752b3` (FTheta uint64 fix)
**Estimated effort:** ~1 day (7-11 h, see §5.5)
**Target PR:** `fix(B2): FTheta cuboid overlay replacement for 4D viz` — independent PR, no overlap with B3

---

## §0 Problem Statement

In `threedgrut_playground/viser_gui_4d.py` 4D viewer:
- **Gaussian backdrop** is rendered by the 3dgut engine using the **FTheta polynomial (fisheye) projection** — trained intrinsics, ~140° FOV with barrel distortion.
- **Cuboid wireframes** (and frustum, track polylines, ego trajectory) are drawn by viser's `add_line_segments`, which performs **pinhole projection** in the browser.

The two projections disagree for the same 3D point. Visually, the cuboid green wireframe drifts off the truck/car it should bound, and the drift grows toward the screen edge where fisheye distortion is largest.

**Fix direction:** Replace viser line_segments path (for cuboid/frustum/track/ego trajectory) with a Python-side overlay image that is forward-projected through the same FTheta polynomial used by the backdrop, then alpha-blended into the backdrop image before `client.scene.set_background_image()`.

---

## §1 Module Decomposition

| Module | Path | Responsibility | Dependencies |
|---|---|---|---|
| **NEW** `FthetaForwardProjector` | `threedgrut_playground/utils/ftheta_projector.py` | 3D world point → FTheta 2D pixel + visibility mask. Pure numpy, no viser/torch dependency. | ckpt `ftheta_dict` 8 keys |
| **NEW** `OverlayRenderer` | `threedgrut_playground/utils/overlay_renderer.py` | Receives 2D polylines + style → outputs `(H, W, 4)` uint8 RGBA buffer. PIL.ImageDraw impl. | PIL (existing dep) |
| **NEW** `Viser4DOverlayCompositor` | `threedgrut_playground/utils/viser_overlay_compositor.py` | Orchestrates: cuboid/frustum/track/ego trajectory edges → projector → renderer → alpha blend into backdrop. | projector + renderer |
| **MODIFY** `viser_gui_4d.py` | same | (1) FTheta mode: invoke compositor before `set_background_image`; (2) FTheta mode: skip `add_line_segments` calls for cuboid/frustum/track/ego trajectory; (3) GUI add "FTheta overlay (debug)" checkbox (default ON). Pinhole mode unchanged. | compositor |
| **NEW** Calibration probe | `scripts/probe_ftheta_overlay.py` | Phase 0 dev tool — runs 4 candidate (FLIP × poly order × linear_cde) combos against a known cuboid vertex; user picks the one that lands correctly in the browser. Output: a log file pinning the winning combo. | projector |
| **NEW** Unit tests | `threedgrut/tests/test_ftheta_projector.py`, `test_overlay_renderer.py`, `test_viser_4d_ftheta_overlay_integration.py` | See §5.1 | projector / renderer / compositor |

**Isolation principle:** `Viser4DOverlayCompositor` does not know viser exists — it takes `(backdrop_ndarray, polylines_dict, c2w, ftheta_dict)` and returns `blended_ndarray`. This lets the entire overlay path be unit-tested without a running viser server.

---

## §2 Data Flow and Coordinate Systems

### 2.1 Per-client per-frame flow

```
[time slider t_us, client.camera]
   │
   ├─ engine.render_pass(client.camera, t_us)
   │     └─ Gaussian backdrop ndarray (H, W, 3) uint8  ──┐
   │                                                       │
   ├─ _collect_overlay_polylines(t_us, client.camera.c2w) │
   │     ├─ cuboid edges:    list[(M_i, 3)] world coords  │
   │     ├─ frustum edges:   list[(M_i, 3)] world coords  │
   │     ├─ track polylines: list[(M_i, 3)] world coords  │
   │     └─ ego trajectory:  list[(M_i, 3)] world coords  │
   │                                                       │
   ├─ FthetaForwardProjector(ftheta_dict).project_polylines(
   │      polylines_world, c2w_opengl=client.camera.c2w, subdivide_n=20)
   │     └─ dict[layer -> list[(uv: (M', 2), visible: (M',) bool)]]
   │                                                       │
   ├─ OverlayRenderer(H, W).render(projected_polylines)   │
   │     └─ RGBA ndarray (H, W, 4) uint8 ─────────────────┤
   │                                                       │
   └─ Viser4DOverlayCompositor.alpha_blend(backdrop, overlay) ──> blended (H, W, 3)
                                                                  ↓
                                            client.scene.set_background_image(blended)
```

### 2.2 Coordinate system convention

| Stage | Frame | Notes |
|---|---|---|
| edges collection | **world** (matches ckpt) | Reuse existing `_build_cuboid_edges` / `_build_frustum_edges` / `_build_track_lines` / `_build_ego_trajectory` geometry verbatim |
| projector entry | **world** + `c2w_opengl` (4×4) | viser client.camera is **OpenGL convention** (+Z backward, +Y up, +X right) |
| projector internal | **camera (OpenCV)** | NCore FTheta training data is **OpenCV convention** (+Z forward, +Y down, +X right). Projector applies axis flip on entry (see §3 Step 0). |
| projector exit | **pixel** (u, v) ∈ [0, W) × [0, H) | + per-vertex `visible: bool` |
| renderer | pixel | PIL.ImageDraw on H×W RGBA buffer |

### 2.3 `c2w` source

In FTheta mode, `client.camera` is already snapped to the ego primary camera pose by `_snap_clients_to_ego` (B1 fix, commit 209886c). So `client.camera.c2w` ≡ ego camera pose, which is what the projector needs.

### 2.4 Edge subdivision

In FTheta fisheye, a 3D straight edge projects to a **curve** in image space. Connecting only the two endpoints with a straight 2D line gives visibly wrong tangents (especially near the screen edge).

**Solution:** Subdivide each edge into `N=20` piecewise-linear segments before projection. After projection, the renderer draws each adjacent (visible, visible) vertex pair.

**N=20 is hardcoded** (not exposed as a tunable). Re-evaluate after visual iteration.

---

## §3 FTheta Forward Projection Math

### 3.1 Algorithm

**Inputs:**
- `points_world: (N, 3) float64`
- `c2w_opengl: (4, 4) float64` — from `client.camera.c2w`
- `ftheta_dict`: 8 keys (`resolution`, `principal_point`, `pixeldist_to_angle_poly`, `angle_to_pixeldist_poly`, `max_angle`, `linear_cde`, `shutter_type`, `reference_poly`)

**Steps:**

```python
# Step 0: OpenGL → OpenCV axis flip
# Exact FLIP matrix pinned by Phase 0 calibration probe (see §5.2).
# Candidates: diag([1, -1, -1, 1]) (likely), diag([1, 1, -1, 1]), or left-mul variant.
c2w_cv = c2w_opengl @ FLIP                       # or FLIP @ c2w_opengl

# Step 1: world → camera (OpenCV)
w2c = np.linalg.inv(c2w_cv)
p_h = np.concatenate([points_world, np.ones((N, 1))], axis=-1)   # (N, 4)
p_cam = (w2c @ p_h.T).T[:, :3]                                    # (N, 3)
x, y, z = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]

# Step 2: ray angle from optical axis (+Z in OpenCV)
r_xy = np.sqrt(x ** 2 + y ** 2)
angle = np.arctan2(r_xy, z)                                       # (N,) ∈ [0, π]

# Step 3: max_angle clip
in_fov = angle <= ftheta_dict['max_angle']

# Step 4: angle → radial pixel distance (forward polynomial)
# Coefficient order (ascending c0 + c1·θ + c2·θ² + ... vs descending)
# pinned by Phase 0 calibration probe.
poly = ftheta_dict['angle_to_pixeldist_poly']
r_pix = np.polyval(poly[::-1], angle)  # or np.polyval(poly, angle) — Phase 0 decides

# Step 5: restore (u, v)
cx, cy = ftheta_dict['principal_point']
safe_r = np.where(r_xy < 1e-9, 1.0, r_xy)
u_offset = np.where(r_xy < 1e-9, 0.0, r_pix * x / safe_r)
v_offset = np.where(r_xy < 1e-9, 0.0, r_pix * y / safe_r)

# Step 6: linear_cde affine correction
# Exact form pinned by Phase 0 (must roundtrip-match the inverse in
# ftheta_intrinsics.py:ftheta_pixels_to_camera_rays).
c, d, e = ftheta_dict['linear_cde']
u_corr = c * u_offset + d * v_offset
v_corr = e * u_offset +     v_offset             # placeholder; Phase 0 pins

u = cx + u_corr
v = cy + v_corr

# Step 7: image bound + behind-camera clip
W, H = ftheta_dict['resolution']
in_bound = (u >= 0) & (u < W) & (v >= 0) & (v < H)
visible = in_fov & in_bound & (z > 0)

return np.stack([u, v], axis=-1), visible
```

### 3.2 Public API

```python
class FthetaForwardProjector:
    def __init__(self, ftheta_dict: dict): ...

    def project_points(
        self,
        points_world: np.ndarray,         # (N, 3) float64
        c2w_opengl: np.ndarray,           # (4, 4) float64
    ) -> tuple[np.ndarray, np.ndarray]:   # (uv: (N, 2), visible: (N,) bool)
        ...

    def project_polylines(
        self,
        polylines_world: list[np.ndarray],  # list of (M_i, 3)
        c2w_opengl: np.ndarray,
        subdivide_n: int = 20,
    ) -> list[tuple[np.ndarray, np.ndarray]]:  # list of (uv: (M_i', 2), visible: (M_i',))
        """Piecewise-linear subdivide each polyline, then project."""
        ...
```

### 3.3 Phase 0 unknowns (pinned by ThinkPad probe before any other implementation)

1. `FLIP` exact matrix and multiplication side
2. `angle_to_pixeldist_poly` coefficient order (ascending vs descending)
3. `linear_cde` exact algebraic form

**Phase 0 is a hard gate** — calibration anchor unit test (`test_calibration_anchor`) cannot be written until Phase 0 outputs land. PR cannot merge without it.

---

## §4 Overlay Composition + viser Integration

### 4.1 Overlay layer style

| Primitive | RGBA color | Width (px) | Z-order (draw order, last = top) | Source |
|---|---|---|---|---|
| ego trajectory | (255, 200, 0, 220) yellow | 2 | 1 (bottom) | `_build_ego_trajectory` |
| track polylines | (180, 180, 180, 180) gray | 1 | 2 | `_build_track_lines` |
| frustum | (0, 220, 220, 220) cyan | 1 | 3 | `_build_frustum_edges` |
| cuboid edges (active) | (0, 255, 0, 255) green | 2 | 4 (top) | `_build_cuboid_edges` |

Initial values mirror current viser line_segments style (if any); iterate during visual verification.

### 4.2 Alpha blend formula

```python
def alpha_blend(backdrop: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """backdrop: (H, W, 3) uint8;  overlay: (H, W, 4) uint8 -> (H, W, 3) uint8"""
    a = overlay[..., 3:4].astype(np.float32) / 255.0
    blended = overlay[..., :3].astype(np.float32) * a + \
              backdrop.astype(np.float32) * (1.0 - a)
    return blended.astype(np.uint8)
```

Vectorized numpy; ~3 ms for a 1920×1080 frame.

### 4.3 `viser_gui_4d.py` changes

**4.3.1 `update()` body (around current set_background_image call, ~line 808):**

```python
# existing: engine.render_pass(...) → backdrop_rgb (H, W, 3) uint8

if self.ftheta_render_wh is not None and self.show_ftheta_overlay.value:
    polylines = self._collect_overlay_polylines(t_us, client.camera.c2w)
    overlay_rgba = self._overlay_compositor.render(
        polylines, h=H, w=W, ftheta_dict=self._ftheta_dict, c2w=client.camera.c2w
    )
    backdrop_rgb = alpha_blend(backdrop_rgb, overlay_rgba)

client.scene.set_background_image(backdrop_rgb)
```

**4.3.2 `add_line_segments` guards (FTheta mode skips them):**

For each of: `_update_active_cuboids`, frustum updater, track polyline updater, ego trajectory updater — wrap the `server.scene.add_line_segments(...)` call:

```python
if self.ftheta_render_wh is None:        # Pinhole mode only
    self.h_xxx = self.server.scene.add_line_segments(...)
```

In FTheta mode the wireframes live entirely in the overlay image; no viser scene primitives are injected.

**4.3.3 GUI:**

In `_build_visibility_gui`, append:

```python
self.show_ftheta_overlay = self.server.gui.add_checkbox(
    "FTheta overlay (debug)",
    initial_value=True,
)
self.show_ftheta_overlay.on_update(lambda _: self._on_time_change(self._current_t_us, source="overlay_toggle"))
```

Turning it off in FTheta mode falls back to no overlay (raw Gaussian backdrop with no wireframes). For "compare against pinhole line_segments" comparison, user switches to Pinhole mode (existing toggle).

### 4.4 Performance budget (ThinkPad RTX 4090, target 10 fps)

| Step | Budget |
|---|---|
| `_collect_overlay_polylines` (CPU dict construction) | < 1 ms |
| Polyline subdivision (N=20) | ~1 ms |
| `project_points` (numpy vectorized) | ~2 ms |
| PIL.ImageDraw (~16 800 segments) | ~5-8 ms |
| Alpha blend (8 MB image) | ~3 ms |
| **Total per client per frame** | **~12-15 ms** |

10 fps = 100 ms/frame budget → 6× headroom. Fallback to cv2 line drawing if PIL is unexpectedly slow.

### 4.5 Edge cases

- **`ftheta_dict is None`** → `self.ftheta_render_wh is None` → overlay path auto-skips → pinhole `add_line_segments` runs → equivalent to pre-B2 behavior. Zero regression for non-FTheta ckpts.
- **`show_ftheta_overlay = False` in FTheta mode** → backdrop renders without wireframes (debug mode). Documented in GUI label.
- **Empty polylines** (no active cuboids at given t_us) → overlay buffer stays fully transparent → alpha blend is identity.

---

## §5 Testing + Acceptance

### 5.1 Mac unit tests (no GPU/viser dependency)

**`threedgrut/tests/test_ftheta_projector.py`:**

| Test | Input | Pass criterion |
|---|---|---|
| `test_project_principal_point` | world point on optical axis, z > 0 | (u, v) ≈ (cx, cy), error < 0.1 px |
| `test_project_unproject_roundtrip` | 100 sampled pixels → `ftheta_pixels_to_camera_rays` (inverse) → forward project | error < 0.5 px (center) / < 2 px (edge) |
| `test_max_angle_clip` | ray angle = max_angle + 0.01 | `visible == False` |
| `test_image_bound_clip` | u or v outside (0, W) × (0, H) | `visible == False` |
| `test_behind_camera_clip` | z < 0 | `visible == False` |
| `test_polyline_subdivision` | 2-point polyline, subdivide_n=10 | 11 vertices, endpoints preserved |
| `test_calibration_anchor` | Phase 0 pinned (FLIP, poly order, linear_cde) + known ego pose + known cuboid vertex → known pixel | error < 5 px (locks the Phase 0 decisions) |

**`threedgrut/tests/test_overlay_renderer.py`:**
- `test_render_single_segment` — draw one polyline, assert nonzero alpha along expected pixels
- `test_alpha_blend_against_known_image` — blend known overlay over known backdrop, byte-exact match
- `test_invisible_vertex_skipped` — polyline where some vertices have `visible=False`, those segments are not drawn

**`threedgrut/tests/test_viser_4d_ftheta_overlay_integration.py`:**
- mock `engine.render_pass` returns flat-gray backdrop; mock ckpt has `ftheta_dict`; call `_overlay_compositor.render()` → assert output RGB differs from input (overlay landed)
- Pinhole-mode mock → assert `_collect_overlay_polylines` never invoked

### 5.2 Phase 0 ThinkPad calibration probe

**Script:** `scripts/probe_ftheta_overlay.py`

**Procedure:**
1. Load ckpt `ckpt_with_ftheta_v3.pt`, extract `ftheta_dict` + first active cuboid at t=0 + ego pose at t=0.
2. For each combo of (FLIP in {diag([1,-1,-1,1]), diag([1,1,-1,1])}) × (poly order in {asc, desc}) × (linear_cde formula in {A, B}):
   - Project cuboid 8 vertices to pixels.
   - Render overlay with that combo + capture screenshot.
3. User opens 8 screenshots in browser, picks the combo where the green wireframe overlaps the truck.
4. Log winning combo to `docs/T8_artifacts/B2_calibration_probe_log.txt`.
5. Hardcode winning combo into `ftheta_projector.py` + `test_calibration_anchor`.

**Phase 0 is a hard gate.** If no combo lands within ~10 px of the truck, the projection algorithm itself is wrong (re-examine §3 against C++ binding source), not just the calibration constants.

### 5.3 ThinkPad visual verification (manual + screenshots)

| Step | Expected |
|---|---|
| 1. Launch viser with `ckpt_with_ftheta_v3.pt`, port 8090, target_fps 10 | Log shows "FTheta mode active" |
| 2. SSH tunnel + browser http://localhost:8090 | Image visible, green cuboid wireframe overlaps trucks/cars |
| 3. Save `docs/T8_artifacts/B2_ftheta_overlay_on.png` | Archive |
| 4. Toggle off "FTheta overlay (debug)" | Cuboids disappear (backdrop only) |
| 5. Save `docs/T8_artifacts/B2_ftheta_overlay_off.png` | Archive comparison |
| 6. Play slider to 5s / 10s / 15s / 19s | Cuboid stays aligned across all timestamps |
| 7. Inspect screen edges (distant cars near corners) | Overlay edges visibly curved (subdivision working) |

**Acceptance criterion (subjective):** Cuboid edges visually trace the truck/car outline; ≤ ~5 px offset near screen center, ≤ ~10 px near edges. Judged by user in browser.

### 5.4 Artifact paths

```
docs/T8_artifacts/
  B2_ftheta_overlay_on.png         # screenshot with overlay on
  B2_ftheta_overlay_off.png        # screenshot with overlay off (control)
  B2_play_5s.png                   # play at 5s
  B2_play_10s.png                  # play at 10s
  B2_play_15s.png                  # play at 15s
  B2_play_19s.png                  # play at 19s
  B2_calibration_probe_log.txt     # Phase 0 winning combo + per-combo screenshots index
```

### 5.5 PR scope + Done Log entry

**PR title:** `fix(B2): FTheta cuboid overlay replacement for 4D viz`

**New files:**
- `threedgrut_playground/utils/ftheta_projector.py`
- `threedgrut_playground/utils/overlay_renderer.py`
- `threedgrut_playground/utils/viser_overlay_compositor.py`
- `scripts/probe_ftheta_overlay.py`
- `threedgrut/tests/test_ftheta_projector.py`
- `threedgrut/tests/test_overlay_renderer.py`
- `threedgrut/tests/test_viser_4d_ftheta_overlay_integration.py`

**Modified files:**
- `threedgrut_playground/viser_gui_4d.py` — 4× `add_line_segments` guards, `update()` overlay invocation, GUI checkbox

**Done Log entry (append to v2_plan.md § 5):**

```
2026-05-DD <commit_hash> fix(B2): FTheta cuboid overlay
- Phase 0 probe pinned: FLIP = <matrix>, poly order = <asc|desc>, linear_cde form = <A|B>
- Mac pytest: N PASS (7 projector + 3 renderer + 1 viser integration)
- ThinkPad visual verification: docs/T8_artifacts/B2_*.png (5 screenshots)
- viser_gui_4d.py FTheta mode now renders cuboid/frustum/track/ego trajectory via overlay
- Pinhole mode behavior unchanged (zero regression for non-FTheta ckpts)
- T8_buglists.md B2 → ✅
```

### 5.6 Effort estimate

| Phase | Hours |
|---|---|
| Phase 0 calibration probe (ThinkPad experiment) | 1-2 |
| `FthetaForwardProjector` + unit tests | 2-3 |
| `OverlayRenderer` + `Viser4DOverlayCompositor` + unit tests | 1-2 |
| `viser_gui_4d.py` integration + GUI checkbox | 1 |
| ThinkPad visual verification + screenshots + iteration | 1-2 |
| Doc sync (v2_plan.md, T8_buglists.md, v2_architecture.md) + commit + PR | 0.5 |
| **Total** | **~7-11 h (1 day)** |

Original T8_buglists.md estimate "~0.5d" is exceeded because:
- (a) scope expanded from "cuboid only" to "cuboid + frustum + track + ego trajectory" (user-approved per §1 scope question),
- (b) Phase 0 calibration probe added (avoids guess-and-rerun on ThinkPad),
- (c) unit test scaffolding added (avoids math regression in future).

---

## §6 Out of Scope (deferred or in other PRs)

- **B3** (dynamic Gaussian distribution diagnosis) — separate brainstorming → spec → plan cycle.
- **Pinhole mode overlay path** — Pinhole mode keeps current `add_line_segments` (per §4 mode-coverage decision). Migrating Pinhole to Python overlay is a hypothetical future PR.
- **Baseline screenshot regression test** — deferred until visual design stabilizes (per §5 testing strategy A).
- **Performance optimization** — current ~12-15 ms / client / frame leaves 6× headroom at 10 fps; optimize only if multi-client (>3) drops below target_fps.

---

## §7 Open Questions

None. All scope decisions resolved during brainstorming. Phase 0 calibration probe outputs are bounded unknowns to be resolved during implementation, not design.
