# Viser OpenCV Rational Overlay Projector Bugfix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Viser cuboid, trajectory, frustum, and label overlays for NCore `OpenCVPinholeCameraModel` cameras use the same OpenCV rational forward projection and validity domain as NCore SDK and the 3DGUT CUDA kernel.

**Architecture:** Keep the existing `PinholeForwardProjector` public interface and replace only its distortion implementation. The projector will evaluate the six radial coefficients as a rational numerator/denominator, include tangential and thin-prism terms, and apply the same `0.8 < icD < 1.2` trust gate used by NCore and `cameraProjections.cuh`. Pure NumPy tests pin the formula on Mac; an inceptio probe compares representative real-camera rays against NCore SDK output.

**Tech Stack:** Python, NumPy, pytest, NCore SDK on inceptio, 3DGUT CUDA camera projection reference.

## Global Constraints

- This is an **overlay-only** task. Do not modify `datasetNcore.py`, photometric masks, training losses, camera parameters, checkpoints, or the 9-camera standard configuration.
- Follow TDD strictly: write each regression test, run it, and observe the expected failure before changing production code.
- Preserve the existing public API:
  - `PinholeForwardProjector(pinhole_dict, world_to_camera_flip=None)`
  - `project_points(points_world, c2w) -> (uv, visible)`
  - `project_polylines(polylines_world, c2w, subdivide_n=4)`
- Preserve ideal-pinhole behavior and existing one-coefficient test behavior by padding missing radial coefficients with zeros to six entries.
- Match the reference formula in:
  - NCore `OpenCVPinholeCameraModel.__compute_distortion()`;
  - `threedgut_tracer/include/3dgut/kernels/cuda/sensors/cameraProjections.cuh:72-118`.
- The six radial coefficients mean `(k1,k2,k3,k4,k5,k6)` where the first three form the numerator and the last three form the denominator. They are **not** a six-term polynomial.
- Apply tangential and thin-prism distortion exactly as NCore/3DGUT do.
- A point is visible only when all are true: camera-frame `z>0`, `0.8 < icD < 1.2`, and the projected point is inside image bounds.
- For radial-invalid points, return the same directional clipping coordinate shape as the reference implementation when practical, but `visible=False` is the required semantic contract.
- Do not change FTheta projector behavior.
- Do not close MC-6 until Mac tests and real NCore SDK parity both pass.
- Commit the implementation and evidence on branch `fix/viser-opencv-rational-overlay`; do not merge to main.

---

## File Structure

### Modified files

- `threedgrut_playground/utils/pinhole_projector.py` — implement OpenCV rational radial model, thin-prism terms, and radial trust gate.
- `threedgrut/tests/test_pinhole_projector.py` — add RED tests for rational denominator, thin-prism, validity gate, short radial arrays, and regression of ideal/tangential paths.
- `docs/viser_mixed_camera_buglist.md` — update MC-6 only after all validation gates pass.
- `.claude/skills/viser-gui-4d/SKILL.md` — record that OpenCVPinhole overlays must use the rational formula and SDK parity, if the file is tracked and editable in the worktree.

### New files

- `scripts/validate_pinhole_projector_ncore_parity.py` — inceptio-only validation tool that loads the b6a9 manifest, compares the NumPy projector against NCore SDK forward projection for representative valid rays, and verifies invalid peripheral rays are rejected consistently.
- `docs/T8_artifacts/C4_opencv_rational_overlay_validation.md` — commands, camera set, parity numbers, tests, and limitations.

---

### Task 1: Pin the OpenCV Rational Formula with RED Tests

**Files:**
- Modify: `threedgrut/tests/test_pinhole_projector.py`

**Interfaces:**
- Consumes: existing `PinholeForwardProjector.project_points()`.
- Produces: regression tests that distinguish rational numerator/denominator from the current incorrect six-term polynomial.

- [ ] **Step 1: Add a six-coefficient rational RED test**

Add a helper with a large image so visibility does not hide numeric projection differences:

```python
def _rational_pinhole_dict():
    return {
        "resolution": np.array([4000, 3000], dtype=np.int64),
        "principal_point": np.array([2000.0, 1500.0]),
        "focal_length": np.array([1000.0, 1000.0]),
        "radial_coeffs": np.array([0.10, 0.02, 0.003, 0.04, 0.005, 0.0006]),
        "tangential_coeffs": np.zeros(2),
        "thin_prism_coeffs": np.zeros(4),
    }
```

Add:

```python
def test_six_radial_coefficients_use_rational_denominator():
    cfg = _rational_pinhole_dict()
    proj = PinholeForwardProjector(cfg)
    point = np.array([[0.5, 0.25, 1.0]])

    uv, visible = proj.project_points(point, np.eye(4))

    x, y = 0.5, 0.25
    r2 = x*x + y*y
    k1, k2, k3, k4, k5, k6 = cfg["radial_coeffs"]
    numerator = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    denominator = 1.0 + r2 * (k4 + r2 * (k5 + r2 * k6))
    scale = numerator / denominator
    expected = np.array([[2000.0 + 1000.0*x*scale,
                          1500.0 + 1000.0*y*scale]])
    np.testing.assert_allclose(uv, expected, atol=1e-9)
    assert visible[0]
```

- [ ] **Step 2: Run the rational test and verify RED**

Run:

```bash
/Users/etendue/repo/3dgrut2/.venv/bin/python -m pytest \
  threedgrut/tests/test_pinhole_projector.py::test_six_radial_coefficients_use_rational_denominator -v
```

Expected: assertion failure because the current implementation treats coefficients 4–6 as higher polynomial powers instead of denominator coefficients.

- [ ] **Step 3: Add a denominator-near-boundary regression test**

Use coefficients that keep `icD` inside `(0.8,1.2)` and assert the exact expected value. This prevents an implementation that merely ignores coefficients 4–6 from passing.

- [ ] **Step 4: Run both tests and preserve the RED output in the validation notes**

Expected: both fail for projection-value mismatch, not import or syntax errors.

---

### Task 2: Pin Tangential, Thin-Prism, and Validity Semantics

**Files:**
- Modify: `threedgrut/tests/test_pinhole_projector.py`

**Interfaces:**
- Produces tests matching NCore/3DGUT distortion deltas and validity gate.

- [ ] **Step 1: Add a thin-prism RED test**

```python
def test_thin_prism_coefficients_match_opencv_model():
    cfg = _identity_pinhole_dict()
    cfg["thin_prism_coeffs"] = np.array([0.01, -0.002, -0.015, 0.003])
    proj = PinholeForwardProjector(cfg)
    point = np.array([[1.0, 0.5, 5.0]])

    uv, visible = proj.project_points(point, np.eye(4))

    x, y = 0.2, 0.1
    r2 = x*x + y*y
    dx = r2 * (0.01 + r2 * -0.002)
    dy = r2 * (-0.015 + r2 * 0.003)
    expected = np.array([[320.0 + 500.0*(x + dx),
                          240.0 + 500.0*(y + dy)]])
    np.testing.assert_allclose(uv, expected, atol=1e-9)
    assert visible[0]
```

Run and verify RED because current code ignores thin prism.

- [ ] **Step 2: Add radial trust-gate RED tests**

Construct one config/point with `icD == 1.3` and another with `icD == 0.7`. Assert `visible=False` even if the resulting coordinate would otherwise lie inside the image.

Example high-scale case:

```python
def test_radial_scale_outside_ncore_trust_interval_is_invalid():
    cfg = _identity_pinhole_dict()
    cfg["resolution"] = np.array([4000, 3000])
    cfg["principal_point"] = np.array([2000.0, 1500.0])
    cfg["radial_coeffs"] = np.array([0.3, 0, 0, 0, 0, 0])
    proj = PinholeForwardProjector(cfg)
    _, visible = proj.project_points(np.array([[1.0, 0.0, 1.0]]), np.eye(4))
    assert not visible[0]
```

- [ ] **Step 3: Add short-array compatibility tests**

Assert:

- zero coefficients or missing key → ideal pinhole;
- `[k1]` behaves as numerator `k1`, denominator zeros;
- arrays with more than six coefficients raise `ValueError` rather than silently changing meaning;
- tangential arrays shorter than two are treated as zeros or rejected consistently with the chosen existing API policy.

- [ ] **Step 4: Run the new tests and verify RED**

Expected failures:

- thin-prism expected coordinate mismatch;
- radial-invalid point currently marked visible;
- any newly specified validation behavior not implemented.

---

### Task 3: Implement the Minimal Rational Projector Fix

**Files:**
- Modify: `threedgrut_playground/utils/pinhole_projector.py`

**Interfaces:**
- Preserves current constructor and methods.
- Adds internal normalized coefficient arrays of fixed sizes 6, 2, and 4.

- [ ] **Step 1: Correct the module documentation**

Replace the incorrect six-term polynomial description with:

```text
icD_num = 1 + k1*r² + k2*r⁴ + k3*r⁶
icD_den = 1 + k4*r² + k5*r⁴ + k6*r⁶
icD = icD_num / icD_den
```

Document tangential, thin-prism, and `0.8 < icD < 1.2` semantics.

- [ ] **Step 2: Normalize coefficient inputs**

Implement a private helper such as:

```python
def _pad_coefficients(values, size, name):
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size > size:
        raise ValueError(f"{name} supports at most {size} coefficients, got {arr.size}")
    return np.pad(arr, (0, size-arr.size))
```

Use it for radial 6, tangential 2, thin-prism 4.

- [ ] **Step 3: Implement the reference distortion formula**

Inside `project_points()` compute:

```python
r2 = x_n*x_n + y_n*y_n
r4 = r2*r2
r6 = r4*r2
num = 1 + k1*r2 + k2*r4 + k3*r6
den = 1 + k4*r2 + k5*r4 + k6*r6
icD = num / den

a1 = 2*x_n*y_n
a2 = r2 + 2*x_n*x_n
a3 = r2 + 2*y_n*y_n

delta_x = p1*a1 + p2*a2 + r2*(s1 + r2*s2)
delta_y = p1*a3 + p2*a1 + r2*(s3 + r2*s4)

x_dist = x_n*icD + delta_x
y_dist = y_n*icD + delta_y
```

Guard denominator zero/non-finite values and mark them invalid.

- [ ] **Step 4: Implement the NCore/3DGUT trust gate**

```python
valid_radial = np.isfinite(icD) & (icD > 0.8) & (icD < 1.2)
```

Final visibility:

```python
visible = (z > 0) & valid_radial & in_bound & np.isfinite(uv).all(axis=1)
```

For invalid radial values, optionally mirror the reference directional clipping coordinate using `hypot(width,height)`, but never mark them visible.

- [ ] **Step 5: Run the focused projector suite**

```bash
/Users/etendue/repo/3dgrut2/.venv/bin/python -m pytest \
  threedgrut/tests/test_pinhole_projector.py -q
```

Expected: all old and new tests pass.

- [ ] **Step 6: Run overlay integration tests**

```bash
/Users/etendue/repo/3dgrut2/.venv/bin/python -m pytest \
  threedgrut/tests/test_viser_4d_ftheta_overlay_integration.py \
  threedgrut/tests/test_viser_gui_4d_camera_switch.py \
  threedgrut/tests/test_viser_gui_4d_cuboid_overlay.py -q
```

Expected: no FTheta or camera-state regression.

- [ ] **Step 7: Commit the pure implementation**

```bash
git add threedgrut_playground/utils/pinhole_projector.py \
        threedgrut/tests/test_pinhole_projector.py
git commit -m "fix(viser): match OpenCV rational overlay projection"
```

---

### Task 4: Add a Real NCore SDK Parity Validator

**Files:**
- Create: `scripts/validate_pinhole_projector_ncore_parity.py`

**Interfaces:**
- CLI inputs:
  - `--manifest PATH` required;
  - `--camera-ids ID [ID ...]` optional;
  - `--stride INT` default `64`;
  - `--valid-mae-threshold FLOAT` default `0.05` pixels.
- Output: per-camera sample count, SDK-valid count, projector-valid agreement, valid-point MAE/max error, and process exit nonzero on mismatch.

- [ ] **Step 1: Implement argument parsing and dataset camera discovery**

Use the established `NCoreDataset(datapath=..., split="train", device="cpu", load_aux_masks=False, n_val_image_subsample=1)` pattern. If camera ids are omitted and the dataset raises the multiple-camera message, parse the listed ids and reopen explicitly.

- [ ] **Step 2: Build representative camera-space rays**

For each OpenCVPinhole camera:

1. sample integer pixels on a stride grid plus center, four edge midpoints, and corners;
2. call `model.pixels_to_camera_rays(pixels)`;
3. call `model.camera_rays_to_pixels(rays)` to obtain SDK pixels and `valid_flag`;
4. feed the rays as world points with identity c2w to `PinholeForwardProjector.project_points()`.

- [ ] **Step 3: Compare valid semantics and coordinates**

Required assertions:

- projector `visible` equals SDK `valid_flag` after applying image-bound semantics;
- for SDK-valid samples, float projection error is below threshold after accounting for SDK integer pixel rounding; compare against original source integer pixels as the primary target;
- for SDK-invalid peripheral samples, projector visibility is false;
- report standard/tele and at least one wide camera separately.

Because SDK `camera_rays_to_pixels()` returns integer pixels, accept a coordinate tolerance consistent with rounding, e.g. max `< 1.0 px`; the script should still report mean and max.

- [ ] **Step 4: Run on inceptio real cameras**

```bash
python scripts/validate_pinhole_projector_ncore_parity.py \
  --manifest /home/inceptio/work/data/inc_b6a9ed61_20s/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9.json \
  --camera-ids \
    camera_front_standard_55fov \
    camera_front_tele_30fov \
    camera_front_wide_120fov \
    camera_cross_left_120fov \
    camera_left_wide_90fov
```

Expected:

- standard/tele nearly all samples valid;
- wide cameras contain SDK-invalid peripheral samples;
- valid-set agreement is exact;
- valid sample coordinate error is within rounding tolerance.

- [ ] **Step 5: Commit the validator**

```bash
git add scripts/validate_pinhole_projector_ncore_parity.py
git commit -m "test(viser): validate pinhole overlay against NCore SDK"
```

---

### Task 5: Documentation and Closure Evidence

**Files:**
- Modify: `docs/viser_mixed_camera_buglist.md`
- Create: `docs/T8_artifacts/C4_opencv_rational_overlay_validation.md`
- Modify: `.claude/skills/viser-gui-4d/SKILL.md` when tracked/editable.

**Interfaces:**
- Produces auditable evidence and closes only MC-6, not MC-10.

- [ ] **Step 1: Write validation evidence**

Record:

- root cause: six radial coefficients incorrectly interpreted as a polynomial;
- corrected rational/tangential/thin-prism formula;
- trust gate;
- RED test names and failure reasons;
- focused test counts;
- inceptio camera ids and parity results;
- explicit scope statement: Gaussian backdrop/training peripheral blur is not fixed by this task.

- [ ] **Step 2: Update MC-6**

Change MC-6 to complete only after parity passes. Keep MC-10 as high-confidence root cause awaiting forward-valid supervision-mask A/B.

- [ ] **Step 3: Update the Viser skill**

Add a durable rule:

```text
OpenCVPinhole six radial coefficients are rational numerator/denominator, not six polynomial powers; overlay projectors must match NCore SDK valid_flag and the 0.8–1.2 radial trust gate.
```

If `.claude/skills` is ignored but the file is tracked, use `git add -u`, not a broad force-add.

- [ ] **Step 4: Run final gates**

```bash
/Users/etendue/repo/3dgrut2/.venv/bin/python -m pytest \
  threedgrut/tests/test_pinhole_projector.py \
  threedgrut/tests/test_viser_4d_ftheta_overlay_integration.py \
  threedgrut/tests/test_viser_gui_4d_camera_switch.py \
  threedgrut/tests/test_viser_gui_4d_cuboid_overlay.py -q

/Users/etendue/repo/3dgrut2/.venv/bin/python -m pytest threedgrut/tests/ -q

git diff --check
python3 -m py_compile \
  threedgrut_playground/utils/pinhole_projector.py \
  scripts/validate_pinhole_projector_ncore_parity.py
```

- [ ] **Step 5: Commit docs and final evidence**

```bash
git add -u
git add docs/T8_artifacts/C4_opencv_rational_overlay_validation.md
git commit -m "docs(viser): close OpenCV rational overlay bug"
```

- [ ] **Step 6: Return a verifiable summary**

Return:

- branch and commit hashes;
- changed files;
- RED and GREEN test outputs;
- inceptio parity command/output summary;
- full-suite result;
- any remaining limitations;
- confirmation that the branch was not merged to main.

---

## Self-Review

- **Scope coverage:** The plan fixes only Viser OpenCV overlay projection and explicitly excludes dataset/training masks.
- **Formula coverage:** Rational numerator/denominator, tangential, thin-prism, denominator safety, and trust gate are all pinned.
- **Backward compatibility:** Ideal pinhole, one-coefficient radial, scalar focal length, flip, empty input, and polyline behavior remain covered.
- **Reference parity:** Mac hand-computed tests plus real NCore SDK parity are both required.
- **Closure discipline:** MC-6 may close; MC-10 remains open pending a separate training A/B.
- **Placeholder scan:** No TBD/TODO or unspecified implementation steps.
- **Type consistency:** Existing projector signatures remain unchanged.
