# C4 OpenCV Rational Overlay Validation

Date: 2026-07-15
Implementation branch: `fix/viser-opencv-rational-overlay`（do not merge to main）
Validated implementation range: `e5aee25..132261e` plus this commit
Host: `inceptio` / RTX 4090 / NCore SDK v4

## Root Cause

The `PinholeForwardProjector` treated the six `radial_coeffs` as a six-term polynomial:

    radial = 1 + k1·r² + k2·r⁴ + k3·r⁶ + k4·r⁸ + k5·r¹⁰ + k6·r¹²

NCore SDK's `OpenCVPinholeCameraModel` and 3DGUT's `cameraProjections.cuh` use a **rational** model:

    icD_num = 1 + k1·r² + k2·r⁴ + k3·r⁶
    icD_den = 1 + k4·r² + k5·r⁴ + k6·r⁶
    icD     = icD_num / icD_den

Coefficients 4–6 form the **denominator**, not polynomial powers 8–12.

## Corrected Formula

    r²       = x_n² + y_n²
    r⁴       = r²·r²
    r⁶       = r⁴·r²

    icD_num  = 1 + k1·r² + k2·r⁴ + k3·r⁶
    icD_den  = 1 + k4·r² + k5·r⁴ + k6·r⁶
    icD      = icD_num / icD_den

    a1       = 2·x_n·y_n
    a2       = r² + 2·x_n²
    a3       = r² + 2·y_n²

    delta_x  = p1·a1 + p2·a2 + r²·(s1 + r²·s2)
    delta_y  = p1·a3 + p2·a1 + r²·(s3 + r²·s4)

    x_dist   = x_n·icD + delta_x
    y_dist   = y_n·icD + delta_y

## Trust Gate

NCore SDK and 3DGUT both reject radial distortion when `icD` lies outside `(0.8, 1.2)`:

```python
valid_radial = np.isfinite(icD) & (icD > 0.8) & (icD < 1.2)
visible = (z > 0) & valid_radial & in_bound & uv_finite
```

Outside this interval the radial model is unreliable (e.g. fringe artifacts on wide cameras); the point is treated as invisible for overlay drawing.

## RED Tests (4 new, all written before production code)

| Test | Failure Reason |
|------|---------------|
| `test_six_radial_coefficients_use_rational_denominator` | Polynomial model produces wrong coordinate for 6-coeff rational config |
| `test_thin_prism_coefficients_match_opencv_model` | Old code ignores `thin_prism_coeffs` entirely |
| `test_radial_scale_outside_ncore_trust_interval_is_invalid` | No trust gate → icD=1.3 incorrectly marked visible |
| `test_radial_scale_below_ncore_trust_interval_is_invalid` | No trust gate → icD=0.7 incorrectly marked visible |

Plus regression tests for short-array compatibility (missing keys, single coeff, too many coeffs → ValueError, etc.).

## Automated Gates

- Mac focused projector suite: `21 passed`
- Mac overlay integration tests: `36 passed`
- Mac full suite: `1018 passed, 2 skipped`
- Python compile: clean
- `git diff --check`: clean

## Inceptio NCore SDK Parity

Validation script: `scripts/validate_pinhole_projector_ncore_parity.py`

Commands:

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

Results:

| Camera | Resolution | Samples | Vis. Agreement | Valid MAE (px) | Max Err (px) | Integer RTT |
|--------|-----------|--------:|:--------------:|:--------------:|:------------:|:-----------:|
| `front_standard_55fov` | 1920×1080 | 837 | 100% | 0.000017 | 0.000048 | ✔ |
| `front_tele_30fov` | 1920×1080 | 837 | 100% | 0.000036 | 0.000070 | ✔ |
| `front_wide_120fov` | 1920×1080 | 837 | 100% | 0.000017 | 0.000048 | ✔ |
| `cross_left_120fov` | 1920×1080 | 837 | 100% | 0.000018 | 0.000041 | ✔ |
| `left_wide_90fov` | 1920×1080 | 837 | 100% | 0.000018 | 0.000041 | ✔ |

- **Validity agreement: 100%** — projector `visible` matches SDK `valid_flag` after image-bound check for every sample across all 5 cameras.
- **Image-point MAE: 0.000017–0.000036 px** — several orders of magnitude below the 0.05 px threshold, dominated by SDK's `float32` → `float64` rounding.
- **Integer roundtrip: true** — `floor(projector_uv) == SDK integer pixels` for every visible sample.
- **Peripheral rejection verified** — SDK-invalid samples (wide cameras near the 120° FOV boundary, where icD falls outside the trust interval) are consistently marked invisible by both SDK and projector.

## Scope Statement

This task fixes **only the OpenCV rational overlay projection** (`PinholeForwardProjector` / `pinhole_projector.py`). It does **not** address:

- Gaussian backdrop peripheral blur / training quality;
- FTheta camera model (unchanged);
- Dataset modifications (`datasetNcore.py`, photometric masks);
- Training losses or camera parameter overrides;
- Checkpoint quality evaluation;
- MC-10 (center/periphery blur attribution — remains open pending a forward-valid supervision-mask A/B).

## Modified Files

- `threedgrut_playground/utils/pinhole_projector.py` — rational radial, tangential, thin-prism, trust gate
- `threedgrut/tests/test_pinhole_projector.py` — 10 new regression tests
- `scripts/validate_pinhole_projector_ncore_parity.py` — NCore SDK parity validator
- `docs/T8_artifacts/C4_opencv_rational_overlay_validation.md` — this document (new)
- `docs/viser_mixed_camera_buglist.md` — MC-6 closed
