# SPDX-License-Identifier: Apache-2.0
"""PIN-FTHETA-1: unit tests for the OpenCV→FTheta fitter.

Tests are pure numpy, Mac-runnable, no NCore SDK needed — they hardcode
the b6a9ed61 camera_front_wide_120fov rational parameters fetched from
inceptio (see this task's diagnostic log).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from threedgrut_playground.utils.ftheta_fitter import (
    _rational_project_ray,
    _fit_monotonic_polynomial,
    fit_ftheta_from_opencv_rational,
    compute_ftheta_remap_and_mask,
    compute_opencv_reference_rays,
    compute_fullimage_angular_error,
)
from threedgrut_playground.utils.ftheta_intrinsics import ftheta_pixels_to_camera_rays
from threedgrut_playground.utils.ftheta_projector import FthetaForwardProjector
from threedgrut_playground.utils.pinhole_projector import PinholeForwardProjector
from threedgrut_playground.utils.projector_common import horner_ascending

# ---------------------------------------------------------------------------
# b6a9ed61 camera_front_wide_120fov OpenCVPinhole params (inceptio, 2026-07-15)
# ---------------------------------------------------------------------------
_B6A9_FRONT_WIDE_PINHOLE: dict = {
    "resolution": np.array([1920, 1080], dtype=np.int64),
    "shutter_type": "ROLLING_TOP_TO_BOTTOM",
    "principal_point": np.array([960.8599853515625, 540.1849975585938], dtype=np.float32),
    "focal_length": np.array([952.8250122070312, 952.9000244140625], dtype=np.float32),
    "radial_coeffs": np.array(
        [3.7687599658966064, 1.61149001121521, 0.0664215013384819,
         4.13346004486084, 2.880429983139038, 0.36570900678634644],
        dtype=np.float32,
    ),
    "tangential_coeffs": np.array(
        [4.691869980888441e-05, 8.77050024428172e-06], dtype=np.float32
    ),
    "thin_prism_coeffs": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
}


# ---------------------------------------------------------------------------
# Test 1: _rational_project_ray — basic projection sanity
# ---------------------------------------------------------------------------
def test_rational_project_ray_on_axis_hits_principal_point():
    """θ=0 ray should project to (cx, cy)."""
    proj = PinholeForwardProjector(_B6A9_FRONT_WIDE_PINHOLE)
    uv, r = _rational_project_ray(proj, np.array([0.0]))
    assert r[0] == pytest.approx(0.0, abs=1e-6)
    assert uv[0, 0] == pytest.approx(proj.cx, abs=1e-6)
    assert uv[0, 1] == pytest.approx(proj.cy, abs=1e-6)


def test_rational_project_ray_r_increases_with_angle():
    """r(θ) must be strictly increasing for a well-behaved lens."""
    proj = PinholeForwardProjector(_B6A9_FRONT_WIDE_PINHOLE)
    theta = np.linspace(0.0, 0.8, 50, dtype=np.float64)
    _, r = _rational_project_ray(proj, theta)
    physical_r = r[np.isfinite(r)]
    assert len(physical_r) < len(r), "test must cross the first physical branch"
    diffs = np.diff(physical_r)
    assert np.all(diffs > 0), f"r(θ) not monotonic: diffs={diffs[diffs < 0][:5]}"


def test_rational_project_ray_positive_x_projects_right_of_center():
    """Ray at (sin θ, 0, cos θ) with θ > 0 → u > cx."""
    proj = PinholeForwardProjector(_B6A9_FRONT_WIDE_PINHOLE)
    uv, _ = _rational_project_ray(proj, np.array([0.5]))
    assert uv[0, 0] > proj.cx


# ---------------------------------------------------------------------------
# Test 2: _fit_monotonic_polynomial
# ---------------------------------------------------------------------------
def test_fit_monotonic_polynomial_linear_exact():
    """Fitting points on a perfect line should give zero residuals."""
    x = np.linspace(0, 1, 100)
    y = 3.0 * x + 1.0  # c0=1, c1=3
    coeffs = _fit_monotonic_polynomial(x, y, degree=1)
    assert coeffs[0] == pytest.approx(1.0, abs=1e-3)
    assert coeffs[1] == pytest.approx(3.0, abs=1e-3)


def test_fit_monotonic_polynomial_cubic_monotonic():
    """A naturally monotonic cubic should fit tightly."""
    x = np.linspace(0, 2, 100)
    y = x**3 + 2 * x  # c0=0, c1=2, c2=0, c3=1
    coeffs = _fit_monotonic_polynomial(x, y, degree=3)
    assert coeffs[0] == pytest.approx(0.0, abs=0.05)
    assert coeffs[1] == pytest.approx(2.0, abs=0.05)
    assert coeffs[3] == pytest.approx(1.0, abs=0.05)
    # Verify monotonic: evaluate at dense samples
    deriv_coeffs = np.array([(k + 1) * coeffs[k + 1] for k in range(3)])
    deriv_vals = horner_ascending(deriv_coeffs, x)
    assert np.all(deriv_vals >= -1e-6), f"derivative negative: {deriv_vals.min()}"


def test_fit_monotonic_polynomial_fallback_stays_normalized_at_extreme_radius():
    """The fallback must not build an r**5 Vandermonde in pixel units."""
    x = np.linspace(0.0, 1.0e8, 200)
    y = np.log1p(x / 1.0e3)
    coeffs = _fit_monotonic_polynomial(
        x, y, degree=5, monotonic_tol=0.0
    )
    predicted = horner_ascending(coeffs, x)
    assert np.isfinite(coeffs).all()
    assert np.isfinite(predicted).all()
    # The old unnormalised fallback is rank deficient here and has >9.7 error.
    assert float(np.max(np.abs(predicted - y))) < 6.0


# ---------------------------------------------------------------------------
# Test 3: fit_ftheta_from_opencv_rational — produces valid 8-key dict
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def b6a9_ftheta():
    """Module-scoped fixture: fit once, reuse across validation tests."""
    return fit_ftheta_from_opencv_rational(_B6A9_FRONT_WIDE_PINHOLE)


_REQUIRED_KEYS = {
    "resolution",
    "shutter_type",
    "principal_point",
    "reference_poly",
    "pixeldist_to_angle_poly",
    "angle_to_pixeldist_poly",
    "max_angle",
    "linear_cde",
}


def test_ftheta_dict_has_all_8_required_keys(b6a9_ftheta):
    assert set(b6a9_ftheta.keys()) == _REQUIRED_KEYS


def test_ftheta_dict_polynomials_are_6_element(b6a9_ftheta):
    """NCore FTheta uses 5th-order polynomials (6 coefficients)."""
    assert len(b6a9_ftheta["angle_to_pixeldist_poly"]) == 6
    assert len(b6a9_ftheta["pixeldist_to_angle_poly"]) == 6


def test_ftheta_dict_max_angle_reasonable(b6a9_ftheta):
    """The physical trust branch ends near 42°, before later rational roots."""
    ma = b6a9_ftheta["max_angle"]
    assert 0.70 <= ma <= 0.80, f"max_angle={ma:.3f} rad outside expected [0.70, 0.80]"


def test_ftheta_dict_principal_point_preserved(b6a9_ftheta):
    """FTheta principal_point must match the OpenCV principal_point."""
    pp = b6a9_ftheta["principal_point"]
    assert float(pp[0]) == pytest.approx(960.86, abs=0.01)
    assert float(pp[1]) == pytest.approx(540.18, abs=0.01)


def test_ftheta_dict_resolution_preserved(b6a9_ftheta):
    res = b6a9_ftheta["resolution"]
    assert int(res[0]) == 1920
    assert int(res[1]) == 1080


def test_ftheta_linear_cde_is_identity(b6a9_ftheta):
    """We don't apply affine distortion correction; polynomial handles it."""
    cde = b6a9_ftheta["linear_cde"]
    np.testing.assert_array_almost_equal(cde, [1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Test 4: polynomial monotonicity
# ---------------------------------------------------------------------------
def test_angle_to_pixeldist_poly_is_monotonic(b6a9_ftheta):
    poly = b6a9_ftheta["angle_to_pixeldist_poly"]
    theta = np.linspace(0, b6a9_ftheta["max_angle"], 200, dtype=np.float64)
    r = horner_ascending(poly, theta)
    diffs = np.diff(r)
    assert np.all(diffs >= -1e-6), f"r(θ) not monotonic at max_angle: {diffs.min():.6f}"


def test_pixeldist_to_angle_poly_is_monotonic(b6a9_ftheta):
    poly = b6a9_ftheta["pixeldist_to_angle_poly"]
    r_max = horner_ascending(
        b6a9_ftheta["angle_to_pixeldist_poly"],
        b6a9_ftheta["max_angle"],
    )
    r = np.linspace(0, r_max, 200, dtype=np.float64)
    theta = horner_ascending(poly, r)
    diffs = np.diff(theta)
    assert np.all(diffs >= -1e-6), f"θ(r) not monotonic: {diffs.min():.6f}"


# ---------------------------------------------------------------------------
# Test 5: polynomial inverse consistency (round-trip in 1D)
# ---------------------------------------------------------------------------
def test_forward_inverse_roundtrip_1d(b6a9_ftheta):
    """θ → r(θ) → θ(r) ≈ θ for samples within max_angle."""
    poly_fwd = b6a9_ftheta["angle_to_pixeldist_poly"]
    poly_inv = b6a9_ftheta["pixeldist_to_angle_poly"]
    theta_in = np.linspace(0, b6a9_ftheta["max_angle"] * 0.95, 100, dtype=np.float64)
    r = horner_ascending(poly_fwd, theta_in)
    theta_out = horner_ascending(poly_inv, r)
    err = np.abs(theta_out - theta_in)
    assert err.max() < 1e-3, f"1D round-trip error: max={err.max():.4f} rad"


# ---------------------------------------------------------------------------
# Test 6: FTheta pixel → ray → pixel round-trip (full image)
# ---------------------------------------------------------------------------
def test_pixel_ray_roundtrip_via_ftheta_projector(b6a9_ftheta):
    """Round-trip coverage must match the trusted physical image domain."""
    valid_mask, error = compute_ftheta_remap_and_mask(b6a9_ftheta)
    H, W = valid_mask.shape
    frac_valid = valid_mask.sum() / (H * W)
    assert 0.62 < frac_valid < 0.65, (
        f"Only {frac_valid*100:.1f}% pixels round-trip; "
        f"max error={error.max():.4f} px, invalid={error[~valid_mask].sum()}"
    )


def test_pixel_roundtrip_center_is_exact(b6a9_ftheta):
    """Principal point pixel must round-trip within 0.1 px."""
    valid_mask, error = compute_ftheta_remap_and_mask(b6a9_ftheta)
    cy = int(b6a9_ftheta["principal_point"][1])
    cx = int(b6a9_ftheta["principal_point"][0])
    assert valid_mask[cy, cx], f"principal point not valid: error={error[cy, cx]:.4f} px"
    # The round-trip error at center is dominated by float32 quantization
    # of the pixel → ray conversion (ftheta_pixels_to_camera_rays returns
    # float32, project_points uses float64 internally) and polynomial fit
    # residuals.  0.1 px is well below the image-level median of 0.04 px.
    assert error[cy, cx] < 0.1, f"principal point error={error[cy, cx]:.4f} px"


# ---------------------------------------------------------------------------
# Test 7: Angular error — FTheta rays vs OpenCV rational reference
# ---------------------------------------------------------------------------
def test_fullimage_angular_error_matches_declared_front_wide_gate(b6a9_ftheta):
    metrics = compute_fullimage_angular_error(
        _B6A9_FRONT_WIDE_PINHOLE, b6a9_ftheta
    )
    assert metrics["mean_deg"] < 0.02
    assert metrics["p95_deg"] < 0.04
    assert metrics["p99_deg"] < 0.08
    assert metrics["max_deg"] < 0.15
    assert not metrics["outer_available"]
    assert np.isnan(metrics["outer_p99_deg"])
    assert metrics["nonradial_floor_mean_deg"] < 0.01
    assert metrics["forward_poly_max_px"] < 1.5
    assert 0.62 < metrics["opencv_inverse_coverage"] < 0.65
    assert metrics["physical_domain_retention"] > 0.99
    assert metrics["opencv_inverse_invalid_count"] > 0
    assert metrics["opencv_roundtrip_max_px"] < 1e-9


# ---------------------------------------------------------------------------
# Test 8: edge cases — empty inputs, invalid dicts
# ---------------------------------------------------------------------------
def test_fit_empty_radial_coeffs_does_not_crash():
    """A pinhole dict with zero radial coeffs (= no distortion) should still fit."""
    plain = {
        "resolution": np.array([640, 480], dtype=np.int64),
        "principal_point": np.array([320.0, 240.0], dtype=np.float32),
        "focal_length": np.array([500.0, 500.0], dtype=np.float32),
    }
    ftheta = fit_ftheta_from_opencv_rational(plain)
    assert set(ftheta.keys()) == _REQUIRED_KEYS
    # For a pure pinhole without distortion, r = f * tan(θ).
    # The FTheta polynomial should approximate this.  At small angles
    # tan(θ) ≈ θ, so the linear coefficient should be ≈ f.
    linear_coeff = ftheta["angle_to_pixeldist_poly"][1]
    assert linear_coeff == pytest.approx(500.0, rel=0.15)


def test_fit_missing_required_key_raises():
    """fit_ftheta must raise with missing focal_length."""
    bad = dict(_B6A9_FRONT_WIDE_PINHOLE)
    del bad["focal_length"]
    with pytest.raises(ValueError):
        fit_ftheta_from_opencv_rational(bad)


def test_ftheta_projector_accepts_fitted_dict(b6a9_ftheta):
    """The FthetaForwardProjector must accept the fitted dict without error."""
    proj = FthetaForwardProjector(b6a9_ftheta)
    assert proj.width == 1920
    assert proj.height == 1080


# ---------------------------------------------------------------------------
# Test 9: compare fitted FTheta polynomial with the ideal pinhole r = f*tan(θ)
# ---------------------------------------------------------------------------
def test_fitted_polynomial_near_ideal_at_small_angles(b6a9_ftheta):
    """At θ < 5°, FTheta polynomial should be close to r = f*θ (equidistant)
    which is close to r ≈ f*tan(θ) ≈ f*θ + f*θ³/3.

    The rational model for this camera is designed for ~120° FOV, so at small
    angles (θ < 0.1 rad ≈ 5.7°) the distortion should be mild and r ≈ f*θ
    within ~5%."""
    poly = b6a9_ftheta["angle_to_pixeldist_poly"]
    f = 952.9  # average focal length
    theta = np.linspace(0.0, 0.08, 30, dtype=np.float64)
    r_fitted = horner_ascending(poly, theta)
    r_ideal = f * theta  # equidistant approximation at small θ
    rel_err = np.abs(r_fitted - r_ideal) / np.maximum(r_ideal, 1e-6)
    # At 0.08 rad the pixel distance is ~76 px.
    assert rel_err[-1] < 0.05, f"rel error at θ=0.08 rad: {rel_err[-1]:.3f}"


def test_survey_single_camera_matches_direct_fullimage_evaluation():
    from scripts.pin_ftheta_camera_survey import evaluate_camera

    source = dict(_B6A9_FRONT_WIDE_PINHOLE)
    source["model_type"] = "OpenCVPinholeCameraModel"
    source["parameters_type"] = "OpenCVPinholeCameraModelParameters"
    survey = evaluate_camera(
        "camera_front_wide_120fov", source
    )
    direct_ftheta = fit_ftheta_from_opencv_rational(_B6A9_FRONT_WIDE_PINHOLE)
    direct_metrics = compute_fullimage_angular_error(
        _B6A9_FRONT_WIDE_PINHOLE, direct_ftheta
    )

    assert set(survey["ftheta_parameters"]) == _REQUIRED_KEYS
    assert survey["source_model_type"] == "OpenCVPinholeCameraModel"
    for key, expected in direct_metrics.items():
        if np.isnan(expected):
            assert survey["fit_metrics"][key] is None
        else:
            assert survey["fit_metrics"][key] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("missing", ["model_type", "parameters_type"])
def test_survey_requires_explicit_source_model_metadata(missing):
    from scripts.pin_ftheta_camera_survey import evaluate_camera

    source = dict(_B6A9_FRONT_WIDE_PINHOLE)
    source["model_type"] = "OpenCVPinholeCameraModel"
    source["parameters_type"] = "OpenCVPinholeCameraModelParameters"
    del source[missing]
    with pytest.raises(ValueError, match=missing):
        evaluate_camera("camera_front_wide_120fov", source)


def test_survey_rejects_non_opencv_models():
    from scripts.pin_ftheta_camera_survey import evaluate_camera

    source = dict(_B6A9_FRONT_WIDE_PINHOLE)
    source["model_type"] = "FThetaCameraModel"
    source["parameters_type"] = "FThetaCameraModelParameters"
    with pytest.raises(ValueError, match="only OpenCVPinholeCameraModel"):
        evaluate_camera("camera_front_wide_120fov", source)


def test_survey_gate_only_allows_unavailable_outer_p99_nan():
    from scripts.pin_ftheta_camera_survey import (
        FIT_GATE_THRESHOLDS,
        evaluate_fit_gate,
    )

    metrics = {key: 0.0 for key in FIT_GATE_THRESHOLDS}
    metrics["outer_available"] = False
    metrics["outer_p99_deg"] = float("nan")
    gate = evaluate_fit_gate(metrics)
    assert gate["passed"]
    assert gate["not_applicable_metrics"] == ["outer_p99_deg"]

    metrics["mean_deg"] = float("nan")
    gate = evaluate_fit_gate(metrics)
    assert not gate["passed"]
    assert "mean_deg" in gate["failed_metrics"]

    metrics["mean_deg"] = 0.0
    metrics["outer_available"] = True
    gate = evaluate_fit_gate(metrics)
    assert not gate["passed"]
    assert "outer_p99_deg" in gate["failed_metrics"]


def test_frozen_nine_camera_artifact_contract_and_provenance():
    root = Path(__file__).resolve().parents[2]
    artifact = json.loads(
        (root / "scripts" / "pin_ftheta_b6a9_9cam_params.json").read_text()
    )
    required = {
        "resolution",
        "shutter_type",
        "principal_point",
        "reference_poly",
        "pixeldist_to_angle_poly",
        "angle_to_pixeldist_poly",
        "max_angle",
        "linear_cde",
    }
    assert len(artifact["camera_order"]) == 9
    assert len(set(artifact["camera_order"])) == 9
    assert artifact["fitter_version"]
    for camera_id in artifact["camera_order"]:
        camera = artifact["cameras"][camera_id]
        assert set(camera["ftheta_parameters"]) == required
        assert len(camera["source_calibration_sha256"]) == 64
        assert camera["source_model_type"] == "OpenCVPinholeCameraModel"
        assert (
            camera["source_parameters_type"]
            == "OpenCVPinholeCameraModelParameters"
        )
        assert camera["fitter_version"] == artifact["fitter_version"]
        assert "opencv_inverse_invalid_count" in camera["fit_metrics"]
        assert "opencv_roundtrip_max_px" in camera["fit_metrics"]
