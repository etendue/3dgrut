# SPDX-License-Identifier: Apache-2.0
"""PIN-FTHETA-1: unit tests for the OpenCV→FTheta fitter.

Tests are pure numpy, Mac-runnable, no NCore SDK needed — they hardcode
the b6a9ed61 camera_front_wide_120fov rational parameters fetched from
inceptio (see this task's diagnostic log).
"""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from threedgrut_playground.utils.ftheta_fitter import (
    _ftheta_own_domain_mask,
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
    """Calibration sampling must continue beyond the pinhole icD guard."""
    proj = PinholeForwardProjector(_B6A9_FRONT_WIDE_PINHOLE)
    theta = np.linspace(0.0, 1.2, 100, dtype=np.float64)
    _, r = _rational_project_ray(proj, theta)
    assert np.isfinite(r).all()
    diffs = np.diff(r)
    assert np.all(diffs > 0), f"r(θ) not monotonic: diffs={diffs[diffs < 0][:5]}"


def test_rational_project_ray_does_not_inherit_pinhole_icd_gate():
    """The 120° camera's 60° edge ray must remain usable for FTheta fitting."""
    proj = PinholeForwardProjector(_B6A9_FRONT_WIDE_PINHOLE)
    uv, r = _rational_project_ray(proj, np.deg2rad(np.array([60.0])))
    assert np.isfinite(uv).all()
    assert np.isfinite(r).all()
    assert r[0] > 900.0


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


def test_fit_monotonic_polynomial_rejects_nonmonotonic_fallback():
    """A finite fallback still fails if its dense derivative turns negative."""
    x = np.linspace(0.0, 1.0e8, 200)
    y = np.log1p(x / 1.0e3)
    with pytest.raises(RuntimeError, match="not strictly monotonic"):
        _fit_monotonic_polynomial(x, y, degree=5, monotonic_tol=0.0)


def test_fit_monotonic_polynomial_allows_valid_deterministic_fallback(monkeypatch):
    """The numerical fallback is allowed when it passes the same dense gate."""
    x = np.linspace(0.0, 1.0, 200)
    y = 2.0 * x

    def nonmonotonic_primary(_x, _y, degree):
        assert degree == 2
        return np.array([-1.0, 0.0, 0.0])  # descending-order -x**2

    monkeypatch.setattr(np, "polyfit", nonmonotonic_primary)
    coeffs = _fit_monotonic_polynomial(x, y, degree=2, monotonic_tol=0.0)
    derivative = horner_ascending(
        np.arange(1, len(coeffs), dtype=np.float64) * coeffs[1:],
        np.linspace(0.0, 1.0, 20_001),
    )
    assert np.isfinite(coeffs).all()
    assert float(derivative.min()) > 0.0


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
    """120° horizontal FOV must not be truncated by the pinhole trust gate."""
    ma = b6a9_ftheta["max_angle"]
    assert ma > np.deg2rad(60.0)
    assert 1.18 <= ma <= 1.25, f"max_angle={ma:.3f} rad outside expected [1.18, 1.25]"


def test_ftheta_own_domain_uses_strict_cuda_max_angle_boundary():
    max_angle = 1.2
    angles = np.array(
        [np.nextafter(max_angle, 0.0), max_angle, np.nextafter(max_angle, np.inf)]
    )
    rays = np.stack(
        [np.sin(angles), np.zeros_like(angles), np.cos(angles)],
        axis=-1,
    )
    np.testing.assert_array_equal(
        _ftheta_own_domain_mask(rays, max_angle),
        [True, False, False],
    )


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
    """FTheta must cover the full raster instead of the pinhole trust disk."""
    valid_mask, error = compute_ftheta_remap_and_mask(b6a9_ftheta)
    H, W = valid_mask.shape
    frac_valid = valid_mask.sum() / (H * W)
    assert frac_valid > 0.999, (
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
    assert metrics["outer_available"]
    assert metrics["outer_p99_deg"] < 0.10
    assert metrics["peripheral_p99_deg"] < 0.10
    assert metrics["nonradial_floor_mean_deg"] < 0.01
    assert metrics["forward_poly_max_px"] < 1.5
    assert metrics["opencv_calibration_domain_coverage"] == pytest.approx(1.0)
    assert metrics["ftheta_own_domain_coverage"] > 0.999
    assert metrics["comparison_intersection_coverage"] > 0.999
    assert (
        metrics["comparison_intersection_retention_of_opencv_calibration_domain"]
        > 0.99
    )
    assert metrics["opencv_calibration_domain_excluded_count"] == 0
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


@pytest.mark.parametrize(
    ("case", "expected_failed_check"),
    [
        ("extra_key", "exact_eight_fields"),
        ("resolution_nonpositive", "resolution_shape_positive_and_preserved"),
        ("resolution_not_preserved", "resolution_shape_positive_and_preserved"),
        ("principal_point_shape", "principal_point_shape_and_preserved"),
        ("principal_point_not_preserved", "principal_point_shape_and_preserved"),
        ("linear_cde_shape", "linear_cde_identity"),
        ("linear_cde_nonidentity", "linear_cde_identity"),
        ("reference_poly_not_fixed", "reference_poly_fixed"),
        ("shutter_invalid", "shutter_type_supported"),
        ("forward_poly_length", "polynomial_lengths"),
        ("inverse_poly_length", "polynomial_lengths"),
        ("max_angle_nonfinite", "max_angle_finite_range"),
        ("max_angle_out_of_range", "max_angle_finite_range"),
    ],
)
def test_survey_structural_hard_gate_reuses_loader_contract(
    case,
    expected_failed_check,
):
    from scripts.pin_ftheta_camera_survey import evaluate_hard_invariants

    root = Path(__file__).resolve().parents[2]
    parameters = json.loads(
        (
            root
            / "scripts"
            / "pin_ftheta_b6a9_7cam_params_v4_full_domain.json"
        ).read_text()
    )["camera_front_wide_120fov"]
    parameters = json.loads(json.dumps(parameters))
    calibration = json.loads(
        (root / "scripts" / "pin_ftheta_b6a9_calibs.json").read_text()
    )["cameras"]["camera_front_wide_120fov"]
    pinhole = {
        key: value
        for key, value in calibration.items()
        if key not in {"model_type", "parameters_type"}
    }
    if case == "extra_key":
        parameters["metadata"] = "forbidden"
    elif case == "resolution_nonpositive":
        parameters["resolution"] = [0, 1080]
    elif case == "resolution_not_preserved":
        parameters["resolution"] = [1919, 1080]
    elif case == "principal_point_shape":
        parameters["principal_point"] = [960.0]
    elif case == "principal_point_not_preserved":
        parameters["principal_point"][0] += 1.0
    elif case == "linear_cde_shape":
        parameters["linear_cde"] = [1.0, 0.0]
    elif case == "linear_cde_nonidentity":
        parameters["linear_cde"] = [1.0, 0.01, 0.0]
    elif case == "reference_poly_not_fixed":
        parameters["reference_poly"] = "ANGLE_TO_PIXELDIST"
    elif case == "shutter_invalid":
        parameters["shutter_type"] = "INVALID"
    elif case == "forward_poly_length":
        parameters["angle_to_pixeldist_poly"] = parameters[
            "angle_to_pixeldist_poly"
        ][:-1]
    elif case == "inverse_poly_length":
        parameters["pixeldist_to_angle_poly"] = parameters[
            "pixeldist_to_angle_poly"
        ][:-1]
    elif case == "max_angle_nonfinite":
        parameters["max_angle"] = float("nan")
    elif case == "max_angle_out_of_range":
        parameters["max_angle"] = math.pi + 0.01
    else:  # pragma: no cover - parametrization guard
        raise AssertionError(case)

    result = evaluate_hard_invariants(
        "camera_front_wide_120fov",
        pinhole,
        parameters,
        {
            "opencv_roundtrip_max_px": 0.0,
            "ftheta_own_domain_excluded_count": 148,
            "opencv_calibration_domain_excluded_count": 0,
        },
    )
    assert not result["passed"]
    assert expected_failed_check in result["failed_invariants"]
    if case not in {
        "linear_cde_nonidentity",
        "reference_poly_not_fixed",
        "max_angle_out_of_range",
        "principal_point_not_preserved",
        "resolution_not_preserved",
    }:
        assert not result["checks"]["loader_eight_field_contract_valid"]


def test_survey_quality_warnings_only_allow_unavailable_outer_p99_nan():
    from scripts.pin_ftheta_camera_survey import (
        FIT_GATE_THRESHOLDS,
        evaluate_quality_warnings,
    )

    metrics = {key: 0.0 for key in FIT_GATE_THRESHOLDS}
    metrics["outer_available"] = False
    metrics["outer_p99_deg"] = float("nan")
    warnings = evaluate_quality_warnings(metrics)
    assert not warnings["has_warnings"]
    assert warnings["not_applicable_metrics"] == ["outer_p99_deg"]

    metrics["mean_deg"] = float("nan")
    warnings = evaluate_quality_warnings(metrics)
    assert warnings["has_warnings"]
    assert "mean_deg" in warnings["warning_metrics"]

    metrics["mean_deg"] = 0.0
    metrics["outer_available"] = True
    warnings = evaluate_quality_warnings(metrics)
    assert warnings["has_warnings"]
    assert "outer_p99_deg" in warnings["warning_metrics"]


@pytest.mark.parametrize(
    ("all_hard_invariants_passed", "expected_exit"),
    [(True, 0), (False, 2)],
)
def test_survey_cli_exit_depends_only_on_hard_invariants(
    tmp_path,
    monkeypatch,
    all_hard_invariants_passed,
    expected_exit,
):
    import scripts.pin_ftheta_camera_survey as survey_module

    calibrations = tmp_path / "calibrations.json"
    calibrations.write_text("{}")
    output = tmp_path / "survey.json"
    fake_survey = {
        "all_hard_invariants_passed": all_hard_invariants_passed,
        "hard_failures": {} if all_hard_invariants_passed else {"camera": ["finite"]},
        "active_subset_hard_invariants_passed": all_hard_invariants_passed,
        "active_hard_failures": (
            {} if all_hard_invariants_passed else {"camera": ["finite"]}
        ),
        # Warnings deliberately remain present in the successful case.
        "quality_warning_cameras": ["camera"],
    }
    monkeypatch.setattr(survey_module, "survey_bundle", lambda _: fake_survey)
    assert survey_module.main(
        [
            "--calibrations",
            str(calibrations),
            "--format",
            "json",
            "--output",
            str(output),
        ]
    ) == expected_exit
    assert json.loads(output.read_text())["quality_warning_cameras"] == ["camera"]


def test_versioned_nine_camera_artifact_contract_and_provenance():
    root = Path(__file__).resolve().parents[2]
    artifact = json.loads(
        (
            root
            / "scripts"
            / "pin_ftheta_b6a9_9cam_survey_v4_full_domain.json"
        ).read_text()
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
    assert len(artifact["active_camera_order"]) == 7
    assert artifact["all_hard_invariants_passed"]
    assert artifact["hard_failures"] == {}
    assert artifact["active_subset_hard_invariants_passed"]
    assert artifact["active_hard_failures"] == {}
    assert artifact["scope"]["nine_camera_survey_role"] == "CALIBRATION_EVIDENCE_ONLY"
    assert not artifact["scope"]["nine_camera_training_approved"]
    assert artifact["excluded_from_runtime_camera_order"] == [
        "camera_front_standard_55fov",
        "camera_front_tele_30fov",
    ]
    assert artifact["quality_warning_cameras"]
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
        assert camera["hard_invariants"]["passed"]
        assert "quality_warnings" in camera
        assert "opencv_calibration_domain_excluded_count" in camera["fit_metrics"]
        assert "ftheta_own_domain_excluded_count" in camera["fit_metrics"]
        assert "comparison_intersection_excluded_count" in camera["fit_metrics"]
        assert "opencv_roundtrip_max_px" in camera["fit_metrics"]


def test_versioned_runtime_artifact_all_seven_dense_derivatives_and_sentinel():
    root = Path(__file__).resolve().parents[2]
    runtime = json.loads(
        (
            root
            / "scripts"
            / "pin_ftheta_b6a9_7cam_params_v4_full_domain.json"
        ).read_text()
    )
    survey = json.loads(
        (
            root
            / "scripts"
            / "pin_ftheta_b6a9_9cam_survey_v4_full_domain.json"
        ).read_text()
    )
    expected_order = [
        "camera_front_wide_120fov",
        "camera_cross_left_120fov",
        "camera_cross_right_120fov",
        "camera_left_wide_90fov",
        "camera_right_wide_90fov",
        "camera_back_rear_wide_90fov",
        "camera_rear_left_70fov",
    ]
    assert list(runtime) == expected_order
    assert runtime == {
        camera_id: survey["cameras"][camera_id]["ftheta_parameters"]
        for camera_id in survey["active_camera_order"]
    }
    assert set(runtime[expected_order[0]]) == _REQUIRED_KEYS
    assert runtime[expected_order[0]]["max_angle"] == pytest.approx(
        1.2087891572930667,
        abs=1e-15,
    )
    assert runtime[expected_order[0]]["max_angle"] != pytest.approx(
        0.7303101158645611,
        abs=1e-12,
    )
    for camera_id in expected_order:
        parameters = runtime[camera_id]
        forward = np.asarray(parameters["angle_to_pixeldist_poly"], dtype=np.float64)
        inverse = np.asarray(parameters["pixeldist_to_angle_poly"], dtype=np.float64)
        max_angle = float(parameters["max_angle"])
        radius_max = float(horner_ascending(forward, max_angle))
        forward_derivative = horner_ascending(
            np.arange(1, len(forward), dtype=np.float64) * forward[1:],
            np.linspace(0.0, max_angle, 20_001),
        )
        inverse_derivative = horner_ascending(
            np.arange(1, len(inverse), dtype=np.float64) * inverse[1:],
            np.linspace(0.0, radius_max, 20_001),
        )
        assert float(forward_derivative.min()) > 0.0, camera_id
        assert float(inverse_derivative.min()) > 0.0, camera_id


def test_active_seven_domain_counts_are_explicit_and_frozen():
    root = Path(__file__).resolve().parents[2]
    survey = json.loads(
        (
            root
            / "scripts"
            / "pin_ftheta_b6a9_9cam_survey_v4_full_domain.json"
        ).read_text()
    )
    expected = {
        "camera_front_wide_120fov": (148, 0),
        "camera_cross_left_120fov": (138, 0),
        "camera_cross_right_120fov": (133, 0),
        "camera_left_wide_90fov": (26_355, 31_681),
        "camera_right_wide_90fov": (44_292, 51_504),
        "camera_back_rear_wide_90fov": (120, 0),
        "camera_rear_left_70fov": (101, 0),
    }
    for camera_id, (ftheta_excluded, opencv_excluded) in expected.items():
        metrics = survey["cameras"][camera_id]["fit_metrics"]
        total = metrics["total_pixel_count"]
        assert metrics["ftheta_own_domain_excluded_count"] == ftheta_excluded
        assert (
            metrics["opencv_calibration_domain_excluded_count"]
            == opencv_excluded
        )
        assert metrics["ftheta_own_domain_count"] + ftheta_excluded == total
        assert metrics["opencv_calibration_domain_count"] + opencv_excluded == total
        assert (
            metrics["comparison_intersection_count"]
            + metrics["comparison_intersection_excluded_count"]
            == total
        )
        assert "invalid_pixel_count" not in metrics
        assert "valid_pixel_count" not in metrics


def test_versioned_runtime_artifact_loader_compatibility_and_provenance():
    from threedgrut.datasets.ftheta_override import load_ftheta_override_parameters

    root = Path(__file__).resolve().parents[2]
    runtime_path = (
        root / "scripts" / "pin_ftheta_b6a9_7cam_params_v4_full_domain.json"
    )
    sidecar = json.loads(
        (
            root
            / "scripts"
            / "pin_ftheta_b6a9_7cam_params_v4_full_domain.provenance.json"
        ).read_text()
    )
    selected = sidecar["camera_order"]
    normalized, fingerprints = load_ftheta_override_parameters(
        runtime_path,
        selected,
    )
    assert list(normalized) == selected
    assert set(fingerprints) == set(selected)
    for relative_path, expected_sha in sidecar["artifacts"].items():
        assert hashlib.sha256((root / relative_path).read_bytes()).hexdigest() == expected_sha
    for relative_path, expected_sha in sidecar["sources"].items():
        assert hashlib.sha256((root / relative_path).read_bytes()).hexdigest() == expected_sha


def test_three_file_export_is_atomic_deterministic_and_loader_compatible(
    tmp_path,
    monkeypatch,
):
    import scripts.export_9cam_ftheta_params as exporter
    from threedgrut.datasets.ftheta_override import load_ftheta_override_parameters

    root = Path(__file__).resolve().parents[2]
    accepted_survey = json.loads(
        (
            root
            / "scripts"
            / "pin_ftheta_b6a9_9cam_survey_v4_full_domain.json"
        ).read_text()
    )
    assert accepted_survey["all_hard_invariants_passed"]
    assert accepted_survey["hard_failures"] == {}
    assert accepted_survey["active_subset_hard_invariants_passed"]
    assert accepted_survey["quality_warning_cameras"]
    monkeypatch.setattr(exporter, "build_artifact", lambda _bundle: accepted_survey)

    calibrations = tmp_path / "calibrations.json"
    calibrations.write_text("{}\n")
    survey_output = tmp_path / "survey.json"
    runtime_output = tmp_path / "runtime.json"
    provenance_output = tmp_path / "provenance.json"
    argv = [
        "--calibrations",
        str(calibrations),
        "--survey-output",
        str(survey_output),
        "--runtime-output",
        str(runtime_output),
        "--provenance-output",
        str(provenance_output),
        "--generated-at",
        "2026-07-18T12:29:38+08:00",
    ]
    assert exporter.main(argv) == 0
    first = tuple(
        path.read_bytes()
        for path in (survey_output, runtime_output, provenance_output)
    )
    assert exporter.main(argv) == 0
    second = tuple(
        path.read_bytes()
        for path in (survey_output, runtime_output, provenance_output)
    )
    assert first == second

    expected_runtime = {
        camera_id: accepted_survey["cameras"][camera_id]["ftheta_parameters"]
        for camera_id in accepted_survey["active_camera_order"]
    }
    assert json.loads(runtime_output.read_text()) == expected_runtime
    normalized, fingerprints = load_ftheta_override_parameters(
        runtime_output,
        accepted_survey["active_camera_order"],
    )
    assert list(normalized) == accepted_survey["active_camera_order"]
    assert set(fingerprints) == set(accepted_survey["active_camera_order"])

    sidecar = json.loads(provenance_output.read_text())
    command = sidecar["generation_command"]
    assert "--survey-output" in command
    assert "--runtime-output" in command
    assert "--provenance-output" in command
    assert "extract" not in command
    assert _sha256(survey_output) in sidecar["artifacts"].values()
    assert _sha256(runtime_output) in sidecar["artifacts"].values()


@pytest.mark.parametrize("preexisting", [True, False])
def test_three_file_transaction_rolls_back_second_replace_failure(
    tmp_path,
    monkeypatch,
    preexisting,
):
    import scripts.export_9cam_ftheta_params as exporter

    destinations = [
        tmp_path / "survey.json",
        tmp_path / "runtime.json",
        tmp_path / "provenance.json",
    ]
    old_payloads = {
        destination: f"old-{index}\n".encode()
        for index, destination in enumerate(destinations)
    }
    if preexisting:
        for destination, payload in old_payloads.items():
            destination.write_bytes(payload)

    real_replace = exporter.os.replace
    replace_count = 0

    def fail_second_replace(source, destination):
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError("injected second replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(exporter.os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="injected second replace failure"):
        exporter._atomic_write_many(
            {
                destination: f"new-{index}\n".encode()
                for index, destination in enumerate(destinations)
            }
        )

    if preexisting:
        assert {
            destination: destination.read_bytes() for destination in destinations
        } == old_payloads
    else:
        assert all(not destination.exists() for destination in destinations)
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.backup"))


@pytest.mark.parametrize("collision_kind", ["duplicate", "calibration", "source"])
def test_exporter_rejects_output_collisions_before_computation_or_write(
    tmp_path,
    monkeypatch,
    collision_kind,
):
    import scripts.export_9cam_ftheta_params as exporter

    root = Path(__file__).resolve().parents[2]
    calibrations = tmp_path / "calibrations.json"
    calibrations.write_text("{}\n")
    fitter_source = root / "threedgrut_playground" / "utils" / "ftheta_fitter.py"
    source_hashes = {
        calibrations: _sha256(calibrations),
        fitter_source: _sha256(fitter_source),
    }
    survey_output = tmp_path / "survey.json"
    runtime_output = tmp_path / "runtime.json"
    provenance_output = tmp_path / "provenance.json"
    if collision_kind == "duplicate":
        runtime_output = survey_output
    elif collision_kind == "calibration":
        survey_output = calibrations
    elif collision_kind == "source":
        provenance_output = fitter_source
    else:  # pragma: no cover - parametrization guard
        raise AssertionError(collision_kind)

    monkeypatch.setattr(
        exporter,
        "build_artifact",
        lambda _bundle: pytest.fail("collision must be rejected before survey"),
    )
    monkeypatch.setattr(
        exporter,
        "_atomic_write_many",
        lambda _payloads: pytest.fail("collision must be rejected before writes"),
    )
    assert exporter.main(
        [
            "--calibrations",
            str(calibrations),
            "--survey-output",
            str(survey_output),
            "--runtime-output",
            str(runtime_output),
            "--provenance-output",
            str(provenance_output),
            "--generated-at",
            "2026-07-18T12:29:38+08:00",
        ]
    ) == 2
    assert {path: _sha256(path) for path in source_hashes} == source_hashes


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exporter_refuses_legacy_targets_without_changing_their_head_hashes(
    tmp_path,
    monkeypatch,
):
    import scripts.export_9cam_ftheta_params as exporter

    root = Path(__file__).resolve().parents[2]
    legacy_seven = root / "scripts" / "pin_ftheta_b6a9_7cam_params.json"
    legacy_nine = root / "scripts" / "pin_ftheta_b6a9_9cam_params.json"
    expected_hashes = {
        legacy_seven: "73965c6d10693b7742c7afb538169990a94d2457890738a7a4f66d13fcd0a450",
        legacy_nine: "2f914d17f69d7f235ddd90abe1d52c3e9e25e383b29491b2e9d7dbef2f162cfa",
    }
    assert {path: _sha256(path) for path in expected_hashes} == expected_hashes
    calibrations = tmp_path / "calibrations.json"
    calibrations.write_text("{}\n")
    monkeypatch.setattr(
        exporter,
        "build_artifact",
        lambda _bundle: pytest.fail("legacy path must be rejected before survey"),
    )

    cases = [
        (legacy_nine, tmp_path / "runtime.json", tmp_path / "provenance.json"),
        (tmp_path / "survey.json", legacy_seven, tmp_path / "provenance.json"),
    ]
    for survey_output, runtime_output, provenance_output in cases:
        assert exporter.main(
            [
                "--calibrations",
                str(calibrations),
                "--survey-output",
                str(survey_output),
                "--runtime-output",
                str(runtime_output),
                "--provenance-output",
                str(provenance_output),
                "--generated-at",
                "2026-07-18T12:29:38+08:00",
            ]
        ) == 2
        assert {path: _sha256(path) for path in expected_hashes} == expected_hashes
