#!/usr/bin/env python3
"""Survey frozen b6a9 OpenCV calibrations against fitted FTheta models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from threedgrut_playground.utils.ftheta_fitter import (  # noqa: E402
    _minimum_polynomial_derivative,
    compute_fullimage_angular_error,
    fit_ftheta_from_opencv_rational,
)
from threedgrut.ftheta_override_contract import (  # noqa: E402
    FTHETA_PARAMETER_KEYS,
    _REFERENCE_POLYNOMIAL_TYPES,
    _SHUTTER_TYPES,
    _validate_ftheta_parameters,
)

FITTER_VERSION = "pin-ftheta-numpy-v4-full-calibration-domain-2026-07-18"

# Predeclared residual thresholds retained from PIN-FTHETA-3/4.  Under the
# user-approved v4 seven-camera approximation these remain visible quality
# warnings, not GPU-blocking hard invariants.  An outer-angle warning applies
# only when a camera actually reaches 55 degrees.
QUALITY_WARNING_THRESHOLDS: dict[str, float] = {
    "nonradial_floor_mean_deg": 0.01,
    "forward_poly_max_px": 1.5,
    "mean_deg": 0.02,
    "p95_deg": 0.04,
    "p99_deg": 0.08,
    "max_deg": 0.15,
    "outer_p99_deg": 0.10,
}

# Backward-compatible import name.  These values are warning thresholds in v4,
# not GPU-blocking representation gates.
FIT_GATE_THRESHOLDS = QUALITY_WARNING_THRESHOLDS

ACTIVE_CAMERA_ORDER = (
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_left_wide_90fov",
    "camera_right_wide_90fov",
    "camera_back_rear_wide_90fov",
    "camera_rear_left_70fov",
)
EXCLUDED_FROM_RUNTIME_CAMERA_ORDER = (
    "camera_front_standard_55fov",
    "camera_front_tele_30fov",
)

# Native-resolution accepted v4 domain oracles.  The two values are explicitly
# FTheta-own-domain excluded pixels and OpenCV-calibration-domain excluded
# pixels; they are never substituted for the comparison-intersection count.
_DOMAIN_EXCLUDED_BASELINES: dict[str, tuple[int, int]] = {
    "camera_front_wide_120fov": (148, 0),
    "camera_cross_left_120fov": (138, 0),
    "camera_cross_right_120fov": (133, 0),
    "camera_left_wide_90fov": (26_355, 31_681),
    "camera_right_wide_90fov": (44_292, 51_504),
    "camera_back_rear_wide_90fov": (120, 0),
    "camera_rear_left_70fov": (101, 0),
}

_SUPPORTED_MODEL_TYPE = "OpenCVPinholeCameraModel"
_SUPPORTED_PARAMETERS_TYPE = "OpenCVPinholeCameraModelParameters"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parameter_dict(camera: dict) -> dict:
    return {
        key: value
        for key, value in camera.items()
        if key not in {"model_type", "parameters_type"}
    }


def evaluate_quality_warnings(metrics: dict) -> dict:
    """Evaluate accepted residual thresholds without blocking execution."""
    checks: dict[str, bool] = {}
    not_applicable: list[str] = []
    for key, threshold in QUALITY_WARNING_THRESHOLDS.items():
        value = float(metrics[key])
        if math.isfinite(value):
            checks[key] = value < threshold
        elif key == "outer_p99_deg" and not metrics["outer_available"]:
            checks[key] = True
            not_applicable.append(key)
        else:
            checks[key] = False
    warnings = [key for key, passed in checks.items() if not passed]
    return {
        "has_warnings": bool(warnings),
        "checks": checks,
        "warning_metrics": warnings,
        "not_applicable_metrics": not_applicable,
    }


def evaluate_fit_gate(metrics: dict) -> dict:
    """Compatibility wrapper for callers migrating from the v3 gate schema."""
    warnings = evaluate_quality_warnings(metrics)
    return {
        **warnings,
        "passed": not warnings["has_warnings"],
        "failed_metrics": list(warnings["warning_metrics"]),
    }


def _all_finite_numbers(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return all(_all_finite_numbers(item) for item in value)
    if isinstance(value, dict):
        return all(_all_finite_numbers(item) for item in value.values())
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return True
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return isinstance(value, str)


def evaluate_hard_invariants(
    camera_id: str,
    pinhole: dict,
    ftheta: dict,
    metrics: dict,
) -> dict:
    """Evaluate structural/domain invariants that block GPU execution."""
    json_parameters = _jsonable(ftheta)
    loader_contract_error: str | None = None
    try:
        normalized = _validate_ftheta_parameters(camera_id, json_parameters)
    except (TypeError, ValueError) as exc:
        normalized = None
        loader_contract_error = str(exc)

    def finite_vector(name: str, length: int) -> np.ndarray | None:
        value = ftheta.get(name)
        try:
            vector = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if vector.shape != (length,) or not np.isfinite(vector).all():
            return None
        return vector

    forward_poly = finite_vector("angle_to_pixeldist_poly", 6)
    inverse_poly = finite_vector("pixeldist_to_angle_poly", 6)
    resolution = ftheta.get("resolution")
    try:
        resolution_values = list(resolution)
    except (TypeError, ValueError):
        resolution_values = []
    resolution_valid = (
        len(resolution_values) == 2
        and all(
            isinstance(value, (int, np.integer))
            and not isinstance(value, bool)
            and int(value) > 0
            for value in resolution_values
        )
    )
    principal_point = finite_vector("principal_point", 2)
    linear_cde = finite_vector("linear_cde", 3)
    raw_max_angle = ftheta.get("max_angle")
    max_angle = (
        float(raw_max_angle)
        if isinstance(raw_max_angle, (int, float, np.integer, np.floating))
        and not isinstance(raw_max_angle, (bool, np.bool_))
        else float("nan")
    )
    max_angle_in_range = math.isfinite(max_angle) and 0.0 < max_angle < math.pi
    forward_min_derivative = (
        _minimum_polynomial_derivative(forward_poly, max_angle)
        if forward_poly is not None and max_angle_in_range
        else float("nan")
    )
    radius_max = (
        float(np.polynomial.polynomial.polyval(max_angle, forward_poly))
        if forward_poly is not None and max_angle_in_range
        else float("nan")
    )
    inverse_min_derivative = (
        _minimum_polynomial_derivative(inverse_poly, radius_max)
        if inverse_poly is not None and math.isfinite(radius_max) and radius_max > 0.0
        else float("nan")
    )

    checks = {
        "loader_eight_field_contract_valid": normalized is not None,
        "exact_eight_fields": set(ftheta) == set(FTHETA_PARAMETER_KEYS),
        "finite_parameters": _all_finite_numbers(ftheta),
        "resolution_shape_positive_and_preserved": resolution_valid
        and resolution_values == list(pinhole.get("resolution", [])),
        "principal_point_shape_and_preserved": principal_point is not None
        and bool(
            np.array_equal(
                principal_point,
                np.asarray(pinhole.get("principal_point", []), dtype=np.float64),
            )
        ),
        "linear_cde_identity": linear_cde is not None
        and bool(np.array_equal(linear_cde, np.array([1.0, 0.0, 0.0]))),
        "reference_poly_fixed": ftheta.get("reference_poly")
        in _REFERENCE_POLYNOMIAL_TYPES
        and ftheta.get("reference_poly") == "PIXELDIST_TO_ANGLE",
        "shutter_type_supported": ftheta.get("shutter_type") in _SHUTTER_TYPES,
        "polynomial_lengths": forward_poly is not None and inverse_poly is not None,
        "max_angle_finite_range": max_angle_in_range,
        "forward_derivative_positive": math.isfinite(forward_min_derivative)
        and forward_min_derivative > 0.0,
        "inverse_derivative_positive": math.isfinite(inverse_min_derivative)
        and inverse_min_derivative > 0.0,
        "opencv_roundtrip_submicropixel": math.isfinite(
            float(metrics["opencv_roundtrip_max_px"])
        )
        and float(metrics["opencv_roundtrip_max_px"]) < 1e-6,
    }
    not_applicable: list[str] = []
    if camera_id in _DOMAIN_EXCLUDED_BASELINES:
        ftheta_excluded, calibration_excluded = _DOMAIN_EXCLUDED_BASELINES[camera_id]
        checks["ftheta_own_domain_excluded_count_baseline"] = (
            int(metrics["ftheta_own_domain_excluded_count"]) == ftheta_excluded
        )
        checks["opencv_calibration_domain_excluded_count_baseline"] = (
            int(metrics["opencv_calibration_domain_excluded_count"])
            == calibration_excluded
        )
    else:
        checks["ftheta_own_domain_excluded_count_baseline"] = True
        checks["opencv_calibration_domain_excluded_count_baseline"] = True
        not_applicable.extend(
            [
                "ftheta_own_domain_excluded_count_baseline",
                "opencv_calibration_domain_excluded_count_baseline",
            ]
        )
    if camera_id == "camera_front_wide_120fov":
        checks["front_wide_full_domain_sentinel"] = (
            max_angle >= 1.20 and not math.isclose(max_angle, 0.7303101158645611)
        )

    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_invariants": failed,
        "not_applicable_invariants": not_applicable,
        "measurements": {
            "loader_contract_error": loader_contract_error,
            "forward_min_derivative": forward_min_derivative,
            "inverse_min_derivative": inverse_min_derivative,
            "forward_radius_at_max_angle": radius_max,
        },
    }


def evaluate_camera(camera_id: str, pinhole_dict: dict) -> dict:
    """Fit and evaluate one OpenCV pinhole calibration at native resolution."""
    for required in ("model_type", "parameters_type"):
        if required not in pinhole_dict:
            raise ValueError(f"{camera_id}: missing required {required!r}")
    if pinhole_dict["model_type"] != _SUPPORTED_MODEL_TYPE:
        raise ValueError(
            f"{camera_id}: only {_SUPPORTED_MODEL_TYPE} can be fitted; "
            f"got {pinhole_dict['model_type']!r}"
        )
    if pinhole_dict["parameters_type"] != _SUPPORTED_PARAMETERS_TYPE:
        raise ValueError(
            f"{camera_id}: expected {_SUPPORTED_PARAMETERS_TYPE}; "
            f"got {pinhole_dict['parameters_type']!r}"
        )
    pinhole = _parameter_dict(pinhole_dict)
    ftheta = fit_ftheta_from_opencv_rational(pinhole)
    metrics = compute_fullimage_angular_error(pinhole, ftheta)

    hard_invariants = evaluate_hard_invariants(camera_id, pinhole, ftheta, metrics)
    quality_warnings = evaluate_quality_warnings(metrics)
    return {
        "camera_id": camera_id,
        "source_model_type": pinhole_dict["model_type"],
        "source_parameters_type": pinhole_dict["parameters_type"],
        "tangential_coeffs": _jsonable(pinhole.get("tangential_coeffs", [0.0, 0.0])),
        "source_calibration_sha256": _canonical_sha256(pinhole),
        "fitter_version": FITTER_VERSION,
        "ftheta_parameters": _jsonable(ftheta),
        "fit_metrics": _jsonable(metrics),
        "hard_invariants": hard_invariants,
        "quality_warnings": quality_warnings,
    }


def survey_bundle(calibration_bundle: dict) -> dict:
    cameras = calibration_bundle["cameras"]
    order = calibration_bundle.get("camera_order", list(cameras))
    if len(order) != 9 or len(set(order)) != 9:
        raise ValueError(f"expected exactly nine unique camera IDs; got {order}")
    missing = set(order) - set(cameras)
    if missing:
        raise ValueError(f"camera_order references missing calibrations: {sorted(missing)}")
    expected_order = list(ACTIVE_CAMERA_ORDER + EXCLUDED_FROM_RUNTIME_CAMERA_ORDER)
    if order != expected_order:
        raise ValueError(
            "camera_order must preserve the frozen active-seven then excluded-two "
            f"scope; expected {expected_order}, got {order}"
        )

    results: dict[str, dict] = {}
    for index, camera_id in enumerate(order, start=1):
        print(f"[{index}/9] surveying {camera_id}", file=sys.stderr, flush=True)
        results[camera_id] = evaluate_camera(camera_id, cameras[camera_id])
    hard_failures = {
        camera_id: results[camera_id]["hard_invariants"]["failed_invariants"]
        for camera_id in order
        if not results[camera_id]["hard_invariants"]["passed"]
    }
    active_hard_failures = {
        camera_id: hard_failures[camera_id]
        for camera_id in ACTIVE_CAMERA_ORDER
        if camera_id in hard_failures
    }
    warning_cameras = [
        camera_id
        for camera_id in order
        if results[camera_id]["quality_warnings"]["has_warnings"]
    ]
    return {
        "schema_version": 2,
        "provenance": calibration_bundle["provenance"],
        "fitter_version": FITTER_VERSION,
        "quality_warning_thresholds": QUALITY_WARNING_THRESHOLDS,
        "active_camera_order": list(ACTIVE_CAMERA_ORDER),
        "excluded_from_runtime_camera_order": list(
            EXCLUDED_FROM_RUNTIME_CAMERA_ORDER
        ),
        "scope": {
            "decision": "SEVEN_CAMERA_PROCEED_WITH_WARNINGS",
            "nine_camera_survey_role": "CALIBRATION_EVIDENCE_ONLY",
            "runtime_gpu_subset_camera_count": len(ACTIVE_CAMERA_ORDER),
            "nine_camera_training_approved": False,
        },
        "v3_invalidation": {
            "fitter_version": "pin-ftheta-numpy-v3-physical-domain-2026-07-17",
            "front_wide_max_angle_rad": 0.7303101158645611,
            "seven_camera_runtime_artifact_sha256": "73965c6d10693b7742c7afb538169990a94d2457890738a7a4f66d13fcd0a450",
            "status": "INVALID_FOR_V4_RUNTIME_GPU_DECISION",
        },
        "camera_order": order,
        "all_hard_invariants_passed": not hard_failures,
        "hard_failures": hard_failures,
        "active_subset_hard_invariants_passed": not active_hard_failures,
        "active_hard_failures": active_hard_failures,
        "quality_warning_cameras": warning_cameras,
        "cameras": results,
    }


def render_markdown(survey: dict) -> str:
    def stats(metrics: dict, prefix: str, suffix: str) -> str:
        values = []
        for name in ("mean", "p50", "p95", "p99", "max"):
            value = metrics[f"{prefix}{name}{suffix}"]
            values.append("N/A" if value is None else f"{value:.5f}")
        return "/".join(values)

    lines = [
        "# PIN-FTHETA 9-Camera Parameter Survey",
        "",
        "## Provenance",
        "",
        f"- Clip: `{survey['provenance']['clip_id']}`",
        f"- Manifest SHA-256: `{survey['provenance']['manifest_sha256']}`",
        f"- Fitter: `{survey['fitter_version']}`",
        "- Evaluation: every native-resolution integer pixel, all azimuths; no spatial downsampling.",
        "- Calibration validity: the complete first monotonic/invertible rational branch from the optical axis. The Pinhole renderer's `0.8 < icD < 1.2` runtime gate is deliberately not applied to FTheta fitting or validation.",
        "- Regions fixed before evaluation: center `r<0.5`, periphery `r>=0.9`, with `r` normalized by image half-diagonal.",
        "- Coverage is measured against the full calibration branch and the native image raster. A roughly 63% wide-camera result is a hard failure, not an expected limitation.",
        "",
        "## Scope and v3 Invalidation",
        "",
        "- This nine-camera survey is calibration evidence only. It does not approve a nine-camera runtime or GPU experiment.",
        "- The runtime/GPU subset contains exactly seven cameras: "
        + ", ".join(f"`{camera_id}`" for camera_id in survey["active_camera_order"])
        + ".",
        "- Excluded from runtime/GPU: "
        + ", ".join(
            f"`{camera_id}`"
            for camera_id in survey["excluded_from_runtime_camera_order"]
        )
        + ". Front-standard and front-tele remain survey-only calibration evidence.",
        "- v3 is invalid for the v4 decision: fitter `pin-ftheta-numpy-v3-physical-domain-2026-07-17`, front-wide `max_angle=0.7303101158645611 rad`, runtime artifact SHA-256 `73965c6d10693b7742c7afb538169990a94d2457890738a7a4f66d13fcd0a450`.",
        "",
        "## Quality Warning Thresholds",
        "",
        "| Metric | Strict threshold |",
        "|---|---:|",
    ]
    for key, threshold in survey["quality_warning_thresholds"].items():
        lines.append(f"| `{key}` | < {threshold} |")
    lines.extend(
        [
            "",
            "## Per-Camera Result",
            "",
            "| Camera | p1 | p2 | nonradial mean/max deg | angular mean/p50/p95/p99/max deg | pixel mean/p50/p95/p99/max px | forward max px | Hard / Quality |",
            "|---|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for camera_id in survey["camera_order"]:
        row = survey["cameras"][camera_id]
        metrics = row["fit_metrics"]
        # p1/p2 are represented by the frozen calibration rather than copied
        # into the fitted parameter artifact; load them from survey provenance
        # is deliberately avoided, so expose their magnitude in source metrics
        # added by main below if present.
        p1, p2 = row["tangential_coeffs"]
        hard = "🟢" if row["hard_invariants"]["passed"] else "🔴"
        quality = "⚠️" if row["quality_warnings"]["has_warnings"] else "clear"
        lines.append(
            f"| `{camera_id}` | {p1:.3e} | {p2:.3e} | "
            f"{metrics['nonradial_floor_mean_deg']:.5f}/{metrics['nonradial_floor_max_deg']:.5f} | "
            f"{stats(metrics, '', '_deg')} | "
            f"{stats(metrics, 'pixel_', '_px')} | "
            f"{metrics['forward_poly_max_px']:.4f} | {hard} / {quality} |"
        )

    lines.extend(["", "## Center and Periphery", ""])
    lines.extend(
        [
            "| Camera | center mean/p50/p95/p99/max deg | peripheral mean/p50/p95/p99/max deg | center mean/p50/p95/p99/max px | peripheral mean/p50/p95/p99/max px |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for camera_id in survey["camera_order"]:
        metrics = survey["cameras"][camera_id]["fit_metrics"]
        lines.append(
            f"| `{camera_id}` | {stats(metrics, 'center_', '_deg')} | "
            f"{stats(metrics, 'peripheral_', '_deg')} | "
            f"{stats(metrics, 'center_pixel_', '_px')} | "
            f"{stats(metrics, 'peripheral_pixel_', '_px')} |"
        )
    lines.extend(["", "## Domain Counts and OpenCV Round-Trip", ""])
    lines.extend(
        [
            "| Camera | FTheta own domain kept / excluded / coverage | OpenCV calibration domain kept / excluded / coverage | comparison intersection kept / excluded / coverage | OpenCV round-trip mean/p50/p95/p99/max px | outer samples |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for camera_id in survey["camera_order"]:
        metrics = survey["cameras"][camera_id]["fit_metrics"]
        roundtrip = "/".join(
            f"{metrics[f'opencv_roundtrip_{name}_px']:.3e}"
            for name in ("mean", "p50", "p95", "p99", "max")
        )
        lines.append(
            f"| `{camera_id}` | "
            f"{metrics['ftheta_own_domain_count']} / {metrics['ftheta_own_domain_excluded_count']} / {metrics['ftheta_own_domain_coverage']:.6f} | "
            f"{metrics['opencv_calibration_domain_count']} / {metrics['opencv_calibration_domain_excluded_count']} / {metrics['opencv_calibration_domain_coverage']:.6f} | "
            f"{metrics['comparison_intersection_count']} / {metrics['comparison_intersection_excluded_count']} / {metrics['comparison_intersection_coverage']:.6f} | "
            f"{roundtrip} | "
            f"{metrics['outer_sample_count']} |"
        )
    tele = survey["cameras"].get("camera_front_tele_30fov")
    if tele is not None:
        tele_metrics = tele["fit_metrics"]
        lines.extend(
            [
                "",
                "## Tele Regression Diagnosis",
                "",
                "The previously observed `8059 px` tele forward residual was a numerical artifact from fitting the pixel-radius Vandermonde in raw units (the `r^5` column is ill-conditioned). It is not a valid physical-branch calibration error. With both primary and fallback least-squares paths normalized, and with later rational roots excluded, the deterministic tele result is "
                f"`{tele_metrics['forward_poly_max_px']:.4f} px`. This is far smaller than 8059 px but remains a reported quality warning. Tele is excluded from the selected seven-camera experiment, and this numerical warning is not a runtime camera-model fallback or a hard v4 invariant failure.",
            ]
        )
    decision = (
        "SEVEN-CAMERA PROCEED WITH WARNINGS"
        if survey["active_subset_hard_invariants_passed"]
        else "SEVEN-CAMERA STOP"
    )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"**{decision}.** Active-subset hard-failure cameras: "
            + (
                ", ".join(f"`{x}`" for x in survey["active_hard_failures"])
                or "none"
            )
            + ".",
            "",
            "Quality-warning cameras: "
            + (
                ", ".join(f"`{x}`" for x in survey["quality_warning_cameras"])
                or "none"
            )
            + ".",
            "",
            "Only active-seven hard invariant failures block this GPU experiment. Residual threshold exceedances remain serialized and visible warnings under the user-approved seven-camera v4 approximation. Front-standard and front-tele remain excluded, and this decision is not nine-camera approval.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibrations",
        type=Path,
        default=_REPO_ROOT / "scripts" / "pin_ftheta_b6a9_calibs.json",
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    with args.calibrations.open() as handle:
        calibrations = json.load(handle)
    survey = survey_bundle(calibrations)
    if args.format == "json":
        rendered = json.dumps(survey, indent=2, sort_keys=True, allow_nan=False) + "\n"
    else:
        rendered = render_markdown(survey)
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(f"wrote {args.output}", file=sys.stderr)
    return 0 if survey["active_subset_hard_invariants_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
