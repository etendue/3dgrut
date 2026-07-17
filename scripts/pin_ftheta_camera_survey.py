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
    compute_fullimage_angular_error,
    fit_ftheta_from_opencv_rational,
)

FITTER_VERSION = "pin-ftheta-numpy-v3-physical-domain-2026-07-17"

# Predeclared before the nine-camera survey is executed.  The first two are
# the PIN-FTHETA-4 representation gate; the angular limits are PIN-FTHETA-3's
# full-image regression envelope.  An outer-angle limit is applied only when a
# camera actually reaches 55 degrees.
FIT_GATE_THRESHOLDS: dict[str, float] = {
    "nonradial_floor_mean_deg": 0.01,
    "forward_poly_max_px": 1.5,
    "mean_deg": 0.02,
    "p95_deg": 0.04,
    "p99_deg": 0.08,
    "max_deg": 0.15,
    "outer_p99_deg": 0.10,
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


def evaluate_fit_gate(metrics: dict) -> dict:
    """Apply the predeclared gate; only unavailable outer-P99 may be NaN."""
    checks: dict[str, bool] = {}
    not_applicable: list[str] = []
    for key, threshold in FIT_GATE_THRESHOLDS.items():
        value = float(metrics[key])
        if math.isfinite(value):
            checks[key] = value < threshold
        elif key == "outer_p99_deg" and not metrics["outer_available"]:
            checks[key] = True
            not_applicable.append(key)
        else:
            checks[key] = False
    failed = [key for key, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_metrics": failed,
        "not_applicable_metrics": not_applicable,
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

    gate = evaluate_fit_gate(metrics)
    return {
        "camera_id": camera_id,
        "source_model_type": pinhole_dict["model_type"],
        "source_parameters_type": pinhole_dict["parameters_type"],
        "tangential_coeffs": _jsonable(pinhole.get("tangential_coeffs", [0.0, 0.0])),
        "source_calibration_sha256": _canonical_sha256(pinhole),
        "fitter_version": FITTER_VERSION,
        "ftheta_parameters": _jsonable(ftheta),
        "fit_metrics": _jsonable(metrics),
        "gate": gate,
    }


def survey_bundle(calibration_bundle: dict) -> dict:
    cameras = calibration_bundle["cameras"]
    order = calibration_bundle.get("camera_order", list(cameras))
    if len(order) != 9 or len(set(order)) != 9:
        raise ValueError(f"expected exactly nine unique camera IDs; got {order}")
    missing = set(order) - set(cameras)
    if missing:
        raise ValueError(f"camera_order references missing calibrations: {sorted(missing)}")

    results: dict[str, dict] = {}
    for index, camera_id in enumerate(order, start=1):
        print(f"[{index}/9] surveying {camera_id}", file=sys.stderr, flush=True)
        results[camera_id] = evaluate_camera(camera_id, cameras[camera_id])
    failed = [camera_id for camera_id in order if not results[camera_id]["gate"]["passed"]]
    return {
        "schema_version": 1,
        "provenance": calibration_bundle["provenance"],
        "fitter_version": FITTER_VERSION,
        "fit_gate_thresholds": FIT_GATE_THRESHOLDS,
        "camera_order": order,
        "all_cameras_passed": not failed,
        "failed_cameras": failed,
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
        "- OpenCV validity: NCore `0.8 < icD < 1.2` and only the first monotonic/invertible branch from the optical axis. Later low-residual roots are invalid.",
        "- Regions fixed before evaluation: center `r<0.5`, periphery `r>=0.9`, with `r` normalized by image half-diagonal.",
        "- Coverage is reported against that physical OpenCV domain. The roughly 63% wide-camera coverage is expected and is not compared with an idealized 100% image domain.",
        "",
        "## Declared Gate",
        "",
        "| Metric | Strict threshold |",
        "|---|---:|",
    ]
    for key, threshold in survey["fit_gate_thresholds"].items():
        lines.append(f"| `{key}` | < {threshold} |")
    lines.extend(
        [
            "",
            "## Per-Camera Result",
            "",
            "| Camera | p1 | p2 | nonradial mean/max deg | angular mean/p50/p95/p99/max deg | pixel mean/p50/p95/p99/max px | forward max px | physical/retained coverage | invalid pixels | Gate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
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
        gate = "🟢" if row["gate"]["passed"] else "🔴"
        lines.append(
            f"| `{camera_id}` | {p1:.3e} | {p2:.3e} | "
            f"{metrics['nonradial_floor_mean_deg']:.5f}/{metrics['nonradial_floor_max_deg']:.5f} | "
            f"{stats(metrics, '', '_deg')} | "
            f"{stats(metrics, 'pixel_', '_px')} | "
            f"{metrics['forward_poly_max_px']:.4f} | "
            f"{metrics['opencv_inverse_coverage']:.6f}/{metrics['physical_domain_retention']:.6f} | "
            f"{metrics['invalid_pixel_count']} | {gate} |"
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
    lines.extend(["", "## Inverse Round-Trip and Invalid Coverage", ""])
    lines.extend(
        [
            "| Camera | OpenCV round-trip p50/p95/p99/max px | OpenCV valid/invalid | comparison valid/invalid | outer samples |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for camera_id in survey["camera_order"]:
        metrics = survey["cameras"][camera_id]["fit_metrics"]
        roundtrip = "/".join(
            f"{metrics[f'opencv_roundtrip_{name}_px']:.3e}"
            for name in ("mean", "p50", "p95", "p99", "max")
        )
        lines.append(
            f"| `{camera_id}` | {roundtrip} | "
            f"{metrics['opencv_inverse_valid_count']}/{metrics['opencv_inverse_invalid_count']} | "
            f"{metrics['valid_pixel_count']}/{metrics['invalid_pixel_count']} | "
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
                f"`{tele_metrics['forward_poly_max_px']:.4f} px`. This is far smaller than 8059 px but still exceeds the predeclared `<1.5 px` gate, so tele remains a real representation blocker.",
            ]
        )
    decision = "PASS" if survey["all_cameras_passed"] else "STOP"
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"**{decision}.** Failed cameras: "
            + (", ".join(f"`{x}`" for x in survey["failed_cameras"]) or "none")
            + ".",
            "",
            "A STOP result blocks GPU training until the failed representation gate is explicitly resolved.",
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
    return 0 if survey["all_cameras_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
