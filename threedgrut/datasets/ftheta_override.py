# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict loading and construction helpers for explicit NCore FTheta overrides."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

FTHETA_PARAMETER_KEYS = frozenset(
    {
        "resolution",
        "shutter_type",
        "principal_point",
        "reference_poly",
        "pixeldist_to_angle_poly",
        "angle_to_pixeldist_poly",
        "max_angle",
        "linear_cde",
    }
)

_SHUTTER_TYPES = frozenset(
    {
        "ROLLING_TOP_TO_BOTTOM",
        "ROLLING_LEFT_TO_RIGHT",
        "ROLLING_BOTTOM_TO_TOP",
        "ROLLING_RIGHT_TO_LEFT",
        "GLOBAL",
    }
)
_REFERENCE_POLYNOMIAL_TYPES = frozenset({"PIXELDIST_TO_ANGLE", "ANGLE_TO_PIXELDIST"})


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key '{key}'")
        result[key] = value
    return result


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _finite_vector(value: Any, field: str, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise TypeError(f"{field} must be a JSON array with exactly {length} entries")
    return [_finite_number(element, f"{field}[{index}]") for index, element in enumerate(value)]


def _validate_ftheta_parameters(camera_id: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"camera '{camera_id}' parameters must be a JSON object")

    actual_keys = set(value)
    missing = sorted(FTHETA_PARAMETER_KEYS - actual_keys)
    unexpected = sorted(actual_keys - FTHETA_PARAMETER_KEYS)
    if missing or unexpected:
        raise ValueError(f"camera '{camera_id}' FTheta keys invalid: missing={missing}, unexpected={unexpected}")

    resolution = value["resolution"]
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(isinstance(element, bool) or not isinstance(element, int) or element <= 0 for element in resolution)
    ):
        raise TypeError("resolution must be a JSON array of two positive integers")

    shutter_type = value["shutter_type"]
    if not isinstance(shutter_type, str) or shutter_type not in _SHUTTER_TYPES:
        raise ValueError(f"shutter_type must be one of {sorted(_SHUTTER_TYPES)}")

    reference_poly = value["reference_poly"]
    if not isinstance(reference_poly, str) or reference_poly not in _REFERENCE_POLYNOMIAL_TYPES:
        raise ValueError(f"reference_poly must be one of {sorted(_REFERENCE_POLYNOMIAL_TYPES)}")

    max_angle = _finite_number(value["max_angle"], "max_angle")
    if max_angle <= 0.0:
        raise ValueError("max_angle must be positive")

    return {
        "resolution": [int(resolution[0]), int(resolution[1])],
        "shutter_type": shutter_type,
        "principal_point": _finite_vector(value["principal_point"], "principal_point", 2),
        "reference_poly": reference_poly,
        "pixeldist_to_angle_poly": _finite_vector(value["pixeldist_to_angle_poly"], "pixeldist_to_angle_poly", 6),
        "angle_to_pixeldist_poly": _finite_vector(value["angle_to_pixeldist_poly"], "angle_to_pixeldist_poly", 6),
        "max_angle": max_angle,
        "linear_cde": _finite_vector(value["linear_cde"], "linear_cde", 3),
    }


def _fingerprint(parameters: Mapping[str, Any]) -> str:
    canonical = json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_ftheta_override_parameters(
    path: str | Path,
    camera_ids: Iterable[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Load an exact camera-id-to-eight-field mapping with no fallback semantics."""

    selected = list(camera_ids)
    if len(selected) != len(set(selected)):
        duplicates = sorted({camera_id for camera_id in selected if selected.count(camera_id) > 1})
        raise ValueError(f"duplicate selected camera ID(s): {duplicates}")
    if not all(isinstance(camera_id, str) and camera_id for camera_id in selected):
        raise TypeError("selected camera IDs must be non-empty strings")

    artifact_path = Path(path).expanduser()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"FTheta parameter artifact is not a file: {artifact_path}")
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid FTheta parameter JSON at {artifact_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise TypeError("FTheta parameter artifact root must be a camera-id mapping")

    selected_set = set(selected)
    artifact_set = set(payload)
    if artifact_set != selected_set:
        raise ValueError(
            "FTheta camera set mismatch: "
            f"missing={sorted(selected_set - artifact_set)}, unexpected={sorted(artifact_set - selected_set)}"
        )

    normalized = {camera_id: _validate_ftheta_parameters(camera_id, payload[camera_id]) for camera_id in selected}
    fingerprints = {camera_id: _fingerprint(normalized[camera_id]) for camera_id in selected}
    return normalized, fingerprints


def build_ftheta_camera_model_parameters(parameters: Mapping[str, Any], *, ncore_data):
    """Build the public NCore FTheta parameter dataclass from a validated dictionary."""

    parameter_type = ncore_data.FThetaCameraModelParameters
    try:
        shutter_type = getattr(ncore_data.ShutterType, parameters["shutter_type"])
        reference_poly = getattr(parameter_type.PolynomialType, parameters["reference_poly"])
    except AttributeError as exc:
        raise ValueError(f"unsupported NCore FTheta enum value: {exc}") from exc

    return parameter_type(
        resolution=np.asarray(parameters["resolution"], dtype=np.uint64),
        shutter_type=shutter_type,
        external_distortion_parameters=None,
        principal_point=np.asarray(parameters["principal_point"], dtype=np.float32),
        reference_poly=reference_poly,
        pixeldist_to_angle_poly=np.asarray(parameters["pixeldist_to_angle_poly"], dtype=np.float32),
        angle_to_pixeldist_poly=np.asarray(parameters["angle_to_pixeldist_poly"], dtype=np.float32),
        max_angle=float(parameters["max_angle"]),
        linear_cde=np.asarray(parameters["linear_cde"], dtype=np.float32),
    )


def build_ftheta_camera_model(
    parameters: Mapping[str, Any],
    *,
    camera_id: str,
    ncore_data,
    ncore_sensors,
    target_resolution: tuple[int, int] | None = None,
):
    """Construct a runtime FTheta model and reject every implicit fallback.

    The injectable NCore surfaces keep this glue unit-testable on CPU-only
    hosts while production passes ``ncore.data`` and ``ncore.sensors``.
    """

    model_parameters = build_ftheta_camera_model_parameters(parameters, ncore_data=ncore_data)
    if target_resolution is not None:
        model_parameters = transform_camera_model_parameters(model_parameters, target_resolution)
    camera_model = ncore_sensors.CameraModel.from_parameters(
        model_parameters,
        device="cpu",
        dtype=torch.float32,
    )
    if not isinstance(camera_model, ncore_sensors.FThetaCameraModel):
        raise TypeError(
            f"FTheta override for camera '{camera_id}' constructed unexpected model "
            f"{type(camera_model).__name__}; refusing native/ideal-pinhole fallback"
        )
    return camera_model


def transform_camera_model_parameters(model_parameters, target_resolution: tuple[int, int]):
    """Scale camera parameters through NCore's image-domain transform contract."""

    target_w, target_h = target_resolution
    if (
        isinstance(target_w, bool)
        or isinstance(target_h, bool)
        or not isinstance(target_w, int)
        or not isinstance(target_h, int)
        or target_w <= 0
        or target_h <= 0
    ):
        raise ValueError(f"target_resolution must contain positive integers, got {target_resolution}")

    source_w = int(model_parameters.resolution[0])
    source_h = int(model_parameters.resolution[1])
    image_domain_scale = (target_w / source_w, target_h / source_h)
    return model_parameters.transform(
        image_domain_scale=image_domain_scale,
        new_resolution=(target_w, target_h),
    )


def extract_ftheta_camera_model_parameters(
    camera_model,
    target_resolution: tuple[int, int],
    *,
    ncore_sensors,
) -> tuple[dict[str, Any], str]:
    """Scale and extract the exact eight-field tracer FTheta contract."""

    if not isinstance(camera_model, ncore_sensors.FThetaCameraModel):
        raise TypeError(
            f"expected FThetaCameraModel, got {type(camera_model).__name__}; refusing pinhole/fisheye fallback"
        )
    scaled_parameters = transform_camera_model_parameters(
        camera_model.get_parameters(),
        target_resolution,
    )
    parameters_dict = {
        "resolution": scaled_parameters.resolution,
        "shutter_type": scaled_parameters.shutter_type.name,
        "principal_point": scaled_parameters.principal_point,
        "reference_poly": scaled_parameters.reference_poly.name,
        "pixeldist_to_angle_poly": scaled_parameters.pixeldist_to_angle_poly,
        "angle_to_pixeldist_poly": scaled_parameters.angle_to_pixeldist_poly,
        "max_angle": scaled_parameters.max_angle,
        "linear_cde": scaled_parameters.linear_cde,
    }
    if set(parameters_dict) != FTHETA_PARAMETER_KEYS:  # defensive contract guard
        raise AssertionError("internal FTheta extraction did not produce the exact eight-field contract")
    return parameters_dict, type(scaled_parameters).__name__


def add_intrinsics_to_batch_dict(
    batch_dict: dict[str, Any],
    intrinsics_result: tuple[dict[str, Any], str] | None,
) -> None:
    """Populate the tracer field selected by the NCore parameter type name."""

    if intrinsics_result is None:
        return
    intrinsics_parameters, model_type_name = intrinsics_result
    batch_dict[f"intrinsics_{model_type_name}"] = intrinsics_parameters
