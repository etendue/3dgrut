# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light validation contract shared by FTheta export and loading."""

from __future__ import annotations

import math
from typing import Any

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
_REFERENCE_POLYNOMIAL_TYPES = frozenset(
    {"PIXELDIST_TO_ANGLE", "ANGLE_TO_PIXELDIST"}
)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _finite_vector(value: Any, field: str, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise TypeError(
            f"{field} must be a JSON array with exactly {length} entries"
        )
    return [
        _finite_number(element, f"{field}[{index}]")
        for index, element in enumerate(value)
    ]


def _validate_ftheta_parameters(camera_id: str, value: Any) -> dict[str, Any]:
    """Normalize the exact eight-field JSON contract used by the loader."""
    if not isinstance(value, dict):
        raise TypeError(f"camera '{camera_id}' parameters must be a JSON object")

    actual_keys = set(value)
    missing = sorted(FTHETA_PARAMETER_KEYS - actual_keys)
    unexpected = sorted(actual_keys - FTHETA_PARAMETER_KEYS)
    if missing or unexpected:
        raise ValueError(
            f"camera '{camera_id}' FTheta keys invalid: "
            f"missing={missing}, unexpected={unexpected}"
        )

    resolution = value["resolution"]
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(
            isinstance(element, bool)
            or not isinstance(element, int)
            or element <= 0
            for element in resolution
        )
    ):
        raise TypeError("resolution must be a JSON array of two positive integers")

    shutter_type = value["shutter_type"]
    if not isinstance(shutter_type, str) or shutter_type not in _SHUTTER_TYPES:
        raise ValueError(f"shutter_type must be one of {sorted(_SHUTTER_TYPES)}")

    reference_poly = value["reference_poly"]
    if (
        not isinstance(reference_poly, str)
        or reference_poly not in _REFERENCE_POLYNOMIAL_TYPES
    ):
        raise ValueError(
            "reference_poly must be one of "
            f"{sorted(_REFERENCE_POLYNOMIAL_TYPES)}"
        )

    max_angle = _finite_number(value["max_angle"], "max_angle")
    if max_angle <= 0.0:
        raise ValueError("max_angle must be positive")

    return {
        "resolution": [int(resolution[0]), int(resolution[1])],
        "shutter_type": shutter_type,
        "principal_point": _finite_vector(
            value["principal_point"], "principal_point", 2
        ),
        "reference_poly": reference_poly,
        "pixeldist_to_angle_poly": _finite_vector(
            value["pixeldist_to_angle_poly"], "pixeldist_to_angle_poly", 6
        ),
        "angle_to_pixeldist_poly": _finite_vector(
            value["angle_to_pixeldist_poly"], "angle_to_pixeldist_poly", 6
        ),
        "max_angle": max_angle,
        "linear_cde": _finite_vector(value["linear_cde"], "linear_cde", 3),
    }
