# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for off-track novel-view ROTATION modes.

Pins the KPI contract introduced for the road off-track task (translation
3m/6m already covered; rotation gate must be 10°/30°/60°):

  * ``yaw_30deg`` / ``yaw_60deg`` are present in ``NOVEL_VIEW_MODES``.
  * ``LEGACY_NOVEL_AVG_MODES`` is byte-for-byte untouched — the historical
    ``mean_novel_lpips_avg`` anchor (B3 0.5962) depends on those 4 modes
    meaning *exactly* lateral_1m/2m + yaw_5/10deg, forever.
  * ``perturb_c2w`` applies the correct rotation magnitude (parsed from the
    mode name) around the camera up axis and keeps the camera position fixed.
  * lateral modes remain a pure translation (no rotation regression).
"""

from __future__ import annotations

import numpy as np
import pytest

from threedgrut.utils.novel_view import (
    LEGACY_NOVEL_AVG_MODES,
    NOVEL_VIEW_MODES,
    _yaw_deg_from_mode,
    perturb_c2w,
)


def _identity_c2w() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def test_rotation_gate_modes_present() -> None:
    # The task requires rotation at 10/30/60deg. 10deg lives in the legacy set.
    assert "yaw_10deg" in NOVEL_VIEW_MODES
    assert "yaw_30deg" in NOVEL_VIEW_MODES
    assert "yaw_60deg" in NOVEL_VIEW_MODES


def test_legacy_avg_modes_untouched() -> None:
    assert LEGACY_NOVEL_AVG_MODES == (
        "lateral_1m",
        "lateral_2m",
        "yaw_5deg",
        "yaw_10deg",
    )
    # New rotation modes must NOT leak into the legacy avg set (would silently
    # redefine mean_novel_lpips_avg and break the historical anchor).
    assert "yaw_30deg" not in LEGACY_NOVEL_AVG_MODES
    assert "yaw_60deg" not in LEGACY_NOVEL_AVG_MODES


@pytest.mark.parametrize(
    "mode,deg",
    [("yaw_5deg", 5.0), ("yaw_10deg", 10.0), ("yaw_30deg", 30.0), ("yaw_60deg", 60.0)],
)
def test_yaw_deg_parsing(mode: str, deg: float) -> None:
    assert _yaw_deg_from_mode(mode) == deg


@pytest.mark.parametrize("deg", [5.0, 10.0, 30.0, 60.0])
def test_yaw_rotation_magnitude_and_fixed_position(deg: float) -> None:
    mode = f"yaw_{int(deg)}deg"
    c2w = _identity_c2w()
    out = perturb_c2w(c2w, mode)
    # Yaw keeps the camera position fixed (rotate in place).
    np.testing.assert_allclose(out[:3, 3], c2w[:3, 3], atol=1e-9)
    # Forward axis (c2w[:3,2]) must rotate by exactly `deg` around the up axis.
    fwd0, fwd1 = c2w[:3, 2], out[:3, 2]
    cos_ang = np.clip(np.dot(fwd0, fwd1) / (np.linalg.norm(fwd0) * np.linalg.norm(fwd1)), -1.0, 1.0)
    ang = np.degrees(np.arccos(cos_ang))
    np.testing.assert_allclose(ang, deg, atol=1e-6)
    # Rotation must be a proper SO(3) matrix (orthonormal, det +1).
    R = out[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-9)


@pytest.mark.parametrize("mode", ["lateral_1m", "lateral_3m", "lateral_6m"])
def test_lateral_modes_are_pure_translation(mode: str) -> None:
    c2w = _identity_c2w()
    out = perturb_c2w(c2w, mode)
    # Rotation block unchanged for lateral modes.
    np.testing.assert_allclose(out[:3, :3], c2w[:3, :3], atol=1e-12)
    meters = float(mode.split("_")[1].replace("m", ""))
    np.testing.assert_allclose(out[:3, 3], c2w[:3, 3] + meters * c2w[:3, 0], atol=1e-9)


def test_yaw_deg_parsing_rejects_non_yaw() -> None:
    with pytest.raises(ValueError):
        _yaw_deg_from_mode("lateral_3m")
