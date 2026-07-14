from __future__ import annotations

import numpy as np
import pytest

from threedgrut_playground.utils.camera_render_state import (
    CameraModelKind,
    interpolate_c2w,
    resolve_camera_render_state,
)


def _poses_identity_to_180deg() -> np.ndarray:
    poses = np.tile(np.eye(4, dtype=np.float64)[None], (2, 1, 1))
    poses[1, :3, :3] = np.diag([-1.0, -1.0, 1.0])
    poses[1, :3, 3] = [10.0, 0.0, 0.0]
    return poses


def _ftheta_dict() -> dict:
    return {
        "resolution": np.array([6, 4], dtype=np.int64),
        "shutter_type": "GLOBAL",
        "principal_point": np.array([3.0, 2.0], dtype=np.float32),
        "reference_poly": "ANGLE_TO_PIXELDIST",
        "pixeldist_to_angle_poly": np.array([0.0, 0.01], dtype=np.float32),
        "angle_to_pixeldist_poly": np.array([0.0, 100.0], dtype=np.float32),
        "max_angle": 1.5,
        "linear_cde": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }


def _opencv_dict() -> dict:
    return {
        "resolution": np.array([6, 4], dtype=np.int64),
        "shutter_type": "GLOBAL",
        "principal_point": np.array([3.0, 2.0], dtype=np.float32),
        "focal_length": np.array([4.0, 4.0], dtype=np.float32),
        "radial_coeffs": np.zeros(6, dtype=np.float32),
        "tangential_coeffs": np.zeros(2, dtype=np.float32),
        "thin_prism_coeffs": np.zeros(4, dtype=np.float32),
    }


def _base_entry() -> dict:
    return {
        "c2w": _poses_identity_to_180deg().astype(np.float32),
        "timestamps_us": np.array([0, 1_000_000], dtype=np.int64),
        "resolution": (6, 4),
        "fov_y_rad": 1.0,
        "ftheta_dict": None,
        "opencv_pinhole_dict": None,
        "opencv_pinhole_rays": None,
    }


def _entry_ftheta() -> dict:
    entry = _base_entry()
    entry["ftheta_dict"] = _ftheta_dict()
    return entry


def _entry_opencv() -> dict:
    entry = _base_entry()
    entry["opencv_pinhole_dict"] = _opencv_dict()
    entry["opencv_pinhole_rays"] = np.zeros((4, 6, 3), dtype=np.float32)
    return entry


def test_resolve_ftheta_sets_only_ftheta_fields():
    state = resolve_camera_render_state("fish", _entry_ftheta(), 500_000)
    assert state.model_kind is CameraModelKind.FTHETA
    assert state.ftheta_dict is not None
    assert state.opencv_pinhole_dict is None
    assert state.opencv_pinhole_rays is None


def test_resolve_opencv_sets_only_opencv_fields():
    state = resolve_camera_render_state("wide", _entry_opencv(), 500_000)
    assert state.model_kind is CameraModelKind.OPENCV_PINHOLE
    assert state.ftheta_dict is None
    assert state.opencv_pinhole_dict is not None
    assert state.opencv_pinhole_rays is not None


def test_interpolate_c2w_midpoint_lerps_translation_and_slerps_rotation():
    sample = interpolate_c2w(_poses_identity_to_180deg(), np.array([0, 1_000_000]), 500_000)
    np.testing.assert_allclose(sample.c2w[:3, 3], [5.0, 0.0, 0.0], atol=1e-6)
    forward = sample.c2w[:3, :3] @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(forward[:2], [0.0, 1.0], atol=1e-5)
    assert sample.left_idx == 0 and sample.right_idx == 1
    assert sample.alpha == pytest.approx(0.5)
    assert sample.interpolated is True


def test_interpolate_c2w_reports_large_source_gap():
    poses = np.tile(np.eye(4, dtype=np.float64)[None], (2, 1, 1))
    sample = interpolate_c2w(poses, np.array([0, 600_000]), 300_000)
    assert sample.source_gap_us == 600_000
    assert sample.nearest_dt_us == 300_000


def test_resolve_rejects_ftheta_and_opencv_both_set():
    entry = _entry_ftheta()
    entry["opencv_pinhole_dict"] = _opencv_dict()
    entry["opencv_pinhole_rays"] = np.zeros((4, 6, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_camera_render_state("bad", entry, 0)
