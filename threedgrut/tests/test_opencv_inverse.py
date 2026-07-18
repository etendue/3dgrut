# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the complete OpenCV rational-model inverse oracle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from threedgrut_playground.utils.opencv_inverse import (
    invert_opencv_full_model,
    opencv_pixels_to_camera_rays,
)
from threedgrut_playground.utils.pinhole_projector import PinholeForwardProjector


_B6A9_FRONT_WIDE_PINHOLE: dict = {
    "resolution": np.array([1920, 1080], dtype=np.int64),
    "shutter_type": "ROLLING_TOP_TO_BOTTOM",
    "principal_point": np.array([960.8599853515625, 540.1849975585938]),
    "focal_length": np.array([952.8250122070312, 952.9000244140625]),
    "radial_coeffs": np.array(
        [
            3.7687599658966064,
            1.61149001121521,
            0.0664215013384819,
            4.13346004486084,
            2.880429983139038,
            0.36570900678634644,
        ]
    ),
    "tangential_coeffs": np.array(
        [4.691869980888441e-05, 8.77050024428172e-06]
    ),
    "thin_prism_coeffs": np.zeros(4),
}


def _frozen_camera(camera_id: str) -> dict:
    path = Path(__file__).resolve().parents[2] / "scripts" / "pin_ftheta_b6a9_calibs.json"
    bundle = json.loads(path.read_text())
    return {
        key: value
        for key, value in bundle["cameras"][camera_id].items()
        if key not in {"model_type", "parameters_type"}
    }


@pytest.fixture(scope="module")
def b6a9_fullimage_inverse():
    width, height = _B6A9_FRONT_WIDE_PINHOLE["resolution"]
    ys, xs = np.mgrid[0:int(height), 0:int(width)]
    uv = np.stack([xs.ravel(), ys.ravel()], axis=-1)
    return invert_opencv_full_model(_B6A9_FRONT_WIDE_PINHOLE, uv)


def test_fullimage_newton_residual_is_machine_precision_on_physical_domain(
    b6a9_fullimage_inverse,
):
    xy, residual = b6a9_fullimage_inverse
    valid = np.isfinite(residual)
    assert 0.62 < float(np.mean(valid)) < 0.65
    assert float(residual[valid].max()) < 1e-12
    assert np.isnan(xy[~valid]).all()
    assert np.isinf(residual[~valid]).all()


def test_forward_then_inverse_recovers_normalized_coordinates():
    xs = np.linspace(-0.6, 0.6, 31)
    ys = np.linspace(-0.3, 0.3, 19)
    xg, yg = np.meshgrid(xs, ys)
    normalized = np.stack([xg.ravel(), yg.ravel()], axis=-1)
    points = np.column_stack(
        [normalized[:, 0], normalized[:, 1], np.ones(len(normalized))]
    )
    projector = PinholeForwardProjector(_B6A9_FRONT_WIDE_PINHOLE)
    uv, _visible = projector.project_points(points, np.eye(4))

    recovered, residual = invert_opencv_full_model(
        _B6A9_FRONT_WIDE_PINHOLE, uv
    )

    np.testing.assert_allclose(recovered, normalized, rtol=0.0, atol=1e-10)
    assert float(residual.max()) < 1e-12


def test_principal_point_inverts_to_optical_axis():
    principal_point = np.asarray(
        _B6A9_FRONT_WIDE_PINHOLE["principal_point"], dtype=np.float64
    )[None, :]
    xy, residual = invert_opencv_full_model(
        _B6A9_FRONT_WIDE_PINHOLE, principal_point
    )
    np.testing.assert_allclose(xy, [[0.0, 0.0]], atol=1e-12)
    assert residual[0] < 1e-12


def test_camera_ray_grid_shape_and_unit_norm():
    rays = opencv_pixels_to_camera_rays(_B6A9_FRONT_WIDE_PINHOLE)
    assert rays.shape == (1080, 1920, 3)
    assert rays.dtype == np.float64
    valid = np.isfinite(rays).all(axis=-1)
    assert 0.62 < float(np.mean(valid)) < 0.65
    np.testing.assert_allclose(np.linalg.norm(rays[valid], axis=-1), 1.0, atol=1e-12)
    assert np.isnan(rays[~valid]).all()


def test_calibration_inverse_is_not_truncated_by_runtime_icd_gate():
    """FTheta conversion uses the full first branch, not pinhole visibility."""
    rays = opencv_pixels_to_camera_rays(
        _B6A9_FRONT_WIDE_PINHOLE,
        enforce_runtime_trust=False,
    )
    valid = np.isfinite(rays).all(axis=-1)
    assert float(np.mean(valid)) == pytest.approx(1.0)
    np.testing.assert_allclose(
        np.linalg.norm(rays[valid], axis=-1),
        1.0,
        atol=1e-12,
    )
    assert np.isfinite(rays[0, 0]).all()
    assert np.isfinite(rays[-1, -1]).all()


@pytest.mark.parametrize(
    "camera_id",
    [
        "camera_front_wide_120fov",
        "camera_cross_left_120fov",
        "camera_cross_right_120fov",
        "camera_left_wide_90fov",
        "camera_right_wide_90fov",
        "camera_back_rear_wide_90fov",
        "camera_rear_left_70fov",
    ],
)
def test_all_seven_calibration_domains_explicitly_exclude_runtime_trust(camera_id):
    """Every active FTheta calibration call must opt out of the icD gate."""
    camera = _frozen_camera(camera_id)
    width, height = camera["resolution"]
    ys, xs = np.mgrid[0:int(height):8, 0:int(width):8]
    uv = np.stack([xs.ravel(), ys.ravel()], axis=-1)
    runtime_xy, runtime_residual = invert_opencv_full_model(camera, uv)
    calibration_xy, calibration_residual = invert_opencv_full_model(
        camera,
        uv,
        enforce_runtime_trust=False,
    )
    runtime_valid = np.isfinite(runtime_residual)
    calibration_valid = np.isfinite(calibration_residual)
    runtime_coverage = float(np.mean(runtime_valid))
    calibration_coverage = float(np.mean(calibration_valid))

    assert runtime_coverage < 0.70, camera_id
    assert calibration_coverage > 0.97, camera_id
    assert calibration_coverage > runtime_coverage + 0.30, camera_id
    assert np.isnan(runtime_xy[~runtime_valid]).all()
    assert np.isnan(calibration_xy[~calibration_valid]).all()
    assert float(calibration_residual[calibration_valid].max()) < 1e-10

    # Side-wide calibration branches still exclude folded/non-positive-
    # Jacobian corners; opting out of icD never means accepting later roots.
    if camera_id in {
        "camera_left_wide_90fov",
        "camera_right_wide_90fov",
    }:
        assert calibration_coverage < 0.995


def test_zero_nonradial_terms_match_radial_reference():
    radial_only = dict(_B6A9_FRONT_WIDE_PINHOLE)
    radial_only["tangential_coeffs"] = np.zeros(2)
    radial_only["thin_prism_coeffs"] = np.zeros(4)

    # Independent dense radial inversion: interpolate the exact forward
    # radius curve, then compare its rays with the 2-D Newton implementation.
    width, height = radial_only["resolution"]
    fx, fy = np.asarray(radial_only["focal_length"], dtype=np.float64)
    cx, cy = np.asarray(radial_only["principal_point"], dtype=np.float64)
    ys, xs = np.mgrid[0:int(height):16, 0:int(width):16]
    uv = np.stack([xs.ravel(), ys.ravel()], axis=-1).astype(np.float64)
    xy, residual = invert_opencv_full_model(radial_only, uv)

    xd = (uv[:, 0] - cx) / fx
    yd = (uv[:, 1] - cy) / fy
    rd = np.hypot(xd, yd)
    # The far image corners invert to ru>2 for this rational calibration.
    ru_grid = np.linspace(0.0, 4.0, 400_001)
    k1, k2, k3, k4, k5, k6 = radial_only["radial_coeffs"]
    r2 = ru_grid * ru_grid
    scale = (1 + k1*r2 + k2*r2**2 + k3*r2**3) / (
        1 + k4*r2 + k5*r2**2 + k6*r2**3
    )
    rd_grid = ru_grid * scale
    ru = np.interp(rd, rd_grid, ru_grid)
    radial_xy = np.zeros_like(xy)
    nonzero = rd > 0
    radial_xy[nonzero, 0] = xd[nonzero] * ru[nonzero] / rd[nonzero]
    radial_xy[nonzero, 1] = yd[nonzero] * ru[nonzero] / rd[nonzero]
    valid = np.isfinite(residual)
    xy = xy[valid]
    radial_xy = radial_xy[valid]
    rays_newton = np.column_stack([xy, np.ones(len(xy))])
    rays_newton /= np.linalg.norm(rays_newton, axis=1, keepdims=True)
    rays_radial = np.column_stack([radial_xy, np.ones(len(radial_xy))])
    rays_radial /= np.linalg.norm(rays_radial, axis=1, keepdims=True)
    dot = np.sum(rays_newton * rays_radial, axis=1)
    cross = np.linalg.norm(np.cross(rays_newton, rays_radial), axis=1)
    angles = np.arctan2(cross, dot)
    assert float(angles.max()) < 1e-8


@pytest.mark.parametrize(
    ("camera_id", "coverage_range"),
    [
        ("camera_left_wide_90fov", (0.62, 0.68)),
        ("camera_right_wide_90fov", (0.62, 0.68)),
        ("camera_front_tele_30fov", (0.999, 1.001)),
    ],
)
def test_wrong_rational_branches_are_not_accepted(camera_id, coverage_range):
    camera = _frozen_camera(camera_id)
    width, height = camera["resolution"]
    ys, xs = np.mgrid[0:int(height):8, 0:int(width):8]
    uv = np.stack([xs.ravel(), ys.ravel()], axis=-1)
    xy, residual = invert_opencv_full_model(camera, uv)
    valid = np.isfinite(residual)
    coverage = float(np.mean(valid))
    assert coverage_range[0] <= coverage <= coverage_range[1]
    assert np.isnan(xy[~valid]).all()
    assert np.isinf(residual[~valid]).all()
    if "wide" in camera_id:
        assert np.count_nonzero(~valid) > 0
