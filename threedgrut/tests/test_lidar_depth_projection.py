# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LiDAR → image-plane projection (Stage 11 T11.B1).

Pure geometry tests with synthetic intrinsics — does NOT load NCore SDK.
"""

import numpy as np
import pytest

from scripts.dump_lidar_depth_map import project_pinhole, ray_depth_from_cam_pts


def test_ray_depth_is_norm_not_z():
    """ray-depth = ‖cam_pts‖, NOT cam_pts.z. drivestudio L759-822 invariant."""
    cam_pts = np.array([[3.0, 4.0, 0.0]])  # x=3, y=4, z=0 → norm=5, z=0
    rd = ray_depth_from_cam_pts(cam_pts)
    assert rd[0] == pytest.approx(5.0, abs=1e-6)


def test_project_pinhole_principal_point():
    """主光轴点 (0, 0, 1) 投到 (cx, cy)。"""
    cam_pts = np.array([[0.0, 0.0, 1.0]])
    intrinsics = dict(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0)
    uv, valid = project_pinhole(cam_pts, intrinsics, (1080, 1920))
    assert valid[0]
    assert uv[0, 0] == pytest.approx(960.0, abs=1e-3)
    assert uv[0, 1] == pytest.approx(540.0, abs=1e-3)


def test_project_pinhole_behind_camera_invalid():
    """z<=0 的点 valid=False。"""
    cam_pts = np.array([[0.0, 0.0, -1.0]])
    intrinsics = dict(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0)
    uv, valid = project_pinhole(cam_pts, intrinsics, (1080, 1920))
    assert not valid[0]


def test_project_pinhole_outside_image_invalid():
    """投到 image 外的点 valid=False。"""
    cam_pts = np.array([[10.0, 10.0, 1.0]])  # fx*x/z = 10000 远超 W=1920
    intrinsics = dict(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0)
    uv, valid = project_pinhole(cam_pts, intrinsics, (1080, 1920))
    assert not valid[0]


def test_multi_point_to_same_pixel_takes_nearest():
    """两个点投到同一像素时，depth_map 取最近的。"""
    from scripts.dump_lidar_depth_map import scatter_depth_map

    uv = np.array([[100.0, 100.0], [100.4, 100.3]])  # 同 floor 像素
    ray_d = np.array([20.0, 5.0])  # 第二个更近
    valid = np.array([True, True])
    dmap = scatter_depth_map(uv, ray_d, valid, H=200, W=200)
    assert dmap[100, 100] == pytest.approx(5.0, abs=1e-6)


def test_scatter_all_invalid_returns_zeros():
    """A frame where 0 LiDAR points are valid → all-zero depth map, no crash."""
    from scripts.dump_lidar_depth_map import scatter_depth_map

    uv = np.array([[50.0, 50.0], [10.0, 10.0]])
    ray_d = np.array([10.0, 5.0])
    valid = np.array([False, False])
    dmap = scatter_depth_map(uv, ray_d, valid, H=100, W=100)
    assert dmap.sum() == 0.0
    assert dmap.shape == (100, 100)


def test_scatter_ignores_nan_in_invalid_points():
    """NaN ray_depth on an invalid point must not corrupt the valid point's pixel."""
    from scripts.dump_lidar_depth_map import scatter_depth_map

    uv = np.array([[50.0, 50.0], [50.0, 50.0]])  # both map to pixel (50,50)
    ray_d = np.array([np.nan, 7.0])  # invalid point carries NaN
    valid = np.array([False, True])
    dmap = scatter_depth_map(uv, ray_d, valid, H=100, W=100)
    assert dmap[50, 50] == 7.0  # valid point's depth, unperturbed
