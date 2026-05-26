# SPDX-License-Identifier: Apache-2.0
"""PinholeForwardProjector unit tests (Mac-runnable, no CUDA/viser).

Pinned by hand-computed expected pixels so any regression to the projection
math (focal length, principal point, distortion, c2w flip) is caught
without GPU / NCore data.
"""
from __future__ import annotations

import numpy as np
import pytest

from threedgrut_playground.utils.ftheta_projector import FLIP_VISER_TO_OPENCV
from threedgrut_playground.utils.pinhole_projector import PinholeForwardProjector


def _identity_pinhole_dict():
    """fx=fy=500, cx=cy=320, no distortion, image 640×480."""
    return {
        "resolution":      np.array([640, 480], dtype=np.int64),
        "principal_point": np.array([320.0, 240.0], dtype=np.float64),
        "focal_length":    np.array([500.0, 500.0], dtype=np.float64),
        "radial_coeffs":   np.array([], dtype=np.float64),
        "tangential_coeffs": np.array([], dtype=np.float64),
    }


def _opencv_identity_c2w():
    """Identity c2w in OpenCV convention. Camera at world origin, +Z forward.
    World point with z>0 is in front of camera.
    """
    return np.eye(4, dtype=np.float64)


# ---- Test 1: optical-axis point hits principal point ---------------------
def test_project_principal_point_default_flip_identity():
    """A point on the +Z optical axis maps to (cx, cy)."""
    proj = PinholeForwardProjector(_identity_pinhole_dict())   # default flip = I
    pts = np.array([[0.0, 0.0, 10.0]])
    uv, vis = proj.project_points(pts, _opencv_identity_c2w())
    assert vis[0]
    assert uv[0, 0] == pytest.approx(320.0, abs=1e-9)
    assert uv[0, 1] == pytest.approx(240.0, abs=1e-9)


# ---- Test 2: square corners at known depth -------------------------------
def test_project_square_corners_no_distortion():
    """A 2×2 m square at depth z=5 should map to cx±200, cy±200 with f=500."""
    proj = PinholeForwardProjector(_identity_pinhole_dict())
    pts = np.array([
        [-2.0, -2.0, 5.0],   # top-left in image (cam +Y is down)
        [+2.0, -2.0, 5.0],
        [-2.0, +2.0, 5.0],
        [+2.0, +2.0, 5.0],
    ])
    uv, vis = proj.project_points(pts, _opencv_identity_c2w())
    assert vis.all()
    # fx * X / Z = 500 * 2 / 5 = 200
    np.testing.assert_allclose(
        uv,
        np.array([
            [320 - 200, 240 - 200],
            [320 + 200, 240 - 200],
            [320 - 200, 240 + 200],
            [320 + 200, 240 + 200],
        ]),
        atol=1e-9,
    )


# ---- Test 3: behind-camera clip ------------------------------------------
def test_behind_camera_clip():
    """Point with z<=0 (behind camera) → visible=False."""
    proj = PinholeForwardProjector(_identity_pinhole_dict())
    pts = np.array([
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0],
    ])
    uv, vis = proj.project_points(pts, _opencv_identity_c2w())
    assert not vis[0]
    assert not vis[1]


# ---- Test 4: out-of-image clip -------------------------------------------
def test_out_of_image_clip():
    """Pixel that lands outside (0, W) × (0, H) → visible=False."""
    proj = PinholeForwardProjector(_identity_pinhole_dict())
    # x/z = 1, so u = 500 * 1 + 320 = 820 > 640
    pts = np.array([[5.0, 0.0, 5.0]])
    uv, vis = proj.project_points(pts, _opencv_identity_c2w())
    assert uv[0, 0] == pytest.approx(820.0, abs=1e-9)
    assert not vis[0]


# ---- Test 5: radial distortion shifts pixel outward ----------------------
def test_radial_distortion_pushes_pixel_outward():
    """k1 > 0 (barrel) on a point off-axis → distorted pixel farther from cx,cy."""
    cfg = _identity_pinhole_dict()
    proj_no = PinholeForwardProjector(cfg)
    cfg_dist = dict(cfg)
    cfg_dist["radial_coeffs"] = np.array([0.1], dtype=np.float64)  # k1=0.1
    proj_dist = PinholeForwardProjector(cfg_dist)
    pts = np.array([[1.0, 1.0, 5.0]])

    uv_no, _ = proj_no.project_points(pts, _opencv_identity_c2w())
    uv_d,  _ = proj_dist.project_points(pts, _opencv_identity_c2w())

    # Both visible; distorted u is farther from cx than undistorted.
    d_no = abs(uv_no[0, 0] - 320.0)
    d_dist = abs(uv_d[0, 0] - 320.0)
    assert d_dist > d_no, f"k1>0 should push outward: d_no={d_no}, d_dist={d_dist}"


# ---- Test 6: tangential distortion adds non-symmetric offset --------------
def test_tangential_distortion_non_symmetric():
    """p1, p2 non-zero on an off-axis point → offset NOT purely along the
    radial direction (this just checks the dist branch runs without nan)."""
    cfg = _identity_pinhole_dict()
    cfg["tangential_coeffs"] = np.array([0.01, -0.02], dtype=np.float64)
    proj = PinholeForwardProjector(cfg)
    pts = np.array([[1.0, 0.5, 5.0]])
    uv, vis = proj.project_points(pts, _opencv_identity_c2w())
    assert vis[0]
    # Undistorted would be u=320+500*0.2=420, v=240+500*0.1=290; with p1/p2
    # the result must differ from that pinhole-pure result.
    assert not np.isclose(uv[0, 0], 420.0)
    assert not np.isclose(uv[0, 1], 290.0)


# ---- Test 7: empty input -------------------------------------------------
def test_project_empty_points():
    """N=0 points input → empty (0, 2) / (0,), no crash."""
    proj = PinholeForwardProjector(_identity_pinhole_dict())
    uv, vis = proj.project_points(np.empty((0, 3)), _opencv_identity_c2w())
    assert uv.shape == (0, 2)
    assert vis.shape == (0,)


# ---- Test 8: viser-flip parity (B2 back-compat) --------------------------
def test_viser_flip_parity_with_opencv_identity():
    """Passing FLIP_VISER_TO_OPENCV with a viser-style c2w should produce
    the same uv as passing identity flip with the matching OpenCV c2w.

    Identity in viser (camera at origin, +Z backward) ≡ identity@diag(1,1,-1,1)
    in OpenCV (camera at origin, +Z forward looking at -Z_world). So a world
    point at z=-10 in viser convention should hit (cx,cy), matching a point
    at z=+10 with identity flip.
    """
    proj_viser = PinholeForwardProjector(
        _identity_pinhole_dict(), world_to_camera_flip=FLIP_VISER_TO_OPENCV)
    proj_cv = PinholeForwardProjector(_identity_pinhole_dict())  # flip = I

    pts_viser_world = np.array([[0.0, 0.0, -10.0]])
    pts_cv_world    = np.array([[0.0, 0.0, +10.0]])

    uv_v, vis_v = proj_viser.project_points(pts_viser_world, np.eye(4))
    uv_c, vis_c = proj_cv.project_points(pts_cv_world, np.eye(4))

    assert vis_v[0] and vis_c[0]
    np.testing.assert_allclose(uv_v, uv_c, atol=1e-9)


# ---- Test 9: project_polylines structure ---------------------------------
def test_project_polylines_split_back():
    """3 polylines in → 3 results out, each with the right subdivided length."""
    proj = PinholeForwardProjector(_identity_pinhole_dict())
    pls = [
        np.array([[0, 0, 5], [1, 0, 5]], dtype=np.float64),
        np.array([[0, 0, 10], [0, 1, 10], [0, 2, 10]], dtype=np.float64),
        np.array([[2, 2, 8]], dtype=np.float64),                            # M=1 edge
    ]
    out = proj.project_polylines(pls, _opencv_identity_c2w(), subdivide_n=4)
    assert len(out) == 3
    assert out[0][0].shape == (1 + 1 * 4, 2)   # M=2 → 5
    assert out[1][0].shape == (1 + 2 * 4, 2)   # M=3 → 9
    assert out[2][0].shape == (1, 2)            # M=1 passthrough


# ---- Test 10: missing required key raises --------------------------------
def test_init_missing_required_key():
    bad = _identity_pinhole_dict()
    del bad["focal_length"]
    with pytest.raises(ValueError, match="missing required keys"):
        PinholeForwardProjector(bad)


# ---- Test 11: scalar focal_length tolerated ------------------------------
def test_scalar_focal_length_accepted():
    """A scalar focal_length (square pixel) should be accepted as fx=fy."""
    cfg = _identity_pinhole_dict()
    cfg["focal_length"] = 500.0
    proj = PinholeForwardProjector(cfg)
    pts = np.array([[0.0, 0.0, 10.0]])
    uv, vis = proj.project_points(pts, _opencv_identity_c2w())
    assert vis[0]
    assert uv[0, 0] == pytest.approx(320.0)
    assert uv[0, 1] == pytest.approx(240.0)
