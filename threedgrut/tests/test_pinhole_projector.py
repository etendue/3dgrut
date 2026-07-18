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
        "resolution": np.array([640, 480], dtype=np.int64),
        "principal_point": np.array([320.0, 240.0], dtype=np.float64),
        "focal_length": np.array([500.0, 500.0], dtype=np.float64),
        "radial_coeffs": np.array([], dtype=np.float64),
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
    proj = PinholeForwardProjector(_identity_pinhole_dict())  # default flip = I
    pts = np.array([[0.0, 0.0, 10.0]])
    uv, vis = proj.project_points(pts, _opencv_identity_c2w())
    assert vis[0]
    assert uv[0, 0] == pytest.approx(320.0, abs=1e-9)
    assert uv[0, 1] == pytest.approx(240.0, abs=1e-9)


# ---- Test 2: square corners at known depth -------------------------------
def test_project_square_corners_no_distortion():
    """A 2×2 m square at depth z=5 should map to cx±200, cy±200 with f=500."""
    proj = PinholeForwardProjector(_identity_pinhole_dict())
    pts = np.array(
        [
            [-2.0, -2.0, 5.0],  # top-left in image (cam +Y is down)
            [+2.0, -2.0, 5.0],
            [-2.0, +2.0, 5.0],
            [+2.0, +2.0, 5.0],
        ]
    )
    uv, vis = proj.project_points(pts, _opencv_identity_c2w())
    assert vis.all()
    # fx * X / Z = 500 * 2 / 5 = 200
    np.testing.assert_allclose(
        uv,
        np.array(
            [
                [320 - 200, 240 - 200],
                [320 + 200, 240 - 200],
                [320 - 200, 240 + 200],
                [320 + 200, 240 + 200],
            ]
        ),
        atol=1e-9,
    )


# ---- Test 3: behind-camera clip ------------------------------------------
def test_behind_camera_clip():
    """Point with z<=0 (behind camera) → visible=False."""
    proj = PinholeForwardProjector(_identity_pinhole_dict())
    pts = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
        ]
    )
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
    uv_d, _ = proj_dist.project_points(pts, _opencv_identity_c2w())

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
    proj_viser = PinholeForwardProjector(_identity_pinhole_dict(), world_to_camera_flip=FLIP_VISER_TO_OPENCV)
    proj_cv = PinholeForwardProjector(_identity_pinhole_dict())  # flip = I

    pts_viser_world = np.array([[0.0, 0.0, -10.0]])
    pts_cv_world = np.array([[0.0, 0.0, +10.0]])

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
        np.array([[2, 2, 8]], dtype=np.float64),  # M=1 edge
    ]
    out = proj.project_polylines(pls, _opencv_identity_c2w(), subdivide_n=4)
    assert len(out) == 3
    assert out[0][0].shape == (1 + 1 * 4, 2)  # M=2 → 5
    assert out[1][0].shape == (1 + 2 * 4, 2)  # M=3 → 9
    assert out[2][0].shape == (1, 2)  # M=1 passthrough


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


# ---- Test 12: six radial coefficients use rational numerator/denominator --
def _rational_pinhole_dict():
    """Large image so visibility does not hide numeric differences."""
    return {
        "resolution": np.array([4000, 3000], dtype=np.int64),
        "principal_point": np.array([2000.0, 1500.0]),
        "focal_length": np.array([1000.0, 1000.0]),
        "radial_coeffs": np.array([0.10, 0.02, 0.003, 0.04, 0.005, 0.0006]),
        "tangential_coeffs": np.zeros(2),
        "thin_prism_coeffs": np.zeros(4),
    }


def test_six_radial_coefficients_use_rational_denominator():
    """OpenCV rational model: icD = (1 + k1·r² + k2·r⁴ + k3·r⁶) /
                                 (1 + k4·r² + k5·r⁴ + k6·r⁶)

    Current code treats k4-k6 as polynomial powers; this test should RED.
    """
    cfg = _rational_pinhole_dict()
    proj = PinholeForwardProjector(cfg)
    point = np.array([[0.5, 0.25, 1.0]])

    uv, visible = proj.project_points(point, np.eye(4))

    x, y = 0.5, 0.25
    r2 = x * x + y * y
    k1, k2, k3, k4, k5, k6 = cfg["radial_coeffs"]
    numerator = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    denominator = 1.0 + r2 * (k4 + r2 * (k5 + r2 * k6))
    scale = numerator / denominator
    expected = np.array(
        [[2000.0 + 1000.0 * x * scale, 1500.0 + 1000.0 * y * scale]]
    )
    np.testing.assert_allclose(uv, expected, atol=1e-9)
    assert visible[0]


# ---- Test 13: denominator-near-boundary regression test -------------------
# ---- Test 14: thin-prism coefficients match OpenCV model ------------------
def test_thin_prism_coefficients_match_opencv_model():
    """Thin-prism terms add r²·(s1 + r²·s2) in x and r²·(s3 + r²·s4) in y.

    Current code ignores thin_prism_coeffs; this test should RED.
    """
    cfg = _identity_pinhole_dict()
    cfg["thin_prism_coeffs"] = np.array([0.01, -0.002, -0.015, 0.003])
    proj = PinholeForwardProjector(cfg)
    point = np.array([[1.0, 0.5, 5.0]])

    uv, visible = proj.project_points(point, np.eye(4))

    x, y = 0.2, 0.1
    r2 = x * x + y * y
    dx = r2 * (0.01 + r2 * -0.002)
    dy = r2 * (-0.015 + r2 * 0.003)
    expected = np.array([[320.0 + 500.0 * (x + dx), 240.0 + 500.0 * (y + dy)]])
    np.testing.assert_allclose(uv, expected, atol=1e-9)
    assert visible[0]


# ---- Test 15: radial scale outside NCore trust interval is invalid ---------
def test_radial_scale_outside_ncore_trust_interval_is_invalid():
    """icD > 1.2 (strong barrel) must mark point invisible.

    Current code has no trust gate; this test should RED.
    """
    cfg = _identity_pinhole_dict()
    cfg["resolution"] = np.array([4000, 3000])
    cfg["principal_point"] = np.array([2000.0, 1500.0])
    cfg["radial_coeffs"] = np.array([0.3, 0, 0, 0, 0, 0])
    proj = PinholeForwardProjector(cfg)
    _, visible = proj.project_points(np.array([[1.0, 0.0, 1.0]]), np.eye(4))
    assert not visible[0], "icD > 1.2 should be invalid"


def test_radial_scale_below_ncore_trust_interval_is_invalid():
    """icD < 0.8 (strong pincushion) must mark point invisible."""
    cfg = _identity_pinhole_dict()
    cfg["resolution"] = np.array([4000, 3000])
    cfg["principal_point"] = np.array([2000.0, 1500.0])
    cfg["radial_coeffs"] = np.array([-0.3, 0, 0, 0, 0, 0])
    proj = PinholeForwardProjector(cfg)
    _, visible = proj.project_points(np.array([[1.0, 0.0, 1.0]]), np.eye(4))
    assert not visible[0], "icD < 0.8 should be invalid"


def test_certified_max_valid_r2_replaces_legacy_icd_trust_gate():
    """PIN-CAM-1c supplies a calibrated ideal-radius certificate to the CUDA
    renderer.  CPU projection users (A3/overlays) must use the same domain,
    otherwise they silently drop valid wide-camera edge projections merely
    because icD exceeds the legacy 0.8..1.2 heuristic.
    """
    cfg = _identity_pinhole_dict()
    cfg["resolution"] = np.array([4000, 3000])
    cfg["principal_point"] = np.array([2000.0, 1500.0])
    cfg["radial_coeffs"] = np.array([0.3, 0, 0, 0, 0, 0])
    cfg["max_valid_r2"] = 1.1
    proj = PinholeForwardProjector(cfg)

    _, visible = proj.project_points(
        np.array(
            [
                [1.0, 0.0, 1.0],  # r2=1.00, icD=1.30: certified valid
                [1.1, 0.0, 1.0],  # r2=1.21: outside certificate
            ]
        ),
        np.eye(4),
    )
    assert visible.tolist() == [True, False]


# ---- Test 16: short array compatibility -----------------------------------
def test_missing_radial_coeffs_ideal_pinhole():
    """Zero radial coefficients -> ideal pinhole."""
    cfg = _identity_pinhole_dict()
    del cfg["radial_coeffs"]
    proj = PinholeForwardProjector(cfg)
    uv, vis = proj.project_points(np.array([[0.0, 0.0, 10.0]]), np.eye(4))
    assert vis[0]
    np.testing.assert_allclose(uv[0], [320.0, 240.0], atol=1e-9)


def test_single_radial_coeff_as_numerator():
    """[k1] should behave as numerator k1, denominator zeros."""
    cfg = _identity_pinhole_dict()
    cfg["radial_coeffs"] = np.array([0.1])
    proj = PinholeForwardProjector(cfg)
    uv, vis = proj.project_points(np.array([[1.0, 1.0, 5.0]]), np.eye(4))
    assert vis[0]
    # With polynomial model r=1+k1*r², numerator/denominator=1+k1*r² too
    # since denominator = 1 + 0 = 1. Both models agree for single coeff.
    x, y = 0.2, 0.2
    r2 = x * x + y * y
    expected_u = 320.0 + 500.0 * x * (1.0 + 0.1 * r2)
    expected_v = 240.0 + 500.0 * y * (1.0 + 0.1 * r2)
    np.testing.assert_allclose(uv[0], [expected_u, expected_v], atol=1e-9)


def test_too_many_radial_coeffs_raises():
    """More than 6 radial coefficients should raise ValueError."""
    cfg = _identity_pinhole_dict()
    cfg["radial_coeffs"] = np.arange(7, dtype=np.float64)
    with pytest.raises(ValueError, match="6"):
        PinholeForwardProjector(cfg)


def test_missing_tangential_coeffs_ideal_pinhole():
    """Missing tangential key -> treat as zeros, ideal pinhole."""
    cfg = _identity_pinhole_dict()
    del cfg["tangential_coeffs"]
    proj = PinholeForwardProjector(cfg)
    uv, vis = proj.project_points(np.array([[0.0, 0.0, 10.0]]), np.eye(4))
    assert vis[0]
    np.testing.assert_allclose(uv[0], [320.0, 240.0], atol=1e-9)


def test_empty_thin_prism_tolerated():
    """Missing or empty thin_prism_coeffs still produces pinhole projection."""
    cfg = _identity_pinhole_dict()
    proj = PinholeForwardProjector(cfg)
    uv, vis = proj.project_points(np.array([[0.0, 0.0, 10.0]]), np.eye(4))
    assert vis[0]
    np.testing.assert_allclose(uv[0], [320.0, 240.0], atol=1e-9)


def test_rational_denominator_near_unity_next():
    """A config where icD ~ 1.0 but denominator != 1 must hit exact pixels.

    This prevents an implementation that merely ignores coefficients 4-6
    from passing.  Neither the numerator nor denominator equals 1.0, but
    the ratio should yield the same result as an ideal pinhole.
    """
    cfg = _rational_pinhole_dict()
    # k1=0.2, k2=0.0, k3=0.0, k4=0.2, k5=0.0, k6=0.0
    cfg["radial_coeffs"] = np.array([0.2, 0.0, 0.0, 0.2, 0.0, 0.0])
    proj = PinholeForwardProjector(cfg)
    point = np.array([[0.5, 0.25, 1.0]])

    uv, visible = proj.project_points(point, np.eye(4))

    # icD = (1 + 0.2·r²) / (1 + 0.2·r²) = 1.0 exactly
    # so the result must equal ideal pinhole
    x, y = 0.5, 0.25
    expected = np.array([[2000.0 + 1000.0 * x, 1500.0 + 1000.0 * y]])
    np.testing.assert_allclose(uv, expected, atol=1e-9)
    assert visible[0]
