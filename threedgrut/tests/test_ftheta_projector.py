# SPDX-License-Identifier: Apache-2.0
"""B2: FthetaForwardProjector unit tests (Mac-runnable, no CUDA/viser).

Calibration anchor at the end pins the Phase 0 probe result
(see docs/T8_artifacts/B2_calibration_probe_log.md). If that test
regresses, the convention has drifted and the overlay will be misaligned.
"""
from __future__ import annotations

import numpy as np
import pytest

from threedgrut_playground.utils.ftheta_intrinsics import (
    ftheta_pixels_to_camera_rays,
)
from threedgrut_playground.utils.ftheta_projector import (
    FLIP_VISER_TO_OPENCV,
    FthetaForwardProjector,
    _horner_ascending,
    _subdivide_polyline,
)


# ---- Synthetic FTheta dict for math-layer tests --------------------------
def _synthetic_ftheta_dict():
    """Approximates the Phase 0 probe ckpt's FTheta intrinsics with a
    simple monotonically-increasing poly so inverse + forward roundtrip
    is well-defined for the math tests.
    """
    return {
        "resolution":              np.array([1920, 1080], dtype=np.int64),
        "shutter_type":            "ROLLING_TOP_TO_BOTTOM",
        "principal_point":         np.array([960.0, 540.0], dtype=np.float32),
        "reference_poly":          "ANGLE_TO_PIXELDIST",
        # angle (rad) → r_pix: ~800 * angle (near-linear fisheye for small θ).
        "angle_to_pixeldist_poly": np.array(
            [0.0, 800.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        # r_pix → angle: ~r_pix / 800 (matching inverse for roundtrip test).
        "pixeldist_to_angle_poly": np.array(
            [0.0, 1.0 / 800.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        # Use a generous 89° max_angle in the synthetic dict so the linear
        # poly (which doesn't compress angles at the edge the way NCore's
        # quintic does) still keeps all image corners in FOV for the
        # roundtrip test. NCore's real ckpt uses ~70° + a non-linear poly.
        "max_angle":               np.pi / 2 - 0.01,
        "linear_cde":              np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }


def _identity_c2w_viser():
    """Camera at world origin, viser convention (+Y down + Z backward).

    With FLIP_VISER_TO_OPENCV = diag([1,1,-1,1]) applied, this becomes an
    OpenCV camera at origin looking down +Z_world. A point at
    world (0, 0, +1) is in front of the OpenCV camera at cam-z = -1...
    wait — c2w_cv = identity @ diag([1,1,-1,1]) = diag([1,1,-1,1]); its
    inverse is the same matrix. So world (x,y,z) → cam (x, y, -z). Hence
    a point with world z < 0 is in front of the camera.
    """
    return np.eye(4, dtype=np.float64)


# ---- Test 1: project on optical axis hits principal point ----------------
def test_project_principal_point():
    """Point on the optical axis (camera-frame x=y=0, z>0) maps to (cx, cy)."""
    ftheta = _synthetic_ftheta_dict()
    proj = FthetaForwardProjector(ftheta)
    # Identity viser c2w → OpenCV c2w = diag([1,1,-1,1]); ego looks at -Z_world.
    # World point (0, 0, -10) → cam (0, 0, +10) ✓ on optical axis.
    pts = np.array([[0.0, 0.0, -10.0]])
    uv, vis = proj.project_points(pts, _identity_c2w_viser())
    assert vis[0]
    assert uv[0, 0] == pytest.approx(960.0, abs=1e-6)
    assert uv[0, 1] == pytest.approx(540.0, abs=1e-6)


# ---- Test 2: forward ↔ inverse roundtrip ---------------------------------
def test_project_unproject_roundtrip():
    """For a grid of pixels, unproject → world ray → re-project should
    return the same pixel (within sub-pixel error).
    """
    ftheta = _synthetic_ftheta_dict()
    proj = FthetaForwardProjector(ftheta)
    rays_hw = ftheta_pixels_to_camera_rays(ftheta)        # (H, W, 3)
    H, W = rays_hw.shape[:2]

    # Sample 100 pixels: 10x10 grid avoiding image borders.
    ys = np.linspace(50, H - 50, 10, dtype=int)
    xs = np.linspace(50, W - 50, 10, dtype=int)
    sample_uv = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)

    # Take camera-space ray dir per pixel (already unit norm in +Z OpenCV).
    cam_dirs = rays_hw[sample_uv[:, 1], sample_uv[:, 0], :]  # (N, 3)

    # Build world points 5 m along each ray, expressed in WORLD frame.
    # With identity viser c2w, world (x,y,z) → cam (x, y, -z).
    # So a cam-frame point P_cam = ray * 5 corresponds to world (x, y, -z).
    world_pts = cam_dirs * 5.0
    world_pts[:, 2] *= -1                                   # invert Z back to world

    uv_back, vis = proj.project_points(world_pts, _identity_c2w_viser())
    assert vis.all(), f"all sampled pixels must roundtrip; visible={vis.sum()}/{len(vis)}"

    err = np.abs(uv_back - sample_uv.astype(np.float64))
    # Synthetic linear poly is exact; tolerance is just float roundoff.
    assert err.max() < 1e-3, f"roundtrip error too large: max={err.max():.4f}px"


# ---- Test 3: max_angle clip ----------------------------------------------
def test_max_angle_clip():
    """Ray angle just past max_angle → visible=False (in_fov gate)."""
    ftheta = _synthetic_ftheta_dict()
    proj = FthetaForwardProjector(ftheta)
    over = ftheta["max_angle"] + 0.01
    # In OpenCV cam frame: (sin(angle), 0, cos(angle)) is on the unit ray
    # at the given angle. World point = cam * 10 (then Z-flip back to world).
    cam_pt = np.array([np.sin(over), 0.0, np.cos(over)]) * 10.0
    world_pt = cam_pt.copy()
    world_pt[2] *= -1
    uv, vis = proj.project_points(world_pt[None], _identity_c2w_viser())
    assert not vis[0], f"angle={over:.3f} > max_angle={ftheta['max_angle']:.3f} must clip"


# ---- Test 4: image bound clip --------------------------------------------
def test_image_bound_clip():
    """Pixel that lands outside (0, W) × (0, H) → visible=False."""
    ftheta = _synthetic_ftheta_dict()
    proj = FthetaForwardProjector(ftheta)
    # Pick a small angle but huge lateral offset → r_pix = 800*angle, but
    # the direction normalization sends it past the image edge. Use an
    # in-FOV angle of 1.0 rad with full lateral component.
    angle = 1.0
    cam_pt = np.array([np.sin(angle), 0.0, np.cos(angle)]) * 10.0
    world_pt = cam_pt.copy()
    world_pt[2] *= -1
    uv, vis = proj.project_points(world_pt[None], _identity_c2w_viser())
    # u_off = 800 * 1.0 = 800 px → u = 960 + 800 = 1760 → still in 1920 bound.
    # Force out-of-bound by larger angle staying in FOV:
    angle2 = 1.20  # near 70° but in FOV
    cam_pt2 = np.array([np.sin(angle2), 0.0, np.cos(angle2)]) * 10.0
    world_pt2 = cam_pt2.copy()
    world_pt2[2] *= -1
    uv2, vis2 = proj.project_points(world_pt2[None], _identity_c2w_viser())
    # u = 960 + 800 * 1.20 = 960 + 960 = 1920 → exactly at right edge, OUT of (0, W).
    assert not vis2[0], f"u={uv2[0, 0]:.1f} should fall outside [0, 1920); vis={vis2}"


# ---- Test 5: behind-camera clip ------------------------------------------
def test_behind_camera_clip():
    """Cam-frame z < 0 (point behind camera) → visible=False even if pixel
    lands inside the image (FTheta poly is symmetric in θ vs π-θ for
    cos(θ)/sin(θ); only the z>0 explicit gate distinguishes).
    """
    ftheta = _synthetic_ftheta_dict()
    proj = FthetaForwardProjector(ftheta)
    # World (0, 0, +10) → cam (0, 0, -10) under identity viser c2w. z<0.
    pts = np.array([[0.0, 0.0, +10.0]])
    uv, vis = proj.project_points(pts, _identity_c2w_viser())
    assert not vis[0]


# ---- Test 6: polyline subdivision ----------------------------------------
def test_polyline_subdivision_endpoints_preserved():
    """N-vertex polyline + subdivide_n=k → 1 + (N-1)*k vertices, with
    endpoints exactly preserved.
    """
    pl = np.array([[0.0, 0.0, 0.0],
                   [1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]], dtype=np.float64)         # N=3
    sub = _subdivide_polyline(pl, n=5)
    assert sub.shape == (1 + 2 * 5, 3), f"got {sub.shape}"
    np.testing.assert_array_equal(sub[0], pl[0])
    np.testing.assert_array_equal(sub[-1], pl[-1])
    # n=1 returns unchanged
    sub1 = _subdivide_polyline(pl, n=1)
    np.testing.assert_array_equal(sub1, pl)


def test_polyline_subdivision_intermediate_linear():
    """Subdivided intermediate vertices should be linear interpolants."""
    pl = np.array([[0.0, 0.0, 0.0],
                   [10.0, 0.0, 0.0]], dtype=np.float64)
    sub = _subdivide_polyline(pl, n=10)
    # Should have 11 vertices at x = 0, 1, 2, ..., 10.
    assert sub.shape == (11, 3)
    np.testing.assert_allclose(sub[:, 0], np.linspace(0, 10, 11), atol=1e-9)


# ---- Test 7: calibration anchor (Phase 0 probe pin) -----------------------
def test_calibration_anchor_phase0_combo_d():
    """Phase 0 (2026-05-24 ThinkPad probe) winning combo:
        c2w_cv = c2w_viser @ diag([1, 1, -1, 1])
        poly evaluation = Horner ascending order
        linear_cde = skipped (≈ identity)

    Anchor data from docs/T8_artifacts/B2_calibration_probe.json (cuboid
    tid=41 'automobile', frame_idx=0 in ckpt_with_ftheta_v2.pt). If this
    test regresses, B2 overlay alignment is broken and the cuboid wireframe
    will drift off the truck/car backdrop.
    """
    # Reproduce Phase 0 probe inputs (snapshot from B2_calibration_probe.json):
    ftheta = {
        "resolution":              np.array([1920, 1080], dtype=np.int64),
        "shutter_type":            "ROLLING_TOP_TO_BOTTOM",
        "principal_point":         np.array([960.31537, 545.4275],
                                            dtype=np.float32),
        "reference_poly":          "ANGLE_TO_PIXELDIST",
        "angle_to_pixeldist_poly": np.array(
            [0.0, 927.5706787109375, 5.75466251373291,
             -37.59929656982422, 24.825389862060547,
             -8.416096687316895], dtype=np.float32),
        "pixeldist_to_angle_poly": np.array(
            [0.0, 1.0782e-3, -8.5288e-9, 5.5028e-11,
             -4.1299e-14, 1.6549e-17], dtype=np.float32),
        "max_angle":               1.2205,
        "linear_cde":              np.array([1.0016, 0.0, 0.0],
                                            dtype=np.float32),
    }
    proj = FthetaForwardProjector(ftheta)

    # Identity viser c2w → c2w_cv = diag([1,1,-1,1]); ego looks at world -Z.
    # Place a small fake cuboid at world (0, ±dy, -75) with:
    #   - bottom vertices at world y > 0  (in viser/OpenCV +Y down ⇒ below ego)
    #   - top    vertices at world y < 0  (above ego)
    # After D-combo projection these should have:
    #   - bottom verts → v > cy   (lower on the image)
    #   - top    verts → v < cy   (higher on the image)
    cuboid_verts_world = np.array([
        # bottom (low world y already-down, so cam-frame y > 0 → v > cy)
        [-1.0, +0.7, -75.0],
        [+1.0, +0.7, -75.0],
        # top (high world -y → cam-frame y < 0 → v < cy)
        [-1.0, -0.7, -75.0],
        [+1.0, -0.7, -75.0],
    ], dtype=np.float64)

    uv, vis = proj.project_points(cuboid_verts_world, _identity_c2w_viser())

    # All 4 verts visible (in FOV ~0.5°, well inside image, z > 0).
    assert vis.all(), f"calibration anchor must project all 4 verts; got vis={vis}"

    # Bottom verts (world y > 0) have v > cy; top verts have v < cy.
    v_bottom = uv[:2, 1]
    v_top    = uv[2:, 1]
    assert v_bottom.min() > ftheta["principal_point"][1], (
        f"bottom verts must be below cy={ftheta['principal_point'][1]:.1f}; "
        f"got v_bottom={v_bottom}"
    )
    assert v_top.max() < ftheta["principal_point"][1], (
        f"top verts must be above cy={ftheta['principal_point'][1]:.1f}; "
        f"got v_top={v_top}"
    )

    # u is symmetric (world ±1 → cam ±1), so the 2 left verts should mirror
    # the 2 right verts around cx.
    np.testing.assert_allclose(uv[0, 0] + uv[1, 0],
                               2.0 * ftheta["principal_point"][0], atol=1e-3)

    # Confirm FLIP_VISER_TO_OPENCV is what we pinned in Phase 0.
    expected = np.diag([1.0, 1.0, -1.0, 1.0])
    np.testing.assert_array_equal(FLIP_VISER_TO_OPENCV, expected,
                                  err_msg="FLIP_VISER_TO_OPENCV changed — "
                                          "Phase 0 calibration is invalidated. "
                                          "Re-run scripts/probe_ftheta_overlay.py.")


# ---- Test 8: horner ascending matches naive sum --------------------------
def test_horner_ascending_matches_polyval():
    """_horner_ascending(poly, x) ≡ sum(poly[k] * x^k for k)."""
    poly = np.array([1.0, 2.0, 3.0, 4.0])    # 1 + 2x + 3x² + 4x³
    x = np.array([0.0, 1.0, 0.5, -1.0])
    expected = poly[0] + poly[1] * x + poly[2] * x ** 2 + poly[3] * x ** 3
    actual = _horner_ascending(poly, x)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


# ---- Test 9: empty input ---------------------------------------------------
def test_project_empty_points():
    """N=0 points input → empty (0, 2) uv and (0,) visible, no crash."""
    proj = FthetaForwardProjector(_synthetic_ftheta_dict())
    uv, vis = proj.project_points(np.empty((0, 3)), _identity_c2w_viser())
    assert uv.shape == (0, 2)
    assert vis.shape == (0,)


# ---- Test 10: project_polylines preserves structure -----------------------
def test_project_polylines_split_back():
    """3 polylines in → 3 results out, each with correct subdivided length."""
    proj = FthetaForwardProjector(_synthetic_ftheta_dict())
    pls = [
        np.array([[0, 0, -5], [1, 0, -5]], dtype=np.float64),       # M=2
        np.array([[0, 0, -10], [0, 1, -10], [0, 2, -10]], dtype=np.float64),  # M=3
        np.array([[2, 2, -8]], dtype=np.float64),                    # M=1 edge case
    ]
    out = proj.project_polylines(pls, _identity_c2w_viser(), subdivide_n=4)
    assert len(out) == 3
    assert out[0][0].shape == (1 + 1 * 4, 2)     # M=2 → 5 verts
    assert out[1][0].shape == (1 + 2 * 4, 2)     # M=3 → 9 verts
    assert out[2][0].shape == (1, 2)              # M=1 → 1 vert passthrough


# ---- Test 11: missing required key raises --------------------------------
def test_init_missing_required_key():
    """ftheta_dict without one of the required 4 keys raises ValueError."""
    bad = _synthetic_ftheta_dict()
    del bad["max_angle"]
    with pytest.raises(ValueError, match="missing required keys"):
        FthetaForwardProjector(bad)
