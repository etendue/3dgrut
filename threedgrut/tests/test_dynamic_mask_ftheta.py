# SPDX-License-Identifier: Apache-2.0
"""B3/Phase B-B2 — FTheta polynomial branch of project_cuboids_to_mask.

The training-side cuboid mask must match the FTheta intrinsics that the
backdrop renderer uses; pinhole AABB collapses to image edges past ±90° FOV
and paints whole columns as dyn, defeating the layer-pruning loss.

The FTheta branch is exercised here with a synthetic but non-linear polynomial
``r_pix = 400 * angle - 100 * angle^2`` (cm at angle≈π/2 ≈ 382 ≪ pinhole
``tan(π/2)*500 = ∞``); the relationship is monotone over [0, π/2] which
mirrors the well-behaved real NCore poly without coupling tests to a specific
camera calibration.
"""
from __future__ import annotations

import math

import pytest
import torch

from threedgrut.layers.dynamic_mask import (
    _corners_to_pixels_ftheta,
    _horner_ascending_torch,
    _normalize_ftheta_params,
    project_cuboids_to_mask,
)


# --- shared fixtures -------------------------------------------------------

def _identity_pose() -> torch.Tensor:
    return torch.eye(4)


def _pinhole_K(fx=400.0, fy=400.0, cx=256.0, cy=256.0) -> torch.Tensor:
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _linear_ftheta(fx_equiv: float = 400.0, cx: float = 256.0, cy: float = 256.0):
    """Linear FTheta poly r_pix = fx_equiv * angle → behaves like a pinhole
    at small angles (r ≈ fx * tan(angle) ≈ fx * angle for small angle)."""
    return {
        "angle_to_pixeldist_poly": [0.0, fx_equiv],
        "principal_point": [cx, cy],
    }


def _nonlinear_ftheta(cx: float = 256.0, cy: float = 256.0):
    """r_pix = 400*angle - 100*angle^2, monotone over [0, π/2], asymptotes
    much earlier than tan(angle)."""
    return {
        "angle_to_pixeldist_poly": [0.0, 400.0, -100.0],
        "principal_point": [cx, cy],
    }


# --- _horner_ascending_torch ----------------------------------------------

def test_horner_ascending_linear():
    poly = torch.tensor([0.0, 5.0])
    x = torch.tensor([0.0, 1.0, 2.0, 3.0])
    out = _horner_ascending_torch(poly, x)
    assert torch.allclose(out, torch.tensor([0.0, 5.0, 10.0, 15.0]))


def test_horner_ascending_quadratic():
    poly = torch.tensor([1.0, 2.0, 3.0])  # 1 + 2x + 3x^2
    x = torch.tensor([0.0, 1.0, 2.0])
    out = _horner_ascending_torch(poly, x)
    # 1+0+0=1, 1+2+3=6, 1+4+12=17
    assert torch.allclose(out, torch.tensor([1.0, 6.0, 17.0]))


def test_horner_constant():
    poly = torch.tensor([7.0])
    x = torch.tensor([0.0, 100.0, -3.0])
    out = _horner_ascending_torch(poly, x)
    assert torch.allclose(out, torch.full_like(x, 7.0))


# --- _normalize_ftheta_params ---------------------------------------------

def test_normalize_ftheta_from_list():
    n = _normalize_ftheta_params(
        {"angle_to_pixeldist_poly": [0.0, 500.0],
         "principal_point": [256.0, 256.0]}, device="cpu",
    )
    assert torch.is_tensor(n["angle_to_pixeldist_poly"])
    assert n["angle_to_pixeldist_poly"].dtype == torch.float32
    assert n["principal_point"].tolist() == [256.0, 256.0]


def test_normalize_ftheta_from_numpy():
    import numpy as np
    raw = {
        "angle_to_pixeldist_poly": np.asarray([0.0, 500.0], dtype=np.float64),
        "principal_point": np.asarray([256.0, 256.0], dtype=np.float64),
    }
    n = _normalize_ftheta_params(raw, device="cpu")
    assert n["angle_to_pixeldist_poly"].dtype == torch.float32
    assert n["principal_point"].dtype == torch.float32


# --- _corners_to_pixels_ftheta micro -------------------------------------

def test_ftheta_corner_on_axis_lands_at_principal_point():
    # Single corner exactly on optical axis: x=0, y=0, z=1.
    corners = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])  # [1, 1, 4]
    fp = _normalize_ftheta_params(_linear_ftheta(), device="cpu")
    u, v = _corners_to_pixels_ftheta(corners, fp)
    assert torch.allclose(u, torch.tensor([[256.0]]))
    assert torch.allclose(v, torch.tensor([[256.0]]))


def test_ftheta_corner_off_axis_matches_polynomial():
    # Corner at x=1, y=0, z=1 → r_xy=1, angle=π/4
    corners = torch.tensor([[[1.0, 0.0, 1.0, 1.0]]])
    fp = _normalize_ftheta_params(_linear_ftheta(fx_equiv=500.0), device="cpu")
    u, v = _corners_to_pixels_ftheta(corners, fp)
    # r_pix = 500 * π/4 ≈ 392.7, u_off = 392.7 * 1/sqrt(1) = 392.7
    expected_u = 256.0 + 500.0 * math.pi / 4
    assert torch.allclose(u, torch.tensor([[expected_u]]), atol=1e-3)
    assert torch.allclose(v, torch.tensor([[256.0]]), atol=1e-3)


# --- project_cuboids_to_mask FTheta path ----------------------------------

def test_ftheta_basic_centered_box_produces_mask():
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([0.0, 0.0, 5.0])
    sizes = torch.tensor([[1.0, 1.0, 1.0]])
    mask = project_cuboids_to_mask(
        pose.unsqueeze(0), sizes,
        K=None, T_world2cam=_identity_pose(),
        H=512, W=512, device="cpu",
        ftheta_params=_linear_ftheta(),
    )
    assert mask.dtype == torch.bool
    # Linear FTheta with fx=400 behaves like pinhole-400 at small angles
    # (half-extent 0.5m at z=5m → half-width ≈ 40 px → AABB ~80×80 = 6400 px).
    n = int(mask.sum().item())
    assert 4500 <= n <= 12000, f"expected ~6400 px mask area, got {n}"


def test_ftheta_vs_pinhole_overlap_at_center_small_angle():
    """Same cuboid + same image → FTheta(linear) ≈ pinhole near optical axis."""
    pose = torch.eye(4); pose[:3, 3] = torch.tensor([0.0, 0.0, 5.0])
    sizes = torch.tensor([[1.0, 1.0, 1.0]])
    K = _pinhole_K(fx=400.0, fy=400.0)
    mask_pin = project_cuboids_to_mask(
        pose.unsqueeze(0), sizes, K, _identity_pose(), H=512, W=512, device="cpu",
    )
    mask_ftheta = project_cuboids_to_mask(
        pose.unsqueeze(0), sizes, None, _identity_pose(), H=512, W=512, device="cpu",
        ftheta_params=_linear_ftheta(fx_equiv=400.0),
    )
    # IoU > 0.7 at small angle; pinhole uses tan(θ), ftheta uses θ — tiny diff
    inter = int((mask_pin & mask_ftheta).sum().item())
    union = int((mask_pin | mask_ftheta).sum().item())
    iou = inter / max(union, 1)
    assert iou > 0.7, f"FTheta vs pinhole IoU={iou:.3f} at center too low"


def test_ftheta_off_axis_aabb_stays_bounded_while_pinhole_spans_edge():
    """Cube at (4, 0, 1) → angle≈76° from axis.

    Pinhole: u = fx * 4 / 1 = 4*400 = 1600 → clamps to W-1, AABB spans hundreds of cols.
    FTheta (non-linear): r_pix(angle≈1.33) = 400*1.33 - 100*1.33^2 = 532 - 177 = 355,
        u_off ≈ 355 * (4/4) = 355, u ≈ 256 + 355 = 611 → still slightly clamps
        but doesn't blow up. AABB area is meaningfully smaller.
    """
    pose = torch.eye(4); pose[:3, 3] = torch.tensor([4.0, 0.0, 1.0])
    sizes = torch.tensor([[0.5, 0.5, 0.5]])
    K = _pinhole_K(fx=400.0, fy=400.0)
    mask_pin = project_cuboids_to_mask(
        pose.unsqueeze(0), sizes, K, _identity_pose(), H=512, W=512, device="cpu",
    )
    mask_ftheta = project_cuboids_to_mask(
        pose.unsqueeze(0), sizes, None, _identity_pose(), H=512, W=512, device="cpu",
        ftheta_params=_nonlinear_ftheta(),
    )
    n_pin = int(mask_pin.sum().item())
    n_ft = int(mask_ftheta.sum().item())
    # FTheta produces a finite, smaller AABB; pinhole bigger (often clamped).
    assert n_pin > n_ft, (
        f"expected pinhole AABB ({n_pin}) > FTheta AABB ({n_ft}) at large off-axis"
    )


def test_ftheta_all_corners_behind_camera_skips_track():
    """Cube fully behind the camera (z<0): FTheta path skips → empty mask."""
    pose = torch.eye(4); pose[:3, 3] = torch.tensor([0.0, 0.0, -3.0])
    sizes = torch.tensor([[1.0, 1.0, 1.0]])
    mask = project_cuboids_to_mask(
        pose.unsqueeze(0), sizes, None, _identity_pose(),
        H=256, W=256, device="cpu",
        ftheta_params=_linear_ftheta(),
    )
    assert int(mask.sum().item()) == 0


def test_ftheta_partial_behind_camera_uses_visible_corners_only():
    """Cube straddling the image plane: some corners z>0, some z<0.

    With pose translation (0, 0, 0.3) and size (1, 1, 1), corners span z∈[-0.2, 0.8].
    4 corners visible (z=+0.8), 4 behind. AABB only uses the visible 4.
    """
    pose = torch.eye(4); pose[:3, 3] = torch.tensor([0.0, 0.0, 0.3])
    sizes = torch.tensor([[1.0, 1.0, 1.0]])
    mask = project_cuboids_to_mask(
        pose.unsqueeze(0), sizes, None, _identity_pose(),
        H=512, W=512, device="cpu",
        ftheta_params=_linear_ftheta(),
    )
    # Visible corners are z=0.8; symmetric in x, y → AABB centered at principal
    # point with finite extent.
    n = int(mask.sum().item())
    assert n > 0, "expected visible corners to produce non-empty AABB"


def test_ftheta_empty_tracks_returns_zero_mask():
    empty_poses = torch.zeros(0, 4, 4)
    empty_sizes = torch.zeros(0, 3)
    mask = project_cuboids_to_mask(
        empty_poses, empty_sizes, None, _identity_pose(),
        H=128, W=128, device="cpu",
        ftheta_params=_linear_ftheta(),
    )
    assert int(mask.sum().item()) == 0
    assert mask.shape == (128, 128)


def test_ftheta_multiple_tracks_union():
    poses = torch.stack([
        torch.eye(4),
        torch.eye(4),
    ])
    poses[0, :3, 3] = torch.tensor([0.0, 0.0, 5.0])
    poses[1, :3, 3] = torch.tensor([0.5, 0.5, 5.0])
    sizes = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    mask = project_cuboids_to_mask(
        poses, sizes, None, _identity_pose(), H=512, W=512, device="cpu",
        ftheta_params=_linear_ftheta(),
    )
    # Two overlapping cuboids should produce more mask than just one.
    mask_one = project_cuboids_to_mask(
        poses[:1], sizes[:1], None, _identity_pose(), H=512, W=512, device="cpu",
        ftheta_params=_linear_ftheta(),
    )
    assert int(mask.sum().item()) >= int(mask_one.sum().item())


def test_ftheta_rejects_missing_intrinsics():
    poses = torch.eye(4).unsqueeze(0)
    sizes = torch.tensor([[1.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="K or ftheta_params"):
        project_cuboids_to_mask(
            poses, sizes, K=None, T_world2cam=_identity_pose(),
            H=64, W=64, device="cpu",
        )
