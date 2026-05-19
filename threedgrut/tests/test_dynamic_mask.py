# SPDX-License-Identifier: Apache-2.0
"""T4.4 unit tests for dynamic_mask.project_cuboids_to_mask.

Pure CPU mock; AABB pixel-count sanity (D5: 10-15% overestimate is acceptable).
"""
from __future__ import annotations

import torch

from threedgrut.layers.dynamic_mask import project_cuboids_to_mask


def _identity_pose() -> torch.Tensor:
    return torch.eye(4)


def _K(fx: float = 500., fy: float = 500., cx: float = 256., cy: float = 256.):
    return torch.tensor([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])


def test_project_cuboid_aabb_pixel_count_centered_unit_box_at_5m():
    """T4.4: 1m³ box at z=5m in front of cam → AABB ≈ 100×100 px.

    Object pose: identity rotation + translation (0, 0, 5).
    Cam: identity world→cam (i.e. cam at world origin facing +Z).
    fx=500, half-extent=0.5m → on-image half-width ≈ 500*0.5/5 = 50 px.
    AABB box-of-cube ≈ 100×100 = 10000 px.
    """
    pose = torch.eye(4); pose[:3, 3] = torch.tensor([0., 0., 5.])
    sizes = torch.tensor([[1., 1., 1.]])
    mask = project_cuboids_to_mask(
        pose.unsqueeze(0), sizes, _K(), _identity_pose(),
        H=512, W=512, device="cpu",
    )
    assert mask.dtype == torch.bool
    assert mask.shape == (512, 512)
    n = mask.sum().item()
    assert 5000 < n < 15000, f"AABB pixel count out of range: {n}"


def test_project_cuboid_empty_tracks_returns_zero_mask():
    """T4.4: T=0 (no active tracks this frame) → all-False mask, no crash."""
    mask = project_cuboids_to_mask(
        torch.zeros(0, 4, 4), torch.zeros(0, 3),
        _K(), _identity_pose(), H=128, W=128, device="cpu",
    )
    assert mask.shape == (128, 128)
    assert mask.sum().item() == 0


def test_project_cuboid_multiple_tracks_union():
    """T4.4: 2 cuboids at different locations → mask is union of their AABBs."""
    # Box A at (0, 0, 5); box B at (2, 0, 5)
    pose_a = torch.eye(4); pose_a[:3, 3] = torch.tensor([0., 0., 5.])
    pose_b = torch.eye(4); pose_b[:3, 3] = torch.tensor([2., 0., 5.])
    poses = torch.stack([pose_a, pose_b])
    sizes = torch.tensor([[1., 1., 1.], [1., 1., 1.]])
    mask = project_cuboids_to_mask(poses, sizes, _K(), _identity_pose(),
                                    H=512, W=512, device="cpu")
    n_union = mask.sum().item()

    # Single-box masks for comparison
    mask_a = project_cuboids_to_mask(
        pose_a.unsqueeze(0), sizes[:1], _K(), _identity_pose(),
        H=512, W=512, device="cpu",
    )
    mask_b = project_cuboids_to_mask(
        pose_b.unsqueeze(0), sizes[1:], _K(), _identity_pose(),
        H=512, W=512, device="cpu",
    )
    # Union ≥ each individual; ≤ sum
    assert n_union >= mask_a.sum().item()
    assert n_union >= mask_b.sum().item()
    assert n_union <= (mask_a.sum() + mask_b.sum()).item()
    # Bitwise: union mask is exactly OR of the two
    assert torch.equal(mask, mask_a | mask_b)


def test_project_cuboid_behind_camera_does_not_crash():
    """T4.4: cuboid behind camera (z<0) → z.clamp(min=0.1) prevents div0,
    AABB may be degenerate but no NaN / no crash."""
    pose = torch.eye(4); pose[:3, 3] = torch.tensor([0., 0., -5.])
    sizes = torch.tensor([[1., 1., 1.]])
    mask = project_cuboids_to_mask(
        pose.unsqueeze(0), sizes, _K(), _identity_pose(),
        H=128, W=128, device="cpu",
    )
    # Mask should be valid bool tensor (possibly all-True or sparse)
    assert mask.dtype == torch.bool
    assert not torch.isnan(mask.float()).any().item()


def test_project_cuboid_image_bounds_clamped():
    """T4.4: large cuboid overflowing image → AABB clipped to (0, W-1) / (0, H-1)."""
    pose = torch.eye(4); pose[:3, 3] = torch.tensor([0., 0., 0.5])  # very close
    sizes = torch.tensor([[10., 10., 10.]])  # huge box
    mask = project_cuboids_to_mask(
        pose.unsqueeze(0), sizes, _K(), _identity_pose(),
        H=64, W=64, device="cpu",
    )
    # AABB clipped; mask should at most cover full image
    assert mask.sum().item() <= 64 * 64


def test_project_cuboid_aabb_increases_with_size():
    """T4.4: doubling cuboid size at same distance ≈ doubles AABB pixel
    count along each axis → 4× area."""
    pose = torch.eye(4); pose[:3, 3] = torch.tensor([0., 0., 5.])
    mask_1 = project_cuboids_to_mask(
        pose.unsqueeze(0), torch.tensor([[1., 1., 1.]]),
        _K(), _identity_pose(), H=512, W=512, device="cpu",
    )
    mask_2 = project_cuboids_to_mask(
        pose.unsqueeze(0), torch.tensor([[2., 2., 2.]]),
        _K(), _identity_pose(), H=512, W=512, device="cpu",
    )
    ratio = mask_2.sum().item() / mask_1.sum().item()
    # Expect ≈4 in idealized parallel projection; perspective at z=5 with box
    # half-extent grown from 0.5 to 1.0 m means near face goes from z=4.5 to
    # z=4.0 (10% closer → 11% wider scale), so total ratio is in [4, 5.5].
    assert 3.5 < ratio < 5.5, f"area ratio {ratio:.2f} not ≈4 for 2× size"
