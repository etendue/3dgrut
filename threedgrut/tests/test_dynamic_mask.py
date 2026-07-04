# SPDX-License-Identifier: Apache-2.0
"""T4.4 unit tests for dynamic_mask.project_cuboids_to_mask.

Pure CPU mock; AABB pixel-count sanity (D5: 10-15% overestimate is acceptable).
"""

from __future__ import annotations

import torch

from threedgrut.layers.dynamic_mask import project_cuboids_to_mask


def _identity_pose() -> torch.Tensor:
    return torch.eye(4)


def _K(fx: float = 500.0, fy: float = 500.0, cx: float = 256.0, cy: float = 256.0):
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def test_project_cuboid_aabb_pixel_count_centered_unit_box_at_5m():
    """T4.4: 1m³ box at z=5m in front of cam → AABB ≈ 100×100 px.

    Object pose: identity rotation + translation (0, 0, 5).
    Cam: identity world→cam (i.e. cam at world origin facing +Z).
    fx=500, half-extent=0.5m → on-image half-width ≈ 500*0.5/5 = 50 px.
    AABB box-of-cube ≈ 100×100 = 10000 px.
    """
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([0.0, 0.0, 5.0])
    sizes = torch.tensor([[1.0, 1.0, 1.0]])
    mask = project_cuboids_to_mask(
        pose.unsqueeze(0),
        sizes,
        _K(),
        _identity_pose(),
        H=512,
        W=512,
        device="cpu",
    )
    assert mask.dtype == torch.bool
    assert mask.shape == (512, 512)
    n = mask.sum().item()
    assert 5000 < n < 15000, f"AABB pixel count out of range: {n}"


def test_project_cuboid_empty_tracks_returns_zero_mask():
    """T4.4: T=0 (no active tracks this frame) → all-False mask, no crash."""
    mask = project_cuboids_to_mask(
        torch.zeros(0, 4, 4),
        torch.zeros(0, 3),
        _K(),
        _identity_pose(),
        H=128,
        W=128,
        device="cpu",
    )
    assert mask.shape == (128, 128)
    assert mask.sum().item() == 0


def test_project_cuboid_multiple_tracks_union():
    """T4.4: 2 cuboids at different locations → mask is union of their AABBs."""
    # Box A at (0, 0, 5); box B at (2, 0, 5)
    pose_a = torch.eye(4)
    pose_a[:3, 3] = torch.tensor([0.0, 0.0, 5.0])
    pose_b = torch.eye(4)
    pose_b[:3, 3] = torch.tensor([2.0, 0.0, 5.0])
    poses = torch.stack([pose_a, pose_b])
    sizes = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    mask = project_cuboids_to_mask(poses, sizes, _K(), _identity_pose(), H=512, W=512, device="cpu")
    n_union = mask.sum().item()

    # Single-box masks for comparison
    mask_a = project_cuboids_to_mask(
        pose_a.unsqueeze(0),
        sizes[:1],
        _K(),
        _identity_pose(),
        H=512,
        W=512,
        device="cpu",
    )
    mask_b = project_cuboids_to_mask(
        pose_b.unsqueeze(0),
        sizes[1:],
        _K(),
        _identity_pose(),
        H=512,
        W=512,
        device="cpu",
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
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([0.0, 0.0, -5.0])
    sizes = torch.tensor([[1.0, 1.0, 1.0]])
    mask = project_cuboids_to_mask(
        pose.unsqueeze(0),
        sizes,
        _K(),
        _identity_pose(),
        H=128,
        W=128,
        device="cpu",
    )
    # Mask should be valid bool tensor (possibly all-True or sparse)
    assert mask.dtype == torch.bool
    assert not torch.isnan(mask.float()).any().item()


def test_project_cuboid_image_bounds_clamped():
    """T4.4: large cuboid overflowing image → AABB clipped to (0, W-1) / (0, H-1)."""
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([0.0, 0.0, 0.5])  # very close
    sizes = torch.tensor([[10.0, 10.0, 10.0]])  # huge box
    mask = project_cuboids_to_mask(
        pose.unsqueeze(0),
        sizes,
        _K(),
        _identity_pose(),
        H=64,
        W=64,
        device="cpu",
    )
    # AABB clipped; mask should at most cover full image
    assert mask.sum().item() <= 64 * 64


def test_project_cuboid_aabb_increases_with_size():
    """T4.4: doubling cuboid size at same distance ≈ doubles AABB pixel
    count along each axis → 4× area."""
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([0.0, 0.0, 5.0])
    mask_1 = project_cuboids_to_mask(
        pose.unsqueeze(0),
        torch.tensor([[1.0, 1.0, 1.0]]),
        _K(),
        _identity_pose(),
        H=512,
        W=512,
        device="cpu",
    )
    mask_2 = project_cuboids_to_mask(
        pose.unsqueeze(0),
        torch.tensor([[2.0, 2.0, 2.0]]),
        _K(),
        _identity_pose(),
        H=512,
        W=512,
        device="cpu",
    )
    ratio = mask_2.sum().item() / mask_1.sum().item()
    # Expect ≈4 in idealized parallel projection; perspective at z=5 with box
    # half-extent grown from 0.5 to 1.0 m means near face goes from z=4.5 to
    # z=4.0 (10% closer → 11% wider scale), so total ratio is in [4, 5.5].
    assert 3.5 < ratio < 5.5, f"area ratio {ratio:.2f} not ≈4 for 2× size"


# ---------------------------------------------------------------------------
# A5 — pinhole behind-camera corner filtering (parity with the FTheta branch)
# ---------------------------------------------------------------------------


def test_pinhole_fully_behind_camera_empty_mask():
    """A5: cuboid entirely behind the camera (all corners z<0) → empty mask.

    Pre-fix behaviour: z.clamp(min=0.1) projected the behind corners with
    huge |u|,|v| in both signs → AABB clamped to the full image → all-True
    mask. That whole-image smear is what T8/B3 originally worked around by
    skipping pinhole in the trainer.
    """
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([0.0, 0.0, -5.0])
    sizes = torch.tensor([[1.0, 1.0, 1.0]])
    mask = project_cuboids_to_mask(
        pose.unsqueeze(0),
        sizes,
        _K(),
        _identity_pose(),
        H=128,
        W=128,
        device="cpu",
    )
    assert int(mask.sum().item()) == 0


def test_pinhole_straddling_plane_aabb_from_visible_corners_only():
    """A5: cuboid straddling the image plane → AABB from z>0 corners only.

    Center (0,0,1.0), size (1,1,2.4) → corners z ∈ {-0.2, +2.2}, x,y = ±0.5.
    Visible corners (z=2.2): u = 256 ± 500*0.5/2.2 ≈ 256 ± 113.6 → cols
    ∈ [142, 369]. Pre-fix, the z=-0.2 corners clamp to z=0.1 → u = 256 ± 2500
    → AABB spans the full 512-px width (column smear).
    """
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([0.0, 0.0, 1.0])
    sizes = torch.tensor([[1.0, 1.0, 2.4]])
    mask = project_cuboids_to_mask(
        pose.unsqueeze(0),
        sizes,
        _K(),
        _identity_pose(),
        H=512,
        W=512,
        device="cpu",
    )
    n = int(mask.sum().item())
    assert n > 0, "visible corners must still produce a mask"
    cols = torch.where(mask.any(dim=0))[0]
    rows = torch.where(mask.any(dim=1))[0]
    assert cols.min().item() >= 140 and cols.max().item() <= 372, (
        f"col extent [{cols.min().item()}, {cols.max().item()}] smeared beyond "
        "the visible-corner AABB — behind-camera corners leaked in"
    )
    assert rows.min().item() >= 140 and rows.max().item() <= 372


def test_pinhole_fully_in_front_unchanged_by_z_filter():
    """A5: all-in-front cuboid → z-filter is a no-op; exact AABB block.

    1m³ box at z=5, fx=fy=500, cx=cy=256: near face z=4.5 → half-extent
    500*0.5/4.5 ≈ 55.6 px → AABB [200, 311]² after long() truncation
    → exactly 112×112 pixels. Pins byte-equivalence of the front path.
    """
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([0.0, 0.0, 5.0])
    sizes = torch.tensor([[1.0, 1.0, 1.0]])
    mask = project_cuboids_to_mask(
        pose.unsqueeze(0),
        sizes,
        _K(),
        _identity_pose(),
        H=512,
        W=512,
        device="cpu",
    )
    assert int(mask.sum().item()) == 112 * 112
    assert mask[200:312, 200:312].all()


def test_pinhole_mixed_tracks_behind_one_dropped_front_one_kept():
    """A5: track A behind (dropped), track B in front (kept) → mask == B-only."""
    pose_behind = torch.eye(4)
    pose_behind[:3, 3] = torch.tensor([0.0, 0.0, -5.0])
    pose_front = torch.eye(4)
    pose_front[:3, 3] = torch.tensor([0.0, 0.0, 5.0])
    sizes = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    mask_both = project_cuboids_to_mask(
        torch.stack([pose_behind, pose_front]),
        sizes,
        _K(),
        _identity_pose(),
        H=512,
        W=512,
        device="cpu",
    )
    mask_front_only = project_cuboids_to_mask(
        pose_front.unsqueeze(0),
        sizes[1:],
        _K(),
        _identity_pose(),
        H=512,
        W=512,
        device="cpu",
    )
    assert torch.equal(mask_both, mask_front_only)


# ---------------------------------------------------------------------------
# A5 — resolve_batch_cuboid_intrinsics: batch → (K, ftheta_params) dispatch
# shared by trainer._maybe_fill_cuboid_mask and render.py class_psnr eval.
# ---------------------------------------------------------------------------


def _pinhole_batch(focal, pp, ftheta=None):
    from types import SimpleNamespace

    return SimpleNamespace(
        intrinsics_OpenCVPinholeCameraModelParameters=(
            None
            if focal is None
            else {
                "focal_length": focal,
                "principal_point": pp,
                "radial_coeffs": [0.0] * 6,
            }
        ),
        intrinsics_FThetaCameraModelParameters=ftheta,
    )


def test_resolve_intrinsics_pinhole_builds_K_from_numpy():
    import numpy as np

    from threedgrut.layers.dynamic_mask import resolve_batch_cuboid_intrinsics

    K, ftheta = resolve_batch_cuboid_intrinsics(
        _pinhole_batch(
            focal=np.asarray([500.0, 501.0]),
            pp=np.asarray([256.0, 257.0]),
        )
    )
    assert ftheta is None
    assert K.shape == (3, 3)
    assert K[0, 0].item() == 500.0 and K[1, 1].item() == 501.0
    assert K[0, 2].item() == 256.0 and K[1, 2].item() == 257.0
    assert K[2, 2].item() == 1.0 and K[0, 1].item() == 0.0


def test_resolve_intrinsics_pinhole_accepts_torch_and_list():
    from threedgrut.layers.dynamic_mask import resolve_batch_cuboid_intrinsics

    K_t, _ = resolve_batch_cuboid_intrinsics(
        _pinhole_batch(
            focal=torch.tensor([400.0, 400.0]),
            pp=torch.tensor([100.0, 200.0]),
        )
    )
    K_l, _ = resolve_batch_cuboid_intrinsics(
        _pinhole_batch(
            focal=[400.0, 400.0],
            pp=[100.0, 200.0],
        )
    )
    assert torch.equal(K_t, K_l)


def test_resolve_intrinsics_ftheta_takes_precedence():
    """FTheta clips must stay byte-identical: when both intrinsics are present
    (defensive; real batches carry one), FTheta wins and K stays None."""
    from threedgrut.layers.dynamic_mask import resolve_batch_cuboid_intrinsics

    fp = {"angle_to_pixeldist_poly": [0.0, 400.0], "principal_point": [256.0, 256.0]}
    K, ftheta = resolve_batch_cuboid_intrinsics(
        _pinhole_batch(
            focal=[500.0, 500.0],
            pp=[256.0, 256.0],
            ftheta=fp,
        )
    )
    assert K is None
    assert ftheta is fp


def test_resolve_intrinsics_neither_returns_none_pair():
    from types import SimpleNamespace

    from threedgrut.layers.dynamic_mask import resolve_batch_cuboid_intrinsics

    K, ftheta = resolve_batch_cuboid_intrinsics(SimpleNamespace())
    assert K is None and ftheta is None
