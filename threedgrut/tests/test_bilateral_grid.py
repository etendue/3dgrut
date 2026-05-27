# SPDX-License-Identifier: Apache-2.0
"""T9.1 unit tests for BilateralGrid color correction module.

Replaces test_exposure.py once trainer wiring migrates; keeps key invariants:
- Identity init = exact identity transform
- per-camera grad isolation
- clamp(0, 1) output
- invalid idx → IndexError
- num_camera < 1 → ValueError
- state_dict roundtrip
- tv_loss == 0 at 1x1x1 identity init
- 1x1x1 fast path == general grid_sample path (math equivalence)
"""
from __future__ import annotations

import pytest
import torch

from threedgrut.correction.bilateral_grid import (
    BilateralGrid,
    color_affine_transform,
    total_variation_loss,
)


def test_zero_init_is_identity_1x1x1():
    """1x1x1 grid, identity init: output ≡ input for any RGB."""
    bg = BilateralGrid(num_camera=3, grid_X=1, grid_Y=1, grid_W=1)
    img = torch.rand(8, 8, 3)
    out = bg(0, img)
    assert torch.allclose(out, img.clamp(0, 1), atol=1e-6)
    out = bg(2, img)
    assert torch.allclose(out, img.clamp(0, 1), atol=1e-6)


def test_zero_init_is_identity_larger_grid():
    """Larger grid (4x4x2), identity init: output ≡ input via grid_sample path."""
    bg = BilateralGrid(num_camera=2, grid_X=4, grid_Y=4, grid_W=2)
    img = torch.rand(8, 8, 3)
    out = bg(0, img)
    assert torch.allclose(out, img.clamp(0, 1), atol=1e-5)


def test_per_camera_grad_isolation():
    """Forwarding camera 0 only flows grad into grids[0], not grids[1]."""
    bg = BilateralGrid(num_camera=4, grid_X=1, grid_Y=1, grid_W=1)
    img = torch.rand(4, 4, 3, requires_grad=False)
    out = bg(1, img)
    loss = out.sum()
    loss.backward()
    assert bg.grids.grad is not None
    grad = bg.grids.grad
    assert grad[1].abs().sum() > 0, "camera 1 grid should receive grad"
    for i in (0, 2, 3):
        assert grad[i].abs().sum() == 0, f"camera {i} grid must not receive grad"


def test_clamp_to_unit_range():
    """If learned affine pushes outside [0, 1], output is clamped."""
    bg = BilateralGrid(num_camera=1)
    # Push gain to 5x → output far above 1
    with torch.no_grad():
        bg.grids.zero_()
        bg.grids[0, 0, 0, 0, 0] = 5.0  # R = 5*R + 0
        bg.grids[0, 5, 0, 0, 0] = 5.0  # G = 5*G + 0
        bg.grids[0, 10, 0, 0, 0] = 5.0  # B = 5*B + 0
    img = torch.full((4, 4, 3), 0.5)
    out = bg(0, img)
    assert (out >= 0).all() and (out <= 1).all()


def test_invalid_camera_idx_raises():
    bg = BilateralGrid(num_camera=3)
    img = torch.rand(2, 2, 3)
    with pytest.raises(IndexError):
        bg(3, img)
    with pytest.raises(IndexError):
        bg(-1, img)


def test_constructor_rejects_zero_cameras():
    with pytest.raises(ValueError):
        BilateralGrid(num_camera=0)


def test_constructor_rejects_zero_grid_dims():
    with pytest.raises(ValueError):
        BilateralGrid(num_camera=2, grid_X=0)
    with pytest.raises(ValueError):
        BilateralGrid(num_camera=2, grid_Y=0)
    with pytest.raises(ValueError):
        BilateralGrid(num_camera=2, grid_W=0)


def test_state_dict_roundtrip():
    """Save state_dict → load into fresh module → identical forward output."""
    bg1 = BilateralGrid(num_camera=3, grid_X=2, grid_Y=2, grid_W=2)
    # Perturb grids
    with torch.no_grad():
        bg1.grids.add_(0.1 * torch.randn_like(bg1.grids))

    bg2 = BilateralGrid(num_camera=3, grid_X=2, grid_Y=2, grid_W=2)
    bg2.load_state_dict(bg1.state_dict())

    img = torch.rand(6, 6, 3)
    for idx in range(3):
        assert torch.allclose(bg1(idx, img), bg2(idx, img), atol=1e-6)


def test_tv_loss_zero_at_1x1x1_identity():
    """TV is 0 for 1x1x1 grid (no spatial neighbors to differ)."""
    bg = BilateralGrid(num_camera=5)
    assert bg.tv_loss().item() == 0.0


def test_tv_loss_zero_at_larger_identity():
    """TV is 0 when all voxels are identity (no spatial variation)."""
    bg = BilateralGrid(num_camera=2, grid_X=4, grid_Y=4, grid_W=2)
    assert bg.tv_loss().item() == 0.0


def test_tv_loss_positive_when_perturbed():
    """TV > 0 once grids deviate spatially."""
    bg = BilateralGrid(num_camera=2, grid_X=4, grid_Y=4, grid_W=2)
    with torch.no_grad():
        bg.grids.add_(0.5 * torch.randn_like(bg.grids))
    assert bg.tv_loss().item() > 0


def test_fast_path_matches_grid_sample_for_constant_grid():
    """At 1x1x1 the fast path is exercised; verify same affine result if we
    feed the same (constant) grid into the general path via a 2x2x2 build."""
    bg_fast = BilateralGrid(num_camera=1, grid_X=1, grid_Y=1, grid_W=1)
    bg_general = BilateralGrid(num_camera=1, grid_X=2, grid_Y=2, grid_W=2)
    # Make the fast-path voxel non-identity, and replicate across general grid.
    perturbed_affine = torch.tensor(
        [[1.2, 0.1, -0.05, 0.02], [0.0, 0.9, 0.05, -0.01], [-0.03, 0.04, 1.1, 0.03]]
    )
    with torch.no_grad():
        bg_fast.grids[0, :, 0, 0, 0] = perturbed_affine.reshape(12)
        # Replicate the same affine across every voxel of the 2x2x2 grid.
        bg_general.grids[0] = (
            perturbed_affine.reshape(12, 1, 1, 1).expand(12, 2, 2, 2).contiguous()
        )

    img = torch.rand(6, 6, 3)
    out_fast = bg_fast(0, img)
    out_general = bg_general(0, img)
    # Both should produce the same affine-transformed-then-clamped output.
    assert torch.allclose(out_fast, out_general, atol=1e-5)


def test_color_affine_transform_identity():
    """color_affine_transform with identity affine returns rgb unchanged."""
    identity = torch.tensor(
        [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]]
    )
    rgb = torch.tensor([0.3, 0.7, 0.2])
    out = color_affine_transform(identity, rgb)
    assert torch.allclose(out, rgb, atol=1e-7)


def test_total_variation_loss_zero_uniform():
    """TV of a uniform tensor is 0."""
    x = torch.ones(2, 12, 4, 4, 4)
    assert total_variation_loss(x).item() == 0.0
