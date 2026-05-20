# SPDX-License-Identifier: Apache-2.0
"""T3.4 unit tests for compute_layered_l1_loss (region-weighted L1).

These tests exercise the pure function without instantiating Trainer, so they
run on Mac CPU without CUDA / NCore SDK / Hydra compose overhead.
"""
from __future__ import annotations

import torch

from threedgrut.model.layered_loss import compute_layered_l1_loss, compute_sky_loss


def _quad_mask(H: int, W: int, quadrant: str) -> torch.Tensor:
    """Return a [H, W] mask covering one quadrant ('tl', 'tr', 'bl', 'br')."""
    m = torch.zeros(H, W)
    h2, w2 = H // 2, W // 2
    if quadrant == "tl": m[:h2, :w2] = 1
    elif quadrant == "tr": m[:h2, w2:] = 1
    elif quadrant == "bl": m[h2:, :w2] = 1
    elif quadrant == "br": m[h2:, w2:] = 1
    return m


def test_compute_layered_l1_loss_v1_fallback_when_no_image_infos():
    """T3.4: image_infos=None → plain .mean() L1 (v1 byte-identical)."""
    pred = torch.ones(1, 4, 4, 3)
    gt = torch.zeros(1, 4, 4, 3)
    loss = compute_layered_l1_loss(pred, gt, image_infos=None)
    assert torch.allclose(loss, torch.tensor(1.0))


def test_compute_layered_l1_loss_v1_fallback_with_valid_mask():
    """T3.4: image_infos=None + valid_mask → masked mean L1."""
    pred = torch.ones(1, 4, 4, 3)
    gt = torch.zeros(1, 4, 4, 3)
    valid = _quad_mask(4, 4, "tl").unsqueeze(0)  # only 4 px valid
    loss = compute_layered_l1_loss(pred, gt, image_infos=None, valid_mask=valid)
    # All valid pixels have L1=1.0, sum=4.0, denom=4 → loss=1.0
    assert torch.allclose(loss, torch.tensor(1.0))


def test_compute_layered_l1_loss_partitions_three_regions():
    """T3.4 main: 4x4 pred=1, gt=0; sky/road/dyn each 4 px → loss = 3.0
    (sum of 3 region means; sky region EXCLUDED from L1; bg region empty)."""
    pred = torch.ones(1, 4, 4, 3)
    gt = torch.zeros(1, 4, 4, 3)
    image_infos = {
        "sky_mask":      _quad_mask(4, 4, "tl").unsqueeze(0),
        "road_mask":     _quad_mask(4, 4, "tr").unsqueeze(0),
        "dyn_mask_sseg": _quad_mask(4, 4, "bl").unsqueeze(0),
    }
    loss = compute_layered_l1_loss(pred, gt, image_infos=image_infos,
                                   min_pixels=1)  # all regions kept
    # Each non-sky region's mean L1 = 1.0; bg region (br quadrant) also has 4 px
    # → bg + road + dyn = 3.0 (sky excluded)
    assert torch.allclose(loss, torch.tensor(3.0)), f"got {loss.item()}"


def test_compute_layered_l1_loss_small_region_skipped():
    """T3.4 D6: region.sum() < min_pixels → that region's contribution = 0."""
    pred = torch.ones(1, 10, 10, 3)
    gt = torch.zeros(1, 10, 10, 3)
    # road occupies only 5 px (< min_pixels=100 default); should be dropped
    road = torch.zeros(1, 10, 10)
    road[0, 0, :5] = 1
    image_infos = {
        "sky_mask":      torch.zeros(1, 10, 10),
        "road_mask":     road,
        "dyn_mask_sseg": torch.zeros(1, 10, 10),
    }
    loss = compute_layered_l1_loss(pred, gt, image_infos=image_infos,
                                   min_pixels=100)
    # bg region = 95 valid px, also < 100 → bg dropped
    # dyn region = 0 px → 0
    # road = 5 px → dropped → 0
    # Expected: all three regions dropped → 0
    assert torch.allclose(loss, torch.tensor(0.0)), f"got {loss.item()}"


def test_compute_layered_l1_loss_dyn_cuboid_takes_precedence_over_sseg():
    """T3.4: when both dyn_mask_cuboid and dyn_mask_sseg present, cuboid wins.
    (Stage 4 T4.4 introduces cuboid mask; Stage 3 falls back to sseg.)"""
    pred = torch.ones(1, 4, 4, 3)
    gt = torch.zeros(1, 4, 4, 3)
    image_infos = {
        "sky_mask":         torch.zeros(1, 4, 4),
        "road_mask":        torch.zeros(1, 4, 4),
        "dyn_mask_sseg":    torch.ones(1, 4, 4),                 # all-1 sseg
        "dyn_mask_cuboid":  _quad_mask(4, 4, "br").unsqueeze(0), # 4 px cuboid
    }
    loss = compute_layered_l1_loss(pred, gt, image_infos=image_infos,
                                   min_pixels=1)
    # If cuboid wins: dyn region = 4 px, bg = 12 px → bg + dyn = 1 + 1 = 2.0
    # If sseg wins:   dyn region = 16 px, bg = 0 → only dyn = 1.0
    assert torch.allclose(loss, torch.tensor(2.0)), (
        f"got {loss.item()} — cuboid should take precedence over sseg"
    )


def test_compute_layered_l1_loss_returns_scalar():
    """T3.4: output is a 0-dim scalar tensor (so .backward() works)."""
    pred = torch.ones(1, 4, 4, 3, requires_grad=True)
    gt = torch.zeros(1, 4, 4, 3)
    image_infos = {
        "sky_mask":      _quad_mask(4, 4, "tl").unsqueeze(0),
        "road_mask":     _quad_mask(4, 4, "tr").unsqueeze(0),
        "dyn_mask_sseg": _quad_mask(4, 4, "bl").unsqueeze(0),
    }
    loss = compute_layered_l1_loss(pred, gt, image_infos=image_infos,
                                   min_pixels=1)
    assert loss.ndim == 0
    loss.backward()
    assert pred.grad is not None


# ============================================================================
# T5.5: sky envmap region L1.
# ============================================================================
def test_compute_sky_loss_zero_when_no_sky_pixels():
    """sky_mask sum < min_pixels → loss is 0 (no NaN, no grad explosion)."""
    rgb_sky = torch.ones(1, 4, 4, 3, requires_grad=True)
    gt = torch.zeros(1, 4, 4, 3)
    loss = compute_sky_loss(rgb_sky, gt, sky_mask=torch.zeros(1, 4, 4),
                            min_pixels=1)
    assert loss.item() == 0.0


def test_compute_sky_loss_none_mask_returns_zero():
    """No mask in the batch → sky_loss returns 0 (graceful no-op)."""
    rgb_sky = torch.ones(1, 4, 4, 3)
    gt = torch.zeros(1, 4, 4, 3)
    loss = compute_sky_loss(rgb_sky, gt, sky_mask=None)
    assert loss.item() == 0.0


def test_compute_sky_loss_uniform_region_arithmetic():
    """rgb_sky - gt = 0.5 over a uniform sky region → loss == 0.5."""
    rgb_sky = torch.full((1, 4, 4, 3), 0.5)
    gt = torch.zeros(1, 4, 4, 3)
    sky_mask = torch.ones(1, 4, 4)
    loss = compute_sky_loss(rgb_sky, gt, sky_mask=sky_mask, min_pixels=1)
    assert torch.allclose(loss, torch.tensor(0.5), atol=1e-6)


def test_compute_sky_loss_only_on_sky_region():
    """Pixels outside sky_mask must not contribute."""
    rgb_sky = torch.ones(1, 4, 4, 3)
    gt = torch.zeros(1, 4, 4, 3)
    sky_mask = _quad_mask(4, 4, "tl").unsqueeze(0)  # 4 of 16 pixels
    loss = compute_sky_loss(rgb_sky, gt, sky_mask=sky_mask, min_pixels=1)
    # |1 - 0| over masked pixels, averaged → 1.0
    assert torch.allclose(loss, torch.tensor(1.0), atol=1e-6)


def test_compute_sky_loss_squeezed_mask_shape():
    """Mask provided already with trailing 1 (shape [B,H,W,1]) must work."""
    rgb_sky = torch.full((1, 4, 4, 3), 0.5)
    gt = torch.zeros(1, 4, 4, 3)
    loss = compute_sky_loss(rgb_sky, gt, sky_mask=torch.ones(1, 4, 4, 1),
                            min_pixels=1)
    assert torch.allclose(loss, torch.tensor(0.5), atol=1e-6)


def test_compute_sky_loss_grad_flows_through_pred():
    """sky_loss must be differentiable wrt rgb_sky params."""
    rgb_sky = torch.ones(1, 4, 4, 3, requires_grad=True)
    gt = torch.zeros(1, 4, 4, 3)
    sky_mask = torch.ones(1, 4, 4)
    loss = compute_sky_loss(rgb_sky, gt, sky_mask=sky_mask, min_pixels=1)
    loss.backward()
    assert rgb_sky.grad is not None
    assert rgb_sky.grad.abs().sum().item() > 0
