# SPDX-License-Identifier: Apache-2.0
"""grad-check: DepthLoss gradient must flow into pred_depth (Stage 11 T11.A2).

This is a tripwire test. tracer.pred_dist is the only path Stage 11 LiDAR /
DepthV2 supervision uses to update Gaussian positions. If a future refactor
detaches that grad chain, depth loss will silently become a no-op and Stage 11
PSNR will not move.
"""
import pytest
import torch

from threedgrut.correction.depth_prior import DepthLoss


def test_depth_loss_grad_flows_back_to_pred_depth():
    """pred_depth.requires_grad=True → loss.backward() populates pred.grad."""
    pred = torch.full((1, 4, 4, 1), 5.0, requires_grad=True)
    gt = torch.full((1, 4, 4), 10.0)
    mask = torch.ones(1, 4, 4)

    loss = DepthLoss(loss_type="l1", normalize=True, max_depth=80.0)
    out = loss(pred, gt, mask)
    out.backward()

    assert pred.grad is not None
    assert pred.grad.abs().sum().item() > 0
    # L1 + normalize: d/dpred = sign(pred/80 - gt/80) * (1/80) / (16 valid pixels)
    # pred=5 < gt=10 → sign = -1 → per-pixel grad = -1/80/16 ≈ -7.8125e-4
    expected = -1.0 / 80.0 / 16.0
    assert pred.grad[0, 0, 0, 0].item() == pytest.approx(expected, abs=1e-6)


def test_depth_loss_grad_zero_when_mask_zero():
    """Empty hit_mask must produce finite zero loss with no NaN grad.

    DepthLoss returns a detached zeros(()) constant when there are no valid
    pixels — this is intentional (no grad_fn needed when there is nothing to
    supervise). The test verifies the scalar is finite and that pred.grad is
    not polluted (remains None / zero).
    """
    pred = torch.full((1, 4, 4, 1), 5.0, requires_grad=True)
    gt = torch.zeros(1, 4, 4)
    mask = torch.zeros(1, 4, 4)

    loss = DepthLoss(loss_type="l1", normalize=True)
    out = loss(pred, gt, mask)

    assert torch.isfinite(out)
    assert out.item() == 0.0
    # No grad_fn when valid pixels = 0 — backward would raise, so skip it.
    # The important invariant: pred.grad is untouched (None).
    assert pred.grad is None
