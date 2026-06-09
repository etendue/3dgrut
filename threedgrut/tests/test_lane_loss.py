# SPDX-License-Identifier: Apache-2.0
"""compute_lane_sharpness_loss — lane band 梯度幅值 L1（纯 torch，Mac 可测）。"""
from __future__ import annotations

import torch

from threedgrut.model.per_class_eval import LANE_CLASS_IDS
from threedgrut.model.lane_loss import compute_lane_sharpness_loss


def _lane_map(H, W, row):
    lane = torch.zeros(H, W, dtype=torch.long)
    lane[row, :] = LANE_CLASS_IDS[0]
    return lane


def test_lane_loss_zero_when_pred_equals_gt():
    H = W = 64
    gt = torch.rand(H, W, 3)
    lane = _lane_map(H, W, 30)
    loss = compute_lane_sharpness_loss(gt.clone(), gt, lane, band_px=8)
    assert torch.is_tensor(loss) and loss.ndim == 0
    assert float(loss.item()) < 1e-6


def test_lane_loss_zero_when_no_lane_pixels():
    H = W = 64
    gt = torch.rand(H, W, 3)
    pred = torch.rand(H, W, 3)
    lane = torch.zeros(H, W, dtype=torch.long)  # 无 lane
    loss = compute_lane_sharpness_loss(pred, gt, lane, band_px=8)
    assert float(loss.item()) == 0.0


def test_lane_loss_positive_and_differentiable():
    H = W = 64
    gt = torch.zeros(H, W, 3)
    gt[30, :] = 1.0  # GT 有一条亮线（强边缘）
    pred = torch.full((H, W, 3), 0.5, requires_grad=True)  # pred 平（无边缘）
    lane = _lane_map(H, W, 30)
    loss = compute_lane_sharpness_loss(pred, gt, lane, band_px=8)
    assert float(loss.item()) > 0.0
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_lane_loss_respects_min_pixels():
    """band < min_pixels → 0（细到不可信时不贡献梯度）。"""
    H = W = 64
    gt = torch.rand(H, W, 3)
    pred = torch.rand(H, W, 3)
    lane = torch.zeros(H, W, dtype=torch.long)
    lane[0, 0] = LANE_CLASS_IDS[0]  # band_px=0 → 1 像素 < 50
    loss = compute_lane_sharpness_loss(pred, gt, lane, band_px=0, min_pixels=50)
    assert float(loss.item()) == 0.0
