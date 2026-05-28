# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DepthLoss (Stage 11 / T11.A1)."""
import pytest
import torch

from threedgrut.correction.depth_prior import DepthLoss, compute_bg_lidar_loss


@pytest.fixture
def synthetic_batch():
    """[B=1, H=4, W=4] 合成数据：左上角 4 像素有 GT depth=10m，其余 0。"""
    pred = torch.full((1, 4, 4, 1), 5.0)  # 全场预测 5m
    gt = torch.zeros(1, 4, 4)
    gt[0, 0:2, 0:2] = 10.0  # 左上角 4 像素 GT=10m
    hit_mask = (gt > 0).float()  # 4 个有效像素
    return pred, gt, hit_mask


def test_l1_basic(synthetic_batch):
    """L1 + normalize：|5/80 - 10/80| = 5/80 = 0.0625，仅 4 有效像素均值。"""
    pred, gt, mask = synthetic_batch
    loss = DepthLoss(loss_type="l1", normalize=True, max_depth=80.0)
    out = loss(pred, gt, mask)
    expected = abs(5.0 / 80.0 - 10.0 / 80.0)
    assert out.item() == pytest.approx(expected, abs=1e-6)


def test_l2_basic(synthetic_batch):
    """L2 + normalize：(5/80 - 10/80)^2 = (1/16)^2 = 1/256。"""
    pred, gt, mask = synthetic_batch
    loss = DepthLoss(loss_type="l2", normalize=True, max_depth=80.0)
    out = loss(pred, gt, mask)
    expected = (5.0 / 80.0 - 10.0 / 80.0) ** 2
    assert out.item() == pytest.approx(expected, abs=1e-6)


def test_gt_with_trailing_dim(synthetic_batch):
    """gt_depth shape [B, H, W, 1] must squeeze to [B, H, W] and produce
    the same loss as the [B, H, W] form (drivestudio also accepts both)."""
    pred, gt, mask = synthetic_batch
    loss = DepthLoss(loss_type="l1", normalize=True, max_depth=80.0)
    out_3d = loss(pred, gt, mask)
    out_4d = loss(pred, gt.unsqueeze(-1), mask)
    assert out_3d.item() == pytest.approx(out_4d.item(), abs=1e-6)


def test_inverse_depth_l2(synthetic_batch):
    """inverse-depth + L2：(1/5 - 1/10)^2 = (0.1)^2 = 0.01。"""
    pred, gt, mask = synthetic_batch
    loss = DepthLoss(loss_type="l2", use_inverse_depth=True, normalize=False)
    out = loss(pred, gt, mask)
    expected = (1.0 / 5.0 - 1.0 / 10.0) ** 2
    assert out.item() == pytest.approx(expected, abs=1e-6)


def test_invalid_gt_filtered():
    """GT < eps (0.01) 或 > max_depth 必须被 hit_mask 滤掉。"""
    pred = torch.full((1, 2, 2, 1), 5.0)
    gt = torch.tensor([[[0.001, 90.0], [10.0, 50.0]]])  # 前两个无效
    mask = torch.ones(1, 2, 2)  # 故意全开，DepthLoss 内部应再过滤
    loss = DepthLoss(loss_type="l1", normalize=True, max_depth=80.0, eps=0.01)
    out = loss(pred, gt, mask)
    # 仅 (10.0, 50.0) 两点参与 → mean(|5/80-10/80|, |5/80-50/80|) = mean(5/80, 45/80)
    expected = (abs(5.0 / 80.0 - 10.0 / 80.0) + abs(5.0 / 80.0 - 50.0 / 80.0)) / 2
    assert out.item() == pytest.approx(expected, abs=1e-6)


def test_zero_valid_pixels_returns_zero():
    """全 mask 0 时返回 0 而非 NaN。"""
    pred = torch.full((1, 2, 2, 1), 5.0)
    gt = torch.zeros(1, 2, 2)
    mask = torch.zeros(1, 2, 2)
    loss = DepthLoss(loss_type="l1", normalize=True)
    out = loss(pred, gt, mask)
    assert out.item() == 0.0
    assert torch.isfinite(out)


def test_smooth_l1():
    """smooth_l1：diff < 1 用平方分支，diff >= 1 用线性分支。"""
    pred = torch.full((1, 2, 2, 1), 5.0)
    gt = torch.tensor([[[10.0, 10.0], [10.0, 10.0]]])
    mask = torch.ones(1, 2, 2)
    loss = DepthLoss(loss_type="smooth_l1", normalize=True, max_depth=80.0)
    out = loss(pred, gt, mask)
    # diff_norm = 5/80 = 0.0625 < 1 → 0.5 * 0.0625^2
    expected = 0.5 * (5.0 / 80.0) ** 2
    assert out.item() == pytest.approx(expected, abs=1e-6)


def test_compute_bg_lidar_loss_sky_far_anchor():
    """sky 区域目标 = max_depth（最远 anchor），其他区域不参与。"""
    pred = torch.full((1, 2, 2, 1), 60.0)  # 预测 60m
    sky_mask = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])  # 仅 (0,0) 是 sky
    loss = compute_bg_lidar_loss(pred, sky_mask, max_depth=80.0)
    expected = ((60.0 / 80.0 - 1.0) ** 2)  # normalized: 60/80 vs 1.0
    assert loss.item() == pytest.approx(expected, abs=1e-6)


def test_compute_bg_lidar_loss_no_sky_returns_zero():
    """sky_mask 全 0 时返回 0。"""
    pred = torch.full((1, 2, 2, 1), 60.0)
    sky_mask = torch.zeros(1, 2, 2)
    loss = compute_bg_lidar_loss(pred, sky_mask, max_depth=80.0)
    assert loss.item() == 0.0
