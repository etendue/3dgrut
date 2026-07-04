# SPDX-License-Identifier: Apache-2.0
"""Unit tests for compute_lidar_psnr (Stage 11 T11.F1)."""

import math

import pytest
import torch

from threedgrut.utils.eval_metrics import compute_lidar_psnr


def test_perfect_prediction_high_psnr():
    """pred == gt over hits → MSE 0 → PSNR very high (capped by 1e-12 eps)."""
    pred = torch.full((1, 4, 4, 1), 10.0)
    gt = torch.full((1, 4, 4), 10.0)
    hit = torch.ones(1, 4, 4)
    p = compute_lidar_psnr(pred, gt, hit, max_depth=100.0)
    assert p > 100.0  # essentially infinite (eps-limited)


def test_known_mse_psnr():
    """pred=10, gt=20 over all hits → MSE=100, PSNR=-10log10(100/10000)=20."""
    pred = torch.full((1, 2, 2, 1), 10.0)
    gt = torch.full((1, 2, 2), 20.0)
    hit = torch.ones(1, 2, 2)
    p = compute_lidar_psnr(pred, gt, hit, max_depth=100.0)
    assert p == pytest.approx(20.0, abs=1e-3)


def test_no_hits_returns_nan():
    """Empty hit_mask → NaN (caller skips so coverage-less frames don't poison mean)."""
    pred = torch.full((1, 2, 2, 1), 10.0)
    gt = torch.zeros(1, 2, 2)
    hit = torch.zeros(1, 2, 2)
    p = compute_lidar_psnr(pred, gt, hit, max_depth=100.0)
    assert math.isnan(p)


def test_gt_out_of_range_excluded():
    """gt=0 (no hit) and gt>max_depth are excluded from the MSE."""
    pred = torch.tensor([[[10.0, 10.0], [10.0, 10.0]]]).unsqueeze(-1)  # [1,2,2,1]
    gt = torch.tensor([[[20.0, 0.0], [150.0, 20.0]]])  # only (0,0) and (1,1) valid
    hit = torch.ones(1, 2, 2)
    # two valid pixels, both MSE=100 → PSNR=20
    p = compute_lidar_psnr(pred, gt, hit, max_depth=100.0)
    assert p == pytest.approx(20.0, abs=1e-3)


def test_accepts_unbatched_shapes():
    """[H,W,1] pred + [H,W] gt/hit also work (render.py may pass single-frame)."""
    pred = torch.full((4, 4, 1), 10.0)
    gt = torch.full((4, 4), 20.0)
    hit = torch.ones(4, 4)
    p = compute_lidar_psnr(pred, gt, hit, max_depth=100.0)
    assert p == pytest.approx(20.0, abs=1e-3)
