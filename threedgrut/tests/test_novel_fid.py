# SPDX-License-Identifier: Apache-2.0
"""E1.4 unit tests for FID/KID eval helpers.

Mac CPU; no Inception forward (that stays in the GPU smoke). Pins the
subset-size adaptation (KID requires subset_size <= n samples; eval splits
are ~74 frames/camera so the default 1000 would crash) and the uint8
conversion feeding torchmetrics' update().
"""
from __future__ import annotations

import torch

from threedgrut.utils.eval_metrics import kid_subset_size, rgb01_to_uint8_chw


def test_kid_subset_size_typical_eval_split():
    assert kid_subset_size(74) == 37


def test_kid_subset_size_caps_at_50():
    assert kid_subset_size(500) == 50
    assert kid_subset_size(101) == 50


def test_kid_subset_size_small_smoke_counts():
    # tiny smokes must still produce a legal subset size (<= n, >= 2)
    assert kid_subset_size(10) == 5
    assert kid_subset_size(5) == 2
    assert kid_subset_size(3) == 2
    assert kid_subset_size(2) == 2


def test_rgb01_to_uint8_chw_shape_dtype_range():
    img = torch.rand(1, 8, 12, 3)  # [B, H, W, 3] in [0, 1]
    out = rgb01_to_uint8_chw(img)
    assert out.shape == (1, 3, 8, 12)
    assert out.dtype == torch.uint8
    assert int(out.min()) >= 0 and int(out.max()) <= 255


def test_rgb01_to_uint8_chw_clamps_out_of_range():
    img = torch.tensor([[[[-0.5, 0.5, 1.7]]]])  # [1, 1, 1, 3]
    out = rgb01_to_uint8_chw(img)
    assert out[0, 0, 0, 0] == 0      # clamped low
    assert out[0, 2, 0, 0] == 255    # clamped high


def test_torchmetrics_fid_kid_constructible():
    """torch-fidelity backend present; instances build without forward."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    fid = FrechetInceptionDistance(feature=2048)
    kid = KernelInceptionDistance(subset_size=10)
    assert fid is not None and kid is not None
