# SPDX-License-Identifier: Apache-2.0
"""T3.1.a / T3.2.a unit tests for NCore aux mask + LiDAR semantic filtering.

These are pure mock tests that exercise the per-pixel mask partitioning logic
and the per-LiDAR-point semantic filter idea without requiring any NCore SDK
or A800-generated aux itar files. Integration verification is deferred to
T3.1.b / T3.2.b A800 smoke.
"""
from __future__ import annotations

import torch

from threedgrut.datasets.ncore_semantic import (
    DYNAMIC_CLASS_IDS,
    ROAD_CLASS_IDS,
    SKY_CLASS_ID,
)


# --- T3.1.a: sseg mask partitioning ---
def _make_sseg_image(H: int = 8, W: int = 8) -> torch.Tensor:
    """8x8 sseg image: 4 quadrants of 16 px each.

    TL = sky (16 px), TR = road (16 px), BL = dynamic (16 px), BR = other (16 px).
    """
    img = torch.full((H, W), 99, dtype=torch.long)  # 99 = "other / unmapped"
    img[:H // 2, :W // 2] = SKY_CLASS_ID
    img[:H // 2, W // 2:] = next(iter(ROAD_CLASS_IDS))
    img[H // 2:, :W // 2] = next(iter(DYNAMIC_CLASS_IDS))
    return img


def test_sky_road_dyn_masks_are_disjoint_partition():
    """T3.1.a: sky/road/dyn 三 mask 任一像素至多命中一类；
    总和 ≤ 1.0，分别 sum 等于 16（4 区每区 16 px）。
    """
    sseg = _make_sseg_image()
    sky = (sseg == SKY_CLASS_ID).float()
    road = torch.isin(sseg, torch.tensor(list(ROAD_CLASS_IDS))).float()
    dyn = torch.isin(sseg, torch.tensor(list(DYNAMIC_CLASS_IDS))).float()

    # disjoint: 每个像素属于至多一类
    assert torch.all((sky + road + dyn) <= 1.0), (
        f"masks overlap: max sum = {(sky + road + dyn).max().item()}"
    )
    # 每区 4*4 = 16 px
    assert sky.sum().item() == 16
    assert road.sum().item() == 16
    assert dyn.sum().item() == 16


def test_road_mask_covers_all_road_class_ids():
    """T3.1.a: ROAD_CLASS_IDS 是 set，road_mask 应同时包含所有 ID（不止 first）。"""
    img = torch.tensor([list(ROAD_CLASS_IDS) + [99]], dtype=torch.long)
    road = torch.isin(img, torch.tensor(list(ROAD_CLASS_IDS))).float()
    # 前 len(ROAD_CLASS_IDS) 个应该 = 1，最后一个（99）应该 = 0
    n_road = len(ROAD_CLASS_IDS)
    assert road[0, :n_road].sum().item() == n_road
    assert road[0, n_road].item() == 0


def test_dynamic_mask_covers_all_dynamic_class_ids():
    """T3.1.a: DYNAMIC_CLASS_IDS 全 8 类（11-18）都进 dyn mask。"""
    img = torch.tensor([list(DYNAMIC_CLASS_IDS)], dtype=torch.long)
    dyn = torch.isin(img, torch.tensor(list(DYNAMIC_CLASS_IDS))).float()
    assert dyn.sum().item() == len(DYNAMIC_CLASS_IDS)


def test_sky_class_id_is_singular():
    """T3.1.a: sky 是唯一 ID 10，不是 set（与 cityscapes 一致）。"""
    assert isinstance(SKY_CLASS_ID, int)
    assert SKY_CLASS_ID == 10


# --- T3.2.a: road / dynamic LiDAR semantic filtering mock ---
def _filter_pts_by_label(pts: torch.Tensor, labels: torch.Tensor,
                         target_ids: frozenset[int]) -> torch.Tensor:
    """Reference filter logic: keep pts whose label ∈ target_ids.

    T3.2.b will live-implement this inside NCoreDataset.get_road_lidar_points()
    via ncore.has_pc_generic_data + isin. This test pins the filter semantics
    so T3.2.b implementation has a behavioral contract.
    """
    mask = torch.isin(labels, torch.tensor(list(target_ids)))
    return pts[mask]


def test_road_lidar_points_filter_keeps_only_road_class():
    """T3.2.a: mock 100 LiDAR 点，50 road / 50 other → 过滤后 50 road 点保留。"""
    pts = torch.randn(100, 3)
    labels = torch.zeros(100, dtype=torch.long)
    labels[:50] = next(iter(ROAD_CLASS_IDS))  # road
    labels[50:] = 99                            # other
    road_pts = _filter_pts_by_label(pts, labels, ROAD_CLASS_IDS)
    assert road_pts.shape == (50, 3)
    # identity: 前 50 个原始 pts 应原样保留（顺序不变，因为 isin mask 保序）
    assert torch.allclose(road_pts, pts[:50])


def test_dynamic_lidar_points_filter_keeps_all_dyn_classes():
    """T3.2.a: 每个 dyn class 各 10 点 → 总共 80 个 (8 class * 10)，全部保留。"""
    n_per_class = 10
    pts_list = [torch.randn(n_per_class, 3) for _ in DYNAMIC_CLASS_IDS]
    labels_list = [
        torch.full((n_per_class,), cid, dtype=torch.long)
        for cid in DYNAMIC_CLASS_IDS
    ]
    pts = torch.cat(pts_list + [torch.randn(20, 3)])  # +20 noise (label 99)
    labels = torch.cat(labels_list + [torch.full((20,), 99, dtype=torch.long)])
    dyn_pts = _filter_pts_by_label(pts, labels, DYNAMIC_CLASS_IDS)
    assert dyn_pts.shape == (n_per_class * len(DYNAMIC_CLASS_IDS), 3)


def test_filter_with_empty_label_set_returns_empty():
    """T3.2.a: 当没有任何点匹配目标 class → 返回 shape (0, 3) 不 crash。"""
    pts = torch.randn(50, 3)
    labels = torch.full((50,), 99, dtype=torch.long)
    out = _filter_pts_by_label(pts, labels, ROAD_CLASS_IDS)
    assert out.shape == (0, 3)
