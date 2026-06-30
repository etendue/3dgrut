# SPDX-License-Identifier: Apache-2.0
"""Task 7 单测 — 旋转矩形 BEV-IoU + 框匹配。纯 numpy，Mac。"""
from __future__ import annotations

from threedgrut.datasets.cuboid_autogen.bev_metric import bev_iou, match_boxes


def test_identical_boxes_iou_one():
    box = (0.0, 0.0, 4.5, 2.0, 0.3)  # (cx, cy, l, w, yaw)
    assert abs(bev_iou(box, box) - 1.0) < 1e-6


def test_half_overlap_axis_aligned():
    a = (0.5, 0.5, 1.0, 1.0, 0.0)
    b = (1.0, 0.5, 1.0, 1.0, 0.0)  # x 重叠 0.5 → 交 0.5 / 并 1.5
    assert abs(bev_iou(a, b) - (0.5 / 1.5)) < 1e-6


def test_disjoint_iou_zero():
    assert bev_iou((0, 0, 1, 1, 0), (10, 10, 1, 1, 0)) == 0.0


def test_match_greedy_by_center():
    auto = [(0, 0, 4.5, 2, 0), (20, 0, 4.5, 2, 0)]
    gt = [(0.3, 0, 4.5, 2, 0)]
    m = match_boxes(auto, gt, max_dist=2.0)
    assert m == [(0, 0)]  # auto[0]↔gt[0]；auto[1] 超距无匹配
