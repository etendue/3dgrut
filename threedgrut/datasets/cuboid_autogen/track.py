# SPDX-License-Identifier: Apache-2.0
"""跨帧关联 + 动静过滤 + size 聚合 + 缺帧插值。纯 numpy，Mac 可测。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from threedgrut.datasets.cuboid_autogen.cluster import wrap_to_pi


@dataclass
class Box:
    ts: int
    center: np.ndarray  # [3] world (x, y, z)
    dim: np.ndarray     # [3] (l, w, h) 全长
    yaw: float


@dataclass
class Track:
    boxes: List[Box] = field(default_factory=list)


def associate(per_frame_boxes, max_center_dist_m, max_yaw_diff_rad, min_track_len) -> List[Track]:
    """连续帧贪心最近中心关联（带 yaw 差约束）；短于 min_track_len 的丢弃。"""
    active: List[Track] = []
    for frame in per_frame_boxes:
        used: set[int] = set()
        for t in active:
            last = t.boxes[-1]
            best, best_d = -1, max_center_dist_m
            for j, b in enumerate(frame):
                if j in used:
                    continue
                d = float(np.linalg.norm(b.center[:2] - last.center[:2]))
                if d <= best_d and abs(wrap_to_pi(b.yaw - last.yaw)) <= max_yaw_diff_rad:
                    best, best_d = j, d
            if best >= 0:
                t.boxes.append(frame[best])
                used.add(best)
        for j, b in enumerate(frame):
            if j not in used:
                active.append(Track([b]))
    return [t for t in active if len(t.boxes) >= min_track_len]
