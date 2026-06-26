# SPDX-License-Identifier: Apache-2.0
"""Task 4/5/6 单测 — 跨帧关联 / 动静过滤 / size 聚合 / 缺帧插值。纯 numpy，Mac。"""
from __future__ import annotations

import numpy as np

from threedgrut.datasets.cuboid_autogen.track import Box, associate


def _b(ts, cx, cy, yaw=0.0):
    return Box(ts=ts, center=np.array([cx, cy, 0.0]),
               dim=np.array([4.5, 2.0, 1.7]), yaw=yaw)


# ---------------- Task 4: associate ----------------

def test_associate_two_tracks_drops_noise():
    rng = np.random.default_rng(0)
    frames = []
    for k in range(10):
        f = [_b(k * 100_000, k * 1.0, 0.0), _b(k * 100_000, k * 1.0, 8.0)]  # 两条直线轨迹
        f.append(_b(k * 100_000, rng.uniform(40, 60), rng.uniform(40, 60)))  # 每帧 1 噪声
        frames.append(f)
    tracks = associate(frames, max_center_dist_m=3.0, max_yaw_diff_rad=0.5, min_track_len=3)
    assert len(tracks) == 2                       # 噪声 < min_track_len 被丢
    assert all(len(t.boxes) == 10 for t in tracks)
    ys = sorted(t.boxes[0].center[1] for t in tracks)
    assert abs(ys[0] - 0.0) < 1e-6 and abs(ys[1] - 8.0) < 1e-6  # 两轨迹不串
