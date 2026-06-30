# SPDX-License-Identifier: Apache-2.0
"""Task 4/5/6 单测 — 跨帧关联 / 动静过滤 / size 聚合 / 缺帧插值。纯 numpy，Mac。"""
from __future__ import annotations

import numpy as np

from threedgrut.datasets.cuboid_autogen.track import (
    Box,
    Track,
    aggregate_size,
    associate,
    interpolate_gaps,
    is_dynamic,
)


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


# ---------------- Task 5: 动静过滤 + size 聚合 ----------------

def test_static_track_filtered():
    rng = np.random.default_rng(1)
    t = Track([_b(k * 100_000, rng.normal(0, 0.05), 0.0) for k in range(11)])  # 1s 内抖动<0.3m
    assert is_dynamic(t, min_speed_mps=0.5) is False


def test_moving_track_kept():
    t = Track([_b(k * 100_000, k * 0.5, 0.0) for k in range(11)])  # 1s 移 5m → 5 m/s
    assert is_dynamic(t, min_speed_mps=0.5) is True


def test_single_frame_is_static():
    assert is_dynamic(Track([_b(0, 0, 0)]), min_speed_mps=0.5) is False


def test_aggregate_size_p90():
    t = Track([_b(0, 0, 0) for _ in range(3)])
    t.boxes[0].dim = np.array([4.0, 2.0, 1.5])
    t.boxes[1].dim = np.array([4.5, 2.1, 1.6])
    t.boxes[2].dim = np.array([5.0, 2.2, 1.7])
    d = aggregate_size(t, q=90)
    expect = np.percentile(np.array([[4, 2, 1.5], [4.5, 2.1, 1.6], [5, 2.2, 1.7]]), 90, axis=0)
    np.testing.assert_allclose(d, expect, atol=1e-9)


# ---------------- Task 6: interpolate_gaps ----------------

def test_interpolate_single_gap():
    t = Track([_b(0, 0.0, 0.0, yaw=0.0), _b(200_000, 2.0, 0.0, yaw=0.4)])  # 缺 100_000
    out = interpolate_gaps(t, frame_timestamps_us=[0, 100_000, 200_000], max_gap=2)
    assert len(out.boxes) == 3
    mid = [b for b in out.boxes if b.ts == 100_000][0]
    np.testing.assert_allclose(mid.center, [1.0, 0.0, 0.0], atol=1e-6)
    assert abs(mid.yaw - 0.2) < 1e-6


def test_no_extrapolation_beyond_ends():
    t = Track([_b(100_000, 1.0, 0.0)])
    out = interpolate_gaps(t, frame_timestamps_us=[0, 100_000, 200_000], max_gap=2)
    assert [b.ts for b in out.boxes] == [100_000]  # 两端不外推
