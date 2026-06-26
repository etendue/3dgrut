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
    cls: int = -1       # lidar-sseg numeric class (13/14/15)；-1=未知/插值帧


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


def is_dynamic(track: Track, min_speed_mps: float) -> bool:
    """速度基动静判定（位移/时长 > min_speed_mps）；<2 帧或 dt≤0 视为静止。

    速度基(非绝对位移)避开 clip 时长依赖——慢车在短 slice 内位移小但仍是动态。
    """
    if len(track.boxes) < 2:
        return False
    disp = float(np.linalg.norm(track.boxes[-1].center[:2] - track.boxes[0].center[:2]))
    dt_s = (track.boxes[-1].ts - track.boxes[0].ts) / 1e6
    if dt_s <= 0:
        return False
    return (disp / dt_s) > min_speed_mps


def aggregate_size(track: Track, q: float = 90) -> np.ndarray:
    """track 内逐轴稳健尺寸聚合（默认 p90），保证框够大装下所有 member 点。"""
    dims = np.stack([b.dim for b in track.boxes], 0)
    return np.percentile(dims, q, axis=0)


def interpolate_gaps(track: Track, frame_timestamps_us, max_gap: int) -> Track:
    """短缺口线性插 center + lerp yaw，保证每相机帧有 obs；不外推 track 两端。

    只在 [首obs_ts, 末obs_ts] 内、且缺口跨帧数 ≤ max_gap 时插值。
    """
    if len(track.boxes) < 2:
        return track
    by_ts = {b.ts: b for b in track.boxes}
    ts_sorted = [int(t) for t in frame_timestamps_us]
    obs_ts = sorted(by_ts)
    lo, hi = obs_ts[0], obs_ts[-1]
    out: list[Box] = []
    for ts in ts_sorted:
        if ts < lo or ts > hi:           # 两端不外推
            continue
        if ts in by_ts:
            out.append(by_ts[ts])
            continue
        prev = max(t for t in obs_ts if t < ts)
        nxt = min(t for t in obs_ts if t > ts)
        gap = sum(1 for f in ts_sorted if prev < f < nxt)
        if gap > max_gap:
            continue
        a, b = by_ts[prev], by_ts[nxt]
        f = (ts - prev) / (nxt - prev)
        center = a.center + f * (b.center - a.center)
        yaw = a.yaw + f * wrap_to_pi(b.yaw - a.yaw)
        out.append(Box(ts=ts, center=center, dim=a.dim.copy(), yaw=float(yaw), cls=a.cls))
    out.sort(key=lambda x: x.ts)
    return Track(out)
