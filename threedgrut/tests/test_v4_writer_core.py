# SPDX-License-Identifier: Apache-2.0
"""Task 9 Mac 核单测 — tracks_to_observations 组装（fake 工厂注入，无 SDK）。

write_cuboids_shard 的真实 SDK 写读在 inceptio round-trip（test_v4_writer_roundtrip）验证。
"""
from __future__ import annotations

import numpy as np

from threedgrut.datasets.cuboid_autogen.track import Box, Track
from threedgrut.datasets.cuboid_autogen.v4_writer import tracks_to_observations


class _FakeBBox:
    def __init__(self, centroid, dim, rot):
        self.centroid = centroid
        self.dim = dim
        self.rot = rot


class _FakeObs:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_obs_fields_pure():
    t = Track([Box(ts=1000, center=np.array([10.0, 3.0, 0.85]),
                   dim=np.array([4.5, 2.0, 1.7]), yaw=0.6)])
    obs = tracks_to_observations(
        [t], track_ids=["auto_0"], ref_frame_id="world", class_name="automobile",
        obs_factory=_FakeObs, bbox_factory=_FakeBBox)
    assert len(obs) == 1
    o = obs[0]
    assert o.track_id == "auto_0"
    assert o.class_id == "automobile"
    assert o.timestamp_us == 1000
    assert o.reference_frame_id == "world"
    assert o.reference_frame_timestamp_us == 1000
    assert o.bbox3.rot == (0.0, 0.0, 0.6)            # 纯 yaw XYZ-euler
    assert tuple(o.bbox3.dim) == (4.5, 2.0, 1.7)
    np.testing.assert_allclose(o.bbox3.centroid, (10.0, 3.0, 0.85))


def test_multi_frame_one_obs_each():
    t = Track([Box(ts=ts, center=np.array([float(ts), 0.0, 0.0]),
                   dim=np.array([4.0, 2.0, 1.5]), yaw=0.0)
               for ts in (0, 100, 200)])
    obs = tracks_to_observations(
        [t], track_ids=["x"], ref_frame_id="world", class_name="bus",
        obs_factory=_FakeObs, bbox_factory=_FakeBBox)
    assert [o.timestamp_us for o in obs] == [0, 100, 200]   # 每 active 帧一条
    assert all(o.reference_frame_id == "world" and o.class_id == "bus" for o in obs)
