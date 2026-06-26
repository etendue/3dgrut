# SPDX-License-Identifier: Apache-2.0
"""Task 10 — V4 cuboid 写读契约回归（inceptio，需真 ncore SDK + 9ae meta）。

固化 O1 spike：tracks_to_observations → write_cuboids_shard → reader append 读回
→ 字段保真（dim/centroid/rot）+ reference_frame_id="world" transform 退化 identity。

Mac 上 9ae meta 不存在 → 自动 skip；inceptio 上 conftest try-import 到真 ncore → 真跑。
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.inceptio

_META = os.environ.get(
    "NCORE_9AE_META",
    "/home/inceptio/work/data/9ae151dc_consolidated/"
    "pai_9ae151dc-e87b-41a7-8e85-71772f9603d7.json",
)


@pytest.mark.skipif(not os.path.exists(_META), reason="9ae meta 不在(非 inceptio)")
def test_cuboid_write_read_roundtrip(tmp_path):
    from ncore.data.v4 import SequenceComponentGroupsReader, SequenceLoaderV4

    from threedgrut.datasets.cuboid_autogen.track import Box, Track
    from threedgrut.datasets.cuboid_autogen.v4_writer import (
        tracks_to_observations,
        write_cuboids_shard,
    )

    # --- 源 loader：拿 seq_id / interval / generic_meta_data / pose_graph ---
    src = SequenceComponentGroupsReader([_META], open_consolidated=True)
    loader = SequenceLoaderV4(
        src, poses_component_group_name="default",
        intrinsics_component_group_name="default", masks_component_group_name="default")
    seq_id = loader.sequence_id
    interval = loader.sequence_timestamp_interval_us
    gmd = loader.generic_meta_data

    # --- 造 1 track / 2 帧（world 系，ts 落在 interval 内）---
    t0 = int(interval.start)
    t1 = t0 + 100_000
    track = Track([
        Box(ts=t0, center=np.array([10.0, 0.0, 0.85]), dim=np.array([4.5, 2.0, 1.7]), yaw=0.3),
        Box(ts=t1, center=np.array([11.0, 0.0, 0.85]), dim=np.array([4.5, 2.0, 1.7]), yaw=0.3),
    ])
    obs = tracks_to_observations(
        [track], track_ids=["auto_0"], ref_frame_id="world", class_name="automobile")

    # --- 写 shard（9ae 有 GT 占 "default" → 用 instance "auto_v0"）---
    shard = write_cuboids_shard(
        obs, out_dir=str(tmp_path / "out"), store_base_name="autocuboids",
        seq_id=seq_id, interval_us=interval, generic_meta_data=gmd,
        group_name="auto_cuboids", component_instance_name="auto_v0")
    assert shard, "write_cuboids_shard 返回空"

    # --- append 读回（显式 cuboids_component_group_name="auto_v0"）---
    rd = SequenceComponentGroupsReader([_META, *map(str, shard)], open_consolidated=True)
    rloader = SequenceLoaderV4(
        rd, poses_component_group_name="default",
        intrinsics_component_group_name="default", masks_component_group_name="default",
        cuboids_component_group_name="auto_v0")
    got = [o for o in rloader.get_cuboid_track_observations() if o.track_id == "auto_0"]
    assert len(got) == 2, f"读回 auto_0 obs 数={len(got)}，应为 2"

    # --- 字段保真：ref="world" → transform("world") 退化 identity ---
    o = sorted(got, key=lambda x: x.timestamp_us)[0]
    wb = o.transform("world", int(o.timestamp_us), rloader.pose_graph).bbox3
    np.testing.assert_allclose(tuple(wb.dim), (4.5, 2.0, 1.7), atol=1e-3)
    np.testing.assert_allclose(tuple(wb.centroid), (10.0, 0.0, 0.85), atol=0.05)
    assert abs(float(wb.rot[2]) - 0.3) < 1e-3
    assert o.class_id == "automobile"
