# SPDX-License-Identifier: Apache-2.0
"""构造 CuboidTrackObservation + 写 NCore V4 cuboids shard。

- ``tracks_to_observations``：组装核，工厂可注入 → Mac 可测（不触发 SDK）。
- ``write_cuboids_shard``：官方 V4 writer（inceptio only，O1 验过的调用）。

模块顶部**不** import ncore，延迟到函数内，保证 Mac import 模块 + 测组装核。
"""
from __future__ import annotations

from typing import List


def tracks_to_observations(tracks, track_ids, ref_frame_id, class_name,
                           obs_factory=None, bbox_factory=None,
                           source=None, source_version="lidar-cluster-v1"):
    """每个 active 帧一条 obs；bbox.rot=(0,0,yaw)（纯 yaw XYZ-euler）。

    工厂默认走 SDK（``ncore.data.CuboidTrackObservation`` / ``BBox3``）；
    传入 fake 工厂即可在 Mac 无 SDK 下测组装逻辑。
    """
    if obs_factory is None:
        import ncore.data as nd
        obs_factory = nd.CuboidTrackObservation
        bbox_factory = nd.BBox3
        if source is None:
            source = nd.LabelSource.AUTOLABEL
    out: List = []
    for tid, t in zip(track_ids, tracks):
        for b in t.boxes:
            cx, cy, cz = (float(v) for v in b.center)
            l, w, h = (float(v) for v in b.dim)
            bbox = bbox_factory(centroid=(cx, cy, cz), dim=(l, w, h),
                                rot=(0.0, 0.0, float(b.yaw)))
            kw = dict(track_id=tid, class_id=class_name, timestamp_us=int(b.ts),
                      reference_frame_id=ref_frame_id,
                      reference_frame_timestamp_us=int(b.ts), bbox3=bbox)
            if source is not None:
                kw.update(source=source, source_version=source_version)
            out.append(obs_factory(**kw))
    return out


def write_cuboids_shard(observations, out_dir, store_base_name, seq_id, interval_us,
                        generic_meta_data, group_name="auto_cuboids",
                        component_instance_name="default", store_type="itar"):
    """官方 V4 writer 写独立 cuboids shard，返回 List[UPath]（O1 验过的三要素）。

    O1 三要素（缺一 append 失败）：
    - ``generic_meta_data``：须对齐源 store（传 ``loader.generic_meta_data``），否则
      reader append 报 "Can't load component store with different generic meta-data"。
    - ``group_name``：须独特非空，否则与源空 group 冲突 "Component group '' loaded multiple times"。
    - ``component_instance_name``：cuboids reader 的 key。**目标 clip 无 GT 用 "default"**
      → ``SequenceLoaderV4`` 默认读到、读取链零改；**9ae 等有 GT 的 clip 用别名**（如
      "auto_v0"）避开 GT 占用的 "default"，验证时显式 ``cuboids_component_group_name`` 读回。
    """
    from upath import UPath
    from ncore.data.v4 import CuboidsComponent, SequenceComponentGroupsWriter

    w = SequenceComponentGroupsWriter(
        output_dir_path=UPath(out_dir), store_base_name=store_base_name,
        sequence_id=seq_id, sequence_timestamp_interval_us=interval_us,
        generic_meta_data=generic_meta_data, store_type=store_type)
    cw = w.register_component_writer(
        CuboidsComponent.Writer, component_instance_name, group_name=group_name)
    cw.store_observations(observations).finalize()
    return w.finalize()
