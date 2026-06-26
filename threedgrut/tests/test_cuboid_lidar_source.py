# SPDX-License-Identifier: Apache-2.0
"""Task 8 — 逐帧动态车辆 LiDAR 点访问集成测试（inceptio，需真 SDK + 9ae aux.lidar-sseg）。

Mac 上 9ae meta 不存在 → skip；inceptio 上构造 NCoreDataset 真跑 iter_vehicle_lidar_frames。
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
def test_iter_vehicle_lidar_frames_9ae():
    import hydra

    from threedgrut import datasets
    from threedgrut.datasets.cuboid_autogen.lidar_source import iter_vehicle_lidar_frames

    with hydra.initialize(config_path="../../configs", version_base=None):
        conf = hydra.compose(
            config_name="apps/ncore_3dgut_mcmc_multilayer",
            overrides=[
                f"path={_META}",
                "trainer.sky_backend=mlp",
                "dataset.load_aux_masks=true",
                "use_lidar_depth=false",
                "use_depth_prior=false",
                "load_lidar_depth_map=false",
            ],
        )
    ds, _ = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)

    n_frames = 0
    seen_ts: dict = {}
    for sid, ts, xyz, labels in iter_vehicle_lidar_frames(ds):
        assert xyz.ndim == 2 and xyz.shape[1] == 3
        assert xyz.shape[0] == labels.shape[0]
        assert set(np.unique(labels).tolist()) <= {13, 14, 15}
        seen_ts.setdefault(sid, []).append(ts)
        n_frames += 1
        if n_frames >= 30:  # 抽样前 30 帧即可
            break

    assert n_frames > 0, "iter_vehicle_lidar_frames 没 yield 任何车辆帧"
    for sid, tss in seen_ts.items():
        assert tss == sorted(tss), f"source {sid} ts 非递增"
    print(f"[iter_vehicle] {n_frames} 帧, sources={list(seen_ts)}")
