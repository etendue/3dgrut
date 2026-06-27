# SPDX-License-Identifier: Apache-2.0
"""逐帧动态车辆 LiDAR 点访问（world frame，按类）。inceptio/SDK only。

复用 ``NCoreDataset._iter_semantic_lidar_frames``（读 aux.lidar-sseg），过滤到
车辆类（13/14/15），保留**逐帧归属**供跨帧 tracking。
"""
from __future__ import annotations

import numpy as np

from threedgrut.datasets.cuboid_autogen.labels import LIDAR_SSEG_TO_CUBOID_CLASS


def iter_vehicle_lidar_frames(dataset):
    """yield ``(source_id, ts_us:int, xyz_world[N,3] np float64, labels[N] np int64)``。

    仅 car/truck/bus（lidar-sseg 13/14/15）点，逐帧（不聚合）。
    """
    vehicle_ids = frozenset(LIDAR_SSEG_TO_CUBOID_CLASS)  # {13, 14, 15}
    for sid, ts, xyz, labels, _color in dataset._iter_semantic_lidar_frames(vehicle_ids):
        yield sid, int(ts), np.asarray(xyz, dtype=np.float64), np.asarray(labels, dtype=np.int64)
