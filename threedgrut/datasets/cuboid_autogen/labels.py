# SPDX-License-Identifier: Apache-2.0
"""lidar-sseg 数字类 → NCore cuboid 字符串类。

值必须逐字等于 ``tracks_loader.DEFAULT_VEHICLE_CLASSES``（由 test 契约 pin），
否则下游 ``obs.class_id in class_filter`` 全 False → obs 静默滤空。

lidar-sseg Cityscapes 调色板（见 ncore_semantic.py）：13=car / 14=truck / 15=bus。
"""
from __future__ import annotations

from typing import Optional

LIDAR_SSEG_TO_CUBOID_CLASS: dict[int, str] = {
    13: "automobile",
    14: "heavy_truck",
    15: "bus",
}


def map_class(numeric_id: int) -> Optional[str]:
    """数字 lidar-sseg 类 → cuboid 字符串类；非车辆返回 None。"""
    return LIDAR_SSEG_TO_CUBOID_CLASS.get(int(numeric_id))
