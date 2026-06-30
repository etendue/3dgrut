# SPDX-License-Identifier: Apache-2.0
"""Task 1 单测 — lidar-sseg 数字类 → cuboid 字符串类映射。

纯 stdlib；Mac CPU 无 NCore SDK 即可跑（conftest stub ncore/sklearn）。
"""
from __future__ import annotations

from threedgrut.datasets.cuboid_autogen.labels import (
    LIDAR_SSEG_TO_CUBOID_CLASS,
    map_class,
)
from threedgrut.datasets.tracks_loader import DEFAULT_VEHICLE_CLASSES


def test_known_vehicle_ids():
    assert map_class(13) == "automobile"
    assert map_class(14) == "heavy_truck"
    assert map_class(15) == "bus"


def test_non_vehicle_returns_none():
    assert map_class(11) is None  # person
    assert map_class(0) is None   # road
    assert map_class(999) is None  # unknown


def test_mapping_values_exactly_match_class_filter():
    # 关键契约：映射出的字符串必须逐字等于 tracks_loader 的 class_filter，
    # 否则 obs.class_id in class_filter 全 False → obs 静默滤空。
    assert set(LIDAR_SSEG_TO_CUBOID_CLASS.values()) == set(DEFAULT_VEHICLE_CLASSES)
