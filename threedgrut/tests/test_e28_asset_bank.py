# SPDX-License-Identifier: Apache-2.0
"""E2.8 Task 1 — AssetBank query (class filter + L2 nearest + fallback ladder)."""

import pytest

from threedgrut.layers.asset_bank import BankMiss, query_bank
from threedgrut.layers.warmstart_metadata import AssetSpec


def _spec(h, cls, dims):
    return AssetSpec(asset_hash=h, ply_file=f"{cls}/{h}/gaussians.ply", label_class=cls, cuboids_dims=tuple(dims))


BUNDLE = {
    "sedan1": _spec("sedan1", "consumer_vehicles", (4.5, 1.8, 1.5)),
    "suv1": _spec("suv1", "consumer_vehicles", (4.9, 2.0, 1.8)),
    "bus1": _spec("bus1", "bus", (12.0, 2.5, 3.2)),
}


def test_same_class_nearest_size():
    # 4.6×1.85×1.55 最接近 sedan1
    h, level = query_bank(BUNDLE, "consumer_vehicles", (4.6, 1.85, 1.55))
    assert h == "sedan1"
    assert level == 0  # same-class exact-ish


def test_one_asset_reused_across_calls():
    # bank 查询不消耗资产：同一资产可被多次返回（区别 bijection）
    h1, _ = query_bank(BUNDLE, "consumer_vehicles", (4.5, 1.8, 1.5))
    h2, _ = query_bank(BUNDLE, "consumer_vehicles", (4.5, 1.8, 1.5))
    assert h1 == h2 == "sedan1"


def test_cross_class_fallback_warns_level1():
    # truck 类 bank 没有 → 跨 class 全局最近 + level 1
    h, level = query_bank(BUNDLE, "truck", (11.5, 2.5, 3.0))
    assert h == "bus1"  # 全局 L2 最近
    assert level == 1


def test_on_miss_skip_raises_bankmiss():
    empty = {}
    with pytest.raises(BankMiss):
        query_bank(empty, "consumer_vehicles", (4.5, 1.8, 1.5), on_miss="skip")


def test_deterministic_tie_break():
    # 两资产等距 → 按 hash 字典序定 deterministic
    b = {"a": _spec("a", "c", (1, 1, 1)), "b": _spec("b", "c", (1, 1, 1))}
    h, _ = query_bank(b, "c", (1, 1, 1))
    assert h == "a"
