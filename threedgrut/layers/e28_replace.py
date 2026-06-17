# SPDX-License-Identifier: Apache-2.0
"""E2.8 — 系统性全替编排：全 vehicle track 枚举 + bank 分配 + 批 align + 替换。

Task 2 = 纯数据分配（Mac 可测）；Task 3 = 加 PLY align + 调
e25_inject.replace_tracks_in_dyn_node 做粒子替换。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from threedgrut.layers.warmstart_metadata import AssetSpec
from threedgrut.layers.asset_bank import query_bank, BankMiss

logger = logging.getLogger(__name__)

# 与 e25_inject_ah_replace._CAR_CLASS 对齐的 NCore autolabel vehicle 类。
VEHICLE_CLASSES = frozenset({
    "automobile", "bus", "truck", "consumer_vehicles", "car", "vehicle",
})


@dataclass(frozen=True)
class AssignRow:
    track: str
    label_class: str
    chosen_asset: str | None
    fallback_level: int | None
    skipped: bool


def is_vehicle(label_class: str) -> bool:
    return str(label_class).lower() in VEHICLE_CLASSES


def assign_assets_to_tracks(
    recon: dict[str, tuple[str, tuple]],
    bundle: dict[str, AssetSpec],
    *,
    on_miss: str = "global",
) -> tuple[dict[str, str], list[AssignRow]]:
    """``recon``: ``{track_name: (label_class, (L,W,H))}``.

    返回 ``({track_name: asset_hash}, [AssignRow])``。非 vehicle track 不分配；
    bank miss + on_miss='skip' → 该 track 留 recon（skipped=True，不 silent）。
    """
    assign: dict[str, str] = {}
    report: list[AssignRow] = []
    for track, (cls, dims) in recon.items():
        if not is_vehicle(cls):
            continue
        try:
            asset_hash, level = query_bank(bundle, str(cls), dims, on_miss=on_miss)
        except BankMiss:
            report.append(AssignRow(track, str(cls), None, None, True))
            logger.warning("e28: track %r kept recon (bank miss)", track)
            continue
        assign[track] = asset_hash
        report.append(AssignRow(track, str(cls), asset_hash, level, False))
    return assign, report
