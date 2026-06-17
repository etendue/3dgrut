# SPDX-License-Identifier: Apache-2.0
"""E2.8 — 系统性全替编排：全 vehicle track 枚举 + bank 分配 + 批 align + 替换。

Task 2 = 纯数据分配（Mac 可测）；Task 3 = 加 PLY align + 调
e25_inject.replace_tracks_in_dyn_node 做粒子替换。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from threedgrut.layers.warmstart_metadata import AssetSpec, resolve_ply_path
from threedgrut.layers.warmstart_ply import (
    AlignedAsset,
    apply_alignment,
    asset_extent,
    compute_axis_alignment,
    load_warmstart_ply,
    subsample_asset,
)
from threedgrut.layers.e25_inject import flip_forward_180, replace_tracks_in_dyn_node
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


def _align_asset(
    spec: AssetSpec,
    bundle_root: Path,
    dims,
    *,
    max_pts: int | None = None,
    flip_forward: bool = True,
    generator=None,
) -> AlignedAsset:
    """PLY → 填 recon live cuboid ``dims`` 的 AlignedAsset（含 180° yaw-flip）。

    搬自 ``scripts/e25_inject_ah_replace.py`` 的 align 循环（A800/inceptio 已验证）：
    load PLY → asset_extent → compute_axis_alignment 填 recon live cuboid
    size/center → flip_forward_180（NCore cuboid forward 与 AH canonical 相反）→
    apply_alignment。``dims`` 用 recon track 的 **live cuboid**，不是 AH metadata。
    ``max_pts`` 给定时按 generator 子采样（控显存）；默认不丢点。
    """
    ply = resolve_ply_path(bundle_root, spec)
    asset = load_warmstart_ply(ply)
    half, center = asset_extent(asset)
    xf = compute_axis_alignment(spec.label_class, dims, half, center)
    if flip_forward:
        xf = flip_forward_180(xf)
    aligned = apply_alignment(asset, xf)
    if max_pts is not None:
        aligned = subsample_asset(aligned, max_pts, generator=generator)
    return aligned


def replace_all_vehicle_tracks(
    ckpt: dict, *, bundle_root, bundle, recon, name_to_id, on_miss="global",
    max_pts: int | None = None, seed: int = 0,
) -> tuple[dict, list[AssignRow]]:
    """全 vehicle track 批替换。``recon``: {track_name:(class,dims)}；
    ``name_to_id``: build_name_to_int_id(tracks)。返回 (ckpt, report)。

    复用 e25_inject.replace_tracks_in_dyn_node（已守护非目标 track + bg/road
    字节不变）。非 vehicle track / skip 的 track 保持 recon 不动。``max_pts`` 给定
    时每 track 子采样到该上限（控显存，deterministic by ``seed``）。
    """
    assign, report = assign_assets_to_tracks(recon, bundle, on_miss=on_miss)
    dyn = ckpt["model"]["gaussians_nodes"]["dynamic_rigids"]
    gen = torch.Generator().manual_seed(seed) if max_pts is not None else None
    aligned_by_id: dict[int, AlignedAsset] = {}
    for track, asset_hash in assign.items():
        spec = bundle[asset_hash]
        dims = recon[track][1]
        aligned_by_id[int(name_to_id[track])] = _align_asset(
            spec, Path(bundle_root), dims, max_pts=max_pts, generator=gen
        )
    if aligned_by_id:
        ckpt["model"]["gaussians_nodes"]["dynamic_rigids"] = \
            replace_tracks_in_dyn_node(dyn, aligned_by_id)
    return ckpt, report


def qa_sanity(dyn_node_after: dict, report: list[AssignRow],
              *, opacity_floor: float = 0.15) -> dict:
    """廉价 sanity 闸（Mac 可跑）。覆盖率 + opacity 防烟雾 + skip 计数。

    opacity_floor 0.15：E2.7 烟雾态 dynamic gaussian opacity≈0.11，正常 AH
    资产应远高于此 → median ≤ floor 判 fail（防烟雾回归）。
    """
    total = len(report)
    skipped = [r for r in report if r.skipped]
    replaced = [r for r in report if not r.skipped]
    coverage = (len(replaced) / total) if total else 1.0
    opacity = torch.sigmoid(dyn_node_after["density"].flatten())
    opacity_median = float(opacity.median()) if opacity.numel() else 0.0
    passed = (opacity_median > opacity_floor) and (len(replaced) > 0)
    return {
        "coverage": coverage,
        "n_total": total,
        "n_replaced": len(replaced),
        "n_skipped": len(skipped),
        "skipped_tracks": [r.track for r in skipped],
        "opacity_median": opacity_median,
        "passed": passed,
    }
