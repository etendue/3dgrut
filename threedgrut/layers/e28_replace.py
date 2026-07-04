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

from threedgrut.layers.asset_bank import BankMiss, query_bank
from threedgrut.layers.e25_inject import flip_forward_180, replace_tracks_in_dyn_node
from threedgrut.layers.warmstart_metadata import AssetSpec, resolve_ply_path
from threedgrut.layers.warmstart_ply import (
    AlignedAsset,
    apply_alignment,
    asset_extent,
    compute_axis_alignment,
    load_warmstart_ply,
    subsample_asset,
)

logger = logging.getLogger(__name__)

# 与 e25_inject_ah_replace._VEHICLE_CLASS_TOKENS 对齐的 NCore autolabel vehicle
# 类标记。**子串**匹配（非精确）—— 真实 autolabel 含 ``heavy_truck`` /
# ``pickup_truck`` 等复合类（E2.5 实测），精确匹配会漏替（E2.8 inceptio convert
# 实测：dynamic_rigids 同时含 automobile + heavy_truck + person）。
VEHICLE_CLASSES = frozenset(
    {
        "automobile",
        "bus",
        "truck",
        "consumer_vehicles",
        "car",
        "vehicle",
    }
)


@dataclass(frozen=True)
class AssignRow:
    track: str
    label_class: str
    chosen_asset: str | None
    fallback_level: int | None
    skipped: bool


def is_vehicle(label_class: str) -> bool:
    """子串匹配（捕获 heavy_truck/pickup_truck 等复合类）；person/VRU/cyclist 不命中。"""
    cls = str(label_class).lower()
    return any(tok in cls for tok in VEHICLE_CLASSES)


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


def select_vehicle_tracks_to_place(
    catalog: dict,
    *,
    min_active_frames: int = 20,
    max_dist_m: float = 40.0,
) -> tuple[dict, dict]:
    """E2.8 insert: pick which vehicle tracks get an AH car (replace ∪ insert).

    ``catalog``: ``{tid: {class, dims, slot, active_frames, min_dist_to_ego,
    present}}`` from ``build_vehicle_catalog``. **Present** tracks (already have
    gaussians → replace) are always kept. **Gaussian-less** tracks are inserted
    only if ``active_frames >= min_active_frames`` AND within ``max_dist_m`` of
    the ego trajectory (active/nearby, 大g 2026-06-17) — drops distant /
    blink-and-gone vehicles. Returns ``(recon {tid:(class,dims)}, name_to_id
    {tid:slot})`` ready for :func:`replace_all_vehicle_tracks` (which treats an
    empty-slot insert as a degenerate replace).
    """
    recon: dict = {}
    name_to_id: dict = {}
    for tid, info in catalog.items():
        keep = info["present"] or (info["active_frames"] >= min_active_frames and info["min_dist_to_ego"] <= max_dist_m)
        if keep:
            recon[tid] = (info["class"], tuple(info["dims"]))
            name_to_id[tid] = info["slot"]
    return recon, name_to_id


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
    ckpt: dict,
    *,
    bundle_root,
    bundle,
    recon,
    name_to_id,
    on_miss="global",
    max_pts: int | None = None,
    seed: int = 0,
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
        ckpt["model"]["gaussians_nodes"]["dynamic_rigids"] = replace_tracks_in_dyn_node(dyn, aligned_by_id)
    return ckpt, report


# ---------------------------------------------------------------------------
# E2.8 cross-source recon fallback (大g 2026-06-17): vehicles the AH bank can't
# size-match (bus/heavy_truck) get their REAL recon gaussian from a sibling
# checkpoint (NCore baseline ckpt has them; NRE USDZ didn't). Same clip → track
# poses match (0.4°) → no frame correction (verified |axis·vel|=0.99).
# ---------------------------------------------------------------------------
_DYN_PARTICLE_KEYS = (
    "positions",
    "rotation",
    "scale",
    "density",
    "features_albedo",
    "features_specular",
    "track_ids",
)


def _sorted_dims_ratio(a, b) -> float:
    """Orientation-agnostic size mismatch: max over sorted dims of the larger/
    smaller ratio. bus(12.5,3.5,3.1) vs pickup(5.8,2.25,1.9) → 2.16."""
    sa = sorted((float(x) for x in a), reverse=True)
    sb = sorted((float(x) for x in b), reverse=True)
    return max(max(x / max(y, 1e-6), y / max(x, 1e-6)) for x, y in zip(sa, sb))


def split_vehicle_tracks_by_ah_match(
    recon_sel: dict,
    bundle: dict,
    *,
    max_size_ratio: float = 1.5,
    on_miss: str = "global",
) -> tuple[dict, list[str]]:
    """Route each selected vehicle track to AH vs cross-source recon by size.

    A track goes to **recon** if its best bank match's :func:`_sorted_dims_ratio`
    exceeds ``max_size_ratio`` (AH would be a stretched wrong vehicle, e.g. a
    5.8 m pickup on a 12.5 m bus) or the bank has nothing. Returns
    ``(ah_recon {tid:(class,dims)}, recon_tids [tid])``.
    """
    ah_recon: dict = {}
    recon_tids: list[str] = []
    for tid, (cls, dims) in recon_sel.items():
        try:
            h, _ = query_bank(bundle, str(cls), dims, on_miss=on_miss)
        except BankMiss:
            recon_tids.append(tid)
            continue
        if _sorted_dims_ratio(dims, bundle[h].cuboids_dims) <= max_size_ratio:
            ah_recon[tid] = (cls, dims)
        else:
            recon_tids.append(tid)
            logger.info(
                "e28: track %r (%s, dims=%s) → recon (AH size mismatch)",
                tid,
                cls,
                tuple(round(float(x), 1) for x in dims),
            )
    return ah_recon, recon_tids


def place_tracks_in_dyn_node(dyn_node: dict, node_tensors_by_id: dict) -> dict:
    """Rebuild dynamic_rigids with pre-built per-track object-local tensors.

    Source-agnostic sibling of :func:`e25_inject.replace_tracks_in_dyn_node`:
    ``node_tensors_by_id`` = ``{int_slot: {positions, rotation, scale, density,
    features_albedo, features_specular}}`` (already in the node's spec_dim).
    Empty-slot ids insert, existing replace, untouched tracks byte-identical.
    """
    track_ids = dyn_node["track_ids"]
    target_ids = sorted(int(t) for t in node_tensors_by_id)
    keep = ~torch.isin(track_ids, torch.tensor(target_ids, dtype=track_ids.dtype))
    new: dict = {}
    for key in ("positions", "rotation", "scale", "density", "features_albedo", "features_specular"):
        kept = dyn_node[key][keep]
        parts = [node_tensors_by_id[t][key] for t in target_ids]
        new[key] = torch.nn.Parameter(torch.cat([kept, *parts], dim=0), requires_grad=False)
    ah_ids = [torch.full((node_tensors_by_id[t]["positions"].shape[0],), t, dtype=track_ids.dtype) for t in target_ids]
    new["track_ids"] = torch.cat([track_ids[keep], *ah_ids], dim=0)
    for k, v in dyn_node.items():
        if k not in _DYN_PARTICLE_KEYS:
            new[k] = v
    return new


def extract_recon_node_tensors(
    recon_dyn: dict,
    recon_sorted_tids: list,
    tid_to_slot: dict,
    spec_dim: int,
) -> dict:
    """Pull object-local node tensors for ``tids`` from a recon ckpt's
    dynamic_rigids, keyed by the TARGET (USDZ) slot. ``features_specular`` is
    padded/truncated to ``spec_dim`` so it concats into the target node.
    """
    rtids = recon_dyn["track_ids"]
    out: dict = {}
    for tid, usdz_slot in tid_to_slot.items():
        if tid not in recon_sorted_tids:
            continue
        mask = rtids == recon_sorted_tids.index(tid)
        n = int(mask.sum())
        if n == 0:
            continue
        nt = {}
        for key in ("positions", "rotation", "scale", "density", "features_albedo"):
            nt[key] = recon_dyn[key][mask].detach().cpu().float().contiguous()
        fs = recon_dyn["features_specular"][mask].detach().cpu().float()
        if fs.shape[1] >= spec_dim:
            fs = fs[:, :spec_dim]
        else:
            fs = torch.cat([fs, torch.zeros(n, spec_dim - fs.shape[1])], dim=1)
        nt["features_specular"] = fs.contiguous()
        out[int(usdz_slot)] = nt
    return out


def inject_recon_tracks(
    ckpt: dict,
    recon_ckpt: dict,
    recon_tids: list,
    name_to_id: dict,
) -> tuple[dict, list[str]]:
    """Inject sibling-ckpt recon gaussians at the USDZ slots for ``recon_tids``.

    For bus/truck the AH bank can't match: their object-local recon gaussians
    live in ``recon_ckpt`` (same clip). Track poses match cross-source so no
    frame fix is needed (verified). Returns ``(ckpt, placed_tids)``.
    """
    dyn = ckpt["model"]["gaussians_nodes"]["dynamic_rigids"]
    spec_dim = dyn["features_specular"].shape[1]
    rdyn = recon_ckpt["model"]["gaussians_nodes"]["dynamic_rigids"]
    rsorted = sorted(recon_ckpt["viz_4d"]["tracks"].keys())
    tid_to_slot = {tid: name_to_id[tid] for tid in recon_tids if tid in name_to_id}
    node_tensors = extract_recon_node_tensors(rdyn, rsorted, tid_to_slot, spec_dim)
    if node_tensors:
        ckpt["model"]["gaussians_nodes"]["dynamic_rigids"] = place_tracks_in_dyn_node(dyn, node_tensors)
    placed = [tid for tid, slot in tid_to_slot.items() if int(slot) in node_tensors]
    return ckpt, placed


def keep_only_track_slots(dyn_node: dict, keep_slots) -> dict:
    """Drop dynamic_rigids gaussians whose ``track_ids`` slot ∉ ``keep_slots``.

    E2.8 clean scene (大g 2026-06-17 bbox test): NRE put some pedestrians in
    dynamic_rigids as oversized smoky blobs (person cuboid 0.6 m but gaussian
    span ~5 m); with vehicle-only cuboid display they render boxless = "asset
    without bbox". Keeping only the placed vehicle slots leaves every cluster
    boxed by its own vehicle cuboid. Non-particle metadata carried over.
    """
    track_ids = dyn_node["track_ids"]
    keep = torch.isin(
        track_ids,
        torch.tensor(sorted(int(s) for s in keep_slots), dtype=track_ids.dtype),
    )
    new: dict = {}
    for key in ("positions", "rotation", "scale", "density", "features_albedo", "features_specular"):
        new[key] = torch.nn.Parameter(dyn_node[key][keep].contiguous(), requires_grad=False)
    new["track_ids"] = track_ids[keep]
    for k, v in dyn_node.items():
        if k not in _DYN_PARTICLE_KEYS:
            new[k] = v
    return new


def qa_sanity(
    dyn_node_after: dict, report: list[AssignRow], *, opacity_floor: float = 0.02, replaced_slots=None
) -> dict:
    """廉价 sanity 闸（Mac 可跑）。覆盖率 + opacity 防退化 + skip 计数。

    opacity_floor 0.02（anti-degenerate）：E2.8 inceptio 实测——NRE 重建的 dynamic
    gaussian per-gaussian opacity 中位数本就只有 ~0.08（recon 0.081 / AH 替换
    0.103 / background 0.056；road 0.988）。plan 原设 0.15「防烟雾」是误标定：
    per-gaussian opacity **无法**区分「0.11 烟雾」与正常值（两者重叠在 ~0.1）。
    故 floor 降为只挡 **near-zero 退化注入**（convention bug → opacity≈0），真
    「烟雾/弥散」感知问题交给 Task 6 的 FID + viser 目视。

    ``replaced_slots`` 给定时 opacity 只统计**替换进来的 AH 粒子**（按 track_ids
    掩码），不被未替换的 recon 行人粒子污染；缺省则统计整节点（back-compat）。
    """
    total = len(report)
    skipped = [r for r in report if r.skipped]
    replaced = [r for r in report if not r.skipped]
    coverage = (len(replaced) / total) if total else 1.0

    op = torch.sigmoid(dyn_node_after["density"].flatten())
    if replaced_slots is not None and "track_ids" in dyn_node_after:
        tids = dyn_node_after["track_ids"].flatten()
        slots = torch.as_tensor(sorted(int(s) for s in replaced_slots), dtype=tids.dtype)
        mask = torch.isin(tids, slots) if slots.numel() else torch.zeros_like(tids, dtype=torch.bool)
        op = op[mask]
    opacity_median = float(op.median()) if op.numel() else 0.0

    passed = (opacity_median > opacity_floor) and (len(replaced) > 0)
    return {
        "coverage": coverage,
        "n_total": total,
        "n_replaced": len(replaced),
        "n_skipped": len(skipped),
        "skipped_tracks": [r.track for r in skipped],
        "opacity_median": opacity_median,
        "opacity_floor": opacity_floor,
        "opacity_scope": "replaced_only" if replaced_slots is not None else "whole_node",
        "passed": passed,
    }
