#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E2.8 编排 driver — USDZ → 拆 → 全 vehicle track 换 AH 资产 → QA sanity。

一条龙（单 clip）：
  ① 拆   convert_usdz_to_ckpt_with_tracks(usdz)
          → 渲染就绪 ckpt（bg+road+dynamic_rigids，deformable 天然丢弃）
            + viz_4d（ego/track poses on camera timeline）
            + recon {tid:(class,dims)} + name_to_id
  ② 配+③ replace_all_vehicle_tracks(ckpt, bank, recon, name_to_id)
          → 全 vehicle track frozen 换 AH 资产（非 vehicle/bg/road 字节不变）
  ⑤ QA   qa_sanity（覆盖率 + 防烟雾 opacity + skip 计数）pass/fail 闸

产物 ``out/<run>/{ckpt_replaced.pt, replace_report.json, qa_sanity.json}``。
sanity 不过 → 退出码 1（不进协调阶段，先查 replace_report）。

协调（harmonizer）+ 定量 QA（NTA-IoU/FID）是 Task 6（``--with-quant``，inceptio GPU）。

用法（inceptio worktree）::

    python scripts/e28_systematic_replace_pipeline.py \
        --usdz       ~/work/nurec_e0/.../artifacts/last.usdz \
        --asset_bank ~/work/nurec_e0/assets/bundle \
        --out_dir    ~/work/output/e28_run \
        --on_miss    global   # 或 skip
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from threedgrut.layers.e28_replace import (
    inject_recon_tracks,
    keep_only_track_slots,
    qa_sanity,
    replace_all_vehicle_tracks,
    select_vehicle_tracks_to_place,
    split_vehicle_tracks_by_ah_match,
)
from threedgrut.layers.warmstart_metadata import load_bundle_metadata
from threedgrut_playground.utils.nre_usdz_viz4d import (
    add_sky_from_recon,
    convert_usdz_to_ckpt_with_tracks,
)


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--usdz", required=True, help="NRE training-checkpoint USDZ (last.usdz)")
    ap.add_argument("--asset_bank", required=True, help="dir with metadata.yaml + plys")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_name", default="ckpt_replaced.pt", help="输出 ckpt 文件名（如 packed_ckpt.pt）")
    ap.add_argument(
        "--on_miss",
        default="global",
        choices=["global", "skip"],
        help="bank miss 策略：global=跨class最近 / skip=保留 recon",
    )
    ap.add_argument(
        "--primary_cam", default="camera_front_wide_120fov", help="viz_4d 共享时间轴的主相机（缺则退首个相机）"
    )
    ap.add_argument("--max_pts", type=int, default=None, help="每 track AH 粒子子采样上限（控显存；默认不丢点）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no_insert",
        dest="insert",
        action="store_false",
        help="只 replace 有 gaussian 的 track；默认 insert: 给 active/附近的 " "无 gaussian vehicle cuboid 也放 AH 车",
    )
    ap.set_defaults(insert=True)
    ap.add_argument(
        "--insert_min_active_frames", type=int, default=20, help="insert 候选: track 至少活跃这么多帧（滤掉一闪而过）"
    )
    ap.add_argument(
        "--insert_max_dist_m",
        type=float,
        default=40.0,
        help="insert 候选: track 到 ego 轨迹最近距离 ≤ 此值（只插附近）",
    )
    ap.add_argument(
        "--recon_ckpt",
        default=None,
        help="跨源 recon fallback ckpt（如 NCore baseline ckpt_30000.pt）："
        "AH size 配不好的大车(bus/truck)从此抽真 recon gaussian 注入",
    )
    ap.add_argument(
        "--max_size_ratio",
        type=float,
        default=1.5,
        help="AH 匹配 size 比上限；超过(如 bus 12.5m vs pickup 5.8m=2.16)→ recon",
    )
    ap.add_argument(
        "--keep_nonvehicle",
        action="store_true",
        help="保留非 vehicle(person/rider/stroller)原 NRE dynamic gaussians；"
        "默认丢弃，dynamic_rigids 只留放置的 vehicle（各有 cuboid 框）",
    )
    ap.add_argument(
        "--no_sky", action="store_true", help="不从 recon_ckpt 跨源 MLP sky_envmap（默认有 recon_ckpt 时加天空）"
    )
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = _parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ① 拆 ----------------------------------------------------------------
    print(f"[e28] ① 拆 USDZ → ckpt+viz_4d: {a.usdz}")
    scene = convert_usdz_to_ckpt_with_tracks(
        a.usdz,
        primary_cam=a.primary_cam,
    )
    ckpt = scene.ckpt
    if a.insert:
        # replace ∪ insert: AH 车放到「有 gaussian 的」+「active/附近无 gaussian 的」
        # 全部 vehicle cuboid track（NRE 只重建了部分车，其余只有 cuboid pose）。
        recon, name_to_id = select_vehicle_tracks_to_place(
            scene.vehicle_catalog,
            min_active_frames=a.insert_min_active_frames,
            max_dist_m=a.insert_max_dist_m,
        )
        n_present = sum(1 for t in recon if scene.vehicle_catalog[t]["present"])
        n_insert = len(recon) - n_present
        print(
            f"[e28]   seq={scene.sequence_id} primary_cam={scene.primary_cam} "
            f"vehicle_catalog={len(scene.vehicle_catalog)} → place {len(recon)} "
            f"(replace {n_present} + insert {n_insert}; "
            f"active≥{a.insert_min_active_frames}f & ≤{a.insert_max_dist_m}m)"
        )
    else:
        recon, name_to_id = scene.recon, scene.name_to_id
        print(
            f"[e28]   seq={scene.sequence_id} primary_cam={scene.primary_cam} "
            f"replace-only present_tracks={len(recon)}"
        )

    # ② 配 + ③ 批注入 -----------------------------------------------------
    bundle = load_bundle_metadata(Path(a.asset_bank) / "metadata.yaml")
    recon_placed: list = []
    if a.recon_ckpt:
        # size-gate: AH 配不好的大车(bus/truck)走跨源 recon，其余走 AH。
        ah_recon, recon_tids = split_vehicle_tracks_by_ah_match(
            recon, bundle, max_size_ratio=a.max_size_ratio, on_miss=a.on_miss
        )
        print(
            f"[e28] ② bank assets={len(bundle)} → AH-match {len(ah_recon)} + "
            f"cross-source recon {len(recon_tids)} {recon_tids}"
        )
    else:
        ah_recon, recon_tids = recon, []
        print(f"[e28] ② bank assets={len(bundle)} → ③ replace/insert {len(ah_recon)} " f"(on_miss={a.on_miss})")

    ckpt, report = replace_all_vehicle_tracks(
        ckpt,
        bundle_root=a.asset_bank,
        bundle=bundle,
        recon=ah_recon,
        name_to_id=name_to_id,
        on_miss=a.on_miss,
        max_pts=a.max_pts,
        seed=a.seed,
    )

    recon_ckpt = None
    if a.recon_ckpt:
        recon_ckpt = torch.load(a.recon_ckpt, weights_only=False, map_location="cpu")
    if recon_ckpt is not None and recon_tids:
        print(f"[e28] ③b 跨源 recon 注入 (from {a.recon_ckpt})")
        ckpt, recon_placed = inject_recon_tracks(ckpt, recon_ckpt, recon_tids, name_to_id)
        missing = sorted(set(recon_tids) - set(recon_placed))
        print(
            f"[e28]   recon 注入 {len(recon_placed)} {sorted(recon_placed)}"
            + (f"; recon_ckpt 也没有 {missing}（留空 cuboid）" if missing else "")
        )

    # ③c 跨源天空（MLP sky_envmap，无需 nvdiffrast）
    if recon_ckpt is not None and not a.no_sky:
        if add_sky_from_recon(ckpt, recon_ckpt):
            print(f"[e28] ③c 跨源 MLP sky_envmap 已加（layers.enabled += sky_envmap）")
        else:
            print(f"[e28] ③c recon_ckpt 无 sky_envmap_state，跳过天空")

    # ④ 清理：默认丢弃非 vehicle(person等)原 NRE dynamic gaussians，只留放置的
    # vehicle 簇 → 每个 dynamic_rigids 簇都被自己的 cuboid 框住（大g bbox 检验）。
    if not a.keep_nonvehicle:
        placed_slots = {name_to_id[r.track] for r in report if not r.skipped}
        placed_slots |= {name_to_id[t] for t in recon_placed if t in name_to_id}
        dyn = ckpt["model"]["gaussians_nodes"]["dynamic_rigids"]
        n0 = dyn["positions"].shape[0]
        ckpt["model"]["gaussians_nodes"]["dynamic_rigids"] = keep_only_track_slots(dyn, placed_slots)
        n1 = ckpt["model"]["gaussians_nodes"]["dynamic_rigids"]["positions"].shape[0]
        print(f"[e28] ④ drop 非 vehicle dynamic gaussians: {n0}→{n1} " f"(只留 {len(placed_slots)} 个 vehicle slot)")

    torch.save(ckpt, out / a.out_name)
    with open(out / "replace_report.json", "w") as f:
        json.dump([asdict(r) for r in report], f, indent=2)
    print(
        f"[e28]   wrote {a.out_name} + replace_report.json "
        f"(AH {sum(1 for r in report if not r.skipped)} + recon {len(recon_placed)} / "
        f"skipped {sum(1 for r in report if r.skipped)})"
    )

    # ⑤ QA sanity 闸 ------------------------------------------------------
    dyn = ckpt["model"]["gaussians_nodes"]["dynamic_rigids"]
    replaced_slots = {name_to_id[r.track] for r in report if not r.skipped}
    qa = qa_sanity(dyn, report, replaced_slots=replaced_slots)
    with open(out / "qa_sanity.json", "w") as f:
        json.dump(qa, f, indent=2)
    print(
        f"[e28] ⑤ QA sanity: coverage={qa['coverage']:.3f} "
        f"opacity_med={qa['opacity_median']:.3f} "
        f"replaced={qa['n_replaced']} skipped={qa['n_skipped']} "
        f"passed={qa['passed']}"
    )
    if not qa["passed"]:
        print("[e28] ✗ QA sanity FAILED — 不进协调阶段，先查 replace_report.json")
        return 1
    print(f"[e28] ✓ done → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
