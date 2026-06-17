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

from threedgrut.layers.warmstart_metadata import load_bundle_metadata
from threedgrut.layers.e28_replace import replace_all_vehicle_tracks, qa_sanity
from threedgrut_playground.utils.nre_usdz_viz4d import convert_usdz_to_ckpt_with_tracks


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--usdz", required=True, help="NRE training-checkpoint USDZ (last.usdz)")
    ap.add_argument("--asset_bank", required=True, help="dir with metadata.yaml + plys")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--on_miss", default="global", choices=["global", "skip"],
                    help="bank miss 策略：global=跨class最近 / skip=保留 recon")
    ap.add_argument("--primary_cam", default="camera_front_wide_120fov",
                    help="viz_4d 共享时间轴的主相机（缺则退首个相机）")
    ap.add_argument("--max_pts", type=int, default=None,
                    help="每 track AH 粒子子采样上限（控显存；默认不丢点）")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = _parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ① 拆 ----------------------------------------------------------------
    print(f"[e28] ① 拆 USDZ → ckpt+viz_4d: {a.usdz}")
    scene = convert_usdz_to_ckpt_with_tracks(
        a.usdz, primary_cam=a.primary_cam,
    )
    ckpt, recon, name_to_id = scene.ckpt, scene.recon, scene.name_to_id
    print(f"[e28]   seq={scene.sequence_id} primary_cam={scene.primary_cam} "
          f"present_tracks={len(recon)}")

    # ② 配 + ③ 批注入 -----------------------------------------------------
    bundle = load_bundle_metadata(Path(a.asset_bank) / "metadata.yaml")
    print(f"[e28] ② bank assets={len(bundle)} → ③ replace all vehicle tracks "
          f"(on_miss={a.on_miss})")
    ckpt, report = replace_all_vehicle_tracks(
        ckpt, bundle_root=a.asset_bank, bundle=bundle, recon=recon,
        name_to_id=name_to_id, on_miss=a.on_miss, max_pts=a.max_pts, seed=a.seed,
    )

    torch.save(ckpt, out / "ckpt_replaced.pt")
    with open(out / "replace_report.json", "w") as f:
        json.dump([asdict(r) for r in report], f, indent=2)
    print(f"[e28]   wrote ckpt_replaced.pt + replace_report.json "
          f"({sum(1 for r in report if not r.skipped)} replaced / "
          f"{sum(1 for r in report if r.skipped)} skipped)")

    # ⑤ QA sanity 闸 ------------------------------------------------------
    dyn = ckpt["model"]["gaussians_nodes"]["dynamic_rigids"]
    replaced_slots = {name_to_id[r.track] for r in report if not r.skipped}
    qa = qa_sanity(dyn, report, replaced_slots=replaced_slots)
    with open(out / "qa_sanity.json", "w") as f:
        json.dump(qa, f, indent=2)
    print(f"[e28] ⑤ QA sanity: coverage={qa['coverage']:.3f} "
          f"opacity_med={qa['opacity_median']:.3f} "
          f"replaced={qa['n_replaced']} skipped={qa['n_skipped']} "
          f"passed={qa['passed']}")
    if not qa["passed"]:
        print("[e28] ✗ QA sanity FAILED — 不进协调阶段，先查 replace_report.json")
        return 1
    print(f"[e28] ✓ done → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
