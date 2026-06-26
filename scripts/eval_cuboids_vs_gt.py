#!/usr/bin/env python3
"""Task 13: 自动 cuboids vs GT cuboids 定量验证（BEV-IoU / precision-recall / center-err）。

在有 GT 的 clip（如 9ae）上：auto loader（cuboids instance=auto_v0）+ GT loader
（cuboids="default"）各经 load_tracks_from_ncore_cuboids 转同一 cam 时间轴的逐帧框，
按中心距贪心匹配，报告工具准确度。

注：auto 只含**动态**车（动静过滤后），GT 含全部车（静+动）→ recall 上限受限于
"动态车/全部车"，故 precision / 匹配对的 center-err / BEV-IoU 更能反映拟合准确度。

Usage (inceptio):
    python scripts/eval_cuboids_vs_gt.py \
        --meta ~/work/data/9ae151dc_consolidated/pai_...json \
        --shard ~/work/9ae_autocuboids/autocuboids.ncore4-auto_cuboids.zarr.itar \
        [--max-dist 2.0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _yaw_of(pose):
    return float(np.arctan2(pose[1, 0], pose[0, 0]))


def _tracks_to_frame_boxes(tracks, num_frames, T=None):
    """{tid:{poses[F,4,4],size,frame_info}} → list[ per-frame list[(cx,cy,l,w,yaw)] ]。

    T（可选 4x4）：把每帧 pose 左乘 T 对齐坐标系。auto 框已在 world-global（点经
    T_world_to_world_global），GT 经 transform("world") 是 world → 须乘 T_world_to_world_global
    才与 auto 同系。
    """
    frame_boxes = [[] for _ in range(num_frames)]
    for t in tracks.values():
        poses = np.asarray(t["poses"])
        size = np.asarray(t["size"])
        info = np.asarray(t["frame_info"])
        l, w = float(size[0]), float(size[1])
        for fi in range(num_frames):
            if not bool(info[fi]):
                continue
            p = poses[fi]
            if T is not None:
                p = T @ p
            frame_boxes[fi].append((float(p[0, 3]), float(p[1, 3]), l, w, _yaw_of(p)))
    return frame_boxes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--meta", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--max-dist", type=float, default=2.0, help="匹配中心距阈 (m)")
    ap.add_argument("--instance-name", default="auto_v0")
    args = ap.parse_args()

    import hydra
    import ncore.data as _nd
    import ncore.data.v4 as v4

    from threedgrut import datasets
    from threedgrut.datasets.cuboid_autogen.bev_metric import bev_iou, match_boxes
    from threedgrut.datasets.cuboid_autogen.cluster import wrap_to_pi
    from threedgrut.datasets.tracks_loader import load_tracks_from_ncore_cuboids

    with hydra.initialize(config_path="../configs", version_base=None):
        conf = hydra.compose(
            config_name="apps/ncore_3dgut_mcmc_multilayer",
            overrides=[
                f"path={args.meta}", "trainer.sky_backend=mlp", "dataset.load_aux_masks=true",
                "+dataset.load_auto_cuboids=true", f"+dataset.auto_cuboids_shard_path={args.shard}",
                f"+dataset.auto_cuboids_instance_name={args.instance_name}",
            ],
        )
    ds, _ = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)
    auto_loader = ds.sequence_loaders[ds.sequence_id]

    # GT loader：独立 reader，cuboids 默认 "default"（GT autolabels）
    gt_loader = v4.SequenceLoaderV4(
        v4.SequenceComponentGroupsReader([args.meta], open_consolidated=True),
        poses_component_group_name="default",
        intrinsics_component_group_name="default",
        masks_component_group_name="default",
    )

    ref_cam = ds.camera_ids[0]
    ref_sensor = ds.sequence_camera_sensors[ds.sequence_id][ref_cam]
    cam_ts = ref_sensor.frames_timestamps_us[:, _nd.FrameTimepoint.END]
    tr = ds.time_range_us
    cam_ts_active = np.asarray(cam_ts)[np.array([int(t) in tr for t in cam_ts])]
    F = int(cam_ts_active.shape[0])

    auto_tracks = load_tracks_from_ncore_cuboids(auto_loader, cam_ts_active)
    gt_tracks = load_tracks_from_ncore_cuboids(gt_loader, cam_ts_active)
    T_glob = np.asarray(ds.T_world_to_world_global)
    auto_fb = _tracks_to_frame_boxes(auto_tracks, F)               # auto 已 world-global
    gt_fb = _tracks_to_frame_boxes(gt_tracks, F, T=T_glob)         # GT world → global 对齐

    tp = fp = fn = 0
    cerr, ious, yerr = [], [], []
    for fi in range(F):
        a, g = auto_fb[fi], gt_fb[fi]
        pairs = match_boxes(a, g, max_dist=args.max_dist)
        tp += len(pairs)
        fp += len(a) - len(pairs)
        fn += len(g) - len(pairs)
        for ai, gi in pairs:
            cerr.append(float(np.hypot(a[ai][0] - g[gi][0], a[ai][1] - g[gi][1])))
            ious.append(bev_iou(a[ai], g[gi]))
            yerr.append(abs(wrap_to_pi(a[ai][4] - g[gi][4])))

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    print(f"=== auto vs GT (BEV match, center-dist <= {args.max_dist}m) ===")
    print(f"frames={F}  GT boxes={sum(len(x) for x in gt_fb)}  auto boxes={sum(len(x) for x in auto_fb)}")
    print(f"TP={tp} FP={fp} FN={fn}")
    print(f"precision={prec:.3f}  recall={rec:.3f}  (recall 受 auto 仅动态/GT 全部车 限制)")
    if cerr:
        print(f"matched: mean center-err={np.mean(cerr):.3f}m  mean BEV-IoU={np.mean(ious):.3f}  "
              f"yaw-MAE={np.degrees(np.mean(yerr)):.1f}deg  (n={len(cerr)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
