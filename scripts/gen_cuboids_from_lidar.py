#!/usr/bin/env python3
"""纯 LiDAR 聚类生成车辆 cuboids → 写 NCore V4 cuboids shard（cuboid_autogen 1a→1e）。

Usage (inceptio, env 3dgrut2):
    python scripts/gen_cuboids_from_lidar.py \
        --meta /home/inceptio/work/data/<clip>/pai_<clip>.json \
        --out  /home/inceptio/work/<clip>_autocuboids \
        [--eps 0.8] [--min-samples 10] [--min-speed 0.5] [--instance-name auto_v0] \
        [--validate-against-gt]

前置：clip 必须有 ``aux.lidar-sseg.zarr.itar``（nre-tools ... --lidar-seg-camvis 生成）。
消费：训练时 `+dataset.load_auto_cuboids=true +dataset.auto_cuboids_shard_path=<shard>`。
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps", type=float, default=0.8, help="DBSCAN eps (m)")
    ap.add_argument("--min-samples", type=int, default=10, help="DBSCAN min_samples")
    ap.add_argument("--min-cluster-pts", type=int, default=15, help="cluster 最少点数")
    ap.add_argument("--min-speed", type=float, default=0.5, help="动静过滤速度阈 (m/s)")
    ap.add_argument("--max-center-dist", type=float, default=3.0, help="关联最大中心距 (m)")
    ap.add_argument("--max-yaw-diff", type=float, default=0.6, help="关联最大 yaw 差 (rad)")
    ap.add_argument("--min-track-len", type=int, default=3, help="最短 track 帧数")
    ap.add_argument("--max-gap", type=int, default=5, help="插值最大缺口帧数")
    ap.add_argument("--store-base-name", default="autocuboids")
    ap.add_argument("--instance-name", default="auto_v0",
                    help="cuboids component instance name（读取时 cuboids_component_group_name）")
    ap.add_argument("--config-name", default="apps/ncore_3dgut_mcmc_multilayer")
    ap.add_argument("--single-track", action="store_true",
                    help="单目标模式: 场景已知单一主车, 每帧取点最多的 cluster 串成 1 条连续 "
                         "trajectory(跨遮挡 gap 插值), 不做多目标关联/动静过滤")
    ap.add_argument("--validate-against-gt", action="store_true",
                    help="对比 GT cuboids 报告 precision/recall/BEV-IoU（Task 13）")
    args = ap.parse_args()

    # --- preflight: aux.lidar-sseg 必须存在 ---
    from threedgrut.datasets.aux_readers import discover_aux_path
    clip_dir = Path(args.meta).parent
    if discover_aux_path(clip_dir, "lidar-sseg") is None:
        print("FATAL: 缺 aux.lidar-sseg.zarr.itar —— 先跑 nre-tools ncore-aux-data "
              "--lidar-seg-camvis 生成逐点语义。", file=sys.stderr)
        return 2

    import hydra

    from threedgrut import datasets
    from threedgrut.datasets.cuboid_autogen.cluster import cluster_points, fit_oriented_box
    from threedgrut.datasets.cuboid_autogen.labels import map_class
    from threedgrut.datasets.cuboid_autogen.lidar_source import iter_vehicle_lidar_frames
    from threedgrut.datasets.cuboid_autogen.track import (
        Box,
        Track,
        aggregate_size,
        associate,
        interpolate_gaps,
        is_dynamic,
    )
    from threedgrut.datasets.cuboid_autogen.v4_writer import (
        tracks_to_observations,
        write_cuboids_shard,
    )

    with hydra.initialize(config_path="../configs", version_base=None):
        conf = hydra.compose(
            config_name=args.config_name,
            overrides=[f"path={args.meta}", "trainer.sky_backend=mlp", "dataset.load_aux_masks=true"],
        )
    print("[gen] building dataset ...", flush=True)
    ds, _ = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)

    # --- 1a→1c: 逐帧动态车辆点 → DBSCAN → 朝向框拟合 ---
    boxes_by_ts: dict[int, list] = defaultdict(list)  # ts -> list[(Box, npts)]
    all_ts: list[int] = []
    n_pts_total = 0
    for _sid, ts, xyz, labels in iter_vehicle_lidar_frames(ds):
        all_ts.append(int(ts))
        n_pts_total += int(xyz.shape[0])
        lab = cluster_points(xyz, eps=args.eps, min_samples=args.min_samples)
        for cid in {int(c) for c in lab.tolist()}:
            if cid < 0:
                continue
            m = lab == cid
            npts = int(m.sum())
            if npts < args.min_cluster_pts:
                continue
            fit = fit_oriented_box(xyz[m])
            if fit is None:
                continue
            center, dim, yaw = fit
            cls = int(np.bincount(labels[m], minlength=16).argmax())
            boxes_by_ts[int(ts)].append(
                (Box(ts=int(ts), center=center, dim=dim, yaw=yaw, cls=cls), npts))

    frame_ts = sorted(boxes_by_ts)
    n_clusters = sum(len(b) for b in boxes_by_ts.values())
    print(f"[gen] {len(frame_ts)} 帧 / {n_clusters} clusters / {n_pts_total} vehicle pts", flush=True)

    # --- 1d: tracking ---
    if args.single_track:
        # 单目标模式: 每帧取点最多的 cluster, 按 ts 串成 1 条连续 trajectory。
        seq = [max(boxes_by_ts[t], key=lambda x: x[1])[0] for t in frame_ts]
        tracks = [Track(seq)]
        print(f"[gen] single-track: {len(frame_ts)} cluster 帧 → 1 track", flush=True)
    else:
        per_frame_boxes = [[b for b, _ in boxes_by_ts[t]] for t in frame_ts]
        tracks = associate(
            per_frame_boxes, max_center_dist_m=args.max_center_dist,
            max_yaw_diff_rad=args.max_yaw_diff, min_track_len=args.min_track_len)
        tracks = [t for t in tracks if is_dynamic(t, min_speed_mps=args.min_speed)]
        print(f"[gen] {len(tracks)} dynamic tracks (after static filter)", flush=True)

    # --- 1e: 统一 size + majority class + 插值 → CuboidTrackObservation ---
    src = ds.sequence_loaders[ds.sequence_id]
    seq_id = src.sequence_id
    interval = src.sequence_timestamp_interval_us
    gmd = src.generic_meta_data

    all_obs = []
    class_hist: dict[str, int] = defaultdict(int)
    for i, t in enumerate(tracks):
        agg = aggregate_size(t)
        for b in t.boxes:
            b.dim = agg
        t = interpolate_gaps(t, sorted(set(all_ts)), max_gap=args.max_gap)
        if args.single_track and len(t.boxes) >= 3:
            # 单车 demo（用户决策：position 主导）：中心轨迹移动平均平滑（消单边回波
            # cluster 中心跳），朝向取平滑轨迹的运动方向（比单边 LiDAR 几何拟合稳得多）。
            C = np.array([b.center for b in t.boxes])
            n = len(t.boxes)
            Cs = np.array([C[max(0, j - 2):min(n, j + 3)].mean(0) for j in range(n)])
            for j, b in enumerate(t.boxes):
                b.center = Cs[j]
            for j, b in enumerate(t.boxes):
                d = Cs[min(n - 1, j + 2)] - Cs[max(0, j - 2)]
                if float(np.hypot(d[0], d[1])) > 1e-3:
                    b.yaw = float(np.arctan2(d[1], d[0]))
        cls_ids = [b.cls for b in t.boxes if b.cls >= 0]
        cls_num = int(np.bincount(cls_ids, minlength=16).argmax()) if cls_ids else 13
        cls_name = map_class(cls_num) or "automobile"
        class_hist[cls_name] += 1
        all_obs.extend(tracks_to_observations(
            [t], track_ids=[f"auto_{i}"], ref_frame_id="world", class_name=cls_name))

    ts_lo = frame_ts[0] if frame_ts else "-"
    ts_hi = frame_ts[-1] if frame_ts else "-"
    print(f"[gen] obs={len(all_obs)}  classes={dict(class_hist)}  ts∈[{ts_lo},{ts_hi}]", flush=True)
    if not all_obs:
        print("[gen] WARNING: 0 obs —— 无动态车辆 track（检查 eps/min-samples/min-speed）",
              file=sys.stderr)

    shard = write_cuboids_shard(
        all_obs, out_dir=args.out, store_base_name=args.store_base_name,
        seq_id=seq_id, interval_us=interval, generic_meta_data=gmd,
        group_name="auto_cuboids", component_instance_name=args.instance_name)
    print(f"[gen] wrote shard: {shard}", flush=True)

    if args.validate_against_gt:
        print("[gen] --validate-against-gt: 见 Task 13（gen_cuboids GT 定量验证）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
