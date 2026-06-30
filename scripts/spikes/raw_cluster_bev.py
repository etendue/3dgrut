#!/usr/bin/env python3
"""诊断: 动静过滤/tracking 前的 raw 逐帧 lidar-sseg 聚类覆盖 BEV。

回答"为什么 auto 框少"——证明原始聚类每帧识别多少物体（动静过滤丢了静止车）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))


def _rect(ax, b, color):
    cx, cy, l, w, yaw = b
    corners = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2], [l/2, w/2]])
    c, s = np.cos(yaw), np.sin(yaw)
    pts = corners @ np.array([[c, -s], [s, c]]).T + [cx, cy]
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", default="/tmp/raw_cluster_bev.png")
    ap.add_argument("--eps", type=float, default=0.8)
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--min-cluster-pts", type=int, default=15)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import hydra

    from threedgrut import datasets
    from threedgrut.datasets.cuboid_autogen.cluster import cluster_points, fit_oriented_box
    from threedgrut.datasets.cuboid_autogen.lidar_source import iter_vehicle_lidar_frames

    with hydra.initialize(config_path="../../configs", version_base=None):
        conf = hydra.compose(
            config_name="apps/ncore_3dgut_mcmc_multilayer",
            overrides=[f"path={args.meta}", "trainer.sky_backend=mlp", "dataset.load_aux_masks=true"])
    ds, _ = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)

    frames = []
    for _sid, ts, xyz, labels in iter_vehicle_lidar_frames(ds):
        lab = cluster_points(xyz, eps=args.eps, min_samples=args.min_samples)
        boxes = []
        for cid in {int(c) for c in lab.tolist()}:
            if cid < 0:
                continue
            m = lab == cid
            if int(m.sum()) < args.min_cluster_pts:
                continue
            fit = fit_oriented_box(xyz[m])
            if fit is not None:
                boxes.append(fit)
        frames.append((int(ts), boxes, xyz))

    counts = [len(b) for _, b, _ in frames]
    print(f"raw clusters/frame: {len(frames)} lidar frames | total {sum(counts)} | "
          f"min {min(counts)} max {max(counts)} mean {np.mean(counts):.1f} | "
          f"frames-with-0: {counts.count(0)}")

    N = len(frames)
    fig, axs = plt.subplots(2, 2, figsize=(18, 18))
    for ax, fi in zip(axs.flat, [N//5, 2*N//5, 3*N//5, 4*N//5]):
        ts, boxes, xyz = frames[fi]
        ax.scatter(xyz[:, 0], xyz[:, 1], s=1, c="lightgray")
        for c, d, y in boxes:  # box = (center[3], dim[3], yaw)
            _rect(ax, (c[0], c[1], d[0], d[1], y), "red")
        ax.set_title(f"lidar frame {fi}: {len(boxes)} raw clusters")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    fig.suptitle("动静过滤/tracking 前 raw lidar-sseg 聚类 (红=cluster box, 灰=车辆点)", fontsize=16)
    fig.savefig(args.out, dpi=70, bbox_inches="tight")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
