#!/usr/bin/env python3
"""目测: 单卡车 cuboid trajectory 俯视图（中心轨迹线 + 每隔几帧的框 + 朝向短杆）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", default="/tmp/truck_traj.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import hydra
    import ncore.data as _nd

    from threedgrut import datasets
    from threedgrut.datasets.tracks_loader import load_tracks_from_ncore_cuboids

    with hydra.initialize(config_path="../../configs", version_base=None):
        conf = hydra.compose(
            config_name="apps/ncore_3dgut_mcmc_multilayer",
            overrides=[f"path={args.meta}", "trainer.sky_backend=mlp",
                       "dataset.load_aux_masks=true", "+dataset.load_auto_cuboids=true",
                       f"+dataset.auto_cuboids_shard_path={args.shard}"])
    ds, _ = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)
    loader = ds.sequence_loaders[ds.sequence_id]
    ref = ds.camera_ids[0]
    rs = ds.sequence_camera_sensors[ds.sequence_id][ref]
    cam_ts = rs.frames_timestamps_us[:, _nd.FrameTimepoint.END]
    tr = ds.time_range_us
    cta = np.asarray(cam_ts)[np.array([int(t) in tr for t in cam_ts])]
    tracks = load_tracks_from_ncore_cuboids(loader, cta)

    fig, ax = plt.subplots(figsize=(15, 13))
    for tid, t in tracks.items():
        poses = np.asarray(t["poses"])
        info = np.asarray(t["frame_info"])
        size = np.asarray(t["size"])
        l, w = float(size[0]), float(size[1])
        act = [fi for fi in range(len(info)) if bool(info[fi])]
        cx = [poses[fi][0, 3] for fi in act]
        cy = [poses[fi][1, 3] for fi in act]
        ax.plot(cx, cy, "-o", ms=3, label=f"track {tid}: {len(act)} 帧, {l:.1f}×{w:.1f} m")
        for fi in act[::max(1, len(act) // 12)]:
            p = poses[fi]
            ccx, ccy = p[0, 3], p[1, 3]
            yaw = np.arctan2(p[1, 0], p[0, 0])
            corners = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2], [l/2, w/2]])
            c, s = np.cos(yaw), np.sin(yaw)
            pts = corners @ np.array([[c, -s], [s, c]]).T + [ccx, ccy]
            ax.plot(pts[:, 0], pts[:, 1], "r-", lw=1.2)
            ax.plot([ccx, ccx + l/2*c], [ccy, ccy + l/2*s], "r-", lw=2)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_xlabel("x (m, world)")
    ax.set_ylabel("y (m, world)")
    ax.set_title("4cab 大卡车 cuboid trajectory — 蓝线=中心轨迹, 红框=每隔几帧, 红短杆=车头朝向")
    fig.savefig(args.out, dpi=80, bbox_inches="tight")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
