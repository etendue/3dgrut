#!/usr/bin/env python3
"""目测: auto cuboids(red) vs GT cuboids(green) 的 BEV 叠加图（几帧）。

直接看自动框的摆放/朝向质量（绕过 viser viewer 的 road LayerSpec 兼容 bug）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))


def _boxes(tracks, fi, T=None):
    out = []
    for t in tracks.values():
        info = np.asarray(t["frame_info"])
        if fi >= info.shape[0] or not bool(info[fi]):
            continue
        p = np.asarray(t["poses"])[fi]
        if T is not None:
            p = T @ p
        s = np.asarray(t["size"])
        out.append((float(p[0, 3]), float(p[1, 3]), float(s[0]), float(s[1]),
                    float(np.arctan2(p[1, 0], p[0, 0]))))
    return out


def _rect(ax, b, color, lw=1.5):
    cx, cy, l, w, yaw = b
    corners = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2], [l/2, w/2]])
    c, s = np.cos(yaw), np.sin(yaw)
    pts = corners @ np.array([[c, -s], [s, c]]).T + [cx, cy]
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw)
    ax.plot([cx, cx + l/2*c], [cy, cy + l/2*s], color=color, lw=lw*0.8)  # 朝向箭杆


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", default="/tmp/cuboid_bev.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import hydra
    import ncore.data as _nd
    import ncore.data.v4 as v4

    from threedgrut import datasets
    from threedgrut.datasets.tracks_loader import load_tracks_from_ncore_cuboids

    with hydra.initialize(config_path="../../configs", version_base=None):
        conf = hydra.compose(
            config_name="apps/ncore_3dgut_mcmc_multilayer",
            overrides=[f"path={args.meta}", "trainer.sky_backend=mlp",
                       "dataset.load_aux_masks=true", "+dataset.load_auto_cuboids=true",
                       f"+dataset.auto_cuboids_shard_path={args.shard}"])
    ds, _ = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)
    auto_loader = ds.sequence_loaders[ds.sequence_id]
    gt_loader = v4.SequenceLoaderV4(
        v4.SequenceComponentGroupsReader([args.meta], open_consolidated=True),
        poses_component_group_name="default", intrinsics_component_group_name="default",
        masks_component_group_name="default")

    ref = ds.camera_ids[0]
    rs = ds.sequence_camera_sensors[ds.sequence_id][ref]
    cam_ts = rs.frames_timestamps_us[:, _nd.FrameTimepoint.END]
    tr = ds.time_range_us
    cta = np.asarray(cam_ts)[np.array([int(t) in tr for t in cam_ts])]
    F = len(cta)
    auto = load_tracks_from_ncore_cuboids(auto_loader, cta)
    gt = load_tracks_from_ncore_cuboids(gt_loader, cta)
    Tg = np.asarray(ds.T_world_to_world_global)

    fig, axs = plt.subplots(2, 2, figsize=(18, 18))
    for ax, fi in zip(axs.flat, [F//5, 2*F//5, 3*F//5, 4*F//5]):
        ab = _boxes(auto, fi)
        gb = _boxes(gt, fi, T=Tg)
        for b in gb:
            _rect(ax, b, "green", lw=1.0)
        for b in ab:
            _rect(ax, b, "red", lw=2.0)
        ax.set_title(f"frame {fi}: auto(red)={len(ab)}  GT(green)={len(gb)}")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        if ab:
            xs = [b[0] for b in ab]
            ys = [b[1] for b in ab]
            ax.set_xlim(min(xs) - 20, max(xs) + 20)
            ax.set_ylim(min(ys) - 20, max(ys) + 20)
    fig.suptitle("auto cuboids (red, 粗) vs GT (green, 细) — BEV 摆放/朝向目测 (短杆=车头方向)",
                 fontsize=16)
    fig.savefig(args.out, dpi=70, bbox_inches="tight")
    print(f"saved {args.out}  (auto {len(auto)} tracks / GT {len(gt)} tracks, F={F})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
