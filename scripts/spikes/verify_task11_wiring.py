#!/usr/bin/env python3
"""Task 11 接线端到端验证（不训练）：dataset(load_auto_cuboids=true) →
loader.get_cuboid_track_observations → load_tracks_from_ncore_cuboids(cam_ts 50ms 匹配)
→ init_dynamic_rigid_layer → 粒子数 > 0。

复用 trainer.py 的 cam_ts_active 构造逻辑，验证 Branch A 接线 + 两个静默失败防线
（class 字符串匹配 / cam_ts 50ms 窗）。
"""
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
    args = ap.parse_args()

    import hydra
    import ncore.data as _nd

    from threedgrut import datasets
    from threedgrut.datasets.tracks_loader import load_tracks_from_ncore_cuboids
    from threedgrut.layers.dynamic_rigid_init import init_dynamic_rigid_layer

    with hydra.initialize(config_path="../../configs", version_base=None):
        conf = hydra.compose(
            config_name="apps/ncore_3dgut_mcmc_multilayer",
            overrides=[
                f"path={args.meta}",
                "trainer.sky_backend=mlp",
                "dataset.load_aux_masks=true",
                "+dataset.load_auto_cuboids=true",
                f"+dataset.auto_cuboids_shard_path={args.shard}",
            ],
        )
    ds, _ = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)
    loader = ds.sequence_loaders[ds.sequence_id]

    n_auto = len(list(loader.get_cuboid_track_observations()))
    print(f"[verify] loader.get_cuboid_track_observations → {n_auto} obs (auto_v0 group)")
    assert n_auto > 0, "FAIL: loader 没读到 auto cuboids（Branch A append / instance 名错）"

    ref_cam = ds.camera_ids[0]
    ref_sensor = ds.sequence_camera_sensors[ds.sequence_id][ref_cam]
    cam_ts = ref_sensor.frames_timestamps_us[:, _nd.FrameTimepoint.END]
    tr = ds.time_range_us
    cam_ts_active = np.asarray(cam_ts)[np.array([int(t) in tr for t in cam_ts])]
    tracks = load_tracks_from_ncore_cuboids(loader, cam_ts_active)
    print(f"[verify] load_tracks_from_ncore_cuboids → {len(tracks)} tracks "
          f"over {cam_ts_active.shape[0]} cam frames")
    assert tracks, "FAIL: 0 tracks（cam_ts 50ms 匹配失败 / class 字符串滤空）"

    dyn_pts, _ = ds.get_dynamic_lidar_points()
    pos, tids, names = init_dynamic_rigid_layer(tracks, dyn_pts)
    print(f"[verify] init_dynamic_rigid_layer → {pos.shape[0]} particles, {len(names)} tracks")
    assert pos.shape[0] > 0, "FAIL: 0 particles（cuboid 中心/尺寸不含动态点）"

    print("[verify] PASS: Branch A 端到端接线通 + dynamic_rigids 粒子 > 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
