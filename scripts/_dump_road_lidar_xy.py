#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A800-only helper: dump semantic-road LiDAR XY to .npy (no training, no GPU).

Mirrors trainer.py:187 dataset construction + trainer.py:395 get_road_lidar_points
so the Phase 2A starvation fact-check (scripts/diagnose_road_starvation.py) can
measure each road particle's distance to the nearest LiDAR road point.

Usage (on A800, conda 3dgrut env):
    python scripts/_dump_road_lidar_xy.py \
        --clip /root/work/yusun/ncore-nurec/data/ncore/clips/<id>/pai_<id>.json \
        --out  /tmp/road_lidar_xy.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config_name", default="apps/ncore_3dgut_mcmc_multilayer")
    args = ap.parse_args()

    import hydra
    from threedgrut import datasets

    with hydra.initialize(config_path="../configs", version_base=None):
        conf = hydra.compose(
            config_name=args.config_name,
            overrides=[f"path={args.clip}", "trainer.sky_backend=mlp"],
        )
    print(f"[dump] dataset.type={conf.dataset.type}  building train_dataset ...", flush=True)
    train_dataset, _ = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)
    pts, _rgb = train_dataset.get_road_lidar_points()  # [M,3] world frame
    xy = pts[:, :2].detach().cpu().numpy().astype(np.float32)
    np.save(args.out, xy)
    print(f"[dump] road LiDAR points: {pts.shape} -> saved XY {xy.shape} to {args.out}", flush=True)
    print(f"[dump] XY bbox: x[{xy[:,0].min():.1f},{xy[:,0].max():.1f}] "
          f"y[{xy[:,1].min():.1f},{xy[:,1].max():.1f}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
