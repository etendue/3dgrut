#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 2A fact-check — WHY does MCMC starve road Gaussians, and where?

Tests the claim "dead road particles concentrate in under-supervised regions,
which include LiDAR coverage gaps" by decomposing it into falsifiable parts:

  T1  death magnitude   — fraction of road particles with opacity <= dead_thresh
  T2  death vs camera   — dead-fraction binned by distance-to-ego-trajectory
                          (ego distance is a proxy for camera supervision:
                          cameras ride the ego and look outward, so far-from-
                          trajectory ground is seen rarely / only at grazing).
  T3  death vs LiDAR    — dead vs alive distance-to-nearest-LiDAR-road-point;
                          tests the specific "LiDAR blind spot" sub-claim.
  T4  disentangle       — 2x2 (near/far ego) x (near/far LiDAR) dead-fraction.
                          If death tracks ego-distance but NOT LiDAR-distance,
                          then "dead == LiDAR gap" is FALSE and the real driver
                          is camera coverage (LiDAR coverage != camera coverage).

Pure CPU. T1/T2 need only the ckpt; T3/T4 need a LiDAR-road-XY .npy
(produced on A800 by scripts/_dump_road_lidar_xy.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _nearest_dist(query_xy: np.ndarray, ref_xy: np.ndarray) -> np.ndarray:
    """Distance from each query point to its nearest ref point (cKDTree)."""
    if query_xy.size == 0 or ref_xy.size == 0:
        return np.full(query_xy.shape[0], np.inf)
    try:
        from scipy.spatial import cKDTree

        d, _ = cKDTree(np.ascontiguousarray(ref_xy)).query(np.ascontiguousarray(query_xy), k=1)
        return np.asarray(d, dtype=np.float64)
    except ImportError:
        out = np.empty(query_xy.shape[0], dtype=np.float64)
        for i in range(0, query_xy.shape[0], 4096):
            c = query_xy[i : i + 4096]
            out[i : i + 4096] = np.sqrt(((c[:, None, :] - ref_xy[None, :, :]) ** 2).sum(-1).min(1))
        return out


def _binned_dead_fraction(dist: np.ndarray, dead: np.ndarray, edges) -> list[dict]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist < hi)
        n = int(m.sum())
        df = float(dead[m].mean()) if n else float("nan")
        rows.append({"lo": float(lo), "hi": (float(hi) if np.isfinite(hi) else None),
                     "n": n, "dead_frac": df})
    return rows


def compute_starvation_stats(
    road_xy: np.ndarray,
    road_opacity: np.ndarray,
    ego_xy: np.ndarray,
    lidar_xy: Optional[np.ndarray] = None,
    *,
    dead_thresh: float = 0.005,
    ego_edges=(0, 3, 6, 10, 15, 20, 30, np.inf),
    lidar_edges=(0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, np.inf),
) -> dict:
    """Decompose road-particle starvation; see module docstring."""
    road_xy = np.asarray(road_xy, dtype=np.float64).reshape(-1, 2)
    road_opacity = np.asarray(road_opacity, dtype=np.float64).reshape(-1)
    ego_xy = np.asarray(ego_xy, dtype=np.float64).reshape(-1, 2)
    if road_xy.shape[0] != road_opacity.shape[0]:
        raise ValueError("road_xy / opacity length mismatch")
    if road_xy.shape[0] == 0 or ego_xy.shape[0] == 0:
        raise ValueError("need >=1 particle and >=1 ego sample")

    dead = road_opacity <= dead_thresh
    out: dict = {
        "dead_thresh": dead_thresh,
        "n_particles": int(road_xy.shape[0]),
        "T1": {
            "dead_frac": float(dead.mean()),
            "n_dead": int(dead.sum()),
            "opacity_pct": {q: float(np.percentile(road_opacity, p))
                            for q, p in (("p10", 10), ("p50", 50), ("p90", 90))},
        },
    }

    ego_dist = _nearest_dist(road_xy, ego_xy)
    out["T2"] = {
        "ego_dist_bins": _binned_dead_fraction(ego_dist, dead, ego_edges),
        "dead_ego_dist_p50": float(np.percentile(ego_dist[dead], 50)) if dead.any() else None,
        "alive_ego_dist_p50": float(np.percentile(ego_dist[~dead], 50)) if (~dead).any() else None,
    }

    if lidar_xy is not None and np.asarray(lidar_xy).size > 0:
        lidar_xy = np.asarray(lidar_xy, dtype=np.float64).reshape(-1, 2)
        lid_dist = _nearest_dist(road_xy, lidar_xy)
        out["T3"] = {
            "lidar_dist_bins": _binned_dead_fraction(lid_dist, dead, lidar_edges),
            "dead_lidar_dist_p50": float(np.percentile(lid_dist[dead], 50)) if dead.any() else None,
            "alive_lidar_dist_p50": float(np.percentile(lid_dist[~dead], 50)) if (~dead).any() else None,
            "dead_lidar_dist_p90": float(np.percentile(lid_dist[dead], 90)) if dead.any() else None,
            "alive_lidar_dist_p90": float(np.percentile(lid_dist[~dead], 90)) if (~dead).any() else None,
        }
        # T4: 2x2 disentangle, median split on each axis
        ego_med = float(np.median(ego_dist))
        lid_med = float(np.median(lid_dist))
        far_ego = ego_dist > ego_med
        far_lid = lid_dist > lid_med
        quad = {}
        for en, em in (("near_ego", ~far_ego), ("far_ego", far_ego)):
            for ln, lm in (("near_lidar", ~far_lid), ("far_lidar", far_lid)):
                m = em & lm
                quad[f"{en}|{ln}"] = {"n": int(m.sum()),
                                      "dead_frac": float(dead[m].mean()) if m.any() else float("nan")}
        out["T4"] = {"ego_median_m": ego_med, "lidar_median_m": lid_med, "quadrants": quad}

    out["_ego_dist"] = ego_dist  # for heatmap (stripped from JSON)
    out["_dead"] = dead
    return out


def _render_dead_map(road_xy, dead, ego_xy, out_path, cell=0.5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(road_xy[~dead, 0], road_xy[~dead, 1], s=0.4, c="#1f77b4", alpha=0.2, label="alive")
    ax.scatter(road_xy[dead, 0], road_xy[dead, 1], s=0.8, c="#d62728", alpha=0.5, label="dead (op<=0.005)")
    ax.plot(ego_xy[:, 0], ego_xy[:, 1], "-", c="lime", lw=1.5, label="ego")
    ax.set_aspect("equal"); ax.set_xlabel("world X (m)"); ax.set_ylabel("world Y (m)")
    ax.set_title("road particles: dead (red) vs alive (blue), ego (green)")
    ax.legend(loc="upper right", fontsize=8, markerscale=8)
    fig.tight_layout(); fig.savefig(str(out_path), dpi=110); plt.close(fig)


def main(argv=None) -> int:
    from threedgrut_playground.utils.diag_ckpt import load_ckpt_cpu
    from scripts.diagnose_road_bev_holes import _extract_layer_xy_opacity, _ego_xy_from_ckpt

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gs_object", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--lidar_road_xy", default=None, help="npy of [M,2] LiDAR road XY (A800-dumped)")
    p.add_argument("--dead_thresh", default=0.005, type=float)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = load_ckpt_cpu(Path(args.gs_object).expanduser().resolve())
    road_xy, road_op = _extract_layer_xy_opacity(ckpt, "road")
    ego_xy = _ego_xy_from_ckpt(ckpt)
    lidar_xy = np.load(args.lidar_road_xy) if args.lidar_road_xy else None

    s = compute_starvation_stats(road_xy, road_op, ego_xy, lidar_xy, dead_thresh=args.dead_thresh)

    print("\n=== Phase 2A road-starvation fact-check ===", flush=True)
    print(f"[T1] road N={s['n_particles']}  dead(op<={args.dead_thresh})={s['T1']['n_dead']} "
          f"({s['T1']['dead_frac']*100:.1f}%)  opacity p10/p50/p90="
          f"{s['T1']['opacity_pct']['p10']:.4f}/{s['T1']['opacity_pct']['p50']:.4f}/{s['T1']['opacity_pct']['p90']:.4f}")
    print(f"\n[T2] dead-fraction by distance-to-ego (camera-supervision proxy):")
    print(f"     {'ego_dist(m)':>14}{'n':>10}{'dead_frac':>11}")
    for r in s["T2"]["ego_dist_bins"]:
        hi = "inf" if r["hi"] is None else f"{r['hi']:.0f}"
        label = f"{r['lo']:.0f}-{hi}"
        print(f"     {label:>14}{r['n']:>10}{r['dead_frac']*100:>10.1f}%")
    print(f"     dead ego-dist p50={s['T2']['dead_ego_dist_p50']:.1f}m  vs  alive p50={s['T2']['alive_ego_dist_p50']:.1f}m")

    if "T3" in s:
        print(f"\n[T3] dead-fraction by distance-to-nearest-LiDAR-road-point:")
        print(f"     {'lidar_dist(m)':>14}{'n':>10}{'dead_frac':>11}")
        for r in s["T3"]["lidar_dist_bins"]:
            hi = "inf" if r["hi"] is None else f"{r['hi']:.2f}"
            label = f"{r['lo']:.2f}-{hi}"
            print(f"     {label:>14}{r['n']:>10}{r['dead_frac']*100:>10.1f}%")
        print(f"     dead lidar-dist p50={s['T3']['dead_lidar_dist_p50']:.2f}m p90={s['T3']['dead_lidar_dist_p90']:.2f}m"
              f"  vs  alive p50={s['T3']['alive_lidar_dist_p50']:.2f}m p90={s['T3']['alive_lidar_dist_p90']:.2f}m")
        print(f"\n[T4] disentangle (median split: ego={s['T4']['ego_median_m']:.1f}m, lidar={s['T4']['lidar_median_m']:.2f}m):")
        print(f"     {'quadrant':>22}{'n':>10}{'dead_frac':>11}")
        for k, v in s["T4"]["quadrants"].items():
            print(f"     {k:>22}{v['n']:>10}{v['dead_frac']*100:>10.1f}%")

    # JSON (strip arrays)
    s_json = {k: v for k, v in s.items() if not k.startswith("_")}
    (out_dir / "starvation_stats.json").write_text(json.dumps(s_json, indent=2))
    _render_dead_map(road_xy, s["_dead"], ego_xy, out_dir / "dead_vs_alive.png")
    print(f"\n[wrote] {out_dir/'starvation_stats.json'} + dead_vs_alive.png", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
