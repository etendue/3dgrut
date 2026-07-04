#!/usr/bin/env python3
"""V3-VIZ.5 — ego trajectory diagnostic for v2 LayeredGaussians checkpoints.

Loads ``ckpt["viz_4d"]["ego"]`` and reports:

  * timestamp monotonicity (sorted? duplicates? gaps?)
  * per-frame ``dxy`` = ||xy[i+1] - xy[i]||  → mean / median / max / outliers
  * per-frame ``dt`` = ts[i+1] - ts[i]       → consistency
  * per-frame ``dxy/dt`` speed (m/s)         → physical sanity check
  * identity / NaN / degenerate ego poses

Writes two diagnostic PNGs:

  ``<out_dir>/ego_traj_dxy.png``   — line plot of dxy and dt vs frame_idx,
                                      outliers highlighted in red.
  ``<out_dir>/ego_traj_xy.png``    — XY top-down trajectory with outlier
                                      segments highlighted.

Possible root causes for the observed "kinks / jumps / discontinuities":
  R1. Timestamps not sorted              → fix: sort ego.poses + timestamps by ts
                                            in threedgrut/viz/metadata.py:_extract_ego.
  R2. viser spline_catmull_rom overshoot → fix: switch to add_line_segments in
                                            viser_gui_4d.py:_add_ego_trajectory.
  R3. Identity / degenerate pose         → fix: raise / skip in _extract_ego.
  R4. Frame is camera c2w, not ego pose  → fix: left-multiply T_vehicle_cam in
                                            _extract_ego (subtract camera offset).

Usage (Mac / CPU; no CUDA):

    python scripts/diagnose_ego_trajectory.py \\
        --gs_object /path/to/ckpt_last.pt \\
        --out_dir /tmp/ego_diag
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _load_ego_block(ckpt_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (poses_c2w[N,4,4], timestamps_us[N], extra dict).

    extra contains primary_camera_id, intrinsics flag, and the full ``ego`` dict.
    """
    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location="cpu")
    viz = ckpt.get("viz_4d")
    if not isinstance(viz, dict):
        raise RuntimeError("ckpt has no 'viz_4d' block — re-inject via threedgrut.viz.inject.")
    ego = viz.get("ego")
    if not isinstance(ego, dict):
        raise RuntimeError("ckpt['viz_4d'] has no 'ego' sub-block.")

    def _np(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    poses = _np(ego.get("poses_c2w"))
    ts = _np(ego.get("frame_timestamps_us"))
    if poses is None or ts is None:
        raise RuntimeError("ckpt['viz_4d']['ego'] missing poses_c2w or frame_timestamps_us")
    poses = poses.astype(np.float32)
    ts = ts.astype(np.int64)
    extra = {
        "primary_camera_id": str(ego.get("primary_camera_id", "primary")),
        "has_ftheta_intrinsics": ego.get("primary_camera_intrinsics_FTheta") is not None,
        "schema_version": int(viz.get("schema_version", 1)),
        "sequence_id": str(viz.get("sequence_id", "unknown")),
    }
    return poses, ts, extra


def _detect_problems(poses: np.ndarray, ts: np.ndarray, outlier_k: float) -> dict:
    """Return dict of detected problems + flagged frame indices."""
    N = int(poses.shape[0])
    xy = poses[:, :2, 3]

    # Monotonicity.
    ts_diff = np.diff(ts)
    n_neg_dt = int((ts_diff < 0).sum())
    n_zero_dt = int((ts_diff == 0).sum())
    is_sorted = bool((ts_diff >= 0).all())

    # dxy / dt / speed.
    if N >= 2:
        dxy = np.linalg.norm(np.diff(xy, axis=0), axis=1)  # (N-1,)
        # speed only where dt > 0 and dt is sane (avoid div-by-zero on swapped frames)
        safe = ts_diff > 0
        speed = np.full(dxy.shape, np.nan, dtype=np.float32)
        speed[safe] = dxy[safe] / (ts_diff[safe].astype(np.float64) * 1e-6)
    else:
        dxy = np.zeros((0,), dtype=np.float32)
        speed = np.zeros((0,), dtype=np.float32)

    # Outlier in geometric *direction discontinuity*, not in speed.
    # The user-observed "kinks" are direction reversals or sharp turns between
    # consecutive segments — not velocity variation. We measure the angle
    # between consecutive XY segment directions; smooth trajectories keep this
    # near 0, kinks produce large angles. Outlier = angle > 60 degrees on a
    # segment whose dxy is non-trivial (> 0.05 m, i.e. ignore stationary jitter).
    if dxy.size > 0:
        med_dxy = float(np.median(dxy))
        mean_dxy = float(np.mean(dxy))
        threshold = max(med_dxy * outlier_k, mean_dxy * 2.0, 1e-6)
    else:
        med_dxy = mean_dxy = threshold = 0.0

    finite_speed = speed[np.isfinite(speed)] if speed.size > 0 else np.empty(0)
    med_speed = float(np.median(finite_speed)) if finite_speed.size > 0 else 0.0

    direction_outlier_idx: list[int] = []
    max_kink_deg = 0.0
    if xy.shape[0] >= 3:
        seg = np.diff(xy, axis=0)  # (N-1, 2)
        seg_len = np.linalg.norm(seg, axis=1)
        # Skip near-zero segments (stationary) when computing angles.
        moving = seg_len > 0.05
        prev_seg = seg[:-1]
        next_seg = seg[1:]
        prev_len = seg_len[:-1]
        next_len = seg_len[1:]
        valid_pair = moving[:-1] & moving[1:]
        cos_angle = np.full(prev_seg.shape[0], 1.0, dtype=np.float64)
        np.divide(
            np.einsum("ij,ij->i", prev_seg, next_seg),
            np.maximum(prev_len * next_len, 1e-9),
            out=cos_angle,
            where=valid_pair,
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        # Outlier at "kink between segments i and i+1" → flagged at edge index i+1.
        kink_threshold_deg = 60.0
        for ki in np.where(valid_pair & (angle_deg > kink_threshold_deg))[0]:
            direction_outlier_idx.append(int(ki + 1))
        max_kink_deg = float(angle_deg[valid_pair].max()) if int(valid_pair.sum()) > 0 else 0.0

    outlier_idx = direction_outlier_idx
    speed_threshold = 0.0  # legacy field, no longer used for outlier flagging

    # Identity / NaN / degenerate.
    eye = np.eye(4, dtype=np.float32)
    identity_idx = []
    nan_idx = []
    nondet_idx = []
    for i in range(N):
        p = poses[i]
        if np.isnan(p).any():
            nan_idx.append(i)
            continue
        if np.allclose(p, eye, atol=1e-6):
            identity_idx.append(i)
            continue
        det = float(np.linalg.det(p[:3, :3]))
        if abs(det - 1.0) > 0.05:
            nondet_idx.append(i)

    # dt cadence pattern: detect rolling-shutter / dropped-frame double-cadence
    # (e.g. 33ms most edges + 66ms occasional). Flags "data sparse but smooth".
    # Uses max/min ratio rather than IQR so minority dt-mode (< 25% of edges)
    # still triggers detection.
    dt_bimodal = False
    if ts_diff.size > 0:
        dt_ms = ts_diff.astype(np.float64) * 1e-3
        valid = dt_ms > 0
        if int(valid.sum()) >= 4:
            d = dt_ms[valid]
            dt_min = float(d.min())
            dt_max = float(d.max())
            if dt_min > 0 and dt_max > dt_min * 1.5:
                # At least 2 distinct samples in both modes (avoid single-outlier).
                small_mode_count = int((d < dt_min * 1.3).sum())
                large_mode_count = int((d > dt_max * 0.77).sum())
                if small_mode_count >= 2 and large_mode_count >= 2:
                    dt_bimodal = True

    return {
        "n_frames": N,
        "is_sorted": is_sorted,
        "n_negative_dt": n_neg_dt,
        "n_zero_dt": n_zero_dt,
        "dxy_mean_m": mean_dxy,
        "dxy_median_m": med_dxy,
        "dxy_max_m": float(dxy.max()) if dxy.size else 0.0,
        "dxy_threshold_m": float(threshold),
        "outlier_jump_indices": outlier_idx,
        "speed_max_mps": float(np.nanmax(speed)) if speed.size else 0.0,
        "speed_mean_mps": float(np.nanmean(speed)) if speed.size else 0.0,
        "speed_median_mps": med_speed,
        "speed_threshold_mps": speed_threshold,
        "max_direction_kink_deg": max_kink_deg,
        "dt_bimodal": dt_bimodal,
        "identity_pose_indices": identity_idx,
        "nan_pose_indices": nan_idx,
        "nondet_rotation_indices": nondet_idx,
        "dxy": dxy,
        "dt_us": ts_diff,
        "speed_mps": speed,
        "xy": xy,
    }


def _render_dxy_plot(report: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dxy = report["dxy"]
    dt_us = report["dt_us"]
    outliers = set(report["outlier_jump_indices"])
    n = dxy.shape[0]
    x = np.arange(n)

    fig, (ax_dxy, ax_dt) = plt.subplots(2, 1, figsize=(12, 6), dpi=100, sharex=True)

    colors = ["#1A99FF" if i not in outliers else "#E60000" for i in range(n)]
    ax_dxy.bar(x, dxy, width=1.0, color=colors, edgecolor="none")
    ax_dxy.axhline(
        report["dxy_median_m"],
        color="#00B050",
        linewidth=1.2,
        linestyle="--",
        label=f"median = {report['dxy_median_m']:.3f} m",
    )
    ax_dxy.axhline(
        report["dxy_threshold_m"],
        color="#E60000",
        linewidth=1.2,
        linestyle="--",
        label=f"outlier ≥ {report['dxy_threshold_m']:.3f} m",
    )
    ax_dxy.set_ylabel("|Δxy| (m)")
    ax_dxy.set_title(
        f"ego per-frame XY displacement  —  {n} edges  "
        f"mean={report['dxy_mean_m']:.3f}m  max={report['dxy_max_m']:.3f}m  "
        f"outliers={len(outliers)}",
        fontsize=10,
    )
    ax_dxy.legend(fontsize=8)
    ax_dxy.grid(True, color="#E0E0E0", linewidth=0.5)

    ax_dt.bar(x, dt_us / 1000.0, width=1.0, color="#888888", edgecolor="none")
    ax_dt.set_ylabel("Δt (ms)")
    ax_dt.set_xlabel("frame index (between i and i+1)")
    ax_dt.set_title(
        f"timestamp interval  —  monotonic={report['is_sorted']}  "
        f"neg_dt={report['n_negative_dt']}  zero_dt={report['n_zero_dt']}",
        fontsize=10,
    )
    ax_dt.grid(True, color="#E0E0E0", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


def _render_xy_plot(report: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xy = report["xy"]
    outliers = report["outlier_jump_indices"]
    identity = report["identity_pose_indices"]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.plot(
        xy[:, 0], xy[:, 1], color="#1A99FF", linewidth=1.0, alpha=0.7, label=f"ego trajectory ({xy.shape[0]} frames)"
    )
    ax.scatter(xy[:, 0], xy[:, 1], s=3, c="#1A99FF", alpha=0.6, zorder=3)

    if xy.shape[0] > 0:
        ax.scatter(
            [xy[0, 0]],
            [xy[0, 1]],
            s=80,
            c="#00B050",
            marker="o",
            edgecolors="black",
            linewidths=1.5,
            label="start",
            zorder=5,
        )
        ax.scatter(
            [xy[-1, 0]],
            [xy[-1, 1]],
            s=80,
            c="#E60000",
            marker="s",
            edgecolors="black",
            linewidths=1.5,
            label="end",
            zorder=5,
        )

    for i in outliers:
        a = xy[i]
        b = xy[i + 1]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#E60000", linewidth=2.5, alpha=0.85, zorder=4)
        ax.scatter([a[0], b[0]], [a[1], b[1]], s=30, c="#E60000", edgecolors="black", linewidths=0.8, zorder=5)

    if identity:
        idx = np.asarray(identity, dtype=np.int64)
        ax.scatter(
            xy[idx, 0],
            xy[idx, 1],
            s=60,
            c="#FFCC00",
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
            label=f"identity pose ({len(identity)})",
        )

    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    ax.set_aspect("equal")
    ax.set_title("ego XY trajectory (red segments = dxy outliers)", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, color="#E0E0E0", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


def _print_summary(report: dict, extra: dict) -> None:
    print("=" * 72)
    print(f"  ego trajectory diagnostic  —  seq={extra['sequence_id']}  " f"schema_v{extra['schema_version']}")
    print("=" * 72)
    print(f"  primary_camera_id           : {extra['primary_camera_id']}")
    print(f"  has_ftheta_intrinsics       : {extra['has_ftheta_intrinsics']}")
    print(f"  n_frames                    : {report['n_frames']}")
    print(f"  timestamps sorted (asc)     : {report['is_sorted']}")
    print(f"    negative-dt edges         : {report['n_negative_dt']}")
    print(f"    zero-dt edges             : {report['n_zero_dt']}")
    print(
        f"  dxy mean / median / max     : "
        f"{report['dxy_mean_m']:.3f} / {report['dxy_median_m']:.3f} / "
        f"{report['dxy_max_m']:.3f} m"
    )
    print(f"  dt cadence bimodal          : {report.get('dt_bimodal', False)}")
    print(f"  max direction-change kink   : " f"{report.get('max_direction_kink_deg', 0.0):.1f}° " f"(threshold 60°)")
    print(f"  kink-outlier count          : {len(report['outlier_jump_indices'])}")
    if report["outlier_jump_indices"]:
        head = report["outlier_jump_indices"][:10]
        suffix = " ..." if len(report["outlier_jump_indices"]) > 10 else ""
        print(f"    first outliers (i→i+1)    : {head}{suffix}")
    print(f"  speed mean / max            : " f"{report['speed_mean_mps']:.2f} / {report['speed_max_mps']:.2f} m/s")
    print(f"  identity poses count        : {len(report['identity_pose_indices'])}")
    print(f"  NaN poses count             : {len(report['nan_pose_indices'])}")
    print(f"  non-det rotation count      : {len(report['nondet_rotation_indices'])}")
    print("=" * 72)


def _diagnose_hypotheses(report: dict, extra: dict) -> list[str]:
    """Map detected problems → root cause hypothesis lines.

    Data-side problems are flagged as R1/R3/R4 (fix in metadata.py:_extract_ego).
    Rendering-side suspicion (R2 — viser spline) is raised when data passes all
    automated checks (clean data + bimodal dt) — the kinks then originate in
    spline_catmull_rom or FTheta overlay projection at irregular sample spacing.
    """
    hints: list[str] = []
    data_clean = True
    if not report["is_sorted"] or report["n_negative_dt"] > 0:
        hints.append(
            "R1 — timestamps not sorted (negative Δt observed). "
            "Fix in threedgrut/viz/metadata.py:_extract_ego by sorting ego.poses "
            "and timestamps by ts before persisting."
        )
        data_clean = False
    if report["n_zero_dt"] > 0:
        hints.append("R1b — duplicate timestamps (Δt=0). Either dedupe or distinguish by " "frame_idx in _extract_ego.")
        data_clean = False
    if report["identity_pose_indices"]:
        hints.append(
            f"R3 — {len(report['identity_pose_indices'])} identity-matrix poses "
            "detected (likely fallback-on-extract-failure). Fix _extract_ego to "
            "raise or skip those frames."
        )
        data_clean = False
    if report["nan_pose_indices"]:
        hints.append(f"R3b — {len(report['nan_pose_indices'])} NaN poses. Investigate " "upstream NCore extractor.")
        data_clean = False
    if report.get("speed_max_mps", 0.0) > 35.0:
        hints.append(
            f"R2/R4 — peak speed {report['speed_max_mps']:.1f} m/s exceeds 35 m/s "
            "(highway upper bound). Either spline_catmull_rom overshoots in viser "
            "(R2 → switch to add_line_segments) or recorded poses are camera c2w "
            "not vehicle ego (R4 → subtract T_cam2vehicle in _extract_ego)."
        )
        data_clean = False
    if report["outlier_jump_indices"]:
        n_out = len(report["outlier_jump_indices"])
        max_kink = report.get("max_direction_kink_deg", 0.0)
        hints.append(
            f"R5 — {n_out} direction-kink outliers (>60° turn between consecutive "
            f"non-stationary segments; max={max_kink:.1f}°). These are geometric "
            "discontinuities — real teleports, dropped frames stitched without "
            "interpolation, or 7-camera concat leakage past primary-camera slice "
            "in _extract_ego. Inspect outlier indices in JSON output."
        )
        data_clean = False

    if data_clean:
        if report.get("dt_bimodal"):
            hints.append(
                "Data is clean. dt cadence is bimodal (likely 33ms regular + 66ms "
                "dropped-frame gaps) — this inflates per-edge dxy proportionally to dt "
                "but the trajectory itself is geometrically smooth. The viser-observed "
                "'kinks' most likely come from R2 (rendering side):"
            )
            hints.append(
                "  → R2a (pinhole mode): scene.add_spline_catmull_rom overshoots "
                "between irregularly-spaced control points. Fix in "
                "viser_gui_4d.py:_add_ego_trajectory L495 by switching to "
                "scene.add_line_segments(positions[:-1], positions[1:]) — exact "
                "straight chords, no overshoot."
            )
            hints.append(
                "  → R2b (FTheta mode): trajectory is image-space overlay via "
                "Viser4DOverlayCompositor (viser_gui_4d.py:815, subdivide_n=3). "
                "Check FTheta projection of points where the trajectory passes "
                "behind the viewer camera (z_cam ≤ 0) — those points should be "
                "clipped before drawing. Suspect "
                "ftheta_projector.py:project_polylines."
            )
        else:
            hints.append(
                "No obvious red flags from automated checks. If viser still shows "
                "kinks, suspect viser-side rendering (spline_catmull_rom passing "
                "through control points with overshoot) — try add_line_segments."
            )
    return hints


def diagnose(args: argparse.Namespace) -> int:
    ckpt_path = Path(args.gs_object).expanduser().resolve()
    if not ckpt_path.exists():
        print(f"[ego-diag] ERROR: ckpt not found: {ckpt_path}", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ego-diag] loading ckpt → {ckpt_path}", flush=True)
    poses, ts, extra = _load_ego_block(ckpt_path)
    print(f"[ego-diag] ego: N={poses.shape[0]}  primary_cam={extra['primary_camera_id']}", flush=True)

    report = _detect_problems(poses, ts, outlier_k=args.outlier_k)
    _print_summary(report, extra)

    hints = _diagnose_hypotheses(report, extra)
    print("Root-cause hypotheses:")
    for h in hints:
        print(f"  • {h}")
    print("")

    dxy_png = out_dir / "ego_traj_dxy.png"
    xy_png = out_dir / "ego_traj_xy.png"
    _render_dxy_plot(report, dxy_png)
    _render_xy_plot(report, xy_png)
    print(f"[ego-diag] wrote {dxy_png}", flush=True)
    print(f"[ego-diag] wrote {xy_png}", flush=True)

    json_out = {
        **{k: v for k, v in report.items() if k not in {"dxy", "dt_us", "speed_mps", "xy"}},
        "hypotheses": hints,
        "extra": extra,
        "ckpt": str(ckpt_path),
    }
    json_path = out_dir / "ego_traj_diag.json"
    with json_path.open("w") as fh:
        json.dump(json_out, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else o)
    print(f"[ego-diag] wrote {json_path}", flush=True)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gs_object", required=True, type=str, help="Path to v2 ckpt (.pt) with viz_4d.ego block.")
    parser.add_argument("--out_dir", required=True, type=str, help="Output directory for PNGs + ego_traj_diag.json.")
    parser.add_argument("--outlier_k", default=5.0, type=float, help="Outlier multiplier on median dxy (default 5×).")
    args = parser.parse_args(argv)
    return diagnose(args)


if __name__ == "__main__":
    sys.exit(main())
