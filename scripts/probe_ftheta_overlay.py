#!/usr/bin/env python3
"""B2 Phase 0 calibration probe — FTheta cuboid forward projection.

Loads a viz_4d ckpt, extracts ego pose + first active cuboid at t=0,
projects the cuboid's 24 edge endpoints through FTheta polynomial under
several candidate (FLIP, poly-order, linear_cde) combinations, and prints
per-combo pixel coordinates + visibility masks.

The "winning" combo is the one whose vertices (a) mostly land within
(0,W) x (0,H) image bounds, (b) are mostly in_fov (angle <= max_angle),
and (c) are mostly z > 0 (in front of camera).

Usage (ThinkPad with conda env 3dgrut2 active):
    python scripts/probe_ftheta_overlay.py \\
        --ckpt /home/yusun/work/ckpts/bug4_v2_full_30k/ckpt_with_ftheta_v2.pt

Dependencies: numpy + torch + (project-local) viz4d_metadata + cuboid.
No CUDA needed; ckpt loads to CPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Make repo root importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from threedgrut_playground.utils.cuboid import cuboid_world_edges
from threedgrut_playground.utils.viz4d_metadata import FourDMetadata


# ----- Candidate combinations ---------------------------------------------
# Standard OpenGL→OpenCV axis flip: c2w_cv = c2w_gl @ diag([1, -1, -1, 1])
# Reason: OpenGL camera convention is +Y up, +Z backward (camera looks at -Z);
# OpenCV is +Y down, +Z forward. The c2w matrix columns store camera basis
# vectors in world frame, so right-multiplying by diag([1,-1,-1,1]) flips
# the Y and Z basis columns to convert between conventions.
FLIP_GL_TO_CV_RIGHT = np.diag([1.0, -1.0, -1.0, 1.0])
FLIP_GL_TO_CV_LEFT  = FLIP_GL_TO_CV_RIGHT  # identity in left-mul slot (would flip world Y/Z, almost certainly wrong)

CANDIDATES = [
    # (name, c2w_transform_callable, poly_eval_callable)
    (
        "A: c2w @ diag([1,-1,-1,1]) + poly ascending (mirror inverse)",
        lambda c2w: c2w @ FLIP_GL_TO_CV_RIGHT,
        lambda poly, theta: _horner_ascending(poly, theta),
    ),
    (
        "B: identity (assume c2w already OpenCV) + poly ascending",
        lambda c2w: c2w.copy(),
        lambda poly, theta: _horner_ascending(poly, theta),
    ),
    (
        "C: c2w @ diag([1,-1,-1,1]) + poly descending (np.polyval default)",
        lambda c2w: c2w @ FLIP_GL_TO_CV_RIGHT,
        lambda poly, theta: np.polyval(poly, theta),
    ),
    (
        "D: c2w @ diag([1,1,-1,1]) (Z-only flip) + poly ascending",
        lambda c2w: c2w @ np.diag([1.0, 1.0, -1.0, 1.0]),
        lambda poly, theta: _horner_ascending(poly, theta),
    ),
]


def _horner_ascending(poly: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate p(x) = poly[0] + poly[1]*x + poly[2]*x^2 + ...

    Mirrors ftheta_intrinsics.py:69-70 convention (asc storage, Horner
    iterating high→low index).
    """
    poly = np.asarray(poly, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)
    for k in range(len(poly) - 1, -1, -1):
        out = out * x + poly[k]
    return out


def _project_one(
    points_world: np.ndarray,        # (N, 3)
    c2w_transformed: np.ndarray,     # (4, 4) — already in OpenCV convention
    ftheta_dict: dict,
    poly_fn,
):
    """Returns (uv: (N,2), visible: (N,) bool, debug: dict)."""
    N = points_world.shape[0]
    w2c = np.linalg.inv(c2w_transformed)
    p_h = np.concatenate([points_world, np.ones((N, 1), dtype=np.float64)], axis=-1)
    p_cam = (w2c @ p_h.T).T[:, :3]

    x, y, z = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]
    r_xy = np.sqrt(x ** 2 + y ** 2)
    angle = np.arctan2(r_xy, z)  # ∈ [0, π]

    poly = np.asarray(ftheta_dict["angle_to_pixeldist_poly"], dtype=np.float64)
    r_pix = poly_fn(poly, angle)

    pp = ftheta_dict["principal_point"]
    cx, cy = float(pp[0]), float(pp[1])
    safe_r = np.where(r_xy < 1e-9, 1.0, r_xy)
    u_off = np.where(r_xy < 1e-9, 0.0, r_pix * x / safe_r)
    v_off = np.where(r_xy < 1e-9, 0.0, r_pix * y / safe_r)
    u = cx + u_off
    v = cy + v_off

    res = ftheta_dict["resolution"]
    W, H = int(res[0]), int(res[1])
    max_angle = float(ftheta_dict["max_angle"])
    in_fov = angle <= max_angle
    in_bound = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    z_pos = z > 0
    visible = in_fov & in_bound & z_pos

    return (
        np.stack([u, v], axis=-1),
        visible,
        {
            "z_min": float(z.min()), "z_max": float(z.max()),
            "angle_min_deg": float(np.degrees(angle.min())),
            "angle_max_deg": float(np.degrees(angle.max())),
            "max_angle_deg": float(np.degrees(max_angle)),
            "n_in_fov": int(in_fov.sum()),
            "n_z_pos": int(z_pos.sum()),
            "n_in_bound": int(in_bound.sum()),
            "n_visible": int(visible.sum()),
            "resolution_WH": [W, H],
        },
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out-json", type=Path, default=None,
                    help="Optional: also dump full results to JSON")
    args = ap.parse_args()

    print(f"[probe] loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    meta = FourDMetadata.from_ckpt(ckpt)
    if meta is None:
        print("[probe] ERROR: ckpt has no viz_4d block")
        return 1
    if not meta.has_ftheta():
        print("[probe] ERROR: ckpt has no FTheta intrinsics (pinhole ckpt)")
        return 1

    ftheta = meta.ego_primary_intrinsics_ftheta
    print(f"[probe] schema_version: {meta.schema_version}")
    print(f"[probe] sequence_id:    {meta.sequence_id}")
    print(f"[probe] ego_poses_c2w:  shape={meta.ego_poses_c2w.shape}")
    print(f"[probe] n_tracks:       {meta.n_tracks()}")
    print(f"[probe] n_frames:       {meta.n_frames()}")
    print(f"[probe] FTheta keys:    {sorted(ftheta.keys())}")
    print(f"[probe] FTheta resolution: {list(ftheta['resolution'])}")
    print(f"[probe] FTheta principal_point: {list(ftheta['principal_point'])}")
    print(f"[probe] FTheta max_angle: {float(ftheta['max_angle']):.4f} rad "
          f"= {np.degrees(float(ftheta['max_angle'])):.1f} deg half-FOV")
    print(f"[probe] FTheta angle_to_pixeldist_poly: "
          f"{np.asarray(ftheta['angle_to_pixeldist_poly'], dtype=np.float64).tolist()}")
    print(f"[probe] FTheta pixeldist_to_angle_poly: "
          f"{np.asarray(ftheta['pixeldist_to_angle_poly'], dtype=np.float64).tolist()}")
    print(f"[probe] FTheta linear_cde: {list(ftheta['linear_cde'])}")

    # Find first active track at frame_idx = 0
    active = meta.active_tracks_at(0)
    if not active:
        print("[probe] ERROR: no active tracks at frame_idx=0")
        return 1
    tid = active[0]
    tdata = meta.tracks[tid]
    pose = tdata["poses"][0]      # (4, 4)
    size = tdata["size"]          # (3,)
    klass = tdata["class"]
    print(f"\n[probe] picked cuboid: tid={tid}  class={klass}")
    print(f"[probe] cuboid pose t = {pose[:3, 3].tolist()}")
    print(f"[probe] cuboid size  = {size.tolist()}")

    edges = cuboid_world_edges(pose, size)  # (12, 2, 3)
    points_world = edges.reshape(-1, 3).astype(np.float64)  # (24, 3)
    print(f"[probe] cuboid 24 edge endpoints (world):")
    for i, p in enumerate(points_world):
        print(f"         v{i:02d}: ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")

    # Ego pose at frame 0 (use ego_poses_c2w[0])
    c2w_gl = meta.ego_poses_c2w[0].astype(np.float64)
    print(f"\n[probe] ego pose[0] (viser/OpenGL convention) translation = "
          f"{c2w_gl[:3, 3].tolist()}")

    results = {}
    for name, c2w_fn, poly_fn in CANDIDATES:
        c2w_cv = c2w_fn(c2w_gl)
        uv, visible, dbg = _project_one(points_world, c2w_cv, ftheta, poly_fn)
        print(f"\n=== Candidate: {name} ===")
        print(f"  cam-frame z range: [{dbg['z_min']:+.2f}, {dbg['z_max']:+.2f}]"
              f"  (n_z_pos={dbg['n_z_pos']}/24)")
        print(f"  ray angle:         [{dbg['angle_min_deg']:.1f}°, "
              f"{dbg['angle_max_deg']:.1f}°]   max_angle={dbg['max_angle_deg']:.1f}°"
              f"   (n_in_fov={dbg['n_in_fov']}/24)")
        print(f"  pixels in image:   n_in_bound={dbg['n_in_bound']}/24"
              f"   (W,H={dbg['resolution_WH']})")
        print(f"  total visible:     {dbg['n_visible']}/24")
        # Identify bottom vs top vertices by world z to confirm Y-axis orientation.
        # In OpenCV image convention: +V_pixel points down → bottom vertices
        # (low world z, assuming roughly level ego) should have LARGER v.
        bottom_idx = points_world[:, 2] < points_world[:, 2].mean()
        v_bottom = uv[bottom_idx, 1]
        v_top    = uv[~bottom_idx, 1]
        print(f"  bottom verts (low world-z) v range: "
              f"[{v_bottom.min():.1f}, {v_bottom.max():.1f}] mean={v_bottom.mean():.1f}")
        print(f"  top    verts (high world-z) v range: "
              f"[{v_top.min():.1f}, {v_top.max():.1f}] mean={v_top.mean():.1f}")
        print(f"  Δv (bottom - top) mean: {v_bottom.mean() - v_top.mean():+.1f}  "
              f"(positive → OpenCV convention: bottom below top)")
        print(f"  visibility mask:   {visible.astype(int).tolist()}")
        print(f"  full (u, v) per vertex:")
        for i in range(uv.shape[0]):
            tag = "bot" if bottom_idx[i] else "top"
            print(f"    v{i:02d} [{tag}]: u={uv[i,0]:7.1f}  v={uv[i,1]:7.1f}")
        results[name] = {
            "uv": uv.tolist(),
            "visible": visible.astype(int).tolist(),
            "v_bottom_mean": float(v_bottom.mean()),
            "v_top_mean":    float(v_top.mean()),
            "delta_v_bottom_minus_top": float(v_bottom.mean() - v_top.mean()),
            **dbg,
        }

    # Verdict
    print("\n=== Verdict (winner = highest n_visible) ===")
    ranked = sorted(results.items(), key=lambda kv: kv[1]["n_visible"], reverse=True)
    for name, r in ranked:
        print(f"  n_visible={r['n_visible']:2d}/24  {name}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)

        def _jsonable(o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(f"unhashable {type(o)}")

        with open(args.out_json, "w") as f:
            json.dump({
                "ckpt": str(args.ckpt),
                "ftheta_resolution": [int(x) for x in ftheta["resolution"]],
                "ftheta_principal_point": [float(x) for x in ftheta["principal_point"]],
                "ftheta_max_angle_rad": float(ftheta["max_angle"]),
                "ftheta_angle_to_pixeldist_poly": np.asarray(
                    ftheta["angle_to_pixeldist_poly"], dtype=np.float64).tolist(),
                "cuboid_tid": tid,
                "cuboid_class": klass,
                "cuboid_pose_translation": [float(x) for x in pose[:3, 3].tolist()],
                "cuboid_size": [float(x) for x in size.tolist()],
                "ego_pose_translation": [float(x) for x in c2w_gl[:3, 3].tolist()],
                "candidates": results,
            }, f, indent=2, default=_jsonable)
        print(f"\n[probe] wrote {args.out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
