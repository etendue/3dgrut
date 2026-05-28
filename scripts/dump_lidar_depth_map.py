# SPDX-License-Identifier: Apache-2.0
"""Offline LiDAR → image-plane depth-map dump (Stage 11 T11.B1).

Iterates every (clip, camera_id, frame) and writes a sparse [H, W] ray-depth
map under aux/lidar_depth/<camera_id>/<timestamp_us>.npz. Loader counterpart is
threedgrut/datasets/aux_readers.py::LidarDepthAuxReader (Task B2).

Pure NumPy + NCore SDK. The projection core (project_pinhole /
ray_depth_from_cam_pts / scatter_depth_map) is import-safe on Mac with no SDK;
the dump_clip driver imports ncore lazily and is a NotImplementedError stub
until it is filled at A800 run-time.

CLI:
    python scripts/dump_lidar_depth_map.py \
        --manifest /path/to/pai_<clip>.json \
        --camera-ids camera_front_wide_120fov ... \
        --out-root /path/to/clip/aux/lidar_depth \
        --max-depth 80.0
"""
import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def ray_depth_from_cam_pts(cam_pts: np.ndarray) -> np.ndarray:
    """Return ‖cam_pts‖ per row — ray-depth, not z-depth.

    Matches tracer.pred_dist semantics (distance along ray from camera origin
    to surface), so the depth loss compares like-for-like.
    """
    return np.linalg.norm(cam_pts, axis=-1)


def project_pinhole(
    cam_pts: np.ndarray,            # [N, 3] in camera frame (right-down-front)
    intrinsics: dict,               # {fx, fy, cx, cy}
    image_shape: Tuple[int, int],   # (H, W)
) -> Tuple[np.ndarray, np.ndarray]:
    """Project Nx3 camera-frame points to image UV. Returns (uv[N,2], valid[N]).

    Valid := (z > 0) ∧ (0 <= u < W) ∧ (0 <= v < H).
    """
    H, W = image_shape
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    z = cam_pts[:, 2]
    valid_z = z > 1e-3
    safe_z = np.where(valid_z, z, 1.0)
    u = np.where(valid_z, fx * cam_pts[:, 0] / safe_z + cx, -1.0)
    v = np.where(valid_z, fy * cam_pts[:, 1] / safe_z + cy, -1.0)
    valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    uv = np.stack([u, v], axis=-1)
    return uv, valid_z & valid_uv


def scatter_depth_map(
    uv: np.ndarray,        # [N, 2] float
    ray_depth: np.ndarray, # [N]    float (ray-depth)
    valid: np.ndarray,     # [N]    bool
    H: int,
    W: int,
) -> np.ndarray:
    """Scatter sparse ray-depth points to a dense [H, W] depth map.

    Conflict resolution: if multiple points fall in the same pixel, keep the
    nearest (smallest ray-depth) — closest surface occludes the rest.
    """
    dmap = np.zeros((H, W), dtype=np.float32)
    # Sort descending so the nearest point is written last and wins.
    order = np.argsort(-ray_depth)
    for i in order:
        if not valid[i]:
            continue
        u, v = uv[i]
        ui, vi = int(np.floor(u)), int(np.floor(v))
        if 0 <= ui < W and 0 <= vi < H:
            dmap[vi, ui] = ray_depth[i]
    return dmap


def dump_clip(
    manifest_path: Path,
    camera_ids: list[str],
    out_root: Path,
    max_depth: float = 80.0,
) -> None:
    """Iterate every frame × every camera; write one npz per (camera, frame).

    Imports the NCore SDK lazily so the projection-core unit tests (which run
    on Mac without NCore) can import this module. The body is a stub filled at
    A800 run-time (Task C1) once we confirm SequenceLoaderV4.get_lidar_sensor's
    actual frame-reading API.
    """
    try:
        import ncore.data.v4  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "scripts/dump_lidar_depth_map.py dump_clip needs the ncore SDK. "
            "Run from A800 (conda env 3dgrut) or a Mac venv with ncore installed."
        ) from e

    raise NotImplementedError(
        "dump_clip body is filled at A800 run-time (Stage 11 Task C1) — needs "
        "the live SequenceLoaderV4 per-frame LiDAR point + pose API."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--camera-ids", nargs="+", required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--max-depth", type=float, default=80.0)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    dump_clip(args.manifest, args.camera_ids, args.out_root, args.max_depth)
