# SPDX-License-Identifier: Apache-2.0
"""scene_manifest tracks → instance_pts_dict loader (T4.1.b).

T8/B3 Phase E fix (2026-05-25): ``load_tracks_from_ncore_cuboids`` previously
stored each cuboid pose as **translation-only identity rotation**, ignoring
``bbox3.rot``. That broke dynamic_rigids end-to-end (init filter, _transform_means
local→world, and project_cuboids_to_mask all assumed object-local frame was
axis-aligned with world). For a yaw=π/2 vehicle, the AABB-style filter
missed most LiDAR points and the rendered Gaussian "footprint" was 90° off.
We now decode ``bbox3.rot`` as intrinsic XYZ Euler radians (probe-confirmed
on NCore v4 manifests: rz spans ±π for vehicle yaw, rx/ry ≈ 0 on flat ground)
and write the full SE(3) into ``poses[fi]``.


Lives in its own module (not in datasetNcore.py) so unit tests can import it
without triggering the NCore SDK / cv2 / kornia chain that datasets/__init__.py
pulls in on Mac.

datasetNcore.py re-exports this at module-level (T4.5) so trainer.init_model
can call `from threedgrut.datasets.datasetNcore import load_tracks_from_manifest`
in line with v2_plan.md's path table.

Output schema mirrors drivestudio's get_init_objects (driving_dataset.py:263-396)
but is rebuilt from scratch — no drivestudio dep, no OmniRe pixel_source coupling:

    {track_id: {
        "pts":        None,                 # filled by T4.2.b dynamic_rigid_init
        "colors":     None,                 # T4.2.b
        "poses":      Tensor[F, 4, 4],      # object → world SE(3) per frame
        "size":       Tensor[3],            # cuboid full extent (not half)
        "frame_info": BoolTensor[F],        # active flag per frame
        "class":      str,                  # "vehicle", "pedestrian", etc.
    }}

NCore manifest shape (T3a.2 verified empty for current clip — tracks field
needs separate generation). When tracks field is missing → returns empty dict
(not a crash; trainer.init_model logs and skips dynamic_rigids layer).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


def euler_xyz_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Extrinsic xyz Euler angles (radians) → 3x3 rotation matrix.

    Matches ``scipy.spatial.transform.Rotation.from_euler("xyz", [rx, ry, rz]).as_matrix()``
    bit-for-bit (within fp tolerance) — scipy's *lowercase* "xyz" is extrinsic,
    i.e. rotate about world X by rx, then world Y by ry, then world Z by rz.
    Pure numpy so Mac venv (no scipy) can still run unit tests.

    Convention (extrinsic xyz):
        R = Rz(rz) · Ry(ry) · Rx(rx)
    applied to a column vector ``v`` as ``R @ v``. For pure-yaw vehicles
    (rx ≈ ry ≈ 0) this collapses to Rz(rz), which is what most NCore
    automobile observations actually exercise.

    Args:
        rx, ry, rz: rotation about the X, Y, Z axes (radians).

    Returns:
        ``[3, 3]`` float64 rotation matrix.
    """
    cx, sx = float(np.cos(rx)), float(np.sin(rx))
    cy, sy = float(np.cos(ry)), float(np.sin(ry))
    cz, sz = float(np.cos(rz)), float(np.sin(rz))
    return np.array(
        [
            [cy * cz,  cz * sy * sx - sz * cx,  cz * sy * cx + sz * sx],
            [sz * cy,  sz * sy * sx + cz * cx,  sz * sy * cx - cz * sx],
            [-sy,      cy * sx,                 cy * cx],
        ],
        dtype=np.float64,
    )


def load_tracks_from_manifest(manifest_path: Union[str, Path]) -> Dict[str, dict]:
    """Parse scene_manifest.tracks → instance_pts_dict.

    Args:
        manifest_path: path to ``pai_<clip>.json`` (or any JSON with a
            top-level ``"tracks"`` array of track dicts).

    Returns:
        Dict keyed by track id; empty if the manifest has no ``tracks`` key
        or the array is empty.

    Raises:
        FileNotFoundError: manifest_path does not exist.
        json.JSONDecodeError: manifest is not valid JSON.
        ValueError: a track dict is missing required fields
            (``id`` / ``poses`` / ``extent`` / ``active_frames``).
    """
    path = Path(manifest_path)
    m = json.loads(path.read_text())

    raw_tracks = m.get("tracks", [])
    out: Dict[str, dict] = {}
    for trk in raw_tracks:
        tid = trk.get("id")
        if tid is None:
            raise ValueError(f"track missing 'id' field: keys={list(trk.keys())}")
        for required in ("poses", "extent", "active_frames"):
            if required not in trk:
                raise ValueError(
                    f"track '{tid}' missing required field '{required}'; "
                    f"keys={list(trk.keys())}"
                )
        poses = torch.tensor(trk["poses"], dtype=torch.float32)
        if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
            raise ValueError(
                f"track '{tid}' poses shape invalid: {tuple(poses.shape)}, "
                f"expected [F, 4, 4]"
            )
        size = torch.tensor(trk["extent"], dtype=torch.float32)
        if size.shape != (3,):
            raise ValueError(
                f"track '{tid}' extent shape invalid: {tuple(size.shape)}, "
                f"expected [3]"
            )
        frame_info = torch.tensor(trk["active_frames"], dtype=torch.bool)
        if frame_info.shape[0] != poses.shape[0]:
            raise ValueError(
                f"track '{tid}' active_frames len {frame_info.shape[0]} "
                f"!= poses F {poses.shape[0]}"
            )
        out[str(tid)] = {
            "pts": None,
            "colors": None,
            "poses": poses,
            "size": size,
            "frame_info": frame_info,
            "class": str(trk.get("class", "vehicle")),
        }
    return out


# Default classes to retain when building tracks from NCore cuboids.
# v2 Stage 4 focuses on vehicle / large-rigid actors only (matches
# dynamic_rigids layer scope). Pedestrians and animals are higher-order
# rigid-deformable and handled in dynamic_deformables (v3).
DEFAULT_VEHICLE_CLASSES: frozenset[str] = frozenset({
    "automobile", "heavy_truck", "bus",
})


def load_tracks_from_ncore_cuboids(
    loader,
    camera_frame_timestamps_us: np.ndarray,
    *,
    class_filter: frozenset[str] = DEFAULT_VEHICLE_CLASSES,
    time_tolerance_us: int = 50_000,  # half typical 30fps frame interval (33ms)
) -> Dict[str, dict]:
    """T4.5: build instance_pts_dict from NCore manifest cuboid autolabels.

    Replaces the mock ``load_tracks_from_manifest(json_path)`` path with the
    real cuboid_track_observations component of the NCore manifest (autolabels
    v2 by default; verified A800 2026-05-19: clip 9ae151dc has 179 unique
    tracks across 13657 observations).

    Per-track pipeline:
      1. groupby track_id
      2. filter by class_filter (default: vehicle classes only — matches
         dynamic_rigids layer scope)
      3. for each NCore camera frame timestamp, find the nearest cuboid obs
         within ``time_tolerance_us``; transform that obs to world frame
         via ``obs.transform("world", ts, pose_graph)``
      4. construct pose = translate-only (identity rot, world centroid)
         + extent (size) + per-frame active flag

    Args:
        loader: NCore SequenceLoaderV4 instance (provides
            ``get_cuboid_track_observations()`` + ``pose_graph``).
        camera_frame_timestamps_us: ``[F]`` per-frame camera END timestamps
            (use sensor.frames_timestamps_us[:, FrameTimepoint.END] — matches
            sseg / lidar-sseg key convention).
        class_filter: which cuboid class_ids to keep. Default = vehicle classes.
        time_tolerance_us: max |ts_cuboid - ts_frame| to consider a match.
            50ms ≈ 1.5 × typical 30fps frame interval.

    Returns:
        ``{track_id: {pts:None, colors:None, poses[F,4,4], size[3],
                       frame_info[F bool], class:str}}``
        — same schema as ``load_tracks_from_manifest`` so downstream
        ``init_dynamic_rigid_layer`` + ``LayeredGaussians(tracks=...)``
        consume it unchanged. Tracks with no obs within tolerance for ANY
        camera frame are dropped.

    Raises:
        AttributeError: loader missing get_cuboid_track_observations or
            pose_graph. Manifest must be a NCore V4 sequence with cuboid
            autolabels (most production clips have these).
    """
    pose_graph = loader.pose_graph
    F = camera_frame_timestamps_us.shape[0]

    # 1. groupby track_id
    by_track: dict = defaultdict(list)
    for obs in loader.get_cuboid_track_observations():
        if obs.class_id in class_filter:
            by_track[obs.track_id].append(obs)

    out: Dict[str, dict] = {}
    for tid, obs_list in by_track.items():
        # Sort obs by timestamp for nearest lookup
        obs_list.sort(key=lambda o: o.timestamp_us)
        obs_ts = np.asarray([o.timestamp_us for o in obs_list], dtype=np.int64)

        poses_np = np.zeros((F, 4, 4), dtype=np.float32)
        poses_np[:] = np.eye(4)
        frame_info_np = np.zeros(F, dtype=bool)
        size_np: Optional[np.ndarray] = None
        class_id: str = obs_list[0].class_id

        for fi, ts in enumerate(camera_frame_timestamps_us):
            ts_int = int(ts)
            # nearest obs by abs(ts_cuboid - ts_frame)
            idx = int(np.argmin(np.abs(obs_ts - ts_int)))
            if abs(int(obs_ts[idx]) - ts_int) > time_tolerance_us:
                continue
            obs = obs_list[idx]
            # transform obs.bbox3 (rig frame) → world frame
            try:
                world_obs = obs.transform("world", ts_int, pose_graph)
                bbox = world_obs.bbox3
            except Exception:
                # transform may fail at clip boundaries (pose_graph
                # extrapolation gap); skip this frame.
                continue
            cx, cy, cz = bbox.centroid
            # T8/B3 Phase E: decode bbox.rot as intrinsic XYZ Euler radians and
            # populate the rotation block of the SE(3) pose. ``bbox.rot`` after
            # ``obs.transform("world", ...)`` is the cuboid local frame's
            # orientation in world frame, which is exactly what _transform_means
            # consumes (``world = R @ local + t``).
            rx, ry, rz = bbox.rot
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = euler_xyz_to_rotation_matrix(rx, ry, rz).astype(np.float32)
            pose[:3, 3] = (cx, cy, cz)
            poses_np[fi] = pose
            frame_info_np[fi] = True
            if size_np is None:
                size_np = np.asarray(bbox.dim, dtype=np.float32)

        if not frame_info_np.any():
            # Track has no observations within tolerance of any camera frame
            # (entirely outside our time window) — drop it.
            continue
        if size_np is None:
            size_np = np.asarray(obs_list[0].bbox3.dim, dtype=np.float32)

        out[str(tid)] = {
            "pts": None,
            "colors": None,
            "poses": torch.from_numpy(poses_np),
            "size": torch.from_numpy(size_np),
            "frame_info": torch.from_numpy(frame_info_np),
            "class": class_id,
            # T4.5: per-pose absolute camera END timestamp (shared across all
            # tracks in this call). Consumers (LayeredGaussians.populate_tracks)
            # store ONE shared timestamp buffer rather than per-track copies.
            "cam_timestamps_us": torch.from_numpy(
                np.asarray(camera_frame_timestamps_us, dtype=np.int64)
            ),
        }
    return out
