# SPDX-License-Identifier: Apache-2.0
"""Pure-CPU dataclass + helpers for parsing ``ckpt['viz_4d']`` blocks (Stage 8).

This module intentionally avoids viser / kaolin / engine imports so it can be
unit-tested on a Mac without GUI dependencies. ``viser_gui_4d.py`` imports
``FourDMetadata`` from here and wires it into the viser scene graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch


FTHETA_REQUIRED_KEYS = frozenset(
    {
        "resolution",
        "shutter_type",
        "principal_point",
        "reference_poly",
        "pixeldist_to_angle_poly",
        "angle_to_pixeldist_poly",
        "max_angle",
        "linear_cde",
    }
)


def _to_np(t: Any) -> Optional[np.ndarray]:
    """torch.Tensor / np.ndarray / list → np.ndarray; None passes through."""
    if t is None:
        return None
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    if isinstance(t, np.ndarray):
        return t
    return np.asarray(t)


def _normalize_resolution(value: Any, *, context: str) -> tuple[int, int]:
    raw = np.asarray(value, dtype=np.int64).reshape(-1)
    if raw.size != 2:
        raise ValueError(f"{context} resolution must contain W,H, got {raw.shape}")
    resolution = (int(raw[0]), int(raw[1]))
    if resolution[0] <= 0 or resolution[1] <= 0:
        raise ValueError(f"{context} resolution must be positive, got {resolution}")
    return resolution


def _normalize_ftheta_dict(value: Any, *, camera_id: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(
            f"FTheta camera '{camera_id}' intrinsics_FTheta must be a dictionary"
        )
    missing = FTHETA_REQUIRED_KEYS.difference(value)
    if missing:
        raise ValueError(
            f"FTheta camera '{camera_id}' missing required keys: {sorted(missing)}"
        )
    out = dict(value)
    out["resolution"] = np.asarray(value["resolution"], dtype=np.int64)
    for key in (
        "principal_point",
        "pixeldist_to_angle_poly",
        "angle_to_pixeldist_poly",
        "linear_cde",
    ):
        out[key] = np.asarray(value[key], dtype=np.float32)
    out["max_angle"] = float(value["max_angle"])
    return out


def _normalize_camera_models(value: Any, *, schema_version: int) -> dict[str, dict]:
    """Parse schema-v3 per-camera contracts and reject ambiguous projection state."""
    if value is None:
        if schema_version >= 3:
            raise ValueError("schema_v3 checkpoint must contain camera_models")
        return {}
    if not isinstance(value, dict):
        raise ValueError("viz_4d.camera_models must be a dictionary")

    out: dict[str, dict] = {}
    for raw_camera_id, raw_contract in value.items():
        camera_id = str(raw_camera_id)
        if not isinstance(raw_contract, dict):
            raise ValueError(f"camera '{camera_id}' contract must be a dictionary")
        model_type = str(raw_contract.get("model_type", ""))
        if model_type not in {"FTheta", "OpenCVPinhole", "IdealPinhole"}:
            raise ValueError(
                f"camera '{camera_id}' has unsupported model_type={model_type!r}"
            )
        if "native_resolution" not in raw_contract:
            raise ValueError(f"camera '{camera_id}' is missing native_resolution")
        resolution = _normalize_resolution(
            raw_contract["native_resolution"], context=f"camera '{camera_id}'"
        )
        ftheta = raw_contract.get("intrinsics_FTheta")
        if model_type == "FTheta":
            ftheta = _normalize_ftheta_dict(ftheta, camera_id=camera_id)
            ftheta_resolution = _normalize_resolution(
                ftheta["resolution"], context=f"camera '{camera_id}' FTheta"
            )
            if ftheta_resolution != resolution:
                raise ValueError(
                    f"camera '{camera_id}' resolution mismatch: "
                    f"native={resolution}, FTheta={ftheta_resolution}"
                )
        elif ftheta is not None:
            raise ValueError(
                f"camera '{camera_id}' carries FTheta intrinsics but "
                f"model_type={model_type}"
            )

        contract = {
            "model_type": model_type,
            "native_resolution": resolution,
        }
        if ftheta is not None:
            contract["intrinsics_FTheta"] = ftheta
        if raw_contract.get("parameter_fingerprint") is not None:
            contract["parameter_fingerprint"] = str(
                raw_contract["parameter_fingerprint"]
            )
        out[camera_id] = contract

    if schema_version >= 3 and not out:
        raise ValueError("schema_v3 checkpoint must contain camera_models")
    return out


@dataclass
class FourDMetadata:
    """In-memory view of ``ckpt['viz_4d']`` (schema versions 1-3).

    All tensors are kept as ``np.ndarray`` (CPU) so render-loop math doesn't
    bounce between torch and numpy. Construct via ``from_ckpt``.
    """

    schema_version: int
    sequence_id: str
    ego_poses_c2w: np.ndarray  # (N, 4, 4) float32, primary camera→world
    ego_frame_timestamps_us: np.ndarray  # (N,) int64
    ego_primary_camera_id: str
    ego_primary_fov_y_rad: float
    ego_primary_aspect: float
    # T8.13 (schema_v2): full FTheta polynomial intrinsics + locked render
    # resolution. None for pinhole / non-FTheta cameras (schema_v1 ckpts).
    # When ``has_ftheta()`` is True the viewer wires this dict into
    # ``Batch.intrinsics_FThetaCameraModelParameters`` for 3dgut UT
    # rasterizer fisheye projection (matching render.py's geometry).
    ego_primary_intrinsics_ftheta: Optional[dict]
    ego_primary_resolution: Optional[tuple]  # (W, H) int tuple
    tracks: dict[str, dict]  # tid → {poses, size, frame_info, class}
    tracks_camera_timestamps_us: np.ndarray  # (F,) int64
    road_xyz: Optional[np.ndarray]
    road_rgb: Optional[np.ndarray]
    dyn_xyz: Optional[np.ndarray]  # legacy world-frame union
    dyn_rgb: Optional[np.ndarray]
    road_n_total: Optional[int]
    dyn_n_total: Optional[int]
    # T8.11 per-track local-frame dyn LiDAR (None for pre-T8.11 ckpts)
    dyn_local_xyz: Optional[np.ndarray]  # (N, 3) float32, object-local
    dyn_track_ids: Optional[np.ndarray]  # (N,) int64, index into dyn_track_names
    dyn_track_names: Optional[list]  # [K] tid strings, idx → name
    initial_c2w: np.ndarray
    t_us_first: int
    t_us_last: int
    ego_rig_poses_c2w: Optional[np.ndarray] = None  # rig/body→world
    # schema_v3: ordered active-camera projection contracts. Empty for v1/v2.
    # Kept as a defaulted tail field so direct v1/v2-style construction by
    # downstream tools remains source-compatible.
    camera_models: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_ckpt(cls, ckpt: dict) -> Optional["FourDMetadata"]:
        """Parse ``ckpt['viz_4d']`` → FourDMetadata. Returns None if absent."""
        viz = ckpt.get("viz_4d") if isinstance(ckpt, dict) else None
        if not viz:
            return None
        schema_version = int(viz.get("schema_version", 1))
        ego = viz.get("ego", {}) or {}
        tracks_in = viz.get("tracks", {}) or {}
        lidar = viz.get("lidar", {}) or {}
        defaults = viz.get("viewer_defaults", {}) or {}

        tracks: dict[str, dict] = {}
        for tid, t in tracks_in.items():
            tracks[tid] = {
                "poses": _to_np(t.get("poses")),
                "size": _to_np(t.get("size")),
                "frame_info": _to_np(t.get("frame_info")).astype(bool),
                "class": str(t.get("class", "unknown")),
            }
        shared_ts = _to_np(viz.get("tracks_camera_timestamps_us"))
        if shared_ts is None or shared_ts.size == 0:
            shared_ts = np.empty((0,), dtype=np.int64)
        else:
            shared_ts = shared_ts.astype(np.int64)

        initial = _to_np(defaults.get("initial_c2w"))
        if initial is None:
            initial = np.eye(4, dtype=np.float32)

        camera_models = _normalize_camera_models(
            viz.get("camera_models"), schema_version=schema_version
        )
        primary_camera_id = str(ego.get("primary_camera_id", "primary"))
        if camera_models and primary_camera_id not in camera_models:
            raise ValueError(
                f"primary camera '{primary_camera_id}' is absent from camera_models"
            )
        primary_ftheta = ego.get("primary_camera_intrinsics_FTheta")
        primary_resolution = (
            tuple(int(x) for x in ego["primary_camera_resolution"])
            if ego.get("primary_camera_resolution") is not None
            else None
        )
        if camera_models:
            primary_contract = camera_models[primary_camera_id]
            if primary_contract["model_type"] == "FTheta":
                # schema_v3 is authoritative; hydrate the v2 aliases even if a
                # producer omitted them so primary-only viewer startup cannot
                # degrade to ideal pinhole.
                primary_ftheta = primary_contract["intrinsics_FTheta"]
                primary_resolution = primary_contract["native_resolution"]
            elif primary_ftheta is not None:
                raise ValueError(
                    f"primary camera '{primary_camera_id}' is marked "
                    f"{primary_contract['model_type']} but carries legacy "
                    "FTheta intrinsics"
                )

        return cls(
            schema_version=schema_version,
            sequence_id=str(viz.get("sequence_id", "unknown")),
            ego_poses_c2w=_to_np(ego.get("poses_c2w")).astype(np.float32),
            ego_rig_poses_c2w=(
                _to_np(ego.get("rig_poses_c2w")).astype(np.float32)
                if ego.get("rig_poses_c2w") is not None
                else None
            ),
            ego_frame_timestamps_us=_to_np(ego.get("frame_timestamps_us")).astype(np.int64),
            ego_primary_camera_id=primary_camera_id,
            ego_primary_fov_y_rad=float(ego.get("primary_camera_fov_y_rad", 0.78)),
            ego_primary_aspect=float(ego.get("primary_camera_aspect", 1.78)),
            ego_primary_intrinsics_ftheta=primary_ftheta,
            ego_primary_resolution=primary_resolution,
            camera_models=camera_models,
            tracks=tracks,
            tracks_camera_timestamps_us=shared_ts,
            road_xyz=_to_np(lidar.get("road_xyz")),
            road_rgb=_to_np(lidar.get("road_rgb")),
            dyn_xyz=_to_np(lidar.get("dynamic_xyz")),
            dyn_rgb=_to_np(lidar.get("dynamic_rgb")),
            road_n_total=lidar.get("road_n_total"),
            dyn_n_total=lidar.get("dynamic_n_total"),
            dyn_local_xyz=_to_np(lidar.get("dynamic_local_xyz")),
            dyn_track_ids=_to_np(lidar.get("dynamic_track_ids")),
            dyn_track_names=lidar.get("dynamic_track_names"),
            initial_c2w=initial.astype(np.float32),
            t_us_first=int(defaults.get("t_us_first", 0)),
            t_us_last=int(defaults.get("t_us_last", 0)),
        )

    # ---- ergonomic accessors ------------------------------------------------
    def n_tracks(self) -> int:
        return len(self.tracks)

    def n_frames(self) -> int:
        return int(self.tracks_camera_timestamps_us.shape[0])

    def has_ftheta(self) -> bool:
        """True if FTheta polynomial intrinsics + matching resolution are
        present and complete (all 8 required keys + (W, H) tuple).

        Viewer uses this to decide between FTheta projection path (3dgut
        UT rasterizer via ``Batch.intrinsics_FThetaCameraModelParameters``)
        and pinhole fallback (kaolin ``Camera.from_args`` fov approx).
        """
        REQUIRED_KEYS = {
            "resolution",
            "shutter_type",
            "principal_point",
            "reference_poly",
            "pixeldist_to_angle_poly",
            "angle_to_pixeldist_poly",
            "max_angle",
            "linear_cde",
        }
        d = self.ego_primary_intrinsics_ftheta
        if d is None or not isinstance(d, dict):
            return False
        if not REQUIRED_KEYS.issubset(d.keys()):
            return False
        if self.ego_primary_resolution is None:
            return False
        return True

    def has_all_camera_ftheta(self) -> bool:
        """Whether schema-v3 declares every active camera as valid FTheta."""
        return bool(self.camera_models) and all(
            contract["model_type"] == "FTheta"
            for contract in self.camera_models.values()
        )

    def has_lidar(self) -> bool:
        return self.road_xyz is not None or self.dyn_xyz is not None or self.dyn_local_xyz is not None

    def has_per_track_dyn_lidar(self) -> bool:
        """True if T8.11 per-track object-local dyn LiDAR is present.

        Enables the viewer's per-frame transform path (LiDAR points follow
        the cuboid). Falls back to a static world-frame snapshot otherwise.
        """
        return (
            self.dyn_local_xyz is not None
            and self.dyn_track_ids is not None
            and self.dyn_track_names is not None
            and self.dyn_local_xyz.shape[0] > 0
        )

    # ---- timeline lookups ---------------------------------------------------
    def lookup_frame_idx(self, t_us: int) -> int:
        """Binary search the shared dynamic-timestamp buffer → nearest frame."""
        ts = self.tracks_camera_timestamps_us
        if ts.size == 0:
            return 0
        idx = int(np.searchsorted(ts, int(t_us)))
        idx = max(0, min(idx, ts.size - 1))
        if idx > 0 and abs(int(ts[idx - 1]) - t_us) < abs(int(ts[idx]) - t_us):
            idx -= 1
        return idx

    def active_tracks_at(self, frame_idx: int) -> list[str]:
        out: list[str] = []
        for tid, t in self.tracks.items():
            mask = t["frame_info"]
            if mask is None or mask.size == 0:
                continue
            if 0 <= frame_idx < mask.size and bool(mask[frame_idx]):
                out.append(tid)
        return out

    def ego_pose_at(self, t_us: int) -> np.ndarray:
        """Nearest-frame ego pose lookup over ``ego_frame_timestamps_us``."""
        ts = self.ego_frame_timestamps_us
        if ts.size == 0 or self.ego_poses_c2w.size == 0:
            return self.initial_c2w
        idx = int(np.searchsorted(ts, int(t_us)))
        idx = max(0, min(idx, ts.size - 1))
        if idx > 0 and abs(int(ts[idx - 1]) - t_us) < abs(int(ts[idx]) - t_us):
            idx -= 1
        return self.ego_poses_c2w[idx]

    def ego_trajectory_positions(self) -> np.ndarray:
        """Return vehicle/rig origins, with legacy camera-center fallback."""
        poses = self.ego_rig_poses_c2w
        if poses is None or poses.size == 0:
            poses = self.ego_poses_c2w
        return poses[:, :3, 3]
