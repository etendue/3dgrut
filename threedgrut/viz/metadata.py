# SPDX-License-Identifier: Apache-2.0
"""Pack 4D viz metadata into v2 LayeredGaussians ckpts (Stage 8 / T8.2).

The on-disk layout produced by ``extract_4d_metadata`` is consumed by
``threedgrut_playground.viser_gui_4d`` (the new 4D viewer) and persisted under
``ckpt["viz_4d"]`` by ``Trainer.save_checkpoint`` when ``conf.viz_4d.enabled``.

Schema (schema_version=1) — see plan §2.1:

    {
        "schema_version":     1,
        "dataset_type":       "ncore",
        "sequence_id":        str,
        "ego": {
            "poses_c2w":               Tensor[N, 4, 4] float32,
            "frame_timestamps_us":     Tensor[N]      int64,
            "primary_camera_id":       str,
            "primary_camera_fov_y_rad": float,
            "primary_camera_aspect":    float,
        },
        "tracks": {tid: {"poses", "size", "frame_info", "class"}},
        "tracks_camera_timestamps_us": Tensor[F] int64,
        "lidar": {
            "road_xyz" / "road_rgb" / "dynamic_xyz" / "dynamic_rgb": Tensor or None,
            "road_n_total" / "road_subsample" / "dynamic_*": int or None,
        },
        "viewer_defaults": {"initial_c2w", "near", "far", "resolution",
                            "t_us_first", "t_us_last"},
    }

Failure model: every section is independently try/except'd. A failing sub-
extractor returns an empty/None placeholder rather than raising so that an
incomplete dataset (e.g. tracks not populated) still produces a partially-
useful viz_4d block.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from threedgrut.utils.logger import logger

SCHEMA_VERSION = 1


# --------------------------------------------------------------------- helpers
def _to_cpu_float32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(dtype=torch.float32, device="cpu").contiguous()


def _to_cpu_int64(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(dtype=torch.int64, device="cpu").contiguous()


def _to_cpu_bool(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(dtype=torch.bool, device="cpu").contiguous()


def _subsample(t: torch.Tensor, k: Optional[int]) -> torch.Tensor:
    """Random subsample first axis to k samples; return unchanged if k>=N or k is None."""
    if k is None or k <= 0 or t.shape[0] <= k:
        return t
    perm = torch.randperm(t.shape[0])[:k]
    return t[perm]


# --------------------------------------------------------------------- ego
def _detect_primary_camera(dataset) -> tuple[str, float, float]:
    """Return (camera_id, fov_y_rad, aspect) for the first camera in dataset.camera_ids.

    Mirrors the FOV math in datasetNcore.create_dataset_camera_visualization
    (line 1531-1546): FTheta uses 2 * max_angle; pinhole-style uses
    2*atan(0.5*h/fy).
    """
    fallback_id = "primary"
    fallback_fov = 0.78  # ~45°
    fallback_aspect = 16.0 / 9.0

    try:
        camera_id = dataset.camera_ids[0]
        seq_id = dataset.sequence_id
        camera_model = dataset.sequence_camera_models[seq_id][camera_id]
        w = float(camera_model.resolution[0].item())
        h = float(camera_model.resolution[1].item())
        aspect = w / h if h > 0 else fallback_aspect

        # FTheta (fisheye-ish) vs pinhole — match dataset's own branching.
        max_angle = getattr(camera_model, "max_angle", None)
        focal_length = getattr(camera_model, "focal_length", None)
        if focal_length is not None:
            fy = float(focal_length[1])
            fov_y = 2.0 * float(np.arctan(0.5 * h / fy)) if fy > 0 else fallback_fov
        elif max_angle is not None:
            fov_y = 2.0 * float(max_angle)
        else:
            fov_y = fallback_fov
        return str(camera_id), float(fov_y), float(aspect)
    except Exception as e:
        logger.warning(f"[viz_4d] primary camera detection fell back to defaults: {e}")
        return fallback_id, fallback_fov, fallback_aspect


def _extract_ego(dataset, conf) -> dict:
    """Pull ego (primary camera) trajectory + timestamps.

    Only the primary camera (camera_ids[0]) is exported. Multi-camera support
    is left to v2.x — viz GUI shows a single moving frustum.
    """
    primary_id, fov_y, aspect = _detect_primary_camera(dataset)

    poses_np = dataset.get_poses()
    poses_c2w = torch.from_numpy(np.asarray(poses_np, dtype=np.float32))

    # Camera frame timestamps follow the same camera_train_frame_indices order
    # as get_poses (which loops over self.camera_ids). We replicate that loop.
    try:
        import ncore  # type: ignore[import-not-found]

        end_idx = ncore.data.FrameTimepoint.END
    except Exception:
        end_idx = 1  # FrameTimepoint.END is column 1 in NCore

    all_ts: list[np.ndarray] = []
    try:
        seq_id = dataset.sequence_id
        for camera_id in dataset.camera_ids:
            frame_indices = dataset.camera_train_frame_indices[camera_id]
            if len(frame_indices) == 0:
                continue
            sensor = dataset.sequence_camera_sensors[seq_id][camera_id]
            cam_ts = np.asarray(sensor.frames_timestamps_us)[
                frame_indices, end_idx
            ].astype(np.int64)
            all_ts.append(cam_ts)
        ts_np = np.concatenate(all_ts) if all_ts else np.empty((0,), dtype=np.int64)
    except Exception as e:
        logger.warning(f"[viz_4d] frame timestamp extraction failed: {e}; using empty")
        ts_np = np.empty((0,), dtype=np.int64)

    return {
        "poses_c2w":                _to_cpu_float32(poses_c2w),
        "frame_timestamps_us":      torch.from_numpy(ts_np),
        "primary_camera_id":        primary_id,
        "primary_camera_fov_y_rad": fov_y,
        "primary_camera_aspect":    aspect,
    }


# --------------------------------------------------------------------- tracks
def _extract_tracks(model) -> tuple[dict, Optional[torch.Tensor]]:
    """Pull dynamic-rigid tracks from a populated LayeredGaussians.

    ``model.tracks_poses`` / ``tracks_active`` are populated by
    ``populate_tracks``; class/size live in ``model.tracks_metadata`` (added
    in T8.2). Returns ``({}, None)`` when the model has no tracks (single-bg
    or road-only LayeredGaussians).
    """
    tracks_poses = getattr(model, "tracks_poses", {}) or {}
    if not tracks_poses:
        return {}, None

    tracks_active = getattr(model, "tracks_active", {}) or {}
    tracks_metadata = getattr(model, "tracks_metadata", {}) or {}

    out: dict[str, dict] = {}
    for tid, poses_buf in tracks_poses.items():
        active_buf = tracks_active.get(tid)
        meta = tracks_metadata.get(tid, {}) or {}
        out[tid] = {
            "poses":      _to_cpu_float32(poses_buf),
            "size":       _to_cpu_float32(meta.get("size", torch.zeros(3)))
                            if not isinstance(meta.get("size"), torch.Tensor)
                            or meta["size"].numel() > 0
                            else _to_cpu_float32(meta["size"]),
            "frame_info": _to_cpu_bool(active_buf)
                            if active_buf is not None
                            else torch.ones(poses_buf.shape[0], dtype=torch.bool),
            "class":      str(meta.get("class", "unknown")),
        }

    shared_ts_buf = getattr(model, "tracks_camera_timestamps_us", None)
    shared_ts = _to_cpu_int64(shared_ts_buf) if shared_ts_buf is not None else None
    return out, shared_ts


# --------------------------------------------------------------------- lidar
def _extract_lidar(dataset, conf, *, road_subsample: Optional[int],
                   dyn_subsample: Optional[int]) -> dict:
    """Pull road + dynamic LiDAR point clouds with optional subsample.

    The functions ``dataset.get_road_lidar_points()`` / ``get_dynamic_lidar_points()``
    return ``(xyz[M, 3] torch.Tensor, rgb[M, 3] | None)``. Subsampling uses
    torch.randperm (deterministic across runs only if a generator is seeded
    upstream).
    """
    out: dict[str, Any] = {
        "road_xyz": None, "road_rgb": None, "road_n_total": None, "road_subsample": None,
        "dynamic_xyz": None, "dynamic_rgb": None,
        "dynamic_n_total": None, "dynamic_subsample": None,
    }

    def _pull(name: str, getter_name: str, k: Optional[int]) -> None:
        try:
            getter = getattr(dataset, getter_name, None)
            if getter is None:
                return
            xyz, rgb = getter()
            if xyz is None or xyz.numel() == 0:
                return
            n_total = int(xyz.shape[0])
            xyz_s = _subsample(xyz, k)
            out[f"{name}_xyz"] = _to_cpu_float32(xyz_s)
            if rgb is not None and rgb.numel() > 0:
                rgb_s = rgb[: xyz_s.shape[0]] if (k is None or n_total <= k) \
                    else rgb[torch.randperm(n_total)[:k]]
                out[f"{name}_rgb"] = _to_cpu_float32(rgb_s)
            out[f"{name}_n_total"] = n_total
            out[f"{name}_subsample"] = int(xyz_s.shape[0])
        except Exception as e:
            logger.warning(f"[viz_4d] LiDAR extraction for '{name}' failed: {e}")

    _pull("road", "get_road_lidar_points", road_subsample)
    _pull("dynamic", "get_dynamic_lidar_points", dyn_subsample)
    return out


# --------------------------------------------------------------------- defaults
def _extract_defaults(ego: dict, conf) -> dict:
    """Build initial viewer config from ego trajectory + conf hints."""
    poses = ego.get("poses_c2w")
    ts = ego.get("frame_timestamps_us")
    initial_c2w = (poses[0].clone() if poses is not None and poses.numel() > 0
                   else torch.eye(4, dtype=torch.float32))
    t_first = int(ts[0].item()) if ts is not None and ts.numel() > 0 else 0
    t_last = int(ts[-1].item()) if ts is not None and ts.numel() > 0 else 0
    viz_conf = conf.get("viz_4d", {}) if hasattr(conf, "get") else {}

    def _val(key: str, default: float) -> float:
        try:
            v = viz_conf.get(key, default) if hasattr(viz_conf, "get") else default
            return float(v)
        except Exception:
            return default

    return {
        "initial_c2w": initial_c2w,
        "near":        _val("default_near", 0.1),
        "far":         _val("default_far", 500.0),
        "resolution":  int(_val("default_resolution", 1024)),
        "t_us_first":  t_first,
        "t_us_last":   t_last,
    }


# --------------------------------------------------------------------- entry
def extract_4d_metadata(model, dataset, conf) -> dict:
    """Top-level: pack ckpt["viz_4d"] dict.

    Args:
        model:    A LayeredGaussians instance (after populate_tracks if any).
        dataset:  NCoreDataset (or duck-typed equivalent exposing get_poses /
                  camera_train_frame_indices / sequence_camera_sensors /
                  sequence_camera_models / get_road_lidar_points etc.).
        conf:     Hydra DictConfig — reads conf.viz_4d.* and conf.dataset.type.

    Returns:
        A pure-CPU dict per the schema in the module docstring. All tensors
        moved to CPU + float32/int64 so the ckpt is portable across GPU types.
    """
    # Read sub-conf safely (DictConfig.get returns None on missing key).
    viz_conf = conf.get("viz_4d", {}) if hasattr(conf, "get") else {}
    include_lidar = bool(
        viz_conf.get("include_lidar", True) if hasattr(viz_conf, "get") else True
    )
    road_subsample = (
        viz_conf.get("lidar_road_subsample", 200_000)
        if hasattr(viz_conf, "get") else 200_000
    )
    dyn_subsample = (
        viz_conf.get("lidar_dynamic_subsample", 100_000)
        if hasattr(viz_conf, "get") else 100_000
    )

    dataset_type = "ncore"
    sequence_id = str(getattr(dataset, "sequence_id", "unknown"))

    ego = _extract_ego(dataset, conf)
    tracks, shared_ts = _extract_tracks(model)
    if include_lidar:
        lidar = _extract_lidar(
            dataset, conf,
            road_subsample=int(road_subsample),
            dyn_subsample=int(dyn_subsample),
        )
    else:
        # include_lidar=False → skip LiDAR entirely; viewer renders without
        # ground-truth point clouds (Gaussian background still works).
        lidar = {
            "road_xyz": None, "road_rgb": None,
            "road_n_total": None, "road_subsample": None,
            "dynamic_xyz": None, "dynamic_rgb": None,
            "dynamic_n_total": None, "dynamic_subsample": None,
        }
    defaults = _extract_defaults(ego, conf)

    out = {
        "schema_version":               SCHEMA_VERSION,
        "dataset_type":                 dataset_type,
        "sequence_id":                  sequence_id,
        "ego":                          ego,
        "tracks":                       tracks,
        "tracks_camera_timestamps_us":  shared_ts,
        "lidar":                        lidar,
        "viewer_defaults":              defaults,
    }
    logger.info(
        f"[viz_4d] packed schema_v{SCHEMA_VERSION} "
        f"({len(tracks)} tracks, ego_N={ego['poses_c2w'].shape[0]}, "
        f"road_pts={lidar.get('road_subsample')})"
    )
    return out
