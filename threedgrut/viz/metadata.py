# SPDX-License-Identifier: Apache-2.0
"""Pack 4D viz metadata into v2 LayeredGaussians ckpts (Stage 8 / T8.2).

The on-disk layout produced by ``extract_4d_metadata`` is consumed by
``threedgrut_playground.viser_gui_4d`` (the new 4D viewer) and persisted under
``ckpt["viz_4d"]`` by ``Trainer.save_checkpoint`` when ``conf.viz_4d.enabled``.

Schema (schema_version=3) — see plan §2.1:

    {
        "schema_version":     3,
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
        "camera_models": {
            camera_id: {
                "model_type": "FTheta" | "OpenCVPinhole" | "IdealPinhole",
                "native_resolution": (W, H),
                "intrinsics_FTheta": {eight-key FTheta dictionary},
                "parameter_fingerprint": str | None,
            },
        },
        "tracks_camera_timestamps_us": Tensor[F] int64,
        "lidar": {
            "road_xyz" / "road_rgb" / "dynamic_xyz" / "dynamic_rgb": Tensor or None,
            "road_n_total" / "road_subsample" / "dynamic_*": int or None,
        },
        "viewer_defaults": {"initial_c2w", "near", "far", "resolution",
                            "t_us_first", "t_us_last"},
    }

Failure model: camera-model extraction is fail-fast because silently omitting an
active FTheta contract would make the viewer fall back to the wrong projection.
The remaining optional sections are independently best-effort: a failing
sub-extractor returns an empty/None placeholder so an incomplete dataset (for
example, tracks not populated) can still produce a partially useful block.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from threedgrut.utils.logger import logger

SCHEMA_VERSION = 3  # PIN-FTHETA: all active camera models + native resolutions

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


def _as_numpy(value: Any, dtype=None) -> np.ndarray:
    """Convert torch/NCore/numpy values without relying on uint64 torch support."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _camera_resolution(camera_model) -> tuple[int, int]:
    raw = _as_numpy(camera_model.resolution, np.int64).reshape(-1)
    if raw.size != 2:
        raise ValueError(f"camera resolution must contain W,H, got shape {raw.shape}")
    resolution = (int(raw[0]), int(raw[1]))
    if resolution[0] <= 0 or resolution[1] <= 0:
        raise ValueError(f"camera resolution must be positive, got {resolution}")
    return resolution


def _extract_ftheta_dict(camera_model) -> Optional[dict]:
    """Return the portable eight-key FTheta contract, or ``None`` for pinhole."""
    if getattr(camera_model, "max_angle", None) is None or not hasattr(
        camera_model, "get_parameters"
    ):
        return None
    params = camera_model.get_parameters()
    # OpenCVFisheye models can also expose max_angle + get_parameters(), but
    # they do not implement the 3DGUT FTheta polynomial contract. Requiring
    # the complete native parameter surface preserves legacy mixed-camera
    # checkpoints without relabelling another fisheye model as FTheta.
    required_attrs = {
        "resolution",
        "shutter_type",
        "principal_point",
        "reference_poly",
        "pixeldist_to_angle_poly",
        "angle_to_pixeldist_poly",
        "max_angle",
        "linear_cde",
    }
    if any(not hasattr(params, key) for key in required_attrs):
        return None
    result = {
        "resolution": _as_numpy(params.resolution, np.int64),
        "shutter_type": params.shutter_type.name,
        "principal_point": _as_numpy(params.principal_point, np.float32),
        "reference_poly": params.reference_poly.name,
        "pixeldist_to_angle_poly": _as_numpy(
            params.pixeldist_to_angle_poly, np.float32
        ),
        "angle_to_pixeldist_poly": _as_numpy(
            params.angle_to_pixeldist_poly, np.float32
        ),
        "max_angle": float(params.max_angle),
        "linear_cde": _as_numpy(params.linear_cde, np.float32),
    }
    missing = FTHETA_REQUIRED_KEYS.difference(result)
    if missing:  # defensive: construction above should make this impossible
        raise ValueError(f"FTheta parameters missing keys: {sorted(missing)}")
    return result


def _legacy_camera_model_type(camera_model) -> str:
    name = type(camera_model).__name__
    if "OpenCVPinhole" in name or hasattr(camera_model, "radial_coeffs"):
        return "OpenCVPinhole"
    return "IdealPinhole"


def _extract_camera_models(dataset) -> dict[str, dict]:
    """Persist every active camera's model contract in checkpoint order.

    ``ftheta_override_enabled`` marks the matched Arm F path. In that mode a
    non-FTheta active camera is a corrupt experiment and must abort checkpoint
    creation; silently writing a pinhole/empty entry would let Viser render an
    ideal-pinhole approximation that training never used.
    """
    sequence_id = dataset.sequence_id
    camera_ids = list(getattr(dataset, "camera_ids", ()) or ())
    models = dataset.sequence_camera_models[sequence_id]
    strict_ftheta = bool(getattr(dataset, "ftheta_override_enabled", False))
    fingerprints = getattr(dataset, "ftheta_parameter_fingerprints", {}) or {}

    out: dict[str, dict] = {}
    for camera_id in camera_ids:
        if camera_id not in models:
            raise ValueError(f"active camera '{camera_id}' has no camera model")
        camera_model = models[camera_id]
        resolution = _camera_resolution(camera_model)
        try:
            ftheta_dict = _extract_ftheta_dict(camera_model)
        except Exception as exc:
            raise ValueError(
                f"active camera '{camera_id}' FTheta extraction failed: {exc}"
            ) from exc

        if strict_ftheta and ftheta_dict is None:
            raise ValueError(
                f"active camera '{camera_id}' is not FTheta while "
                "ftheta_override_enabled=True"
            )

        model_type = (
            "FTheta"
            if ftheta_dict is not None
            else _legacy_camera_model_type(camera_model)
        )
        entry: dict[str, Any] = {
            "model_type": model_type,
            "native_resolution": resolution,
        }
        if ftheta_dict is not None:
            ftheta_resolution = tuple(
                int(x) for x in np.asarray(ftheta_dict["resolution"]).reshape(-1)
            )
            if ftheta_resolution != resolution:
                raise ValueError(
                    f"active camera '{camera_id}' resolution mismatch: "
                    f"model={resolution}, FTheta={ftheta_resolution}"
                )
            entry["intrinsics_FTheta"] = ftheta_dict
            if camera_id in fingerprints:
                entry["parameter_fingerprint"] = str(fingerprints[camera_id])
        out[str(camera_id)] = entry
    return out


# --------------------------------------------------------------------- ego
def _detect_primary_camera(dataset):
    """Return (camera_id, fov_y_rad, aspect, ftheta_dict, resolution).

    T8.13: tuple expanded from 3 to 5. ``ftheta_dict`` is the 8-key
    params_dict that NCoreDataset.get_camera_intrinsics produces for
    FTheta cameras (see datasetNcore.py:1467-1477) — it is consumed
    verbatim by the 3dgut UT rasterizer in threedgut_tracer/tracer.py:471
    via ``_3dgut_plugin.fromFThetaCameraModelParameters``. None for
    pinhole / non-FTheta cameras (viewer falls back to pinhole approx).
    ``resolution`` is the (W, H) int tuple of the trained camera —
    viser_gui_4d locks render dims to this for FTheta ckpts because
    principal_point is in pixel coords.

    FTheta detection is duck-typed (``get_parameters`` + ``max_angle``)
    so Mac tests don't need the NCore SDK; production NCore datasets
    satisfy the same surface.

    Mirrors the FOV math in datasetNcore.create_dataset_camera_visualization
    (line 1531-1546): FTheta uses 2 * max_angle; pinhole-style uses
    2*atan(0.5*h/fy).
    """
    fallback_id = "primary"
    fallback_fov = 0.78  # ~45°
    fallback_aspect = 16.0 / 9.0
    fallback_resolution = (1, 1)

    try:
        camera_id = dataset.camera_ids[0]
        seq_id = dataset.sequence_id
        camera_model = dataset.sequence_camera_models[seq_id][camera_id]
        w, h = _camera_resolution(camera_model)
        aspect = (w / h) if h > 0 else fallback_aspect
        resolution = (w, h)

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

        # T8.13: extract FTheta polynomial params_dict for ckpt persistence.
        # Duck-type on get_parameters() + max_angle so this works without
        # NCore SDK (Mac tests). The 8 keys mirror datasetNcore.py:1467-1477.
        ftheta_dict = None
        if max_angle is not None and hasattr(camera_model, "get_parameters"):
            try:
                # NCore returns resolution as uint64. Convert through numpy so
                # torch's lack of uint64 support cannot silently erase FTheta.
                ftheta_dict = _extract_ftheta_dict(camera_model)
            except Exception as e:
                logger.warning(
                    f"[viz_4d] FTheta intrinsics extraction failed: {e}; "
                    "ftheta_dict=None"
                )
                ftheta_dict = None

        return str(camera_id), float(fov_y), float(aspect), ftheta_dict, resolution
    except Exception as e:
        logger.warning(f"[viz_4d] primary camera detection fell back to defaults: {e}")
        return fallback_id, fallback_fov, fallback_aspect, None, fallback_resolution


def _extract_ego(dataset, conf) -> dict:
    """Pull ego (primary camera) trajectory + timestamps.

    Only the primary camera (camera_ids[0]) is exported. The dataset's
    ``get_poses()`` concatenates per-camera frame poses in ``camera_ids``
    order — so the first ``N_primary`` rows belong to the primary camera and
    form the correct time-ordered ego trajectory. Including all cameras
    (Bug B6) produced 2623-frame piecewise paths where 0-5s was front-wide,
    5-10s was rear-tele, etc. — viewer Play visibly jumped between camera
    viewpoints instead of following one continuous ego trail. We slice the
    primary camera's prefix so ``ego_poses_c2w.shape[0] ==
    n_frames(primary_camera)`` and matches the primary's frame timestamps.
    Multi-camera trajectories are left to v2.x.
    """
    primary_id, fov_y, aspect, ftheta_dict, resolution = _detect_primary_camera(dataset)

    poses_np = dataset.get_poses()
    poses_full = np.asarray(poses_np, dtype=np.float32)

    # Camera frame timestamps follow the same camera_train_frame_indices order
    # as get_poses (which loops over self.camera_ids). Only the primary
    # camera's slice is exported as the ego trajectory.
    try:
        import ncore  # type: ignore[import-not-found]

        end_idx = ncore.data.FrameTimepoint.END
    except Exception:
        end_idx = 1  # FrameTimepoint.END is column 1 in NCore

    n_primary = 0
    ts_np = np.empty((0,), dtype=np.int64)
    try:
        seq_id = dataset.sequence_id
        primary_frame_indices = dataset.camera_train_frame_indices.get(primary_id)
        if primary_frame_indices is not None and len(primary_frame_indices) > 0:
            n_primary = len(primary_frame_indices)
            sensor = dataset.sequence_camera_sensors[seq_id][primary_id]
            ts_np = np.asarray(sensor.frames_timestamps_us)[primary_frame_indices, end_idx].astype(np.int64)
    except Exception as e:
        logger.warning(
            f"[viz_4d] primary-camera timestamp extraction failed: {e}; " f"falling back to empty ts + full poses"
        )

    # Sanity: get_poses() places primary camera first when N_primary > 0.
    # Slice or fall back to the whole array if our count is off.
    if n_primary > 0 and poses_full.shape[0] >= n_primary:
        poses_c2w = torch.from_numpy(poses_full[:n_primary])
    else:
        poses_c2w = torch.from_numpy(poses_full)

    # Camera poses remain the rendering/frustum contract. The UI trajectory,
    # however, represents the vehicle and therefore needs the NCore rig origin.
    rig_poses_c2w = None
    if ts_np.size > 0:
        try:
            pose_graph = dataset.sequence_loaders[dataset.sequence_id].pose_graph
            rig_native = pose_graph.evaluate_poses(
                "rig", "world", ts_np.astype(np.uint64)
            )
            rig_native = np.asarray(rig_native, dtype=np.float64).reshape(-1, 4, 4)
            T_w2wg = np.asarray(
                getattr(dataset, "T_world_to_world_global", np.eye(4)),
                dtype=np.float64,
            )
            rig_world_global = np.einsum("ij,njk->nik", T_w2wg, rig_native)
            if rig_world_global.shape[0] == poses_c2w.shape[0]:
                rig_poses_c2w = torch.from_numpy(rig_world_global.astype(np.float32))
        except Exception as e:
            logger.warning(
                f"[viz_4d] rig trajectory extraction failed: {e}; "
                "legacy camera-center fallback remains available"
            )

    return {
        "poses_c2w": _to_cpu_float32(poses_c2w),
        "rig_poses_c2w": (
            _to_cpu_float32(rig_poses_c2w) if rig_poses_c2w is not None else None
        ),
        "frame_timestamps_us": torch.from_numpy(ts_np),
        "primary_camera_id": primary_id,
        "primary_camera_fov_y_rad": fov_y,
        "primary_camera_aspect": aspect,
        # T8.13: full FTheta polynomial intrinsics for viser_gui_4d → 3dgut UT
        # rasterizer fisheye projection. None for pinhole / non-FTheta cameras
        # (viewer falls back to existing pinhole approximation path).
        "primary_camera_intrinsics_FTheta": ftheta_dict,
        "primary_camera_resolution": resolution,
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
            "poses": _to_cpu_float32(poses_buf),
            "size": (
                _to_cpu_float32(meta.get("size", torch.zeros(3)))
                if not isinstance(meta.get("size"), torch.Tensor) or meta["size"].numel() > 0
                else _to_cpu_float32(meta["size"])
            ),
            "frame_info": (
                _to_cpu_bool(active_buf) if active_buf is not None else torch.ones(poses_buf.shape[0], dtype=torch.bool)
            ),
            "class": str(meta.get("class", "unknown")),
        }

    shared_ts_buf = getattr(model, "tracks_camera_timestamps_us", None)
    shared_ts = _to_cpu_int64(shared_ts_buf) if shared_ts_buf is not None else None
    return out, shared_ts


# --------------------------------------------------------------------- lidar
def _model_to_instance_pts_dict(model) -> dict:
    """Build the {tid: {poses, size, frame_info}} dict expected by
    ``init_dynamic_rigid_layer`` from a populated LayeredGaussians model."""
    out: dict[str, dict] = {}
    tracks_poses = getattr(model, "tracks_poses", {}) or {}
    tracks_active = getattr(model, "tracks_active", {}) or {}
    tracks_metadata = getattr(model, "tracks_metadata", {}) or {}
    for tid, poses in tracks_poses.items():
        active = tracks_active.get(tid)
        meta = tracks_metadata.get(tid, {})
        size = meta.get("size")
        if size is None:
            size = torch.zeros(3, dtype=torch.float32)
        out[tid] = {
            "poses": poses,
            "size": size,
            "frame_info": active if active is not None else torch.ones(poses.shape[0], dtype=torch.bool),
        }
    return out


def _extract_lidar(dataset, model, conf, *, road_subsample: Optional[int], dyn_pts_per_track: int) -> dict:
    """Pull road LiDAR (static) + dynamic LiDAR (per-track object-local).

    Road LiDAR stays in world frame (it IS the static ground). Dynamic LiDAR
    is re-projected into each track's object-local frame via
    ``init_dynamic_rigid_layer``, so the viewer can transform back to world
    every frame using the current track pose — keeping points glued to the
    moving cuboid.

    Schema:
      road_xyz / road_rgb:                      world-frame, static
      dynamic_local_xyz / dynamic_track_ids:    object-local per-track
      dynamic_track_names:                      idx → tid name mapping
      dynamic_xyz / dynamic_rgb (legacy):       world-frame union (v1 viewer
                                                fallback; deprecated for
                                                animated playback)
    """
    out: dict[str, Any] = {
        "road_xyz": None,
        "road_rgb": None,
        "road_n_total": None,
        "road_subsample": None,
        # New per-track local schema (T8.11)
        "dynamic_local_xyz": None,
        "dynamic_track_ids": None,
        "dynamic_track_names": None,
        "dynamic_pts_per_track": None,
        # Legacy world-frame (v1 viewer fallback)
        "dynamic_xyz": None,
        "dynamic_rgb": None,
        "dynamic_n_total": None,
        "dynamic_subsample": None,
    }

    # ---- road: world-frame static ----
    try:
        getter = getattr(dataset, "get_road_lidar_points", None)
        if getter is not None:
            xyz, rgb = getter()
            if xyz is not None and xyz.numel() > 0:
                n_total = int(xyz.shape[0])
                xyz_s = _subsample(xyz, road_subsample)
                out["road_xyz"] = _to_cpu_float32(xyz_s)
                if rgb is not None and rgb.numel() > 0:
                    rgb_s = (
                        rgb[: xyz_s.shape[0]]
                        if (road_subsample is None or n_total <= road_subsample)
                        else rgb[torch.randperm(n_total)[:road_subsample]]
                    )
                    out["road_rgb"] = _to_cpu_float32(rgb_s)
                out["road_n_total"] = n_total
                out["road_subsample"] = int(xyz_s.shape[0])
    except Exception as e:
        logger.warning(f"[viz_4d] road LiDAR extraction failed: {e}")

    # ---- dynamic: per-track object-local (T8.11) ----
    try:
        getter = getattr(dataset, "get_dynamic_lidar_points", None)
        if getter is None:
            return out
        dyn_xyz_world, dyn_rgb = getter()
        if dyn_xyz_world is None or dyn_xyz_world.numel() == 0:
            return out

        # Populate per-track object-local via the same routine the trainer's
        # dynamic_rigid layer uses (mutates instance_pts_dict in place).
        from threedgrut.layers.dynamic_rigid_init import init_dynamic_rigid_layer

        instance_pts_dict = _model_to_instance_pts_dict(model)
        if not instance_pts_dict:
            # No tracks → keep just the legacy world-frame union for fallback.
            out["dynamic_xyz"] = _to_cpu_float32(dyn_xyz_world)
            out["dynamic_n_total"] = int(dyn_xyz_world.shape[0])
            out["dynamic_subsample"] = int(dyn_xyz_world.shape[0])
            return out
        local_pts, track_ids, track_names = init_dynamic_rigid_layer(
            instance_pts_dict,
            dyn_xyz_world,
            max_pts_per_track=dyn_pts_per_track,
        )
        out["dynamic_local_xyz"] = _to_cpu_float32(local_pts)
        out["dynamic_track_ids"] = _to_cpu_int64(track_ids)
        out["dynamic_track_names"] = list(track_names)
        out["dynamic_pts_per_track"] = int(dyn_pts_per_track)
        out["dynamic_n_total"] = int(dyn_xyz_world.shape[0])
        out["dynamic_subsample"] = int(local_pts.shape[0])
    except Exception as e:
        logger.warning(f"[viz_4d] dynamic LiDAR extraction failed: {e}")

    return out


# --------------------------------------------------------------------- defaults
def _extract_defaults(ego: dict, conf) -> dict:
    """Build initial viewer config from ego trajectory + conf hints."""
    poses = ego.get("poses_c2w")
    ts = ego.get("frame_timestamps_us")
    initial_c2w = poses[0].clone() if poses is not None and poses.numel() > 0 else torch.eye(4, dtype=torch.float32)
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
        "near": _val("default_near", 0.1),
        "far": _val("default_far", 500.0),
        "resolution": int(_val("default_resolution", 1024)),
        "t_us_first": t_first,
        "t_us_last": t_last,
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
    include_lidar = bool(viz_conf.get("include_lidar", True) if hasattr(viz_conf, "get") else True)
    road_subsample = viz_conf.get("lidar_road_subsample", 200_000) if hasattr(viz_conf, "get") else 200_000
    # T8.11: dynamic LiDAR is now per-track (object-local) so the cap is per
    # track, not total. 5000 pts/track × ~30-100 tracks ≈ 150-500K total, on
    # par with the old 100K total but enough headroom for dense long-active
    # tracks. Backward-compatible config key:
    # `lidar_dynamic_pts_per_track` first, fall back to legacy
    # `lidar_dynamic_subsample` / 20 as a heuristic if user only set total.
    dyn_pts_per_track = viz_conf.get("lidar_dynamic_pts_per_track", None) if hasattr(viz_conf, "get") else None
    if dyn_pts_per_track is None:
        legacy_total = viz_conf.get("lidar_dynamic_subsample", 100_000) if hasattr(viz_conf, "get") else 100_000
        # rough split — driving clips average ~30-50 tracks, want ~5K each
        dyn_pts_per_track = max(1_000, int(legacy_total) // 20)

    dataset_type = "ncore"
    sequence_id = str(getattr(dataset, "sequence_id", "unknown"))

    # This is intentionally outside the section-level best-effort fallbacks.
    # An explicit FTheta training arm must never write an incomplete camera
    # contract that the viewer could later interpret as ideal pinhole.
    camera_models = _extract_camera_models(dataset)
    ego = _extract_ego(dataset, conf)
    tracks, shared_ts = _extract_tracks(model)
    if include_lidar:
        lidar = _extract_lidar(
            dataset,
            model,
            conf,
            road_subsample=int(road_subsample),
            dyn_pts_per_track=int(dyn_pts_per_track),
        )
    else:
        # include_lidar=False → skip LiDAR entirely; viewer renders without
        # ground-truth point clouds (Gaussian background still works).
        lidar = {
            "road_xyz": None,
            "road_rgb": None,
            "road_n_total": None,
            "road_subsample": None,
            "dynamic_local_xyz": None,
            "dynamic_track_ids": None,
            "dynamic_track_names": None,
            "dynamic_pts_per_track": None,
            "dynamic_xyz": None,
            "dynamic_rgb": None,
            "dynamic_n_total": None,
            "dynamic_subsample": None,
        }
    defaults = _extract_defaults(ego, conf)

    out = {
        "schema_version": SCHEMA_VERSION,
        "dataset_type": dataset_type,
        "sequence_id": sequence_id,
        "camera_models": camera_models,
        "ego": ego,
        "tracks": tracks,
        "tracks_camera_timestamps_us": shared_ts,
        "lidar": lidar,
        "viewer_defaults": defaults,
    }
    logger.info(
        f"[viz_4d] packed schema_v{SCHEMA_VERSION} "
        f"({len(tracks)} tracks, cameras={len(camera_models)}, "
        f"ego_N={ego['poses_c2w'].shape[0]}, "
        f"road_pts={lidar.get('road_subsample')})"
    )
    return out
