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
    cam_pts: np.ndarray,  # [N, 3] in camera frame (right-down-front)
    intrinsics: dict,  # {fx, fy, cx, cy}
    image_shape: Tuple[int, int],  # (H, W)
) -> Tuple[np.ndarray, np.ndarray]:
    """Project Nx3 camera-frame points to image UV. Returns (uv[N,2], valid[N]).

    Valid := (z > 1e-3) ∧ (0 <= u < W) ∧ (0 <= v < H).
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
    uv: np.ndarray,  # [N, 2] float
    ray_depth: np.ndarray,  # [N]    float (ray-depth)
    valid: np.ndarray,  # [N]    bool
    H: int,
    W: int,
) -> np.ndarray:
    """Scatter sparse ray-depth points to a dense [H, W] depth map.

    Conflict resolution: if multiple points fall in the same pixel, keep the
    nearest (smallest ray-depth) — closest surface occludes the rest.

    Invalid points (valid=False) are dropped BEFORE sorting so NaN / no-return
    ray_depth values cannot perturb the argsort order of the valid points
    (np.argsort with NaN is platform-dependent). All-invalid input returns an
    all-zero map.
    """
    dmap = np.zeros((H, W), dtype=np.float32)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return dmap
    sub_depth = ray_depth[valid_idx]
    # Descending so the nearest point is written last and wins the pixel.
    order = valid_idx[np.argsort(-sub_depth)]
    for i in order:
        u, v = uv[i]
        ui, vi = int(np.floor(u)), int(np.floor(v))
        if 0 <= ui < W and 0 <= vi < H:
            dmap[vi, ui] = ray_depth[i]
    return dmap


def _accumulate_lidar_world_points(
    dataset,
    ts_cam_us: int,
    time_window_us: int,
) -> np.ndarray:
    """Aggregate world-frame LiDAR points within ``±time_window_us`` of the
    camera frame timestamp, in the world-global frame.

    Densifies the single-sweep cloud (LiDAR is ~10 Hz vs ~30 Hz cameras) while
    bounding dynamic-object smear via the time window. If no sweep falls in the
    window, falls back to the single nearest sweep.

    Returns ``[N, 3]`` float64 world-global points (empty ``[0, 3]`` if no
    LiDAR source has any sweep).
    """
    sid = dataset.sequence_id
    sources = dataset.sequence_point_clouds_sources[sid]
    source_ids = dataset.sequence_point_clouds_source_ids[sid]
    pose_graph = dataset.sequence_loaders[sid].pose_graph
    T_wg = np.asarray(dataset.T_world_to_world_global, dtype=np.float64)
    R_wg, t_wg = T_wg[:3, :3], T_wg[:3, 3:4]

    chunks: list[np.ndarray] = []
    # Track the globally-nearest sweep for the fallback (no sweep in window).
    best_dt = None
    best_ref = None  # (source, pc_idx)

    for source_id in source_ids:
        source = sources[source_id]
        pc_ts = np.asarray(source.pc_timestamps_us)
        for pc_idx in range(len(pc_ts)):
            ts_us = int(pc_ts[pc_idx])
            dt = abs(ts_us - ts_cam_us)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_ref = (source, pc_idx)
            if dt <= time_window_us:
                pc = source.get_pc(pc_idx)
                pc_world = pc.transform("world", pc.reference_frame_timestamp_us, pose_graph)
                xyz_w = np.asarray(pc_world.xyz, dtype=np.float64)
                xyz_wg = (R_wg @ xyz_w.T + t_wg).T
                chunks.append(xyz_wg)

    if chunks:
        return np.concatenate(chunks, axis=0)

    # Fallback: single nearest sweep (window too tight / camera outside LiDAR span).
    if best_ref is not None:
        source, pc_idx = best_ref
        pc = source.get_pc(pc_idx)
        pc_world = pc.transform("world", pc.reference_frame_timestamp_us, pose_graph)
        xyz_w = np.asarray(pc_world.xyz, dtype=np.float64)
        return (R_wg @ xyz_w.T + t_wg).T

    return np.zeros((0, 3), dtype=np.float64)


def _project_and_depth(
    xyz_world: np.ndarray,  # [N, 3] world-global
    c2w: np.ndarray,  # [4, 4] OpenCV camera-to-world
    params_dict: dict,
    model_type_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Project world points to image plane and compute ray-depth.

    Dispatches by ``model_type_name``. Returns
    ``(uv[N,2], ray_depth[N], visible[N], H, W)``. ``visible`` already encodes
    the in-FOV / in-bounds / in-front test for the chosen model.

    Ray-depth = ‖cam_pts‖ with ``w2c = inv(c2w)`` (NCore c2w is OpenCV, so no
    extra flip). This matches the tracer.pred_dist semantics consumed by the
    depth loss.
    """
    H = int(params_dict["resolution"][1])
    W = int(params_dict["resolution"][0])

    # Camera-frame points for ray-depth (shared across all models; OpenCV c2w).
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
    pts_h = np.concatenate([xyz_world, np.ones((len(xyz_world), 1), dtype=np.float64)], axis=1)
    cam_pts = (w2c @ pts_h.T).T[:, :3]
    ray_depth = ray_depth_from_cam_pts(cam_pts)

    if model_type_name == "FThetaCameraModelParameters":
        from threedgrut_playground.utils.ftheta_projector import FthetaForwardProjector

        ftheta_dict = {
            "resolution": params_dict["resolution"],
            "principal_point": params_dict["principal_point"],
            "angle_to_pixeldist_poly": np.asarray(params_dict["angle_to_pixeldist_poly"]),
            "max_angle": params_dict["max_angle"],
        }
        # NCore c2w is already OpenCV (+Y down, +Z forward) → identity flip.
        proj = FthetaForwardProjector(ftheta_dict, world_to_camera_flip=np.eye(4))
        uv, visible = proj.project_points(xyz_world, c2w)
        return uv, ray_depth, np.asarray(visible, dtype=bool), H, W

    if model_type_name == "OpenCVPinholeCameraModelParameters":
        pp = np.asarray(params_dict["principal_point"], dtype=np.float64)
        fl = np.asarray(params_dict["focal_length"], dtype=np.float64)
        intrinsics = {"fx": float(fl[0]), "fy": float(fl[1]), "cx": float(pp[0]), "cy": float(pp[1])}
        # project_pinhole ignores radial/tangential distortion (acceptable for a
        # sparse depth supervision signal; the 5 standard NCore cams are FTheta).
        uv, valid = project_pinhole(cam_pts, intrinsics, (H, W))
        return uv, ray_depth, valid, H, W

    raise NotImplementedError(
        f"_project_and_depth: model_type_name={model_type_name!r} not supported "
        f"(only FTheta + OpenCV pinhole implemented for the depth dump)."
    )


def dump_clip(
    manifest_path: Path,
    camera_ids: list[str],
    out_root: Path,
    max_depth: float = 80.0,
    time_window_us: int = 60000,
    max_frames: int | None = None,
) -> None:
    """Iterate every frame × every camera; write one npz per (camera, frame).

    Per (camera, camera-frame): aggregate world-frame LiDAR points within
    ``±time_window_us`` of the camera END timestamp, project them onto the
    image plane (FTheta / pinhole), scatter the per-point ray-depth into a
    sparse ``[H, W]`` float32 map (nearest-wins), clamp depths > ``max_depth``
    to 0 (no-hit), and write ``<out_root>/<camera_id>/<ts_end_us>.npz`` with a
    single ``"depth"`` key (the LidarDepthAuxReader contract).

    The frame key is the **END** timestamp (``FrameTimepoint.END``) so the
    dumped maps align 1:1 with the timestamp the dataset uses to read aux
    masks / depth in ``NCoreDataset.__getitem__`` (verified: sseg aux keys are
    END timestamps).

    Imports the NCore SDK lazily so the projection-core unit tests (which run
    on Mac without NCore) can import this module.
    """
    try:
        import ncore.data  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "scripts/dump_lidar_depth_map.py dump_clip needs the ncore SDK. "
            "Run from A800 (conda env 3dgrut) or a Mac venv with ncore installed."
        ) from e

    from threedgrut.datasets.datasetNcore import NCoreDataset

    dataset = NCoreDataset(
        datapath=str(manifest_path),
        device="cpu",
        split="train",
        camera_ids=list(camera_ids),
        downsample=1.0,
        load_aux_masks=False,
    )
    sid = dataset.sequence_id
    out_root = Path(out_root)

    logger.info(
        "dump_clip: seq=%s cameras=%s time_window=±%dus max_depth=%.1fm out=%s",
        sid,
        list(camera_ids),
        time_window_us,
        max_depth,
        out_root,
    )

    summary: dict[str, tuple[int, float]] = {}  # camera_id -> (n_frames, mean_valid_px)

    for camera_id in camera_ids:
        camera_sensor = dataset.sequence_camera_sensors[sid][camera_id]
        camera_model = dataset.sequence_camera_models[sid][camera_id]
        W = int(camera_model.resolution[0].item())
        H = int(camera_model.resolution[1].item())
        res = dataset._get_camera_model_parameters_for_resolution(camera_id, W, H)
        if res is None:
            logger.warning("dump_clip: camera %s has no intrinsics; skipping.", camera_id)
            continue
        params_dict, model_type_name = res
        if model_type_name == "OpenCVFisheyeCameraModelParameters":
            logger.warning(
                "dump_clip: camera %s is OpenCV fisheye; not implemented — skipping.",
                camera_id,
            )
            continue

        cam_out_dir = out_root / camera_id
        cam_out_dir.mkdir(parents=True, exist_ok=True)

        n_frames = int(np.asarray(camera_sensor.frames_timestamps_us).shape[0])
        if max_frames is not None:
            n_frames = min(n_frames, int(max_frames))

        valid_px_total = 0
        for frame_idx in range(n_frames):
            ts_end_us = int(camera_sensor.frames_timestamps_us[frame_idx, ncore.data.FrameTimepoint.END])
            xyz_world = _accumulate_lidar_world_points(dataset, ts_end_us, time_window_us)
            if len(xyz_world) == 0:
                dmap = np.zeros((H, W), dtype=np.float32)
            else:
                c2w = dataset._get_start_end_poses_world_global(camera_sensor, frame_idx)[0]
                uv, ray_depth, visible, Hh, Ww = _project_and_depth(xyz_world, c2w, params_dict, model_type_name)
                # Clamp far returns to no-hit (depth > max_depth → drop).
                visible = visible & (ray_depth <= max_depth)
                dmap = scatter_depth_map(uv, ray_depth, visible, Hh, Ww)

            n_valid = int(np.count_nonzero(dmap))
            valid_px_total += n_valid
            np.savez_compressed(cam_out_dir / f"{ts_end_us}.npz", depth=dmap)

            if frame_idx % 50 == 0 or frame_idx == n_frames - 1:
                logger.info(
                    "  [%s] frame %d/%d ts=%d valid_px=%d (%.3f%%)",
                    camera_id,
                    frame_idx + 1,
                    n_frames,
                    ts_end_us,
                    n_valid,
                    100.0 * n_valid / (H * W),
                )

        mean_valid = valid_px_total / max(n_frames, 1)
        summary[camera_id] = (n_frames, mean_valid)
        logger.info(
            "dump_clip: camera %s done — %d frames, mean valid px/frame=%.0f (%.3f%%)",
            camera_id,
            n_frames,
            mean_valid,
            100.0 * mean_valid / (H * W),
        )

    logger.info("=== dump_clip summary (seq=%s) ===", sid)
    for cam, (nf, mv) in summary.items():
        logger.info("  %s: %d frames, mean valid px/frame=%.0f", cam, nf, mv)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--camera-ids", nargs="+", required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--max-depth", type=float, default=80.0)
    p.add_argument(
        "--time-window-us",
        type=int,
        default=60000,
        help="±window (microseconds) around each camera frame to accumulate "
        "LiDAR sweeps (default 60000 = ±60ms). Falls back to single "
        "nearest sweep if none in window.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap frames per camera (smoke / sanity-check). Default: all.",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    dump_clip(
        args.manifest,
        args.camera_ids,
        args.out_root,
        max_depth=args.max_depth,
        time_window_us=args.time_window_us,
        max_frames=args.max_frames,
    )
