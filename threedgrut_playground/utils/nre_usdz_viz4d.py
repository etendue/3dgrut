# SPDX-License-Identifier: Apache-2.0
"""E2.8 — USDZ (NRE training-checkpoint flavour) → renderable 3dgrut2 ckpt dict
with dynamic_rigids 4D wiring + viz_4d, ready for per-track AH replacement.

Why this module exists
----------------------
``nre_usdz_loader.build_native_ckpt`` (E2.7, this branch) translates the NRE
``checkpoint.ckpt`` gaussians into a 3dgrut2 ckpt dict, but only wires the
*static* layers (background/road) by default and leaves dynamic_rigids carrying
a raw ``_nre_cuboid_ids`` (no ``track_ids`` buffer, no ``viz_4d`` block). The
end-to-end dynamic wiring (cuboid_id → sorted-tid remap + per-track pose
resample onto the camera timeline + viz_4d assembly) lived only in
``fervent-knuth-d25fe9``'s ``nurec_usdz_loader.py`` — and that one reads the
``volume.nurec`` USDZ flavour, NOT our ``checkpoint.ckpt`` flavour.

This module bridges the gap: it reuses this branch's ``build_native_ckpt``
(correct gaussian source for our USDZ) and ports fervent-knuth's *proven*
pure-python rig/track/viz4d wiring (``rig_trajectories.json`` carries the camera
timeline + ftheta intrinsics + ``world_to_nre``).

**Coordinate frame (E2.7 golden rule + 2026-06-17 实测，E2.7-B/C 反复踩坑)**:
bg/road gaussians live in the NRE near-origin frame (median ~5m) and MUST be
shifted by ``-world_to_nre.translation`` (≈ +38m x for 9ae151dc) into the NCore
world frame; ego (rig_trajectories c2w[0]=[2.15,0.03,1.44], 实测本就 NCore
world) + track poses (NCore world) + dynamic_rigids (object-local, placed by
track_pose at render) are NOT shifted. (The ``omni:nurec:offset`` from
volume.usda — a different transform — IS a no-op; do not confuse the two.)
Skipping the bg/road shift desyncs the static scene from cars/cameras by ~38m.
The result is a ckpt dict that:

* loads through ``engine.load_3dgrt_object`` (LayeredGaussians route),
* renders dynamic vehicles in place along the timeline, and
* exposes ``recon = {tid: (label_class, dims)}`` + ``name_to_id`` so E2.8's
  ``replace_all_vehicle_tracks`` can swap every vehicle track for an AH asset.

The pure parsers (``parse_rig_trajectories`` / ``build_viz4d_dict`` /
``build_ftheta_dict``) carry no cuda/hydra import (Mac/synthetic testable);
``convert_usdz_to_ckpt_with_tracks`` needs the 3dgrut env + GPU (build_native_ckpt
builds a reference MoG on cuda).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from threedgrut_playground.utils.nre_usdz_loader import (
    build_native_ckpt,
    extract_nre_checkpoint,
    parse_sequence_tracks,
    parse_volume_usda_track_order,
    read_usdz_member_bytes,
    resample_track_to_timeline,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Ported pure-python rig / viz4d helpers (fervent-knuth-d25fe9, proven).
# --------------------------------------------------------------------------- #
def short_cam_id(cam_key: str) -> str:
    """'camera_front_wide_120fov@clipgt-<uuid>' → 'camera_front_wide_120fov'."""
    return cam_key.split("@", 1)[0]


def build_ftheta_dict(camera_model_params: dict) -> dict:
    """rig_trajectories camera_model.parameters → viewer ftheta_dict.

    Key set/dtypes mirror viser_gui_4d._load_multi_cam_poses so the engine
    treats USDZ cameras exactly like NCore ones. 6-coeff polys pass through
    unchanged (ftheta Horner eval is length-agnostic).
    """
    p = camera_model_params
    return {
        "resolution": np.asarray(p["resolution"], dtype=np.int64),
        "shutter_type": str(p["shutter_type"]),
        "principal_point": np.asarray(p["principal_point"], dtype=np.float32),
        "reference_poly": str(p["reference_poly"]),
        "pixeldist_to_angle_poly": np.asarray(p["pixeldist_to_angle_poly"], dtype=np.float32),
        "angle_to_pixeldist_poly": np.asarray(p["angle_to_pixeldist_poly"], dtype=np.float32),
        "max_angle": float(p["max_angle"]),
        "linear_cde": np.asarray(p["linear_cde"], dtype=np.float32),
    }


@dataclass
class RigInfo:
    """Parsed rig_trajectories.json: per-camera pose tables + rig trajectory."""
    sequence_id: str
    cams: dict
    T_rig_worlds: np.ndarray
    T_rig_world_timestamps_us: np.ndarray
    world_to_nre: Optional[np.ndarray] = None


def parse_rig_trajectories(rt_json: dict, *, invert_sensor: bool = False) -> RigInfo:
    """rig_trajectories.json → RigInfo (cams keyed by short logical name).

    Handles the measured ``(F, 2, 4, 4)`` exposure-start/end rig-pose pairs
    (takes END to match END timestamps) and flat ``(F, 4, 4)`` layouts.
    """
    traj = rt_json["rig_trajectories"][0]
    sequence_id = str(traj.get("sequence_id", "unknown"))
    frame_poses = traj.get("cameras_frame_T_rig_worlds", {}) or {}
    frame_ts = traj.get("cameras_frame_timestamps_us", {}) or {}

    cams: dict = {}
    for cam_key, calib in (rt_json.get("camera_calibrations", {}) or {}).items():
        rig_mats = frame_poses.get(cam_key)
        ts_windows = frame_ts.get(cam_key)
        if rig_mats is None or ts_windows is None:
            logger.info("camera %s has no per-frame poses/timestamps — skipped", cam_key)
            continue
        sensor = np.asarray(calib["T_sensor_rig"], dtype=np.float64)
        if invert_sensor:
            sensor = np.linalg.inv(sensor)
        rig_mats = np.asarray(rig_mats, dtype=np.float64)
        if rig_mats.ndim == 4 and rig_mats.shape[1:] == (2, 4, 4):
            rig_mats = rig_mats[:, 1]  # exposure END
        else:
            rig_mats = rig_mats.reshape(-1, 4, 4)
        c2w = (rig_mats @ sensor).astype(np.float32)
        ts = np.asarray(ts_windows, dtype=np.int64).reshape(len(c2w), -1)[:, -1]  # END

        cam_model = calib.get("camera_model", {}) or {}
        params = cam_model.get("parameters")
        ftheta = None
        resolution = None
        fov_y_rad = 1.5708
        if params:
            resolution = (int(params["resolution"][0]), int(params["resolution"][1]))
            if str(cam_model.get("type", "")).lower() == "ftheta":
                ftheta = build_ftheta_dict(params)
                fov_y_rad = 2.0 * float(params["max_angle"])
        cid = short_cam_id(str(calib.get("logical_sensor_name") or cam_key))
        cams[cid] = {
            "c2w": c2w,
            "timestamps_us": ts,
            "ftheta_dict": ftheta,
            "resolution": resolution,
            "fov_y_rad": fov_y_rad,
        }

    world_to_nre = None
    w2n = rt_json.get("world_to_nre")
    if isinstance(w2n, dict) and "matrix" in w2n:
        world_to_nre = np.asarray(w2n["matrix"], dtype=np.float64)

    return RigInfo(
        sequence_id=sequence_id,
        cams=cams,
        T_rig_worlds=np.asarray(traj.get("T_rig_worlds", []), dtype=np.float64).reshape(-1, 4, 4),
        T_rig_world_timestamps_us=np.asarray(traj.get("T_rig_world_timestamps_us", []), dtype=np.int64),
        world_to_nre=world_to_nre,
    )


def resolve_primary_cam(rig: RigInfo, primary_cam: str) -> str:
    """primary_cam if present, else first camera (deterministic by sorted key)."""
    if primary_cam in rig.cams:
        return primary_cam
    if not rig.cams:
        raise ValueError("rig has no usable cameras")
    fallback = sorted(rig.cams.keys())[0]
    logger.warning("primary cam %r absent; falling back to %r", primary_cam, fallback)
    return fallback


def build_viz4d_dict(
    rig: RigInfo,
    tracks: list,
    *,
    primary_cam: str,
    sequence_id: Optional[str] = None,
) -> dict:
    """Assemble a ckpt['viz_4d']-shaped dict (schema_version 2).

    Shared timeline = primary camera's per-frame exposure-END timestamps, so
    ego poses, the time slider, and every resampled cuboid track share one clock.
    """
    primary_cam = resolve_primary_cam(rig, primary_cam)
    cam = rig.cams[primary_cam]
    timeline = np.asarray(cam["timestamps_us"], dtype=np.int64)

    tracks_out: dict = {}
    for tr in tracks:
        poses, frame_info = resample_track_to_timeline(tr.poses7, tr.ts_us, timeline)
        # torch tensors (not numpy): engine.load_3dgrt_object / render.py auto-hook
        # call model.populate_tracks(viz_4d.tracks) → _populate_tracks_impl does
        # poses.to(float32) / frame_info.to(bool) — numpy has no .to(). Matches the
        # trainer-written viz_4d format. FourDMetadata.from_ckpt._to_np converts
        # them back to numpy for the viewer overlays.
        tracks_out[tr.tid] = {
            "poses": torch.as_tensor(poses, dtype=torch.float32),
            "size": torch.as_tensor(np.asarray(tr.dims), dtype=torch.float32),
            "frame_info": torch.as_tensor(frame_info, dtype=torch.bool),
            "class": str(tr.label_class),
        }

    W, H = cam["resolution"] if cam["resolution"] else (1920, 1080)
    return {
        "schema_version": 2,
        "sequence_id": sequence_id or rig.sequence_id,
        "ego": {
            "poses_c2w": np.asarray(cam["c2w"], dtype=np.float32),
            "frame_timestamps_us": timeline,
            "primary_camera_id": primary_cam,
            "primary_camera_fov_y_rad": float(cam["fov_y_rad"]),
            "primary_camera_aspect": float(W) / float(H),
            "primary_camera_intrinsics_FTheta": cam["ftheta_dict"],
            "primary_camera_resolution": (W, H),
        },
        "tracks": tracks_out,
        "tracks_camera_timestamps_us": timeline,
        "lidar": {},
        "viewer_defaults": {
            "initial_c2w": np.asarray(cam["c2w"][0], dtype=np.float32),
            "t_us_first": int(timeline[0]),
            "t_us_last": int(timeline[-1]),
        },
    }


# --------------------------------------------------------------------------- #
# track_ids remap (cuboid_id → sorted-tid slot) — pure, numpy.
# --------------------------------------------------------------------------- #
def cuboid_ids_to_track_ids(cuboid_ids: np.ndarray, track_order: list) -> tuple[np.ndarray, list]:
    """Per-gaussian cuboid index → sorted-tid slot (populate_tracks convention).

    Returns ``(track_ids (N,) int64, sorted_tids)``. ``sorted_tids[slot]`` is the
    tid string; ``name_to_id[tid] = sorted_tids.index(tid)``.
    """
    sorted_tids = sorted(set(track_order))
    cid_to_sorted = np.array([sorted_tids.index(t) for t in track_order], dtype=np.int64)
    cuboid_ids = np.asarray(cuboid_ids, dtype=np.int64)
    if cuboid_ids.size and int(cuboid_ids.max()) >= len(track_order):
        raise IndexError(
            f"gaussian_cuboid_ids max {int(cuboid_ids.max())} >= track_order len "
            f"{len(track_order)} — inconsistent USDZ container."
        )
    return cid_to_sorted[cuboid_ids], sorted_tids


# 静态层（NRE 帧），需 NRE→world translate。dynamic_rigids 是 object-local（render
# 时由 track_pose 放置），绝不平移——否则 double-translate 把所有车堆到 +38m。
_STATIC_NRE_LAYERS = ("background", "road")


def apply_nre_to_world_translate(
    gaussians_nodes: dict,
    world_to_nre: Optional[np.ndarray],
    *,
    static_layers: tuple[str, ...] = _STATIC_NRE_LAYERS,
) -> np.ndarray:
    """把 NRE-帧静态层 gaussians 平移到 NCore world 帧（E2.7 golden 规则）。

    translate = ``-world_to_nre.matrix[:3,3]``（9ae151dc ≈ +38m x）。只动
    ``static_layers``（background/road）的 positions；**dynamic_rigids 跳过**
    （object-local，render_pass 用 track_pose 放置，平移会 double-translate）；
    track poses + ego（rig_trajectories，实测本就 NCore world）也不在此处理。
    返回实际应用的 translate (3,)（world_to_nre 缺失 → 零位移 no-op）。
    """
    if world_to_nre is None:
        logger.warning("apply_nre_to_world_translate: world_to_nre absent → no-op")
        return np.zeros(3, dtype=np.float32)
    T = np.asarray(world_to_nre, dtype=np.float64).reshape(4, 4)
    if not np.allclose(T[:3, :3], np.eye(3), atol=1e-4):
        logger.warning(
            "world_to_nre rotation NOT identity (R=%s); translate-only align "
            "insufficient — visual skew possible (E2.7 同警告)", T[:3, :3].tolist()
        )
    translate = (-T[:3, 3]).astype(np.float32)
    tt = torch.as_tensor(translate, dtype=torch.float32)
    for layer in static_layers:
        node = gaussians_nodes.get(layer)
        if node is None or "positions" not in node:
            continue
        p = node["positions"]
        with torch.no_grad():
            shifted = p.detach().cpu() + tt
        node["positions"] = (
            torch.nn.Parameter(shifted.contiguous(), requires_grad=False)
            if isinstance(p, torch.nn.Parameter) else shifted.contiguous()
        )
        logger.info("nre→world: shifted layer %r by %s", layer, translate.tolist())
    return translate


@dataclass
class UsdzScene:
    """Output of :func:`convert_usdz_to_ckpt_with_tracks`."""
    ckpt: dict
    recon: dict = field(default_factory=dict)      # {tid: (label_class, (L,W,H))}
    name_to_id: dict = field(default_factory=dict)  # {tid: sorted-tid slot int}
    primary_cam: str = ""
    sequence_id: str = ""


# --------------------------------------------------------------------------- #
# Orchestrator (needs 3dgrut env + GPU via build_native_ckpt).
# --------------------------------------------------------------------------- #
def _track_order_from_usdz(usdz_path: str | Path) -> list:
    """Parse cuboid declaration order → tid list from the USDZ usda member."""
    last_err: Optional[Exception] = None
    for member in ("volume.usda", "sequence_tracks.usda"):
        try:
            text = read_usdz_member_bytes(usdz_path, member).decode("utf-8", errors="replace")
        except FileNotFoundError as e:
            last_err = e
            continue
        order = parse_volume_usda_track_order(text)
        if order:
            logger.info("track_order parsed from %s (%d entries)", member, len(order))
            return order
    raise ValueError(
        f"USDZ {usdz_path}: no volume.usda/sequence_tracks.usda with track prims "
        f"(last error: {last_err!r})"
    )


def convert_usdz_to_ckpt_with_tracks(
    usdz_path: str | Path,
    *,
    primary_cam: str = "camera_front_wide_120fov",
    layers: tuple[str, ...] = ("background", "road", "dynamic_rigids"),
    albedo_mode: str = "dc",
    clip_radius_m: float = 1500.0,
    clip_scale_m: float = 20.0,
) -> UsdzScene:
    """USDZ → renderable native ckpt dict (CPU) + recon + name_to_id.

    The returned ckpt's tensors are on CPU (frozen surgery + torch.save are
    CPU-side); ``build_native_ckpt`` builds them on cuda, so we move to CPU.
    ``dynamic_rigids`` gains a ``track_ids`` buffer and the ckpt gains a
    ``viz_4d`` block (ego + per-track poses on the shared camera timeline).
    ``recon`` only contains tids whose gaussians are actually present.
    """
    usdz_path = Path(usdz_path)
    nre = extract_nre_checkpoint(usdz_path)
    state_dict = nre["state_dict"]
    global_step = int(nre.get("global_step", 0))

    ckpt = build_native_ckpt(
        state_dict,
        layers=tuple(layers),
        experiment_name=usdz_path.stem,
        albedo_mode=albedo_mode,
        clip_radius_m=clip_radius_m,
        clip_scale_m=clip_scale_m,
        global_step=global_step,
    )
    ckpt = _ckpt_to_cpu(ckpt)

    gn = ckpt["model"]["gaussians_nodes"]
    recon: dict = {}
    name_to_id: dict = {}
    rig = parse_rig_trajectories(
        json.loads(read_usdz_member_bytes(usdz_path, "rig_trajectories.json"))
    )
    # 坐标对齐（E2.7 golden 规则 + 2026-06-17 实测）：bg/road gaussians 在 NRE
    # 近原点帧，需 +(-world_to_nre.translation)≈+38m 搬到 NCore world；ego
    # (rig_trajectories，实测 c2w[0]=[2.15,0.03,1.44] 本就 NCore world) + track
    # poses (NCore world) + dynamic_rigids (object-local) 均不动。漏此步会让静态
    # 场景与车/相机错位 ~38m（E2.7-B/C 反复踩坑）。
    applied_translate = apply_nre_to_world_translate(gn, rig.world_to_nre)
    resolved_cam = resolve_primary_cam(rig, primary_cam)
    tracks_raw = parse_sequence_tracks(
        json.loads(read_usdz_member_bytes(usdz_path, "sequence_tracks.json"))
    )
    ckpt["viz_4d"] = build_viz4d_dict(rig, tracks_raw, primary_cam=resolved_cam)

    if "dynamic_rigids" in gn:
        dyn = gn["dynamic_rigids"]
        cuboid_ids = dyn.pop("_nre_cuboid_ids", None)
        if cuboid_ids is None:
            raise ValueError(
                "dynamic_rigids node has no _nre_cuboid_ids — build_native_ckpt "
                "did not carry gaussian_cuboid_ids (cannot wire per-track replacement)."
            )
        cuboid_ids = (cuboid_ids.cpu().numpy() if torch.is_tensor(cuboid_ids)
                      else np.asarray(cuboid_ids))
        track_order = _track_order_from_usdz(usdz_path)
        track_ids, sorted_tids = cuboid_ids_to_track_ids(cuboid_ids, track_order)
        dyn["track_ids"] = torch.as_tensor(track_ids, dtype=torch.long)

        present = {int(s) for s in np.unique(track_ids).tolist()}
        tracks_meta = ckpt["viz_4d"]["tracks"]
        for tid in sorted_tids:
            slot = sorted_tids.index(tid)
            if slot not in present:
                continue
            meta = tracks_meta.get(tid)
            if meta is None:
                logger.warning("present tid %r missing from sequence_tracks — skipped", tid)
                continue
            dims = tuple(float(x) for x in np.asarray(meta["size"]).flatten()[:3])
            recon[tid] = (str(meta["class"]), dims)
            name_to_id[tid] = slot

    return UsdzScene(
        ckpt=ckpt,
        recon=recon,
        name_to_id=name_to_id,
        primary_cam=resolved_cam,
        sequence_id=rig.sequence_id,
    )


def _ckpt_to_cpu(ckpt: dict) -> dict:
    """Move every gaussians_nodes tensor/Parameter to CPU (frozen-surgery side)."""
    for node in ckpt["model"]["gaussians_nodes"].values():
        for k, v in list(node.items()):
            if torch.is_tensor(v):
                node[k] = (torch.nn.Parameter(v.detach().cpu(), requires_grad=False)
                           if isinstance(v, torch.nn.Parameter) else v.detach().cpu())
    return ckpt
