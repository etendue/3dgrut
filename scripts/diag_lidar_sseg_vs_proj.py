# SPDX-License-Identifier: Apache-2.0
"""A1 step-0 — reconcile lidar-sseg/camvis aux stats against geometric projection.

Motivation (v5 task A1): the E5.0 diagnosis said "92.9% of lidar points are
ignore because the 3 forward cameras can't see them", yet the per-camera
lidar→image projection previews (thinkpad inc_b6a9ed61_proj) show the road
densely covered by points in those same cameras. Before spending hours
regenerating aux for 6 cameras, this script decides between:

  A) coverage problem  — ignore faithfully mirrors camvis==0, visible points
     are labeled correctly → just add the side cameras (original A1 plan);
  B) aux generation bug — points visible in cameras still labeled ignore
     (P(ignore|camvis>0) high) or points landing on road *pixels* labeled
     ignore/vegetation → fix aux generation first, then regenerate.

Stages:
  hist  read aux.lidar-sseg + aux.lidar-camvis itars only (no dataset):
        label histogram, camvis value semantics, P(ignore | camvis).
  proj  sample N lidar sweeps, project points into each camera with the
        repo's own projection (scripts/dump_lidar_depth_map._project_and_depth),
        cross-tab in-image visibility × lidar-sseg label × camvis, and — for
        cameras that already have camera-sseg aux — the label histogram of
        points landing on road *pixels* (the direct bug discriminator).

Runs on inceptio (needs ncore SDK). Read-only; writes only the report JSON.

CLI:
    python scripts/diag_lidar_sseg_vs_proj.py \
        --manifest ~/work/data/inc_b6a9ed61_20s/<clip>/<clip>.json \
        --stage all \
        --cameras camera_front_wide_120fov camera_cross_left_120fov \
                  camera_cross_right_120fov camera_left_wide_90fov \
                  camera_right_wide_90fov camera_back_rear_wide_90fov \
        --sseg-cameras camera_front_wide_120fov camera_cross_left_120fov \
                       camera_cross_right_120fov \
        --n-sweeps 3 --out /tmp/diag_lidar_sseg.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling script import

from threedgrut.datasets.aux_readers import (  # noqa: E402
    LidarSsegAuxReader,
    SsegAuxReader,
    _open_itar_zarr,
    discover_aux_path,
)

logger = logging.getLogger("diag_lidar_sseg")

IGNORE = 255
ROAD_CLASS_IDS = (0, 1)  # road, sidewalk — mirrors threedgrut/datasets/ncore_semantic.py


# ---------------------------------------------------------------------------
# camvis access — layout from scripts/merge_lidar_aux.py:
#   /aux/lidar_camera_visibility/<lidar_id>/<ts_us> : (N_pts, 1) uint8
# ---------------------------------------------------------------------------

def _open_camvis_group(clip_dir: Path):
    """Return (group, lidar_id) for the camvis component, or (None, None)."""
    path = discover_aux_path(clip_dir, "lidar-camvis")
    if path is None:
        return None, None
    root = _open_itar_zarr(path)
    aux = root["aux"]
    comp_names = [k for k in aux.group_keys()]
    # Expect exactly one visibility-ish component; be tolerant of naming.
    vis_names = [k for k in comp_names if "vis" in k] or comp_names
    grp = aux[vis_names[0]]
    lidar_ids = list(grp.group_keys())
    return grp[lidar_ids[0]], lidar_ids[0]


def _lidar_sseg_group_meta(clip_dir: Path):
    """Return (lidar_id, stuff_classes) from the lidar-sseg store attrs."""
    path = discover_aux_path(clip_dir, "lidar-sseg")
    root = _open_itar_zarr(path)
    grp = root["aux/lidar_semantic_segmentation"]
    lidar_id = list(grp.group_keys())[0]
    attrs = dict(grp[lidar_id].attrs)
    return lidar_id, list(attrs.get("stuff_classes", []))


def _hist_named(counts: np.ndarray, classes: list, top: int = 8) -> list:
    total = int(counts.sum())
    out = []
    for cid in np.argsort(-counts):
        if counts[cid] == 0 or len(out) >= top:
            break
        name = (
            "ignore" if cid == IGNORE
            else (classes[cid] if cid < len(classes) else f"class_{cid}")
        )
        out.append({
            "class_id": int(cid), "name": str(name),
            "count": int(counts[cid]),
            "pct": round(100.0 * counts[cid] / max(total, 1), 3),
        })
    return out


# ---------------------------------------------------------------------------
# stage hist
# ---------------------------------------------------------------------------

def stage_hist(clip_dir: Path) -> dict:
    lidar_id, classes = _lidar_sseg_group_meta(clip_dir)
    sseg_reader = LidarSsegAuxReader(discover_aux_path(clip_dir, "lidar-sseg"))
    camvis_grp, camvis_lidar_id = _open_camvis_group(clip_dir)

    sseg_reader._ensure_open()
    label_grp = sseg_reader._lidar_group(lidar_id)
    ts_keys = sorted(label_grp.array_keys(), key=int)

    label_counts = np.zeros(256, dtype=np.int64)
    camvis_counter: Counter = Counter()
    # Joint counts for the verdict quantities.
    n_ign_vis0 = n_vis0 = n_ign_vispos = n_vispos = 0
    n_len_mismatch = n_missing_camvis = 0

    for key in ts_keys:
        labels = sseg_reader.read(lidar_id, int(key))
        label_counts += np.bincount(labels, minlength=256)

        if camvis_grp is None or key not in camvis_grp:
            n_missing_camvis += 1
            continue
        camvis = np.asarray(camvis_grp[key][...]).ravel()
        vals, cnts = np.unique(camvis, return_counts=True)
        for v, c in zip(vals.tolist(), cnts.tolist()):
            camvis_counter[int(v)] += int(c)
        if camvis.shape[0] != labels.shape[0]:
            n_len_mismatch += 1
            continue
        ign = labels == IGNORE
        vis0 = camvis == 0
        n_vis0 += int(vis0.sum())
        n_ign_vis0 += int((ign & vis0).sum())
        n_vispos += int((~vis0).sum())
        n_ign_vispos += int((ign & (~vis0)).sum())

    total_pts = int(label_counts.sum())
    road_pts = int(label_counts[list(ROAD_CLASS_IDS)].sum())
    report = {
        "lidar_id": lidar_id,
        "n_frames": len(ts_keys),
        "total_points": total_pts,
        "label_hist_top": _hist_named(label_counts, classes),
        "road_sidewalk_points": road_pts,
        "road_sidewalk_pct": round(100.0 * road_pts / max(total_pts, 1), 4),
        "ignore_pct": round(100.0 * label_counts[IGNORE] / max(total_pts, 1), 3),
        "camvis_value_hist": {
            str(k): int(v) for k, v in sorted(camvis_counter.items())
        },
        "camvis_frames_missing": n_missing_camvis,
        "camvis_len_mismatch_frames": n_len_mismatch,
        "P(ignore|camvis==0)": round(n_ign_vis0 / max(n_vis0, 1), 4),
        "P(ignore|camvis>0)": round(n_ign_vispos / max(n_vispos, 1), 4),
        "n_camvis0": n_vis0,
        "n_camvis_pos": n_vispos,
    }
    return report


# ---------------------------------------------------------------------------
# stage proj
# ---------------------------------------------------------------------------

def stage_proj(
    manifest: Path,
    cameras: list[str],
    sseg_cameras: list[str],
    n_sweeps: int,
) -> dict:
    import ncore.data  # noqa: F401  (fail fast when SDK missing)
    from threedgrut.datasets.datasetNcore import NCoreDataset

    from dump_lidar_depth_map import _project_and_depth  # sibling script

    clip_dir = manifest.parent
    lidar_id, classes = _lidar_sseg_group_meta(clip_dir)
    sseg_lidar_reader = LidarSsegAuxReader(discover_aux_path(clip_dir, "lidar-sseg"))
    camvis_grp, _ = _open_camvis_group(clip_dir)
    cam_sseg_reader = (
        SsegAuxReader(discover_aux_path(clip_dir, "sseg"))
        if discover_aux_path(clip_dir, "sseg") else None
    )

    dataset = NCoreDataset(
        datapath=str(manifest), device="cpu", split="train",
        camera_ids=list(cameras), downsample=1.0, load_aux_masks=False,
    )
    sid = dataset.sequence_id
    sources = dataset.sequence_point_clouds_sources[sid]
    source = sources[lidar_id]
    pose_graph = dataset.sequence_loaders[sid].pose_graph
    T_wg = np.asarray(dataset.T_world_to_world_global, dtype=np.float64)

    pc_ts = np.asarray(source.pc_timestamps_us, dtype=np.int64)
    sweep_idxs = sorted(set(
        int(i) for i in np.linspace(0, len(pc_ts) - 1, num=n_sweeps).round()
    ))

    per_sweep = []
    for pc_idx in sweep_idxs:
        ts_us = int(pc_ts[pc_idx])
        if not sseg_lidar_reader.has_frame(lidar_id, ts_us):
            per_sweep.append({"pc_idx": pc_idx, "ts_us": ts_us, "error": "no lidar-sseg frame"})
            continue
        labels = sseg_lidar_reader.read(lidar_id, ts_us)
        pc = source.get_pc(pc_idx)
        pc_world = pc.transform("world", pc.reference_frame_timestamp_us, pose_graph)
        xyz_w = np.asarray(pc_world.xyz, dtype=np.float64)
        if xyz_w.shape[0] != labels.shape[0]:
            per_sweep.append({
                "pc_idx": pc_idx, "ts_us": ts_us,
                "error": f"len mismatch pts={xyz_w.shape[0]} labels={labels.shape[0]}",
            })
            continue
        xyz_wg = (T_wg[:3, :3] @ xyz_w.T + T_wg[:3, 3:4]).T
        camvis = None
        if camvis_grp is not None and str(ts_us) in camvis_grp:
            cv = np.asarray(camvis_grp[str(ts_us)][...]).ravel()
            camvis = cv if cv.shape[0] == labels.shape[0] else None

        sweep_rep = {
            "pc_idx": pc_idx, "ts_us": ts_us, "n_points": int(xyz_w.shape[0]),
            "label_hist_top": _hist_named(
                np.bincount(labels, minlength=256), classes, top=5),
            "cameras": {},
        }
        union_visible = np.zeros(labels.shape[0], dtype=bool)

        for cam in cameras:
            camera_sensor = dataset.sequence_camera_sensors[sid][cam]
            camera_model = dataset.sequence_camera_models[sid][cam]
            W = int(np.asarray(camera_model.resolution).ravel()[0])
            H = int(np.asarray(camera_model.resolution).ravel()[1])
            res = dataset._get_camera_model_parameters_for_resolution(cam, W, H)
            if res is None:
                sweep_rep["cameras"][cam] = {"error": "no intrinsics"}
                continue
            params_dict, model_type_name = res
            if model_type_name == "OpenCVFisheyeCameraModelParameters":
                sweep_rep["cameras"][cam] = {"error": "fisheye not supported"}
                continue

            end_ts = np.asarray(
                camera_sensor.frames_timestamps_us[:, ncore.data.FrameTimepoint.END],
                dtype=np.int64,
            )
            frame_idx = int(np.argmin(np.abs(end_ts - ts_us)))
            dt_ms = float(end_ts[frame_idx] - ts_us) / 1000.0
            c2w = dataset._get_start_end_poses_world_global(camera_sensor, frame_idx)[0]

            uv, ray_depth, visible, Hh, Ww = _project_and_depth(
                xyz_wg, c2w, params_dict, model_type_name,
            )
            union_visible |= visible
            vis_labels = labels[visible]
            cam_rep = {
                "frame_idx": frame_idx, "dt_ms": round(dt_ms, 1),
                "model": model_type_name.replace("CameraModelParameters", ""),
                "n_visible": int(visible.sum()),
                "visible_pct": round(100.0 * visible.mean(), 2),
                "visible_label_hist_top": _hist_named(
                    np.bincount(vis_labels, minlength=256), classes, top=5),
                "visible_road_pts": int(np.isin(vis_labels, ROAD_CLASS_IDS).sum()),
                "visible_ignore_pct": round(
                    100.0 * float((vis_labels == IGNORE).mean()) if len(vis_labels) else 0.0, 2),
            }
            if camvis is not None:
                cam_rep["visible_camvis_pos_pct"] = round(
                    100.0 * float((camvis[visible] > 0).mean()) if visible.any() else 0.0, 2)

            # Road-pixel cross-check: sample the camera sseg at projected uv.
            if cam_sseg_reader is not None and cam in sseg_cameras:
                sseg_ts = int(end_ts[frame_idx])
                try:
                    sseg_img = cam_sseg_reader.read(cam, sseg_ts)
                except KeyError:
                    try:  # some aux versions key by START timestamp
                        sseg_img = cam_sseg_reader.read(cam, int(
                            camera_sensor.frames_timestamps_us[frame_idx, ncore.data.FrameTimepoint.START]))
                    except KeyError:
                        sseg_img = None
                if sseg_img is not None:
                    Hs, Ws = sseg_img.shape[:2]
                    ui = np.floor(uv[:, 0] * (Ws / Ww)).astype(np.int64)
                    vi = np.floor(uv[:, 1] * (Hs / Hh)).astype(np.int64)
                    inb = visible & (ui >= 0) & (ui < Ws) & (vi >= 0) & (vi < Hs)
                    on_road_px = np.zeros_like(inb)
                    on_road_px[inb] = np.isin(sseg_img[vi[inb], ui[inb]], ROAD_CLASS_IDS)
                    road_px_labels = labels[on_road_px]
                    cam_rep["pts_on_road_pixels"] = int(on_road_px.sum())
                    cam_rep["road_px_label_hist_top"] = _hist_named(
                        np.bincount(road_px_labels, minlength=256), classes, top=5)
                    n_rp = max(len(road_px_labels), 1)
                    cam_rep["road_px_labeled_road_pct"] = round(
                        100.0 * float(np.isin(road_px_labels, ROAD_CLASS_IDS).sum()) / n_rp, 2)
                    cam_rep["road_px_labeled_ignore_pct"] = round(
                        100.0 * float((road_px_labels == IGNORE).sum()) / n_rp, 2)
            sweep_rep["cameras"][cam] = cam_rep

        # Sweep-level: geometric union visibility vs labels/camvis.
        ign = labels == IGNORE
        sweep_rep["geom_union_visible_pct"] = round(100.0 * union_visible.mean(), 2)
        sweep_rep["P(ignore|geom_visible)"] = round(
            float(ign[union_visible].mean()) if union_visible.any() else 0.0, 4)
        sweep_rep["P(ignore|geom_invisible)"] = round(
            float(ign[~union_visible].mean()) if (~union_visible).any() else 0.0, 4)
        if camvis is not None:
            sweep_rep["camvis_pos_pct"] = round(100.0 * float((camvis > 0).mean()), 2)
            sweep_rep["geom_visible_but_camvis0_pct"] = round(
                100.0 * float((union_visible & (camvis == 0)).mean()), 2)
        per_sweep.append(sweep_rep)

    return {"sweeps": per_sweep, "cameras": cameras, "sseg_cameras": sseg_cameras}


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--stage", choices=["hist", "proj", "all"], default="all")
    p.add_argument("--cameras", nargs="+", default=[
        "camera_front_wide_120fov", "camera_cross_left_120fov",
        "camera_cross_right_120fov", "camera_left_wide_90fov",
        "camera_right_wide_90fov", "camera_back_rear_wide_90fov",
    ])
    p.add_argument("--sseg-cameras", nargs="+", default=[
        "camera_front_wide_120fov", "camera_cross_left_120fov",
        "camera_cross_right_120fov",
    ])
    p.add_argument("--n-sweeps", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("/tmp/diag_lidar_sseg.json"))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    clip_dir = args.manifest.parent
    report: dict = {"manifest": str(args.manifest)}
    if args.stage in ("hist", "all"):
        logger.info("=== stage hist ===")
        report["hist"] = stage_hist(clip_dir)
        logger.info(json.dumps(report["hist"], indent=2, ensure_ascii=False))
    if args.stage in ("proj", "all"):
        logger.info("=== stage proj ===")
        report["proj"] = stage_proj(
            args.manifest, args.cameras, args.sseg_cameras, args.n_sweeps,
        )
        logger.info(json.dumps(report["proj"], indent=2, ensure_ascii=False))

    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("report written → %s", args.out)


if __name__ == "__main__":
    main()
