#!/usr/bin/env python3
"""V3-VIZ.1b — BEV stitched diagnostic for v2 LayeredGaussians checkpoints.

Builds a per-frame top-down mosaic by inverse-perspective-mapping the rig's
5/7 cameras' raw images onto the world ground plane at z = ego_z, then
overlays the same cuboid footprints + per-layer Gaussian centers as the
scatter-version diagnostic (``scripts/diagnose_layered_bev.py``). The IPM
backdrop gives the eye a visual anchor — lane markings, parked cars, road
texture — making it obvious whether a cuboid (or a layer's particles) is
correctly aligned with what the cameras actually see.

Requires the NCore SDK (image arrays + per-camera c2w + FTheta intrinsics).
Run on ThinkPad / a800-x2; rsync ``out_dir`` PNGs back to the laptop for review.

Usage:

    python scripts/diagnose_layered_bev_stitched.py \\
        --gs_object   /home/yusun/work/ckpts/B3_30k_20260525/ckpt_last.pt \\
        --dataset_path /path/to/manifest.json \\
        --out_dir     /tmp/b3_bev_stitched \\
        --frame_range 0:10 \\
        --xy_range_m 30 \\
        --res_mpp    0.10
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


def _parse_frame_range(spec: Optional[str], n_frames: int) -> list[int]:
    if spec is None or spec.strip() == "":
        return list(range(n_frames))
    if ":" in spec:
        lo_s, hi_s = spec.split(":", 1)
        lo = int(lo_s) if lo_s else 0
        hi = int(hi_s) if hi_s else n_frames
        return list(range(max(0, lo), min(n_frames, hi)))
    f = int(spec)
    return [f] if 0 <= f < n_frames else []


def _extract_ftheta_dict_from_model(camera_model) -> dict:
    """Pull the 8-key FTheta dict from a CameraModel.

    Mirrors threedgrut/viz/metadata.py:_detect_primary_camera so the BEV
    projector is fed exactly the same intrinsics the trainer used.
    """
    params = camera_model.get_parameters()
    return {
        "resolution": np.asarray(params.resolution, dtype=np.int64),
        "shutter_type": params.shutter_type.name,
        "principal_point": np.asarray(params.principal_point, dtype=np.float32),
        "reference_poly": params.reference_poly.name,
        "pixeldist_to_angle_poly": np.asarray(params.pixeldist_to_angle_poly, dtype=np.float32),
        "angle_to_pixeldist_poly": np.asarray(params.angle_to_pixeldist_poly, dtype=np.float32),
        "max_angle": float(params.max_angle),
        "linear_cde": np.asarray(params.linear_cde, dtype=np.float32),
    }


def _nearest_frame_index(
    sensor,
    target_ts_us: int,
    end_col: int,
) -> tuple[int, int]:
    """Return (camera_frame_index, actual_ts) of frame nearest to target_ts."""
    ts_all = np.asarray(sensor.frames_timestamps_us)[:, end_col].astype(np.int64)
    idx = int(np.argmin(np.abs(ts_all - target_ts_us)))
    return idx, int(ts_all[idx])


def diagnose(args: argparse.Namespace) -> int:
    import imageio.v3 as iio
    import ncore

    from threedgrut.datasets.datasetNcore import NCoreDataset
    from threedgrut_playground.utils.bev_renderer import (
        build_inputs_from_metadata,
        render_bev_frame,
    )
    from threedgrut_playground.utils.bev_stitcher import (
        BEVStitcher,
        CameraRig,
        default_azimuth,
    )
    from threedgrut_playground.utils.diag_ckpt import (
        dyn_local_to_world_at_frame,
        extract_dyn_track_ids,
        extract_layer_positions,
        load_ckpt_cpu,
    )
    from threedgrut_playground.utils.viz4d_metadata import FourDMetadata

    ckpt_path = Path(args.gs_object).expanduser().resolve()
    if not ckpt_path.exists():
        print(f"[bev-stitch] ERROR: ckpt not found: {ckpt_path}", file=sys.stderr)
        return 2

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not dataset_path.exists():
        print(f"[bev-stitch] ERROR: dataset manifest not found: {dataset_path}", file=sys.stderr)
        return 3

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ckpt (CPU-only, no torch.cuda) -------------------------------
    print(f"[bev-stitch] loading ckpt → {ckpt_path}", flush=True)
    ckpt = load_ckpt_cpu(ckpt_path)
    meta = FourDMetadata.from_ckpt(ckpt)
    if meta is None:
        print("[bev-stitch] ERROR: ckpt has no viz_4d block", file=sys.stderr)
        return 4

    layer_positions = extract_layer_positions(ckpt)
    static_layers = {n: p for n, p in layer_positions.items() if n != "dynamic_rigids"}
    dyn_local = layer_positions.get("dynamic_rigids")
    dyn_track_ids = extract_dyn_track_ids(ckpt)
    sorted_track_names = sorted(meta.tracks.keys())

    n_frames_ego = int(meta.ego_poses_c2w.shape[0])
    frames = _parse_frame_range(args.frame_range, n_frames_ego)
    if not frames:
        print(
            f"[bev-stitch] ERROR: no frames in range '{args.frame_range}' " f"(n_frames={n_frames_ego})",
            file=sys.stderr,
        )
        return 5

    # --- Load dataset (NCore SDK required) ----------------------------------
    print(f"[bev-stitch] loading NCoreDataset → {dataset_path}", flush=True)
    if args.camera_ids:
        cam_id_list = [c.strip() for c in args.camera_ids.split(",") if c.strip()]
    else:
        # 5 default wide-coverage cameras (skip tele 30fov which have narrow ground footprint).
        cam_id_list = [
            "camera_front_wide_120fov",
            "camera_cross_left_120fov",
            "camera_cross_right_120fov",
            "camera_rear_left_70fov",
            "camera_rear_right_70fov",
        ]
    # Minimal config: split='train', device='cpu', sample_full_image=True.
    dataset = NCoreDataset(
        datapath=str(dataset_path),
        split="train",
        device="cpu",
        sample_full_image=True,
        camera_ids=cam_id_list,
        load_aux_masks=False,
    )
    seq_id = dataset.sequence_id
    camera_ids = list(dataset.camera_ids)
    print(f"[bev-stitch] cameras: {camera_ids}", flush=True)

    end_col = int(ncore.data.FrameTimepoint.END)
    start_tp = ncore.data.FrameTimepoint.START  # NCore expects enum, not int

    # Build CameraRig list + BEVStitcher.
    rigs: list[CameraRig] = []
    sensors = {}
    for cid in camera_ids:
        cam_model = dataset.sequence_camera_models[seq_id][cid]
        cam_sensor = dataset.sequence_camera_sensors[seq_id][cid]
        try:
            ftheta_dict = _extract_ftheta_dict_from_model(cam_model)
        except Exception as e:
            print(f"[bev-stitch] WARN: cam {cid} not FTheta — skipping ({e})", flush=True)
            continue
        # Image HW from camera model resolution.
        W_img = int(cam_model.resolution[0].item())
        H_img = int(cam_model.resolution[1].item())
        rigs.append(
            CameraRig(
                camera_id=cid,
                ftheta_dict=ftheta_dict,
                azimuth_deg=default_azimuth(cid),
                image_hw=(H_img, W_img),
            )
        )
        sensors[cid] = cam_sensor

    if not rigs:
        print("[bev-stitch] ERROR: no usable cameras (no FTheta intrinsics found)", file=sys.stderr)
        return 6

    stitcher = BEVStitcher(
        rigs,
        bev_xy_range_m=args.xy_range_m,
        bev_resolution_m_per_px=args.res_mpp,
    )
    print(
        f"[bev-stitch] BEV {stitcher.bev_w}×{stitcher.bev_h} px "
        f"@ {args.res_mpp} m/px  ({stitcher.xy_range_m * 2:.0f}m square)",
        flush=True,
    )
    print(
        f"[bev-stitch] sequence={meta.sequence_id}  n_frames_ego={n_frames_ego}  " f"n_tracks={meta.n_tracks()}",
        flush=True,
    )
    for name, p in layer_positions.items():
        print(f"  layer {name:<22} particles={p.shape[0]:>10d}", flush=True)
    print(f"[bev-stitch] rendering {len(frames)} frames → {out_dir}", flush=True)

    # Apply NCore-world → world_global transform so c2w matches ckpt convention.
    T_w2wg = np.asarray(dataset.T_world_to_world_global, dtype=np.float64)

    manifest: list[dict] = []
    for fi in frames:
        ego_t_us = int(meta.ego_frame_timestamps_us[fi])
        ego_pose = np.asarray(meta.ego_poses_c2w[fi], dtype=np.float64)
        ego_xy = ego_pose[:2, 3].astype(np.float64)
        # ego_pose stores PRIMARY CAMERA position in world; cameras are mounted
        # ~1.5 m above ground. Subtract --ground_offset_m to put the BEV plane
        # on the actual road surface (where IPM is meaningful).
        ego_z = float(ego_pose[2, 3]) - float(args.ground_offset_m)

        cam_c2w: dict[str, np.ndarray] = {}
        cam_images: dict[str, np.ndarray] = {}
        for rig in rigs:
            cid = rig.camera_id
            sensor = sensors[cid]
            try:
                cf_idx, _ts_actual = _nearest_frame_index(sensor, ego_t_us, end_col)
                c2w_native = sensor.get_frames_T_source_target(
                    source_node=sensor.sensor_id,
                    target_node="world",
                    frame_indices=[cf_idx],
                    frame_timepoint=start_tp,
                )
                c2w_native = np.asarray(c2w_native, dtype=np.float64).reshape(-1, 4, 4)[0]
                # Transform to world_global to match ckpt convention.
                c2w_wg = T_w2wg @ c2w_native
                img = sensor.get_frame_image_array(cf_idx)
            except Exception as e:
                if args.verbose:
                    print(f"[bev-stitch]   cam {cid} frame {fi}: skipped ({e})", flush=True)
                continue
            cam_c2w[cid] = c2w_wg
            cam_images[cid] = np.asarray(img)

        bg_rgb, _coverage = stitcher.stitch_frame(
            cam_c2w,
            cam_images,
            ego_xy,
            ego_z,
        )
        extent = stitcher.world_xy_extent(ego_xy)

        # Build cuboid + Gaussian overlay inputs (reuse scatter renderer).
        frame_layer_positions = dict(static_layers)
        if dyn_local is not None and dyn_track_ids is not None:
            frame_layer_positions["dynamic_rigids"] = dyn_local_to_world_at_frame(
                dyn_local,
                dyn_track_ids,
                sorted_track_names,
                meta.tracks,
                fi,
            )

        inputs = build_inputs_from_metadata(
            meta,
            frame_layer_positions,
            fi,
            z_window_m=args.z_window_m,
        )

        title = (
            f"frame {fi}/{n_frames_ego - 1}  seq={meta.sequence_id}  "
            f"cuboids={len(inputs.cuboids)}  cams={len(cam_c2w)}"
        )
        out_img = render_bev_frame(
            inputs,
            xy_range_m=args.xy_range_m,
            grid_step_m=args.grid_step_m,
            dpi=args.dpi,
            show_labels=not args.no_labels,
            title=title,
            backdrop_rgb=bg_rgb,
            backdrop_xy_extent=extent,
        )
        out_path = out_dir / f"bev_stitched_{fi:04d}.png"
        iio.imwrite(str(out_path), out_img)

        manifest.append(
            {
                "frame_idx": fi,
                "ego_t_us": ego_t_us,
                "out_path": str(out_path.relative_to(out_dir)),
                "n_cuboids": len(inputs.cuboids),
                "n_cameras_with_image": len(cam_c2w),
            }
        )
        if args.verbose:
            print(
                f"[bev-stitch]   frame {fi:>4d}: cuboids={len(inputs.cuboids):>3d}  "
                f"cams={len(cam_c2w):>2d}  → {out_path.name}",
                flush=True,
            )

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w") as fh:
        json.dump(
            {
                "ckpt": str(ckpt_path),
                "dataset": str(dataset_path),
                "sequence_id": meta.sequence_id,
                "cameras": camera_ids,
                "xy_range_m": args.xy_range_m,
                "res_mpp": args.res_mpp,
                "n_frames": len(manifest),
                "frames": manifest,
            },
            fh,
            indent=2,
        )

    print(f"[bev-stitch] wrote {len(manifest)} PNGs + {manifest_path}", flush=True)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gs_object", required=True, type=str, help="Path to v2 LayeredGaussians ckpt (.pt).")
    parser.add_argument("--dataset_path", required=True, type=str, help="Path to NCore manifest .json.")
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--frame_range", default=None, type=str, help='Frame slice e.g. "0:10", "5", or empty for all.')
    parser.add_argument(
        "--xy_range_m", default=30.0, type=float, help="Half-width of BEV square around current ego (default 30 m)."
    )
    parser.add_argument(
        "--res_mpp", default=0.10, type=float, help="BEV resolution in meters per pixel (default 0.10)."
    )
    parser.add_argument("--grid_step_m", default=10.0, type=float)
    parser.add_argument("--z_window_m", default=10.0, type=float)
    parser.add_argument("--dpi", default=100, type=int)
    parser.add_argument(
        "--ground_offset_m",
        default=1.5,
        type=float,
        help="Subtract from ego_pose Z to estimate road height " "(camera mounted ~1.5 m above ground; default 1.5).",
    )
    parser.add_argument("--no_labels", action="store_true")
    parser.add_argument(
        "--camera_ids",
        default=None,
        type=str,
        help="Comma-separated camera IDs; default uses the 5 " "wide-coverage (120/70fov) cameras.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    return diagnose(args)


if __name__ == "__main__":
    sys.exit(main())
