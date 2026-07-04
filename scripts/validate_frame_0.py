#!/usr/bin/env python3
"""T8/B3 Phase E.10 — Frame-0 alignment validation.

Goal: before another A800 retrain, verify the FIRST FRAME setup is consistent.
Projects everything that should land on/near a vehicle onto the same image
through the same FTheta polynomial:

  out_0_gt.png             — raw GT image (sanity)
  out_1_cuboids.png        — GT + cuboid wireframes (FTheta-projected 8 edges)
  out_2_sseg.png           — GT + sseg dynamic-class pixels (red tint)
  out_3_lidar_init.png     — GT + bg LiDAR (blue dots) + dyn LiDAR (red dots)
  out_4_gaussian_init.png  — GT + bg Gaussian centers (blue) + dyn Gaussian centers (red)
                             (post-init only; for post-train state, pass --ckpt)

A consistent first frame means:
  * cuboid wireframes land ON cars (not in mid-air)
  * sseg dyn pixels cover the same cars
  * dyn LiDAR init dots cluster inside the cuboids
  * bg LiDAR dots avoid the cuboid interiors

If those four overlays disagree, every downstream training step inherits the
mismatch — much cheaper to fix here than after a 30k retrain.

Usage (A800)::

    python scripts/validate_frame_0.py \\
        --config_name apps/ncore_3dgut_mcmc_v2_full_4dviz_dynfix \\
        --path /root/work/yusun/ncore-nurec/data/ncore/clips/9ae*/pai_*.json \\
        --frame_idx 0 \\
        --output_dir /tmp/E10_frame0_init

After 1k training::

    python scripts/validate_frame_0.py \\
        --config_name ... --path ... \\
        --ckpt /path/to/ckpt_last.pt \\
        --output_dir /tmp/E10_frame0_after_1k
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Hydra config compose
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# PIL for image draw overlays
from PIL import Image, ImageDraw

_CONFIG_DIR = str(_REPO_ROOT / "configs")


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


def _draw_dots(
    im: Image.Image,
    uv: np.ndarray,
    visible: np.ndarray,
    color: tuple[int, int, int],
    radius: int = 2,
    subsample: int = -1,
) -> None:
    """Overlay dots at (u, v) pixel coords where visible == True. Mutates ``im``."""
    draw = ImageDraw.Draw(im)
    if subsample > 0 and uv.shape[0] > subsample:
        idx = np.random.default_rng(0).choice(uv.shape[0], subsample, replace=False)
        uv = uv[idx]
        visible = visible[idx]
    for (u, v), vis in zip(uv, visible):
        if not vis:
            continue
        x = int(round(float(u)))
        y = int(round(float(v)))
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline=None)


def _draw_polyline(
    im: Image.Image, uv: np.ndarray, visible: np.ndarray, color: tuple[int, int, int], width: int = 2
) -> None:
    """Overlay a polyline; skips segments where either endpoint is invisible."""
    draw = ImageDraw.Draw(im)
    for i in range(uv.shape[0] - 1):
        if not visible[i] or not visible[i + 1]:
            continue
        x0, y0 = float(uv[i, 0]), float(uv[i, 1])
        x1, y1 = float(uv[i + 1, 0]), float(uv[i + 1, 1])
        draw.line([(x0, y0), (x1, y1)], fill=color, width=width)


def _draw_mask_overlay(im: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.4) -> None:
    """Tint pixels where ``mask > 0.5`` with ``color`` at the given alpha."""
    arr = np.array(im, dtype=np.float32)
    mask_bool = mask > 0.5
    tint = np.asarray(color, dtype=np.float32)
    arr[mask_bool] = (1.0 - alpha) * arr[mask_bool] + alpha * tint
    im.paste(Image.fromarray(arr.astype(np.uint8)))


def _gt_image_to_pil(rgb_gt: torch.Tensor) -> Image.Image:
    """Convert ``[H, W, 3]`` float (0..1) tensor → PIL RGB image."""
    arr = rgb_gt.detach().cpu().numpy()
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--config_name",
        required=True,
        type=str,
        help="Hydra config name, e.g. apps/ncore_3dgut_mcmc_v2_full_4dviz_dynfix",
    )
    p.add_argument("--path", required=True, type=str, help="dataset manifest path (overrides conf.path)")
    p.add_argument("--frame_idx", type=int, default=0, help="frame index to render against (default 0)")
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="optional ckpt to load instead of fresh init " "(post-training validation)",
    )
    p.add_argument(
        "--bg_subsample", type=int, default=5000, help="random subsample of bg dots to overlay (default 5000)"
    )
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Compose conf
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        conf = compose(
            config_name=args.config_name,
            overrides=[f"path={args.path}", "trainer.sky_backend=mlp"],
        )

    # 2. Build dataset via the same factory the trainer uses (matches all the
    # NCore-specific kwargs: load_aux_masks, val_frame_interval, downsample,
    # cam/lidar IDs, etc.).
    import threedgrut.datasets as datasets

    dataset, _val_dataset = datasets.make(conf.dataset.type, conf, ray_jitter=None)

    # 3. Build LayeredGaussians + init layers
    from threedgrut.layers.layered_model import LayeredGaussians
    from threedgrut.layers.registry import specs_from_config

    specs = specs_from_config(conf)
    scene_extent = float(dataset.get_scene_extent())
    model = LayeredGaussians(conf, specs=specs, scene_extent=scene_extent)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if args.ckpt is not None:
        print(f"[frame0] loading ckpt {args.ckpt}", flush=True)
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.init_from_checkpoint(ckpt, setup_optimizer=False)
        viz_4d = ckpt.get("viz_4d")
        if isinstance(viz_4d, dict) and viz_4d.get("tracks"):
            tracks_dict = viz_4d["tracks"]
            shared_ts = viz_4d.get("tracks_camera_timestamps_us")
            if shared_ts is not None:
                first_tid = next(iter(tracks_dict))
                tracks_dict[first_tid]["cam_timestamps_us"] = shared_ts
            model.populate_tracks(tracks_dict)
    else:
        # Init bg from LiDAR (mirror trainer.setup_training T3 path L327-356).
        from threedgrut.datasets.utils import PointCloud

        pc = PointCloud.from_sequence(
            list(dataset.get_point_clouds(step_frame=1, non_dynamic_points_only=True)),
            device="cpu",
        )
        n_init = conf.initialization.get("num_points", len(pc.xyz_end))
        if n_init < len(pc.xyz_end):
            rng = torch.Generator().manual_seed(int(conf.get("seed_initialization", 0)))
            idxs = torch.randperm(len(pc.xyz_end), generator=rng)[:n_init]
            pc = pc.selected_idxs(idxs)
        observer_points = torch.tensor(
            dataset.get_observer_points(),
            dtype=torch.float32,
            device=device,
        )
        if "background" in model.layers:
            model.layers["background"].init_from_lidar(pc, observer_points)
            print(
                f"[frame0] bg layer initialized: " f"{model.layers['background'].num_gaussians} particles", flush=True
            )

        # Init road (BEV-grid + KNN, optional). init_road_layer returns
        # (positions, rotations, scales, densities, colors) so we unpack.
        if "road" in model.layers:
            from threedgrut.layers.road_init import init_road_layer

            road_pts, _road_rgb = dataset.get_road_lidar_points()
            traj_pts = torch.tensor(
                dataset.get_observer_points(),
                dtype=torch.float32,
                device=device,
            )
            r_pos, r_rot, r_sca, r_den, r_col = init_road_layer(
                road_pts.to(device),
                traj_pts,
            )
            model.init_layer_from_points(
                "road",
                r_pos,
                rotations=r_rot,
                scales=r_sca,
                densities=r_den,
                colors=r_col,
                setup_optimizer=False,
            )
            print(f"[frame0] road layer initialized: {r_pos.shape[0]} particles", flush=True)

        # Init dyn_rigids + populate tracks
        if "dynamic_rigids" in model.layers:
            import ncore.data as _nd

            from threedgrut.datasets.tracks_loader import load_tracks_from_ncore_cuboids
            from threedgrut.layers.dynamic_rigid_init import init_dynamic_rigid_layer

            loader = dataset.sequence_loaders[dataset.sequence_id]
            ref_cam = dataset.camera_ids[0]
            ref_sensor = dataset.sequence_camera_sensors[dataset.sequence_id][ref_cam]
            cam_ts = ref_sensor.frames_timestamps_us[:, _nd.FrameTimepoint.END]
            time_range = dataset.time_range_us
            in_window = np.array([int(t) in time_range for t in cam_ts])
            cam_ts_active = np.asarray(cam_ts)[in_window]
            tracks = load_tracks_from_ncore_cuboids(loader, cam_ts_active)
            print(f"[frame0] loaded {len(tracks)} tracks", flush=True)
            if tracks:
                model.populate_tracks(tracks)
                # Run dyn init on CPU — fewer surprises around tiny per-track
                # tensor catenations on the GPU, and the workload is tiny.
                dyn_pts, _ = dataset.get_dynamic_lidar_points()
                dyn_pts_cpu = dyn_pts.detach().cpu()
                tracks_cpu = {}
                for tid, info in tracks.items():
                    tracks_cpu[tid] = {
                        "pts": None,
                        "colors": None,
                        "poses": info["poses"].detach().cpu(),
                        "size": info["size"].detach().cpu(),
                        "frame_info": info["frame_info"].detach().cpu(),
                        "class": info["class"],
                    }
                d_pos, d_track_ids, _ = init_dynamic_rigid_layer(
                    tracks_cpu,
                    dyn_pts_cpu,
                    max_pts_per_track=5_000,
                )
                model.init_layer_from_points(
                    "dynamic_rigids",
                    d_pos.to(device),
                    track_ids=d_track_ids.to(device),
                    setup_optimizer=False,
                )
                print(f"[frame0] dyn_rigids layer initialized: {d_pos.shape[0]} particles", flush=True)

    # 4. Get frame batch (use frame_idx)
    raw_batch = dataset[args.frame_idx]
    gpu_batch = dataset.get_gpu_batch_with_intrinsics(raw_batch)
    H, W = int(gpu_batch.rgb_gt.shape[1]), int(gpu_batch.rgb_gt.shape[2])
    print(
        f"[frame0] frame {args.frame_idx} | H={H} W={W} | "
        f"ts_us={getattr(gpu_batch, 'timestamp_us', -1)} | "
        f"cam_idx={getattr(gpu_batch, 'camera_idx', -1)}",
        flush=True,
    )

    ftheta_dict = getattr(gpu_batch, "intrinsics_FThetaCameraModelParameters", None)
    if ftheta_dict is None:
        raise RuntimeError(
            "Batch has no intrinsics_FThetaCameraModelParameters — " "current validation only supports FTheta cameras."
        )
    T_c2w = gpu_batch.T_to_world[0].detach().cpu().numpy()

    # 5. Build FTheta projector (numpy)
    from threedgrut_playground.utils.ftheta_projector import FthetaForwardProjector

    # ftheta_dict from NCore: angle_to_pixeldist_poly list, principal_point list,
    # resolution list, max_angle scalar. FthetaForwardProjector accepts dict directly.
    projector = FthetaForwardProjector(ftheta_dict)

    # IMPORTANT: FthetaForwardProjector expects c2w in *viser convention*
    # (Y-down, Z-backward) and applies FLIP_VISER_TO_OPENCV internally. The
    # dataset's T_to_world is already in OpenCV convention (Y-down, Z-forward),
    # so we PRE-undo the projector's flip by sending c2w @ FLIP^-1.
    # FLIP_VISER_TO_OPENCV = diag([1, 1, -1, 1]), self-inverse.
    flip = np.diag([1.0, 1.0, -1.0, 1.0])
    c2w_for_projector = T_c2w @ flip  # projector applies @ flip again → cancels

    # 6. GT image → PIL
    rgb_gt = gpu_batch.rgb_gt[0]  # [H, W, 3]
    pil_gt = _gt_image_to_pil(rgb_gt)
    pil_gt.save(args.output_dir / "out_0_gt.png")
    print(f"[frame0] saved {args.output_dir / 'out_0_gt.png'}", flush=True)

    # 7. Cuboid wireframes (FTheta-projected)
    pil_cuboids = pil_gt.copy()
    from threedgrut_playground.utils.cuboid import (
        UNIT_CUBE_EDGES,
        class_color,
        cuboid_world_edges,
    )

    n_cuboids_drawn = 0
    if model.tracks_poses:
        idx = model._resolve_pose_idx(
            int(getattr(gpu_batch, "timestamp_us", -1)),
            args.frame_idx,
        )
        tracks_metadata = getattr(model, "tracks_metadata", {})
        for tid in sorted(model.tracks_poses.keys()):
            active = model.tracks_active.get(tid)
            if active is None or idx >= int(active.shape[0]) or not bool(active[idx]):
                continue
            meta = tracks_metadata.get(tid, {}) or {}
            size = meta.get("size")
            if size is None:
                continue
            size_np = size.detach().cpu().numpy()
            pose_np = model.tracks_poses[tid][idx].detach().cpu().numpy()
            edges = cuboid_world_edges(pose_np, size_np)  # (12, 2, 3)
            # Subdivide each edge for proper FTheta curvature
            polylines = []
            for seg in edges:
                t_lin = np.linspace(0, 1, 21)
                pts = seg[0:1] + (seg[1:2] - seg[0:1]) * t_lin[:, None]
                polylines.append(pts.astype(np.float64))
            results = projector.project_polylines(polylines, c2w_for_projector, subdivide_n=1)
            cls = meta.get("class", "unknown")
            col_f = class_color(str(cls))
            col = tuple(int(round(c * 255)) for c in col_f)
            for uv, vis in results:
                _draw_polyline(pil_cuboids, uv, vis, col, width=2)
            n_cuboids_drawn += 1
    pil_cuboids.save(args.output_dir / "out_1_cuboids.png")
    print(
        f"[frame0] saved {args.output_dir / 'out_1_cuboids.png'} " f"({n_cuboids_drawn} active cuboids drawn)",
        flush=True,
    )

    # 8. Sseg dynamic mask overlay
    pil_sseg = pil_gt.copy()
    image_infos = getattr(gpu_batch, "image_infos", None) or {}
    print(f"[frame0] image_infos keys: {list(image_infos.keys())}", flush=True)
    for k, v in image_infos.items():
        if hasattr(v, "shape"):
            print(f"  - {k}: shape={tuple(v.shape)} dtype={v.dtype}")
    dyn_sseg = image_infos.get("dyn_mask_sseg")
    if dyn_sseg is not None:
        # Image_infos masks come from the dataloader collated batch:
        # shape is [B, H, W] (B=1 in our case). Squeeze to [H, W].
        mask_t = dyn_sseg
        while mask_t.dim() > 2:
            mask_t = mask_t.squeeze(0)
        mask_np = mask_t.detach().cpu().numpy()
        # Sanity check: dataset sometimes stores mask transposed relative to
        # image. Image PIL array is (H, W, 3); mask must be (H, W).
        if mask_np.shape != (H, W):
            print(
                f"[frame0] sseg mask shape {mask_np.shape} differs from " f"image (H={H}, W={W}); transposing.",
                flush=True,
            )
            mask_np = mask_np.T
        _draw_mask_overlay(pil_sseg, mask_np, color=(255, 0, 0), alpha=0.45)
    pil_sseg.save(args.output_dir / "out_2_sseg.png")
    print(f"[frame0] saved {args.output_dir / 'out_2_sseg.png'}", flush=True)

    # 9. LiDAR init projection (bg + dyn at frame idx). Use Gaussian-center
    # positions directly: bg/road centers are world-frame, dyn centers are
    # object-local + need pose transform.
    pil_lidar = pil_gt.copy()
    bg_layer = model.layers["background"]
    bg_pos_np = bg_layer.positions.detach().cpu().numpy()
    uv_bg, vis_bg = projector.project_points(bg_pos_np, c2w_for_projector)
    _draw_dots(pil_lidar, uv_bg, vis_bg, color=(80, 120, 255), radius=1, subsample=args.bg_subsample)

    if "dynamic_rigids" in model.layers:
        dyn = model.layers["dynamic_rigids"]
        track_ids_buf = getattr(dyn, "track_ids", None)
        if track_ids_buf is not None and dyn.positions.numel() > 0:
            dyn_local = dyn.positions.detach().cpu().numpy()
            t_ids = track_ids_buf.detach().cpu().numpy()
            track_keys = sorted(model.tracks_poses.keys())
            dyn_world = np.zeros_like(dyn_local)
            for i, tid in enumerate(track_keys):
                sel = t_ids == i
                if not sel.any():
                    continue
                active = model.tracks_active.get(tid)
                if active is None:
                    continue
                idx_t = model._resolve_pose_idx(
                    int(getattr(gpu_batch, "timestamp_us", -1)),
                    args.frame_idx,
                )
                if idx_t >= int(active.shape[0]) or not bool(active[idx_t]):
                    continue
                pose = model.tracks_poses[tid][idx_t].detach().cpu().numpy()
                pts_h = np.concatenate(
                    [dyn_local[sel], np.ones((int(sel.sum()), 1))],
                    axis=-1,
                )
                dyn_world[sel] = (pose @ pts_h.T).T[:, :3]
            uv_dyn, vis_dyn = projector.project_points(dyn_world, c2w_for_projector)
            _draw_dots(pil_lidar, uv_dyn, vis_dyn, color=(255, 80, 80), radius=2)
    pil_lidar.save(args.output_dir / "out_3_lidar_init.png")
    print(
        f"[frame0] saved {args.output_dir / 'out_3_lidar_init.png'} " f"(bg subsampled to {args.bg_subsample} dots)",
        flush=True,
    )

    # 10. Combined Gaussian-center overlay (same as #9 since centers == LiDAR
    # init points at iter 0; differs after training).
    # In ckpt mode the layer positions have drifted, so this is the post-train
    # projected centers.
    pil_lidar.save(args.output_dir / "out_4_gaussian_centers.png")
    print(
        f"[frame0] saved {args.output_dir / 'out_4_gaussian_centers.png'} " f"(== out_3 unless --ckpt is passed)",
        flush=True,
    )

    print("[frame0] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
