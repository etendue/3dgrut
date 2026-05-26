#!/usr/bin/env python3
"""Compare raw + cuboid vs reconstructed-Gaussian + cuboid for one frame.

Renders the primary camera at a given frame index two ways:

  (A) Raw image (NCore decode) + FTheta-projected cuboid wireframes
  (B) Reconstructed image (LayeredGaussians via engine) + same wireframes

Both panels use the same camera c2w + intrinsics + cuboid set, so any
discrepancy between (A) and (B) is a scene-reconstruction error
(Gaussian model has placed content at the wrong world position), not a
cuboid bug. Side-by-side PNG saved to ``--output``.

Run on GPU machine (ThinkPad / a800-x2) — engine needs CUDA.

Usage::

    python scripts/compare_raw_vs_recon_cuboid.py \\
        --config_name apps/ncore_3dgut_mcmc_v2_full_4dviz_dynfix \\
        --path /home/yusun/work/data/9ae*/pai_*.json \\
        --ckpt /home/yusun/work/ckpts/B3_30k_20260525/ckpt_last.pt \\
        --frame_idx 20 \\
        --output /tmp/compare_frame_20.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _draw_polyline(pil_img: Image.Image, uv: np.ndarray, vis: np.ndarray,
                   color: tuple, width: int = 2) -> int:
    """Draw projected polyline; returns drawn-segment count (segments where
    both endpoints are visible)."""
    if uv.shape[0] < 2:
        return 0
    draw = ImageDraw.Draw(pil_img)
    drawn = 0
    for i in range(uv.shape[0] - 1):
        if bool(vis[i]) and bool(vis[i + 1]):
            draw.line(
                [(float(uv[i, 0]), float(uv[i, 1])),
                 (float(uv[i + 1, 0]), float(uv[i + 1, 1]))],
                fill=color, width=width,
            )
            drawn += 1
    return drawn


def _draw_caption(pil: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    pad = 6
    tw = draw.textlength(text, font=font)
    box = [(pad, pad), (pad + tw + 2 * pad, pad + 28)]
    draw.rectangle(box, fill=(0, 0, 0, 220))
    draw.text((pad * 2, pad + 2), text, fill=(255, 255, 255), font=font)


def _draw_cuboids_on(pil: Image.Image, projector, c2w_opencv: np.ndarray,
                     cuboid_set: list[tuple[str, np.ndarray, tuple]],
                     subdivide_n: int = 20) -> int:
    drawn_cuboids = 0
    for tid, edges, color in cuboid_set:
        polylines = [edges[i].astype(np.float64) for i in range(12)]
        results = projector.project_polylines(
            polylines, c2w_opencv, subdivide_n=subdivide_n)
        segs = 0
        for uv, vis in results:
            segs += _draw_polyline(pil, uv, vis, color, width=2)
        if segs > 0:
            drawn_cuboids += 1
    return drawn_cuboids


def _collect_active_cuboid_edges(meta, t_us: int
                                 ) -> list[tuple[str, np.ndarray, tuple]]:
    from threedgrut_playground.utils.cuboid import (
        cuboid_world_edges, instance_color,
    )
    frame_idx = meta.lookup_frame_idx(t_us)
    out: list[tuple[str, np.ndarray, tuple]] = []
    for tid in meta.active_tracks_at(frame_idx):
        t = meta.tracks[tid]
        poses = t["poses"]
        if poses is None or frame_idx >= poses.shape[0]:
            continue
        pose = poses[frame_idx]
        size = t["size"] if t["size"] is not None else np.array(
            [1.0, 1.0, 1.0], dtype=np.float32)
        edges = cuboid_world_edges(pose, size)
        col_f = instance_color(tid)
        col = tuple(int(round(c * 255)) for c in col_f)
        out.append((tid, edges, col))
    return out


def main(argv=None) -> int:
    import ncore.data as _nd
    from hydra import compose, initialize_config_dir
    from threedgrut_playground.utils.ftheta_projector import (
        FthetaForwardProjector,
    )
    from threedgrut_playground.utils.viz4d_metadata import FourDMetadata

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config_name", required=True, type=str)
    p.add_argument("--path", required=True, type=str,
                   help="NCore manifest path")
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--frame_idx", type=int, default=20,
                   help="primary-camera frame index (default 20)")
    p.add_argument("--camera_id", default="camera_front_wide_120fov", type=str)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load dataset (NCore).
    print(f"[compare] loading dataset via Hydra config={args.config_name}",
          flush=True)
    with initialize_config_dir(
            config_dir=str(_REPO_ROOT / "configs"), version_base=None):
        conf = compose(
            config_name=args.config_name,
            overrides=[f"path={args.path}", "trainer.sky_backend=mlp"],
        )
    import threedgrut.datasets as datasets
    dataset, _val = datasets.make(conf.dataset.type, conf, ray_jitter=None)
    seq_id = dataset.sequence_id
    camera_sensor = dataset.sequence_camera_sensors[seq_id][args.camera_id]

    # 2) Load ckpt + metadata.
    print(f"[compare] loading ckpt → {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    meta = FourDMetadata.from_ckpt(ckpt)
    if meta is None or not meta.tracks:
        print("[compare] ERROR: ckpt has no viz_4d / tracks", file=sys.stderr)
        return 2

    # 3) Resolve frame: use primary camera's train-frame indices so frame_idx
    #    matches the ego/cuboid timeline (524 frames for B3_30k).
    train_indices = dataset.camera_train_frame_indices.get(args.camera_id)
    if train_indices is None or len(train_indices) <= args.frame_idx:
        print(f"[compare] ERROR: frame_idx {args.frame_idx} out of range "
              f"(n_train={len(train_indices) if train_indices is not None else 0})",
              file=sys.stderr)
        return 3
    cam_frame_idx = int(train_indices[args.frame_idx])
    ts_array = np.asarray(camera_sensor.frames_timestamps_us)
    t_us = int(ts_array[cam_frame_idx, int(_nd.FrameTimepoint.END)])
    print(f"[compare] frame_idx={args.frame_idx} → cam_frame_idx={cam_frame_idx}  "
          f"t_us={t_us}", flush=True)

    # 4) Raw image.
    rgb_arr = dataset._decode_image(camera_sensor, cam_frame_idx)
    H_img, W_img = int(rgb_arr.shape[0]), int(rgb_arr.shape[1])
    pil_raw = Image.fromarray(rgb_arr, mode="RGB")
    print(f"[compare] raw image {W_img}×{H_img}", flush=True)

    # 5) c2w (OpenCV) and FTheta intrinsics for projecting cuboids.
    _T_start, T_c2w_end = dataset._get_start_end_poses_world_global(
        camera_sensor, cam_frame_idx)
    T_c2w_opencv = np.asarray(T_c2w_end, dtype=np.float64)
    intrinsics_result = dataset._get_camera_model_parameters_for_resolution(
        args.camera_id, render_w=W_img, render_h=H_img)
    if intrinsics_result is None:
        print("[compare] ERROR: cannot extract intrinsics for primary camera",
              file=sys.stderr)
        return 4
    ftheta_dict, model_type_name = intrinsics_result
    if "FTheta" not in model_type_name:
        print(f"[compare] ERROR: primary camera is not FTheta "
              f"(model={model_type_name})", file=sys.stderr)
        return 5
    # NCore _get_camera_model_parameters_for_resolution already returns the
    # 8-key dict expected by FthetaForwardProjector.
    projector = FthetaForwardProjector(
        ftheta_dict, world_to_camera_flip=np.eye(4))   # OpenCV in, no flip

    cuboid_set = _collect_active_cuboid_edges(meta, t_us)
    print(f"[compare] active cuboids @ t_us: {len(cuboid_set)}", flush=True)

    n_drawn_raw = _draw_cuboids_on(pil_raw, projector, T_c2w_opencv,
                                   cuboid_set)
    _draw_caption(pil_raw,
                  f"RAW + cuboid  |  frame {args.frame_idx}  |  "
                  f"{n_drawn_raw}/{len(cuboid_set)} cuboids drawn")

    # 6) Engine render reconstructed image — let the engine load ckpt itself
    # (avoids breaking its OptiX BVH state by injecting a separate model).
    print(f"[compare] loading engine on CUDA …", flush=True)
    from threedgrut_playground.engine import Engine3DGRUT
    engine = Engine3DGRUT(
        gs_object=args.ckpt,
        mesh_assets_folder=str(_REPO_ROOT / "threedgrut_playground" / "assets"),
        default_config=args.config_name,
        envmap_assets_folder=str(_REPO_ROOT / "threedgrut_playground" / "assets"),
    )
    # FTheta ckpts MUST use fisheye render path (mirrors viser_gui_4d L1303).
    engine.camera_type = "Fisheye"
    print(f"[compare] engine.camera_type set to Fisheye", flush=True)
    # Ensure tracks_poses populated on engine's model (viser_gui_4d does this
    # at viewer init via _load_metadata path).
    if "tracks" in ckpt.get("viz_4d", {}) and hasattr(engine.scene_mog, "populate_tracks"):
        tracks_dict = ckpt["viz_4d"]["tracks"]
        shared_ts = ckpt["viz_4d"].get("tracks_camera_timestamps_us")
        if shared_ts is not None and len(tracks_dict) > 0:
            first_tid = next(iter(tracks_dict))
            tracks_dict[first_tid]["cam_timestamps_us"] = shared_ts
        engine.scene_mog.populate_tracks(tracks_dict)

    # Build kaolin Camera at primary camera pose.
    from kaolin.render.camera import Camera
    # viser_gui_4d._snap_clients_to_ego sets client.camera.wxyz directly from
    # the OpenCV ego pose (no convention flip). get_c2w then returns the same
    # OpenCV c2w. So the engine actually consumes OpenCV c2w as view_matrix —
    # despite the "viser" naming in the projector path. Use NCore raw c2w
    # directly.
    c2w_for_engine = T_c2w_opencv
    # FOV: viser uses a 90° kaolin-frame FOV when ckpt is FTheta and the
    # actual fisheye distortion comes from fisheye_intrinsics. Mirror that.
    fov_y = float(np.deg2rad(90.0))
    # kaolin Camera.from_args expects view_matrix = world-to-camera (the
    # standard "view matrix" convention in graphics). NCore + viser pass c2w;
    # invert for kaolin.
    w2c = np.linalg.inv(c2w_for_engine).astype(np.float32)
    kaolin_camera = Camera.from_args(
        view_matrix=w2c,
        fov=fov_y,
        width=W_img, height=H_img,
        near=0.1, far=500.0,
        dtype=torch.float32,
        device=engine.device,
    )
    # Use the ckpt-embedded ftheta dict (T8.13 schema_v2) — that's what
    # the viser viewer's render path consumes via meta.ego_primary_intrinsics_ftheta.
    ftheta_for_engine = (meta.ego_primary_intrinsics_ftheta
                         if meta.ego_primary_intrinsics_ftheta is not None
                         else ftheta_dict)
    print(f"[compare] engine fisheye_intrinsics keys: "
          f"{sorted(ftheta_for_engine.keys())}", flush=True)
    out = engine.render_pass(
        kaolin_camera, is_first_pass=True,
        timestamp_us=t_us,
        fisheye_intrinsics=ftheta_for_engine,
    )
    rgba = torch.cat([out["rgb"], out["opacity"]], dim=-1)
    rgba = torch.clamp(rgba, 0.0, 1.0)
    img_recon = (rgba[0, :, :, :3] * 255).to(torch.uint8).cpu().numpy()
    pil_recon = Image.fromarray(img_recon, mode="RGB")
    print(f"[compare] reconstructed image {pil_recon.size}", flush=True)

    n_drawn_recon = _draw_cuboids_on(pil_recon, projector, T_c2w_opencv,
                                     cuboid_set)
    _draw_caption(pil_recon,
                  f"RECON + cuboid  |  frame {args.frame_idx}  |  "
                  f"{n_drawn_recon}/{len(cuboid_set)} cuboids drawn")

    # 7) Side-by-side composite (top = raw, bottom = recon, stacked vertically
    # so a single image is easy to scroll).
    if pil_recon.size != pil_raw.size:
        pil_recon = pil_recon.resize(pil_raw.size)
    combined = Image.new("RGB",
                         (pil_raw.width, pil_raw.height * 2 + 8),
                         color=(20, 20, 22))
    combined.paste(pil_raw, (0, 0))
    combined.paste(pil_recon, (0, pil_raw.height + 8))
    combined.save(args.output)
    print(f"[compare] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
