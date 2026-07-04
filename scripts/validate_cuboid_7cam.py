#!/usr/bin/env python3
"""7-camera cuboid projection validator (B2 follow-up).

Goal: confirm that the cuboid wireframe projection algorithm is correct on
**every** NCore v4 camera (mix of FTheta fisheye + OpenCV pinhole), not
just the single ego-primary FTheta camera that the viser viewer renders.

For a given timestamp ``--t_us`` (default = ckpt's ``t_us_first``), the
script:

  1. Loads the NCore dataset and the model checkpoint (for cuboid poses).
  2. For each of the 7 cameras:
       a. Finds the camera frame whose END timestamp is closest to ``t_us``.
       b. Pulls the raw image + ``T_camera_to_world`` (OpenCV convention).
       c. Picks the right projector based on the camera model:
            - ``FthetaForwardProjector(flip=I)`` for FTheta cameras
            - ``PinholeForwardProjector(flip=I)`` for OpenCV pinhole / fisheye
       d. Projects every active cuboid's 12 wireframe edges and draws them
          on a copy of the raw image (per-track instance color).
  3. Tiles the 7 per-camera PNGs into a 2×4 grid for at-a-glance inspection.

Both single-camera PNGs and the grid land in ``--output_dir``.

A "correct" output:

  - Cuboid edges hug the cars in each camera that sees them.
  - FTheta (fisheye) cameras show **curved** edges at image periphery.
  - Pinhole cameras show **straight** edges.
  - Rear-facing cameras do NOT show cuboids that belong to a forward camera.

Usage (a800-x2)::

    python scripts/validate_cuboid_7cam.py \\
        --config_name apps/ncore_3dgut_mcmc_v2_full_4dviz_dynfix \\
        --path /root/work/yusun/ncore-nurec/data/ncore/clips/9ae*/pai_*.json \\
        --ckpt /root/work/yusun/ncore-nurec/output/<run>/checkpoints/ckpt_last.pt \\
        --output_dir /tmp/cuboid_7cam_validate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import ncore.data as _nd
from hydra import compose, initialize_config_dir
from PIL import Image, ImageDraw, ImageFont

_CONFIG_DIR = str(_REPO_ROOT / "configs")


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------


def _draw_polyline(im: Image.Image, uv: np.ndarray, visible: np.ndarray, color: tuple, width: int = 2) -> int:
    """Overlay a polyline; skip segments where either endpoint is invisible.

    Returns the number of segments actually drawn — used to decide whether
    a cuboid is "visible in this camera" for the per-cam counter.
    """
    draw = ImageDraw.Draw(im)
    drawn = 0
    for i in range(uv.shape[0] - 1):
        if not visible[i] or not visible[i + 1]:
            continue
        x0, y0 = float(uv[i, 0]), float(uv[i, 1])
        x1, y1 = float(uv[i + 1, 0]), float(uv[i + 1, 1])
        draw.line([(x0, y0), (x1, y1)], fill=color, width=width)
        drawn += 1
    return drawn


def _draw_caption(im: Image.Image, text: str) -> None:
    """Stamp a top-left caption with a translucent box behind it for legibility."""
    draw = ImageDraw.Draw(im, mode="RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except (IOError, OSError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    w_t = bbox[2] - bbox[0]
    h_t = bbox[3] - bbox[1]
    draw.rectangle([6, 6, 6 + w_t + 14, 6 + h_t + 14], fill=(0, 0, 0, 160))
    draw.text((13, 10), text, fill=(255, 255, 255, 255), font=font)


# --------------------------------------------------------------------------
# Camera frame lookup
# --------------------------------------------------------------------------


def _find_nearest_camera_frame(camera_sensor, t_us: int) -> tuple[int, int]:
    """Return (camera_frame_index, frame_END_ts_us) closest to ``t_us``."""
    ts = camera_sensor.frames_timestamps_us[:, _nd.FrameTimepoint.END]
    ts = np.asarray(ts, dtype=np.int64)
    idx = int(np.argmin(np.abs(ts - int(t_us))))
    return idx, int(ts[idx])


def _build_projector(intrinsics: dict, model_type_name: str):
    """Pick FTheta vs Pinhole projector. NCore raw c2w is OpenCV — flip=I."""
    from threedgrut_playground.utils.ftheta_projector import FthetaForwardProjector
    from threedgrut_playground.utils.pinhole_projector import PinholeForwardProjector

    if model_type_name == "FThetaCameraModelParameters":
        return FthetaForwardProjector(intrinsics, world_to_camera_flip=np.eye(4)), "ftheta", 20
    if model_type_name in (
        "OpenCVPinholeCameraModelParameters",
        "OpenCVFisheyeCameraModelParameters",
    ):
        return PinholeForwardProjector(intrinsics, world_to_camera_flip=np.eye(4)), "pinhole", 4
    return None, "unknown", 1


# --------------------------------------------------------------------------
# Cuboid collection (from ckpt)
# --------------------------------------------------------------------------


def _collect_active_cuboid_edges(meta, t_us: int) -> list[tuple[str, np.ndarray, tuple]]:
    """Walk ``FourDMetadata`` tracks, return per-track ``(tid, (12,2,3), color)``
    for every active cuboid at the frame nearest ``t_us``."""
    from threedgrut_playground.utils.cuboid import cuboid_world_edges, instance_color

    frame_idx = meta.lookup_frame_idx(t_us)
    out: list[tuple[str, np.ndarray, tuple]] = []
    for tid in meta.active_tracks_at(frame_idx):
        t = meta.tracks[tid]
        poses = t["poses"]
        if poses is None or frame_idx >= poses.shape[0]:
            continue
        pose = poses[frame_idx]
        size = t["size"] if t["size"] is not None else np.array([1.0, 1.0, 1.0], dtype=np.float32)
        edges = cuboid_world_edges(pose, size)  # (12, 2, 3)
        col_f = instance_color(tid)
        col = tuple(int(round(c * 255)) for c in col_f)
        out.append((tid, edges, col))
    return out


# --------------------------------------------------------------------------
# Per-camera rendering
# --------------------------------------------------------------------------


def _render_camera_view(
    dataset,
    camera_id: str,
    t_us: int,
    cuboid_edges: list[tuple[str, np.ndarray, tuple]],
) -> tuple[Image.Image, dict]:
    """For one camera_id at one t_us, decode the raw image, project cuboid
    edges, and return ``(PIL image, stats dict)``."""
    seq_id = dataset.sequence_id
    camera_sensor = dataset.sequence_camera_sensors[seq_id][camera_id]

    cam_frame_idx, frame_ts_us = _find_nearest_camera_frame(camera_sensor, t_us)
    drift_us = abs(frame_ts_us - int(t_us))

    # Raw image (uint8 HWC at downsampled resolution).
    rgb_arr = dataset._decode_image(camera_sensor, cam_frame_idx)
    H, W = int(rgb_arr.shape[0]), int(rgb_arr.shape[1])
    pil = Image.fromarray(rgb_arr, mode="RGB")

    # Intrinsics scaled to this resolution.
    intrinsics_result = dataset._get_camera_model_parameters_for_resolution(camera_id, render_w=W, render_h=H)
    if intrinsics_result is None:
        _draw_caption(pil, f"{camera_id} | UNSUPPORTED CAMERA MODEL")
        return pil, {
            "model": "unsupported",
            "drawn_cuboids": 0,
            "total_cuboids": len(cuboid_edges),
            "drift_us": drift_us,
        }
    intrinsics, model_type_name = intrinsics_result

    projector, model_short, default_sub = _build_projector(intrinsics, model_type_name)
    if projector is None:
        _draw_caption(pil, f"{camera_id} | {model_type_name} (no projector)")
        return pil, {"model": model_short, "drawn_cuboids": 0, "total_cuboids": len(cuboid_edges), "drift_us": drift_us}

    # c2w in OpenCV convention. NCore returns (START, END) poses for the
    # rolling-shutter window — pick END because the cuboid timestamp lookup
    # in the ckpt also uses END timestamps, so this is the temporally
    # consistent choice for a single-pose sanity overlay.
    _T_start, T_c2w_end = dataset._get_start_end_poses_world_global(camera_sensor, cam_frame_idx)
    T_c2w = np.asarray(T_c2w_end, dtype=np.float64)

    # Project + draw every cuboid; count those with at least one visible edge.
    drawn_cuboids = 0
    for tid, edges, color in cuboid_edges:
        polylines = [edges[i].astype(np.float64) for i in range(12)]
        results = projector.project_polylines(polylines, T_c2w, subdivide_n=default_sub)
        cuboid_drawn_segments = 0
        for uv, vis in results:
            cuboid_drawn_segments += _draw_polyline(pil, uv, vis, color, width=2)
        if cuboid_drawn_segments > 0:
            drawn_cuboids += 1

    _draw_caption(
        pil,
        f"{camera_id}  |  {model_short.upper()}  |  "
        f"{drawn_cuboids}/{len(cuboid_edges)} cuboids  |  "
        f"Δt={drift_us/1000:.1f}ms",
    )
    return pil, {
        "model": model_short,
        "drawn_cuboids": drawn_cuboids,
        "total_cuboids": len(cuboid_edges),
        "drift_us": drift_us,
    }


# --------------------------------------------------------------------------
# Grid composition
# --------------------------------------------------------------------------


def _tile_grid(panels: list[Image.Image], cols: int = 4) -> Image.Image:
    """Tile panels into a ``rows × cols`` grid (panel sizes can differ; each
    cell is sized to the max width / height across panels)."""
    if not panels:
        raise ValueError("no panels to tile")
    rows = (len(panels) + cols - 1) // cols
    cell_w = max(p.width for p in panels)
    cell_h = max(p.height for p in panels)
    grid = Image.new("RGB", (cell_w * cols, cell_h * rows), color=(20, 20, 22))
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        # Center each panel within its cell.
        x = c * cell_w + (cell_w - p.width) // 2
        y = r * cell_h + (cell_h - p.height) // 2
        grid.paste(p, (x, y))
    return grid


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--config_name", required=True, help="Hydra config (e.g. apps/ncore_3dgut_mcmc_v2_full_4dviz_dynfix)"
    )
    p.add_argument("--path", required=True, help="NCore manifest path (overrides conf.path)")
    p.add_argument("--ckpt", required=True, help="checkpoint path (provides cuboid poses via viz_4d block)")
    p.add_argument("--t_us", type=int, default=None, help="timestamp (μs) to validate at; default = ckpt's t_us_first")
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--cols", type=int, default=4, help="grid columns (default 4)")
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Compose Hydra conf (matches validate_frame_0.py style).
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        conf = compose(
            config_name=args.config_name,
            overrides=[f"path={args.path}", "trainer.sky_backend=mlp"],
        )

    # 2. Build dataset via the factory (matches all NCore-specific kwargs).
    import threedgrut.datasets as datasets

    dataset, _val = datasets.make(conf.dataset.type, conf, ray_jitter=None)
    print(f"[7cam] dataset ready | cameras = {dataset.camera_ids}", flush=True)

    # 3. Load ckpt → FourDMetadata for cuboid poses + sizes.
    from threedgrut_playground.utils.viz4d_metadata import FourDMetadata

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    meta: Optional[FourDMetadata] = FourDMetadata.from_ckpt(ckpt)
    if meta is None or meta.n_tracks() == 0:
        raise RuntimeError(
            f"checkpoint {args.ckpt} has no viz_4d / tracks — cuboid poses "
            "must come from a 4D-trained ckpt with viz_4d block."
        )

    t_us = int(args.t_us) if args.t_us is not None else int(meta.t_us_first)
    frame_idx = meta.lookup_frame_idx(t_us)
    active = meta.active_tracks_at(frame_idx)
    print(
        f"[7cam] t_us={t_us} | frame_idx={frame_idx} | " f"active_cuboids={len(active)}/{meta.n_tracks()}", flush=True
    )

    cuboid_edges = _collect_active_cuboid_edges(meta, t_us)
    if not cuboid_edges:
        print("[7cam] WARNING: no active cuboids at this t_us — grid will be raw images only.", flush=True)

    # 4. Per-camera render.
    panels: list[Image.Image] = []
    stats: list[dict] = []
    for cam_id in dataset.camera_ids:
        try:
            pil, st = _render_camera_view(dataset, cam_id, t_us, cuboid_edges)
        except Exception as e:
            print(f"[7cam] ERROR camera {cam_id}: {e}", flush=True)
            # Substitute a black panel + caption so the grid stays aligned.
            blk = Image.new("RGB", (800, 600), color=(40, 0, 0))
            _draw_caption(blk, f"{cam_id}  |  ERROR: {e}")
            pil = blk
            st = {"model": "error", "drawn_cuboids": 0, "total_cuboids": len(cuboid_edges), "drift_us": -1}
        out_path = args.output_dir / f"cam_{cam_id}.png"
        pil.save(out_path)
        print(
            f"[7cam] {cam_id:>40s}  model={st['model']:<10s}  "
            f"drawn={st['drawn_cuboids']}/{st['total_cuboids']}  "
            f"Δt={st['drift_us']/1000:.1f}ms  → {out_path.name}",
            flush=True,
        )
        panels.append(pil)
        stats.append({"camera_id": cam_id, **st})

    # 5. Tile into one grid PNG.
    grid = _tile_grid(panels, cols=args.cols)
    grid_path = args.output_dir / "out_grid.png"
    grid.save(grid_path)
    print(f"[7cam] grid saved: {grid_path}  ({grid.size[0]}×{grid.size[1]})", flush=True)

    # 6. Summary line — quick "did anything project?" sanity gate.
    n_drawn = sum(s["drawn_cuboids"] for s in stats)
    n_seen = sum(1 for s in stats if s["drawn_cuboids"] > 0)
    print(
        f"[7cam] SUMMARY: {n_seen}/{len(stats)} cameras saw at least 1 cuboid; "
        f"total drawn cuboid-overlays = {n_drawn}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
