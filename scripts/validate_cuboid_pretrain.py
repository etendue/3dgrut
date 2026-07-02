#!/usr/bin/env python3
"""Pre-training cuboid geometry validator (NO checkpoint required).

Purpose: on a *new* NCore v4 clip, confirm that camera pose, cuboid pose and
the world<->camera transformation are all correct **before** spending any GPU
on training. It loads vehicle cuboid autolabels straight from the manifest via
``load_tracks_from_ncore_cuboids`` — the exact same path the ``dynamic_rigids``
layer uses at train time — then projects each active cuboid's 12-edge wireframe
onto the raw camera images and tiles the requested cameras side-by-side.

Why no ckpt (unlike ``validate_cuboid_7cam.py``): cuboid world poses come from
the manifest, and ``pose_adjustment`` is default-off, so a trained ckpt's
cuboid poses are identical to these. Verifying them here == verifying the
transformation the trainer will consume.

A "correct" output:
  - Cuboid edges hug the cars in every camera that sees them.
  - Pinhole cameras draw straight edges; FTheta (fisheye) cameras curve at the
    periphery (this clip's 3-cam subset is all pinhole).
  - A car visible in front_wide should sit at plausible left/right positions in
    cross_left / cross_right, not mirrored or 90-degrees-off.

Usage (inceptio)::

    python scripts/validate_cuboid_pretrain.py \\
        --config_name apps/ncore_3dgut_mcmc_multilayer \\
        --path ~/work/data/inc_b6a9ed61_20s/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9.json \\
        --camera_ids camera_front_wide_120fov camera_cross_left_120fov camera_cross_right_120fov \\
        --n_frames 4 \\
        --output_dir /tmp/cuboid_pretrain_inc_b6a9
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from PIL import Image  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
import ncore.data as _nd  # noqa: E402

# Reuse the battle-tested projection / drawing / tiling helpers from the 7-cam
# validator so the per-camera projection math is byte-identical.
from validate_cuboid_7cam import _render_camera_view, _tile_grid  # noqa: E402

from threedgrut.datasets.tracks_loader import load_tracks_from_ncore_cuboids  # noqa: E402
from threedgrut_playground.utils.cuboid import cuboid_world_edges, instance_color  # noqa: E402

_CONFIG_DIR = str(_REPO_ROOT / "configs")


def _edges_at_frame(tracks: dict, frame_idx: int
                    ) -> list[tuple[str, np.ndarray, tuple]]:
    """Build a ``validate_cuboid_7cam``-style ``[(tid, (12,2,3), color)]`` list
    from a ``tracks`` dict, keeping only tracks active at ``frame_idx``."""
    out: list[tuple[str, np.ndarray, tuple]] = []
    for tid, t in tracks.items():
        fi = t["frame_info"]
        if frame_idx >= int(fi.shape[0]) or not bool(fi[frame_idx]):
            continue
        pose = t["poses"][frame_idx].cpu().numpy().astype(np.float64)
        size = t["size"].cpu().numpy().astype(np.float64)
        edges = cuboid_world_edges(pose, size)  # (12, 2, 3)
        col_f = instance_color(tid)
        col = tuple(int(round(c * 255)) for c in col_f)
        out.append((tid, edges, col))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config_name", required=True,
                   help="Hydra config (e.g. apps/ncore_3dgut_mcmc_multilayer)")
    p.add_argument("--path", required=True, help="NCore manifest json path")
    p.add_argument("--camera_ids", nargs="+", required=True,
                   help="cameras to render, first one drives the frame sampling")
    p.add_argument("--class_filter", nargs="+", default=None,
                   help="cuboid classes to keep (default: automobile/heavy_truck/bus)")
    p.add_argument("--n_frames", type=int, default=4,
                   help="how many evenly-spaced reference frames to sample")
    p.add_argument("--ts_mode", default="ref_nearest",
                   choices=["ref_nearest", "per_camera_interp"],
                   help="A2: ref_nearest = legacy (ref-camera timeline, nearest "
                        "obs); per_camera_interp = union timeline + lerp/slerp — "
                        "each camera panel draws the pose interpolated at its "
                        "OWN frame END timestamp")
    p.add_argument("--output_dir", required=True, type=Path)
    args = p.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cam_override = "[" + ",".join(args.camera_ids) + "]"
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        conf = compose(
            config_name=args.config_name,
            overrides=[
                f"path={args.path}",
                f"dataset.camera_ids={cam_override}",
                "trainer.sky_backend=mlp",
            ],
        )

    import threedgrut.datasets as datasets
    dataset, _val = datasets.make(conf.dataset.type, conf, ray_jitter=None)
    seq_id = dataset.sequence_id
    loader = dataset.sequence_loaders[seq_id]
    print(f"[pretrain] dataset ready | cameras = {dataset.camera_ids}", flush=True)

    # Reference camera (first requested) drives the tracks pose index: its END
    # timestamps are the [F] array we sample cuboid observations against.
    ref_cam = args.camera_ids[0]
    ref_sensor = dataset.sequence_camera_sensors[seq_id][ref_cam]
    ref_ts_end = np.asarray(
        ref_sensor.frames_timestamps_us[:, _nd.FrameTimepoint.END], dtype=np.int64)
    F = int(ref_ts_end.shape[0])

    kw = {}
    if args.class_filter:
        kw["class_filter"] = frozenset(args.class_filter)
    from threedgrut.datasets.tracks_loader import (
        CUBOID_TS_MODES,
        build_cuboid_frame_timeline_us,
    )
    if args.ts_mode == "per_camera_interp":
        timeline = np.asarray(
            build_cuboid_frame_timeline_us(dataset, args.ts_mode), dtype=np.int64)
    else:
        timeline = ref_ts_end  # legacy: unwindowed ref-camera END timestamps
    tracks = load_tracks_from_ncore_cuboids(
        loader, timeline, pose_time_mode=CUBOID_TS_MODES[args.ts_mode], **kw)
    classes = sorted({t["class"] for t in tracks.values()})
    n_active = sum(int(t["frame_info"].sum()) for t in tracks.values())
    print(f"[pretrain] tracks={len(tracks)} classes={classes} "
          f"total_active_(track,frame)={n_active} ref_frames={F}", flush=True)
    if not tracks:
        print("[pretrain] WARNING: no cuboids after class filter — nothing to draw.",
              flush=True)

    idxs = sorted(set(np.linspace(0, F - 1, args.n_frames, dtype=int).tolist()))
    grid_summ: list[str] = []
    for fi in idxs:
        t_us = int(ref_ts_end[fi])
        cuboid_edges = _edges_at_frame(tracks, fi) if args.ts_mode == "ref_nearest" else []
        panels: list[Image.Image] = []
        for cam_id in args.camera_ids:
            if args.ts_mode == "per_camera_interp":
                # A2: draw the pose interpolated at THIS camera's own frame END
                # timestamp (nearest to the ref sample time), read back from the
                # union timeline the tracks were populated on.
                sensor_c = dataset.sequence_camera_sensors[seq_id][cam_id]
                ts_c = np.asarray(
                    sensor_c.frames_timestamps_us[:, _nd.FrameTimepoint.END],
                    dtype=np.int64)
                tc = int(ts_c[int(np.argmin(np.abs(ts_c - t_us)))])
                uidx = int(np.clip(
                    np.searchsorted(timeline, tc), 0, len(timeline) - 1))
                edges_cam = _edges_at_frame(tracks, uidx)
                query_us = tc
            else:
                edges_cam, query_us = cuboid_edges, t_us
            try:
                pil, st = _render_camera_view(dataset, cam_id, query_us, edges_cam)
            except Exception as e:  # keep the grid aligned on a single-cam error
                blk = Image.new("RGB", (960, 540), color=(40, 0, 0))
                from validate_cuboid_7cam import _draw_caption
                _draw_caption(blk, f"{cam_id} | ERROR: {e}")
                pil, st = blk, {"model": "error", "drawn_cuboids": 0,
                                "total_cuboids": len(edges_cam), "drift_us": -1}
            panels.append(pil)
            print(f"[pretrain] f{fi:03d} t={t_us} {cam_id:>32s} "
                  f"model={st['model']:<9s} drawn={st['drawn_cuboids']}/{st['total_cuboids']} "
                  f"dt={st['drift_us']/1000:.1f}ms", flush=True)
        grid = _tile_grid(panels, cols=len(args.camera_ids))
        gp = args.output_dir / f"grid_f{fi:03d}_t{t_us}.png"
        grid.save(gp)
        grid_summ.append(str(gp))
        print(f"[pretrain] saved {gp}  ({grid.size[0]}x{grid.size[1]})", flush=True)

    print(f"[pretrain] DONE — {len(grid_summ)} grids in {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
