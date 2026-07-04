#!/usr/bin/env python3
"""V3-VIZ.1 — BEV diagnostic for v2 LayeredGaussians checkpoints.

Renders one BEV PNG per ego frame, layering:

  * ego trajectory (green polyline) + current ego marker + heading arrow
  * cuboid footprints (per-class color outline) with ``t<tid> | <class>`` labels
  * per-layer Gaussian centers (background=gray, road=blue, dynamic_rigids=red,
    dynamic_deformables=yellow; sky_envmap has no particles)

Purpose: user reviews PNG sequences to spot
  (1) cuboid fit on dynamic objects
  (2) "dynamic Gaussians erroneously in background layer" — bg gray dots inside
      a red cuboid footprint = misclassification
  (3) "road Gaussians erroneously in background layer" — bg gray dots covering
      what should be road (compare with blue road dots)

CPU-only (Mac / ThinkPad). Loads ckpt on CPU, projects all positions, calls
``bev_renderer`` per frame. Recommended frame budget: 5-20 frames for spot
checks, full sequence for batch review.

Usage:

    python scripts/diagnose_layered_bev.py \\
        --gs_object /path/to/ckpt_last.pt \\
        --out_dir /tmp/bev_diag \\
        --frame_range 0:20 \\
        --xy_range_m 60 \\
        --dpi 100

Output: ``<out_dir>/bev_<frame_idx:04d>.png`` and a manifest ``manifest.json``.
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
    """Parse ``"0:20"`` / ``"5"`` / ``""`` / None → list of frame indices."""
    if spec is None or spec.strip() == "":
        return list(range(n_frames))
    if ":" in spec:
        lo_s, hi_s = spec.split(":", 1)
        lo = int(lo_s) if lo_s else 0
        hi = int(hi_s) if hi_s else n_frames
        lo = max(0, lo)
        hi = min(n_frames, hi)
        return list(range(lo, hi))
    # Single frame.
    f = int(spec)
    if 0 <= f < n_frames:
        return [f]
    return []


def _frame_idx_for_ego(meta, frame_idx: int) -> int:
    """Clamp ego frame index. ego frames and dynamic frames share the same
    timeline post T8.13 (ego_frame_timestamps_us aligned w/ tracks_camera_timestamps_us
    for primary camera)."""
    return max(0, min(int(frame_idx), int(meta.ego_poses_c2w.shape[0]) - 1))


def diagnose(args: argparse.Namespace) -> int:
    import imageio.v3 as iio

    from threedgrut_playground.utils.bev_renderer import (
        build_inputs_from_metadata,
        render_bev_frame,
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
        print(f"[bev-diag] ERROR: ckpt not found: {ckpt_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bev-diag] loading ckpt → {ckpt_path}", flush=True)
    ckpt = load_ckpt_cpu(ckpt_path)
    meta = FourDMetadata.from_ckpt(ckpt)
    if meta is None:
        print("[bev-diag] ERROR: ckpt has no viz_4d block", file=sys.stderr)
        return 3

    layer_positions = extract_layer_positions(ckpt)
    static_layers = {n: p for n, p in layer_positions.items() if n != "dynamic_rigids"}
    dyn_local = layer_positions.get("dynamic_rigids")
    dyn_track_ids = extract_dyn_track_ids(ckpt)
    sorted_track_names = sorted(meta.tracks.keys())
    if dyn_local is not None and dyn_local.size > 0 and dyn_track_ids is None:
        print(
            f"[bev-diag] WARNING: dynamic_rigids has {dyn_local.shape[0]} particles "
            "but no track_ids buffer in ckpt — pre-T8/B3-Phase-E.4 ckpts cannot map "
            "particles to tracks, so dyn_rigids will NOT be drawn. Use a ckpt saved "
            "after commit 92dbf42 (B3_30k) for full dyn visualization.",
            flush=True,
        )

    n_frames_ego = int(meta.ego_poses_c2w.shape[0])
    frames = _parse_frame_range(args.frame_range, n_frames_ego)
    if not frames:
        print(
            f"[bev-diag] ERROR: no frames in range '{args.frame_range}' " f"(n_frames={n_frames_ego})", file=sys.stderr
        )
        return 4

    print(
        f"[bev-diag] sequence={meta.sequence_id}  n_frames_ego={n_frames_ego}  " f"n_tracks={meta.n_tracks()}",
        flush=True,
    )
    print(f"[bev-diag] layer particle counts:", flush=True)
    for name, p in layer_positions.items():
        print(f"  - {name:<22} {p.shape[0]:>10d}", flush=True)
    print(f"[bev-diag] rendering {len(frames)} frames → {out_dir}", flush=True)

    manifest: list[dict] = []
    for fi in frames:
        ego_fi = _frame_idx_for_ego(meta, fi)

        # Build per-frame layer positions: copy static layers, derive dyn from local.
        frame_layer_positions = dict(static_layers)
        if dyn_local is not None and dyn_track_ids is not None:
            frame_layer_positions["dynamic_rigids"] = dyn_local_to_world_at_frame(
                dyn_local,
                dyn_track_ids,
                sorted_track_names,
                meta.tracks,
                ego_fi,
            )

        inputs = build_inputs_from_metadata(
            meta,
            frame_layer_positions,
            ego_fi,
            z_window_m=args.z_window_m,
        )

        title = f"frame {ego_fi}/{n_frames_ego - 1}  " f"seq={meta.sequence_id}  " f"cuboids={len(inputs.cuboids)}"
        img = render_bev_frame(
            inputs,
            xy_range_m=args.xy_range_m,
            grid_step_m=args.grid_step_m,
            dpi=args.dpi,
            show_labels=not args.no_labels,
            title=title,
        )
        out_path = out_dir / f"bev_{ego_fi:04d}.png"
        iio.imwrite(str(out_path), img)
        manifest.append(
            {
                "frame_idx": ego_fi,
                "out_path": str(out_path.relative_to(out_dir)),
                "n_cuboids": len(inputs.cuboids),
                "n_bg_pts": int(inputs.layer_positions_xy.get("background", np.empty((0, 2))).shape[0]),
                "n_road_pts": int(inputs.layer_positions_xy.get("road", np.empty((0, 2))).shape[0]),
                "n_dyn_pts": int(inputs.layer_positions_xy.get("dynamic_rigids", np.empty((0, 2))).shape[0]),
            }
        )
        if args.verbose:
            m = manifest[-1]
            print(
                f"[bev-diag]   frame {ego_fi:>4d}: cuboids={m['n_cuboids']:>3d}  "
                f"bg={m['n_bg_pts']:>7d}  road={m['n_road_pts']:>6d}  "
                f"dyn={m['n_dyn_pts']:>5d}  → {out_path.name}",
                flush=True,
            )

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w") as fh:
        json.dump(
            {
                "ckpt": str(ckpt_path),
                "sequence_id": meta.sequence_id,
                "xy_range_m": args.xy_range_m,
                "z_window_m": args.z_window_m,
                "n_frames": len(manifest),
                "frames": manifest,
            },
            fh,
            indent=2,
        )

    print(f"[bev-diag] wrote {len(manifest)} PNGs + {manifest_path}", flush=True)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gs_object", required=True, type=str, help="Path to v2 LayeredGaussians ckpt (.pt).")
    parser.add_argument("--out_dir", required=True, type=str, help="Output directory for BEV PNGs + manifest.json.")
    parser.add_argument(
        "--frame_range", default=None, type=str, help='Frame slice e.g. "0:20", "10", or empty for all.'
    )
    parser.add_argument(
        "--xy_range_m", default=60.0, type=float, help="Half-width of BEV square around current ego (default 60 m)."
    )
    parser.add_argument("--grid_step_m", default=10.0, type=float, help="Background grid step (default 10 m).")
    parser.add_argument(
        "--z_window_m", default=10.0, type=float, help="|z - ego_z| filter for Gaussian scatter (default 10 m)."
    )
    parser.add_argument("--dpi", default=100, type=int)
    parser.add_argument("--no_labels", action="store_true", help="Hide cuboid labels (faster + less clutter).")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    return diagnose(args)


if __name__ == "__main__":
    sys.exit(main())
