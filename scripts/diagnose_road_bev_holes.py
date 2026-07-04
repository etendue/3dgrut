#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 2A — quantify BEV road-surface holes on a LayeredGaussians ckpt.

Splits "the road looks holey from straight above" into two measurable modes
over the ego-corridor (see ``threedgrut_playground/utils/bev_holes.py``):

  * **B-type geometry hole** — corridor cell with no road particle at all.
  * **A-type transparency hole** — corridor cell that has road particle(s) but
    the strongest is below an opacity floor (present yet invisible top-down).

Because the *full* render stacks road over background, a cell only looks like a
real hole in viser if NEITHER road NOR bg covers it. So we report three passes:

    road-only   — the road layer's own coverage (the thing we want to fix)
    bg-only     — how much background is currently carrying the road region
    road ∪ bg   — what the user actually sees top-down (combined coverage)

CPU-only (Mac / ThinkPad / A800). No NCore SDK / GPU needed — reads particle
tensors straight out of the ckpt via diag_ckpt.

Usage:
    python scripts/diagnose_road_bev_holes.py \\
        --gs_object /path/to/ckpt_last.pt \\
        --out_dir   /tmp/road_holes \\
        --cell_size 0.5 --corridor_half_width 12 \\
        --opacity_floors 0.05,0.1,0.3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _extract_layer_xy_opacity(ckpt: dict, layer: str):
    """Return (xy [N,2] float64, opacity [N] float64) for a layer, or (None, None)."""
    model_block = ckpt.get("model") if isinstance(ckpt.get("model"), dict) else None
    if model_block is not None and isinstance(model_block.get("gaussians_nodes"), dict):
        nodes = model_block["gaussians_nodes"]
    elif isinstance(ckpt.get("gaussians_nodes"), dict):
        nodes = ckpt["gaussians_nodes"]
    else:
        raise RuntimeError("ckpt has no 'gaussians_nodes' block.")
    payload = nodes.get(layer)
    if not isinstance(payload, dict):
        return None, None

    import torch

    pos = payload.get("positions")
    if pos is None:
        return None, None
    xy = pos.detach().cpu().numpy().reshape(-1, 3)[:, :2].astype(np.float64)

    dens = None
    for k in ("density", "densities", "opacity", "opacities"):
        if k in payload and payload[k] is not None:
            dens = payload[k]
            break
    if dens is None:
        raise RuntimeError(f"layer '{layer}' has no density field; keys={list(payload)}")
    dens = dens.detach().cpu().numpy().reshape(-1)
    opacity = _sigmoid(dens)  # density_activation = sigmoid (configs/base_gs.yaml)
    return xy, opacity


def _ego_xy_from_ckpt(ckpt: dict) -> np.ndarray:
    from threedgrut_playground.utils.viz4d_metadata import FourDMetadata

    meta = FourDMetadata.from_ckpt(ckpt)
    if meta is None:
        raise RuntimeError("ckpt has no viz_4d block → cannot recover ego trajectory.")
    c2w = np.asarray(meta.ego_poses_c2w)  # [F, 4, 4]
    return c2w[:, :3, 3][:, :2].astype(np.float64)


def _render_heatmaps(stats, ego_xy, out_dir: Path, tag: str, floor_for_cov: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x0, y0 = stats.x0, stats.y0
    cs = stats.cell_size
    extent = [x0, x0 + stats.nx * cs, y0, y0 + stats.ny * cs]
    # grids are [nx, ny] (axis0=x); imshow wants [row=y, col=x] → transpose
    count = stats.count_grid.T.astype(float)
    maxop = stats.maxop_grid.T
    corr = stats.corridor_mask_grid.T

    # mask out-of-corridor cells to NaN for clarity
    count_c = np.where(corr, count, np.nan)
    maxop_c = np.where(corr, maxop, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, grid, title, vmax in (
        (axes[0], count_c, f"{tag}: road particle count / cell", None),
        (axes[1], maxop_c, f"{tag}: max opacity / cell (floor={floor_for_cov:g})", 1.0),
    ):
        im = ax.imshow(grid, origin="lower", extent=extent, aspect="equal", cmap="viridis", vmax=vmax)
        ax.plot(ego_xy[:, 0], ego_xy[:, 1], "-", color="red", lw=1.2, label="ego")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("world X (m)")
        ax.set_ylabel("world Y (m)")
        ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = out_dir / f"heatmap_{tag}.png"
    fig.savefig(str(out_path), dpi=110)
    plt.close(fig)
    return out_path


def main(argv: Optional[list[str]] = None) -> int:
    from threedgrut_playground.utils.bev_holes import compute_bev_hole_stats
    from threedgrut_playground.utils.diag_ckpt import load_ckpt_cpu

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gs_object", required=True, type=str)
    p.add_argument("--out_dir", required=True, type=str)
    p.add_argument("--cell_size", default=0.5, type=float)
    p.add_argument("--corridor_half_width", default=12.0, type=float)
    p.add_argument("--opacity_floors", default="0.05,0.1,0.3", type=str)
    p.add_argument("--no_heatmap", action="store_true")
    args = p.parse_args(argv)

    floors = tuple(float(f) for f in args.opacity_floors.split(",") if f.strip())
    ckpt_path = Path(args.gs_object).expanduser().resolve()
    if not ckpt_path.exists():
        print(f"[road-holes] ERROR: ckpt not found: {ckpt_path}", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[road-holes] loading {ckpt_path}", flush=True)
    ckpt = load_ckpt_cpu(ckpt_path)
    ego_xy = _ego_xy_from_ckpt(ckpt)

    road_xy, road_op = _extract_layer_xy_opacity(ckpt, "road")
    if road_xy is None:
        print("[road-holes] ERROR: no 'road' layer in ckpt", file=sys.stderr)
        return 3
    bg_xy, bg_op = _extract_layer_xy_opacity(ckpt, "background")

    kw = dict(cell_size=args.cell_size, corridor_half_width=args.corridor_half_width, opacity_floors=floors)

    print(
        f"[road-holes] ego frames={ego_xy.shape[0]}  road N={road_xy.shape[0]}  "
        f"bg N={0 if bg_xy is None else bg_xy.shape[0]}",
        flush=True,
    )

    results = {}
    results["road"] = compute_bev_hole_stats(road_xy, road_op, ego_xy, **kw)
    if bg_xy is not None:
        results["bg"] = compute_bev_hole_stats(bg_xy, bg_op, ego_xy, **kw)
        comb_xy = np.concatenate([road_xy, bg_xy], axis=0)
        comb_op = np.concatenate([road_op, bg_op], axis=0)
        results["road_union_bg"] = compute_bev_hole_stats(comb_xy, comb_op, ego_xy, **kw)

    # ---- print headline table ----
    f0 = floors[0]
    fkey = f"{f0:g}"
    print("\n=== Phase 2A BEV road-hole quantification ===", flush=True)
    print(f"cell={args.cell_size}m  corridor=±{args.corridor_half_width}m  " f"floors={list(floors)}\n", flush=True)
    rd = results["road"]
    print("road layer opacity percentiles (sigmoid(density)):", flush=True)
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in rd.opacity_percentiles.items()), flush=True)
    print(
        f"\n{'pass':<16}{'corr_cells':>11}{'B_geom_hole':>13}" f"{'A_transp@'+fkey:>13}{'opaque_cov@'+fkey:>14}",
        flush=True,
    )
    for name in ("road", "bg", "road_union_bg"):
        s = results.get(name)
        if s is None:
            continue
        print(
            f"{name:<16}{s.n_corridor_cells:>11d}{s.b_geometry_hole_rate:>13.3f}"
            f"{s.a_transparency_hole_rate[fkey]:>13.3f}{s.opaque_coverage[fkey]:>14.3f}",
            flush=True,
        )
    # full opaque-coverage sweep across floors for the road & combined passes
    print("\nopaque coverage by floor (fraction of corridor rendered opaque):", flush=True)
    for name in ("road", "road_union_bg"):
        s = results.get(name)
        if s is None:
            continue
        cov = "  ".join(f"f={k}:{v:.3f}" for k, v in s.opaque_coverage.items())
        print(f"  {name:<14} {cov}", flush=True)

    # ---- dump JSON ----
    stats_json = {name: s.to_dict() for name, s in results.items()}
    stats_json["_ckpt"] = str(ckpt_path)
    (out_dir / "road_hole_stats.json").write_text(json.dumps(stats_json, indent=2))
    print(f"\n[road-holes] wrote {out_dir/'road_hole_stats.json'}", flush=True)

    # ---- heatmaps ----
    if not args.no_heatmap:
        for name in ("road", "road_union_bg"):
            s = results.get(name)
            if s is None:
                continue
            hp = _render_heatmaps(s, ego_xy, out_dir, name, f0)
            print(f"[road-holes] wrote {hp}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
