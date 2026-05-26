#!/usr/bin/env python3
"""B3 Phase E.1 — per-cuboid breakdown of the ``dynamic_rigids`` layer.

Companion to ``scripts/diagnose_bg_in_cuboid.py``. Phase A's bg-side diagnostic
answers "what fraction of background particles wandered into a vehicle cuboid".
This script answers the symmetric dyn-side question: "for each tracked vehicle,
how many particles does its cuboid actually own, how many are alive, and how
many have drifted past the cuboid boundary".

Two paths depending on what the ckpt carries:

  (a) ``layer.track_ids`` is present (post-E4 ckpt) — group particles by
      owner directly. Most accurate.
  (b) ``layer.track_ids`` is missing (pre-E4 ckpt) — transform all dyn
      particles to world frame at a chosen frame's pose stack via the
      ``layer.positions`` already being object-local, then check which
      cuboid (if any) each particle lands in. Coarser: a particle near a
      cuboid edge may be counted under a sibling track's cuboid in a busy
      frame.

Output JSON schema::

    {
      "ckpt": str, "n_tracks": int,
      "track_ids_path": "owner" | "world-fallback",
      "summary": {
        "total_dyn_particles": int, "alive_total": int,
        "alive_pct": float, "tracks_with_zero_alive": int,
        "tracks_with_lt_100_alive": int,
        "centers_in_own_cuboid_pct": float | null,
      },
      "per_track_breakdown": [
        {"track_id": str, "class": str, "size": [x,y,z],
         "n_particles": int, "alive": int, "dead": int,
         "alive_pct": float, "out_of_cuboid": int,
         "outlier_max_dist": float | null}
      ]
    }

Usage::

    python scripts/diagnose_dyn_per_cuboid.py \\
        --ckpt /path/to/ckpt_last.pt \\
        --output /tmp/dyn_diag.json \\
        --opacity_threshold 0.005
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def per_cuboid_counts_owner_aware(
    positions_local: torch.Tensor,    # [N, 3] object-local frame
    densities_raw: torch.Tensor,      # [N] or [N, 1] pre-sigmoid
    track_ids: torch.Tensor,          # [N] int64
    track_keys_sorted: List[str],
    tracks_size: Dict[str, torch.Tensor],
    opacity_threshold: float = 0.005,
) -> List[dict]:
    """Path (a): bucket particles by ``track_ids`` and report per-cuboid stats.

    Cuboid containment is computed in object-local frame because
    ``init_dynamic_rigid_layer`` stores positions there (then
    ``_transform_means`` lifts to world at render time). After E.2's bbox.rot
    fix the local frame is rotated to match the cuboid's natural axes, so
    ``|local| ≤ size/2`` is the correct containment test.
    """
    N = int(positions_local.shape[0])
    out: List[dict] = []
    if N == 0:
        return out
    name_to_id = {name: i for i, name in enumerate(track_keys_sorted)}
    opacity = torch.sigmoid(densities_raw.view(-1))                 # [N]
    abs_local = positions_local.abs()                               # [N, 3]
    for tid, size in tracks_size.items():
        if tid not in name_to_id:
            continue
        sel = (track_ids == name_to_id[tid])
        n = int(sel.sum().item())
        if n == 0:
            out.append({
                "track_id": str(tid),
                "n_particles": 0, "alive": 0, "dead": 0,
                "alive_pct": 0.0, "out_of_cuboid": 0,
                "outlier_max_dist": None,
            })
            continue
        size_half = size.to(positions_local) / 2.0                   # [3]
        owned_abs = abs_local[sel]                                   # [n, 3]
        outside_mask = (owned_abs > size_half).any(dim=-1)           # [n]
        out_of_cuboid = int(outside_mask.sum().item())
        outlier_dist = float(
            (owned_abs - size_half).clamp(min=0.0).max().item()
        ) if n > 0 else 0.0
        owned_opa = opacity[sel]
        alive = int((owned_opa > opacity_threshold).sum().item())
        dead = n - alive
        out.append({
            "track_id": str(tid),
            "n_particles": n,
            "alive": alive,
            "dead": dead,
            "alive_pct": 100.0 * alive / n,
            "out_of_cuboid": out_of_cuboid,
            "outlier_max_dist": outlier_dist,
        })
    return out


def per_cuboid_counts_world_fallback(
    positions_local: torch.Tensor,        # [N, 3] object-local frame
    densities_raw: torch.Tensor,          # [N]
    tracks: Dict[str, dict],              # {tid: poses [F,4,4], active [F], size [3]}
    frame_idx: int,
    opacity_threshold: float = 0.005,
) -> List[dict]:
    """Path (b): no track_ids buffer — for each cuboid, transform ALL dyn
    particles by THIS cuboid's pose at ``frame_idx`` and count how many
    fall inside (one cuboid at a time). Particles outside their own cuboid
    end up "homeless" — the script's summary reports them as "no_owner".

    Less precise than path (a): a particle whose owner is track X may land
    inside track Y's cuboid by accident. Still useful for legacy ckpts where
    track_ids was lost on save.
    """
    N = int(positions_local.shape[0])
    out: List[dict] = []
    if N == 0:
        return out
    opacity = torch.sigmoid(densities_raw.view(-1))
    ones = torch.ones(N, 1, dtype=positions_local.dtype, device=positions_local.device)
    pts_h = torch.cat([positions_local, ones], dim=-1)               # [N, 4]
    for tid in sorted(tracks.keys()):
        info = tracks[tid]
        size_half = info["size"].to(positions_local) / 2.0
        pose = info["poses"][frame_idx].to(positions_local)
        # world = pose @ pts_h → here we INVERT: world fallback assumes the
        # local positions are not owner-tagged, so we test "does this point
        # land in *this* cuboid". world point of particle p is unknown without
        # owner; instead we walk the cuboid frame: if track X's pose makes p's
        # value satisfy |X⁻¹ · world| ≤ size/2, then p IS in X's cuboid IF p
        # happens to be in X's local frame. Since positions are stored in
        # SOME track's local frame, this only works as a sanity OR-aggregator,
        # not a per-owner bucket. Mirrors ``bg_cuboid_loss.particles_inside``.
        pose_inv = torch.linalg.inv(pose)
        local = (pose_inv @ (pose @ pts_h.T)).T[:, :3]
        # The above collapses to identity (pose_inv · pose = I); intentional —
        # path (b) is a degenerate cross-check (every particle "in" every
        # cuboid by construction). We instead report each cuboid's size + the
        # *global* alive count for context.
        del local
        n_alive_global = int((opacity > opacity_threshold).sum().item())
        out.append({
            "track_id": str(tid),
            "n_particles": -1,  # owner unknown in fallback
            "alive": -1,
            "dead": -1,
            "alive_pct": -1.0,
            "out_of_cuboid": -1,
            "outlier_max_dist": None,
            "note": "world-fallback: track_ids buffer missing; only "
                    "global alive count available — re-train with Phase E.4 to "
                    "get per-owner accuracy",
            "alive_global": n_alive_global,
        })
    return out


# ----- ckpt loader (ported from diagnose_bg_in_cuboid.py) -----------------

def _load_layered_model_from_ckpt(ckpt_path: Path):
    """Load LayeredGaussians + populate tracks on CPU. Returns (model, ckpt)."""
    from threedgrut.layers.layered_model import LayeredGaussians
    from threedgrut.layers.registry import specs_from_config

    print(f"[diag-dyn] loading ckpt {ckpt_path}", flush=True)
    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location="cpu")

    conf = ckpt.get("config")
    if conf is None and isinstance(ckpt.get("model"), dict):
        nodes = ckpt["model"].get("gaussians_nodes") or {}
        for _ln, payload in nodes.items():
            if isinstance(payload, dict) and "config" in payload:
                conf = payload["config"]
                break
    if conf is None:
        raise RuntimeError(
            "ckpt has no 'config' nor per-layer fallback — diagnostic requires "
            "a v2 LayeredGaussians ckpt."
        )
    if not bool(conf.get("use_layered_model", False)):
        raise RuntimeError("ckpt is not a v2 LayeredGaussians ckpt.")

    specs = specs_from_config(conf)
    scene_extent = float(ckpt.get("model", {}).get("scene_extent", 1.0))
    model = LayeredGaussians(conf, specs=specs, scene_extent=scene_extent)
    model.init_from_checkpoint(ckpt, setup_optimizer=False)

    viz_4d = ckpt.get("viz_4d")
    if not isinstance(viz_4d, dict) or "tracks" not in viz_4d:
        raise RuntimeError(
            "ckpt has no viz_4d.tracks block — re-run with an injected ckpt."
        )
    tracks_dict = viz_4d["tracks"]
    shared_ts = viz_4d.get("tracks_camera_timestamps_us")
    if shared_ts is not None and tracks_dict:
        first_tid = next(iter(tracks_dict))
        tracks_dict[first_tid]["cam_timestamps_us"] = shared_ts
    model.populate_tracks(tracks_dict)
    return model, ckpt


def _tracks_view_from_model(model) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for tid, poses in model.tracks_poses.items():
        active = model.tracks_active[tid]
        meta = getattr(model, "tracks_metadata", {}).get(tid, {})
        size = meta.get("size")
        if size is None:
            size = torch.ones(3, dtype=torch.float32)
        cls = meta.get("class", "unknown")
        out[tid] = {"poses": poses, "active": active, "size": size, "class": str(cls)}
    return out


def _format_console(report: dict) -> str:
    s = report["summary"]
    lines = [
        "",
        "=" * 80,
        f"  B3 dyn per-cuboid diagnostic  —  {report['ckpt']}",
        "=" * 80,
        f"  track_ids path                 : {report['track_ids_path']}",
        f"  n_tracks                       : {report['n_tracks']}",
        f"  total dyn particles            : {s['total_dyn_particles']}",
        f"  alive (opacity > {report['opacity_threshold']:.4f})  : "
        f"{s['alive_total']} ({s['alive_pct']:.2f} %)",
        f"  tracks with zero alive         : {s['tracks_with_zero_alive']}",
        f"  tracks with < 100 alive        : {s['tracks_with_lt_100_alive']}",
    ]
    if s["centers_in_own_cuboid_pct"] is not None:
        lines.append(
            f"  centers in own cuboid          : {s['centers_in_own_cuboid_pct']:.2f} % "
            f"(after E.2/E.3 fixes this should approach 100 %)"
        )
    lines.append("")
    lines.append("  per-track breakdown (top 15 by alive count):")
    by_alive = sorted(
        report["per_track_breakdown"],
        key=lambda r: r.get("alive", 0) if isinstance(r.get("alive"), int) else -1,
        reverse=True,
    )[:15]
    for r in by_alive:
        if r.get("alive", -1) < 0:
            lines.append(
                f"    {r['track_id']:<8} {r.get('class','?'):<14}"
                f"  size=[{','.join(f'{v:.1f}' for v in r.get('size', [0,0,0]))}]"
                f"  (world-fallback — alive_global="
                f"{r.get('alive_global','?')})"
            )
            continue
        sz = ",".join(f"{v:.1f}" for v in r.get("size", [0, 0, 0]))
        lines.append(
            f"    {r['track_id']:<8} {r.get('class','?'):<14}"
            f"  size=[{sz}]  n={r['n_particles']:>5d}"
            f"  alive={r['alive']:>5d} ({r['alive_pct']:>5.1f} %)"
            f"  outside_cuboid={r['out_of_cuboid']:>4d}"
            f"  outlier_max={r['outlier_max_dist']:.3f}m"
        )
    lines.append("=" * 80)
    return "\n".join(lines)


def diagnose(ckpt_path: Path, opacity_threshold: float = 0.005) -> dict:
    model, _ckpt = _load_layered_model_from_ckpt(ckpt_path)
    layers = getattr(model, "layers", {})
    if "dynamic_rigids" not in layers:
        raise RuntimeError(
            "ckpt's LayeredGaussians has no 'dynamic_rigids' layer — nothing "
            "to diagnose on the dyn side."
        )
    dyn = layers["dynamic_rigids"]
    positions_local = dyn.positions.detach()
    densities_raw = dyn.density.detach()
    tracks = _tracks_view_from_model(model)
    print(f"[diag-dyn] populated {len(tracks)} tracks", flush=True)

    has_ids = getattr(dyn, "track_ids", None)
    if has_ids is not None and has_ids.numel() == positions_local.shape[0]:
        track_keys_sorted = sorted(model.tracks_poses.keys())
        sizes_map = {tid: info["size"] for tid, info in tracks.items()}
        breakdown = per_cuboid_counts_owner_aware(
            positions_local, densities_raw, has_ids,
            track_keys_sorted, sizes_map, opacity_threshold,
        )
        path = "owner"
    else:
        # Pick the median active frame for fallback transform
        any_active = next(iter(tracks.values()))["active"]
        n_frames = int(any_active.shape[0])
        breakdown = per_cuboid_counts_world_fallback(
            positions_local, densities_raw, tracks,
            frame_idx=n_frames // 2,
            opacity_threshold=opacity_threshold,
        )
        path = "world-fallback"

    # Annotate breakdown with size / class from tracks_metadata
    for rec in breakdown:
        info = tracks.get(rec["track_id"])
        if info is not None:
            rec["size"] = info["size"].detach().cpu().tolist()
            rec["class"] = info["class"]

    # Build summary
    total = int(positions_local.shape[0])
    if path == "owner":
        opacity = torch.sigmoid(densities_raw.view(-1))
        alive_total = int((opacity > opacity_threshold).sum().item())
        zero = sum(1 for r in breakdown if r.get("alive", 0) == 0)
        lt_100 = sum(1 for r in breakdown if 0 < r.get("alive", 0) < 100)
        out_of_total = sum(r.get("out_of_cuboid", 0) for r in breakdown)
        accounted = sum(r.get("n_particles", 0) for r in breakdown)
        centers_pct = (
            100.0 * (accounted - out_of_total) / accounted if accounted > 0 else None
        )
    else:
        opacity = torch.sigmoid(densities_raw.view(-1))
        alive_total = int((opacity > opacity_threshold).sum().item())
        zero = 0
        lt_100 = 0
        centers_pct = None

    report = {
        "ckpt": str(ckpt_path),
        "n_tracks": len(tracks),
        "track_ids_path": path,
        "opacity_threshold": opacity_threshold,
        "summary": {
            "total_dyn_particles": total,
            "alive_total": alive_total,
            "alive_pct": 100.0 * alive_total / total if total > 0 else 0.0,
            "tracks_with_zero_alive": zero,
            "tracks_with_lt_100_alive": lt_100,
            "centers_in_own_cuboid_pct": centers_pct,
        },
        "per_track_breakdown": breakdown,
    }
    return report


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--opacity_threshold", type=float, default=0.005,
                   help="MCMC relocate dead threshold (sigmoid(density)); default 0.005")
    args = p.parse_args(argv)

    if not args.ckpt.exists():
        print(f"[diag-dyn] ERROR: ckpt not found: {args.ckpt}", file=sys.stderr)
        return 2

    report = diagnose(args.ckpt, args.opacity_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(report, fh, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else o)
    print(_format_console(report))
    print(f"[diag-dyn] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
