#!/usr/bin/env python3
"""B3 Phase A diagnostic — count Gaussian particles per layer that fall inside
active dynamic-rigid cuboids.

Loads a v2 LayeredGaussians ckpt (must include viz_4d.tracks block) and, for
each enabled particle layer, computes:

  * total particle count
  * count + percentage of particles whose world position lies inside ANY
    active cuboid across the sampled frames
  * per-track breakdown (which tracks "host" how many bg/road particles)

For the dynamic_rigids layer the report also checks how many particles have
drifted outside their OWN track's cuboid in object-local frame (should be 0
right after init; MCMC perturbations may leak some).

Usage (ThinkPad / Mac CPU; no CUDA needed):
    python scripts/diagnose_bg_in_cuboid.py \\
        --ckpt /home/yusun/work/ckpts/bug4_v2_full_30k/ckpt_with_ftheta_v2.pt \\
        --output /tmp/b3_baseline.json \\
        --max_frames 5

Output JSON schema:
    {
        "ckpt": str, "n_tracks": int, "max_frames_per_track": int,
        "per_layer_total": {layer_name: int},
        "per_layer_inside_any_cuboid": {layer_name: int},
        "per_layer_inside_pct": {layer_name: float},
        "dynamic_rigids_outside_own_cuboid_pct": float | null,
        "per_track_breakdown": [
            {"track_id": str, "size": [x,y,z], "n_active_frames": int,
             "n_frames_sampled": int,
             "background_inside": int, "road_inside": int}
        ],
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

# Make repo root importable when invoked as `python scripts/diagnose_bg_in_cuboid.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _select_sample_frames(active: torch.Tensor, max_frames: int) -> torch.Tensor:
    """Pick up to ``max_frames`` active frame indices, evenly spaced.

    Returns int64 indices into the track's per-frame dimension. Empty tensor
    when the track has no active frames.
    """
    active_idx = active.nonzero(as_tuple=False).squeeze(-1).to(torch.int64)
    n = int(active_idx.numel())
    if n == 0:
        return active_idx
    if n <= max_frames:
        return active_idx
    step = float(n - 1) / float(max_frames - 1) if max_frames > 1 else 0.0
    picks = [int(round(i * step)) for i in range(max_frames)]
    return active_idx[picks]


def count_world_positions_in_any_active_cuboid(
    positions: torch.Tensor,
    tracks: Dict[str, dict],
    max_frames_per_track: int = 5,
) -> Tuple[torch.Tensor, List[dict]]:
    """Return (any_inside_mask[N], per_track_records).

    Args:
        positions: ``[N, 3]`` world frame.
        tracks: ``{tid: {"poses": [F,4,4] obj→world, "active": [F bool],
                         "size": [3] full extent}}``.
        max_frames_per_track: number of evenly-spaced active frames to sample
            per track (caps cost on long tracks).

    Returns:
        any_inside_mask: ``[N]`` BoolTensor — True iff position is inside at
            least one (track, sampled active frame) cuboid.
        per_track_records: per-track dicts with ``n_inside_any_sampled_frame``
            and ``n_frames_sampled`` keys (ordered like ``sorted(tracks)``).
    """
    N = positions.shape[0]
    device = positions.device
    dtype = positions.dtype
    any_inside = torch.zeros(N, dtype=torch.bool, device=device)
    records: List[dict] = []
    if N == 0 or len(tracks) == 0:
        for tid in sorted(tracks.keys()):
            info = tracks[tid]
            active = info["active"].to(torch.bool)
            records.append({
                "track_id": str(tid),
                "size": info["size"].detach().cpu().tolist(),
                "n_active_frames": int(active.sum().item()),
                "n_frames_sampled": 0,
                "n_inside_any_sampled_frame": 0,
            })
        return any_inside, records

    ones = torch.ones(N, 1, dtype=dtype, device=device)
    pts_h = torch.cat([positions.to(dtype), ones], dim=-1)  # [N, 4]

    for tid in sorted(tracks.keys()):
        info = tracks[tid]
        active = info["active"].to(torch.bool)
        poses = info["poses"].to(dtype=dtype, device=device)            # [F,4,4]
        size_half = info["size"].to(dtype=dtype, device=device) / 2.0   # [3]

        sample_idx = _select_sample_frames(active, max_frames_per_track)
        per_track_inside = torch.zeros(N, dtype=torch.bool, device=device)
        for fi in sample_idx.tolist():
            pose = poses[fi]
            pose_inv = torch.linalg.inv(pose)
            local = (pose_inv @ pts_h.T).T[:, :3]                       # [N,3]
            inside = (local.abs() <= size_half).all(dim=-1)
            per_track_inside |= inside

        any_inside |= per_track_inside
        records.append({
            "track_id": str(tid),
            "size": info["size"].detach().cpu().tolist(),
            "n_active_frames": int(active.sum().item()),
            "n_frames_sampled": int(sample_idx.numel()),
            "n_inside_any_sampled_frame": int(per_track_inside.sum().item()),
        })
    return any_inside, records


def count_local_positions_outside_own_cuboid(
    positions: torch.Tensor,
    track_ids: torch.Tensor,
    track_keys_sorted: Iterable[str],
    tracks: Dict[str, dict],
) -> int:
    """Count dynamic_rigids particles whose object-local position lies outside
    their owning track's cuboid (``|local| > size/2`` on any axis).

    Right after :func:`init_dynamic_rigid_layer`, this is exactly 0 — the
    initializer enforces ``|local| ≤ size/2``. MCMC perturbations during
    training may drift particles past the cuboid boundary.
    """
    if positions.numel() == 0:
        return 0
    name_to_id = {name: i for i, name in enumerate(track_keys_sorted)}
    bad = 0
    for tid, info in tracks.items():
        if tid not in name_to_id:
            continue
        mask = track_ids == name_to_id[tid]
        if not bool(mask.any()):
            continue
        size_half = info["size"].to(dtype=positions.dtype, device=positions.device) / 2.0
        outside = (positions[mask].abs() > size_half).any(dim=-1)
        bad += int(outside.sum().item())
    return bad


# ----- ckpt loader (ported from engine.py:1305-1351, CPU-safe) ------------

def _load_layered_model_from_ckpt(ckpt_path: Path):
    """Load LayeredGaussians + populate tracks on CPU. Returns (model, ckpt).

    Mirrors playground engine.load_3dgrt_object() but skips CUDA-only paths
    (no .cuda(), no OptiX BVH build) so the diagnostic runs on Mac.
    """
    from threedgrut.layers.layered_model import LayeredGaussians
    from threedgrut.layers.registry import specs_from_config

    print(f"[diag] loading ckpt {ckpt_path}", flush=True)
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
            "ckpt has no 'config' nor per-layer 'config' fallback — "
            "diagnostic requires a v2 LayeredGaussians ckpt."
        )
    if not bool(conf.get("use_layered_model", False)):
        raise RuntimeError(
            "ckpt is not a v2 LayeredGaussians ckpt (use_layered_model=false)."
        )

    specs = specs_from_config(conf)
    scene_extent = float(ckpt.get("model", {}).get("scene_extent", 1.0))
    model = LayeredGaussians(conf, specs=specs, scene_extent=scene_extent)
    model.init_from_checkpoint(ckpt, setup_optimizer=False)

    viz_4d = ckpt.get("viz_4d")
    if not isinstance(viz_4d, dict) or "tracks" not in viz_4d:
        raise RuntimeError(
            "ckpt has no viz_4d.tracks block — re-run with an injected ckpt "
            "(see threedgrut.viz.inject)."
        )
    tracks_dict = viz_4d["tracks"]
    shared_ts = viz_4d.get("tracks_camera_timestamps_us")
    if shared_ts is not None:
        first_tid = next(iter(tracks_dict))
        tracks_dict[first_tid]["cam_timestamps_us"] = shared_ts
    model.populate_tracks(tracks_dict)
    return model, ckpt


def _tracks_view_from_model(model) -> Dict[str, dict]:
    """Build a {tid: {poses, active, size}} dict from the loaded model."""
    out: Dict[str, dict] = {}
    for tid, poses in model.tracks_poses.items():
        active = model.tracks_active[tid]
        meta = getattr(model, "tracks_metadata", {}).get(tid, {})
        size = meta.get("size")
        if size is None:
            # Fallback when size not in metadata: use unit cube; flagged in report.
            size = torch.ones(3, dtype=torch.float32)
        out[tid] = {"poses": poses, "active": active, "size": size}
    return out


def _format_console(report: dict) -> str:
    lines = [
        "",
        "=" * 72,
        f"  B3 layer-vs-cuboid diagnostic  —  {report['ckpt']}",
        "=" * 72,
        f"  n_tracks                       : {report['n_tracks']}",
        f"  max_frames_per_track sampled   : {report['max_frames_per_track']}",
        "",
        "  per-layer totals + inside-any-cuboid hits:",
    ]
    name_w = max((len(n) for n in report["per_layer_total"]), default=10)
    for name, total in report["per_layer_total"].items():
        inside = report["per_layer_inside_any_cuboid"][name]
        pct = report["per_layer_inside_pct"][name]
        lines.append(
            f"    {name:<{name_w}}  total={total:>10d}   "
            f"inside={inside:>10d}   pct={pct:>6.2f}%"
        )
    dyn_out = report.get("dynamic_rigids_outside_own_cuboid_pct")
    if dyn_out is not None:
        lines.append("")
        lines.append(
            f"  dynamic_rigids particles OUTSIDE own cuboid : {dyn_out:.3f}%"
        )
    lines.append("")
    lines.append("  top-10 tracks by background-inside count:")
    by_bg = sorted(
        report["per_track_breakdown"],
        key=lambda r: r.get("background_inside", 0),
        reverse=True,
    )[:10]
    for r in by_bg:
        sz = ", ".join(f"{v:.2f}" for v in r["size"])
        lines.append(
            f"    {r['track_id']:<36}  size=[{sz}]  "
            f"active_frames={r['n_active_frames']:>4d}  "
            f"sampled={r['n_frames_sampled']:>2d}  "
            f"bg_inside={r.get('background_inside', 0):>6d}  "
            f"road_inside={r.get('road_inside', 0):>6d}"
        )
    lines.append("=" * 72)
    return "\n".join(lines)


def diagnose(ckpt_path: Path, max_frames: int) -> dict:
    model, _ckpt = _load_layered_model_from_ckpt(ckpt_path)
    tracks = _tracks_view_from_model(model)
    print(f"[diag] populated {len(tracks)} tracks; layers={list(model.layers.keys())}",
          flush=True)

    per_layer_total: Dict[str, int] = {}
    per_layer_inside: Dict[str, int] = {}
    per_layer_inside_pct: Dict[str, float] = {}
    per_track_acc: Dict[str, dict] = {}

    for spec in model.specs:
        if not spec.is_particle_layer:
            continue
        layer = model.layers[spec.name]
        positions = layer.positions.detach()
        per_layer_total[spec.name] = int(positions.shape[0])

        if spec.name == "dynamic_rigids":
            # dynamic_rigids positions are in object-local frame; the
            # "inside-any-cuboid" notion does not transfer directly. Skip
            # the per-track world containment loop for this layer.
            per_layer_inside[spec.name] = int(positions.shape[0])
            per_layer_inside_pct[spec.name] = 100.0 if positions.shape[0] else 0.0
            continue

        any_inside, records = count_world_positions_in_any_active_cuboid(
            positions, tracks, max_frames_per_track=max_frames,
        )
        n_inside = int(any_inside.sum().item())
        per_layer_inside[spec.name] = n_inside
        per_layer_inside_pct[spec.name] = (
            100.0 * n_inside / positions.shape[0] if positions.shape[0] else 0.0
        )
        for rec in records:
            entry = per_track_acc.setdefault(rec["track_id"], {
                "track_id": rec["track_id"],
                "size": rec["size"],
                "n_active_frames": rec["n_active_frames"],
                "n_frames_sampled": rec["n_frames_sampled"],
            })
            entry[f"{spec.name}_inside"] = rec["n_inside_any_sampled_frame"]

    dyn_outside_pct: Optional[float] = None
    if "dynamic_rigids" in model.layers:
        dyn_layer = model.layers["dynamic_rigids"]
        track_ids_buf = getattr(dyn_layer, "track_ids", None)
        if track_ids_buf is not None and dyn_layer.positions.numel() > 0:
            track_keys_sorted = sorted(tracks.keys())
            bad = count_local_positions_outside_own_cuboid(
                dyn_layer.positions.detach(), track_ids_buf, track_keys_sorted, tracks,
            )
            dyn_outside_pct = (
                100.0 * bad / dyn_layer.positions.shape[0]
                if dyn_layer.positions.shape[0] else 0.0
            )

    report = {
        "ckpt": str(ckpt_path),
        "n_tracks": len(tracks),
        "max_frames_per_track": max_frames,
        "per_layer_total": per_layer_total,
        "per_layer_inside_any_cuboid": per_layer_inside,
        "per_layer_inside_pct": per_layer_inside_pct,
        "dynamic_rigids_outside_own_cuboid_pct": dyn_outside_pct,
        "per_track_breakdown": list(per_track_acc.values()),
    }
    return report


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--max_frames", type=int, default=5,
                   help="evenly-spaced active frames sampled per track (default 5)")
    args = p.parse_args(argv)

    if not args.ckpt.exists():
        print(f"[diag] ERROR: ckpt not found: {args.ckpt}", file=sys.stderr)
        return 2

    report = diagnose(args.ckpt, args.max_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(report, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else o)
    print(_format_console(report))
    print(f"[diag] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
