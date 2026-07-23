#!/usr/bin/env python3
"""Hide background Gaussians selected by multi-view road projection evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def projection_candidate_mask(
    counts: dict[str, np.ndarray],
    *,
    road_field: str,
    min_road_hits: int,
    protect_field: str,
    max_protected_hits: int,
    min_road_visible_ratio: float,
) -> np.ndarray:
    if road_field not in {"road_center_hits", "road_footprint_hits"}:
        raise ValueError("invalid road_field")
    if protect_field not in {"protected_center_hits", "protected_footprint_hits"}:
        raise ValueError("invalid protect_field")
    road_hits = counts[road_field]
    protected_hits = counts[protect_field]
    visible_hits = counts["visible_hits"]
    if not (road_hits.shape == protected_hits.shape == visible_hits.shape):
        raise ValueError("projection count arrays must have the same shape")
    ratio = road_hits / np.maximum(visible_hits, 1)
    return (
        (road_hits >= int(min_road_hits))
        & (protected_hits <= int(max_protected_hits))
        & (ratio >= float(min_road_visible_ratio))
    )


def filter_checkpoint(
    checkpoint: dict,
    candidate_mask: np.ndarray,
    metadata: dict,
    *,
    density_logit: float = -100.0,
    alive_threshold: float = 0.005,
) -> dict:
    background = checkpoint["model"]["gaussians_nodes"]["background"]
    mask = torch.as_tensor(candidate_mask, dtype=torch.bool, device=background["density"].device)
    if mask.numel() != background["positions"].shape[0]:
        raise ValueError("candidate mask must match background Gaussian count")
    alive = torch.sigmoid(background["density"].detach().reshape(-1)) > alive_threshold
    metadata = dict(metadata)
    metadata.update(
        n_background=int(mask.numel()),
        n_candidates=int(mask.sum()),
        candidate_fraction=float(mask.float().mean()) if mask.numel() else 0.0,
        n_alive_candidates=int((mask & alive).sum()),
        density_logit=float(density_logit),
    )
    with torch.no_grad():
        background["density"][mask] = float(density_logit)
    checkpoint["mcro_bg_projection_filter"] = metadata
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--road-field",
        choices=("road_center_hits", "road_footprint_hits"),
        default="road_footprint_hits",
    )
    parser.add_argument("--min-road-hits", type=int, default=1)
    parser.add_argument(
        "--protect-field",
        choices=("protected_center_hits", "protected_footprint_hits"),
        default="protected_center_hits",
    )
    parser.add_argument("--max-protected-hits", type=int, default=0)
    parser.add_argument("--min-road-visible-ratio", type=float, default=0.0)
    parser.add_argument("--density-logit", type=float, default=-100.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    loaded = np.load(args.counts)
    counts = {name: loaded[name] for name in loaded.files}
    mask = projection_candidate_mask(
        counts,
        road_field=args.road_field,
        min_road_hits=args.min_road_hits,
        protect_field=args.protect_field,
        max_protected_hits=args.max_protected_hits,
        min_road_visible_ratio=args.min_road_visible_ratio,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = {
        "counts_path": str(args.counts),
        "road_field": args.road_field,
        "min_road_hits": args.min_road_hits,
        "protect_field": args.protect_field,
        "max_protected_hits": args.max_protected_hits,
        "min_road_visible_ratio": args.min_road_visible_ratio,
    }
    filter_checkpoint(checkpoint, mask, metadata, density_logit=args.density_logit)
    report = checkpoint["mcro_bg_projection_filter"]
    report_path = args.out.with_suffix(args.out.suffix + ".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not args.dry_run:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.out)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
