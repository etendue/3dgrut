#!/usr/bin/env python3
"""Create a render-only checkpoint with road-relative background candidates hidden."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from threedgrut.model.road_region import (
    build_confident_road_surface,
    road_surface_gaussian_candidates,
    summarize_confident_road_surface,
)


def build_candidate_mask(
    background: dict,
    road: dict,
    *,
    cell_size: float,
    min_support: int,
    max_xy_distance: float,
    max_z_dispersion: float,
    relative_height_min: float,
    relative_height_max: float,
    sigma_multiplier: float,
    chunk_size: int = 200_000,
) -> tuple[torch.Tensor, dict]:
    surface = build_confident_road_surface(
        road["positions"].detach(),
        cell_size=cell_size,
        min_support=min_support,
        max_xy_distance=max_xy_distance,
        max_z_dispersion=max_z_dispersion,
    )
    positions = background["positions"].detach()
    scales = background["scale"].detach().exp()
    rotations = background["rotation"].detach()
    mask = torch.zeros(positions.shape[0], dtype=torch.bool, device=positions.device)
    candidate_heights = []
    candidate_sigma_z = []
    n_surface_valid = 0
    for start in range(0, positions.shape[0], chunk_size):
        stop = min(start + chunk_size, positions.shape[0])
        candidate, details = road_surface_gaussian_candidates(
            positions[start:stop],
            scales[start:stop],
            rotations[start:stop],
            surface,
            relative_height_min=relative_height_min,
            relative_height_max=relative_height_max,
            sigma_multiplier=sigma_multiplier,
        )
        mask[start:stop] = candidate
        n_surface_valid += int(details["surface_valid"].sum())
        candidate_heights.append(details["relative_height"][candidate])
        candidate_sigma_z.append(details["sigma_z"][candidate])

    heights = torch.cat(candidate_heights) if candidate_heights else torch.empty(0)
    sigma_z = torch.cat(candidate_sigma_z) if candidate_sigma_z else torch.empty(0)
    report = {
        "surface": summarize_confident_road_surface(surface),
        "relative_height_min": float(relative_height_min),
        "relative_height_max": float(relative_height_max),
        "sigma_multiplier": float(sigma_multiplier),
        "n_background": int(positions.shape[0]),
        "n_surface_valid": n_surface_valid,
        "n_candidates": int(mask.sum()),
        "candidate_fraction": float(mask.float().mean()) if mask.numel() else 0.0,
    }
    if heights.numel():
        report.update(
            candidate_relative_height_p10=float(torch.quantile(heights, 0.1)),
            candidate_relative_height_p50=float(torch.quantile(heights, 0.5)),
            candidate_relative_height_p90=float(torch.quantile(heights, 0.9)),
            candidate_sigma_z_p10=float(torch.quantile(sigma_z, 0.1)),
            candidate_sigma_z_p50=float(torch.quantile(sigma_z, 0.5)),
            candidate_sigma_z_p90=float(torch.quantile(sigma_z, 0.9)),
        )
    return mask, report


def filter_checkpoint(
    checkpoint: dict,
    candidate_mask: torch.Tensor,
    report: dict,
    *,
    density_logit: float = -100.0,
    alive_threshold: float = 0.005,
) -> dict:
    background = checkpoint["model"]["gaussians_nodes"]["background"]
    opacity_before = torch.sigmoid(background["density"].detach().reshape(-1))
    report = dict(report)
    report.update(
        density_logit=float(density_logit),
        n_alive_candidates=int((candidate_mask & (opacity_before > alive_threshold)).sum()),
    )
    with torch.no_grad():
        background["density"][candidate_mask] = float(density_logit)
    checkpoint["mcro_bg_road_relative_filter"] = report
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--cell-size", type=float, default=0.5)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--max-xy-distance", type=float, default=1.0)
    parser.add_argument("--max-z-dispersion", type=float, default=0.25)
    parser.add_argument("--relative-height-min", type=float, required=True)
    parser.add_argument("--relative-height-max", type=float, required=True)
    parser.add_argument("--sigma-multiplier", type=float, default=2.0)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--density-logit", type=float, default=-100.0)
    parser.add_argument("--dry-run", action="store_true", help="Report candidates without saving a checkpoint")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    layers = checkpoint["model"]["gaussians_nodes"]
    mask, report = build_candidate_mask(
        layers["background"],
        layers["road"],
        cell_size=args.cell_size,
        min_support=args.min_support,
        max_xy_distance=args.max_xy_distance,
        max_z_dispersion=args.max_z_dispersion,
        relative_height_min=args.relative_height_min,
        relative_height_max=args.relative_height_max,
        sigma_multiplier=args.sigma_multiplier,
        chunk_size=args.chunk_size,
    )
    filter_checkpoint(checkpoint, mask, report, density_logit=args.density_logit)
    report_path = args.report or args.out.with_suffix(args.out.suffix + ".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(checkpoint["mcro_bg_road_relative_filter"], indent=2, allow_nan=True) + "\n"
    )
    print(json.dumps(checkpoint["mcro_bg_road_relative_filter"], indent=2, allow_nan=True))
    if args.dry_run:
        print("Dry run: checkpoint was not written")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.out)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
