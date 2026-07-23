#!/usr/bin/env python3
"""Report confident road-surface coverage and background relative heights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from threedgrut.model.road_region import (
    build_confident_road_surface,
    query_confident_road_surface,
    summarize_confident_road_surface,
)


def checkpoint_layer_positions(checkpoint: dict, layer: str) -> torch.Tensor:
    try:
        return checkpoint["model"]["gaussians_nodes"][layer]["positions"]
    except KeyError as error:
        raise KeyError(f"checkpoint does not contain particle layer {layer!r}") from error


def analyze_surface(
    road_positions: torch.Tensor,
    background_positions: torch.Tensor,
    *,
    cell_size: float,
    min_support: int,
    max_xy_distance: float,
    max_z_dispersion: float,
    chunk_size: int = 200_000,
) -> dict:
    surface = build_confident_road_surface(
        road_positions,
        cell_size=cell_size,
        min_support=min_support,
        max_xy_distance=max_xy_distance,
        max_z_dispersion=max_z_dispersion,
    )
    relative_heights = []
    valid_support = []
    valid_dispersion = []
    for start in range(0, background_positions.shape[0], chunk_size):
        positions = background_positions[start : start + chunk_size]
        road_z, valid, support, dispersion = query_confident_road_surface(
            positions[:, :2], surface
        )
        relative_heights.append((positions[:, 2] - road_z)[valid])
        valid_support.append(support[valid])
        valid_dispersion.append(dispersion[valid])
    relative = torch.cat(relative_heights) if relative_heights else torch.empty(0)
    support = torch.cat(valid_support) if valid_support else torch.empty(0, dtype=torch.int64)
    dispersion = torch.cat(valid_dispersion) if valid_dispersion else torch.empty(0)

    report = summarize_confident_road_surface(surface)
    report.update(
        n_road_particles=int(road_positions.shape[0]),
        n_background_particles=int(background_positions.shape[0]),
        n_background_surface_valid=int(relative.numel()),
        background_surface_valid_fraction=(
            float(relative.numel() / background_positions.shape[0])
            if background_positions.shape[0]
            else 0.0
        ),
        road_particle_z_min=(float(road_positions[:, 2].min()) if road_positions.numel() else float("nan")),
        road_particle_z_max=(float(road_positions[:, 2].max()) if road_positions.numel() else float("nan")),
    )
    if relative.numel():
        report.update(
            background_relative_height_p01=float(torch.quantile(relative, 0.01)),
            background_relative_height_p10=float(torch.quantile(relative, 0.10)),
            background_relative_height_p50=float(torch.quantile(relative, 0.50)),
            background_relative_height_p90=float(torch.quantile(relative, 0.90)),
            background_relative_height_p99=float(torch.quantile(relative, 0.99)),
            background_surface_support_p10=float(torch.quantile(support.float(), 0.10)),
            background_surface_dispersion_p90=float(torch.quantile(dispersion, 0.90)),
        )
        for low, high in ((-0.25, 0.15), (-0.15, 0.10), (-0.10, 0.08)):
            key = f"candidate_{low:+.2f}_{high:+.2f}"
            count = int(((relative >= low) & (relative <= high)).sum())
            report[f"{key}_count"] = count
            report[f"{key}_fraction_of_background"] = count / background_positions.shape[0]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cell-size", type=float, default=0.5)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--max-xy-distance", type=float, default=1.0)
    parser.add_argument("--max-z-dispersion", type=float, default=0.25)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    report = analyze_surface(
        checkpoint_layer_positions(checkpoint, "road"),
        checkpoint_layer_positions(checkpoint, "background"),
        cell_size=args.cell_size,
        min_support=args.min_support,
        max_xy_distance=args.max_xy_distance,
        max_z_dispersion=args.max_z_dispersion,
        chunk_size=args.chunk_size,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
