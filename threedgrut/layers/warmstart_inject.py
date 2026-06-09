# SPDX-License-Identifier: Apache-2.0
"""P1.4 asset-harvester warm-start — trainer-seam orchestrator.

Ties metadata → PLY load → AH-1 align → subsample → merge-with-LiDAR into one
``build_warmstart_layer_inputs`` call the trainer makes when
``layers.overrides.dynamic_rigids.warmstart_ply_bundle`` is set. Kept as a pure
function (only side effect: reading PLY files from disk) so it is CPU-testable
against the demo bundle without spinning up the full trainer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from threedgrut.layers.dynamic_rigid_init import merge_warmstart_with_lidar
from threedgrut.layers.warmstart_metadata import (
    load_bundle_metadata,
    map_assets_to_tracks,
    resolve_ply_path,
)
from threedgrut.layers.warmstart_ply import (
    apply_alignment,
    asset_extent,
    assets_to_layer_inputs,
    compute_axis_alignment,
    load_warmstart_ply,
    subsample_asset,
)


def _dims_for(track: dict, fallback, use_track_size: bool):
    if use_track_size and track is not None and track.get("size") is not None:
        size = track["size"]
        size = size.tolist() if hasattr(size, "tolist") else list(size)
        return tuple(float(x) for x in size)
    return tuple(float(x) for x in fallback)


def build_warmstart_layer_inputs(
    *,
    bundle_path,
    mapping,
    tracks: dict,
    track_names: list,
    lidar_positions: torch.Tensor,
    lidar_track_ids: torch.Tensor,
    scale_prior,
    density_init: float,
    mode: str = "replace",
    max_pts_per_track: int = 5000,
    seed: int = 0,
    use_track_size: bool = True,
) -> Optional[dict]:
    """Build merged ``init_layer_from_points`` kwargs for warm-started tracks.

    Returns ``None`` (caller keeps the LiDAR-only path) when no asset maps to any
    track. Each mapped track's asset is aligned to the LIVE NCore cuboid
    (``track["size"]`` when ``use_track_size``), subsampled, and merged with its
    LiDAR particles per ``mode``. ``track_names`` is the sorted track-key order
    from ``init_dynamic_rigid_layer`` so warm track ids match the LiDAR ones.
    """
    if not mapping:
        return None

    meta_path = Path(bundle_path)
    if meta_path.is_dir():
        meta_path = meta_path / "metadata.yaml"
    bundle_root = meta_path.parent
    bundle = load_bundle_metadata(meta_path)
    asset_map = map_assets_to_tracks(bundle, tracks, mapping)
    if not asset_map:
        return None

    name_to_id = {k: i for i, k in enumerate(track_names)}
    gen = torch.Generator().manual_seed(int(seed))
    aligned_list: list[tuple[int, object]] = []
    for track_key, spec in asset_map.items():
        if track_key not in name_to_id:
            raise KeyError(
                f"warm-start track {track_key!r} not in track_names "
                f"(len {len(track_names)}); cannot assign an integer track id"
            )
        asset = load_warmstart_ply(resolve_ply_path(bundle_root, spec))
        dims = _dims_for(tracks.get(track_key), spec.cuboids_dims, use_track_size)
        half, center = asset_extent(asset)
        xf = compute_axis_alignment(spec.label_class, dims, half, center)
        aligned = apply_alignment(asset, xf)
        aligned = subsample_asset(aligned, max_pts_per_track, generator=gen)
        aligned_list.append((name_to_id[track_key], aligned))

    warm = assets_to_layer_inputs(aligned_list)
    merged = merge_warmstart_with_lidar(
        lidar_positions, lidar_track_ids, warm,
        max_pts_per_track=max_pts_per_track, scale_prior=scale_prior,
        density_init=density_init, mode=mode, generator=gen,
    )
    if merged is not None:
        # Protected warm-start (C2): the integer ids of the asset-mapped tracks.
        # Consumed by the trainer → init_layer_from_points(protected_track_ids=)
        # so MCMC leaves these tracks' injected geometry alone.
        warm_ids = sorted({tid for tid, _ in aligned_list})
        merged["warm_track_ids"] = torch.tensor(warm_ids, dtype=torch.long)
    return merged
