# SPDX-License-Identifier: Apache-2.0
"""V3-R2 bg-in-road opacity penalty (pure functions).

Diagnosis: ~750k alive `background` particles blanket the road surface and
dominate its rendering, leaving the dedicated `road` layer (opacity ~0.014)
invisible. This module penalizes the opacity of background particles that sit
on the road surface so MCMC relocate_gaussians marks them dead and the road
layer can take over via photometric loss. Mirrors bg_cuboid_loss.py
(grad flows only through density; spatial test is no_grad; lambda warmup).

Road region is defined by a precomputed BEV height field: bin road-layer XY
into `cell_size`-meter cells, store per-cell median Z = local ground height.
A bg particle is "on road" iff its XY lands in an occupied cell and
|z - ground_z| < z_band.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def build_road_height_field(road_positions: torch.Tensor, cell_size: float = 1.0) -> Dict:
    """Build a BEV ground-height field from road particle positions.

    Args:
        road_positions: [M,3] world-frame road-layer positions.
        cell_size: BEV cell size in meters.
    Returns dict with at least:
        "xy_min": [2] float (grid origin, world XY of cell (0,0))
        "cell_size": float
        "grid_z": [H,W] float — per-cell ground height (median Z of road pts
                  in cell); 0 where unoccupied.
        "occupied": [H,W] bool — True where >=1 road particle fell in the cell.
    For empty input, return grid_z/occupied as empty or all-False so
    query_ground_z reports everything invalid.
    Build under torch.no_grad (it's geometry, never differentiated).
    """
    with torch.no_grad():
        M = road_positions.shape[0]
        device = road_positions.device
        dtype = road_positions.dtype

        # Handle empty input
        if M == 0:
            empty_grid = torch.zeros(0, 0, dtype=dtype, device=device)
            return {
                "xy_min": torch.zeros(2, dtype=dtype, device=device),
                "xy_max": torch.zeros(2, dtype=dtype, device=device),
                "cell_size": cell_size,
                "grid_z": empty_grid,
                "occupied": torch.zeros(0, 0, dtype=torch.bool, device=device),
            }

        xy = road_positions[:, :2]  # [M, 2]
        z = road_positions[:, 2]  # [M]

        xy_min = xy.min(dim=0).values  # [2]
        xy_max = xy.max(dim=0).values  # [2]

        # Number of cells in each dimension (at least 1)
        span = xy_max - xy_min
        H = max(1, int(torch.ceil(span[0] / cell_size).item()) + 1)
        W = max(1, int(torch.ceil(span[1] / cell_size).item()) + 1)

        # Compute integer cell index for each road particle
        idx_x = torch.floor((xy[:, 0] - xy_min[0]) / cell_size).long().clamp(0, H - 1)
        idx_y = torch.floor((xy[:, 1] - xy_min[1]) / cell_size).long().clamp(0, W - 1)
        flat_idx = idx_x * W + idx_y  # [M] flat cell index

        # Build grid_z by computing per-cell median Z.
        # We iterate over unique cells — M ~200k but this runs ONCE at setup.
        grid_z = torch.zeros(H * W, dtype=dtype, device=device)
        occupied = torch.zeros(H * W, dtype=torch.bool, device=device)

        unique_cells = flat_idx.unique()
        for cell in unique_cells:
            mask = flat_idx == cell
            z_in_cell = z[mask]
            median_z = z_in_cell.median()
            grid_z[cell] = median_z
            occupied[cell] = True

        grid_z = grid_z.view(H, W)
        occupied = occupied.view(H, W)

        return {
            "xy_min": xy_min,
            "xy_max": xy_max,
            "cell_size": cell_size,
            "grid_z": grid_z,
            "occupied": occupied,
        }


def query_ground_z(
    positions_xy: torch.Tensor,  # [N, 2] world XY
    height_field: Dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Look up ground Z for each query XY.

    Args:
        positions_xy: [N,2] world XY.
        height_field: output of build_road_height_field.
    Returns:
        ground_z: [N] float — ground height of the containing cell (0 if invalid).
        valid:    [N] bool  — True iff the XY fell in an occupied cell.
    No grad. Use integer cell indexing (floor((xy - xy_min)/cell_size)),
    bounds-check against grid shape, and gate by `occupied`.
    """
    with torch.no_grad():
        grid_z = height_field["grid_z"]
        occupied = height_field["occupied"]
        xy_min = height_field["xy_min"]
        cell_size = float(height_field["cell_size"])

        N = positions_xy.shape[0]
        device = positions_xy.device
        dtype = positions_xy.dtype

        ground_z = torch.zeros(N, dtype=dtype, device=device)
        valid = torch.zeros(N, dtype=torch.bool, device=device)

        # Handle empty grid
        if grid_z.numel() == 0:
            return ground_z, valid

        H, W = grid_z.shape

        # Compute cell indices — vectorised O(N)
        xy_min_dev = xy_min.to(device=device, dtype=dtype)
        ix = torch.floor((positions_xy[:, 0] - xy_min_dev[0]) / cell_size).long()
        iy = torch.floor((positions_xy[:, 1] - xy_min_dev[1]) / cell_size).long()

        # Bounds check
        in_bounds = (ix >= 0) & (ix < H) & (iy >= 0) & (iy < W)

        if in_bounds.any():
            ix_clamp = ix[in_bounds].clamp(0, H - 1)
            iy_clamp = iy[in_bounds].clamp(0, W - 1)
            occ_sel = occupied[ix_clamp, iy_clamp]  # [K] bool
            gz_sel = grid_z[ix_clamp, iy_clamp]  # [K] float

            # Build full-size mask: in_bounds AND occupied
            valid_full = torch.zeros(N, dtype=torch.bool, device=device)
            in_bounds_idx = in_bounds.nonzero(as_tuple=True)[0]
            valid_full[in_bounds_idx] = occ_sel

            ground_z[valid_full] = gz_sel[occ_sel]
            valid = valid_full

        return ground_z, valid


def build_confident_road_surface(
    road_positions: torch.Tensor,
    *,
    cell_size: float = 0.5,
    min_support: int = 3,
    max_xy_distance: float = 1.0,
    max_z_dispersion: float = 0.25,
) -> Dict:
    """Build a locally interpolated road surface with confidence grids.

    Small holes may be filled from road points within ``max_xy_distance``.
    The grid never extends beyond the observed road XY bounding box, and a
    cell is valid only when its support and Z dispersion pass their gates.
    """
    if not cell_size > 0:
        raise ValueError("cell_size must be positive")
    if min_support <= 0:
        raise ValueError("min_support must be positive")
    if max_xy_distance < 0:
        raise ValueError("max_xy_distance must be non-negative")
    if max_z_dispersion < 0:
        raise ValueError("max_z_dispersion must be non-negative")

    with torch.no_grad():
        device = road_positions.device
        dtype = road_positions.dtype
        if road_positions.shape[0] == 0:
            empty_f = torch.zeros(0, 0, dtype=dtype, device=device)
            empty_i = torch.zeros(0, 0, dtype=torch.int64, device=device)
            empty_b = torch.zeros(0, 0, dtype=torch.bool, device=device)
            return {
                "xy_min": torch.zeros(2, dtype=dtype, device=device),
                "xy_max": torch.zeros(2, dtype=dtype, device=device),
                "cell_size": float(cell_size),
                "grid_z": empty_f,
                "support": empty_i,
                "dispersion": empty_f,
                "occupied": empty_b,
                "valid": empty_b,
                "min_support": int(min_support),
                "max_xy_distance": float(max_xy_distance),
                "max_z_dispersion": float(max_z_dispersion),
            }

        xy = road_positions[:, :2]
        z = road_positions[:, 2]
        xy_min = xy.min(dim=0).values
        xy_max = xy.max(dim=0).values
        span = xy_max - xy_min
        height = max(1, int(torch.ceil(span[0] / cell_size).item()) + 1)
        width = max(1, int(torch.ceil(span[1] / cell_size).item()) + 1)
        ix = torch.floor((xy[:, 0] - xy_min[0]) / cell_size).long().clamp(0, height - 1)
        iy = torch.floor((xy[:, 1] - xy_min[1]) / cell_size).long().clamp(0, width - 1)
        flat = ix * width + iy

        count = torch.zeros(height * width, dtype=dtype, device=device)
        sum_z = torch.zeros_like(count)
        sum_z2 = torch.zeros_like(count)
        count.scatter_add_(0, flat, torch.ones_like(z, dtype=dtype))
        sum_z.scatter_add_(0, flat, z)
        sum_z2.scatter_add_(0, flat, z.square())
        count = count.view(height, width)
        sum_z = sum_z.view(height, width)
        sum_z2 = sum_z2.view(height, width)
        occupied = count > 0

        radius_cells = int(math.ceil(max_xy_distance / cell_size))
        coords = torch.arange(-radius_cells, radius_cells + 1, device=device, dtype=dtype)
        gx, gy = torch.meshgrid(coords, coords, indexing="ij")
        circle = (gx.square() + gy.square()).sqrt() * cell_size <= max_xy_distance + 1e-9
        kernel = circle.to(dtype=dtype)[None, None]

        def aggregate(values: torch.Tensor) -> torch.Tensor:
            return F.conv2d(values[None, None], kernel, padding=radius_cells)[0, 0]

        local_count = aggregate(count)
        local_sum_z = aggregate(sum_z)
        local_sum_z2 = aggregate(sum_z2)
        safe_count = local_count.clamp_min(1.0)
        grid_z = local_sum_z / safe_count
        variance = (local_sum_z2 / safe_count - grid_z.square()).clamp_min(0.0)
        dispersion = variance.sqrt()
        valid = (local_count >= int(min_support)) & (dispersion <= float(max_z_dispersion))
        return {
            "xy_min": xy_min,
            "xy_max": xy_max,
            "cell_size": float(cell_size),
            "grid_z": grid_z,
            "support": local_count.round().to(torch.int64),
            "dispersion": dispersion,
            "occupied": occupied,
            "valid": valid,
            "min_support": int(min_support),
            "max_xy_distance": float(max_xy_distance),
            "max_z_dispersion": float(max_z_dispersion),
        }


def query_confident_road_surface(
    positions_xy: torch.Tensor,
    surface: Dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Query local road Z, validity, support count and Z dispersion."""
    with torch.no_grad():
        grid_z = surface["grid_z"]
        support_grid = surface["support"]
        dispersion_grid = surface["dispersion"]
        valid_grid = surface["valid"]
        n_queries = positions_xy.shape[0]
        device = positions_xy.device
        dtype = positions_xy.dtype
        z = torch.zeros(n_queries, dtype=dtype, device=device)
        valid = torch.zeros(n_queries, dtype=torch.bool, device=device)
        support = torch.zeros(n_queries, dtype=torch.int64, device=device)
        dispersion = torch.zeros(n_queries, dtype=dtype, device=device)
        if grid_z.numel() == 0:
            return z, valid, support, dispersion

        height, width = grid_z.shape
        xy_min = surface["xy_min"].to(device=device, dtype=dtype)
        xy_max = surface["xy_max"].to(device=device, dtype=dtype)
        cell_size = float(surface["cell_size"])
        ix = torch.floor((positions_xy[:, 0] - xy_min[0]) / cell_size).long()
        iy = torch.floor((positions_xy[:, 1] - xy_min[1]) / cell_size).long()
        in_bounds = (
            (positions_xy[:, 0] >= xy_min[0])
            & (positions_xy[:, 0] <= xy_max[0])
            & (positions_xy[:, 1] >= xy_min[1])
            & (positions_xy[:, 1] <= xy_max[1])
            & (ix >= 0)
            & (ix < height)
            & (iy >= 0)
            & (iy < width)
        )
        if in_bounds.any():
            query_indices = in_bounds.nonzero(as_tuple=True)[0]
            qx = ix[in_bounds]
            qy = iy[in_bounds]
            cell_valid = valid_grid[qx, qy]
            accepted = query_indices[cell_valid]
            z[accepted] = grid_z[qx[cell_valid], qy[cell_valid]].to(dtype=dtype)
            support[accepted] = support_grid[qx[cell_valid], qy[cell_valid]]
            dispersion[accepted] = dispersion_grid[qx[cell_valid], qy[cell_valid]].to(dtype=dtype)
            valid[accepted] = True
        return z, valid, support, dispersion


def summarize_confident_road_surface(surface: Dict) -> Dict[str, float | int]:
    """Return serializable coverage and geometry diagnostics."""
    valid = surface["valid"]
    occupied = surface["occupied"]
    grid_z = surface["grid_z"]
    if grid_z.numel() == 0:
        return {
            "n_cells": 0,
            "n_occupied_cells": 0,
            "n_valid_cells": 0,
            "n_filled_hole_cells": 0,
            "valid_fraction": 0.0,
            "z_min": float("nan"),
            "z_max": float("nan"),
        }
    valid_z = grid_z[valid]
    return {
        "n_cells": int(grid_z.numel()),
        "n_occupied_cells": int(occupied.sum()),
        "n_valid_cells": int(valid.sum()),
        "n_filled_hole_cells": int((valid & ~occupied).sum()),
        "valid_fraction": float(valid.float().mean()),
        "z_min": float(valid_z.min()) if valid_z.numel() else float("nan"),
        "z_max": float(valid_z.max()) if valid_z.numel() else float("nan"),
    }


def compute_bg_road_opacity_penalty(
    bg_positions: torch.Tensor,  # [N,3] world frame
    bg_density_raw: torch.Tensor,  # [N] or [N,1] pre-sigmoid nn.Parameter
    height_field: Dict,
    z_band: float,
    lambda_val: float,
) -> torch.Tensor:
    """Scalar loss = lambda * mean( sigmoid(bg_density) * on_road_mask ).

    on_road_mask[i] = valid_cell(xy_i) AND |z_i - ground_z(xy_i)| < z_band,
    computed under torch.no_grad (piecewise-constant in positions; we only
    want to pull opacity down, not drag positions). Gradient flows through
    bg_density_raw only. Returns torch.zeros(()) on bg_positions.device when
    lambda_val==0 or N==0 or height field empty.

    NOTE the mean is over ALL N bg particles (matching bg_cuboid_loss
    convention: mean(opacity * mask)), so the test expecting 0.5/3 for 1 of 3
    on-road particles holds.
    """
    device = bg_positions.device
    N = bg_positions.shape[0]

    if lambda_val == 0.0 or N == 0 or height_field["grid_z"].numel() == 0:
        return torch.zeros((), device=device, dtype=bg_density_raw.dtype)

    assert bg_density_raw.reshape(-1).shape[0] == N, (
        f"compute_bg_road_opacity_penalty: bg_density_raw has {bg_density_raw.reshape(-1).shape[0]} "
        f"elements but bg_positions has {N} rows — they must match."
    )

    # Spatial mask is piecewise-constant in positions — no grad through it.
    with torch.no_grad():
        positions_xy = bg_positions[:, :2].detach()
        positions_z = bg_positions[:, 2].detach()

        ground_z, valid = query_ground_z(positions_xy, height_field)

        # on_road: inside an occupied cell AND within z_band of ground
        z_dist = (positions_z - ground_z).abs()
        on_road = valid & (z_dist < z_band)

    mask_f = on_road.to(dtype=bg_density_raw.dtype)

    opacity = torch.sigmoid(bg_density_raw.view(-1))  # [N]
    loss = (opacity * mask_f).mean()
    return float(lambda_val) * loss
