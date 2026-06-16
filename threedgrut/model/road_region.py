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
from typing import Dict, Optional, Tuple

import torch


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
                "cell_size": cell_size,
                "grid_z": empty_grid,
                "occupied": torch.zeros(0, 0, dtype=torch.bool, device=device),
            }

        xy = road_positions[:, :2]   # [M, 2]
        z  = road_positions[:, 2]    # [M]

        xy_min = xy.min(dim=0).values          # [2]
        xy_max = xy.max(dim=0).values          # [2]

        # Number of cells in each dimension (at least 1)
        span = xy_max - xy_min
        H = max(1, int(torch.ceil(span[0] / cell_size).item()) + 1)
        W = max(1, int(torch.ceil(span[1] / cell_size).item()) + 1)

        # Compute integer cell index for each road particle
        idx_x = torch.floor((xy[:, 0] - xy_min[0]) / cell_size).long().clamp(0, H - 1)
        idx_y = torch.floor((xy[:, 1] - xy_min[1]) / cell_size).long().clamp(0, W - 1)
        flat_idx = idx_x * W + idx_y            # [M] flat cell index

        # Build grid_z by computing per-cell median Z.
        # We iterate over unique cells — M ~200k but this runs ONCE at setup.
        grid_z    = torch.zeros(H * W, dtype=dtype, device=device)
        occupied  = torch.zeros(H * W, dtype=torch.bool, device=device)

        unique_cells = flat_idx.unique()
        for cell in unique_cells:
            mask = flat_idx == cell
            z_in_cell = z[mask]
            median_z = z_in_cell.median()
            grid_z[cell] = median_z
            occupied[cell] = True

        grid_z   = grid_z.view(H, W)
        occupied = occupied.view(H, W)

        return {
            "xy_min": xy_min,
            "cell_size": cell_size,
            "grid_z": grid_z,
            "occupied": occupied,
        }


def query_ground_z(
    positions_xy: torch.Tensor,   # [N, 2] world XY
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
        grid_z   = height_field["grid_z"]
        occupied = height_field["occupied"]
        xy_min   = height_field["xy_min"]
        cell_size = float(height_field["cell_size"])

        N = positions_xy.shape[0]
        device = positions_xy.device
        dtype  = positions_xy.dtype

        ground_z = torch.zeros(N, dtype=dtype, device=device)
        valid    = torch.zeros(N, dtype=torch.bool, device=device)

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
            occ_sel  = occupied[ix_clamp, iy_clamp]       # [K] bool
            gz_sel   = grid_z[ix_clamp, iy_clamp]         # [K] float

            # Build full-size mask: in_bounds AND occupied
            valid_full = torch.zeros(N, dtype=torch.bool, device=device)
            in_bounds_idx = in_bounds.nonzero(as_tuple=True)[0]
            valid_full[in_bounds_idx] = occ_sel

            ground_z[valid_full] = gz_sel[occ_sel]
            valid = valid_full

        return ground_z, valid


def compute_on_road_mask(
    positions: torch.Tensor,   # [N,3] world frame
    height_field: Dict,
    z_band: float,
    z_ceil: Optional[float] = None,
) -> torch.Tensor:
    """[N] bool — True iff xy falls in an occupied road cell AND z is within the
    vertical band of local ground.

    z_ceil is None (default): symmetric thin band ``|z - ground_z| < z_band``
        (V3-R2 / Task1 — grounded bg that steals road rendering).
    z_ceil given: asymmetric full-height column
        ``ground_z - z_band <= z < ground_z + z_ceil`` (E3.6 Task2 — also captures
        air-region bg above the road, the novel-view ghost source the thin band
        misses; z_band stays a thin floor so sub-road structure is left alone).

    Shared primitive used by:
      (a) the bg-in-road opacity penalty's on_road test (loss → push opacity down);
      (b) E3.6 bg-init road exclusion (keep = ~mask → bg never seeded on road).
    Pure and no_grad (piecewise-constant in positions). All-False when grid /
    positions empty.
    """
    with torch.no_grad():
        positions_xy = positions[:, :2]
        positions_z = positions[:, 2]
        ground_z, valid = query_ground_z(positions_xy, height_field)
        dz = positions_z - ground_z
        if z_ceil is None:
            in_band = dz.abs() < z_band
        else:
            in_band = (dz > -z_band) & (dz < z_ceil)
        on_road = valid & in_band
    return on_road


def compute_bg_road_opacity_penalty(
    bg_positions: torch.Tensor,    # [N,3] world frame
    bg_density_raw: torch.Tensor,  # [N] or [N,1] pre-sigmoid nn.Parameter
    height_field: Dict,
    z_band: float,
    lambda_val: float,
    z_ceil: Optional[float] = None,
) -> torch.Tensor:
    """Scalar loss = lambda * mean( sigmoid(bg_density) * on_road_mask ).

    on_road_mask[i] = valid_cell(xy_i) AND z_i within the vertical band of ground
    (symmetric |z-ground|<z_band when z_ceil is None — V3-R2 default; else the
    full-height column ground-z_band .. ground+z_ceil — E3.6 Task2 air-region),
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
    # Shared primitive with E3.6 bg-init road exclusion (keep = ~on_road).
    on_road = compute_on_road_mask(bg_positions, height_field, z_band, z_ceil=z_ceil)
    mask_f = on_road.to(dtype=bg_density_raw.dtype)

    opacity = torch.sigmoid(bg_density_raw.view(-1))    # [N]
    loss = (opacity * mask_f).mean()
    return float(lambda_val) * loss
