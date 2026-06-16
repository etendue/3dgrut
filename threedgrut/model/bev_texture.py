# SPDX-License-Identifier: Apache-2.0
"""E3.3 BEV road texture grid (pure functions).

Road color as a planar BEV texture instead of per-gaussian SH: a road particle's
world xy bilinearly samples a learnable [H,W,C] feature grid glued to the road
height field. Any extrapolated view re-samples the SAME planar texture →
view-independent, correct off-track (parameterization-level fix for the aperture
problem; cf. ExtraGS Road Surface Gaussians).

The grid shares xy_min / cell_size / H×W with build_road_height_field built from
the same road points, so one xy indexes both ground_z and texture color.
"""
from __future__ import annotations
from typing import Dict

import torch

from threedgrut.model.road_region import build_road_height_field


def build_bev_feature_grid(
    road_positions: torch.Tensor,   # [M,3] world frame
    road_features: torch.Tensor,    # [M,C] per-point feature (albedo / RGB DC)
    cell_size: float = 1.0,
) -> Dict:
    """Bin per-point road features into a BEV grid (per-cell median), aligned
    with build_road_height_field(road_positions, cell_size).

    Returns {xy_min, cell_size, grid_feature[H,W,C], occupied[H,W]}. Built under
    no_grad (init-time seeding); the returned grid_feature is meant to be wrapped
    in an nn.Parameter and learned, and sampled differentiably at train time.
    """
    hf = build_road_height_field(road_positions, cell_size)
    xy_min = hf["xy_min"]
    occupied = hf["occupied"]
    H, W = occupied.shape
    C = int(road_features.shape[1]) if road_features.dim() == 2 else 1
    device = road_features.device
    dtype = road_features.dtype
    with torch.no_grad():
        grid_feature = torch.zeros(H, W, C, dtype=dtype, device=device)
        M = road_positions.shape[0]
        if M == 0 or H == 0 or W == 0:
            return {"xy_min": xy_min, "cell_size": cell_size,
                    "grid_feature": grid_feature, "occupied": occupied}
        xy = road_positions[:, :2].to(device)
        xy_min_d = xy_min.to(device=device, dtype=xy.dtype)
        idx_x = torch.floor((xy[:, 0] - xy_min_d[0]) / cell_size).long().clamp(0, H - 1)
        idx_y = torch.floor((xy[:, 1] - xy_min_d[1]) / cell_size).long().clamp(0, W - 1)
        flat = idx_x * W + idx_y
        flat_grid = grid_feature.view(H * W, C)
        for cell in flat.unique():
            mask = flat == cell
            flat_grid[cell] = road_features[mask].median(dim=0).values
        grid_feature = flat_grid.view(H, W, C)
    return {"xy_min": xy_min, "cell_size": cell_size,
            "grid_feature": grid_feature, "occupied": occupied}


def sample_bev_feature_bilinear(grid: Dict, query_xy: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample grid_feature [H,W,C] at world query_xy [N,2] → [N,C].

    Cell-center aligned (cell (i,j) center at xy_min + (i+0.5, j+0.5)*cell_size).
    Out-of-grid neighbors contribute 0, so a fully-outside query returns 0.
    Differentiable in grid_feature (the learnable road texture).
    """
    grid_feature = grid["grid_feature"]          # [H,W,C]
    cell_size = float(grid["cell_size"])
    H, W, C = grid_feature.shape
    N = query_xy.shape[0]
    if H == 0 or W == 0 or N == 0:
        return torch.zeros(N, C, dtype=grid_feature.dtype, device=grid_feature.device)
    xy_min = grid["xy_min"].to(device=query_xy.device, dtype=query_xy.dtype)
    fx = (query_xy[:, 0] - xy_min[0]) / cell_size - 0.5
    fy = (query_xy[:, 1] - xy_min[1]) / cell_size - 0.5
    x0 = torch.floor(fx).long()
    y0 = torch.floor(fy).long()
    x1, y1 = x0 + 1, y0 + 1
    wx = (fx - x0.to(fx.dtype)).unsqueeze(-1)    # [N,1]
    wy = (fy - y0.to(fy.dtype)).unsqueeze(-1)
    gf = grid_feature.reshape(H * W, C)

    def gather(ix, iy):
        valid = (ix >= 0) & (ix < H) & (iy >= 0) & (iy < W)
        flat = ix.clamp(0, H - 1) * W + iy.clamp(0, W - 1)
        v = gf[flat]                              # [N,C]
        return v * valid.unsqueeze(-1).to(v.dtype)

    f00 = gather(x0, y0)
    f01 = gather(x0, y1)
    f10 = gather(x1, y0)
    f11 = gather(x1, y1)
    return (f00 * (1 - wx) * (1 - wy) + f01 * (1 - wx) * wy
            + f10 * wx * (1 - wy) + f11 * wx * wy)
