# SPDX-License-Identifier: Apache-2.0
"""E3.3 BEV feature grid: learnable 2D road texture sampled by gaussian XY.

Reuses road_region.build_road_height_field's BEV conventions (xy_min /
cell_size / [H,W] grid). The grid is a learnable [1,C,H,W] Parameter; road
gaussians sample it bilinearly by their world XY → [N,C] feature (C=3 → RGB →
SH DC). Pure PyTorch (F.grid_sample), autograd reproducible on Mac CPU — no
CUDA kernel (the pre-bake route: color is computed in Python, written back to
features_albedo, and the renderer kernel is untouched).

Axis convention (MUST match road_region.build_road_height_field):
    road-x = positions[:,0] → grid dim 0 (H)
    road-y = positions[:,1] → grid dim 1 (W)
F.grid_sample expects grid_coords[...,0]→W and [...,1]→H (reversed), so the
normalized grid is built as coord[...,0]=road-y(W), coord[...,1]=road-x(H).
Getting this wrong transposes the texture 90° and only shows up on GPU.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_SH_C0 = 0.28209479177387814  # matches utils/render.py C0 + layered_model._SH_C0


def build_bev_feature_grid(
    road_positions: torch.Tensor,             # [M,3] world-frame road positions
    cell_size: float = 1.0,
    n_channels: int = 3,
    init_rgb: Optional[torch.Tensor] = None,  # [3] or [M,3] in [0,1]; mean→fill
) -> Dict:
    """Build a learnable BEV feature grid over the road XY extent.

    xy_min / H / W reuse build_road_height_field's formula (same frame), so the
    grid lines up cell-for-cell with the road height field. The grid is
    nn.Parameter[1, C, H, W] (grid_sample NCHW layout), requires_grad=True.
    init: init_rgb given → fill every cell with its (per-channel) mean; else
    0.5 grey. Empty input → H=W=0, grid [1,C,0,0] (sample returns zeros).
    """
    device = road_positions.device
    dtype = road_positions.dtype if road_positions.numel() else torch.float32

    with torch.no_grad():
        M = road_positions.shape[0]
        if M == 0:
            xy_min = torch.zeros(2, dtype=dtype, device=device)
            H = W = 0
        else:
            xy = road_positions[:, :2]
            xy_min = xy.min(dim=0).values
            xy_max = xy.max(dim=0).values
            span = xy_max - xy_min
            H = max(1, int(torch.ceil(span[0] / cell_size).item()) + 1)
            W = max(1, int(torch.ceil(span[1] / cell_size).item()) + 1)

        if init_rgb is not None and init_rgb.numel() > 0:
            ir = init_rgb.to(device=device, dtype=dtype)
            if ir.dim() == 2:
                ir = ir.mean(dim=0)                       # [M,3] → [3]
            fill = ir.reshape(-1)                         # [C]
            assert fill.shape[0] == n_channels, (
                f"init_rgb has {fill.shape[0]} channels, expected {n_channels}")
            grid_t = fill.view(1, n_channels, 1, 1).expand(
                1, n_channels, H, W).clone()
        else:
            grid_t = torch.full((1, n_channels, H, W), 0.5,
                                dtype=dtype, device=device)

    grid = nn.Parameter(grid_t)                           # requires_grad=True
    return {
        "xy_min": xy_min,
        "cell_size": float(cell_size),
        "H": H,
        "W": W,
        "n_channels": n_channels,
        "grid": grid,
    }


def sample_bev_feature(
    positions_xy: torch.Tensor,   # [N,2] world XY
    grid_struct: Dict,
) -> torch.Tensor:                # [N,C], autograd flows to grid_struct["grid"]
    """Bilinear-sample the BEV grid at world XY.

    cell_f = (xy - xy_min)/cell_size (road-x→H, road-y→W), normalized to
    grid_sample NDC with align_corners=True so integer cell coord i hits grid
    node i. size==1 dims map to NDC 0 (avoids the (size-1) div-by-zero).
    Empty grid or N==0 → zeros [N,C]. padding_mode='border' clamps OOB queries.
    """
    grid = grid_struct["grid"]
    n_channels = grid_struct["n_channels"]
    N = positions_xy.shape[0]
    device = positions_xy.device
    H, W = grid_struct["H"], grid_struct["W"]

    if H == 0 or W == 0 or N == 0:
        return torch.zeros(N, n_channels, dtype=grid.dtype, device=device)

    xy_min = grid_struct["xy_min"].to(device=device, dtype=grid.dtype)
    cell_size = float(grid_struct["cell_size"])
    xy = positions_xy.to(dtype=grid.dtype)

    cell_x = (xy[:, 0] - xy_min[0]) / cell_size   # continuous coord along H
    cell_y = (xy[:, 1] - xy_min[1]) / cell_size   # continuous coord along W

    def _to_ndc(c: torch.Tensor, size: int) -> torch.Tensor:
        if size <= 1:
            return torch.zeros_like(c)            # single node → NDC center
        return 2.0 * c / (size - 1) - 1.0

    ndc_h = _to_ndc(cell_x, H)
    ndc_w = _to_ndc(cell_y, W)

    # grid_sample axis order is reversed: coord[...,0]=W(x), [...,1]=H(y)
    grid_coords = torch.stack([ndc_w, ndc_h], dim=-1).view(1, 1, N, 2)
    sampled = F.grid_sample(
        grid, grid_coords, mode="bilinear",
        align_corners=True, padding_mode="border",
    )  # [1, C, 1, N]
    return sampled.view(n_channels, N).transpose(0, 1).contiguous()  # [N, C]


def bev_feature_to_sh_dc(feat_rgb: torch.Tensor) -> torch.Tensor:   # [N,3]→[N,3]
    """RGB feature → features_albedo SH DC band: (rgb - 0.5)/C0.

    Inverse of SH2RGB (dc*C0 + 0.5), matching init_layer_from_points /
    utils.render.RGB2SH, so a grid initialized at the road colour mean bakes to
    the original DC mean (smooth takeover, no colour jump at enable time).
    """
    return (feat_rgb - 0.5) / _SH_C0
