# SPDX-License-Identifier: Apache-2.0
"""E3.3 unit tests for BEV road texture grid (build + bilinear sample).

Pure-CPU. The grid bins road-layer point features (albedo) into a BEV grid that
is geometrically aligned with build_road_height_field (same xy_min / cell_size /
H×W), so a road particle's xy can index both ground_z and its planar texture
color. The renderer (E3.3 TaskB) replaces road per-gaussian SH-DC with a
bilinear sample of this grid → view-independent planar road texture, correct
under any extrapolated view.
"""
from __future__ import annotations

import torch

from threedgrut.model.bev_texture import (
    build_bev_feature_grid,
    sample_bev_feature_bilinear,
)
from threedgrut.model.road_region import build_road_height_field


def test_build_bev_feature_grid_aligns_with_height_field():
    """grid_feature [H,W,C] shares xy_min/cell_size/occupancy with the height
    field built from the SAME road points (so xy indexes both ground_z + color)."""
    xy = torch.stack(torch.meshgrid(
        torch.linspace(0,4,9), torch.linspace(0,4,9), indexing="ij"), dim=-1).reshape(-1,2)
    pos = torch.cat([xy, torch.zeros(xy.shape[0],1)], dim=-1)
    feat = torch.zeros(pos.shape[0], 3)
    feat[:, 0] = pos[:, 0] / 4.0      # R ramps with world X
    grid = build_bev_feature_grid(pos, feat, cell_size=1.0)
    hf = build_road_height_field(pos, cell_size=1.0)
    assert grid["grid_feature"].shape[:2] == hf["grid_z"].shape
    assert grid["grid_feature"].shape[-1] == 3
    assert torch.equal(grid["occupied"], hf["occupied"])
    assert torch.allclose(grid["xy_min"], hf["xy_min"])
    # occupied cells carry the ramp: R rises along the X axis (dim 0 of the grid)
    gf, occ = grid["grid_feature"], grid["occupied"]
    r_lo = gf[0][occ[0]][:, 0].mean()      # low-X row
    r_hi = gf[-1][occ[-1]][:, 0].mean()    # high-X row
    assert r_hi > r_lo, f"ramp not preserved: lo={float(r_lo)} hi={float(r_hi)}"


def test_sample_bev_feature_bilinear_recovers_cell_color():
    """Sampling an interior point of a uniform texture returns that color;
    fully-outside xy returns the default fill (0). Gradients flow to grid_feature."""
    # uniform blue road plane over many occupied cells
    xy = torch.stack(torch.meshgrid(
        torch.linspace(0,6,13), torch.linspace(0,6,13), indexing="ij"), dim=-1).reshape(-1,2)
    pos = torch.cat([xy, torch.zeros(xy.shape[0],1)], dim=-1)
    feat = torch.zeros(pos.shape[0], 3); feat[:, 2] = 1.0   # blue
    grid = build_bev_feature_grid(pos, feat, cell_size=1.0)
    grid["grid_feature"].requires_grad_(True)
    q = torch.tensor([[3.0, 3.0], [100.0, 100.0]])   # interior, far-outside
    out = sample_bev_feature_bilinear(grid, q)
    assert out.shape == (2, 3)
    assert torch.allclose(out[0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-4), f"interior {out[0]}"
    assert torch.allclose(out[1], torch.zeros(3), atol=1e-6), "outside → default 0"
    out.sum().backward()
    assert grid["grid_feature"].grad is not None and grid["grid_feature"].grad.abs().sum() > 0
