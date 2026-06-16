# SPDX-License-Identifier: Apache-2.0
"""V3-R2 unit tests for bg-in-road opacity penalty pure functions."""
from __future__ import annotations

import math
import torch

from threedgrut.model.road_region import (
    build_road_height_field,
    query_ground_z,
    compute_bg_road_opacity_penalty,
)


def test_build_height_field_basic():
    # road points on a flat plane z=0, spread over a 4x4 m area
    xy = torch.stack(torch.meshgrid(
        torch.linspace(0, 4, 9), torch.linspace(0, 4, 9), indexing="ij"), dim=-1).reshape(-1, 2)
    z = torch.zeros(xy.shape[0], 1)
    road = torch.cat([xy, z], dim=-1)
    hf = build_road_height_field(road, cell_size=1.0)
    # occupied mask should cover the [0,4]x[0,4] region
    assert hf["grid_z"].shape == hf["occupied"].shape
    assert hf["occupied"].any()
    assert hf["cell_size"] == 1.0


def test_build_height_field_empty():
    road = torch.zeros(0, 3)
    hf = build_road_height_field(road, cell_size=1.0)
    assert hf["occupied"].sum() == 0 or hf["grid_z"].numel() == 0


def test_query_ground_z_flat():
    xy = torch.stack(torch.meshgrid(
        torch.linspace(0, 10, 21), torch.linspace(0, 10, 21), indexing="ij"), dim=-1).reshape(-1, 2)
    road = torch.cat([xy, torch.full((xy.shape[0],1), 2.0)], dim=-1)  # plane z=2
    hf = build_road_height_field(road, cell_size=1.0)
    q = torch.tensor([[5.0, 5.0], [3.0, 7.0]])
    gz, valid = query_ground_z(q, hf)
    assert valid.all()
    assert torch.allclose(gz, torch.full((2,), 2.0), atol=1e-4)


def test_query_ground_z_outside_returns_invalid():
    xy = torch.stack(torch.meshgrid(
        torch.linspace(0, 4, 9), torch.linspace(0, 4, 9), indexing="ij"), dim=-1).reshape(-1, 2)
    road = torch.cat([xy, torch.zeros(xy.shape[0],1)], dim=-1)
    hf = build_road_height_field(road, cell_size=1.0)
    q = torch.tensor([[100.0, 100.0]])  # far outside
    gz, valid = query_ground_z(q, hf)
    assert not valid.any()


def test_penalty_zero_when_lambda_zero():
    road = torch.cat([torch.zeros(4,2), torch.zeros(4,1)], dim=-1)
    hf = build_road_height_field(road, cell_size=1.0)
    bgp = torch.randn(10, 3)
    bgd = torch.zeros(10, requires_grad=True)
    L = compute_bg_road_opacity_penalty(bgp, bgd, hf, z_band=0.4, lambda_val=0.0)
    assert float(L) == 0.0


def test_penalty_targets_on_road_bg_only():
    # road plane z=0 over [0,4]x[0,4]
    xy = torch.stack(torch.meshgrid(
        torch.linspace(0,4,9), torch.linspace(0,4,9), indexing="ij"), dim=-1).reshape(-1,2)
    road = torch.cat([xy, torch.zeros(xy.shape[0],1)], dim=-1)
    hf = build_road_height_field(road, cell_size=1.0)
    # bg particles: #0 ON road (z=0.1, within band), #1 high above (z=10, sky), #2 outside XY
    bgp = torch.tensor([[2.0,2.0,0.1],[2.0,2.0,10.0],[100.0,100.0,0.0]])
    bgd = torch.zeros(3, requires_grad=True)  # sigmoid(0)=0.5 each
    L = compute_bg_road_opacity_penalty(bgp, bgd, hf, z_band=0.4, lambda_val=1.0)
    # only particle #0 contributes: mean over 3 = 0.5/3
    assert torch.allclose(L, torch.tensor(0.5/3), atol=1e-5), f"got {float(L)}"


def test_penalty_grad_flows_to_density_only():
    xy = torch.stack(torch.meshgrid(
        torch.linspace(0,4,9), torch.linspace(0,4,9), indexing="ij"), dim=-1).reshape(-1,2)
    road = torch.cat([xy, torch.zeros(xy.shape[0],1)], dim=-1)
    hf = build_road_height_field(road, cell_size=1.0)
    bgp = torch.tensor([[2.0,2.0,0.1]], requires_grad=True)
    bgd = torch.zeros(1, requires_grad=True)
    L = compute_bg_road_opacity_penalty(bgp, bgd, hf, z_band=0.4, lambda_val=1.0)
    L.backward()
    assert bgd.grad is not None and bgd.grad.abs().sum() > 0
    # positions must NOT receive gradient (mask is no_grad)
    assert bgp.grad is None or bgp.grad.abs().sum() == 0


def test_penalty_shape_mismatch_raises():
    import pytest as _pt
    road = torch.cat([torch.zeros(4,2), torch.zeros(4,1)], dim=-1)
    hf = build_road_height_field(road, cell_size=1.0)
    bgp = torch.zeros(5,3); bgd = torch.zeros(3, requires_grad=True)  # mismatch 5 vs 3
    with _pt.raises(Exception):
        compute_bg_road_opacity_penalty(bgp, bgd, hf, z_band=0.4, lambda_val=1.0)


# ─── E3.6 Task1: bg-init road exclusion (geometric, reuses height field) ───

def test_on_road_mask_flags_grounded_road_cell_points():
    """compute_on_road_mask: True iff xy in occupied road cell AND |z-ground|<z_band.

    This is the shared primitive for (a) bg-init road exclusion (keep=~mask)
    and (b) the existing opacity penalty's on_road test. bg LiDAR points and
    road LiDAR points come from DIFFERENT sources (get_point_clouds vs
    lidar-sseg), so exclusion must be GEOMETRIC via the road height field,
    not a per-point sseg-label index.
    """
    from threedgrut.model.road_region import compute_on_road_mask
    # road plane z=0 over [0,4]x[0,4]
    xy = torch.stack(torch.meshgrid(
        torch.linspace(0,4,9), torch.linspace(0,4,9), indexing="ij"), dim=-1).reshape(-1,2)
    road = torch.cat([xy, torch.zeros(xy.shape[0],1)], dim=-1)
    hf = build_road_height_field(road, cell_size=1.0)
    pts = torch.tensor([
        [2.0,   2.0,   0.1],   # #0 grounded in road cell      -> True
        [2.0,   2.0,  10.0],   # #1 high air above road        -> False (|z|>band)
        [100.0, 100.0, 0.0],   # #2 outside grid               -> False (not road cell)
        [2.0,   2.0,  -0.2],   # #3 just below ground in band   -> True
    ])
    mask = compute_on_road_mask(pts, hf, z_band=0.4)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [True, False, False, True], f"got {mask.tolist()}"
    # bg-init keep mask is the complement; together they partition all points
    keep = ~mask
    assert (mask | keep).all() and not (mask & keep).any()
