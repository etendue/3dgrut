# SPDX-License-Identifier: Apache-2.0
"""E3.3 unit tests for BEV feature grid pure functions (Task 0).

Covers build_bev_feature_grid / sample_bev_feature / bev_feature_to_sh_dc.
Pure PyTorch (no Hydra, no CUDA) — mirrors test_road_region.py so it runs on
Mac CPU. The axis-order test (road-x→grid H, road-y→grid W vs grid_sample's
(coord[...,0]=W, coord[...,1]=H)) is the critical pin: a transpose bug only
surfaces on GPU training otherwise.
"""
from __future__ import annotations

import torch

from threedgrut.model.bev_texture import (
    build_bev_feature_grid,
    sample_bev_feature,
    bev_feature_to_sh_dc,
)
from threedgrut.model.road_region import build_road_height_field

_C0 = 0.28209479177387814


def _flat_road(xs, ys, z=0.0):
    """road points = cartesian product xs × ys at height z, shape [M,3]."""
    xs = torch.as_tensor(xs, dtype=torch.float32)
    ys = torch.as_tensor(ys, dtype=torch.float32)
    xy = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), dim=-1).reshape(-1, 2)
    return torch.cat([xy, torch.full((xy.shape[0], 1), float(z))], dim=-1)


# --- build: extent / shape conventions --------------------------------------
def test_build_grid_extent_matches_road_region():
    road = _flat_road(torch.linspace(0, 4, 9), torch.linspace(0, 4, 9))
    hf = build_road_height_field(road, cell_size=1.0)
    g = build_bev_feature_grid(road, cell_size=1.0, n_channels=3)
    assert torch.allclose(g["xy_min"], hf["xy_min"])
    assert (g["H"], g["W"]) == tuple(hf["grid_z"].shape)
    assert g["cell_size"] == 1.0
    assert g["grid"].shape == (1, 3, g["H"], g["W"])


def test_build_grid_empty():
    road = torch.zeros(0, 3)
    g = build_bev_feature_grid(road, cell_size=1.0, n_channels=3)
    # empty extent → zero-sized grid; sampling returns zeros, never throws
    q = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = sample_bev_feature(q, g)
    assert out.shape == (2, 3)
    assert torch.equal(out, torch.zeros(2, 3))


def test_grid_is_learnable_parameter():
    road = _flat_road([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    g = build_bev_feature_grid(road, cell_size=1.0, n_channels=3)
    assert isinstance(g["grid"], torch.nn.Parameter)
    assert g["grid"].requires_grad is True


def test_build_grid_init_from_rgb_mean():
    road = _flat_road([0.0, 1.0], [0.0, 1.0])
    init_rgb = torch.tensor([0.2, 0.4, 0.6])
    g = build_bev_feature_grid(road, cell_size=1.0, n_channels=3, init_rgb=init_rgb)
    # every grid cell filled with the per-channel mean rgb
    for c in range(3):
        assert torch.allclose(g["grid"][0, c],
                              torch.full_like(g["grid"][0, c], float(init_rgb[c])))
    # rgb→SH-DC conversion matches the C0 convention
    assert torch.allclose(bev_feature_to_sh_dc(init_rgb.view(1, 3)),
                          (init_rgb.view(1, 3) - 0.5) / _C0)


# --- sample: axis order (CRITICAL) + bilinear + border ----------------------
def test_sample_at_node_matches_axis_order():
    # road x∈{0,1} (→H=2), y∈{0,1,2} (→W=3): non-square catches a transpose
    road = _flat_road([0.0, 1.0], [0.0, 1.0, 2.0])
    g = build_bev_feature_grid(road, cell_size=1.0, n_channels=1)
    assert (g["H"], g["W"]) == (2, 3)
    # grid[h,w] = h*10 + w — asymmetric so a swapped axis gives a wrong value
    g["grid"].data = torch.tensor([[[[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]]])  # [1,1,2,3]
    # query each node's world XY: x = xy_min_x + h*cell, y = xy_min_y + w*cell
    q, expected = [], []
    for h in range(2):
        for w in range(3):
            q.append([float(h), float(w)])
            expected.append(float(h * 10 + w))
    out = sample_bev_feature(torch.tensor(q), g)
    assert torch.allclose(out.view(-1), torch.tensor(expected)), \
        f"axis-order/sample mismatch: got {out.view(-1).tolist()}"


def test_sample_bilinear_midpoint():
    road = _flat_road([0.0, 1.0], [0.0, 1.0])  # H=W=2
    g = build_bev_feature_grid(road, cell_size=1.0, n_channels=1)
    g["grid"].data = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])  # [1,1,2,2]
    out = sample_bev_feature(torch.tensor([[0.5, 0.5]]), g)  # dead center
    assert torch.allclose(out.view(-1), torch.tensor([1.5]))  # mean of 4 corners


def test_sample_outside_uses_border():
    road = _flat_road([0.0, 1.0], [0.0, 1.0])
    g = build_bev_feature_grid(road, cell_size=1.0, n_channels=1)
    g["grid"].data = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
    out = sample_bev_feature(torch.tensor([[100.0, 100.0]]), g)  # far outside
    assert torch.isfinite(out).all()
    assert torch.allclose(out.view(-1), torch.tensor([3.0]))  # clamped to far corner


# --- gradients --------------------------------------------------------------
def test_sample_gradcheck():
    road = _flat_road([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    g = build_bev_feature_grid(road, cell_size=1.0, n_channels=2)
    grid_d = g["grid"].detach().double().requires_grad_(True)
    q = torch.tensor([[0.3, 0.7], [1.2, 1.5]], dtype=torch.double)

    def f(grid_param):
        gs = dict(g)
        gs["grid"] = grid_param
        return sample_bev_feature(q, gs)

    assert torch.autograd.gradcheck(f, (grid_d,), atol=1e-4)


def test_sample_grad_flows_to_grid_only():
    road = _flat_road([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    g = build_bev_feature_grid(road, cell_size=1.0, n_channels=3)
    q = torch.tensor([[0.5, 1.5], [2.0, 0.0]])  # leaf, no requires_grad
    sample_bev_feature(q, g).sum().backward()
    assert g["grid"].grad is not None and g["grid"].grad.abs().sum() > 0
    assert q.grad is None  # coordinates are not differentiated


# --- color mapping ----------------------------------------------------------
def test_feature_to_sh_dc_roundtrip():
    rgb = torch.tensor([[0.2, 0.4, 0.6], [0.9, 0.1, 0.5]])
    dc = bev_feature_to_sh_dc(rgb)
    assert torch.allclose(dc, (rgb - 0.5) / _C0)
    # SH2RGB inverse: dc*C0 + 0.5 recovers rgb
    assert torch.allclose(dc * _C0 + 0.5, rgb, atol=1e-6)


# --- edge: single cell (no div-by-zero in (size-1) normalization) -----------
def test_sample_single_cell_no_div0():
    road = _flat_road([0.5], [0.5])  # one point → H=W=1
    g = build_bev_feature_grid(road, cell_size=1.0, n_channels=3)
    assert (g["H"], g["W"]) == (1, 1)
    g["grid"].data = torch.tensor([[[[7.0]], [[8.0]], [[9.0]]]])  # [1,3,1,1]
    out = sample_bev_feature(torch.tensor([[0.5, 0.5], [3.0, -2.0]]), g)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[0], torch.tensor([7.0, 8.0, 9.0]))
