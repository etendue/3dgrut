# SPDX-License-Identifier: Apache-2.0
"""V3-R2 unit tests for bg-in-road opacity penalty pure functions."""

from __future__ import annotations

import math

import pytest
import torch

from threedgrut.model.road_region import (
    build_confident_road_surface,
    build_road_height_field,
    compute_bg_road_opacity_penalty,
    query_confident_road_surface,
    query_ground_z,
    road_surface_gaussian_candidates,
    summarize_confident_road_surface,
)


def test_build_height_field_basic():
    # road points on a flat plane z=0, spread over a 4x4 m area
    xy = torch.stack(torch.meshgrid(torch.linspace(0, 4, 9), torch.linspace(0, 4, 9), indexing="ij"), dim=-1).reshape(
        -1, 2
    )
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
    xy = torch.stack(
        torch.meshgrid(torch.linspace(0, 10, 21), torch.linspace(0, 10, 21), indexing="ij"), dim=-1
    ).reshape(-1, 2)
    road = torch.cat([xy, torch.full((xy.shape[0], 1), 2.0)], dim=-1)  # plane z=2
    hf = build_road_height_field(road, cell_size=1.0)
    q = torch.tensor([[5.0, 5.0], [3.0, 7.0]])
    gz, valid = query_ground_z(q, hf)
    assert valid.all()
    assert torch.allclose(gz, torch.full((2,), 2.0), atol=1e-4)


def test_query_ground_z_outside_returns_invalid():
    xy = torch.stack(torch.meshgrid(torch.linspace(0, 4, 9), torch.linspace(0, 4, 9), indexing="ij"), dim=-1).reshape(
        -1, 2
    )
    road = torch.cat([xy, torch.zeros(xy.shape[0], 1)], dim=-1)
    hf = build_road_height_field(road, cell_size=1.0)
    q = torch.tensor([[100.0, 100.0]])  # far outside
    gz, valid = query_ground_z(q, hf)
    assert not valid.any()


def test_penalty_zero_when_lambda_zero():
    road = torch.cat([torch.zeros(4, 2), torch.zeros(4, 1)], dim=-1)
    hf = build_road_height_field(road, cell_size=1.0)
    bgp = torch.randn(10, 3)
    bgd = torch.zeros(10, requires_grad=True)
    L = compute_bg_road_opacity_penalty(bgp, bgd, hf, z_band=0.4, lambda_val=0.0)
    assert float(L) == 0.0


def test_penalty_targets_on_road_bg_only():
    # road plane z=0 over [0,4]x[0,4]
    xy = torch.stack(torch.meshgrid(torch.linspace(0, 4, 9), torch.linspace(0, 4, 9), indexing="ij"), dim=-1).reshape(
        -1, 2
    )
    road = torch.cat([xy, torch.zeros(xy.shape[0], 1)], dim=-1)
    hf = build_road_height_field(road, cell_size=1.0)
    # bg particles: #0 ON road (z=0.1, within band), #1 high above (z=10, sky), #2 outside XY
    bgp = torch.tensor([[2.0, 2.0, 0.1], [2.0, 2.0, 10.0], [100.0, 100.0, 0.0]])
    bgd = torch.zeros(3, requires_grad=True)  # sigmoid(0)=0.5 each
    L = compute_bg_road_opacity_penalty(bgp, bgd, hf, z_band=0.4, lambda_val=1.0)
    # only particle #0 contributes: mean over 3 = 0.5/3
    assert torch.allclose(L, torch.tensor(0.5 / 3), atol=1e-5), f"got {float(L)}"


def test_penalty_grad_flows_to_density_only():
    xy = torch.stack(torch.meshgrid(torch.linspace(0, 4, 9), torch.linspace(0, 4, 9), indexing="ij"), dim=-1).reshape(
        -1, 2
    )
    road = torch.cat([xy, torch.zeros(xy.shape[0], 1)], dim=-1)
    hf = build_road_height_field(road, cell_size=1.0)
    bgp = torch.tensor([[2.0, 2.0, 0.1]], requires_grad=True)
    bgd = torch.zeros(1, requires_grad=True)
    L = compute_bg_road_opacity_penalty(bgp, bgd, hf, z_band=0.4, lambda_val=1.0)
    L.backward()
    assert bgd.grad is not None and bgd.grad.abs().sum() > 0
    # positions must NOT receive gradient (mask is no_grad)
    assert bgp.grad is None or bgp.grad.abs().sum() == 0


def test_penalty_shape_mismatch_raises():
    import pytest as _pt

    road = torch.cat([torch.zeros(4, 2), torch.zeros(4, 1)], dim=-1)
    hf = build_road_height_field(road, cell_size=1.0)
    bgp = torch.zeros(5, 3)
    bgd = torch.zeros(3, requires_grad=True)  # mismatch 5 vs 3
    with _pt.raises(Exception):
        compute_bg_road_opacity_penalty(bgp, bgd, hf, z_band=0.4, lambda_val=1.0)


def test_confident_surface_flat_and_small_hole_fill():
    # Occupied cells at x=0 and x=2 support the empty x=1 cell.
    road = torch.tensor(
        [[0.1, 0.1, 0.0], [0.2, 0.1, 0.0], [2.3, 0.1, 0.0], [2.4, 0.1, 0.0]]
    )
    surface = build_confident_road_surface(
        road,
        cell_size=1.0,
        min_support=2,
        max_xy_distance=1.1,
        max_z_dispersion=0.01,
    )
    z, valid, support, dispersion = query_confident_road_surface(
        torch.tensor([[1.2, 0.1], [3.2, 0.1]]), surface
    )
    assert valid.tolist() == [True, False]
    assert z[0].item() == pytest.approx(0.0)
    assert support[0].item() == 4
    assert dispersion[0].item() == pytest.approx(0.0)
    summary = summarize_confident_road_surface(surface)
    assert summary["n_filled_hole_cells"] >= 1


def test_confident_surface_tracks_slope():
    road = torch.tensor(
        [[x + dx, 0.1, 0.2 * x] for x in range(5) for dx in (0.1, 0.2, 0.3)]
    )
    surface = build_confident_road_surface(
        road,
        cell_size=1.0,
        min_support=3,
        max_xy_distance=0.1,
        max_z_dispersion=0.01,
    )
    z, valid, _, _ = query_confident_road_surface(torch.tensor([[3.2, 0.1]]), surface)
    assert valid.item()
    assert z.item() == pytest.approx(0.6)


def test_confident_surface_rejects_height_discontinuity():
    road = torch.tensor(
        [[0.1, 0.1, 0.0], [0.2, 0.1, 0.0], [0.3, 0.1, 1.0], [0.4, 0.1, 1.0]]
    )
    surface = build_confident_road_surface(
        road,
        cell_size=1.0,
        min_support=3,
        max_xy_distance=0.1,
        max_z_dispersion=0.2,
    )
    _, valid, support, _ = query_confident_road_surface(torch.tensor([[0.2, 0.1]]), surface)
    assert not valid.item()
    assert support.item() == 0


def test_confident_surface_empty_and_parameter_validation():
    empty = build_confident_road_surface(torch.empty(0, 3))
    z, valid, support, dispersion = query_confident_road_surface(torch.zeros(2, 2), empty)
    assert not valid.any()
    assert not z.any() and not support.any() and not dispersion.any()
    with pytest.raises(ValueError, match="cell_size"):
        build_confident_road_surface(torch.zeros(1, 3), cell_size=0)


def test_confident_surface_is_permutation_stable():
    road = torch.tensor(
        [[0.1, 0.1, 0.0], [0.2, 0.1, 0.1], [1.1, 0.1, 0.2], [1.2, 0.1, 0.3]]
    )
    a = build_confident_road_surface(road, cell_size=1.0, min_support=1)
    b = build_confident_road_surface(
        road[torch.tensor([2, 0, 3, 1])], cell_size=1.0, min_support=1
    )
    assert torch.equal(a["support"], b["support"])
    assert torch.allclose(a["grid_z"], b["grid_z"])
    assert torch.allclose(a["dispersion"], b["dispersion"])


def test_gaussian_extent_reaches_road_when_center_does_not():
    road = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    surface = build_confident_road_surface(
        road, cell_size=1.0, min_support=3, max_xy_distance=0.1
    )
    positions = torch.tensor([[0.1, 0.0, -2.0], [0.1, 0.0, -2.0]])
    scales = torch.tensor([[0.1, 0.1, 1.1], [0.1, 0.1, 0.2]])
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    candidate, details = road_surface_gaussian_candidates(
        positions,
        scales,
        rotations,
        surface,
        relative_height_min=-0.1,
        relative_height_max=0.1,
        sigma_multiplier=2.0,
    )
    assert candidate.tolist() == [True, False]
    assert details["relative_height"].tolist() == pytest.approx([-2.0, -2.0])


def test_rotated_gaussian_uses_world_vertical_extent():
    road = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    surface = build_confident_road_surface(
        road, cell_size=1.0, min_support=3, max_xy_distance=0.1
    )
    positions = torch.tensor([[0.1, 0.0, -1.0]])
    scales = torch.tensor([[1.0, 0.1, 0.1]])
    # 90 degrees around Y maps the long local X axis onto world Z.
    rotations = torch.tensor([[2**-0.5, 0.0, 2**-0.5, 0.0]])
    candidate, details = road_surface_gaussian_candidates(
        positions,
        scales,
        rotations,
        surface,
        relative_height_min=-0.1,
        relative_height_max=0.1,
        sigma_multiplier=1.0,
    )
    assert candidate.item()
    assert details["sigma_z"].item() == pytest.approx(1.0, abs=1e-5)
