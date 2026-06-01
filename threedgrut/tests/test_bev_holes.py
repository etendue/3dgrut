# SPDX-License-Identifier: Apache-2.0
"""Mac/CPU unit tests for the BEV road-hole quantifier (Phase 2A diagnostic).

Pure-numpy; no ckpt / torch / GPU. Validates the A-type (transparency) vs
B-type (geometry) hole classification on synthetic, hand-checkable scenes.
"""
from __future__ import annotations

import numpy as np
import pytest

from threedgrut_playground.utils.bev_holes import compute_bev_hole_stats


def _tile_rect(x_lo, x_hi, y_lo, y_hi, step, opacity):
    """Dense grid of particles over a rectangle, constant opacity."""
    xs = np.arange(x_lo, x_hi + 1e-9, step)
    ys = np.arange(y_lo, y_hi + 1e-9, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    xy = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    op = np.full(xy.shape[0], float(opacity))
    return xy, op


def _ego_line(x_lo=0.0, x_hi=20.0, step=1.0):
    xs = np.arange(x_lo, x_hi + 1e-9, step)
    return np.stack([xs, np.zeros_like(xs)], axis=-1)


def test_all_transparent_gives_full_A_rate_and_zero_coverage():
    # Tile beyond ego range so the whole corridor is occupied (b-rate == 0).
    xy, op = _tile_rect(-3, 23, -2.5, 2.5, step=0.4, opacity=0.01)
    ego = _ego_line()
    s = compute_bev_hole_stats(
        xy, op, ego, cell_size=0.5, corridor_half_width=1.5, opacity_floors=(0.1,)
    )
    assert s.b_geometry_hole_rate == 0.0          # every corridor cell has a particle
    assert s.a_transparency_hole_rate["0.1"] == 1.0  # ...but all below floor
    assert s.opaque_coverage["0.1"] == 0.0


def test_all_opaque_gives_full_coverage_zero_holes():
    xy, op = _tile_rect(-3, 23, -2.5, 2.5, step=0.4, opacity=0.9)
    ego = _ego_line()
    s = compute_bev_hole_stats(
        xy, op, ego, cell_size=0.5, corridor_half_width=1.5, opacity_floors=(0.3,)
    )
    assert s.b_geometry_hole_rate == 0.0
    assert s.a_transparency_hole_rate["0.3"] == 0.0
    assert s.opaque_coverage["0.3"] == 1.0


def test_geometry_hole_when_half_corridor_untiled():
    # Particles only over x in [0,10]; ego runs to x=20 → far half is empty.
    xy, op = _tile_rect(0, 10, -2.5, 2.5, step=0.4, opacity=0.9)
    ego = _ego_line(0, 20, 1.0)
    s = compute_bev_hole_stats(
        xy, op, ego, cell_size=0.5, corridor_half_width=1.5, opacity_floors=(0.3,)
    )
    # roughly half the corridor (x in (10,20]) has no particle
    assert s.b_geometry_hole_rate > 0.3
    # the occupied half should still be opaque
    assert s.a_transparency_hole_rate["0.3"] == 0.0


def test_far_particle_outside_corridor_is_ignored():
    xy, op = _tile_rect(-3, 23, -2.5, 2.5, step=0.4, opacity=0.9)
    ego = _ego_line()
    base = compute_bev_hole_stats(
        xy, op, ego, cell_size=0.5, corridor_half_width=1.5, opacity_floors=(0.3,)
    )
    # add one particle 500 m away — far outside the corridor
    xy2 = np.concatenate([xy, np.array([[500.0, 500.0]])], axis=0)
    op2 = np.concatenate([op, np.array([0.9])], axis=0)
    s = compute_bev_hole_stats(
        xy2, op2, ego, cell_size=0.5, corridor_half_width=1.5, opacity_floors=(0.3,)
    )
    assert s.n_particles == base.n_particles + 1
    # corridor-restricted stats unchanged
    assert s.n_corridor_cells == base.n_corridor_cells
    assert s.n_corridor_occupied == base.n_corridor_occupied
    assert s.b_geometry_hole_rate == base.b_geometry_hole_rate
    assert s.opaque_coverage["0.3"] == base.opaque_coverage["0.3"]


def test_opacity_percentiles_match_numpy():
    op = np.linspace(0.0, 1.0, 1001)  # p50 == 0.5
    xy = np.stack([np.linspace(0, 20, 1001), np.zeros(1001)], axis=-1)
    ego = _ego_line()
    s = compute_bev_hole_stats(xy, op, ego, cell_size=0.5, corridor_half_width=5.0)
    assert s.opacity_percentiles["p50"] == pytest.approx(0.5, abs=1e-6)
    assert s.opacity_percentiles["mean"] == pytest.approx(0.5, abs=1e-6)


def test_partial_transparency_rate_is_fraction_of_occupied():
    # Two disjoint occupied columns: one opaque (0.9), one transparent (0.02).
    # Each column tiled densely so it fills its own corridor cells.
    op_xy, op_op = _tile_rect(0, 5, -2.5, 2.5, step=0.4, opacity=0.9)
    tr_xy, tr_op = _tile_rect(15, 20, -2.5, 2.5, step=0.4, opacity=0.02)
    xy = np.concatenate([op_xy, tr_xy], axis=0)
    op = np.concatenate([op_op, tr_op], axis=0)
    # ego covers both columns AND the gap between them (10..15 empty → B-type),
    # restrict to the two tiled spans by tiling the gap too? No — keep the gap a
    # geometry hole and assert A-type is computed over occupied cells only.
    ego = _ego_line(0, 20, 1.0)
    s = compute_bev_hole_stats(
        xy, op, ego, cell_size=0.5, corridor_half_width=1.5, opacity_floors=(0.1,)
    )
    # A-type is over occupied cells; opaque col ~half, transparent col ~half
    assert 0.35 < s.a_transparency_hole_rate["0.1"] < 0.65
    # the empty gap shows up as geometry holes
    assert s.b_geometry_hole_rate > 0.0


def test_grid_bounded_to_corridor_not_particle_sprawl():
    # Regression: bg particles sprawl ±500 m (sky/far field) but ego spans 20 m.
    # The grid must be bounded to the ego corridor, not the particle bbox, or
    # nx*ny explodes (OOM). Far-field particles are dropped, not clipped onto
    # boundary cells.
    rng = np.random.RandomState(0)
    sprawl_xy = rng.uniform(-500, 500, size=(50_000, 2))
    sprawl_op = np.full(50_000, 0.5)
    ego = _ego_line(0, 20, 1.0)
    s = compute_bev_hole_stats(
        sprawl_xy, sprawl_op, ego, cell_size=0.5, corridor_half_width=12.0
    )
    assert s.nx * s.ny < 20_000  # ~ (20+24)/0.5 * (24)/0.5, not 1000/0.5 squared
    assert s.n_particles == 50_000  # percentiles still over all particles


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_bev_hole_stats(
            np.zeros((10, 2)), np.zeros(9), _ego_line()
        )


def test_empty_inputs_raise():
    with pytest.raises(ValueError):
        compute_bev_hole_stats(np.zeros((0, 2)), np.zeros(0), _ego_line())
    with pytest.raises(ValueError):
        compute_bev_hole_stats(np.zeros((5, 2)), np.zeros(5), np.zeros((0, 2)))
