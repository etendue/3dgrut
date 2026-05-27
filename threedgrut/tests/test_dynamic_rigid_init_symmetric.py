# SPDX-License-Identifier: Apache-2.0
"""V3-L5 unit tests for ``symmetric_axis`` augmentation in
``init_dynamic_rigid_layer``.

Coverage:
    (a) ``symmetric_axis=None`` byte-identical to pre-V3-L5 baseline (regression pin)
    (b) ``symmetric_axis='Y'`` mirror count == original (before subsample cap)
    (c) ``max_pts_per_track`` truncation still bounds output when mirror doubles
        the raw count
    (d) mirrored points have correct sign-flipped axis
    (e) invalid ``symmetric_axis`` raises ValueError
    (f) empty track (no LiDAR inside cuboid) is robust to symmetric_axis
"""
from __future__ import annotations

import pytest
import torch

from threedgrut.layers.dynamic_rigid_init import init_dynamic_rigid_layer


def _identity_track(F: int = 1, size=(4.0, 2.0, 1.5)) -> dict:
    eye = torch.eye(4).expand(F, 4, 4).clone()
    return {
        "pts": None, "colors": None,
        "poses": eye, "size": torch.tensor(list(size)),
        "frame_info": torch.ones(F, dtype=torch.bool), "class": "vehicle",
    }


# --- (a) baseline regression pin --------------------------------------------
def test_symmetric_axis_none_matches_baseline():
    """symmetric_axis=None must produce byte-identical output to the
    pre-V3-L5 baseline call (no symmetric_axis kwarg)."""
    torch.manual_seed(0)
    tracks = {"v0": _identity_track()}
    pts = torch.tensor([
        [1.0, 0.5, 0.3], [-1.0, -0.5, -0.3], [1.5, 0.9, 0.7],
        [0.0, 0.0, 0.0], [1.9, 0.5, -0.5],
    ])
    out_a = init_dynamic_rigid_layer(dict(tracks), pts.clone(),
                                     max_pts_per_track=100)
    out_b = init_dynamic_rigid_layer(dict(tracks), pts.clone(),
                                     max_pts_per_track=100,
                                     symmetric_axis=None)
    assert torch.equal(out_a[0], out_b[0])
    assert torch.equal(out_a[1], out_b[1])
    assert out_a[2] == out_b[2]


# --- (b) Y-mirror doubles count when below cap ------------------------------
def test_symmetric_axis_y_doubles_count_below_cap():
    """Below max_pts_per_track, Y-mirror should output exactly 2× the
    baseline count."""
    tracks_base = {"v0": _identity_track()}
    tracks_sym = {"v0": _identity_track()}
    pts = torch.tensor([
        [1.0, 0.5, 0.3], [-1.0, -0.5, -0.3], [1.5, 0.9, 0.7],
        [0.0, 0.4, 0.0], [1.9, 0.1, -0.5],
    ])
    pos_base, _, _ = init_dynamic_rigid_layer(
        tracks_base, pts, max_pts_per_track=100
    )
    pos_sym, _, _ = init_dynamic_rigid_layer(
        tracks_sym, pts, max_pts_per_track=100, symmetric_axis='Y'
    )
    # All 5 LiDAR points are inside the 4×2×1.5 cuboid → baseline = 5, sym = 10.
    assert pos_base.shape[0] == 5
    assert pos_sym.shape[0] == 10


# --- (c) cap bounds output when mirror would overflow -----------------------
def test_symmetric_axis_respects_max_pts_per_track():
    """Mirror is applied BEFORE subsample, so max_pts_per_track is the
    hard ceiling regardless of mirror doubling."""
    torch.manual_seed(0)
    tracks = {"v0": _identity_track()}
    # 8 LiDAR points all inside cuboid → 16 after Y-mirror, capped at 6.
    pts = torch.tensor([
        [0.5, 0.5, 0.3], [-0.5, -0.5, -0.3], [1.5, 0.9, 0.7],
        [0.0, 0.4, 0.0], [1.9, 0.1, -0.5], [-1.9, -0.1, 0.5],
        [0.3, 0.7, 0.2], [-0.3, -0.7, -0.2],
    ])
    pos, ids, _ = init_dynamic_rigid_layer(
        tracks, pts, max_pts_per_track=6, symmetric_axis='Y'
    )
    assert pos.shape[0] == 6
    assert ids.shape[0] == 6


# --- (d) mirrored points have correct sign flip -----------------------------
def test_symmetric_axis_y_flips_y_coordinate():
    """When pts are deliberately asymmetric on Y, the mirrored copy must
    contain the same set of points with Y negated."""
    tracks = {"v0": _identity_track()}
    # All points have y>0 strictly so the baseline has zero y<=0 points,
    # and the mirror should add exactly those y<0 negated copies.
    pts = torch.tensor([
        [0.5, 0.3, 0.0],
        [-0.5, 0.5, 0.1],
        [1.0, 0.7, -0.2],
    ])
    pos, _, _ = init_dynamic_rigid_layer(
        tracks, pts, max_pts_per_track=100, symmetric_axis='Y'
    )
    # 3 baseline + 3 mirrored = 6 rows. Sort by y so the test is stable
    # regardless of internal cat order.
    assert pos.shape[0] == 6
    sorted_y = sorted(pos[:, 1].tolist())
    # Expect 3 positive y and 3 negative y, paired in magnitude.
    pos_y = [y for y in sorted_y if y > 0]
    neg_y = [y for y in sorted_y if y < 0]
    assert len(pos_y) == 3
    assert len(neg_y) == 3
    # Magnitudes should match (sorted on each side).
    assert pos_y == sorted([-y for y in neg_y])


# --- (e) invalid axis raises ------------------------------------------------
@pytest.mark.parametrize("bad", ["y", "W", "yy", "", "0"])
def test_symmetric_axis_invalid_value_raises(bad):
    tracks = {"v0": _identity_track()}
    pts = torch.zeros(1, 3)
    with pytest.raises(ValueError, match="symmetric_axis"):
        init_dynamic_rigid_layer(tracks, pts, symmetric_axis=bad)


# --- (f) empty track is robust ----------------------------------------------
def test_symmetric_axis_empty_track_no_crash():
    """A track with no LiDAR points inside its cuboid must still return
    cleanly (track_pts.shape[0] == 0 → no mirror, no crash)."""
    tracks = {"v0": _identity_track()}
    # All points outside the 4×2×1.5 cuboid
    pts = torch.tensor([
        [10.0, 0.0, 0.0], [-10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
    ])
    pos, ids, names = init_dynamic_rigid_layer(
        tracks, pts, max_pts_per_track=100, symmetric_axis='Y'
    )
    assert pos.shape == (0, 3)
    assert ids.shape == (0,)
    assert names == ["v0"]


# --- (g) X / Z axes also work -----------------------------------------------
@pytest.mark.parametrize("axis,col", [("X", 0), ("Y", 1), ("Z", 2)])
def test_symmetric_axis_all_axes_flip_correct_coord(axis, col):
    tracks = {"v0": _identity_track()}
    pts = torch.tensor([[0.5, 0.3, 0.2]])  # 1 asymmetric point
    pos, _, _ = init_dynamic_rigid_layer(
        tracks, pts, max_pts_per_track=100, symmetric_axis=axis
    )
    assert pos.shape == (2, 3)
    # One row is the original (0.5, 0.3, 0.2); the other has `col` negated.
    sums = pos.sum(dim=0)
    # On the mirror axis the two rows cancel to 0; other two axes sum to 2×orig.
    assert abs(float(sums[col])) < 1e-6
    other_axes = [i for i in range(3) if i != col]
    for i in other_axes:
        expected = 2.0 * float(pts[0, i])
        assert abs(float(sums[i]) - expected) < 1e-6
