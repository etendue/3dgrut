# SPDX-License-Identifier: Apache-2.0
"""T8/B3 — unit tests for bg_cuboid_loss helpers.

All functions are pure tensor ops (no MoG / no LayeredGaussians coupling),
so the tests exercise them directly with hand-built inputs.
"""

from __future__ import annotations

import math

import pytest
import torch

from threedgrut.model.bg_cuboid_loss import (
    clamp_layer_positions_to_cuboids,
    collect_active_cuboids_for_frame,
    compute_bg_cuboid_opacity_penalty,
    lambda_schedule,
    particles_inside_any_cuboid_mask,
)

# --- lambda_schedule -------------------------------------------------------


def test_lambda_schedule_zero_step():
    assert lambda_schedule(0, lambda_max=0.05, warmup_iters=5000) == 0.0


def test_lambda_schedule_half_warmup():
    assert lambda_schedule(2500, lambda_max=0.05, warmup_iters=5000) == pytest.approx(0.025)


def test_lambda_schedule_at_warmup_end():
    assert lambda_schedule(5000, lambda_max=0.05, warmup_iters=5000) == pytest.approx(0.05)


def test_lambda_schedule_past_warmup_held():
    assert lambda_schedule(123_456, lambda_max=0.05, warmup_iters=5000) == pytest.approx(0.05)


def test_lambda_schedule_negative_step():
    assert lambda_schedule(-1, lambda_max=0.05, warmup_iters=5000) == 0.0


def test_lambda_schedule_zero_warmup_means_immediate():
    # Edge case: if no warmup, full lambda from step 1 onwards.
    assert lambda_schedule(0, lambda_max=0.05, warmup_iters=0) == 0.0
    assert lambda_schedule(1, lambda_max=0.05, warmup_iters=0) == pytest.approx(0.05)


# --- collect_active_cuboids_for_frame -------------------------------------


def _make_track(
    F: int, active_idx: list[int], pose_tx: float = 0.0, size: tuple[float, float, float] = (2.0, 2.0, 2.0)
):
    poses = torch.eye(4).unsqueeze(0).repeat(F, 1, 1)
    for f in range(F):
        poses[f, 0, 3] = float(pose_tx)  # constant translation
    active = torch.zeros(F, dtype=torch.bool)
    for i in active_idx:
        active[i] = True
    return {
        "poses": poses,
        "active": active,
        "size": torch.tensor(size, dtype=torch.float32),
    }


def test_collect_active_one_track_one_frame():
    t = _make_track(F=10, active_idx=[3])
    poses, sizes = collect_active_cuboids_for_frame(
        {"t0": t["poses"]},
        {"t0": t["active"]},
        {"t0": t["size"]},
        frame_idx=3,
    )
    assert poses.shape == (1, 4, 4)
    assert sizes.shape == (1, 3)
    assert sizes[0].tolist() == [2.0, 2.0, 2.0]


def test_collect_active_inactive_frame_returns_empty():
    t = _make_track(F=10, active_idx=[3])
    poses, sizes = collect_active_cuboids_for_frame(
        {"t0": t["poses"]},
        {"t0": t["active"]},
        {"t0": t["size"]},
        frame_idx=5,
    )
    assert poses.shape == (0, 4, 4)
    assert sizes.shape == (0, 3)


def test_collect_active_two_tracks_only_one_active():
    t0 = _make_track(F=10, active_idx=[3], pose_tx=0.0)
    t1 = _make_track(F=10, active_idx=[5], pose_tx=10.0)
    poses, sizes = collect_active_cuboids_for_frame(
        {"t0": t0["poses"], "t1": t1["poses"]},
        {"t0": t0["active"], "t1": t1["active"]},
        {"t0": t0["size"], "t1": t1["size"]},
        frame_idx=3,
    )
    assert poses.shape == (1, 4, 4)
    assert poses[0, 0, 3].item() == 0.0  # t0's translation


def test_collect_active_both_active_sorted_order():
    t0 = _make_track(F=10, active_idx=[3], pose_tx=0.0)
    t1 = _make_track(F=10, active_idx=[3], pose_tx=10.0)
    poses, _ = collect_active_cuboids_for_frame(
        {"t1": t1["poses"], "t0": t0["poses"]},  # intentionally unsorted dict
        {"t0": t0["active"], "t1": t1["active"]},
        {"t0": t0["size"], "t1": t1["size"]},
        frame_idx=3,
    )
    assert poses.shape == (2, 4, 4)
    # Sorted order: t0 first (x=0), t1 second (x=10)
    assert poses[0, 0, 3].item() == 0.0
    assert poses[1, 0, 3].item() == 10.0


def test_collect_active_out_of_range_frame_skipped():
    t = _make_track(F=10, active_idx=[3])
    poses, _ = collect_active_cuboids_for_frame(
        {"t0": t["poses"]},
        {"t0": t["active"]},
        {"t0": t["size"]},
        frame_idx=100,
    )
    assert poses.shape == (0, 4, 4)


# --- particles_inside_any_cuboid_mask -------------------------------------


def test_inside_any_cuboid_basic():
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # inside identity-cuboid at origin
            [5.0, 0.0, 0.0],  # outside
            [10.5, 0.0, 0.0],  # inside the translated cuboid
        ]
    )
    pose0 = torch.eye(4)
    pose1 = torch.eye(4)
    pose1[0, 3] = 10.0
    poses = torch.stack([pose0, pose1])
    sizes = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    mask = particles_inside_any_cuboid_mask(positions, poses, sizes)
    assert mask.tolist() == [True, False, True]


def test_inside_any_cuboid_no_active_returns_all_false():
    positions = torch.randn(100, 3)
    mask = particles_inside_any_cuboid_mask(
        positions,
        torch.zeros(0, 4, 4),
        torch.zeros(0, 3),
    )
    assert mask.shape == (100,)
    assert not bool(mask.any())


def test_inside_any_cuboid_empty_positions():
    poses = torch.eye(4).unsqueeze(0)
    sizes = torch.tensor([[2.0, 2.0, 2.0]])
    mask = particles_inside_any_cuboid_mask(torch.zeros(0, 3), poses, sizes)
    assert mask.shape == (0,)


# --- compute_bg_cuboid_opacity_penalty ------------------------------------


def test_penalty_zero_lambda_returns_zero():
    bg_positions = torch.randn(10, 3)
    bg_density = torch.randn(10)
    poses = torch.eye(4).unsqueeze(0)
    sizes = torch.tensor([[2.0, 2.0, 2.0]])
    loss = compute_bg_cuboid_opacity_penalty(
        bg_positions,
        bg_density,
        poses,
        sizes,
        lambda_val=0.0,
    )
    assert loss.item() == 0.0


def test_penalty_all_outside_returns_zero():
    bg_positions = torch.tensor([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]])
    bg_density = torch.tensor([0.0, 0.0])  # sigmoid(0)=0.5
    poses = torch.eye(4).unsqueeze(0)
    sizes = torch.tensor([[2.0, 2.0, 2.0]])
    loss = compute_bg_cuboid_opacity_penalty(
        bg_positions,
        bg_density,
        poses,
        sizes,
        lambda_val=0.05,
    )
    assert loss.item() == 0.0


def test_penalty_all_inside_returns_lambda_times_mean_sigmoid():
    # All 4 particles inside a [2,2,2] cuboid at origin. Density=0 → sigmoid=0.5.
    bg_positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [-0.5, 0.0, 0.5],
            [0.0, 0.5, -0.5],
        ]
    )
    bg_density = torch.zeros(4)  # sigmoid(0) = 0.5
    poses = torch.eye(4).unsqueeze(0)
    sizes = torch.tensor([[2.0, 2.0, 2.0]])
    loss = compute_bg_cuboid_opacity_penalty(
        bg_positions,
        bg_density,
        poses,
        sizes,
        lambda_val=0.1,
    )
    # mean(0.5 * 1.0) = 0.5; * lambda 0.1 → 0.05
    assert loss.item() == pytest.approx(0.05, abs=1e-5)


def test_penalty_partial_inside_uses_only_inside_fraction():
    bg_positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # inside
            [100.0, 0.0, 0.0],  # outside
        ]
    )
    bg_density = torch.zeros(2)  # sigmoid=0.5
    poses = torch.eye(4).unsqueeze(0)
    sizes = torch.tensor([[2.0, 2.0, 2.0]])
    loss = compute_bg_cuboid_opacity_penalty(
        bg_positions,
        bg_density,
        poses,
        sizes,
        lambda_val=1.0,
    )
    # mean(0.5*1 + 0.5*0)/2 = 0.25; * lambda 1.0 → 0.25
    assert loss.item() == pytest.approx(0.25, abs=1e-5)


def test_penalty_gradient_flows_through_density_only():
    bg_positions = torch.tensor([[0.0, 0.0, 0.0]])
    bg_density = torch.zeros(1, requires_grad=True)
    poses = torch.eye(4).unsqueeze(0)
    sizes = torch.tensor([[2.0, 2.0, 2.0]])
    loss = compute_bg_cuboid_opacity_penalty(
        bg_positions,
        bg_density,
        poses,
        sizes,
        lambda_val=1.0,
    )
    loss.backward()
    # ∂loss/∂density = ∂(sigmoid(d))/∂d = sigmoid(d)(1-sigmoid(d)) = 0.5*0.5 = 0.25
    assert bg_density.grad.item() == pytest.approx(0.25, abs=1e-5)


def test_penalty_returns_scalar_zero_for_empty_positions():
    bg_density = torch.zeros(0)
    loss = compute_bg_cuboid_opacity_penalty(
        torch.zeros(0, 3),
        bg_density,
        torch.eye(4).unsqueeze(0),
        torch.tensor([[2.0, 2.0, 2.0]]),
        lambda_val=1.0,
    )
    assert loss.shape == ()
    assert loss.item() == 0.0


def test_penalty_density_2d_shape_handled():
    """MoG.density is sometimes [N, 1]; loss should reshape transparently."""
    bg_positions = torch.tensor([[0.0, 0.0, 0.0]])
    bg_density = torch.zeros(1, 1)
    poses = torch.eye(4).unsqueeze(0)
    sizes = torch.tensor([[2.0, 2.0, 2.0]])
    loss = compute_bg_cuboid_opacity_penalty(
        bg_positions,
        bg_density,
        poses,
        sizes,
        lambda_val=0.1,
    )
    # Same as scalar density case: mean(0.5 * 1.0) * 0.1 = 0.05
    assert loss.item() == pytest.approx(0.05, abs=1e-5)


# --- clamp_layer_positions_to_cuboids -------------------------------------


def test_clamp_all_inside_no_changes():
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [-0.5, -0.5, -0.5],
        ]
    )
    track_ids = torch.zeros(3, dtype=torch.long)
    sizes = {"t0": torch.tensor([2.0, 2.0, 2.0])}
    n = clamp_layer_positions_to_cuboids(positions, track_ids, ["t0"], sizes)
    assert n == 0
    # positions unchanged
    assert positions[0].tolist() == [0.0, 0.0, 0.0]
    assert positions[1].tolist() == [0.5, 0.5, 0.5]


def test_clamp_some_outside_get_clipped():
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # inside
            [2.0, 0.0, 0.0],  # outside x by 1.0
            [0.0, -1.5, 0.0],  # outside y by 0.5
        ]
    )
    track_ids = torch.zeros(3, dtype=torch.long)
    sizes = {"t0": torch.tensor([2.0, 2.0, 2.0])}  # half = 1.0
    n = clamp_layer_positions_to_cuboids(positions, track_ids, ["t0"], sizes)
    assert n == 2
    # clamped to boundary
    assert positions[1].tolist() == [1.0, 0.0, 0.0]
    assert positions[2].tolist() == [0.0, -1.0, 0.0]


def test_clamp_per_track_routing():
    positions = torch.tensor(
        [
            [3.0, 0.0, 0.0],  # owned by t0 (half=1.0) → outside, clamped to 1
            [3.0, 0.0, 0.0],  # owned by t1 (half=5.0) → inside, no change
        ]
    )
    track_ids = torch.tensor([0, 1], dtype=torch.long)
    sizes = {
        "t0": torch.tensor([2.0, 2.0, 2.0]),
        "t1": torch.tensor([10.0, 10.0, 10.0]),
    }
    n = clamp_layer_positions_to_cuboids(positions, track_ids, ["t0", "t1"], sizes)
    assert n == 1
    assert positions[0, 0].item() == 1.0
    assert positions[1, 0].item() == 3.0


def test_clamp_empty_positions():
    n = clamp_layer_positions_to_cuboids(
        torch.zeros(0, 3),
        torch.zeros(0, dtype=torch.long),
        ["t0"],
        {"t0": torch.tensor([2.0, 2.0, 2.0])},
    )
    assert n == 0
