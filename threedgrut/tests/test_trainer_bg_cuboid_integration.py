# SPDX-License-Identifier: Apache-2.0
"""T8/B3 — integration tests for the bg-cuboid penalty wiring.

We can't import the full ``threedgrut.trainer.Trainer`` on Mac (it requires
``addict`` + the CUDA renderer), so this module covers the *integration*
between (a) :mod:`threedgrut.model.bg_cuboid_loss` helpers and (b) the
:class:`LayeredMCMCStrategy` clamp gate. Trainer-level wiring is exercised
on A800 during the 5k smoke; here we pin the conf-flag plumbing.

The two pieces covered:

1. ``LayeredMCMCStrategy._maybe_clamp_dynamic_rigids`` —
   * no-op when ``dyn_clamp_to_cuboid=false``
   * actually clamps when ``dyn_clamp_to_cuboid=true`` AND model has
     dynamic_rigids layer + track_ids + tracks_metadata.size

2. End-to-end "frame → penalty" chain that mimics trainer's call sequence:
   ``collect_active_cuboids_for_frame`` → ``compute_bg_cuboid_opacity_penalty``.
"""
from __future__ import annotations

import types

import pytest
import torch
from omegaconf import OmegaConf

from threedgrut.model.bg_cuboid_loss import (
    collect_active_cuboids_for_frame,
    compute_bg_cuboid_opacity_penalty,
    lambda_schedule,
)


# -----------------------------------------------------------------------------
# 1. LayeredMCMCStrategy._maybe_clamp_dynamic_rigids
# -----------------------------------------------------------------------------

def _make_mock_dyn_layer(track_ids: list[int], positions: torch.Tensor):
    """Mimic ``MixtureOfGaussians`` minimally: positions Parameter + track_ids buffer."""
    layer = types.SimpleNamespace()
    layer.positions = torch.nn.Parameter(positions.clone(), requires_grad=False)
    layer.track_ids = torch.tensor(track_ids, dtype=torch.long)
    return layer


def _make_mock_model(dyn_layer, tracks_metadata: dict, tracks_poses: dict):
    model = types.SimpleNamespace()
    model.layers = {"dynamic_rigids": dyn_layer}
    model.tracks_metadata = tracks_metadata
    model.tracks_poses = tracks_poses
    return model


def _build_strategy(conf, model):
    """Bind ``LayeredMCMCStrategy._maybe_clamp_dynamic_rigids`` to a stub
    holding just (config, model) — we don't construct the full strategy
    (which would need MCMCStrategy CUDA init), only exercise the gate.
    """
    from threedgrut.strategy.layered_mcmc import LayeredMCMCStrategy

    stub = types.SimpleNamespace()
    stub.config = conf
    stub.model = model
    bound = LayeredMCMCStrategy._maybe_clamp_dynamic_rigids.__get__(stub)
    return bound, stub


def test_dyn_clamp_off_by_default_byte_identical():
    """Without the conf gate, positions stay exactly as written."""
    layer = _make_mock_dyn_layer(
        track_ids=[0, 0, 0],
        positions=torch.tensor([
            [3.0, 0.0, 0.0],      # would be clamped if enabled (half=1.0)
            [0.0, 1.5, 0.0],
            [-2.0, 0.0, 0.0],
        ]),
    )
    model = _make_mock_model(
        dyn_layer=layer,
        tracks_metadata={"t0": {"size": torch.tensor([2.0, 2.0, 2.0])}},
        tracks_poses={"t0": torch.eye(4).unsqueeze(0)},
    )
    conf = OmegaConf.create({"trainer": {}})  # no bg_dyn_cuboid_penalty section
    clamp_fn, _ = _build_strategy(conf, model)
    pos_before = layer.positions.data.clone()
    clamp_fn()
    assert torch.equal(layer.positions.data, pos_before)


def test_dyn_clamp_explicitly_disabled_byte_identical():
    layer = _make_mock_dyn_layer(
        track_ids=[0, 0, 0],
        positions=torch.tensor([[3.0, 0.0, 0.0], [0.0, 1.5, 0.0], [-2.0, 0.0, 0.0]]),
    )
    model = _make_mock_model(
        dyn_layer=layer,
        tracks_metadata={"t0": {"size": torch.tensor([2.0, 2.0, 2.0])}},
        tracks_poses={"t0": torch.eye(4).unsqueeze(0)},
    )
    conf = OmegaConf.create({
        "trainer": {"bg_dyn_cuboid_penalty": {"dyn_clamp_to_cuboid": False}}
    })
    clamp_fn, _ = _build_strategy(conf, model)
    pos_before = layer.positions.data.clone()
    clamp_fn()
    assert torch.equal(layer.positions.data, pos_before)


def test_dyn_clamp_enabled_pulls_outside_back_to_boundary():
    layer = _make_mock_dyn_layer(
        track_ids=[0, 0, 0],
        positions=torch.tensor([
            [3.0, 0.0, 0.0],      # outside x → clamped to 1.0
            [0.5, 0.5, 0.5],      # inside → unchanged
            [-2.0, 0.0, 0.0],     # outside x → clamped to -1.0
        ]),
    )
    model = _make_mock_model(
        dyn_layer=layer,
        tracks_metadata={"t0": {"size": torch.tensor([2.0, 2.0, 2.0])}},  # half=1
        tracks_poses={"t0": torch.eye(4).unsqueeze(0)},
    )
    conf = OmegaConf.create({
        "trainer": {"bg_dyn_cuboid_penalty": {"dyn_clamp_to_cuboid": True}}
    })
    clamp_fn, _ = _build_strategy(conf, model)
    clamp_fn()
    pos = layer.positions.data
    assert pos[0].tolist() == [1.0, 0.0, 0.0]
    assert pos[1].tolist() == [0.5, 0.5, 0.5]
    assert pos[2].tolist() == [-1.0, 0.0, 0.0]


def test_dyn_clamp_skipped_when_no_track_ids_buffer():
    layer = types.SimpleNamespace()
    layer.positions = torch.nn.Parameter(torch.zeros(3, 3), requires_grad=False)
    # no track_ids attribute → clamp should silently no-op
    model = _make_mock_model(
        dyn_layer=layer,
        tracks_metadata={"t0": {"size": torch.tensor([2.0, 2.0, 2.0])}},
        tracks_poses={"t0": torch.eye(4).unsqueeze(0)},
    )
    conf = OmegaConf.create({
        "trainer": {"bg_dyn_cuboid_penalty": {"dyn_clamp_to_cuboid": True}}
    })
    clamp_fn, _ = _build_strategy(conf, model)
    pos_before = layer.positions.data.clone()
    clamp_fn()  # should not raise
    assert torch.equal(layer.positions.data, pos_before)


def test_dyn_clamp_skipped_when_no_layer():
    model = types.SimpleNamespace()
    model.layers = {}  # no dynamic_rigids
    model.tracks_metadata = {}
    model.tracks_poses = {}
    conf = OmegaConf.create({
        "trainer": {"bg_dyn_cuboid_penalty": {"dyn_clamp_to_cuboid": True}}
    })
    clamp_fn, _ = _build_strategy(conf, model)
    clamp_fn()  # should not raise


def test_dyn_clamp_skipped_when_empty_positions():
    layer = types.SimpleNamespace()
    layer.positions = torch.nn.Parameter(torch.zeros(0, 3), requires_grad=False)
    layer.track_ids = torch.zeros(0, dtype=torch.long)
    model = _make_mock_model(
        dyn_layer=layer,
        tracks_metadata={"t0": {"size": torch.tensor([2.0, 2.0, 2.0])}},
        tracks_poses={"t0": torch.eye(4).unsqueeze(0)},
    )
    conf = OmegaConf.create({
        "trainer": {"bg_dyn_cuboid_penalty": {"dyn_clamp_to_cuboid": True}}
    })
    clamp_fn, _ = _build_strategy(conf, model)
    clamp_fn()  # should not raise


# -----------------------------------------------------------------------------
# 2. End-to-end "frame → penalty" mimicking trainer.get_losses
# -----------------------------------------------------------------------------

def _make_bg_layer(positions: torch.Tensor, density: torch.Tensor):
    layer = types.SimpleNamespace()
    layer.positions = torch.nn.Parameter(positions.clone())
    layer.density = torch.nn.Parameter(density.clone())
    return layer


def test_end_to_end_zero_when_lambda_warmup_at_zero():
    """At step=0, lambda=0 → loss=0 regardless of cuboid contents."""
    bg = _make_bg_layer(
        positions=torch.tensor([[0.0, 0.0, 0.0]]),
        density=torch.zeros(1, 1),
    )
    tracks_poses = {"t0": torch.eye(4).unsqueeze(0)}
    tracks_active = {"t0": torch.tensor([True])}
    tracks_size = {"t0": torch.tensor([2.0, 2.0, 2.0])}

    lam = lambda_schedule(0, lambda_max=0.05, warmup_iters=5000)
    assert lam == 0.0
    poses, sizes = collect_active_cuboids_for_frame(
        tracks_poses, tracks_active, tracks_size, frame_idx=0,
    )
    loss = compute_bg_cuboid_opacity_penalty(
        bg.positions, bg.density, poses, sizes, lambda_val=lam,
    )
    assert loss.item() == 0.0


def test_end_to_end_penalty_at_full_warmup():
    """At step=warmup, lambda=lambda_max; inside-cuboid bg yields measurable loss."""
    bg = _make_bg_layer(
        positions=torch.tensor([
            [0.0, 0.0, 0.0],      # inside
            [10.0, 0.0, 0.0],     # outside (track at origin)
        ]),
        density=torch.zeros(2, 1),  # sigmoid(0) = 0.5
    )
    tracks_poses = {"t0": torch.eye(4).unsqueeze(0)}
    tracks_active = {"t0": torch.tensor([True])}
    tracks_size = {"t0": torch.tensor([2.0, 2.0, 2.0])}

    lam = lambda_schedule(5000, lambda_max=0.05, warmup_iters=5000)
    assert lam == pytest.approx(0.05)
    poses, sizes = collect_active_cuboids_for_frame(
        tracks_poses, tracks_active, tracks_size, frame_idx=0,
    )
    loss = compute_bg_cuboid_opacity_penalty(
        bg.positions, bg.density, poses, sizes, lambda_val=lam,
    )
    # 1 of 2 particles inside; sigmoid(0)=0.5 → mean=0.25; * 0.05 = 0.0125
    assert loss.item() == pytest.approx(0.0125, abs=1e-6)


def test_end_to_end_inactive_track_no_penalty():
    """When the only track is inactive at this frame, no penalty applies."""
    bg = _make_bg_layer(
        positions=torch.tensor([[0.0, 0.0, 0.0]]),
        density=torch.zeros(1, 1),
    )
    tracks_poses = {"t0": torch.eye(4).unsqueeze(0)}
    tracks_active = {"t0": torch.tensor([False])}  # inactive at frame 0
    tracks_size = {"t0": torch.tensor([2.0, 2.0, 2.0])}

    poses, sizes = collect_active_cuboids_for_frame(
        tracks_poses, tracks_active, tracks_size, frame_idx=0,
    )
    assert poses.shape[0] == 0  # no active → empty
    loss = compute_bg_cuboid_opacity_penalty(
        bg.positions, bg.density, poses, sizes, lambda_val=0.05,
    )
    assert loss.item() == 0.0


def test_end_to_end_gradient_lowers_density_inside_cuboid():
    """One step of gradient descent on the penalty pushes inside-cuboid density
    down (raw density param decreases → sigmoid(density) → less opacity)."""
    inside_pos = torch.tensor([[0.0, 0.0, 0.0]])
    outside_pos = torch.tensor([[10.0, 0.0, 0.0]])
    bg = _make_bg_layer(
        positions=torch.cat([inside_pos, outside_pos], dim=0),
        density=torch.zeros(2, 1),  # both start at 0.5 opacity
    )
    tracks_poses = {"t0": torch.eye(4).unsqueeze(0)}
    tracks_active = {"t0": torch.tensor([True])}
    tracks_size = {"t0": torch.tensor([2.0, 2.0, 2.0])}
    poses, sizes = collect_active_cuboids_for_frame(
        tracks_poses, tracks_active, tracks_size, frame_idx=0,
    )

    optim = torch.optim.SGD([bg.density], lr=1.0)
    optim.zero_grad()
    loss = compute_bg_cuboid_opacity_penalty(
        bg.positions, bg.density, poses, sizes, lambda_val=1.0,
    )
    loss.backward()
    optim.step()

    # Inside particle's density decreased; outside particle unchanged.
    d_after = bg.density.data
    assert d_after[0, 0].item() < 0.0, "inside-cuboid raw density should drop"
    assert d_after[1, 0].item() == pytest.approx(0.0, abs=1e-6), \
        "outside-cuboid raw density should not change"
