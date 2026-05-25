# SPDX-License-Identifier: Apache-2.0
"""T8/B3 Phase E.3 — fused_view suppresses inactive-track particles.

Phase E audit found that ``tracks_loader.load_tracks_from_ncore_cuboids``
initializes inactive frames with ``np.eye(4)`` poses; ``_transform_means``
therefore dumped every inactive-track particle to world origin (0, 0, 0)
when fused_view ran. On a 70-track clip with only ~5-10 active per frame,
60+ tracks' worth of particles polluted the render at world origin every
single frame. This module pins the fix:

  1. ``_transform_means_and_active`` returns BOTH world positions AND a
     per-particle active mask derived from ``tracks_active`` at the resolved
     frame.
  2. ``fused_view`` consumes the active mask to push density of
     inactive-owner particles to -50 (sigmoid ≈ 1.9e-22 → effectively zero
     opacity, OptiX still processes the splat but renders nothing).
  3. The original ``_transform_means`` API is unchanged for backward compat
     with the three T4.3 transform tests.
"""
from __future__ import annotations

import os

import pytest
import torch
from hydra import compose, initialize_config_dir

from threedgrut.layers.layered_model import LayeredGaussians
from threedgrut.layers.registry import specs_from_config

_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs")
)


@pytest.fixture(scope="module")
def real_conf():
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="apps/ncore_3dgut_mcmc")


def _with_dyn_layer(conf):
    from copy import deepcopy
    c = deepcopy(conf)
    c.layers = {"enabled": ["background", "dynamic_rigids"]}
    return c


def _make_model_with_two_tracks(conf, alice_active_at_frame_0: bool,
                                bob_active_at_frame_0: bool):
    """Build a LayeredGaussians with two tracks (alice, bob) where
    each has known activity at frame 0."""
    pose_a = torch.eye(4); pose_a[:3, 3] = torch.tensor([10.0, 0.0, 0.0])
    pose_b = torch.eye(4); pose_b[:3, 3] = torch.tensor([0.0, 20.0, 0.0])
    tracks = {
        "alice": {
            "poses": pose_a.unsqueeze(0).expand(3, 4, 4).clone(),
            "active": torch.tensor(
                [alice_active_at_frame_0, True, True], dtype=torch.bool,
            ),
            "size": torch.tensor([2.0, 2.0, 2.0]),
            "cam_timestamps_us": torch.tensor([1000, 2000, 3000], dtype=torch.int64),
        },
        "bob": {
            "poses": pose_b.unsqueeze(0).expand(3, 4, 4).clone(),
            "active": torch.tensor(
                [bob_active_at_frame_0, True, True], dtype=torch.bool,
            ),
            "size": torch.tensor([2.0, 2.0, 2.0]),
            "cam_timestamps_us": torch.tensor([1000, 2000, 3000], dtype=torch.int64),
        },
    }
    model = LayeredGaussians(conf, specs=specs_from_config(conf), scene_extent=10.0,
                             tracks=tracks)
    model.init_layer_from_points("background", torch.zeros(2, 3), setup_optimizer=False)
    # Each track owns 3 particles (object-local position = origin so we can
    # easily verify which world translation gets applied).
    positions = torch.zeros(6, 3)
    track_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64)  # 0=alice, 1=bob
    model.init_layer_from_points(
        "dynamic_rigids", positions, track_ids=track_ids, setup_optimizer=False,
    )
    return model


# -----------------------------------------------------------------------------
# _transform_means_and_active
# -----------------------------------------------------------------------------

def test_transform_means_and_active_all_active(real_conf):
    """Both tracks active at frame 0 → active_mask all True."""
    conf = _with_dyn_layer(real_conf)
    model = _make_model_with_two_tracks(conf, True, True)
    local = model.layers["dynamic_rigids"].positions
    track_ids = model.layers["dynamic_rigids"].track_ids
    world, active = model._transform_means_and_active(local, track_ids, frame_id=0)
    assert world.shape == (6, 3)
    assert active.shape == (6,)
    assert active.dtype == torch.bool
    assert bool(active.all())


def test_transform_means_and_active_mixed(real_conf):
    """alice inactive, bob active at frame 0 → first 3 mask=False, last 3 True."""
    conf = _with_dyn_layer(real_conf)
    model = _make_model_with_two_tracks(conf, False, True)
    local = model.layers["dynamic_rigids"].positions
    track_ids = model.layers["dynamic_rigids"].track_ids
    world, active = model._transform_means_and_active(local, track_ids, frame_id=0)
    assert active.tolist() == [False, False, False, True, True, True]


def test_transform_means_and_active_world_positions_correct(real_conf):
    """Even for inactive particles, world positions still get computed
    (suppression happens at the fused_view density-override level)."""
    conf = _with_dyn_layer(real_conf)
    model = _make_model_with_two_tracks(conf, False, True)
    local = model.layers["dynamic_rigids"].positions  # all zeros
    track_ids = model.layers["dynamic_rigids"].track_ids
    world, _active = model._transform_means_and_active(local, track_ids, frame_id=0)
    # alice particles (0,1,2) get alice's translation (10, 0, 0); bob's get
    # bob's (0, 20, 0). Identity rotation × zero local → translation only.
    assert torch.allclose(world[:3], torch.tensor([10.0, 0.0, 0.0]).expand(3, 3).to(world))
    assert torch.allclose(world[3:], torch.tensor([0.0, 20.0, 0.0]).expand(3, 3).to(world))


def test_transform_means_unchanged_returns_only_positions(real_conf):
    """The original API stays callable (3 T4.3 tests rely on a single-tensor
    return)."""
    conf = _with_dyn_layer(real_conf)
    model = _make_model_with_two_tracks(conf, True, True)
    local = model.layers["dynamic_rigids"].positions
    track_ids = model.layers["dynamic_rigids"].track_ids
    world = model._transform_means(local, track_ids, frame_id=0)
    assert torch.is_tensor(world)
    assert world.shape == (6, 3)


# -----------------------------------------------------------------------------
# fused_view density override
# -----------------------------------------------------------------------------

_DENSITY_SENTINEL = -50.0


def _set_dyn_density(model, raw_value: float):
    """Helper: overwrite the dynamic_rigids layer's raw density (pre-sigmoid)."""
    n = model.layers["dynamic_rigids"].positions.shape[0]
    model.layers["dynamic_rigids"].density = torch.nn.Parameter(
        torch.full((n, 1), float(raw_value))
    )


def test_fused_view_keeps_density_when_all_tracks_active(real_conf):
    """All-active → fused view density matches layer density unchanged."""
    conf = _with_dyn_layer(real_conf)
    model = _make_model_with_two_tracks(conf, True, True)
    _set_dyn_density(model, raw_value=0.5)
    fv = model.fused_view(frame_id=0)
    # bg has 2 particles + dyn has 6 = 8 total
    assert fv["density"].shape == (8, 1)
    # dyn rows are last 6
    assert torch.allclose(fv["density"][2:], torch.full((6, 1), 0.5))


def test_fused_view_suppresses_density_for_inactive_owners(real_conf):
    """alice inactive at frame 0 → her 3 dyn particles get density=-50 in fused
    view, while bob's 3 keep their original density."""
    conf = _with_dyn_layer(real_conf)
    model = _make_model_with_two_tracks(conf, False, True)
    _set_dyn_density(model, raw_value=0.5)
    fv = model.fused_view(frame_id=0)
    # bg = 2 particles (rows 0, 1); dyn = 6 (rows 2..7)
    dyn_density = fv["density"][2:]
    assert dyn_density.shape == (6, 1)
    # First 3 (alice, inactive) → suppressed; last 3 (bob, active) → original
    assert torch.allclose(dyn_density[:3], torch.full((3, 1), _DENSITY_SENTINEL))
    assert torch.allclose(dyn_density[3:], torch.full((3, 1), 0.5))


def test_fused_view_all_inactive_frame_suppresses_everything(real_conf):
    """Both tracks inactive at frame 0 → all 6 dyn density entries are -50."""
    conf = _with_dyn_layer(real_conf)
    model = _make_model_with_two_tracks(conf, False, False)
    _set_dyn_density(model, raw_value=2.0)
    fv = model.fused_view(frame_id=0)
    dyn_density = fv["density"][2:]
    assert torch.allclose(dyn_density, torch.full((6, 1), _DENSITY_SENTINEL))


def test_fused_view_positions_use_transformed_world_coords(real_conf):
    """Verify that fused_view still transforms positions (E.2 + E.3 together)
    — alice active at frame 0 with pose translating to (10, 0, 0), her
    particles' world positions should be (10, 0, 0) for the local=zeros case."""
    conf = _with_dyn_layer(real_conf)
    model = _make_model_with_two_tracks(conf, True, False)
    _set_dyn_density(model, raw_value=0.5)
    fv = model.fused_view(frame_id=0)
    # Dyn positions are rows 2..7 (after bg rows 0-1)
    dyn_pos = fv["positions"][2:]
    # alice rows 0-2 in dyn → world (10, 0, 0); bob rows 3-5 → world (0, 20, 0)
    # though bob is inactive, his world position is still computed (suppression
    # happens at density, not position).
    assert torch.allclose(dyn_pos[:3], torch.tensor([10.0, 0.0, 0.0]).expand(3, 3).to(dyn_pos))
    assert torch.allclose(dyn_pos[3:], torch.tensor([0.0, 20.0, 0.0]).expand(3, 3).to(dyn_pos))


def test_fused_view_no_transform_when_no_timestamp_or_frame(real_conf):
    """When neither timestamp_us nor frame_id is given, density override
    should NOT fire (no active mask is available)."""
    conf = _with_dyn_layer(real_conf)
    model = _make_model_with_two_tracks(conf, False, False)
    _set_dyn_density(model, raw_value=0.5)
    # No frame_id, no timestamp_us → dyn density unchanged
    fv = model.fused_view()
    dyn_density = fv["density"][2:]
    assert torch.allclose(dyn_density, torch.full((6, 1), 0.5))


def test_sigmoid_of_sentinel_is_essentially_zero():
    """Sanity-check the -50 sentinel: sigmoid(-50) is below MCMC's relocate
    dead threshold (0.005) by ~20 orders of magnitude, so the renderer's
    opacity will be effectively zero."""
    val = float(torch.sigmoid(torch.tensor(-50.0)).item())
    assert val < 1e-15  # ≈ 1.9e-22 in practice
