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
    world, active, _rot = model._transform_means_and_active(local, track_ids, frame_id=0)
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
    world, active, _rot = model._transform_means_and_active(local, track_ids, frame_id=0)
    assert active.tolist() == [False, False, False, True, True, True]


def test_transform_means_and_active_world_positions_correct(real_conf):
    """Even for inactive particles, world positions still get computed
    (suppression happens at the fused_view density-override level)."""
    conf = _with_dyn_layer(real_conf)
    model = _make_model_with_two_tracks(conf, False, True)
    local = model.layers["dynamic_rigids"].positions  # all zeros
    track_ids = model.layers["dynamic_rigids"].track_ids
    world, _active, _rot = model._transform_means_and_active(local, track_ids, frame_id=0)
    # alice particles (0,1,2) get alice's translation (10, 0, 0); bob's get
    # bob's (0, 20, 0). Identity rotation × zero local → translation only.
    assert torch.allclose(world[:3], torch.tensor([10.0, 0.0, 0.0]).expand(3, 3).to(world))
    assert torch.allclose(world[3:], torch.tensor([0.0, 20.0, 0.0]).expand(3, 3).to(world))


def test_transform_means_and_active_rotation_composition_identity_pose(real_conf):
    """Identity pose rotation → q_world = q_pose ⊗ q_local should equal q_local
    (up to sign — quat double cover). Verify the returned rotations match the
    input rotations under identity pose."""
    conf = _with_dyn_layer(real_conf)
    model = _make_model_with_two_tracks(conf, True, True)
    local = model.layers["dynamic_rigids"].positions
    track_ids = model.layers["dynamic_rigids"].track_ids
    # default per-particle rotation is identity (1, 0, 0, 0)
    rot_local = model.layers["dynamic_rigids"].rotation
    _, _, rot_world = model._transform_means_and_active(
        local, track_ids, rotations_local=rot_local, frame_id=0,
    )
    assert rot_world is not None
    assert rot_world.shape == rot_local.shape
    # identity pose × identity local → identity world (within sign)
    expected = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand_as(rot_world)
    assert torch.allclose(rot_world.abs(), expected.abs().to(rot_world), atol=1e-5)


def test_transform_means_and_active_rotation_composition_yaw(real_conf):
    """Non-trivial pose rotation (yaw=π/2) composes correctly: w_world = q_pose ⊗ q_local.

    With q_local = identity (1, 0, 0, 0), q_world should equal q_pose
    (rotation about Z by π/2 → quat = (cos π/4, 0, 0, sin π/4))."""
    import math
    # Build tracks dict explicitly with a yaw=π/2 pose
    yaw = math.pi / 2.0
    cz, sz = math.cos(yaw), math.sin(yaw)
    pose_yaw = torch.tensor([
        [cz, -sz, 0.0, 0.0],
        [sz,  cz, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    tracks = {
        "v0": {
            "poses": pose_yaw.unsqueeze(0).expand(3, 4, 4).clone(),
            "active": torch.tensor([True, True, True], dtype=torch.bool),
            "size": torch.tensor([2.0, 2.0, 2.0]),
            "cam_timestamps_us": torch.tensor([1000, 2000, 3000], dtype=torch.int64),
        }
    }
    conf = _with_dyn_layer(real_conf)
    model = LayeredGaussians(conf, specs=specs_from_config(conf), scene_extent=10.0,
                             tracks=tracks)
    model.init_layer_from_points("background", torch.zeros(2, 3), setup_optimizer=False)
    positions = torch.zeros(4, 3)
    track_ids = torch.zeros(4, dtype=torch.int64)
    model.init_layer_from_points(
        "dynamic_rigids", positions, track_ids=track_ids, setup_optimizer=False,
    )
    rot_local = model.layers["dynamic_rigids"].rotation  # all identity (1,0,0,0)
    _, _, rot_world = model._transform_means_and_active(
        positions, track_ids, rotations_local=rot_local, frame_id=0,
    )
    # Expected world quat = yaw-π/2 Z rotation = (cos π/4, 0, 0, sin π/4)
    expected_w = math.cos(math.pi / 4)
    expected_z = math.sin(math.pi / 4)
    # All 4 particles share the same track, so all rotations should be identical
    for i in range(4):
        assert math.isclose(float(rot_world[i, 0]), expected_w, abs_tol=1e-5), \
            f"q_world[{i}].w = {rot_world[i, 0]} (expected {expected_w})"
        assert math.isclose(float(rot_world[i, 3]), expected_z, abs_tol=1e-5), \
            f"q_world[{i}].z = {rot_world[i, 3]} (expected {expected_z})"
        # x, y components should be ~0
        assert abs(float(rot_world[i, 1])) < 1e-5
        assert abs(float(rot_world[i, 2])) < 1e-5


def test_transform_means_and_active_rotation_omitted_returns_none(real_conf):
    """When ``rotations_local`` is None (backward compat), returned rotations
    is None — preserves the API for legacy / unit-test callers."""
    conf = _with_dyn_layer(real_conf)
    model = _make_model_with_two_tracks(conf, True, True)
    local = model.layers["dynamic_rigids"].positions
    track_ids = model.layers["dynamic_rigids"].track_ids
    world, active, rot = model._transform_means_and_active(local, track_ids, frame_id=0)
    assert rot is None


def test_fused_view_applies_rotation_composition(real_conf):
    """fused_view should override the rotation field with q_world for the
    dynamic_rigids layer when tracks are populated + a frame is provided."""
    import math
    yaw = math.pi
    pose_pi = torch.eye(4)
    pose_pi[0, 0] = math.cos(yaw); pose_pi[0, 1] = -math.sin(yaw)
    pose_pi[1, 0] = math.sin(yaw); pose_pi[1, 1] = math.cos(yaw)
    tracks = {
        "v0": {
            "poses": pose_pi.unsqueeze(0).expand(3, 4, 4).clone(),
            "active": torch.tensor([True, True, True], dtype=torch.bool),
            "size": torch.tensor([2.0, 2.0, 2.0]),
            "cam_timestamps_us": torch.tensor([1000, 2000, 3000], dtype=torch.int64),
        }
    }
    conf = _with_dyn_layer(real_conf)
    model = LayeredGaussians(conf, specs=specs_from_config(conf), scene_extent=10.0,
                             tracks=tracks)
    model.init_layer_from_points("background", torch.zeros(2, 3), setup_optimizer=False)
    model.init_layer_from_points(
        "dynamic_rigids", torch.zeros(3, 3),
        track_ids=torch.zeros(3, dtype=torch.int64), setup_optimizer=False,
    )
    fv = model.fused_view(frame_id=0)
    # bg rows 0-1 keep their identity rotation; dyn rows 2-4 should have the
    # yaw=π quat ≈ (0, 0, 0, 1) (or sign-flipped (0, 0, 0, -1))
    dyn_rot = fv["rotation"][2:]
    for i in range(3):
        assert abs(float(dyn_rot[i, 0])) < 1e-4, f"yaw=π should give w≈0; got {dyn_rot[i, 0]}"
        assert abs(float(dyn_rot[i, 3])) > 0.99, \
            f"yaw=π should give |z|≈1; got {dyn_rot[i, 3]}"


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


def test_fused_view_no_time_uses_per_track_first_active_fallback(real_conf):
    """E.2.c: when neither timestamp_us nor frame_id is given (inference
    free camera), each track falls back to its FIRST ACTIVE FRAME so the
    user sees an "all visible actors" composite scene instead of dyn
    particles snapping to world origin.

    Build tracks where alice is active only at frame 1 (NOT 0) and bob is
    active only at frame 2. fused_view() with no args should transform
    each track's particles by THEIR first-active pose.
    """
    pose_a_f0 = torch.eye(4)
    pose_a_f1 = torch.eye(4); pose_a_f1[:3, 3] = torch.tensor([10.0, 0.0, 0.0])
    pose_a_f2 = torch.eye(4)
    pose_b_f0 = torch.eye(4)
    pose_b_f1 = torch.eye(4)
    pose_b_f2 = torch.eye(4); pose_b_f2[:3, 3] = torch.tensor([0.0, 20.0, 0.0])
    tracks = {
        "alice": {
            "poses": torch.stack([pose_a_f0, pose_a_f1, pose_a_f2]),
            "active": torch.tensor([False, True, False], dtype=torch.bool),
            "size": torch.tensor([2.0, 2.0, 2.0]),
            "cam_timestamps_us": torch.tensor([1000, 2000, 3000], dtype=torch.int64),
        },
        "bob": {
            "poses": torch.stack([pose_b_f0, pose_b_f1, pose_b_f2]),
            "active": torch.tensor([False, False, True], dtype=torch.bool),
            "size": torch.tensor([2.0, 2.0, 2.0]),
            "cam_timestamps_us": torch.tensor([1000, 2000, 3000], dtype=torch.int64),
        },
    }
    conf = _with_dyn_layer(real_conf)
    model = LayeredGaussians(conf, specs=specs_from_config(conf), scene_extent=10.0,
                             tracks=tracks)
    model.init_layer_from_points("background", torch.zeros(2, 3), setup_optimizer=False)
    positions = torch.zeros(6, 3)
    track_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64)
    model.init_layer_from_points(
        "dynamic_rigids", positions, track_ids=track_ids, setup_optimizer=False,
    )
    _set_dyn_density(model, raw_value=0.5)

    # No frame_id, no timestamp_us → E.2.c fallback fires
    fv = model.fused_view()
    dyn_pos = fv["positions"][2:]
    # alice's first active = frame 1 → translation (10, 0, 0)
    # bob's first active = frame 2 → translation (0, 20, 0)
    assert torch.allclose(dyn_pos[:3], torch.tensor([10.0, 0.0, 0.0]).expand(3, 3).to(dyn_pos))
    assert torch.allclose(dyn_pos[3:], torch.tensor([0.0, 20.0, 0.0]).expand(3, 3).to(dyn_pos))
    # Both tracks have at least one active frame → all particles considered
    # active → density unchanged (no -50 sentinel).
    dyn_density = fv["density"][2:]
    assert torch.allclose(dyn_density, torch.full((6, 1), 0.5))


def test_fused_view_no_time_track_with_zero_active_marked_inactive(real_conf):
    """E.2.c: tracks with NO active frames at all → particles still get density
    suppressed, mirroring the existing inactive-track behaviour."""
    pose = torch.eye(4); pose[:3, 3] = torch.tensor([5.0, 0.0, 0.0])
    tracks = {
        "alice": {
            "poses": pose.unsqueeze(0).expand(3, 4, 4).clone(),
            "active": torch.tensor([False, False, False], dtype=torch.bool),  # no active
            "size": torch.tensor([2.0, 2.0, 2.0]),
            "cam_timestamps_us": torch.tensor([1000, 2000, 3000], dtype=torch.int64),
        },
    }
    conf = _with_dyn_layer(real_conf)
    model = LayeredGaussians(conf, specs=specs_from_config(conf), scene_extent=10.0,
                             tracks=tracks)
    model.init_layer_from_points("background", torch.zeros(2, 3), setup_optimizer=False)
    model.init_layer_from_points(
        "dynamic_rigids", torch.zeros(3, 3),
        track_ids=torch.zeros(3, dtype=torch.int64), setup_optimizer=False,
    )
    _set_dyn_density(model, raw_value=0.5)
    fv = model.fused_view()
    # No-active-frame track → suppressed via density override
    dyn_density = fv["density"][2:]
    assert torch.allclose(dyn_density, torch.full((3, 1), _DENSITY_SENTINEL))


def test_sigmoid_of_sentinel_is_essentially_zero():
    """Sanity-check the -50 sentinel: sigmoid(-50) is below MCMC's relocate
    dead threshold (0.005) by ~20 orders of magnitude, so the renderer's
    opacity will be effectively zero."""
    val = float(torch.sigmoid(torch.tensor(-50.0)).item())
    assert val < 1e-15  # ≈ 1.9e-22 in practice
