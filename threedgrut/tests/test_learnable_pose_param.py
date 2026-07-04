# SPDX-License-Identifier: Apache-2.0
"""V3 Stage A — learnable cuboid pose Parameter contract tests.

Verifies the surfaces touched by the V3 Stage A refactor on
``LayeredGaussians`` (registers wxyz quat + trans Parameters per track instead
of a frozen ``_track_pose_<tid>`` buffer when
``conf.trainer.learnable_pose.enabled=true``):

1. ``_quat_wxyz_to_rotmat`` round-trips against ``_rotmat_to_quat_wxyz``.
2. ``populate_tracks`` registers ``nn.Parameter`` (vs ``register_buffer``)
   based on the conf flag.
3. ``tracks_poses`` is a ``@property`` that survives a state_dict round-trip
   (root cause for observations #321/#349/#851 — Python-dict mirror going
   stale after ckpt load).
4. ``_compose_pose_for_track`` reconstructs the original GT pose at init
   (no drift before optimization).
5. Adam step on a synthetic photometric-like loss actually moves the trans
   Parameter (gradient plumbing works end-to-end).
6. Resume guard: a second ``populate_tracks`` call after pose optimization
   does NOT overwrite the learned Parameter.
7. ``track_quat`` Parameters survive a state_dict save → load round-trip.
8. Property keys are returned in sorted order (matches the
   ``init_layer_from_points`` ``sorted(tracks_poses.keys())`` convention
   used to assign per-particle ``track_ids``).

Engine isolation note: same as ``test_engine_layered_load.py`` — we don't
construct a full Engine (kaolin / OptiX absent on Mac); the unit under test
is ``LayeredGaussians`` itself.
"""

import os

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir

from threedgrut.layers.layer_spec import LayerSpec

_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs"))


@pytest.fixture(scope="module")
def conf_learnable_off():
    """Baseline (v2) conf: ``learnable_pose.enabled=false`` — buffer path."""
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(
            config_name="apps/ncore_3dgut_mcmc",
            overrides=["trainer.learnable_pose.enabled=false"],
        )


@pytest.fixture(scope="module")
def conf_learnable_on():
    """V3 Stage A conf: ``learnable_pose.enabled=true`` — Parameter path."""
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(
            config_name="apps/ncore_3dgut_mcmc",
            overrides=["trainer.learnable_pose.enabled=true"],
        )


def _viz4d_tracks_dict(F: int = 4) -> dict:
    """Build a track dict matching ``ckpt["viz_4d"]["tracks"]`` schema.

    Mirrors ``test_engine_layered_load._viz4d_tracks_dict`` but adds a
    nontrivial yaw rotation to each frame so the quat ↔ rotmat round-trip
    actually exercises non-identity matrices (else q ≈ [1,0,0,0] hides bugs).
    """
    tracks = {}
    for i, tid in enumerate(["t0", "t1"]):
        poses = torch.eye(4).repeat(F, 1, 1)
        # Per-frame yaw rotation around Z so each frame has a distinct R.
        for f in range(F):
            yaw = torch.tensor(0.1 * (f + 1) + 0.5 * i, dtype=torch.float32)
            c, s = torch.cos(yaw), torch.sin(yaw)
            poses[f, 0, 0] = c
            poses[f, 0, 1] = -s
            poses[f, 1, 0] = s
            poses[f, 1, 1] = c
            poses[f, 0, 3] = torch.arange(F, dtype=torch.float32)[f] + i * 100.0
        tracks[tid] = {
            "poses": poses,
            "size": torch.tensor([2.0, 1.5, 4.5], dtype=torch.float32),
            "frame_info": torch.ones(F, dtype=torch.bool),
            "class": "automobile" if i == 0 else "heavy_truck",
        }
    return tracks


# ─── 1. quat ↔ rotmat round-trip ────────────────────────────────────────────


def test_quat_wxyz_to_rotmat_round_trip():
    """rot → quat → rot identity on random SO(3) elements via axis-angle."""
    from threedgrut.layers.layered_model import (
        _quat_wxyz_to_rotmat,
        _rotmat_to_quat_wxyz,
    )

    torch.manual_seed(42)
    max_err = 0.0
    for _ in range(50):
        axis = torch.randn(3)
        axis = axis / axis.norm()
        angle = torch.rand(1).item() * 6.0 - 3.0  # ∈ (-3, 3) covers ±π
        K = torch.tensor(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        R = torch.eye(3) + torch.sin(torch.tensor(angle)) * K + (1 - torch.cos(torch.tensor(angle))) * (K @ K)
        q = _rotmat_to_quat_wxyz(R)
        R2 = _quat_wxyz_to_rotmat(q)
        max_err = max(max_err, (R - R2).abs().max().item())
    assert max_err < 1e-5, f"max round-trip error {max_err:.2e} > 1e-5"


def test_quat_wxyz_to_rotmat_batched_shape():
    """Batched input ``[..., 4]`` → ``[..., 3, 3]``."""
    from threedgrut.layers.layered_model import _quat_wxyz_to_rotmat

    q = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(4, 5, 4).contiguous()
    R = _quat_wxyz_to_rotmat(q)
    assert R.shape == (4, 5, 3, 3)
    eye = torch.eye(3).expand(4, 5, 3, 3)
    assert torch.allclose(R, eye, atol=1e-6)


# ─── 2. populate_tracks: Parameter vs buffer based on flag ──────────────────


def test_populate_tracks_registers_buffer_when_disabled(conf_learnable_off):
    """Default v2 path: ``_track_pose_<tid>`` registered as a non-trainable buffer."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1000)]
    model = LayeredGaussians(conf_learnable_off, specs=specs, scene_extent=1.0)
    tracks = _viz4d_tracks_dict(F=4)
    tracks[next(iter(tracks))]["cam_timestamps_us"] = torch.tensor([1000, 2000, 3000, 4000], dtype=torch.int64)
    model.populate_tracks(tracks)

    # Buffer mode: pose buffer exists, quat/trans Parameter do NOT.
    assert "_track_pose_t0" in model._buffers
    assert "_track_pose_t1" in model._buffers
    assert "_track_quat_t0" not in model._parameters
    assert "_track_trans_t0" not in model._parameters
    # Active mask is always a buffer.
    assert "_track_active_t0" in model._buffers


def test_populate_tracks_registers_parameter_when_enabled(conf_learnable_on):
    """V3 Stage A path: ``_track_quat_<tid>`` + ``_track_trans_<tid>`` are
    Parameters; ``_track_pose_gt_<tid>`` is a frozen buffer for resume."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1000)]
    model = LayeredGaussians(conf_learnable_on, specs=specs, scene_extent=1.0)
    tracks = _viz4d_tracks_dict(F=4)
    tracks[next(iter(tracks))]["cam_timestamps_us"] = torch.tensor([1000, 2000, 3000, 4000], dtype=torch.int64)
    model.populate_tracks(tracks)

    # Learnable mode: quat[F, 4] and trans[F, 3] Parameters per track.
    assert "_track_quat_t0" in model._parameters
    assert "_track_trans_t0" in model._parameters
    assert "_track_quat_t1" in model._parameters
    assert "_track_trans_t1" in model._parameters
    assert model._track_quat_t0.shape == (4, 4)
    assert model._track_trans_t0.shape == (4, 3)
    assert isinstance(model._track_quat_t0, nn.Parameter)
    assert isinstance(model._track_trans_t0, nn.Parameter)
    assert model._track_quat_t0.requires_grad is True

    # Legacy pose buffer NOT present (we don't double-store).
    assert "_track_pose_t0" not in model._buffers
    # Frozen GT pose stored for resume / future viz diff.
    assert "_track_pose_gt_t0" in model._buffers
    assert model._track_pose_gt_t0.shape == (4, 4, 4)
    # Active mask still a buffer in both modes.
    assert "_track_active_t0" in model._buffers


# ─── 3. tracks_poses property survives state_dict round-trip ────────────────


def test_tracks_poses_property_survives_state_dict_round_trip(conf_learnable_off):
    """Root cause for observations #321/#349/#851: the old Python-dict mirror
    went stale after ckpt load. ``tracks_poses`` is now a ``@property`` derived
    from the registered buffers/parameters, so it always reflects current
    state.
    """
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1000)]
    src = LayeredGaussians(conf_learnable_off, specs=specs, scene_extent=1.0)
    tracks = _viz4d_tracks_dict(F=4)
    tracks[next(iter(tracks))]["cam_timestamps_us"] = torch.tensor([1000, 2000, 3000, 4000], dtype=torch.int64)
    src.populate_tracks(tracks)
    state = src.state_dict()

    # Build a fresh model and load state into it (the path that used to leave
    # tracks_poses empty). We must populate first so buffer slots exist for
    # load_state_dict to fill.
    dst = LayeredGaussians(conf_learnable_off, specs=specs, scene_extent=1.0)
    dst.populate_tracks(tracks)
    dst.load_state_dict(state)

    # Property recomputes from buffers — never empty after load.
    assert set(dst.tracks_poses.keys()) == {"t0", "t1"}
    assert dst.tracks_poses["t0"].shape == (4, 4, 4)
    # Values match the source's pose buffer.
    assert torch.allclose(dst.tracks_poses["t0"], src.tracks_poses["t0"])


def test_tracks_poses_property_keys_sorted(conf_learnable_off):
    """``init_layer_from_points`` assigns per-particle ``track_ids`` via
    ``sorted(tracks_poses.keys())``. The property MUST return keys in sorted
    order for indexing to round-trip (observation #677 ordering risk)."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1000)]
    model = LayeredGaussians(conf_learnable_off, specs=specs, scene_extent=1.0)

    # Insert in reverse order to test that registration order doesn't leak.
    F = 3
    tracks = {}
    for tid in ["zzz", "aaa", "mmm"]:
        tracks[tid] = {
            "poses": torch.eye(4).repeat(F, 1, 1),
            "size": torch.tensor([1.0, 1.0, 1.0]),
            "frame_info": torch.ones(F, dtype=torch.bool),
        }
    tracks[next(iter(tracks))]["cam_timestamps_us"] = torch.tensor([100, 200, 300], dtype=torch.int64)
    model.populate_tracks(tracks)

    assert list(model.tracks_poses.keys()) == ["aaa", "mmm", "zzz"]
    assert list(model.tracks_active.keys()) == ["aaa", "mmm", "zzz"]


# ─── 4. _compose_pose_for_track reconstructs GT at init ─────────────────────


def test_compose_pose_matches_gt_at_init_learnable(conf_learnable_on):
    """In learnable mode, immediately after populate_tracks, composing pose
    from quat/trans Parameter should match the input GT pose closely."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1000)]
    model = LayeredGaussians(conf_learnable_on, specs=specs, scene_extent=1.0)
    tracks = _viz4d_tracks_dict(F=4)
    gt_pose_t0 = tracks["t0"]["poses"].clone()
    tracks[next(iter(tracks))]["cam_timestamps_us"] = torch.tensor([1000, 2000, 3000, 4000], dtype=torch.int64)
    model.populate_tracks(tracks)

    # Per-frame composition matches GT to single-precision tolerance
    # (rotmat→quat→rotmat round-trip + float32 SO(3) reconstruction).
    for idx in range(4):
        composed = model._compose_pose_for_track("t0", idx)
        assert torch.allclose(composed, gt_pose_t0[idx], atol=1e-5), (
            f"frame {idx}: composed vs gt diff " f"max={(composed - gt_pose_t0[idx]).abs().max().item():.2e}"
        )


def test_compose_pose_buffer_path_returns_slice(conf_learnable_off):
    """Buffer mode: _compose_pose_for_track returns the raw pose slice (no
    quat detour). Verifies the legacy fast path is preserved."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1000)]
    model = LayeredGaussians(conf_learnable_off, specs=specs, scene_extent=1.0)
    tracks = _viz4d_tracks_dict(F=4)
    tracks[next(iter(tracks))]["cam_timestamps_us"] = torch.tensor([1000, 2000, 3000, 4000], dtype=torch.int64)
    model.populate_tracks(tracks)
    for idx in range(4):
        assert torch.equal(
            model._compose_pose_for_track("t0", idx),
            tracks["t0"]["poses"][idx],
        )


# ─── 5. Adam step on synthetic loss moves the trans Parameter ───────────────


def test_pose_optimizer_step_changes_trans_parameter(conf_learnable_on):
    """Smoke test of the full forward gradient path: build a synthetic loss
    that depends on a track's composed pose, backward, step → trans changes.
    Mimics what photometric loss flowing through _transform_means_and_active
    would do."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1000)]
    model = LayeredGaussians(conf_learnable_on, specs=specs, scene_extent=1.0)
    tracks = _viz4d_tracks_dict(F=4)
    tracks[next(iter(tracks))]["cam_timestamps_us"] = torch.tensor([1000, 2000, 3000, 4000], dtype=torch.int64)
    model.populate_tracks(tracks)

    trans_before = model._track_trans_t0.detach().clone()
    quat_before = model._track_quat_t0.detach().clone()

    opt = torch.optim.Adam(
        [
            {"params": [model._track_quat_t0, model._track_quat_t1], "lr": 1e-3},
            {"params": [model._track_trans_t0, model._track_trans_t1], "lr": 1e-2},
        ]
    )

    # Push the trans of t0 frame 1 toward a target offset.
    target = torch.tensor([5.0, 7.0, 0.0])
    for _ in range(20):
        opt.zero_grad()
        pose = model._compose_pose_for_track("t0", 1)
        loss = (pose[:3, 3] - target).pow(2).sum()
        loss.backward()
        opt.step()

    trans_after = model._track_trans_t0.detach()
    quat_after = model._track_quat_t0.detach()
    assert not torch.allclose(trans_after, trans_before), "trans Parameter did not move under optimization"
    # The target-aligned frame should be closer to the target than at init.
    init_dist = (trans_before[1] - target).norm()
    final_dist = (trans_after[1] - target).norm()
    assert final_dist < init_dist, f"trans did not approach target ({init_dist} → {final_dist})"
    # Quat may also drift slightly (Adam touches non-frame-1 too because of
    # the loss formulation), but rotation matrix should remain orthogonal
    # after the normalize-in-_compose_pose step.
    composed = model._compose_pose_for_track("t0", 1)
    R = composed[:3, :3]
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-4), "composed rotation lost orthogonality"


# ─── 6. Resume guard: re-populate doesn't clobber learned Parameter ─────────


def test_resume_guard_preserves_learned_parameter(conf_learnable_on):
    """If a learnable-mode model has already trained pose Parameter, a second
    populate_tracks call (engine reload path) must NOT overwrite it back to GT.
    """
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1000)]
    model = LayeredGaussians(conf_learnable_on, specs=specs, scene_extent=1.0)
    tracks = _viz4d_tracks_dict(F=4)
    tracks[next(iter(tracks))]["cam_timestamps_us"] = torch.tensor([1000, 2000, 3000, 4000], dtype=torch.int64)
    model.populate_tracks(tracks)

    # Simulate that training moved trans by adding a small delta in-place.
    with torch.no_grad():
        model._track_trans_t0.add_(torch.tensor([10.0, 20.0, 30.0]))
    moved = model._track_trans_t0.detach().clone()

    # Second populate (e.g. engine reload of same clip).
    model.populate_tracks(tracks)

    # Resume guard fires: Parameter still equals the moved value, not GT.
    assert torch.allclose(
        model._track_trans_t0.detach(), moved
    ), "resume guard failed — populate_tracks overwrote learned Parameter"


def test_buffer_mode_repopulate_replaces(conf_learnable_off):
    """Sanity: buffer-mode behavior unchanged (re-populate overwrites)."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1000)]
    model = LayeredGaussians(conf_learnable_off, specs=specs, scene_extent=1.0)
    tracks_v1 = _viz4d_tracks_dict(F=4)
    tracks_v1[next(iter(tracks_v1))]["cam_timestamps_us"] = torch.tensor([1000, 2000, 3000, 4000], dtype=torch.int64)
    model.populate_tracks(tracks_v1)
    assert model._track_pose_t0.shape == (4, 4, 4)

    tracks_v2 = _viz4d_tracks_dict(F=7)
    tracks_v2[next(iter(tracks_v2))]["cam_timestamps_us"] = torch.tensor(
        [10, 20, 30, 40, 50, 60, 70], dtype=torch.int64
    )
    model.populate_tracks(tracks_v2)
    assert model._track_pose_t0.shape == (7, 4, 4)


# ─── 7. state_dict round-trip preserves Parameter values ────────────────────


def test_state_dict_round_trip_learnable_pose(conf_learnable_on):
    """save → load round-trip preserves the per-track quat/trans Parameter
    values (they ride in ``state_dict()`` automatically as nn.Parameter).
    """
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1000)]
    src = LayeredGaussians(conf_learnable_on, specs=specs, scene_extent=1.0)
    tracks = _viz4d_tracks_dict(F=4)
    tracks[next(iter(tracks))]["cam_timestamps_us"] = torch.tensor([1000, 2000, 3000, 4000], dtype=torch.int64)
    src.populate_tracks(tracks)

    # Train one step so quat/trans differ from GT.
    with torch.no_grad():
        src._track_trans_t0.add_(torch.tensor([1.0, 2.0, 3.0]))
        src._track_quat_t1.add_(torch.tensor([0.0, 0.01, 0.0, 0.0]))

    state = src.state_dict()
    src_trans = src._track_trans_t0.detach().clone()
    src_quat = src._track_quat_t1.detach().clone()

    dst = LayeredGaussians(conf_learnable_on, specs=specs, scene_extent=1.0)
    dst.populate_tracks(tracks)  # registers slot with GT init
    dst.load_state_dict(state)

    assert torch.allclose(dst._track_trans_t0.detach(), src_trans)
    assert torch.allclose(dst._track_quat_t1.detach(), src_quat)
    # Both endpoints are Parameters.
    assert isinstance(dst._track_quat_t0, nn.Parameter)
    assert isinstance(dst._track_trans_t0, nn.Parameter)


# ─── 7b. trainer-format ckpt round-trip (the real production path) ──────────


def test_trainer_format_ckpt_round_trip_learnable_pose(conf_learnable_on):
    """The trainer doesn't use nn.Module.state_dict() — it calls
    ``model.get_model_parameters()`` and writes a structured dict (per-layer
    MoG params + sky_envmap_state + layered_track_state etc.). On load it
    calls ``model.init_from_checkpoint(ckpt)`` which dispatches per layer.

    Before V3 Stage A this path did NOT carry the LayeredGaussians-level
    track buffers — fine in legacy mode (populate_tracks re-derives them
    from manifest each session) but FATAL for learnable mode where the
    Adam-updated Parameters are the only persistent copy of refined pose.

    This test reproduces the trainer save/load contract for learnable mode:
        src → get_model_parameters() → ckpt dict
        dst → populate_tracks(fresh GT) → init_from_checkpoint(ckpt)
        assert dst._track_quat_<tid> == src._track_quat_<tid>
    """
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1000)]
    src = LayeredGaussians(conf_learnable_on, specs=specs, scene_extent=1.0)
    tracks = _viz4d_tracks_dict(F=4)
    tracks[next(iter(tracks))]["cam_timestamps_us"] = torch.tensor([1000, 2000, 3000, 4000], dtype=torch.int64)
    src.populate_tracks(tracks)
    src.setup_optimizer_for_test()  # get_model_parameters() asserts optimizer != None

    # Simulate training: drift quat + trans away from GT.
    with torch.no_grad():
        src._track_trans_t0.add_(torch.tensor([10.0, 20.0, 30.0]))
        src._track_quat_t1.add_(torch.tensor([0.0, 0.05, 0.0, 0.0]))
    src_trans_t0 = src._track_trans_t0.detach().clone()
    src_quat_t1 = src._track_quat_t1.detach().clone()

    # Trainer-format save: get_model_parameters wraps under "model" key.
    ckpt = {"model": src.get_model_parameters()}

    assert (
        "layered_track_state" in ckpt["model"]
    ), "get_model_parameters did NOT emit layered_track_state — Stage A ckpt persistence broken"
    # Spot-check that the wxyz quat and trans Parameters AND _track_pose_gt_
    # and _track_active_ buffers are all present.
    expected = {
        "_track_quat_t0",
        "_track_quat_t1",
        "_track_trans_t0",
        "_track_trans_t1",
        "_track_pose_gt_t0",
        "_track_pose_gt_t1",
        "_track_active_t0",
        "_track_active_t1",
    }
    missing = expected - set(ckpt["model"]["layered_track_state"].keys())
    assert not missing, f"layered_track_state missing entries: {missing}"

    # Trainer-format load: fresh dst + populate_tracks (GT) + init_from_checkpoint.
    dst = LayeredGaussians(conf_learnable_on, specs=specs, scene_extent=1.0)
    dst.populate_tracks(tracks)
    # Before load, dst's quat/trans equal GT (not drifted).
    pre_load_trans_t0 = dst._track_trans_t0.detach().clone()
    assert not torch.allclose(
        pre_load_trans_t0, src_trans_t0
    ), "pre-load dst should still be GT, not src's drifted state"
    dst.init_from_checkpoint(ckpt, setup_optimizer=False)

    # After load: dst's Parameters match src (refined pose restored).
    assert torch.allclose(
        dst._track_trans_t0.detach(), src_trans_t0
    ), "trans Parameter not restored from ckpt via trainer-format path"
    assert torch.allclose(
        dst._track_quat_t1.detach(), src_quat_t1
    ), "quat Parameter not restored from ckpt via trainer-format path"
    # Still nn.Parameter, not buffer.
    assert isinstance(dst._track_quat_t0, nn.Parameter)
    assert isinstance(dst._track_trans_t0, nn.Parameter)


def test_trainer_format_ckpt_round_trip_buffer_mode(conf_learnable_off):
    """Buffer mode: trainer ckpt path round-trips the _track_pose_<tid> +
    _track_active_<tid> buffers too (regression guard — before V3 Stage A
    these were re-derived from manifest each session, so 'not in ckpt' was
    fine; after the fix they ARE in the ckpt and should match)."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1000)]
    src = LayeredGaussians(conf_learnable_off, specs=specs, scene_extent=1.0)
    tracks = _viz4d_tracks_dict(F=4)
    tracks[next(iter(tracks))]["cam_timestamps_us"] = torch.tensor([1000, 2000, 3000, 4000], dtype=torch.int64)
    src.populate_tracks(tracks)
    src.setup_optimizer_for_test()

    ckpt = {"model": src.get_model_parameters()}
    assert "_track_pose_t0" in ckpt["model"].get("layered_track_state", {})
    assert "_track_active_t0" in ckpt["model"].get("layered_track_state", {})

    dst = LayeredGaussians(conf_learnable_off, specs=specs, scene_extent=1.0)
    dst.populate_tracks(tracks)
    dst.init_from_checkpoint(ckpt, setup_optimizer=False)

    assert torch.equal(dst._track_pose_t0, src._track_pose_t0)
    assert torch.equal(dst._track_active_t0, src._track_active_t0)


# ─── 8. _gather_active_tracks_for_batch detaches in learnable mode ──────────
# (This test lives in test_learnable_pose_param rather than
# test_trainer_* because it only exercises the model's property + a tiny
# stand-in for the trainer's snapshot step; trainer construction needs CUDA.)


def test_property_detach_snapshot_breaks_gradient(conf_learnable_on):
    """The trainer's ``_gather_active_tracks_for_batch`` snapshots the
    ``tracks_poses`` property dict into detached tensors before passing
    downstream. Verify the snapshot pattern itself: detached tensors do
    NOT carry gradients back to the source Parameters.
    """
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1000)]
    model = LayeredGaussians(conf_learnable_on, specs=specs, scene_extent=1.0)
    tracks = _viz4d_tracks_dict(F=4)
    tracks[next(iter(tracks))]["cam_timestamps_us"] = torch.tensor([1000, 2000, 3000, 4000], dtype=torch.int64)
    model.populate_tracks(tracks)

    # Live property tensor IS gradient-tracking.
    live = model.tracks_poses["t0"]
    assert live.requires_grad is True

    # Snapshot (what trainer._gather_active_tracks_for_batch does).
    snapshot = {tid: p.detach() for tid, p in model.tracks_poses.items()}
    snap_t0 = snapshot["t0"]
    assert snap_t0.requires_grad is False

    # Snapshot tensor has no autograd link to the source Parameter.
    assert snap_t0.grad_fn is None
    # And the source Parameter does in fact carry gradient (sanity).
    model._track_trans_t0.grad = None
    live_loss = live[:, :3, 3].pow(2).sum()
    live_loss.backward()
    assert model._track_trans_t0.grad is not None
    assert model._track_trans_t0.grad.abs().max().item() > 0.0
