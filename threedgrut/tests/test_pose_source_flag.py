# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V3 Stage D.2 — pose_source flag unit tests.

Pins:
  * ``set_pose_source("learned"|"gt")`` toggles the route ``_compose_pose_*``
    takes; default is "learned".
  * ``pose_source == "gt"`` AND ``_track_pose_gt_<tid>`` exists → returns
    the frozen GT buffer slice/value (no gradient).
  * ``pose_source == "gt"`` AND no ``_track_pose_gt_`` (e.g. buffer-only
    legacy ckpt) → falls through to learned route.
  * Bad source string raises ValueError.

These guarantees back ``scripts/render_learned_vs_gt.py`` — the only
consumer toggles this flag between two forward passes on the same model.
"""
import os
import sys

import pytest
import torch
from hydra import compose, initialize_config_dir

from threedgrut.layers.layer_spec import LayerSpec

# Mirror test_learnable_pose_param.py layout.
_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs")
)


@pytest.fixture(scope="module")
def conf_learnable_on():
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(
            config_name="apps/ncore_3dgut_mcmc",
            overrides=["trainer.learnable_pose.enabled=true"],
        )


@pytest.fixture(scope="module")
def conf_learnable_off():
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(
            config_name="apps/ncore_3dgut_mcmc",
            overrides=["trainer.learnable_pose.enabled=false"],
        )


def _viz4d_tracks_dict(F: int = 4) -> dict:
    """Build a track dict matching ``ckpt["viz_4d"]["tracks"]`` schema."""
    tracks = {}
    for i, tid in enumerate(["t0", "t1"]):
        poses = torch.eye(4).repeat(F, 1, 1)
        for f in range(F):
            yaw = torch.tensor(0.1 * (f + 1) + 0.5 * i, dtype=torch.float32)
            c, s = torch.cos(yaw), torch.sin(yaw)
            poses[f, 0, 0] =  c; poses[f, 0, 1] = -s
            poses[f, 1, 0] =  s; poses[f, 1, 1] =  c
            poses[f, 0, 3] = float(f) + i * 100.0
        tracks[tid] = {
            "poses":      poses,
            "size":       torch.tensor([2.0, 1.5, 4.5], dtype=torch.float32),
            "frame_info": torch.ones(F, dtype=torch.bool),
            "class":      "automobile" if i == 0 else "heavy_truck",
        }
    return tracks


def _populated_learnable_model(conf, F=4):
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1000)]
    model = LayeredGaussians(conf, specs=specs, scene_extent=1.0)
    tracks = _viz4d_tracks_dict(F=F)
    tracks[next(iter(tracks))]["cam_timestamps_us"] = torch.tensor(
        [1000 * (i + 1) for i in range(F)], dtype=torch.int64
    )
    model.populate_tracks(tracks)
    return model


# ─── 1. Default is "learned"; setter validates ──────────────────────────────


def test_pose_source_default_is_learned(conf_learnable_on):
    model = _populated_learnable_model(conf_learnable_on)
    assert model.pose_source == "learned"


def test_set_pose_source_rejects_unknown_value(conf_learnable_on):
    model = _populated_learnable_model(conf_learnable_on)
    with pytest.raises(ValueError, match="must be 'learned' or 'gt'"):
        model.set_pose_source("ground_truth")
    with pytest.raises(ValueError):
        model.set_pose_source("")
    # Valid round-trip still works.
    model.set_pose_source("gt")
    assert model.pose_source == "gt"
    model.set_pose_source("learned")
    assert model.pose_source == "learned"


# ─── 2. gt route returns _track_pose_gt_ slice when learnable + gt exists ───


def test_compose_for_track_gt_returns_gt_buffer(conf_learnable_on):
    """Manually drift the learned quat/trans; pose_source='gt' must still
    return the frozen GT pose buffer (unaffected by the drift)."""
    model = _populated_learnable_model(conf_learnable_on, F=4)
    pose_gt_t0_frame2 = model._track_pose_gt_t0[2].clone()

    # Drift learned params heavily (simulate post-training state).
    with torch.no_grad():
        model._track_trans_t0.add_(torch.tensor([10.0, 20.0, 30.0]))
        model._track_quat_t0.add_(torch.tensor([0.0, 0.5, 0.5, 0.5]))

    # learned route picks up the drift
    model.set_pose_source("learned")
    learned = model._compose_pose_for_track("t0", 2)
    assert not torch.allclose(learned[:3, 3], pose_gt_t0_frame2[:3, 3], atol=1e-3), \
        "drift should be visible in learned route"

    # gt route ignores the drift
    model.set_pose_source("gt")
    got_gt = model._compose_pose_for_track("t0", 2)
    assert torch.allclose(got_gt, pose_gt_t0_frame2, atol=1e-6), \
        f"gt route should match _track_pose_gt_t0[2] exactly, got\n{got_gt}\nvs\n{pose_gt_t0_frame2}"


def test_compose_all_frames_gt_returns_full_gt_buffer(conf_learnable_on):
    """Batched variant: shape [F, 4, 4] under gt route equals _track_pose_gt_."""
    model = _populated_learnable_model(conf_learnable_on, F=4)
    expected = model._track_pose_gt_t1.clone()

    # Drift learned params.
    with torch.no_grad():
        model._track_trans_t1.add_(torch.tensor([1.0, 2.0, 3.0]))

    model.set_pose_source("gt")
    got = model._compose_pose_all_frames("t1")
    assert got.shape == (4, 4, 4)
    assert torch.allclose(got, expected, atol=1e-6)


# ─── 3. gt route doesn't require grad ───────────────────────────────────────


def test_gt_route_no_grad(conf_learnable_on):
    """The gt buffer is registered with persistent=True and is NOT a Parameter
    → returned tensor must not require grad (so we can render without an
    autograd graph)."""
    model = _populated_learnable_model(conf_learnable_on, F=4)
    model.set_pose_source("gt")
    p = model._compose_pose_for_track("t0", 1)
    assert not p.requires_grad
    p_all = model._compose_pose_all_frames("t0")
    assert not p_all.requires_grad


# ─── 4. learned route still uses Parameters ─────────────────────────────────


def test_learned_route_uses_parameter(conf_learnable_on):
    model = _populated_learnable_model(conf_learnable_on, F=4)
    model.set_pose_source("learned")
    p = model._compose_pose_for_track("t0", 1)
    # Composed from quat/trans Parameter → requires_grad True.
    assert p.requires_grad


# ─── 5. tracks_poses @property reflects flag ────────────────────────────────


def test_tracks_poses_property_respects_pose_source(conf_learnable_on):
    model = _populated_learnable_model(conf_learnable_on, F=4)
    gt_t0 = model._track_pose_gt_t0.clone()

    # Drift.
    with torch.no_grad():
        model._track_trans_t0.add_(torch.tensor([5.0, 5.0, 5.0]))

    model.set_pose_source("learned")
    learned_dict = model.tracks_poses
    assert not torch.allclose(learned_dict["t0"], gt_t0, atol=1e-3)

    model.set_pose_source("gt")
    gt_dict = model.tracks_poses
    assert torch.allclose(gt_dict["t0"], gt_t0, atol=1e-6)


# ─── 6. Buffer-only mode (no _track_pose_gt_) → gt falls through to buffer ──


def test_gt_falls_through_in_buffer_mode(conf_learnable_off):
    """In legacy buffer mode there is no ``_track_pose_gt_<tid>``; setting
    pose_source='gt' must NOT crash — it falls through to the buffer route
    (which is itself the GT in that mode, since nothing learns)."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1000)]
    model = LayeredGaussians(conf_learnable_off, specs=specs, scene_extent=1.0)
    tracks = _viz4d_tracks_dict(F=4)
    tracks[next(iter(tracks))]["cam_timestamps_us"] = torch.tensor(
        [1000, 2000, 3000, 4000], dtype=torch.int64
    )
    model.populate_tracks(tracks)

    # No _track_pose_gt_t0 in buffer mode.
    assert getattr(model, "_track_pose_gt_t0", None) is None
    # _track_pose_t0 IS the active pose (and the only pose).
    expected = model._track_pose_t0[2].clone()

    model.set_pose_source("gt")
    got = model._compose_pose_for_track("t0", 2)
    assert torch.allclose(got, expected, atol=1e-6)
