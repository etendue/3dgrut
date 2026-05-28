# SPDX-License-Identifier: Apache-2.0
"""V3-L8/L9 unit tests for the warmup gate + Adam optimizer attachment.

Coverage:
    (a) ``maybe_activate_track_params`` returns False before warmup, True
        once on the warmup step, False forever after.
    (b) After activation ``requires_grad`` is True on registered tables.
    (c) ``setup_optimizer`` attaches a ``_track_optim`` Adam when either
        table is registered; the optimizer's param_groups carry the
        correct lr from spec.extra.
    (d) ``setup_optimizer`` does NOT attach ``_track_optim`` when both
        toggles are OFF (regression pin for ckpt schema).
    (e) ``LayeredGaussians.optimizer`` _LayeredOptimizerView includes
        ``_track_optim.param_groups`` in its aggregate view (so the
        trainer's ``self.model.optimizer.step()`` updates the tables).

The conftest pattern matches test_track_albedo_scale_params.py.
"""
from __future__ import annotations

import os

import pytest
import torch
from hydra import compose, initialize_config_dir

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.layers.layered_model import LayeredGaussians

_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs")
)


@pytest.fixture(scope="module")
def real_conf():
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="apps/ncore_3dgut_mcmc")


def _build_model(real_conf, *, albedo: bool, scale: bool, warmup: int = 50,
                 albedo_lr: float = 1e-5, scale_lr: float = 1e-5):
    extra = {"track_warmup_steps": warmup,
             "track_albedo_lr": albedo_lr, "track_scale_lr": scale_lr}
    if albedo:
        extra["optimize_track_albedo"] = True
    if scale:
        extra["optimize_track_scale"] = True
    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="dynamic_rigids", layer_id=2, max_n_particles=200_000,
                  scale_prior=(0.05, 0.05, 0.05), extra=extra),
    ]
    eye = torch.eye(4).expand(5, 4, 4).clone()
    tracks = {
        "alice": {"poses": eye.clone(), "active": torch.ones(5, dtype=torch.bool),
                  "class": "automobile", "size": torch.tensor([4.0, 2.0, 1.5])},
        "bob":   {"poses": eye.clone(), "active": torch.ones(5, dtype=torch.bool),
                  "class": "automobile", "size": torch.tensor([4.0, 2.0, 1.5])},
    }
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0,
                             tracks=tracks)
    model.init_layer_from_points("background", torch.randn(5, 3),
                                 setup_optimizer=False)
    track_names = sorted(tracks.keys())
    all_pts = []
    all_ids = []
    for tid in track_names:
        all_pts.append(torch.zeros(4, 3))
        all_ids.append(torch.full((4,), track_names.index(tid), dtype=torch.long))
    model.init_layer_from_points("dynamic_rigids",
                                 torch.cat(all_pts),
                                 track_ids=torch.cat(all_ids),
                                 setup_optimizer=False)
    return model


# --- (a) warmup gate timing ------------------------------------------------
def test_maybe_activate_track_params_warmup_timing(real_conf):
    model = _build_model(real_conf, albedo=True, scale=True, warmup=10)
    # Before warmup: returns False, requires_grad stays False
    for step in [0, 1, 5, 9]:
        assert model.maybe_activate_track_params(step) is False
        assert model._track_albedo_table.requires_grad is False
        assert model._track_log_scale_table.requires_grad is False
    # At step == warmup: returns True (exactly once), flips requires_grad
    assert model.maybe_activate_track_params(10) is True
    assert model._track_albedo_table.requires_grad is True
    assert model._track_log_scale_table.requires_grad is True
    # After warmup: returns False (no double-flip), requires_grad stays True
    for step in [11, 100, 5000]:
        assert model.maybe_activate_track_params(step) is False
        assert model._track_albedo_table.requires_grad is True
        assert model._track_log_scale_table.requires_grad is True


def test_maybe_activate_track_params_off_mode_is_noop(real_conf):
    """OFF mode (no tables) → maybe_activate is a cheap no-op returning False."""
    model = _build_model(real_conf, albedo=False, scale=False)
    for step in [0, 10, 1000]:
        assert model.maybe_activate_track_params(step) is False
    assert not hasattr(model, "_track_albedo_table")
    assert not hasattr(model, "_track_log_scale_table")


# --- (b/c/d) optimizer attach -----------------------------------------------
def test_setup_optimizer_attaches_track_optim_when_enabled(real_conf):
    model = _build_model(real_conf, albedo=True, scale=True,
                         albedo_lr=2.0e-5, scale_lr=3.0e-5)
    # Per-layer particle optimizers were already set during init_layer_from_points;
    # explicit setup_optimizer() triggers sky_envmap + V3-L8/L9 attach.
    model.setup_optimizer()
    assert hasattr(model, "_track_optim")
    opt = model._track_optim
    assert isinstance(opt, torch.optim.Adam)
    # Two param groups (albedo + scale), named correctly with their LRs.
    by_name = {g.get("name"): g for g in opt.param_groups}
    assert "track_albedo" in by_name
    assert "track_log_scale" in by_name
    assert by_name["track_albedo"]["lr"] == pytest.approx(2.0e-5)
    assert by_name["track_log_scale"]["lr"] == pytest.approx(3.0e-5)


def test_setup_optimizer_skips_track_optim_when_off(real_conf):
    model = _build_model(real_conf, albedo=False, scale=False)
    model.setup_optimizer()
    assert not hasattr(model, "_track_optim") or model._track_optim is None


def test_setup_optimizer_albedo_only_skips_scale_group(real_conf):
    model = _build_model(real_conf, albedo=True, scale=False)
    model.setup_optimizer()
    assert hasattr(model, "_track_optim")
    by_name = {g.get("name") for g in model._track_optim.param_groups}
    assert "track_albedo" in by_name
    assert "track_log_scale" not in by_name


# --- (e) optimizer view aggregates extras -----------------------------------
def test_layered_optimizer_view_includes_track_optim(real_conf):
    """``LayeredGaussians.optimizer`` is the trainer's entry point; it
    must surface ``_track_optim.param_groups`` so ``optimizer.step()``
    updates the tables once warmup flips ``requires_grad``."""
    model = _build_model(real_conf, albedo=True, scale=True)
    model.setup_optimizer()
    view = model.optimizer
    # The view's param_groups must include at least one group whose
    # `params` list is the albedo table (and one for log_scale).
    albedo_id = id(model._track_albedo_table)
    log_scale_id = id(model._track_log_scale_table)
    found_ids = set()
    for g in view.param_groups:
        for p in g["params"]:
            found_ids.add(id(p))
    assert albedo_id in found_ids
    assert log_scale_id in found_ids


def test_layered_optimizer_view_no_track_optim_when_off(real_conf):
    """Symmetric to setup_optimizer OFF: view ignores _track_optim absence."""
    model = _build_model(real_conf, albedo=False, scale=False)
    model.setup_optimizer()
    view = model.optimizer
    # No track_albedo / track_log_scale group names; only sub-layer groups.
    group_names = {g.get("name") for g in view.param_groups}
    assert "track_albedo" not in group_names
    assert "track_log_scale" not in group_names


# --- (f) optimizer.step actually mutates the tables once activated ---------
def test_track_optim_step_mutates_table_after_warmup(real_conf):
    model = _build_model(real_conf, albedo=True, scale=False, warmup=0,
                         albedo_lr=1e-2)
    model.setup_optimizer()
    # Activate immediately (warmup=0).
    model.maybe_activate_track_params(0)
    table = model._track_albedo_table
    assert table.requires_grad is True
    # Inject a manual grad so Adam.step() has something to apply.
    table.grad = torch.full_like(table, 1.0)
    val_before = table.detach().clone()
    model._track_optim.step()
    val_after = table.detach().clone()
    # Adam with grad=1, lr=1e-2 → table shifts by ~-lr after one step
    # (sign matches grad direction). Just check non-identity.
    assert not torch.equal(val_before, val_after)
