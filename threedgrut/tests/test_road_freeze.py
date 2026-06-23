# SPDX-License-Identifier: Apache-2.0
"""Tests for NuRec-style road geometry freeze (E0.5 port, 2026-06-22).

Two mechanisms:
  1. per-layer ABSOLUTE lr override (LayerSpec.{positions,density,rotation,
     scale,features_albedo}_lr) → LayeredGaussians._apply_layer_lr_overrides
     sets the named param-group lr and drops its scheduler (else scheduler_step
     silently overwrites). Freezes road geometry, keeps albedo learning.
  2. strategy.exclude_layer_ids → LayeredMCMCStrategy skips that layer's
     add/relocate/perturb so the road particle set stays constant.

LayerSpec + registry parts are pure (always run); the model-method test is
guarded by importorskip in case the heavy module can't import on this host.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.layers.registry import specs_from_config


def test_layerspec_per_layer_lr_fields_default_none() -> None:
    s = LayerSpec(name="road", layer_id=2, max_n_particles=200000)
    for f in ("positions_lr", "density_lr", "rotation_lr", "scale_lr", "features_albedo_lr"):
        assert getattr(s, f) is None, f"{f} should default None"


def test_registry_override_routes_per_layer_lr_into_spec() -> None:
    # The CLI form ++layers.overrides.road.positions_lr=1e-6 must land on the spec.
    conf = OmegaConf.create({
        "layers": {
            "enabled": ["background", "road"],
            "overrides": {"road": {
                "positions_lr": 1e-6, "density_lr": 1e-4,
                "rotation_lr": 1e-4, "scale_lr": 1e-4,
            }},
        }
    })
    specs = specs_from_config(conf)
    road = next(s for s in specs if s.name == "road")
    assert road.positions_lr == 1e-6
    assert road.density_lr == 1e-4
    assert road.rotation_lr == 1e-4
    assert road.scale_lr == 1e-4
    assert road.features_albedo_lr is None  # not set → road still learns colour
    bg = next(s for s in specs if s.name == "background")
    assert bg.positions_lr is None  # only road overridden


def _mock_layer():
    # param-group lr values mimic post-setup_optimizer state (positions already
    # × scene_extent=57). schedulers has 'positions' (base_gs schedules it).
    return SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=[
            {"name": "positions", "lr": 0.0016 * 57.0},
            {"name": "density", "lr": 0.05},
            {"name": "rotation", "lr": 1e-3},
            {"name": "scale", "lr": 5e-3},
            {"name": "features_albedo", "lr": 2.5e-3},
        ]),
        schedulers={"positions": (lambda step: 0.001)},
    )


def test_apply_layer_lr_overrides_absolute_and_drops_scheduler() -> None:
    LayeredGaussians = pytest.importorskip(
        "threedgrut.layers.layered_model"
    ).LayeredGaussians
    layer = _mock_layer()
    spec = LayerSpec(
        name="road", layer_id=2, max_n_particles=200000,
        positions_lr=1e-6, density_lr=1e-4, rotation_lr=1e-4, scale_lr=1e-4,
    )
    # method doesn't use self → pass None
    LayeredGaussians._apply_layer_lr_overrides(None, layer, spec)
    lrs = {pg["name"]: pg["lr"] for pg in layer.optimizer.param_groups}
    assert lrs["positions"] == 1e-6   # ABSOLUTE, overrides the ×scene_extent value
    assert lrs["density"] == 1e-4
    assert lrs["rotation"] == 1e-4
    assert lrs["scale"] == 1e-4
    assert lrs["features_albedo"] == 2.5e-3  # untouched → road keeps learning colour
    assert "positions" not in layer.schedulers  # dropped, else scheduler_step overwrites 1e-6


def test_apply_layer_lr_overrides_noop_when_all_none() -> None:
    LayeredGaussians = pytest.importorskip(
        "threedgrut.layers.layered_model"
    ).LayeredGaussians
    layer = _mock_layer()
    spec = LayerSpec(name="road", layer_id=2, max_n_particles=200000)
    LayeredGaussians._apply_layer_lr_overrides(None, layer, spec)
    lrs = {pg["name"]: pg["lr"] for pg in layer.optimizer.param_groups}
    assert abs(lrs["positions"] - 0.0016 * 57.0) < 1e-12  # untouched
    assert "positions" in layer.schedulers  # scheduler kept


# ---------------------------------------------------------------------------
# E3.2.5 ③b — HARD rotation freeze (recon-studio zero_ground_gradients port).
# lr-override (mechanism 1 above) only shrinks the step; Adam momentum still
# drifts a 1e-4 rotation over 30k steps → the disc tilts (spec §5 "未锁法线"
# = a named roadoff-freeze failure cause). freeze_rotation_grad zeroes the
# road rotation grad in _post_backward (after backward / before optimizer.step),
# killing both the update AND the Adam momentum source → the identity-quat
# normal-vertical disc is truly locked. Default False → byte-identical no-op.
# ---------------------------------------------------------------------------


def test_layerspec_freeze_rotation_grad_default_false() -> None:
    s = LayerSpec(name="road", layer_id=2, max_n_particles=200000)
    assert s.freeze_rotation_grad is False


def test_registry_routes_freeze_rotation_grad() -> None:
    conf = OmegaConf.create({
        "layers": {
            "enabled": ["background", "road"],
            "overrides": {"road": {"freeze_rotation_grad": True}},
        }
    })
    specs = specs_from_config(conf)
    road = next(s for s in specs if s.name == "road")
    bg = next(s for s in specs if s.name == "background")
    assert road.freeze_rotation_grad is True
    assert bg.freeze_rotation_grad is False  # only road overridden


def _mock_strategy_self(specs, layers):
    """Minimal mock for LayeredMCMCStrategy._post_backward.

    The method reads only self.model.layers (a dict) + self.specs, so a
    SimpleNamespace stand-in lets us exercise the grad-zero without building
    the heavy strategy (sub-strategies / MoG optimizers).
    """
    return SimpleNamespace(model=SimpleNamespace(layers=layers), specs=specs)


def _layer_with_rot_grad(n: int):
    rot = torch.zeros(n, 4, requires_grad=True)
    rot.grad = torch.ones(n, 4)  # mimic post-backward populated grad
    return SimpleNamespace(rotation=rot), rot


def test_post_backward_zeros_road_rotation_grad() -> None:
    LayeredMCMCStrategy = pytest.importorskip(
        "threedgrut.strategy.layered_mcmc"
    ).LayeredMCMCStrategy
    road_layer, road_rot = _layer_with_rot_grad(5)
    bg_layer, bg_rot = _layer_with_rot_grad(3)
    layers = {"road": road_layer, "background": bg_layer}
    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600000),
        LayerSpec(name="road", layer_id=1, max_n_particles=200000,
                  freeze_rotation_grad=True),
    ]
    mock_self = _mock_strategy_self(specs, layers)
    out = LayeredMCMCStrategy._post_backward(
        mock_self, step=1, scene_extent=1.0, train_dataset=None
    )
    assert torch.count_nonzero(road_rot.grad) == 0  # road rotation grad zeroed
    assert torch.count_nonzero(bg_rot.grad) == bg_rot.grad.numel()  # bg untouched
    assert out is False  # no scene-structure change


def test_post_backward_noop_when_freeze_false() -> None:
    LayeredMCMCStrategy = pytest.importorskip(
        "threedgrut.strategy.layered_mcmc"
    ).LayeredMCMCStrategy
    road_layer, road_rot = _layer_with_rot_grad(5)
    layers = {"road": road_layer}
    specs = [LayerSpec(name="road", layer_id=1, max_n_particles=200000)]  # default False
    mock_self = _mock_strategy_self(specs, layers)
    LayeredMCMCStrategy._post_backward(
        mock_self, step=1, scene_extent=1.0, train_dataset=None
    )
    assert torch.count_nonzero(road_rot.grad) == road_rot.grad.numel()  # untouched
