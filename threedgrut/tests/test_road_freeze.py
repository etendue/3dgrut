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
