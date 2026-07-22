# SPDX-License-Identifier: Apache-2.0
"""E0.5 audit follow-up: ``LayerSpec.scale_lr_mult`` must reach the per-layer
Adam. It was dead config from T1.2 until the 2026-06-11 recipe audit
(docs/superpowers/specs/2026-06-11-e05-nurec-vs-multilayer-recipe-diff.md,
"顺手发现") found nothing consumed it.

Coverage:
    (a) mult != 1 scales the 'scale' param-group lr of that layer.
    (b) a sibling layer with default mult keeps the conf lr (per-layer
        isolation).
    (c) every other param group of the mult layer is untouched
        (positions still gets the scene_extent multiply, nothing else moves).
    (d) a conf that schedules the scale group + mult != 1 fails loud:
        MoG.scheduler_step would silently overwrite the multiplier each
        step, so that combination must be rejected at setup time.
    (e) MoG.scheduler_step does not clobber the multiplied lr (scale is
        unscheduled in base_gs.yaml -- pins the invariant (d) relies on).
    (f) registry road default is 1.0 (identity): wiring the field changed
        no anchor-run recipe. 2026-06-11 decision -- the historical 0.2
        never took effect, so E3 experiments opt in explicitly via
        ``++layers.overrides.road.scale_lr_mult=...`` instead of having
        the default silently flip mid-baseline.
"""

from __future__ import annotations

import os

import pytest
import torch
from hydra import compose, initialize_config_dir

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.layers.layered_model import LayeredGaussians

_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs"))

_SCENE_EXTENT = 10.0


@pytest.fixture(scope="module")
def real_conf():
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="apps/ncore_3dgut_mcmc")


def _specs(road_mult: float) -> list[LayerSpec]:
    return [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(
            name="road",
            layer_id=1,
            max_n_particles=200_000,
            scale_prior=(0.1, 0.1, 0.001),
            scale_lr_mult=road_mult,
            mask_field="road_mask",
        ),
    ]


def _init_model(conf, specs: list[LayerSpec]) -> LayeredGaussians:
    model = LayeredGaussians(conf, specs=specs, scene_extent=_SCENE_EXTENT)
    for s in specs:
        model.init_layer_from_points(s.name, torch.randn(8, 3), setup_optimizer=True)
    return model


def _group(optimizer, name: str) -> dict:
    return next(g for g in optimizer.param_groups if g["name"] == name)


# --- (a) the multiplier reaches the scale group ------------------------------
def test_scale_lr_mult_applies_to_scale_group(real_conf):
    model = _init_model(real_conf, _specs(road_mult=0.2))
    base = float(real_conf.optimizer.params.scale.lr)
    road_lr = _group(model.layers["road"].optimizer, "scale")["lr"]
    assert road_lr == pytest.approx(0.2 * base)


# --- (b) sibling layer with default mult is isolated -------------------------
def test_default_mult_layer_keeps_conf_lr(real_conf):
    model = _init_model(real_conf, _specs(road_mult=0.2))
    base = float(real_conf.optimizer.params.scale.lr)
    bg_lr = _group(model.layers["background"].optimizer, "scale")["lr"]
    assert bg_lr == pytest.approx(base)


# --- (c) only the scale group of the mult layer moves -------------------------
def test_other_param_groups_untouched(real_conf):
    model = _init_model(real_conf, _specs(road_mult=0.2))
    road_opt = model.layers["road"].optimizer
    params_conf = real_conf.optimizer.params
    # positions keeps the v1 scene_extent multiply -- and nothing else.
    assert _group(road_opt, "positions")["lr"] == pytest.approx(float(params_conf.positions.lr) * _SCENE_EXTENT)
    for name in ("density", "rotation", "features_albedo", "features_specular"):
        assert _group(road_opt, name)["lr"] == pytest.approx(float(getattr(params_conf, name).lr)), name


# --- (d) scheduled scale group + mult != 1 must fail loud ---------------------
def _conf_with_scale_scheduler():
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(
            config_name="apps/ncore_3dgut_mcmc",
            overrides=[
                "+scheduler.scale.type=exp",
                "+scheduler.scale.lr_init=0.005",
                "+scheduler.scale.lr_final=0.0005",
                "+scheduler.scale.max_steps=1000",
            ],
        )


def test_scheduled_scale_group_with_mult_fails_loud():
    conf = _conf_with_scale_scheduler()
    with pytest.raises(ValueError, match="scale_lr_mult"):
        _init_model(conf, _specs(road_mult=0.2))


def test_scheduled_scale_group_with_identity_mult_is_allowed():
    conf = _conf_with_scale_scheduler()
    _init_model(conf, _specs(road_mult=1.0))  # must not raise


# --- (e) scheduler_step keeps the multiplied lr -------------------------------
def test_scheduler_step_keeps_multiplied_scale_lr(real_conf):
    model = _init_model(real_conf, _specs(road_mult=0.2))
    road = model.layers["road"]
    base = float(real_conf.optimizer.params.scale.lr)
    pos_before = _group(road.optimizer, "positions")["lr"]
    road.scheduler_step(500)
    # sanity: the positions scheduler did run...
    assert _group(road.optimizer, "positions")["lr"] != pos_before
    # ...and the multiplied scale lr survived.
    assert _group(road.optimizer, "scale")["lr"] == pytest.approx(0.2 * base)


# --- (f) registry default is identity (baseline parity) -----------------------
def test_registry_road_default_mult_is_identity():
    from threedgrut.layers.registry import STANDARD_LAYERS

    assert STANDARD_LAYERS["road"].scale_lr_mult == 1.0


def _resume_specs(*, road_scale_lr: float | None) -> list[LayerSpec]:
    return [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(
            name="road",
            layer_id=1,
            max_n_particles=200_000,
            scale_prior=(0.1, 0.1, 0.001),
            scale_lr=road_scale_lr,
            mask_field="road_mask",
        ),
    ]


def _checkpoint_with_custom_saved_road_lr(real_conf, saved_lr: float) -> dict:
    source = _init_model(real_conf, _resume_specs(road_scale_lr=None))
    _group(source.layers["road"].optimizer, "scale")["lr"] = saved_lr
    return {"gaussians_nodes": source.get_model_parameters()["gaussians_nodes"]}


def test_absolute_scale_lr_override_wins_after_optimizer_resume(real_conf):
    checkpoint = _checkpoint_with_custom_saved_road_lr(real_conf, saved_lr=0.005)
    target = LayeredGaussians(
        real_conf,
        specs=_resume_specs(road_scale_lr=1.0e-4),
        scene_extent=_SCENE_EXTENT,
    )
    target.init_from_checkpoint(checkpoint, setup_optimizer=True)
    assert _group(target.layers["road"].optimizer, "scale")["lr"] == pytest.approx(1.0e-4)


def test_resume_without_absolute_override_preserves_saved_lr(real_conf):
    checkpoint = _checkpoint_with_custom_saved_road_lr(real_conf, saved_lr=0.0042)
    target = LayeredGaussians(
        real_conf,
        specs=_resume_specs(road_scale_lr=None),
        scene_extent=_SCENE_EXTENT,
    )
    target.init_from_checkpoint(checkpoint, setup_optimizer=True)
    assert _group(target.layers["road"].optimizer, "scale")["lr"] == pytest.approx(0.0042)
