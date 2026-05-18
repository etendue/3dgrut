# SPDX-License-Identifier: Apache-2.0
"""Pure-Python unit tests for LayerSpec dataclass + layers/registry.

These tests deliberately avoid importing torch / MixtureOfGaussians / hydra
compose so they can run on a developer laptop without CUDA. The heavier
container/checkpoint contract tests live in test_layered_gaussians.py and
require the A800 environment.

Coverage:
  T1.2 part 1 (LayerSpec extended fields)  -- test_layer_spec_*
  T1.2 part 2 (registry STANDARD_LAYERS)   -- test_registry_*
  T1.2 part 3 (specs_from_config factory)  -- test_specs_from_config_*
"""
from dataclasses import FrozenInstanceError

import pytest
from omegaconf import OmegaConf

from threedgrut.layers.layer_spec import LayerSpec


# --------------------------------------------------------------- T1.2 part 1
def test_layer_spec_frozen_immutable():
    """LayerSpec 是 frozen dataclass：运行时改字段抛 FrozenInstanceError"""
    spec = LayerSpec(name="background", layer_id=0, max_n_particles=100)
    with pytest.raises(FrozenInstanceError):
        spec.name = "road"


def test_layer_spec_full_field_defaults():
    """T1.2 字段：未显式传时使用默认值，传时透传"""
    minimal = LayerSpec(name="background", layer_id=0, max_n_particles=100)
    assert minimal.scale_prior == (0.1, 0.1, 0.1)
    assert minimal.scale_lr_mult == 1.0
    assert minimal.mask_field is None
    assert minimal.is_particle_layer is True
    assert minimal.density_init == 0.1

    explicit = LayerSpec(
        name="road", layer_id=1, max_n_particles=200_000,
        scale_prior=(0.1, 0.1, 0.001), scale_lr_mult=0.2,
        mask_field="road_mask", is_particle_layer=True, density_init=0.05,
    )
    assert explicit.scale_prior == (0.1, 0.1, 0.001)
    assert explicit.scale_lr_mult == 0.2
    assert explicit.mask_field == "road_mask"
    assert explicit.density_init == 0.05


# --------------------------------------------------------------- T1.2 part 2
def test_registry_standard_layers_complete():
    """STANDARD_LAYERS 含 5 个 v2 标准层"""
    from threedgrut.layers.registry import STANDARD_LAYERS
    assert set(STANDARD_LAYERS.keys()) == {
        "background", "road", "dynamic_rigids",
        "dynamic_deformables", "sky_envmap",
    }


def test_registry_specs_have_unique_ids():
    """每个 layer 的 layer_id 唯一"""
    from threedgrut.layers.registry import STANDARD_LAYERS
    ids = [s.layer_id for s in STANDARD_LAYERS.values()]
    assert len(ids) == len(set(ids))


def test_registry_particle_flags_correct():
    """sky_envmap 和 dynamic_deformables 不是粒子层"""
    from threedgrut.layers.registry import STANDARD_LAYERS
    assert STANDARD_LAYERS["background"].is_particle_layer is True
    assert STANDARD_LAYERS["road"].is_particle_layer is True
    assert STANDARD_LAYERS["dynamic_rigids"].is_particle_layer is True
    assert STANDARD_LAYERS["sky_envmap"].is_particle_layer is False
    assert STANDARD_LAYERS["dynamic_deformables"].is_particle_layer is False


def test_registry_road_layer_has_flat_scale_prior():
    """Road 层 scale_prior 第三维必须远小于前两维（thin-disc Z-lock 约定）"""
    from threedgrut.layers.registry import STANDARD_LAYERS
    sx, sy, sz = STANDARD_LAYERS["road"].scale_prior
    assert sz < sx
    assert sz < sy
    assert STANDARD_LAYERS["road"].mask_field == "road_mask"


# --------------------------------------------------------------- T1.2 part 3
def test_specs_from_config_filters_enabled():
    """specs_from_config 只返回 enabled list 中的 layer，保持顺序"""
    from threedgrut.layers.registry import specs_from_config
    conf = OmegaConf.create({"layers": {"enabled": ["background", "road"]}})
    specs = specs_from_config(conf)
    assert [s.name for s in specs] == ["background", "road"]


def test_specs_from_config_unknown_layer_raises():
    """enabled 含未知 layer name → 明确 ValueError 提示可选名"""
    from threedgrut.layers.registry import specs_from_config
    conf = OmegaConf.create({"layers": {"enabled": ["background", "bogus"]}})
    with pytest.raises(ValueError, match="bogus"):
        specs_from_config(conf)


def test_specs_from_config_defaults_to_single_background():
    """conf 无 layers 字段 → fallback 单 background"""
    from threedgrut.layers.registry import specs_from_config
    conf = OmegaConf.create({})
    specs = specs_from_config(conf)
    assert len(specs) == 1
    assert specs[0].name == "background"
