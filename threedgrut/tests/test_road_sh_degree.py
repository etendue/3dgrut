# SPDX-License-Identifier: Apache-2.0
"""V3-R1.1 unit tests for per-layer SH degree override via LayerSpec.sh_degree."""
from __future__ import annotations

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.layers.registry import STANDARD_LAYERS


def test_layerspec_default_sh_degree_is_none():
    """sh_degree defaults to None → use global progressive_training.max_n_features."""
    spec = LayerSpec(name="x", layer_id=0, max_n_particles=100)
    assert spec.sh_degree is None


def test_layerspec_accepts_explicit_sh_degree():
    """Non-default integer value round-trips through frozen dataclass construction."""
    spec = LayerSpec(name="x", layer_id=0, max_n_particles=100, sh_degree=1)
    assert spec.sh_degree == 1


def test_road_layer_sh_degree_is_1():
    """V3-R1.1 acceptance: road layer caps SH at degree 1 (DC + 3 linear)."""
    assert STANDARD_LAYERS["road"].sh_degree == 1


def test_background_layer_sh_degree_default():
    """Background layer keeps default (None = use global)."""
    assert STANDARD_LAYERS["background"].sh_degree is None


def test_specs_from_config_can_override_sh_degree():
    """yaml override (layers.overrides.road.sh_degree) reaches the spec — the
    only production path for changing the registry default."""
    from omegaconf import OmegaConf
    from threedgrut.layers.registry import specs_from_config

    conf = OmegaConf.create({
        "layers": {
            "enabled": ["road"],
            "overrides": {"road": {"sh_degree": 3}},
        }
    })
    specs = specs_from_config(conf)
    assert specs[0].sh_degree == 3
