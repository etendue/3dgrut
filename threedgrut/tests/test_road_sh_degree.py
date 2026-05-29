# SPDX-License-Identifier: Apache-2.0
"""V3-R1.1 unit tests for per-layer SH degree override via LayerSpec.sh_degree."""
from __future__ import annotations

import os

import pytest
import torch

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.layers.registry import STANDARD_LAYERS
from threedgrut.utils.misc import sh_degree_to_specular_dim


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


# ----------------------------------------------------------------------- conf

_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs")
)


@pytest.fixture(scope="module")
def real_conf():
    """Hydra-composed full conf from apps/ncore_3dgut_mcmc_multilayer.

    Module-scoped so Tracer CUDA compile (mocked in conftest.py) happens at
    most once per test session.
    """
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="apps/ncore_3dgut_mcmc_multilayer")


def test_init_layer_from_points_road_uses_spec_sh_degree(real_conf):
    """V3-R1.1: road layer gets sh_degree=1 width, not global degree-3 width.

    init_layer_from_points with setup_optimizer=False allocates features_specular
    with sh_degree_to_specular_dim(spec.sh_degree) = 9 when spec.sh_degree=1,
    not sh_degree_to_specular_dim(layer.max_n_features) = 45 for degree-3.
    """
    from threedgrut.layers.layered_model import LayeredGaussians
    from threedgrut.layers.registry import specs_from_config

    from omegaconf import OmegaConf

    # Override enabled layers to just road for this test
    conf = OmegaConf.merge(
        real_conf,
        OmegaConf.create({"layers": {"enabled": ["road"]}}),
    )
    specs = specs_from_config(conf)
    assert specs[0].name == "road"
    assert specs[0].sh_degree == 1, "precondition: road spec must have sh_degree=1"

    lg = LayeredGaussians(conf=conf, specs=specs, scene_extent=10.0)
    N = 16
    positions = torch.randn(N, 3)
    lg.init_layer_from_points("road", positions, setup_optimizer=False)

    road_layer = lg.layers["road"]
    expected_specular_dim = sh_degree_to_specular_dim(1)  # 9
    assert road_layer.features_specular.shape == (N, expected_specular_dim), (
        f"road features_specular shape = {road_layer.features_specular.shape}, "
        f"expected ({N}, {expected_specular_dim}) for sh_degree=1; "
        f"got degree-3 width ({sh_degree_to_specular_dim(3)}) instead?"
    )


def test_init_layer_from_points_background_uses_global_sh_degree(real_conf):
    """V3-R1.1 companion: background layer (spec.sh_degree=None) inherits global
    conf.model.progressive_training.max_n_features (degree 3 → dim 45)."""
    from threedgrut.layers.layered_model import LayeredGaussians
    from threedgrut.layers.registry import specs_from_config
    from omegaconf import OmegaConf

    conf = OmegaConf.merge(
        real_conf,
        OmegaConf.create({"layers": {"enabled": ["background"]}}),
    )
    specs = specs_from_config(conf)
    assert specs[0].name == "background"
    assert specs[0].sh_degree is None, "precondition: background spec.sh_degree must be None"

    lg = LayeredGaussians(conf=conf, specs=specs, scene_extent=10.0)
    N = 8
    positions = torch.randn(N, 3)
    lg.init_layer_from_points("background", positions, setup_optimizer=False)

    bg_layer = lg.layers["background"]
    global_max = conf.model.progressive_training.max_n_features  # 3
    expected_specular_dim = sh_degree_to_specular_dim(global_max)  # 45
    assert bg_layer.features_specular.shape == (N, expected_specular_dim), (
        f"background features_specular shape = {bg_layer.features_specular.shape}, "
        f"expected ({N}, {expected_specular_dim}) for global sh_degree={global_max}"
    )
