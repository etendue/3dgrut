# SPDX-License-Identifier: Apache-2.0
"""V3-R1.1 unit tests for LayerSpec.sh_degree field.

NOTE: per-layer SH-degree reduction by shrinking features_specular is
DISABLED (reserved for future freeze-based redesign).  The field still
exists on LayerSpec and round-trips through the dataclass / yaml config, but
init_layer_from_points always uses layer.max_n_features (the global degree)
so that the fused-view renderer's uniform-width invariant is preserved.
"""

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


def test_road_layer_sh_degree_is_none_reserved():
    """V3-R1.1 (disabled): road layer's sh_degree is None (reserved/unused).

    Per-layer SH-degree reduction by shrinking features_specular is
    incompatible with the fused-view renderer: all particle layers must share
    one specular width (the renderer uses the reference layer's max_n_features).
    A future freeze-based approach would consume LayerSpec.sh_degree — the
    field stays in place so that redesign is a small change.
    """
    assert STANDARD_LAYERS["road"].sh_degree is None


def test_background_layer_sh_degree_default():
    """Background layer keeps default (None = use global)."""
    assert STANDARD_LAYERS["background"].sh_degree is None


def test_specs_from_config_can_override_sh_degree():
    """yaml override (layers.overrides.road.sh_degree) reaches the spec — the
    only production path for changing the registry default."""
    from omegaconf import OmegaConf

    from threedgrut.layers.registry import specs_from_config

    conf = OmegaConf.create(
        {
            "layers": {
                "enabled": ["road"],
                "overrides": {"road": {"sh_degree": 3}},
            }
        }
    )
    specs = specs_from_config(conf)
    assert specs[0].sh_degree == 3


# ----------------------------------------------------------------------- conf

_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs"))


@pytest.fixture(scope="module")
def real_conf():
    """Hydra-composed full conf from apps/ncore_3dgut_mcmc_multilayer.

    Module-scoped so Tracer CUDA compile (mocked in conftest.py) happens at
    most once per test session.
    """
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="apps/ncore_3dgut_mcmc_multilayer")


def test_init_layer_from_points_road_uses_global_sh_degree(real_conf):
    """Fused-renderer constraint: road layer uses the GLOBAL SH degree width.

    The fused-view renderer concatenates ALL particle layers' features_specular
    into one tensor and evaluates a single global SH degree from the reference
    layer's max_n_features.  Therefore every layer — including road — must
    allocate features_specular with sh_degree_to_specular_dim(layer.max_n_features)
    = 45 for global degree-3, regardless of spec.sh_degree.

    LayerSpec.sh_degree is reserved for a future freeze-based redesign (keep
    width 45, zero+freeze road's order>=2 coefficients) and is currently unused
    by init_layer_from_points.
    """
    from omegaconf import OmegaConf

    from threedgrut.layers.layered_model import LayeredGaussians
    from threedgrut.layers.registry import specs_from_config

    # Override enabled layers to just road for this test
    conf = OmegaConf.merge(
        real_conf,
        OmegaConf.create({"layers": {"enabled": ["road"]}}),
    )
    specs = specs_from_config(conf)
    assert specs[0].name == "road"
    assert specs[0].sh_degree is None, "precondition: road spec.sh_degree must be None (reserved/unused)"

    lg = LayeredGaussians(conf=conf, specs=specs, scene_extent=10.0)
    N = 16
    positions = torch.randn(N, 3)
    lg.init_layer_from_points("road", positions, setup_optimizer=False)

    road_layer = lg.layers["road"]
    global_max = conf.model.progressive_training.max_n_features  # 3
    expected_specular_dim = sh_degree_to_specular_dim(global_max)  # 45
    assert road_layer.features_specular.shape == (N, expected_specular_dim), (
        f"road features_specular shape = {road_layer.features_specular.shape}, "
        f"expected ({N}, {expected_specular_dim}) for global sh_degree={global_max}; "
        f"fused renderer requires uniform specular width across all layers."
    )


def test_init_layer_from_points_background_uses_global_sh_degree(real_conf):
    """V3-R1.1 companion: background layer (spec.sh_degree=None) inherits global
    conf.model.progressive_training.max_n_features (degree 3 → dim 45)."""
    from omegaconf import OmegaConf

    from threedgrut.layers.layered_model import LayeredGaussians
    from threedgrut.layers.registry import specs_from_config

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
