# SPDX-License-Identifier: Apache-2.0
"""T1.1 contract tests for LayeredGaussians container.

These tests rely on the real Hydra config (`apps/ncore_3dgut_mcmc`) because
`MixtureOfGaussians.__init__` constructs a real Tracer that reads many config
keys. The Tracer's CUDA extension is JIT-compiled once and cached.

To skip optimizer setup in init_from_checkpoint (which would need the full
optimizer/scheduler conf trees), tests pass `setup_optimizer=False`.

The 6 per-particle Parameters use `max_n_features=3` from the real config,
so `features_specular` has dim `sh_degree_to_specular_dim(3) = 45`.
"""

import os

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.utils.misc import sh_degree_to_specular_dim


# ----------------------------------------------------------------------- conf
_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs")
)


@pytest.fixture(scope="module")
def real_conf():
    """Hydra-composed full conf from apps/ncore_3dgut_mcmc.

    Module-scoped so Tracer CUDA compile happens at most once per test session.
    """
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="apps/ncore_3dgut_mcmc")


def _v1_shape_dict(N: int, conf) -> dict:
    """Build a v1-shape flat ckpt sub-dict for one layer.

    Matches `MixtureOfGaussians.get_model_parameters()` output (minus optimizer
    state) so `init_from_checkpoint(setup_optimizer=False)` accepts it.
    """
    sh_deg = conf.model.progressive_training.max_n_features
    specular_dim = sh_degree_to_specular_dim(sh_deg)
    return {
        "positions":             torch.nn.Parameter(torch.randn(N, 3)),
        "rotation":              torch.nn.Parameter(torch.randn(N, 4)),
        "scale":                 torch.nn.Parameter(torch.randn(N, 3)),
        "density":               torch.nn.Parameter(torch.randn(N, 1)),
        "features_albedo":       torch.nn.Parameter(torch.randn(N, 3)),
        "features_specular":     torch.nn.Parameter(torch.randn(N, specular_dim)),
        "background":            {},
        "n_active_features":     0,
        "max_n_features":        sh_deg,
        "scene_extent":          10.0,
        # progressive_training=True when n_active < max_n_features, so these are required
        "feature_dim_increase_interval": conf.model.progressive_training.increase_frequency,
        "feature_dim_increase_step":     conf.model.progressive_training.increase_step,
    }


# ----------------------------------------------------------------------- tests
def test_layered_gaussians_init_with_single_background_layer(real_conf):
    """空构造 → 仅 background 层 → 行为等价 v1 单 model"""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1_000_000)]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    assert "background" in model.layers
    assert model.layers["background"].num_gaussians == 0
    assert len(model.layers) == 1


def test_layered_gaussians_init_with_multiple_layers(real_conf):
    """多层 spec → 每层各一个 MixtureOfGaussians"""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="road",       layer_id=1, max_n_particles=200_000),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    assert set(model.layers.keys()) == {"background", "road"}


def test_layered_gaussians_get_model_parameters_nested(real_conf):
    """get_model_parameters 返回 NRE schema: gaussians_nodes.<name>"""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1_000_000)]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    model.setup_optimizer_for_test()

    params = model.get_model_parameters()
    assert "gaussians_nodes" in params
    assert "background" in params["gaussians_nodes"]
    assert "positions" in params["gaussians_nodes"]["background"]
    assert "scene_extent" in params


def test_load_v1_checkpoint_routes_to_background(real_conf):
    """v1 ckpt (无 'gaussians_nodes' 嵌套) → 全部塞 background layer"""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1_000_000)]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)

    v1_ckpt = _v1_shape_dict(N=100, conf=real_conf)
    model.init_from_checkpoint(v1_ckpt, setup_optimizer=False)

    assert model.layers["background"].positions.shape == (100, 3)


def test_load_v2_checkpoint_dispatches_per_layer(real_conf):
    """v2 ckpt (NRE schema: 'model.gaussians_nodes.<name>') → 分发到对应 layer"""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=1_000_000),
        LayerSpec(name="road",       layer_id=1, max_n_particles=200_000),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    v2_ckpt = {
        "global_step": 30000,
        "model": {
            "gaussians_nodes": {
                "background": _v1_shape_dict(N=600, conf=real_conf),
                "road":       _v1_shape_dict(N=200, conf=real_conf),
            },
        },
    }
    model.init_from_checkpoint(v2_ckpt, setup_optimizer=False)
    assert model.layers["background"].positions.shape == (600, 3)
    assert model.layers["road"].positions.shape == (200, 3)


def test_load_v2_checkpoint_warns_on_missing_layer(real_conf):
    """ckpt 缺某层 → warn + skip，不 raise"""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=1_000_000),
        LayerSpec(name="road",       layer_id=1, max_n_particles=200_000),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    v2_ckpt = {
        "global_step": 30000,
        "model": {
            "gaussians_nodes": {
                "background": _v1_shape_dict(N=600, conf=real_conf),
                # road missing
            },
        },
    }
    model.init_from_checkpoint(v2_ckpt, setup_optimizer=False)
    assert model.layers["background"].positions.shape == (600, 3)
    assert model.layers["road"].positions.shape == (0, 3)


# ----------------------------------------------------------- T1.3 / T1.4 additions
def test_v1_ckpt_resume_without_background_layer_raises(real_conf):
    """T1.3: layers.enabled=['road'] + v1 ckpt → 明确报错指向 layers.enabled"""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="road", layer_id=1, max_n_particles=200_000)]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)

    v1_ckpt = _v1_shape_dict(N=100, conf=real_conf)
    with pytest.raises(ValueError, match="layers.enabled"):
        model.init_from_checkpoint(v1_ckpt, setup_optimizer=False)


def test_v1_ckpt_resume_with_background_layer_works(real_conf):
    """T1.3: layers.enabled=['background','road'] + v1 ckpt → 全部塞 background"""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="road",       layer_id=1, max_n_particles=200_000),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)

    v1_ckpt = _v1_shape_dict(N=100, conf=real_conf)
    model.init_from_checkpoint(v1_ckpt, setup_optimizer=False)
    assert model.layers["background"].positions.shape == (100, 3)
    assert model.layers["road"].positions.shape == (0, 3)


def test_multi_layer_ckpt_roundtrip(real_conf):
    """T1.4: 2 层 LayeredGaussians: save → load 后各层 tensor 字节级一致"""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="road", layer_id=1, max_n_particles=200_000),
    ]
    model_a = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)

    src_ckpt = {
        "global_step": 30000,
        "model": {
            "gaussians_nodes": {
                "background": _v1_shape_dict(N=300, conf=real_conf),
                "road":       _v1_shape_dict(N=150, conf=real_conf),
            },
        },
    }
    model_a.init_from_checkpoint(src_ckpt, setup_optimizer=False)

    saved = model_a.get_model_parameters()
    assert "gaussians_nodes" in saved
    assert set(saved["gaussians_nodes"].keys()) == {"background", "road"}

    model_b = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    model_b.init_from_checkpoint(
        {"gaussians_nodes": saved["gaussians_nodes"]},
        setup_optimizer=False,
    )

    for layer_name in ["background", "road"]:
        for attr in [
            "positions", "rotation", "scale", "density",
            "features_albedo", "features_specular",
        ]:
            t_a = getattr(model_a.layers[layer_name], attr)
            t_b = getattr(model_b.layers[layer_name], attr)
            assert torch.equal(t_a.data, t_b.data), (
                f"Roundtrip mismatch at {layer_name}.{attr}: "
                f"shapes {t_a.shape} vs {t_b.shape}"
            )
