# SPDX-License-Identifier: Apache-2.0
"""Contract tests for MCMCStrategy._get_add_cap() (T2.1), LayeredMCMCStrategy (T2.2),
layered_mcmc.yaml inheritance (T2.3), and T2.4 invariant tests.

sys.modules stubs and the MCMCStrategy.__init__ no-CUDA patch are installed by
conftest.py before any test in this directory is collected. See conftest.py for
the full rationale (I-1 fix: prevents collection-order dependency).
"""

import os

import pytest
from hydra import compose, initialize_config_dir


# ----------------------------------------------------------------------- conf
_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs")
)


@pytest.fixture(scope="module")
def real_conf():
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="apps/ncore_3dgut_mcmc")


# --- T2.1: _get_add_cap hook ---
def test_mcmc_get_add_cap_defaults_to_conf(real_conf):
    """T2.1: MCMCStrategy._get_add_cap() returns conf.strategy.add.max_n_gaussians by default.

    Uses __new__ to bypass __init__ (which JIT-compiles CUDA, unavailable on Mac).
    """
    from threedgrut.strategy.mcmc import MCMCStrategy

    strat = MCMCStrategy.__new__(MCMCStrategy)
    strat.conf = real_conf
    assert strat._get_add_cap() == real_conf.strategy.add.max_n_gaussians




# --- T2.2: LayeredMCMCStrategy sub-strategy array ---
def test_layered_mcmc_holds_sub_strategy_per_particle_layer(real_conf):
    """T2.2: each is_particle_layer=True layer gets one MCMCStrategy; non-particle layers skipped."""
    from threedgrut.layers.layer_spec import LayerSpec
    from threedgrut.layers.layered_model import LayeredGaussians
    from threedgrut.strategy.layered_mcmc import LayeredMCMCStrategy

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(
            name="road",
            layer_id=1,
            max_n_particles=200_000,
            scale_prior=(0.1, 0.1, 0.001),
            scale_lr_mult=0.2,
            mask_field="road_mask",
        ),
        LayerSpec(name="sky_envmap", layer_id=-1, max_n_particles=0, is_particle_layer=False),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    strat = LayeredMCMCStrategy(real_conf, model, specs)
    assert set(strat.sub_strategies.keys()) == {"background", "road"}
    assert "sky_envmap" not in strat.sub_strategies


def test_layered_mcmc_sub_uses_per_layer_cap(real_conf):
    """T2.2: each sub-strategy's _get_add_cap() returns spec.max_n_particles."""
    from threedgrut.layers.layer_spec import LayerSpec
    from threedgrut.layers.layered_model import LayeredGaussians
    from threedgrut.strategy.layered_mcmc import LayeredMCMCStrategy

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="road", layer_id=1, max_n_particles=200_000),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    strat = LayeredMCMCStrategy(real_conf, model, specs)
    assert strat.sub_strategies["background"]._get_add_cap() == 600_000
    assert strat.sub_strategies["road"]._get_add_cap() == 200_000


def test_layered_mcmc_single_bg_uses_one_sub_strategy(real_conf):
    """T2.2: single-bg mode produces exactly one sub-strategy that is an MCMCStrategy instance.

    Structural invariant: sub.model is model.layers['background'] (the layer MoG, not the
    LayeredGaussians wrapper). This test verifies structural identity only — it does NOT
    guarantee byte-identical training output to a bare MCMCStrategy on the same model.
    """
    from threedgrut.layers.layer_spec import LayerSpec
    from threedgrut.layers.layered_model import LayeredGaussians
    from threedgrut.strategy.layered_mcmc import LayeredMCMCStrategy
    from threedgrut.strategy.mcmc import MCMCStrategy

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=600_000)]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    strat = LayeredMCMCStrategy(real_conf, model, specs)
    assert len(strat.sub_strategies) == 1
    assert isinstance(strat.sub_strategies["background"], MCMCStrategy)
    # Key invariant: sub.model references the layer MoG, not the LayeredGaussians wrapper.
    assert strat.sub_strategies["background"].model is model.layers["background"]



# --- T2.3: yaml inheritance ---
def test_layered_mcmc_yaml_inherits_mcmc_defaults():
    """T2.3: layered_mcmc.yaml inherits mcmc.yaml; only `method` differs."""
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg_mcmc = compose(
            config_name="apps/ncore_3dgut_mcmc",
            overrides=["strategy=mcmc"],
        )
        cfg_layered = compose(
            config_name="apps/ncore_3dgut_mcmc",
            overrides=["strategy=layered_mcmc", "use_layered_model=true"],
        )

    assert cfg_mcmc.strategy.method == "MCMCStrategy"
    assert cfg_layered.strategy.method == "LayeredMCMCStrategy"
    # Training hyper-params inherited unchanged
    assert cfg_mcmc.strategy.binom_n_max == cfg_layered.strategy.binom_n_max
    assert cfg_mcmc.strategy.relocate.frequency == cfg_layered.strategy.relocate.frequency
    assert cfg_mcmc.strategy.perturb.noise_lr == cfg_layered.strategy.perturb.noise_lr



# --- T2.4: invariants ---
def test_no_cross_layer_migration_structural(real_conf):
    """T2.4: structural identity check — sub.model is pinned to its layer's MoG.

    This is a *structural identity* verification only: it asserts that each
    sub-strategy's .model reference points to the correct per-layer
    MixtureOfGaussians, which is the architectural guarantee that MCMC
    operations cannot migrate particles across layers.

    NOTE: this test does NOT call post_optimizer_step() and therefore does
    NOT dynamically verify relocate/add/perturb behaviour. That dynamic
    verification (running actual MCMC steps and checking no particle crosses
    a layer boundary) requires a CUDA environment and is deferred to the A800
    controller batch at the end of Stage 2.
    """
    from threedgrut.layers.layered_model import LayeredGaussians
    from threedgrut.layers.layer_spec import LayerSpec
    from threedgrut.strategy.layered_mcmc import LayeredMCMCStrategy
    from threedgrut.tests.test_layered_gaussians import _v1_shape_dict

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="road",       layer_id=1, max_n_particles=200_000),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    model.init_from_checkpoint(
        {"gaussians_nodes": {
            "background": _v1_shape_dict(N=100, conf=real_conf),
            "road":       _v1_shape_dict(N=50,  conf=real_conf),
        }},
        setup_optimizer=False,
    )
    model.setup_optimizer_for_test()

    strat = LayeredMCMCStrategy(real_conf, model, specs)
    assert strat.sub_strategies["background"].model.num_gaussians == 100
    assert strat.sub_strategies["road"].model.num_gaussians == 50
    assert strat.sub_strategies["background"].model is model.layers["background"]
    assert strat.sub_strategies["road"].model is model.layers["road"]


def test_init_densification_buffer_dispatches_to_all_subs(real_conf, monkeypatch):
    """T2.4: init_densification_buffer must broadcast to every sub-strategy."""
    from threedgrut.layers.layered_model import LayeredGaussians
    from threedgrut.layers.layer_spec import LayerSpec
    from threedgrut.strategy.layered_mcmc import LayeredMCMCStrategy

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="road",       layer_id=1, max_n_particles=200_000),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    strat = LayeredMCMCStrategy(real_conf, model, specs)

    call_log: list[str] = []
    for name, sub in strat.sub_strategies.items():
        monkeypatch.setattr(
            sub, "init_densification_buffer",
            lambda ckpt, n=name: call_log.append(n)
        )
    strat.init_densification_buffer(checkpoint=None)
    assert sorted(call_log) == ["background", "road"]


def test_make_sub_conf_does_not_mutate_parent(real_conf):
    """T2.2 carry-over (M-2): _make_sub_conf returns an independent conf;
    modifying the sub must not leak back into the parent."""
    from threedgrut.layers.layer_spec import LayerSpec
    from threedgrut.strategy.layered_mcmc import LayeredMCMCStrategy

    spec = LayerSpec(name="background", layer_id=0, max_n_particles=123_456)
    original_cap = real_conf.strategy.add.max_n_gaussians
    sub = LayeredMCMCStrategy._make_sub_conf(real_conf, spec)
    assert sub.strategy.add.max_n_gaussians == 123_456
    assert real_conf.strategy.add.max_n_gaussians == original_cap
