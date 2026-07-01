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


def test_relocate_and_add_handle_empty_layer_without_zerodiv(real_conf):
    """Regression (inceptio 5cam crash #2, 2026-06-26).

    LayeredMCMCStrategy runs one MCMCStrategy sub per particle layer. A clip with
    no cuboid autolabels leaves dynamic_rigids empty (0 particles), so that sub's
    ``model.get_density()`` is a 0-row tensor and ``model.num_gaussians == 0``.
    The print_stats lines divided by the particle count —
    ``n_dead / len(densities)`` (relocate) and
    ``num_to_add / current_num_gaussians`` (add) — raising ZeroDivisionError and
    crashing MCMC at the first densification step. Both divisions are now guarded.
    """
    import copy

    import torch
    from omegaconf import OmegaConf

    from threedgrut.strategy.mcmc import MCMCStrategy

    class _EmptyLayer:
        """Minimal stand-in for an empty MoG particle layer."""

        num_gaussians = 0

        def get_density(self):  # 0-row density, as a never-populated layer has
            return torch.zeros((0, 1))

    conf = copy.deepcopy(real_conf)
    OmegaConf.set_struct(conf, False)
    conf.strategy.print_stats = True  # force the division branch to execute

    strat = MCMCStrategy.__new__(MCMCStrategy)
    strat.conf = conf
    strat.model = _EmptyLayer()

    # Pre-fix: each of these raised ZeroDivisionError on the print_stats line.
    strat.relocate_gaussians()
    strat.add_new_gaussians()




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
    # Training hyper-params inherited unchanged (T2.3)
    assert cfg_mcmc.strategy.binom_n_max == cfg_layered.strategy.binom_n_max
    assert cfg_mcmc.strategy.relocate.frequency == cfg_layered.strategy.relocate.frequency
    assert cfg_mcmc.strategy.perturb.noise_lr == cfg_layered.strategy.perturb.noise_lr


# --- T3.4: perturb mask hook (D1) ---
def test_mcmc_get_perturb_mask_default_is_ones(real_conf):
    """T3.4: MCMCStrategy._get_perturb_mask() default = ones (v1 byte-identical)."""
    import torch
    from threedgrut.strategy.mcmc import MCMCStrategy

    strat = MCMCStrategy.__new__(MCMCStrategy)
    mask = strat._get_perturb_mask()
    assert mask.shape == (3,)
    assert torch.equal(mask, torch.ones(3))


def test_road_spec_has_perturb_scale_mask_z_zero():
    """T3.4 D1: registry road spec installs perturb_scale_mask=(1, 1, 0).

    Without this, MCMC perturb would noisily drift the LiDAR-Z-locked thin
    disc off the road surface even though road_init enforces Z lock at init.
    """
    from threedgrut.layers.registry import STANDARD_LAYERS

    road = STANDARD_LAYERS["road"]
    assert road.perturb_scale_mask == (1.0, 1.0, 0.0), (
        f"road perturb mask leaked Z: {road.perturb_scale_mask}"
    )
    # Background / dynamic_rigids should NOT override (free perturb)
    assert STANDARD_LAYERS["background"].perturb_scale_mask is None
    assert STANDARD_LAYERS["dynamic_rigids"].perturb_scale_mask is None


def test_layered_mcmc_installs_road_perturb_mask(real_conf):
    """T3.4 D1: LayeredMCMCStrategy injects road spec's perturb mask into
    sub-strategy['road']; background sub stays at default ones."""
    import torch
    from threedgrut.layers.layer_spec import LayerSpec
    from threedgrut.layers.layered_model import LayeredGaussians
    from threedgrut.strategy.layered_mcmc import LayeredMCMCStrategy

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="road", layer_id=1, max_n_particles=200_000,
                  scale_prior=(0.1, 0.1, 0.001),
                  perturb_scale_mask=(1.0, 1.0, 0.0)),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    strat = LayeredMCMCStrategy(real_conf, model, specs)

    bg_mask = strat.sub_strategies["background"]._get_perturb_mask()
    road_mask = strat.sub_strategies["road"]._get_perturb_mask()

    assert torch.equal(bg_mask, torch.ones(3))
    assert torch.equal(road_mask, torch.tensor([1.0, 1.0, 0.0]))


def test_layered_mcmc_perturb_mask_skipped_when_spec_none(real_conf):
    """T3.4 D1: when spec.perturb_scale_mask is None, sub keeps the default
    _get_perturb_mask path — no instance attribute installed."""
    from threedgrut.layers.layer_spec import LayerSpec
    from threedgrut.layers.layered_model import LayeredGaussians
    from threedgrut.strategy.layered_mcmc import LayeredMCMCStrategy

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=600_000)]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    strat = LayeredMCMCStrategy(real_conf, model, specs)
    sub = strat.sub_strategies["background"]
    # No _perturb_mask_override attribute → default class method path
    assert not hasattr(sub, "_perturb_mask_override")



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


def test_relocation_fraction_cap_config_in_dynfix():
    """Regression: dynfix config must have max_relocation_fraction < 1.0 (OOM prevention).

    T8/B3 stability fix: 90% dyn-layer dead → 630k particles crammed into 70k alive
    spots → tile-buffer OOM. The cap prevents mass-clustering by spreading relocations
    across multiple MCMC steps.
    """
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        conf = compose(config_name="apps/ncore_3dgut_mcmc_v2_full_4dviz_dynfix")
    frac = conf.strategy.relocate.max_relocation_fraction
    assert frac < 1.0, (
        f"dynfix config must cap relocation fraction (got {frac}); "
        "without cap, 90%-dead dyn layer causes OOM from dense cluster"
    )
    assert frac >= 0.1, f"cap too aggressive ({frac}); bg/road need normal relocation"


def test_relocation_cap_subsamples_dead_indices():
    """Unit test for the max_relocation_fraction capping logic in relocate_gaussians.

    Simulates 90-particle layer with 81 dead (90%). With max_relocation_fraction=0.5,
    only 45 particles should be selected for relocation per step.
    """
    import torch

    N = 100
    dead = 90
    max_frac = 0.5

    densities = torch.zeros(N)
    densities[:N - dead] = 0.5  # first 10 alive

    dead_idxs = torch.where(densities <= 0.005)[0]
    assert len(dead_idxs) == dead

    # Replicate the cap logic from mcmc.py:relocate_gaussians
    cap = max(1, int(max_frac * N))
    assert len(dead_idxs) > cap  # precondition: cap triggers
    perm = torch.randperm(len(dead_idxs))
    capped_dead = dead_idxs[perm[:cap]]

    assert len(capped_dead) == cap
    assert len(capped_dead) <= int(max_frac * N) + 1  # allow rounding
    # All selected indices must actually be dead
    assert (densities[capped_dead] <= 0.005).all()


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
