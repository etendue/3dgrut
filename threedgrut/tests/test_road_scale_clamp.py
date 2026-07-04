# SPDX-License-Identifier: Apache-2.0
"""V3-R1.2 unit tests for per-layer scale clamp + anisotropy via LayerSpec
and LayeredMCMCStrategy._maybe_clamp_road_scales integration."""

from __future__ import annotations

import math
import os

import pytest
import torch
from hydra import compose, initialize_config_dir

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.layers.registry import STANDARD_LAYERS


def test_layerspec_default_scale_clamps_are_none():
    spec = LayerSpec(name="x", layer_id=0, max_n_particles=100)
    assert spec.scale_xy_max is None
    assert spec.scale_z_max is None
    assert spec.anisotropy_ratio_max is None


def test_road_layer_clamps_set():
    """V3-R1.2 acceptance: road layer caps scale (XY <= 0.3m, Z <= 0.05m)
    and anisotropy ratio (max/min eigenvalue <= 8x)."""
    s = STANDARD_LAYERS["road"]
    assert s.scale_xy_max == 0.3
    assert s.scale_z_max == 0.05
    assert s.anisotropy_ratio_max == 8.0


def test_background_layer_clamps_not_set():
    s = STANDARD_LAYERS["background"]
    assert s.scale_xy_max is None
    assert s.scale_z_max is None
    assert s.anisotropy_ratio_max is None


# ---------------------------------------------------------------------------
# Integration tests: LayeredMCMCStrategy._maybe_clamp_road_scales
# ---------------------------------------------------------------------------

_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs"))


@pytest.fixture(scope="module")
def real_conf():
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="apps/ncore_3dgut_mcmc")


def test_maybe_clamp_road_scales_clamps_oversized(real_conf):
    """V3-R1.2 integration: _maybe_clamp_road_scales enforces XY<=0.3, Z<=0.05,
    ratio<=8 on the road layer after being called directly."""
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
            scale_xy_max=0.3,
            scale_z_max=0.05,
            anisotropy_ratio_max=8.0,
        ),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    strat = LayeredMCMCStrategy(real_conf, model, specs)

    road_layer = model.layers["road"]
    N = 20
    # Deliberately oversized: exp(0) = 1.0m >> 0.3m XY, 1.0m >> 0.05m Z.
    with torch.no_grad():
        road_layer.scale.data = torch.zeros(N, 3)

    strat._maybe_clamp_road_scales()

    out_exp = torch.exp(road_layer.scale.detach())
    assert torch.all(out_exp[:, 0] <= 0.3 + 1e-6), "XY-X not clamped"
    assert torch.all(out_exp[:, 1] <= 0.3 + 1e-6), "XY-Y not clamped"
    assert torch.all(out_exp[:, 2] <= 0.05 + 1e-6), "Z not clamped"
    ratio = out_exp.max(dim=-1).values / out_exp.min(dim=-1).values
    assert torch.all(ratio <= 8.0 + 1e-5), "Anisotropy ratio not clamped"


def test_maybe_clamp_road_scales_needle_anisotropy(real_conf):
    """V3-R1.2 integration: a needle-shaped particle (Z very thin, XY normal)
    gets its anisotropy corrected by raising the Z floor."""
    from threedgrut.layers.layered_model import LayeredGaussians
    from threedgrut.strategy.layered_mcmc import LayeredMCMCStrategy

    specs = [
        LayerSpec(
            name="road",
            layer_id=1,
            max_n_particles=200_000,
            scale_xy_max=0.3,
            scale_z_max=0.05,
            anisotropy_ratio_max=8.0,
        ),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    strat = LayeredMCMCStrategy(real_conf, model, specs)

    road_layer = model.layers["road"]
    N = 5
    # needle: XY=0.1m (log=-2.30), Z=0.001m (log=-6.91) → ratio=100x >> 8x
    with torch.no_grad():
        road_layer.scale.data = (
            torch.tensor([math.log(0.1), math.log(0.1), math.log(0.001)]).unsqueeze(0).expand(N, 3).clone()
        )

    strat._maybe_clamp_road_scales()

    out_exp = torch.exp(road_layer.scale.detach())
    ratio = out_exp.max(dim=-1).values / out_exp.min(dim=-1).values
    assert torch.all(ratio <= 8.0 + 1e-5), f"ratio={ratio} exceeds 8x"
    # XY should be untouched (0.1 <= 0.3)
    assert torch.allclose(out_exp[:, 0], torch.tensor(0.1), atol=1e-5)
    assert torch.allclose(out_exp[:, 1], torch.tensor(0.1), atol=1e-5)


def test_post_optimizer_step_clamps_road_scales(real_conf):
    """V3-R1.2 integration: road scale clamp fires through the real
    _post_optimizer_step public path, not just _maybe_clamp_road_scales directly.

    Uses step=1 so relocate/add/perturb cadence checks are all False
    (all three start at iteration 500 per configs/strategy/mcmc.yaml).
    The clamp call at the end of _post_optimizer_step runs unconditionally.
    """
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
            scale_xy_max=0.3,
            scale_z_max=0.05,
            anisotropy_ratio_max=8.0,
        ),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    strat = LayeredMCMCStrategy(real_conf, model, specs)

    road_layer = model.layers["road"]
    N = 20
    # Deliberately oversized: exp(0) = 1.0m >> 0.3m XY, 1.0m >> 0.05m Z.
    with torch.no_grad():
        road_layer.scale.data = torch.zeros(N, 3)

    # Drive the real public path using step=0.
    # check_step_condition requires step > start_iteration, so:
    #   - perturb (start=0): 0 > 0 = False → no CUDA call
    #   - relocate (start=500): 0 > 500 = False → skipped
    #   - add (start=500): 0 > 500 = False → skipped
    # The _maybe_clamp_road_scales call at the end runs unconditionally.
    strat._post_optimizer_step(step=0, scene_extent=10.0, train_dataset=None, batch=None, writer=None)

    out_exp = torch.exp(road_layer.scale.detach())
    assert torch.all(out_exp[:, 0] <= 0.3 + 1e-6), "XY-X not clamped via _post_optimizer_step"
    assert torch.all(out_exp[:, 1] <= 0.3 + 1e-6), "XY-Y not clamped via _post_optimizer_step"
    assert torch.all(out_exp[:, 2] <= 0.05 + 1e-6), "Z not clamped via _post_optimizer_step"


def test_maybe_clamp_road_scales_does_not_touch_background(real_conf):
    """V3-R1.2 integration: background layer (all clamp fields None) is
    byte-identical before and after _maybe_clamp_road_scales."""
    from threedgrut.layers.layered_model import LayeredGaussians
    from threedgrut.strategy.layered_mcmc import LayeredMCMCStrategy
    from threedgrut.tests.test_layered_gaussians import _v1_shape_dict

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(
            name="road",
            layer_id=1,
            max_n_particles=200_000,
            scale_xy_max=0.3,
            scale_z_max=0.05,
            anisotropy_ratio_max=8.0,
        ),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    model.init_from_checkpoint(
        {
            "gaussians_nodes": {
                "background": _v1_shape_dict(N=30, conf=real_conf),
                "road": _v1_shape_dict(N=15, conf=real_conf),
            }
        },
        setup_optimizer=False,
    )
    model.setup_optimizer_for_test()

    strat = LayeredMCMCStrategy(real_conf, model, specs)

    # Snapshot background scale before clamp
    bg_scale_before = model.layers["background"].scale.detach().clone()
    # Put road in oversized state so the clamp actually fires
    with torch.no_grad():
        model.layers["road"].scale.data = torch.zeros(15, 3)

    strat._maybe_clamp_road_scales()

    bg_scale_after = model.layers["background"].scale.detach()
    assert torch.equal(bg_scale_before, bg_scale_after), "background scale was mutated by _maybe_clamp_road_scales"
