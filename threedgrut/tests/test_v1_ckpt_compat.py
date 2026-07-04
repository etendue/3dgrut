# SPDX-License-Identifier: Apache-2.0
"""Bit-level v1 <-> layered checkpoint compatibility test.

Proves that loading the same v1-shape ckpt via:
    (a) MixtureOfGaussians.init_from_checkpoint(ckpt)
    (b) LayeredGaussians.init_from_checkpoint(ckpt)  (routes to bg layer)
produces byte-identical Parameter tensors on the 6 per-particle fields.

This is the T1.1 safety net: it guarantees v1 ckpts resumed via the layered
path behave exactly as v1, which is the basis of the A800 smoke equivalence
claim (24.12 +/- 0.05 dB).
"""

import torch

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.layers.layered_model import LayeredGaussians
from threedgrut.model.model import MixtureOfGaussians

from .test_layered_gaussians import _v1_shape_dict, real_conf  # noqa: F401


def test_v1_ckpt_loaded_into_background_layer_matches_single_model(real_conf):
    """Same v1 ckpt -> MoG vs LayeredGaussians[background]: 6 Parameters byte-equal."""
    v1_ckpt = _v1_shape_dict(N=500, conf=real_conf)

    # single-model path
    single = MixtureOfGaussians(real_conf, scene_extent=10.0)
    single.init_from_checkpoint(v1_ckpt, setup_optimizer=False)

    # layered path (single bg spec)
    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1_000_000)]
    layered = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    layered.init_from_checkpoint(v1_ckpt, setup_optimizer=False)

    for field in [
        "positions",
        "rotation",
        "scale",
        "density",
        "features_albedo",
        "features_specular",
    ]:
        assert torch.equal(
            getattr(single, field).detach(),
            getattr(layered.layers["background"], field).detach(),
        ), f"Mismatch in {field}"


def test_single_bg_bridge_exposes_layer_attributes(real_conf):
    """LayeredGaussians (single bg) should forward .positions / .num_gaussians to bg layer.

    Verifies the __getattr__ bridge that lets the existing Trainer +
    MCMCStrategy code paths see LayeredGaussians as a drop-in MoG.
    """
    v1_ckpt = _v1_shape_dict(N=42, conf=real_conf)
    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1_000_000)]
    layered = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    layered.init_from_checkpoint(v1_ckpt, setup_optimizer=False)

    # Read-path bridge
    assert layered.num_gaussians == 42
    assert layered.positions.shape == (42, 3)
    assert torch.equal(
        layered.positions.detach(),
        layered.layers["background"].positions.detach(),
    )


def test_single_bg_bridge_setattr_writes_to_bg_layer(real_conf):
    """Strategy-style setattr(model, 'positions', new_param) must update bg layer's Parameter.

    MCMCStrategy._update_param_with_optimizer does this after cat()-ing new
    particles. If the write landed on LayeredGaussians directly instead of bg,
    the renderer (which reads bg.positions) would see stale tensors.
    """
    v1_ckpt = _v1_shape_dict(N=10, conf=real_conf)
    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1_000_000)]
    layered = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    layered.init_from_checkpoint(v1_ckpt, setup_optimizer=False)

    new_positions = torch.nn.Parameter(torch.zeros(20, 3))
    setattr(layered, "positions", new_positions)

    # bg layer's positions must be the new tensor (not stale).
    assert layered.layers["background"].positions.shape == (20, 3)
    assert torch.equal(
        layered.layers["background"].positions.detach(),
        torch.zeros(20, 3),
    )
    # Reading via bridge returns the same.
    assert torch.equal(layered.positions.detach(), torch.zeros(20, 3))
