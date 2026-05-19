# SPDX-License-Identifier: Apache-2.0
"""Standard layer registry for v2 LayeredGaussians.

STANDARD_LAYERS captures the 5 v2 layers' default specs. Trainer reads
conf.layers.enabled (list of names) and uses specs_from_config() to filter
the registry into the runtime spec list passed to LayeredGaussians.

To add a new layer:
  1. Append a LayerSpec to STANDARD_LAYERS with a fresh layer_id.
  2. Add its name to apps yaml `layers.enabled`.
"""
from __future__ import annotations

from typing import List

from omegaconf import DictConfig

from threedgrut.layers.layer_spec import LayerSpec


STANDARD_LAYERS: dict[str, LayerSpec] = {
    "background": LayerSpec(
        name="background", layer_id=0, max_n_particles=600_000,
        scale_prior=(0.1, 0.1, 0.1),
    ),
    "road": LayerSpec(
        name="road", layer_id=1, max_n_particles=200_000,
        scale_prior=(0.1, 0.1, 0.001), scale_lr_mult=0.2,
        mask_field="road_mask",
        perturb_scale_mask=(1.0, 1.0, 0.0),  # T3.4 D1: Z lock during MCMC perturb
    ),
    "dynamic_rigids": LayerSpec(
        name="dynamic_rigids", layer_id=2, max_n_particles=200_000,
        scale_prior=(0.05, 0.05, 0.05),
        mask_field="dynamic_mask",
    ),
    "dynamic_deformables": LayerSpec(
        name="dynamic_deformables", layer_id=3, max_n_particles=0,
        scale_prior=(0.0, 0.0, 0.0),
        is_particle_layer=False,  # v2 stub: no particles allocated
    ),
    "sky_envmap": LayerSpec(
        name="sky_envmap", layer_id=-1, max_n_particles=0,
        scale_prior=(0.0, 0.0, 0.0),
        mask_field="sky_mask", is_particle_layer=False,
    ),
}


def specs_from_config(conf: DictConfig) -> List[LayerSpec]:
    """Build runtime spec list from conf.layers.enabled.

    Args:
        conf: top-level Hydra conf with optional conf.layers.enabled list.
              Falls back to ["background"] if not present (v1 single-layer mode).

    Returns:
        List of LayerSpec preserving the order given in conf.layers.enabled.

    Raises:
        ValueError: if conf.layers.enabled contains a name not in STANDARD_LAYERS.
    """
    enabled = list(conf.get("layers", {}).get("enabled", ["background"]))
    specs: List[LayerSpec] = []
    for name in enabled:
        if name not in STANDARD_LAYERS:
            raise ValueError(
                f"Unknown layer name '{name}' in conf.layers.enabled. "
                f"Available: {sorted(STANDARD_LAYERS.keys())}"
            )
        specs.append(STANDARD_LAYERS[name])
    return specs
