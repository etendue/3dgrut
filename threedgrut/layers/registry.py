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

import dataclasses
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
        # T5.4: backend selects cubemap (default, nvdiffrast) vs mlp (fallback).
        # Trainer reads conf.trainer.sky_backend / sky_resolution and overrides
        # these defaults at LayeredGaussians construction time.
        extra={"backend": "cubemap", "resolution": 128},
    ),
}


def specs_from_config(conf: DictConfig) -> List[LayerSpec]:
    """Build runtime spec list from conf.layers.enabled.

    Args:
        conf: top-level Hydra conf with optional conf.layers.enabled list.
              Falls back to ["background"] if not present (v1 single-layer mode).
              T8/B3: also reads ``conf.layers.overrides.<name>.<field>`` to
              tweak per-layer registry defaults (e.g. ``max_n_particles``)
              from yaml without forking STANDARD_LAYERS.

    Returns:
        List of LayerSpec preserving the order given in conf.layers.enabled.

    Raises:
        ValueError: if conf.layers.enabled contains a name not in STANDARD_LAYERS.
    """
    layers_conf = conf.get("layers", {}) or {}
    enabled = list(layers_conf.get("enabled", ["background"]))
    overrides = layers_conf.get("overrides", {}) or {}
    valid_fields = {f.name for f in dataclasses.fields(LayerSpec)}
    specs: List[LayerSpec] = []
    for name in enabled:
        if name not in STANDARD_LAYERS:
            raise ValueError(
                f"Unknown layer name '{name}' in conf.layers.enabled. "
                f"Available: {sorted(STANDARD_LAYERS.keys())}"
            )
        spec = STANDARD_LAYERS[name]
        per_layer = overrides.get(name)
        if per_layer is not None:
            # Filter to known fields so a typo doesn't silently no-op.
            patch = {}
            for k, v in dict(per_layer).items():
                if k not in valid_fields:
                    raise ValueError(
                        f"Unknown LayerSpec field '{k}' in "
                        f"conf.layers.overrides.{name}; valid: {sorted(valid_fields)}"
                    )
                patch[k] = v
            if patch:
                spec = dataclasses.replace(spec, **patch)
        specs.append(spec)
    return specs
