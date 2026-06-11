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


# V3-L5/L8/L9: keys that ``layers.overrides.<name>.<field>`` may supply
# beyond the LayerSpec dataclass fields. These are passed through into
# ``LayerSpec.extra`` (merged with existing extras) so user yaml can flip
# NuRec-style dynamic_rigids tricks without modifying the dataclass schema.
# Add new keys here when a new ``extra``-routed knob is introduced.
_EXTRA_OVERRIDE_KEYS: frozenset[str] = frozenset({
    "symmetric_axis",          # V3-L5  ('X' | 'Y' | 'Z' | None)
    "optimize_track_albedo",   # V3-L8  (bool)
    "optimize_track_scale",    # V3-L9  (bool)
    "track_warmup_steps",      # V3-L5/L8/L9 shared warmup (int)
    "track_albedo_lr",         # V3-L8 optimizer LR
    "track_scale_lr",          # V3-L9 optimizer LR
    "n_fourier_albedo_terms",  # P1.3b 4D-SH time-varying albedo terms (int, default 1 = DC-only)
})


STANDARD_LAYERS: dict[str, LayerSpec] = {
    "background": LayerSpec(
        name="background", layer_id=0, max_n_particles=600_000,
        scale_prior=(0.1, 0.1, 0.1),
    ),
    "road": LayerSpec(
        name="road", layer_id=1, max_n_particles=200_000,
        # scale_lr_mult sat at 0.2 from T1.2 but was dead config (no consumer)
        # until the 2026-06-11 E0.5 recipe audit wired it up in
        # LayeredGaussians._apply_scale_lr_mult. Default stays identity so no
        # anchor recipe changes underfoot; E3 road-freeze experiments opt in
        # via ++layers.overrides.road.scale_lr_mult=0.02 (official NuRec road
        # scales lr 1e-4 over base 5e-3) or 0.2 (historical T1.2 intent).
        scale_prior=(0.1, 0.1, 0.001), scale_lr_mult=1.0,
        mask_field="road_mask",
        perturb_scale_mask=(1.0, 1.0, 0.0),  # T3.4 D1: Z lock during MCMC perturb
        scale_xy_max=0.3, scale_z_max=0.05,   # V3-R1.2
        anisotropy_ratio_max=8.0,             # V3-R1.2
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
            # V3-L5/L8/L9: keys in _EXTRA_OVERRIDE_KEYS are routed into the
            # ``extra`` dict (merged on top of registry defaults) so user
            # yaml can flip NuRec-style knobs without bloating the
            # LayerSpec dataclass schema.
            patch: dict = {}
            extra_patch: dict = {}
            for k, v in dict(per_layer).items():
                if k in valid_fields:
                    patch[k] = v
                elif k in _EXTRA_OVERRIDE_KEYS:
                    extra_patch[k] = v
                else:
                    raise ValueError(
                        f"Unknown LayerSpec field '{k}' in "
                        f"conf.layers.overrides.{name}; valid: "
                        f"{sorted(valid_fields)}; extras: "
                        f"{sorted(_EXTRA_OVERRIDE_KEYS)}"
                    )
            if extra_patch:
                # Merge over the registry default ``extra`` (e.g. sky_envmap's
                # {"backend": "cubemap", "resolution": 128}) so we never drop
                # a baked-in default by listing only one new key.
                merged_extra = dict(spec.extra or {})
                merged_extra.update(extra_patch)
                patch["extra"] = merged_extra
            if patch:
                spec = dataclasses.replace(spec, **patch)
        specs.append(spec)
    return specs
