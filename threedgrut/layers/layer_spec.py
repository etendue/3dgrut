# SPDX-License-Identifier: Apache-2.0
"""Minimal LayerSpec for T1.1.

Full spec (scale_prior, mask_field, is_particle_layer, lr_mult, ...) lands in T1.2.
T1.1 only needs name + layer_id + max_n_particles to bootstrap the container.

Layer naming aligned with NRE ckpt schema:
    model.gaussians_nodes.<name>
where <name> in {"background", "road", "dynamic_rigids", "dynamic_deformables"}.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerSpec:
    """Descriptive configuration of one Gaussian layer.

    Frozen because spec is configuration and must not mutate at runtime.
    """

    name: str
    layer_id: int
    max_n_particles: int
