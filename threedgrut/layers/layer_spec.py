# SPDX-License-Identifier: Apache-2.0
"""LayerSpec: frozen descriptor of one Gaussian layer.

Layer naming aligned with NRE ckpt schema:
    model.gaussians_nodes.<name>
where <name> in {"background", "road", "dynamic_rigids", "dynamic_deformables"}.

T1.2 expanded the 3-field T1.1 minimal spec to 8 fields covering scale prior,
mask gating, lr_mult and particle/non-particle distinction so that Stage 2
(LayeredMCMC) and Stage 3 (Road) can read all knobs from a single dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LayerSpec:
    """Descriptive configuration of one Gaussian layer.

    Frozen because spec is configuration and must not mutate at runtime.

    Fields:
        name: NRE-aligned layer name, e.g. "background"/"road"/"dynamic_rigids".
        layer_id: stable integer id used for per-particle layer_id buffer.
        max_n_particles: per-layer particle cap consumed by LayeredMCMCStrategy.
        scale_prior: log-space initial scale (sx, sy, sz). Road layer uses
            (0.1, 0.1, 0.001) to enforce thin-disc Z-lock; default isotropic.
        scale_lr_mult: optimizer LR multiplier on the scale parameter group.
            Road layer uses 0.2 so the Z-lock survives optimization.
        mask_field: which mask in image_infos this layer's loss is gated by
            (e.g. "road_mask", "dynamic_mask"). None = no mask gating.
        is_particle_layer: False for sky_envmap / dynamic_deformables (v2 stub)
            -- skipped by LayeredMCMCStrategy and fused_view.
        density_init: log-space initial density for new particles.
    """

    name: str
    layer_id: int
    max_n_particles: int
    scale_prior: tuple[float, float, float] = (0.1, 0.1, 0.1)
    scale_lr_mult: float = 1.0
    mask_field: str | None = None
    is_particle_layer: bool = True
    density_init: float = 0.1
    # T3.4 D1: per-axis multiplier on MCMC positional perturb noise. Road
    # uses (1, 1, 0) so the LiDAR-Z-locked thin disc cannot drift in Z under
    # MCMC perturb. None = no override (LayeredMCMCStrategy leaves the sub's
    # default _get_perturb_mask=ones in place).
    perturb_scale_mask: tuple[float, float, float] | None = None
    # T5.4: backend-specific knobs for non-particle layers. Currently used by
    # the sky_envmap layer to carry {"backend": "cubemap"|"mlp", "resolution":
    # int}. ``compare=False`` keeps LayerSpec hashable even though dict isn't,
    # because the auto-generated __eq__/__hash__ skip this field.
    extra: dict = field(default_factory=dict, compare=False)
