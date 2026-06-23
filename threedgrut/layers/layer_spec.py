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
        scale_lr_mult: per-layer multiplier on the 'scale' param-group lr,
            applied once after the layer optimizer is built
            (LayeredGaussians._apply_scale_lr_mult; wired 2026-06-11 -- dead
            config from T1.2 until then). Incompatible with a conf lr
            scheduler on the scale group (fails loud at setup). Registry
            defaults are identity everywhere; E3 road-freeze experiments
            enable it via ++layers.overrides.road.scale_lr_mult=....
        mask_field: which mask in image_infos this layer's loss is gated by
            (e.g. "road_mask", "dynamic_mask"). None = no mask gating.
        is_particle_layer: False for sky_envmap / dynamic_deformables (v2 stub)
            -- skipped by LayeredMCMCStrategy and fused_view.
        density_init: log-space initial density for new particles.
        sh_degree: RESERVED/unused. Per-layer SH degree cap (incompatible with fused renderer; see field comment).
        scale_xy_max: linear-metre upper bound on XY scale, compared against exp(scale_log); None disables.
        scale_z_max: linear-metre upper bound on Z scale, compared against exp(scale_log); None disables.
        anisotropy_ratio_max: cap on max/min scale eigenvalue ratio; None disables.
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
    # RESERVED / currently unused. Per-layer SH-degree reduction by shrinking
    # features_specular is incompatible with the fused-view renderer (all
    # particle layers must share one specular width; the renderer uses the
    # reference layer's max_n_features). A future freeze-based approach (keep
    # width 45, zero+freeze road's order>=2 coefficients) would consume this
    # field. Leaving the field in place so that redesign is a small change.
    sh_degree: int | None = None
    # V3-R1.2: per-layer scale upper bounds in LINEAR units (physical metres).
    # A later clamp compares these against exp(scale_log) -- i.e. the physical
    # scale, NOT the raw log-space parameter -- after every MCMC
    # post_optimizer_step. None disables. Road uses (0.3, 0.05): XY 0.3m ~=
    # lane-stripe width x 2; Z 0.05m keeps the disc thin on the LiDAR-Z surface.
    scale_xy_max: float | None = None
    scale_z_max: float | None = None
    # V3-R1.2: per-layer anisotropy ratio cap (max scale eigenvalue /
    # min scale eigenvalue). Prevents needle-shaped Gaussians that
    # overfit to a single training-camera direction. None disables.
    # Road layer uses 8.0 -- generous enough for elongated lane stripes
    # yet bounded enough to suppress hair-thin novel-view artifacts.
    anisotropy_ratio_max: float | None = None
    # E3 road-freeze (NuRec port 2026-06-22): ABSOLUTE per-layer lr overrides.
    # When not None, LayeredGaussians._apply_layer_lr_overrides sets that param
    # group's lr to this absolute value (after MoG's ×scene_extent on positions)
    # and drops any conf scheduler on it (else MoG.scheduler_step overwrites it
    # every step — the silent no-op _apply_scale_lr_mult warns about). NuRec road
    # recipe freezes geometry via positions 1e-6 / density·rotation·scale 1e-4,
    # leaving features_albedo at normal lr so road only learns colour. None =
    # leave that group untouched (default → byte-identical for non-road layers).
    positions_lr: float | None = None
    density_lr: float | None = None
    rotation_lr: float | None = None
    scale_lr: float | None = None
    features_albedo_lr: float | None = None
    # E3.2.5① (recon-studio ground-disk init, 2026-06-22): # of nearest road
    # LiDAR points whose Z is medianed per BEV grid cell in road_init. Trainer
    # passes this as init_road_layer(knn_k=...). 1 (default) = legacy
    # nearest-single-point (byte-identical off baseline); 5 = on (median-reject
    # LiDAR outlier spikes → ~8mm dense disc). Road-only knob; ignored elsewhere.
    road_init_knn_k: int = 1
    # E3.2.5③b (recon-studio zero_ground_gradients port, 2026-06-22): when True,
    # LayeredMCMCStrategy._post_backward zeroes this layer's rotation grad after
    # backward / before optimizer.step — killing both the update and the Adam
    # momentum source so the identity-quat (normal-vertical) disc is truly
    # locked. Stronger than rotation_lr (1e-4 lr still drifts via momentum over
    # 30k steps). Default False → byte-identical no-op for every other layer.
    freeze_rotation_grad: bool = False
    # T5.4: backend-specific knobs for non-particle layers. Currently used by
    # the sky_envmap layer to carry {"backend": "cubemap"|"mlp", "resolution":
    # int}. ``compare=False`` keeps LayerSpec hashable even though dict isn't,
    # because the auto-generated __eq__/__hash__ skip this field.
    extra: dict = field(default_factory=dict, compare=False)


def particle_layer_names_excluding(specs, exclude_layer_names) -> list[str]:
    """Phase 2A: particle-layer names in spec order, minus ``exclude_layer_names``.

    Pure (torch-free) so it unit-tests on a minimal env. Used by
    ``LayeredGaussians.get_density_excluding`` to build the opacity-reg density
    over every particle layer except the exempted ones (e.g. road), so
    ``lambda_opacity`` stops starving a structured/LiDAR-init layer.
    """
    exclude = set(exclude_layer_names or ())
    return [s.name for s in specs if s.is_particle_layer and s.name not in exclude]
