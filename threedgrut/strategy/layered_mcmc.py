# SPDX-License-Identifier: Apache-2.0
"""LayeredMCMCStrategy (T2.2): per-layer MCMC densification via sub-strategy array.

Design: holds one MCMCStrategy instance per particle layer; each sub's model
field points at LayeredGaussians.layers[name] (an independent MixtureOfGaussians
with its own optimizer). post_optimizer_step iterates sub-strategies → naturally
gives per-layer cap + scoped relocate/add/perturb + zero cross-layer migration.

Single-bg mode: only one sub-strategy → structurally identical to v1 MCMCStrategy
operating on the same MoG (validated by test_layered_mcmc_single_bg_uses_one_sub_strategy;
note: structural identity only, not byte-identical training output).

Non-particle layers (is_particle_layer=False, e.g. sky_envmap) are skipped:
they have no MoG particles to densify.
"""
from __future__ import annotations

import torch
from omegaconf import OmegaConf

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.layers.layered_model import LayeredGaussians
from threedgrut.model.bg_cuboid_loss import clamp_layer_positions_to_cuboids
from threedgrut.strategy.base import BaseStrategy
from threedgrut.strategy.mcmc import MCMCStrategy
from threedgrut.utils.logger import logger


class LayeredMCMCStrategy(BaseStrategy):
    """Per-layer MCMC densification driven by LayerSpec.max_n_particles."""

    def __init__(self, conf, model: LayeredGaussians, specs: list[LayerSpec]) -> None:
        super().__init__(config=conf, model=model)
        self.specs = list(specs)
        self.sub_strategies: dict[str, MCMCStrategy] = {}
        for spec in self.specs:
            if not spec.is_particle_layer:
                continue
            sub_conf = self._make_sub_conf(conf, spec)
            sub = MCMCStrategy(sub_conf, model.layers[spec.name])
            self._install_perturb_mask(sub, spec)
            self.sub_strategies[spec.name] = sub
        logger.info(
            f"LayeredMCMC: {len(self.sub_strategies)} sub-strategies for "
            f"layers {list(self.sub_strategies.keys())}"
        )

    @staticmethod
    def _install_perturb_mask(sub: MCMCStrategy, spec: LayerSpec) -> None:
        """T3.4 D1: bind spec.perturb_scale_mask onto the sub's _get_perturb_mask.

        If spec.perturb_scale_mask is None, leave the default (returns ones)
        so the sub's perturb behaviour is byte-identical with v1 MCMCStrategy.
        Otherwise install a per-sub override that returns the spec mask as a
        CPU tensor (gets .to(noise.device) inside perturb_gaussians).
        """
        if spec.perturb_scale_mask is None:
            return
        mask = torch.tensor(spec.perturb_scale_mask, dtype=torch.float32)
        # Instance-level override; default at the class still returns ones for
        # other subs / when bypassed via super().
        sub._perturb_mask_override = mask
        sub._get_perturb_mask = lambda: sub._perturb_mask_override

    @staticmethod
    def _make_sub_conf(conf, spec: LayerSpec):
        """Deep-copy conf and override add.max_n_gaussians for this layer.

        Uses OmegaConf.to_container + OmegaConf.create to produce an independent
        config object so modifying it does not affect the parent config.
        """
        sub = OmegaConf.create(OmegaConf.to_container(conf, resolve=False))
        sub.strategy.add.max_n_gaussians = spec.max_n_particles
        return sub

    def init_densification_buffer(self, checkpoint: dict | None = None) -> None:
        for sub in self.sub_strategies.values():
            sub.init_densification_buffer(checkpoint)

    def _post_optimizer_step(
        self, step: int, scene_extent: float, train_dataset, batch=None, writer=None
    ) -> bool:
        any_updated = False
        for name, sub in self.sub_strategies.items():
            updated = sub._post_optimizer_step(
                step, scene_extent, train_dataset, batch, writer
            )
            any_updated = any_updated or updated
        # T8/B3 — dynamic_rigids hard constraint: clamp positions back into
        # owner cuboid after MCMC perturb/add. Pure no-op when conf gate off
        # or when the layer / metadata are missing.
        self._maybe_clamp_dynamic_rigids()
        return any_updated

    def _maybe_clamp_dynamic_rigids(self) -> None:
        """In-place clamp dynamic_rigids positions to ``|local| ≤ size/2``.

        Gated by ``conf.trainer.bg_dyn_cuboid_penalty.dyn_clamp_to_cuboid``
        (default false) so v2 baseline training stays byte-identical until
        the dynfix yaml flips it on.
        """
        trainer_conf = getattr(self.config, "trainer", None)
        if trainer_conf is None:
            return
        cfg = (
            trainer_conf.get("bg_dyn_cuboid_penalty", None)
            if hasattr(trainer_conf, "get")
            else getattr(trainer_conf, "bg_dyn_cuboid_penalty", None)
        )
        if cfg is None:
            return
        enabled = (
            cfg.get("dyn_clamp_to_cuboid", False) if hasattr(cfg, "get")
            else getattr(cfg, "dyn_clamp_to_cuboid", False)
        )
        if not enabled:
            return

        model = self.model
        layers = getattr(model, "layers", None)
        if layers is None or "dynamic_rigids" not in layers:
            return
        dyn = layers["dynamic_rigids"]
        track_ids_buf = getattr(dyn, "track_ids", None)
        if track_ids_buf is None or dyn.positions.numel() == 0:
            return
        tracks_meta = getattr(model, "tracks_metadata", {})
        sizes_map: dict = {}
        for tid, meta in tracks_meta.items():
            if isinstance(meta, dict) and "size" in meta:
                sizes_map[tid] = meta["size"]
        if not sizes_map:
            return
        # The track-id integer assignment in dynamic_rigid_init mirrors
        # sorted(instance_pts_dict.keys()); populate_tracks uses the same
        # sort. tracks_poses keys are the authoritative source post-load.
        tracks_poses = getattr(model, "tracks_poses", {})
        track_keys_sorted = sorted(tracks_poses.keys()) if tracks_poses else sorted(sizes_map.keys())
        with torch.no_grad():
            clamp_layer_positions_to_cuboids(
                dyn.positions.data, track_ids_buf, track_keys_sorted, sizes_map,
            )

    def suspend(self) -> None:
        super().suspend()
        for sub in self.sub_strategies.values():
            sub.suspend()
