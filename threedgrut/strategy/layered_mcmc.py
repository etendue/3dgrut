# SPDX-License-Identifier: Apache-2.0
"""LayeredMCMCStrategy (T2.2): per-layer MCMC densification via sub-strategy array.

Design: holds one MCMCStrategy instance per particle layer; each sub's model
field points at LayeredGaussians.layers[name] (an independent MixtureOfGaussians
with its own optimizer). post_optimizer_step iterates sub-strategies → naturally
gives per-layer cap + scoped relocate/add/perturb + zero cross-layer migration.

Single-bg mode: only one sub-strategy → behavior byte-identical to v1
MCMCStrategy (validated by test_layered_mcmc_single_bg_equivalent_to_v1).

Non-particle layers (is_particle_layer=False, e.g. sky_envmap) are skipped:
they have no MoG particles to densify.
"""
from __future__ import annotations

from typing import List, Optional

from omegaconf import OmegaConf

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.layers.layered_model import LayeredGaussians
from threedgrut.strategy.base import BaseStrategy
from threedgrut.strategy.mcmc import MCMCStrategy
from threedgrut.utils.logger import logger


class LayeredMCMCStrategy(BaseStrategy):
    """Per-layer MCMC densification driven by LayerSpec.max_n_particles."""

    def __init__(self, conf, model: LayeredGaussians, specs: List[LayerSpec]) -> None:
        super().__init__(config=conf, model=model)
        self.specs = list(specs)
        self.sub_strategies: dict[str, MCMCStrategy] = {}
        for spec in self.specs:
            if not spec.is_particle_layer:
                continue
            sub_conf = self._make_sub_conf(conf, spec)
            self.sub_strategies[spec.name] = MCMCStrategy(
                sub_conf, model.layers[spec.name]
            )
        logger.info(
            f"LayeredMCMC: {len(self.sub_strategies)} sub-strategies for "
            f"layers {list(self.sub_strategies.keys())}"
        )

    @staticmethod
    def _make_sub_conf(conf, spec: LayerSpec):
        """Deep-copy conf and override add.max_n_gaussians for this layer.

        Uses OmegaConf.to_container + OmegaConf.create to produce an independent
        config object so modifying it does not affect the parent config.
        """
        sub = OmegaConf.create(OmegaConf.to_container(conf, resolve=False))
        sub.strategy.add.max_n_gaussians = spec.max_n_particles
        return sub

    def init_densification_buffer(self, checkpoint: Optional[dict] = None) -> None:
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
        return any_updated

    def suspend(self) -> None:
        super().suspend()
        for sub in self.sub_strategies.values():
            sub.suspend()
