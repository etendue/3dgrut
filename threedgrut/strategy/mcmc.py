# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MCMC implementation was adpoted from gSplat library (https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/strategy/mcmc.py),
# which is based on the original implementation https://github.com/ubc-vision/3dgs-mcmc that uderlines the work
#
# 3D Gaussian Splatting as Markov Chain Monte Carlo by
# Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Weiwei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi and Kwang Moo Yie
#
# If you use this code in your research, please cite the above works.

import math
from typing import Optional, Tuple

import torch

from threedgrut.model.model import MixtureOfGaussians
from threedgrut.strategy.base import BaseStrategy
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import _multinomial_sample, check_step_condition


@torch.no_grad()
def _sanitize_relocation(
    new_densities: torch.Tensor,   # [N, 1] kernel output (activated space)
    new_scales: torch.Tensor,      # [N, 3] kernel output (activated space)
    donor_densities: torch.Tensor,  # [N, 1] donor originals (activated space)
    donor_scales: torch.Tensor,     # [N, 3] donor originals (activated space)
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """A1 — contain non-finite/non-positive relocation-kernel outputs.

    compute_relocation_tensor (MCMC Eq.9 binomial math) can emit NaN/Inf or
    zero/negative densities/scales at opacity→1 boundaries. The log inverse
    activations downstream turn those into NaN/-inf parameters, silently
    poisoning the layer (observed inc_b6a9 6-cam R1 2026-07-02: 60% of
    background densities NaN by ~3k steps; the poison never crosses the loss
    so pred-based guards can't catch it). Any bad ROW (density or any scale
    component non-finite or ≤0) falls back to the donor's original values —
    relocation degenerates to a plain copy for that row.

    Returns (densities, scales, n_bad_rows); inputs are not modified.
    """
    bad = (
        ~torch.isfinite(new_densities).all(dim=-1)
        | (new_densities <= 0).any(dim=-1)
        | ~torch.isfinite(new_scales).all(dim=-1)
        | (new_scales <= 0).any(dim=-1)
    )  # [N]
    n_bad = int(bad.sum())
    if n_bad == 0:
        return new_densities, new_scales, 0
    out_d = torch.where(bad.unsqueeze(-1), donor_densities, new_densities)
    out_s = torch.where(bad.unsqueeze(-1), donor_scales, new_scales)
    return out_d, out_s, n_bad

_mcmc_plugin = None


def load_mcmc_plugin():
    global _mcmc_plugin
    if _mcmc_plugin is None:
        try:
            from . import lib_mcmc_cc as gaussian_mcmc

            _mcmc_plugin = gaussian_mcmc  # type: ignore
        except ImportError:
            from threedgrut.strategy.src.setup_mcmc import setup_mcmc

            _mcmc_plugin = setup_mcmc()


class MCMCStrategy(BaseStrategy):
    """Densification and prunning strategy that follows the paper:

    `3D Gaussian Splatting as Markov Chain Monte Carlo <https://arxiv.org/abs/2404.09591>`

    MCMC Strategy interprets the training process of placing and optimizing Gaussians
    as a sampling process.

    Specifically, it periodically:
    - Moves "dead" Gaussians (low opacity) to the location of "live" Gaussians (high opacity).
    - Adds covariance dependent noise to the positions of the Gaussians.
    - Introduces new Gaussians sampled based on the opacity distribution.

    """

    def __init__(self, config, model: MixtureOfGaussians) -> None:
        super().__init__(config=config, model=model)

        load_mcmc_plugin()
        # Precompute the look up table for binomial coefficients (Eq 9 in the MCMC paper)
        self.binoms = torch.tensor(
            [
                [math.comb(n, k) if k <= n else 0 for k in range(config.strategy.binom_n_max)]
                for n in range(config.strategy.binom_n_max)
            ],
            dtype=torch.float32,
            device=self.model.device,
        )

    def _post_optimizer_step(self, step: int, scene_extent: float, train_dataset, batch=None, writer=None) -> bool:
        # Relocate dead gaussians to the alive areas
        if check_step_condition(
            step,
            self.conf.strategy.relocate.start_iteration,
            self.conf.strategy.relocate.end_iteration,
            self.conf.strategy.relocate.frequency,
        ):
            self.relocate_gaussians()

        # Add new Gaussians if the maximum number has not been reached
        if check_step_condition(
            step,
            self.conf.strategy.add.start_iteration,
            self.conf.strategy.add.end_iteration,
            self.conf.strategy.add.frequency,
        ):
            self.add_new_gaussians()

        # Perturb the positions of the Gaussians
        if check_step_condition(
            step,
            self.conf.strategy.perturb.start_iteration,
            self.conf.strategy.perturb.end_iteration,
            self.conf.strategy.perturb.frequency,
        ):
            self.perturb_gaussians()

        return True

    @torch.no_grad()
    def relocate_gaussians(self) -> None:
        # Get the per Gaussian densities and scales (after sigmoid)
        densities = self.model.get_density()
        # Find the dead indices
        dead_idxs = torch.where(densities <= self.conf.strategy.opacity_threshold)[0]
        alive_idxs = torch.where(densities > self.conf.strategy.opacity_threshold)[0]
        n_dead_gaussians = len(dead_idxs)
        n_dead_total = n_dead_gaussians  # pre-cap count, for truthful logging

        # Cap relocation to avoid super-dense clusters when a layer collapses
        # (e.g. dynamic_rigids at 90% dead → 630k particles crammed into 70k spots
        # → tile-buffer OOM in the renderer). max_relocation_fraction=1.0 disables.
        max_frac = float(getattr(self.conf.strategy.relocate, "max_relocation_fraction", 1.0))
        if max_frac < 1.0 and n_dead_gaussians > 0:
            cap = max(1, int(max_frac * len(densities)))
            if n_dead_gaussians > cap:
                perm = torch.randperm(n_dead_gaussians, device=dead_idxs.device)
                dead_idxs = dead_idxs[perm[:cap]]
                n_dead_gaussians = cap

        # A1-guard: a fully-dead layer has no alive donors — multinomial over
        # an empty probability tensor aborts with CUDA invalid-configuration
        # (observed 2026-07-02 inc_b6a9 6-cam R1, layer collapsed by the first
        # relocation step). Skip relocation and surface the collapse loudly;
        # the layer size in the log identifies which layer it is.
        if n_dead_gaussians and len(alive_idxs) == 0:
            # NOTE: report the PRE-cap dead count — n_dead_gaussians may have
            # been subsampled to max_relocation_fraction above, which reads
            # misleadingly (e.g. "400000/1000000" when ALL 1M are dead).
            logger.warning(
                f"[A1] relocate skipped: layer fully dead "
                f"({n_dead_total}/{len(densities)} particles at or below "
                f"opacity_threshold={self.conf.strategy.opacity_threshold}) — "
                f"no alive donors to sample from; layer will not recover via "
                f"MCMC this step"
            )
            return

        if n_dead_gaussians:
            sampled_idxs, new_densities, new_scales = self.sample_new_gaussians(n_dead_gaussians, alive_idxs)

            def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
                if name == "density":
                    param[sampled_idxs] = new_densities
                elif name == "scale":
                    param[sampled_idxs] = new_scales
                param[dead_idxs] = param[sampled_idxs]
                return torch.nn.Parameter(param, requires_grad=param.requires_grad)

            def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
                v[sampled_idxs] = 0
                return v

            self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)

            # T4.5 fix: dead particles inherit track_id from their alive donor
            # so the reborn particle sits in the right cuboid's local frame.
            if hasattr(self.model, "track_ids") and self.model.track_ids is not None:
                self.model.track_ids[dead_idxs] = self.model.track_ids[sampled_idxs]

        if self.conf.strategy.print_stats:
            # Guard div-by-zero: an empty particle layer (e.g. dynamic_rigids on a
            # clip with no cuboid autolabels) has len(densities)==0, and the
            # per-layer LayeredMCMC sub-strategy still runs relocate every step.
            n_total = len(densities)
            pct = (n_dead_gaussians / n_total * 100) if n_total else 0.0
            logger.info(f"Relocated {n_dead_gaussians} ({pct:.2f}%) gaussians")

    def _get_add_cap(self) -> int:
        """Maximum total particle count for add_new_gaussians. Override in subclasses
        (e.g. LayeredMCMCStrategy) to scope to a single layer."""
        return self.conf.strategy.add.max_n_gaussians

    def _get_perturb_mask(self) -> torch.Tensor:
        """[3] elementwise multiplier on the positional perturb noise (xyz).

        Default = ones (v1 byte-identical: noise unaffected).
        LayeredMCMCStrategy overrides this per sub to honour LayerSpec.
        perturb_scale_mask — e.g. road layer uses (1, 1, 0) so MCMC perturb
        cannot drift the thin disc off its LiDAR-Z-locked surface (T3.4 D1).
        """
        return torch.ones(3)

    @torch.no_grad()
    def add_new_gaussians(self) -> None:
        # Get the current number of gaussians
        current_num_gaussians = self.model.num_gaussians
        target_num_gaussians = min(self._get_add_cap(), int(1.05 * current_num_gaussians))
        num_gaussians_to_add = max(0, target_num_gaussians - current_num_gaussians)

        if num_gaussians_to_add:
            sampled_idxs, new_densities, new_scales = self.sample_new_gaussians(num_gaussians_to_add)

            def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
                if name == "density":
                    param[sampled_idxs] = new_densities
                elif name == "scale":
                    param[sampled_idxs] = new_scales
                param_new = torch.cat([param, param[sampled_idxs]])
                return torch.nn.Parameter(param_new, requires_grad=param.requires_grad)

            def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
                v_new = torch.zeros((len(sampled_idxs), *v.shape[1:]), device=v.device)
                return torch.cat([v, v_new])

            self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)

            # T4.5 fix: sync per-particle non-Parameter buffers (e.g. track_ids
            # for dynamic_rigids layer) that _update_param_with_optimizer only
            # touches Parameter fields. Without this, fused_view sees a layer
            # whose positions.shape[0] > track_ids.shape[0] → crash.
            if hasattr(self.model, "track_ids") and self.model.track_ids is not None:
                self.model.track_ids = torch.cat(
                    [self.model.track_ids, self.model.track_ids[sampled_idxs]]
                )

        if self.conf.strategy.print_stats:
            # Guard div-by-zero: an empty particle layer has
            # current_num_gaussians==0 (and num_gaussians_to_add==0), but the
            # per-layer LayeredMCMC sub-strategy still runs add every step.
            pct = (num_gaussians_to_add / current_num_gaussians * 100) if current_num_gaussians else 0.0
            logger.info(
                f"Added {num_gaussians_to_add} ({pct:.2f}%) gaussians"
            )

    @torch.no_grad()
    def perturb_gaussians(self) -> None:
        covariance = self.model.get_covariance()
        positions = self.model.get_positions()
        densities = self.model.get_density()

        current_lr = 0.0
        for param_gorup in self.model.optimizer.param_groups:
            if param_gorup["name"] == "positions":
                current_lr = param_gorup["lr"]

        def op_sigmoid(x: torch.Tensor, k: int = 100, x0: float = 0.995) -> torch.Tensor:
            return 1 / (1 + torch.exp(-k * (x - x0)))

        # Current positional learning rate multiplied by the config paramater scale
        noise = (
            torch.randn_like(positions) * (op_sigmoid(1 - densities)) * self.conf.strategy.perturb.noise_lr * current_lr
        )
        noise = torch.bmm(covariance, noise.unsqueeze(-1)).squeeze(-1)

        # T3.4 D1: per-axis mask on positional noise. Default is ones (v1
        # byte-identical); LayeredMCMC road sub overrides to (1, 1, 0).
        noise = noise * self._get_perturb_mask().to(noise.device).to(noise.dtype)

        self.model.positions.add_(noise)

    def sample_new_gaussians(
        self, num_gaussians: int, valid_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        densities = self.model.get_density()
        scales = self.model.get_scale()

        if valid_indices is None:
            valid_indices = torch.arange(0, int(densities.shape[0]), device=densities.device, dtype=torch.int32)

        probabilities = densities[valid_indices].flatten()  # ensure its shape is [N,]

        # Sample the locations to which the dead Gaussians will be moved proportional to the opacity of the alive Gaussians
        sampled_idxs = _multinomial_sample(probabilities, num_gaussians, replacement=True)
        sampled_idxs = valid_indices[sampled_idxs]

        ratios = (
            (torch.bincount(sampled_idxs)[sampled_idxs] + 1).clamp_(min=1, max=self.conf.strategy.binom_n_max).int()
        )

        # A1-guard (input): donor opacity can saturate to exactly 1.0 in
        # float; the relocation kernel's (1-o) terms then divide by zero.
        donor_densities = densities[sampled_idxs].clamp(
            max=1.0 - torch.finfo(torch.float32).eps
        )
        donor_scales = scales[sampled_idxs]

        new_densities, new_scales = _mcmc_plugin.compute_relocation_tensor(
            donor_densities.contiguous(),
            donor_scales.contiguous(),
            ratios.contiguous(),
            self.binoms,
            self.conf.strategy.binom_n_max,
        )

        # A1-guard (output): the kernel's Eq.9 binomial math can still emit
        # non-finite / non-positive values at opacity boundaries; the log
        # inverse activations below turn those into NaN/-inf PARAMETERS,
        # silently poisoning the layer (inc_b6a9 6-cam R1: 60% of background
        # densities NaN by ~3k steps — never crosses the loss, so the
        # pred-based drop guard can't see it). Bad rows fall back to the
        # donor's original values (plain copy semantics).
        new_densities, new_scales, n_bad = _sanitize_relocation(
            new_densities, new_scales, donor_densities, donor_scales,
        )
        if n_bad:
            logger.warning(
                f"[A1] relocation kernel emitted {n_bad}/{int(ratios.shape[0])} "
                f"non-finite/non-positive rows — fell back to donor copy"
            )

        new_densities = self.model.density_activation_inv(
            torch.clamp(
                new_densities, max=1.0 - torch.finfo(torch.float32).eps, min=self.conf.strategy.opacity_threshold
            )
        )

        new_scales = self.model.scale_activation_inv(new_scales)

        return sampled_idxs, new_densities, new_scales
