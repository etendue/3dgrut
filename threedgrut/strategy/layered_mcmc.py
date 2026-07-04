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
from threedgrut.model.road_reg import (
    bg_in_road_slab_mask,
    build_road_bev_height,
    clamp_layer_scales,
    project_bg_road_hits,
)
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
        # A1: cached BEV road-height field for background-slab exclusion (built
        # lazily on first use; road layer is ~frozen so it stays valid).
        self._road_bev = None
        logger.info(
            f"LayeredMCMC: {len(self.sub_strategies)} sub-strategies for " f"layers {list(self.sub_strategies.keys())}"
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

    def _post_backward(self, step: int, scene_extent: float, train_dataset, batch=None, writer=None) -> bool:
        """E3.2.5③b: zero rotation grads for layers with freeze_rotation_grad.

        Runs after loss.backward() and BEFORE optimizer.step()/zero_grad()
        (trainer post_backward seam). Zeroing the grad here kills BOTH the
        gradient step AND the Adam momentum source, truly locking the road
        disc's identity-quat (normal vertical) — the recon-studio
        ``zero_ground_gradients`` equivalent that lr-override alone cannot
        achieve (a 1e-4 rotation lr still drifts via Adam momentum over 30k
        steps; spec §5 lists "未锁法线 → disc tilts" as a roadoff-freeze
        failure cause). Default (every spec freeze_rotation_grad=False) →
        byte-identical no-op. Returns False (no scene-structure change).
        """
        layers = getattr(self.model, "layers", None)
        if layers is None:
            return False
        for spec in self.specs:
            if not getattr(spec, "freeze_rotation_grad", False):
                continue
            layer = layers.get(spec.name) if hasattr(layers, "get") else layers[spec.name]
            if layer is None:
                continue
            rot = getattr(layer, "rotation", None)
            if rot is not None and rot.grad is not None:
                rot.grad.zero_()
        return False

    def _post_optimizer_step(self, step: int, scene_extent: float, train_dataset, batch=None, writer=None) -> bool:
        # E3 road-freeze (NuRec `strategy.exclude_layer_ids`): layers listed here
        # are fully exempt from MCMC — no add/relocate/perturb/prune, so their
        # particle set stays exactly as initialized. Default empty → baseline
        # byte-identical. CLI: ++strategy.exclude_layer_ids=[road].
        exclude = set(getattr(getattr(self.conf, "strategy", None), "exclude_layer_ids", None) or [])
        any_updated = False
        for name, sub in self.sub_strategies.items():
            if name in exclude:
                continue
            updated = sub._post_optimizer_step(step, scene_extent, train_dataset, batch, writer)
            any_updated = any_updated or updated
        # T8/B3 — dynamic_rigids hard constraint: clamp positions back into
        # owner cuboid after MCMC perturb/add. Pure no-op when conf gate off
        # or when the layer / metadata are missing.
        self._maybe_clamp_dynamic_rigids()
        # V3-R1.2 — road-layer scale clamp (XY/Z upper bound + anisotropy ratio).
        # No-op for layers whose LayerSpec leaves all 3 clamp fields None.
        self._maybe_clamp_road_scales()
        # A1 — hard-exclude background gaussians from the thin road slab so the
        # frozen road layer owns the road surface. No-op unless enabled.
        self._maybe_exclude_bg_from_road_slab()
        # A2 — image-space variant: project bg centers into this training camera
        # and clamp those landing on road-mask pixels (catches floating bg the
        # 3D slab misses). No-op unless projection_enabled.
        self._maybe_project_clamp_bg_density(batch)
        return any_updated

    def _maybe_clamp_dynamic_rigids(self) -> None:
        """In-place clamp dynamic_rigids positions to ``|local| ≤ size/2``.

        Gated by ``conf.trainer.bg_dyn_cuboid_penalty.dyn_clamp_to_cuboid``
        (default false) so v2 baseline training stays byte-identical until
        the dynfix yaml flips it on.
        """
        # BaseStrategy stores config under ``self.conf`` (base.py:25).
        trainer_conf = getattr(self.conf, "trainer", None)
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
            cfg.get("dyn_clamp_to_cuboid", False) if hasattr(cfg, "get") else getattr(cfg, "dyn_clamp_to_cuboid", False)
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
                dyn.positions.data,
                track_ids_buf,
                track_keys_sorted,
                sizes_map,
            )

    def _maybe_clamp_road_scales(self) -> None:
        """V3-R1.2: in-place clamp per-layer scale params for layers whose
        LayerSpec sets any of scale_xy_max / scale_z_max / anisotropy_ratio_max.

        Road uses (0.3m, 0.05m, 8x); all other layers leave the three fields
        None, so clamp_layer_scales is a no-op and this loop skips them —
        byte-identical training for non-road layers.
        """
        layers = getattr(self.model, "layers", None)
        if layers is None:
            return
        for spec in self.specs:
            if not spec.is_particle_layer:
                continue
            if spec.scale_xy_max is None and spec.scale_z_max is None and spec.anisotropy_ratio_max is None:
                continue
            layer = layers.get(spec.name) if hasattr(layers, "get") else layers[spec.name]
            if layer is None or layer.scale.numel() == 0:
                continue
            with torch.no_grad():
                # copy_ (not .data=) bumps the param version so autograd/optimizer stay consistent
                layer.scale.copy_(clamp_layer_scales(layer.scale.detach(), spec))

    def _maybe_exclude_bg_from_road_slab(self) -> None:
        """A1: hard-clamp opacity of background gaussians inside the road slab
        to ~0 so the frozen road layer owns the road surface. Gradient-free,
        runs every optimizer step. No-op unless
        ``strategy.bg_road_slab_exclude.enabled`` is true.
        """
        strat = getattr(self.conf, "strategy", None)
        cfg = getattr(strat, "bg_road_slab_exclude", None) if strat is not None else None
        if cfg is None or not getattr(cfg, "enabled", False):
            return
        layers = getattr(self.model, "layers", None)
        if not layers or "background" not in layers or "road" not in layers:
            return
        bg = layers["background"]
        road = layers["road"]
        if road.positions.shape[0] == 0 or bg.positions.shape[0] == 0:
            return

        mode = str(getattr(cfg, "mode", "clamp"))
        if mode != "clamp":
            raise NotImplementedError(f"bg_road_slab_exclude.mode='{mode}' not implemented in A1 (use 'clamp')")

        if getattr(self, "_road_bev", None) is None:
            self._road_bev = build_road_bev_height(road.positions.detach(), cell=float(getattr(cfg, "cell", 0.20)))
        mask = bg_in_road_slab_mask(bg.positions.detach(), self._road_bev, band_z=float(getattr(cfg, "band_z", 0.15)))
        # Observability (review caveat): log how many bg gaussians are clamped so
        # the A/B run shows the exclusion is firing (and not fighting photometric
        # gradients). Throttled: first call + every 500th.
        n_excl = int(mask.sum())
        self._bg_excl_calls = getattr(self, "_bg_excl_calls", 0) + 1
        if self._bg_excl_calls == 1 or self._bg_excl_calls % 500 == 0:
            logger.info(
                f"[A1] bg_road_slab_exclude: clamped {n_excl} background gaussians " f"(call #{self._bg_excl_calls})"
            )
        if n_excl == 0:
            return
        with torch.no_grad():
            bg.density[mask] = float(getattr(cfg, "clamp_value", -50.0))

    def _maybe_project_clamp_bg_density(self, batch) -> None:
        """A2: project background gaussian centers into the current training
        camera and hard-clamp the opacity of those landing on road-mask pixels.
        Catches floating bg above the road that the 3D slab (A1) misses (caught
        by WHERE it projects, not its 3D height). Gradient-free, every step.
        No-op unless ``strategy.bg_road_slab_exclude.projection_enabled`` is true.
        """
        strat = getattr(self.conf, "strategy", None)
        cfg = getattr(strat, "bg_road_slab_exclude", None) if strat is not None else None
        if cfg is None or not getattr(cfg, "projection_enabled", False):
            return
        if batch is None:
            return
        image_infos = getattr(batch, "image_infos", None)
        road_mask_t = image_infos.get("road_mask") if isinstance(image_infos, dict) else None
        T_to_world = getattr(batch, "T_to_world", None)
        if road_mask_t is None or T_to_world is None:
            return
        layers = getattr(self.model, "layers", None)
        if not layers or "background" not in layers:
            return
        bg = layers["background"]
        if bg.positions.shape[0] == 0:
            return

        # Camera model: FTheta (PAI) or OpenCVPinhole (NCore inceptio).
        ftheta = getattr(batch, "intrinsics_FThetaCameraModelParameters", None)
        pinhole = getattr(batch, "intrinsics_OpenCVPinholeCameraModelParameters", None)
        if ftheta is not None:
            intr, model_type = ftheta, "ftheta"
        elif pinhole is not None:
            intr, model_type = pinhole, "pinhole"
        else:
            return
        # Projectors expect numpy arrays in the dict; coerce any tensor values.
        intr_np = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in intr.items()}

        rm = road_mask_t.detach().cpu().numpy()
        T_c2w = T_to_world[0].detach().cpu().numpy()
        bg_xyz = bg.positions.detach().cpu().numpy()
        hits = project_bg_road_hits(bg_xyz, T_c2w, intr_np, model_type, rm)

        n_hit = int(hits.sum())
        self._bg_proj_calls = getattr(self, "_bg_proj_calls", 0) + 1
        if self._bg_proj_calls == 1 or self._bg_proj_calls % 500 == 0:
            logger.info(
                f"[A2] bg_road_proj_clamp: clamped {n_hit} background gaussians " f"(call #{self._bg_proj_calls})"
            )
        if n_hit == 0:
            return
        hit_t = torch.from_numpy(hits).to(bg.density.device)
        with torch.no_grad():
            bg.density[hit_t] = float(getattr(cfg, "clamp_value", -50.0))

    def suspend(self) -> None:
        super().suspend()
        for sub in self.sub_strategies.values():
            sub.suspend()
