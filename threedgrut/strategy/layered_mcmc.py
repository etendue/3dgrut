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

import math

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
from threedgrut.model.road_ownership import apply_bg_road_exclusion
from threedgrut.model.road_projection_candidates import (
    accumulate_projection_counts,
    make_projection_counts,
)
from threedgrut.model.road_region import build_road_height_field
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
        # Cached BEV fields for legacy A1 and MCRO B2 ownership exclusion.
        self._road_bev = None
        self._mcro_road_height_field = None
        self.last_bg_road_exclusion_stats = None
        self.last_bg_road_duplicate_stats = None
        self._bg_road_duplicate_visibility = None
        self._bg_road_duplicate_counts = None
        self._bg_road_duplicate_n_samples = 0
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
        # Projection evidence is tied to live particle row indices.  It is
        # intentionally not checkpointed; resume rebuilds it from new views.
        self._reset_bg_road_duplicate_evidence()
        for sub in self.sub_strategies.values():
            sub.init_densification_buffer(checkpoint)

    def set_step_outputs(self, outputs: dict | None) -> None:
        """Retain only B12's background visibility mask for the current step."""
        self._bg_road_duplicate_visibility = None
        cfg = self._bg_road_duplicate_cfg()
        if cfg is None or not bool(getattr(cfg, "enabled", False)):
            return
        visibility = outputs.get("mog_visibility") if isinstance(outputs, dict) else None
        if not torch.is_tensor(visibility):
            return
        visibility = visibility.detach().reshape(-1).bool()
        layers = getattr(self.model, "layers", None)
        if layers is None or "background" not in layers:
            return
        n_background = int(layers["background"].positions.shape[0])
        if visibility.numel() == n_background:
            self._bg_road_duplicate_visibility = visibility
            return
        try:
            layer_mask = self.model.get_layer_mask("background")
        except (AttributeError, ValueError):
            return
        if layer_mask.numel() == visibility.numel():
            self._bg_road_duplicate_visibility = visibility[layer_mask]

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
        # B12 must consume visibility before a relocation changes the meaning
        # of fused particle row indices.  A recycle action deliberately feeds
        # candidates into the MCMC dead pool below, so relocation can replace
        # them from legitimate live donors in the same step.
        duplicate_updated = self._maybe_apply_bg_road_duplicate_exclusion(step, batch, writer)
        any_updated = False
        for name, sub in self.sub_strategies.items():
            if name in exclude:
                continue
            updated = sub._post_optimizer_step(step, scene_extent, train_dataset, batch, writer)
            any_updated = any_updated or updated
            if name == "background" and getattr(sub, "last_relocation_count", 0) > 0:
                # Relocation reuses dead row indices for new particles; any
                # incomplete evidence window referred to the old row owners.
                self._reset_bg_road_duplicate_evidence()
        any_updated = any_updated or duplicate_updated
        # T8/B3 — dynamic_rigids hard constraint: clamp positions back into
        # owner cuboid after MCMC perturb/add. Pure no-op when conf gate off
        # or when the layer / metadata are missing.
        self._maybe_clamp_dynamic_rigids()
        # V3-R1.2 — road-layer scale clamp (XY/Z upper bound + anisotropy ratio).
        # No-op for layers whose LayerSpec leaves all 3 clamp fields None.
        self._maybe_clamp_road_scales()
        formal_enabled = self._maybe_apply_bg_road_exclusion(step, batch, writer)
        ownership_stats = self.last_bg_road_exclusion_stats
        if ownership_stats is not None and (
            ownership_stats["n_recycled"] > 0 or ownership_stats["n_footprint_shrunk"] > 0
        ):
            # Scale changes alter bounds; force acceleration-structure rebuild.
            any_updated = True
        # Preserve the historical experimental gates when formal B2 is off.
        if not formal_enabled:
            self._maybe_exclude_bg_from_road_slab()
            self._maybe_project_clamp_bg_density(batch)
        return any_updated

    def _bg_road_duplicate_cfg(self):
        layers_conf = getattr(self.conf, "layers", None)
        return getattr(layers_conf, "bg_road_duplicate_exclusion", None) if layers_conf is not None else None

    def _reset_bg_road_duplicate_evidence(self) -> None:
        self._bg_road_duplicate_counts = None
        self._bg_road_duplicate_n_samples = 0
        self._bg_road_duplicate_visibility = None

    @staticmethod
    def _resolve_training_road_mask(batch):
        image_infos = getattr(batch, "image_infos", None)
        if not isinstance(image_infos, dict):
            return None
        road_mask = image_infos.get("road_mask")
        if road_mask is not None:
            return road_mask
        semantic_sseg = image_infos.get("semantic_sseg")
        if semantic_sseg is None:
            return None
        return (semantic_sseg == 0) | (semantic_sseg == 1)

    @torch.no_grad()
    def _maybe_apply_bg_road_duplicate_exclusion(self, step: int, batch, writer=None) -> bool:
        """Accumulate B12 projection evidence and mutate at window boundaries."""
        self.last_bg_road_duplicate_stats = None
        cfg = self._bg_road_duplicate_cfg()
        if cfg is None or not bool(getattr(cfg, "enabled", False)):
            self._bg_road_duplicate_visibility = None
            return False

        every = int(getattr(cfg, "sample_every_steps", 10))
        window_samples = int(getattr(cfg, "window_samples", 8))
        warmup = int(getattr(cfg, "warmup_steps", 500))
        if every <= 0 or window_samples <= 0:
            raise ValueError(
                "layers.bg_road_duplicate_exclusion sample_every_steps and " "window_samples must be positive"
            )
        if step < warmup or (step - warmup) % every:
            self._bg_road_duplicate_visibility = None
            return False

        layers = getattr(self.model, "layers", None)
        if layers is None or "background" not in layers:
            self._reset_bg_road_duplicate_evidence()
            return False
        bg = layers["background"]
        road_mask = self._resolve_training_road_mask(batch)
        intrinsics = getattr(batch, "intrinsics_OpenCVPinholeCameraModelParameters", None)
        pose = getattr(batch, "T_to_world", None)
        visibility = self._bg_road_duplicate_visibility
        self._bg_road_duplicate_visibility = None
        required = (
            bg.positions.detach(),
            bg.get_scale().detach(),
            road_mask,
            intrinsics,
            pose,
            visibility,
        )
        if any(value is None for value in required):
            return False
        tensor_values = (required[0], required[1], road_mask, pose)
        if any(torch.is_tensor(value) and not bool(torch.isfinite(value).all()) for value in tensor_values):
            logger.warning(f"[MCRO B12] non-finite projection input at step={step}; evidence sample skipped")
            return False

        n_background = int(bg.positions.shape[0])
        if (
            self._bg_road_duplicate_counts is None
            or self._bg_road_duplicate_counts["visible_hits"].numel() != n_background
        ):
            self._bg_road_duplicate_counts = make_projection_counts(n_background, bg.positions.device)
            self._bg_road_duplicate_n_samples = 0
        accumulate_projection_counts(
            self._bg_road_duplicate_counts,
            positions_world=bg.positions.detach(),
            scales_linear=bg.get_scale().detach(),
            T_camera_to_world=pose,
            intrinsics=intrinsics,
            road_mask=road_mask,
            mog_visibility=visibility,
            erosion_px=int(getattr(cfg, "road_mask_erosion_px", 8)),
            protection_margin_px=int(getattr(cfg, "protection_margin_px", 16)),
            footprint_sigma=float(getattr(cfg, "footprint_sigma", 2.0)),
            max_footprint_px=float(getattr(cfg, "max_footprint_px", 48.0)),
            chunk_size=int(getattr(cfg, "chunk_size", 100_000)),
        )
        self._bg_road_duplicate_n_samples += 1
        if self._bg_road_duplicate_n_samples < window_samples:
            return False

        counts = self._bg_road_duplicate_counts
        alive = bg.get_density().detach().reshape(-1) > float(getattr(cfg, "opacity_threshold", 0.005))
        visible = counts["visible_hits"] >= int(getattr(cfg, "min_visible_hits", 1))
        road = counts["road_footprint_hits"] >= int(getattr(cfg, "min_road_hits", 1))
        protected = counts["protected_center_hits"] > int(getattr(cfg, "max_protected_hits", 0))
        candidate = alive & visible & road & ~protected
        stats = {
            "n_window_samples": int(self._bg_road_duplicate_n_samples),
            "n_visible": int((alive & visible).sum().item()),
            "n_road_candidates": int((alive & visible & road).sum().item()),
            "n_protected": int((alive & visible & road & protected).sum().item()),
            "n_recycled": 0,
            "n_decayed": 0,
            "n_footprint_shrunk": 0,
        }
        action = str(getattr(cfg, "action", "recycle"))
        if action == "recycle":
            bg.density[candidate] = float(getattr(cfg, "dead_density_raw", -50.0))
            stats["n_recycled"] = int(candidate.sum().item())
        elif action == "density_decay":
            bg.density[candidate] -= float(getattr(cfg, "density_decay", 5.0))
            stats["n_decayed"] = int(candidate.sum().item())
        elif action == "footprint_shrink":
            factor = float(getattr(cfg, "footprint_shrink_factor", 0.5))
            minimum = float(getattr(cfg, "min_footprint_scale", 1e-4))
            if not 0.0 < factor < 1.0 or minimum <= 0.0:
                raise ValueError("B12 footprint shrink requires factor in (0,1) and positive minimum")
            floor = torch.full_like(bg.scale[candidate], math.log(minimum))
            shrunk = torch.maximum(bg.scale[candidate] + math.log(factor), floor)
            bg.scale[candidate] = shrunk
            stats["n_footprint_shrunk"] = int(candidate.sum().item())
        else:
            raise ValueError(
                "layers.bg_road_duplicate_exclusion.action must be " "recycle, density_decay, or footprint_shrink"
            )
        self.last_bg_road_duplicate_stats = stats
        logger.info(f"[MCRO B12] projection ownership window step={step}: {stats}")
        if writer is not None:
            for name, value in stats.items():
                writer.add_scalar(f"mcro/bg_road_duplicate_exclusion/{name}", value, step)
        self._reset_bg_road_duplicate_evidence()
        return bool(candidate.any())

    def _maybe_apply_bg_road_exclusion(self, step: int, batch, writer=None) -> bool:
        """MCRO B2 formal GPU path; returns whether its config gate is enabled."""
        self.last_bg_road_exclusion_stats = None
        layers_conf = getattr(self.conf, "layers", None)
        cfg = getattr(layers_conf, "bg_road_exclusion", None) if layers_conf is not None else None
        if cfg is None or not bool(getattr(cfg, "enabled", False)):
            return False
        every_k = int(getattr(cfg, "every_k_steps", 10))
        if every_k <= 0:
            raise ValueError("layers.bg_road_exclusion.every_k_steps must be positive")
        if step % every_k:
            return True
        layers = getattr(self.model, "layers", None)
        if layers is None or "background" not in layers or "road" not in layers:
            return True
        road = layers["road"]
        if self._mcro_road_height_field is None:
            self._mcro_road_height_field = build_road_height_field(
                road.positions.detach(), cell_size=float(getattr(cfg, "cell_size", 1.0))
            )
        stats = apply_bg_road_exclusion(layers["background"], self._mcro_road_height_field, batch, cfg)
        self.last_bg_road_exclusion_stats = stats
        calls = getattr(self, "_mcro_bg_road_calls", 0) + 1
        self._mcro_bg_road_calls = calls
        if calls == 1 or calls % 50 == 0:
            logger.info(f"[MCRO B2] bg-road exclusion call={calls}: {stats}")
        if writer is not None:
            for name, value in stats.items():
                writer.add_scalar(f"mcro/bg_road_exclusion/{name}", value, step)
        return True

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
