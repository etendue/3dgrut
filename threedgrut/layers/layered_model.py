# SPDX-License-Identifier: Apache-2.0
"""LayeredGaussians: container of per-layer MixtureOfGaussians.

T1.1 scope: container skeleton + ckpt interop only.
    - Each layer is a full MixtureOfGaussians instance.
    - get_model_parameters() emits NRE-aligned nested schema:
        {"gaussians_nodes": {layer_name: <MoG params>}, "scene_extent": float}
    - init_from_checkpoint() auto-detects three input shapes:
        (a) ckpt["model"]["gaussians_nodes"][name]  -- NRE / v2 with outer wrap
        (b) ckpt["gaussians_nodes"][name]           -- v2 already unwrapped
        (c) flat v1: ckpt["positions"], ...         -- legacy, routed to "background"
    - Missing layers in ckpt: warn + skip (do not raise).

T2/T3/T4 scope (NOT in T1.1):
    - fused_view(cur_frame) -> flat tensors for renderer
    - per-frame dynamic pose transforms
    - LayeredMCMCStrategy hooks
    - per-layer optimizers (T1.1 keeps single-optimizer compat via bridge)

Single-bg-layer bridge: when there is exactly one layer named "background",
LayeredGaussians transparently forwards attribute access (positions, rotation,
optimizer, renderer, num_gaussians, get_density(), etc.) to that layer. This
keeps the existing Trainer + MCMCStrategy code paths working without
modification for T1.1 smoke. T2 replaces this with explicit fused-view logic.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import sh_degree_to_specular_dim


# Degree-0 SH coefficient: 1 / (2 * sqrt(pi)). Used to convert RGB in [0, 1]
# to the DC term of the SH expansion in init_layer_from_points.
_SH_C0 = 0.28209479177387814


class _LayeredOptimizerView:
    """Lightweight optimizer-like wrapper that fans calls across per-layer Adams.

    LayeredGaussians.optimizer returns one of these in multi-layer mode so the
    Trainer's main loop ``self.model.optimizer.step()`` works unchanged. In
    single-bg mode the property short-circuits to the bg layer's own optimizer
    (byte-identical with the v1 path; this wrapper is not used).
    """

    def __init__(self, layers: nn.ModuleDict) -> None:
        self._layers = layers

    def step(self) -> None:
        for layer in self._layers.values():
            opt = getattr(layer, "optimizer", None)
            if opt is not None:
                opt.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for layer in self._layers.values():
            opt = getattr(layer, "optimizer", None)
            if opt is not None:
                opt.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self) -> list[dict]:
        groups: list[dict] = []
        for layer in self._layers.values():
            opt = getattr(layer, "optimizer", None)
            if opt is not None:
                groups.extend(opt.param_groups)
        return groups


# Per-particle parameter names forwarded to the single background layer when in
# single-layer mode (so MCMCStrategy's setattr(model, name, new_param) writes
# to bg layer's Parameter rather than registering one on LayeredGaussians).
_FORWARD_PARAM_NAMES = frozenset({
    "positions", "rotation", "scale", "density",
    "features_albedo", "features_specular",
})


class LayeredGaussians(nn.Module):
    """Drop-in replacement for MixtureOfGaussians when use_layered_model=True."""

    def __init__(self, conf, specs: List[LayerSpec], scene_extent: float,
                 *, tracks: Optional[dict] = None):
        super().__init__()
        # Use object.__setattr__ for non-Module fields to bypass our custom
        # __setattr__ forwarding logic below.
        object.__setattr__(self, "conf", conf)
        object.__setattr__(self, "specs", list(specs))
        object.__setattr__(self, "scene_extent", scene_extent)
        # nn.ModuleDict so .to(device) / state_dict() recurse into layers.
        self.layers: nn.ModuleDict = nn.ModuleDict()
        for spec in specs:
            self.layers[spec.name] = MixtureOfGaussians(conf, scene_extent)

        # T4.0: per-clip dynamic-rigid track poses, registered as nn.Buffer so
        # they ride along with .to(device) / state_dict() / .cuda(). v2 assumes
        # single-clip training (D3); multi-clip would require re-construct.
        # Stored separately as flat names "_track_pose_<tid>" / "_track_active_<tid>"
        # (PyTorch register_buffer disallows '.' in names) plus mirror Python
        # dicts for ergonomic per-track lookup by name.
        object.__setattr__(self, "tracks_poses", {})    # {tid: Tensor[F, 4, 4]}
        object.__setattr__(self, "tracks_active", {})   # {tid: BoolTensor[F]}
        if tracks is not None:
            for tid, info in tracks.items():
                poses = info["poses"] if isinstance(info, dict) else info[0]
                active = (info["active"] if isinstance(info, dict) and "active" in info
                          else info.get("frame_info") if isinstance(info, dict)
                          else info[1])
                buf_pose_name = f"_track_pose_{tid}"
                buf_active_name = f"_track_active_{tid}"
                self.register_buffer(buf_pose_name, poses, persistent=True)
                self.register_buffer(buf_active_name, active.to(torch.bool),
                                     persistent=True)
                self.tracks_poses[tid] = getattr(self, buf_pose_name)
                self.tracks_active[tid] = getattr(self, buf_active_name)

    # ------------------------------------------------------------------ checkpoint

    def get_model_parameters(self) -> dict:
        """Return NRE-aligned sub-dict.

        Output (consumed by Trainer.save_checkpoint, which adds outer wrappers
        like 'global_step', 'epoch', strategy state, post_processing):

            {
                "gaussians_nodes": {layer_name: <MoG.get_model_parameters()>},
                "scene_extent": float,
            }

        After Trainer writes this under the 'model' key, the on-disk ckpt is:
            ckpt["model"]["gaussians_nodes"]["background"]["positions"]
        matching NRE (nvcr.io/nvidia/nre/nre-ga:latest) output format.
        """
        return {
            "gaussians_nodes": {
                name: layer.get_model_parameters()
                for name, layer in self.layers.items()
            },
            "scene_extent": self.scene_extent,
        }

    def init_from_checkpoint(self, checkpoint: dict, setup_optimizer: bool = True):
        """Dispatch checkpoint into per-layer MoG. Accepts three input shapes:

        (a) NRE-wrapped v2:    checkpoint["model"]["gaussians_nodes"][name]
        (b) already-unwrapped: checkpoint["gaussians_nodes"][name]
        (c) v1 legacy flat:    checkpoint["positions"], ... (no nesting)
                               -> route entire dict into layers["background"]

        Missing layers warn + skip. Extra ckpt keys ignored silently.
        """
        nodes_dict = None
        if (
            "model" in checkpoint
            and isinstance(checkpoint["model"], dict)
            and "gaussians_nodes" in checkpoint["model"]
        ):
            nodes_dict = checkpoint["model"]["gaussians_nodes"]
        elif "gaussians_nodes" in checkpoint:
            nodes_dict = checkpoint["gaussians_nodes"]

        if nodes_dict is not None:
            # v2 path: dispatch per layer.
            for name, layer in self.layers.items():
                if name not in nodes_dict:
                    logger.warning(
                        f"[ckpt] Layer '{name}' not found in checkpoint "
                        f"['gaussians_nodes']; keeping it empty (warn+skip)."
                    )
                    continue
                layer.init_from_checkpoint(
                    nodes_dict[name], setup_optimizer=setup_optimizer
                )
            return

        # v1 legacy path: route entire flat dict into background.
        if "background" not in self.layers:
            enabled = [s.name for s in self.specs]
            raise ValueError(
                f"v1 checkpoint detected (flat schema with 'positions' key) "
                f"but 'background' layer is not in conf.layers.enabled "
                f"(got {enabled}). Add 'background' to layers.enabled to "
                f"resume v1 checkpoints."
            )
        n_particles = checkpoint["positions"].shape[0]
        logger.info(
            f"[v1->v2] Detected v1-shape checkpoint ({n_particles} particles); "
            f"routing all into layer 'background'."
        )
        self.layers["background"].init_from_checkpoint(
            checkpoint, setup_optimizer=setup_optimizer
        )

    # ------------------------------------------------------------------ test helpers

    def setup_optimizer_for_test(self):
        """Minimal optimizer attach for unit tests; avoids touching Trainer.

        Plays the role of MoG.setup_optimizer() but skips conf-driven schedulers
        and per-name LR multipliers (the test only needs get_model_parameters()
        to pass its assert).
        """
        for layer in self.layers.values():
            layer.optimizer = torch.optim.Adam(
                [
                    {"params": [layer.positions],         "name": "positions"},
                    {"params": [layer.rotation],          "name": "rotation"},
                    {"params": [layer.scale],             "name": "scale"},
                    {"params": [layer.density],           "name": "density"},
                    {"params": [layer.features_albedo],   "name": "features_albedo"},
                    {"params": [layer.features_specular], "name": "features_specular"},
                ],
                lr=1e-3,
            )

    # ------------------------------------------------------------------ single-layer bridge
    # When the container holds exactly one "background" layer (T1.1 default),
    # forward attribute reads and Parameter writes through to that layer so the
    # existing Trainer + MCMCStrategy + renderer code paths continue working.

    def forward(self, *args, **kwargs):
        """Forward the render call to the single bg layer in single-layer mode.

        `nn.Module.__call__` resolves `forward` via class-level lookup, which
        bypasses our `__getattr__` bridge entirely. Define forward explicitly
        so `self.model(batch, train=True, frame_id=...)` works in single-bg
        T1.1 mode. Multi-layer mode lands in T2 with a fused-view forward.
        """
        bg = self._single_bg_layer()
        if bg is None:
            raise NotImplementedError(
                "LayeredGaussians.forward only defined for single-bg-layer mode "
                "(T1.1 scope). Multi-layer fused-view forward lands in T2."
            )
        return bg(*args, **kwargs)

    def _single_bg_layer(self):
        """Return the bg layer iff this is a single-bg-layer setup, else None."""
        modules = self.__dict__.get("_modules", {})
        layers = modules.get("layers")
        if layers is not None and len(layers) == 1 and "background" in layers:
            return layers["background"]
        return None

    # ------------------------------------------------------------------ T2.5 fused-view
    def fused_view(self, frame_id: int | None = None) -> dict[str, torch.Tensor]:
        """Return concat of 6 per-particle parameters across all particle layers.

        Single-bg-layer mode: short-circuits to direct attribute access (returns
        the bg layer's Parameters themselves, byte-identical to the v1 path).

        Multi-layer mode: returns `torch.cat(..., dim=0)` across layers in
        `self.specs` order (particle layers only).

        T4.3 dynamic-rigid handling: when a particle layer is named
        ``dynamic_rigids``, its `positions` are stored in object-local frame.
        Before concat, they are routed through ``_transform_means`` using the
        current frame's per-track pose (sourced from buffers registered in
        T4.0). When ``frame_id is None`` (e.g. inference), the dynamic layer's
        positions are passed through unchanged with a TODO marker for Stage 8
        "nearest active frame" fallback (D4).
        """
        bg = self._single_bg_layer()
        if bg is not None:
            return {n: getattr(bg, n) for n in _FORWARD_PARAM_NAMES}

        pieces: dict[str, list[torch.Tensor]] = {n: [] for n in _FORWARD_PARAM_NAMES}
        for spec in self.specs:
            if not spec.is_particle_layer:
                continue
            layer = self.layers[spec.name]
            for n in _FORWARD_PARAM_NAMES:
                v = getattr(layer, n)
                if (
                    n == "positions"
                    and spec.name == "dynamic_rigids"
                    and frame_id is not None
                    and hasattr(layer, "track_ids")
                    and len(self.tracks_poses) > 0
                ):
                    v = self._transform_means(v, layer.track_ids, frame_id)
                pieces[n].append(v)
        return {n: torch.cat(pieces[n], dim=0) for n in _FORWARD_PARAM_NAMES}

    # ------------------------------------------------------------------ T4.3 transform_means
    def _transform_means(
        self,
        positions_local: torch.Tensor,
        track_ids: torch.Tensor,
        frame_id: int,
    ) -> torch.Tensor:
        """Apply per-particle ``object → world`` SE(3) using current frame's
        per-track pose.

        Pose source: ``self.tracks_poses`` buffer dict registered in T4.0.
        Particle-to-track routing: ``track_ids`` per-particle int buffer
        registered by ``init_layer_from_points("dynamic_rigids", ..., track_ids=...)``.
        Track-name → int-id mapping: ``sorted(self.tracks_poses.keys())``
        (must match the order used by ``init_dynamic_rigid_layer``).

        Args:
            positions_local: ``[N, 3]`` object-local positions.
            track_ids:       ``[N]`` int64; values in ``[0, len(tracks_poses))``.
            frame_id:        per-clip frame index.

        Returns:
            ``[N, 3]`` world-frame positions.
        """
        track_names = sorted(self.tracks_poses.keys())
        # Stack poses for this frame: [K, 4, 4]
        pose_stack = torch.stack(
            [self.tracks_poses[name][frame_id] for name in track_names]
        )
        pose_per_pt = pose_stack[track_ids]                                  # [N, 4, 4]
        R = pose_per_pt[:, :3, :3]                                            # [N, 3, 3]
        t = pose_per_pt[:, :3, 3]                                             # [N, 3]
        return (R @ positions_local.to(R.dtype).unsqueeze(-1)).squeeze(-1) + t

    # ------------------------------------------------------------------ T3.0 init
    def init_layer_from_points(
        self,
        layer_name: str,
        positions: torch.Tensor,
        *,
        colors: Optional[torch.Tensor] = None,
        rotations: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        densities: Optional[torch.Tensor] = None,
        track_ids: Optional[torch.Tensor] = None,
        observer_pts: Optional[torch.Tensor] = None,
        setup_optimizer: bool = True,
    ) -> None:
        """Initialize one named layer's MoG parameters from a point cloud.

        Spec-aware defaults: when scales / densities / colors / rotations are
        omitted, fall back to ``LayerSpec.scale_prior`` (log-applied) /
        ``LayerSpec.density_init`` / identity quat / neutral gray (0.5).

        Args:
            layer_name: must be in ``self.layers``.
            positions: ``[N, 3]``. World frame for background / road;
                object-local frame for dynamic_rigids (pair with ``track_ids``).
            colors:    ``[N, 3]`` in [0, 1]; default neutral gray.
            rotations: ``[N, 4]`` quat wxyz; default identity ``(1, 0, 0, 0)``.
            scales:    ``[N, 3]`` log-space; default ``torch.log(spec.scale_prior)``.
            densities: ``[N, 1]`` log-space; default ``spec.density_init``.
            track_ids: ``[N]`` int64; only meaningful for dynamic_rigids.
                Registered as a persistent buffer ``track_ids`` on the layer.
            observer_pts: reserved; ignored in T3.0 (full Parameter path,
                no observer-distance scale estimation).
            setup_optimizer: when True (default), wire the per-layer Adam via
                ``layer.set_optimizable_parameters()`` + ``layer.setup_optimizer()``.
                Tests that bypass CUDA conf paths pass False.
        """
        if layer_name not in self.layers:
            raise ValueError(
                f"unknown layer '{layer_name}', enabled = {list(self.layers)}"
            )
        spec = next(s for s in self.specs if s.name == layer_name)
        layer = self.layers[layer_name]
        N = positions.shape[0]
        dtype = torch.float32

        if rotations is None:
            rotations = torch.zeros(N, 4, dtype=dtype)
            rotations[:, 0] = 1.0
        if scales is None:
            s_phys = torch.tensor(list(spec.scale_prior), dtype=dtype)
            scales = torch.log(s_phys).expand(N, 3).clone()
        if densities is None:
            densities = torch.full((N, 1), float(spec.density_init), dtype=dtype)
        if colors is None:
            colors = torch.full((N, 3), 0.5, dtype=dtype)

        features_albedo = (colors.to(dtype=dtype) - 0.5) / _SH_C0
        num_specular_dims = sh_degree_to_specular_dim(layer.max_n_features)
        features_specular = torch.zeros((N, num_specular_dims), dtype=dtype)

        # Tensors keep their incoming device. Caller (Trainer) is responsible
        # for putting them on GPU; tests stay on CPU. Note layer.device is
        # hardcoded to "cuda" in MoG.__init__ but is only consulted by code
        # paths that allocate new tensors (not by the assignments below).
        layer.positions         = nn.Parameter(positions.to(dtype=dtype))
        layer.rotation          = nn.Parameter(rotations.to(dtype=dtype))
        layer.scale             = nn.Parameter(scales.to(dtype=dtype))
        layer.density           = nn.Parameter(densities.to(dtype=dtype))
        layer.features_albedo   = nn.Parameter(features_albedo)
        layer.features_specular = nn.Parameter(features_specular)

        if setup_optimizer:
            layer.set_optimizable_parameters()
            layer.setup_optimizer()
            layer.validate_fields()

        if track_ids is not None:
            layer.register_buffer("track_ids", track_ids.long(), persistent=True)

    # ------------------------------------------------------------------ T3.0 optimizer view
    @property
    def optimizer(self):
        """Drop-in optimizer view across all layers.

        Single-bg mode: return the bg layer's own optimizer directly, so
        ``self.model.optimizer.step()`` from the trainer is byte-identical
        with v1 (no wrapper allocation, no fan-out cost).

        Multi-layer mode: return a ``_LayeredOptimizerView`` that fans
        ``step()`` / ``zero_grad()`` / ``param_groups`` across each layer's
        sub-optimizer.
        """
        bg = self._single_bg_layer()
        if bg is not None and getattr(bg, "optimizer", None) is not None:
            return bg.optimizer
        return _LayeredOptimizerView(self.layers)

    def get_layer_mask(self, name: str) -> torch.Tensor:
        """Return Bool mask of shape [N_total] selecting particles of layer `name`.

        Mask layout follows the same particle-layer concat order as `fused_view`
        (specs order, non-particle layers skipped). Used by per-layer loss /
        region gating downstream.
        """
        particle_layers = [s for s in self.specs if s.is_particle_layer]
        if name not in {s.name for s in particle_layers}:
            raise ValueError(
                f"unknown layer '{name}' (or it is a non-particle layer); "
                f"particle layers: {[s.name for s in particle_layers]}"
            )

        total = 0
        target_start, target_end = -1, -1
        for spec in particle_layers:
            n_local = self.layers[spec.name].num_gaussians
            if spec.name == name:
                target_start, target_end = total, total + n_local
            total += n_local

        device = self.layers[name].positions.device
        mask = torch.zeros(total, dtype=torch.bool, device=device)
        mask[target_start:target_end] = True
        return mask

    def __getattr__(self, name):
        # nn.Module.__getattr__ is only invoked when normal attribute lookup
        # fails (so layers/specs/conf/scene_extent etc. handled by base class
        # first). Forward to bg layer when in single-layer mode.
        try:
            return super().__getattr__(name)
        except AttributeError:
            bg = self._single_bg_layer()
            if bg is not None:
                return getattr(bg, name)
            raise

    def __setattr__(self, name, value):
        # Redirect Parameter writes (e.g. MCMCStrategy.setattr(model, "positions", new))
        # to the single bg layer so its Parameter identity stays in sync with the
        # optimizer state. Non-bridged names fall through to nn.Module.__setattr__.
        if name in _FORWARD_PARAM_NAMES:
            bg = self._single_bg_layer()
            if bg is not None:
                setattr(bg, name, value)
                return
        super().__setattr__(name, value)
