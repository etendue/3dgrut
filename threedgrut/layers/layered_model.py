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

from typing import List

import torch
import torch.nn as nn

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.logger import logger


# Per-particle parameter names forwarded to the single background layer when in
# single-layer mode (so MCMCStrategy's setattr(model, name, new_param) writes
# to bg layer's Parameter rather than registering one on LayeredGaussians).
_FORWARD_PARAM_NAMES = frozenset({
    "positions", "rotation", "scale", "density",
    "features_albedo", "features_specular",
})


class LayeredGaussians(nn.Module):
    """Drop-in replacement for MixtureOfGaussians when use_layered_model=True."""

    def __init__(self, conf, specs: List[LayerSpec], scene_extent: float):
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
            raise ValueError(
                "v1 checkpoint detected (flat schema) but no 'background' layer "
                "configured. Add 'background' to layers.enabled."
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

    def _single_bg_layer(self):
        """Return the bg layer iff this is a single-bg-layer setup, else None."""
        modules = self.__dict__.get("_modules", {})
        layers = modules.get("layers")
        if layers is not None and len(layers) == 1 and "background" in layers:
            return layers["background"]
        return None

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
