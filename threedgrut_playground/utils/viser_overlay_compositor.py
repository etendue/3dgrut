# SPDX-License-Identifier: Apache-2.0
"""B2: Viser4DOverlayCompositor — glue between FTheta forward projection,
PIL overlay rendering, and the viser_gui_4d backdrop image.

The compositor is the *only* point of contact between the overlay path and
the viser viewer. It knows nothing about viser itself: it takes a backdrop
ndarray + world-space polylines + the active client's c2w, and returns a
blended ndarray ready for ``client.scene.set_background_image``.

This isolation lets the entire FTheta overlay path be unit-tested on Mac
without a running viser server (see test_viser_4d_ftheta_overlay_integration).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .ftheta_projector import FthetaForwardProjector
from .overlay_renderer import OverlayLayer, OverlayRenderer, RGBAColor, alpha_blend


@dataclass
class PolylineLayerSpec:
    """One semantic layer of world-space polylines, before projection.

    ``polylines_world`` is a list of (M, 3) float ndarrays. Each polyline
    will be subdivided (default n=20) before projection so fisheye-curved
    edges render with smooth tangents instead of straight chord shortcuts.
    """
    name: str
    polylines_world: list[np.ndarray] = field(default_factory=list)
    color: RGBAColor = (0, 255, 0, 255)
    width: int = 1


class Viser4DOverlayCompositor:
    """Build (project → render → blend) chain for one or more FTheta overlay
    layers. Reused across clients and frames; stateless apart from the
    cached projector + renderer (both depend only on ftheta_dict + resolution).

    Layers are drawn in registration order (last layer = topmost). Typical
    order for B2: ego_trajectory → tracks → frustum → cuboids (cuboids on top).
    """

    def __init__(
        self,
        ftheta_dict: dict,
        height: int,
        width: int,
        subdivide_n: int = 20,
    ):
        self.projector = FthetaForwardProjector(ftheta_dict)
        self.renderer = OverlayRenderer(height=height, width=width)
        self.subdivide_n = int(subdivide_n)
        self.height = int(height)
        self.width = int(width)

    def composite(
        self,
        backdrop_rgb: np.ndarray,         # (H, W, 3) uint8 from engine
        layers_world: Sequence[PolylineLayerSpec],
        c2w_viser: np.ndarray,            # (4, 4) viser client camera
    ) -> np.ndarray:
        """Project each layer, render to RGBA, alpha-blend onto backdrop.

        Returns: (H, W, 3) uint8.

        No-op fast path when ``layers_world`` is empty or every layer has
        zero polylines.
        """
        if backdrop_rgb.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"backdrop shape {backdrop_rgb.shape[:2]} doesn't match "
                f"compositor resolution ({self.height}, {self.width}); "
                f"check ftheta_render_wh / engine output."
            )

        total_pl = sum(len(L.polylines_world) for L in layers_world)
        if total_pl == 0:
            return backdrop_rgb  # nothing to draw, skip projection + blend

        render_layers: list[OverlayLayer] = []
        for spec in layers_world:
            if not spec.polylines_world:
                continue
            projected = self.projector.project_polylines(
                spec.polylines_world, c2w_viser, subdivide_n=self.subdivide_n)
            render_layers.append(OverlayLayer(
                name=spec.name,
                polylines=projected,
                color=spec.color,
                width=spec.width,
            ))

        overlay_rgba = self.renderer.render(render_layers)
        return alpha_blend(backdrop_rgb, overlay_rgba)
