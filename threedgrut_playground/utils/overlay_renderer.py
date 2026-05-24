# SPDX-License-Identifier: Apache-2.0
"""B2: PIL-based 2D polyline overlay renderer for FTheta cuboid/frustum/track
overlays in viser_gui_4d. Pure CPU; no torch, no viser.

Draws projected polylines (returned by ``FthetaForwardProjector.project_polylines``)
into a transparent RGBA buffer that ``Viser4DOverlayCompositor`` then alpha-
blends into the engine's Gaussian backdrop image.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw


RGBAColor = tuple[int, int, int, int]


@dataclass
class OverlayLayer:
    """One semantic layer of polylines (e.g., all cuboid edges, or all
    frustum edges). All polylines in a layer share color + width; layers
    are drawn in registration order so the last-added layer is on top.

    ``polylines`` is the projector output: list of (uv: (M, 2) float,
    visible: (M,) bool). Segments where either endpoint has visible=False
    are skipped.
    """
    name: str
    polylines: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    color: RGBAColor = (0, 255, 0, 255)
    width: int = 1


class OverlayRenderer:
    """Render a sequence of OverlayLayer into an (H, W, 4) uint8 RGBA buffer.

    Buffer starts fully transparent (alpha=0). Each line segment with both
    endpoints ``visible=True`` is drawn with the layer's color + width.
    """

    def __init__(self, height: int, width: int):
        self.height = int(height)
        self.width = int(width)

    def render(self, layers: Sequence[OverlayLayer]) -> np.ndarray:
        """Render layers in registration order; returns (H, W, 4) uint8."""
        img = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        for layer in layers:
            color = layer.color
            width = max(1, int(layer.width))
            for uv, visible in layer.polylines:
                if uv.shape[0] < 2:
                    continue
                # Draw each adjacent (visible, visible) pair as a line segment.
                # Skipping the polyline ends where visibility drops avoids
                # spurious diagonal lines crossing through occluded regions.
                u = uv[:, 0]
                v = uv[:, 1]
                vis = visible
                for i in range(len(uv) - 1):
                    if not (vis[i] and vis[i + 1]):
                        continue
                    draw.line(
                        [(float(u[i]),     float(v[i])),
                         (float(u[i + 1]), float(v[i + 1]))],
                        fill=color,
                        width=width,
                    )
        return np.asarray(img, dtype=np.uint8)


def alpha_blend(backdrop_rgb: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    """Standard premultiplied-style alpha compositing of an RGBA overlay
    on top of an RGB backdrop.

    Args:
        backdrop_rgb:   (H, W, 3) uint8.
        overlay_rgba:   (H, W, 4) uint8.

    Returns:
        (H, W, 3) uint8 blended image.

    No-op fast path when overlay is fully transparent (max alpha == 0).
    """
    if backdrop_rgb.shape[:2] != overlay_rgba.shape[:2]:
        raise ValueError(
            f"shape mismatch: backdrop={backdrop_rgb.shape} "
            f"overlay={overlay_rgba.shape}"
        )
    if backdrop_rgb.dtype != np.uint8 or overlay_rgba.dtype != np.uint8:
        raise ValueError(
            f"dtype must be uint8: backdrop={backdrop_rgb.dtype} "
            f"overlay={overlay_rgba.dtype}"
        )
    if overlay_rgba[..., 3].max() == 0:
        return backdrop_rgb  # nothing to blend

    a = overlay_rgba[..., 3:4].astype(np.float32) / 255.0
    blended = (
        overlay_rgba[..., :3].astype(np.float32) * a
        + backdrop_rgb.astype(np.float32) * (1.0 - a)
    )
    return blended.astype(np.uint8)
