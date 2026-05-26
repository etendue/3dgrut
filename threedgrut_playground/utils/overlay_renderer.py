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
        """Render layers in registration order; returns (H, W, 4) uint8.

        Perf: ``draw.line`` accepts a list of points and draws one
        connected polyline in a single C-level call (10-50× faster than
        looping in Python and calling ``draw.line`` per segment). We
        split the input polyline into maximal contiguous visible runs
        and emit one batched ``draw.line`` per run.
        """
        img = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        for layer in layers:
            color = layer.color
            width = max(1, int(layer.width))
            for uv, visible in layer.polylines:
                M = uv.shape[0]
                if M < 2:
                    continue
                # Walk the polyline, collecting maximal contiguous visible
                # runs (≥ 2 verts each), then draw each run in one call.
                run_start = None
                for i in range(M):
                    if visible[i]:
                        if run_start is None:
                            run_start = i
                    else:
                        if run_start is not None and i - run_start >= 2:
                            _draw_run(draw, uv, run_start, i, color, width)
                        run_start = None
                if run_start is not None and M - run_start >= 2:
                    _draw_run(draw, uv, run_start, M, color, width)
        return np.asarray(img, dtype=np.uint8)


def _draw_run(draw, uv, i0, i1, color, width):
    """Emit one batched draw.line for the slice uv[i0:i1] (i1 exclusive)."""
    # PIL accepts a flat list of (x, y) tuples for a connected polyline.
    # Convert once; ndarray slicing + tolist is fast even for ~10k verts.
    pts = uv[i0:i1].tolist()
    draw.line([(p[0], p[1]) for p in pts], fill=color, width=width)


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
