# SPDX-License-Identifier: Apache-2.0
"""Static ego-mask construction from visual polygon / fisheye-circle specs
(Phase C P0.3).

The ego vehicle is static w.r.t. each camera (mirrors do not fold), so one
static mask per camera can be reused for every frame. This module turns
Claude-annotated polygon vertices (+ optional fisheye imaging-circle) into a
``(H, W)`` bool ego mask (True = ego / invalid pixel), and composes a full
per-camera mask set — optionally *reinforcing* an existing nre-tools itar mask
(union) so pixel-accurate vehicle body is kept and only omissions (mirrors)
are added by hand.

Coordinate convention: polygon vertices and the fisheye center are ``(x, y) =
(col, row)``; returned masks are indexed ``mask[row, col]``. Pure numpy + PIL —
no ncore SDK, so unit-testable on Mac CPU.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

Point = Tuple[float, float]


def rasterize_polygons(polygons: Sequence[Sequence[Point]], hw: Tuple[int, int]) -> np.ndarray:
    """Fill each polygon (list of ``(x, y)`` vertices) and return their union.

    Returns a ``(H, W)`` bool array; empty ``polygons`` -> all False. Filling
    uses ``PIL.ImageDraw.polygon`` (interior + boundary).
    """
    H, W = int(hw[0]), int(hw[1])
    if not polygons:
        return np.zeros((H, W), dtype=bool)
    img = Image.new("1", (W, H), 0)
    draw = ImageDraw.Draw(img)
    for poly in polygons:
        pts = [(int(round(x)), int(round(y))) for x, y in poly]
        draw.polygon(pts, fill=1)
    return np.asarray(img, dtype=bool)


def rasterize_fisheye_outer(center_xy: Point, radius: float, hw: Tuple[int, int]) -> np.ndarray:
    """Return ``(H, W)`` bool with True *outside* the imaging circle.

    The fisheye black vignette (outside the circle) carries no image content
    and is masked like ego. Points with ``dist <= radius`` (inside / on the
    circle) are False.
    """
    H, W = int(hw[0]), int(hw[1])
    cx, cy = float(center_xy[0]), float(center_xy[1])
    yy, xx = np.ogrid[:H, :W]
    return (xx - cx) ** 2 + (yy - cy) ** 2 > float(radius) ** 2


def build_camera_mask(
    hw: Tuple[int, int],
    polygons: Optional[Sequence[Sequence[Point]]] = None,
    fisheye_circle: Optional[Tuple[float, float, float]] = None,
    base_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Union of ``base_mask`` (reinforce), rasterized ``polygons`` and the
    fisheye outer region into one ``(H, W)`` bool ego mask.

    Any of the three sources may be None/empty and is then skipped;
    ``fisheye_circle`` is ``(cx, cy, radius)``. All-None -> all False.
    """
    H, W = int(hw[0]), int(hw[1])
    mask = np.zeros((H, W), dtype=bool)
    if base_mask is not None:
        mask |= np.asarray(base_mask, dtype=bool)
    if polygons:
        mask |= rasterize_polygons(polygons, (H, W))
    if fisheye_circle is not None:
        cx, cy, r = fisheye_circle
        mask |= rasterize_fisheye_outer((cx, cy), r, (H, W))
    return mask
