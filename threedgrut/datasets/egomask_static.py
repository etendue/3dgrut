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

import warnings
from typing import Dict, Optional, Sequence, Set, Tuple

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


def compose_egomask_set(
    visual_specs: Dict[str, dict],
    existing_reader,
    hw: Tuple[int, int],
    reinforce_cams: Set[str],
    skip_cams: Set[str],
) -> Dict[str, np.ndarray]:
    """Compose the per-camera static ego-mask set to write into the itar.

    Args:
        visual_specs: ``{camera_id: {"polygons": [[(x,y),...], ...],
            "fisheye_circle": (cx, cy, r) or None}}`` — Claude-annotated shapes.
        existing_reader: an ``EgomaskAuxReader`` (needs ``has_camera`` /
            ``read_static_mask``) supplying the base mask for reinforced cameras.
        hw: ``(H, W)`` of the target masks.
        reinforce_cams: cameras whose existing itar mask is unioned with the
            visual polygons (pixel-accurate body kept, omissions added). Each
            MUST be present in ``existing_reader`` or a ``KeyError`` is raised.
        skip_cams: cameras that get NO mask (absent from the returned dict, so
            no itar entry -> ``resolve_ego_valid_mask`` returns all-valid).

    Returns:
        ``{camera_id: (H, W) bool}`` for every camera in ``visual_specs`` except
        those in ``skip_cams``.
    """
    result: Dict[str, np.ndarray] = {}
    for cam, spec in visual_specs.items():
        if cam in skip_cams:
            warnings.warn(f"compose_egomask_set: '{cam}' in skip_cams — ignoring its visual spec")
            continue
        base = None
        if cam in reinforce_cams:
            if not existing_reader.has_camera(cam):
                raise KeyError(
                    f"compose_egomask_set: reinforce camera '{cam}' not present in existing egomask reader"
                )
            base = existing_reader.read_static_mask(cam)
        result[cam] = build_camera_mask(
            hw,
            polygons=spec.get("polygons"),
            fisheye_circle=spec.get("fisheye_circle"),
            base_mask=base,
        )
    return result
