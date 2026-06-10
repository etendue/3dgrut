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
    will be subdivided (per-layer ``subdivide_n``, default 20) before
    projection so fisheye-curved edges render with smooth tangents instead
    of straight chord shortcuts.

    Use a high ``subdivide_n`` (e.g. 20) for *short* 2-vertex edges where
    the fisheye-induced curvature between endpoints is large (cuboid edges).
    Use a low value (e.g. 2-3) for already-dense multi-vertex polylines
    (track trajectories, ego trajectory) where each segment is short and
    the per-frame subdivision cost dominates.

    ``labels_world`` (BUG-1b) is a list of (anchor_xyz (3,), text) world-space
    text labels. Anchors are projected through the SAME FTheta polynomial as
    the polylines; invisible anchors (behind camera / outside FOV / out of
    bounds) are dropped. Rendered in the layer color with a black stroke.
    """
    name: str
    polylines_world: list[np.ndarray] = field(default_factory=list)
    color: RGBAColor = (0, 255, 0, 255)
    width: int = 1
    subdivide_n: int = 20
    labels_world: list[tuple[np.ndarray, str]] = field(default_factory=list)
    font_size: int = 18


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
        world_to_camera_flip: "np.ndarray | None" = None,
    ):
        """``world_to_camera_flip`` is forwarded to FthetaForwardProjector.

        None keeps the projector's legacy default (FLIP_VISER_TO_OPENCV).
        BUG-1 (2026-06-10): the viewer must pass ``np.eye(4)`` — the c2w it
        feeds composite() is the SAME matrix the engine renders the backdrop
        with, whose viewing direction is the +Z column (FTheta rays have
        rz=cos(theta)>0, see ftheta_intrinsics.ftheta_pixels_to_camera_rays).
        The legacy Z-flip pointed the overlay camera 180° away from the
        backdrop camera, so wireframes were mirror-projections of the tracks
        BEHIND the ego — plausibly placed on a fore-aft symmetric street,
        which is how the original B2 probe mis-calibrated it. Verified on
        inceptio: bus track 405 (12 m ahead, visible in the backdrop)
        projects to its rendered position with flip=I and is fully invisible
        (0/8 corners) with the legacy flip; GT raw-camera validation
        (validate_cuboid_7cam, flip=I) hugs the real vehicles.
        """
        self.projector = FthetaForwardProjector(
            ftheta_dict, world_to_camera_flip=world_to_camera_flip)
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

        total_pl = sum(len(L.polylines_world) + len(L.labels_world)
                       for L in layers_world)
        if total_pl == 0:
            return backdrop_rgb  # nothing to draw, skip projection + blend

        render_layers: list[OverlayLayer] = []
        for spec in layers_world:
            if not spec.polylines_world and not spec.labels_world:
                continue
            # Spec's per-layer subdivide_n overrides the compositor default
            # so callers can tune cuboid (high) vs trajectory (low) separately.
            n_sub = spec.subdivide_n if spec.subdivide_n else self.subdivide_n
            projected = (self.projector.project_polylines(
                spec.polylines_world, c2w_viser, subdivide_n=n_sub)
                if spec.polylines_world else [])
            # BUG-1b: labels share the projector with the wireframe, so text
            # and box can never separate again. Invisible anchors dropped.
            texts: list[tuple[float, float, str]] = []
            if spec.labels_world:
                anchors = np.stack([np.asarray(a, dtype=np.float64)
                                    for a, _ in spec.labels_world])
                uv, vis = self.projector.project_points(anchors, c2w_viser)
                for (_, text), (u, v), ok in zip(spec.labels_world, uv, vis):
                    if bool(ok):
                        texts.append((float(u), float(v), str(text)))
            render_layers.append(OverlayLayer(
                name=spec.name,
                polylines=projected,
                color=spec.color,
                width=spec.width,
                texts=texts,
                font_size=spec.font_size,
            ))

        overlay_rgba = self.renderer.render(render_layers)
        return alpha_blend(backdrop_rgb, overlay_rgba)
