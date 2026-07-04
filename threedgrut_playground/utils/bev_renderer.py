# SPDX-License-Identifier: Apache-2.0
"""BEV (bird's-eye view) renderer for V3-VIZ.1 diagnostic.

Pure numpy + matplotlib (no torch / no GPU / no viser). Renders a single frame:
ego trajectory + current ego marker + cuboid footprints (with labels) + per-layer
Gaussian center scatter, all in BEV / world-XY plane.

Used by both:
  - ``scripts/diagnose_layered_bev.py`` (CLI per-frame PNG output)
  - ``threedgrut_playground/viser_gui_4d.py`` (V3-VIZ.4 embedded BEV panel,
    not yet wired up — V3-VIZ.4 work)

Color map (from cuboid.class_color + per-layer convention):
  background       : gray  #888888  alpha 0.15  size 0.5
  road             : blue  #3399FF  alpha 0.30  size 0.5
  dynamic_rigids   : red   #FF3333  alpha 0.50  size 0.8
  dynamic_deformables: yellow #FFCC00 alpha 0.50 size 0.8
  sky_envmap       : not drawn (no particles)
  cuboid outline   : per-class (automobile=blue, heavy_truck=orange,
                                bus=purple, unknown=gray) — see cuboid.class_color
  ego trajectory   : green #00B050
  ego current      : red cross + heading triangle
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

# Layer style. Tuple = (color_hex, alpha, marker_size_px)
_LAYER_STYLE: dict[str, tuple[str, float, float]] = {
    "background": ("#888888", 0.15, 0.5),
    "road": ("#3399FF", 0.30, 0.5),
    "dynamic_rigids": ("#FF3333", 0.50, 0.8),
    "dynamic_deformables": ("#FFCC00", 0.50, 0.8),
}

# Cuboid class outline colors (matches threedgrut_playground.utils.cuboid.class_color).
_CLASS_COLOR: dict[str, str] = {
    "automobile": "#1A99FF",  # blue
    "heavy_truck": "#FF8019",  # orange
    "bus": "#CC4DD9",  # purple
    "unknown": "#A6A6A6",  # gray
}


@dataclass
class BEVRenderInputs:
    """Pre-extracted data for one BEV frame. Constructed by the CLI driver."""

    ego_xy_trajectory: np.ndarray  # (N, 2) float32 — full ego XY polyline
    ego_current_xy: np.ndarray  # (2,) float32 — current ego position
    ego_current_heading_xy: np.ndarray  # (2,) float32 — unit heading vector
    cuboids: list[dict]  # [{tid, class, footprint_xy (4,2), center_xy}]
    layer_positions_xy: dict[str, np.ndarray]  # layer_name → (M, 2) float32 XY


def _ego_heading_from_c2w(c2w: np.ndarray) -> np.ndarray:
    """Extract a unit XY heading vector from a 4x4 ego c2w pose.

    NCore ego/camera convention: camera looks down its +Z axis (OpenCV / RDF).
    We project Z onto XY and normalize. Returns (0, 1) when Z is nearly vertical.
    """
    z_world = c2w[:3, 2]
    heading = np.array([z_world[0], z_world[1]], dtype=np.float32)
    n = float(np.linalg.norm(heading))
    if n < 1e-6:
        return np.array([0.0, 1.0], dtype=np.float32)
    return heading / n


def _cuboid_footprint_xy(pose: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Project a cuboid bottom face to world XY. Returns (4, 2) polygon.

    Uses the 4 bottom corners (z = -size_z/2 in object-local) and applies
    the SE(3) pose. Yaw is captured by R; small roll/pitch are projected away.
    """
    sx, sy, sz = (float(s) for s in size.reshape(3))
    # Bottom face corners in object-local frame (CCW from -x,-y).
    local_corners = np.array(
        [
            [-sx / 2, -sy / 2, -sz / 2],
            [sx / 2, -sy / 2, -sz / 2],
            [sx / 2, sy / 2, -sz / 2],
            [-sx / 2, sy / 2, -sz / 2],
        ],
        dtype=np.float32,
    )
    R = pose[:3, :3]
    t = pose[:3, 3]
    world = local_corners @ R.T + t  # (4, 3)
    return world[:, :2].astype(np.float32)


def _cuboid_center_top_xy(pose: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Project the cuboid top-center to world XY (for label position)."""
    sz = float(size[2])
    local = np.array([0.0, 0.0, sz / 2], dtype=np.float32)
    world = pose[:3, :3] @ local + pose[:3, 3]
    return world[:2].astype(np.float32)


def build_inputs_from_metadata(
    meta,  # FourDMetadata
    layer_positions: Mapping[str, np.ndarray],  # world-frame XY positions per layer
    frame_idx: int,
    *,
    z_window_m: float = 10.0,
) -> BEVRenderInputs:
    """Pull all per-frame BEV inputs out of FourDMetadata + layer positions.

    ``layer_positions`` are full world-frame (N, 3) per-particle positions; this
    helper:
      * projects to (M, 2) XY
      * filters by |z - ego_z| <= ``z_window_m`` to drop sky / sub-ground points
      * skips ``sky_envmap`` (no particles in v2)

    For ``dynamic_rigids`` the caller must transform the object-local positions
    to world-frame at ``frame_idx`` before passing in (see
    ``diag_ckpt.dyn_local_to_world_at_frame``).
    """
    ego_poses = meta.ego_poses_c2w
    if ego_poses.shape[0] == 0:
        ego_xy_traj = np.empty((0, 2), dtype=np.float32)
        ego_current = np.zeros(2, dtype=np.float32)
        ego_heading = np.array([0.0, 1.0], dtype=np.float32)
        ego_z = 0.0
    else:
        ego_xy_traj = ego_poses[:, :2, 3].astype(np.float32)
        clamped_idx = max(0, min(int(frame_idx), ego_poses.shape[0] - 1))
        ego_current = ego_poses[clamped_idx, :2, 3].astype(np.float32)
        ego_heading = _ego_heading_from_c2w(ego_poses[clamped_idx])
        ego_z = float(ego_poses[clamped_idx, 2, 3])

    # Cuboids at this frame.
    cuboids: list[dict] = []
    for tid in meta.active_tracks_at(frame_idx):
        track = meta.tracks[tid]
        pose = track["poses"][frame_idx]
        size = track["size"]
        footprint = _cuboid_footprint_xy(pose, size)
        cls_name = str(track.get("class", "unknown"))
        cuboids.append(
            {
                "tid": str(tid),
                "class": cls_name,
                "footprint_xy": footprint,  # (4, 2)
                "label_xy": _cuboid_center_top_xy(pose, size),  # (2,)
            }
        )

    # Layer positions: filter by Z window, project to XY.
    layer_xy: dict[str, np.ndarray] = {}
    for name, pos in layer_positions.items():
        if name == "sky_envmap" or pos is None or pos.size == 0:
            continue
        if name not in _LAYER_STYLE:
            continue
        z = pos[:, 2]
        z_mask = np.abs(z - ego_z) <= z_window_m
        layer_xy[name] = pos[z_mask, :2].astype(np.float32)

    return BEVRenderInputs(
        ego_xy_trajectory=ego_xy_traj,
        ego_current_xy=ego_current,
        ego_current_heading_xy=ego_heading,
        cuboids=cuboids,
        layer_positions_xy=layer_xy,
    )


def render_bev_frame(
    inputs: BEVRenderInputs,
    *,
    xy_range_m: float = 60.0,
    grid_step_m: float = 10.0,
    figsize: tuple[float, float] = (10.0, 10.0),
    dpi: int = 100,
    show_labels: bool = True,
    title: Optional[str] = None,
    backdrop_rgb: Optional[np.ndarray] = None,
    backdrop_xy_extent: Optional[tuple[float, float, float, float]] = None,
) -> np.ndarray:
    """Render a single BEV frame to an (H, W, 3) uint8 array.

    matplotlib is imported lazily so the renderer module remains importable
    even when matplotlib is not installed (e.g. minimal CI environments that
    only run schema unit tests).

    Args:
        backdrop_rgb: optional ``(H, W, 3)`` uint8 image drawn under all
            overlays — typically a 5-camera IPM BEV stitch from V3-VIZ.1b.
        backdrop_xy_extent: ``(xmin, xmax, ymin, ymax)`` world-frame extent of
            ``backdrop_rgb``. Required when ``backdrop_rgb`` is given.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless safe
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrow, Polygon

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cx, cy = float(inputs.ego_current_xy[0]), float(inputs.ego_current_xy[1])

    if backdrop_rgb is not None and backdrop_xy_extent is not None:
        # imshow uses image-y flipped (origin='upper') by default; we want
        # +Y up so the BEV is right-handed.
        ax.imshow(backdrop_rgb, extent=backdrop_xy_extent, origin="lower", interpolation="nearest", zorder=0)
        ax.set_facecolor("white")
    else:
        # Grid (light gray, every grid_step_m).
        ax.set_xticks(np.arange(cx - xy_range_m, cx + xy_range_m + 1, grid_step_m))
        ax.set_yticks(np.arange(cy - xy_range_m, cy + xy_range_m + 1, grid_step_m))
        ax.grid(True, color="#E0E0E0", linewidth=0.5, zorder=0)
        ax.set_facecolor("white")

    # Layer scatter (drawn first so cuboid outlines + ego sit on top).
    layer_draw_order = ["background", "road", "dynamic_rigids", "dynamic_deformables"]
    for name in layer_draw_order:
        xy = inputs.layer_positions_xy.get(name)
        if xy is None or xy.size == 0:
            continue
        color, alpha, s = _LAYER_STYLE[name]
        ax.scatter(
            xy[:, 0], xy[:, 1], s=s, c=color, alpha=alpha, edgecolors="none", zorder=2, label=f"{name} ({xy.shape[0]})"
        )

    # Ego full trajectory (green line).
    traj = inputs.ego_xy_trajectory
    if traj.shape[0] >= 2:
        ax.plot(traj[:, 0], traj[:, 1], color="#00B050", linewidth=1.5, alpha=0.9, zorder=4, label="ego trajectory")

    # Cuboid footprints (polygons + labels).
    for cu in inputs.cuboids:
        color = _CLASS_COLOR.get(cu["class"], _CLASS_COLOR["unknown"])
        poly = Polygon(cu["footprint_xy"], closed=True, edgecolor=color, facecolor="none", linewidth=1.8, zorder=5)
        ax.add_patch(poly)
        if show_labels:
            lx, ly = float(cu["label_xy"][0]), float(cu["label_xy"][1])
            ax.text(
                lx,
                ly,
                f"t{cu['tid']} | {cu['class']}",
                fontsize=6,
                color=color,
                ha="center",
                va="bottom",
                zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.5),
            )

    # Ego current + heading.
    ax.plot(cx, cy, marker="x", color="#E60000", markersize=10, markeredgewidth=2.5, zorder=7, label="ego (current)")
    hx, hy = inputs.ego_current_heading_xy
    arrow_len = max(2.0, xy_range_m * 0.04)
    ax.add_patch(
        FancyArrow(
            cx,
            cy,
            float(hx) * arrow_len,
            float(hy) * arrow_len,
            width=arrow_len * 0.15,
            head_width=arrow_len * 0.45,
            head_length=arrow_len * 0.45,
            color="#E60000",
            length_includes_head=True,
            zorder=7,
        )
    )

    ax.set_xlim(cx - xy_range_m, cx + xy_range_m)
    ax.set_ylim(cy - xy_range_m, cy + xy_range_m)
    ax.set_aspect("equal")
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

    # Scale annotation (bottom-right).
    ax.text(
        0.98,
        0.02,
        f"{int(xy_range_m * 2)} m × {int(xy_range_m * 2)} m",
        transform=ax.transAxes,
        fontsize=7,
        color="#404040",
        ha="right",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="#C0C0C0", pad=2.0),
    )

    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)
    return img
