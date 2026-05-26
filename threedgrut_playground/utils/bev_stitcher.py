# SPDX-License-Identifier: Apache-2.0
"""V3-VIZ.1b — BEV stitcher: project 5-cam raw images onto z=ground plane.

Builds a top-down BEV mosaic by inverse-perspective-mapping (IPM) each
camera's raw image onto a world-frame ground plane at z = ego_z. Uses the
existing FthetaForwardProjector so the fisheye distortion is handled
identically to the training-time / playground projection paths.

Per-camera coverage is angle-based: each BEV cell is assigned to whichever
camera's principal direction (azimuth in vehicle frame) is closest. This
matches the rig layout (front_wide front, cross_left ~90° left, etc.) and
keeps stitching cheap (no per-frame photometric blending).

Pure numpy + cv2 + FthetaForwardProjector — no torch / no GPU. NCore SDK
required only by the *caller* (to fetch raw image arrays + per-frame c2w);
the stitcher itself takes pre-extracted arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from .ftheta_projector import FthetaForwardProjector


@dataclass
class CameraRig:
    """Per-camera intrinsics + nominal vehicle-frame azimuth (for coverage assignment).

    ``azimuth_deg`` is the principal-axis yaw in vehicle frame (forward = 0°,
    left = +90°, right = -90°, rear = 180°). Used to assign BEV cells to the
    "closest-looking" camera. Approximated from camera_id (front/rear/cross_*)
    when not provided explicitly.
    """
    camera_id: str
    ftheta_dict: dict
    azimuth_deg: float
    image_hw: tuple[int, int]   # (H, W) of raw image — for clipping pixel coords


_DEFAULT_AZIMUTH: dict[str, float] = {
    "camera_front_wide_120fov":  0.0,
    "camera_front_wide":         0.0,
    "camera_front":              0.0,
    "camera_front_tele":         0.0,
    "camera_front_tele_30fov":   0.0,
    "camera_cross_left_120fov":  90.0,
    "camera_cross_left":         90.0,
    "camera_cross_right_120fov": -90.0,
    "camera_cross_right":        -90.0,
    "camera_rear_120fov":        180.0,
    "camera_rear":               180.0,
    "camera_rear_tele":          180.0,
    "camera_rear_tele_30fov":    180.0,
    "camera_rear_left_70fov":    135.0,
    "camera_rear_right_70fov":   -135.0,
}


def default_azimuth(camera_id: str) -> float:
    """Best-effort azimuth lookup; returns 0.0 (forward) for unknown ids."""
    if camera_id in _DEFAULT_AZIMUTH:
        return _DEFAULT_AZIMUTH[camera_id]
    cid = camera_id.lower()
    if "front" in cid:
        return 0.0
    if "rear" in cid:
        return 180.0
    if "left" in cid:
        return 90.0
    if "right" in cid:
        return -90.0
    return 0.0


class BEVStitcher:
    """Pre-compute BEV grid + per-camera projectors; stitch one frame at a time.

    Convention:
      * BEV grid in world XY plane at z = ego_z (passed per-frame).
      * Output image: BEV_H × BEV_W RGB, with axes oriented so +X (world) maps
        to image column (left → right), +Y (world) maps to image row (bottom
        → top). Image origin at (xmin, ymin) of the world-frame square.
      * Coverage: each cell labeled with the index into ``rigs`` that
        contributed its color (255 = uncovered).
    """

    def __init__(
        self,
        rigs: list[CameraRig],
        *,
        bev_xy_range_m: float = 30.0,
        bev_resolution_m_per_px: float = 0.10,
    ):
        if len(rigs) == 0:
            raise ValueError("BEVStitcher needs at least one camera rig.")
        self.rigs = rigs
        self.xy_range_m = float(bev_xy_range_m)
        self.res_mpp = float(bev_resolution_m_per_px)

        # Output image dims: edge length = 2 * xy_range_m / res_mpp
        side = int(round(2.0 * self.xy_range_m / self.res_mpp))
        if side <= 0:
            raise ValueError(
                f"bev_xy_range_m / bev_resolution_m_per_px → {side} pixels; "
                "increase range or decrease resolution.")
        self.bev_w = side
        self.bev_h = side

        # Per-camera FTheta projector — projects (N, 3) world-frame XYZ to
        # raw-image pixels. c2w from NCoreDataset is OpenCV convention
        # (+Y down, +Z forward), so we skip the viser→opencv flip by passing
        # identity. See ftheta_projector.py:28-33 for the documented flag.
        self.projectors: dict[str, FthetaForwardProjector] = {}
        identity_flip = np.eye(4, dtype=np.float64)
        for rig in rigs:
            self.projectors[rig.camera_id] = FthetaForwardProjector(
                rig.ftheta_dict, world_to_camera_flip=identity_flip,
            )

        # Pre-build the BEV grid offset template (vehicle-centered, in m).
        # ``offsets[i, j] = (dx, dy)`` where (dx, dy) is the world-frame offset
        # from ego center for pixel (i_row, j_col). +Y up convention.
        ys = np.linspace(-self.xy_range_m, self.xy_range_m, side, dtype=np.float64)
        xs = np.linspace(-self.xy_range_m, self.xy_range_m, side, dtype=np.float64)
        gx, gy = np.meshgrid(xs, ys)
        # Image row 0 should map to ymin (bottom). gy is already increasing
        # along row axis so flip if needed when assembling final image —
        # we leave gy as-is here and flip in stitch_frame so the final
        # PNG is upright (matches matplotlib origin='lower').
        self._grid_offsets = np.stack([gx, gy], axis=-1)        # (H, W, 2)

        # Precompute per-cell azimuth in vehicle frame and the camera index
        # whose nominal azimuth is closest (used for stitching coverage).
        cell_az_deg = np.degrees(np.arctan2(gy, gx))            # (H, W)
        camera_az = np.array([r.azimuth_deg for r in rigs], dtype=np.float64)
        # Angular distance, wrapped to [0, 180].
        d = np.abs(cell_az_deg[..., None] - camera_az[None, None, :])
        d = np.minimum(d, 360.0 - d)
        self._best_camera_idx = np.argmin(d, axis=-1).astype(np.int16)   # (H, W)

    def stitch_frame(
        self,
        camera_c2w: Mapping[str, np.ndarray],   # cam_id → (4, 4) world c2w
        camera_images: Mapping[str, np.ndarray],  # cam_id → (H_img, W_img, 3) uint8
        ego_xy_world: np.ndarray,               # (2,) — BEV center
        ego_z_world: float,                     # ground plane z (world)
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stitch one BEV frame.

        Returns:
            stitched_rgb: ``(H_bev, W_bev, 3)`` uint8 in row-major (image row 0
                = bottom of world-Y axis, so origin='lower' in matplotlib).
            coverage_mask: ``(H_bev, W_bev)`` uint8 — camera rig index per
                cell (255 = uncovered).
        """
        H_bev, W_bev = self.bev_h, self.bev_w
        rgb = np.zeros((H_bev, W_bev, 3), dtype=np.uint8)
        coverage = np.full((H_bev, W_bev), 255, dtype=np.uint8)

        # World-frame grid (H_bev, W_bev, 3) of (xw, yw, ego_z).
        gx = self._grid_offsets[..., 0] + float(ego_xy_world[0])
        gy = self._grid_offsets[..., 1] + float(ego_xy_world[1])
        gz = np.full_like(gx, ego_z_world, dtype=np.float64)
        world_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)  # (N, 3)

        for rig_i, rig in enumerate(self.rigs):
            cid = rig.camera_id
            c2w = camera_c2w.get(cid)
            img = camera_images.get(cid)
            if c2w is None or img is None:
                continue
            projector = self.projectors[cid]

            # Cells assigned to this camera by nominal azimuth.
            cell_mask = (self._best_camera_idx == rig_i).ravel()
            if not cell_mask.any():
                continue
            sel_pts = world_pts[cell_mask]                  # (M, 3)

            uv, visible = projector.project_points(sel_pts, c2w)
            # Filter visible cells; uncovered cells remain black.
            if not visible.any():
                continue

            uv_v = uv[visible]                              # (K, 2) — float pixels
            H_img, W_img = rig.image_hw

            # Bilinear sample directly via per-corner index (cv2.remap caps
            # each map dim at SHRT_MAX which a single-row K>32k map exceeds).
            u = np.clip(uv_v[:, 0], 0, W_img - 1.0001)
            v = np.clip(uv_v[:, 1], 0, H_img - 1.0001)
            u0 = u.astype(np.int32); u1 = u0 + 1
            v0 = v.astype(np.int32); v1 = v0 + 1
            du = (u - u0).astype(np.float32)[:, None]
            dv = (v - v0).astype(np.float32)[:, None]
            img_f = img.astype(np.float32)
            c00 = img_f[v0, u0]
            c01 = img_f[v0, u1]
            c10 = img_f[v1, u0]
            c11 = img_f[v1, u1]
            sampled = (
                (1 - du) * (1 - dv) * c00
                + du * (1 - dv) * c01
                + (1 - du) * dv * c10
                + du * dv * c11
            ).astype(np.uint8)                              # (K, 3)

            # Scatter back into the flat output buffer.
            sel_idx = np.nonzero(cell_mask)[0]
            target_idx = sel_idx[visible]
            rgb.reshape(-1, 3)[target_idx] = sampled
            coverage.ravel()[target_idx] = np.uint8(rig_i)

        # Flip vertically so image row 0 is the bottom of world-Y (matches
        # imshow(origin='lower') in matplotlib).
        return rgb[::-1].copy(), coverage[::-1].copy()

    def world_xy_extent(self, ego_xy_world: np.ndarray) -> tuple[float, float, float, float]:
        """Return (xmin, xmax, ymin, ymax) world extent of the BEV image.

        Use as matplotlib ``extent`` arg for imshow(origin='lower').
        """
        cx, cy = float(ego_xy_world[0]), float(ego_xy_world[1])
        return (cx - self.xy_range_m, cx + self.xy_range_m,
                cy - self.xy_range_m, cy + self.xy_range_m)
