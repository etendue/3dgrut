# SPDX-License-Identifier: Apache-2.0
"""B2: FTheta forward projection — world 3D → image 2D for the viser overlay path.

Mirror of ``ftheta_intrinsics.py:ftheta_pixels_to_camera_rays`` (the inverse
path used by the engine raygen). Used by ``Viser4DOverlayCompositor`` to draw
cuboid / frustum / track / ego-trajectory wireframes on top of the Gaussian
backdrop in FTheta mode, so that wireframes and the backdrop share the same
fisheye projection (B2 fix in T8_buglists.md).

Calibration constants pinned by Phase 0 probe on ThinkPad with
``ckpt_with_ftheta_v2.pt``; see ``docs/T8_artifacts/B2_calibration_probe_log.md``
for the per-candidate analysis. The viser/ckpt ego pose convention is
``+Y down + Z backward`` (Y already matches OpenCV image axis), so the
GL→CV flip only inverts Z:
    c2w_cv = c2w_viser @ diag([1, 1, -1, 1])

Pure numpy; no torch, no viser, no kaolin — fully Mac-testable.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .projector_common import horner_ascending, subdivide_polyline


# Phase 0 calibration: see docs/T8_artifacts/B2_calibration_probe_log.md
# This flip maps a c2w in viser convention (+Y down, +Z backward) to OpenCV
# convention (+Y down, +Z forward). For c2w already in OpenCV convention
# (e.g. NCoreDataset's T_camera_to_world for the raw-camera projection path),
# pass ``world_to_camera_flip=np.eye(4)`` to skip the flip.
FLIP_VISER_TO_OPENCV: np.ndarray = np.diag([1.0, 1.0, -1.0, 1.0])


# Back-compat aliases (older imports may reach in for these symbols).
_horner_ascending = horner_ascending
_subdivide_polyline = subdivide_polyline


class FthetaForwardProjector:
    """Projects world-space 3D points to pixels through the same FTheta
    polynomial used by the engine's 3dgut UT rasterizer.

    Stateless apart from caching the parsed ``ftheta_dict`` entries.
    Thread-safe (no mutation after __init__).

    Example
    -------
    >>> proj = FthetaForwardProjector(ftheta_dict)
    >>> uv, visible = proj.project_points(points_world, c2w_viser)
    >>> # uv: (N, 2) float64 pixels; visible: (N,) bool

    For polyline drawing (cuboid edges etc.), use ``project_polylines`` which
    handles the piecewise subdivision needed because a 3D straight line
    projects to a *curve* under fisheye.
    """

    def __init__(
        self,
        ftheta_dict: dict,
        world_to_camera_flip: Optional[np.ndarray] = None,
    ):
        """Parse ftheta_dict into numpy arrays. Validates required keys.

        ``world_to_camera_flip`` is a 4×4 right-multiplied onto ``c2w`` before
        inversion, mapping the caller's c2w convention onto OpenCV
        (+Y down, +Z forward). Defaults to ``FLIP_VISER_TO_OPENCV`` for
        backward compatibility with the viser viewer path. Pass
        ``np.eye(4)`` when c2w is already in OpenCV convention (NCore raw
        camera ``T_camera_to_world``).
        """
        REQUIRED = {
            "resolution", "principal_point",
            "angle_to_pixeldist_poly", "max_angle",
        }
        missing = REQUIRED - set(ftheta_dict.keys())
        if missing:
            raise ValueError(f"ftheta_dict missing required keys: {sorted(missing)}")

        res = ftheta_dict["resolution"]
        self.width  = int(res[0])
        self.height = int(res[1])

        pp = ftheta_dict["principal_point"]
        self.cx = float(pp[0])
        self.cy = float(pp[1])

        self.angle_to_pixeldist_poly = np.asarray(
            ftheta_dict["angle_to_pixeldist_poly"], dtype=np.float64)
        self.max_angle = float(ftheta_dict["max_angle"])
        # linear_cde intentionally skipped — see ftheta_intrinsics.py:50-57.

        if world_to_camera_flip is None:
            world_to_camera_flip = FLIP_VISER_TO_OPENCV
        flip = np.asarray(world_to_camera_flip, dtype=np.float64)
        if flip.shape != (4, 4):
            raise ValueError(
                f"world_to_camera_flip must be (4, 4); got {flip.shape}")
        self._flip = flip

    def project_points(
        self,
        points_world: np.ndarray,         # (N, 3) float64-compatible
        c2w_viser: np.ndarray,            # (4, 4) viser/ckpt convention (+Y down, +Z backward)
    ) -> tuple[np.ndarray, np.ndarray]:
        """3D world points → (uv: (N, 2) pixels, visible: (N,) bool).

        ``visible[i]`` is True iff:
          - point is in front of camera (cam-frame z > 0 after Z flip),
          - ray angle from optical axis ≤ ``max_angle`` (within fisheye FOV),
          - projected pixel falls within (0, W) × (0, H).
        """
        pts = np.asarray(points_world, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points_world must be (N, 3); got {pts.shape}")
        c2w = np.asarray(c2w_viser, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"c2w_viser must be (4, 4); got {c2w.shape}")

        c2w_cv = c2w @ self._flip
        w2c = np.linalg.inv(c2w_cv)

        N = pts.shape[0]
        if N == 0:
            return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=bool)

        p_h = np.concatenate([pts, np.ones((N, 1), dtype=np.float64)], axis=-1)
        p_cam = (w2c @ p_h.T).T[:, :3]
        x, y, z = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]

        r_xy = np.sqrt(x * x + y * y)
        angle = np.arctan2(r_xy, z)                                # ∈ [0, π]

        r_pix = horner_ascending(self.angle_to_pixeldist_poly, angle)

        safe_r = np.where(r_xy < 1e-9, 1.0, r_xy)
        u_off = np.where(r_xy < 1e-9, 0.0, r_pix * x / safe_r)
        v_off = np.where(r_xy < 1e-9, 0.0, r_pix * y / safe_r)

        u = self.cx + u_off
        v = self.cy + v_off
        uv = np.stack([u, v], axis=-1)

        in_fov   = angle <= self.max_angle
        in_bound = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        z_pos    = z > 0
        visible  = in_fov & in_bound & z_pos
        return uv, visible

    def project_polylines(
        self,
        polylines_world: Sequence[np.ndarray],   # list of (M_i, 3)
        c2w_viser: np.ndarray,                   # (4, 4)
        subdivide_n: int = 20,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Project each polyline after piecewise-linear subdivision.

        Each input polyline of M vertices becomes 1 + (M-1)*subdivide_n
        vertices in 3D, then projects to pixels. Renderer is responsible
        for skipping segments where either endpoint has ``visible=False``.

        ``subdivide_n=1`` returns the original endpoints (no subdivision).

        Returns: list of (uv: (M', 2), visible: (M',)) per input polyline.
        """
        if subdivide_n < 1:
            raise ValueError(f"subdivide_n must be >= 1; got {subdivide_n}")

        # Concat all subdivided polylines into one batched project call, then split.
        all_pts: list[np.ndarray] = []
        lengths: list[int] = []
        for pl in polylines_world:
            pl = np.asarray(pl, dtype=np.float64)
            if pl.ndim != 2 or pl.shape[1] != 3:
                raise ValueError(f"each polyline must be (M, 3); got {pl.shape}")
            if pl.shape[0] < 2:
                lengths.append(pl.shape[0])
                if pl.shape[0] == 1:
                    all_pts.append(pl)
                continue
            sub = subdivide_polyline(pl, subdivide_n)
            lengths.append(sub.shape[0])
            all_pts.append(sub)

        if not all_pts:
            return []

        cat = np.concatenate(all_pts, axis=0)
        uv_all, vis_all = self.project_points(cat, c2w_viser)

        out: list[tuple[np.ndarray, np.ndarray]] = []
        cursor = 0
        for L in lengths:
            if L == 0:
                out.append((np.empty((0, 2), dtype=np.float64),
                            np.empty((0,), dtype=bool)))
                continue
            out.append((uv_all[cursor:cursor + L], vis_all[cursor:cursor + L]))
            cursor += L
        return out


# NOTE: ``_subdivide_polyline`` and ``_horner_ascending`` now live in
# ``projector_common``; module-level aliases above keep older imports working.
