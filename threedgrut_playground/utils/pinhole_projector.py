# SPDX-License-Identifier: Apache-2.0
"""Forward projection for OpenCV pinhole cameras (with radial / tangential
distortion) — world 3D → image 2D, numpy only.

Mirror of ``ftheta_projector.py`` for NCore's ``OpenCVPinholeCameraModel``
(and, when distortion is absent, the trivial pinhole case used by the viser
viewer's default browser projection). Used by the 7-camera cuboid overlay
validation script (``scripts/validate_cuboid_7cam.py``) so the same call
shape works for both FTheta and pinhole cameras.

Distortion model matches OpenCV's ``cv2.projectPoints`` formulation:
    r²       = x_n² + y_n²
    radial   = 1 + k1·r² + k2·r⁴ + k3·r⁶ + k4·r⁸ + k5·r¹⁰ + ...
    x_dist   = x_n · radial + 2·p1·x_n·y_n + p2·(r² + 2·x_n²)
    y_dist   = y_n · radial + p1·(r² + 2·y_n²) + 2·p2·x_n·y_n

NCore stores ``radial_coeffs`` as an ascending list (k1, k2, k3, ...) and
``tangential_coeffs`` as ``[p1, p2]``. ``thin_prism_coeffs`` is currently
ignored (rare on NCore v4 vehicle cameras); add it if a real camera needs it.

Pure numpy; no torch, no viser, no kaolin — Mac-testable.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .projector_common import subdivide_polyline


class PinholeForwardProjector:
    """Projects world-space 3D points to pixels through an OpenCV pinhole
    + optional polynomial distortion.

    Defaults to ``world_to_camera_flip = identity`` because the canonical use
    case is the NCore raw-camera validation path, where ``T_camera_to_world``
    is already in OpenCV convention (+Y down, +Z forward). Pass
    ``FLIP_VISER_TO_OPENCV`` (from ``ftheta_projector``) if you need to feed
    a viser-convention c2w.

    Example
    -------
    >>> proj = PinholeForwardProjector(pinhole_dict)            # NCore raw
    >>> uv, visible = proj.project_points(points_world, T_c2w)
    """

    def __init__(
        self,
        pinhole_dict: dict,
        world_to_camera_flip: Optional[np.ndarray] = None,
    ):
        REQUIRED = {"resolution", "principal_point", "focal_length"}
        missing = REQUIRED - set(pinhole_dict.keys())
        if missing:
            raise ValueError(
                f"pinhole_dict missing required keys: {sorted(missing)}")

        res = pinhole_dict["resolution"]
        self.width = int(res[0])
        self.height = int(res[1])

        pp = pinhole_dict["principal_point"]
        self.cx = float(pp[0])
        self.cy = float(pp[1])

        fl = pinhole_dict["focal_length"]
        # NCore's OpenCVPinholeCameraModel stores focal_length as a 2-vector
        # (fx, fy); also tolerate a scalar (square pixel).
        fl_arr = np.atleast_1d(np.asarray(fl, dtype=np.float64))
        if fl_arr.size == 1:
            self.fx = self.fy = float(fl_arr[0])
        else:
            self.fx = float(fl_arr[0])
            self.fy = float(fl_arr[1])

        self.radial_coeffs = np.asarray(
            pinhole_dict.get("radial_coeffs", []), dtype=np.float64).ravel()
        self.tangential_coeffs = np.asarray(
            pinhole_dict.get("tangential_coeffs", []), dtype=np.float64).ravel()

        if world_to_camera_flip is None:
            world_to_camera_flip = np.eye(4)
        flip = np.asarray(world_to_camera_flip, dtype=np.float64)
        if flip.shape != (4, 4):
            raise ValueError(
                f"world_to_camera_flip must be (4, 4); got {flip.shape}")
        self._flip = flip

    def project_points(
        self,
        points_world: np.ndarray,         # (N, 3)
        c2w: np.ndarray,                  # (4, 4)
    ) -> tuple[np.ndarray, np.ndarray]:
        """3D world points → (uv: (N, 2) pixels, visible: (N,) bool).

        ``visible[i]`` is True iff cam-frame z > 0 (in front of the camera)
        AND projected pixel is inside the image rectangle.
        """
        pts = np.asarray(points_world, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points_world must be (N, 3); got {pts.shape}")
        c2w_arr = np.asarray(c2w, dtype=np.float64)
        if c2w_arr.shape != (4, 4):
            raise ValueError(f"c2w must be (4, 4); got {c2w_arr.shape}")

        N = pts.shape[0]
        if N == 0:
            return (np.empty((0, 2), dtype=np.float64),
                    np.empty((0,), dtype=bool))

        c2w_cv = c2w_arr @ self._flip
        w2c = np.linalg.inv(c2w_cv)

        p_h = np.concatenate([pts, np.ones((N, 1), dtype=np.float64)], axis=-1)
        p_cam = (w2c @ p_h.T).T[:, :3]
        x, y, z = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]

        # Guard z → 0 to avoid div-by-zero; the visibility mask filters z ≤ 0
        # downstream so the bogus uv from these points is never drawn.
        safe_z = np.where(np.abs(z) < 1e-9, 1.0, z)
        x_n = x / safe_z
        y_n = y / safe_z

        if self.radial_coeffs.size > 0 or self.tangential_coeffs.size > 0:
            r2 = x_n * x_n + y_n * y_n
            # radial = 1 + k1·r² + k2·r⁴ + k3·r⁶ + ...
            radial = np.ones_like(r2)
            r_pow = r2.copy()
            for k in self.radial_coeffs:
                radial = radial + k * r_pow
                r_pow = r_pow * r2

            if self.tangential_coeffs.size >= 2:
                p1 = float(self.tangential_coeffs[0])
                p2 = float(self.tangential_coeffs[1])
                x_dist = (x_n * radial
                          + 2.0 * p1 * x_n * y_n
                          + p2 * (r2 + 2.0 * x_n * x_n))
                y_dist = (y_n * radial
                          + p1 * (r2 + 2.0 * y_n * y_n)
                          + 2.0 * p2 * x_n * y_n)
            else:
                x_dist = x_n * radial
                y_dist = y_n * radial
        else:
            x_dist = x_n
            y_dist = y_n

        u = self.fx * x_dist + self.cx
        v = self.fy * y_dist + self.cy
        uv = np.stack([u, v], axis=-1)

        z_pos = z > 0
        in_bound = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        visible = z_pos & in_bound
        return uv, visible

    def project_polylines(
        self,
        polylines_world: Sequence[np.ndarray],
        c2w: np.ndarray,
        subdivide_n: int = 4,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Project each polyline after piecewise-linear subdivision.

        Pinhole projection maps 3D straight lines to 2D straight lines (in
        the absence of distortion), so ``subdivide_n`` defaults to 4 — just
        enough to catch the sub-pixel curvature introduced by radial
        distortion at large field angles. For FTheta-grade fisheye see
        ``FthetaForwardProjector.project_polylines`` (subdivide_n=20).
        """
        if subdivide_n < 1:
            raise ValueError(f"subdivide_n must be >= 1; got {subdivide_n}")

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
        uv_all, vis_all = self.project_points(cat, c2w)

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
