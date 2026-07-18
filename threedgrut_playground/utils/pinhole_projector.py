# SPDX-License-Identifier: Apache-2.0
"""Forward projection for OpenCV pinhole cameras — world 3D → image 2D, numpy only.

Mirror of ``ftheta_projector.py`` for NCore's ``OpenCVPinholeCameraModel``
(and, when distortion is absent, the trivial pinhole case used by the viser
viewer's default browser projection). Used by the 7-camera cuboid overlay
validation script (``scripts/validate_cuboid_7cam.py``) so the same call
shape works for both FTheta and pinhole cameras.

Distortion model matches NCore / 3DGUT / OpenCV's rational model (not the
older six-term polynomial):

    r²       = x_n² + y_n²
    r⁴       = r²·r²
    r⁶       = r⁴·r²

    icD_num  = 1 + k1·r² + k2·r⁴ + k3·r⁶
    icD_den  = 1 + k4·r² + k5·r⁴ + k6·r⁶
    icD      = icD_num / icD_den

    a1       = 2·x_n·y_n
    a2       = r² + 2·x_n²
    a3       = r² + 2·y_n²

    delta_x  = p1·a1 + p2·a2 + r²·(s1 + r²·s2)
    delta_y  = p1·a3 + p2·a1 + r²·(s3 + r²·s4)

    x_dist   = x_n·icD + delta_x
    y_dist   = y_n·icD + delta_y

The six radial coefficients (k1,k2,k3,k4,k5,k6) are **not** a six-term
polynomial — the first three form the numerator, the last three form the
denominator.  The "rational" name comes from this numerator/denominator form.

A point is visible (visible=True) only when **all** of:
  - camera-frame z > 0
  - radial projection is finite and valid under either the supplied calibrated
    ``max_valid_r2`` prefix or, when absent, legacy icD interval (0.8, 1.2)
  - projected pixel is within the image rectangle
  - uv coordinates are finite

The legacy 0.8 < icD < 1.2 trust gate matches NCore SDK's
``OpenCVPinholeCameraModel.__compute_distortion()`` and 3DGUT's
``cameraProjections.cuh:72-118``.  Outside this interval the radial model
is unreliable (e.g. fringe artifacts on wide cameras) and the point is
treated as invalid for overlay drawing.  PIN-CAM-1c camera dictionaries carry
``max_valid_r2`` instead; that calibrated ideal-radius certificate matches the
CUDA renderer and replaces the coarse legacy icD heuristic.

NCore stores ``radial_coeffs`` as (k1,k2,k3,k4,k5,k6),
``tangential_coeffs`` as (p1,p2), and ``thin_prism_coeffs`` as (s1,s2,s3,s4).

Pure numpy; no torch, no viser, no kaolin — Mac-testable.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .projector_common import subdivide_polyline

# Maximum coefficient counts
_MAX_RADIAL = 6
_MAX_TANGENTIAL = 2
_MAX_THIN_PRISM = 4

# NCore/3DGUT radial trust interval
_RADIAL_TRUST_LOWER = 0.8
_RADIAL_TRUST_UPPER = 1.2


def _pad_coefficients(values, size, name):
    """Normalise an optional coefficient array to fixed *size*, padding with zeros.

    Parameters
    ----------
    values : array-like or None
        Input coefficients (e.g. [k1, k2]) or None/missing.
    size : int
        Desired output length.
    name : str
        Human-readable name for error messages.

    Returns
    -------
    np.ndarray
        1-D float64 array of length *size*.

    Raises
    ------
    ValueError
        If *values* has more than *size* elements.
    """
    if values is None:
        return np.zeros(size, dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size > size:
        raise ValueError(f"{name} supports at most {size} coefficients, got {arr.size}")
    return np.pad(arr, (0, size - arr.size))


class PinholeForwardProjector:
    """Projects world-space 3D points to pixels through an OpenCV pinhole
    + rational / tangential / thin-prism distortion.

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
            raise ValueError(f"pinhole_dict missing required keys: {sorted(missing)}")

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

        self.radial_coeffs = _pad_coefficients(
            pinhole_dict.get("radial_coeffs"), _MAX_RADIAL, "radial_coeffs"
        )
        self.tangential_coeffs = _pad_coefficients(
            pinhole_dict.get("tangential_coeffs"), _MAX_TANGENTIAL, "tangential_coeffs"
        )
        self.thin_prism_coeffs = _pad_coefficients(
            pinhole_dict.get("thin_prism_coeffs"), _MAX_THIN_PRISM, "thin_prism_coeffs"
        )
        max_valid_r2 = pinhole_dict.get("max_valid_r2")
        if max_valid_r2 is None:
            self.max_valid_r2 = None
        else:
            self.max_valid_r2 = float(np.asarray(max_valid_r2).reshape(()))
            if not np.isfinite(self.max_valid_r2) or self.max_valid_r2 < 0.0:
                raise ValueError("max_valid_r2 must be finite and non-negative")

        if world_to_camera_flip is None:
            world_to_camera_flip = np.eye(4)
        flip = np.asarray(world_to_camera_flip, dtype=np.float64)
        if flip.shape != (4, 4):
            raise ValueError(f"world_to_camera_flip must be (4, 4); got {flip.shape}")
        self._flip = flip

    def project_points(
        self,
        points_world: np.ndarray,  # (N, 3)
        c2w: np.ndarray,  # (4, 4)
    ) -> tuple[np.ndarray, np.ndarray]:
        """3D world points → (uv: (N, 2) pixels, visible: (N,) bool).

        ``visible[i]`` is True iff:
          - camera-frame z > 0 (in front of the camera)
          - radial projection is finite and inside ``max_valid_r2`` when
            supplied, otherwise icD is in the NCore trust interval (0.8, 1.2)
          - projected pixel is inside the image rectangle
          - uv coordinates are finite
        """
        pts = np.asarray(points_world, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points_world must be (N, 3); got {pts.shape}")
        c2w_arr = np.asarray(c2w, dtype=np.float64)
        if c2w_arr.shape != (4, 4):
            raise ValueError(f"c2w must be (4, 4); got {c2w_arr.shape}")

        N = pts.shape[0]
        if N == 0:
            return (np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=bool))

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

        # ---- Rational radial distortion (NCore/3DGUT compatible) ----------
        k1, k2, k3, k4, k5, k6 = (
            self.radial_coeffs[0],
            self.radial_coeffs[1],
            self.radial_coeffs[2],
            self.radial_coeffs[3],
            self.radial_coeffs[4],
            self.radial_coeffs[5],
        )

        r2 = x_n * x_n + y_n * y_n
        r4 = r2 * r2
        r6 = r4 * r2

        num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        den = 1.0 + k4 * r2 + k5 * r4 + k6 * r6

        # Guard denominator zero / near-zero
        den_safe = np.where(np.abs(den) < 1e-12, 1.0, den)
        icD = num / den_safe
        icD = np.where(np.abs(den) < 1e-12, np.inf, icD)

        # ---- Tangential distortion ----------------------------------------
        p1 = self.tangential_coeffs[0]
        p2 = self.tangential_coeffs[1]
        a1 = 2.0 * x_n * y_n
        a2 = r2 + 2.0 * x_n * x_n
        a3 = r2 + 2.0 * y_n * y_n
        delta_x_tan = p1 * a1 + p2 * a2
        delta_y_tan = p1 * a3 + p2 * a1

        # ---- Thin-prism distortion ----------------------------------------
        s1, s2, s3, s4 = (
            self.thin_prism_coeffs[0],
            self.thin_prism_coeffs[1],
            self.thin_prism_coeffs[2],
            self.thin_prism_coeffs[3],
        )
        delta_x_tp = r2 * (s1 + r2 * s2)
        delta_y_tp = r2 * (s3 + r2 * s4)

        # ---- Combined distorted coordinates -------------------------------
        x_dist = x_n * icD + delta_x_tan + delta_x_tp
        y_dist = y_n * icD + delta_y_tan + delta_y_tp

        u = self.fx * x_dist + self.cx
        v = self.fy * y_dist + self.cy
        uv = np.stack([u, v], axis=-1)

        # ---- Visibility mask ----------------------------------------------
        z_pos = z > 0
        if self.max_valid_r2 is not None:
            # PIN-CAM-1c: use the same calibrated ideal-radius prefix supplied
            # to the CUDA renderer.  This intentionally replaces (rather than
            # intersects) the coarse 0.8..1.2 icD heuristic, which rejects
            # valid wide-camera edge rays.
            valid_radial = np.isfinite(icD) & np.isfinite(r2) & (r2 <= self.max_valid_r2)
        else:
            valid_radial = np.isfinite(icD) & (icD > _RADIAL_TRUST_LOWER) & (icD < _RADIAL_TRUST_UPPER)
        in_bound = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        uv_finite = np.isfinite(uv).all(axis=1)
        visible = z_pos & valid_radial & in_bound & uv_finite
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
                out.append((np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=bool)))
                continue
            out.append((uv_all[cursor : cursor + L], vis_all[cursor : cursor + L]))
            cursor += L
        return out
