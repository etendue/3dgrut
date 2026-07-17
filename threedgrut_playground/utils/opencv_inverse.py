# SPDX-License-Identifier: Apache-2.0
"""Machine-precision inverse for the complete OpenCV rational camera model.

The production camera contract contains rational radial, tangential, and
thin-prism terms.  A one-dimensional radial lookup therefore cannot be used as
the truth oracle for a full image: it misses the azimuth-dependent terms.  This
module uses a dense radial lookup only as an initial guess and then solves the
two-dimensional distortion equations with Newton iterations.

Pure NumPy; intended for calibration validation on CPU-only Mac hosts.
"""

from __future__ import annotations

import numpy as np

from .pinhole_projector import PinholeForwardProjector

_NEWTON_ITERATIONS = 8
_INVERSE_CHUNK_SIZE = 262_144
_RADIAL_BRANCH_SAMPLES = 32_769
_RADIAL_BRANCH_SCAN_MAX = 4.0
_RADIAL_TRUST_LOWER = 0.8
_RADIAL_TRUST_UPPER = 1.2
_RESIDUAL_TOLERANCE = 1e-10


def _distort_normalized(
    projector: PinholeForwardProjector,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the exact normalized-coordinate distortion used by the projector."""
    k1, k2, k3, k4, k5, k6 = projector.radial_coeffs
    p1, p2 = projector.tangential_coeffs
    s1, s2, s3, s4 = projector.thin_prism_coeffs

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        radial = numerator / denominator

    a1 = 2.0 * x * y
    a2 = r2 + 2.0 * x * x
    a3 = r2 + 2.0 * y * y
    delta_x = p1 * a1 + p2 * a2 + r2 * (s1 + r2 * s2)
    delta_y = p1 * a3 + p2 * a1 + r2 * (s3 + r2 * s4)
    return x * radial + delta_x, y * radial + delta_y


def _radial_scale(
    projector: PinholeForwardProjector, radius: np.ndarray
) -> np.ndarray:
    """Return the rational radial scale ``icD`` without non-radial terms."""
    k1, k2, k3, k4, k5, k6 = projector.radial_coeffs
    r2 = np.asarray(radius, dtype=np.float64) ** 2
    numerator = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    denominator = 1.0 + k4 * r2 + k5 * r2**2 + k6 * r2**3
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return numerator / denominator


def _first_physical_branch_limit(projector: PinholeForwardProjector) -> float:
    """Return the conservative end of the physical branch from the axis.

    A small Newton residual does not identify which inverse of a folded
    rational curve was found.  The production OpenCV contract also requires
    ``0.8 < icD < 1.2``.  Starting at the optical axis, this function stops at
    the first sample that either leaves that trust interval or makes
    ``r_distorted = r * icD`` non-increasing.  Later roots are mathematical
    branches, not camera rays.
    """
    radius = np.linspace(
        0.0,
        _RADIAL_BRANCH_SCAN_MAX,
        _RADIAL_BRANCH_SAMPLES,
        dtype=np.float64,
    )
    scale = _radial_scale(projector, radius)
    distorted_radius = radius * scale
    increasing = np.concatenate(
        [np.array([True]), np.diff(distorted_radius) > 0.0]
    )
    trusted = (
        np.isfinite(scale)
        & (scale > _RADIAL_TRUST_LOWER)
        & (scale < _RADIAL_TRUST_UPPER)
        & np.isfinite(distorted_radius)
        & increasing
    )
    invalid = np.flatnonzero(~np.logical_and.accumulate(trusted))
    if invalid.size == 0:
        return float(radius[-1])
    return float(radius[int(invalid[0])])


def _distortion_jacobian_determinant(
    projector: PinholeForwardProjector,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Finite-difference determinant of the complete 2-D distortion map."""
    hx = 1e-6 * (1.0 + np.abs(x))
    hy = 1e-6 * (1.0 + np.abs(y))
    fx_plus, fy_plus = _distort_normalized(projector, x + hx, y)
    fx_minus, fy_minus = _distort_normalized(projector, x - hx, y)
    j00 = (fx_plus - fx_minus) / (2.0 * hx)
    j10 = (fy_plus - fy_minus) / (2.0 * hx)
    fx_plus, fy_plus = _distort_normalized(projector, x, y + hy)
    fx_minus, fy_minus = _distort_normalized(projector, x, y - hy)
    j01 = (fx_plus - fx_minus) / (2.0 * hy)
    j11 = (fy_plus - fy_minus) / (2.0 * hy)
    return j00 * j11 - j01 * j10


def opencv_forward_domain_mask(
    projector: PinholeForwardProjector,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Identify normalized points on NCore's first physical forward branch."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    radius = np.hypot(x, y)
    scale = _radial_scale(projector, radius)
    determinant = _distortion_jacobian_determinant(projector, x, y)
    branch_limit = _first_physical_branch_limit(projector)
    return (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(scale)
        & (scale > _RADIAL_TRUST_LOWER)
        & (scale < _RADIAL_TRUST_UPPER)
        & (radius < branch_limit)
        & np.isfinite(determinant)
        & (determinant > 0.0)
    )


def _radial_lut_initial_guess(
    projector: PinholeForwardProjector,
    xd: np.ndarray,
    yd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the radial component on its first monotonic branch."""
    rd = np.hypot(xd, yd)
    ru_grid = np.linspace(
        0.0, _RADIAL_BRANCH_SCAN_MAX, _RADIAL_BRANCH_SAMPLES, dtype=np.float64
    )
    radial_x = ru_grid * _radial_scale(projector, ru_grid)
    branch_limit = _first_physical_branch_limit(projector)
    stop = int(np.searchsorted(ru_grid, branch_limit, side="left"))
    stop = max(stop, 2)
    ru = np.interp(rd, radial_x[:stop], ru_grid[:stop])

    scale = np.divide(ru, rd, out=np.ones_like(ru), where=rd > 0.0)
    return xd * scale, yd * scale


def _invert_chunk(
    projector: PinholeForwardProjector,
    xd: np.ndarray,
    yd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = _radial_lut_initial_guess(projector, xd, yd)

    for _ in range(_NEWTON_ITERATIONS):
        fx, fy = _distort_normalized(projector, x, y)
        residual_x = fx - xd
        residual_y = fy - yd

        # Central finite differences match the validation plan and keep this
        # oracle independent of a hand-derived Jacobian.  The scale-aware step
        # avoids cancellation near the optical axis and over-large steps at
        # the image corners.
        hx = 1e-6 * (1.0 + np.abs(x))
        hy = 1e-6 * (1.0 + np.abs(y))
        fx_plus, fy_plus = _distort_normalized(projector, x + hx, y)
        fx_minus, fy_minus = _distort_normalized(projector, x - hx, y)
        j00 = (fx_plus - fx_minus) / (2.0 * hx)
        j10 = (fy_plus - fy_minus) / (2.0 * hx)
        fx_plus, fy_plus = _distort_normalized(projector, x, y + hy)
        fx_minus, fy_minus = _distort_normalized(projector, x, y - hy)
        j01 = (fx_plus - fx_minus) / (2.0 * hy)
        j11 = (fy_plus - fy_minus) / (2.0 * hy)

        determinant = j00 * j11 - j01 * j10
        solvable = np.isfinite(determinant) & (np.abs(determinant) >= 1e-14)
        safe_determinant = np.where(solvable, determinant, 1.0)
        step_x = (j11 * residual_x - j01 * residual_y) / safe_determinant
        step_y = (-j10 * residual_x + j00 * residual_y) / safe_determinant
        step_x[~solvable] = np.nan
        step_y[~solvable] = np.nan
        x -= step_x
        y -= step_y

    final_x, final_y = _distort_normalized(projector, x, y)
    residual = np.hypot(final_x - xd, final_y - yd)
    physical = (
        np.isfinite(residual)
        & (residual < _RESIDUAL_TOLERANCE)
        & opencv_forward_domain_mask(projector, x, y)
    )
    x[~physical] = np.nan
    y[~physical] = np.nan
    residual[~physical] = np.inf
    return x, y, residual


def invert_opencv_full_model(
    pinhole_dict: dict,
    uv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert distorted pixels into undistorted normalized coordinates.

    Parameters
    ----------
    pinhole_dict:
        OpenCV rational parameter dictionary accepted by
        :class:`PinholeForwardProjector`.
    uv:
        Float pixel coordinates with shape ``(N, 2)``.

    Returns
    -------
    xy, residual:
        Undistorted normalized coordinates ``(N, 2)`` and final normalized
        distortion-equation residual ``(N,)``.  Pixels without a solution on
        NCore's trusted first branch are represented by ``NaN`` coordinates
        and an infinite residual.  A small residual on a later folded branch
        is deliberately not accepted.
    """
    pixels = np.asarray(uv, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"uv must have shape (N, 2); got {pixels.shape}")
    if not np.isfinite(pixels).all():
        raise ValueError("uv must contain only finite coordinates")

    projector = PinholeForwardProjector(pinhole_dict)
    xd_all = (pixels[:, 0] - projector.cx) / projector.fx
    yd_all = (pixels[:, 1] - projector.cy) / projector.fy
    xy = np.empty((len(pixels), 2), dtype=np.float64)
    residual = np.empty(len(pixels), dtype=np.float64)

    for start in range(0, len(pixels), _INVERSE_CHUNK_SIZE):
        stop = min(start + _INVERSE_CHUNK_SIZE, len(pixels))
        x, y, chunk_residual = _invert_chunk(
            projector, xd_all[start:stop], yd_all[start:stop]
        )
        xy[start:stop, 0] = x
        xy[start:stop, 1] = y
        residual[start:stop] = chunk_residual
    return xy, residual


def opencv_pixels_to_camera_rays(pinhole_dict: dict) -> np.ndarray:
    """Return exact float64 unit rays for the invertible integer pixels.

    Pixels outside a rational model's physical/invertible branch are returned
    as ``NaN`` rays.  This is intentional: some b6a9 wide calibrations contain
    a denominator pole outside their forward-valid domain.  Downstream survey
    code must report the resulting coverage instead of inventing a ray on an
    arbitrary rational branch.
    """
    projector = PinholeForwardProjector(pinhole_dict)
    ys, xs = np.mgrid[0:projector.height, 0:projector.width]
    uv = np.stack([xs.ravel(), ys.ravel()], axis=-1)
    xy, residual = invert_opencv_full_model(pinhole_dict, uv)
    rays = np.column_stack([xy, np.ones(len(xy), dtype=np.float64)])
    valid = (
        np.isfinite(residual)
        & (residual < _RESIDUAL_TOLERANCE)
        & np.isfinite(xy).all(axis=1)
    )
    rays[~valid] = np.nan
    rays[valid] /= np.linalg.norm(rays[valid], axis=1, keepdims=True)
    return rays.reshape(projector.height, projector.width, 3)
