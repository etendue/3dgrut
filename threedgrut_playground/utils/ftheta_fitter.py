# SPDX-License-Identifier: Apache-2.0
"""PIN-FTHETA-1: Fit a monotonic FTheta polynomial model from OpenCV rational params.

Converts a ``PinholeForwardProjector`` view of an ``OpenCVPinholeCameraModel``
(6-coeff rational radial + tangential + thin-prism distortion) into an 8-key
``FThetaCameraModelParameters`` dict suitable for the 3dgut UT rasterizer.

Fitting strategy
  1. Sample N rays at uniform angles θ ∈ [0, θ_max] on the +X half-plane.
  2. Project each ray through the OpenCV rational model → pixel (u, v).
  3. Compute pixel distance r = ‖(u, v) − (cx, cy)‖.
  4. Fit a 5th-order monotonic polynomial to the (θ, r) mapping.
  5. Fit the inverse polynomial to the (r, θ) mapping.
  6. Determine max_angle as the largest θ whose projection lands inside
     the image rectangle (with a small margin so edge pixels are usable).
  7. The fitted FTheta dict mirrors the 8-key shape produced by
     ``threedgrut.datasets.datasetNcore.NCoreDataset`` for native FTheta
     sensors: {resolution, shutter_type, principal_point, reference_poly,
               pixeldist_to_angle_poly, angle_to_pixeldist_poly, max_angle,
               linear_cde}.

Pure numpy; no torch, no viser, no kaolin — Mac-testable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .pinhole_projector import PinholeForwardProjector
from .projector_common import horner_ascending

# NCore FTheta polynomial degree (0…5 → 6 coefficients).
_FTHETA_POLY_DEGREE: int = 5

# Sampling density: 200 rays between 0 and θ_max gives enough resolution
# to capture the non-linear behaviour of wide-angle rational distortion
# without overfitting noise.
_N_ANGLE_SAMPLES: int = 200

# Margin (in pixels) from the image border when determining max_angle.
# Prevents edge pixels being declared "in FOV" when the polynomial fitting
# can't reliably hit them.
_EDGE_MARGIN_PX: int = 3

# Hard-gate density used after either the primary polynomial fit or the
# deterministic least-squares fallback.  Checking only the original 200 fit
# samples can miss a negative derivative between samples.
_DENSE_MONOTONIC_SAMPLES: int = 20_001


def _minimum_polynomial_derivative(
    coeffs: np.ndarray,
    domain_max: float,
    *,
    n_samples: int = _DENSE_MONOTONIC_SAMPLES,
) -> float:
    """Return the dense minimum derivative on ``[0, domain_max]``."""
    coefficients = np.asarray(coeffs, dtype=np.float64)
    if coefficients.ndim != 1 or len(coefficients) < 2:
        raise ValueError("coeffs must contain at least constant and linear terms")
    if not np.isfinite(coefficients).all():
        return float("nan")
    maximum = float(domain_max)
    if not np.isfinite(maximum) or maximum <= 0.0:
        return float("nan")
    derivative = np.arange(1, len(coefficients), dtype=np.float64) * coefficients[1:]
    samples = np.linspace(0.0, maximum, n_samples, dtype=np.float64)
    values = horner_ascending(derivative, samples)
    return float(np.min(values)) if np.isfinite(values).all() else float("nan")


def _rational_project_ray(
    proj: PinholeForwardProjector,
    angle_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame rays at given angles through the rational model.

    Each ray is (sin(θ), 0, cos(θ)) in camera space — on the +X half-plane
    at distance z=1.  The projector then applies inverse-projection →
    rational distortion → pixel coordinates.

    Returns (uv, r_pix) where uv is (N, 2) and r_pix is pixel distance from
    principal point.
    """
    from .opencv_inverse import _distort_normalized

    angles = np.asarray(angle_rad, dtype=np.float64)
    x = np.tan(angles)
    y = np.zeros_like(x)
    xd, yd = _distort_normalized(proj, x, y)
    uv = np.stack([proj.fx * xd + proj.cx, proj.fy * yd + proj.cy], axis=-1)

    # Compute pixel distance from principal point.  This calibration path
    # deliberately evaluates the raw rational curve beyond the production
    # OpenCV projector's 0.8 < icD < 1.2 visibility guard.  That guard is a
    # runtime safety policy for the pinhole renderer, not part of the lens
    # calibration, and inheriting it here would truncate the fitted FTheta
    # field of view at exactly the limitation this conversion is meant to
    # remove.
    du = uv[:, 0] - proj.cx
    dv = uv[:, 1] - proj.cy
    r_pix = np.sqrt(du * du + dv * dv)

    # Keep only forward, finite rays here.  Fold rejection is handled from the
    # raw r(theta) monotonic prefix in fit_ftheta_from_opencv_rational(),
    # independently of the pinhole runtime trust interval.
    usable = (
        (np.cos(angles) > 0.0)
        & np.isfinite(uv).all(axis=-1)
        & np.isfinite(r_pix)
    )
    uv[~usable] = np.nan
    r_pix[~usable] = np.nan

    return uv, r_pix


def _fit_monotonic_polynomial(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    monotonic_tol: float = 1e-6,
) -> np.ndarray:
    """Fit a polynomial y = Σ c_k * x^k with a strict monotonicity check.

    Uses numpy.polyfit (ascending-degree coefficients) then verifies that
    the derivative on the sampled domain is ≥ 0 (or ≤ 0).

    If the raw fit is non-monotonic, falls back to a constrained least-
    squares approximation that forces the polynomial through (0, 0) and
    bounds the linear coefficient to be positive.  (The constant term for
    pixeldist_from_angle must be zero — a ray at θ=0 projects to the
    principal point.)

    Returns coefficients in ascending order: [c0, c1, ..., c_{degree}].
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError(f"x and y must be same-length 1-D arrays; got {x.shape}, {y.shape}")
    if len(x) < degree + 1:
        raise ValueError(f"need at least {degree + 1} samples for degree {degree}")

    # Fit in a dimensionless domain.  The inverse fit uses pixel radii near
    # 1,100, where r**5 is O(1e15); an unnormalised Vandermonde matrix silently
    # loses the precision this calibration oracle is meant to measure.
    x_scale = float(np.max(np.abs(x)))
    y_scale = float(np.max(np.abs(y)))
    if x_scale == 0.0:
        return np.array([float(np.mean(y))] + [0.0] * degree, dtype=np.float64)
    if y_scale == 0.0:
        return np.zeros(degree + 1, dtype=np.float64)
    x_normalized = x / x_scale
    y_normalized = y / y_scale
    coeffs_normalized_desc = np.polyfit(x_normalized, y_normalized, degree)
    coeffs_normalized = coeffs_normalized_desc[::-1].astype(np.float64)
    powers = np.arange(degree + 1, dtype=np.float64)
    coeffs = coeffs_normalized * y_scale / np.power(x_scale, powers)

    # The hard camera contract requires a strictly positive derivative over
    # the complete used domain.  Evaluate densely rather than only at the fit
    # samples, which can miss a between-sample fold.
    minimum_derivative = _minimum_polynomial_derivative(coeffs, x_scale)
    if np.isfinite(minimum_derivative) and minimum_derivative > monotonic_tol:
        return coeffs

    # ---- Fallback: force monotonic via pinning (0,0) + positive slope ----
    # This is a band-aid; a proper monotonic-spline fit would be better
    # but adds a dependency (scipy) we don't want in the pure-numpy path.
    # In practice the rational model at 200 samples gives a well-behaved
    # curve that polyfit handles without needing the fallback.

    # Build the fallback in the same dimensionless domain as the primary fit.
    # In particular, inverse fits can have x_max in the thousands of pixels;
    # using x**5 here recreates the ill-conditioned path normalization was
    # meant to eliminate.
    # Constrain c0 = 0 (ray at θ=0 → r=0), then solve for c1..c_{degree}
    # with a non-negativity constraint on c1 (the linear term must be
    # positive for r to grow with θ).
    A = np.zeros((len(x_normalized), degree + 1), dtype=np.float64)
    for k in range(degree + 1):
        A[:, k] = x_normalized**k

    # Solve: min ||A @ c - y||  subject to c0=0, c1 >= 0.
    # Equivalent to: min ||A[:, 1:] @ c' - y|| with c1 >= 0.
    A_sub = A[:, 1:]  # (N, degree)
    c_sub, _residuals, _rank, _s = np.linalg.lstsq(
        A_sub, y_normalized, rcond=None
    )

    # Force the linear term to be non-negative.
    if c_sub[0] < 0:
        # Clamp and re-solve without it.
        c_sub[0] = 0.0
        if degree >= 2:
            A_sub2 = A[:, 2:]  # drop both c0 and c1
            c_sub2, _, _, _ = np.linalg.lstsq(
                A_sub2, y_normalized, rcond=None
            )
            c_sub[1:] = c_sub2
        else:
            # Only c0 and c1 — with both zero there's nothing to fit.
            pass

    coeffs_fallback_normalized = np.zeros(degree + 1, dtype=np.float64)
    coeffs_fallback_normalized[1:] = c_sub  # c0 = 0 already
    coeffs_fallback = (
        coeffs_fallback_normalized * y_scale / np.power(x_scale, powers)
    )
    fallback_minimum_derivative = _minimum_polynomial_derivative(
        coeffs_fallback,
        x_scale,
    )
    if (
        not np.isfinite(fallback_minimum_derivative)
        or fallback_minimum_derivative <= monotonic_tol
    ):
        raise RuntimeError(
            "deterministic polynomial fallback is not strictly monotonic over "
            f"the used domain: min_derivative={fallback_minimum_derivative!r}"
        )
    return coeffs_fallback


def fit_ftheta_from_opencv_rational(
    pinhole_dict: dict,
    n_samples: int = _N_ANGLE_SAMPLES,
    edge_margin_px: int = _EDGE_MARGIN_PX,
) -> dict:
    """Fit an FTheta polynomial model from OpenCV rational camera parameters.

    Parameters
    ----------
    pinhole_dict : dict
        OpenCV pinhole intrinsics as produced by
        ``NCoreDataset._get_camera_model_parameters_for_resolution``.
        Required keys: resolution, principal_point, focal_length.
        Optional: radial_coeffs, tangential_coeffs, thin_prism_coeffs.
    n_samples : int
        Number of angular samples between 0 and max_angle.
    edge_margin_px : int
        Pixel margin from image border when determining FOV extent.

    Returns
    -------
    ftheta_dict : dict
        8-key FTheta intrinsics dict:
        {resolution, shutter_type, principal_point, reference_poly,
         pixeldist_to_angle_poly, angle_to_pixeldist_poly, max_angle,
         linear_cde}
    """
    proj = PinholeForwardProjector(pinhole_dict)

    W = proj.width
    H = proj.height
    cx = proj.cx
    cy = proj.cy

    # Determine the maximum usable angle: the angle at which the projected
    # pixel distance from the principal point reaches the image corner
    # distance minus a margin.
    # Image corners: the four corners relative to principal point.
    corners_r = np.array([
        np.sqrt((0 - cx) ** 2 + (0 - cy) ** 2),
        np.sqrt((W - cx) ** 2 + (0 - cy) ** 2),
        np.sqrt((0 - cx) ** 2 + (H - cy) ** 2),
        np.sqrt((W - cx) ** 2 + (H - cy) ** 2),
    ])
    r_max_target = corners_r.max() - edge_margin_px

    # Scan angles to find θ_max via the rational model.  Start from 0
    # and increment until the projected pixel distance exceeds r_max_target
    # or the projection goes invalid.
    theta_scan = np.linspace(0.0, np.pi / 2, 500, dtype=np.float64)
    _, r_scan = _rational_project_ray(proj, theta_scan)

    # Stay on the first monotonic branch.  Rational calibrations can contain a
    # denominator pole outside their intended image domain; selecting the last
    # r<target sample after such a fold produces enormous, meaningless
    # coefficients (notably for b6a9 side-wide and tele cameras).
    increasing_step = np.isfinite(np.diff(r_scan)) & (np.diff(r_scan) > 0.0)
    monotonic_prefix = np.concatenate(
        [np.array([True]), np.logical_and.accumulate(increasing_step)]
    )
    valid = monotonic_prefix & np.isfinite(r_scan) & (r_scan < r_max_target)
    if not valid.any():
        raise RuntimeError("No valid projection within image bounds — check pinhole params")

    max_angle = float(theta_scan[valid][-1])

    # ---- Fit angle → pixel distance ----------------------------------------
    theta_fine = np.linspace(0.0, max_angle, n_samples, dtype=np.float64)
    _, r_fine = _rational_project_ray(proj, theta_fine)

    # Drop any NaN/Inf samples (shouldn't happen at well-behaved angles near
    # the center, but defensive).
    good = np.isfinite(r_fine) & (r_fine >= 0)
    theta_good = theta_fine[good]
    r_good = r_fine[good]

    if len(theta_good) < _FTHETA_POLY_DEGREE + 1:
        raise RuntimeError(
            f"Only {len(theta_good)} valid angle samples; need at least "
            f"{_FTHETA_POLY_DEGREE + 1} for a degree-{_FTHETA_POLY_DEGREE} fit"
        )

    angle_to_pixeldist_poly = _fit_monotonic_polynomial(theta_good, r_good, _FTHETA_POLY_DEGREE)

    # ---- Fit pixel distance → angle (inverse) ------------------------------
    # Sample pixel distances uniformly and invert via the rational model as
    # ground truth.  The inverse mapping is r → θ where θ = arctan2(r_pix, f)
    # for the ideal pinhole, but with rational distortion it's non-linear.
    # We can't directly project from pixel distance to angle, so we use the
    # (θ, r) pairs from the forward fit, swap axes, and re-fit.
    pixeldist_to_angle_poly = _fit_monotonic_polynomial(r_good, theta_good, _FTHETA_POLY_DEGREE)

    # ---- Build the FTheta dict in NCore's 8-key convention ------------------
    ftheta_dict: dict = {
        "resolution": pinhole_dict["resolution"],
        "shutter_type": pinhole_dict.get("shutter_type", "ROLLING_TOP_TO_BOTTOM"),
        "principal_point": pinhole_dict["principal_point"],
        "reference_poly": "PIXELDIST_TO_ANGLE",
        "pixeldist_to_angle_poly": pixeldist_to_angle_poly,
        "angle_to_pixeldist_poly": angle_to_pixeldist_poly,
        "max_angle": max_angle,
        # NCore FTheta stores a 3-vector (c, d, e) for affine distortion
        # correction applied before the polynomial.  We set it to identity
        # (no correction) — the polynomial already handles the full mapping.
        "linear_cde": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }

    return ftheta_dict


def compute_ftheta_remap_and_mask(
    ftheta_dict: dict,
    resolution: Optional[tuple[int, int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate an FTheta remap table and valid-pixel mask.

    For each output pixel (u, v) in the FTheta canvas, compute the
    corresponding camera-space ray direction via the FTheta polynomial
    (pixel → angle → ray), then project it back through the *fitted*
    FTheta forward projection to verify the round-trip.

    This is NOT an image remap (input-to-output pixel mapping).  It is a
    validity mask that answers: "does every pixel in the FTheta raster
    project to a consistent ray and re-project to within ε of itself?"

    Parameters
    ----------
    ftheta_dict : dict
        8-key FTheta intrinsics (from ``fit_ftheta_from_opencv_rational``).
    resolution : (W, H) tuple, optional
        Override the canvas resolution.  Defaults to the resolution in
        ``ftheta_dict``.

    Returns
    -------
    valid_mask : np.ndarray of shape (H, W), dtype bool
        True for FTheta pixels whose round-trip error ≤ 0.5 px.
    roundtrip_error : np.ndarray of shape (H, W), dtype float64
        Pixel-level round-trip error map for diagnostic visualization.
    """
    from .ftheta_intrinsics import ftheta_pixels_to_camera_rays
    from .ftheta_projector import FthetaForwardProjector

    if resolution is None:
        res = ftheta_dict["resolution"]
        W, H = int(res[0]), int(res[1])
    else:
        W, H = resolution

    # Step 1: pixel → camera ray (FTheta polynomial inverse)
    rays_hw = ftheta_pixels_to_camera_rays(ftheta_dict)  # (H, W, 3)

    # Step 2: camera ray → pixel (FTheta polynomial forward)
    # For the round-trip test we use the fitted FTheta projector with
    # identity flip (camera-frame world points, no viser convention flip).
    proj_ft = FthetaForwardProjector(ftheta_dict, world_to_camera_flip=np.eye(4))

    # Build world points 5 m along each ray.  With identity flip,
    # world (x, y, z) → cam (x, y, z), so a cam-ray (rx, ry, rz) at
    # distance 5 is world (rx, ry, rz) * 5.
    # But wait — FTheta rays are camera-space directions with +Z forward.
    # c2w_viser = identity → c2w_cv = identity @ diag([1,1,-1,1]).
    # So world (rx, ry, rz) * 5 → cam (rx, ry, -rz) * 5.
    # That's WRONG — rz > 0 would become cam -z → behind camera.
    #
    # Actually, let's trace through more carefully.  With default flip
    # FLIP_VISER_TO_OPENCV = diag([1,1,-1,1]):
    #   c2w_cv = c2w_viser @ diag([1,1,-1,1])
    #   w2c  = inv(c2w_cv)
    #   p_cam = w2c @ p_world
    #
    # With c2w_viser = I:
    #   c2w_cv = diag([1,1,-1,1])
    #   w2c = diag([1,1,-1,1])  (self-inverse)
    #   p_cam = diag([1,1,-1,1]) @ p_world
    #         = (x_world, y_world, -z_world)
    #
    # With identity flip (world_to_camera_flip=np.eye(4)), the projector
    # treats c2w as OpenCV convention (+Y down, +Z forward).
    # A camera ray (rx, ry, rz) with +Z forward, at distance 5 m, is:
    #   camera frame: (5*rx, 5*ry, 5*rz)
    #   world frame with c2w=I: (5*rx, 5*ry, 5*rz) — identical.
    # The projector does: w2c = I, p_cam = I @ p_world → z = 5*rz > 0 ✓.
    world_pts = (rays_hw * 5.0).astype(np.float64)  # (H, W, 3)

    # Flatten for batched projection
    N = H * W
    pts_flat = world_pts.reshape(N, 3).astype(np.float64)
    uv_flat, vis_flat = proj_ft.project_points(pts_flat, np.eye(4, dtype=np.float64))

    # Round-trip error: original pixel vs re-projected pixel
    ys, xs = np.mgrid[0:H, 0:W]
    orig_uv = np.stack([xs.ravel(), ys.ravel()], axis=-1).astype(np.float64)
    error = np.sqrt(np.sum((uv_flat - orig_uv) ** 2, axis=1)).reshape(H, W)

    # Valid mask: round-trip error ≤ 0.5 px AND visible in FTheta.
    # The 0.5 px threshold accounts for float32→float64 quantization in
    # ftheta_pixels_to_camera_rays (float32 rays) and 5th-order polynomial
    # fit residuals.  The median error is ~0.04 px for a well-fitted model.
    valid_mask = (error <= 0.5) & vis_flat.reshape(H, W)

    return valid_mask, error


def compute_opencv_reference_rays(
    pinhole_dict: dict,
    ftheta_dict: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the historical *radial-interpolation* comparison rays.

    .. warning::
       This helper is not a full OpenCV truth oracle.  It collapses every
       azimuth to one radial profile and is therefore blind to tangential,
       thin-prism, and unequal-focal-length error.  New validation must use
       :func:`compute_fullimage_angular_error`.

    For each FTheta pixel, compute the difference between the FTheta ray and
    the OpenCV rational ray.  Returns two (H, W, 3) arrays of camera-space rays
    so the caller can compute angular error per pixel.

    This is the core comparison metric: a perfect FTheta fit would produce
    rays that exactly match the OpenCV rational model's rays at every pixel.
    """
    from .ftheta_intrinsics import ftheta_pixels_to_camera_rays

    # FTheta rays: pixel → ray via FTheta polynomial
    rays_ftheta = ftheta_pixels_to_camera_rays(ftheta_dict)  # (H, W, 3), float32

    # OpenCV rational rays: we compute them by starting from the known
    # mapping: for each pixel, the rational model's inverse gives us a ray.
    # Rather than inverting the rational model ourselves (which requires
    # solving the rational distortion equation), we pre-sample the
    # θ → r mapping much more finely and interpolate.
    #
    # Approach: for each pixel at distance r from principal point,
    # interpolate the angle θ(r) from the finely-sampled rational mapping.
    # The ray direction is then (sin(θ)*du/r, sin(θ)*dv/r, cos(θ)) if
    # r > 0, and (0, 0, 1) at the principal point.
    proj = PinholeForwardProjector(pinhole_dict)
    W = proj.width
    H = proj.height

    # Sample 5000 angles for high-resolution interpolation.
    theta_fine = np.linspace(0.0, ftheta_dict["max_angle"], 5000, dtype=np.float64)
    _, r_fine = _rational_project_ray(proj, theta_fine)

    # For each pixel, compute r and interpolate θ.
    ys, xs = np.mgrid[0:H, 0:W]
    du = xs.astype(np.float64) - proj.cx
    dv = ys.astype(np.float64) - proj.cy
    r_pix = np.sqrt(du * du + dv * dv)

    # Interpolate: for pixels where r exceeds max(r_fine), clip to max angle.
    theta_per_pixel = np.interp(r_pix.ravel(), r_fine, theta_fine).reshape(H, W)

    sin_t = np.sin(theta_per_pixel)
    cos_t = np.cos(theta_per_pixel)
    eps = 1e-12
    norm = np.maximum(r_pix, eps)
    rx = sin_t * du / norm
    ry = sin_t * dv / norm
    rz = cos_t

    # Fix principal point: r=0 → ray = (0, 0, 1)
    at_center = r_pix < eps
    rx[at_center] = 0.0
    ry[at_center] = 0.0
    rz[at_center] = 1.0

    rays_rational = np.stack([rx, ry, rz], axis=-1).astype(np.float32)
    return rays_ftheta, rays_rational


def _ftheta_pixels_to_camera_rays_float64(ftheta_dict: dict) -> np.ndarray:
    """Float64 version of the fitted FTheta pixel-to-ray mapping."""
    resolution = np.asarray(ftheta_dict["resolution"])
    width, height = int(resolution[0]), int(resolution[1])
    cx, cy = np.asarray(ftheta_dict["principal_point"], dtype=np.float64)
    poly = np.asarray(ftheta_dict["pixeldist_to_angle_poly"], dtype=np.float64)
    ys, xs = np.mgrid[0:height, 0:width]
    du = xs.astype(np.float64) - cx
    dv = ys.astype(np.float64) - cy
    pixel_distance = np.hypot(du, dv)
    theta = horner_ascending(poly, pixel_distance)
    direction_scale = np.divide(
        np.sin(theta), pixel_distance,
        out=np.zeros_like(theta), where=pixel_distance > 0.0,
    )
    rays = np.stack(
        [direction_scale * du, direction_scale * dv, np.cos(theta)], axis=-1
    )
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def _ftheta_own_domain_mask(rays: np.ndarray, max_angle: float) -> np.ndarray:
    """Return the strict FTheta domain used by the CUDA forward projector."""
    directions = np.asarray(rays, dtype=np.float64)
    theta = np.arctan2(
        np.linalg.norm(directions[..., :2], axis=-1),
        directions[..., 2],
    )
    return (
        np.isfinite(directions).all(axis=-1)
        & np.isfinite(theta)
        & (theta >= 0.0)
        & (theta < float(max_angle))
    )


def _percentiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            f"{prefix}mean": float("nan"),
            f"{prefix}p50": float("nan"),
            f"{prefix}p95": float("nan"),
            f"{prefix}p99": float("nan"),
            f"{prefix}max": float("nan"),
        }
    return {
        f"{prefix}mean": float(np.mean(finite)),
        f"{prefix}p50": float(np.percentile(finite, 50)),
        f"{prefix}p95": float(np.percentile(finite, 95)),
        f"{prefix}p99": float(np.percentile(finite, 99)),
        f"{prefix}max": float(np.max(finite)),
    }


def compute_fullimage_angular_error(
    pinhole_dict: dict,
    ftheta_dict: dict,
    outer_deg: float = 55.0,
) -> dict[str, float]:
    """Measure a fitted FTheta model against the complete OpenCV model.

    Metrics cover every integer pixel and all azimuths.  Spatial regions use
    the same image half-diagonal normalisation as PIN-AB-1: center is ``r<0.5``
    and periphery is ``r>=0.9``.  ``outer_*`` uses the true camera-ray angle.
    """
    from .opencv_inverse import (
        _distort_normalized,
        opencv_calibration_domain_mask,
        opencv_pixels_to_camera_rays,
    )

    # Compare against the complete first invertible calibration branch.  The
    # production pinhole renderer's icD trust gate is intentionally excluded:
    # carrying it into this oracle would approve the same truncated FOV that
    # the FTheta conversion exists to remove.
    true_rays = opencv_pixels_to_camera_rays(
        pinhole_dict,
        enforce_runtime_trust=False,
    )
    fitted_rays = _ftheta_pixels_to_camera_rays_float64(ftheta_dict)
    if fitted_rays.shape != true_rays.shape:
        raise ValueError(
            f"pinhole/FTheta resolution mismatch: {true_rays.shape} vs {fitted_rays.shape}"
        )

    ftheta_in_domain = _ftheta_own_domain_mask(
        fitted_rays,
        float(ftheta_dict["max_angle"]),
    )
    projector = PinholeForwardProjector(pinhole_dict)
    safe_fitted_z = np.where(
        np.abs(fitted_rays[..., 2]) < 1e-15, 1.0, fitted_rays[..., 2]
    )
    fitted_x = fitted_rays[..., 0] / safe_fitted_z
    fitted_y = fitted_rays[..., 1] / safe_fitted_z
    fitted_forward_valid = opencv_calibration_domain_mask(
        projector, fitted_x, fitted_y
    )
    comparison_valid = (
        np.isfinite(true_rays).all(axis=-1)
        & np.isfinite(fitted_rays).all(axis=-1)
        & ftheta_in_domain
        & fitted_forward_valid
    )
    dot = np.sum(true_rays * fitted_rays, axis=-1)
    angular_error_deg = np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))
    angular_error_deg[~comparison_valid] = np.nan
    true_angle_deg = np.rad2deg(
        np.arctan2(np.linalg.norm(true_rays[..., :2], axis=-1), true_rays[..., 2])
    )

    height, width = angular_error_deg.shape
    cx, cy = np.asarray(pinhole_dict["principal_point"], dtype=np.float64)
    ys, xs = np.mgrid[0:height, 0:width]
    half_diagonal = np.hypot(width / 2.0, height / 2.0)
    radius_normalized = np.hypot(xs - cx, ys - cy) / half_diagonal
    center_mask = radius_normalized < 0.5
    peripheral_mask = radius_normalized >= 0.9
    outer_mask = true_angle_deg >= outer_deg

    metrics = _percentiles(angular_error_deg, "")
    metrics.update(_percentiles(angular_error_deg[center_mask], "center_"))
    metrics.update(_percentiles(angular_error_deg[peripheral_mask], "peripheral_"))
    outer_stats = _percentiles(angular_error_deg[outer_mask], "outer_")
    metrics.update(outer_stats)
    # Preserve the names declared by the plan.
    metrics["mean_deg"] = metrics.pop("mean")
    metrics["p50_deg"] = metrics.pop("p50")
    metrics["p95_deg"] = metrics.pop("p95")
    metrics["p99_deg"] = metrics.pop("p99")
    metrics["max_deg"] = metrics.pop("max")
    for region in ("center", "peripheral", "outer"):
        for stat in ("mean", "p50", "p95", "p99", "max"):
            old_key = f"{region}_{stat}"
            metrics[f"{region}_{stat}_deg"] = metrics.pop(old_key)

    # Pixel-domain error: take each fitted FTheta ray through the exact OpenCV
    # forward equations and compare with the pixel that generated it.
    xd, yd = _distort_normalized(projector, fitted_x, fitted_y)
    reprojected_u = projector.fx * xd + projector.cx
    reprojected_v = projector.fy * yd + projector.cy
    pixel_error = np.hypot(reprojected_u - xs, reprojected_v - ys)
    pixel_error[~comparison_valid] = np.nan
    pixel_stats = _percentiles(pixel_error, "pixel_")
    metrics.update({f"{key}_px": value for key, value in pixel_stats.items()})
    for region, mask in (("center", center_mask), ("peripheral", peripheral_mask)):
        stats = _percentiles(pixel_error[mask], f"{region}_pixel_")
        metrics.update({f"{key}_px": value for key, value in stats.items()})

    # Non-radial representability floor: compare the full oracle with the
    # circular model used by the +X fitter (fx for both image axes and no
    # tangential/thin-prism terms).  This intentionally includes fx != fy.
    radial_only = dict(pinhole_dict)
    fx = float(np.atleast_1d(np.asarray(pinhole_dict["focal_length"]))[0])
    radial_only["focal_length"] = np.array([fx, fx], dtype=np.float64)
    radial_only["tangential_coeffs"] = np.zeros(2, dtype=np.float64)
    radial_only["thin_prism_coeffs"] = np.zeros(4, dtype=np.float64)
    radial_rays = opencv_pixels_to_camera_rays(
        radial_only,
        enforce_runtime_trust=False,
    )
    floor_deg = np.rad2deg(
        np.arccos(np.clip(np.sum(true_rays * radial_rays, axis=-1), -1.0, 1.0))
    )
    floor_stats = _percentiles(floor_deg, "nonradial_floor_")
    metrics.update({f"{key}_deg": value for key, value in floor_stats.items()})

    # One-dimensional forward polynomial residual is reported separately from
    # full-image pixel mismatch so the two failure sources cannot be conflated.
    theta = np.linspace(0.0, float(ftheta_dict["max_angle"]), 20_001)
    _uv, true_radius = _rational_project_ray(projector, theta)
    fitted_radius = horner_ascending(
        np.asarray(ftheta_dict["angle_to_pixeldist_poly"], dtype=np.float64), theta
    )
    forward_error = np.abs(fitted_radius - true_radius)
    forward_stats = _percentiles(forward_error, "forward_poly_")
    metrics.update({f"{key}_px": value for key, value in forward_stats.items()})

    # Keep the three domains separate.  Generic valid/invalid names previously
    # conflated FTheta's own max-angle cone, the OpenCV calibration branch, and
    # their comparison intersection.
    opencv_calibration_domain = np.isfinite(true_rays).all(axis=-1)
    total_pixel_count = int(true_rays.shape[0] * true_rays.shape[1])
    ftheta_own_domain_count = int(np.count_nonzero(ftheta_in_domain))
    opencv_calibration_domain_count = int(
        np.count_nonzero(opencv_calibration_domain)
    )
    comparison_intersection_count = int(np.count_nonzero(comparison_valid))
    metrics["total_pixel_count"] = total_pixel_count
    metrics["ftheta_own_domain_count"] = ftheta_own_domain_count
    metrics["ftheta_own_domain_excluded_count"] = (
        total_pixel_count - ftheta_own_domain_count
    )
    metrics["ftheta_own_domain_coverage"] = float(
        ftheta_own_domain_count / total_pixel_count
    )
    metrics["opencv_calibration_domain_count"] = opencv_calibration_domain_count
    metrics["opencv_calibration_domain_excluded_count"] = (
        total_pixel_count - opencv_calibration_domain_count
    )
    metrics["opencv_calibration_domain_coverage"] = float(
        opencv_calibration_domain_count / total_pixel_count
    )
    metrics["comparison_intersection_count"] = comparison_intersection_count
    metrics["comparison_intersection_excluded_count"] = (
        total_pixel_count - comparison_intersection_count
    )
    metrics["comparison_intersection_coverage"] = float(
        comparison_intersection_count / total_pixel_count
    )
    metrics["comparison_intersection_retention_of_opencv_calibration_domain"] = (
        float(comparison_intersection_count / opencv_calibration_domain_count)
        if opencv_calibration_domain_count
        else float("nan")
    )

    # The exact inverse must return to the generating integer pixel on the
    # physical branch.  Report this separately from fitted-FTheta pixel error
    # so invalid coverage cannot disappear behind percentile filtering.
    true_z = true_rays[..., 2]
    safe_true_z = np.where(np.abs(true_z) < 1e-15, 1.0, true_z)
    true_x = true_rays[..., 0] / safe_true_z
    true_y = true_rays[..., 1] / safe_true_z
    true_xd, true_yd = _distort_normalized(projector, true_x, true_y)
    roundtrip_error = np.hypot(
        projector.fx * true_xd + projector.cx - xs,
        projector.fy * true_yd + projector.cy - ys,
    )
    roundtrip_error[~opencv_calibration_domain] = np.nan
    roundtrip_stats = _percentiles(roundtrip_error, "opencv_roundtrip_")
    metrics.update({f"{key}_px": value for key, value in roundtrip_stats.items()})
    metrics["outer_sample_count"] = int(np.count_nonzero(outer_mask & comparison_valid))
    metrics["outer_available"] = bool(metrics["outer_sample_count"] > 0)
    metrics["max_true_angle_deg"] = float(np.nanmax(true_angle_deg))
    return metrics
