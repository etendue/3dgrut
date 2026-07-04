# SPDX-License-Identifier: Apache-2.0
"""Shared numpy helpers for forward-projection modules (FTheta + Pinhole).

Extracted from ``ftheta_projector.py`` so the new ``pinhole_projector.py``
(and any future camera model) can reuse the same subdivision logic without
crossing module boundaries.

Pure numpy; no torch, no viser, no kaolin — Mac-testable.
"""

from __future__ import annotations

import numpy as np


def horner_ascending(poly: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate ``p(x) = poly[0] + poly[1]*x + poly[2]*x^2 + ...``

    NCore stores polynomial coefficients in **ascending** order; Horner
    iterates from highest-degree index down to 0 for numerical stability.
    """
    out = np.zeros_like(x, dtype=np.float64)
    for k in range(len(poly) - 1, -1, -1):
        out = out * x + poly[k]
    return out


def subdivide_polyline(pl: np.ndarray, n: int) -> np.ndarray:
    """Insert ``(n-1)`` intermediate vertices on each segment of a polyline.

    ``M``-vertex input → ``(1 + (M-1)*n)``-vertex output. ``n=1`` returns the
    input unchanged. Endpoints are preserved exactly.

    Used so a 3D line that projects to a *curve* under fisheye (FTheta) is
    drawn with smooth tangents instead of straight chord shortcuts. For
    pinhole the curvature is zero, so callers should pass small ``n``.
    """
    if n < 1:
        raise ValueError(f"subdivide_n must be >= 1; got {n}")
    if n == 1:
        return pl.copy()
    M = pl.shape[0]
    if M < 2:
        return pl.copy()
    t = np.linspace(0.0, 1.0, n, endpoint=False)  # (n,)
    a = pl[:-1]  # (M-1, 3)
    b = pl[1:]  # (M-1, 3)
    seg = a[:, None, :] + (b - a)[:, None, :] * t[None, :, None]  # (M-1, n, 3)
    flat = seg.reshape(-1, 3)  # ((M-1)*n, 3)
    return np.concatenate([flat, pl[-1:]], axis=0)  # +1 endpoint
