# SPDX-License-Identifier: Apache-2.0
"""Numpy quaternion / pose helpers for viser scene primitives.

Viser 1.0 expects ``wxyz`` (w, x, y, z) quaternions. Our pose tensors are
C2W [4, 4] matrices (camera-to-world). This file converts between the two
without pulling scipy / kaolin into the viewer.
"""

from __future__ import annotations

import numpy as np


def mat_to_wxyz(c2w: np.ndarray) -> np.ndarray:
    """Convert a 4x4 (or 3x3) rotation matrix to a (w, x, y, z) quaternion.

    Uses the Shepperd / Markley method with sign disambiguation for numerical
    stability: pick the diagonal element with the largest magnitude as the
    pivot. Matches the viser convention used in viser.transforms.SO3.

    Args:
        c2w: ``(4, 4)`` or ``(3, 3)`` ``np.ndarray``.

    Returns:
        ``np.ndarray`` of shape ``(4,)`` dtype float32, ordered ``(w, x, y, z)``.
    """
    R = np.asarray(c2w)[:3, :3].astype(np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float32)
    # Canonicalize sign: pick the hemisphere with w >= 0 so two equivalent
    # quaternions don't oscillate across frames.
    if q[0] < 0.0:
        q = -q
    # Normalize to unit length; round-off from large matrices can drift.
    q /= max(np.linalg.norm(q), 1e-12)
    return q
