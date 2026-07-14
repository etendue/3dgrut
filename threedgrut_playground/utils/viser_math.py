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


def wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    """Convert a unit ``(w, x, y, z)`` quaternion to a homogeneous 4x4 matrix."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"q must have shape (4,), got {q.shape}")
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return out


def slerp_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Shortest-arc spherical interpolation between normalized wxyz quaternions."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    if q0.shape != (4,) or q1.shape != (4,):
        raise ValueError(f"q0/q1 must have shape (4,), got {q0.shape}/{q1.shape}")
    q0 = q0 / max(np.linalg.norm(q0), 1e-12)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        out = (1.0 - float(alpha)) * q0 + float(alpha) * q1
        return out / max(np.linalg.norm(out), 1e-12)
    theta = float(np.arccos(dot))
    out = (
        np.sin((1.0 - float(alpha)) * theta) * q0
        + np.sin(float(alpha) * theta) * q1
    ) / np.sin(theta)
    return out / max(np.linalg.norm(out), 1e-12)
