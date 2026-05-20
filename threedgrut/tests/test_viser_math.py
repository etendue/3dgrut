"""Task D contract tests — mat_to_wxyz (Stage 8 viewer's pose→quaternion)."""
from __future__ import annotations

import numpy as np
import pytest

from threedgrut_playground.utils.viser_math import mat_to_wxyz


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Inverse of mat_to_wxyz (wxyz → 3x3 rotation matrix), for round-trip check."""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array(
        [[1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
         [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
         [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]],
        dtype=np.float64,
    )


def test_identity_pose_yields_unit_quat():
    q = mat_to_wxyz(np.eye(4))
    assert q.shape == (4,)
    assert q.dtype == np.float32
    np.testing.assert_allclose(q, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                                atol=1e-6)


def test_90deg_x_rotation():
    """Rotating about +x by 90° → q = (cos45°, sin45°, 0, 0)."""
    R = np.eye(4, dtype=np.float64)
    R[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    q = mat_to_wxyz(R)
    np.testing.assert_allclose(
        q, np.array([0.7071068, 0.7071068, 0.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )


def test_roundtrip_random_rotations():
    """For 50 random rotations, mat→wxyz→mat reconstructs within float tol."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        # Sample uniform-random rotation via QR of random gaussian matrix.
        Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = Q
        q = mat_to_wxyz(pose)
        R_back = _quat_to_mat(q.astype(np.float64))
        np.testing.assert_allclose(R_back, Q, atol=1e-5)


def test_canonical_sign_positive_w():
    """Two quaternions q and -q are equivalent rotations; ensure we always
    pick the q with w >= 0 to avoid sign flipping between frames."""
    # 180° rotation about z (boundary case where sign can flip).
    R = np.eye(4)
    R[:3, :3] = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    q = mat_to_wxyz(R)
    # The canonical 180°-z quaternion is (0, 0, 0, 1); allow w very near 0.
    assert q[0] >= 0.0
