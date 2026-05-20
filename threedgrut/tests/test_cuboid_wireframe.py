"""Task E contract tests — UNIT_CUBE_EDGES + cuboid_world_edges."""
from __future__ import annotations

import numpy as np
import pytest

from threedgrut_playground.utils.cuboid import (
    UNIT_CUBE_EDGES,
    class_color,
    cuboid_world_edges,
    instance_color,
)


def test_unit_cube_edges_shape():
    assert UNIT_CUBE_EDGES.shape == (12, 2, 3)
    assert UNIT_CUBE_EDGES.dtype == np.float32


def test_unit_cube_vertex_range():
    """Each vertex coord should be exactly +/-0.5 (unit cube centered at origin)."""
    vals = np.unique(UNIT_CUBE_EDGES.reshape(-1))
    np.testing.assert_array_equal(np.sort(vals), np.array([-0.5, 0.5]))


def test_unit_cube_edges_unique_pairs():
    """12 edges, each connecting two distinct vertices. Each undirected pair
    should appear exactly once."""
    pairs = set()
    for a, b in UNIT_CUBE_EDGES:
        ta = tuple(a.tolist())
        tb = tuple(b.tolist())
        pair = frozenset([ta, tb])
        assert pair not in pairs, f"duplicate edge {pair}"
        pairs.add(pair)
    assert len(pairs) == 12


def test_world_edges_identity_pose_unit_size():
    """pose=I + size=(1,1,1) → unchanged unit cube edges."""
    pose = np.eye(4, dtype=np.float32)
    size = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    out = cuboid_world_edges(pose, size)
    np.testing.assert_allclose(out, UNIT_CUBE_EDGES, atol=1e-6)


def test_world_edges_translation_only():
    """Pose pure translate → all 8 vertices shifted by t, shape preserved."""
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [10.0, 20.0, 30.0]
    size = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    out = cuboid_world_edges(pose, size)
    expected = UNIT_CUBE_EDGES + np.array([10.0, 20.0, 30.0])
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_world_edges_size_scale():
    """size=(Lx, Ly, Lz) scales each axis independently."""
    pose = np.eye(4, dtype=np.float32)
    size = np.array([2.0, 1.0, 3.0], dtype=np.float32)
    out = cuboid_world_edges(pose, size)
    # Bounding extent should be [-1, 1]×[-0.5, 0.5]×[-1.5, 1.5].
    pts = out.reshape(-1, 3)
    np.testing.assert_allclose(pts[:, 0].min(), -1.0, atol=1e-6)
    np.testing.assert_allclose(pts[:, 0].max(),  1.0, atol=1e-6)
    np.testing.assert_allclose(pts[:, 1].min(), -0.5, atol=1e-6)
    np.testing.assert_allclose(pts[:, 1].max(),  0.5, atol=1e-6)
    np.testing.assert_allclose(pts[:, 2].min(), -1.5, atol=1e-6)
    np.testing.assert_allclose(pts[:, 2].max(),  1.5, atol=1e-6)


def test_world_edges_rotation_z90():
    """Rotate cuboid 90° about z; x-extent should swap with y-extent."""
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    size = np.array([4.0, 2.0, 1.0], dtype=np.float32)
    out = cuboid_world_edges(pose, size)
    pts = out.reshape(-1, 3)
    # After +z 90°, world-x range becomes [-1, 1] (was y-extent / 2 = 1)
    # and world-y range becomes [-2, 2] (was x-extent / 2 = 2).
    np.testing.assert_allclose(pts[:, 0].max() - pts[:, 0].min(), 2.0, atol=1e-5)
    np.testing.assert_allclose(pts[:, 1].max() - pts[:, 1].min(), 4.0, atol=1e-5)


def test_class_color_known_and_unknown():
    auto = class_color("automobile")
    assert isinstance(auto, tuple) and len(auto) == 3
    assert all(0.0 <= c <= 1.0 for c in auto)
    # Unknown class falls through to gray default.
    assert class_color("totally_made_up_class") == class_color("unknown")


def test_instance_color_deterministic():
    """Same track_id yields same color across calls."""
    a = instance_color("track_42")
    b = instance_color("track_42")
    c = instance_color("track_43")
    assert a == b
    assert a != c  # extremely unlikely collision for these two ids
    assert all(0.0 <= ch <= 1.0 for ch in a)
