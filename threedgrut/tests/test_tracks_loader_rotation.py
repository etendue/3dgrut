# SPDX-License-Identifier: Apache-2.0
"""T8/B3 Phase E.2 — tracks_loader bbox3.rot integration.

NCore ``BBox3.rot`` is intrinsic XYZ Euler radians (probe-confirmed 2026-05-25:
rz spans ±π for vehicle yaw, rx/ry ≈ 0 on flat ground). Phase B baseline
shipped ``pose[:3,:3] = I``, dropping all car/truck rotations. This module
pins:

  * ``euler_xyz_to_rotation_matrix`` matches scipy's intrinsic XYZ exactly
    for selected angles (identity, pure-yaw 90°/180°, mixed small angles).
  * ``load_tracks_from_ncore_cuboids`` writes a full SE(3) pose containing
    both centroid translation AND rotation derived from bbox.rot.
"""

from __future__ import annotations

import math
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from threedgrut.datasets.tracks_loader import (
    euler_xyz_to_rotation_matrix,
    load_tracks_from_ncore_cuboids,
)

# -----------------------------------------------------------------------------
# euler_xyz_to_rotation_matrix
# -----------------------------------------------------------------------------


def test_euler_xyz_identity():
    R = euler_xyz_to_rotation_matrix(0.0, 0.0, 0.0)
    assert np.allclose(R, np.eye(3), atol=1e-12)


def test_euler_xyz_yaw_90_degrees():
    """rz = π/2 → rotates X-axis into Y-axis (right-hand rule about Z)."""
    R = euler_xyz_to_rotation_matrix(0.0, 0.0, math.pi / 2.0)
    expected = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(R, expected, atol=1e-10)


def test_euler_xyz_yaw_180_degrees():
    """rz = π → X and Y flipped, Z unchanged."""
    R = euler_xyz_to_rotation_matrix(0.0, 0.0, math.pi)
    expected = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(R, expected, atol=1e-10)


def test_euler_xyz_pitch_90_degrees():
    """ry = π/2 → +X axis maps to -Z axis (intrinsic XYZ)."""
    R = euler_xyz_to_rotation_matrix(0.0, math.pi / 2.0, 0.0)
    # R @ [1,0,0]^T should be [0, 0, -1]
    out = R @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(out, np.array([0.0, 0.0, -1.0]), atol=1e-10)


def test_euler_xyz_roll_90_degrees():
    """rx = π/2 → +Y axis maps to +Z axis."""
    R = euler_xyz_to_rotation_matrix(math.pi / 2.0, 0.0, 0.0)
    out = R @ np.array([0.0, 1.0, 0.0])
    assert np.allclose(out, np.array([0.0, 0.0, 1.0]), atol=1e-10)


def test_euler_xyz_small_angles_skew_approximation():
    """For small angles, R ≈ I + skew(ω) where ω = (rx, ry, rz)."""
    rx, ry, rz = 0.00418, -0.00045, -0.00587  # actual NCore probe value (obs 0)
    R = euler_xyz_to_rotation_matrix(rx, ry, rz)
    skew = np.array(
        [
            [0.0, -rz, ry],
            [rz, 0.0, -rx],
            [-ry, rx, 0.0],
        ]
    )
    expected = np.eye(3) + skew
    # 1e-4 tolerance: I + skew(ω) is the first-order term of the exponential map.
    # For |ω| ≈ 0.007 rad, second-order error ≈ 0.5 |ω|² ≈ 2.5e-5, with mixed
    # rx*ry cross terms adding similar magnitude → atol 1e-4 covers it.
    assert np.allclose(R, expected, atol=1e-4)


def test_euler_xyz_mixed_angles_matches_extrinsic_xyz_reference():
    """Pin extrinsic-xyz semantics (scipy lowercase "xyz") for non-degenerate
    mixed angles. Reference matrix computed manually as Rz(rz) · Ry(ry) · Rx(rx).

    For (rx=0.1, ry=0.2, rz=0.3) the intrinsic-XYZ vs extrinsic-xyz matrices
    differ by ≈ 6 % in some elements — large enough to catch silent convention
    swaps in future refactors.
    """
    rx, ry, rz = 0.1, 0.2, 0.3
    # Hand-computed Rz · Ry · Rx (extrinsic xyz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    reference = Rz @ Ry @ Rx
    actual = euler_xyz_to_rotation_matrix(rx, ry, rz)
    assert np.allclose(actual, reference, atol=1e-12)


def test_euler_xyz_orthogonal_determinant_one():
    """All outputs should be proper rotation matrices: R^T R = I, det(R) = +1."""
    for rx, ry, rz in [
        (0.0, 0.0, 0.0),
        (0.1, 0.2, 0.3),
        (0.0, 0.0, math.pi / 3),
        (math.pi / 4, 0.0, math.pi / 6),
        (0.01, -0.02, 3.1),  # NCore-like car yaw
    ]:
        R = euler_xyz_to_rotation_matrix(rx, ry, rz)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10), f"R^T R != I for ({rx}, {ry}, {rz})"
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10), f"det(R) != 1 for ({rx}, {ry}, {rz})"


# -----------------------------------------------------------------------------
# load_tracks_from_ncore_cuboids — Phase E.2 pose-with-rotation
# -----------------------------------------------------------------------------


class _FakeBBox3:
    """Mock NCore BBox3 (centroid + dim + rot tuple)."""

    def __init__(self, centroid, dim, rot):
        self.centroid = tuple(centroid)
        self.dim = tuple(dim)
        self.rot = tuple(rot)


class _FakeObs:
    def __init__(self, track_id, class_id, ts_us, bbox):
        self.track_id = track_id
        self.class_id = class_id
        self.timestamp_us = ts_us
        self.bbox3 = bbox

    def transform(self, target_frame, ts_int, pose_graph):
        """Stub: world-frame obs == rig-frame obs (poses already in target frame)."""
        assert target_frame == "world"
        return self  # delegate; tests build the rig-frame bbox already in world


class _FakeLoader:
    def __init__(self, observations):
        self._obs = observations
        self.pose_graph = MagicMock()

    def get_cuboid_track_observations(self):
        return iter(self._obs)


def test_load_tracks_writes_rotation_into_pose():
    """A car at world (10, 5, 1.6) with yaw = π/2 produces pose with Rz(π/2)."""
    bbox = _FakeBBox3(
        centroid=(10.0, 5.0, 1.6),
        dim=(4.5, 2.0, 1.7),
        rot=(0.0, 0.0, math.pi / 2.0),
    )
    obs = _FakeObs(track_id="42", class_id="automobile", ts_us=1_000_000, bbox=bbox)
    loader = _FakeLoader([obs])
    ts = np.asarray([1_000_000], dtype=np.int64)
    out = load_tracks_from_ncore_cuboids(loader, ts)

    assert "42" in out
    pose = out["42"]["poses"][0].numpy()  # [4, 4]
    assert pose.shape == (4, 4)
    # translation
    assert np.allclose(pose[:3, 3], np.array([10.0, 5.0, 1.6]), atol=1e-5)
    # rotation: yaw π/2 → Rz(π/2)
    expected_R = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(pose[:3, :3], expected_R, atol=1e-5)
    assert pose[3, 3] == 1.0


def test_load_tracks_identity_rotation_for_zero_euler():
    """rot = (0, 0, 0) → pose rotation block is exactly I."""
    bbox = _FakeBBox3(centroid=(1.0, 2.0, 3.0), dim=(4.0, 2.0, 1.5), rot=(0.0, 0.0, 0.0))
    obs = _FakeObs(track_id="7", class_id="automobile", ts_us=0, bbox=bbox)
    out = load_tracks_from_ncore_cuboids(_FakeLoader([obs]), np.asarray([0], dtype=np.int64))
    pose = out["7"]["poses"][0].numpy()
    assert np.allclose(pose[:3, :3], np.eye(3), atol=1e-6)
    assert np.allclose(pose[:3, 3], [1.0, 2.0, 3.0], atol=1e-6)


def test_load_tracks_filter_inactive_frame_keeps_identity():
    """Frame outside time tolerance stays with the np.eye(4) initial pose."""
    bbox = _FakeBBox3(centroid=(5.0, 0.0, 1.0), dim=(4.0, 2.0, 1.5), rot=(0.0, 0.0, math.pi))
    obs = _FakeObs(track_id="9", class_id="automobile", ts_us=1_000_000, bbox=bbox)
    # cam_ts = [0us, 1Mus, 10Mus]; obs at 1Mus is in range, 0us and 10Mus are >50ms off
    ts = np.asarray([0, 1_000_000, 10_000_000], dtype=np.int64)
    out = load_tracks_from_ncore_cuboids(
        _FakeLoader([obs]),
        ts,
        time_tolerance_us=50_000,
    )
    poses = out["9"]["poses"].numpy()  # [3, 4, 4]
    active = out["9"]["frame_info"].numpy()
    assert active.tolist() == [False, True, False]
    # frame 1 (active) has the real pose with yaw π
    expected_R_pi = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
    assert np.allclose(poses[1, :3, :3], expected_R_pi, atol=1e-5)
    # frames 0 and 2 (inactive) keep the np.eye(4) sentinel (Bug E2 territory —
    # this test pins the *current* behaviour so E.3 can change it deliberately).
    assert np.allclose(poses[0, :3, :3], np.eye(3))
    assert np.allclose(poses[2, :3, :3], np.eye(3))


def test_load_tracks_drops_track_with_no_active_frames():
    """All cam_ts outside tolerance → track dropped from output."""
    bbox = _FakeBBox3(centroid=(0.0, 0.0, 0.0), dim=(4.0, 2.0, 1.5), rot=(0.0, 0.0, 0.0))
    obs = _FakeObs(track_id="z", class_id="automobile", ts_us=50_000_000, bbox=bbox)
    ts = np.asarray([0, 1_000_000], dtype=np.int64)
    out = load_tracks_from_ncore_cuboids(_FakeLoader([obs]), ts, time_tolerance_us=10_000)
    assert "z" not in out


def test_load_tracks_class_filter():
    """Non-vehicle classes are dropped (T4.5 scope = automobile/heavy_truck/bus)."""
    bbox_car = _FakeBBox3((0.0, 0.0, 0.0), (4.0, 2.0, 1.5), (0.0, 0.0, 0.0))
    bbox_ped = _FakeBBox3((1.0, 0.0, 0.0), (0.5, 0.5, 1.7), (0.0, 0.0, 0.0))
    obs_car = _FakeObs("car1", "automobile", 0, bbox_car)
    obs_ped = _FakeObs("ped1", "person", 0, bbox_ped)
    out = load_tracks_from_ncore_cuboids(
        _FakeLoader([obs_car, obs_ped]),
        np.asarray([0], dtype=np.int64),
    )
    assert "car1" in out
    assert "ped1" not in out


def test_load_tracks_pose_is_consistent_se3():
    """Round-trip: world point at the cuboid origin should map back to (0,0,0)
    in object-local frame via pose^{-1}."""
    bbox = _FakeBBox3(
        centroid=(20.0, -3.0, 1.5),
        dim=(4.5, 2.0, 1.7),
        rot=(0.01, -0.005, 1.234),  # representative car yaw
    )
    obs = _FakeObs("t", "automobile", 0, bbox)
    out = load_tracks_from_ncore_cuboids(_FakeLoader([obs]), np.asarray([0], dtype=np.int64))
    pose = out["t"]["poses"][0].numpy()
    centroid_world = np.array([20.0, -3.0, 1.5, 1.0])
    object_local = np.linalg.inv(pose) @ centroid_world
    assert np.allclose(object_local[:3], np.zeros(3), atol=1e-5)


def test_load_tracks_rot_rotates_local_into_world():
    """A point along the cuboid's +X (length) axis with yaw=π/2 should land
    along world +Y (since the cuboid's local +X is rotated to world +Y)."""
    bbox = _FakeBBox3(
        centroid=(0.0, 0.0, 0.0),
        dim=(4.0, 2.0, 1.5),
        rot=(0.0, 0.0, math.pi / 2.0),
    )
    obs = _FakeObs("t", "automobile", 0, bbox)
    out = load_tracks_from_ncore_cuboids(_FakeLoader([obs]), np.asarray([0], dtype=np.int64))
    pose = out["t"]["poses"][0].numpy()
    R = pose[:3, :3]
    # local (+1, 0, 0) → world should be (0, +1, 0) after yaw 90°
    out_world = R @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(out_world, np.array([0.0, 1.0, 0.0]), atol=1e-5)
