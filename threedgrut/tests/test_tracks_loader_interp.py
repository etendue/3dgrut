# SPDX-License-Identifier: Apache-2.0
"""A2 — per-camera END-timestamp cuboid pose refinement (interp mode).

The v5 A2 fix: ``load_tracks_from_ncore_cuboids`` historically snapped every
camera frame to the *nearest* cuboid observation (50ms tolerance) on the ref
camera's timeline — cross cameras whose exposure differs by up to ~100ms from
the ref camera render the cuboid at a stale pose (≈2m at 20 m/s). The
``pose_time_mode="interp"`` path interpolates the observation trajectory
(translation lerp + rotation slerp) to the exact query timestamp instead.

Default (``pose_time_mode="nearest"``) must stay byte-identical — pinned here
by comparing against an explicit-default call; the pre-existing rotation tests
in test_tracks_loader_rotation.py are the wider regression net.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from unittest.mock import MagicMock

from threedgrut.datasets.tracks_loader import (
    euler_xyz_to_rotation_matrix,
    interp_pose_to_ts,
    load_tracks_from_ncore_cuboids,
)


# --- fixtures (mirror test_tracks_loader_rotation.py) -----------------------

class _FakeBBox3:
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
        assert target_frame == "world"
        return self


class _FakeLoader:
    def __init__(self, observations):
        self._obs = observations
        self.pose_graph = MagicMock()


    def get_cuboid_track_observations(self):
        return iter(self._obs)


def _pose(x: float, yaw: float) -> np.ndarray:
    p = np.eye(4)
    p[:3, :3] = euler_xyz_to_rotation_matrix(0.0, 0.0, yaw)
    p[0, 3] = x
    return p


def _uniform_track_obs(track_id="t1", speed=20.0, yaw_rate=0.2, n=11, dt_us=100_000):
    """n obs at 10Hz: x = speed * t, yaw = yaw_rate * t (t seconds)."""
    obs = []
    for i in range(n):
        ts = i * dt_us
        t_s = ts * 1e-6
        obs.append(_FakeObs(
            track_id, "automobile", ts,
            _FakeBBox3((speed * t_s, 0.0, 1.0), (4.5, 2.0, 1.7),
                       (0.0, 0.0, yaw_rate * t_s)),
        ))
    return obs


# --- interp_pose_to_ts pure function ----------------------------------------

def test_interp_linear_translation_and_yaw():
    """Uniform motion: query between obs → exact lerp position + slerp yaw."""
    obs_ts = np.asarray([0, 1_000_000], dtype=np.int64)
    poses = np.stack([_pose(0.0, 0.0), _pose(20.0, 0.2)])
    out = interp_pose_to_ts(obs_ts, poses, 400_000, max_extrapolation_us=50_000)
    assert out is not None
    assert abs(out[0, 3] - 8.0) < 1e-4
    expected_R = euler_xyz_to_rotation_matrix(0.0, 0.0, 0.08)
    assert np.allclose(out[:3, :3], expected_R, atol=1e-5)


def test_interp_exact_obs_timestamp_returns_that_pose():
    obs_ts = np.asarray([0, 1_000_000], dtype=np.int64)
    poses = np.stack([_pose(0.0, 0.0), _pose(20.0, 0.2)])
    out = interp_pose_to_ts(obs_ts, poses, 1_000_000, max_extrapolation_us=0)
    assert np.allclose(out, poses[1], atol=1e-7)


def test_interp_out_of_range_clamps_within_tolerance():
    """Query 30ms past the last obs (tol 50ms) → constant-extrapolate last pose."""
    obs_ts = np.asarray([0, 1_000_000], dtype=np.int64)
    poses = np.stack([_pose(0.0, 0.0), _pose(20.0, 0.2)])
    out = interp_pose_to_ts(obs_ts, poses, 1_030_000, max_extrapolation_us=50_000)
    assert out is not None
    assert np.allclose(out, poses[1], atol=1e-7)


def test_interp_out_of_range_beyond_tolerance_returns_none():
    obs_ts = np.asarray([0, 1_000_000], dtype=np.int64)
    poses = np.stack([_pose(0.0, 0.0), _pose(20.0, 0.2)])
    assert interp_pose_to_ts(obs_ts, poses, 1_060_000, max_extrapolation_us=50_000) is None
    assert interp_pose_to_ts(obs_ts, poses, -60_000, max_extrapolation_us=50_000) is None


def test_interp_quat_double_cover_takes_short_arc():
    """yaw +3.0 → -3.0 rad crosses ±π; slerp must go through π (short arc),
    not spin backwards through 0."""
    obs_ts = np.asarray([0, 1_000_000], dtype=np.int64)
    poses = np.stack([_pose(0.0, 3.0), _pose(0.0, -3.0)])
    out = interp_pose_to_ts(obs_ts, poses, 500_000, max_extrapolation_us=0)
    expected_R = euler_xyz_to_rotation_matrix(0.0, 0.0, math.pi)
    assert np.allclose(out[:3, :3], expected_R, atol=1e-4)


def test_interp_single_obs_within_tolerance():
    obs_ts = np.asarray([500_000], dtype=np.int64)
    poses = _pose(5.0, 0.1)[None]
    out = interp_pose_to_ts(obs_ts, poses, 520_000, max_extrapolation_us=50_000)
    assert np.allclose(out, poses[0], atol=1e-7)
    assert interp_pose_to_ts(obs_ts, poses, 600_000, max_extrapolation_us=50_000) is None


# --- load_tracks_from_ncore_cuboids pose_time_mode --------------------------

def test_interp_mode_fixes_known_camera_offset():
    """Camera ts = obs grid + 40ms. nearest → 0.8m stale error @20 m/s;
    interp → exact position (uniform motion ⇒ lerp is exact)."""
    obs = _uniform_track_obs()
    cam_ts = np.asarray([40_000, 140_000, 240_000], dtype=np.int64)

    out_nearest = load_tracks_from_ncore_cuboids(
        _FakeLoader(obs), cam_ts, pose_time_mode="nearest")
    out_interp = load_tracks_from_ncore_cuboids(
        _FakeLoader(obs), cam_ts, pose_time_mode="interp")

    # nearest snaps query 40ms → obs at 0ms → x=0 (0.8m stale)
    x_nearest = out_nearest["t1"]["poses"][0, 0, 3].item()
    assert abs(x_nearest - 0.0) < 1e-5
    # interp lands on the true position 20 m/s * 0.04 s = 0.8
    x_interp = out_interp["t1"]["poses"][0, 0, 3].item()
    assert abs(x_interp - 0.8) < 1e-4
    # yaw likewise: 0.2 rad/s * 0.04 s = 0.008
    R_interp = out_interp["t1"]["poses"][0, :3, :3].numpy()
    assert np.allclose(
        R_interp, euler_xyz_to_rotation_matrix(0.0, 0.0, 0.008), atol=1e-5)
    # all frames active in both modes (40ms < 50ms tolerance)
    assert out_interp["t1"]["frame_info"].numpy().all()
    assert out_nearest["t1"]["frame_info"].numpy().all()


def test_default_mode_is_nearest_and_identical():
    """Omitting pose_time_mode ≡ pose_time_mode='nearest' (byte-identical)."""
    obs = _uniform_track_obs()
    cam_ts = np.asarray([40_000, 140_000, 940_000], dtype=np.int64)
    out_default = load_tracks_from_ncore_cuboids(_FakeLoader(obs), cam_ts)
    out_nearest = load_tracks_from_ncore_cuboids(
        _FakeLoader(obs), cam_ts, pose_time_mode="nearest")
    for tid in out_default:
        assert torch.equal(out_default[tid]["poses"], out_nearest[tid]["poses"])
        assert torch.equal(out_default[tid]["frame_info"], out_nearest[tid]["frame_info"])
        assert torch.equal(out_default[tid]["cam_timestamps_us"],
                           out_nearest[tid]["cam_timestamps_us"])


def test_interp_mode_inactive_outside_obs_range():
    """Query 200ms past the last obs (tol 50ms) → frame inactive; track with
    some frames in range keeps those active."""
    obs = _uniform_track_obs(n=3)  # obs at 0 / 100ms / 200ms
    cam_ts = np.asarray([100_000, 400_000], dtype=np.int64)
    out = load_tracks_from_ncore_cuboids(
        _FakeLoader(obs), cam_ts, pose_time_mode="interp")
    active = out["t1"]["frame_info"].numpy()
    assert active.tolist() == [True, False]
    # inactive frame keeps the eye(4) sentinel (same convention as nearest)
    assert np.allclose(out["t1"]["poses"][1].numpy(), np.eye(4))


def test_interp_mode_unknown_mode_raises():
    obs = _uniform_track_obs(n=2)
    cam_ts = np.asarray([0], dtype=np.int64)
    try:
        load_tracks_from_ncore_cuboids(
            _FakeLoader(obs), cam_ts, pose_time_mode="banana")
        raised = False
    except ValueError:
        raised = True
    assert raised
