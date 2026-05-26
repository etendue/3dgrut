# SPDX-License-Identifier: Apache-2.0
"""Unit tests for scripts/diagnose_ego_trajectory.py (V3-VIZ.5).

CPU / Mac runnable, no real ckpt required.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _import_diagnose_module():
    """Import scripts/diagnose_ego_trajectory.py as a module."""
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "scripts" / "diagnose_ego_trajectory.py"
    spec = importlib.util.spec_from_file_location("diagnose_ego_trajectory", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_clean_poses(n: int, dx: float = 1.0):
    """N straight-line poses moving along +X at constant dt=100ms with a
    non-identity rotation so frame 0 doesn't false-flag as identity."""
    poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    # Rotate -90deg about Y → camera looks +X (NCore-style). R != I, so even
    # at t=0 the pose differs from the identity matrix.
    R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
    poses[:, :3, :3] = R
    poses[:, 0, 3] = np.arange(n, dtype=np.float32) * dx
    ts = (np.arange(n) * 100_000).astype(np.int64)
    return poses, ts


def test_detect_clean_trajectory_has_no_problems():
    diag = _import_diagnose_module()
    poses, ts = _make_clean_poses(20, dx=2.0)
    r = diag._detect_problems(poses, ts, outlier_k=5.0)
    assert r["n_frames"] == 20
    assert r["is_sorted"] is True
    assert r["n_negative_dt"] == 0
    assert r["n_zero_dt"] == 0
    assert r["outlier_jump_indices"] == []
    assert r["identity_pose_indices"] == []
    assert r["nan_pose_indices"] == []
    assert np.isclose(r["dxy_mean_m"], 2.0, atol=1e-5)
    # Speed = 2 m / 0.1 s = 20 m/s.
    assert np.isclose(r["speed_mean_mps"], 20.0, atol=0.1)


def test_detect_outlier_jump():
    diag = _import_diagnose_module()
    poses, ts = _make_clean_poses(10, dx=1.0)
    poses[5, 0, 3] = 100.0  # inject a 95m jump at frame 5 → direction reverse at 6
    r = diag._detect_problems(poses, ts, outlier_k=5.0)
    # Direction reverses at edges 4→5 (forward jump) and 5→6 (backward jump),
    # so kink is detected at frame 5 (between segments 4→5 and 5→6).
    assert 5 in r["outlier_jump_indices"]
    assert r["max_direction_kink_deg"] > 60.0
    assert r["dxy_max_m"] > 90.0
    assert r["speed_max_mps"] > 100.0  # 95 m / 0.1 s


def test_detect_sharp_turn_kink():
    """Real 90° kink should be flagged as direction-change outlier.

    Trajectory:  (0,0) → (1,0) → (2,0) → (3,0) → (4,0) → (4,1) → (4,2) → ...
    Kink is at frame 4 (between segment 3→4 going +X and 4→5 going +Y).
    """
    diag = _import_diagnose_module()
    n = 10
    poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
    poses[:, :3, :3] = R
    poses[:5, 0, 3] = np.arange(5, dtype=np.float32)   # (0..4, 0)
    poses[5:, 0, 3] = 4.0                              # x stays at 4
    poses[5:, 1, 3] = np.arange(1, n - 4, dtype=np.float32)  # (4, 1), (4, 2), ...
    ts = (np.arange(n) * 100_000).astype(np.int64)
    r = diag._detect_problems(poses, ts, outlier_k=5.0)
    # Kink between segments 3→4 (+X) and 4→5 (+Y), flagged at edge 4.
    assert 4 in r["outlier_jump_indices"]
    assert r["max_direction_kink_deg"] >= 89.0


def test_bimodal_dt_does_not_flag_outliers():
    """Smooth motion at bimodal cadence (33ms/66ms) should NOT flag outliers.

    Regression for the false-positive seen on B3_30k where 119/523 edges were
    flagged just because dt-correlated dxy varies between {33ms-step, 66ms-step}.
    """
    diag = _import_diagnose_module()
    n = 30
    v = 2.0  # m/s constant velocity
    ts = np.zeros(n, dtype=np.int64)
    dt_pattern = [33_000] * 6 + [66_000]  # period-7 cadence pattern
    for i in range(1, n):
        ts[i] = ts[i - 1] + dt_pattern[(i - 1) % 7]
    poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
    poses[:, :3, :3] = R
    poses[:, 0, 3] = (ts.astype(np.float64) * 1e-6 * v).astype(np.float32)
    r = diag._detect_problems(poses, ts, outlier_k=5.0)
    assert r["dt_bimodal"] is True
    assert r["outlier_jump_indices"] == []
    assert np.isclose(r["speed_median_mps"], v, atol=0.2)


def test_detect_unsorted_timestamps():
    diag = _import_diagnose_module()
    poses, ts = _make_clean_poses(5, dx=1.0)
    ts = ts.copy()
    ts[2], ts[3] = ts[3], ts[2]  # swap timestamps
    r = diag._detect_problems(poses, ts, outlier_k=5.0)
    assert r["is_sorted"] is False
    assert r["n_negative_dt"] >= 1


def test_detect_zero_dt():
    diag = _import_diagnose_module()
    poses, ts = _make_clean_poses(5, dx=1.0)
    ts = ts.copy()
    ts[3] = ts[2]  # duplicate timestamp
    r = diag._detect_problems(poses, ts, outlier_k=5.0)
    assert r["n_zero_dt"] == 1


def test_detect_identity_poses():
    diag = _import_diagnose_module()
    poses, ts = _make_clean_poses(5, dx=1.0)
    poses[2] = np.eye(4, dtype=np.float32)  # identity at frame 2 → flagged
    r = diag._detect_problems(poses, ts, outlier_k=5.0)
    assert 2 in r["identity_pose_indices"]
    # Clean poses (with rotation block) are not identity; only the injected one is.
    assert 0 not in r["identity_pose_indices"]


def test_detect_nan_poses():
    diag = _import_diagnose_module()
    poses, ts = _make_clean_poses(5, dx=1.0)
    poses[3, 0, 3] = np.nan
    r = diag._detect_problems(poses, ts, outlier_k=5.0)
    assert 3 in r["nan_pose_indices"]


def test_hypotheses_picks_R1_for_unsorted():
    diag = _import_diagnose_module()
    poses, ts = _make_clean_poses(5, dx=1.0)
    ts = ts.copy()
    ts[1], ts[2] = ts[2], ts[1]
    r = diag._detect_problems(poses, ts, outlier_k=5.0)
    hints = diag._diagnose_hypotheses(r, {"primary_camera_id": "front_wide"})
    assert any("R1" in h for h in hints)


def test_hypotheses_clean_path_emits_residual_hint():
    diag = _import_diagnose_module()
    poses, ts = _make_clean_poses(20, dx=2.0)
    r = diag._detect_problems(poses, ts, outlier_k=5.0)
    hints = diag._diagnose_hypotheses(r, {"primary_camera_id": "front_wide"})
    # Clean dataset still emits an explanatory hint (no auto red flags).
    assert len(hints) >= 1
    assert any("No obvious red flags" in h or "viser" in h for h in hints)
