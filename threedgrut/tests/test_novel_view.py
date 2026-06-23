# SPDX-License-Identifier: Apache-2.0
"""T8.5.3 unit tests for novel-view pose perturbations.

Pure math; CPU only; runs in <50ms.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from threedgrut.utils.novel_view import (
    LEGACY_NOVEL_AVG_MODES,
    NOVEL_VIEW_MODES,
    perturb_c2w,
    perturb_shutter_pair,
    perturb_batch_shutter_pair_torch,
)


def _identity_c2w():
    """c2w = identity (camera at origin, right=X, down=Y, forward=Z)."""
    return np.eye(4, dtype=np.float64)


def test_legacy_avg_modes_frozen_at_four():
    """E1.1: mean_novel_lpips_avg 历史口径永远只聚合这 4 档（B3 锚 0.5962）。"""
    assert LEGACY_NOVEL_AVG_MODES == (
        "lateral_1m", "lateral_2m", "yaw_5deg", "yaw_10deg",
    )


def test_modes_constant_has_all_eight_modes():
    """E1.1 加 lateral_3m/6m 外推档 + PR #34 road off-track 加 yaw_30/60deg。
    前 4 元素保持历史顺序（LEGACY avg 锚），随后是 E1.1 两档与 off-track 两档。"""
    assert NOVEL_VIEW_MODES == (
        "lateral_1m", "lateral_2m", "yaw_5deg", "yaw_10deg",
        "lateral_3m", "lateral_6m",
        "yaw_30deg", "yaw_60deg",
    )
    assert NOVEL_VIEW_MODES[:4] == LEGACY_NOVEL_AVG_MODES


def test_lateral_3m_is_triple_of_1m():
    m = _identity_c2w()
    m[:3, 3] = [10.0, 20.0, 30.0]
    delta1 = perturb_c2w(m, "lateral_1m")[:3, 3] - m[:3, 3]
    delta3 = perturb_c2w(m, "lateral_3m")[:3, 3] - m[:3, 3]
    assert np.allclose(delta3, 3.0 * delta1)


def test_lateral_6m_shifts_along_right_axis_rotation_unchanged():
    m = _identity_c2w()
    out = perturb_c2w(m, "lateral_6m")
    assert np.allclose(out[:3, 3], [6.0, 0.0, 0.0])
    assert np.allclose(out[:3, :3], np.eye(3))


def test_lateral_6m_shutter_pair_rigid():
    """新档必须走 perturb_shutter_pair 的通用 lateral 分支：start/end 同 delta。"""
    start = _identity_c2w()
    end = _identity_c2w()
    end[:3, 3] = [0.0, 0.0, 0.01]
    R_end = np.array([
        [np.cos(0.01), 0.0, np.sin(0.01)],
        [0.0, 1.0, 0.0],
        [-np.sin(0.01), 0.0, np.cos(0.01)],
    ])
    end[:3, :3] = R_end
    new_s, new_e = perturb_shutter_pair(start, end, "lateral_6m")
    delta_s = new_s[:3, 3] - start[:3, 3]
    delta_e = new_e[:3, 3] - end[:3, 3]
    assert np.allclose(delta_s, delta_e, atol=1e-9)
    assert np.allclose(np.linalg.norm(delta_s), 6.0)


def test_lateral_1m_shifts_position_along_right_axis():
    m = _identity_c2w()
    out = perturb_c2w(m, "lateral_1m")
    # right axis of identity c2w is +X; expect position +1 m in X.
    assert np.allclose(out[:3, 3], [1.0, 0.0, 0.0])
    # rotation unchanged
    assert np.allclose(out[:3, :3], np.eye(3))


def test_lateral_2m_double_of_lateral_1m_translation():
    m = _identity_c2w()
    m[:3, 3] = [10.0, 20.0, 30.0]  # nontrivial anchor position
    out1 = perturb_c2w(m, "lateral_1m")
    out2 = perturb_c2w(m, "lateral_2m")
    delta1 = out1[:3, 3] - m[:3, 3]
    delta2 = out2[:3, 3] - m[:3, 3]
    assert np.allclose(delta2, 2.0 * delta1)


def test_lateral_follows_camera_right_axis_when_rotated():
    """Camera yaw'd 90° CCW (looking +X instead of +Z): right axis should
    now be -Z in world; lateral_1m must shift -Z by 1m."""
    # Build c2w that looks down +X: rotate identity 90° around world-up (-Y).
    theta = np.pi / 2
    R = np.array([
        [np.cos(theta), 0.0, np.sin(theta)],
        [0.0, 1.0, 0.0],
        [-np.sin(theta), 0.0, np.cos(theta)],
    ])
    m = np.eye(4)
    m[:3, :3] = R
    # After rotating identity (which had right=+X) by R, new right = R @ +X
    # = (cos, 0, -sin) = (0, 0, -1). lateral_1m moves position by (0,0,-1).
    out = perturb_c2w(m, "lateral_1m")
    assert np.allclose(out[:3, 3], [0.0, 0.0, -1.0])


def test_yaw_5deg_rotates_rotation_matrix_position_unchanged():
    m = _identity_c2w()
    m[:3, 3] = [5.0, 0.0, 7.0]
    out = perturb_c2w(m, "yaw_5deg")
    # Position invariant
    assert np.allclose(out[:3, 3], m[:3, 3])
    # Rotation matrix changed by 5° around -y axis (world-up for identity).
    # Forward axis (c2w[:3,2]) should rotate by 5° in the xz plane.
    fwd_before = m[:3, 2]
    fwd_after = out[:3, 2]
    cos5 = np.cos(np.deg2rad(5.0))
    assert np.isclose(np.dot(fwd_before, fwd_after), cos5, atol=1e-6)


def test_yaw_10deg_is_larger_rotation_than_5deg():
    m = _identity_c2w()
    fwd0 = m[:3, 2]
    out5 = perturb_c2w(m, "yaw_5deg")
    out10 = perturb_c2w(m, "yaw_10deg")
    cos5 = np.dot(fwd0, out5[:3, 2])
    cos10 = np.dot(fwd0, out10[:3, 2])
    # cos(10°) < cos(5°), i.e., 10° is the bigger angle
    assert cos10 < cos5


def test_perturb_c2w_rejects_invalid_mode():
    with pytest.raises(ValueError):
        perturb_c2w(_identity_c2w(), "lateral_999m")


def test_perturb_c2w_accepts_torch_input():
    m = torch.eye(4, dtype=torch.float32)
    out = perturb_c2w(m, "lateral_1m")
    assert isinstance(out, np.ndarray)
    assert np.allclose(out[:3, 3], [1.0, 0.0, 0.0])


def test_perturb_c2w_accepts_batched_1_4_4_input():
    m = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    out = perturb_c2w(m, "lateral_1m")
    assert out.shape == (4, 4)
    assert np.allclose(out[:3, 3], [1.0, 0.0, 0.0])


def test_shutter_pair_lateral_preserves_rigid_shift():
    """T_to_world_end must shift by the SAME world delta as T_to_world,
    not by the end-frame's right axis (which can differ from start's)."""
    start = _identity_c2w()
    end = _identity_c2w()
    end[:3, 3] = [0.0, 0.0, 0.01]  # tiny forward shutter motion
    # Rotate end slightly so end's right axis would differ from start's
    R_end = np.array([
        [np.cos(0.01), 0.0, np.sin(0.01)],
        [0.0, 1.0, 0.0],
        [-np.sin(0.01), 0.0, np.cos(0.01)],
    ])
    end[:3, :3] = R_end
    new_s, new_e = perturb_shutter_pair(start, end, "lateral_1m")
    delta_s = new_s[:3, 3] - start[:3, 3]
    delta_e = new_e[:3, 3] - end[:3, 3]
    assert np.allclose(delta_s, delta_e, atol=1e-9), (
        "shutter-start and shutter-end must shift by the SAME world delta"
    )


def test_shutter_pair_yaw_rotates_end_around_start_origin():
    """Yaw mode pivots end pose around start position so the trajectory
    rotates as a rigid body (not just spinning end in place)."""
    start = _identity_c2w()
    start[:3, 3] = [1.0, 2.0, 3.0]
    end = start.copy()
    end[:3, 3] = [1.1, 2.0, 3.0]  # 10 cm shutter shift along world-X
    new_s, new_e = perturb_shutter_pair(start, end, "yaw_5deg")
    # Start pose: position unchanged
    assert np.allclose(new_s[:3, 3], start[:3, 3])
    # End pose: should still be ~10 cm from start (rigid rotation around start)
    dist_before = np.linalg.norm(end[:3, 3] - start[:3, 3])
    dist_after = np.linalg.norm(new_e[:3, 3] - new_s[:3, 3])
    assert np.isclose(dist_before, dist_after, atol=1e-9)


def test_perturb_batch_shutter_pair_torch_preserves_shape_dtype_device():
    T_s = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    T_e = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    T_e[0, :3, 3] = torch.tensor([0.0, 0.0, 0.01])
    new_s, new_e = perturb_batch_shutter_pair_torch(T_s, T_e, "lateral_1m")
    assert new_s.shape == (1, 4, 4)
    assert new_e.shape == (1, 4, 4)
    assert new_s.dtype == T_s.dtype
    assert new_e.dtype == T_e.dtype


def test_perturb_batch_rejects_wrong_shape():
    T = torch.eye(4, dtype=torch.float32)  # (4, 4) without batch dim
    with pytest.raises(ValueError):
        perturb_batch_shutter_pair_torch(T, T, "lateral_1m")
