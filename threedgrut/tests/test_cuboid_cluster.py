# SPDX-License-Identifier: Apache-2.0
"""Task 2/3 单测 — DBSCAN 包装 + 朝向框拟合。

纯 numpy；Mac CPU 无 sklearn 真实调用（conftest stub sklearn → cluster_points
测试用 monkeypatch 替身；fit_oriented_box 不依赖 sklearn）。
"""
from __future__ import annotations

import numpy as np

import threedgrut.datasets.cuboid_autogen.cluster as C


# ---------------- Task 2: cluster_points ----------------

def test_cluster_points_labels_contract(monkeypatch):
    class _FakeDBSCAN:
        def __init__(self, eps, min_samples):
            pass

        def fit(self, X):
            self.labels_ = np.array([0, 0, -1, 1, 1, 1])  # 含 noise=-1
            return self

    monkeypatch.setattr(C, "DBSCAN", _FakeDBSCAN)
    labels = C.cluster_points(np.zeros((6, 3)), eps=0.5, min_samples=3)
    assert labels.shape == (6,)
    assert labels.dtype == np.int64
    assert set(labels.tolist()) == {0, 1, -1}  # noise 保留 -1


def test_cluster_points_empty_input():
    labels = C.cluster_points(np.zeros((0, 3)), eps=0.5, min_samples=3)
    assert labels.shape == (0,)
    assert labels.dtype == np.int64


# ---------------- Task 3: fit_oriented_box ----------------

def _make_box_surface(center, dim, yaw, n=3000, noise=0.01, seed=0):
    """采样盒 6 面点云（已知 center/dim/yaw），用于验证几何中心+尺寸+yaw 恢复。"""
    rng = np.random.default_rng(seed)
    half = np.asarray(dim) / 2.0
    faces = [(0, +1), (0, -1), (1, +1), (1, -1), (2, +1), (2, -1)]
    per = n // 6
    pts = []
    for axis, sign in faces:
        p = (rng.random((per, 3)) * 2 - 1) * half
        p[:, axis] = sign * half[axis]
        pts.append(p)
    p = np.concatenate(pts, 0) + rng.normal(0, noise, (per * 6, 3))
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return p @ R.T + np.asarray(center)


def test_fit_full_surface_recovers_geometry():
    center = (10.0, 3.0, 0.85)
    dim = (4.5, 2.0, 1.7)
    yaw = 0.6
    xyz = _make_box_surface(center, dim, yaw)
    c, d, y = C.fit_oriented_box(xyz)
    np.testing.assert_allclose(c, center, atol=0.05)   # 几何中心
    np.testing.assert_allclose(d, dim, atol=0.10)      # (l,w,h)，l>w 顺序
    assert abs(C.wrap_to_pi(y - yaw)) < 0.05           # yaw mod π
    assert d[0] >= d[1]                                 # 长轴=principal


def test_fit_degenerate_too_few_points():
    assert C.fit_oriented_box(np.zeros((3, 3))) is None


def test_canonicalize_yaw_idempotent_and_range():
    for a in np.linspace(-3.0, 3.0, 25):
        y = C.canonicalize_yaw(a)
        assert -np.pi / 2 - 1e-9 <= y < np.pi / 2 + 1e-9
        assert abs(C.wrap_to_pi(C.canonicalize_yaw(y) - y)) < 1e-9
