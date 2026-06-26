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
