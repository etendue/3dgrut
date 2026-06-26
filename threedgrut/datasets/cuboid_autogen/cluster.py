# SPDX-License-Identifier: Apache-2.0
"""逐帧 DBSCAN 聚类 + 朝向框拟合。

- ``cluster_points`` 用 ``sklearn.cluster.DBSCAN``（inceptio 真实；Mac conftest
  stub sklearn，单测以 monkeypatch 替身覆盖）。
- ``fit_oriented_box`` 纯 numpy，Mac 真测（Task 3 追加）。
"""
from __future__ import annotations

import numpy as np

try:
    from sklearn.cluster import DBSCAN  # noqa: F401  (Mac stub 时为占位，测试 monkeypatch 覆盖)
except Exception:  # pragma: no cover
    DBSCAN = None


def cluster_points(xyz: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """对点云做 DBSCAN，返回每点 label（int64，-1 = noise）。空输入返回 (0,)。"""
    if xyz.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz).labels_
    return np.asarray(labels, dtype=np.int64)


def wrap_to_pi(a: float) -> float:
    """把角度归一到 (-π, π]。"""
    return (a + np.pi) % (2 * np.pi) - np.pi


def canonicalize_yaw(yaw: float) -> float:
    """归一到 [-π/2, π/2)；±π 平移不交换长短轴（保持 principal=长轴）。"""
    y = wrap_to_pi(yaw)
    if y >= np.pi / 2:
        y -= np.pi
    elif y < -np.pi / 2:
        y += np.pi
    return float(y)


def fit_oriented_box(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    """拟合朝向框 → (center[3] 几何中心, dim[3]=(l,w,h) 全长, yaw)；<4 点返回 None。

    BEV 上 PCA 取最大特征向量为 yaw，旋进 box 系取 min/max 得 extent 与几何中心
    （非点云质心，对齐 init filter 的 ``|local| ≤ size/2`` 几何中心假设），旋回 world；
    z 取 min/max 中点与高度。
    """
    if xyz.shape[0] < 4:
        return None
    xy = xyz[:, :2].astype(np.float64)
    mean = xy.mean(0)
    centered = xy - mean
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    principal = evecs[:, int(np.argmax(evals))]          # 长轴方向
    yaw = float(np.arctan2(principal[1], principal[0]))
    c, s = np.cos(-yaw), np.sin(-yaw)
    rot = centered @ np.array([[c, -s], [s, c]]).T        # 旋进 box 系
    lo, hi = rot.min(0), rot.max(0)
    extent = hi - lo                                       # (l, w)
    center_box = (lo + hi) / 2.0                           # box 系几何中心
    ci, si = np.cos(yaw), np.sin(yaw)
    center_xy = mean + center_box @ np.array([[ci, -si], [si, ci]]).T  # 旋回 world
    z = xyz[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    center = np.array([center_xy[0], center_xy[1], (zmin + zmax) / 2.0], dtype=np.float64)
    dim = np.array([float(extent[0]), float(extent[1]), zmax - zmin], dtype=np.float64)
    return center, dim, canonicalize_yaw(yaw)
