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

    BEV 上用**最小面积包围矩形**（min-area rect，1° 暴力扫 mod π/2）定朝向 —— 比 PCA
    对单边/L-shape LiDAR 车点鲁棒（O2：PCA 主轴对稀疏单边回波 yaw-MAE 差）。取矩形几何
    中心（非点云质心，对齐 init filter 的 ``|local| ≤ size/2`` 假设），旋回 world；
    z 取 min/max 中点与高度。
    """
    if xyz.shape[0] < 4:
        return None
    xy = xyz[:, :2].astype(np.float64)
    mean = xy.mean(0)
    centered = xy - mean
    best = None
    for k in range(90):  # mod π/2 足够（矩形 AABB 面积 π/2 周期）
        a = k * (np.pi / 180.0)
        c, s = np.cos(-a), np.sin(-a)
        rot = centered @ np.array([[c, -s], [s, c]]).T
        lo, hi = rot.min(0), rot.max(0)
        ext = hi - lo
        area = float(ext[0] * ext[1])
        if best is None or area < best[0]:
            best = (area, a, lo, hi)
    _, ang, lo, hi = best
    ext = hi - lo
    center_box = (lo + hi) / 2.0
    ci, si = np.cos(ang), np.sin(ang)
    center_xy = mean + center_box @ np.array([[ci, -si], [si, ci]]).T  # 旋回 world
    if ext[0] >= ext[1]:                 # 长轴对齐 → yaw
        l, w, yaw = float(ext[0]), float(ext[1]), ang
    else:
        l, w, yaw = float(ext[1]), float(ext[0]), ang + np.pi / 2.0
    z = xyz[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    center = np.array([center_xy[0], center_xy[1], (zmin + zmax) / 2.0], dtype=np.float64)
    dim = np.array([l, w, zmax - zmin], dtype=np.float64)
    return center, dim, canonicalize_yaw(yaw)
