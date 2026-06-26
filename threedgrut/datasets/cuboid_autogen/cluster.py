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
