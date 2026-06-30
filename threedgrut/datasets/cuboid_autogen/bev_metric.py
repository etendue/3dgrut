# SPDX-License-Identifier: Apache-2.0
"""旋转矩形 BEV-IoU + 框匹配（纯 numpy，Mac 真测）。用于 9ae GT 定量验证。

凸多边形交用 Sutherland–Hodgman 裁剪 + shoelace 面积，确定性可单测。
框表示：``(cx, cy, l, w, yaw)``，l 沿 yaw 方向，w 垂直。
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _corners(cx, cy, l, w, yaw) -> np.ndarray:
    dx, dy = l / 2.0, w / 2.0
    loc = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]], dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    return loc @ np.array([[c, -s], [s, c]]).T + np.array([cx, cy], dtype=np.float64)


def _signed_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ccw(p: np.ndarray) -> np.ndarray:
    return p if _signed_area(p) >= 0 else p[::-1]


def _clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """用凸多边形 clip(CCW) 裁剪 subject，返回交多边形顶点。"""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12

    def inter(p1, p2, a, b):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = a
        x4, y4 = b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-12:
            return p2
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])

    out = list(subject)
    n = len(clip)
    for i in range(n):
        a, b = clip[i], clip[(i + 1) % n]
        inp, out = out, []
        if not inp:
            break
        for j in range(len(inp)):
            cur, prv = inp[j], inp[j - 1]
            if inside(cur, a, b):
                if not inside(prv, a, b):
                    out.append(inter(prv, cur, a, b))
                out.append(cur)
            elif inside(prv, a, b):
                out.append(inter(prv, cur, a, b))
    return np.array(out) if out else np.zeros((0, 2))


def bev_iou(boxA, boxB) -> float:
    """两旋转矩形的 BEV IoU。"""
    pa, pb = _ccw(_corners(*boxA)), _ccw(_corners(*boxB))
    inter = _clip(pa, pb)
    if inter.shape[0] < 3:
        return 0.0
    ia = abs(_signed_area(inter))
    ua = abs(_signed_area(pa)) + abs(_signed_area(pb)) - ia
    return float(ia / ua) if ua > 0 else 0.0


def match_boxes(auto: List, gt: List, max_dist: float) -> List[Tuple[int, int]]:
    """贪心按中心距匹配 auto→gt，返回 [(auto_idx, gt_idx)]（升序）。"""
    pairs = []
    for gi, g in enumerate(gt):
        for ai, a in enumerate(auto):
            d = float(np.hypot(a[0] - g[0], a[1] - g[1]))
            if d <= max_dist:
                pairs.append((d, ai, gi))
    pairs.sort()
    used_a: set[int] = set()
    used_g: set[int] = set()
    out = []
    for _d, ai, gi in pairs:
        if ai in used_a or gi in used_g:
            continue
        used_a.add(ai)
        used_g.add(gi)
        out.append((ai, gi))
    return sorted(out)
