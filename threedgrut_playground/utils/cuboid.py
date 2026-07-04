# SPDX-License-Identifier: Apache-2.0
"""Unit cube wireframe geometry for viser cuboid rendering (Stage 8 / T8.5).

Pure numpy; no torch, no viser. Tested in isolation.
"""

from __future__ import annotations

import numpy as np

# 8 unit-cube vertices in (±0.5)^3 order. Vertex 0 is (-0.5, -0.5, -0.5);
# bit 0 = +x, bit 1 = +y, bit 2 = +z when index is binary-encoded.
_UNIT_CUBE_VERTS = np.array(
    [
        [-0.5, -0.5, -0.5],  # 0  (- - -)
        [0.5, -0.5, -0.5],  # 1  (+ - -)
        [-0.5, 0.5, -0.5],  # 2  (- + -)
        [0.5, 0.5, -0.5],  # 3  (+ + -)
        [-0.5, -0.5, 0.5],  # 4  (- - +)
        [0.5, -0.5, 0.5],  # 5  (+ - +)
        [-0.5, 0.5, 0.5],  # 6  (- + +)
        [0.5, 0.5, 0.5],
    ],  # 7  (+ + +)
    dtype=np.float32,
)

# 12 edges: 4 along x (low-z bottom + high-z top), 4 along y, 4 along z.
_UNIT_CUBE_EDGE_INDICES = np.array(
    [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),  # +x edges
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),  # +y edges
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ],  # +z edges
    dtype=np.int64,
)

UNIT_CUBE_EDGES: np.ndarray = _UNIT_CUBE_VERTS[_UNIT_CUBE_EDGE_INDICES]
"""Unit cube wireframe edges: shape ``(12, 2, 3)``, ``float32``.

Each row ``UNIT_CUBE_EDGES[i] = [v_start, v_end]`` is a 3D segment in object-
local frame centered at the origin with extent 1.
"""


def cuboid_world_edges(pose: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Transform a unit cube wireframe to world frame.

    Args:
        pose: ``(4, 4)`` SE(3) object-to-world transform. Translate-only pose
              (rot block == I) yields an axis-aligned box.
        size: ``(3,)`` full-extent box size ``(Lx, Ly, Lz)``.

    Returns:
        ``(12, 2, 3)`` float32 world-frame segments.
    """
    pose = np.asarray(pose, dtype=np.float32)
    size = np.asarray(size, dtype=np.float32).reshape(3)
    local = UNIT_CUBE_EDGES * size[None, None, :]  # (12, 2, 3)
    flat = local.reshape(-1, 3)  # (24, 3)
    R = pose[:3, :3]
    t = pose[:3, 3]
    world_flat = flat @ R.T + t  # (24, 3)
    return world_flat.reshape(12, 2, 3).astype(np.float32)


def class_color(class_name: str) -> tuple[float, float, float]:
    """Stable color per cuboid class (used to tint per-track trajectory polylines)."""
    palette = {
        "automobile": (0.10, 0.60, 1.00),  # blue
        "heavy_truck": (1.00, 0.50, 0.10),  # orange
        "bus": (0.80, 0.30, 0.85),  # purple
        "unknown": (0.65, 0.65, 0.65),  # gray
    }
    return palette.get(class_name, palette["unknown"])


def instance_color(track_id: str) -> tuple[float, float, float]:
    """Deterministic instance color from track_id via hash → HSV → RGB.

    Used for per-cuboid wireframe so each instance reads distinctly even when
    classes overlap. Hash-stable across runs and machines.
    """
    # FNV-1a 32-bit on the utf-8 bytes — deterministic + dependency-free.
    h = 2166136261
    for b in track_id.encode("utf-8"):
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    hue = (h & 0xFFFF) / 65535.0  # [0, 1)
    sat = 0.7
    val = 0.9
    # HSV→RGB
    i = int(hue * 6.0)
    f = hue * 6.0 - i
    p = val * (1.0 - sat)
    q = val * (1.0 - f * sat)
    t = val * (1.0 - (1.0 - f) * sat)
    table = [(val, t, p), (q, val, p), (p, val, t), (p, q, val), (t, p, val), (val, p, q)]
    return table[i % 6]
