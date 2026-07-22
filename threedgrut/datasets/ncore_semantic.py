# SPDX-License-Identifier: Apache-2.0
"""NCore aux sseg / lidar-sseg semantic class IDs.

T3.1.a 占位实现：NCore aux 数据通过 `nre-tools ncore-aux-data` 用 mask2former
后端生成，该后端默认输出 Cityscapes class palette（19 类）。

| ID | Class       | 用途                          |
|----|-------------|-------------------------------|
| 0  | road        | road_mask                     |
| 1  | sidewalk    | road_mask (含人行道作为路面广义) |
| 2  | building    | bg                            |
| 3  | wall        | bg                            |
| 4  | fence       | bg                            |
| 5  | pole        | bg                            |
| 6  | traffic light| bg                           |
| 7  | traffic sign| bg                            |
| 8  | vegetation  | bg                            |
| 9  | terrain     | bg                            |
| 10 | sky         | sky_mask                      |
| 11 | person      | dyn_mask (sseg-based fallback) |
| 12 | rider       | dyn_mask                      |
| 13 | car         | dyn_mask                      |
| 14 | truck       | dyn_mask                      |
| 15 | bus         | dyn_mask                      |
| 16 | train       | dyn_mask                      |
| 17 | motorcycle  | dyn_mask                      |
| 18 | bicycle     | dyn_mask                      |

TODO(T3.1.b A800 对账): NRE 的 mask2former 实际输出 palette 可能与 Cityscapes
标准有偏移；T3.1.b 集成测时必须读一帧 sseg.zarr.itar 抽 unique values 验证。
若不匹配，本文件改成从 scene_manifest['component_stores'] 读 class table，
或在 nre-tools docker 输出里查 palette JSON。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# 唯一 sky class
SKY_CLASS_ID: int = 10

# 路面（含人行道作为广义可行驶区域，参考 drivestudio road 层定义）
ROAD_CLASS_IDS: frozenset[int] = frozenset({0, 1})

# 可移动物（11-18 = 行人 + 车辆类）
DYNAMIC_CLASS_IDS: frozenset[int] = frozenset({11, 12, 13, 14, 15, 16, 17, 18})

LIDAR_IGNORE_LABEL: int = 255


def filter_init_points_by_semantics(
    xyz: np.ndarray,
    color: Optional[np.ndarray],
    *,
    labels: Optional[np.ndarray],
    dynamic_flags: Optional[np.ndarray],
    non_dynamic_points_only: bool,
    exclude_semantic_class_ids: frozenset[int],
) -> tuple[np.ndarray, Optional[np.ndarray], dict[str, int]]:
    """Filter an initialization cloud while retaining unlabeled points."""
    xyz = np.asarray(xyz)
    color_array = None if color is None else np.asarray(color)
    n_input = int(xyz.shape[0])
    keep_dynamic = np.ones(n_input, dtype=bool)
    if non_dynamic_points_only and dynamic_flags is not None:
        dynamic_array = np.asarray(dynamic_flags).reshape(-1)
        if dynamic_array.shape[0] != n_input:
            raise ValueError("dynamic_flags must align 1:1 with point cloud")
        keep_dynamic = dynamic_array != 1

    xyz = xyz[keep_dynamic]
    if color_array is not None:
        color_array = color_array[keep_dynamic]

    labels_after_dynamic: Optional[np.ndarray] = None
    if labels is not None:
        labels_array = np.asarray(labels).reshape(-1)
        if labels_array.shape[0] != n_input:
            raise ValueError("semantic labels must align 1:1 with point cloud")
        labels_after_dynamic = labels_array[keep_dynamic]

    if labels_after_dynamic is None:
        exclude_mask = np.zeros(xyz.shape[0], dtype=bool)
        unknown_mask = np.ones(xyz.shape[0], dtype=bool)
    else:
        exclude_mask = np.isin(labels_after_dynamic, np.asarray(sorted(exclude_semantic_class_ids)))
        unknown_mask = labels_after_dynamic == LIDAR_IGNORE_LABEL

    keep_semantic = ~exclude_mask
    xyz = xyz[keep_semantic]
    if color_array is not None:
        color_array = color_array[keep_semantic]

    remaining_labels = None if labels_after_dynamic is None else labels_after_dynamic[keep_semantic]
    n_intersection = (
        0
        if remaining_labels is None
        else int(np.isin(remaining_labels, np.asarray(sorted(exclude_semantic_class_ids))).sum())
    )
    stats = {
        "n_input_points": n_input,
        "n_dynamic_removed": int((~keep_dynamic).sum()),
        "n_bg_points": int(xyz.shape[0]),
        "n_road_points": int(exclude_mask.sum()),
        "n_excluded_road_class": int(exclude_mask.sum()),
        "n_unknown_kept": int((unknown_mask & keep_semantic).sum()),
        "n_intersection": n_intersection,
    }
    return xyz, color_array, stats
