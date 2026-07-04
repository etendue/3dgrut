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

# 唯一 sky class
SKY_CLASS_ID: int = 10

# 路面（含人行道作为广义可行驶区域，参考 drivestudio road 层定义）
ROAD_CLASS_IDS: frozenset[int] = frozenset({0, 1})

# 可移动物（11-18 = 行人 + 车辆类）
DYNAMIC_CLASS_IDS: frozenset[int] = frozenset({11, 12, 13, 14, 15, 16, 17, 18})
