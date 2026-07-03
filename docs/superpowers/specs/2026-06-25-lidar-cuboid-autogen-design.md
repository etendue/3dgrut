# Design Spec：纯 LiDAR 聚类自动生成车辆 cuboids → V4 shard → dynamic_rigids 初始化

- **日期**：2026-06-25
- **状态**：Design（待 writing-plans 拆实现计划）
- **作者**：大g + Claude（brainstorming）
- **关联**：dynamic_rigids 层初始化（`threedgrut/layers/dynamic_rigid_init.py`）、`tracks_loader.py`、`datasetNcore.py`

---

## 1. 背景与目标

inceptio 的 NCore clip **没有 dynamic object 的 cuboid autolabels**（V4 cuboids
component 为空）。而 dynamic_rigids 层需要每个动态车辆的 cuboid 轨迹
（带 track-id 的逐帧 SE(3) pose + 尺寸）来**初始化刚体高斯**：
`init_dynamic_rigid_layer` 用每帧 cuboid pose 把动态 LiDAR 点变换到物体局部系、
按 `|local| ≤ size/2` 裁剪得到 object-local 粒子（[dynamic_rigid_init.py:40](../../../threedgrut/layers/dynamic_rigid_init.py)）。

**目标**：离线为任意 clip 生成车辆 cuboid 轨迹，写成 NCore V4 cuboids shard，
让现有读取/初始化链零改消费。

**精度定位（宽松）**：cuboid 只做**初始化**，pose 在训练中被
`trainer.pose_adjustment` / `learnable_pose` 更新（见项目 CLAUDE.md）。因此
框中心/尺寸/朝向的小误差、yaw 180° 歧义都能被训练吸收 —— 不追高精度。

---

## 2. 关键决策（brainstorming 已定）

| # | 决策 | 选择 | 理由 |
|---|------|------|------|
| D1 | pipeline 路线 | **纯 LiDAR 聚类**（不用 SAM2/图像） | `aux.lidar-sseg` 已给每点语义标签，"哪些点是车"已知，无需图像分割。zero-shot、无 domain gap、CPU 即可 |
| D2 | 注入方式 | **写回 NCore V4 二进制**（autolabels component） | 与真 autolabel 同源同格式，读取/初始化链零改 |
| D3 | 写入形态 | **独立 `cuboids.zarr.itar` shard**（不重写 base store） | V4 无 in-place append；但 multi-shard 架构允许独立 component-group shard（aux.sseg 即先例） |
| D4 | 类别范围 | **只车辆**（automobile/heavy_truck/bus） | 精确对齐 `DEFAULT_VEHICLE_CLASSES` + dynamic_rigids scope；行人/骑行属未实现的 dynamic_deformables |
| D5 | 生成器跑哪 | **inceptio**（数据+SDK 都在，聚类是 CPU） | — |
| D6 | 实现首步 | **O1 spike 优先**（先打通空 shard 写→读 round-trip） | 最高风险的 V4 写读契约最先证伪 |

---

## 3. 现状证据（代码 + SDK 实测）

**读取链（trainer 当前路径，零改目标）：**
- [trainer.py:456](../../../threedgrut/trainer.py) 调
  `load_tracks_from_ncore_cuboids(loader, cam_ts_active)`。
- [tracks_loader.py:153](../../../threedgrut/datasets/tracks_loader.py)
  从 `loader.get_cuboid_track_observations()` 读 obs，
  `obs.transform("world", ts, pose_graph)` 转世界系，
  解码 `bbox.centroid`(几何中心) / `bbox.rot`(XYZ-euler) / `bbox.dim` →
  `{tid: {poses[F,4,4], size[3], frame_info[F], class, cam_timestamps_us}}`。
- 缺 cuboids → [trainer.py:463](../../../threedgrut/trainer.py) 只 warning，
  layer 留空不崩。

**loader 构造（multi-shard 证据）：**
- [datasetNcore.py:314](../../../threedgrut/datasets/datasetNcore.py)：
  `SequenceLoaderV4(SequenceComponentGroupsReader([sequence_meta_file_path], …), …)`
  —— reader 接受 **store path 列表**。
- [datasetNcore.py:100-106](../../../threedgrut/datasets/datasetNcore.py) 注释：
  aux.*.zarr.itar 作为独立 shard 被纳入 reader。
- [datasetNcore.py:1419](../../../threedgrut/datasets/datasetNcore.py)：已有
  "bypass SequenceLoaderV4 直读 `aux.lidar-sseg.zarr.itar`" 先例（供单元1a 复用）。

**写接口（inceptio `ncore` SDK 实测签名）：**
```python
# ncore.data
CuboidTrackObservation(
    track_id: str, class_id: str, timestamp_us: int,
    reference_frame_id: str, reference_frame_timestamp_us: int,
    bbox3: BBox3, source: LabelSource = None, source_version: Optional[str] = None)
BBox3(centroid: (x,y,z), dim: (l,w,h), rot: (rx,ry,rz))   # rot = XYZ-euler
LabelSource = {AUTOLABEL, EXTERNAL, GT_ANNOTATION, GT_SYNTHETIC, UNKNOWN}

# ncore.data.v4
CuboidsComponent.Writer.store_observations(List[CuboidTrackObservation]) -> Self
CuboidsComponent.Writer.finalize()
SequenceComponentGroupsWriter(...)            # 写 shard
SequenceComponentGroupsReader([paths], ...)   # 读多 shard
```
我们生成的框：`source=LabelSource.AUTOLABEL, source_version="lidar-cluster-v1"`。

---

## 4. 架构

```
[单元1] 离线生成器 (inceptio, CPU)
  clip → 1a 逐帧动态LiDAR点(按类) → 1b 聚类 → 1c 朝向框拟合
       → 1d 跨帧track+动静过滤 → 1e CuboidTrackObservation[] → 写 cuboids.zarr.itar
                         │
[单元2] datasetNcore 发现层 (唯一改动)
  load_auto_cuboids=true 且存在 shard → SequenceComponentGroupsReader 纳入
                         │
[单元3] 读取/初始化链 (零改)
  get_cuboid_track_observations() → load_tracks_from_ncore_cuboids
  → init_dynamic_rigid_layer → dynamic_rigids 高斯初始化
                         │
[单元4] 验证：round-trip 单测 + 1k smoke + viser 目检
```

---

## 5. 组件与文件清单

| 单元 | 文件 | 职责 | 依赖 | 可测 |
|------|------|------|------|------|
| 1b/1c | `threedgrut/datasets/cuboid_autogen/cluster.py`（新） | 逐帧 DBSCAN 聚类 + L-shape/PCA 朝向框拟合 | numpy（+可选 sklearn/scipy） | Mac 纯函数单测 |
| 1d | `threedgrut/datasets/cuboid_autogen/track.py`（新） | 跨帧 IoU/中心距关联 → track-id + 动静过滤 + size 聚合 | numpy | Mac 纯函数单测 |
| 1a | `threedgrut/datasets/cuboid_autogen/lidar_source.py`（新） | 逐帧动态点访问（读 aux.lidar-sseg，保留帧归属+per-point class） | NCore SDK | inceptio |
| 1e | `threedgrut/datasets/cuboid_autogen/v4_writer.py`（新） | 构造 CuboidTrackObservation + 写独立 cuboids shard | NCore SDK | inceptio round-trip |
| CLI | `scripts/gen_cuboids_from_lidar.py`（新） | 串联 1a→1e 的可复用 CLI（任意 clip） | 上述 | — |
| 单元2 | `threedgrut/datasets/datasetNcore.py`（改） | `load_auto_cuboids` 开关 + reader 纳入 cuboids shard | — | smoke |

**core 纯函数（cluster.py/track.py）与 SDK 隔离**，沿用 tracks_loader.py
"独立模块避免 SDK/cv2 chain，Mac 可 import 单测" 的先例。

---

## 6. 坐标 / 时间不变量（错了静默失败）

1. **world frame 全程**：LiDAR 点已 transform 到 world → cuboid 存 world，
   `reference_frame_id="world"`，下游 `obs.transform("world",…)` 退化 identity。
2. **centroid = 几何中心**（非 bottom-center）：init filter `|local|≤size/2` 假设
   几何中心；ncore skill 亦明确要求。
3. **rot = XYZ-euler**，车辆纯 yaw → `(0,0,yaw)`，对齐
   `euler_xyz_to_rotation_matrix`（[tracks_loader.py:50](../../../threedgrut/datasets/tracks_loader.py)）。
4. **timestamp = camera END ts**（非 LiDAR sweep ts）：
   `load_tracks_from_ncore_cuboids` 按 cam_ts 最近邻匹配（tol 50ms），
   obs 的 `timestamp_us` / `reference_frame_timestamp_us` 必须落在匹配窗内。
5. **class_id 数字→字符串映射**：lidar-sseg 是 13/14/15（car/truck/bus），
   写入前映射成 `{"automobile","heavy_truck","bus"}`（`DEFAULT_VEHICLE_CLASSES`），
   否则 `class_filter` 滤空。映射表在 1e。

---

## 7. 错误处理 / 边界

- 无 vehicle 点 / 无动态 track → 写 0-obs shard → 现有"空 layer warning 不崩"兜底。
- shard 不存在 + `load_auto_cuboids=false`（默认）→ 完全回退现有行为，零影响。
- 同 track 跨帧 size 抖动 → 取 track 内稳健聚合的**单一 size**（刚体假设）。
- 写读契约由 round-trip 单测在写入阶段拦截（不到 smoke 才发现）。

---

## 8. 测试策略

- **Mac 纯函数单测**（无 GPU/SDK）：
  - `test_cuboid_cluster.py`：合成点云（已知 yaw/中心/尺寸的车形点团）验证
    1b 聚类分离 + 1c 拟合精度 + euler 约定。
  - `test_cuboid_track.py`：合成多帧轨迹验证 1d 关联正确、动静过滤（静止 track 被丢）、size 聚合。
- **inceptio round-trip 单测**（需 SDK）：
  - `test_cuboid_v4_roundtrip.py`：store_observations 写 → SequenceComponentGroupsReader
    读回 → 字段逐一比对，pin 住 V4 cuboid 写读契约（**即 O1 spike**）。
- **inceptio 1k smoke**：开 `load_auto_cuboids`，确认 dynamic_rigids 粒子数 > 0、
  训练不崩、metrics.json 正常。
- **viser 目检**（viser-gui-4d skill）：dynamic actor 摆放/朝向合理性。

---

## 9. 实现顺序（writing-plans 细化）

1. **O1 spike**（最高风险先证伪）：inceptio 上写一个**空/最小** cuboids shard →
   `SequenceComponentGroupsReader([meta, cuboids_shard])` 读回 →
   `get_cuboid_track_observations()` 拿到 obs。**确认 shard 发现机制**
   （独立 path vs component_group 名）+ 字段 round-trip。这步定死单元2 与 1e 的接口。
2. 单元2：`load_auto_cuboids` 开关 + reader 纳入逻辑（基于 step1 结论）。
3. 1a：逐帧动态点访问（读 aux.lidar-sseg）。
4. 1b/1c/1d：聚类→拟合→tracking 纯函数 + Mac 单测（TDD）。
5. 1e + CLI：串联写 shard。
6. inceptio 1k smoke + viser 目检。

---

## 10. 开放点（实现期验证，不阻塞 design）

- **O1**：cuboids shard 被 reader 发现的确切机制 → step1 spike 确认。
- **O2**：L-shape vs PCA 朝向鲁棒性阈值 + 动静位移阈值 → 真实 clip 调。
- **O3**：tracker 关联度量（BEV IoU vs 中心距）+ 缺帧插值策略。
- **O4**：class 数字→字符串映射表（13→automobile…）需对齐 NCore 类名约定。

---

## 11. YAGNI（明确不做）

- 不做行人/骑行（只车）。
- 不做 SAM2 / 图像分支（纯 LiDAR）。
- 不做 in-place 重写 base store（独立 shard）。
- 不追高精度（训练会更新 pose）。
- 不做轨迹的 learnable 优化（那是现有 `pose_adjustment` 的职责）。
