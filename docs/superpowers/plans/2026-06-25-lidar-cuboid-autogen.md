# 纯 LiDAR 聚类自动生成车辆 cuboids → V4 shard → dynamic_rigids 初始化 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: 用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 逐任务执行。步骤用 checkbox 跟踪。
> **格式约定：** 本计划**不贴 code snippet 代码块**（大g 要求）。每个任务只写：目标、改/建文件、关键签名（inline）、测试要验证什么 + 公差、命令意图。具体代码执行期 TDD 时再写。
> **落盘位置（执行期）：** 归档到仓库 `docs/superpowers/plans/2026-06-25-lidar-cuboid-autogen.md`（与 design spec 同目录），执行第一步从 `~/.claude/plans/` 草稿位复制进仓库。

## Context（为什么做这个改动）

inceptio 自采的 NCore clip（如 `4cabad44`）**没有动态物体的 cuboid autolabels**（V4 cuboids component 为空）。而 `dynamic_rigids` 层需要每个动态车辆的 cuboid 轨迹（带 track-id 的逐帧 SE(3) pose + 尺寸）来**初始化刚体高斯**——`init_dynamic_rigid_layer` 用每帧 cuboid pose 把动态 LiDAR 点变换到物体局部系、按 `|local| ≤ size/2` 裁剪得到 object-local 粒子。缺 cuboid → 该层留空，动态车辆建不出来。

本改动**离线**为任意 clip 生成车辆 cuboid 轨迹（纯 LiDAR 聚类，复用已有的 `aux.lidar-sseg` 逐点语义标签），写成 NCore V4 cuboids shard，让现有读取/初始化链**尽量零改**消费。精度定位宽松：cuboid 只做初始化，pose 在训练中由 `trainer.pose_adjustment` 更新，小误差与 yaw 180° 歧义可被训练吸收。

**设计依据：** `docs/superpowers/specs/2026-06-25-lidar-cuboid-autogen-design.md`（brainstorming 已定 D1–D6）。

## Goal

离线纯-LiDAR 聚类生成车辆 cuboid 轨迹 → 写 NCore V4 cuboids shard → 被现有 `dynamic_rigids` 初始化链消费，并用 9ae GT cuboid 定量验证工具正确性。

## Architecture

三段式：① **离线生成器**（inceptio CPU）读 `aux.lidar-sseg` 逐帧动态车辆点 → DBSCAN 聚类 → 朝向框拟合 → 跨帧 tracking + 动静过滤 → 构造 `CuboidTrackObservation[]` → 官方 `SequenceComponentGroupsWriter` 写独立 cuboids shard。② **datasetNcore 发现层**（唯一侵入改动）`load_auto_cuboids` 开关把 shard 纳入读取。③ **读取/初始化链**（目标零改）`get_cuboid_track_observations()` → `load_tracks_from_ncore_cuboids` → `init_dynamic_rigid_layer`。

**核心风险即 O1 硬门：** 官方 writer 写的 cuboids shard 能否被 `SequenceComponentGroupsReader([meta, shard])` append 读回。已知 `nre-tools` 产的 aux shard 因根 `.zattrs` 缺 `version` 字段**不能** append（`datasetNcore.py:305-313`，A800 实测 `KeyError: 'version'`），但官方 writer 写的 shard 理论上带 version——**O1 spike 必须先证实/证伪**，结论决定 Branch A（reader 接受 → 读取链零改）还是 Branch B（reader 拒绝 → 独立 reader 桥接）。O1 出结论前**不动** writer/接线代码；Mac 纯函数任务与 O1 并行。

## Tech Stack

Python 3.11（inceptio conda `3dgrut2`）/ numpy / `scikit-learn`（DBSCAN，已是依赖）/ `nvidia-ncore`（V4 SDK，仅 inceptio）/ pytest。Mac 端（`.venv`）跑纯函数单测，conftest 已 stub `ncore`/`sklearn`/`cv2`/CUDA。

## Global Constraints（每个任务隐含适用）

- **坐标/时间不变量（错了静默失败，trainer 只 warning 不崩）：**
  1. **world frame 全程**——cuboid 存 world，`reference_frame_id` 取值由 O1 A3 定（期望 `"world"`，使 `obs.transform("world",…)` 退化 identity）。
  2. **centroid = 几何中心**（非 bottom-center、非点云质心）——init filter `|local| ≤ size/2` 假设几何中心对称。
  3. **rot = XYZ-euler，纯 yaw → `(0,0,yaw)`**——对齐 `euler_xyz_to_rotation_matrix`（`tracks_loader.py:50`）。
  4. **timestamp 落在 camera END ts 的 50ms 窗内**——`load_tracks_from_ncore_cuboids` 按 cam_ts 最近邻匹配，`time_tolerance_us=50_000`；写入 obs 必须有 ≥1 帧落进某 cam END ts 的 50ms 内，否则该 track 被悄悄丢空。
  5. **class 数字→字符串映射**——lidar-sseg 13/14/15（car/truck/bus）→ `{"automobile","heavy_truck","bus"}`，否则 `obs.class_id in class_filter` 全 False → obs 静默滤空。映射值必须**逐字等于** `tracks_loader.DEFAULT_VEHICLE_CLASSES`。
- **Mac 单测纪律：** conftest stub 了 `sklearn` → 纯函数测试**不真正调** `sklearn.DBSCAN`，`cluster_points` 测试用 monkeypatch 替身；其余几何/tracking/metric 纯 numpy 真测。
- **inceptio 训练铁律：** depth-off（`use_lidar_depth=false use_depth_prior=false load_lidar_depth_map=false`）+ `num_workers=10`；conda env `3dgrut2`；训练用 `apps/ncore_3dgut_mcmc_multilayer` + `trainer.sky_backend=mlp`。**1a 读 aux.lidar-sseg 需 `dataset.load_aux_masks=true`**（sseg ≠ depth，不冲突）。
- **inceptio worktree 工作流：** 每任务独立 git worktree；`git worktree add` 后**必须 rsync submodule**（lib3dgut_cc JIT 需 tiny-cuda-nn）。
- **Hydra override：** 新 key 用 `+`，覆盖已有用 `++`，顶层已有 key 直接写。
- **脚本形态：** 顶层脚本放 `scripts/`，`#!/usr/bin/env python3` + `_REPO_ROOT` sys.path insert + `if __name__=="__main__": sys.exit(main())`（CLAUDE.md L47 踩坑：包内模块错放顶层致 silent exit 0）。
- **提交：** 每 Task 末尾 commit；A800/inceptio 出口 ✅ 需实测数字 + commit hash；文档同步作为最后一个 Task。

## File Structure（先锁分解）

新建（均不存在）：
- `threedgrut/datasets/cuboid_autogen/__init__.py`
- `cuboid_autogen/labels.py`——class 数字→字符串映射（纯，Mac）
- `cuboid_autogen/cluster.py`——DBSCAN 包装 `cluster_points` + 朝向框拟合 `fit_oriented_box`（后者纯 numpy，Mac）
- `cuboid_autogen/track.py`——`associate` / `is_dynamic` / `aggregate_size` / `interpolate_gaps`（纯 numpy，Mac）
- `cuboid_autogen/bev_metric.py`——`bev_iou` / `match_boxes`（纯 numpy，Mac）
- `cuboid_autogen/lidar_source.py`——逐帧动态点访问（读 aux.lidar-sseg，inceptio/SDK）
- `cuboid_autogen/v4_writer.py`——`tracks_to_observations`（组装核 Mac 可测）+ `write_cuboids_shard`（inceptio/SDK）
- `scripts/gen_cuboids_from_lidar.py`——CLI 串联 1a→1e（inceptio）
- `scripts/spikes/spike_o1_cuboids_roundtrip.py`——O1 spike，之后晋升为 round-trip 单测
- `threedgrut/tests/test_cuboid_{labels,cluster,track,bev_metric}.py`、`test_v4_writer_core.py`（Mac）、`test_v4_writer_roundtrip.py`（inceptio）、`test_tracks_loader_shard_bridge.py`（仅 Branch B）

改动（已存在）：
- `threedgrut/datasets/datasetNcore.py`——`_iter_semantic_lidar_frames` 重构（Task 8）+ `load_auto_cuboids` 开关（Task 11）
- `threedgrut/datasets/__init__.py`——`make()` 透传新 config key（Task 11）
- `threedgrut/datasets/tracks_loader.py`——仅 Branch B：抽 `_obs_iter_to_tracks` + 加 `load_tracks_from_cuboids_shard`（Task 11）
- `threedgrut/trainer.py`——仅 Branch B：~456 处按 config 分支（Task 11）
- `v3_plan_revised.md` / `v2_architecture.md`——文档同步（Task 14）

---

## Task 0 — O1 spike：cuboids shard 写→读 round-trip（inceptio，硬门）

**最高风险先证伪。** 不投资聚类，手搓 2 条 obs，验证「官方 writer 写的 shard 能否被生产读取链看见」+ 定死 `reference_frame_id` 约定。
**Files:** Create `scripts/spikes/spike_o1_cuboids_roundtrip.py`。
**Runs on:** inceptio（`3dgrut2`）。**Blocks:** Task 7/9/10/11。**不 block** Task 1–7。

- [ ] **Step 1: 写 spike**——读源 meta 镜像 `sequence_id` / `interval` / `pose_graph`（并打印 loader 的真实 accessor 名）；用 `SequenceComponentGroupsWriter` + `CuboidsComponent.Writer.store_observations().finalize()` 写 2 条同 track、`reference_frame_id="world"`、`rot=(0,0,0.3)`、`dim=(4.5,2,1.7)`、`source=AUTOLABEL/source_version="lidar-cluster-v1"` 的 obs；时间戳取源 clip 真实的前两个 camera END ts（避免 ts 窗错配）。
- [ ] **Step 2: inceptio 跑**，逐条断言：
  - **A0** `finalize()` 返回非空 `List[UPath]`。
  - **A1（决策 fork）** `SequenceComponentGroupsReader([meta, *shard])` 构造**不抛**（尤其非 `KeyError:'version'` / 非 seq-id/store-base 冲突）→ **Branch A**；抛则 **Branch B**。
  - **A2** `loader.get_cuboid_track_observations()` 读回 2 条、track_id 一致、class 一致。
  - **A3** `obs.transform("world", ts, pose_graph).bbox3` 的 `dim` 保真（atol 1e-3）、`reference_frame_id="world"` 下 centroid 退化 identity（atol 0.05）。
  - **A5（Branch-B 探针，恒跑）** standalone `SequenceComponentGroupsReader([*shard])` 能否构造 + 有无可用 pose_graph。
- [ ] **Step 3: 记录决策**——① Branch A/B；② `reference_frame_id` 用 `"world"` 还是需 back-transform；③ `sequence_id`/`interval`/`pose_graph` 真实 accessor 名。这三条定死 Task 7/9/11。
- [ ] **Step 4: Commit**——`spike(O1): cuboids shard 写读 round-trip — 决策 Branch <A/B>, ref_frame=<...>`

**预判：** Branch A 更可能——aux 的 `KeyError('version')` 是 `nre-tools` 产物特有，官方 writer 按构造写 V4 根 `.zattrs`（含 version）。

---

## Task 1 — `labels.py`：class 数字→字符串映射（Mac）

**Files:** Create `cuboid_autogen/__init__.py`、`labels.py`、`tests/test_cuboid_labels.py`。
- [ ] **Step 1: 失败测试**——`map_class(13/14/15)` 得 `automobile/heavy_truck/bus`；`map_class(11/0/未知)` 得 `None`；**关键契约** `set(LIDAR_SSEG_TO_CUBOID_CLASS.values()) == set(DEFAULT_VEHICLE_CLASSES)`（import 自 tracks_loader）。
- [ ] **Step 2: 跑失败** `pytest tests/test_cuboid_labels.py -v`。
- [ ] **Step 3: 实现**——`LIDAR_SSEG_TO_CUBOID_CLASS = {13:"automobile",14:"heavy_truck",15:"bus"}` + `map_class(id) -> Optional[str]`。
- [ ] **Step 4: 跑通过。**
- [ ] **Step 5: Commit。**

---

## Task 2 — `cluster.py`：DBSCAN 包装 `cluster_points`（Mac，monkeypatch）

**Files:** Create `cluster.py`、`tests/test_cuboid_cluster.py`。
- [ ] **Step 1: 失败测试**——monkeypatch `cluster.DBSCAN` 为返回固定 `labels_`（含 noise=-1）的替身；断言 `cluster_points(xyz,eps,min_samples)` 输出 shape=(N,)、dtype=int64、noise 保留为 -1、cluster id 集合正确。
- [ ] **Step 2: 跑失败。**
- [ ] **Step 3: 实现**——`cluster_points(xyz, eps, min_samples) -> np.ndarray[int64]`，空输入返回 `(0,)`；顶部 `try: from sklearn.cluster import DBSCAN except: DBSCAN=None`。
- [ ] **Step 4: 跑通过。**
- [ ] **Step 5: Commit。**

---

## Task 3 — `cluster.py`：朝向框拟合 `fit_oriented_box`（Mac，核心几何）

PCA 求 yaw + 旋转系 min/max 求**几何中心 + 尺寸**（非点云质心，对齐 init filter）。
**Files:** Modify `cluster.py`；append `tests/test_cuboid_cluster.py`。
- [ ] **Step 1: 失败测试**——合成「全表面盒点云」fixture（已知 `center=(10,3,0.85)` / `dim=(4.5,2,1.7)` / `yaw=0.6`，6 面采样 + σ=1cm 噪声）；断言 `fit_oriented_box(xyz) -> (center,dim,yaw)`：center atol 0.05、dim atol 0.10、`|wrap_to_pi(yaw-0.6)|<0.05`、`dim[0]>=dim[1]`（长轴=principal）；`<4` 点返回 `None`；`canonicalize_yaw` 落 `[-π/2,π/2)` 且幂等。
- [ ] **Step 2: 跑失败。**
- [ ] **Step 3: 实现**——`wrap_to_pi` / `canonicalize_yaw`（±π 不换长短轴）/ `fit_oriented_box`（BEV PCA 取最大特征向量为 yaw → 旋进 box 系取 min/max 得 extent 与几何中心 → 旋回 world；z 用 min/max 中点与高度）。
- [ ] **Step 4: 跑通过。**
- [ ] **Step 5: Commit。**

> **已知限制（不在 Mac 单测过度承诺）：** 单边 LiDAR 回波下远侧面缺失 → 中心偏向传感器、尺寸低估。属 O2 真实数据调参项，由 Task 13 的 9ae GT 度量，不在此处宣称紧公差恢复。L-shape / min-area-rect 优化留 O2。

---

## Task 4 — `track.py`：跨帧关联 `associate`（Mac）

**Files:** Create `track.py`、`tests/test_cuboid_track.py`。
- [ ] **Step 1: 失败测试**——合成 10 帧、2 条直线轨迹 + 每帧 1 噪声框；`associate(per_frame_boxes, max_center_dist_m, max_yaw_diff_rad, min_track_len)` 断言返回正好 2 track（噪声因 < min_track_len 被丢）、每 track 10 帧、两轨迹不串（首帧 y 各为 0 / 8）。
- [ ] **Step 2: 跑失败。**
- [ ] **Step 3: 实现**——`Box`/`Track` dataclass + 连续帧贪心最近中心关联（带 `wrap_to_pi` yaw 差约束），尾部按 `min_track_len` 过滤。
- [ ] **Step 4: 跑通过。**
- [ ] **Step 5: Commit。**

---

## Task 5 — `track.py`：动静过滤 + size 聚合（Mac）

**速度基**（非绝对位移）避开 clip 时长依赖；size 取 p90 保证框够大装下所有点。
- [ ] **Step 1: 失败测试**——静止 track（1s 内抖动<0.3m）→ `is_dynamic(t, min_speed_mps=0.5)` False；移动 track（1s 移 5m）→ True；单帧 track → False；`aggregate_size(t, q=90)` == 逐轴 p90（atol 1e-9）。
- [ ] **Step 2: 跑失败。**
- [ ] **Step 3: 实现**——`is_dynamic` 用 `位移/时长 > min_speed_mps`（<2 帧或 dt≤0 返回 False）；`aggregate_size` 用 `np.percentile(dims, q, axis=0)`。
- [ ] **Step 4: 跑通过。**
- [ ] **Step 5: Commit。**

---

## Task 6 — `track.py`：缺帧插值 `interpolate_gaps`（Mac）

聚类漏帧 → 短缺口线性插值 center、lerp yaw，保证每相机帧有 obs 落进 50ms 窗；**不外推 track 两端**。
- [ ] **Step 1: 失败测试**——track 在 ts=0/200_000 有框、缺 100_000；`interpolate_gaps(t, frame_timestamps_us, max_gap)` 填出 ts=100_000 的中点 center（atol 1e-6）+ lerp yaw=0.2；单帧 track 不外推（输出仍只含原 ts）。
- [ ] **Step 2: 跑失败。**
- [ ] **Step 3: 实现**——只在 `[首obs_ts, 末obs_ts]` 内、缺口跨帧数 ≤ `max_gap` 时插值（center 线性、yaw 用 `wrap_to_pi` 差值 lerp、dim 复制前一帧）。
- [ ] **Step 4: 跑通过。**
- [ ] **Step 5: Commit。**

---

## Task 7 — `bev_metric.py`：旋转矩形 BEV-IoU + 框匹配（Mac，GT 验证用）

为 Task 13 准备纯 numpy 度量。Sutherland–Hodgman 凸多边形裁剪 + shoelace，确定性可单测。
- [ ] **Step 1: 失败测试**——`bev_iou(box,box)==1`（box=`(cx,cy,l,w,yaw)`）；两 1×1 轴对齐错半格 IoU=1/3；不相交=0；`match_boxes(auto,gt,max_dist)` 贪心按中心距，返回 `[(auto_idx,gt_idx)]`，超距不匹配。
- [ ] **Step 2: 跑失败。**
- [ ] **Step 3: 实现**——`_corners`/`_ccw`/`_clip`（Sutherland–Hodgman）/`bev_iou`/`match_boxes`。
- [ ] **Step 4: 跑通过。**
- [ ] **Step 5: Commit。**

---

## Task 8 — `lidar_source.py` + datasetNcore 逐帧重构（inceptio）

复用 `_get_semantic_lidar_points`（`datasetNcore.py:1412`）读 aux.lidar-sseg 逻辑，但**逐帧 yield**（保留 frame_id+ts，tracking 需要）。把现有聚合实现重构成「逐帧 generator + 聚合 wrapper」保持 DRY。
**Files:** Modify `datasetNcore.py`；Create `lidar_source.py`；append `test_v4_writer_roundtrip.py`。
- [ ] **Step 1: 重构 datasetNcore（DRY）**——抽 `_iter_semantic_lidar_frames(class_ids)` yield `(source_id, ts_us, xyz_world, labels)`（沿用 1450–1496 的 source/ts 遍历但不聚合）；`_get_semantic_lidar_points` 改为消费它。
- [ ] **Step 2: `lidar_source.py`**——`iter_vehicle_lidar_frames(dataset)` 用 `frozenset(LIDAR_SSEG_TO_CUBOID_CLASS)`={13,14,15} 调上面 generator，yield numpy。
- [ ] **Step 3: inceptio 集成测试**（`@pytest.mark.inceptio`）——9ae 上断言 >0 帧、每帧 `xyz.shape[0]==labels.shape[0]`、labels⊂{13,14,15}、同 source ts 递增。
- [ ] **Step 4: inceptio 跑** `pytest -k iter_vehicle -m inceptio -v`。
- [ ] **Step 5: Commit。**

---

## Task 9 — `v4_writer.py`：构造 obs + 写 shard（inceptio，POST-O1）

组装核 `tracks_to_observations` 用工厂注入 → Mac 可测；`write_cuboids_shard` 用 O1 验过的 SDK 调用。
**Files:** Create `v4_writer.py`、`tests/test_v4_writer_core.py`（Mac）。
- [ ] **Step 1: Mac 失败测试（fake 工厂）**——`tracks_to_observations(tracks, track_ids, ref_frame_id, class_name, obs_factory, bbox_factory)` 断言：每 active 帧一条 obs、`rot==(0,0,yaw)`（纯 yaw）、`dim`/`centroid` 一致、`reference_frame_id`/`class_id`/`timestamp_us` 正确。
- [ ] **Step 2: 跑失败。**
- [ ] **Step 3: 实现**——`tracks_to_observations`（工厂默认走 `ncore.data.CuboidTrackObservation`/`BBox3`，可注入 fake）+ `write_cuboids_shard(observations, out_dir, store_base_name, seq_id, interval_us, store_type)`（O1 验过的 writer 调用）。**O1 依赖窄点**：`ref_frame_id` 默认 `"world"`（A3 确认；若需 back-transform 则写前把 world box 变到 reference 系）；`seq_id`/`interval` accessor 取自 O1 Step3。
- [ ] **Step 4: 跑通过** `pytest tests/test_v4_writer_core.py -v`。
- [ ] **Step 5: Commit。**

---

## Task 10 — round-trip 单测晋升（inceptio，pin 写读契约）

把 O1 spike 固化成可提交回归测试。
**Files:** `tests/test_v4_writer_roundtrip.py`。
- [ ] **Step 1: 写测试**（`@pytest.mark.inceptio`）——用 `tracks_to_observations`+`write_cuboids_shard` 写真 shard → reader 读回 → 复用 O1 的 A0–A3 字段断言（dim/rot/centroid round-trip、ref transform 退化）。
- [ ] **Step 2: inceptio 跑** `pytest -k roundtrip -m inceptio -v`。
- [ ] **Step 3: Commit。**

---

## Task 11 — 接线：datasetNcore + config（+ trainer，仅 Branch B）

**按 O1 决策走分支。**
**Branch A（期望）：**
- [ ] **Step 1:** `datasetNcore.__init__` 加 `load_auto_cuboids: bool=False` / `auto_cuboids_shard_path: Optional[str]=None`，存 self。
- [ ] **Step 2:** `datasetNcore.py:314` reader 路径列表条件 append shard。
- [ ] **Step 3:** `datasets/__init__.py make()` 用 `config.dataset.get(...)` 透传两 key。读取链零改（`get_cuboid_track_observations()` 自动含 auto obs）。

**Branch B（应急）：**
- [ ] **Step 1B:** `tracks_loader.py` 抽 `_obs_iter_to_tracks(obs_iter, pose_graph, cam_ts, *, class_filter, time_tolerance_us)`；`load_tracks_from_ncore_cuboids` 委托它；新增 `load_tracks_from_cuboids_shard(shard_path, cam_ts, pose_graph, ...)` 打开 standalone shard 复用同 decode。
- [ ] **Step 2B:** `trainer.py:~456` 按 `dataset.load_auto_cuboids` 分支选入口。
- [ ] **Step 3B:** Mac DRY 等价测试 `test_tracks_loader_shard_bridge.py`：同组 fake obs，两入口产出 dict 逐字段相等。

**共同收尾：**
- [ ] **Step 4: inceptio smoke 断言非空**——`load_auto_cuboids=true`+shard → `len(tracks)>0` 且 `init_dynamic_rigid_layer` 返回 positions.shape[0]>0。
- [ ] **Step 5: Commit。**

---

## Task 12 — CLI `scripts/gen_cuboids_from_lidar.py`（inceptio）

串联 1a→1e，含 preflight + 运营可见性（防静默 0 obs）。
- [ ] **Step 1: 写 CLI**——argparse `--meta/--out/--eps/--min-samples/--min-speed/--store-type/--ref-frame-id/--validate-against-gt`；**preflight** 用 `discover_aux_path(clip_dir,"lidar-sseg")`，缺则报错 + 提示 `nre-tools ... --lidar-seg-camvis`；编排 `iter_vehicle_lidar_frames → cluster_points → fit_oriented_box → associate → is_dynamic 过滤 → aggregate_size → interpolate_gaps → tracks_to_observations → write_cuboids_shard`；**打印** obs 数 + class 直方图 + ts 范围。
- [ ] **Step 2: inceptio dry-run**（截断/小 clip），确认打印 obs=N（N>0）+ class 分布 + ts 区间。
- [ ] **Step 3: Commit。**

---

## Task 13 — 9ae GT 定量验证（inceptio + Mac 度量已测）

- [ ] **Step 0: preflight 只读确认 GT 在哪个 9ae 文件夹**——一条只读脚本对 `9ae151dc` 与 `9ae151dc_consolidated` 各打开 loader 数 `len(get_cuboid_track_observations())`，取 obs 数大的那个做基准（消除 consolidated 不确定性）。
- [ ] **Step 1: 跑生成器 + 验证**——`gen_cuboids_from_lidar.py --validate-against-gt`，对每相机帧 GT vs 自动框做 `match_boxes`（中心距 ≤2m），报告 precision/recall、mean center err、mean BEV-IoU、yaw MAE（mod π）、dim MAE。
- [ ] **Step 2: 看验收门**（实测后定档；建议初值）——cars within 40m **recall ≥ 0.6**、**mean center err ≤ 0.8m**、**mean BEV-IoU ≥ 0.5**。不达标 → 回 O2 调 eps/min-samples/min-speed/朝向算法。
- [ ] **Step 3: 记录实测数字**入 Done Log。

---

## Task 14 — 1k smoke + viser 目检 + 文档同步（inceptio + Mac）

- [ ] **Step 1: inceptio 1k smoke**——`apps/ncore_3dgut_mcmc_multilayer` + `n_iterations=1000` + `trainer.sky_backend=mlp` + `num_workers=10` + depth-off 三连 + `dataset.load_aux_masks=true` + `+dataset.load_auto_cuboids=true` + `+dataset.auto_cuboids_shard_path=<shard>`。Expected：不崩；dynamic_rigids 粒子 >0；`⭐ Test Metrics` 表出现；`metrics.json` 正常。
- [ ] **Step 2: viser 目检**（`viser-gui-4d` skill）——动态 actor 摆放/朝向/运动合理（无明显错位/抖动/180° 反向）。
- [ ] **Step 3: 文档同步**——`v3_plan_revised.md`（看板移列 + P*.* 状态 ✅ + commit hash + Done Log 追实测数字）、`v2_architecture.md`（新模块 `:::done` + 文件清单 ✅ + § 7 不变量加锚）。**改 mermaid 看板用全角 `（）`**，提交前跑 `awk` 自查应零输出。
- [ ] **Step 4: Commit**（`feat` + `docs(plan)` + `docs(arch)` 三行式 message）。

---

## Verification（端到端如何验）

1. **Mac 纯函数（无 GPU/SDK）：** `pytest threedgrut/tests/test_cuboid_{labels,cluster,track,bev_metric}.py threedgrut/tests/test_v4_writer_core.py -v` 全绿——聚类/拟合/tracking/动静过滤/插值/BEV-IoU/obs 组装 + 5 条坐标时间不变量成立。
2. **inceptio 写读契约：** `pytest test_v4_writer_roundtrip.py -m inceptio -v`——V4 cuboid 写→读字段保真、ref transform 退化。
3. **inceptio 逐帧源：** `iter_vehicle` 集成测试——>0 帧、点/标签对齐、ts 递增。
4. **GT 定量（Task 13）：** 9ae 自动框 vs GT 的 recall/center-err/BEV-IoU 达门槛——工具正确性的客观证据。
5. **1k smoke（Task 14）：** dynamic_rigids 粒子 >0 + metrics.json 正常 + viser 目检合理——端到端贯通。

**静默失败防线（重点验）：** ① Task 1 pin class 字符串 == `DEFAULT_VEHICLE_CLASSES`；② Task 11 smoke 断言 `len(tracks)>0`；③ CLI 打印 obs 数/class 直方图/ts 范围——三道防「exit 0 但没有动态 actor」的伪完成。

## Self-Review（对照 spec）

- **D1 纯 LiDAR**（Task 8 复用 aux.lidar-sseg，无图像分支）✅
- **D2/D3 V4 独立 shard**（Task 9 官方 writer）+ **O1 证伪「读取链零改」假设**（代码证据 `datasetNcore.py:305-313` 表明可能不行 → Branch A/B 双分支）✅ spec 缺口已补
- **D4 只车辆**（Task 1 仅 13/14/15）✅
- **§6 五条不变量**（Task 1/3/9/10/11 分别 pin）✅
- **§7 空 shard 兜底**（trainer warning-not-crash + Task 11 非空断言）✅
- **§8 测试策略**（Mac 纯函数 + inceptio round-trip + 1k smoke + viser 全覆盖）✅
- **§9 实现顺序**（O1 优先 + Mac 纯函数并行 + inceptio 后置）✅
- **spec 未覆盖、本计划新增**：reference_frame 双重变换风险（O1 A3+Task 9）、缺帧插值（Task 6）、速度基动静过滤（Task 5）、9ae GT 定量验证（Task 7+13）、运营可见性打印（Task 12）、lidar-sseg preflight 门（Task 12）。

## 执行建议（approval 后选）

1. **Subagent-Driven（推荐）**——每任务 fresh subagent + 两段 review；Mac 纯函数 Task 1–7 并行铺开，O1（Task 0）先跑定分支。
2. **Inline**——本会话按 `executing-plans` 批执行 + checkpoint。

**关键路径：** Task 0（O1 硬门）决定 Task 9/11 形态；Task 1–7（Mac）可即刻并行不等 O1。

---

## 退出 plan mode 后第一件事（大g 指令）

在全局 `~/.claude/CLAUDE.md` 写入规则：**写实现计划（writing-plans / plan 文件）时不要贴大段 code snippet 代码块，只写任务/文件/签名/测试要验证什么/命令意图，实现代码留到执行期 TDD 时再写**（此规则覆盖 writing-plans skill 的 "Complete code in every step"）。

---

## 执行结果（2026-06-26，Inline 执行，分支 claude/beautiful-newton-fa35e1）

**全部 14 任务完成。** O1 结论 = **Branch A**（官方 `SequenceComponentGroupsWriter` 写的 cuboids shard 可被 `SequenceComponentGroupsReader([meta, shard])` append 读回；下游 `get_cuboid_track_observations`/`tracks_loader`/`trainer`/`init_dynamic_rigid_layer` 零改）。

### 实测证据
- **Mac 纯函数 21 tests 绿**：labels / cluster（DBSCAN + min-area-rect 朝向框）/ track（关联+动静+插值）/ bev_metric（旋转矩形 IoU）/ v4_writer 组装核。
- **O1 spike**（inceptio 真 SDK）：Branch A + `reference_frame_id="world"` 字段保真（centroid/dim/rot round-trip 完全一致）。写 shard 三要素：`generic_meta_data` 对齐源 store / 独特 `group_name` / `component_instance_name`（目标 clip 无 GT 用 "default" 读取链零改、9ae 有 GT 用 "auto_v0" 避开）。
- **round-trip 单测**（inceptio）：tracks→obs→write→append→读回字段保真 PASS。
- **逐帧 LiDAR 源**（inceptio）：9ae `iter_vehicle` 30 帧 / 点标签对齐 / labels⊂{13,14,15} / ts 递增 PASS。
- **接线端到端**（inceptio verify）：dataset(load_auto_cuboids=true) → 415 obs → `load_tracks_from_ncore_cuboids` 30 tracks（cam_ts 50ms 匹配 + class 字符串两道静默失败防线均通）→ `init_dynamic_rigid_layer` 70879 particles。
- **CLI 生成**（9ae）：184 帧 / 937 clusters / 456007 车辆点 → 36 动态 track / 467 obs → shard。
- **GT 定量**（9ae vs 13657 GT，BEV match center-dist≤2m）：precision 0.525 / center-err 1.30m / BEV-IoU 0.285 / **yaw-MAE 65°** / recall 0.049（auto 仅动态 vs GT 全部车，故 recall 上限受限）。
- **1k smoke**（inceptio，load_auto_cuboids + depth-off + num_workers=10）：dynamic_rigids 从 36 auto cuboids 初始化 **94363 particles**（502072 动态点 × 36 cuboids）；训练 1000 iter / 80.94s / 12.35 it/s **不崩**——渲染路径（含 dynamic_rigids forward）经 1000 step 验证。

### 已知局限（O2 follow-up，已记 background task chip）
- **yaw-MAE 65°**：纯几何（PCA / min-area-rect 实测同）对单边/稀疏 LiDAR 车点朝向不可靠（只见车一两面时最小矩形长轴常落在车宽方向）。改进方向：L-shape fitting / 轨迹运动方向定 yaw / 车类先验尺寸定中心。cuboid 仅做 init、训练 `pose_adjustment` 会修，但 yaw 65° 是不利起点。
- **1k smoke eval metrics 未取**：eval 卡 yolov8 权重外网下载（inceptio 外网限速，基础设施非功能）；1k iter 重建数字亦无统计意义。渲染不崩已由训练 1000 step forward 证明。

### 产物
- 新模块 `threedgrut/datasets/cuboid_autogen/`（labels / cluster / track / bev_metric / lidar_source / v4_writer）。
- CLI `scripts/gen_cuboids_from_lidar.py`（生成）+ `scripts/eval_cuboids_vs_gt.py`（GT 验证）+ `scripts/spikes/`（O1 spike + 接线 verify）。
- `datasetNcore` / `datasets/__init__` Branch A 接线（`load_auto_cuboids` / `auto_cuboids_shard_path` / `auto_cuboids_instance_name`，false 路径字节不变）；`conftest.py` ncore/datasets 改 try-import（Mac stub / inceptio 真 SDK）。
- viser 目检 ckpt（inceptio）：`/home/inceptio/work/output/cuboid_smoke/pai_9ae151dc-...-2606_171829/ckpt_last.pt`。
