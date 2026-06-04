# 可行性分析：用 asset-harvester 生成的高斯注入/替换 3dgrut 的前景 actor

- **日期**：2026-06-04
- **类型**：可行性诊断（**不含实施计划**）
- **触发**：诊断 doc [`2026-06-04-v3-actor-centric-perf-diagnosis.md`](2026-06-04-v3-actor-centric-perf-diagnosis.md) 锁定 v3 目标=「前景 actor per-class 重建质量」后，用户提出 alternative —— 用 asset-harvester 生成 per-object 高斯，塞进/替换 3dgrut 训练出的 dynamic（车）+ deformable（人）高斯。
- **范围边界**：本文只判可行性与取舍，不落实施。沿用诊断 doc 的 v3=重建质量 / v4=编辑仿真 的分界。
- **状态**：技术底盘两边已查实；asset-harvester 可运行性已由 `asset-harvester-verify` 实测确认；结论待用户复核。

---

## 0. TL;DR

**这不是一个问题，是三个，结论各不相同：**

1. **车辆（dynamic_rigids）**：✅ **高度可行，但必须走 warm-start（注入当初始化、继续多帧训练），不能 frozen drop-in。** asset-harvester 的扩散先验补全了「驾驶日志里永远看不到的那一面」——这正是 3dgrut 多帧 photometric 训练**根本解决不了**的死角（未观测面=blob）。两者**互补不是竞争**。

2. **行人（dynamic_deformables）**：⚠️ **asset-harvester 是错的工具。** 3dgrut 这一层是空壳，asset-harvester 又只产**静态**资产（不会走路）。正解是把 **DriveStudio 的 SMPL-LBS deformable 高斯**移植进来，asset-harvester 至多当「从无到有的静态 blob 垫脚石」。

3. **单帧 vs 多帧的表征差异（用户核心顾虑）**：✅ **差异确实很大，且这恰好证明 frozen 会退化、warm-start 会赢。** frozen drop-in 丢掉 3dgrut 已经赢的三栏（本场景光照匹配 / view-dependent SH / 多视角整合）；warm-start 用同一套多帧训练把这三栏全拿回来，同时白拿扩散补全的完整几何。

**一句话**：asset-harvester 的正确定位**不是「替换训练结果」，而是「给训练一个补全了未观测面的高质量起点」**。对车，这是高 ROI 的 v3 动作；对人，换 DriveStudio 法。frozen drop-in 留给 v4 编辑。

---

## 1. 命题拆解：为什么是三个问题

用户原话「替换 dynamic gaussian、deformable gaussian」隐含三个被混在一起的子命题，可行性天差地别：

| 子问题 | 3dgrut 现状 | asset-harvester 适配度 | 本文章节 |
|---|---|---|---|
| **车辆** automobile/truck/bus | `dynamic_rigids` 层，边界清晰、**有现成注入口** | ✅ 强匹配（同构高斯，格式机械可转） | §4 |
| **行人/骑行** person/rider | `dynamic_deformables` = **空壳层**（`is_particle_layer=False`，无模块/无初始化/无渲染） | ⚠️ 弱匹配（只产**静态**资产） | §5 |
| **单帧 asset vs 多帧训练** | 跨整个 clip 多帧拟合的静态 SH（degree 3） | 决定 warm-start vs frozen 的关键 | §3 |

---

## 2. 两边技术底盘（证据锚点）

### 2.1 asset-harvester 产出什么

- **管道**：NCore 解析（object-centric 多视图裁剪）→ SparseViewDiT 多视图扩散（1–4 稀疏视角 → 16 一致视角）→ TokenGS feed-forward lifting → NuRec 坐标变换 + metadata。
- **产物**：标准 3DGS PLY，每高斯 `[xyz(3) + opacity(1, 已 sigmoid) + scale(3, log) + rot quat(4, wxyz) + RGB(3, =f_dc·SH_C0+0.5)]`。**纯 DC-SH（无高阶 view-dependent SH）**。
  - 关键文件：`asset-harvester/asset_harvester/tokengs/ply_io.py`、`.../utils/gaussians.py`、`.../utils/orient_gaussians_for_nurec.py`、`.../utils/generate_external_assets_metadata.py`
- **类别**：`VRU_pedestrians→person`、`consumer_vehicles→automobile`，及 bus/truck（`ncore_parser/ncore_object_parser.py`）。
- **关键性质**：**完全 feed-forward 的静态 canonical 资产**——无骨骼/无变形/无动画/无时间维。单卡 RTX 4090 ~17s/对象。
- **坐标/尺度**：Objaverse 归一化系（物体中心在原点，按 `max(lwh)·bbox_size`, 默认 `bbox_size=0.8` 归一化），保存前缩回原尺度；NuRec 插入需 Y 轴 90° 旋转。metadata 带 `cuboids_dims=[L,W,H]` + `label_class`。

### 2.2 3dgrut 前景层

- **分层**：`LayeredGaussians` 容器，各层是独立 `MixtureOfGaussians`，渲染时 `fused_view()` 拼接（[layered_model.py:926](../../../threedgrut/layers/layered_model.py)）。enabled 默认 `[background, road, dynamic_rigids, sky_envmap]`（[ncore_3dgut_mcmc_multilayer.yaml:53](../../../configs/apps/ncore_3dgut_mcmc_multilayer.yaml)）。
- **dynamic_rigids = 边界清晰、可注入的 actor 集合**：
  - 参数 schema：`positions[N,3]` / `rotation[N,4 wxyz]` / `scale[N,3 log]` / `density[N,1 log]` / `features_albedo[N,3 DC-SH]` / `features_specular[N, ·]`（SH degree 3）（[model.py:160](../../../threedgrut/model/model.py)、[base_gs.yaml:162](../../../configs/base_gs.yaml) `max_n_features:3`）。
  - **坐标系=object-local**（X 前 / Y 左 / Z 上），per-frame SE(3) pose 在渲染时由 `_transform_means_and_active()` 应用（[layered_model.py:1092](../../../threedgrut/layers/layered_model.py)）。
  - **per-actor 身份**由 `track_ids` buffer 切分，可 mask 单独访问/替换。
  - **现成注入口**：`init_layer_from_points("dynamic_rigids", positions, rotations, scales, densities, colors, track_ids, setup_optimizer=True)`（[layered_model.py:1250](../../../threedgrut/layers/layered_model.py)）。
  - 颜色约定：`colors = features_albedo·SH_C0 + 0.5`。
- **dynamic_deformables = 空壳**：仅在 registry 注册（`is_particle_layer=False`），无模块实例、无初始化、无渲染（[registry.py:55](../../../threedgrut/layers/registry.py)）。Stage 16 计划用 permuto hash-grid + FullyFusedMLP **形变网络**（非 SMPL），仍是 stretch/未实现（v3_plan.md Stage 16）。

---

## 3. 核心问题：单帧 asset vs 多帧训练，差异有多大（回答用户 Q2）

用户的顾虑：3dgrut 里一个 gaussian 活在整个 clip 的多帧中，训练出的是「多帧拟合」结果，而 asset-harvester 是单帧 feed-forward——直接替换是否差异很大？

**查实结论：差异很大，且方向是「frozen 会退化、warm-start 会赢」。**

3dgrut 训练出的车 gaussian 到底多捕捉了什么（[layered_model.py:926](../../../threedgrut/layers/layered_model.py) fused_view：同一组 object-local 高斯按 per-frame pose 变换后渲进 actor 出现的**每一帧**，photometric loss 全程累加）：

| 信息维度 | 3dgrut 多帧训练 | asset-harvester 单帧 |
|---|---|---|
| **未观测几何**（车背面/人侧面） | ❌ 只能重建看到的；看不到的=blob | ✅ **扩散先验幻觉补全 360°** |
| **本场景光照/曝光匹配** | ✅ 直接对**本 clip 像素**优化 | ❌ 来自扩散先验，跨场景会「贴上去」违和 |
| **view-dependent 高光** | ✅ SH degree 3 specular（±30° 内有效） | ❌ 仅 DC-SH，无高阶视角依赖 |
| **多视角整合** | ✅ actor 出现的每一帧累加 loss | ❌ 单次 feed-forward 快照 |
| **时变光照**（阴影→阳光） | ❌ baseline 也没有（Fourier SH 是 Stage 13b **未启用**计划，诊断还建议停掉） | ❌ 同样没有 |

**关键转折：**
- **frozen drop-in** 直接用 asset-harvester 外观当最终结果 → **丢掉上表中 3dgrut 已经赢的三栏**（本场景光照、view-dependent、多视角整合）。对 v3 的 class PSNR 多半是**退化**：车看起来像 P 上去的，开过光照变化处尤其明显，且没有 view-dependent 高光。
- **warm-start** 把 asset-harvester 当**初始化注入**，再用**同一套多帧 photometric + pose 训练**重新拟合 SH/光照/视角依赖 → **恢复了**那三栏，同时白拿扩散补全的完整几何。

> 所以用户直觉完全对：单纯替换（frozen）差异大且多半变差。但这恰恰说明 asset-harvester 的价值不在「替换训练」，而在**补上多帧训练根本补不了的未观测面**——它和训练是互补关系。这一点直接打在诊断 doc §2.4 / §3 锁定的「未观测面 blob」死角上。

---

## 4. 车辆（dynamic_rigids）可行性 —— ✅ 高度可行（warm-start）

### 4.1 格式 / 坐标转换映射（机械层，well-defined）

| 字段 | asset-harvester | 3dgrut 期望 | 转换 |
|---|---|---|---|
| opacity | sigmoid 后 `[0,1]` | `density` log/logit 前体 | `density = logit(opacity)` |
| color | RGB `=f_dc·SH_C0+0.5` | `features_albedo`（DC-SH） | `features_albedo = (RGB−0.5)/SH_C0` |
| 高阶 SH | **无** | `features_specular`（SH degree 3） | **补零**（训练时再学出来） |
| scale | log-space | log-space | 直接（注意单位/尺度还原，见下） |
| rotation | wxyz quat | wxyz quat | 直接（注意 canonical 朝向对齐，见下） |
| 坐标系 | Objaverse 归一化 | object-local（X前Y左Z上）+ 米制 | **旋转对齐 + 用 `cuboids_dims` 还原米制尺度** |
| 身份 | per-asset PLY + metadata | `track_ids` | 按 track ↔ asset 映射赋 `track_ids` |

机械字段（opacity/color/scale/rotation）转换是确定性的、低风险。**真正的工程难点是坐标/尺度对齐**（§4.3）。

### 4.2 两种注入模式

| 模式 | 做法 | 适用 | 判断 |
|---|---|---|---|
| **warm-start**（推荐 v3） | asset 当初始化注入 dynamic_rigids，**保持 nn.Parameter 可学**，继续多帧 photometric + pose 训练 | v3 重建质量 | ✅ 恢复光照/SH/多视角，几何补全白拿；对 asset 质量不完美**鲁棒**（训练会修） |
| **frozen drop-in** | 注入即冻结，asset 外观=最终渲染 | v4 编辑/换车/删插不留痕 | ⏸️ 丢多帧整合（§3），v3 多半退化；asset 质量缺陷**直接暴露** |

### 4.3 工程风险（warm-start）

1. **per-track 坐标/尺度对齐**（头号风险）：Objaverse 归一化 canonical → 3dgrut object-local 的旋转必须对齐（asset-harvester canonical 朝向 vs NCore cuboid Euler XYZ 解码朝向，[dynamic_rigid_init.py:35](../../../threedgrut/layers/dynamic_rigid_init.py)），尺度用 `cuboids_dims` 还原。对齐错 → 车浮空/错位/缩放错。**`asset-harvester-verify` 已证 cuboids_dims 真实可用**（§6）。
2. **变长粒子注入的 plumbing**：注入改变粒子数，需 `setup_optimizer()` 重置 Adam state、`LayeredMCMCStrategy` 重新同步计数/索引、checkpoint 兼容（`track_ids` 存进 ckpt）。已知工程，非阻塞。
3. **外观域差**：扩散先验光照 ≠ 本场景。warm-start 下由训练消化；仍建议保留 per-track albedo bias（V3-L8）协同。
4. **输入获取**：需 per-track 抽出 object-centric 稀疏视角喂 asset-harvester——它自带 `ncore_parser` 正好做这件事，与 `asset-harvester-verify` 已跑通的流程一致。

### 4.4 判决（车辆）

**warm-start 高度可行且高 ROI**：它攻击的是 photometric 训练**结构上补不了**的未观测面 blob，直接抬 class PSNR，且与诊断 Phase 1（track-pose 收尾、per-track 外观）**协同不冲突**——asset-harvester 给「完整几何 + 外观先验」，track-pose 修运动，per-track bias 修色偏。**frozen drop-in 不适合 v3**，留给 v4。

---

## 5. 行人（dynamic_deformables）可行性 —— ⚠️ 换 DriveStudio 法

### 5.1 三方法对比

| 方法 | 机制 | 会走路？ | 输入需求 | 成熟度 | 判断 |
|---|---|---|---|---|---|
| **asset-harvester 静态 + 刚性摆放** | 静态 canonical 人按 per-frame pose 刚性放置 | ❌ 帧间鬼影/肢体错位 | 稀疏视角 | 已验证产出 | 仅「从无到有的静态 blob 垫脚石」，**非 deformable 解** |
| **DriveStudio SMPL-LBS**（推荐） | canonical 高斯长在 SMPL mesh 上，per-frame 24 关节四元数经 LBS 蒙皮变形 | ✅ 关节动画 | **SMPL pose/shape**（HMR2 预测）+ trans | **成熟、代码在手** | 真正的 deformable 解 |
| **3dgrut Stage 16**（计划） | permuto hash-grid + MLP 形变网络 | ✅（理论） | canonical + 时间 | ⬜ 未实现、无人体先验 | 更通用但未验证，结构性弱于 SMPL |

### 5.2 DriveStudio SMPL 法的代价与可行性

- **机制**（`drivestudio/models/nodes/smpl.py:158`、`drivestudio/models/human_body.py:83`）：canonical 高斯在 SMPL 6890 顶点细分网格上初始化；per-frame body pose `theta[B,24,4]` → LBS `T=W@A` → `deformed = R·canonical + t`。端到端 photometric + 时间平滑 + voxel-deformer 正则。
- **输入**：per-frame SMPL `pose(24 quat) / betas(10) / trans(3)`，由 HMR2 从 RGB 自动预测（`drivestudio/datasets/tools/extract_smpl.py`）。NCore 有多相机 + 2D human mask，利于 HMR2 拟合，但需：HMR2 在 NCore 相机上跑通 + 全局运动估计 + 坐标系对齐。
- **移植代价**：把 `SMPLNodes` 的 canonical+LBS+正则移植进 3dgrut 那个空的 `dynamic_deformables` 层（替换 stub），复用 `MixtureOfGaussians` 渲染。中-重工程，但 DriveStudio 就在工作目录里、方法成熟。

### 5.3 判决（行人）

**asset-harvester 不是行人的答案**（静态本质 ≠ 会走路）。正解：**DriveStudio SMPL-LBS 移植**填进 dynamic_deformables 空壳，优先级高于 3dgrut 自带 Stage 16 hash-grid+MLP（后者无人体先验、未验证）。asset-harvester 至多在「rigid 垫脚石」阶段提供静态人 blob，或为 SMPL canonical 高斯提供外观 init（边际收益，非必需）。

---

## 6. 实际可运行性 —— 闸门已解（`asset-harvester-verify` 实测）

> 「现在就能做」而非「理论可行」：asset-harvester 已在**真实 NCore/NuRec 驾驶数据**上端到端跑通。

| 证据 | 内容 | 来源 |
|---|---|---|
| 端到端跑通 | RTX 4090（vast.ai），17s/对象，权重 `AH_{multiview_diffusion,tokengs_lifting,camera_estimator}.safetensors` | `asset-harvester-verify/HANDOVER.md` |
| 真实产出 | **3 车 + 3 人**，各 ~99k 高斯、5–5.7 MB；cuboid 真实（车 4.47×1.82×1.43m / 人 ~0.6×0.6×1.7m） | `asset-harvester-verify/verify_assets/bundle/*.ply` + `metadata.yaml` |
| 未观测面补全 | 80 帧 360° orbit 视频肉眼证实完整几何（扩散幻觉质量） | `.../harvester_3d_lifted.mp4` |
| 输入来源 | 1–4 视角稀疏观测，**从 NuRec dynamic gaussian 导出的 object-level** | `HANDOVER.md` |

**两个必须写明的限定：**
1. **无量化指标**：质量靠 orbit 视频肉眼判断，**没有 PSNR/LPIPS** 对本 clip GT。→ 对 warm-start **不致命**（训练会修）；但若想做 frozen drop-in，需先补量化评测。
2. **NuRec apply-schema 卡住 ≠ 本路径受阻**：verify 里卡在 NuRec **运行时编辑**的 `--edit-assets` JSON schema（NuRec 26.02 是两阶段 runtime replace，非重训）。**我们的 3dgrut warm-start 路径根本不经过 NuRec**——直接把 PLY 转换后注入 `init_layer_from_points`，所以那个 last-mile blocker 与本可行性**无关**。

---

## 7. 与诊断 doc 的衔接 + 推荐路线

本 alternative **不与诊断正交，反而精准命中**诊断锁定的「前景 actor class 质量」轴：

| 诊断 Phase | 原计划 | asset-harvester alternative 的位置 |
|---|---|---|
| Phase 0 测量 | 新建 per-class 评测 | 不变（warm-start 前后对比需要它） |
| Phase 1 车辆 | track-pose 收尾 + per-track 外观 | **+ asset-harvester warm-start**：补未观测面 blob，与 track-pose / per-track bias **协同** |
| Phase 2 行人 | rigid 垫脚石 → deformable | **改用 DriveStudio SMPL 法**移植 dynamic_deformables；asset-harvester 至多当静态 blob 垫脚石 |
| v4 编辑 | inpaint / 软分割 | **asset-harvester frozen drop-in 归这里**（换车/删插不留痕） |

**推荐路线（按 ROI / 对齐度）：**
1. **车辆 asset-harvester warm-start**（v3 高 ROI，可运行性已证，攻击训练补不了的死角）。
2. **行人改道 DriveStudio SMPL 移植**（asset-harvester 在此是错工具）。
3. **frozen drop-in 推迟到 v4**（编辑/仿真目标，需先补量化评测 + 域适配）。

---

## 8. 关键风险 / 未验证项清单

- [ ] **坐标/尺度对齐**：Objaverse 归一化 canonical → 3dgrut object-local 的旋转对齐 + `cuboids_dims` 米制还原，per-track 正确性（头号风险）。
- [ ] **变长粒子注入 plumbing**：`setup_optimizer` / MCMC strategy resync / ckpt 兼容。
- [ ] **量化质量未知**：asset-harvester 输出无 PSNR/LPIPS vs 本 clip GT；warm-start 鲁棒，frozen 需先补测。
- [ ] **外观域差**：扩散先验光照 vs 本场景曝光；warm-start + per-track bias 消化。
- [ ] **行人 SMPL 链路**：HMR2 在 NCore 相机跑通 + 全局运动估计 + DriveStudio SMPLNodes 移植进空壳层。
- [ ] **per-track 输入抽取**：asset-harvester `ncore_parser` 对本仓库 NCore manifest 的适配（verify 用的是 NuRec USDZ 导出，路径可能不同）。

---

## 9. 待决定的后续

- [ ] 本可行性是否转实施 plan？若转，**先做车辆 warm-start 的最小验证**（1–2 track，注入 → 5k smoke，对比 class PSNR）。
- [ ] 行人是否立项 DriveStudio SMPL 移植（独立较大工程，可单开 spec）。
- [ ] frozen drop-in / v4 编辑是否需要单独 spec（含量化评测补齐）。
