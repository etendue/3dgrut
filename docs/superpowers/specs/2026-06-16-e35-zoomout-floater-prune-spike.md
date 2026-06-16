# E3.5 — zoom-out floater prune / clean（后处理几何 prune spike）

- **编号**：E3.5（新增，归 E3 表示侧；2026-06-16 大g 提议）
- **状态**：⬜ Spike 提纲（待排期）
- **一句话目标**：对已训练 ckpt 做**后处理几何 prune**，砍掉 zoom-out / 拉远 / 俯视视角下暴露的漂浮重影（floater），**不重训、不卡 R9**，并立一个 floater 量化锚。
- **2026-06-16 review 加固**（deep-research + codebase 三路核实后）：① prune 改 **「几何候选 ∩ 低贡献 gate」双闸门**（堵 alpha-抵消型 floater 砍了破守护线的风险，见 § 2）；② road 上方 `z>Δ` 由「判决」降为「候选」（堵合法高物体误砍，见 § 2 / Task 0）；③ 交付物重排 —— **zoom-out 量化锚 + R10 探针列 P0**（不依赖 prune 调参成败），守护线达标列 P1（见 § 3）；④ 订正三处 codebase 表述（68%/24% 属 v3 诊断非 docstring、road 豁免走 LayeredGaussians 分层而非 `exempt_layers`、0.4m 非硬编码），并登记三个现成基建利好。

---

## 1. 背景与证据

- **现象（大g 2026-06-16 提出）**：当前 ckpt zoom out 后有大量重影、漂浮 artifacts。
- **codebase 现状（2026-06-16 核实）**：baseline 走 MCMC（[`threedgrut/strategy/layered_mcmc.py`](threedgrut/strategy/layered_mcmc.py)），靠 **relocate 处理「死粒子」（低 opacity）**（`max_relocation_fraction=0.4`，`exempt_layers_opacity_reg=[road]`）。
- **真空地带**：zoom-out floater 大多是**高 opacity、几何深度错**的高斯——在训练 ego 轨迹（贴地前视）被遮挡或 alpha 抵消，拉远/俯视才暴露。**opacity-based relocate 抓不到它们**。当前 pipeline 完全没有针对这类的 prune 通道。

### 核心洞察

floater 本质 = **几何欠约束 + scale-space aliasing**：训练只有地面附近前视轨迹 → 空中/远处 severely under-constrained → 高斯放在错误深度只要 reproject 到训练视角对得上就不被惩罚 → zoom out 即成漂浮重影。它们「投影对、深度错」，所以**只能用几何 / 多视一致性信号抓，不能用 opacity 抓**。

### 核心洞察 B —— road 兜底 bg floater（2026-06-16 大g 观察 + 双重证据）

**docstring 实锤**（[`road_region.py`](threedgrut/model/road_region.py) L2-14 核实）：**~750k bg 粒子毯覆盖路面、主导其渲染，专门的 `road` 层 opacity 仅 ~0.014（几乎隐形）**；V3-R2 opacity penalty（实际函数 `compute_bg_road_opacity_penalty`）判据为 `|z − ground_z| < z_band` 的**贴地窄带**——**窄带之上的空气区悬浮 bg 完全零约束 = novel 鬼影主源**（docstring + 代码直接证实的真空地带）。

**v3 诊断实测**（非 docstring，来源 [`v4_plan.md`](v4_plan.md) § 6 / § 305「2026-06-11 外推诊断」）：penalty 已压路面区 bg −86%，但 **road 自盖仍 68% / bg 替补仍 24%**。

对照 NuRec「删 road 层 = 黑洞」（源头所有权切分），我方是「bg 兜底 + 事后窄带驱逐」。这批**空气区 bg 兜底粒子是不受 road 冻结五件套约束的自由高斯，是 off-track 退化的重要嫌疑**——给本卡一个最对症的 prune **候选**信号（注：「候选」非「判决」，须再过 § 2 双闸门——bg 也可能兜底跨街杆 / 低垂枝叶等合法高物体，由 Task 0 ownership 诊断先甄别）。

---

## 2. 方法（后处理 prune 脚本 `scripts/e35_floater_prune.py`）

**管线**：读 ckpt → 用 [`model.py`](threedgrut/model/model.py) 的 `get_positions/get_scale/get_density/get_covariance` 取参数 → **双闸门选 prune 集** → mask → 写回 pruned ckpt/ply（`export_ply`）。

### ⭐ 核心设计：双闸门（geometry 候选 ∩ 低贡献 gate）

deep-research（StableGS, arXiv 2503.18458）的 floater 根因：floater 的混合颜色与背景误差**相消达伪平衡**后 opacity 梯度消失 —— 即 **floater 在训练视角往往是「有贡献」的**（参与 alpha 合成、抵消误差）。据此把 floater 分两类，**只有一类后处理硬砍是安全的**：

| floater 类型 | 训练视角累计 alpha 贡献 | 后处理硬砍 | 归属 |
|---|---|---|---|
| **遮挡型**（藏在不透明物后） | ≈ 0 | ✅ 安全，守护线稳 | **本卡 prune** |
| **alpha-抵消型**（漂在物体前 / 半透明） | 高 | ❌ 砍了训练 PSNR 必掉（可能 ≫ 0.1dB） | 留训练期正则（E3.1 / StableGS Dual-Opacity） |

⚠️ **关键张力**：下表 depth 一致性信号（「中心深度 ≪ LiDAR depth → 漂在物体前方空中」）**恰恰命中 alpha-抵消型**——也就是守护线最易破的那类。因此 prune 集 **= geometry 候选 ∩ 低贡献 gate（必要条件，非加权项）**：

> **prune 当且仅当**：`(road 空气区 bg ∨ depth 不一致 ∨ scale 异常 ∨ KNN 孤立)` **AND** `跨训练视角累计 alpha 贡献 < τ_vis`。

geometry 信号负责「可疑」，低贡献 gate 负责「砍了不破守护线」——把守护线从「赌阈值」变成「结构性保证」。

**闸门 1 — geometry 候选信号**（命中任一即「可疑」，非直接判决）：

| 信号 | 判据 | 抓什么 |
|---|---|---|
| **road 区空气区 bg** ★★ | **bg 层**粒子 xy 落在 road occupied cell 且 `z > ground_z + Δ`（`Δ` = penalty z_band 之上盲区下界，**E3.1 待 A/B，本卡先取保守上界**如 0.4m），复用 `query_ground_z` | road 兜底 bg 鬼影（最对症，攻 layer ownership 根因）|
| **LiDAR/多视 depth 一致性** ★ | 高斯中心投影到训练相机，中心深度 ≪ 该像素 LiDAR depth GT（[`datasetNcore.py`](threedgrut/datasets/datasetNcore.py) / [`aux_readers.py`](threedgrut/datasets/aux_readers.py)）→ 漂在物体前方空中。⚠️ **此信号偏 alpha-抵消型，强依赖闸门 2** | 「几何位置错」的高斯 |
| **scale aspect** | `max_scale/min_scale` 过大（细长针状）或绝对 `max_scale` 过大 | 朝相机的细长 splat / 糊脸巨片 |
| **空间孤立度** | KNN 距离，孤立无邻居 | 空中孤立 floater |

**闸门 2 — 低贡献 gate**（**必要条件**，与闸门 1 取交集）：

| gate | 判据 | 作用 |
|---|---|---|
| **可见性 / 贡献度** ★★ | 跨训练视角累计 alpha 贡献 `< τ_vis`（Mini-Splatting / PUP-3DGS 思路）| **守护线护栏**：只放行训练视角贡献低的（遮挡型），挡住 alpha-抵消型 |

> depth 一致性依赖 LiDAR 覆盖；远场 / 天空无 LiDAR → 用孤立度 + scale + sky-mask 补。**注意 zoom-out floater 多在远场 / 空中（LiDAR 打不到）→ 那里主要靠 孤立度 + scale + 闸门 2，depth 信号对最关键目标区可能失效。**

### 分层 prune（订正：走 LayeredGaussians 分层，非复用 exempt_layers）

`LayeredGaussians` 本就**逐层存储**（`model.layers["road"]` / `["background"]` 各一个 MoG）→ **逐层取 positions、road 层整层跳过**即可。注意 `exempt_layers_opacity_reg` 仅豁免 **opacity L1 正则**（走 `get_density_excluding()`），**不是**通用层级 prune 基建，本卡不复用它，而是直接按层迭代。

### 现成基建（核实确认，省一半实现风险）

- **prune + optimizer state 一致性**：复用 [`strategy/base.py`](threedgrut/strategy/base.py) `_update_param_with_optimizer()`（`gs.py:prune_gaussians_opacity` 即范例）→ 删行后 Adam m/v 同步、写回 ckpt 可正常 load。
- **高斯中心投影（含 FTheta 鱼眼）**：复用 [`scripts/dump_lidar_depth_map.py`](scripts/dump_lidar_depth_map.py) `project_pinhole` / `_project_and_depth` / `ray_depth_from_cam_pts`（NCore 鱼眼，自写易错）。
- **LiDAR depth / sky-road mask**：`LidarDepthAuxReader.read(camera_id, ts_us) → [H,W] ray-depth（0=无命中）`、`sky_mask` / `road_mask`（[`aux_readers.py`](threedgrut/datasets/aux_readers.py) / [`datasetNcore.py`](threedgrut/datasets/datasetNcore.py)）。
- **road 高度场**：`query_ground_z(xy[N,2], hf) → (ground_z, valid)` + `build_road_height_field` 的 `occupied[H,W]`（[`road_region.py`](threedgrut/model/road_region.py)）现成。

---

## 3. 验收指标（按交付优先级重排）

> **重排原则**：zoom-out 量化锚 + R10 探针 + ownership 诊断**不依赖 prune 调参是否一次到位** → 列 **P0（本卡下限保证）**；prune 真砍 + 守护线达标依赖阈值调对 → 列 **P1**。即使 P1 没调到位，P0 三件已是独立有价值产出（v4 目前缺 zoom-out 维度量化）。

### P0 — 不依赖 prune 成败的独立产物

- **floater 量化锚（zoom-out proxy）★**：**先 pin 死一组 zoom-out / 俯视 / pull-back 相机轨迹**（可复现），指标 = **sky-intrusion rate**（sky-mask 区域内 alpha > 阈值像素比例）+ background depth 方差 / 离群双峰。**这是 v4 当前空白**——E1.1 只测 lateral_3m/6m 横向外推，无 zoom-out / 俯视维度。先立锚、再谈 prune。
- **R10 硬切探针**（本卡核心副产物）：砍 road 区空气区 bg 后，**路面区域是否出洞 / 变暗**（守护渲染对照）。不塌 → 空气区 bg 纯鬼影可砍、喂 E3.1/E3.6；塌 → 量化 road 层覆盖缺口（喂 E3.6 takeover）。**后处理 prune = R10 不重训的廉价探针**，且**绕开 R9**（不碰 road spec/yaml）→ 在 E3.1/E3.2 被 R9 暂缓期间仍能推进 road 方向并产出决策证据。
- **layer ownership 诊断**（Task 0，复现大g 观察）：分层渲染贡献——删 road 层后路面区渲染缺失率（NuRec≈100% 黑洞 vs 我方实测），road-only / bg-only alpha 占比图。**先回答「bg 在 road 上方兜底的是鬼影还是合法高物体」**再定 prune 判据。复用分层基建（[`overlay_renderer.py`](threedgrut_playground/utils/overlay_renderer.py) / `test_layered_gaussians.py` / [`bev_renderer.py`](threedgrut_playground/utils/bev_renderer.py)）。

### P1 — prune 实际执行 + 守护

- **保真守护**（硬线）：训练轨迹 interp PSNR/SSIM 退化 **≤ 0.1 dB**。**由 § 2 双闸门（低贡献 gate 必要条件）结构性保证**，而非仅靠调阈值。
- **zoom-out 锚前后对照**：prune 前后 sky-intrusion rate / depth 方差改善量。
- **目视存档** + **效率**（prune X% 粒子，渲染提速 Y%）。

---

## 4. 成本 / GPU / gate

- **成本**：纯后处理，**0.5–1d**（现成基建复用后估计更可信：optimizer 同步 + FTheta 投影 + 高度场查询均现成）。信号计算部分可 CPU，渲染验证需 GPU。**不重训**。
- **GPU**：inceptio 单卡（验证渲染）。
- **gate**：无硬 gate（与 ①road / ②actor 正交）；**关键卡位 = 不卡 R9**（不碰 road spec/yaml），是 E3.1/E3.2 被 R9 暂缓期间唯一能推进 road 方向的卡；可选复用 E1.x eval 基建做 zoom-out 渲染。
- **风险**：
  - ① **alpha-抵消型 floater 砍了破守护线**（最大风险，§ 2 双闸门的低贡献 gate 必要条件结构性堵住；depth 信号尤其依赖此 gate）。
  - ② **road 上方合法高物体误砍**（跨街杆 / 低垂枝叶 / 车顶落进 occupied cell）→ z>Δ 仅作候选，须过闸门 2 + KNN 孤立；Task 0 ownership 诊断先甄别。
  - ③ depth 信号依赖 LiDAR 覆盖，远场 / 空中失效 → 靠孤立度 + scale + 闸门 2 兜。
  - ④ layered ckpt 逐层 prune、road 层整层跳过（走 LayeredGaussians 分层，非 `exempt_layers`）。

---

## 5. TDD 任务拆解骨架

- **Task 0（诊断先行）**：layer ownership 诊断——分层渲染 road-only/bg-only + 删 road 层缺失率（复现大g「删 road=黑洞」对照，量化我方 bg 替补率），**并回答「bg 在 road 上方兜底的是鬼影还是合法高物体」**（决定 § 2 闸门 1 的 z>Δ 候选边界）。复用 [`overlay_renderer.py`](threedgrut_playground/utils/overlay_renderer.py) / `test_layered_gaussians.py`。**所有 road 工作的基线，半天。**
- **Task 1**：信号函数单测——**闸门 1**（road 空气区 bg ★★ / depth 一致性 / scale aspect / KNN 孤立）+ **闸门 2 低贡献 gate**（跨视角累计 alpha 贡献 `< τ_vis`），合成高斯 fixture；**单测须 pin 住「alpha-抵消型 floater（depth 错但高贡献）被闸门 2 挡下、不进 prune 集」这个 case**（防守护线回归）。
- **Task 2**：prune 脚本读 ckpt + 写回单测（粒子数对、未 prune 高斯参数不变、逐层 mask 正确、**road 层整层不动**）。**复用 `_update_param_with_optimizer` 保 optimizer state 一致**（[`strategy/base.py`](threedgrut/strategy/base.py)）；投影复用 [`scripts/dump_lidar_depth_map.py`](scripts/dump_lidar_depth_map.py)（含 FTheta）。
- **Task 3（P0 锚先行）**：**先 pin zoom-out / 俯视相机轨迹**，sky-intrusion / depth 方差 floater 指标 + R10 出洞探针函数单测。
- **Task 4**：inceptio 对真 ckpt 跑 —— 先出 **P0**（zoom-out 锚 baseline + R10 探针：路面塌不塌 + ownership 诊断）→ 再跑 **P1**（双闸门 prune → 守护线 ≤0.1dB + zoom-out 锚前后对照 + 目视存档）。
- **Task 5**：文档回填；按 R10 探针结果决定 —— 不塌则升级**训练期正则**（喂 E3.1）；塌则量化覆盖缺口（喂 E3.6 takeover）。

---

## 6. 与其它方向的关系

- **定位（deep-research 校准）**：后处理 prune 是「**信号探针 + 廉价验证**」，**非终态解**——floater 的终态解是「不产生」（训练期正则）。本卡验证有效的信号 → 升级为训练期约束：① 空气区信号 → **E3.1** penalty（本卡是其**不卡 R9 的后处理前哨**）；② 几何 / scale 信号 → 对应 deep-research 主线的 **MCMC `λ_o`（opacity L1）+ `λ_Σ`（scale 正则）** 与 **StableGS Dual-Opacity 几何正则**。
- 与 **① E3.3 BEV road** 同属「表示侧、减少伪影暴露」；E3.1 空气区 penalty 是本卡 prune 信号的**训练期版本**（从「事后清」升级到「不产生」）。
- 与 **② E2.5 actor 替换** 正交（一个治全局静态 floater，一个治 actor 弱观测面）。
- **可并行的更轻实验**（deep-research 提示）：MCMC `λ_Σ` / `λ_o` 是更轻的「不产生 floater」训练期 A/B，但**给不了 zoom-out 量化锚与 R10 探针** → 本卡不可替代。
- **推荐先做**：成本最低、不卡 R9、最快出可见效果 + 立 zoom-out 盲区量化锚。
