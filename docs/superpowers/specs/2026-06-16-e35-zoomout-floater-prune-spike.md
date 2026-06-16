# E3.5 — zoom-out floater prune / clean（后处理几何 prune spike）

- **编号**：E3.5（新增，归 E3 表示侧；2026-06-16 大g 提议）
- **状态**：⬜ Spike 提纲（待排期）
- **一句话目标**：对已训练 ckpt 做**后处理几何 prune**，砍掉 zoom-out / 拉远 / 俯视视角下暴露的漂浮重影（floater），**不重训、不卡 R9**，并立一个 floater 量化锚。

---

## 1. 背景与证据

- **现象（大g 2026-06-16 提出）**：当前 ckpt zoom out 后有大量重影、漂浮 artifacts。
- **codebase 现状（2026-06-16 核实）**：baseline 走 MCMC（[`threedgrut/strategy/layered_mcmc.py`](threedgrut/strategy/layered_mcmc.py)），靠 **relocate 处理「死粒子」（低 opacity）**（`max_relocation_fraction=0.4`，`exempt_layers_opacity_reg=[road]`）。
- **真空地带**：zoom-out floater 大多是**高 opacity、几何深度错**的高斯——在训练 ego 轨迹（贴地前视）被遮挡或 alpha 抵消，拉远/俯视才暴露。**opacity-based relocate 抓不到它们**。当前 pipeline 完全没有针对这类的 prune 通道。

### 核心洞察

floater 本质 = **几何欠约束 + scale-space aliasing**：训练只有地面附近前视轨迹 → 空中/远处 severely under-constrained → 高斯放在错误深度只要 reproject 到训练视角对得上就不被惩罚 → zoom out 即成漂浮重影。它们「投影对、深度错」，所以**只能用几何 / 多视一致性信号抓，不能用 opacity 抓**。

### 核心洞察 B —— road 兜底 bg floater（2026-06-16 大g 观察 + 代码实锤）

[`road_region.py`](threedgrut/model/road_region.py) docstring 实锤：**~750k bg 粒子毯覆盖路面、主导其渲染，专门的 `road` 层 opacity 仅 ~0.014（几乎隐形）**。V3-R2 `bg_road_penalty` 已压路面区 bg −86%，但 road 自盖仍 68% / **bg 替补仍 24%**，且 penalty 只管贴地 ±0.4m 窄带——**路面上方 0.4m 以上空气区的悬浮 bg 完全零约束 = novel 鬼影主源**。对照 NuRec「删 road 层 = 黑洞」（源头所有权切分），我方是「bg 兜底 + 事后窄带驱逐」。这批**空气区 bg 兜底粒子是不受 road 冻结五件套约束的自由高斯，正是 off-track 退化真凶**——给本卡一个最对症的 prune 信号。

---

## 2. 方法（后处理 prune 脚本 `scripts/e35_floater_prune.py`）

读 ckpt → 用 [`model.py`](threedgrut/model/model.py) 的 `get_positions/get_scale/get_density/get_covariance` 取参数 → 算多信号打分 → 加权阈值 → mask → 写回 pruned ckpt/ply（`export_ply`）。**分层 prune，road 层豁免**（已有 `exempt_layers`）。

| 信号 | 判据 | 抓什么 |
|---|---|---|
| **road 区空气区 bg** ★★ | **bg 层**粒子 xy 落在 road height field occupied cell 且 `z > ground_z + 0.4m`（`bg_road_penalty` z_band 之上的盲区），复用 [`road_region.py`](threedgrut/model/road_region.py) `query_ground_z` | road 兜底 bg 鬼影（最对症，攻 layer ownership 根因）|
| **LiDAR/多视 depth 一致性** ★ | 高斯中心投影到训练相机，中心深度 ≪ 该像素 LiDAR depth GT（[`datasetNcore.py`](threedgrut/datasets/datasetNcore.py) / [`aux_readers.py`](threedgrut/datasets/aux_readers.py)）→ 漂在物体前方空中 | 「几何位置错」的高斯 |
| **scale aspect** | `max_scale/min_scale` 过大（细长针状）或绝对 `max_scale` 过大 | 朝相机的细长 splat / 糊脸巨片 |
| **空间孤立度** | KNN 距离，孤立无邻居 | 空中孤立 floater |
| **可见性 / 贡献度** | 跨训练视角累计 alpha 贡献低（Mini-Splatting / PUP-3DGS 思路）| 冗余 / 低贡献雾 |

> depth 一致性依赖 LiDAR 覆盖；远场 / 天空无 LiDAR → 用孤立度 + scale + sky-mask 补。

---

## 3. 验收指标

- **保真守护**（硬线）：训练轨迹 interp PSNR/SSIM 退化 **≤ 0.1 dB**（prune 不能伤训练分布内质量）。
- **R10 硬切探针**（本卡核心副产物）：砍 road 区空气区 bg 后，**路面区域是否出洞/变暗**（守护渲染对照）。不塌 → 空气区 bg 纯鬼影可砍、喂 E3.1/E3.6；塌 → 量化 road 层覆盖缺口（喂 E3.6 takeover）。**后处理 prune = R10 不重训的廉价探针。**
- **layer ownership 诊断**（复现大g 观察）：分层渲染贡献——删 road 层后路面区渲染缺失率（NuRec≈100% 黑洞 vs 我方实测），road-only / bg-only alpha 占比图。复用分层基建（[`overlay_renderer.py`](threedgrut_playground/utils/overlay_renderer.py) / `test_layered_gaussians.py` / [`bev_renderer.py`](threedgrut_playground/utils/bev_renderer.py)）。
- **floater 量化锚（zoom-out proxy）**：固定一组 zoom-out / 俯视 / pull-back 相机渲染，prune 前后比 —— **sky-intrusion rate**（sky-mask 区域内 alpha > 阈值像素比例）+ background depth 方差 / 离群双峰。
- **目视存档** + **效率**（prune X% 粒子，渲染提速 Y%）。

---

## 4. 成本 / GPU / gate

- **成本**：纯后处理，**0.5–1d**。信号计算部分可 CPU，渲染验证需 GPU。**不重训**。
- **GPU**：inceptio 单卡（验证渲染）。
- **gate**：无硬 gate（与 ①road / ②actor 正交）；可选复用 E1.x eval 基建做 zoom-out 渲染。
- **风险**：① 过度 prune 伤训练分布内质量（守护线挡）② depth 信号依赖 LiDAR 覆盖 ③ layered ckpt 分层 prune，road 豁免。

---

## 5. TDD 任务拆解骨架

- **Task 0（诊断先行）**：layer ownership 诊断——分层渲染 road-only/bg-only + 删 road 层缺失率（复现大g「删 road=黑洞」对照，量化我方 bg 替补率），复用 [`overlay_renderer.py`](threedgrut_playground/utils/overlay_renderer.py) / `test_layered_gaussians.py`。**所有 road 工作的基线，半天。**
- **Task 1**：floater 信号函数单测（**road 区空气区 bg** ★★ / depth 一致性 / scale aspect / KNN 孤立），合成高斯 fixture。
- **Task 2**：prune 脚本读 ckpt + 写回单测（粒子数对、未 prune 高斯参数不变、分层 mask 正确、road 层不动）。
- **Task 3**：sky-intrusion / floater 指标 + R10 出洞探针函数单测。
- **Task 4**：inceptio 对真 ckpt 跑 —— 砍 road 区空气区 bg → R10 探针（路面塌不塌）+ zoom-out 渲染前后对照（守护线 + floater 指标 + 目视存档）。
- **Task 5**：文档回填；按 R10 探针结果决定 —— 不塌则升级**训练期正则**（喂 E3.1）；塌则量化覆盖缺口（喂 E3.6 takeover）。

---

## 6. 与其它方向的关系

- 与 **① E3.3 BEV road** 同属「表示侧、减少伪影暴露」；E3.1 空气区 penalty 是本卡 prune 信号的**训练期版本**（后处理验证有效的信号 → 训练期正则化，从「事后清」升级到「不产生」）。
- 与 **② E2.5 actor 替换** 正交（一个治全局静态 floater，一个治 actor 弱观测面）。
- **推荐先做**：成本最低、不卡 R9、最快出可见效果 + 立 zoom-out 盲区量化锚。
