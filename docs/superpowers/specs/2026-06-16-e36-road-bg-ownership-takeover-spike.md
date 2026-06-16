# E3.6 — road/bg 所有权切分 + road 层 takeover（根因 spike）

- **编号**：E3.6（新增，归 E3 表示侧；2026-06-16 大g 观察催生）
- **状态**：⬜ Spike 提纲（待排期）
- **一句话目标**：让 road 层**独占** road 区域渲染（删 road 层 ≈ 黑洞，对齐 NuRec），根治「bg 兜底 road」——这是 ① E3.3 BEV road 与 ③ E3.5 floater prune 共同的根因前置。

---

## 1. 背景与证据（代码级实锤）

- **现状（[`road_region.py`](threedgrut/model/road_region.py) docstring）**：~750k alive **bg 粒子毯覆盖路面、主导其渲染**，专门的 `road` 层 **opacity 仅 ~0.014（几乎隐形）**。
- **已做（V3-R2 `bg_road_penalty`，multilayer.yaml `enabled:true / lambda 0.1 / z_band 0.4m`）**：路面区 bg 粒子 **−86%**，但 road 自盖仍 **68%**、**bg 替补仍 24%**；且 `exempt_layers_opacity_reg:[road]` 救 road 不死（road dead 29.1%→0.2%）。
- **两个未解（plan 自标）**：① 残余耦合**非纯寄生——bg 在补 road 的洞**，硬切出洞（R10）；② penalty 只管贴地 ±0.4m 窄带，**路面上方 0.4m 以上空气区悬浮 bg 零约束 = novel 鬼影主源**。
- **NuRec 对照（大g 2026-06-16 观察）**：删 road 层 = 黑洞 → bg 在 road 区 opacity≈0 = **源头所有权切分**（E0.5 Top-5 ② bg init 剔 road 类点）。我方是「bg 兜底 + 事后窄带驱逐」，本末倒置。

### 核心洞察

road 外推退化的真凶是 **layer takeover 不彻底**：road 层 opacity 0.014 撑不起渲染 → bg 自由高斯兜底 → 这些不受 road 冻结五件套约束的 bg 粒子在 road 区过拟合训练视角 → off-track 炸开。**鸡生蛋困境**：压 bg 就出洞（road 撑不住），救 road 不死但 road 仍弱。NuRec 用 **ground-mesh init（road 天生几何对、覆盖足）+ bg init 剔 road（源头不进）** 破解。

---

## 2. 方法（源头切分 + 覆盖补足，顺序是关键）

| 步 | 做法 | 落点 |
|---|---|---|
| **① bg 源头剔 road** | 初始化时 bg 不在 road 区/road 类点放粒子（对齐 E0.5 ②），替代「先放再 penalty」| init / [`datasetNcore.py`](threedgrut/datasets/datasetNcore.py) sseg road 类 |
| **② road 层 takeover 补足** ★ | road 层 opacity/覆盖加强：ground-mesh init 强化 + road 粒子定向加密（P-CAP 思路）+ 提 road 层目标 opacity，先让 road 撑起 68%→~100% | road 层 init + [`road_reg.py`](threedgrut/model/road_reg.py) |
| **③ 安全撤 bg（不出洞）** | **先补 road 再撤 bg** 的顺序；空气区 bg penalty 扩到全高度（不止 0.4m 窄带，吞并 E3.1）| `bg_road_penalty` z_band → 全高度 + trainer |

> 顺序铁律：先 ② 让 road 接管，再 ③ 撤 bg，否则触发 R10 出洞。E3.5 的 R10 探针正是本卡 ③ 的廉价预演。

---

## 3. 验收指标

- **layer ownership（主）**：删 road 层缺失率 **我方现状 → 逼近 NuRec≈100% 黑洞**；road 区 **bg 替补率 24% → <5%**；road 层 opacity 0.014 → 正常量级。
- **不出洞（R10 硬线）**：路面区无洞/无变暗（守护渲染）。
- **外推**：3m/6m lane grad_corr 改善（去掉 bg 兜底 floater 后）+ 空气区鬼影消除。
- **守护线**：interp **cc ≥ 24.7 / grad_corr 0.744 不退**。

---

## 4. 成本 / GPU / gate

- **成本**：**~2d**（init 改造 + 多次短训验 takeover 不出洞）。
- **GPU**：inceptio depth-off + `num_workers=10`（铁律）；先立 depth-off baseline 锚。
- **gate**：E1 锚 ✅；与 **R9（PR #24 road spec 单一来源）** 强相关——本卡动 road init/spec，**必须先定 PR #24 去留**。
- **风险**：① 顺序错 → R10 出洞 ② bg 剔 road 误删该留的近地 bg ③ road 加密 vs 内存 ④ takeover 后 road 区光照/阴影由谁承载。

---

## 5. TDD 任务拆解骨架

- **Task 0**：复用 E3.5 Task 0 的 layer ownership 诊断作 before 基线（删 road 缺失率 / bg 替补率）。
- **Task 1**：bg init 剔 road 函数单测（road 类点/road 区粒子被排除，其它不动）。
- **Task 2**：road takeover 补足（定向加密 / opacity 目标）+ 全高度空气区 bg penalty 单测。
- **Task 3**：inceptio 短训 A/B —— 「先补 road 再撤 bg」 vs 现状，验 takeover + 不出洞 + ownership 指标。
- **Task 4**：全量训 + E1 外推指标；文档回填。

---

## 6. 与其它方向的关系

- **E3.3 BEV road 的前置**：road 必须先 takeover，BEV 平面纹理才有意义。
- **吞并 / 升级 E3.1**：E3.1 空气区 penalty 是窄带→全高度的中间态；本卡 ③ 直接做全高度 + 顺序保障，E3.1 可并入。
- **与 E3.5 互证**：E3.5 后处理 prune 是本卡 ③（撤 bg）的不重训探针；E3.5 探明「砍空气区 bg 出不出洞」直接喂本卡顺序设计。
- **根因层级最高**：建议 road 轴执行序 = E3.5 诊断/探针 → **E3.6 takeover** → E3.3 BEV 平面化。
