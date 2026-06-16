# E3.3 — BEV 纹理平面化（road 表示侧根治 spike）

- **编号**：E3.3（v4 plan 已立，backlog 转正；本文为执行 spike 提纲）
- **状态**：⬜ Spike 提纲（待排期）
- **一句话目标**：road 颜色从 per-gaussian SH 改为 **BEV feature grid / 纹理图采样**、真正贴在高度场平面 → **外推天然正确**（参数化级根治 aperture problem），复刻 NuRec road off-track 不退化的能力。

---

## 1. 背景与证据

- **NuRec road off-track 不退化的根因（E0.5 实测）= road 几何冻结五件套**：① ground-mesh init ② lr 1e-6 冻结 ③ MCMC 豁免 ④ z-scale 压扁 + 平整正则 ⑤ DC-only SH。本质是把 road 从「自由高斯」降维成**贴在已知平面上、冻结的、Lambertian 纹理**——aperture problem 的细长高斯过拟合单视角光线方向的自由度被锁死。
- **我方差距（E0.4 双向对照实测）**：interp 近景我方**反超**官方（B3 grad_corr 0.739 vs 0.659，full-res + lane loss 更锐），但 **3m 横移官方 +0.05 corr / +3.9dB band_psnr，6m 两家同崩 ~0.30**。即训练分布内更锐、一出轨迹立刻退化 → E1.5 结论「差距是结构性配方、非调参」。
- **同思路参考**：ExtraGS Road Surface Gaussians / NuRec road 处理。

### 核心洞察

E3.1/E3.2 是「短刀」（空气区 penalty + DC-only freeze，把退化曲线右移一档），**E3.3 是参数化级根治**：road 颜色不再属于「会在新视角炸开的高斯」，而是一张贴在高度场上的 2D 纹理，**任何外推视角都只是对同一张平面纹理的重采样 → 天然正确**。

### ⚠ 前置依赖：road 层 takeover（2026-06-16 大g 观察催生）

`road_region.py` 实锤：我方 road 层 **opacity 仅 ~0.014**，路面渲染主要由 ~750k bg 兜底（bg 替补 24%、空气区 bg 鬼影零约束）。**若 road 层撑不起 road 渲染，BEV 平面纹理做得再好也白做**——退化来自 bg floater 不是 road 层。因此 **E3.3 必须以 E3.6（road/bg 所有权切分 + takeover 补足）为前置**，或把「road 层 takeover」纳入本卡 Task 0。否则 BEV grid 只是给一个几乎隐形的层换了颜色参数化。

---

## 2. 方法（复用 `road_region.py` BEV 基建）

[`road_region.py`](threedgrut/model/road_region.py) 已有 `build_road_height_field`（BEV `grid_z` 高度场，per-cell median Z）+ `query_ground_z`。**扩展 `grid_z → grid_feature`**：

1. **几何锚定**：road 高斯位置 `z = query_ground_z(xy)`，scale 压扁贴地（部分已是 road 冻结配方）。
2. **颜色 BEV 化**：新增 BEV feature grid `[H,W,C]`（learnable）；road 高斯渲染时按其 `xy` 双线性采样 grid feature → tiny MLP / 直接 RGB → **替代 per-gaussian SH DC**。
3. **view-independent**：road 高斯不再各自存 SH → 自动满足 E3.2 DC-only。
4. **渐进**：先小网格 spike（大 `cell_size` + 单 clip 短训）验训练稳定，再全量。

> 渲染路径侵入性是最大工程不确定项：颜色查询走 CUDA kernel 还是 python 层 pre-bake，spike Task 1 先定。

---

## 3. 验收指标

- **外推**（主）：3m/6m **lane grad_corr 提升**（目标逼近官方 +0.05、缩小 6m 同崩），road 区 band_psnr 外推档提升。
- **守护线**（硬）：interp **cc ≥ 24.7 / grad_corr 0.744 不退**（v4 全局守护）。
- 口径同 E1（外推测量门六档 + lane warp 指标）。

---

## 4. 成本 / GPU / gate

- **成本**：**~3d**（含渲染路径改造 + 多次短训迭代）。
- **GPU**：inceptio **depth-off + `num_workers=10`**（内存铁律）；数字不可与 A800 lidar-on 锚跨机比，须先在 inceptio 立 depth-off baseline 锚。
- **gate**：E1 锚 ✅；E3.3 动 road 表示 → 仍受 **R9（PR #24 road spec 单一来源）** 影响，开工前须定 PR #24 去留。
- **风险**：① BEV grid 分辨率 vs 内存/过拟合 ② road↔bg/lane 边界过渡 ③ 渲染路径改造侵入大 ④ DC-only 假设 Lambertian → 湿滑路面 / 镜面反光 view-dependent 丢失。

---

## 5. TDD 任务拆解骨架

- **Task 0**：BEV feature grid 模块（build / 双线性 sample / 梯度）单测，复用 `road_region.py` grid 约定。
- **Task 1**：road 渲染颜色路径接 BEV grid（小 fixture，前向 + 反传梯度通），**先定 kernel vs pre-bake**。
- **Task 2**：小网格短训 spike（inceptio）—— 训练稳定性 + interp 守护线。
- **Task 3**：全量训 + E1 外推指标（3m/6m lane grad_corr / band_psnr）。
- **Task 4**：文档回填（plan §6 Done Log + arch 文件清单/不变量）。

---

## 6. 与其它方向的关系

- 与 **③ E3.5 floater prune** 同属表示侧；E3.3 治 road 这一层，E3.5 治全局静态 floater。
- E3.3 若成功，可能让 E3.1/E3.2 短刀降级为次要（参数化根治 > 正则补丁）。
- 与 **② actor 替换** 正交（road vs actor）。
- **根治性最强但工程量最大**；建议小网格 spike 先验训练稳定再决定是否全量。
