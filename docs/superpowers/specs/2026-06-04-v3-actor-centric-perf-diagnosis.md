# v3 性能路线诊断：从「背景/全局」转向「前景 actor class 质量」

- **日期**：2026-06-04
- **类型**：战略诊断 + 优先级建议（**不含实施计划**）
- **触发**：用户两问 —— ①v3 提升性能还有哪些代办项；②是不是陷入 local minimum、有无其它路径
- **范围边界**：编辑/仿真（删/插 actor 不留痕）**是 v4 目标，不是 v3**；v3 只追重建质量
- **状态**：诊断结论已与用户对齐；是否转 plan 待定

---

## 0. TL;DR

**不是「调参到顶」的局部最优，而是「优化错了轴」的局部最优。** 计划阶梯把绝大多数算力投在背景/全局质量（cc_psnr 已到 26 区间），而用户真正要的是**前景 actor 的 per-class 重建质量：车辆 + 行人/骑行 + 道路（尤其车道线）**，背景模糊可接受。

最高 ROI 的动作（按对齐目标排）：
1. **先把 per-class actor 指标测出来**（现有工具只能测车辆，行人/车道线要新建评测）
2. **sseg 精修动静边界**（修「1/4 辆车其实是 background」的 bleed，直接涨 class PSNR）
3. **track-pose 完整版收尾**（已证 +1.68 class_psnr，但要修 −0.61 cc 退化）
4. **行人从「完全没有」做到「有」**（rigid 垫脚石 → deformable，Stage 16 从 V4 stretch 提到主线）
5. **把粒子预算从背景挪给前景**（bg 1M vs actor 200K，预算重分配）

应当**停止/降级**：Stage 10 sky / Stage 12 MCMC 移植 / 13b Fourier+DINOv2 / Stage 14 大部分 mask / DiFix 当救世主 / 深度监督 —— 全是背景或与目标错配的轴。

---

## 1. 修正后的目标函数

经多轮澄清，v3 的真实成功指标是：

> **最大化前景 actor 的 per-class 重建质量：车辆 + 行人/骑行 + 道路/车道线。背景模糊可接受。**

与 plan 字面写的「novel-view PSNR ≥ 30 全局主 KPI」**几乎正交**。两个错配：

- **目标错配**：plan 优化全局/背景；用户要前景 actor。
- **测量错配**：plan 的 novel-view PSNR **没有 GT、根本测不准**（现只能报 LPIPS，且追的 Δ≈0.004 在噪声地板）；而 per-class 重建质量**有 GT、可测**。

车道线还有一个**测量陷阱**：road 区 PSNR 被大片平整沥青主导，**测不出车道线锐度**。必须单独用 lane/marking mask 上的 PSNR/LPIPS、或道路 BEV-crop 的 LPIPS（感知对线条更敏感）。

---

## 2. 诊断：为什么当前是 local minimum

### 2.1 计划的最大预算项已经塌了
§1.3 的 novel-view 预算 `baseline + 9.5 dB`，但拆开看最大两笔已废：

| 来源 | 计划声称 | 实测 | 状态 |
|---|---:|---:|---|
| Stage 11 深度监督 | **+3.0** | ≈0（dense −0.0045 / sparse −0.004 反向）| ❌ 三次实验证伪 |
| Stage 15 DiFix | +2.0 | repro **+0.30**，novel 未测 | 🟡 低 50-70% 且关键数没测 |

深度监督失败的机理已讲死：它绑死在训练相机上、只约束**几何**不约束**外观**；dense 下 RGB 已 pin 住几何（深度冗余），无相机方向连几何也补不了。novel-view / 前景外观缺的是**辐射信息**，深度给不了。

### 2.2 元教训重复三次
项目史上超参调优最大增益 **+0.04 dB**（T12 SH clamp）；唯二真实大跃迁都来自**重构物理问题**：
- V3-R2 bg-in-road penalty **+0.65 dB**（赶走入侵路面的 75 万 bg 粒子）
- Phase 2A road 豁免 opacity（填路面洞）

而剩余 backlog 的 Stage 12/13b/14 **大部分是 NuRec 超参移植** —— 正是历史回报≈0 的那一类。

### 2.3 KPI 与目标错配（核心）
计划主线（Stage 10 sky / 12 MCMC / 13b Fourier / 14 mask）服务的是**背景/全局**；用户要的是**前景 actor**。这就是 local minimum 的本质 —— 一直在爬错的山。

### 2.4 测量基建只覆盖了一半
- 车辆 class PSNR：[`class_psnr.py`](../../../threedgrut/model/class_psnr.py) 现成，但**基于 cuboid，只能测车辆**（tracks 只有车辆类）。
- 行人 class PSNR：**测不了**（行人无 cuboid/track），要新建 sseg-based per-class 评测（[`ncore_semantic.py`](../../../threedgrut/datasets/ncore_semantic.py) 已有 person=11 / rider=12 / bicycle=18 类表）。
- 车道线：无专门指标。

---

## 3. 按 actor 类的现状与缺口

| 目标类 | 现状建模 | class PSNR | 缺口 | 计划位置 |
|---|---|---:|---|---|
| **车辆**（automobile/truck/bus）| rigid `dynamic_rigids` track | ~17.3 → **18.96**(poseopt) | 中：track-pose 收尾 + 边界 + per-track 外观 | 散在 13a/13b，半成品 |
| **行人/骑行/rider** | **完全没建模**（被 `DEFAULT_VEHICLE_CLASSES` 排除，归 dynamic_deformables=Stage16）| **≈ 地板** | **最大单一缺口** | 当 V4 stretch 扔了 |
| **道路/车道线** | V3-R2 + Phase2A 已让 road 主导路面 | road 区 PSNR 不可信（沥青主导）| 中：高频线条锐度 + 测量口径 | 散在已完成 + Stage 10/13b |
| 背景 | bg 1M 粒子层 | 26 区间 | —（用户：不重要）| **计划的主体** ← 错配 |

车辆 poseopt 实测（V3-L7 Run B, sym5cam 30k）：

| 指标 | baseline | poseopt | Δ |
|---|---:|---:|---:|
| raw psnr_masked | 15.29 | 17.34 | **+2.06 ★** |
| class_psnr（动态车辆区）| 17.28 | 18.96 | **+1.68 ★** |
| cc_psnr_masked | 26.04 | 25.43 | **−0.61 ⚠️**（待完整版 reg 修）|
| novel-view | — | — | 未测 |

> 注：track-pose（`trainer.pose_adjustment.*` opt-in）**已实现并合进 main**（[base_gs.yaml](../../../configs/base_gs.yaml) + [multilayer_poseopt.yaml](../../../configs/apps/ncore_3dgut_mcmc_multilayer_poseopt.yaml)，render.py reload 已修），看板的 T13a.4 ⬜ 是**过时状态**。完整版还差 fix_first/last + temporal smooth + pose prior + novel eval。

---

## 4. 动静分割问题（用户第二/三问）

### 4.1 代码现状（坐实痛点）
- **无任何速度/静止分类**：所有有 cuboid 的 actor 一律进 `dynamic_rigids`，不管动不动 → 停的车/站的人被强行当动态训。
- **bg-in-cuboid penalty 是「杀死」而非「遮挡」**（[`bg_cuboid_loss.py`](../../../threedgrut/model/bg_cuboid_loss.py)）：把 active cuboid 内的 bg 粒子 opacity 压到 `<0.005` 让 MCMC relocate 当死粒子搬走 → 车扫过的走廊里 bg 全清空 → **删车=黑洞**（机制坐实）。
- **分割边界=cuboid AABB**（注释自承高估 10-15%，斜车角落欠覆盖）→ 「1/4 辆车其实是 background」既漏又溢的来源。

### 4.2 想法（按对 v3 重建质量的 ROI 排；v3/v4 分界已标）

| # | 想法 | 服务目标 | v3/v4 |
|---|---|---|---|
| **①** | **sseg 精修边界**：cuboid 定「哪个 track」，sseg 的 car/person mask 定「哪些像素」，求交。动态 loss 只路由到 sseg-actor 像素（边界锐利），AABB 内非 actor 部分（影子/车底地面）还给 bg。| **车辆+行人 class PSNR 直接涨** + 边界干净（同时为 v4 编辑打底）| **v3 主力** |
| **②** | **速度门控**：按 track 实际位移分类（poses 已存，V3-D4 `track_min_speed=1.4 m/s` 现成阈值）。动的留 dynamic_rigids；停的烘焙进静态层。| 静态 actor 质量/效率（同时产出干净动静分解，为 v4 打底）| **v3 轻量** |
| **③** | **遮挡式 bg**（penalty 改「只 mask loss 不杀粒子」+ 深度合成）| 仅道路/车道线在 actor 移开帧的连续性 | v3 可选 / v4 必需 |
| **④** | **生成式补遮挡地面**（inpaint）| 删/插 actor 不留痕 | **v4**（编辑目标）|
| **⑤** | **学习式软分割**（per-gaussian 动/静软归属 photometric 学，cuboid+sseg 当先验）| 最干净分解 | **v4** |

> **v3↔v4 衔接**：①② 是「既涨 v3 重建质量、又天然产出干净动静分解」的双赢项 —— 把它们做扎实，v4 的编辑/仿真（③④⑤）就有了干净的分解地基，不必返工。

---

## 5. 整合优先级（建议）

**Phase 0 — 把目标测出来（前置，便宜，无新训练）**
- 车辆 class PSNR：跑现有 `class_psnr.py`。
- 行人/骑行 class PSNR：**新建 sseg-based per-class 评测**。
- 车道线指标：lane/marking mask PSNR/LPIPS 或 BEV-crop LPIPS。
→ 把含糊的「30dB」换成**每类 actor 的真实数字+缺口**。

**Phase 1 — 高 ROI / 已验证（车辆 + 边界）**
1. **① sseg 精修动静边界** —— ROI/工程比最好的单项。
2. **track-pose 完整版 T13a.4**（补 reg 修 −0.61 cc 退化）—— 已证 +1.68。
3. per-track albedo/scale + per-track 粒子上限（13b L8/L9 / 13a L6）。

**Phase 2 — 最大缺口但工程重（行人）**
4. 行人重建：先 rigid track 垫脚石（从「没有」到「有粗 blob」验证抬升），再上完整 **deformable（Stage 16 提到主线）**。

**Phase 3 — 道路/车道线**
5. road 当 2D 纹理问题：沿车道线定向加密 / 平面 feature grid（非堆 Fourier 时间维）；③ 遮挡式 bg 保 actor 移开帧路面连续。

**停 / 降级（别再烧 A800 30k）**：Stage 10 sky / Stage 12 MCMC 移植 / 13b Fourier+DINOv2 / Stage 14 大部分 mask / DiFix 当救世主 / 深度监督 / ④ inpaint（后者转 v4）。

**战略性容量重分配**：背景不重要，但 bg 是 **1M 粒子**大头、actor 才 200K（70 车）。MCMC per-layer cap / 粒子预算应向 actor 倾斜 —— 砍 bg、补车辆/行人/车道线。又一次「容量竞争重分配」，同 V3-R2 套路（从「位置所有权」换成「预算所有权」），低成本高对齐。

---

## 6. v4 衔接（编辑 / 仿真）

编辑/仿真（删/插 actor 不留痕）= **v4 目标**。v3 阶段：
- 做 ①② 时**优先选能产出干净动静分解的实现**，为 v4 打底。
- ③（遮挡式 bg，不杀粒子）、④（inpaint 遮挡地面）、⑤（学习式软分割）→ **v4 backlog**。
- 把 v3 测出的 per-class 指标 + 干净分解作为 v4 的起点。

---

## 7. 关键事实 / 证据锚点

- 车辆-only tracks：`DEFAULT_VEHICLE_CLASSES = {automobile, heavy_truck, bus}`（[tracks_loader.py:148](../../../threedgrut/datasets/tracks_loader.py)），行人/动物注释明说归 dynamic_deformables（Stage 16，未做）。
- cc_psnr_masked 的 mask **只屏蔽 ego 车身**（`gpu_batch.mask`，[trainer.py:999](../../../threedgrut/trainer.py)），**不屏蔽行人** → 未建模行人确实计入误差。
- class_psnr **基于 cuboid AABB**（[class_psnr.py](../../../threedgrut/model/class_psnr.py)）→ 只测车辆。
- bg-in-cuboid penalty「杀死」机制（[bg_cuboid_loss.py](../../../threedgrut/model/bg_cuboid_loss.py)）→ 删 actor 黑洞根因。
- 深度监督三实验证伪：v3_plan.md § Done Log「Stage 11 后续 — 通道隔离」+「稀疏视角 ablation」。
- V3-R2 +0.65 / Phase 2A：v3_plan.md § Done Log「V3-R1+V3-R2」+「Phase 2A」。
- DiFix repro +0.30：v3_plan.md § Done Log「Stage A.4 Real ckpt Δ-PSNR」。
- track-pose +1.68 class_psnr / −0.61 cc：v3_plan.md § Done Log「V3-L7」。

---

## 8. 待决定的后续

- [ ] 是否把本诊断转成实施 plan（用户此前选「只要诊断，暂不落 plan」）。
- [ ] 修 v3_plan.md 看板过时项（T13a.4 track-pose 实为已做 opt-in；行人/Stage 16 优先级；停止清单）。
- [ ] 深挖候选：① sseg 边界精修具体改法 / 行人 rigid 垫脚石 vs deformable 工程量对比 / Phase 0 per-class 评测扩展设计。
