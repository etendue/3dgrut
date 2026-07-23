# Road ownership bugfix 报告

**日期：** 2026-07-23
**结论：** 第一阶段修复解决了 background 位于 road 前方时的显式侵蚀，
并修复了续训状态与分层学习率的两个真实 bug。B12 随后补齐 duplicate
测量、road-relative surface 和多视角 attribution。5s E arm 可把 duplicate
mean 降低 92.2%，用户在 Viser 中接受其“近中距 road、远距 background”
视觉分工并授权补跑 20s/30k。正式训练将 duplicate mean 降低 85.0%，但
duplicate pixel fraction 仅降低 36.0%，foreground alpha 与完整 KPI 严重
回退。因此只合入通用诊断和默认关闭的实验基础设施，不晋级训练 recipe，
不标记 Full-Fix 完成。

## 1. 问题拆分

Road ownership 不是一个单一问题，而是三个相互独立的环节：

1. **初始化归属：** road/sidewalk 语义点是否同时进入 background 和 road。
2. **训练稳定性：** road 粒子能否存活、扩张补洞，并在续训后保持正确的 scale/density 学习率。
3. **最终成像归属：** 即使 background 粒子中心不在 road 前方，其 Gaussian footprint 仍可能在 road 像素上贡献道路纹理。

前两项改善并不自动保证第三项成立。本轮最初的 guard 只覆盖“background 深度位于 road 深度之前”的像素，因此会漏掉共面、稍后方和中心落在 road mask 外但 footprint 覆盖 road 的重复几何。

## 2. 技术修改与结论

| 修改 | 原理 | 实测/验证结果 | 状态 | 是否合入 `main` |
|---|---|---|---|---|
| checkpoint 分层统计、固定 crop/radial 报告、per-camera telemetry | 把粒子数量、分层 KPI、径向退化和相机采样从主观观察变成可复现数据 | 本地回归通过；默认关闭的 telemetry 不改变历史训练路径 | 成功，通用诊断能力 | 是 |
| layer filter、alpha/road-mask/sky/depth 导出、ownership evaluator | 独立渲染各层并在同一 road mask 内统计贡献与深度关系 | 形成 R0、R2、R6 的统一评测口径，并定位 raw bg alpha 的深度混淆 | 成功，但现有 ownership 指标仍不完整 | 是 |
| B1 semantic-disjoint initialization | background 初始化时剔除已属于 road/sidewalk 的语义 LiDAR 点；无标签点保留并记录统计 | 功能和边界测试通过；避免初始点集直接重叠，但不能阻止后续训练重新形成重复道路 | 实现成功，不是完整修复 | 是，默认关闭 |
| B2 background-road exclusion | 用 road 层构建局部高度场；background 中心落入 road slab/投影时回收 density，只有 footprint 命中时收缩 XY scale | 5s R2 中 raw bg alpha `0.57379→0.25427`（−55.7%），road P10 `0.90327→0.93595`；正式 R2 的前景 bg alpha 相对 R0 `0.00259963→0.00012346`（−95.25%） | 对“前景侵蚀”成功；对 bg-only 重复道路不完整 | 是，默认关闭 |
| B2 footprint 收缩下限 | 重复命中时将 XY scale 收缩到物理下限，不再让 `exp(log_scale)` 下溢为 0 | 避免 MCMC relocation 中出现 `log(0)` 和全 tensor 污染；单测覆盖重复收缩 | 成功的稳定性修复 | 是 |
| depth-aware ownership 与 foreground-only guard | 同时导出 road-only/background-only expected depth，只把 background 在 road 前方的 alpha 视为侵蚀 | 排除了“远处建筑/植被被 raw bg alpha 误判为 road 侵蚀”的假阳性 | 指标纠偏成功，但漏检共面/后方重复道路 | 是，并在报告中明确局限 |
| B3 road responsibility loss | road-only forward，在 road mask 内推动 road alpha/RGB，限制梯度只更新 density/appearance | 5s 实验相对 R2 回退，未证明能解决归属；额外 forward 增加训练复杂度 | 失败/不晋级 | 否 |
| R4：从 step 0 锁低 road scale LR | 试图避免 road Gaussian 变得过小 | road P10 降至 `0.03284`；早期没有足够扩张补洞能力 | 失败 | 否 |
| R5：7k 后只锁 scale LR | 先扩张再固定几何尺寸 | 15k road P10 `0.2797`，低于 R2 的 `0.3083`；road alive `97.7%→95.0%`，density 高 LR 仍持续杀死粒子 | 失败 | 否 |
| R6：7k 后同时锁 scale/density LR | 前 7k 允许补洞，之后以 `1e-4` 同时稳定几何和 opacity | 30k road alive `99.32%`，road P10 `0.51569`，质量守护线通过 | road 稳定性成功，但 ownership 不完整 | 只合入通用 LR/resume bugfix；不合入实验 recipe/driver |
| 恢复 dynamic tracks 后再加载 checkpoint | 续训模型必须先重建 checkpoint 中的动态 track 参数结构，再装载 state dict | 修复含动态层 checkpoint 的恢复顺序错误；回归测试覆盖 | 成功的独立 bugfix | 是 |
| 续训后重应用当前 LayerSpec 的绝对 LR | optimizer state 会从 checkpoint 恢复旧 LR；必须在恢复后用当前配置覆盖 scale/density 等绝对 LR | checkpoint 中验证 road scale/density LR 均为 `1e-4`；测试覆盖当前配置优先级 | 成功的独立 bugfix | 是 |
| background world-Z 过滤工具 | 不改变 tensor 形状，仅把指定 Z 一侧的 background density 置为极低值，用于 render-only 因果 A/B | 成功定位重复道路主要来自负 Z background；绝对 Z 本身不适合作为正式规则 | 成功的诊断工具 | 是，仅诊断用途 |

## 3. R6 结果与为什么仍不能结案

最终 R6 30k checkpoint：

`/home/inceptio/work/output/mcro_b8_ownership/mcro_r6_warm7k_geom1e4_20s_30000/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-2207_171412/ckpt_last.pt`

工程指标：

| 指标 | R6 30k |
|---|---:|
| road alive | 99.32% |
| road alpha P10 | 0.51569 |
| bg-in-front-of-road alpha mean | 0.00015578 |
| sky-on-road | 0 |
| front-wide CC-PSNR | 22.8242 dB |
| road PSNR | 28.3206 dB |
| road LPIPS | 0.24697 |
| frozen guard | 6/6 PASS |

这些结果说明：road 层已经能存活并覆盖道路，background 位于 road 前方的明显遮挡也受到抑制。但是 Viser 的 `background-only` 仍能渲染出完整道路，说明另有一组 background Gaussian 在复制 road 的颜色/纹理。

frame 216 的 Z 过滤因果实验进一步确认：

| A/B | 被隐藏的 background 粒子 | 观察 |
|---|---:|---|
| 隐藏 `z < 0` | 158,297（15.83%） | background-only 中的道路基本消失 |
| 隐藏 `z > 0` | 841,703（84.17%） | background-only 中道路仍完整可见 |

这说明重复道路主要由负 Z 的 background 粒子承担。但不能把 `z < 0` 直接写成生产规则：该序列的 road 本身有坡度，world Z 跨越约 `−2.89m～+0.89m`；绝对 Z 会把正常低洼建筑、植被、路缘或其它路段一起误删。正式规则必须基于局部 road surface 的相对高度。

## 4. B2 为什么没有排除全部 bg-only 道路

B2 的粒子判定以 background Gaussian 的**中心**为主：

- 中心位于 road height-field slab 内，或中心投影落入 road mask，才回收 density；
- 中心不在 mask、只有少量采样 footprint 命中时，只收缩 XY scale；
- 当前 ownership guard 只统计 background depth 严格位于 road depth 前方的像素。

因此下面几类仍会漏过：

1. 与 road 共面、但深度数值略靠后的 background；
2. 中心在 road mask/height-field validity 外，较大 footprint 延伸到 road 的 background；
3. 采样点没有覆盖真实椭圆 footprint 与 road 的交叠区域；
4. 在训练视角不明显、换视角后才覆盖 road 的重复几何；
5. 已经学到道路颜色但 alpha/depth 排序未触发 foreground guard 的 background。

所以“background-road exclusion 成功”应准确表述为：**它成功抑制了所定义候选集中的前景侵蚀，但候选集没有覆盖全部 bg-only 道路成像责任。**

## 5. `main` 集成边界

本次合入：

- 冻结基线、统一诊断/评测/guard 工具；
- B1/B2 的 feature-gated 实现，默认均为 `false`；
- B2 数值稳定性修复；
- dynamic track resume 顺序修复；
- 当前分层绝对 LR 在 resume 后正确生效；
- background Z filter 诊断工具。

本次不合入：

- B3 responsibility loss；
- R4/R5/R6 实验 recipe 与训练 driver；
- 把 world `z < 0` 当成正式过滤规则；
- 声称 Road ownership Full-Fix 已完成的旧计划状态。

所有新训练行为保持默认关闭，因而 `main` 的历史配置不会自动启用 MCRO 实验机制。

## 6. 后续工作

后续任务不再优化“前景 background alpha”，而是直接建立并降低 **background 对 road 像素的重复道路贡献**。实施依据见 [后续计划](superpowers/plans/2026-07-23-road-relative-background-exclusion.md)。

## 7. B12 根因调查结果

### 7.1 新指标确认原 guard 的盲区

旧指标 `bg_in_front_of_road_alpha_mean` 只回答“background 是否挡在 road
前面”，不能回答“background-only 是否在复制 road 纹理”。B12 新增的
`bg_road_duplicate_alpha_mean/p90/pixel_fraction` 在 eroded road domain
直接比较 background-only 与 road/GT 的颜色贡献，不要求 background depth
必须更近。

R6 原始 front-wide held-out：

| 指标 | R6 |
|---|---:|
| duplicate alpha mean | 0.430204 |
| duplicate pixel fraction | 0.927802 |
| valid road pixels | 18,365,105 |

`drop z<0` 可消除约 98.9% duplicate，而 `drop z>0` 只消除约 0.1%。
这解释了 Viser 中的观察，但不改变“禁止用绝对 Z 作为生产规则”的结论：
同一段 road 的 world-Z 实测横跨 `−2.89m～+0.89m`。

### 7.2 中心相对高度不是主要充分条件

局部 road surface 有 73,525 个有效 cells，grid validity 为 99.76%。
然而 background 中心落入窄 relative-Z slab 的粒子很少，extent 扫描也只
减少 8.7%～13.0% duplicate。问题主要来自投影 footprint 和跨视角成像责任，
不是简单的“粒子中心位于 road 平面附近”。

### 7.3 硬删除能去道路，也会删除有效场景

screen-space attribution 找到 1,320 个严格候选，其中 1,318 个 alive。
过滤后 duplicate `0.430204→0.054017`（−87.44%），证明候选确实承担重复
道路成像；但完整六相机评测同时出现：

- front-wide CC-PSNR −2.63 dB；
- 六相机平均 CC-PSNR −0.93 dB；
- road PSNR −3.49 dB。

所以这些不是“纯垃圾粒子”。它们在 front-wide 的 road 像素上是 duplicate，
同时在其它像素或视角承担建筑、植被、路缘等有效纹理。Gaussian 的 density
是跨视角共享参数，硬删会同时取消两种责任。

## 8. B12 技术修改：成功、失败与集成边界

| 技术修改 | 原理 | 结果 | 是否进入 `main` |
|---|---|---|---|
| lossless layer RGB/GT/alpha/depth dump | 避免 8-bit 图像和 depth gate 隐藏弱 duplicate | 新指标稳定复现 R6 视觉问题 | 是 |
| duplicate evaluator 与正确像素分母 | 对所有 eroded-road 有效像素统计，包括零 background 贡献 | R6、Z A/B 和各训练 arm 可统一比较 | 是 |
| confident local road surface | 用局部支持、距离和高度离散度约束有效域 | 证明绝对 Z 不可用，并排除 center-slab 主因 | 是 |
| footprint / multi-view projection attribution | 统计粒子在 road 与 protected pixels 的真实屏幕贡献 | 成功找到因果候选，并揭示跨视角责任冲突 | 是，诊断用途 |
| checkpoint projection filter | 保持 tensor shape，只降低候选 density | duplicate −87.44%，但 KPI 严重回退 | 是，仅 render-only 诊断；不得作为修复 |
| 多视角 projection recycle/decay/shrink | 只 mutation 未被其它视角保护的 road 候选 | C2 质量接近基线，但 duplicate 仅 −58.6% | 保留默认关闭；不晋级 recipe |
| appearance-gated screen transfer | 在 road 像素抑制与道路颜色相似的 bg，并可推动 road 接管 | E duplicate −92.2%，但多相机/foreground 回退 | 不作为默认或正式 recipe |
| isolated-render exposure detach | layer loss 训练 Gaussian 时不误更新共享 BilateralGrid | 修复一个实验路径中的真实梯度泄漏 | 若保留 screen-transfer 基础设施则必须同时保留 |
| 5s A/B driver | 统一 seed、相机、depth-off、workers、评测 | 发现第一版 A/B/C 错把 road LR 从 step 0 锁低 | 只保留修正后的 warm 对照；旧结果不得用于晋级 |

所有新增训练行为均默认关闭。`main` 不应自动启用 projection mutation 或
screen transfer，也不应收录任何把 world `z<0` 当正式规则的配置。

## 9. B12 5s 实验结果

第一版 A/B/C 与 R6 语义不一致：它从 step 0 把 road scale/density LR 锁到
`1e-4`，而 R6 是先 warm 7k 再锁。5s 实验本应让 road 全程正常学习。
修正后的 A2 比错误 A 的 front-wide CC-PSNR 高 0.746 dB、road PSNR 高
0.654 dB，因此最终只以 A2 为基线。

| arm | 机制 | duplicate | 降幅 | mean CC-PSNR | road PSNR | road LPIPS | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| A2 | 正确 warm baseline | 0.291202 | — | 20.3838 | 25.8312 | 0.26350 | 基线 |
| D | bg + road screen transfer | 0.085828 | 70.5% | 18.7218 | 23.9643 | 0.27498 | 全部门失败 |
| E | bg-only screen loss | 0.022859 | 92.2% | 19.2785 | 24.9359 | 0.26855 | duplicate 通过，质量/foreground 失败 |
| C2 | warm + protected projection recycle | 0.120437 | 58.6% | 20.3347 | 25.5669 | 0.26662 | 质量接近，duplicate 失败 |

C2 相对 A2：

| 相机 | CC-PSNR 变化 |
|---|---:|
| front-wide | −0.644 dB |
| cross-left | +0.155 dB |
| cross-right | +0.250 dB |
| rear-left | +0.183 dB |
| rear-right | −0.326 dB |
| back-wide | +0.047 dB |

C2 的六相机平均只下降 0.049 dB，说明 multi-view protection 的方向正确；
但 rear-right 略超 `−0.3 dB` 门，front-wide 明显回退，duplicate 降幅也低于
80%。E 则从反方向证明：只要全局压低这些粒子，duplicate 可以消失，但有效
跨视角内容必然一起丢失。

## 10. 5s 阶段判定

- Task 1–5 完成；
- 本地 B12 相关测试：57 passed；
- C2 GPU smoke/5s：5000 steps，10.43 it/s，无 OOM/nonfinite；
- 没有 arm 同时通过 duplicate、六相机 KPI、foreground 和 road coverage 门；
- 按自动晋级规则不启动 Task 6；后续仅在用户视觉授权后补跑；
- Road ownership 状态仍是“部分修复”，不是 Full-Fix。

后续不能继续只调 density decay/recycle 强度。需要先为冲突粒子建立
per-particle/per-view contribution ledger，并尝试 representation split/clone：
把“front-wide 的 road duplicate”与“其它视角的有效 background”拆给不同
Gaussian。新机制仍须先通过同一套 5s 三门，再允许消耗正式训练资源。

## 11. 用户视觉授权后的 20s/30k 正式复测

用户在 5s E arm 的 Viser 中观察到：

- road-only 在远处截断；
- 远处道路由 background 接管；
- 近处 background-only duplicate 明显受到抑制；
- 该视觉分工主观上可以接受。

因此按用户明确要求，使用同一 E 配方运行六相机 20s/30k：

- background-only screen loss，`lambda=1.0`；
- warmup 1000、every 4 steps；
- road RGB/alpha 辅助项均为 0；
- depth-off、`num_workers=10`；
- checkpoint 7k/15k/30k。

训练完成，commit `bfcb881`，30,000 steps，31 epochs，3903.53 秒，
7.69 it/s，无 OOM/nonfinite。显存最高约 23,240 MiB，最低剩余约
844 MiB。正式目录：

`/home/inceptio/work/output/mcro_b12_task6_screen_20s/mcro_b12_e_bgonly_screen_20s_30000/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-2307_163233/`

### 11.1 正式 KPI

| 指标 | E 20s/30k | R6 30k | 变化 |
|---|---:|---:|---:|
| front-wide CC-PSNR | 19.3627 dB | 22.8242 dB | −3.4615 dB |
| road PSNR | 23.2836 dB | 28.3206 dB | −5.0370 dB |
| road LPIPS | 0.30563 | 0.24697 | +0.05866 |
| road coverage P10 | 0.46956 | 0.51569 | −0.04613 |
| foreground bg alpha mean | 0.04822 | 0.000156 | 显著回退 |
| sky-on-road | 0 | 0 | 持平 |

六相机 mean CC-PSNR 为 15.7369 dB，per-camera CC-PSNR：

- front-wide 19.3627；
- cross-left 14.9172；
- cross-right 13.9033；
- rear-left 16.5176；
- rear-right 14.9987；
- back-wide 14.5822。

### 11.2 正式 ownership

| 指标 | E 20s/30k | R6 | 降幅/变化 |
|---|---:|---:|---:|
| duplicate alpha mean | 0.06467 | 0.43020 | −85.0% |
| duplicate pixel fraction | 0.59380 | 0.92780 | −36.0% |
| raw bg-on-road alpha mean | 0.86009 | — | 仍很高 |
| bg-in-front-of-road alpha mean | 0.04822 | 0.000156 | 约 309× |

这组数字解释了视觉现象：screen loss 主要改变 background 在 road 像素上的
颜色相似性，使 duplicate score 变小；它没有让 background 的 alpha 从 road
像素退出。远处合入 background 并非无成本的责任交接，而伴随大量
background-on-road alpha 和显著的多相机质量损失。

### 11.3 最终结论

正式 20s/30k 证明 E arm 的 duplicate mean 改善可以扩展到长训练，但整体
工程门失败：

- duplicate pixel fraction 未达到 80% 降幅；
- foreground ownership 严重回退；
- front-wide、road 和其它相机 KPI 明显下降；
- 显存余量过低，不适合作为常规正式 recipe。

因此 E arm checkpoint 保留用于视觉与根因研究，不晋级默认配置。最终 Viser
人工观察可以决定这种分层外观是否有研究价值，但不能覆盖上述工程失败。
