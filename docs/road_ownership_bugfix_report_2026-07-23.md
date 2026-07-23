# Road ownership bugfix 报告

**日期：** 2026-07-23
**结论：** 本轮修复解决了 background 位于 road 前方时的显式侵蚀，并修复了续训状态与分层学习率的两个真实 bug；但没有消除 `background-only` 中与 road 共面或稍后方的重复道路成像。因此本轮按“部分修复”合入 `main`，不标记 Full-Fix 完成。

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
