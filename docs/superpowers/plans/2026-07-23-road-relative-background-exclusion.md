# Road-relative background duplicate exclusion 计划

## Summary

目标是消除 `background-only` 中的重复道路成像，同时保留建筑、植被、路缘、车辆、sky 和其它相机质量。本任务是 Road ownership 的 B12 后续，不处理 1cam/6cam 清晰度或训练预算问题。

核心原则：**禁止使用绝对 world Z 作为删除规则。** 候选粒子必须以局部 road surface 为参照：

`h_relative = z_background - z_road(x, y)`

先做 checkpoint 上的 render-only 候选扫描，证明哪些粒子真正承担重复道路成像；只有阈值和保护门有证据后，才接入默认关闭的训练机制。

## 冻结输入与验收口径

冻结参考：

- R0 产物：`/home/inceptio/work/output/mcro_r0_ownership/`
- R6 30k checkpoint：`/home/inceptio/work/output/mcro_b8_ownership/mcro_r6_warm7k_geom1e4_20s_30000/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-2207_171412/ckpt_last.pt`
- 基线帧、crop、相机与评测口径：`docs/mcro_baseline_freeze.md`
- 现有工程门：`configs/eval/mcro_ownership_guards.json`

新增的主验收指标不能再只用 `bg_in_front_of_road_alpha_mean`。至少增加：

- `bg_road_duplicate_alpha_mean/p90`：在 eroded road domain 内，background 对道路重复成像的贡献；共面和稍后方也必须计入；
- `bg_road_duplicate_pixel_fraction`：超过贡献阈值的 road 像素比例；
- `road_coverage_p10`：不得低于 `0.37`，目标保持 R6 的 `0.51569` 附近；
- 现有 foreground bg alpha、sky、full/road KPI guards 全部保留；
- 建筑、植被、路缘保护区域的 background alpha/RGB 差异与固定 crop；
- 六台相机分别报告，不能只看 front-wide 平均值。

晋级门：相对 R6，重复道路贡献至少下降 80%，road coverage 和现有 6 项 guard 全部通过；非 road 保护区域不得出现可见删除，per-camera CC-PSNR 不得下降超过 0.3 dB。最终还需在 Viser 逐层检查 front/side/rear。

## Task 1：补齐“重复道路”测量

目的：先修复 evaluator 的盲区，避免再次出现 guard 全过但视觉问题仍在。

实现：

- 扩展 layer render dump，保留 background/road 的 alpha、expected depth、RGB contribution 和有效深度域；
- 对 road mask 做 configurable erosion，排除路缘与语义边界不确定区域；
- 定义不依赖“background 必须在 road 前方”的 duplicate 指标；
- 同时保留 foreground 指标，区分“遮挡 road”和“复制 road”两个问题；
- 在 frame 216 和冻结 held-out 帧上验证：原 R6 必须被新指标判为失败，`drop z<0` 诊断 arm 必须显示显著改善，`drop z>0` 不得误判为改善。

测试：共面、前方、稍后方、无有效 depth、road mask 边界和空像素分别覆盖；聚合必须按有效像素计数，不能按帧均值二次平均。

出口：新指标能复现 Viser 结论，否则停止，不进入过滤机制。

## Task 2：构建有置信度的局部 road surface

目的：用道路坡面而非 `z=0` 区分道路附近的 background。

实现：

- 复用 `road_region.build_road_height_field`，将固定 1m occupied-cell 查询升级为可配置的局部插值/邻域支持；
- 每个查询返回 `z_road`、validity、邻域样本数和局部离散度；
- validity 只允许在 road 点的支持域内，禁止无约束外插到建筑/植被区域；
- 对稀疏洞允许小范围插值，但设置最大 XY 距离和最大局部 Z 离散度；
- 报告 road surface 的 world-Z 范围、坡度、空洞率和各相机可见覆盖率。

测试：平路、斜坡、非连续高程、边界外查询、稀疏洞、空 road 层和不同 cell size。结果必须对粒子顺序稳定。

出口：冻结 road 区域具有足够 validity，且不会把 road 域扩张到已标注的非 road 保护区域。

## Task 3：render-only 候选扫描

目的：不训练、不永久修改原 checkpoint，找出能去掉重复道路且不破坏场景的最小候选集。

候选必须同时考虑：

1. background XY 位于高置信 road surface 支持域；
2. `h_relative` 位于待扫描的非对称窄 slab；
3. Gaussian 的真实/保守 footprint 与 eroded road mask 有贡献，而非只检查中心像素；
4. 在多个冻结视角上实际产生 background alpha/RGB contribution；
5. 对建筑、植被、路缘保护 mask 没有显著贡献。

首轮仅作为搜索范围扫描相对高度，例如 `[-0.25, +0.15]m`、`[-0.15, +0.10]m`、`[-0.10, +0.08]m`；最终阈值由贡献曲线和保护门选择，不在代码中假定固定值。

每个 arm 保持 checkpoint tensor shape 不变，只将候选 density 置低并输出：粒子数/比例、相对高度分布、每相机重复道路指标、完整 KPI、保护 crop 和四联图。必须额外比较：

- center-only 与 footprint-aware；
- 单帧候选与多帧一致候选；
- road-layer height field 与原始 road LiDAR surface；
- frame 216、冻结 held-out 帧、lateral 3m/6m。

出口：至少一个 render-only arm 达到新增重复指标门和全部保护门。没有 arm 达标时，不把过滤写入训练；转向 footprint/contribution attribution 继续诊断。

## Task 4：训练期 B12 机制

目的：让训练过程中产生的重复 background 也持续退出 road 成像责任，而不是只修已有 checkpoint。

接口默认关闭：

```yaml
layers:
  bg_road_duplicate_exclusion:
    enabled: false
    every_k_steps: 10
    surface_cell_size: 0.5
    min_surface_support: 3
    max_surface_xy_distance: 1.0
    relative_height_min: null
    relative_height_max: null
    road_mask_erosion_px: 8
    min_contribution: 0.0
    action: recycle  # recycle | density_decay | footprint_shrink
```

行为要求：

- 只处理 Task 3 证据支持的 road-relative + contribution 候选；
- 优先比较 density decay/recycle，footprint shrink 只用于中心在保护域外但 footprint 侵入的情况；
- post-MCMC 再检查，防止 relocation 把 background 移回 road；
- 统计候选、受保护、recycle、decay、shrink 数量及相对高度分布；
- nonfinite 时该步不执行 ownership mutation；
- resume 后重建 surface/cache，不能把旧坐标缓存直接复用；
- `enabled=false` 必须与当前 `main` 行为回归等价。

不得使用：全局 raw background alpha 惩罚、绝对 `z<0` 删除、无 validity 的 road surface 外插、仅依据中心投影的硬删除。

## Task 5：测试与 5s 机制 A/B

本地测试覆盖：

- default-off parity；
- 共面、稍前、稍后粒子均由 relative slab 正确判定；
- 中心在 road 外但椭圆 footprint 覆盖 road；
- 保护区域优先于删除候选；
- 多相机贡献聚合、mask erosion、无效 depth/surface；
- MCMC 后再次进入 road 的粒子；
- resume/cache 重建；
- repeated shrink 数值下限与 nonfinite 原子跳过。

5s 实验保持同一 seed、camera set、depth-off、`num_workers=10`、训练窗和 optimizer steps：

- A：当前 R6 稳定性 recipe，不启用 B12；
- B：A + 最小证据 slab + density decay；
- C：A + 最小证据 slab + recycle；
- D：仅当 B/C 遗留 footprint intrusion 时，增加 footprint-aware action。

每臂先统一 render/eval，再比较新增 duplicate 指标、原 6 项 guard、每相机 KPI 和保护 crop。只晋级满足全部门且动作最小的 arm。

## Task 6：20s 正式训练与视觉验收

使用晋级 arm 运行 20s/30k，保存 7k/15k/30k checkpoint。每个阶段检查：

- road alive、scale/density 分布和当前实际 optimizer LR；
- duplicate、foreground、road coverage、sky；
- 六相机 full/road/non-road KPI；
- lateral 3m/6m 与固定文字、车辆、路缘、建筑、植被 crop；
- background-only、road-only、background+road 和 full 四联图。

训练完成后启动 Viser，逐层检查 front/side/rear。只有新增 duplicate 门、现有工程门和人工视觉三者同时通过，才标记 Road ownership Full-Fix 完成并考虑将 B12 配方晋级；否则保留 feature flag，不改变默认配置。

## 决策树

- 新指标不能识别 R6 的视觉问题：先修 evaluator，不训练。
- relative-Z arm 去掉道路但伤建筑/植被：收紧 surface validity/保护 mask，不用绝对 Z 补救。
- center-only 无效、footprint-aware 有效：问题是 Gaussian extent，不扩大 slab。
- 5s 有效、20s 后复发：检查 post-MCMC enforcement 和多视角候选一致性。
- ownership 达标但 road coverage 下降：先稳定 road density/scale，不把责任还给 background。
- 全部门通过但 Viser 仍见重复道路：任务仍未完成，增加失败帧到冻结集合后继续判因。
