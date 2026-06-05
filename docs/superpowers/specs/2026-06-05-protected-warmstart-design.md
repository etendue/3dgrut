# Protected Warm-Start — 设计 spec（P1.4 提质）

> 状态：设计已与用户对齐（2026-06-05 brainstorming），待 review → 写实施 plan。
> 依赖：P1.4 warm-start 注入引擎（PR #18，branch `claude/interesting-mccarthy-03c6be`，**未进 main**）。
> 关联：[asset-harvester 可行性](2026-06-04-asset-harvester-gaussian-injection-feasibility.md)、[v3 actor 诊断](2026-06-04-v3-actor-centric-perf-diagnosis.md)、[v3_plan_revised.md](../../../v3_plan_revised.md) P1.4 / §3。

---

## 1. 背景与核心洞察（why）

P1.4 首次 5k A/B 拿到 **automobile class_psnr +0.730 dB**（B warm-start 22.06 vs A LiDAR 21.33），证明 warm-start 有效。但 viser 目视发现注入车 **spiky/糊、未观测面残缺**，质量没发挥出来。

诊断（代码佐证）确诊两个机制在**侵蚀 asset 最有价值的部分——未观测面几何**（车背/侧面，相机永远看不到，asset-harvester 扩散补全的那部分）：

1. **MCMC `relocate_gaussians`**（[mcmc.py:114](../../../threedgrut/strategy/mcmc.py)）：opacity 跌破阈值的「dead」高斯被搬到高误差的观测面（用 alive 高斯覆盖）。未观测面高斯**永远不被渲染 → 没有 photometric 梯度撑 opacity**。
2. **opacity L1 正则**（[trainer.py:1199](../../../threedgrut/trainer.py)）：持续把所有高斯 opacity 往下压，只有被渲染到的（观测面）有梯度顶回去。

合力 → 未观测面 opacity 单调衰减 → 跌破阈值 → 被 relocate 抹掉。**MCMC 在 warm-start 这里是反作用**（它假设"靠 photometric 长几何"，但 asset 的价值恰恰是 photometric **补不了**的未观测面）。

**首次实验还叠了第二个限流**：默认 `warmstart_max_pts_per_track=5000` 把每车 ~10万 harvested 高斯砍到 5000（丢 95%），注入总数 150077 = 5车×5000 warm + 65 track×LiDAR。MCMC 再从稀疏种子瞎长回 6–9万 → spiky。

---

## 2. 设计原则（对齐结论）

> **保护未观测面（asset 的独有价值），同时让观测面贴合真车 + 场景光照。**

利用「**梯度只流向被渲染的高斯**」这一性质，做精细区分（不需显式标注哪些高斯是观测/未观测）：

| 区域 | 处理 | 机制 |
|---|---|---|
| **观测面**（相机看得到） | **几何 + 颜色都让 Adam 梯度学**（用户已同意）→ 贴合真车像素 + 场景光照 | 有 photometric 梯度 → 自然精修 |
| **未观测面**（车背/侧，asset 补的） | **冻结**（不被 MCMC 搬走、不被 opacity 正则压衰减） | 无梯度 → 只要不让 MCMC/正则动它，就自然不变 |

**关键修正（vs 用户初版"只调 albedo"）**：asset ≠ 场景里那辆真车（生成式补全 + AH-1 近似对齐），观测面几何会有 misalign。若几何全冻死只学 albedo，**观测面 PSNR 被锁死**。所以观测面几何**也要**梯度精修。未观测面靠"没梯度"自然冻住，不需显式冻结。

---

## 3. Protected warm-start 配方（what）

| # | 改动 | 类型 | 作用 | 锚点 |
|---|---|---|---|---|
| C1 | **保留 asset 细节**：`warmstart_max_pts_per_track` 5000 → **~50000**（5车×5万=25万）| 配置 | 不靠 MCMC 长，必须保住 asset 几何 | registry default + CLI |
| C2 | **dynamic_rigids 关 MCMC** densify/relocate（**仅 warm track，推荐**；whole-layer 为简化 fallback）| 小代码 | 不搬走/不增删 asset 高斯，护未观测面 | [layered_mcmc.py](../../../threedgrut/strategy/layered_mcmc.py) sub_strategies per-layer |
| C3 | **关 opacity 正则**：`exempt_layers_opacity_reg: [road, dynamic_rigids]`（机制现成，road 已用）| 配置 | 未观测面 opacity 不被压衰减 | [config:132](../../../configs/apps/ncore_3dgut_mcmc_multilayer.yaml) + [trainer.py:1202](../../../threedgrut/trainer.py) |
| C4 | **保留 Adam 梯度精修** geom（pos/scale/rot）+ 外观（albedo；可选 SH/per-track albedo bias）| 默认/配置 | 观测面贴合真车 + 场景光照 | 默认开 |
| C5 | iters：无 MCMC 重构、纯精修好 init → **可能 5–10k 即够**（30k 作上界对照）| 配置 | 省 GPU | CLI |

**保留**：cuboid 位置 clamp（`dyn_clamp_to_cuboid`，把车钳在 cuboid 内）—— 对 protected 模式无害，防梯度把车推出框。

### 粒子预算
5车×~5万 = 25万 + 65 LiDAR track ≈ 12.5万 → dynamic_rigids ~37.5万。MCMC 关掉后 `max_n_particles=300k` cap 不再由 MCMC 强制（cap 是 strategy 预算）。需确认 init/optimizer 路径支持 >cap 的变长注入（warmstart 引擎已支持变长）；必要时把 `dynamic_rigids.max_n_particles` 调到 400k。

---

## 4. C2 实现方案（关 MCMC，per-track vs whole-layer）

- **推荐 per-track 保护**：在 `MCMCStrategy.relocate_gaussians` 把 warm track 的粒子排除出 `dead_idxs`（`dead_idxs = dead_idxs[~isin(track_ids[dead_idxs], warm_ids)]`），并跳过对它们的 add/perturb。model 需一个 `_warmstart_protected_track_ids` buffer（注入时写入）。**保持 65 个 LiDAR track 的 MCMC 正常**（它们 sparse，需要 MCMC densify）。
- **简化 fallback（whole-layer 关）**：LayeredMCMCStrategy 跳过 dynamic_rigids 的 sub-strategy（一个 `disable_layers` 集合）。更简单，但非 warm 的 LiDAR 车也失去 MCMC → 那些车可能轻退；**A/B 评测须用 per-track class_psnr（只看 5 warm 车）以隔离**。

> 决策：优先 per-track（隔离干净、对 65 LiDAR track 无副作用）；若 per-track 工程量超预期，退 whole-layer + per-track eval。

---

## 5. 验证（success criteria：指标 + 视觉 兼顾）

**A/B（同 clip 从头，A800 双卡并行）**：
- **A 对照**：现配方 LiDAR-only（或首轮 B）。
- **B protected**：C1–C5 配方。
- iters：先 **10k**（protected 收敛快），不足再 30k。

**量化**：automobile class_psnr（B vs A）。若用 whole-layer 关 MCMC → 需 **per-track class_psnr**（只统计 5 warm track）以隔离 warm-start 效应。守护 cc_psnr_masked ≥ 24.7。

**视觉**：viser（原版 3dgut + `--replaced_track_ids 14,16,17,27,67` + dynamic_replaced 开关）目视：
- 未观测面（车背/侧）**保住**（不再残缺/被抹）。
- 观测面**清晰、不 spiky**（梯度精修 + 无 MCMC floater）。

**出口**：B 的 class_psnr ≥ 首轮 B(22.06) 且视觉明显改善 → protected warm-start 坐实，回写 v3_plan_revised.md §3 + Done Log。

---

## 6. 改动文件清单

| 文件 | 改动 | C# |
|---|---|---|
| `configs/apps/ncore_3dgut_mcmc_multilayer.yaml` | `exempt_layers_opacity_reg: [road, dynamic_rigids]`；（可选）`dynamic_rigids.max_n_particles` 400k | C3 / 预算 |
| `threedgrut/layers/registry.py` 或 CLI | `warmstart_max_pts_per_track` 默认/override → 50000 | C1 |
| `threedgrut/strategy/mcmc.py` + `layered_mcmc.py` | per-track MCMC 保护（排除 warm track 出 relocate）；warm-track buffer | C2 |
| `threedgrut/layers/layered_model.py` / `warmstart_inject.py` | 注入时写 `_warmstart_protected_track_ids` buffer + ckpt 持久化 | C2 |
| `threedgrut/model/per_class_eval.py` / `render.py`（若走 whole-layer 简化路） | per-track class_psnr 输出 | 验证 |

**测试纪律**（CLAUDE.md C.10）：per-track MCMC 保护先写单测（warm track 粒子在 relocate 后数量/track_id 不变）pin 住，再上 A800。

---

## 7. 风险

| ID | 风险 | 缓解 |
|---|---|---|
| PR1 | 关 MCMC 后观测面欠致密（细节不够）| asset 提供 ~5万/车，足够；观测面靠梯度精修而非 densify |
| PR2 | 关 opacity 正则 → 未观测面留一堆半透明杂点，novel-view 偶现 floater | cuboid clamp 约束位置；必要时给 dynamic 单独一个弱 opacity floor 而非 L1 decay |
| PR3 | whole-layer 关 MCMC 伤非 warm 车 | 优先 per-track；否则 per-track eval 隔离 |
| PR4 | >cap 变长注入破 optimizer/ckpt | warmstart 引擎已支持变长；加注入 roundtrip 单测 |
| PR5 | asset 观测面 misalign 大，梯度精修拉不回 | AH-1 已验证 containment/朝向；必要时上精确协方差对齐（L4，本 spec 外） |

---

## 8. 不在本 spec（YAGNI / 后续）
- 精确各向异性协方差对齐（L4）—— 仅当 PR5 成为瓶颈再做。
- 重新 harvest（更多视角/分辨率）—— 仅当证明 raw asset 是天花板。
- frozen drop-in（几何全冻）—— v4 编辑目标，非本 spec。
