# 3DGRUT v3 — NuRec 缺口闭合 + 新视角扰动鲁棒性 · 可执行计划

> **配套文档**：`v2_plan.md`（v2 已结题，本计划承接 § 14 全部 V3-* / V4 任务种子）·`v2_architecture.md`（v2 模块/流程图）·`v3_architecture.md`（待 Stage 9 T9.0 任务创建）
> **评估输入**：`/Users/etendue/repo/3dgrut2/according-to-oss-sim-roadmap-md-how-zazzy-harbor.md`（V3-1..V3-4 pillar 视角）
> **本文档作用**：把 v2_plan.md § 14 全部 V3-* / V4 任务种子落到具体 Stage + 任务卡，承担 v3 整体看板与状态跟踪。

---

## 0. 目标与 KPI

### 0.1 v3 核心方向（用户补充澄清，2026-05-22）

**v2 Stage 8 viser_gui_4d 交互式可视化实证发现**：
- ✅ **Reconstructed view**（训练视角附近）：质量不错，cc_psnr_masked 24.70 dB，与 v2 设计预期一致
- ❌ **Novel view degradation 严重**：一旦视角偏离（位置 ±1m+ / 角度 ±5°+），质量大幅下降——天空爆黑洞、动态对象漂移、反射 view-dependent 失效、几何远景模糊

**v3 主目标重新定位**：把 **novel view generation 质量提升到 ~30 dB**，作为 v3 主 KPI。
Reconstructed cc_psnr_masked 退为辅指标——只要不显著退化（≥ v2 24.70）即可，**不再追求训练视角 30/34 dB**。

### 0.2 KPI 双档（主 KPI 切换为 novel-view PSNR）

| 档位 | 触发 Stage | **★ Novel-view PSNR ★** | Novel Sky PSNR | Novel Dynamic PSNR | Reconstructed cc_psnr_masked (辅) | LPIPS 改善 |
|---|---|---:|---:|---:|---:|---:|
| v2 baseline | Stage 7 (旧, 非对称 5cam) | **~ 待 Stage 8.5 实测**（估测 18-22 dB） | 严重黑洞 | 严重漂移 | 24.70 | baseline |
| **v3 baseline (T8.5.7)** | Stage 8.5 (对称 5cam 30k) | 待 T8.5.3/4 测 | 待测 | 待测 | **26.04** ★ (E2b, +1.34 vs Stage 7) | baseline |
| **v3 保守门槛（必达）** | Stage 15 出口 | **≥ 28.0** | ≥ 28 | ≥ 25 | ≥ v3 baseline (26.04, 不退化) | ≥ 25% |
| **v3 进取目标** | Stage 17 出口 | **≥ 30.0** ★ 用户目标 | ≥ 30 | ≥ 28 | ≥ 26.5 | ≥ 35% |
| NuRec 理论极限（留 v4） | — | ~32-34（含 DiFix 专有数据） | — | — | 36.28 reconstructed | — |

**双档说明**：
- **保守门槛 Stage 15**（Cosmos-DiFix 渐进蒸馏后）必达 novel-view ≥ 28，否则 v3 不结题。
- **进取目标 Stage 17 完成 ≥ 30** — 用户明确目标，v3 应努力达成（虽然标 stretch 但是用户主线诉求）
- Reconstructed cc_psnr_masked 不再设激进目标——只监控不退化（≥ 24.70）。v3 训练增加监督可能短期拉低 reconstructed（trade-off for novel-view），可接受 24.0 容忍下限。
- NuRec 36.28 是 **reconstructed view** 数字；其 novel-view PSNR 未公开，估测 ~32-34，需 DiFix 专有数据，留 v4。

### 0.3 v3 不做（明确排除，转 v4 backlog）

- NuRec 专有 DiFix 训练数据集复现（v3 用开源 Cosmos-DiFix NGC 公开 checkpoint，novel-view 上限差 +2-4 dB）
- 跨 clip 大规模联训（v3 仍单 clip 训练）
- USDZ 打包（V1-6 独立工作包，与 v3 解耦）
- Marching Cubes mesh 导出（V1-5 独立工作包）
- **追求训练视角 NuRec 36.28 reconstructed**——v3 关心的是 novel-view，不优化训练视角到极致

### 0.4 v3 baseline（v2 出口实测数 + Stage 8.5 必须补测 novel-view）

| 维度 | v2 Stage 7 实测 | v3 Stage 9 起点 |
|---|---:|---:|
| `mean_psnr` (full, reconstructed) | 23.78（exposure OFF） | 用 exposure OFF baseline |
| `mean_psnr_masked` (reconstructed) | 25.76（exposure OFF, Stage 7 非对称 5cam） / **15.29**（exposure ON, T8.5.7 对称 5cam 30k —— exposure 退化已知问题, V3-P1 修复） | 用对称 5cam baseline |
| `mean_cc_psnr_masked` (reconstructed) | 24.70（Stage 7 非对称 5cam, σ < 0.2 dB） / **26.04** ★（T8.5.7 对称 5cam 30k） | v3 辅 KPI baseline 更新为 26.04 |
| **Novel-view PSNR (±2m / ±5° hold-out)** | **❓ Stage 8.5 必测 ★** | **v3 主 KPI baseline — 待实测确认** |
| Sky region PSNR (reconstructed) | Stage 5 出口 ≥ 30，30k 训练后衰减 | Stage 10 重新达标 |
| **Novel Sky region** | **❓ 视觉验证已知严重黑洞** | Stage 8.5 量化 |
| **Novel Dynamic region** | **❓ 视觉验证已知严重漂移** | Stage 8.5 量化 |
| 4 层粒子规模 | bg 1M + road 200K + dyn 200K (70 tracks) + sky MLP | 维持 |
| 训练速度 | 9.80 it/s @ A800 | 维持（v3 增加监督会略降，目标 ≥ 7 it/s） |
| ckpt baseline 路径 | `a800-x2:/root/work/yusun/ncore-nurec/output/stage7_noexp_20260521-102930/.../ckpt_30000.pt`（991 MB） | v3 Stage 9 从此 ckpt 续训 |

**Stage 8.5 强制前置任务**：必须实测 v2 baseline 的 **novel-view PSNR（±1m / ±2m / ±5° / ±10° pose 扰动 4 档）** 作为 v3 起点，否则 v3 进展无法量化。

---

## 1. 项目看板（Kanban）

> 状态：⬜ Todo · 🟡 In Progress · 🔵 Review · ✅ Done · ⏸ Blocked · ⏭ Skip

### 1.1 顶层看板（Mermaid Kanban）

```mermaid
%%{init: {'theme':'base'}}%%
kanban
    Backlog
        [T8.5.1 R3/R4 投影参数校对 + 校对值入档]
        [T8.5.2 D6/D7 cuboid padding 设计草案]
        [T8.5.3 ★ Novel-view pose 生成器 + hold-out 验证集]
        [T8.5.4 ★ v2 baseline novel-view PSNR 实测]
        [T8.5.5 A800 5k smoke 验证]
        [T9.0 v3_architecture.md 创建]
        [T9.1 V3-P1.a 双边网格 1×1×1 grid]
        [T9.2 V3-P1.b ExposureModel L2 reg]
        [T9.3 V3-P1.c 训练 forward 与 eval 同套 bilateral grid]
        [T9.4 V3-P1.d 健康度监控]
        [T9.5 A800 5k smoke + 30k 出口]
        [T10.1 V3-L10 sky envmap inpaint]
        [T10.2 V3-L11 sRGB↔linear gamma]
        [T10.3 V3-L12 sky_envmap warm-up]
        [T10.4 A800 Stage 10 出口]
        [T11.1 V3-T8 trainer 每步 ray batch]
        [T11.2 V3-T9 LiDAR ray loss head]
        [T11.3 V3-R2 lidar_divergence]
        [T11.4 V3-D1 DepthAnythingV2 prior]
        [T11.5 V3-E1 val_lidar=true]
        [T11.6 A800 Stage 11 出口]
        [T12.1 V3-T2 opacity_threshold]
        [T12.2 V3-T3 binom_n_max + noise_lr]
        [T12.3 V3-T4 add/relocate 双阶上限]
        [T12.4 V3-T5 StepFunCosineAnnealingLR]
        [T12.5 V3-T6 SequentialLR]
        [T12.6 V3-T7 per-layer LR]
        [T12.7 V3-T1.basic PERTURB hook]
        [T12.8 A800 Stage 12 出口]
        [T13a.1 V3-L4 ignore_classes]
        [T13a.2 V3-L5 symmetric_axis]
        [T13a.3 V3-L6 per-track cap]
        [T13a.4 V3-L7 track-pose 联合优化]
        [T13a.5 V3-D6 cuboid LiDAR padding]
        [T13a.6 V3-D7 cuboid camera padding]
        [T13a.7 A800 Stage 13a 出口]
        [T13b.1 V3-L1 fourier_features_dim=5]
        [T13b.2 V3-L2 road fourier_features_dim]
        [T13b.3 V3-L3 scale_pos_lr_by_scene_extent]
        [T13b.4 V3-L8 optimize_track_albedo]
        [T13b.5 V3-L9 optimize_track_scale]
        [T13b.6 V3-D2 MoG extra_signal]
        [T13b.7 A800 Stage 13b 出口]
        [T14.1 V3-D3 sseg CE loss]
        [T14.2 V3-D4 场景流 mask]
        [T14.3 V3-D5 交通灯 mask]
        [T14.4 V3-D8 相机 mask]
        [T14.5 V3-D9 帧 mask]
        [T14.6 V3-P2 valid_pixel_mask]
        [T14.7 A800 Stage 14 出口]
        [T15.1 V3-T1.full PERTURB cuboid]
        [T15.2 V3-Cosmos.a DiFix checkpoint]
        [T15.3 V3-Cosmos.b novel-view pose 生成器]
        [T15.4 V3-Cosmos.c 渐进蒸馏]
        [T15.5 V3-Cosmos.d color_transfer]
        [T15.6 V3-E3 hold-out 验证集]
        [T15.7 A800 Stage 15 出口 ★]
        [T16.1 V4-Deform.a hash-grid]
        [T16.2 V4-Deform.b FullyFusedMLP]
        [T16.3 V4-Deform.c canonical xyz]
        [T16.4 V4-Deform.d deformnet 渐进]
        [T16.5 A800 Stage 16 出口]
        [T17.1 V3-R1 3DGRUT 复合 renderer]
        [T17.2 V3-E2 per-class cPSNR]
        [T17.3 A800 Stage 17 出口 ★]
        [T18.1 WP_V3_Report.md]
        [T18.2 v3 双档判定]
        [T18.3 v3_plan/architecture 同步]

    In Progress

    Review

    Blocked

    Done
        [T8.5.7 ★ V3-E4 5-cam vs 7-cam → 切对称 5-cam 30k 实测]
```

> Flowchart 在 VSCode/Markdown 中可正常渲染；若不支持则退化为文本列表。

降级表（同源数据）：

| 列 | 任务数 | 关键项 |
|---|---:|---|
| Backlog ⬜ | **57** | Stage 8.5 (5) + 9 (6) + 10 (4) + 11 (6) + 12 (8) + 13a (7) + 13b (7) + 14 (7) + 15 (7) + 16 (5) + 17 (3) + 18 (3) — 含 T9.0 架构图任务 |
| Done ✅ | **1** | T8.5.7 V3-E4 (5-cam vs 7-cam KPI 对照 + 对称 5-cam 切换) |
| In Progress 🟡 | 0 | — |
| Review 🔵 | 0 | — |
| Blocked ⏸ | 0 | — |
| Done ✅ | 0 | v3 启动中 |

### 1.2 任务级看板（按 T*.* 编号）

> 进度状态：⬜ Todo · 🟡 In Progress · 🔵 Review · ✅ Done · ⏸ Blocked · ⏭ Skip
> "改动 / 新增" 列在任务完成后填实际 commit 短 hash 与改动文件

| ID | Stage | Subtask | V3-* 锚 | 估时(d) | 状态 | 改动 / 新增 |
|---|---|---|---|---:|:---:|---|
| **T8.5.1** | 8.5 | R3/R4 投影参数 (`min_projected_ray_radius` ≈ √(1/3), `image_margin_factor=0.1`) 校对 + 校对值入档 | V3-R3/R4 | 0.5 | ⬜ | — |
| **T8.5.2** | 8.5 | cuboid padding 设计草案 + LayerSpec 字段预留（不实现，给 T13a.5/6 用） | V3-D6/D7 设计 | 0.5 | ⬜ | — |
| **T8.5.3** ★ | 8.5 | Novel-view pose 生成器 + hold-out 验证集（±1m / ±2m / ±5° / ±10° 4 档 pose 扰动）+ render.py 接入 | NEW `threedgrut/utils/novel_view.py` | 1.5 | ⬜ | — |
| **T8.5.4** ★ | 8.5 | v2 baseline novel-view PSNR 实测（4 档 × {full / sky / dyn / bg 区域}）+ metrics.json novel_* 字段定义 | A800 + `render.py` eval | 1 | ⬜ | — |
| **T8.5.5** | 8.5 | A800 5k smoke 验证投影校对无回归（reconstructed cc_psnr_masked ≥ 24.7 不退化 + novel-view baseline 入档） | — | 1 | ⬜ | — |
| **T8.5.7** ★ | 8.5 | V3-E4 7-cam vs 5-cam KPI 对照实验 + per-camera PSNR breakdown 工具 + 对称 5-cam 切换 | V3-E4 (新) | 2 | ✅ | dd6c39f + 0ffd738 + 6e14059 |
| **T9.0** | 9 | v3_architecture.md 创建（v2_architecture.md 1:1 镜像 + v3 新增模块占位） | docs | 1 | ⬜ | — |
| **T9.1** | 9 | V3-P1.a 双边网格 1×1×1 grid（按 camera_id）port Recon-Studio | V3-P1 | 1.5 | ⬜ | — |
| **T9.2** | 9 | V3-P1.b ExposureModel L2 reg + lr cosine decay + 2-stage freeze (step > 2000) | V3-P1 | 1 | ⬜ | — |
| **T9.3** | 9 | V3-P1.c 训练 forward 与 eval `color_correct_affine` 用同一套 bilateral grid（消除 raw vs cc 分歧） | V3-P1 | 0.5 | ⬜ | — |
| **T9.4** | 9 | V3-P1.d 健康度监控 — 训练日志 `exposure_a.std` + raw/cc PSNR ratio，> 2 dB 警报 | V3-P1 | 0.5 | ⬜ | — |
| **T9.5** | 9 | A800 5k smoke + 30k 出口 — cc_psnr_masked ≥ 26.5 dB + raw vs cc 差 ≤ 2 dB | V3-P1 出口 | 1.5 | ⬜ | — |
| **T10.1** | 10 | V3-L10 sky envmap inpaint hole-filling — threshold 0.05 + kernel 10 | V3-L10 | 1 | ⬜ | — |
| **T10.2** | 10 | V3-L11 sRGB↔linear gamma 合成 — composite_in_linear_space=false | V3-L11 | 0.5 | ⬜ | — |
| **T10.3** | 10 | V3-L12 sky_envmap 前 1k 步冻结 warm-up — min_grad_updates=1000 | V3-L12 | 0.5 | ⬜ | — |
| **T10.4** | 10 | A800 Stage 10 出口 — Sky region PSNR ≥ 30 dB + 新视角无黑洞（视觉验证） | — | 1.5 | ⬜ | — |
| **T11.1/2** | 11 | **改 image-space**: LiDAR depth + bg_lidar loss head（`depth_prior.py`，复用 tracer pred_dist，非 ray-space） | V3-T8/9 | 1.5 | ✅ | `ae36867`+`3b091d8`+`eb4433f` |
| **T11.3** | 11 | V3-R2 lidar_divergence cone 抗锯齿 — **defer**（出口不依赖，tracer Slang 改动大） | V3-R2 | 1 | ⏭️ defer | — |
| **T11.4** | 11 | V3-D1 DepthAnythingV2 metric depth prior — reader + dataset 接入 + depth loss head | V3-D1 | 2 | ✅ | `f6e4e52`+`f12c304` |
| **T11.5** | 11 | V3-E1 `mean_lidar_psnr` — render.py+trainer 双路 + 三层 eval-path 修复 | V3-E1 | 0.5 | ✅ | `c368b0c`+`f33dc89`+`9ac0d51` |
| **T11.6** | 11 | A800 30k 出口 — **工程✓ KPI✗**: cc_psnr_masked 25.98 不退化, novel-view LPIPS Δ-0.0015 无提升, lidar_psnr 20.68 | — | 2 | 🟡 见 Done Log | `yaml opt-in` |
| **T12.1** | 12 | V3-T2 opacity_threshold=0.005 与 NuRec 校对，记录当前 mcmc.py 实际值 | V3-T2 | 0.5 | ⬜ | — |
| **T12.2** | 12 | V3-T3 binom_n_max=51 / noise_lr=5000 校对 | V3-T3 | 0.5 | ⬜ | — |
| **T12.3** | 12 | V3-T4 add/relocate 双阶上限 — layered_mcmc.yaml 加 add_cap_ratio=0.9 / overall=2M | V3-T4 | 1 | ⬜ | — |
| **T12.4** | 12 | V3-T5 StepFunCosineAnnealingLR 新 scheduler — 供轨迹标定 / albedo / 形变网络 | V3-T5 | 1 | ⬜ | — |
| **T12.5** | 12 | V3-T6 SequentialLR Constant→Linear→Cosine | V3-T6 | 1 | ⬜ | — |
| **T12.6** | 12 | V3-T7 per-layer LR 校对 — position 组 vs 特征组 + γ=0.9998465 | V3-T7 | 0.5 | ⬜ | — |
| **T12.7** | 12 | V3-T1.basic PERTURB hook 简化版（不含 cuboid clip，留 T15.1） | V3-T1 部分 | 0.5 | ⬜ | — |
| **T12.8** | 12 | A800 Stage 12 出口 — cc_psnr_masked ≥ 28.5 dB + MCMC 收敛监控曲线 | — | 1 | ⬜ | — |
| **T13a.1** | 13a | V3-L4 background ignore_classes_from_layers=[road] — layered loss 加层级排他 mask | V3-L4 | 0.5 | ⬜ | — |
| **T13a.2** | 13a | V3-L5 DynamicRigid symmetric_axis='Y' — 镜像粒子对称先验 + 镜像约束 reg | V3-L5 | 1.5 | ⬜ | — |
| **T13a.3** | 13a | V3-L6 DynamicRigid 5000 pts/track + 全层 300K cap | V3-L6 | 0.5 | ⬜ | — |
| **T13a.4** | 13a | V3-L7 track-pose 联合优化 — fix_first/last + warm start ≥ 500 + 可学习 Δpose | V3-L7 | 3 | ⬜ | — |
| **T13a.5** | 13a | V3-D6 cuboid LiDAR padding [0.5, 0.5, 0.25] m — T4.4 dynamic_mask 加膨胀 | V3-D6 | 0.5 | ⬜ | — |
| **T13a.6** | 13a | V3-D7 cuboid camera padding [1.0, 1.0, 0.25] m | V3-D7 | 0.5 | ⬜ | — |
| **T13a.7** | 13a | A800 Stage 13a 出口 — cc_psnr_masked ≥ 29.2 + dynamic region PSNR ≥ 26 | — | 1.5 | ⬜ | — |
| **T13b.1** | 13b | V3-L1 background fourier_features_dim=5 时间编码 | V3-L1 | 1 | ⬜ | — |
| **T13b.2** | 13b | V3-L2 road fourier_features_dim=1 | V3-L2 | 0.5 | ⬜ | — |
| **T13b.3** | 13b | V3-L3 LayerSpec scale_pos_lr_by_scene_extent 字段 + trainer 接 | V3-L3 | 0.5 | ⬜ | — |
| **T13b.4** | 13b | V3-L8 optimize_track_albedo — per-track SH bias + Constant→Linear→Cosine LR | V3-L8 | 1.5 | ⬜ | — |
| **T13b.5** | 13b | V3-L9 optimize_track_scale — per-track scale offset 同 L8 | V3-L9 | 1 | ⬜ | — |
| **T13b.6** | 13b | V3-D2 MoG extra_signal 20 维通道 + dataset DINOv2 feat reader + 背景层接入 | V3-D2 | 2.5 | ⬜ | — |
| **T13b.7** | 13b | A800 Stage 13b 出口 — cc_psnr_masked ≥ 29.7 dB | — | 1 | ⬜ | — |
| **T14.1** | 14 | V3-D3 sseg 直读 logits + sky/road/dyn aux CE loss head (21 类 softmax) | V3-D3 | 1.5 | ⬜ | — |
| **T14.2** | 14 | V3-D4 场景流 mask — track_min_speed=1.4 m/s + dilate 20 px | V3-D4 | 1 | ⬜ | — |
| **T14.3** | 14 | V3-D5 交通灯 / 闪烁光源 mask — 21 px dilation（与 D4 同管线） | V3-D5 | 0.5 | ⬜ | — |
| **T14.4** | 14 | V3-D8 相机 mask 30 iter dilation — mask 合并管线统一 | V3-D8 | 0.5 | ⬜ | — |
| **T14.5** | 14 | V3-D9 帧 mask 10 iter dilation | V3-D9 | 0.5 | ⬜ | — |
| **T14.6** | 14 | V3-P2 valid_pixel_mask 多源汇入 + dilation 合并 | V3-P2 | 1 | ⬜ | — |
| **T14.7** | 14 | A800 Stage 14 出口 — cc_psnr_masked ≥ 30.0 + LPIPS 改善 ≥ 15% | — | 1.5 | ⬜ | — |
| **T15.1** | 15 | V3-T1.full PERTURB cuboid clip — move_outside_of_cuboid=false（粒子+noise 后投回 cuboid） | V3-T1 完整 | 1 | ⬜ | — |
| **T15.2** | 15 | V3-Cosmos.a Cosmos-DiFix NGC 公开 checkpoint 下载 + 本地缓存策略 + 加载封装 | V3-Cosmos | 2 | ⬜ | — |
| **T15.3** | 15 | V3-Cosmos.b 复用 T8.5.3 novel-view pose 生成器（已 Stage 8.5 实现） | V3-Cosmos | 0.25 | ⬜ | — |
| **T15.4** | 15 | V3-Cosmos.c 渐进蒸馏调度 — start_epoch=16 / full_novel_view_by_epoch=22 / 50% 训练视角 + 50% 新视角 | V3-Cosmos | 2.5 | ⬜ | — |
| **T15.5** | 15 | V3-Cosmos.d use_color_transfer=true — DiFix 输出色彩传输到 GT 域 | V3-Cosmos | 1 | ⬜ | — |
| **T15.6** | 15 | V3-E3 hold-out 新视角验证集（与 Cosmos pose 生成器复用） + metrics 新字段 | V3-E3 | 1 | ⬜ | — |
| **T15.7** | 15 | A800 Stage 15 出口（**保守门槛**） — cc_psnr_masked ≥ 30.0 dB + hold-out novel-view PSNR ≥ 27 | — ★ | 2 | ⬜ | — |
| **T16.1** | 16 (stretch) | V4-Deform.a permuto hash-grid encoding 16 层 | V4 | 4 | ⬜ | — |
| **T16.2** | 16 (stretch) | V4-Deform.b FullyFusedMLP 64×1 形变网络 | V4 | 3 | ⬜ | — |
| **T16.3** | 16 (stretch) | V4-Deform.c canonical xyz + smoothness_frame_steps=5 | V4 | 2 | ⬜ | — |
| **T16.4** | 16 (stretch) | V4-Deform.d deformnet_start_iteration=1000 + 渐进 10→16 hash level | V4 | 3 | ⬜ | — |
| **T16.5** | 16 (stretch) | A800 Stage 16 出口 — 行人 region PSNR ≥ 28 + cc_psnr_masked ≥ 32.5 | — | 3 | ⬜ | — |
| **T17.1** ★ | 17 ★ | V3-R1 v2_full.yaml 切到 3DGRUT 复合 renderer + 配置 secondary ray（反射 view-dependent 改善 novel view） | V3-R1 | 1.5 | ⬜ | — |
| **T17.2** ★ | 17 ★ | V3-E2 evaluator per-class PSNR / SSIM / LPIPS × 4 档 novel pose 拆解（sky / road / dyn / bg） | V3-E2 | 1.5 | ⬜ | — |
| **T17.3** ★ | 17 ★ | A800 Stage 17 出口 — **novel-view PSNR (4 档平均) ≥ 30.0 ★ 用户进取主目标 ★** + per-class report 完整 | — | 1 | ⬜ | — |
| **T18.1** | 18 | WP_V3_Report.md 编写（镜像 WP_V2_Report.md 结构） | docs | 1.5 | ⬜ | — |
| **T18.2** | 18 | v3 双档判定（保守 ≥ 30 / 进取 ≥ 34）+ V4 backlog 转出 | docs | 0.5 | ⬜ | — |
| **T18.3** | 18 | v3_plan.md / v3_architecture.md 最终同步 + git commit | docs | 0.5 | ⬜ | — |

### 1.3 当前 Stage 状态汇总（**主 KPI: novel-view PSNR ↑**；reconstructed 为辅 KPI）

| Stage | 主题 | 任务数 (Done / Total) | **Novel-view PSNR 出口 ★** | Reconstructed (辅, 不退化) | 状态 |
|---:|---|---:|:---:|:---:|:---:|
| 8.5 | 投影校对 + cuboid 草案 + **novel-view baseline 实测** | 0/5 | **baseline 入档**（≈ 待实测） | ≥ 24.7 (持平) | ⬜ Todo |
| 9 | V3-P1 双边网格 + ExposureModel 修复 | 0/6 | baseline + 0.5 (反作用) | ≥ 24.7（raw/cc 差 ≤ 2 dB） | ⬜ Todo |
| 10 | Sky envmap inpaint + gamma + warm-up | 0/4 | **+1.5（Sky novel 不爆黑洞最大头）** | ≥ 24.7 | ⬜ Todo |
| 11 | LiDAR ray + DepthAnythingV2 几何先验 | 0/6 | **+3.0（几何稳定性核心）** | ≥ 24.7 | ⬜ Todo |
| 12 | MCMC + scheduler 增强 | 0/8 | +0.3（cap baseline 修正） | ≥ 24.7 | ⬜ Todo |
| 13a | Track-pose + symmetric + cuboid padding | 0/7 | **+1.5（dynamic novel 不漂移）** | ≥ 24.7（dyn ≥ 26） | ⬜ Todo |
| 13b | per-track albedo/scale + Fourier + DINOv2 | 0/7 | +0.5（每 track 外观稳定） | ≥ 24.7 | ⬜ Todo |
| 14 | 辅助 mask 管线（LPIPS 主） | 0/7 | +0.2（LPIPS 主） | ≥ 24.7（LPIPS -15%） | ⬜ Todo |
| **15** ★ | **Cosmos-DiFix 渐进蒸馏 + PERTURB cuboid**（保守门槛） | 0/7 | **+2.0 → ≥ 28.0 ★ 保守门槛必达** | ≥ 24.7 | ⬜ Todo |
| 16 (stretch) | DynamicDeformable hash-grid（行人/骑行） | 0/5 | +1.5（行人 novel 改善） | ≥ 24.0（容忍轻退） | ⬜ Todo |
| 17 (★ 用户进取目标 ★) | 3DGRUT secondary ray + per-class cPSNR | 0/3 | **+1.5 → ≥ 30.0 ★ 用户主目标** | ≥ 26.5 | ⬜ Todo |
| 18 | V3 结题报告 + 双档判定 | 0/3 | 保守 ✅ / 进取 ✅ / V4 转出 | — | ⬜ Todo |
| **总计** | — | **0/68** | — | — | — |

> Novel-view PSNR 累计预算：baseline + 0.5 - 0 + 1.5 + 3.0 + 0.3 + 1.5 + 0.5 + 0.2 + 2.0 = baseline + 9.5 dB
> 若 v2 baseline novel-view ≈ 19 dB（Stage 8.5 待实测验证），Stage 15 出口 ≈ 28.5（保守门槛 28 达成）；Stage 17 出口 ≈ 30.0（进取目标达成）。
> Stage 16 行人/骑行通常不计入主 novel-view 平均（受动态权重小），实际累积 +1.5 仅在行人 region 体现。

### 1.4 任务依赖图（关键路径）

```mermaid
flowchart LR
  classDef todo fill:#f5f5f5,stroke:#999,color:#333
  classDef stretch fill:#fff7e6,stroke:#f5a623,color:#333
  classDef milestone fill:#e6f4ff,stroke:#0070f3,color:#000,font-weight:bold

  v2["v2 Stage 7 baseline<br/>recon cc_psnr_masked 24.70<br/>novel-view ❓ Stage 8.5 实测"]:::milestone

  S85["Stage 8.5 ★<br/>R3/R4 + cuboid 草案<br/>+ novel-view pose 生成器<br/>+ v2 baseline novel-view 实测"]:::milestone
  S9["Stage 9<br/>V3-P1 ExposureModel 修复<br/>novel +0.5 (反作用小) / raw≈cc"]:::todo
  S10["Stage 10<br/>Sky inpaint+gamma+warm-up<br/>novel +1.5 / Sky novel ≥28"]:::todo
  S11["Stage 11<br/>LiDAR + DepthV2 深度监督 (image-space)<br/>工程✓ 但 novel-view 无提升; opt-in yaml"]:::done
  S12["Stage 12<br/>MCMC + scheduler 增强<br/>novel +0.3"]:::todo
  S13a["Stage 13a<br/>Track-pose + symmetric + cuboid padding<br/>novel +1.5 / dyn novel 不漂移"]:::todo
  S13b["Stage 13b<br/>per-track albedo/scale + Fourier + DINOv2<br/>novel +0.5"]:::todo
  S14["Stage 14<br/>Aux mask 管线 + valid_pixel_mask<br/>novel +0.2 (LPIPS 主)"]:::todo
  S15["Stage 15 ★保守门槛<br/>Cosmos-DiFix + PERTURB cuboid<br/>novel +2.0 → ≥28.0 必达 ★"]:::milestone

  S16["Stage 16 (stretch)<br/>V4 DynamicDeformable hash-grid<br/>行人 novel ≥28"]:::stretch
  S17["Stage 17 ★ 用户进取主目标 ★<br/>3DGRUT secondary ray + per-class cPSNR<br/>novel +1.5 → ≥30.0 ★"]:::milestone
  S18["Stage 18<br/>WP_V3_Report.md + 双档判定"]:::milestone

  v2 --> S85 --> S9 --> S10 --> S11 --> S12 --> S13a --> S13b --> S14 --> S15
  S15 --> S17 --> S18
  S15 -.可选并行.-> S16 --> S18

  V4["V4 backlog<br/>(NuRec 专有 DiFix 数据 / 跨 clip 联训 / DynamicDeformable 若 stretch 失败)"]:::stretch
  S18 -.转出.-> V4
```

---

## 2. Stage 详细任务卡

> 每个 Stage 任务卡包含：**触发条件 / 任务清单 / 改动文件预期 / 验收准则 / A800 验证脚本预设 / 风险与 fallback**。
> 所有任务卡只描述目标 / 改动文件 / 验收准则，**不放代码**（按 CLAUDE.md 全局约束）。

### 2.0 Stage 8.5 — 渲染管线投影校对 + Novel-view baseline 实测（健康检查 + ★ v3 主 KPI baseline）

**触发条件**：v2 Stage 7 ckpt 已固化（`stage7_noexp_20260521-102930/ckpt_30000.pt`）。

**核心变化（用户反馈驱动 2026-05-22）**：
Stage 8.5 不再仅是健康检查 — 增加 **novel-view pose 生成器 + v2 baseline novel-view PSNR 实测** 两项关键任务。原 Stage 15 V3-Cosmos.b 的 pose 生成器（T15.3）前置到这里，让 Stage 9-15 每一个出口都能用同一套 hold-out 验证集报告 novel-view PSNR。**没有 novel-view baseline 实测数据，整个 v3 无法量化主 KPI 进展。**

**任务清单**：

| Task | 描述 | 预期改动文件 |
|---|---|---|
| T8.5.1 | V3-R3 `min_projected_ray_radius ≈ √(1/3) = 0.5477` + V3-R4 `image_margin_factor=0.1` 校对值入档；与 NuRec parsed_config.yaml 对照 | `threedgut_tracer/`, `configs/render/3dgut.yaml` |
| T8.5.2 | V3-D6/D7 cuboid padding LayerSpec 字段预留（不实现）：`dynamic_rigid_pad_lidar`, `dynamic_rigid_pad_camera` | `threedgrut/layers/layer_spec.py` |
| **T8.5.3 ★** | **Novel-view pose 生成器 + hold-out 验证集**：基于训练 ego trajectory 生成 4 档扰动 pose：±1m / ±2m 平移、±5° / ±10° 旋转；render.py eval 路径接入（按 4 档分别报告 PSNR / SSIM / LPIPS） | NEW `threedgrut/utils/novel_view.py`, MOD `render.py` |
| **T8.5.4 ★** | **v2 baseline novel-view PSNR 实测**：用 T8.5.3 hold-out 集 + v2 Stage 7 ckpt 在 A800 上跑 eval；按 4 档 × {full / sky region / dyn region / bg region} 拆解 PSNR；metrics.json 新增 `novel_psnr_<档>_<region>` 字段命名规范 | A800 + `render.py` |
| T8.5.5 | A800 5k smoke 验证投影校对无回归：reconstructed cc_psnr_masked ≥ 24.7 + novel-view PSNR 4 档基线入档 | A800 |

**验收准则**：
- R3/R4 校对值与 NuRec 一致或解释差异写入 v3_architecture.md
- A800 5k smoke reconstructed cc_psnr_masked ≥ 24.7（v2 baseline 持平，σ < 0.2 dB）
- LayerSpec 新字段不破坏 v2 ckpt 加载（roundtrip 测试通过）
- **★ v2 baseline novel-view PSNR 4 档实测入档**（这是 v3 主 KPI 起点；用户先验估测 18-22 dB，实测确认后写入 § 0.2 KPI 表）
- **★ Novel-view 4 档拆解显示 Sky novel / Dynamic novel 退化严重**（确认用户视觉观察的量化证据）

**A800 验证脚本预设**：
- T8.5.4 用 v2 Stage 7 ckpt + 4 档 hold-out pose（每档至少 20 帧）跑 eval，输出 `metrics_v2_baseline_novel.json`
- T8.5.5 用 v3 Stage 8.5 配置（仅 R3/R4 校对 + LayerSpec 字段加） 续 v2 ckpt 跑 5k smoke

**风险与 fallback**：
- T8.5.3 pose 生成器边界 case：novel pose 落到 cuboid 内 / 远离 ego trajectory → 用 ego trajectory ±N% 范围 sampling 而非全空间扰动
- T8.5.4 novel-view PSNR 全部低于 15 dB（极度退化）→ 视觉验证是否 pose 生成有 bug；若确认 v2 真的这么差，说明 v3 baseline 起点更低，KPI 目标需重新校准
- R3/R4 校对若发现 v2 baseline 存在系统偏差 → v2 Stage 7 24.70 数字需打折扣，KPI 表的 v2 baseline 列需要更新

---

### 2.1 Stage 9 — V3-P1 双边网格 + ExposureModel 退化修复 ★

**触发条件**：Stage 8.5 验收通过。

**背景**：v2 Stage 7 实证 ExposureModel 在 30k 长训中退化优化失控（raw psnr_masked 15.63 vs 关掉后 25.76，+10.13 dB；但 cc_psnr_masked 几乎不变 24.75 ↔ 24.70）。退化路径：高斯学个大概 + exposure 把色彩偏差全 compensate（14 参数 vs 几百万高斯，更快收敛）。

**任务清单**：

| Task | 描述 | 预期改动文件 |
|---|---|---|
| T9.0 | v3_architecture.md 创建（v2_architecture.md 1:1 镜像 + v3 新增模块占位） | `v3_architecture.md` (NEW) |
| T9.1 | V3-P1.a Recon-Studio 双边网格 1×1×1 grid 直接 port 替换 affine `exp(a)·img+b` | `threedgrut/correction/exposure.py` (改造) 或 NEW `bilateral_grid.py` |
| T9.2 | V3-P1.b 加约束防退化：bilateral grid params L2 reg + lr cosine decay + 2-stage freeze (step > 2000 freeze) | `threedgrut/correction/exposure.py`, `trainer.py` |
| T9.3 | V3-P1.c 训练 forward 与 eval `color_correct_affine` 用同一套 bilateral grid（消除 raw vs cc 分歧） | `trainer.py`, `render.py`（eval 路径同步） |
| T9.4 | V3-P1.d 健康度监控：训练日志 `exposure_a.std` + raw/cc PSNR ratio，> 2 dB 警报 | `trainer.py` log hook |
| T9.5 | A800 5k smoke + 30k 出口验证 | A800 |

**验收准则**：
- **★ Novel-view PSNR (4 档平均) ≥ v2 baseline + 0.5 dB**（V3-P1 主要解决 exposure 不直接改善 novel-view 几何，预期反作用小）
- Reconstructed `mean_cc_psnr_masked` ≥ 24.7（不退化于 v2）
- raw `mean_psnr_masked` 与 cc `mean_cc_psnr_masked` 差 ≤ 2 dB（exposure 健康度，**这是 V3-P1 真正主验收点**）
- 30k 长训 `exposure_a.std` 单调收敛（不发散）
- v1/v2 ckpt 加载兼容性测试通过（affine 模式仍可启用）
- Mac CPU pytest 单测覆盖：bilateral grid 0 初始化 ≈ affine identity / 2-stage freeze 触发 / raw/cc 同步

**A800 验证脚本预设**：
- 5k smoke：1-cam 5k step 验证 bilateral grid forward + raw vs cc gap
- 30k 出口：7-cam 30k step + use_exposure=true，对比 T7.3 / T7.3.b baseline

**风险与 fallback**：
- 若 bilateral grid 1×1×1 仍发生退化优化 → 试 2×2×2 grid 增加约束（NuRec 默认 1×1×1，但更复杂可能稳定）；或 freeze step 提前到 1000
- 若 raw vs cc 差仍 > 2 dB → 排查 `color_correct_affine` 与 bilateral grid 双重作用，确保 eval 路径只 apply 一次
- Stretch goal：3 dB 提升（→ 27.7 dB）但保守 1.8 dB 即可触发 Stage 10

---

### 2.2 Stage 10 — Sky envmap 增强（inpaint + gamma + warm-up）

**触发条件**：Stage 9 出口 cc_psnr_masked ≥ 26.5。

**背景**：v2 Stage 5 出口 Sky region PSNR ≥ 30 但在 30k 训练后衰减（30k baseline 中 Sky 区域已无独立监控）。NuRec 用 inpaint + gamma 合成 + warm-up 三件套防止新视角天空黑洞。

**任务清单**：

| Task | 描述 | 预期改动文件 |
|---|---|---|
| T10.1 | V3-L10 inpaint hole-filling — `should_inpaint=true` + threshold 0.05 + kernel 10（关键 — 新视角不爆黑洞） | `threedgrut/correction/sky_envmap.py` |
| T10.2 | V3-L11 `composite_in_linear_space=false` — trainer blend 路径加 sRGB↔linear | `threedgrut/correction/sky_envmap.py`, `trainer.py` (blend) |
| T10.3 | V3-L12 sky_envmap 前 1k 步冻结 warm-up — `min_grad_updates=1000` | `trainer.py` |
| T10.4 | A800 Stage 10 出口验证（含新视角验证） | A800 |

**验收准则**：
- Reconstructed Sky region PSNR ≥ 30 dB（30k 长训后维持，不衰减）
- **★ Novel-view PSNR (4 档平均) ≥ baseline + 1.5 dB ★**（Sky 不爆黑洞是 novel-view 头号头号收益点）
- **★ Novel Sky region PSNR ≥ 28 dB ★**（Stage 8.5 baseline 视觉验证已知严重黑洞，本 Stage 必须量化改善）
- Reconstructed cc_psnr_masked 不回退（≥ 24.7）
- ±1 m / ±2 m 测试 pose Sky 区域无黑洞 / 大块伪影（视觉验证 + LPIPS）
- Mac CPU 单测：inpaint hole-filling 收敛 / linear-space blend 数值正确性

**A800 验证脚本预设**：
- 30k 训练 + Sky region 独立 PSNR 报告
- 5 张 ±1 m holdout 视角渲染 + 视觉检查

**风险与 fallback**：
- 若 inpaint 引入新视角接缝（与 background 交界 LPIPS 升高） → kernel 缩小到 5；threshold 调到 0.02
- nvdiffrast 仍不可用（v2 Stage 5 T5.1 已确认）→ 维持 MLP sky backend，inpaint 在 MLP 路径上等价实现

---

### 2.3 Stage 11 — LiDAR ray 监督 + lidar_divergence + DepthAnythingV2 prior

**触发条件**：Stage 10 出口 Sky region PSNR ≥ 30。

**背景**：NuRec §3 / §5 最重要的两个特色之一（与 DiFix 并列）。LiDAR ray 监督是 NuRec 主训练同等权重的 supervision。DepthAnythingV2 提供稠密 prior，与稀疏 LiDAR 真值锚定互补。Plan agent 修正：D1 与 T8/T9 必须同 Stage 启动。

**任务清单**：

| Task | 描述 | 预期改动文件 |
|---|---|---|
| T11.1 | V3-T8 trainer step 加 LiDAR ray batch — `camera_rays=6144 + lidar_rays=2048` 1:1 比例 | `trainer.py`, `threedgrut/datasets/datasetNcore.py`（LiDAR ray reader） |
| T11.2 | V3-T9 LiDAR depth/intensity ray loss head（NuRec 主训练同等权重） | `trainer.py` (loss), `threedgrut/model/layered_loss.py` |
| T11.3 | V3-R2 lidar_divergence=0.002 rad cone 抗锯齿 — tracer 端 expose | `threedgrt_tracer/` config |
| T11.4 | V3-D1 DepthAnythingV2 metric depth prior — dataset reader + trainer depth loss head | `threedgrut/datasets/datasetNcore.py`, `trainer.py`, NEW `threedgrut/correction/depth_prior.py` |
| T11.5 | V3-E1 val_lidar=true — LiDAR domain PSNR 独立报告 + metrics.json 新字段 | `render.py`, `trainer.py` (eval) |
| T11.6 | A800 Stage 11 出口验证 | A800 |

**验收准则**：
- **★ Novel-view PSNR (4 档平均) ≥ baseline + 3.0 dB ★**（**Stage 11 是 novel-view 几何稳定性核心 Stage** — LiDAR ray 提供稀疏真值锚定 + DepthV2 提供稠密 prior 是 view extrapolation 不糊的最关键约束）
- LiDAR domain PSNR ≥ 25 dB（NuRec 参考值；独立 metrics.json 字段 `mean_lidar_psnr`）
- Reconstructed cc_psnr_masked 不退化（≥ 24.7；可能因引入 LiDAR/depth 多任务训练而轻退至 24.0-24.5，可接受）
- DepthAnythingV2 depth loss 收敛单调（不发散）
- LiDAR ray batch 与 camera ray batch 速度损失 < 30%（A800 it/s ≥ 6.8）
- Mac CPU 单测：LiDAR ray reader / depth loss head / val_lidar metrics 字段

**A800 验证脚本预设**：
- 5k smoke：验证 LiDAR + depth loss 收敛
- 30k 出口：7-cam 30k step + LiDAR ray + DepthV2 prior

**风险与 fallback**：
- LiDAR ray 在 Sky 区域 NaN/inf（Stage 10 已修，但残余风险）→ valid mask 排除 sky；或 LiDAR depth > 100 m clip
- DepthAnythingV2 与 LiDAR 度量不一致（DepthV2 是 scale-shift invariant 相对深度，NuRec 用 metric 版本）→ Stage 11 用 NuRec metric 微调版；fallback 用 scale 对齐 head
- A800 it/s 跌破 5 → LiDAR ray batch 降到 1024；或每 2 step 一次 LiDAR loss

---

### 2.4 Stage 12 — MCMC + 训练策略增强（固定 baseline 密度）

**触发条件**：Stage 11 出口 cc_psnr_masked ≥ 28.0。

**背景**：Plan agent 修正：MCMC densification 影响整个高斯分布，应在 Stage 13a/b 多层 per-track 参数前固定 baseline 分布，否则 L1..L9 基于错误密度学习。V3-T1 PERTURB cuboid 完整版留 Stage 15（与 V3-E3 强耦合），Stage 12 只做简化 hook。

**任务清单**：

| Task | 描述 | 预期改动文件 |
|---|---|---|
| T12.1 | V3-T2 `opacity_threshold=0.005` 与 NuRec 校对，记录当前 mcmc.py 实际值 | `threedgrut/strategy/mcmc.py` |
| T12.2 | V3-T3 `binom_n_max=51` / `noise_lr=5000` 校对 | `threedgrut/strategy/mcmc.py` |
| T12.3 | V3-T4 add/relocate 双阶上限 — `add_cap_ratio=0.9` / overall=2M | `configs/strategy/layered_mcmc.yaml` |
| T12.4 | V3-T5 StepFunCosineAnnealingLR 新 scheduler — 供轨迹标定 / albedo / 形变网络 | NEW `threedgrut/utils/schedulers.py` |
| T12.5 | V3-T6 SequentialLR Constant→Linear→Cosine | 同上 |
| T12.6 | V3-T7 per-layer LR 校对 — position 组 vs 特征组 + γ=0.9998465 | `trainer.py` optimizer init |
| T12.7 | V3-T1.basic PERTURB hook 简化版（不含 cuboid clip，留 T15.1） | `threedgrut/strategy/mcmc.py` |
| T12.8 | A800 Stage 12 出口验证 | A800 |

**验收准则**：
- **Novel-view PSNR (4 档平均) ≥ baseline + 3.3 dB**（+0.3 vs Stage 11；MCMC 改进对 novel-view 间接收益）
- Reconstructed cc_psnr_masked ≥ 24.7（不退化）
- MCMC 收敛监控曲线无 collapse（每层 Gaussian 数 ≤ 配置 cap）
- 新 scheduler 在 5k smoke 上 LR 单调下降到 ≈ 配置末端值
- Mac CPU 单测：scheduler 端点正确性 / per-layer LR group 隔离

**A800 验证脚本预设**：
- 5k smoke：验证 LR scheduler + MCMC cap 触发
- 30k 出口：7-cam 30k step

**风险与 fallback**：
- 若 per-layer LR γ 不收敛 → 用 v2 fused_adam 默认 γ；γ=0.9998465 是 NuRec 经验值
- MCMC cap_max 增大引入显存爆 → 单层 cap 维持 v2 200K，仅引入 add_cap_ratio 比例参数

---

### 2.5 Stage 13a — 多层 + Track-pose 联合优化 + cuboid padding

**触发条件**：Stage 12 出口 cc_psnr_masked ≥ 28.5。

**背景**：v2 明确"不做 track pose 学习"（用 GT），v3 Stage 13a 是 v2.x 留的下一步。symmetric_axis 'Y' 给车辆左右镜像粒子，per-track cap 5000 是 NuRec 经验值。cuboid padding D6/D7 是 L4 / L7 的前置（agent 修正）。

**任务清单**：

| Task | 描述 | 预期改动文件 |
|---|---|---|
| T13a.1 | V3-L4 background `ignore_classes_from_layers=[road]` — layered loss 加层级排他 mask | `threedgrut/model/layered_loss.py` |
| T13a.2 | V3-L5 DynamicRigid `symmetric_axis='Y'` — 对称粒子初始化 + 镜像约束 reg | `threedgrut/layers/dynamic_rigid_init.py`, `layered_loss.py` |
| T13a.3 | V3-L6 DynamicRigid 5000 pts/track + 全层 300K cap | `threedgrut/layers/dynamic_rigid_init.py`, `configs/strategy/layered_mcmc.yaml` |
| T13a.4 | V3-L7 track-pose 联合优化 — fix_first/last + warm start ≥ 500 + 可学习 Δpose（与 Sequential warm-up 配合） | `threedgrut/layers/layered_model.py`, `trainer.py` |
| T13a.5 | V3-D6 cuboid LiDAR padding `[0.5, 0.5, 0.25]` m — T4.4 dynamic_mask 加膨胀 | `threedgrut/datasets/datasetNcore.py` (mask) |
| T13a.6 | V3-D7 cuboid camera padding `[1.0, 1.0, 0.25]` m | 同上 |
| T13a.7 | A800 Stage 13a 出口验证 | A800 |

**验收准则**：
- **★ Novel-view PSNR (4 档平均) ≥ baseline + 4.8 dB ★**（+1.5 vs Stage 12；dynamic novel 不漂移是 v3 第二关键收益点）
- **★ Novel Dynamic region PSNR ≥ 25 dB ★**（Stage 8.5 baseline 视觉验证已知 dyn novel 严重漂移）
- Reconstructed Dynamic region PSNR ≥ 26 dB
- Reconstructed cc_psnr_masked ≥ 24.7
- track-pose 学到的 Δpose 端点固定（`‖Δpose[0]‖ ≤ 1e-4` 和 `‖Δpose[-1]‖ ≤ 1e-4`）
- 镜像约束 reg loss 单调下降（不发散）
- Mac CPU 单测：fix_first/last 端点 / per-track cap / cuboid padding

**A800 验证脚本预设**：
- 5k smoke：验证 track-pose warm start 不爆梯度
- 30k 出口：7-cam 30k step + region PSNR 拆解（基础版，per-class 工具留 Stage 17）

**风险与 fallback**：
- track-pose 学到 collapse（Δpose → 0 或远离）→ 增加 fix_first/last 权重 10×；warm start 提到 1000 步
- 对称约束在非对称车辆（如 SUV vs 卡车）回退 PSNR → symmetric_axis 设为 per-track 可选

---

### 2.6 Stage 13b — per-track albedo/scale + Fourier time + extra_signal 背景层

**触发条件**：Stage 13a 出口 cc_psnr_masked ≥ 29.2 + dyn ≥ 26。

**任务清单**：

| Task | 描述 | 预期改动文件 |
|---|---|---|
| T13b.1 | V3-L1 background `fourier_features_dim=5` 时间编码 | `threedgrut/layers/layer_spec.py`, `model/model.py`, NEW Fourier encoding |
| T13b.2 | V3-L2 road `fourier_features_dim=1`（轻量） | 同上 |
| T13b.3 | V3-L3 LayerSpec `scale_pos_lr_by_scene_extent` 字段 + trainer 接 | `threedgrut/layers/layer_spec.py`, `trainer.py` |
| T13b.4 | V3-L8 `optimize_track_albedo` — per-track SH bias + Constant→Linear→Cosine LR（用 T12.5 scheduler） | `threedgrut/layers/layered_model.py` |
| T13b.5 | V3-L9 `optimize_track_scale` — per-track scale offset | 同上 |
| T13b.6 | V3-D2 MoG `extra_signal` 20 维通道 + dataset DINOv2 feat reader + 背景层接入 | `threedgrut/model/model.py`, `datasets/datasetNcore.py` |
| T13b.7 | A800 Stage 13b 出口验证 | A800 |

**验收准则**：
- **Novel-view PSNR (4 档平均) ≥ baseline + 5.3 dB**（+0.5 vs Stage 13a）
- Reconstructed cc_psnr_masked ≥ 24.7
- Fourier time encoding 在不同时间帧产生可观察的色彩/亮度变化（visualisation）
- per-track albedo 学到的 SH bias 与 track-pose 学习不冲突（两组参数 covariance 监控）
- DINOv2 feat 加载速度不显著拖慢训练（每 epoch < 10% overhead）

**A800 验证脚本预设**：
- DINOv2 feat 预计算缓存（per-frame .pt）+ dataset reader 测试
- 30k 出口 + Fourier time 视觉验证

**风险与 fallback**：
- DINOv2 feat 存储爆盘（7-cam × 600 帧 × 20 维 × 各分辨率）→ 用 float16 + per-frame .pt 压缩
- Fourier time 在静态场景 over-fitting（学到时间噪声）→ Fourier feat dim 降到 3；或 lambda 缩小

---

### 2.7 Stage 14 — 辅助数据通道 + mask 管线（LPIPS 主导）

**触发条件**：Stage 13b 出口 cc_psnr_masked ≥ 29.7。

**背景**：mask 管线增强主要改善 LPIPS（感知质量），PSNR 增益较小（Plan agent 评估 +0.3~0.7 dB）。但对结题 ≥ 30.0 仍是必要的最后 0.3 dB。

**任务清单**：

| Task | 描述 | 预期改动文件 |
|---|---|---|
| T14.1 | V3-D3 sseg 直读 logits + sky/road/dyn aux CE loss head (21 类 softmax) | `datasets/datasetNcore.py`, `trainer.py` |
| T14.2 | V3-D4 场景流 mask — `track_min_speed=1.4` m/s + dilate 20 px | `datasets/datasetNcore.py` (mask), aux_readers |
| T14.3 | V3-D5 交通灯 / 闪烁光源 mask — 21 px dilation | 同上 |
| T14.4 | V3-D8 相机 mask 30 iter dilation | `datasets/aux_readers.py` |
| T14.5 | V3-D9 帧 mask 10 iter dilation | 同上 |
| T14.6 | V3-P2 valid_pixel_mask 多源汇入 + dilation 合并 | `datasets/datasetNcore.py` |
| T14.7 | A800 Stage 14 出口验证 | A800 |

**验收准则**：
- **Novel-view PSNR (4 档平均) ≥ baseline + 5.5 dB**（+0.2 vs Stage 13b；mask 管线主要改 LPIPS）
- LPIPS 改善 ≥ 15%（vs v2 baseline，**这是 Stage 14 真正主验收点**）
- Reconstructed cc_psnr_masked ≥ 24.7
- valid_pixel_mask 与 dilation 后无 holes / 边缘 artifact（视觉验证）
- Mac CPU 单测：sseg logits CE loss / mask dilation 数值正确性 / valid_pixel_mask 汇总

**A800 验证脚本预设**：
- 30k 出口 + LPIPS 拆解

**风险与 fallback**：
- mask dilation 过度引入 over-smoothing → dilation 减半（D8: 15 iter / D9: 5 iter）
- sseg CE loss 与主 photometric loss 不平衡 → CE lambda 调小到 0.01

---

### 2.8 Stage 15 — Cosmos-DiFix 渐进蒸馏 + 新视角扰动 + PERTURB cuboid 完整版 ★

**触发条件**：Stage 14 出口 cc_psnr_masked ≥ 30.0。**保守门槛 Stage**。

**背景**：NuRec §7 主力（与 LiDAR ray 并列预期最大 PSNR 提升点）。Plan agent 修正：DiFix 在 cc_psnr_masked 净增益 +0.8~1.5 dB（不是 +2，因为 cc 已 mask 出动态区）。V3-T1 完整 PERTURB cuboid clip 与 V3-E3 新视角扰动强耦合，同 Stage。

**任务清单**：

| Task | 描述 | 预期改动文件 |
|---|---|---|
| T15.1 | V3-T1.full PERTURB cuboid clip — `move_outside_of_cuboid=false`（粒子 + noise 后投回 cuboid） | `threedgrut/strategy/mcmc.py`, `strategy/src/` |
| T15.2 | V3-Cosmos.a Cosmos-DiFix NGC 公开 checkpoint 下载 + 本地缓存策略 + 加载封装 | NEW `threedgrut/correction/difix.py`, NEW `scripts/download_difix.sh` |
| T15.3 | V3-Cosmos.b 新视角 pose 生成器 — ±2 m 平移 + 小旋转 | `threedgrut/utils/novel_view.py` (NEW) |
| T15.4 | V3-Cosmos.c 渐进蒸馏调度 — `start_epoch=16` / `full_novel_view_by_epoch=22` / 50% 训练视角 + 50% 新视角 | `trainer.py` |
| T15.5 | V3-Cosmos.d `use_color_transfer=true` — DiFix 输出色彩传输到 GT 域 | `correction/difix.py` |
| T15.6 | V3-E3 hold-out 新视角验证集（与 Cosmos pose 生成器复用）+ metrics 新字段 | `render.py` (eval) |
| T15.7 | A800 Stage 15 出口验证（**保守门槛**） | A800 |

**验收准则（★ v3 保守门槛 ★）**：
- **★ Novel-view PSNR (4 档平均) ≥ 28.0 dB ★ — v3 保守门槛必达**（+2.0 vs Stage 14；DiFix 是直接为 novel view 设计的伪影修复器，预期最大单 Stage 收益）
- **★ Novel Sky region novel-view PSNR ≥ 28** ★ + **Novel Dynamic novel-view PSNR ≥ 25** ★
- Reconstructed cc_psnr_masked ≥ 24.7（不退化）
- DiFix 蒸馏前后 v3 训练 loss 不 collapse（progressive epoch 16 启动后 loss 平稳）
- PERTURB cuboid 后粒子位置 100% 在 cuboid 内（投回约束验证）
- Mac CPU 单测：novel-view pose 生成正确性 / DiFix forward shape / PERTURB cuboid clip

**A800 验证脚本预设**：
- DiFix checkpoint 下载 + smoke test（forward shape + 输出范围）
- 30k 出口（DiFix 在 step 16k epoch 启动）+ hold-out novel-view 5 张视觉验证

**风险与 fallback**：
- DiFix NGC checkpoint 不可访问（无 NGC 账号 / sandbox）→ 用 Cosmos-DiFix-public HuggingFace 版本（如有）；若都不可用，转 V4 backlog，Stage 15 出口降为 cc_psnr_masked ≥ 29.5 + 配合 stretch Stage 16 补
- DiFix 输出引入 over-saturation / hallucination → use_color_transfer=true 强制色彩对齐；novel-view 比例从 50% 降到 30%
- 30 dB 门槛仍未达 → 重新评估前序 Stage 误差累积；可能需要 Stage 9/11/13a re-run
- 若 Stage 15 出口 cc_psnr_masked < 30.0 dB → v3 **不结题**，回滚到 Stage 14 出口（cc_psnr_masked 30.0）作为保守结题，DiFix 转 V4

---

### 2.9 Stage 16 (stretch) — DynamicDeformable hash-grid（V4 主力进 v3）

**触发条件**：Stage 15 出口 cc_psnr_masked ≥ 30.0 已结题。Stage 16 是 stretch，不阻塞结题。

**背景**：NuRec V4 主力，行人/骑行/形变 actor 重建关键模块。complexity 极高（hash-grid + FullyFusedMLP + permuto encoding + canonical xyz + smoothness_frame_steps）。失败转 V4 不影响 v3 保守结题。

**任务清单**：

| Task | 描述 | 预期改动文件 |
|---|---|---|
| T16.1 | V4-Deform.a permuto hash-grid encoding 16 层（依赖 `thirdparty/tiny-cuda-nn/`） | NEW `threedgrut/layers/dynamic_deformable.py` |
| T16.2 | V4-Deform.b FullyFusedMLP 64×1 形变网络 | 同上 |
| T16.3 | V4-Deform.c canonical xyz + `smoothness_frame_steps=5` | 同上 |
| T16.4 | V4-Deform.d `deformnet_start_iteration=1000` + 渐进 10→16 hash level + `optimize_canonical_xyz` | 同上 |
| T16.5 | A800 Stage 16 出口验证 | A800 |

**验收准则（stretch）**：
- 行人 / 骑行 region **novel-view PSNR ≥ 28 dB**（per-class，依赖 Stage 17 评估工具或临时手动拆解；这是 Stage 16 主验收点）
- 行人 / 骑行 reconstructed region PSNR ≥ 30 dB
- v3 主 KPI Novel-view PSNR (4 档平均) ≥ baseline + 7.5（即 Stage 15 出口 + 1.5），但若失败不阻塞结题
- 形变网络梯度回流到 hash-grid + MLP（数值一致性测试）
- v2 / v3 Stage 15 ckpt 加载兼容（DynamicDeformable layer 可选 enable）

**A800 验证脚本预设**：
- 5k smoke：形变网络梯度健康检查
- 30k 出口（deformnet 在 step 1000 启动）+ 行人 region 视觉验证

**风险与 fallback**：
- hash-grid + FullyFusedMLP 显存爆 → 降到 8 层 hash + MLP 32×1
- permuto encoding 编译失败 → 用普通 multi-resolution hash（drivestudio 备份）
- 形变网络与 v2 LayeredGaussians fused_view collapse → DynamicDeformable layer 在 fused_view 加 `if spec.enabled_deform_render` 守卫
- **Stretch 失败转 V4**：Stage 16 不达标不阻塞 Stage 18 结题

---

### 2.10 Stage 17 — 3DGRUT 复合 renderer + per-class cPSNR ★ 用户进取主目标 ★

**触发条件**：Stage 15 出口已结题（保守门槛 novel-view ≥ 28 达成）。

**用户进取目标定位（2026-05-22 澄清）**：
本 Stage 是 v3 **进取目标 novel-view PSNR ≥ 30 的最后冲刺 Stage**，**不再是 stretch**。3DGRUT secondary ray 处理反射 / 折射类 view-dependent 效果——这是除 DiFix 之外另一个直接改善 novel view 的核心机制（v1 已有 OptiX 实现但 v2 未启用）。

**任务清单**：

| Task | 描述 | 预期改动文件 |
|---|---|---|
| T17.1 | V3-R1 v2_full.yaml 切到 3DGRUT 复合 renderer + 配置 secondary ray（反射 view-dependent 改善 novel view 关键） | `configs/apps/ncore_3dgut_mcmc_v3_full.yaml`, `threedgrut/render/` |
| T17.2 | V3-E2 evaluator per-class PSNR / SSIM / LPIPS 拆解（sky / road / dyn / bg）+ 每档 novel-view 分别报告 | `render.py` (eval), `trainer.py` (compute_metrics) |
| T17.3 | A800 Stage 17 出口验证 | A800 |

**验收准则（★ v3 进取目标 ★）**：
- **★ Novel-view PSNR (4 档平均) ≥ 30.0 dB ★ — 用户进取主目标**（+2.0 vs Stage 15 出口 28.0；secondary ray 在反射、汽车玻璃、湿路面场景预期 +1.5；per-class 拆解精细化预期 +0.5）
- Novel-view PSNR 在 ±2m / ±10° 极端档单独 ≥ 28 dB（不再仅 4 档平均，极端 pose 也必须达标）
- per-class report 完整（sky / road / dynamic / background 各独立 PSNR / SSIM / LPIPS × 4 档 pose）
- Reconstructed cc_psnr_masked ≥ 26.5（轻提升于 v2 是可以接受的副作用）
- 3DGRUT 复合 renderer secondary ray 不引入数值不稳定（5k smoke loss 平稳）

**A800 验证脚本预设**：
- 5k smoke：验证 3DGRUT 复合 renderer + secondary ray 不发散
- 30k 出口 + per-class × 4 档 novel-view metrics 完整 dump
- 与 Stage 15 ckpt 渲染对比：反射区域（车窗 / 湿路）视觉验证

**风险与 fallback**：
- 3DGRUT secondary ray 在某些 NCore 场景 collapse（OptiX edge case）→ 关闭 secondary ray，只启用复合 renderer 基础部分
- per-class PSNR 拆解 mask 不对齐 → 用 Stage 14 valid_pixel_mask + Stage 13a cuboid padding 后的 dynamic_mask 重新对齐
- 若 ≥ 30 进取目标未达 → 在 Stage 18 报告"进取目标未达 X.X dB"，v3 仍按 Stage 15 保守门槛结题

---

### 2.11 Stage 18 — V3 结题报告 + 双档判定 + V4 转出

**触发条件**：Stage 15 出口已通过（保守门槛达成）。

**任务清单**：

| Task | 描述 | 预期改动文件 |
|---|---|---|
| T18.1 | WP_V3_Report.md 编写（镜像 WP_V2_Report.md 结构）— KPI 三组对比 + 每 Stage 实测 + 关键技术发现 + V4 backlog | NEW `WP_V3_Report.md` |
| T18.2 | v3 双档判定 — 保守 ✅ ≥ 30 / 进取 ✅ ≥ 34（或 stretch 未达说明）/ V4 backlog 转出 | `WP_V3_Report.md`, `v3_plan.md` § 7.8 |
| T18.3 | v3_plan.md / v3_architecture.md 最终同步 + git commit | docs |

**验收准则**：
- WP_V3_Report.md 完整覆盖 Stage 8.5-18（或 8.5-15 保守 + 16/17 stretch 状态说明）
- v3 双档判定明确写入（保守 30.0 实测 / 进取 34.0 实测 or 未达）
- V4 backlog 明确（DynamicDeformable 若未达 / NuRec 36.28 极限对齐 / 跨 clip 联训 / 专有 DiFix 数据集）
- 所有 v3_plan.md 任务状态从 ⬜ 转为 ✅ 或 ⏭

---

## 3. 关键架构改动（前置说明）

### 3.1 v3 新增模块（与 v2_architecture.md 模式一致）

| 模块 | Stage | 类型 | 说明 |
|---|---|---|---|
| `threedgrut/correction/exposure.py` 改造 | 9 | 替换 | affine `exp(a)·img+b` → bilateral grid 1×1×1 |
| `threedgrut/correction/sky_envmap.py` 扩展 | 10 | 扩展 | + inpaint hole-filling + sRGB↔linear blend + warm-up freeze |
| `threedgrut/correction/depth_prior.py` | 11 | NEW | DepthAnythingV2 metric depth prior loss head |
| `threedgrut/utils/schedulers.py` | 12 | NEW | StepFunCosineAnnealingLR + SequentialLR Constant→Linear→Cosine |
| `threedgrut/layers/layered_model.py` track-pose 扩展 | 13a | 扩展 | track-pose 联合优化 + fix_first/last + warm start |
| `threedgrut/layers/dynamic_rigid_init.py` symmetric 扩展 | 13a | 扩展 | symmetric_axis='Y' 镜像粒子 + 镜像约束 reg |
| `threedgrut/utils/fourier_time.py` | 13b | NEW | Fourier features time encoding（fourier_features_dim） |
| `threedgrut/utils/novel_view.py` | 15 | NEW | ±2 m 平移 + 小旋转 pose 生成器 |
| `threedgrut/correction/difix.py` | 15 | NEW | Cosmos-DiFix NGC checkpoint 加载 + 渐进蒸馏调度 |
| `threedgrut/layers/dynamic_deformable.py` | 16 (stretch) | NEW | permuto hash-grid + FullyFusedMLP 64×1 + canonical xyz |

### 3.2 复用 / 移植第三方代码清单（NuRec / Recon-Studio / drivestudio）

| 来源 | 用途 | Stage |
|---|---|---|
| Recon-Studio bilateral grid | T9.1 双边网格 port | 9 |
| drivestudio sky envmap inpaint | T10.1 hole-filling 实现参考 | 10 |
| NuRec parsed_config.yaml | T8.5.1 R3/R4 / T12.1-T12.7 校对值 | 8.5 / 12 |
| DepthAnythingV2 weights (HuggingFace) | T11.4 metric depth prior | 11 |
| DINOv2 weights (Facebook Research) | T13b.6 extra_signal 20 维 | 13b |
| Cosmos-DiFix NGC checkpoint | T15.2 渐进蒸馏 | 15 |
| `thirdparty/tiny-cuda-nn/` permuto | T16.1-T16.2 hash-grid + FullyFusedMLP | 16 |

### 3.3 与 v2 LayeredGaussians 的兼容性策略

**强制约束**：所有 v3 新模块按 v2 风格 `isinstance` 守卫 + config flag 默认 False，v1 / v2 ckpt 仍可加载。

| 兼容性维度 | 策略 |
|---|---|
| v1 ckpt 加载 | bilateral grid / inpaint / DiFix / DynamicDeformable 全部默认 disabled，v1 ckpt 自动走 affine / 无 inpaint / 无 DiFix / 无 deform 路径 |
| v2 ckpt 加载 | 全部 v3 新字段在 LayerSpec / ExposureModel 中可选；v2 ckpt 加载时 fallback 到 v2 行为（与 v2 测试一致） |
| 训练流程 | v3 新功能通过 config flag 启用（`trainer.use_bilateral_grid=true`, `trainer.use_difix=true`, etc.）；默认 config 与 v2 一致 |
| 单测 | v3 不破坏 v2 现有 200 个测试；每个新模块新增独立单测 + roundtrip 测试 |

---

## 4. 风险登记表（Risk Log）

| ID | 风险描述 | 触发条件 | 影响范围 | 缓解措施 | 责任 Stage |
|---|---|---|---|---|---|
| R1 | bilateral grid 仍发生 ExposureModel 退化优化 | Stage 9 30k 训练 raw vs cc 差仍 > 2 dB | Stage 9 出口失败 | freeze 提前到 step 1000 + 2×2×2 grid + 严格 lr cosine | 9 |
| R2 | LiDAR ray 在 Sky 区域 NaN/inf | Stage 11 训练梯度爆 | Stage 11 失败回滚 | valid mask 排除 sky / LiDAR depth > 100 m clip | 10 → 11 |
| R3 | DepthAnythingV2 度量不一致 | Stage 11 depth loss 不收敛 | Stage 11 出口降级 | 用 scale-shift align head；或换 DepthV2 metric 微调版 | 11 |
| R4 | track-pose 学习 collapse | Stage 13a Δpose 端点失约束 | dynamic region PSNR 不达标 | fix_first/last 权重 10× + warm start 提到 1000 步 | 13a |
| R5 | DINOv2 feat 存储爆盘 | Stage 13b DINOv2 缓存 > 200 GB | A800 盘满训练失败 | float16 + per-frame .pt + LRU 缓存 | 13b |
| R6 | DiFix NGC checkpoint 不可访问 | Stage 15 下载失败 | **结题门槛威胁** | 用 HuggingFace 公开版；若都失败，转 V4 backlog 并降级 Stage 15 出口 | 15 |
| R7 | DiFix 引入 hallucination | Stage 15 视觉验证伪影 | LPIPS / novel-view PSNR 回退 | use_color_transfer=true + novel-view 比例 50% → 30% | 15 |
| R8 | **保守门槛 cc_psnr_masked 30.0 未达** | Stage 15 出口失败 | **v3 不结题** | 回滚到 Stage 14 出口（应已 30.0）作为保守结题，DiFix 转 V4；重新评估 Stage 9/11/13a 误差累积 | 15 |
| R9 | DynamicDeformable 编译 / 训练失败 | Stage 16 hash-grid / permuto 不可用 | Stretch 失败 | 转 V4；用普通 multi-resolution hash 替换 permuto | 16 |
| R10 | 训练速度跌破 5 it/s | LiDAR ray + DiFix + Deform 累积 overhead | 30k 训练时间 > 60 min KPI | LiDAR ray batch 减半 + DiFix 每 N epoch 一次 + Deform 渐进 hash level | 11/15/16 |
| R11 | A800 显存不足 | 多层增强 + DINOv2 feat + DiFix 渲染 | OOM crash | 显存分析 + checkpoint 重启 + bg 层 cap 降到 800K | 13b 起 |
| R12 | NuRec parsed_config.yaml 校对值找不到 | T8.5.1 / T12.1-T12.7 无参考 | 校对失败 | 用 v2 默认值 + 实证 ablation 找 sweet spot | 8.5 / 12 |
| **R13** ★ | **v2 baseline novel-view PSNR 严重低于估测**（实测 < 16 dB） | T8.5.4 实测发现 | **v3 KPI 目标体系需重新校准** | (1) 视觉验证 pose 生成器无 bug;(2) 若确认 v2 真的这么差，把保守门槛降为 baseline + 8 dB / 进取改为 baseline + 10 dB 而非绝对 28/30;(3) Stage 11 LiDAR/DepthV2 几何先验权重加倍 | 8.5 |
| **R14** ★ | **Stage 17 进取目标 ≥ 30 未达** | 3DGRUT secondary ray + per-class 仍差 1-2 dB | 用户主诉求未满足 | (1) 重审 Stage 11 几何先验是否充分;(2) DiFix novel-view 比例从 50% 上调到 70%;(3) novel-view pose 范围缩小到 ±1m/±5° 满足较容易场景 | 17 |
| **R15** | novel-view PSNR 与 reconstructed 严重不一致（reconstructed 高 / novel 低） | 任意 Stage 9-14 | 过拟合训练视角 | LiDAR / DepthV2 / DiFix 等几何与新视角监督权重提升；reduce SH degree | 11 起 |

---

## 5. Done Log

> 按 Stage 顺序追加。每条包含：日期 + commit hash + Stage / Task ID + 实际改动摘要 + 关键验收数据（实测 PSNR / it/s / 耗时）。
> v3 启动后填充。初始为空。

### Stage 11 — LiDAR + DepthAnythingV2 image-space 深度监督（2026-05-30）

**路线**：改走 **image-space**（drivestudio 风格，复用 tracer 的 `pred_dist`），非 plan 原定 ray-space 旁路 forward —— tracer 已逐像素返回深度，省一遍 forward。T11.3（lidar_divergence cone）defer，出口不依赖。

**Commits（branch `worktree-stage11-lidar-depthv2`）**：
- `ae36867` T11.A1 `DepthLoss` + `compute_bg_lidar_loss`（`threedgrut/correction/depth_prior.py`）+ 9 单测
- `3b091d8`+`f512fb4`+`eb4433f` T11.A2 trainer 三 loss 接入 + grad-check tripwire + NameError 回归（`compute_bg_lidar_loss` 提 module 级 import + AST guard）
- `7c50f99` per-head depth loss TB 记录（`loss/lidar_depth`、`loss/bg_lidar`、`loss/depth_prior`）
- `f24fb36`+`f721044` T11.B1 LiDAR→image 投影脚本（`scripts/dump_lidar_depth_map.py`）+ NaN/遮挡鲁棒
- `9c39f61`+`8836928` T11.B2 `LidarDepthAuxReader`/`DepthV2AuxReader`（npz-per-frame）+ 有界缓存
- `f12c304` T11.C1 dump_clip 主体（FthetaForwardProjector 复用）+ datasetNcore train 分支注入
- `f6e4e52` T11.D1 DepthV2 下载 + dump（`scripts/dump_depth_priors.py`，HF `Depth-Anything-V2-Metric-Outdoor-Large-hf`）
- `c368b0c`+`f33dc89`+`9ac0d51` T11.F1 `mean_lidar_psnr`（`threedgrut/utils/eval_metrics.py`，render.py+trainer 双路）+ **三层 eval-path 修复**（make_test 工厂 + val/test `__getitem__` 分支转发 depth flags）
- yaml 可配置选项（`configs/apps/ncore_3dgut_mcmc_multilayer.yaml`，默认全 OFF opt-in）

**实测（A800，clip 9ae151dc，sym5cam 30k，vs baseline `v3_kpi_sym5cam_30k`）**：
| 指标 | baseline | Stage11(+depth) | 判定 |
|---|---|---|---|
| cc_psnr_masked | 26.04 | **25.98**（Δ-0.06） | **不退化 ✓** |
| novel-view LPIPS (4模式avg) | 0.6022 | **0.6007**（Δ-0.0015 噪声级） | **无提升 ✗** |
| 视觉 A/B（viser 3dgut） | — | **肉眼无差别** | 印证无提升 |
| mean_lidar_psnr | — | 20.68（< 25 NuRec ref） | 口径未校准 |
| it/s | — | 4.68（30k≈107min） | — |

**结论**：⚠️ **工程链路完整且正确**（dump→注入→3 loss→eval 全验证），但 **depth 监督在当前配置下对 novel-view 无可观测效果**，主 KPI（+3.0）未达成。量化（LPIPS）+ 视觉 A/B 一致。
**根因（疑）**：`lidar_w_decay=1.0` 让 LiDAR λ 到 30k 衰到 ~2.4% + depth_prior inverse-depth 早饱和（27→0 by step 1000）→ **监督训练中途消失**；且 30k/5cam 下 RGB 信号已 pin 住几何（高斯 LiDAR 初始化），先验改进空间小。呼应 plan 风险 R13/R14。
**已沉淀为 opt-in yaml 选项**（默认 OFF，baseline 字节等价）。**调参待验方向**：`lidar_w_decay=0`（全程监督）+ 提 `lambda_depth_prior` + depth_prior 换不饱和 loss。

**数据资产发现（A800 schema 探查）**：clip 内 NuRec nre-tools 已预生成 `aux.depth.zarr.itar`（4.7GB，每相机 dense f16 深度，疑即 DepthV2 metric，未来可直读省 dump）+ `aux.lidar-camvis.zarr.itar`（每点 uint8 可见性，非 per-pixel 深度）。**Stage 13a 增强机会**：scatter 前用 camvis 做遮挡剔除 + lidar-sseg 剔动态点（消移动车拖影，关联车道线退化）。

### V3-VIZ — 可视化诊断 + viser GUI 增强（2026-05-26）

可视化基础设施 sub-stage，承担 v3 训练前的"看清楚问题"诉求，复用 B3_30k ckpt 不重跑训练。

- **V3-VIZ.1** BEV 散点诊断：`scripts/diagnose_layered_bev.py` + `threedgrut_playground/utils/bev_renderer.py`，CPU/Mac 可跑，输出每帧 BEV PNG（ego + cuboid + 各 layer Gaussian 中心散点按 layer 着色）。
- **V3-VIZ.1b** BEV stitched：`scripts/diagnose_layered_bev_stitched.py` + `threedgrut_playground/utils/bev_stitcher.py`，5 相机 FTheta IPM (z=0 vehicle plane) 拼成 BEV 底图，cuboid + Gaussian 叠加其上。ThinkPad 批量跑了 500 帧 (`/tmp/b3_bev_stitched_full/`, 455MB)。
- **V3-VIZ.2** viser cuboid label：每个 active cuboid 顶角加 "tID | class" 3D 文本 billboard，跟随 Active cuboids 可见性。FTheta + pinhole 两条路径都生效。
- **V3-VIZ.3** viser 多相机 dropdown + Follow Camera：从 `--dataset_path` 自动加载 7 相机的 per-frame c2w + FTheta intrinsics + FOV，dropdown 切换时同时 snap viewer pose + 改 engine FTheta intrinsics + 改 client FOV，跟 Follow Ego 互斥。
- **V3-VIZ.5** ego trajectory 诊断：`scripts/diagnose_ego_trajectory.py` + 单测。结论：B3_30k ckpt trajectory 数据干净（max kink 2.4°），dt 双峰 (33ms/66ms) 为正常 rolling-shutter 模式，"乱七八糟"来自 viser 渲染端。
- **架构统一**：cuboid wireframe / label / ego_trajectory / track_trajectories **全部走 viser 3D scene primitive**（移除 FTheta image-space overlay 路径中的 cuboid+ego+track 块）。ego 从 `add_spline_catmull_rom` 改成 `add_line_segments`（避免 dt 双峰处的样条过冲）。Trade-off：3D primitives 用 viser 内部 pinhole 投影，跟 FTheta backdrop 在画面边缘可能有几像素到几十像素漂移；好处是 cuboid + label + trajectory 之间互相完全对齐，没有 FTheta overlay 的 behind-camera 剪枝 bug。
- **Dichotomy 验证**：`scripts/validate_cuboid_7cam.py` 在 5 相机原始图上投影 cuboid 全部精确贴合实际车辆 → 证明 cuboid pose 数据 + FTheta 投影算法 100% 正确。
- **Diagnostic 工具**：`scripts/diagnose_bg_in_cuboid.py` 跑 B3_30k 结果 → bg 层 1.58% (15,845)、road 层 10.53% (21,067) 粒子落入 cuboid 内，dyn → bg/road 错分残留（V3 backlog 等待 V3-P2 修复）。
- **Mac/CPU 单测**：`test_bev_renderer.py` (5 cases) + `test_diagnose_ego_trajectory.py` (9 cases) 全绿。
- **遗留**：(a) ego/track trajectory 视觉上仍有少量异常（用户报告，未完全定位）；(b) Follow Ego toggle 偶发不自动更新（中段反馈，未修）；(c) `compare_raw_vs_recon_cuboid.py` recon 渲染破（kaolin Camera convention 跟引擎不一致，留作后续调试）；(d) V3-VIZ.4（viser 嵌入 BEV panel）未做（P2）。

**关联未来 task**：V3-P2 — 修 dyn→bg/road 错分（cuboid-exclusion mask 加严 / 训练后 post-process 粒子迁移）。

---

### V3-E4.1 — Renderer.from_checkpoint 还原 tracks_poses + scene_extent（2026-05-27）

**Commit**: `<next>` （render.py L143-178 fix + tests/test_render_reload_parity.py 新增 6 测试）

**背景**：T8.5.7 commit 0ffd738 加了 `logger.warning` 标注 standalone `Renderer.from_checkpoint()` 比 train-end-of-train eval 低 **~3 dB**，但未定位根因。Phase 8.5 T8.5.4 必须用 v2 ckpt 跑 standalone novel-view eval 拿 v3 主 KPI baseline，**不修这个 bug 则 baseline 数字偏低 ~3 dB，整个 v3 KPI 校准会错**。

**根因定位（5cam 5k smoke 历史数据复现 + ckpt inspection bisection）**：

| 指标 | train-end (live model) | reload (修复前) | Δ |
|---|---:|---:|---:|
| `mean_psnr_masked` (raw) | 21.42 | 19.48 | **−1.94 dB** |
| `mean_cc_psnr_masked` | 23.36 | 20.40 | **−2.96 dB** |

raw psnr 也掉 ~2 dB 排除了"eval 漏 apply exposure"假设（两条 eval 路径都不 apply ExposureModel）。Ckpt inspection 暴露两个真实根因：

1. **tracks_poses 未恢复**（主因）：训练时 `trainer.py:433` 在 dataset ready 后调 `model.populate_tracks(tracks)` 把 70 个 dynamic_rigid track 的 per-frame poses (`Tensor[599,4,4]` × 70) 写入 model buffer；但 `render.py:from_checkpoint` 路径**完全不调** populate_tracks → 300K dyn Gaussians 留在 canonical pose（不跟随车辆运动）→ 所有含动态车辆的帧 raw psnr 显著下降。Tracks pose 数据本身存在 `ckpt["viz_4d"]["tracks"][<tid>]["poses"]` + `ckpt["viz_4d"]["tracks_camera_timestamps_us"]`，只是 render.py 没读。
2. **scene_extent 传 None 给 ctor**（次因）：`render.py:164` 传 `LayeredGaussians(conf, specs, scene_extent=None)` 而 ckpt 里有 `model.scene_extent=1.103`。MoG.init_from_checkpoint 会从 sub-dict 覆盖回正确值，所以单测 / inference 行为对 model 本身无影响，但 `model.scene_extent`（顶层）保持 None 与 live 训练态不一致，未来代码引用 `model.scene_extent` 会回归。

**修复**：把 `playground/engine.py:1340-1360` 已有的正确 pattern 复制到 `threedgrut/render.py:143-178`：
- Ctor 传 `scene_extent = float(checkpoint.get("model", {}).get("scene_extent", 1.0))`
- `init_from_checkpoint` 后从 `viz_4d.tracks + tracks_camera_timestamps_us` 调 `populate_tracks`
- 用 `viz_4d` 缺失 / `tracks_camera_timestamps_us` 缺失的两层 guard 兼容 v1 + pre-T8.2 ckpts
- 删掉 0ffd738 的 misleading warning（warning 文本怀疑是 exposure_state，实际无关）

**验收（A800 sym5cam 30k ckpt, byte-identical ★）**：

| 指标 | train-end | reload (修复后) | Δ |
|---|---:|---:|---:|
| `mean_psnr_masked` | 15.2878 | **15.2878** | **0.0000** ★ |
| `mean_cc_psnr_masked` | 26.0436 | **26.0436** | **0.0000** ★ |
| `mean_cc_psnr` | 23.8795 | **23.8795** | 0.0000 |
| per_camera (5 个 × 6 metric) | 全 | 全 | 全 0 ★ |

远超 plan 验收门槛（差 < 0.1 dB）→ 修复完全闭合 gap，T8.5.4 可以直接启动。

**测试**：
- `tests/test_render_reload_parity.py` 新增 6 测试：4 contract（viz_4d → populate_tracks 调用 / ctor scene_extent / no-viz_4d / shared_ts 缺失）+ 2 source-level guard（greppy 检查 render.py 不会被回归性误删）— Mac venv 0.08s 全绿
- 既有 58 个 layered / render_per_camera / track_ids / engine_layered_load 测试无回归

**关联未来 task**：
- **下一步 V3-P1 (Stage 9 T9.1-T9.5)**：bilateral grid 替换 ExposureModel + L2 reg + 2-stage freeze（Plan agent 现已确认 V3-P1 范围按原计划走，T9.3 走路线 B = eval 也 apply correction）
- Phase 1 决策门通过：V3-E4.1 与 ExposureModel 无关，V3-P1 实施不需调整范围

---

### Stage 8.5 启动条目

### T8.5.3 / T8.5.4 — Novel-view pose 生成器 + v3 baseline LPIPS 实测（2026-05-27）

**Commits**: `b27d09e` (T8.5.3 code) + `<next>` (T8.5.4 数据回填 Done Log + KPI 表更新)

**任务**:
- T8.5.3: NEW `threedgrut/utils/novel_view.py` 4 模式扰动 (lateral_1m / lateral_2m / yaw_5deg / yaw_10deg) + render.py `--novel-view` flag + render_all 4 模式循环 + metrics.json `mean_novel_lpips_*` 字段
- T8.5.4: A800 sym5cam 30k ckpt 上跑全 5cam 375 帧 × 5 渲染 (1 anchor + 4 novel) = 1875 renders, 14 min wall-clock

**关键设计决策（用户对齐, 2026-05-27 16:00）**:
- **Anchor**: 训练帧 c2w
- **4 档幅度**: v3_plan 默认 (±1m / ±2m 横向平移 + ±5° / ±10° yaw 旋转)
- **GT 策略**: 不算 PSNR vs anchor GT — 小扰动下 image content 因 parallax 变化，PSNR 是 noise floor 而非 view-extrapolation 质量。**只算 LPIPS + 视觉验证**；PSNR 留到 Stage 15 DiFix synthesized GT 可用时。
- **无 region mask**: 不算 per-region PSNR 就不需要 mask 拆解
- **Baseline ckpt**: T8.5.7 对称 5cam 30k（v3 实际 baseline）而非 v2 Stage 7 旧非对称 5cam

**Rolling-shutter integrity**: T_to_world 和 T_to_world_end 用 **同一 world-frame delta** 同步扰动（rigid trajectory shift）。lateral 用 start frame 的 right axis 算两端平移；yaw 用 start frame 的 up axis、绕 start 位置 pivot end frame。`perturb_shutter_pair()` 处理，单测覆盖 `test_shutter_pair_lateral_preserves_rigid_shift` + `test_shutter_pair_yaw_rotates_end_around_start_origin`。

**T8.5.4 关键结果（v3 Stage 8.5 baseline ★）**:

| 维度 | LPIPS |
|---|---:|
| Anchor `mean_lpips`（重建质量参考） | **0.3702** |
| Anchor `mean_cc_lpips` | 0.3432 |
| Anchor `mean_lpips_masked` | 0.3286 |
| **Novel-view avg (4 modes)** | **0.6022** ★ |
| Novel `lateral_1m` | 0.5838 |
| Novel `yaw_5deg` | 0.5895 |
| Novel `lateral_2m` | 0.6168 |
| Novel `yaw_10deg` | 0.6188 |

**baseline 解读**:
- Novel-view LPIPS 比 anchor LPIPS 劣化 **+0.23**（相对 +58-67%）
- 4 模式 LPIPS 单调随扰动幅度增加（1m < 2m, 5° < 10°）—— 几何上合理
- 平移与旋转幅度对 LPIPS 影响相当：1m ≈ 5°, 2m ≈ 10°
- 重建 anchor `mean_cc_psnr_masked = 26.0436` 与 train-end metrics.json byte-identical（V3-E4.1 fix 持稳）

**v3 主 KPI 起点确认**:
- v3 进取目标：把 novel-view LPIPS **拉回接近 anchor LPIPS 0.37**（即 Stage 9-17 累计改善 ~0.23）
- 主要改善源：Stage 10 sky inpaint（novel sky 不爆黑洞）+ Stage 11 LiDAR/DepthV2 几何先验 + Stage 15 DiFix 蒸馏

**视觉验证**:
- 4 模式 × 5 帧样本图保存在 `<out_dir>/ours_30000/novel_view/<mode>/{00000-00004}.png`
- 抽样肉眼对照：lateral_2m 明显侧移 / yaw_10deg 明显旋转；底部 ego mask 在 novel pose 下错位形成黑椭圆（已知边界 case，所有 mode 受影响相同，不破坏相对 baseline）

**Edge case 与已知限制**:
- ego mask 沿用 anchor 帧的 mask（在 novel pose 下不严格准确），导致 LPIPS 数字略偏高。Stage 10/11 阶段 mask 管线 V3-D8/D9/P2 改进时一并修
- LPIPS 是相对感知度量，绝对数字不直接对标 NuRec 论文（NuRec 报的是 reconstructed PSNR），与 v2 baseline 也无对照数字（v2 没跑过 novel-view）

**测试**:
- `tests/test_novel_view.py` 13 测试（CPU，0.03s）：4 模式扰动数学 + rolling-shutter rigid trajectory + torch wrapper
- 26 个 render-related 测试全绿（test_novel_view.py + test_render_reload_parity.py + test_render_per_camera.py）

**关联未来 task**:
- Stage 10 (T10.1-T10.4): Sky envmap inpaint — 期望降 novel_lpips_lateral_2m + novel_lpips_yaw_10deg 最多（sky 区域劣化最严重）
- Stage 11 (T11.1-T11.6): LiDAR + DepthV2 几何先验 — 期望降所有 4 模式平均（几何稳定）
- Stage 15 (T15.2-T15.5): Cosmos-DiFix 渐进蒸馏 — 直接为 novel view 设计的伪影修复器，期望降全部模式 + 最终拉回接近 anchor 0.37

---

### V3-L7 / V3-StageA — 学习式 track-pose 30k 预研（Run B, ThinkPad, 2026-05-27）

**Commits**: `6b84d54` (Merge stageA into main) + `bb49bc5` (V3-StageB temporal smoothness scaffolding) + `e902bf6` (StageB device fix) — 全部在 origin/main，源代码来自 `.claude/worktrees/v3-learnable-track-pose-stageA/` 已合并

**任务背景**：T13a.4 / V3-L7 完整版需 fix_first/last + warm start + temporal smooth reg + pose prior reg；stageA 先把基础架构落下（learnable Parameter 替换 buffer + ckpt 持久化 + freeze-until-iter 调度），跑一次预研看 photometric loss 自校准 NCore cuboid GT 的潜力。

**配置（Run B = `poseopt_on_30k_freeze10k`）**：

```yaml
trainer.learnable_pose:
  enabled: true
  lr_rotation: 1.0e-05               # 极保守 LR (refine, 不重学)
  lr_translation: 0.0001
  freeze_until_iter: 10000           # 步 0-9999 freeze, 步 10000-30000 联合优化
  lambda_temporal_smooth_trans: 0.0  # ⏸ stageA 暂不启用
  lambda_temporal_smooth_rot: 0.0    # ⏸ stageA 暂不启用 (T13a.4 完整版需要)
  lambda_pose_prior_trans: 0.0       # ⏸ 同上
  lambda_pose_prior_rot: 0.0         # ⏸ 同上
```

**代码改动机制**（[layered_model.py:321-330](.claude/worktrees/v3-learnable-track-pose-stageA/threedgrut/layers/layered_model.py:321)）：
- v2 baseline: 每 track 一个 `_track_pose_<tid>` buffer `[F, 4, 4]`，GT 不可学
- stageA：拆成 `_track_quat_<tid>` Parameter `[F, 4]` (wxyz) + `_track_trans_<tid>` Parameter `[F, 3]`，渲染时归一化 quat → R + trans 组合 SE(3)，photometric loss 反向传播到这 70 tracks × 599 frames
- 同时保留 frozen `_track_pose_gt_<tid>` buffer 用于 resume detection / 未来 viz diff
- ckpt 顶层新增 `learnable_pose_state` key（独立 nn.Module 容器 `model.layered_track_state`）

**关键结果（Run B, sym5cam 30k, ThinkPad RTX 4090, 12:56 完成）**:

| 指标 | T8.5.7 baseline (no poseopt) | Run B (poseopt + freeze10k) | Δ |
|---|---:|---:|---:|
| `mean_psnr_masked` (raw) | 15.288 | **17.344** | **+2.06 dB ★** |
| `mean_cc_psnr_masked` | 26.044 | 25.431 | **−0.61 dB** ⚠️ |
| `mean_class_psnr` (per-cuboid 动态车辆区域) | 17.276 | **18.958** | **+1.68 dB ★** |
| `class_psnr_n_low_15db` | 1298 | (待确认) | — |

**Ckpt 持久化验证 ✅**（`scripts/inspect_poseopt_ckpt.py` 走 Run B `ckpt_30000.pt`）：
- 顶层 `learnable_pose_state` key 存在
- 70 个 `_track_quat_*` Parameter (599, 4)；std min 0.36 / mean 0.44 / max 0.50 — 单位 quat 帧间正常变化
- 70 个 `_track_trans_*` Parameter (599, 3)；std min 2.6 m / mean 22.4 m / max 88 m — 合理车辆位移
- 70 个 `_track_pose_gt_*` frozen buffer 保留 — resume / diff 用
- 0 个 v2 `_track_pose_*` 旧 buffer — 完全切到 learnable
- → 历史 memory 1416/1461 "Silent Data Loss: Learnable Track Pose Parameters Not Persisted" 告警**已修复**

**关键观察 + 解读**:
1. **raw psnr +2.06 dB**：主要来自 dynamic_rigid Gaussians 不再被错误的 NCore cuboid annotation pose 拖累（自动 tracking 标注本来就 ~5-10cm / ~1° noise），photometric loss 在 freeze 解锁后让 SE(3) 自校准
2. **class_psnr +1.68 dB**：per-cuboid 动态车辆区域直接受益，与 raw 改善方向一致
3. **cc_psnr -0.61 dB**：cc affine 修正吸收 tone 偏差，learnable_pose 改善几何不直接受益；且 lr_pose=1e-5 量级训练 20k 步可能轻微扰动 ExposureModel 优化平衡 — 与 V3-P1 退化 mode 一致逻辑
4. **trans std 88 m max** 是合理的高速车辆 20 秒位移，**不**是 over-fitting

**已知 caveat（影响后续标 T13a.4 ✅）**:
- ⚠️ stageA 仅启用 freeze 调度 + 基础 SE(3) Parameter 化；T13a.4 完整版还要：
  - **fix_first/last**: Δpose 端点固定（防 collapse）
  - **warm start ≥ 500**: freeze 解锁后渐进释放 LR
  - **temporal_smooth reg**: 相邻帧 Δpose 平滑（StageB `bb49bc5` 已写好但 lambda=0 未启用）
  - **pose_prior reg**: Δpose vs GT 偏离的 L2 约束
- ⚠️ Run B ckpt reload 必须用 stageA worktree 的 render.py（含 `learnable_pose_state` 加载链）；**当前 main 分支的 render.py（V3-E4.1 fix 后）也不支持** `learnable_pose_state` 加载——加载到 buffer-mode model 会回退到 frozen GT pose，raw psnr 退回 baseline 15.29
- ⚠️ Novel-view（T8.5.3）也尚未在 Run B ckpt 上跑过；T13a.4 完整出口要在 Stage 13a 时再 novel-view eval

**V3-L7 完整版 T13a.4 状态**: ⬜ Todo (依然在 Backlog)。stageA 预研只解锁了 +2.06 dB raw 的潜力证据，**完整 V3-L7 的 +1.5 dB novel-view 增益预算仍需 fix_first/last + warm start + smooth/prior reg 一起验证**。

**关联未来 task**:
- StageB (`bb49bc5`) 已加 temporal smoothness reg 代码，下次实验把 lambda 打开（如 0.01）跑 Run C 看 cc_psnr 是否回归（解决 cc_psnr -0.61 dB 退化）
- T13a.4 标 ✅ 的前提：fix_first/last + warm start + smooth reg + pose prior 都启用且 ablation 数据完整 + main 分支 render.py 支持 `learnable_pose_state` reload

---

### T8.5.7 / V3-E4 — 5-cam vs 7-cam KPI 对照 + 对称 5-cam 切换（2026-05-27）

**Commits**: dd6c39f (Phase 1 代码) → 0ffd738 (standalone reload fix + 知识警告) → `<next>` (对称 5-cam 切换 + 文档同步)

**动机**：v2 multilayer 默认 5-cam ring 是非对称的 `[front_wide_120, rear_tele_30, cross_left_120, cross_right_120, rear_left_70]` — 用 30° 窄视角 rear-tele 占位后向，缺右后 70°，左右不对称。原计划是"5-cam vs 加 2 个相机 (rear_right_70 + front_tele_30)"对比，但用户指出 baseline 选择本身就有问题：真正的对称环视 ring 应该是 `[front_wide_120, cross_left_120, cross_right_120, rear_left_70, rear_right_70]`。三组 30k 实测后结论清晰，**应切对称 5-cam 而不是 7-cam**。

**实验设计**：

| 实验 | iter | camera_ids | wall-clock |
|---|---:|---|---:|
| **E1a** 原非对称 5cam | 5000 | front_wide + rear_tele_30 + cross_l + cross_r + rear_l | ~9 min |
| **E1b** 7cam | 5000 | E1a + front_tele_30 + rear_right_70 | ~13 min |
| **E1c** 7cam per-cam step parity | 7000 | E1b 同 camera_ids | ~18 min |
| **E2a** 7cam 30k | 30000 | 全 7 个相机 | ~80 min |
| **E2b** 对称 5cam 30k | 30000 | front_wide + cross_l + cross_r + rear_l + **rear_right_70** | ~75 min |

**关键工具**：dd6c39f commit 新增 render.py `metrics.json["per_camera"]` 字段 + `--eval-cameras` hydra filter。配合 NCoreDataset val frame 切分按 `frame_idx % 8 == 0`（与 camera 数无关），同一 camera_id 在不同 train-set 配置下 val 帧集完全一致，**train-time metrics.json 的 per_camera 字典就是天然的同名公平对比工具，不需要 standalone re-eval**。

**核心结果 — 3 组在 4 公共相机上 cc_psnr_masked 对照（dB）**：

| Camera | E1a 5cam(原)5k | E2a 7cam 30k | E2b **对称5cam 30k** |
|---|---:|---:|---:|
| front_wide_120 | 20.19 | 21.24 | **21.75** |
| cross_left_120 | 24.86 | 26.71 | **27.01** |
| cross_right_120 | 26.87 | 28.08 | **28.58** |
| rear_left_70 | 22.33 | 24.74 | **25.10** |
| **4-cam mean** | **23.56** | **25.19** | **25.61** ★ |
| **Δ vs E2b** | -2.05 | -0.42 | baseline |

**rear-back 相机选择关键证据**：

| 配置 | 相机 | cc_psnr_masked |
|---|---|---:|
| E1a 非对称 5cam | rear_tele_30 | 22.56 |
| E2a 7cam（含 rear_tele_30） | rear_tele_30 | 23.38 |
| **E2b 对称 5cam** | **rear_right_70** | **27.78** |
| E2a 7cam（也含 rear_right_70） | rear_right_70 | 26.99 |

**`rear_right_70` 比 `rear_tele_30` 高 +5.2 dB** — 后向用 30° 窄视角是 baseline 的主要短板。

**结论 + 行动**：
1. **7-cam 在所有公共相机上劣于对称 5-cam，平均 -0.42 dB** — 多相机几何约束没补偿训练量摊薄（每相机 1/7 vs 1/5）
2. **对称 5-cam vs 原非对称 5-cam 大幅提升 +2.05 dB** — 主要来自 `rear_right_70` 替换 `rear_tele_30`
3. **切换 multilayer.yaml 默认 camera_ids → 对称 5-cam** (本 commit)
4. 后续 baseline 数字以 E2b cc_psnr_masked **26.04** (global, 5cam test set) 替换 v2 Stage 7 的 24.70

**Independent observation**: 30k 训练相比 5k smoke，`raw psnr_masked` 反而下降（E2a 18.35 / E2b 15.29 vs E1a 21.42），但 `cc_psnr_masked` 大幅上升 → ExposureModel 学到了让 raw RGB 偏离 GT 的 tone shift，cc affine fit 能拉回但说明 exposure 退化严重。**这正是 V3-P1 (T9.1-T9.4) 双边网格 + L2 reg + 2-stage freeze 要修的问题**。本 task 仅记录现象。

**Caveat — standalone reload 已知问题（V3-E4.1 follow-up）**：
- `Renderer.from_checkpoint()` 对 LayeredGaussians ckpt 加载时缺少 ExposureModel state（`exposure_state` key）和可能的 sky_envmap warmup buffers 恢复，导致重新加载评估比 train-end-of-train 自带 metrics.json 低 ~3 dB
- commit 0ffd738 加了 logger.warning 明确告知 caller，但加载链路本身未完整修复
- 本 task 结论不依赖 standalone reload（用 train-time metrics.json per_camera 直接对比），但 V3 后续 novel-view eval / cross-clip eval 需要 standalone reload 时必须先解决 V3-E4.1

**测试**：tests/test_render_per_camera.py 7 个新测试 + tests/test_trainer_masked_metrics.py 等 23 个既有测试无回归（70/70 通过 on Mac venv）。

---

### V3-L5 + V3-L8 + V3-L9 — NuRec dynamic_rigids tricks（2026-05-27）

Branch: `v3-l589-on-main` (基于 origin/main, push 至 https://github.com/etendue/3dgrut/tree/v3-l589-on-main)

**Commits**（cherry-pick from worktree-feat-v3-l589-symmetric onto origin/main）:
- `c46cf63` feat(V3-L5/L8/L9): NuRec dynamic_rigids tricks — symmetric_axis + per-track albedo/scale
- `f5ea26d` fix(V3-L589): use Hydra ++ to override existing yaml defaults
- `9d2e0ff` chore(V3-L589): vast.ai bootstrap + 5k smoke A/B helper scripts
- `46eba02` docs(CLAUDE): Vast.ai 远程执行环境章节 + 5k A/B 实战经验

**实装范围**（默认全 OFF，通过 `layers.overrides.dynamic_rigids.<key>` 翻转）：

| Trick | 文件 | 行为 |
|---|---|---|
| **V3-L5** `symmetric_axis: 'Y'` | `threedgrut/layers/dynamic_rigid_init.py` | LiDAR object-local 点云 y 取负后 concat（subsample cap=5000 前） |
| **V3-L8** `optimize_track_albedo` | `threedgrut/layers/layered_model.py` | `_track_albedo_table[K,3]` 加到 features_albedo DC SH; warmup=500 翻 requires_grad |
| **V3-L9** `optimize_track_scale` | `threedgrut/layers/layered_model.py` | `_track_log_scale_table[K,1]` 加到 scale (log-space); 同 warmup |
| 基础设施 | `registry.py` extra-key 白名单 / `trainer.py` warmup hook / `render.py` 4 metrics 字段 / `multilayer.yaml` 默认 OFF | — |

**Mac 单测**：v3-l589-on-main worktree **133 PASS / 0 regression**（43 新增/扩展测试 + main 新增 test_render_per_camera.py 都过）。

#### 5k smoke A/B（vast.ai RTX 4090, **非对称 5cam** with rear_tele_30，2026-05-27 早期）

vast 38003930 RTX 4090, 10.97 it/s, 7.6 min train + 8 min eval per run。Note: 这是在 main 切对称 5cam 之前跑的，**baseline 不能直接与 30k 对比**。

| 指标 | 5k Baseline (OFF) | 5k Experiment (ON) | Δ |
|---|---:|---:|---:|
| mean_cc_psnr_masked | 23.290 | 23.536 | **+0.246** |
| mean_class_psnr | 20.295 | 20.296 | +0.001 |
| heavy_truck_psnr | 21.563 | 21.709 | **+0.146** |
| bus_psnr | 22.161 | 22.309 | **+0.149** |
| class_psnr_n_low_15db | 500 | 435 | **−65 (−13%) ✅** |
| track_albedo_l2_mean | null | 6.9e-4 | (warmup 不足) |
| track_log_scale_std | null | 3.4e-3 | (warmup 不足) |

**5k 结论**：V3-L5 信号正向（heavy_truck/bus +0.15, n_low_15db −13%）；V3-L8/L9 未充分激活（track_*_std ≈ 0）。

#### 30k full A/B（A800 GPU 1, **对称 5cam** with rear_right_70，main baseline 26.04）

A800 SXM4-80GB, 6.89 it/s, 72.5 min train + 8 min eval per run。Baseline `mean_cc_psnr_masked = 25.99 dB` **与 main report E2b 30k=26.04 仅差 −0.05 dB → 实验环境与 main 完全一致 ✅**。

| 指标 | 30k Baseline (OFF, 对称5cam) | 30k Experiment (ON) | Δ | 5k Δ 对照 |
|---|---:|---:|---:|---|
| mean_psnr_masked | 15.68 | 15.41 | −0.27 | (5k: −0.32 同向) |
| **mean_cc_psnr_masked** | **25.99** | **26.03** | **+0.04** | (5k: +0.25 ↓信号严重弱化) |
| mean_cc_psnr | 23.88 | 23.88 | 0.00 | (5k: +0.23 ↓) |
| **mean_class_psnr** | 17.73 | 17.61 | **−0.12** | (5k: 0.00 ↓**反向**) |
| automobile_psnr | 17.65 | 17.51 | −0.14 | (5k: −0.01 ↓加剧) |
| heavy_truck_psnr | 17.87 | **17.99** | +0.12 | (5k: +0.15 ≈) |
| bus_psnr | 19.86 | 19.88 | +0.02 | (5k: +0.15 ↓) |
| **class_psnr_n_low_15db** | 1042 | **1260** | **+218 (+21%) ❌** | (5k: −65 (−13%) ✅ **反转！**) |
| (per-camera) front_wide_120 cc_psnr_masked | 21.82 | 21.78 | −0.04 | — |
| (per-camera) rear_right_70 cc_psnr_masked | 27.50 | 27.77 | +0.27 | — |

**⚠️ 30k 结果与 5k 趋势反转**：

| 现象 | 5k smoke | 30k full | 解读 |
|---|---|---|---|
| n_low_15db | **−13%** ✅ | **+21%** ❌ | 5k 把"困难"车辆区域救回，30k 反而推垮更多 |
| heavy_truck Δ | +0.146 | +0.12 | 量级差不多但 30k 偏弱 |
| bus Δ | +0.149 | +0.02 | 30k 几乎归零 |
| mean_cc_psnr_masked Δ | +0.25 | +0.04 | 30k 几乎归零 |

⚠️ **A800 是 rsync mirror，experiment 30k 的 eval 在另一个并行 task (`multilayer_poseopt` 16:45 启动) overwrite render.py 之后跑**，所以 experiment metrics.json 缺失 4 个 V3 diagnostic 字段 (`symmetric_axis`, `track_albedo_l2_mean`, `track_log_scale_mean`, `track_log_scale_std`)。但**训练本身不受影响** — baseline metrics.json 含 V3 null 字段证明 render.py 在 baseline 跑时正确；experiment 训练时 ⚡ V3-L8/L9 warmup at step 500 实测触发证明 trainer.py + layered_model.py 是 V3 版本。**如需 diagnostic 数值，可从 ckpt 后处理（experiment ckpt `_track_albedo_table` / `_track_log_scale_table` Parameter 仍在）**。

**Run 信息**：
- A800 baseline: `/root/work/yusun/ncore-nurec/output/v3_L589_baseline_30k_2605_27_144607/`
- A800 experiment: `/root/work/yusun/ncore-nurec/output/v3_L589_on_30k_2605_27_144607/`
- 本地 archive: `/Users/etendue/repo/report/v3_L589_30k_results/{baseline_OFF,experiment_ON}/`
- 5k vast archive: `/Users/etendue/repo/report/v3_L589_5k_results/`
- Plan 文件：`/Users/etendue/.claude/plans/users-etendue-repo-report-nurec-av-usdz-crystalline-toast.md`

**结论 + Follow-up 任务（V3-L589 不可直接 merge to main，需要 ablation 后再定）**：

1. **V3-L5/L8/L9 全开** 在 30k 长跑出现副作用（n_low_15db +21%），不能合 main
2. **可能成因**：V3-L8/L9 (per-track albedo/scale) 在 30k 才真正激活（warmup=500 后 29500 步 cosine decay LR 1e-5 累积），但**学过头**让 dyn 层 track-level offset 推到错误方向，加上 V3-L5 init 增强翻倍 dyn 粒子，让 cuboid 内分布不稳
3. **关键 ablation 必须分开做**：
   - **V3-L5b** sym=Y only（albedo=false, scale=false） — 隔离 init 增强单独效果，30k 数据是否仍维持 5k 的 +0.15 dB 趋势？
   - **V3-L8b** albedo only — 单独激活 albedo 在 30k 的效果
   - **V3-L9b** scale only — 同上 for scale
   - **V3-L8/L9 调参**：warmup_steps 500→5000, lr 1e-5→1e-6，避免学过头
4. **5k smoke 信号过于早期**，30k 才暴露真实效果 — **未来类似 NuRec tricks 评估必须 30k 起步**，不要再用 5k smoke 作 acceptance gate

**关联未来 task**：V3-L5b（sym only 30k）；V3-L8b / V3-L9b（单独激活 30k）；V3-L8/L9 LR ablation（warmup ↑ + lr ↓）。

---

## 6. v3 工作流补充（在 CLAUDE.md 基础上）

CLAUDE.md 已有 v2 工作流（A800 远程执行 + 把关清单 A/B/C + 任务完成同步文档）继续适用。v3 增加以下补充：

### 6.1 A800 双档 KPI 验证流程（novel-view 主 KPI + reconstructed 辅 KPI）

每个 Stage 出口必须按 **novel-view PSNR 主 KPI** + **reconstructed cc_psnr_masked 辅 KPI** 双重验证：

1. **必读字段**（每个 Stage 出口跑完）：
   - `mean_cc_psnr_masked`（reconstructed 辅 KPI，必须 ≥ 24.7 不退化）
   - **`novel_psnr_<档>_<region>` × 16 字段**（4 档 pose × {full / sky / dyn / bg}）
   - `mean_novel_psnr_avg`（4 档平均，**v3 主 KPI**）

2. **保守门槛达成判定**（Stage 15）：
   - `mean_novel_psnr_avg ≥ 28.0` ★
   - Novel Sky region ≥ 28 + Novel Dynamic region ≥ 25
   - reconstructed cc_psnr_masked ≥ 24.7
   - 写入 v3_plan.md § 5 Done Log + commit hash

3. **★ 进取目标达成判定（Stage 17，用户进取主线）**：
   - `mean_novel_psnr_avg ≥ 30.0` ★ 用户主目标
   - ±2m / ±10° 极端档单独 ≥ 28
   - reconstructed cc_psnr_masked ≥ 26.5

4. **保守门槛失败处理**（Stage 15 `mean_novel_psnr_avg < 28.0`）：
   - v3 **不结题**
   - 回滚 Stage 14 出口作为保守结题基线（baseline + 5.5 dB）
   - DiFix / 形变 / 评估工具 全部转 V4
   - Stage 18 WP_V3_Report.md 报告"v3 保守门槛未达，回滚 Stage 14 结题，V4 backlog 扩大"

5. **Stage 16 (stretch only) 触发**：用户决策，不自动启动。stretch 失败不阻塞结题。

### 6.2 DiFix / DepthAnythingV2 / DINOv2 模型依赖管理

- **DepthAnythingV2**：Stage 11 启动时 `scripts/download_depth_anything_v2.sh`（NEW）下载到 `models/depth_anything_v2/`，加入 .gitignore；HuggingFace 公开
- **DINOv2**：Stage 13b 启动时 `scripts/download_dinov2.sh`（NEW），HuggingFace 公开
- **Cosmos-DiFix**：Stage 15 启动时 `scripts/download_difix.sh`（NEW），NGC 公开（需 NGC API key）
- 三个模型本地缓存路径在 v3_architecture.md 文档化；A800 与 Mac 各自下载，rsync 不传

### 6.3 Stage 16 stretch 触发条件 + Stage 17 进取主线说明

- **Stage 17 不再是 stretch** — 用户进取目标 novel-view ≥ 30 的最后冲刺，应在 Stage 15 出口达成后继续推进
- **Stage 16 是 stretch only**（DynamicDeformable 复杂度极高 + 主要针对行人/骑行 region）：
  - 触发判定由用户决策（不自动启动）
  - 若用户不启动 stretch，Stage 15 → Stage 17 → Stage 18 直推结题
  - Stretch 失败不阻塞结题，转 V4 backlog
- Stage 16 与 Stage 17 **可并行启动**（依赖图无相互依赖）

### 6.4 ThinkPad RTX 4090 部署（v3 视觉验收辅助主机）

**定位**：A800 是 Ampere datacenter SKU（RT cores 被阉割），OptiX dlopen segfault → **A800 做不了 Gaussian 渲染**，viser_gui_4d 视觉验收只能走 `--no_gaussian_render`（仅 scene primitives）。**ThinkPad RTX 4090**（消费级 RTX，有完整 RT cores）作为本地视觉验收主机，替代 vast.ai RTX 4090（$0.6-0.9/hr）做 viser_gui_4d / playground / render.py 视觉对比；**A800 仍是 v3 训练主力**，ThinkPad 只做视觉验收，不跑训练。

**主机参数**：
- SSH alias `thinkpad`（user `yusun`），LAN IP 10.8.31.45
- Ubuntu 24.04，repo `/home/yusun/repo/3dgrut2/`
- 双 GPU：`cuda:0` = RTX 4090 24 GiB（torch FASTEST_FIRST 排第一，与 `nvidia-smi` 列顺序相反），`cuda:1` = Quadro T2000 Max-Q（不用）
- **必须 `CUDA_VISIBLE_DEVICES=0` 才走 4090**

**conda env 激活模板**（与 A800 模式对齐）：
```bash
ssh thinkpad 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate 3dgrut2 && cd ~/repo/3dgrut2 && CUDA_VISIBLE_DEVICES=0 <cmd>'
```
- env 名：`3dgrut2`（torch 2.11.0+cu128 / Python 3.11 / kaolin 0.17.0）
- `PYTHONNOUSERSITE=1` 已在 env config vars 永久锁定（防 `~/.local` site-packages 影子）
- gcc-11 / g++-11 已装 + 默认 selected（install_env.sh `WITH_GCC11=1` 路径）

**踩坑 tcnn pin（必读）**：
- `install_env.sh` 走 `git submodule update --init --recursive`，但 ThinkPad 是 **rsync mirror（无 .git）** → 改用手动 `git clone --recursive https://github.com/NVlabs/tiny-cuda-nn.git` 拉的是 **master HEAD**（如 2026-05-23 拿到 `749dd70`），比仓库期望 commit 新；
- 新版 tcnn `common.h:105` 强制要 `-DTCNN_HALF_PRECISION` define，而 `threedgut_tracer/setup_3dgut.py` 没传 → 首次 `viser_gui_4d` / `train.py` 触发 JIT 编译时 `lib3dgut_cc` fail：
  ```
  error: #error "TCNN_HALF_PRECISION is undefined. The build system must define this explicitly."
  ```
- **修复**（一次性，pin 到 Mac 仓库锁定的 commit）：
  ```bash
  ssh thinkpad 'cd ~/repo/3dgrut2/thirdparty/tiny-cuda-nn \
    && git checkout 075158a70b87dba8729188a9cadc9411cfa4b71d \
    && git submodule update --init --recursive \
    && rm -rf ~/.cache/torch_extensions/py311_cu128/lib3dgut_cc'
  ```
  cutlass / fmt 子模块也会同步回 `1eb6355` / `b0c8263`。pin 后首次 JIT 编译 lib3dgut_cc + libplayground_cc 共 ~1-2 分钟，编译产物缓存在 `~/.cache/torch_extensions/py311_cu128/`，后续启动秒级。

**ckpt 拷贝路径**（v2 inject 产物 → ThinkPad）：
- A800 inject 产物（`python -m threedgrut.viz.inject --ckpt <old> --dataset_path <manifest> --out /tmp/<new>`）写在 A800 `/tmp/`，**重启即丢**，inject 后必须立刻 `rsync` 到持久存储；
- Mac 本地 `/tmp/ckpt_with_ftheta.pt`（T8.13 inject 完 rsync 下来的 schema_v2 + FTheta，949 MB）是离线 fallback；
- **Mac → ThinkPad 直传**（不走 A800）：
  ```bash
  scp /tmp/ckpt_with_ftheta.pt thinkpad:~/work/ckpts/v2_ftheta/ckpt_with_viz_4d_v2.pt
  ```
  ThinkPad 直连不到 A800，A800 → ThinkPad 必须用 `scp -3 a800-x2:<path> thinkpad:<path>`（Mac 做 SSH stream relay，不落盘）。

**viser_gui_4d 启动 + Mac 隧道访问**：
```bash
# 1. ThinkPad 拉起 viser（4090, 端口 8090）
ssh thinkpad 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate 3dgrut2 \
  && cd ~/repo/3dgrut2 \
  && CUDA_VISIBLE_DEVICES=0 nohup python threedgrut_playground/viser_gui_4d.py \
       --gs_object ~/work/ckpts/v2_ftheta/ckpt_with_viz_4d_v2.pt \
       --port 8090 --target_fps 10 > /tmp/viser_4d.log 2>&1 & disown'

# 2. Mac 端 SSH tunnel（一行起，stay alive 直到手动 kill）
ssh -fN -L 8090:localhost:8090 thinkpad
# → 浏览器打开 http://localhost:8090
```
- ckpt schema_v2 + FTheta 时启动日志必有：`[T8.13] FTheta intrinsics 已加载 (resolution=(...), max_angle=...rad). GUI resolution slider 已锁定到训练分辨率。`
- ThinkPad 是 RTX 4090（有 RT cores），**不要**加 `--no_gaussian_render`（那是 A800 / A100 专用 bypass）；
- 关闭：`ssh thinkpad pkill -f viser_gui_4d` + Mac 端 `pkill -f "ssh.*8090"`。

**实测锚点（2026-05-23 v2 验收）**：T8.13 inject 产物 `ckpt_with_viz_4d_v2.pt`（schema_v2 + FTheta 8-key + resolution=(1920,1080) + max_angle=1.221rad + 31 tracks + 51 ego poses）在 ThinkPad 4090 跑 viser_gui_4d：lib3dgut_cc + libplayground_cc JIT 编译通过 → viser 0.0.0.0:8090 listening → 4090 GPU mem 2732 / 24564 MiB。tcnn pin 是唯一 thinkpad-only blocker。

---

## 7. NuRec 对齐表（移植自 v2_plan.md § 14）

> 本节是 v2_plan.md § 14.1-14.9 的 v3 对齐版。原文为 v3 任务种子，本节列出每个种子分配到的 v3 Stage 与状态。

### 7.1 § 14.1 辅助数据通道 → v3 Stage 11 / 14

| NuRec trick | v3 Stage | v3 任务 ID |
|---|:---:|---|
| DepthAnythingV2 度量深度 prior | 11 | T11.4 (V3-D1) |
| DINOv2 背景层 extra_signal | 13b | T13b.6 (V3-D2) |
| Mask2Former seg-logits aux CE loss | 14 | T14.1 (V3-D3) |
| 场景流 mask | 14 | T14.2 (V3-D4) |
| 交通灯 / 闪烁光源 mask | 14 | T14.3 (V3-D5) |
| Cuboid LiDAR padding | 13a | T13a.5 (V3-D6) |
| Cuboid camera padding | 13a | T13a.6 (V3-D7) |
| 相机 mask 30 iter dilation | 14 | T14.4 (V3-D8) |
| 帧 mask 10 iter dilation | 14 | T14.5 (V3-D9) |

### 7.2 § 14.2 多层场景分解 → v3 Stage 10 / 13a / 13b / 16

| NuRec trick | v3 Stage | v3 任务 ID |
|---|:---:|---|
| Background fourier_features_dim=5 | 13b | T13b.1 (V3-L1) |
| Road fourier_features_dim=1 | 13b | T13b.2 (V3-L2) |
| Road scale_pos_lr_by_scene_extent=false | 13b | T13b.3 (V3-L3) |
| Background ignore_classes_from_layers=[road] | 13a | T13a.1 (V3-L4) |
| DynamicRigid symmetric_axis='Y' | 13a | T13a.2 (V3-L5) |
| DynamicRigid 5000 pts/track + 300K | 13a | T13a.3 (V3-L6) |
| Track-pose 联合优化 | 13a | T13a.4 (V3-L7) |
| optimize_track_albedo | 13b | T13b.4 (V3-L8) |
| optimize_track_scale | 13b | T13b.5 (V3-L9) |
| DynamicDeformable hash-grid 形变场 | 16 (stretch) | T16.1-T16.4 (V4) |
| Sky envmap should_inpaint=true | 10 | T10.1 (V3-L10) |
| Sky envmap composite_in_linear_space=false | 10 | T10.2 (V3-L11) |
| Sky envmap min_grad_updates=1000 | 10 | T10.3 (V3-L12) |

### 7.3 § 14.3 训练策略 → v3 Stage 11 / 12 / 15

| NuRec trick | v3 Stage | v3 任务 ID |
|---|:---:|---|
| PERTURB move_outside_of_cuboid=false 简化版 | 12 | T12.7 (V3-T1.basic) |
| PERTURB cuboid clip 完整版 | 15 | T15.1 (V3-T1.full) |
| opacity_threshold=0.005 | 12 | T12.1 (V3-T2) |
| binom_n_max / noise_lr | 12 | T12.2 (V3-T3) |
| add/relocate 双阶上限 | 12 | T12.3 (V3-T4) |
| StepFunCosineAnnealingLR | 12 | T12.4 (V3-T5) |
| SequentialLR | 12 | T12.5 (V3-T6) |
| per-layer LR + γ=0.9998465 | 12 | T12.6 (V3-T7) |
| camera_rays=6144 + lidar_rays=2048 | 11 | T11.1 (V3-T8) |
| LiDAR ray supervision loss head | 11 | T11.2 (V3-T9) |

### 7.4 § 14.4 渲染管线 → v3 Stage 8.5 / 11 / 17

| NuRec trick | v3 Stage | v3 任务 ID |
|---|:---:|---|
| 3DGRUT k-buffer 复合 renderer | 17 (stretch) | T17.1 (V3-R1) |
| lidar_divergence=0.002 rad | 11 | T11.3 (V3-R2) |
| min_projected_ray_radius=0.5477 | 8.5 | T8.5.1 (V3-R3) |
| image_margin_factor=0.1 | 8.5 | T8.5.1 (V3-R4) |

### 7.5 § 14.5 后处理 → v3 Stage 9 / 14 / 15

| NuRec trick | v3 Stage | v3 任务 ID |
|---|:---:|---|
| Cosmos-DiFix 渐进蒸馏 | 15 | T15.2-T15.5 (V3-Cosmos) |
| 双边网格 1×1×1 + ExposureModel 修复（整合） | 9 | T9.1-T9.5 (V3-P1) |
| valid_pixel_mask 多源 + dilation | 14 | T14.6 (V3-P2) |

### 7.6 § 14.7 评估 → v3 Stage 11 / 15 / 17

| NuRec trick | v3 Stage | v3 任务 ID |
|---|:---:|---|
| val_lidar=true LiDAR domain PSNR | 11 | T11.5 (V3-E1) |
| per-class cPSNR / SSIM | 17 (stretch) | T17.2 (V3-E2) |
| 新视角扰动验证集 | 15 | T15.6 (V3-E3) |

### 7.7 § 14.8 优先级排序回归对照（novel-view 主目标重新校准 2026-05-22）

> 用户澄清：v3 主目标是 novel-view PSNR ≥ 30，不是 reconstructed view。重新按 **novel-view PSNR 贡献** 排序：

| 新 v3 优先级 | NuRec trick | v3 Stage | Novel-view 预期增益 | 与 v2 § 14.8 对比 |
|---:|---|---|---:|---|
| **1 ★** | **LiDAR ray 监督 + DepthAnythingV2 几何先验** | **11** | **+3.0** dB | 原 #2 → 升 #1（几何稳定性是 view extrapolation 不糊的根本） |
| **2 ★** | **Cosmos-DiFix 渐进蒸馏** | **15** | **+2.0** dB | 原 #1 → 降 #2（伪影修复但依赖前置 Stage 几何） |
| **3 ★** | **3DGRUT secondary ray + per-class cPSNR** | **17 ★** | **+2.0** dB | 原 #7/末（v3-R1 stretch）→ 升 #3 用户进取主线 |
| 4 | Track-pose 联合优化 + dynamic 改善 | 13a | +1.5 | 原 #4 维持 |
| 5 | Sky envmap inpaint + gamma | 10 | +1.5（主要 Sky novel 不爆黑洞） | 原 #3 维持 |
| 6 | DynamicDeformable hash-grid（行人/骑行 novel） | 16 (stretch) | +1.5 行人 region | 原 V4 → V3 stretch 维持 |
| 7 | symmetric_axis + per-track albedo/scale | 13b | +0.5 | 原 #5 拆 |
| 8 | MCMC PERTURB cuboid（防粒子飞出训练空间） | 15 (完整版) | +0.5（与 DiFix 协同） | 原 #8 维持 |
| 9 | 双边网格替换 affine（ExposureModel 修复） | 9 | +0.5（间接 — exposure 主要影响 reconstructed） | 原 #7 → 降级（novel 主线下优先级降低，但仍必做修复） |
| 10 | extra_signal DINOv2 + Fourier time | 13b | +0.3（间接） | 原 #6 → 降级 |
| 11 | mask 管线膨胀细节 | 14 | +0.2（LPIPS 主） | 原 #10 维持 |

**核心方向**：原 v2 § 14.8 优先级是按 reconstructed PSNR 提升排序的；v3 主目标改为 novel-view 后，**几何监督类（LiDAR + DepthV2 + DiFix + secondary ray）跃居 Top 3，应用类视觉细节修复（bilateral grid / DINOv2 / mask）降级**。

### 7.8 V4 backlog（v3 之后留给 V4 / NuRec 极限对齐的任务）

- **DynamicDeformable hash-grid 形变场（行人/骑行）**：若 Stage 16 stretch 未启动或失败，转 V4
- **NuRec 专有 DiFix 训练数据集复现**：v3 用开源版，v4 用专有版预期 +1-2 dB
- **跨 clip 大规模联训**：v3 单 clip，v4 多 clip 联训（10+ clips）
- **C++ tracer 大规模改动**：v3 仅 Python concat + 配置层 3DGRUT 复合，v4 改 OptiX kernel 加 layer-aware ray
- **NuRec 36.28 PSNR 极限对齐**：v3 进取目标 34.0 + DiFix 专有数据 + 跨 clip + tracer 改动 ≈ 36+
- **USDZ 打包 + Marching Cubes mesh 导出**：V1-5 / V1-6 独立工作包，与 v3 解耦
- **Cuboid 轨迹清洗 / 跨 clip 一致性 / 大型场景拼接**：v4 数据层增强
- **Sky envmap nvdiffrast cubemap 路径**（v2 Stage 5 用 MLP fallback）：若 nvdiffrast 在 A800 conda env 可用，v4 完成 NuRec 严格对齐

---

> **本计划状态**：v3 启动准备就绪。Stage 8.5 任务卡完整。等待用户触发 Stage 8.5 实际执行。
> **下一步**：用户触发后，按 Stage 8.5 → 9 → 10 → ... → 15（必达）→ 16/17（stretch）→ 18 顺序执行，每个 Stage 完成后按 CLAUDE.md 工作流同步 v3_plan.md（看板 / 任务级表 / Stage 状态 / Done Log）+ v3_architecture.md。
