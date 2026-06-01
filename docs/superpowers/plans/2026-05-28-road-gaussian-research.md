# Road 层车道线在大位移 / 大角度 novel-view 下模糊变形：根因分析 + 方案清单

**产出形态**：调研报告 + 排序后方案清单（不绑定 task 编号）
**时间窗口**：Phase 1 目标 1 周内 5k smoke 见效
**编写时间**：2026-05-28
**关联文档**：[v2_plan.md](../../repo/3dgrut2/v2_plan.md), [v3_plan.md](../../repo/3dgrut2/v3_plan.md), [v2_architecture.md](../../repo/3dgrut2/v2_architecture.md)

---

## 1. Context

3DGRUT2 项目采用分层高斯（LayeredGaussians: road / background / sky / dynamic）+ MCMC densification + BilateralGrid 色彩校正。V3 KPI 是 novel-view PSNR ≥ 28 dB（横移 ±1m/±2m、yaw ±5°/±10°）。当前 road 层车道线在大位移 / 大角度 novel-view 下出现**模糊（高频丢失）**与**变形（几何漂移）**双重劣化，cc_psnr_masked 与 novel-view LPIPS 之间存在显著 gap（v2 30k baseline novel-view LPIPS 0.6022, raw PSNR 14.37 dB）。

车道线是路面高对比度高频细长结构（典型宽度 10–15 cm），同时具有强方向性（沿车道纵向）。它对 view-dependent 颜色、densification 各向异性、平面约束三者同时敏感，是衡量 road 层质量的天然 stress test。

本研究目标：定位根因 → 列出 SOTA 候选 → 给出 1 周窗口下的推荐执行路线 + 后续升级路径。

---

## 2. 问题表象

| 扰动模式 | 表现 | 视觉特征 |
|---|---|---|
| `lateral_2m` | 车道线**横向位置漂移**，边界模糊 | 黑白边界羽化、白线弯曲 |
| `yaw_10deg` | 车道线**纵向断裂、变形** | 远端车道线"消失"或"分叉" |
| `lateral_1m + yaw_5deg` | 中等劣化，模糊主导 | 对比度下降、饱和度损失 |

cc_psnr_masked（重建视角）≈ 26 dB 而 novel-view raw PSNR ≈ 14 dB，gap 12 dB 表明问题不在"渲染保真度"而在"novel-view 外推鲁棒性"。

---

## 3. 根因分析（6 因）

证据来自三路 Explore agent 对源码的扫描。

### A. **View-Dependent SH 在窄训练相机锥下 overfit**（主因）

- **证据**：[threedgrut/model/model.py:142-166](../../repo/3dgrut2/threedgrut/model/model.py) road 层继承 `max_n_features=3`（3 阶 SH 全开）
- **机理**：训练相机为单车前向 5-cam ring（窄方位角覆盖），SH 系数在未观测视角方向上自由外推 → ±2m 横移即触发 extrapolation failure
- **车道线特异**：高对比黑白条纹 → SH 系数容差极小，外推稍偏即"灰化"
- **预期收益（降至 degree 1）**：novel-view PSNR +0.5–1.2 dB（参考 Spec-Gaussian / DropAnSH-GS）

### B. **MCMC perturb 无 XY 各向异性锁，细长高斯在平面内"胖化"**（次因）

- **证据**：
  - [threedgrut/layers/registry.py:27-32](../../repo/3dgrut2/threedgrut/layers/registry.py) road `perturb_scale_mask=(1, 1, 0)` 仅锁 Z
  - [threedgrut/strategy/mcmc.py:223-225](../../repo/3dgrut2/threedgrut/strategy/mcmc.py) perturb = `sigmoid(1-density) × noise_lr × LR × covariance @ randn` — 低 opacity 高斯获更大扰动
- **机理**：road 层 `scale_lr_mult=0.2` 衰减学习率但**初始 scale_prior=(0.1, 0.1, 0.001)** 是各向同性 XY，训练中 XY 缩放无上限 → 车道线高斯在平面内膨胀模糊
- **预期收益（加 anisotropy clamp / Effective Rank reg）**：+0.3–0.8 dB

### C. **缺平面 / 法向监督**（结构性）

- **证据**：[threedgrut/model/losses.py](../../repo/3dgrut2/threedgrut/model/losses.py), [threedgrut/layers/layered_loss.py:78](../../repo/3dgrut2/threedgrut/layers/layered_loss.py) 仅 L1+SSIM，无 normal/planar/depth smoothness
- **机理**：road 高斯 normal 应全部朝上（0, 0, 1）但 rotation 参数无显式约束 → 大 yaw 下 normal 漂移 → 车道线"翻面"
- **与 Stage 11 关系**：DepthAnythingV2 metric depth + LiDAR ray loss 已规划，但**法向 / planar loss 未规划**

### D. **无 mip-filter / LOD**（远端高频损失）

- **证据**：[configs/render/3dgut.yaml](../../repo/3dgrut2/configs/render/3dgut.yaml) 启用 `particle_kernel_degree=2`、`ut_kappa=0.0`，但无 frequency-domain filter
- **机理**：单高斯 footprint 在像素 < 1 时未做低通 → aliasing & 远端车道线"断点"
- **预期收益（加 Mip-Splatting filter）**：novel-view 远端 +0.3–0.7 dB

### E. **BilateralGrid 跨通道 affine 在高对比黑白上降饱和**（轻度）

- **证据**：[threedgrut/correction/bilateral_grid.py](../../repo/3dgrut2/threedgrut/correction/bilateral_grid.py) 3×4 per-camera affine 引入 RGB 混合
- **机理**：车道线 [0,0,0] vs [255,255,255] 经 cross-channel affine 后对比度下降；T9.3 已修 train/eval 一致但**未约束 affine 单调性**
- **预期收益（加 luminance-only 模式或 lane mask exclude）**：+0.1–0.3 dB（边际）

### F. **Road / Background 在地面 Z 重叠 → CUDA 排序在大 yaw 下 tie-breaking 翻转**（偶发）

- **证据**：[threedgrut/layers/layered_model.py:220-284](../../repo/3dgrut2/threedgrut/layers/layered_model.py) `_FusedView` concat 多层后交给 CUDA 排序；road/bg 都密集分布于地面附近
- **机理**：当两层高斯中心 Z 差异 < 排序数值精度时，大 yaw 视角下排序顺序翻转 → alpha 合成抖动
- **预期收益（深度 epsilon 隔离 / explicit layer order）**：+0.0–0.3 dB（仅极端 case）

---

## 4. SOTA 方案族（按相关度 × 改动量 排序）

### 4.1 平面化 Gaussian / Surfel（针对根因 B+C）

| 方案 | 核心思路 | 改动量 | 预期 ROI | 与现栈兼容 |
|---|---|---|---|---|
| **2DGS (Huang 2024)** | 用 2D 椭圆 surfel 替代 3D 椭球，天然贴面 | 高（替换 rasterizer 或 road 单层切换） | road 区 +1–2 dB | ⚠️ road 单层切换可行，需 layered_model 双 renderer |
| **PGSR (Chen 2024)** | 3DGS + 强各向异性 + 单视图法向监督 | 中 | +0.8–1.5 dB | ✅ 与 MCMC 兼容 |
| **AutoSplat flat-Gaussian** | road/sky 强制 flat (scale_z 上限 + scale_xy 上限) | 低 | +0.5–1.2 dB | ✅ 仅加 loss + clamp |
| **Effective Rank reg** | 防止针状高斯，所有 scale eigenvalue 比值 ≤ K | 低 | +0.3–0.8 dB | ✅ 一行 reg loss |

### 4.2 View-Dependent 失真消除（针对根因 A）

| 方案 | 核心思路 | 改动量 | 预期 ROI |
|---|---|---|---|
| **Road 层 SH 降至 degree 1** | 仅保留 DC + 3 个 linear coef | 低（一行配置） | +0.5–1.2 dB |
| **SH dropout (DropAnSH-GS)** | 训练中随机屏蔽高阶 SH coef | 低 | +0.3–0.7 dB |
| **Virtual view augmentation** | 训练 batch 合成 ±0.5m 中间相机做 unsupervised L1/perceptual | 中 | +0.5–1.5 dB |
| **Spec-Gaussian ASG** | 各向异性球形高斯替代 SH（处理高光） | 高 | +0.5–1 dB（湿滑路面） |

### 4.3 几何先验 / 法向约束（针对根因 C，与 Stage 11 合流）

| 方案 | 核心思路 | 改动量 | 预期 ROI |
|---|---|---|---|
| **DepthAnythingV2 法向监督** | mono depth → normal → road 高斯 normal 监督 | 中（Stage 11 已规划 depth） | +0.5–1 dB |
| **LiDAR plane fitting prior** | LiDAR road points 拟合局部 plane → 高斯法向监督 | 中（Stage 11 已规划 LiDAR） | +0.5–1.2 dB |
| **DN-Splatter 几何一致 loss** | depth + normal 联合监督 + edge-aware smoothness | 中 | +0.8–1.5 dB |
| **TCLC-GS / LiHi-GS** | LiDAR ray supervision 的 SOTA 配方 | 中 | +1–2 dB |

### 4.4 Densification 优化（针对根因 B）

| 方案 | 核心思路 | 改动量 | 预期 ROI |
|---|---|---|---|
| **Anisotropy clamp** | road scale 最大 eigenvalue / 最小 ≤ ratio_max | 低 | +0.3–0.6 dB |
| **Scale upper bound** | road xy scale ≤ 0.3m, z ≤ 0.05m | 低 | +0.2–0.5 dB |
| **MCMC noise 各向异性 mask** | `perturb_scale_mask` 改为 (0.5, 0.5, 0) | 低 | +0.2–0.4 dB |

### 4.5 Anti-aliasing（针对根因 D）

| 方案 | 核心思路 | 改动量 | 预期 ROI |
|---|---|---|---|
| **Mip-Splatting filter** | 3D smoothing filter + 2D mip filter | 中（CUDA kernel 改动，gsplat 已集成） | +0.3–0.7 dB |
| **Multi-resolution training** | 多尺度训练正则化高频 | 低 | +0.2–0.5 dB |

### 4.6 自动驾驶场景专门工作（参考）

- **AutoSplat** (Khan 2024)：显式 flat road/sky Gaussian，本报告 4.1 已吸收
- **DHGS** (2024)：SDF 约束路面，复杂
- **MTGS** (2024)：multi-traversal supervision，依赖多次采集
- **Para-Lane** (benchmark)：横移 PSNR 专用 benchmark
- **StreetGaussians / OmniRe**：dynamic 处理优秀，road 处理一般
- **RoGS** (2024)：road-specific Gaussian，与本项目分层思路相似可借鉴

---

## 5. 推荐路线（Phase 1：1 周窗口）

**目标**：5k smoke 上 novel-view raw PSNR 从 14.4 dB 提升到 **16.5+ dB**（+2.0 dB），cc_psnr_masked 不降（≥ 24 dB 守护线）。

### Phase 1 方案组合（按引入顺序，每步独立 5k smoke 验证）

| Step | 方案 | 根因 | 改动量 | 预期增益 |
|---|---|---|---|---|
| **P1.1** | Road 层 SH degree: 3 → 1 | A | 1 行 config | +0.5–1.2 dB |
| **P1.2** | Anisotropy clamp + scale upper bound (road) | B | ~30 行（loss + clamp hook） | +0.3–0.8 dB |
| **P1.3** | Effective Rank reg (road only) | B | ~20 行（reg loss） | +0.2–0.5 dB |
| **P1.4** | Virtual view augmentation（±0.5m 横移合成 L1） | A | ~80 行（dataloader + loss） | +0.5–1.5 dB |
| **合计** | | | | **+1.5–4.0 dB** |

**为什么这套组合**：
- 全部为 **loss / config / 超参** 级改动，零 rasterizer 修改 → 与 1 周窗口匹配
- 覆盖主因 A + 次因 B，预期累积收益足以达到 Phase 1 目标
- 每步可独立 ablation，失败可回滚

### Phase 1 不做但保留作 Phase 2 候选

- Mip-Splatting filter（CUDA 改动，留待 Stage 11 后）
- BilateralGrid 改 luminance-only（仅 +0.1–0.3 dB，性价比低）
- Layer depth epsilon 隔离（偶发问题，先观测再说）

---

## 6. Phase 2 候选（2-4 周窗口）

**触发条件**：Phase 1 达标但 novel-view PSNR 仍 < 22 dB，或视觉上车道线变形未根本解决。

### Phase 2A：几何监督合流（与 v3_plan Stage 11 对齐）

- **2A.1** DepthAnythingV2 法向监督（road 高斯 normal → ground plane up）
- **2A.2** LiDAR plane fitting prior（局部 plane fit → normal 监督）
- **2A.3** DN-Splatter edge-aware smoothness loss
- 预期 +1–2 dB

### Phase 2B：Road 层架构升级（surfel）

- **2B.1** Road 单层切换 2DGS surfel（其他层保 3DGS）
- 需要：`layered_model.py` 支持双 renderer / 双 rasterizer 调度
- 与 2A 法向监督天然耦合（surfel 法向是显式参数）
- 预期 road 区 +1–2 dB（叠加 2A 后）
- **改动评估**：~500-800 行 + 1-2 周
- **风险**：与 MCMC perturb 兼容性需重新验证

### Phase 2C：Mip-Splatting 抗锯齿

- 改 3DGRUT CUDA 内核或切到 gsplat 后端
- 与 2A/2B 正交，可独立叠加
- 预期 +0.3–0.7 dB

---

## 7. 与 v3_plan.md 现有规划的关系

| v3_plan 现有 task | 本报告对应 | 是否冲突 |
|---|---|---|
| V3-P1.a/b/c (T9.1-9.3) BilateralGrid | 已完成，本报告根因 E 涉及 | 无冲突 |
| Stage 11 LiDAR + DepthAnythingV2 (+3.0 dB) | Phase 2A 几何监督 | ✅ 合流，本报告补充法向监督维度 |
| Stage 13a (T13a.2 symmetric_axis) | 可用于车道线中心线约束 | 可与 Phase 1 P1.2 联动 |
| Stage 15 Cosmos-DiFix (+2.0 dB) | 视频扩散修复，与本报告正交 | 无冲突 |
| Stage 17 3DGRUT secondary ray (+2.0 dB) | 渲染层改进，与本报告正交 | 无冲突 |

**建议**：Phase 1（本报告 §5）作为 **Stage 9 → Stage 11 之间的快速胜利**插入，task 编号建议 V3-R1.a/b/c/d（R for Road），不挤占 Stage 11 LiDAR/depth 任务序列。

---

## 8. 验证方法

### 8.1 单测层验证（每步 P1.* 改动落地必跑）

```bash
cd /Users/etendue/repo/3dgrut2 && source .venv/bin/activate
pytest threedgrut/tests/ -k "novel_view or road or layer" -v
```

预期：现有 58 个回归 test 全过 + 新增针对各步改动的回归 test。

### 8.2 Mac 1k smoke（每步 P1.* 改动落地）

```bash
python train.py --config-name apps/ncore_3dgut_mcmc_multilayer \
  n_iterations=1000 \
  path=<test_clip>/pai_<clip>.json \
  trainer.sky_backend=mlp \
  experiment_name=v3r1_p1_<step>_1k_smoke
```

观察：训练 loss 收敛、ckpt 写出、reload parity（T9.1 已建立的 parity 标准）。

### 8.3 A800 5k smoke + novel-view eval（每步 P1.* 改动 ✅ 必要）

```bash
ssh a800-x2 'export PATH=/root/miniforge3/envs/3dgrut/bin:$PATH \
  && export CUDA_VISIBLE_DEVICES=0 \
  && cd /root/work/yusun/repo/3dgrut \
  && python train.py --config-name apps/ncore_3dgut_mcmc_multilayer \
    n_iterations=5000 \
    path=<clip>/pai_<clip>.json \
    trainer.sky_backend=mlp \
    experiment_name=v3r1_p1_<step>_5k_a800 \
  && python render.py --checkpoint=<ckpt> --novel_view=true'
```

### 8.4 KPI 验收矩阵（每步 5k smoke 后回报）

| Metric | Baseline (T9.3) | P1.1 target | P1.2 target | P1.3 target | P1.4 target | Phase 1 exit |
|---|---|---|---|---|---|---|
| novel-view raw PSNR | 14.4 dB | ≥ 14.9 | ≥ 15.2 | ≥ 15.4 | ≥ 16.0 | **≥ 16.5** |
| novel-view LPIPS | 0.602 | ≤ 0.59 | ≤ 0.58 | ≤ 0.57 | ≤ 0.55 | **≤ 0.55** |
| cc_psnr_masked (守护) | 26.0 dB | ≥ 24.0 | ≥ 24.0 | ≥ 24.0 | ≥ 24.0 | **≥ 24.0** |
| road 高斯数 | 200K | ~200K | ~190K | ~180K | ~180K | 不显著变化 |

**回滚条件**：任一指标低于守护线（cc_psnr_masked < 24.0 dB）或 novel-view LPIPS 升高 → 当步骤回滚，分析原因。

### 8.5 视觉验证（必做）

- 横移 lateral_2m / yaw_10deg 的渲染图与 GT 对比
- 车道线放大区域（中心 + 远端）人工对比
- 视频对比 baseline vs Phase 1 final

---

## 8.6 Nvidia Nurec 路面层参数对照（外部实证）

Nvidia Nurec pipeline 对 road 层用一组"扁平定向圆盘"参数实证了"亚像素级车道线清晰度"的关键设计——把所有车道线 / 井盖 / 路面补丁压扁到接近 2D 平面。3dgrut2 现状已经匹配 6/8 个参数：

| Nurec 设计 | 3dgrut2 现状 | 状态 | 备注 |
|---|---|---|---|
| `default_scale=(0.1, 0.1, 0.001)` | `LayerSpec.scale_prior=(0.1, 0.1, 0.001)` | ✅ | [registry.py:29](/Users/etendue/repo/3dgrut2/threedgrut/layers/registry.py) |
| `num_points=200,000` | `max_n_particles=200_000` | ✅ | 同上 |
| `class_labels=[road]` | `mask_field="road_mask"` | ✅ | 5-tier sseg mask |
| `scale LR 1e-3` (bg 5e-3) | `scale_lr_mult=0.2` × 全局 `scale.lr=0.005` = 1e-3 | ✅ 等价 | [registry.py:29](/Users/etendue/repo/3dgrut2/threedgrut/layers/registry.py) |
| `z_offset=0` | `perturb_scale_mask=(1, 1, 0)` 锁 Z | ✅ 更强 | T3.4 D1 |
| **`fourier_features_dim=1`** (bg=5) | **SH degree=3 全开** | ❌ **P1.1** | 等价问题：road view-dep 表达力过高 |
| **训练期 scale 保持扁平** | **无 scale clamp** | ❌ **P1.2** | 初始扁但 MCMC 训练中会"胖化" |
| `scale_pos_lr_by_scene_extent=false` | (3DGRUT 默认无 scene-extent 缩放) | n/a | — |

**核心解读**：
- Nurec 的 `fourier_features_dim=1` 与 3dgrut2 SH degree 概念等价（都是 view-direction encoding 维度）。Nurec 实测 road 层用 1 维 Fourier features 已足，与 background 5 维形成 5x 差距 → 直接支持本报告 **P1.1（road SH 3 → 1）** 的方向选择。
- Nurec 文档原话："**每个高斯被压扁成一个贴地的扁平定向圆盘。这是车道线、井盖、路面补丁在新视角下保持锐利的根本原因 —— 体积型粒子无法在地表实现亚像素级清晰度。**"——这强烈支持本报告 **P1.2（XY/Z scale clamp + anisotropy ratio）** 的必要性：初始扁盘不够，必须有运行期约束防止 MCMC 训练把 disc 胖化为体积粒子。
- 3dgrut2 已经无意识地吸收了 Nurec 6/8 个设计，Phase 1 的 P1.1 + P1.2 正是补齐剩余 2 个缺口；Phase 2 的几何监督（DepthAnythingV2 + LiDAR plane prior，对应 v3_plan Stage 11）则与 Nurec depth supervision 思路同源。

**对 Phase 1 优先级的影响**：P1.1 + P1.2 从"理论推断"升级为"外部实证强背书"，应作为 Phase 1 不可省略的两步（不可只挑其中一个）。

## 8.7 与 v3_plan Stage 11 深度监督的关系

v3_plan.md 当前规划的 T11.1–T11.6（Stage 11 LiDAR + DepthAnythingV2）：

| Task | 内容 | 状态 |
|---|---|---|
| T11.1 | V3-T8 trainer 每步 6144 cam ray + 2048 LiDAR ray batch (1:1) | ⬜ Todo |
| T11.2 | V3-T9 LiDAR depth/intensity ray loss head (NuRec 主训练同等权重) | ⬜ Todo |
| T11.3 | V3-R2 lidar_divergence=0.002 rad cone 抗锯齿 | ⬜ Todo |
| **T11.4** | **V3-D1 DepthAnythingV2 metric depth prior reader + depth loss head** | ⬜ **Todo** |
| T11.5 | V3-E1 val_lidar=true — LiDAR PSNR 独立报告 | ⬜ Todo |
| T11.6 | A800 Stage 11 出口 — cc_psnr_masked ≥ 28.0 dB | ⬜ Todo |

**所有 Stage 11 任务目前都是 ⬜ Todo**（kanban "In Progress 🟡 = 0"），还未启动。

**关系定位**：
- 本报告 **Phase 1（V3-R1.a–d）** 是 Stage 9 → Stage 11 之间的"快速胜利"插入，不依赖 LiDAR / DepthAnythingV2，预期 +1.5–4.0 dB
- 本报告 **Phase 2A** = Stage 11（T11.1–T11.6）的 **road 法向监督扩展**，目标在 Stage 11 LiDAR depth + DepthAnythingV2 已经接入后，**复用同一个 depth reader / loss head**，额外加一条 road-normal supervision（mono depth → normal → 约束 road 高斯 rotation）
- 因此 Phase 1 + Stage 11 的执行顺序：**Phase 1 → Stage 11 (T11.1–T11.6) → Phase 2A road-normal 扩展**。Phase 1 不能等 Stage 11，因为 Phase 1 是不依赖 LiDAR/DepthV2 的 loss/clamp 级改动，1 周窗口可独立验证；Stage 11 是 2-3 周窗口的几何监督底座。

## 9. 关键参考文献

| 主题 | 论文 | 备注 |
|---|---|---|
| Surfel / 平面化 | Huang et al. 2024, **2D Gaussian Splatting**, SIGGRAPH | road 单层切换候选 |
| Anti-aliasing | Yu et al. 2024, **Mip-Splatting**, CVPR | Phase 2C |
| Planar prior | Chen et al. 2024, **PGSR** | 中间路线 |
| 自动驾驶 road | Khan et al. 2024, **AutoSplat** | Phase 1 P1.2 思路源 |
| 法向 + 深度 | Turkulainen et al. 2024, **DN-Splatter** | Phase 2A |
| LiDAR 监督 | **TCLC-GS** / **LiHi-GS** 2024 | Stage 11 合流 |
| Road specific | **RoGS** 2024 | 类似分层思路 |
| Anisotropy | **Effective Rank reg** in GaussianSplat papers | Phase 1 P1.3 |
| SH dropout | **DropAnSH-GS** | Phase 1 P1.1 备选 |
| Benchmark | **Para-Lane** | 横移 PSNR 专用 |

---

## 10. 下一步行动建议

本调研产出为 **report-only**，不直接落代码。建议你阅读后：

1. **确认 Phase 1 推荐路线**（§5）的 4 步是否符合预期
2. 若同意，下一会话用 `/superpowers:writing-plans` 把 Phase 1 拆成具体 task（V3-R1.a 到 V3-R1.d），接入 v3_plan.md kanban
3. Phase 2 视 Phase 1 结果决定是否启动 surfel 改动
4. 与 Stage 11 (LiDAR + DepthAnythingV2) 的合流时机：建议 Phase 1 完成后立刻插入 Stage 11，因 Phase 2A 几何监督可与 Stage 11 共享 reader / loader 代码

---

**附：本报告生成过程**
- 3 路 Explore agent 并行扫描 [threedgrut/layers/](../../repo/3dgrut2/threedgrut/layers/), [threedgrut/strategy/](../../repo/3dgrut2/threedgrut/strategy/), [threedgrut/model/](../../repo/3dgrut2/threedgrut/model/), [threedgrut/utils/novel_view.py](../../repo/3dgrut2/threedgrut/utils/novel_view.py), [configs/](../../repo/3dgrut2/configs/), [v3_plan.md](../../repo/3dgrut2/v3_plan.md)
- 1 路 general-purpose agent 通过 WebSearch 调研 2DGS / Mip-Splatting / PGSR / AutoSplat / DN-Splatter 等
- 用户在 plan mode 通过 AskUserQuestion 确认了产出形态 / SH 取舍 / 改动范围 / 时间窗口
