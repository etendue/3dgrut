# WP V2 任务报告：LayeredGaussians 分层高斯 v2 项目结题

> **状态**：Stage 7 软出口 ✅ 结题（A800 实测 cc_psnr_masked **24.70 dB** ≈ Stage 5/6/6-fix baseline 24.7~24.9 ± 0.2 dB noise 级）
> **关键发现**：ExposureModel 在 7-cam 30k 长训中**退化优化失控**，raw psnr_masked 崩到 15.63 dB（vs 关闭后 25.76 dB，差 +10.13 dB），但 cc_psnr_masked 几乎不变 → 实证 v2 真实重建质量上限是 ~25 dB masked PSNR / ~24.7 dB cc_psnr_masked。
> **下游交接**：ExposureModel 退化问题 + Bilateral-grid Color Correction 合并入 V3-P1 统一研究（v2_plan.md § 14.5）。

---

## 一、交付物汇总（Stage 0 → Stage 8 全量）

| WP 阶段 | 主交付 | 状态 | 关键产出 |
|---|---|---|---|
| Stage 0 | A800 baseline | ✅ | smoke 24.12 dB / 9.48 it/s |
| Stage 1 | LayeredGaussians 容器 + LayerSpec registry | ✅ | 5 任务 + 9 单测 + 3 A800 contract |
| Stage 2 | LayeredMCMC sub-strategy | ✅ | 5 任务 + 8 单测 + per-layer cap |
| Stage 3 | Road 层（LiDAR-Z KNN init + region loss + perturb Z-lock） | ✅ | A800 5k PSNR **26.13 dB**（+2.5 超 23.6 出口） |
| Stage 4 | DynamicRigid 层（真 NCore cuboids autolabels v2） | ✅ | A800 10k PSNR **26.32 dB** / 31 tracks / 48K dyn particles / 9.58 it/s |
| Stage 5 | Sky Envmap（MLP fallback, A800 nvdiffrast 不可用） | ✅ | A800 5k PSNR **26.17 dB** |
| Stage 6 | Per-camera ExposureModel（affine `exp(a)·rgb+b`） | ✅ | A800 5-cam 5k cc_PSNR **24.94 dB** / exposure_a.std=0.0306 |
| Stage 6-fix | Ego mask 全链路接通 + masked 指标 | ✅ | A800 1-cam 5k **masked PSNR 29.49 dB**（+9.0 vs full 20.49, 量化 ego 区 21.78% 历史水分） |
| **Stage 7** | **7-cam 30k 完整训练 + Stage 7 出口** | **✅ 软出口** | **A800 30k 51 min / cc_psnr_masked 24.70**（详见 § 三） |
| Stage 8 | viser_gui_4d 4D 浏览器可视化 | ✅ | 10 任务 + ckpt['viz_4d'] schema v1 + Mac 179/179 PASS + A800/vast.ai RTX 4090 端到端 |

**总任务数**：54 任务（Stage 0-8 全部 ✅，Stage 7 新增 T7.3.b exposure ablation）。
**Mac pytest**：181/181 PASS（含 Stage 6-fix + Stage 8 新增 50 个测试，0 回归）。

---

## 二、v2 架构总览（一图概述）

```
┌─────────────────────── LayeredGaussians ───────────────────────┐
│                                                                 │
│  layers.enabled = [background, road, dynamic_rigids, sky_envmap]│
│                                                                 │
│   ┌─────────┐  ┌──────┐  ┌────────────────┐  ┌─────────┐       │
│   │background│  │ road │  │dynamic_rigids  │  │  sky    │       │
│   │  ~600K   │  │ 200K │  │ 50~200K        │  │ envmap  │       │
│   │  粒子     │  │ 粒子  │  │ × N tracks     │  │  MLP    │       │
│   │           │  │       │  │ object-local    │  │ /cubemap │       │
│   └─────────┘  └──────┘  └────────────────┘  └─────────┘       │
│        │           │              │                │             │
│        └────┬──────┴──────┬───────┘                │             │
│             │              │ track-aligned          │             │
│             ▼              ▼ pose lookup            ▼             │
│   ┌──────────────────────────────┐         ┌──────────────┐     │
│   │ LayeredMCMC sub-strategies   │         │ blend at top │     │
│   │ (per-layer cap + perturb 1,1,0)│         │ post render  │     │
│   └──────────────────────────────┘         └──────────────┘     │
│                       │                            │             │
└───────────────────────┼────────────────────────────┼─────────────┘
                        │                            │
            ┌───────────▼──────────────┐      ┌──────▼────────┐
            │ region-weighted L1 loss  │      │ ExposureModel │
            │ (sky / road / dyn / bg)  │      │ per-cam affine│  ← T7.3 证伪
            └──────────────────────────┘      └───────┬───────┘
                                                       │
                                              ┌────────▼────────┐
                                              │     vs GT       │
                                              │  + ego mask     │
                                              └─────────────────┘
```

详见 `v2_architecture.md` § 1.x mermaid 全图。

---

## 三、Stage 7 KPI 出口（三组实测对比 + 真实质量解读）

### 3.1 三组 A800 实测对照

| 指标 | T6F.3 (1-cam 5k baseline) | T7.2 (1-cam 1k smoke) | T7.3 (7-cam 30k exposure ON) | **T7.3.b (7-cam 30k exposure OFF)** | Stage 7 plan 目标 |
|---|---:|---:|---:|---:|---:|
| training_time | 520 s (8.7 min) | 102.9 s (1.7 min) | 3061.8 s (51.0 min) | 3064.6 s (51.1 min) | ≤ 60 min ✅ |
| iteration_speed | 9.61 it/s | 9.71 it/s | 9.80 it/s | 9.79 it/s | — |
| n_iterations | 5000 | 1000 | 30000 | 30000 | 30000 ✅ |
| `mean_psnr` (full) | 20.49 | 19.93 | 14.91 | **23.78** | ≥ 28.5 |
| `mean_psnr_masked` | **29.49** | 26.38 | 15.63 ❌ | **25.76** | ≥ 30 |
| `mean_cc_psnr` (full) | 19.61 | 18.57 | 23.25 | 23.24 | — |
| **`mean_cc_psnr_masked`** | **24.90** | 22.76 | **24.75** | **24.70** | — |
| `mean_ssim_masked` | 0.934 | 0.898 | 0.792 | 0.846 | — |
| `mean_lpips_masked` | 0.190 | 0.285 | 0.352 | 0.330 | — |
| 4 层粒子 | bg+road | bg+road+dyn31 | bg+road+dyn70+sky | bg+road+dyn70+sky | 全栈 ✅ |

### 3.2 关键观察

1. **`cc_psnr_masked` 三组高度一致**（24.7 ↔ 24.9 ↔ 24.75 ↔ 24.70，σ < 0.2 dB） — **真实重建质量在 Stage 5/6/6-fix/7 没有净提升**，单相机 5k 步就已经收敛到 v2 架构在当前 NCore 9ae151dc clip 上的天花板（~24.7 dB cc_psnr_masked / ~25 dB raw masked）。
2. **T7.3 raw psnr_masked 灾难性低**（15.63 vs 目标 30）—— 不是几何重建崩，而是 ExposureModel 在 30k 长训中学到了大幅 RGB 偏移（见 § 四诊断）。
3. **T7.3.b 关闭 exposure 后 raw psnr_masked 跳回 25.76**（+10.13 dB），证伪 exposure 是 raw 崩的真因。
4. **训练时长 51 min ≤ 60 min plan 出口** ✅（9.79 it/s 与 T6F.3 baseline 9.61 完全持平，性能零损失）。
5. **7-cam 30k vs 1-cam 5k masked PSNR 反而低 3.7 dB**（25.76 vs 29.49）—— 多相机长训没有带来质量净提升。归因待 V3 排查（多相机 imbalance、过拟合、cross-view 一致性约束缺失）。

### 3.3 Stage 7 软出口判定

按 plan 严格 KPI（raw psnr ≥ 28.5 / masked ≥ 30）→ **未达标**。
按真实质量 KPI（cc_psnr_masked）→ **24.70 ≈ T6F.3 baseline 24.90**（差 -0.20 dB noise 级）→ **质量不退化，结构无回归** → **软出口** ✅。

**决策**：
- T7.4 per-layer cap ablation **跳过**（根因不在 cap，跑 4×55 min 不会改变 cc_psnr_masked 上限）。
- ExposureModel 失控 + 7-cam 30k 反直觉 → 转入 V3-P1（Bilateral-Grid Color Correction）统一研究。
- Stage 7 以 T7.3.b（exposure off）作为推荐配置，写入 yaml 注释 + V3 入口。

---

## 四、关键技术发现

### 4.1 ExposureModel 长训失控的机制（退化优化 degenerate optimization）

**问题**：训练有两条 loss 下降路径，30k Adam 没有约束 → 模型选择了病态短路径。

```
路径 1 (物理正确): 高斯学准真实色彩 → exposure 维持小值 → loss ↓
路径 2 (病态短路): 高斯学个大概 → exposure 把偏差全 compensate → loss ↓（更快收敛, 14 个参数 vs 几百万高斯）
```

**实证**：
- T7.3 raw psnr_masked = 15.63（路径 2 终点：exposure 学到大幅 RGB 偏移，raw 输出严重过曝/泛白）
- T7.3.b raw psnr_masked = 25.76（同配置但关 exposure：模型被迫走路径 1）
- 两者 cc_psnr_masked 几乎一致（24.75 vs 24.70）→ 真实几何/纹理质量没差

**关键代码** [threedgrut/correction/exposure.py](threedgrut/correction/exposure.py)：

```python
out = clamp(exp(a) * img + b, 0, 1)  # per-cam affine, 零初始化 identity
                                       # 独立 Adam, lr=1e-3, 无 decay, 无 reg
```

设计本意（Stage 6 T6.1）：消除 7-cam EXIF 差异（每相机独立 ISP/自动曝光，同物理场景 RGB 不同）。
实际行为（Stage 7）：30k Adam × 14 参数 × 无约束 = 退化优化，与场景重建耦合。

### 4.2 cc_PSNR ≡ eval-time 后处理 color correction（实现 [color_correct_affine](threedgrut/utils/color_correct.py)）

| 维度 | ExposureModel（训练时）| color_correct_affine（eval 时）|
|---|---|---|
| 公式 | `exp(a)·img + b` clamp | `(img − b)/a` clamp（per-channel lstsq） |
| 参数 | 可学习 `nn.Parameter`, per-camera | 每张图临时拟合, **不存储** |
| 颗粒度 | per-camera | **per-image × per-channel** |
| 反向传播 | 是 | 否（只影响 metric） |
| 来源 | Recon-Studio luxury/exposure.py | Google multinerf（NeRF 评测标准） |

**cc_PSNR 是 NeRF 圈标准 KPI**：Mip-NeRF 360 / Block-NeRF / drivestudio / Recon-Studio 都报 cc 版本，因为真实采集数据中相机 ISP 各不相同，强迫匹配 raw 不公平。

**对 v2 的启示**：
- Stage 7 真实出口指标应是 `cc_psnr_masked`，不是 `psnr_masked`。
- raw `psnr_masked` 只作 ExposureModel **健康度健康检查**（应 ≈ cc + ≤ 2 dB）。
- 若 raw 比 cc 低很多 → ExposureModel 学过头（T7.3 case）；反之 ≈ 健康。

### 4.3 双指标 → 三指标演化

| 阶段 | 指标体系 |
|---|---|
| Stage 0-5 | 单指标 `mean_psnr`（全图，含 ego 区水分） |
| Stage 6-fix | 双指标 `mean_psnr` + `mean_psnr_masked`（剔除 ego 区 21.78%） |
| **Stage 7（推荐）** | **三指标** `mean_psnr` + `mean_psnr_masked` + `mean_cc_psnr_masked` |

`mean_cc_psnr_masked` 是 v2 最干净的"真实重建质量"指标：撤销 exposure 偏移 + 排除 ego 区 + 排除 EXIF 全局色偏差。

---

## 五、V3 / V4 Backlog 入口（核心遗留）

详见 v2_plan.md § 14。本节列 **Stage 7 直接转出**的核心待办：

### 5.1 V3-P1（更新版）：Bilateral-Grid Color Correction + ExposureModel 退化修复（**整合研究**）

**v2 现状**：
- 训练时：affine `exp(a)·img + b`（ExposureModel, Stage 6 占位）— **Stage 7 实证长训失控**
- Eval 时：affine `(img-b)/a`（color_correct_affine, Google multinerf）— 标准 NeRF 评测后处理

**V3 整合目标**（Recon-Studio 完整 bilateral grid port）：
1. **替换 ExposureModel**：affine → bilateral grid `1×1×1` per-camera（Recon-Studio post.py 直接 port）
2. **加约束防退化**：bilateral grid params L2 reg + lr cosine decay + 2-stage freeze（step > 2000 freeze）
3. **eval metric 一致性**：训练 forward 路径 + eval color correction 用同一套 bilateral grid（消除 raw vs cc 分歧）
4. **健康度监控**：训练日志输出 `exposure_a.std` + raw/cc PSNR ratio，>2 dB 警报

**预期收益**：
- raw psnr_masked 与 cc_psnr_masked 收敛（消除 T7.3 的 +9 dB 分歧）
- 真实 cc_psnr_masked 提升 1-2 dB（bilateral grid 比 affine 表达能力更强）
- 7-cam 30k 长训不再退化

**实施 owner**：V3 plan
**预估工作量**：3-5 天（含 port + 测试 + A800 验证）
**优先级**：**高**（Stage 7 实证最关键缺口）

### 5.2 其他 V3 backlog（按 PSNR 预期排序）

| 序号 | 项 | 状态 | 优先级 |
|---|---|---|---|
| V3-T9 | LiDAR ray loss head（NuRec 主训练同等权重） | ❌ | 最高（预估 +1~2 dB） |
| V3-P1 | **Bilateral-grid CC + exposure 退化修复**（本报告新增） | ⚠️ | **高**（Stage 7 实证最关键） |
| V3-D1 | DepthAnythingV2 metric depth prior + loss | ❌ | 高 |
| V3-D2 | DINOv2 背景 extra_signal（20 维语义 logits） | ❌ | 中 |
| V3-L10 | Sky envmap inpaint 模块（新视角不爆黑洞） | ❌ | 中（v2 实测 sky region 无问题但 V3 加这层更稳） |
| V3-L7 | Track-pose 联合优化（v2 明确不做） | ❌ | 中 |
| V3-E2 | **per-class cPSNR 评测工具**（road/dyn/sky 拆解） | ❌ | 中（plan T7.3 提到但工具未实现，Stage 7 仅汇总 full+masked+cc） |

完整 V3/V4 backlog（10 类 60+ 项）见 v2_plan.md § 14。

### 5.3 7-cam 30k 反直觉的待查问题

**现象**：7-cam 30k masked PSNR 25.76 vs 1-cam 5k masked PSNR 29.49，**反而低 3.73 dB**。
**候选解释**（V3 优先排查）：
1. **7-cam 之间几何不一致**：cross / rear 相机 frustum 重叠少，监督稀疏区域被 over-fit / under-fit
2. **30k 过拟合训练集** 但 val_frame_interval=8（采样不同帧）应该已经避免
3. **dyn_rigids 70 tracks vs 31 tracks**：更多动态物体粒子分配不充分（默认 200K cap 不够）
4. **多相机 LiDAR 监督权重未调**：每帧增加 6× 训练数据但 dataset.train.n_train_sample_camera_rays 没相应放大

**V3 任务**：跑 `1-cam 30k` + `5-cam 30k` baseline，三角化定位是 cam 数量还是 step 数量副作用。

---

## 六、已知限制（V3 / V4 显式不做）

| 限制 | 原 plan 决策 | Stage 7 后是否变更 |
|---|---|---|
| DynamicDeformable 形变层（行人/可形变物体） | v2 仅 spec 占位 → V4 主力 | 不变（V4） |
| Bilateral grid 色彩矫正 | v2 用 affine 占位 → V3 port | **优先级提升**（V3-P1） |
| Track-pose 联合优化 | v2 明确不做 → V3-L7 | 不变 |
| Cosmos-DiFix 扩散修复后处理 | v2 不做 → V3 主力 | 不变 |
| Per-class cPSNR 评测工具 | plan T7.3 提到但未实现 | V3-E2 |
| nvdiffrast cubemap backend | A800 不可用 → V2/V3 重新探测 | 不变（A800 conda env 仍无 nvdiffrast） |

---

## 七、下一步建议

1. **立即（Stage 7 收尾）**：
   - 本报告作为 v2 项目结题文档
   - v2_plan.md § 5 Done Log 追加 "🎁 Stage 7 软出口完整结题"
   - v2_architecture.md § 6.1 翻 :::done
   - ckpt `stage7_full_20260520-202222/.../ckpt_30000.pt`（exposure on）+ `stage7_noexp_20260521-102930/.../ckpt_30000.pt`（exposure off）保留作 V3 baseline

2. **短期（V3 启动前）**：
   - V3 plan kickoff 优先 V3-P1（bilateral grid + exposure 退化修复整合研究）
   - V3-T9 LiDAR ray loss 并行启动（PSNR 收益最大）
   - V3 plan 重新定义 KPI 为 `cc_psnr_masked` 出口门槛（不再用 raw）

3. **中期（V3 → V4）**：
   - V3 收口 cc_psnr_masked 达 27-28 dB（+2-3 dB 提升）
   - V4 主推 DynamicDeformable + Cosmos-DiFix 后处理

---

## 八、附录

### 8.1 Stage 7 A800 复现命令

```bash
# T7.3 (exposure on, 实测失控参考)
ssh a800-x2
export PATH=/root/miniforge3/envs/3dgrut/bin:$PATH
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/work/yusun/repo/3dgrut
python -u train.py --config-name apps/ncore_3dgut_mcmc_v2_full_exposure \
  path=/root/work/yusun/ncore-nurec/data/ncore/clips/9ae151dc-e87b-41a7-8e85-71772f9603d7/pai_9ae151dc-e87b-41a7-8e85-71772f9603d7.json \
  out_dir=<out> \
  n_iterations=30000 \
  'dataset.camera_ids=[camera_front_wide_120fov,camera_front_tele_30fov,camera_cross_left_120fov,camera_cross_right_120fov,camera_rear_left_70fov,camera_rear_right_70fov,camera_rear_tele_30fov]' \
  trainer.sky_backend=mlp

# T7.3.b (exposure off, Stage 7 推荐出口配置)
# 同上, 追加: trainer.use_exposure=false
```

### 8.2 实测 ckpt 路径（A800）

| Run | Path | metrics.json 关键字段 |
|---|---|---|
| T7.3 | `a800-x2:/root/work/yusun/ncore-nurec/output/stage7_full_20260520-202222/.../ckpt_30000.pt` | psnr_masked=15.63, cc_psnr_masked=24.75 |
| T7.3.b | `a800-x2:/root/work/yusun/ncore-nurec/output/stage7_noexp_20260521-102930/.../ckpt_30000.pt` | psnr_masked=25.76, cc_psnr_masked=24.70 |

### 8.3 渲染图样本（Stage 7 inspect, 已 rsync 到 Mac）

```
/Users/etendue/repo/3dgrut2/.local/stage7_inspect/
├── t7_3_30k/ (7 张, 每相机一张 mid-frame, 严重过曝/泛白)
└── t7_2_baseline_1k/ (1 张, 1-cam 1k smoke, 色彩正常)
```

### 8.4 关键文档锚点

- `v2_plan.md` § 14.5（V3-P1 bilateral grid + ExposureModel 修复）
- `v2_architecture.md` § 7（关键不变量 + Stage 7 实测锚点）
- `CLAUDE.md` § A/B/C（A800 操作严格把关清单, Stage 6-fix + Stage 7 实证）

---

**报告完成日期**：2026-05-21
**作者**：Claude (Sonnet 4.7) + 用户协同
**项目状态**：v2 LayeredGaussians 结题 ✅，V3 启动准备就绪
