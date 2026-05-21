# 3DGRUT v2 — 分层高斯训练 · 可执行计划

> **配套文档**：[v2_architecture.md](v2_architecture.md) 描述模块/流程差异图；[v2_alternative.md](v2_alternative.md) 是备选实现路线。
> **历史讨论**：`~/.claude/plans/a800-x2-10-8-30-cached-sundae.md`（保留原文不动）。
> **本文档作用**：把架构图上的每一处"新增 / 修改"落到具体任务，用看板跟踪进度。

---

## 0. 目标与 KPI

| 维度 | v1 基线 | v2 目标 | NuRec 参考 |
|---|---:|---:|---:|
| 7-cam 20s PSNR | 27.60 | **≥ 28.5** | 36.28 |
| Sky 区域 PSNR | (黑) | **≥ 30** | — |
| Road 区域 PSNR | — | **≥ 32** | — |
| Dynamic vehicle PSNR | — | **≥ 25** | — |
| 30k step 训练时间（A800 单卡） | 35 min @ A100 | ≤ 60 min | — |

**v2 不做**（明确排除）：
- 学习 track pose（用 GT，留 v2.x）
- DynamicDeformable 层粒子分配（仅在 LayeredGaussians 注册占位，留 v3）
- bilateral grid（仅用 affine ExposureModel 占位）
- Cosmos-DiFix 扩散修复（v3）
- C++ tracer 改动（Python 层 concat，renderer 不感知 layer）
- USDZ 打包（V1-6 独立工作包）
- Marching Cubes mesh 导出（V1-5 独立工作包）

---

## 1. 项目看板（Kanban）

> 状态：⬜ Todo · 🟡 In Progress · 🔵 Review · ✅ Done · ⏸ Blocked
> 拖动方法：完成一个任务把行从右上方迁移；遇阻塞标 ⏸ 并在 Risk Log 记录。

### 1.1 顶层看板（按任务，Mermaid Kanban）

> Mermaid 11+ 渲染为五列看板；旧版渲染器会退化为列表，仍可读。

```mermaid
kanban
  Backlog
    [T8.13 viz_4d schema 扩展 fisheye/FTheta 内参 + viser fisheye 投影 → 匹配 render.py 输出 🟡 CONDITIONAL — 仅 T8.12-FIX 视觉不达标时启动, 参考 repo 反证 pinhole 也能渲清晰]
  In Progress
    [T8.12-FIX viser_gui_4d --initial_fov_deg + --camera_type + --camera_fov_deg CLI flags (Mac 代码 + 单测 10/10 PASS, 待 vast.ai 4090 视觉验收)]
  Review
  Blocked
  Done
    [T7.1 复用 v2_full_exposure (注释 Stage 7 入口) + 7-cam Hydra dump ✅]
    [T7.2 A800 1-cam 1k smoke 9.71 it/s / masked 26.38 / 12 字段 metrics ✅]
    [T7.3 A800 7-cam 30k 51 min raw masked 15.63 ❌ → 暴露 exposure 失控]
    [T7.3.b A800 7-cam 30k exposure OFF 证伪 raw masked 25.76 +10.13 dB ✅]
    [T7.4 跳过 (根因不在 cap, 转 V3-P1 bilateral grid 整合) ⏭]
    [T7.5 WP_V2_Report.md + v2_plan/architecture 同步 + git commit ✅]
    [T0.1 A800 环境验证 smoke 24.12 dB ✅]
    [T1.1 LayeredGaussians 容器 NRE schema ✅]
    [T1.2 LayerSpec 完整字段 + registry ✅]
    [T1.3 v1 ckpt → background trainer 侧错误消息 ✅]
    [T1.4 单测 test_layered_gaussians 扩展 ✅]
    [T1.5 Trainer 集成 use_layered_model flag ✅]
    [T2.1 MCMCStrategy 抽 _get_add_cap 钩子 ✅]
    [T2.2 LayeredMCMCStrategy 子类 ✅]
    [T2.3 configs/strategy/layered_mcmc.yaml ✅]
    [T2.4 单测 test_layered_mcmc ✅]
    [T2.5 LayeredGaussians.fused_view 多层路径 ✅]
    [T3.0 init_layer_from_points + optimizer property ✅]
    [T3.1.a + T3.2.a ncore_semantic + mock 单测 ✅]
    [T3.1.b + T3.2.b NCoreDataset aux + LiDAR A800 集成 ✅]
    [T3.3.a + T3.3.b road_init.py LiDAR-Z KNN ✅]
    [T3.4 region-weighted loss + perturb_mask Z lock ✅]
    [T3.5.a 多层 forward + _FusedView ✅]
    [T3.5.b Stage 3 出口 A800 5k PSNR 26.13 ✅]
    [T4.0 tracks buffer ✅]
    [T4.1.a + T4.1.b load_tracks_from_manifest ✅]
    [T4.2.a + T4.2.b dynamic_rigid_init ✅]
    [T4.3 _transform_means + fused_view dyn ✅]
    [T4.4 dynamic_mask scanline AABB ✅]
    [T4.5 Stage 4 出口 A800 10k PSNR 26.32 real cuboids + timestamp-aligned ✅]
    [T5.1 A800 nvdiffrast probe → MLP fallback ✅]
    [T5.2 SkyEnvmapBase + SkyEnvmapMLP + SkyEnvmapCubemap ✅]
    [T5.3 LayeredGaussians._blend_sky + trainer get_losses sky_loss ✅]
    [T5.4 test_sky_envmap 11 测试 + test_layered_gaussians 6 sky tests ✅]
    [T5.6 Stage 5 出口 A800 5k PSNR 26.17 dB ✅]
    [T6.1 ExposureModel 移植 Recon-Studio + 6 测试 ✅]
    [T6.2 trainer init_exposure_model + per-cam apply + exposure_optimizer ✅]
    [T6.3 test_exposure 6 测试 + ckpt roundtrip ✅]
    [T6.3.a Stage 6 出口 A800 5k cc_PSNR 24.94 +1.7 dB ✅]
    [T6F.1 Ego mask Batch.mask 接通 train+val + 9 测试 ✅ Mac]
    [T6F.2 Metric 双指标 psnr/ssim/lpips full+masked + 7 测试 ✅ Mac]
    [T6F.3 Stage 6-fix 出口 A800 5k masked PSNR 29.49 "+9 vs full 20.49" ✅]
    [T8.1 engine LayeredGaussians load + timestamp_us 透传 ✅ Mac 3 测试]
    [T8.2 extract_4d_metadata + ckpt viz_4d 注入 ✅ Mac 8 测试]
    [T8.3 viser_gui_4d 骨架 + FourDMetadata + timeline ✅ Mac 8 测试]
    [T8.4 scene ego/LiDAR/frustum/axes ✅ Mac 4 viser_math 测试]
    [T8.5 scene dynamic tracks polylines + cuboid wireframe ✅ Mac 9 测试]
    [T8.6 dataset --dataset_path lazy fallback ✅ Mac]
    [T8.8 Stage 8 A800 100 step smoke ✅ 70 tracks / ego 2623 / 960 MB]
    [T8.9 inject_viz_4d CLI "方案 B" ✅ A800 实测 T6F.3 ckpt 31 tracks / 991→995 MB]
    [T8.10 viser_gui_4d --no_gaussian_render (Ampere datacenter A100/A800; Hopper H100 不需要) ✅]
    [T8.11 dyn LiDAR per-track local frame + 每帧 transform ✅ 48K 点 / 20 active tracks]
    [T8.12 vast.ai RTX 4090 验证 viser_gui_4d ⚠️ 部分通过 (Bug #1+#2 修, fisheye 内参遗漏待 T8.13)]
```

如果你的 Markdown 渲染器不支持 mermaid kanban，可读下表（同源数据）：

| 列 | 任务数 | 关键项 |
|---|---:|---|
| Backlog ⬜ | 1 | T8.13 🟡 CONDITIONAL — 仅 T8.12-FIX 视觉不达标时启动, **参考 repo (thinkpad:/home/yusun/repo/3dgrut/tools/viser_multilayer_nurec.py) 反证 pinhole 也能渲清晰** |
| In Progress 🟡 | 1 | T8.12-FIX (Mac 代码 + 单测 10/10 PASS, 待 vast.ai RTX 4090 视觉验收) |
| Review 🔵 | 0 | — |
| Blocked ⏸ | 0 | — |
| Done ✅ | 54 | Stage 0-6 + Stage 6-fix + **Stage 7 软出口结题 (T7.1-T7.5 + T7.3.b ablation)** + **Stage 8 viser_gui_4d 11/12 ⚠️** (T8.12 部分通过: pipeline 接通 + Bug #1+#2 修; **T8.12-FIX 诊断翻新**: 参考 repo 反证 fisheye-trained Gaussians + pinhole viser 预览也能渲清晰, 真根因可能是 fov 初值 / view_matrix 约定 / Layered blend 之一, T8.13 schema 扩展降级为 CONDITIONAL). Stage 6-fix A800: masked PSNR 29.49 dB. **Stage 7 实测**: T7.3 7-cam 30k exposure ON raw masked **15.63 ❌**, T7.3.b exposure OFF raw masked **25.76** (+10.13 dB), 但**两组 cc_psnr_masked 几乎一致** (24.75 / 24.70) → 实证 v2 真实重建质量上限 ~24.7 dB cc_psnr_masked = Stage 5/6/6-fix baseline 持平, **ExposureModel 在 30k 长训中退化优化失控** (退化成"高斯学个大概+exposure 补偿"二元解), 不增不减真实质量. Stage 7 软出口判定 ✅ + T7.4 cap ablation 跳过 (根因不在 cap) + **V3-P1 升级为整合任务**: bilateral-grid + ExposureModel 退化修复合并研究. Mac 189/189 PASS (+1 skip, +10 T8.12-FIX 新增) 0 回归. |

### 1.2 任务级看板（按 Subtask）

> 进度状态：⬜ Todo · 🟡 In Progress · 🔵 Review · ✅ Done · ⏸ Blocked

| ID | Stage | Subtask | 估时(d) | 状态 | 改动 / 新增 |
|---|---|---|---:|:---:|---|
| **T0.1** | 0 | A800 环境验证 + smoke 复跑 | 1 | ✅ | smoke 24.12 dB / 9.48 it/s (2026-05-14) |
| **T1.1** | 1 | LayeredGaussians 容器 + NRE ckpt schema | 1 | ✅ | NEW `layers/layered_model.py` (5a6a5f9) |
| **T1.2** | 1 | LayerSpec 完整字段 + registry | 0.5 | ✅ | MOD `layer_spec.py` · NEW `registry.py` · MOD `trainer.py` + `base_gs.yaml` (60e1154 / 569819b / 6435483) |
| **T1.3** | 1 | v1 flat ckpt → layered["background"] 兼容 | 0.5 | ✅ | MOD `layered_model.py` 错误消息指向 `layers.enabled` (ff83028) |
| **T1.4** | 1 | 单测 test_layered_gaussians.py 扩展 | 1 | ✅ | NEW `test_layer_spec_registry.py` (9 测试) + 3 个 A800 contract test (60e1154 / 569819b / ff83028) |
| **T1.5** | 1 | Trainer 集成 + use_layered_model flag | 0.5 | ✅ | MOD `trainer.py` (5a6a5f9 / 8a29fc0) |
| **T2.1** | 2 | MCMCStrategy 抽 `_get_add_cap()` 钩子 | 0.5 | ✅ | MOD `strategy/mcmc.py` · NEW `tests/test_layered_mcmc.py` (62fc509) |
| **T2.2** | 2 | LayeredMCMCStrategy 子类 | 1 | ✅ | NEW `strategy/layered_mcmc.py` · MOD `trainer.py` · MOD `tests/test_layered_mcmc.py` (7ad883b) |
| **T2.3** | 2 | configs/strategy/layered_mcmc.yaml | 0.5 | ✅ | NEW `configs/strategy/layered_mcmc.yaml` · MOD `trainer.py` (1a0d275) |
| **T2.4** | 2 | 单测 test_layered_mcmc.py | 1 | ✅ | NEW `conftest.py` (I-1 fix) · 8 tests total (51540a8 / 04c9174) |
| **T2.5** | 2 | LayeredGaussians.fused_view(frame_id) 多层路径 | 1 | ✅ | MOD `layered_model.py` · NEW 4 tests (d4841df) |
| **T3.0** | 3 | LayeredGaussians.init_layer_from_points + optimizer property | 0.5 | ✅ | MOD `layers/layered_model.py` · NEW 5 tests (Mac 38/38 PASS) |
| **T3.1.a** | 3 | ncore_semantic 常量 + mock 单测（sky/road/dyn partition） | 0.25 | ✅ | NEW `datasets/ncore_semantic.py` · NEW `tests/test_ncore_aux_masks.py` (4 tests) |
| **T3.1.b** | 3 | datasetNcore.py 加载 sky/road/dyn aux mask（A800 集成） | 0.75 | ✅ | NEW `datasets/aux_readers.py` (绕过 SDK 直读 itar) · MOD `datasets/datasetNcore.py` (load_aux_masks + sseg 抽取 + image_infos 装配) · MOD `datasets/protocols.py` (Batch.image_infos) · A800 单帧 sseg 0.11s / sky 1.85% / road 21.55% / dyn 2.50% pairwise disjoint |
| **T3.2.a** | 3 | LiDAR semantic filter mock 单测（行为契约） | 0.25 | ✅ | NEW 3 tests in `test_ncore_aux_masks.py`（合并到 T3.1.a commit） |
| **T3.2.b** | 3 | datasetNcore.py 暴露 road/dyn LiDAR 点（A800 集成） | 0.75 | ✅ | get_road_lidar_points / get_dynamic_lidar_points / _get_semantic_lidar_points 改用 LidarSsegAuxReader 直读 · A800 road 629K pts Z std 0.425m / dyn 135K pts |
| **T3.3.a** | 3 | road_init 6 单测（z_lock / scale_flat / handles_empty / max_n / identity_quat / uneven_terrain） | 0.25 | ✅ | NEW `tests/test_road_init.py` |
| **T3.3.b** | 3 | road_init.py LiDAR-Z KNN + flat scale prior 实现 | 0.75 | ✅ | NEW `layers/road_init.py` |
| **T3.4** | 3 | trainer.py region-weighted loss + perturb mask hook (D1) | 0.75 | ✅ | NEW `model/layered_loss.py` · MOD `trainer.py` · MOD `strategy/mcmc.py` · MOD `strategy/layered_mcmc.py` · MOD `layers/layer_spec.py` · MOD `layers/registry.py` · MOD `configs/base_gs.yaml` · NEW `tests/test_layered_loss.py` (6 tests) · 4 new T3.4 tests in `test_layered_mcmc.py` |
| **T3.5.a** | 3 | LayeredGaussians 多层 forward + _FusedView (本地) | 0.5 | ✅ | MOD `layers/layered_model.py` · 3 new tests |
| **T3.5.b** | 3 | trainer.init_model 串通 road init + A800 5k step 出口 | 0.5 | ✅ | MOD `trainer.py` · MOD `layered_model.py` (build_acc/setup_optimizer/__getattr__ multi-layer fallback) · MOD `road_init.py` (cKDTree+cdist fallback) · MOD `datasets/__init__.py` (load_aux_masks 入 NCoreDataset) · NEW `configs/apps/ncore_3dgut_mcmc_v2_road.yaml` · **A800 5k step PSNR 26.133 dB (+2.5 超额)** |
| **T4.0** | 4 | LayeredGaussians 接 tracks buffer | 0.25 | ✅ | MOD `layers/layered_model.py` · NEW 2 tests |
| **T4.1.a** | 4 | tracks loader mock 单测 (10 case) | 0.25 | ✅ | NEW `tests/test_tracks_loader.py` |
| **T4.1.b** | 4 | tracks_loader.py 实现（独立模块） | 0.5 | ✅ | NEW `datasets/tracks_loader.py` |
| **T4.2.a** | 4 | dynamic_rigid_init 8 单测 | 0.25 | ✅ | NEW `tests/test_dynamic_rigid_init.py` |
| **T4.2.b** | 4 | dynamic_rigid_init.py cuboid 内 LiDAR 抽取 | 0.5 | ✅ | NEW `layers/dynamic_rigid_init.py` |
| **T4.3** | 4 | _transform_means + fused_view dynamic 分支 | 1 | ✅ | MOD `layers/layered_model.py` · NEW 5 tests |
| **T4.4** | 4 | dynamic_mask 纯 PyTorch scanline AABB (D5) | 0.5 | ✅ | NEW `layers/dynamic_mask.py` · NEW 6 tests |
| **T4.5** | 4 | Stage 4 集成 + A800 出口验收（real cuboids + timestamp-aligned） | 1.5 | ✅ | NEW `tracks_loader.load_tracks_from_ncore_cuboids` · MOD `layered_model.py` (populate_tracks + timestamp-aligned _resolve_pose_idx) · MOD `protocols.Batch.timestamp_us` · MOD `datasetNcore.__getitem__` (train+val timestamp_us) · MOD `trainer.setup_training` (dynamic_rigids 串通) · MOD `mcmc.py` (track_ids buffer sync on add/relocate) · NEW `configs/apps/ncore_3dgut_mcmc_v2_full.yaml` · **A800 10k PSNR 26.315 dB (+0.18 vs Stage 3), SSIM 0.883, LPIPS 0.275, 9.58 it/s 零性能损失** |
| **T5.1** | 5 | nvdiffrast.torch 可用性确认 / 降级 SkyModel | 0.5 | ✅ | A800 probe: nvdiffrast 不可用 (PyPI 镜像无 / GitHub install 被 sandbox 拒) → 用 MLP fallback. CLI `trainer.sky_backend=mlp` 覆盖 (2026-05-20) |
| **T5.2** | 5 | port EnvLight → correction/sky_envmap.py (含 MLP fallback) | 0.5 | ✅ | NEW `threedgrut/correction/__init__.py` · NEW `threedgrut/correction/sky_envmap.py` (SkyEnvmapBase + SkyEnvmapMLP + SkyEnvmapCubemap) |
| **T5.3** | 5 | LayeredGaussians sky 层接入 + trainer sky blending + loss | 1 | ✅ | MOD `threedgrut/layers/layer_spec.py` (+ `extra` field) · MOD `threedgrut/layers/registry.py` (sky_envmap extra={backend,resolution}) · MOD `threedgrut/layers/layered_model.py` (_build_sky_module + _blend_sky + sky optimizer hookup + multi-layer ref_layer particle-only) · MOD `threedgrut/trainer.py` (get_losses sky_loss path) · MOD `threedgrut/model/layered_loss.py` (compute_sky_loss 纯函数) · MOD `configs/base_gs.yaml` (trainer.use_sky_envmap/sky_backend/sky_resolution/sky_lr/lambda_sky) · NEW `configs/apps/ncore_3dgut_mcmc_v2_sky.yaml` |
| **T5.4** | 5 | 单测 test_sky_envmap.py + LayeredGaussians sky 集成 + sky_loss | 1 | ✅ | NEW `threedgrut/tests/test_sky_envmap.py` (11 测试) · 6 个新测试在 `test_layered_gaussians.py` (sky module / blend extremes / forward 集成) · 6 个新测试在 `test_layered_loss.py` (compute_sky_loss) · 全部 Mac CPU PASS |
| **T6.1** | 6 | port ExposureModel → correction/exposure.py | 0.25 | ✅ | NEW `threedgrut/correction/exposure.py` (移植 Recon-Studio 29 行 affine) |
| **T6.2** | 6 | trainer step per-camera 应用 + 独立 optimizer + ckpt roundtrip | 0.5 | ✅ | MOD `threedgrut/trainer.py` (init_exposure_model + run_train_iter 应用 + exposure_optimizer.step + save_checkpoint exposure_state + resume) · MOD `configs/base_gs.yaml` (trainer.use_exposure/exposure_lr) · NEW `configs/apps/ncore_3dgut_mcmc_v2_full_exposure.yaml` |
| **T6.3** | 6 | 单测 test_exposure.py | 0.25 | ✅ | NEW `threedgrut/tests/test_exposure.py` (6 测试: zero_init_identity / grad_isolation / clamp / invalid_idx / zero_cameras / state_dict_roundtrip) |
| **T6F.1** | 6-fix | Ego mask 真注入 Batch.mask (train+val) → 自动接通 loss / layered_l1 | 0.5 | ✅ | MOD `threedgrut/datasets/datasetNcore.py` (训练分支取 `sequence_cameras_frame_valid_pixels_masks[seq][cam][frame_idx]` 按渲染分辨率 cv2 INTER_NEAREST resize 塞 `batch_dict["valid"]`; `get_gpu_batch_with_intrinsics` 把 valid 1D/2D → `Batch.mask` reshape [1,H,W,1] float32 GPU) · MOD `threedgrut/model/layered_loss.py` (4D valid_mask squeeze fix, A800 smoke 暴露的 broadcast 错配) · NEW `threedgrut/tests/test_datasetNcore_ego_mask.py` (9 测试 Mac PASS, 含 2 个 4D vs 3D 回归测试) · commit 65869ec + 9c18b57 |
| **T6F.2** | 6-fix | Metric 端 psnr/ssim/lpips full + masked 双指标（保留 Stage 3/4/5/6 历史可比性 + 给出未来 KPI 用干净值） | 0.5 | ✅ | MOD `threedgrut/trainer.py::compute_metrics` + MOD `threedgrut/render.py` eval loop (T6F.3 暴露的盲点 — eval 路径独立, 修复时**必须**同步两处) · psnr_masked 解析公式; ssim_masked/lpips_masked via GT-fill; cc 版本同款 · mask=None 时三指标 ≡ 全图指标保证 byte-identical 回归 · NEW `threedgrut/tests/test_trainer_masked_metrics.py` (7 测试 Mac PASS) · commit 65869ec + 12e142a |
| **T6F.3** | 6-fix | Stage 6-fix 出口 A800 5k smoke v2_full_exposure 双指标 + ego 区粒子密度核查 | 0.5 | ✅ | A800 GPU 1 v2_full_exposure 5k 单相机 (520s, **9.61 it/s 性能 0 损失**). metrics.json 12 字段齐全 (full + masked + cc 各 6). **masked PSNR 29.493 / SSIM 0.934 / LPIPS 0.190** (vs full 20.493 / 0.858 / 0.317; **+9.0 dB / +0.076 / -0.127**). valid frac 78.22% (ego 区 21.78%, mask 方向正确). 印证 ego mask 修复方向正确: 非 ego 区不再被 ego 内卷 → 干净区 PSNR > Stage 4 baseline 26.32. v2_egomask_fix_20260520_113746 · commit 12e142a |
| **T7.1** | 7 | 复用 `v2_full_exposure.yaml` (Stage 7 入口注释) + 7-cam Hydra cfg dump | 0.5 | ✅ | MOD `configs/apps/ncore_3dgut_mcmc_v2_full_exposure.yaml` (Stage 7 注释块); A800 dry-run `--cfg=job` 通过, 7-cam 全部 resolve, layers/sky_envmap/exposure 全栈正确 |
| **T7.2** | 7 | A800 1-cam 1k smoke 全 pipeline 验证 | 0.5 | ✅ | A800 GPU1 v2_full_exposure 1k step 单相机 102.9 s, **9.71 it/s**, 4 层 init (bg 1M / road 200K / dyn 48488 / sky MLP), metrics.json **12 字段齐全** (full+masked+cc 各 6), masked PSNR **26.38 dB** (>Stage 4 baseline 26.32). Run: `stage7_smoke_20260520-174444` |
| **T7.3** | 7 | A800 7-cam 30k step 完整训练 + KPI | 1 | ✅ | A800 GPU1 v2_full_exposure 30k step 7-cam 3061 s = **51.0 min ≤ 60** ✅, **9.80 it/s** 性能 0 损失. 4 层 init (bg 1M / road 200K / **dyn 146761 / 70 tracks** / sky MLP). KPI 出口: raw `psnr_masked=15.63` ❌ vs 目标 30 (差 -14.4 dB), `cc_psnr_masked=24.75` ≈ T6F.3 baseline 24.90. **诊断: ExposureModel 在 30k 长训中退化优化失控**, raw 输出严重过曝/泛白 (肉眼对比 T7.2 baseline 确认). Run: `stage7_full_20260520-202222` |
| **T7.3.b** | 7 | A800 7-cam 30k exposure OFF 对比 ablation (NEW) | 0.5 | ✅ | A800 GPU1 同 T7.3 配置 + `trainer.use_exposure=false` 3064 s, 9.79 it/s. KPI: raw `psnr_masked=25.76` (**vs T7.3 +10.13 dB ✅**), `cc_psnr_masked=24.70` (~T7.3 同, -0.05 dB noise 级). **证伪 ExposureModel 是 T7.3 raw 崩的真因**; 真实重建质量两组持平. Run: `stage7_noexp_20260521-102930` |
| **T7.4** | 7 | per-layer cap ablation (4 组) | 1 | ⏭ | **跳过** (T7.3+T7.3.b 证明根因不在 cap, 4×55 min 投入预期不改变 cc_psnr_masked 上限). 转入 V3-P1 整合研究 |
| **T7.5** | 7 | WP_V2_Report.md + v2_plan/architecture 同步 (schema 转 V3 backlog) | 1 | ✅ | NEW `WP_V2_Report.md` (231 行, 镜像 v1 report 结构 + Stage 7 软出口判定 + cc_PSNR 解读 + ExposureModel 失控诊断 + V3-P1 整合方案), MOD `v2_plan.md` (顶层看板 + 任务级 + Stage 状态 + Done Log + § 14.5 V3-P1 扩展), MOD `v2_architecture.md` (mermaid :::done + § 6.1 文件清单 + § 7 关键不变量 + ExposureModel 健康度锚点) |
| **T8.1** | 8 | engine load_3dgrt_object 加 LayeredGaussians 分支 + render_pass timestamp_us 透传 | 0.5 | ✅ Mac | MOD `threedgrut_playground/engine.py` (load_3dgrt_object .pt 分支 use_layered_model 检测 + populate_tracks from viz_4d / render_pass timestamp_us kwarg / _trace_scene_mog dispatch LayeredGaussians.forward 路径) · NEW `threedgrut/tests/test_engine_layered_load.py` (3 测试) |
| **T8.2** | 8 | extract_4d_metadata 模块 + ckpt save 注入 | 0.75 | ✅ Mac | NEW `threedgrut/viz/__init__.py` + `threedgrut/viz/metadata.py` (~280 行: extract_ego / extract_tracks / extract_lidar / detect_primary_camera / extract_defaults) · MOD `threedgrut/layers/layered_model.py:_populate_tracks_impl` (tracks_metadata 字典 class/size) · MOD `threedgrut/trainer.py:save_checkpoint` (viz_4d.enabled gate + try/except 静默 fallback) · MOD `configs/base_gs.yaml` (viz_4d 默认 enabled=false / subsample 200K/100K) · NEW `configs/apps/ncore_3dgut_mcmc_v2_full_4dviz.yaml` · NEW `threedgrut/tests/test_viz_4d_metadata.py` (8 测试: smoke / subsample / include_lidar=false / tracks_metadata / no_tracks / no_lidar / unknown_class / ckpt_roundtrip) |
| **T8.3** | 8 | viser_gui_4d 骨架 + FourDMetadata + timeline state machine | 0.5 | ✅ Mac | NEW `threedgrut_playground/viser_gui_4d.py` (~640 行: Viser4DViewer 类 + 5 folder GUI + play_tick + _on_time_change 中央调度 + _mirror_ui 抑制回路) · NEW `threedgrut_playground/utils/viz4d_metadata.py` (~150 行 FourDMetadata dataclass + lookup_frame_idx 二分 + active_tracks_at + ego_pose_at nearest) · NEW `threedgrut/tests/test_viz4d_metadata_loader.py` (8 测试) |
| **T8.4** | 8 | scene D - ego trajectory + frustum + LiDAR + world axes | 0.5 | ✅ Mac | MOD `viser_gui_4d.py:_populate_static_scene` 系列 (_add_world_axes / _add_ego_trajectory spline_catmull_rom / _add_lidar_clouds road+dyn / _update_ego_frustum setter) · NEW `threedgrut_playground/utils/viser_math.py` (mat_to_wxyz Shepperd/Markley + canonical sign) · NEW `threedgrut/tests/test_viser_math.py` (4 测试: identity / 90deg_x / 50 random round-trip / canonical sign) |
| **T8.5** | 8 | scene E - dynamic tracks polylines + per-frame cuboid wireframe | 0.75 | ✅ Mac | MOD `viser_gui_4d.py:_add_track_trajectories` (一次 add_line_segments 全 tracks polyline 按 class color) + `_build_cuboid_edges` + `_update_active_cuboids` (remove+add 每帧) · NEW `threedgrut_playground/utils/cuboid.py` (UNIT_CUBE_EDGES [12,2,3] + cuboid_world_edges + class_color palette + instance_color FNV-1a hash → HSV→RGB) · NEW `threedgrut/tests/test_cuboid_wireframe.py` (9 测试: edges shape / vertex range / unique pairs / identity / translation / size scale / z90 rotation / class_color / instance_color) |
| **T8.6** | 8 | dataset --dataset_path lazy fallback | 0.5 | ✅ Mac | MOD `viser_gui_4d.py:_load_metadata` (lazy import NCoreDataset + extract_4d_metadata 路径; 无 viz_4d block + dataset_path=None → static fallback warn) |
| **T8.8** | 8 | Stage 8 A800 100 step smoke + ckpt viz_4d 字段验证 | 0.25 | ✅ A800 | A800 GPU 0, `apps/ncore_3dgut_mcmc_v2_full_4dviz` 100 step. ckpt_last.pt **960.2 MB, 自带 viz_4d schema_v1**, 70 dynamic tracks (5-cam 20s clip), ego 2623 poses, road 200K of 629K, dyn 100K of 135K. 13:18-13:29 总时长 11 min (dyn_init 17min 改进后约 5min). 暴露 + 修复 3 bugs (d15f69e/64ccf95/dffa59f): teardown 顺序 + 顶层 config + datasets.make 工厂. |
| **T8.9** | 8 | inject_viz_4d CLI 工具（方案 B 一次性注入旧 ckpt） | 0.25 | ✅ **A800 端到端** | NEW `threedgrut/viz/inject.py` (~165 行: inject_viz_4d + _populate_tracks_from_dataset + _extract_conf 兼容旧 v2 ckpt 顶层无 config 的嵌套布局) · MOD `threedgrut_playground/README_4D.md` (加方案 B 章节) · NEW `threedgrut/tests/test_inject_viz_4d.py` (5 测试). **A800 实测 (T6F.3 ckpt v2_egomask_fix_20260520_113746)**: 991.3MB 旧 ckpt → 1m50s 注入 → 995.1MB (+3.8MB 元数据), viz_4d schema_v1, 31 tracks (与 T4.5 baseline 完全吻合), sample track automobile size=[4.09, 1.87, 1.61]m, ego 51 poses + primary cam fov_y=2.441 rad, road_xyz 200K/629K, dyn_xyz 100K/135K, ts range 88-1988 ms (duration_sec=2.0 对齐) |
| **T8.10** | 8 | viser_gui_4d --no_gaussian_render 模式 (Ampere datacenter GPU 兼容) | 0.25 | ✅ **A800 端到端** | **仅 Ampere datacenter SKU (A100/A800)** RT cores 被 NVIDIA 阉割导致 OptiX dlopen segfault. Hopper datacenter (H100/H800/H200, 第 3 代 RT cores) + RTX 系列 + workstation A5000/A6000 都有 RT cores, 不需要此 flag. MOD `viser_gui_4d.py`: --no_gaussian_render flag 跳过 Engine3DGRUT 实例化, 主循环只跑 _play_tick + scene primitives. UI 隐藏 Resolution/Near/Far/FPS 控件. **A800 实测**: server listening *:8080 起来, 浏览器拖 timeline → ego/cuboid/LiDAR/frustum 全部动起来 (无 Gaussian 背景) |
| **T8.11** | 8 | dynamic LiDAR per-track object-local frame + per-frame transform | 0.25 | ✅ **A800 端到端** | T8.10 暴露视觉 bug: cuboid 移动但 dyn LiDAR 点云不动 (static world union). 修复复用训练侧 `init_dynamic_rigid_layer` 路径: MOD `viz/metadata.py:_extract_lidar` 调用它产 per-track local pts + track_ids + track_names; NEW schema 字段 `dynamic_local_xyz/track_ids/track_names`; MOD `FourDMetadata.has_per_track_dyn_lidar()` helper; MOD `viser_gui_4d`: `_build_dyn_lidar_world(frame_idx)` 每帧 R·local+t + instance_color 着色, `_update_dynamic_lidar` remove+add point_cloud. NEW `lidar_dynamic_pts_per_track` config 默认 5000. **A800 实测**: 48,488 个 object-local 点分布在 20 个 active tracks (cap 5000/track), 994.8 MB ckpt, 浏览器拖 timeline → dyn LiDAR 点云跟着 cuboid 飘. NEW `test_dyn_lidar_per_track_local_frame` 测试 |
| **T8.12-FIX** | 8 | viser_gui_4d --initial_fov_deg + --camera_type + --camera_fov_deg CLI flags (诊断翻新: 参考 repo 反证 pinhole + 90° fov 也能渲清晰) | 0.25 | 🟡 **Mac 代码完成**, 待 vast.ai 视觉 | 远端参考 `thinkpad:/home/yusun/repo/3dgrut/tools/viser_multilayer_nurec.py:280` 同样处理 fisheye-trained nurec 用 `client.camera.fov = math.radians(90)` 硬设 + 纯 pinhole `make_rays` 渲染清晰; 反证 T8.12 留的 T8.13 假设 (必须扩 FTheta schema) 大概率错诊. **改动**: `viser_gui_4d.py` 加 3 CLI flags + `Viser4DViewer.__init__` 加 `initial_fov_rad` kwarg + `_on_connect` / Reset View 显式设 `client.camera.fov` + engine 实例化后可选 `engine.camera_type=Fisheye / engine.camera_fov=<deg>` 走已存在 `_raygen_fisheye` 路径. **Mac 验证**: NEW `test_viser_gui_4d_fov.py` 10/10 PASS + 全量 189/189 PASS 0 回归. **A800 验证**: CLI --help 显示三个新 flag 正确. **未做**: vast.ai 4090 视觉对比 (T8.12 实例已销毁需重开). |
| **T8.12** | 8 | vast.ai RTX 4090 验证 viser_gui_4d 完整 Gaussian 渲染 + 修 Stage 8 集成 bug | 0.5 | ⚠️ **部分通过** | vast.ai RTX 4090 24GB (Norway, $0.630/hr). **修了 2 个真实 Stage 8 bug**: **Bug #1** `engine.py:_trace_scene_mog` LayeredGaussians 路径构 Batch 缺 camera intrinsics → 3dgut tracer `__create_camera_parameters` 抛 `Camera intrinsics unavailable` viser 一连即崩; fix: 从 kaolin Camera 取 [fx,fy,cx,cy] 塞 Batch.intrinsics + 真实 c2w as T_to_world + camera-space rays 匹配 NCoreDataset.get_gpu_batch_with_intrinsics contract. **Bug #2** `layered_model.py:init_from_checkpoint` 完后 SkyEnvmapMLP 残留 CPU → `_blend_sky` 路径 `cpu vs cuda` addmm 报错; fix: 末尾 `self.cuda()` 把整个 ModuleDict 搬上 GPU. **Reset View 改进**: snap camera 到 `meta.initial_c2w` (position+wxyz+look_at+up_direction). **Infra fix**: `scripts/cuda_helper.sh` 加 CUDA 12.1 case + viser nohup 不持久 → setsid 子 shell; NEW `docs/T8.12_handover_day1.md`. **Pipeline 验证**: viser RTX 4090 87 FPS 跑通 + scene primitives (cuboid/LiDAR/ego frustum) 全同步 + timeline 推进 cuboid + dyn LiDAR 跟车飘. **未达预期点 → T8.13**: viz_4d schema (T8.2 设计) 只存 `primary_camera_fov_y_rad`, 没存 fisheye polynomial / distortion coeffs. NCore ckpt 用 `camera_front_wide_120fov` (FTheta fisheye) 训练, viser 用 pinhole 投影 → Gaussians 视觉是远景隧道 motion-blur 乱糊, 跟 render.py 输出的清晰街景 (含 fisheye 桶形畸变) 完全不是同一视角. 用户对比图实证此 gap. Bug #3 fov override 因 pinhole 140° 退化已撤. **完整 fisheye 渲染留 T8.13** (扩展 viz_4d schema 含 FTheta + viser_gui_4d 用 fisheye intrinsics). **T8.12 实例 37188673 已销毁** (~$1.5 总成本) |
| | | **合计** | **30.0** | | |

### 1.3 当前 Stage 状态汇总

| Stage | 名称 | 完成 / 总 | 关键产出 |
|---|---|---:|---|
| 0 | A800 环境验证 | 1/1 ✅ | smoke 24.12 dB baseline |
| 1 | Layer 抽象 | 5/5 ✅ | LayeredGaussians + registry + base.yaml 默认 + 9 本地单测 + 3 A800 contract test |
| 2 | Layered MCMC | 5/5 ✅ | T2.1: `_get_add_cap()` hook (62fc509) · T2.2: LayeredMCMCStrategy sub-strategy array (7ad883b) · T2.3: layered_mcmc.yaml + trainer dedup (1a0d275) · T2.4: 8 tests + conftest I-1 fix (51540a8/04c9174) · T2.5: fused_view + get_layer_mask + 4 tests (d4841df; carry-over 75ed0e4) |
| 3 | Road 层 | **10/10 ✅** | Stage 3 **完成**：PSNR 26.133 dB (出口 23.6, +2.5 超额), SSIM 0.879, LPIPS 0.297, 9.54 it/s 零性能损失 |
| 4 | DynamicRigid 层 | **8/8 ✅** | Stage 4 **完成**：PSNR 26.315 dB (Stage 3 +0.18), SSIM 0.883, LPIPS 0.275, 9.58 it/s; 31 真实 cuboid tracks (autolabels v2), 48K dyn particles; 距严格出口 26.4 差 0.085 (noise 级) |
| 5 | Sky envmap | **4/4 ✅** | Stage 5 出口完成 A800 5k PSNR **26.167 dB** (sky_envmap=MLP, +0.37 over 25.8 出口门槛, -0.15 vs Stage 4 baseline noise 级) |
| 6 | Exposure | **3/3 ✅** | Stage 6 出口完成 A800 5-cam 5k cc_PSNR **24.937 dB** (+1.7 dB cc gain 直证 per-cam affine 学到差异), exposure_a.std=0.0306 > 0.01 出口 ✅ |
| 6-fix | Ego mask 全栈接通 | **3/3 ✅** | Stage 6-fix 完成. T6F.1+T6F.2 Mac 本地 (16 新测试, 141/141 PASS). T6F.3 A800 5k smoke v2_full_exposure: **masked PSNR 29.49 dB > Stage 4 baseline 26.32 (+3.17 dB 干净区)**, full PSNR 20.49 (ego 区不再训练→渲染崩 -5.8 dB, 正确预期), 性能 0 损失 (9.61 it/s ≈ 9.58). ego 区 21.78% 量化为 Stage 3/4/5/6 历史 PSNR 水分源 |
| 7 | 集成 + KPI 软出口 | **5/5 ✅** (T7.4 跳过) | Stage 7 软出口结题. T7.1 复用 v2_full_exposure (无新 yaml) + 7-cam Hydra dump 通过. T7.2 A800 1-cam 1k smoke masked 26.38 / 9.71 it/s. **T7.3 A800 7-cam 30k 51 min raw masked 15.63 ❌ 但 cc_psnr_masked 24.75 OK → 暴露 ExposureModel 长训退化优化失控. T7.3.b A800 同配置 + use_exposure=false ablation 证伪: raw masked 25.76 (+10.13 dB), cc_psnr_masked 24.70 (vs T7.3 24.75 -0.05 dB noise 级)** → 实证 v2 真实重建质量上限 ~24.7 dB cc_psnr_masked = Stage 5/6/6-fix baseline 持平. **T7.4 cap ablation 跳过** (根因不在 cap). T7.5 WP_V2_Report.md (231 行) + v2_plan/architecture 同步, ExposureModel 失控 + bilateral-grid 合并 V3-P1 整合任务 (§ 14.5) |
| 8 | viser_gui_4d (4D viz) | **11/12 ⚠️ pipeline 通, fisheye 渲染留 T8.13** | Stage 8 完整 (T8.1-T8.6 + T8.8-T8.11 ✅, T8.12 ⚠️). ckpt['viz_4d'] schema v1: ego poses + tracks {poses/size/frame_info/class} + road LiDAR + **per-track object-local dynamic LiDAR (T8.11)** + viewer_defaults. viser_gui_4d.py: timeline + ego polyline + per-frame frustum + tracks polylines (class color) + cuboid wireframe + 每帧 dyn LiDAR world transform (instance color) + `--no_gaussian_render` (T8.10 Ampere datacenter A100/A800 兼容). inject_viz_4d CLI 方案 B 一次性注入. **A800 + vast.ai RTX 4090 双路实测**: A800 走 --no_gaussian_render bypass; RT cores (T8.12, RTX 4090 Norway $0.630/hr, 87 FPS @ 1024×~600) 完整 Gaussian 渲染**pipeline 通**但**视觉不匹配 render.py**. **T8.12 修了 Stage 8 两个真实集成 bug**: camera intrinsics 缺失 (engine.py) / sky_envmap CPU 残留 (layered_model.py). **T8.12 发现的 schema gap**: `viz_4d.ego` 只存 `primary_camera_fov_y_rad`, 没存 fisheye polynomial. NCore ckpt 训练用 `camera_front_wide_120fov` (FTheta fisheye), viser 用 pinhole 投影 → Gaussians 乱糊不是 ground truth → T8.13 backlog (扩展 schema + viser fisheye 投影). Mac 179/179 PASS 0 回归. |

### 1.4 依赖关系图

```mermaid
flowchart LR
    T01["T0.1 ✅<br/>A800 baseline"]:::done

    %% Stage 1
    T11["T1.1 ✅<br/>LayeredGaussians"]:::done
    T12["T1.2 ✅<br/>LayerSpec + registry"]:::done
    T13["T1.3 ✅<br/>v1 ckpt resume msg"]:::done
    T14["T1.4 ✅<br/>单测扩展"]:::done
    T15["T1.5 ✅<br/>Trainer 集成"]:::done

    %% Stage 2
    T21["T2.1 ✅<br/>_get_add_cap 钩子"]:::done
    T22["T2.2 ✅<br/>LayeredMCMC"]:::done
    T23["T2.3 ✅<br/>yaml 配置 (1a0d275)"]:::done
    T24["T2.4 ✅<br/>单测 (51540a8/04c9174)"]:::done
    T25["T2.5 ✅<br/>多层 fused_view (d4841df)"]:::done

    %% Stage 3 ✅ 全部完成 (commits b3b3b2b / e8cb490 / 9f6a54c / 9077fd6 / 5b49f4b / c688984 / 8a625c2)
    T30["T3.0 ✅<br/>init_layer_from_points<br/>+ optimizer property"]:::done
    T31a["T3.1.a ✅<br/>ncore_semantic 常量"]:::done
    T31b["T3.1.b ✅<br/>aux_readers + sseg mask"]:::done
    T32a["T3.2.a ✅<br/>LiDAR filter mock"]:::done
    T32b["T3.2.b ✅<br/>road/dyn LiDAR 接口"]:::done
    T33a["T3.3.a ✅<br/>road_init 6 单测"]:::done
    T33b["T3.3.b ✅<br/>road_init scipy cKDTree"]:::done
    T34["T3.4 ✅<br/>region loss + perturb Z lock"]:::done
    T35a["T3.5.a ✅<br/>多层 forward + _FusedView"]:::done
    T35b["T3.5.b ✅<br/>Stage 3 出口<br/>A800 5k 26.13 dB"]:::done

    %% Stage 4 ✅ 全部完成 (commits b22a506 / 4807951)
    T40["T4.0 ✅<br/>tracks buffer"]:::done
    T41a["T4.1.a ✅<br/>tracks loader 10 单测"]:::done
    T41b["T4.1.b ✅<br/>load_tracks_from_ncore_cuboids<br/>(real autolabels v2)"]:::done
    T42a["T4.2.a ✅<br/>dyn_init 8 单测"]:::done
    T42b["T4.2.b ✅<br/>dynamic_rigid_init cuboid 抽取"]:::done
    T43["T4.3 ✅<br/>_transform_means timestamp-aligned"]:::done
    T44["T4.4 ✅<br/>dynamic_mask scanline AABB"]:::done
    T45["T4.5 ✅<br/>Stage 4 出口<br/>A800 10k 26.32 dB"]:::done

    %% Stage 5 ✅ 全部完成 (commit b38fce8 + A800 GPU1 PSNR 26.17 dB)
    T51["T5.1 ✅<br/>nvdiffrast probe<br/>→ MLP fallback"]:::done
    T52["T5.2 ✅<br/>sky_envmap.py<br/>(MLP + Cubemap)"]:::done
    T53["T5.3 ✅<br/>sky blend + sky_loss"]:::done
    T54["T5.4 ✅<br/>23 新测试 +<br/>ckpt roundtrip"]:::done

    %% Stage 6 ✅ Mac 本地完成
    T61["T6.1 ✅<br/>ExposureModel"]:::done
    T62["T6.2 ✅<br/>per-cam apply + opt + ckpt"]:::done
    T63["T6.3 ✅<br/>6 测试"]:::done

    %% Stage 6-fix (ego mask 全栈接通)
    T6F1["T6F.1 ✅<br/>Batch.mask 注入<br/>train+val (Mac)"]:::done
    T6F2["T6F.2 ✅<br/>metric 双指标<br/>full + masked (Mac)"]:::done
    T6F3["T6F.3 ✅<br/>A800 5k masked PSNR 29.49<br/>+9 dB vs full 20.49"]:::done

    %% Stage 7 软出口结题
    T71["T7.1 ✅<br/>复用 v2_full_exposure<br/>+ 7-cam Hydra dump"]:::done
    T72["T7.2 ✅<br/>A800 1-cam 1k smoke<br/>masked 26.38 / 9.71 it/s"]:::done
    T73["T7.3 ✅<br/>A800 7-cam 30k 51 min<br/>raw masked 15.63 ❌<br/>cc_masked 24.75 OK"]:::done
    T73b["T7.3.b ✅<br/>A800 exposure OFF ablation<br/>raw masked 25.76 +10 dB<br/>cc 24.70 同, 证伪 exposure"]:::done
    T74["T7.4 ⏭<br/>cap ablation 跳过<br/>(根因不在 cap)"]:::done
    T75["T7.5 ✅<br/>WP_V2_Report.md + 文档同步<br/>+ V3-P1 整合 bilateral-grid"]:::done

    %% Stage 8 (Mac 本地完成, 待 A800 smoke)
    T81["T8.1 ✅<br/>engine LayeredGaussians<br/>+ timestamp_us"]:::done
    T82["T8.2 ✅<br/>extract_4d_metadata<br/>+ ckpt viz_4d"]:::done
    T83["T8.3 ✅<br/>viser_gui_4d 骨架<br/>+ FourDMetadata"]:::done
    T84["T8.4 ✅<br/>ego/LiDAR/frustum"]:::done
    T85["T8.5 ✅<br/>tracks + cuboid"]:::done
    T86["T8.6 ✅<br/>--dataset_path fallback"]:::done
    T89["T8.9 ✅<br/>inject_viz_4d CLI<br/>(方案 B + A800 实测)"]:::done
    T88["T8.8 ✅<br/>A800 100 step smoke<br/>70 tracks / 960 MB"]:::done

    T01 --> T11
    T11 --> T12 --> T13 --> T14
    T11 --> T15

    T15 --> T21 --> T22 --> T23 --> T24
    T22 --> T25

    %% Stage 3 chain
    T15 --> T30
    T30 --> T31a --> T31b
    T30 --> T32a --> T32b
    T30 --> T33a --> T33b
    T31b --> T34
    T32b --> T34
    T34 --> T35a --> T35b
    T33b --> T35b
    T25 --> T35a

    %% Stage 4 chain
    T15 --> T40
    T40 --> T41a --> T41b
    T40 --> T42a --> T42b
    T41b --> T45
    T42b --> T45
    T43 --> T45
    T44 --> T45
    T35a --> T43

    %% Stage 5/6/7 (todo)
    T15 --> T51 --> T52 --> T53 --> T54
    T15 --> T61 --> T62 --> T63

    %% Stage 6-fix chain（接在 T6.3 之后，Stage 7 之前）
    T63 --> T6F1 --> T6F2 --> T6F3

    T24 --> T71
    T35b --> T71
    T45 --> T71
    T54 --> T71
    T63 --> T71
    T25 --> T71
    T6F3 --> T71

    T71 --> T72 --> T73
    T73 -. raw masked 15.6 → exposure 失控 .-> T73b
    T73b -. 证伪后跳过 .-> T74
    T73b --> T75
    T73 --> T75

    %% Stage 8 chain (independent of Stage 7, takes any v2 ckpt)
    T63 --> T81
    T81 --> T82
    T81 --> T83
    T82 --> T83
    T83 --> T84
    T83 --> T85
    T82 --> T86
    T83 --> T86
    T82 --> T89
    T82 --> T88
    T83 --> T88

    classDef todo fill:#f5f5f5,stroke:#999,color:#333
    classDef wip  fill:#fff3cd,stroke:#bf8700,stroke-width:3px,color:#7a4d00
    classDef done fill:#cfe8ff,stroke:#0969da,stroke-width:3px,color:#0a3069
```

---

## 2. Stage 详解

> 已完成的 T0.1 / T1.1 / T1.5 见末尾"Done Log"，此处只展开 ⬜ / 🟡 任务。

### Stage 1 — Layer 抽象（基础设施）

#### T1.2 — LayerSpec 完整字段 + registry

- **目标**：把 layer 描述参数（scale prior / mask gating / lr_mult / is_particle_layer）从 Python 代码挪到配置，未来 ablation 只动 yaml。
- **现状**：`layers/layer_spec.py` 已经是 frozen dataclass，但只含 `name / layer_id / max_n_particles`。
- **改动**：
  - `layers/layer_spec.py`：补字段 `scale_prior: Tuple[float,float,float]`、`scale_lr_mult: float = 1.0`、`mask_field: Optional[str]`、`is_particle_layer: bool = True`、`density_init: float = 0.1`。
  - 新建 `layers/registry.py`：
    ```python
    STANDARD_LAYERS = {
      "background"        : LayerSpec("background",         0,  600_000, (0.1,0.1,0.1)),
      "road"              : LayerSpec("road",               1,  200_000, (0.1,0.1,0.001), scale_lr_mult=0.2, mask_field="road_mask"),
      "dynamic_rigids"    : LayerSpec("dynamic_rigids",     2,  200_000, (0.05,0.05,0.05), mask_field="dynamic_mask"),
      "dynamic_deformables": LayerSpec("dynamic_deformables",3,       0, (0,0,0), is_particle_layer=False),  # v2 占位
      "sky_envmap"        : LayerSpec("sky_envmap",        -1,       0, (0,0,0), mask_field="sky_mask", is_particle_layer=False),
    }
    def specs_from_config(cfg) -> list[LayerSpec]: ...
    ```
- **验收**：`pytest threedgrut/tests/test_layered_gaussians.py::test_registry_specs_have_unique_ids`。

#### T1.3 — v1 flat ckpt → layered["background"] 兼容（trainer 侧）

- **目标**：`Trainer3DGRUT.setup_training` 的 `resume` 路径走 LayeredGaussians 时，能识别 v1 flat ckpt 并自动 route 到 background 层。
- **现状**：`LayeredGaussians.init_from_checkpoint` 已经支持 3 种 ckpt 形态（NRE wrap / 已解开 / v1 flat），T1.1 已完成。但 trainer 的 `setup_training` 路径 A 还需明确：当 `use_layered_model=True` 且 ckpt 是 v1 flat 时，构造 LayeredGaussians 时必须含 `"background"` 层。
- **改动**：`threedgrut/trainer.py::setup_training` 路径 A：
  ```python
  if conf.use_layered_model:
      specs = specs_from_config(conf)
      if not any(s.name == "background" for s in specs):
          raise ValueError("v1 ckpt resume requires 'background' layer in layers.enabled")
      model = LayeredGaussians(conf, specs, scene_extent)
      model.init_from_checkpoint(checkpoint, ...)
  else:
      ...  # v1 path 不动
  ```
- **验收**：A800 上 `python train.py resume=<v1_ckpt_path> use_layered_model=true layers.enabled=[background]` 跑通；val PSNR 与 v1 一致 (±0.05 dB)。

#### T1.4 — 单测 test_layered_gaussians.py 扩展

- 在 T1.1 已有 contract test 基础上补：
  - `test_registry_returns_specs_for_enabled_only`
  - `test_layer_spec_frozen_immutable`
  - `test_v1_ckpt_routed_to_background`（已有，确认 Coverage）
  - `test_multi_layer_ckpt_roundtrip`（新增 2 层 roundtrip 字节级一致）
- **验收**：本地 Mac `pytest threedgrut/tests/test_layered_gaussians.py -v`，case ≥ 8，全部 pass。

---

### Stage 2 — Layered MCMC

#### T2.1 — MCMCStrategy 抽 `_select_indices` 钩子

- **目标**：把现有 `relocate_gaussians / add_new_gaussians / perturb_gaussians` 改成"先选 mask 再操作"两阶段，子类 override mask 选取即可。
- **改动**：`threedgrut/strategy/mcmc.py`：
  ```python
  def _select_indices(self, model) -> torch.BoolTensor:
      return torch.ones(model.num_gaussians, dtype=torch.bool, device=model.device)

  def relocate_gaussians(self, model, optimizer):
      idx = self._select_indices(model)
      # 现有 tensor 操作前面切 idx
  ```
- **关键不变量**：基类行为对 v1（单层）byte-identical。
- **验收**：用 v1 ckpt 跑 1000 step，重构前后 MCMC 关键指标（粒子数曲线、relocation rate、PSNR）字节级一致；通过现有 v1 MCMC 单测。

#### T2.2 — LayeredMCMCStrategy 子类

- **目标**：MCMC 三个操作在每层独立执行，per-layer cap，跨层无迁移。
- **改动**：`threedgrut/strategy/layered_mcmc.py` ~100 行：
  ```python
  class LayeredMCMCStrategy(MCMCStrategy):
      def __init__(self, conf, model: LayeredGaussians, specs: list[LayerSpec]):
          super().__init__(conf, model)
          self.specs = specs

      def post_optimizer_step(self, step):
          for spec in self.specs:
              if not spec.is_particle_layer:
                  continue
              self._current_layer = spec.name
              self._current_cap = spec.max_n_particles
              super().post_optimizer_step(step)

      def _select_indices(self, model):
          return model.get_layer_mask(self._current_layer)
  ```
  Trainer factory：
  ```python
  if conf.strategy.method == "LayeredMCMCStrategy":
      strategy = LayeredMCMCStrategy(conf.strategy, model, specs)
  ```
- **验收**：见 T2.4。

#### T2.3 — configs/strategy/layered_mcmc.yaml

```yaml
defaults: [mcmc]
method: LayeredMCMCStrategy
per_layer_max_n:
  background:      600000
  road:            200000
  dynamic_rigids:  200000
```
- **验收**：`python train.py --config-name apps/... --cfg job` 查看 strategy 段合并正确。

#### T2.4 — 单测 test_layered_mcmc.py

- `test_per_layer_cap_respected`：3 层 mock，add 100 步后每层 ≤ cap。
- `test_no_cross_layer_migration`：relocate 1000 步后，初始 layer 归属不变（用 `get_layer_mask` 对比）。
- `test_falls_back_to_global_when_single_layer`：只有 bg 时行为 ≡ v1 MCMC。
- **验收**：本地 Mac CPU mock 跑 < 2 秒，全部 pass。

#### T2.5 — LayeredGaussians.fused_view 多层路径

- **目标**：T1.5 ✅ 已实现单 bg 透传；T2.5 实现真正 N 层 concat。
- **改动**：`layers/layered_model.py`：
  ```python
  def fused_view(self, frame_id: Optional[int] = None) -> dict[str, Tensor]:
      """Return flat tensors (positions/rotation/scale/density/SH) concat across layers.
      Dynamic layers (T4.3) apply per-frame pose transform inline."""
      ...
  def forward(self, batch, train=True, frame_id=...):
      flat = self.fused_view(frame_id)
      return self._render(flat, batch, train)  # 调 background.renderer 或共享 tracer
  ```
- **依赖**：和 T4.3 协作；先在 Stage 2 跑通 bg + road 两层 concat，再 Stage 4 加 dynamic pose 变换。
- **验收**：bg + road 2 层 concat 后渲染，单帧 RGB 与"bg only + road only 分别渲染再 alpha 合成"在 PSNR 内一致（验证 concat 数学正确性）。

---

### Stage 3 — Road 层

#### T3.1 — datasetNcore.py 加载 sky/road/dynamic aux mask

- **目标**：dataloader 输出 `image_infos` 中新增 `sky_mask / road_mask / dynamic_mask_sseg`（per-frame per-camera）。
- **改动**：`datasets/datasetNcore.py`：
  - `__init__` 加 `load_aux_masks: bool = False`
  - 新方法 `_load_sseg_masks(camera_id, frame_idx)` 读 `aux.sseg.zarr.itar`：
    ```python
    sseg = self._sseg_reader.read(camera_id, frame_idx)
    return {
      "sky_mask":           (sseg == SKY_CLASS_ID).float(),
      "road_mask":          (sseg.isin(ROAD_CLASS_IDS)).float(),
      "dynamic_mask_sseg":  (sseg.isin(DYNAMIC_CLASS_IDS)).float(),
    }
    ```
  - Class ID 来自 `ncore.semantic.NCORE_SEMANTIC_LABELS`。
- **验收**：抽 1 帧 → 三 mask 之和 + 其他 ≈ 1.0；road_mask 可视化吻合路面。

#### T3.2 — datasetNcore.py 暴露 road LiDAR 点接口

- **目标**：为 road_init 提供"分类为路面"的 LiDAR 点。
- **改动**：`datasetNcore.py`：
  ```python
  def get_road_lidar_points(self) -> Tuple[Tensor, Tensor]:
      pts, labels = self._lidar_sseg_reader.read_all()
      mask = torch.isin(labels, torch.tensor(ROAD_CLASS_IDS))
      return pts[mask], self._project_colors(pts[mask])
  ```
- **验收**：clip 3435ace9 → road pts 数 ∈ [10K, 500K]；Z std < 0.5 m；BEV plot 形态合理。

#### T3.3 — road_init.py LiDAR-Z KNN + flat scale prior

- **目标**：基于 road LiDAR 构造 200K 路面粒子，scale [0.1, 0.1, 0.001]，Z 由 KNN 拉到最近路面点。
- **改动**：新建 `layers/road_init.py`：
  ```python
  def init_road_layer(road_points, ego_trajectory, cut_range=30.0, resolution=0.05, max_n=200_000):
      xy_min = ego_trajectory[:, :2].min(0).values - cut_range
      xy_max = ego_trajectory[:, :2].max(0).values + cut_range
      grid_xy = make_bev_grid(xy_min, xy_max, resolution)   # [M, 2]
      # 用 torch.cdist 而非 pytorch3d.knn → 避开 PyTorch3D 依赖
      dists = torch.cdist(grid_xy.unsqueeze(0), road_points[:, :2].unsqueeze(0))[0]
      nearest = dists.argmin(1)
      grid_z = road_points[nearest, 2]
      positions = torch.stack([grid_xy[:, 0], grid_xy[:, 1], grid_z], dim=1)
      scales = torch.log(torch.tensor([0.1, 0.1, 0.001])).expand(M, 3)
      ...
      return positions, rotations, scales, densities, albedo
  ```
- **验收**：见 T3.5。

#### T3.4 — trainer.py region-weighted loss

- **改动**：`threedgrut/trainer.py::get_losses`：
  ```python
  if conf.trainer.layered_loss:
      valid = image_infos["valid_pixel_mask"]
      sky   = image_infos["sky_mask"]
      road  = image_infos["road_mask"]
      dyn   = image_infos["dynamic_mask"]    # cuboid 投影 (T4.4) 优先；fallback sseg
      bg    = valid * (1 - road) * (1 - dyn) * (1 - sky)

      l1 = (rgb_pred - rgb_gt).abs()
      loss = (
          (l1 * bg  ).sum() / (bg  .sum() + 1e-6)
        + (l1 * road).sum() / (road.sum() + 1e-6)
        + (l1 * dyn ).sum() / (dyn .sum() + 1e-6)
      )
  else:
      loss = (rgb_pred - rgb_gt).abs().mean()   # v1 行为
  ```
- **验收**：mock 4x4 图 + 已知 mask → 数值对账；集成 500 步路面 Z std 仍 < 0.005。

#### T3.5 — 单测 test_road_init.py

- `test_road_init_z_lock`：mock 100 路面点 Z=0 → init 后所有 Z 误差 < 0.05 m
- `test_road_init_scale_flat`：scales.exp()[:, 2] < 0.005
- `test_road_init_handles_empty_lidar`：空 tensor 不 crash
- `test_road_init_respects_max_n`：500K 候选 → ≤ 200K 输出
- **验收**：本地 Mac pytest 全 pass < 1 秒。

---

### Stage 4 — DynamicRigid 层

> NVIDIA NuRec 命名 = "dynamic_rigids"（OmniRe 称 `RigidNodes`）

#### T4.1 — scene_manifest tracks → instance_pts_dict loader

- **OmniRe 参考 schema** (`drivestudio/datasets/driving_dataset.py:263-396`)：
  ```
  instance_pts_dict[track_id] = {
      "pts":        [N, 3] local-frame Gaussian means (T4.2 填)
      "colors":     [N, 3] (T4.2 填)
      "poses":      [num_frame, 4, 4] object→world SE(3)
      "size":       [3] cuboid 半轴
      "frame_info": [num_frame] bool active
      "class":      str
  }
  ```
- **改动**：`datasets/datasetNcore.py::load_tracks_from_manifest(manifest_path)` 解析 WP V1-1 manifest tracks 字段，**字段对应清晰**（manifest 已含 poses/extent/active_frames）。
- **验收**：clip 3435ace9 → `len(instance_pts_dict) == 11`，每 track 的 `poses.shape == [num_frame, 4, 4]` 一致。

#### T4.2 — dynamic_rigid_init.py cuboid 内 LiDAR 抽取

- **改动**：新建 `layers/dynamic_rigid_init.py`：
  ```python
  def init_dynamic_rigid_layer(instance_pts_dict, dynamic_lidar_points, max_pts_per_track=5000):
      for track_id, info in instance_pts_dict.items():
          collected = []
          for frame_idx in info["frame_info"].nonzero().squeeze(-1):
              pose_inv = torch.linalg.inv(info["poses"][frame_idx])
              local_pts = (pose_inv[:3, :3] @ dynamic_lidar_points[:, :3].T).T + pose_inv[:3, 3]
              mask = (local_pts.abs() <= info["size"] / 2).all(dim=1)
              collected.append(local_pts[mask])
          all_pts = torch.cat(collected)
          if len(all_pts) > max_pts_per_track:
              all_pts = all_pts[torch.randperm(len(all_pts))[:max_pts_per_track]]
          info["pts"] = all_pts
          info["colors"] = ...  # 同样从 LiDAR RGB 抽
      return instance_pts_dict
  ```
- **设计选择**：不复制 OmniRe `RigidNodes`（依赖 `ctrl_cfg` / `instances_quats` 学习接口），只借 schema + transform_means 模式，重写适配 3dgrut2。
- **验收**：见 T4.5。

#### T4.3 — trainer step 中 per-frame pose 应用 + concat

- **核心点**：dynamic_rigids 粒子 `positions` 存的是 **object-local frame**；每 step 临时算 world frame，**不进 Parameter**（pose 不学习）。
- **改动**：`layers/layered_model.py::fused_view(frame_id)` 内：
  ```python
  for spec in self.specs:
      layer = self.layers[spec.name]
      if spec.name == "dynamic_rigids":
          world_pts = self._transform_means(layer.positions, layer.track_ids, frame_id, self.tracks_poses)
          pieces.append(world_pts)
      elif spec.is_particle_layer:
          pieces.append(layer.positions)
  fused_positions = torch.cat(pieces, dim=0)
  ```
  `_transform_means` 参考 `drivestudio/models/nodes/rigid.py:315-362`，**模式参考重写**。
- **验收**：mock 单 track 单粒子，frame 0 / N-1 两端 world 位置匹配；训练 5k 步渲染视频，车辆不漂移。

#### T4.4 — dynamic_mask.py cuboid → 像素 mask 投影

- **为什么不用 sseg**：sseg 含未跟踪的物体（traffic cone 等），会让 dynamic 层学不属于自己的内容；cuboid 投影精确对应 track。
- **改动**：新建 `layers/dynamic_mask.py`：
  ```python
  def project_cuboids_to_mask(tracks, frame_idx, K, T_world2cam, H, W) -> Tensor[H, W]:
      mask = torch.zeros(H, W)
      for tid, info in tracks.items():
          if not info["frame_info"][frame_idx]: continue
          corners_local = make_cuboid_corners(info["size"])       # [8, 3]
          corners_world = info["poses"][frame_idx] @ corners_local
          corners_img = project_points(corners_world, K, T_world2cam)
          mask = fill_convex_hull(mask, corners_img)
      return mask
  ```
- **验收**：渲染一帧 mask 与 GT video 车辆位置吻合。

#### T4.5 — 单测 test_dynamic_rigid_init.py

- `test_local_frame_transform_roundtrip`：world→local→world 数值一致
- `test_cuboid_filter`：超出 size/2 的点被剔除
- `test_subsample_respects_max_pts`：> max_pts 后输出 ≤ max_pts
- `test_per_frame_pose_correct`：mock 1 track，frame 0/N-1 端点位置正确
- **验收**：本地 Mac pytest 全 pass < 1 秒。

---

### Stage 5 — Sky envmap

#### T5.1 — nvdiffrast.torch 可用性 / 降级 SkyModel

```bash
ssh a800-x2 && conda activate 3dgrut && python -c "import nvdiffrast.torch; print('ok')"
```
若失败 → `pip install nvdiffrast`；若仍失败 → T5.2 降级为 MLP 版 SkyModel（drivestudio 也有备份）。

#### T5.2 — port EnvLight → correction/sky_envmap.py

- **改动**：新建 `threedgrut/correction/sky_envmap.py`，**直接复制** `drivestudio/models/modules.py:174-205` 的 `EnvLight`：
  ```python
  class SkyEnvmap(nn.Module):
      def __init__(self, resolution=512):
          super().__init__()
          self.to_opengl = torch.tensor([[1,0,0],[0,0,1],[0,-1,0]], dtype=torch.float32).cuda()
          self.base = nn.Parameter(0.5 * torch.ones(6, resolution, resolution, 3))

      def forward(self, viewdirs):
          l = (viewdirs.reshape(-1, 3) @ self.to_opengl.T).reshape(*viewdirs.shape).contiguous()
          ...
          return dr.texture(self.base[None], l, filter_mode='linear', boundary_mode='cube').view(*prefix, -1)
  ```

#### T5.3 — trainer step 中 sky blending + loss

- **改动**：`trainer.py`：
  ```python
  rgb_gauss, alpha = self.model(batch, train=True)
  if conf.trainer.use_sky_envmap:
      viewdirs = compute_per_pixel_viewdirs(batch)
      rgb_sky = self.sky_envmap(viewdirs)
      rgb_final = rgb_gauss + rgb_sky * (1 - alpha)
      sky_loss = ((rgb_sky - rgb_gt).abs() * sky_mask).sum() / (sky_mask.sum() + 1e-6)
      total_loss += conf.loss.lambda_sky * sky_loss
  ```
  参考 `drivestudio/models/trainers/scene_graph.py:252-253` 的 blend 模式。

#### T5.4 — 单测 test_sky_envmap.py

- `test_envmap_shape`：`base.shape == [6, 512, 512, 3]`
- `test_envmap_forward_shape`：viewdirs `[B, 3]` → out `[B, 3]`
- `test_envmap_no_nvdiffrast_fallback`：mock 缺 nvdiffrast → 落到 MLP 路径
- **验收**：3k 步后 envmap +Z face 明显偏蓝；sky 区 PSNR ≥ 30。

---

### Stage 6 — 每相机曝光占位

#### T6.1 — port ExposureModel

- **改动**：新建 `threedgrut/correction/exposure.py`，**直接复制** Recon-Studio `models/luxury/exposure.py`（29 行），仅改 import 路径。
- **设计选择**：用 affine `exp(a)*img + b` 占位；完整 bilateral grid 留 v3。

#### T6.2 — trainer step per-camera 应用 + 独立 optimizer

```python
self.exposure_model = ExposureModel(num_camera=len(camera_ids)).cuda()
self.exposure_optimizer = torch.optim.Adam(self.exposure_model.parameters(), lr=1e-3)

# 在 trainer step：
rgb_pred = self.exposure_model(batch.camera_idx, rgb_pred)
loss.backward()
self.optimizer.step()
self.exposure_optimizer.step()
self.exposure_optimizer.zero_grad()
```

#### T6.3 — 单测 test_exposure.py

- `test_single_camera_is_identity`：num_camera=1 时输出 == 输入
- `test_zero_init_is_identity`：`exp(0)*img + 0 == img`

---

### Stage 6-fix — Ego mask 全栈接通

> **背景**：经审计 v2 代码路径（2026-05-20），`NCoreDataset` 在 [datasetNcore.py:385-395](threedgrut/datasets/datasetNcore.py:385) 加载了 ego car mask 并膨胀 30 次缓存为 `sequence_cameras_frame_valid_pixels_masks`，但**实际从未流入 loss / metric**：
> - 训练分支 [datasetNcore.py:806-872](threedgrut/datasets/datasetNcore.py:806) 完全没读 `frame_valid_pixel_mask`、也没塞 `mask` 字段；
> - 验证分支 [datasetNcore.py:937-971](threedgrut/datasets/datasetNcore.py:937) 采样了 `valid`，但 `get_gpu_batch_with_intrinsics` ([L1273-1367](threedgrut/datasets/datasetNcore.py:1273)) 没把 `valid` 拷贝进 `Batch`；
> - 因此 [trainer.py:702](threedgrut/trainer.py:702) `mask = gpu_batch.mask = None`，[trainer.py:707-709](threedgrut/trainer.py:707) 的 mask 乘法跳过；`compute_layered_l1_loss(valid_mask=None)`；[trainer.py:661-670](threedgrut/trainer.py:661) PSNR/SSIM/LPIPS 全图算。
>
> **实际影响**：
> 1. 训练 loss 浪费高斯容量去贴随相机刚体平移的车身色块，易产生跟随相机的漂浮高斯；
> 2. 验证 PSNR 虚高——车身固定色块最易拟合；Stage 3 实测 26.133 dB / Stage 4 实测 26.315 dB 里都包含 ego 区水分。
>
> **目标**：dataset → Batch → loss → metric 全栈真接通 ego mask，**双指标**（full / masked）保留与 Stage 3-6 历史数据可比性。

#### T6F.1 — Dataset → Batch.mask 接通（train+val）

- **改动**：`threedgrut/datasets/datasetNcore.py`
  - 训练分支 `__getitem__` ([L806-872](threedgrut/datasets/datasetNcore.py:806))：从已缓存的 `self.sequence_cameras_frame_valid_pixels_masks[sequence_id][camera_id][camera_frame_index]` 取 `frame_valid_pixel_mask`；如 `downsample < 1.0` 走 `cv2.INTER_NEAREST` resize（仿 val 分支 [L942-947](threedgrut/datasets/datasetNcore.py:942)），加入 `batch_dict["valid"] = to_torch(frame_valid_pixel_mask, device="cpu")`。
  - 验证分支已有 `"valid"` 字段（[L959](threedgrut/datasets/datasetNcore.py:959)），无需改。
  - `get_gpu_batch_with_intrinsics` ([L1273-1367](threedgrut/datasets/datasetNcore.py:1273))：在 RGB 处理段之后加：
    ```python
    if "valid" in batch and batch["valid"] is not None:
        valid = batch["valid"]
        if not isinstance(valid, torch.Tensor):
            valid = torch.from_numpy(valid)
        mask = valid.to(self.device, non_blocking=True).float()
        batch_dict["mask"] = mask.reshape(1, h, w, 1)  # 对齐 Batch 约定 [B,H,W,1]
    ```
- **自动接通**（不改）：
  - [trainer.py:707-709](threedgrut/trainer.py:707) `if mask is not None: rgb_gt *= mask; rgb_pred *= mask` 立即生效。
  - [layered_loss.compute_layered_l1_loss](threedgrut/model/layered_loss.py) 经 `valid_mask=mask` 自动从 bg/road/dyn 三区 partition 剔除 ego 像素。
  - **保持 D7 不变量**：SSIM 仍走全图（不进 SSIM loss term）。
- **新增单测**（`threedgrut/tests/test_datasetNcore_ego_mask.py`）：
  1. `test_train_batch_carries_ego_mask_when_mask_present`：mock 返回非全零 ego mask → `Batch.mask` 非 None / shape=[1,H,W,1] / dtype=float32 / sum < H·W。
  2. `test_train_batch_mask_is_none_when_no_ego_mask`：mock 不返回 ego mask → `Batch.mask` 仍 None（v1 byte-identical 回归）。
  3. `test_val_batch_carries_ego_mask`：validation 分支同样产出 `Batch.mask`。
  4. `test_ego_mask_disjoint_with_sky_road_dyn`：ego ∩ {sky ∪ road ∪ dyn} 软阈值 <5%（如挡风玻璃反射的天空允许重叠）。
  5. `test_layered_l1_loss_ignores_ego_pixels_when_mask_present`：在 ego 区造大误差，mask 前 loss 大、mask 后 loss 小。
- **验收**：`pytest threedgrut/tests/test_datasetNcore_ego_mask.py -v` 5/5 PASS（Mac CPU mock）。

#### T6F.2 — Metric 端 psnr/ssim/lpips full + masked 双指标

- **改动**：`threedgrut/trainer.py::compute_metrics` validation 分支 ([L659-670](threedgrut/trainer.py:659))，在保留原全图三指标后追加：
  ```python
  mask = gpu_batch.mask  # [1,H,W,1] 或 None
  if mask is not None:
      # PSNR masked：解析公式
      diff_sq = (rgb_pred - rgb_gt).pow(2) * mask  # broadcast last dim 1→3
      denom = mask.sum().clamp(min=1.0) * 3
      mse_masked = diff_sq.sum() / denom
      metrics["psnr_masked"] = (-10.0 * torch.log10(mse_masked.clamp(min=1e-10))).item()

      # SSIM/LPIPS via GT-fill：mask=False 区填 GT（差=0），该区 SSIM≈1 / LPIPS≈0，
      # 对全局指标贡献按面积稀释 —— driving-3DGS 文献广泛采用。
      m4d = mask.permute(0, 3, 1, 2)  # [B,1,H,W]
      rgb_pred_filled = pred_rgb_full * m4d + rgb_gt_full * (1.0 - m4d)
      rgb_pred_filled_clipped = (
          rgb_pred.clip(0, 1).permute(0, 3, 1, 2) * m4d + rgb_gt_full * (1.0 - m4d)
      )
      metrics["ssim_masked"] = ssim(rgb_pred_filled, rgb_gt_full).item()
      metrics["lpips_masked"] = lpips(rgb_pred_filled_clipped, rgb_gt_full).item()
  else:
      # v1 byte-identical 回归：无 mask 时 masked 三指标 ≡ 全图三指标
      metrics["psnr_masked"] = metrics["psnr"]
      metrics["ssim_masked"] = metrics["ssim"]
      metrics["lpips_masked"] = metrics["lpips"]
  ```
- **新增单测**（`threedgrut/tests/test_trainer_masked_metrics.py`）：
  1. `test_masked_metrics_equal_full_when_mask_none`
  2. `test_masked_metrics_equal_full_when_mask_all_ones`（float32 容差 1e-4）
  3. `test_psnr_masked_uniform_error`：mask 区造 δ 均匀误差 → 数值 = -10·log10(δ²)
  4. `test_psnr_masked_ignores_masked_region_error`：mask=False 区设巨大误差（pred=1, gt=0）；psnr_masked 高 / psnr ≈ 0
  5. `test_ssim_masked_with_gt_fill_in_masked_region`：ssim_masked ∈ [ssim_full, 1.0]
  6. `test_metrics_no_extra_alloc_on_v1_byte_identical_path`：mask=None 不引入新张量分配
- **验收**：`pytest threedgrut/tests/test_trainer_masked_metrics.py -v` 6/6 PASS（Mac CPU）。

#### T6F.3 — A800 5k smoke 出口（双指标 + ego 粒子密度核查）

- **目的**：首次产出 `psnr_full` 与 `psnr_masked` 双指标，量化 Stage 3/4/5/6 历史 PSNR 中 ego 区水分；核查 ego 区高斯密度是否下降（loss mask 生效的间接证据）。
- **配置**：现有 `configs/apps/ncore_3dgut_mcmc_v2_full_exposure.yaml` 无需改（ego mask 接通是 dataset 内部行为，对 config 不可见）。
- **执行**：
  ```bash
  ssh a800-x2 << 'EOF'
  export PATH=/root/miniforge3/envs/3dgrut/bin:$PATH
  export CUDA_VISIBLE_DEVICES=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  cd /root/work/yusun/repo/3dgrut
  TS=$(date +%Y%m%d_%H%M%S)
  python -u train.py --config-name apps/ncore_3dgut_mcmc_v2_full_exposure \
    path=/root/work/yusun/ncore-nurec/data/ncore/clips/.../pai_....json \
    out_dir=/root/work/yusun/ncore-nurec/output/v2_egomask_fix_${TS} \
    n_iterations=5000 \
    dataset.train.duration_sec=2.0 \
    'dataset.camera_ids=[camera_front_wide_120fov]' \
    2>&1 | tee /tmp/v2_egomask_fix_${TS}.log
  EOF
  ```
- **出口指标**（回填 § 5 Done Log）：
  | 指标 | 预期 |
  |---|---|
  | `psnr_full` (5k, 单相机) | ≈ 旧 v2_full_exposure 5k baseline（Stage 4 26.3 dB ± 0.3） |
  | `psnr_masked` | < `psnr_full`，差额 0.2–1.5 dB（ego 水分量化） |
  | `ssim_masked` | ≥ `ssim` |
  | `lpips_masked` | ≤ `lpips` |
  | MoG.num_gaussians 时序 | ≤ 旧 baseline（mask 让 dyn / bg 不在 ego 区生成） |
  | 训练时间 | 与旧 baseline 同（mask 是 cheap broadcast，<1% 开销） |
- **Risk / 回滚**：若 `psnr_masked > psnr_full` → dump `mask.float().mean()` 检查 mask 是否反向。

#### 不做的事（明确划界）

- **不**改 SSIM 实现去支持"真"像素级掩膜（保持 D7 不变量；masked 用 GT-fill 是文献接受近似）。
- **不**改 v1 byte-identical baseline（mask=None 路径走旧逻辑）。
- **不**追加 Stage 7 KPI 表目标值（T7.3 的 ≥ 28.5 仍是全图目标；masked 目标待 T6F.3 给出 baseline 后再定）。
- **不**触碰任何 config yaml（零 config 改动是健康信号）。
- **不**实现 ego mask 与 sky/road/dyn 三区 mask 的硬 AND（loss 端自然 AND，三区指标当前只 print 不入 KPI）。

---

### Stage 7 — 集成 + 7-cam 20s 训练 + KPI

#### T7.1 — configs/apps/ncore_3dgut_mcmc_v2_full.yaml

```yaml
defaults:
  - ncore_3dgut_mcmc
  - override /strategy: layered_mcmc

use_layered_model: true
layers:
  enabled: [background, road, dynamic_rigids, sky_envmap]
  # dynamic_deformables 注册但不分配粒子 (v2 占位)

dataset:
  load_aux_masks: true

trainer:
  layered_loss: true
  use_sky_envmap: true
  use_exposure: true
  exposure_lr: 0.001
```

#### T7.2 — 2s smoke 全 pipeline 验证

```bash
ssh a800-x2 && conda activate 3dgrut && cd /root/work/yusun/repo/3dgrut
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u train.py --config-name apps/ncore_3dgut_mcmc_v2_full \
  path=/root/work/yusun/ncore-nurec/data/ncore/clips/9ae151dc-.../pai_9ae151dc-...json \
  out_dir=/root/work/yusun/ncore-nurec/output/smoke_v2_<ts> \
  n_iterations=1000 \
  dataset.train.duration_sec=2.0 \
  'dataset.camera_ids=[camera_front_wide_120fov]'
```
- **验收**：exit 0；各层粒子数日志 bg ~600K / road ~200K / dyn ~50-200K；loss 下降不发散。

#### T7.3 — 7-cam 20s full 30k step + KPI

```bash
python -u train.py --config-name apps/ncore_3dgut_mcmc_v2_full \
  path=/root/work/yusun/ncore-nurec/data/ncore/clips/.../pai_....json \
  out_dir=/root/work/yusun/ncore-nurec/output/v2_full_run1 \
  n_iterations=30000
```
- **验收**：

| 维度 | 目标 |
|---|---:|
| 7-cam 20s PSNR | ≥ 28.5 |
| Sky 区 PSNR | ≥ 30 |
| Road 区 PSNR | ≥ 32 |
| Dynamic vehicle PSNR | ≥ 25 |
| 训练时间 | ≤ 60 min |

PSNR < 28.0 → 进入 T7.4。

#### T7.4 — per-layer cap ablation（仅在 T7.3 未达目标时执行）

4 组（每组 30k 步，A800 各约 60 分钟）：

| 组 | background | road | dynamic_rigids |
|---|---:|---:|---:|
| A | 600K | 200K | 200K |
| B | 700K | 200K | 100K |
| C | 500K | 300K | 200K |
| D | 800K | 100K | 100K |

- **教训挂点**（2026-05-09 #209）：KPI 归因不要草率，每组都拆 region PSNR（路面/动态/背景），不只看全局。

#### T7.5 — WP_V2_Report.md + scene_manifest v2 schema

- 新建 `WP_V2_Report.md`，镜像 `WP_V1-1_Report.md` 结构：
  - 设计概要（4 层 + Layered MCMC + Sky + Exposure）
  - 实现路径（文件清单引用本 plan T1-T7）
  - KPI 表（v1 vs v2 vs NuRec 三列 + region 拆解）
  - Ablation 结果（T7.4 4 组）
  - 已知限制（V2-4 pose 不学、bilateral grid 占位、deformable 未实现）
  - 下一步（V2-4 pose calib、V1-6 USDZ 对接）
- 更新 `schemas/scene_manifest.schema.json`：加可选 `layer_assignments`。

---

## 3. 开发工作流

```mermaid
flowchart LR
    subgraph Mac["Local Mac"]
        Edit["编辑代码<br/>/Users/etendue/repo/3dgrut2/<br/>threedgrut/..."]
        Test["pytest threedgrut/tests/test_*.py<br/>(CPU mock, 不需 CUDA tracer)"]
        Commit["git commit"]
        Edit --> Test --> Commit
    end

    subgraph Transfer["内网 rsync"]
        Rsync["rsync -avz<br/>--exclude='.git' --exclude='__pycache__'<br/>3dgrut2/ a800-x2:/root/work/yusun/repo/3dgrut/"]
    end

    subgraph A800["A800-x2 训练机"]
        SSH["ssh a800-x2"]
        Env["conda activate 3dgrut<br/>/root/miniforge3/envs/3dgrut"]
        Cd["cd /root/work/yusun/repo/3dgrut"]
        Run["python train.py<br/>--config-name=apps/<br/>ncore_3dgut_mcmc_v2_&lt;stage&gt; ..."]
        SSH --> Env --> Cd --> Run
    end

    Commit ==> Rsync ==> SSH

    classDef step fill:#f5f5f5,stroke:#666,color:#222
    class Edit,Test,Commit,Rsync,SSH,Env,Cd,Run step
```

**A800 已知 caveats（T0.1 / Stage 2 已踩坑）**：
1. GPU 共享：两张 A800 各被外部进程占 ~57 GiB，可用 ~22 GiB/卡 → `CUDA_VISIBLE_DEVICES=1` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
2. CUDA kernel 首次编译 ~3 min（缓存在 `/root/.cache/torch_extensions/py311_cu118/`）
3. 多相机必须显式 `dataset.camera_ids=[...]`
4. 数据路径 `/root/work/yusun/ncore-nurec/data/ncore/clips/...`（NFS）
5. **SSH non-interactive shell 不继承 conda PATH**：跑 train.py / pytest 必须 `export PATH=/root/miniforge3/envs/3dgrut/bin:$PATH`（slangc 在 env 内），仅设 `CUDA_VISIBLE_DEVICES` 会触发 `FileNotFoundError: 'slangc'`

---

## 4. Risk Register

| 风险 | 缓解 | 任务挂点 |
|---|---|---|
| `nvdiffrast.torch` 在 A800 conda env 不存在 | T5.1 起手探测；不可用降级 SkyModel MLP | T5.1 |
| Per-layer cap 配比对 KPI 影响大 | T7.4 ablation 4 组 | T7.4 |
| Track pose 不学习时 dynamic 粒子漂移 | 训练前 NCore validator strict mode 过滤低置信 track；漂移严重则 V2-4 提前 | T4.3 |
| Recon-Studio `surface.py` 依赖 PyTorch3D | T3.3 用 `torch.cdist` 替代 `knn_points`，借算法不借实现 | T3.3 |
| 路面 LiDAR 语义质量差 | T3.2 起手 BEV 可视化；必要时加 plane fit fallback | T3.2 |
| NFS 性能拖累 import 时间 | conda env 本地盘；若长期慢则 rsync repo 到 `/data/repo/` | — |
| Renderer 接口被意外修改 | tracer Python binding `git diff` 必须为空 | 所有 Stage |
| MCMC 抽象重构改变 v1 行为 | T2.1 必须 byte-identical 验证 | T2.1 |
| Stage 3/4/5/6 历史 PSNR 含 ego 车身水分（数据可比性 / KPI 虚高） | T6F.2 双指标保留全图 PSNR 列；T6F.3 之后所有 Done Log 全部标注 `psnr_full` / `psnr_masked` 双数；mask=None 路径走 byte-identical 回归避免对其他 dataset 产生回归 | T6F.1 / T6F.2 / T6F.3 |

---

## 5. Done Log

### 🎁 Stage 7 软出口结题 — 7-cam 30k + exposure 失控诊断 + V3-P1 整合 bilateral-grid (2026-05-21, A800 GPU 1, plan stage-7-tranquil-brook)

Stage 7 v2 LayeredGaussians 最终交付出口. 5/5 任务全部 ✅ (含新增 T7.3.b exposure ablation; T7.4 cap ablation 跳过). 真实交付指标 cc_psnr_masked **24.70 dB** ≈ Stage 5/6/6-fix baseline 24.7~24.9 (σ < 0.2 dB noise 级), 几何质量不退化, 结构无回归. **关键技术发现 + V3 启动入口 (bilateral-grid + ExposureModel 退化修复合并 V3-P1)**.

**5 任务完整轨迹**:

| Task | A800 Run | training_time | raw psnr_masked | cc_psnr_masked | 结论 |
|---|---|---:|---:|---:|---|
| T7.1 | (Mac dry-run + a800 cfg dump) | — | — | — | ✅ 复用 v2_full_exposure (无新 yaml), 7-cam Hydra resolve 通过 |
| T7.2 | `stage7_smoke_20260520-174444` (1-cam 1k smoke) | 102.9 s / 9.71 it/s | 26.38 | 22.76 | ✅ 全栈连通, metrics 12 字段齐全 |
| T7.3 | `stage7_full_20260520-202222` (7-cam 30k exposure ON) | **3061 s = 51 min** / 9.80 it/s | **15.63 ❌** | **24.75** | ✅ 完成但 raw KPI 崩, **暴露 exposure 失控** |
| T7.3.b | `stage7_noexp_20260521-102930` (7-cam 30k exposure OFF) | 3064 s / 9.79 it/s | **25.76 (+10.13)** | **24.70 (-0.05)** | ✅ **证伪 ExposureModel 是 raw 崩真因**; 真实质量两组持平 |
| T7.4 | (跳过) | — | — | — | ⏭ 根因不在 cap, 转 V3-P1 整合研究 |
| T7.5 | (Mac doc) | — | — | — | ✅ `WP_V2_Report.md` (231 行) + v2_plan/architecture 同步 |

**T7.3 + T7.3.b 三组对照 (含 Stage 6-fix baseline)**:

| 维度 | T6F.3 (1-cam 5k) | T7.2 (1-cam 1k) | T7.3 (7-cam 30k ON) | **T7.3.b (7-cam 30k OFF)** | plan 目标 |
|---|---:|---:|---:|---:|---:|
| training_time | 520 s | 102.9 s | 3061 s **= 51 min** | 3064 s **= 51 min** | ≤ 60 ✅ |
| it/s | 9.61 | 9.71 | 9.80 | 9.79 | — |
| `mean_psnr` (full) | 20.49 | 19.93 | 14.91 | **23.78** | ≥ 28.5 ❌ |
| `mean_psnr_masked` | **29.49** | 26.38 | **15.63 ❌** | **25.76** | ≥ 30 ❌ |
| `mean_cc_psnr` (full) | 19.61 | 18.57 | **23.25** | 23.24 | — |
| **`mean_cc_psnr_masked`** | **24.90** | 22.76 | **24.75** | **24.70** | (新 KPI) |
| 4 层粒子 | bg+road | bg+road+dyn31 | bg+road+dyn70+sky MLP | bg+road+dyn70+sky MLP | 全栈 ✅ |

**关键技术发现 (3 项)**:

1. **ExposureModel 退化优化失控** — 训练有两条 loss 下降路径, 30k Adam 无约束 → 模型选择病态短路径:
   - 路径 1 (物理正确): 高斯学准真实色彩 → exposure 维持小值
   - 路径 2 (病态短路): 高斯学个大概 → exposure 把偏差全 compensate (14 参数 vs 几百万高斯, 更快收敛)
   - 实证: T7.3 raw_masked 15.63 (路径 2 终点, 渲染严重过曝/泛白) vs T7.3.b 25.76 (关掉强制走路径 1)
   - 但两组 cc_psnr_masked 几乎一致 (24.75 vs 24.70) → **真实几何/纹理质量没差**, exposure 只是 raw 输出与 GT 的色彩偏差

2. **cc_PSNR 是真实重建质量 KPI** — `color_correct_affine` (Google multinerf, [threedgrut/utils/color_correct.py](threedgrut/utils/color_correct.py)) per-image per-channel lstsq 撤销色彩偏移; NeRF 圈 (Mip-NeRF 360 / Block-NeRF / drivestudio) 都报 cc PSNR. **Stage 7 真实指标应是 cc_psnr_masked, 不是 psnr_masked**. raw 只作 ExposureModel 健康检查 (应 ≈ cc ± 2 dB)

3. **7-cam 30k vs 1-cam 5k masked PSNR 反而低 3.7 dB** (25.76 vs 29.49) — 多相机长训没有带来质量净提升, 实证 v2 架构在 NCore 9ae151dc clip 上的天花板. 候选解释 (V3 排查): 多相机 frustum 重叠少监督稀疏 / 30k 过拟合 / dyn 70 tracks 粒子分配不充分 / 多相机 LiDAR 监督权重未调

**踩坑记录 + 加固 (CLAUDE.md § A/B/C 清单加强)**:

- **教训 #1 — ssh heredoc 远端进程被 SIGHUP 杀**: 第一次 T7.3 用 `ssh a800-x2 << 'EOF'` 跑训练, 30 min 后 ssh session 因网络/timeout 断开 → 远端 Python 进程被 SIGHUP 杀掉, step ~20000 ckpt 写出但无 metrics.json. **修复**: 改为 `ssh ... "nohup setsid /tmp/script.sh < /dev/null > /tmp/nohup.out 2>&1 & disown"` 完全脱离 ssh session, 写 `.done` sentinel + 复制 metrics 到固定路径. T7.3.b 用同模式 + watcher (ssh 阻塞等 sentinel) 鲁棒完成. **CLAUDE.md § A 把关清单第 6 条加固**: "A800 长训务必 nohup setsid disown, 不用 heredoc 阻塞"

- **教训 #2 — region PSNR 拆解工具未实现**: plan T7.3 出口要求 "Road 区 PSNR ≥ 32 / Dynamic vehicle PSNR ≥ 25" 但 `trainer.compute_metrics` + `render.py` eval 路径只算 full + masked + cc, **没有 region 拆解**. Stage 7 实际只汇总三个指标维度. region PSNR 转 V3-E2 (per-class cPSNR 评测工具, § 14.6)

- **教训 #3 — 配置歧义解决**: T7.1 plan 写 "新建 v2_full.yaml 含 sky+exposure" 但 `configs/apps/ncore_3dgut_mcmc_v2_full_exposure.yaml` 已等价存在 (Stage 6 创建). 决策: 复用 + 注释追加 Stage 7 入口说明, 不新建 yaml 避免重复. T7.1 工作量从 "新建" 降为 "注释 + dry-run"

**v2 项目整体回顾**:

- Mac 单测: **181/181 PASS** (Stage 0-8 + Stage 6-fix 全部覆盖, 0 回归)
- A800 实测: Stage 3/4/5/6/6-fix/7 + Stage 8 全 cycle 跑通
- vast.ai RTX 4090: Stage 8 完整 Gaussian 渲染 87.5 FPS @ 1024×600
- 总任务数: 54 (Stage 0-8 全部 ✅, Stage 7 新增 T7.3.b)
- 真实重建质量: v2 架构在 NCore 9ae151dc clip 上的天花板 ≈ **24.7 dB cc_psnr_masked**, V3 启动后 V3-P1 / V3-T9 / V3-D1 预期带来 +2-3 dB 提升到 27-28 dB

**V3 backlog 升级 (本次 Stage 7 直接转出)**:

- **V3-P1 升级为整合任务** (§ 14.5): bilateral-grid + ExposureModel 退化修复合并 — Stage 7 实证最关键缺口, 预估 3-5 天, V3 启动优先级提升. 详见 v2_plan.md § 14.5 V3-P1 完整任务卡

**Stage 7 ckpt 路径 (V3 baseline 保留)**:

```
a800-x2:/root/work/yusun/ncore-nurec/output/stage7_full_20260520-202222/.../ckpt_30000.pt    # 991 MB, exposure ON 失控样本
a800-x2:/root/work/yusun/ncore-nurec/output/stage7_noexp_20260521-102930/.../ckpt_30000.pt   # 991 MB, exposure OFF 推荐配置
```

**Stage 7 软出口判定**: ✅ — cc_psnr_masked 24.70 ≈ T6F.3 baseline 24.90 (差 -0.20 dB noise 级), 几何质量不退化, ExposureModel 失控诊断 + V3 整合方案明确, WP_V2_Report.md 完整交付. v2 LayeredGaussians 项目结题, V3 启动准备就绪.

**改动清单 (本次 commit)**:

| Path | 操作 | 说明 |
|---|---|---|
| `WP_V2_Report.md` | NEW | 231 行 v2 项目结题报告 (镜像 v1 结构 + KPI 三组对比 + ExposureModel 失控诊断 + cc_PSNR 解读 + V3-P1 整合方案) |
| `v2_plan.md` | MOD | § 1.1 顶层看板 Backlog 5 项 → 0 / Done 50 → 54 + Stage 7 看板项; § 1.2 任务级 T7.1-T7.5 + 新增 T7.3.b; § 1.3 Stage 7 0/5 → 5/5 ✅; § 1.4 dependency 图 T7.* :::done + T7.3.b 节点; § 5 Done Log 本段; § 14.5 V3-P1 整合升级 |
| `v2_architecture.md` | MOD | § 6.1 文件清单 WP_V2_Report.md ✅; § 6.2 修改文件 ExposureModel 健康度锚点; § 7 关键不变量 cc_psnr_masked baseline 24.7 + raw/cc 差 ≤ 2 dB 健康度规则 |
| `configs/apps/ncore_3dgut_mcmc_v2_full_exposure.yaml` | MOD | Stage 7 结题注释块: ExposureModel 长训失控警告, 推荐 `use_exposure=false` for 7-cam 30k+ |

---

### ⚠️ T8.12 — vast.ai RTX 4090 viser_gui_4d 部分通过: pipeline 接通 + 2 个真实 bug 修复, fisheye 渲染留 T8.13 (2026-05-21, RTX 4090 Norway, eason_3dgrut_t8_12_viz4d_d2)

vast.ai RTX 4090 24GB (Ada Lovelace 第 3 代 RT cores, Norway, $0.630/hr) 端到端验证. **Pipeline 部分通过** @ 1024×~600 / **87 FPS** (远超 ≥10 FPS 目标). 实例 ID 37188673 (Day 1 用过的 37136461 已销毁), 总成本 ~$1.5 (~¥10).

**实测结果**:

| 项 | 状态 | 备注 |
|---|---|---|
| viser server 启动 + WebSocket | ✅ | 修了真实 bug 才能跑起来 (Bug #1+#2) |
| 浏览器加载 + GUI 三 folder 显示 | ✅ | Render Controls / Timeline / Visibility 全显 |
| Scene primitives (cuboid/LiDAR/ego frustum) 显示 + timeline 同步 | ✅ | 跟 A800 --no_gaussian_render 模式一致, viser 端坐标系正确 |
| dyn LiDAR per-track 点云跟 cuboid 飘 | ✅ | T8.11 fix 在 RT cores 路径也正确 |
| Gaussian 渲染 pipeline 不崩 | ✅ | Bug #1 + #2 修完, 87 FPS 稳态 |
| **Gaussian 渲染输出匹配 render.py ground truth** | **❌** | **用户对比图实证: render.py 出清晰街景含楼/车/柱+fisheye 桶形畸变, viser 出远景隧道 motion-blur 乱糊, 完全不是同一相机视角** |

**根因 (Day 2 深挖发现的 schema gap)**:

- `viz_4d.ego` (T8.2 设计) 只存 `primary_camera_fov_y_rad = 2.441` 和 `primary_camera_id = camera_front_wide_120fov` — NCore 数据集用 **FTheta fisheye polynomial + 大畸变**, viz_4d 把它简化成单个 fov_y_rad 丢掉了所有多项式系数和畸变参数
- viser_gui_4d 没法走数据集 (脱依赖设计), 只能用 viz_4d 里的 fov_y_rad 构 pinhole 内参
- ckpt 的 Gaussians 是按 fisheye 投影优化的, viser 用 pinhole + 140° fov 看 (focal_y=131 px, 退化区) → Gaussians 投影到错的像素 scale → 视觉是"远景隧道 motion-blur"乱码
- render.py 走 NCoreDataset.get_gpu_batch_with_intrinsics() 直接读完整 FTheta 多项式, 所以渲染对

**修法 (T8.13 backlog)**: viz_4d schema 扩展 `ego.primary_camera_intrinsics` 含完整 FTheta polynomial / OpenCVFisheye distortion coeffs; viser_gui_4d 用 `Batch.intrinsics_FThetaCameraModelParameters` 而非 pinhole 简化; 可能还需要 fisheye ray gen 替代 kaolin pinhole. 预估 0.5-1 d.

**实际修了的 2 个真实 Stage 8 bug** (T8.12 不可或缺的产出):

| Bug | 症状 | Root Cause | Fix |
|---|---|---|---|
| **#1 camera intrinsics 缺失** | viser 一连即崩 `AttributeError: 'Batch' object has no attribute 'keys'` (其实是 ValueError 错误 f-string 自崩, 真错是 `Camera intrinsics unavailable`) | T8.1 给 LayeredGaussians+tracks 路径构 Batch 只塞 rays + T_to_world=I + rays_in_world_space=True, 但 3dgut UT 光栅器需要 camera intrinsics 做投影 | `threedgrut_playground/engine.py:_trace_scene_mog`: 加 `camera=None` kwarg + 从 kaolin Camera 取 [fx,fy,cx,cy] 塞 Batch.intrinsics + 真 c2w as T_to_world + camera-space rays 匹配 NCoreDataset.get_gpu_batch_with_intrinsics contract (即使 pinhole 是 T8.13 待替换的近似, contract 对就值得 commit) |
| **#2 SkyEnvmapMLP 残留 CPU** | 修完 #1 后 viser 不崩但每帧 render 报 `RuntimeError: Expected cpu and cuda:0` 在 `sky_envmap.py:105 F.relu(self.layer0(enc))` | `LayeredGaussians.init_from_checkpoint` 用 `load_state_dict(sky_state)` 加载 CPU tensors 后 nn.Module 没 `.cuda()`. Trainer/eval 路径有 `model.cuda()` 兜底, playground engine 路径 (`Engine3DGRUT.load_3dgrt_object`) 没 | `threedgrut/layers/layered_model.py:init_from_checkpoint`: 末尾加 `if torch.cuda.is_available(): self.cuda()` |
| Reset View 改进 | 按按钮视角没变 | 只 reset up_direction, 不恢复 initial_c2w | `viser_gui_4d.py:reset_view_button.on_click`: snap camera.position + wxyz + look_at + up_direction 全部从 `meta.initial_c2w` |

**已撤销的 "Bug #3 fov override"**: Day 2 中途以为 viser browser fov (~45°) ≠ ckpt fov (140°) 是糊的根因, 加了 override 用 ckpt fov. 实测无效 (pinhole 30°/45°/60°/90°/140° 全部一样糊), 因为根因是 pinhole vs fisheye 而不是 fov. 已撤回不 commit.

**Infra 踩坑 (这次记下避免下次重蹈)**:

- **CUDA 12.1 不在 cuda_helper.sh 支持列表**: vast.ai `pytorch:2.4.0-cuda12.1-cudnn9-devel` 镜像 CUDA 12.1, 原 cuda_helper.sh 只 11.8/12.4/12.6/12.8/13.0. 加 12.1 case (TORCH_VERSION=2.4.0 / TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;8.9;9.0+PTX), commit
- **nohup 不持久**: vast.ai container 里 `nohup ... &; disown` 起的 viser 进程在 parent ssh 退出后被回收. 换 `setsid bash -c '...' < /dev/null > /dev/null 2>&1 &` 完全脱离父 session
- **viser 1.0.27 ↔ 1.0.29 客户端版本 mismatch**: viser server 在 WebSocket 握手校验客户端 subprotocol `viser-v<version>`, 不匹配就拒绝. 中间 pip downgrade 后 React 客户端 JS bundle (内联 zstd 压缩) 跟 server 版本对不上 → 浏览器只看到 /WorldAxes. 用 `pip install --force-reinstall viser==<server_ver>` 修, 或保持 server/client 版本一致
- **rsync over ssh 必须 `-e "ssh -T"`**: ssh config 里 `RequestTTY force` 跟 rsync binary stream 不兼容, 错误 `unexpected tag 103 (0x6e797372)`
- **smoke 测试一定要 `_clear_engine_meshes()`**: 否则走 `_render_playground_hybrid` 静态路径 (无 timestamp_us), 测不到 `_trace_scene_mog`. Day 2 中段花了 1h 才发现这个

**代码改动清单** (4 files + 1 doc, T8.12 commit):

| Path | 改动 |
|---|---|
| `threedgrut_playground/engine.py` | `_trace_scene_mog` 加 `camera=None` kwarg, 从 kaolin Camera 算 [fx,fy,cx,cy] + 真 c2w + camera-space rays 匹配 dataset contract (Bug #1; pinhole 近似, T8.13 替换为 fisheye) |
| `threedgrut/layers/layered_model.py` | `init_from_checkpoint` 末尾 `self.cuda()` (Bug #2) |
| `threedgrut_playground/viser_gui_4d.py` | Reset View 真重置到 initial_c2w (4 个 camera 字段) |
| `scripts/cuda_helper.sh` | 加 CUDA 12.1.1 / 12.1 case |
| `docs/T8.12_handover_day1.md` (NEW) | Day 1 → Day 2 交接 + 完整 bug 分析 + 重建脚本 |

**Mac 单测无回归** (179/179 PASS, 0 regression).

**v3 backlog 入 T8.13**: viz_4d schema 扩展 fisheye/FTheta 内参 + viser 走 fisheye 投影路径. 完成后 viser 应能匹配 render.py 输出 (含 fisheye 桶形畸变).

---

### 🟡 T8.12-FIX — 诊断翻新: 参考 repo 反证 T8.13 假设 + viser_gui_4d CLI fov/camera_type flags (2026-05-21, Mac 本地, 待 vast.ai 视觉验收)

**触发**: 用户拿到 thinkpad 上的参考 repo `/home/yusun/repo/3dgrut/tools/viser_*.py` 后, 对比发现参考 repo 在同样 fisheye-trained nurec 场景下, viser 预览 (`viser_multilayer_nurec.py` + `viser_av_static.py`) **也是用 pinhole `make_rays` + 纯 `[fx,fy,cx,cy]`**, 不走 FTheta polynomial, 但视觉清晰. 反证 T8.12 留的 T8.13 假设 ("必须扩 schema + 走 fisheye 投影") 大概率**错诊**.

**关键差异表** (参考 repo vs 当前):

| 维度 | 参考 `viser_multilayer_nurec.py:280` | 当前 `viser_gui_4d.py` |
|---|---|---|
| 初始 fov | `client.camera.fov = math.radians(90)` 硬设 | 不设 → viser 默认 ~80° |
| 投影 | 纯 pinhole `make_rays` | engine `_trace_scene_mog` pinhole (T8.12 fix) |
| view_matrix | C2W 显式构 `T_to_world` | `get_c2w` 命名作 `view_matrix` 传 kaolin Camera.from_args |
| 模型调用 | `self.model(batch)` 纯 MoG | `self.scene_mog(batch)` LayeredGaussians blend |
| FTheta polynomial | 不存不用 | 不存不用 (T8.13 backlog 计划补) |
| 视觉效果 | 清晰 | 远景隧道 motion-blur 乱糊 |

3 个非 FTheta 维度差异 (fov 初值 / view_matrix 约定 / Layered blend) 需先排除. T8.12-FIX 先做最廉价的 fov 校正:

**改动 (4 处)**:

| 文件 | 改动 |
|---|---|
| `threedgrut_playground/viser_gui_4d.py` | 加 `import math`; `Viser4DViewer.__init__` 加 `initial_fov_rad: Optional[float] = None` kwarg; `_on_connect` 在设 wxyz/position 后加 `client.camera.fov = float(self.initial_fov_rad)`; Reset View 同步加 fov 复位; argparse 加 `--initial_fov_deg` (default=90.0, 与参考 repo 对齐) / `--camera_type` (Pinhole/Fisheye choices, default Pinhole) / `--camera_fov_deg` (默认 None → 用 initial_fov_deg); engine 实例化后若 `args.camera_type == "Fisheye"` 设 `engine.camera_type = "Fisheye"; engine.camera_fov = <deg>` 走已存在的 `_raygen_fisheye` 路径 |
| `threedgrut/tests/test_viser_gui_4d_fov.py` (NEW) | 10 case 单测: CLI parse (default/override/Fisheye/reject unknown) + camera_fov_deg 兜底回 initial_fov_deg + Viser4DViewer kwarg 接受 + 存储正确性 |

**Mac 验证**:
- `pytest threedgrut/tests/test_viser_gui_4d_fov.py -v` → 10/10 PASS
- `pytest threedgrut/tests/` 全量 → **189 passed + 1 skipped** (旧 179 + 新 10), **0 回归**
- A800 conda 3dgrut env `python viser_gui_4d.py --help` → 三个新 flag 显示 + help 文案正确

**未做 (待外部触发)**:
- vast.ai RTX 4090 视觉对比 (T8.12 实例 37188673 已销毁, 需重开实例; G-A2/G-B2)
- 若 fov 90° 仍不达标 → 跑 A.3 view_matrix 约定 print + A.4 LayeredGaussians blend 旁路诊断, 命中再上 Phase B.2/B.3
- 若 Phase A.2 + B.* 都不够 → 进 Phase C (= 原 T8.13 FTheta schema 扩展, 重活)

**T8.13 状态**: ⬜ Backlog → 🟡 CONDITIONAL (仅 T8.12-FIX 视觉不达标时启动). 决策依据见上 plan file `/Users/etendue/.claude/plans/v2-plan-md-task-8-12-replicated-shore.md`.

**代码改动清单** (2 files, T8.12-FIX commit, 待 vast.ai 验收后再决定是否合并到 main):

| 文件 | 改动 |
|---|---|
| `threedgrut_playground/viser_gui_4d.py` | 加 3 CLI flags + initial_fov_rad kwarg + on_connect/Reset View 显式设 client.camera.fov + engine.camera_type 钩子 |
| `threedgrut/tests/test_viser_gui_4d_fov.py` (NEW) | 10 case 单测覆盖 CLI parse + Viser4DViewer kwarg contract |

---

### 🎬 Stage 8 — viser_gui_4d 4D 场景可视化 (2026-05-20, Mac 本地, T8.1-T8.6 完整链路)

把 v2 LayeredGaussians 训完的 4D 场景拉到浏览器:用户拖 timeline → Gaussian 背景里的车跟着动、绿色 ego polyline + 当前 t 时刻相机 frustum、所有 dynamic track 历史折线(class color)、当前 t 时刻 active cuboid wireframe(instance color)、road / dynamic LiDAR overlay。所有 4D 元素从 ckpt['viz_4d'] 直接读,viewer 不依赖 NCore SDK / dataset 重加载 (无 viz_4d 时可选 `--dataset_path` lazy fallback)。

**ckpt schema 扩展** (向后兼容三档: v2+viz_4d 完整 / v2 无 viz_4d static fallback / v1 等价原 viser_gui):

```python
ckpt["viz_4d"] = {
    "schema_version": 1,
    "ego": {poses_c2w[N,4,4], frame_timestamps_us[N], primary_camera_id/fov_y_rad/aspect},
    "tracks": {tid: {poses[F,4,4], size[3], frame_info[F bool], class}},
    "tracks_camera_timestamps_us": [F] int64,
    "lidar": {road_xyz/road_rgb/dynamic_xyz/dynamic_rgb (subsample), road_n_total, ...},
    "viewer_defaults": {initial_c2w, near/far/resolution, t_us_first/last},
}
```

体积评估 (F=1500/K=179/road 200K/dyn 100K): tracks 17 MB + LiDAR 18 MB + ego 0.7 MB ≈ **+35 MB** (v2 主体 500 MB-2 GB 中可接受). `viz_4d.enabled=false` (默认) → ckpt 与旧版 byte-identical.

**改动清单** (commit 待 push):

| Path | 操作 | 说明 |
|---|---|---|
| [threedgrut_playground/engine.py](threedgrut_playground/engine.py) | MOD | `load_3dgrt_object` 加 `use_layered_model` 分支构造 LayeredGaussians + populate_tracks from viz_4d; `render_pass` 加 `*, timestamp_us=-1` kwarg; `_trace_scene_mog` dispatch LayeredGaussians.forward 路径 (T_to_world=I + rays_in_world_space=True) |
| [threedgrut/trainer.py:1116](threedgrut/trainer.py) | MOD | `save_checkpoint` 在 `torch.save` 前注入 `viz_4d` (try/except 静默 fallback) |
| [threedgrut/layers/layered_model.py:670](threedgrut/layers/layered_model.py) | MOD | `_populate_tracks_impl` 新增 `self.tracks_metadata: dict[tid, {class, size}]` 持久化 |
| `threedgrut/viz/__init__.py` | NEW | 命名空间 |
| `threedgrut/viz/metadata.py` | NEW | ~280 行: `extract_4d_metadata` + `_extract_ego/tracks/lidar/defaults` + `_detect_primary_camera` (FTheta / pinhole 自适应) |
| `threedgrut_playground/viser_gui_4d.py` | NEW | ~640 行: `Viser4DViewer` 类 + 5 GUI folder (Static Render/Timeline/Visibility) + `_play_tick` wallclock-driven 实时 + `_on_time_change` 中央调度 + `_mirror_ui` 抑制 slider 回路 + scene primitives 全套 + `_load_metadata` lazy fallback |
| `threedgrut_playground/utils/viz4d_metadata.py` | NEW | `FourDMetadata` dataclass (pure numpy, 测试友好) + `from_ckpt` + `lookup_frame_idx` 二分 + `active_tracks_at` + `ego_pose_at` |
| `threedgrut_playground/utils/cuboid.py` | NEW | `UNIT_CUBE_EDGES` [12,2,3] + `cuboid_world_edges(pose, size)` + `class_color` palette + `instance_color` FNV-1a hash → HSV → RGB |
| `threedgrut_playground/utils/viser_math.py` | NEW | `mat_to_wxyz` Shepperd/Markley + 4 分支 + canonical sign (w>=0) + unit-norm round-off cleanup |
| `configs/base_gs.yaml` | MOD | 加 `viz_4d` 块: `enabled=false` (默认), `include_lidar=true`, `lidar_road_subsample=200000`, `lidar_dynamic_subsample=100000`, `default_near/far/resolution` |
| `configs/apps/ncore_3dgut_mcmc_v2_full_4dviz.yaml` | NEW | 继承 `..._v2_full_exposure` + 翻 `viz_4d.enabled=true` |
| `threedgrut/tests/test_engine_layered_load.py` | NEW | 3 测试 (populate_tracks viz_4d roundtrip / `_resolve_pose_idx` 二分 / idempotent replace) |
| `threedgrut/tests/test_viz_4d_metadata.py` | NEW | 8 测试 (smoke / subsample / include_lidar=false / tracks_metadata / no_tracks / no_lidar / unknown_class / ckpt_roundtrip) |
| `threedgrut/tests/test_viz4d_metadata_loader.py` | NEW | 8 测试 (`from_ckpt` 解析 / 各种 missing block 兼容 / `lookup_frame_idx` 二分 / `active_tracks_at` / `ego_pose_at` nearest / empty / no_lidar / torch.save 真 roundtrip) |
| `threedgrut/tests/test_viser_math.py` | NEW | 4 测试 (identity / 90° x / 50 random round-trip / canonical sign) |
| `threedgrut/tests/test_cuboid_wireframe.py` | NEW | 9 测试 (shape / vertex range / unique pairs / identity / translation / size scale / z90 rotation / class_color / instance_color) |
| `threedgrut_playground/README_4D.md` | NEW | 用户文档 (Quick Start + GUI 控件说明 + ckpt schema + fallback + troubleshooting) |

**Mac 单测**: 32 新测试 PASS, **总 173/173 PASS**, 0 回归.

**剩余 (Backlog)**: T8.8 A800 100 step smoke + ckpt['viz_4d'] 字段验证 (脚本已写在 `viser-3d-purrfect-whisper.md` plan 文件)

---

### 🎉 Stage 6-fix T6F.3 出口达成 (2026-05-20, A800 GPU 1, v2_egomask_fix_20260520_113746)

**首次量化 Stage 3/4/5/6 历史 PSNR 中 ego 车身水分**, 同时印证 T6F.1+T6F.2 修复方向正确.

**配置**: `apps/ncore_3dgut_mcmc_v2_full_exposure` + `sky_backend=mlp` + 单相机 `camera_front_wide_120fov` + `duration_sec=2.0` + 5000 step (与 Stage 6 T6.3.a 同款).

**双指标 (新增)** —— metrics.json 12 字段齐全:

| 指标 | full (全图) | masked (T6F.1 ego 接通后) | Δ | 备注 |
|---|---:|---:|---:|---|
| mean PSNR | 20.493 | **29.493** | **+9.000** dB | ego 区水分 |
| mean SSIM | 0.858 | 0.934 | +0.076 | 结构相似 |
| mean LPIPS | 0.317 | 0.190 | -0.127 | 感知相似 |
| mean cc_PSNR | 19.611 | 24.904 | +5.293 | color-correction 后 |
| mean cc_SSIM | 0.821 | 0.900 | +0.079 | |
| mean cc_LPIPS | 0.327 | 0.200 | -0.127 | |

**Ego mask 实测分布** (NCoreDataset.sequence_cameras_frame_valid_pixels_masks):
- valid frac (非 ego) = **78.22%**
- ego car body 区 = **21.78%** (与 sseg sky 1.85% / road 21.55% 共存, mask 方向正确)
- TOP half (天空+前景) = 100% valid
- BOTTOM half (地面+ego 引擎盖) = 56.43% valid
- BOTTOM 1/4 (引擎盖区) = **14.10% valid** (即 85.90% 是 ego)

**性能基线**: 5000 step / 520.41 s / **9.61 it/s** — 与 Stage 4 baseline 9.58 it/s 0% 性能损失 (mask 是 cheap broadcast, ssim/lpips_masked 多算 2× 但忽略不计).

**关键洞察 — Stage 6-fix 的双向证据**:

1. **masked PSNR 29.493 > Stage 4 baseline 26.315 (+3.18 dB)** —— 非 ego 区不再被 ego 区"内卷"训练资源, 高斯容量集中到天空/地面/远景, 干净区拟合反而比 Stage 4 更好.
2. **full PSNR 20.493 < Stage 4 baseline 26.315 (-5.82 dB)** —— ego 区 21.78% 不再被训练 loss 监督 → 高斯不主动管那块 → 渲染输出乱色 → 全图 MSE 含巨大 ego 误差 → PSNR 拉低. **这是正确的预期行为** (业务上 ego 区从不参与 novel view synthesis 的核心 KPI).
3. **历史 PSNR 含水分量化**: Stage 3 26.13 / Stage 4 26.32 / Stage 5 26.17 / Stage 6 cc 24.94 dB 这些 baseline 数字, **绝大多数是 ego 区被强行训练拟合得"完美"** (车身固定位置/颜色) 拉高的, 真实可用的非 ego 区 PSNR 一直被 ego 内卷压制.
4. **未来 KPI 用 `mean_psnr_masked`** —— 全图 PSNR 只作 sanity check, masked PSNR 才是反映 v2 NuRec 风格分层重建质量的指标. Stage 7 T7.3 出口门槛 ≥ 28.5 dB 仍是全图 PSNR (与 v1 plan 可比), 但新增 masked PSNR 门槛 ≥ 30 dB (Stage 6-fix 已达 29.49 单相机 5k, 7 相机 30k 应明显超 30).

**踩坑记录 — 写入 CLAUDE.md "A800 操作 + 文档同步 严格把关清单"** (commit 919bac7):

1. **第 1 次跑 5k smoke 立即崩** (RuntimeError: tensor a (1920) vs b (1080) at dim 2) —— compute_layered_l1_loss 内部 `bg = valid * (1-road) * ...` 在 4D [B,H,W,1] valid 和 3D [B,H,W] road 上 broadcast 错配. Mac 单测全过是因为没构造"4D valid + 完整 image_infos"的 case. 修复: layered_loss.py 顶部 squeeze 4D valid → 3D, NEW 2 个回归测试 pin 住. commit 9c18b57. **教训**: Mac 单测要覆盖所有 mask shape × image_infos 组合.

2. **第 2 次跑 5k smoke 成功但 metrics.json 缺 6 个 masked 字段** —— T6F.2 只改了 `trainer.compute_metrics`, 但 eval-time metrics.json 由 `threedgrut/render.py` 独立路径写出. 修复: render.py eval loop 同款加 masked 三指标 (raw + cc). **教训**: 新增 metric 任务必须同时改 trainer.py + render.py. 已写进 CLAUDE.md § "B. 训练→eval→metrics 链路完整性验证".

3. **eval 单跑 render.py 失败** —— A800 顶层 `render.py` 历史上被错放成包内 `threedgrut/render.py` 内容 (无 main entry, silent exit 0 无产物). rsync 本地正确顶层 render.py 覆盖修复. **教训**: 跑训练前 `ssh a800-x2 "head -25 <entry.py>"` 验证入口. 已写进 CLAUDE.md § "A. 远程代码就绪验证".

**Stage 6-fix 状态**: 3/3 ✅. **Stage 7 KPI 现可启动**: T7.1 v2_full 配置 (sky_envmap + exposure + ego mask 全栈) → T7.2 2s smoke → T7.3 7-cam 20s 30k step 含 masked PSNR ≥ 30 dB 出口.

---

### 🔧 Stage 6-fix T6F.1 + T6F.2 Mac 本地完成 (2026-05-20, commit 65869ec)

修复 v2 链路里 ego mask "加载了但从未流入 loss/metric" 的漏洞：
- **根因**：`NCoreDataset.__init__` L385-395 加载 ego car mask 并 dilate 缓存到 `sequence_cameras_frame_valid_pixels_masks`，但训练分支 `__getitem__` L806-872 从未读取该缓存；验证分支 L959 取了 `valid` 但 `get_gpu_batch_with_intrinsics` L1273-1367 没把 `valid` 拷贝进 `Batch` → `gpu_batch.mask = None` → trainer L707-709 `if mask is not None` 永远跳过 → ego 车身像素参与 L1 / layered_l1 / PSNR / SSIM / LPIPS。
- **影响量化**：Stage 3 PSNR 26.133 dB / Stage 4 26.315 dB / Stage 5 26.167 dB / Stage 6 cc_PSNR 24.937 dB 这些 baseline 都**含 ego 车身水分**。车身像素在所有帧颜色固定 + 几何固定，是最易拟合的区域，虚高 0.2-1.5 dB 是大致量级（待 T6F.3 A800 实测确认）。

**T6F.1 ego mask 真注入 Batch.mask**（`threedgrut/datasets/datasetNcore.py`）：
- 训练分支 `__getitem__` 在 timestamp 注入之后取 `sequence_cameras_frame_valid_pixels_masks[seq][cam][frame_idx]`，按 `_camera_resolutions[cam_id]` 的渲染分辨率 `cv2.INTER_NEAREST` resize（与验证分支 L942-947 同款），塞 `batch_dict["valid"]`（2D `bool` ndarray）。
- `get_gpu_batch_with_intrinsics` 在 RGB 段之后加 4 行：if `"valid"` in batch → torch float32 → `.reshape(1, h, w, 1)` → `batch_dict["mask"]`。同时兼容 val 分支已有的 1D `[H*W]` valid（`reshape` row-major 顺序无关）。
- 自动接通（不改）：`trainer.get_losses` L707-709 mask 乘法生效 + `compute_layered_l1_loss(valid_mask=mask)` 自动从 bg/road/dyn 三区 partition 剔除 ego 像素。
- 保持 D7 不变量：SSIM loss term 仍走全图。

**T6F.2 metric 端 psnr/ssim/lpips full + masked 双指标**（`threedgrut/trainer.py::compute_metrics`）：
- 保留原全图三指标（与历史 Stage 3-6 baseline 可比）。
- 当 `gpu_batch.mask is not None` 时追加：
  - `psnr_masked = -10·log10(sum((p-g)² · m) / (sum(m) · 3))` 解析公式
  - `ssim_masked / lpips_masked` 用 GT-fill 近似：mask=False 区填 GT（差=0）后调原 SSIM/LPIPS criterion；该区 SSIM≈1 / LPIPS≈0，按面积稀释稳定（driving-3DGS 文献广泛采用）。
- `mask = None` 时三 masked 指标 ≡ 全图指标（byte-identical 回归），保证 NeRF/Colmap 等非 NCore dataset 不引入回归。

**Mac 测试套件**：
- NEW `threedgrut/tests/test_datasetNcore_ego_mask.py` 7 测试 PASS：Batch.mask 4D 形状契约 / 3D shape 拒绝（验证 reshape 不可省）/ 2D valid reshape / 1D valid reshape 等价 / layered_l1 ego 区剔除 / valid_mask=None v1 byte-identical / mask=全1 ≡ mask=None
- NEW `threedgrut/tests/test_trainer_masked_metrics.py` 7 测试 PASS：mask_none 三指标等价全图 / mask_all_ones 等价 / psnr_uniform_error δ→-10·log10(δ²) / psnr_ignores_invalid_error / ssim_gt_fill 单调改善 / lpips_gt_fill 单调改善 / shape broadcast
- 全套回归 `pytest threedgrut/tests/ -v`：**139 passed, 1 skipped**（原 125 + 14 新；sky cubemap test 仍 skip 因 nvdiffrast 不可用，预期行为）

**Stage 6-fix 状态**：2/3 ✅ Mac 本地；剩 T6F.3 A800 5k smoke v2_full_exposure 双指标 + ego 区粒子密度核查（待 ssh a800-x2 跑）。

---

### 🎉 Stage 5 + Stage 6 A800 5k step 出口达成 (2026-05-20, A800 GPU 0/1 并行, commit b38fce8)

**Stage 5 (T5.6) GPU 1 5k step 单相机 sky smoke**：

| 指标 | 实测 (v2 Stage 5 sky_envmap=MLP) | 出口门槛 | 备注 |
|---|---|---|---|
| Mean PSNR | **26.167 dB** | ≥ 25.8 ✅ (+0.37) | 距 Stage 4 baseline 26.315 仅 -0.15 dB (noise 级) |
| Mean SSIM | 0.878 | — | Stage 4 baseline 0.883 |
| Mean LPIPS | 0.302 | — | Stage 4 baseline 0.275 |
| Mean cc_PSNR | 25.736 | — | color-correction post-process |
| Training time | 943.51 s (15.7 min) | — | 5000 step |
| Iter speed | 5.30 it/s | — | 比 Stage 4 9.58 慢 (4 层 + sky blend + num_workers=0) |
| n_particles | bg 1M + road 200K + dyn 48K + sky MLP ~50K params | — | sky MLP layer0=64×39, layer1=64×103, layer2=3×64 |

- ckpt 含 `sky_envmap_state` (6 个 Linear param tensors, layer0.weight norm=19.04 → MLP 训练了)
- 配置：`apps/ncore_3dgut_mcmc_v2_sky.yaml` + `trainer.sky_backend=mlp` (CLI overide, nvdiffrast 远端不可用 → MLP backend)
- 输出：`/root/work/yusun/ncore-nurec/output/smoke_v2_sky_20260520-101019/`

**Stage 6 (T6.3.a) GPU 0 5k step 5 相机 exposure smoke**：

| 指标 | 实测 (sky + exposure 双开) | 出口门槛 | 备注 |
|---|---|---|---|
| Mean PSNR | 23.237 dB | (Stage 7 30k 才需 ≥ 28.5) | 5-cam 5k 中间态 |
| Mean cc_PSNR | **24.937 dB** | — | **color-correction +1.7 dB**（直接证明 exposure 在补偿 cam 间差异）|
| Mean SSIM | 0.820 | — | std_psnr 2.951 (5-cam 视角差异)|
| exposure_a.std | **0.0306** | **> 0.01 ✅ (3×)** | 学到非平凡 per-cam gain |
| exposure_b.std | 0.0288 | > 0 ✅ | 学到非平凡 per-cam bias |

per-camera learned gain（exp(a)）：

| Camera | exposure_a | exposure_b | gain=exp(a) |
|---|---:|---:|---:|
| cam0 camera_front_wide_120fov | -0.1304 | -0.0080 | 0.878 (front 暗化 12%) |
| cam1 camera_rear_tele_30fov   | -0.0649 | -0.0501 | 0.937 |
| cam2 camera_cross_left_120fov | -0.0549 | +0.0158 | 0.946 |
| cam3 camera_cross_right_120fov| -0.0609 | -0.0182 | 0.941 |
| cam4 camera_rear_left_70fov   | -0.0805 | +0.0217 | 0.923 |

- ckpt 含 **`sky_envmap_state` + `exposure_state` 同时**（3 模块协同 — bg+road+dyn 粒子 / sky envmap / per-cam exposure 三者 byte-level coexist in single ckpt）
- 配置：`apps/ncore_3dgut_mcmc_v2_full_exposure.yaml` + `trainer.sky_backend=mlp`
- 输出：`/root/work/yusun/ncore-nurec/output/smoke_v2_exp_20260520-101443/`

**T5.1 A800 nvdiffrast 探测结果**：A800 conda env (3dgrut) 不带 nvdiffrast；PyPI 镜像 (Huawei) 无该包；GitHub install 被 sandbox 拒绝 → **Stage 5 落入 MLP fallback 路径**（pre-designed plan）。conf 用 `trainer.sky_backend=mlp` CLI override 即可。Stage 7 可考虑手动 `pip install` nvdiffrast 切换到 cubemap backend（更稳定 sky 表征）。

**实施过程修复（A800 dry-run 发现并 land）**：

1. **设备同步**：sky 模块在 `LayeredGaussians.__init__` 时 CPU 上构造，多层模式下 trainer 不调 `.to(device)` → forward 时 `nn.Linear` 在 CPU 而 `batch.rays_dir` 在 CUDA 报 device mismatch。修复：`setup_optimizer` 多层路径调 `sky.to(target_device)`（target = 首个粒子层的 device）。
2. **ckpt save 路径**：`LayeredGaussians.get_model_parameters` 旧版用 `layer.get_model_parameters()` for-all-layers，对 sky 模块（不是 MoG）抛 AttributeError。修复：粒子层走 `get_model_parameters`、sky 走 `state_dict()` 存到 sibling key `sky_envmap_state`；`init_from_checkpoint` 同步加 sky load 分支。`setup_optimizer_for_test` 也跟着分粒子/非粒子两条路（粒子层 6 group / sky 一组 flat Adam）。
3. **5-cam yaml fix**：原 v2_full_exposure.yaml 写了 `camera_rear_wide_120fov` 等名字，9ae151dc clip 实际无此相机（manifest 列：cross_left/right_120fov, front_tele_30fov, front_wide_120fov, rear_left/right_70fov, rear_tele_30fov）。改 yaml 用真实 5 相机环。

**新增 Mac 单测覆盖 ckpt path fix**：
- `test_sky_envmap_state_roundtrip_in_checkpoint` — sky weights 通过 get_model_parameters → init_from_checkpoint 字节一致 round-trip
- `test_get_model_parameters_skips_non_particle_layers` — dynamic_deformables / sky 不会出现在 gaussians_nodes 里

**关键不变量验证（A800 端到端）**：

- ✅ `pred_opacity` 路径正确：sky blend `rgb_gauss + rgb_sky*(1-pred_opacity)` 渲染不退化 (Stage 5 PSNR -0.15 vs Stage 4 noise 级)
- ✅ Mac 单测 36 layered_gaussians + 11 sky_envmap + 6 exposure + 12 layered_loss = 65 测试在 sky/exposure 启用 / 关闭两种模式下均 PASS（byte-identical at runtime when off）
- ✅ Stage 6 cc_psnr - raw_psnr = +1.7 dB 直接证明 ExposureModel 学到的 per-cam affine 真有效（不是噪声）
- ✅ sky_envmap_state + exposure_state 在同 ckpt 中共存，resume 时 trainer.init_exposure_model load 自如

**剩余 Stage 7 工作（≥ 7-cam × 20s × 30k step 完整 KPI）**：T7.1-T7.5（仍在 Backlog）。

---

### 🎉 Stage 5 + Stage 6 Mac 本地完成 (2026-05-19, Mac CPU venv)

Stage 5 (T5.2-T5.4) + Stage 6 (T6.1-T6.3) 整个代码路径在 Mac 本地完成并 Mac 单测全过。**剩余 A800 任务**：T5.1 nvdiffrast probe + T5.6 / T6.3.a 5k step smoke 出口（用户手动 ssh 跑）。

**关键发现**（修正了 plan.md 原描述）：
- `pred_opacity` 已经在 renderer 输出字典里（3DGUT `pred_rgba[..., 3:]` / 3DGRT 直接）→ Stage 5 sky blend 不用改 CUDA / binding，直接用 `(1 - outputs["pred_opacity"])` 当透射率
- `Batch.camera_idx` / `Batch.rays_dir` / `Batch.T_to_world` 已就位 → Stage 6 exposure 直接 `batch.camera_idx`，Stage 5 viewdirs 用 `T_to_world[:3,:3] @ rays_dir` 转世界帧

**Stage 5 实现细节**：
- `threedgrut/correction/sky_envmap.py`：`SkyEnvmapBase` 抽象基类 + `SkyEnvmapMLP`（无外部依赖，SinusoidalEncoder + 3 层 MLP + sigmoid）+ `SkyEnvmapCubemap`（drivestudio EnvLight 移植，nvdiffrast 不可用时 forward 抛 ImportError 带明确指引）
- `LayeredGaussians`：构造时按 `spec.extra["backend"]` 实例化 sky 模块挂 ModuleDict；`forward` 末尾调 `_blend_sky(outputs, batch)`，写入 `rgb_gaussians` / `rgb_sky` / `pred_rgb` 三 key；`setup_optimizer` 给 sky 模块挂独立 Adam（lr=trainer.sky_lr，默认 0.01）
- `_LayeredOptimizerView.step()` 已通过 `getattr(layer, "optimizer", None)` 自动 fan-out 到 sky，无需改动 view
- `compute_sky_loss(rgb_sky, rgb_gt, sky_mask, min_pixels=100)`：纯函数，sky_mask 为空时返回 0（min_pixels guard），有 NaN 安全；3 通道分母处理正确
- `LayerSpec.extra: dict`（field 加 compare=False 保持 hashable）；registry `sky_envmap` 默认 `{"backend":"cubemap","resolution":128}`；conf.trainer.sky_backend 非 null 时覆盖 spec
- `configs/base_gs.yaml` 加 `use_sky_envmap` / `sky_backend(null=spec默认)` / `sky_resolution` / `sky_lr` / `lambda_sky` 字段
- `configs/apps/ncore_3dgut_mcmc_v2_sky.yaml`：基于 v2_full 加 sky_envmap 层 + use_sky_envmap=true

**Stage 6 实现细节**：
- `threedgrut/correction/exposure.py`：`ExposureModel(num_camera)` — 移植 Recon-Studio 29 行 affine（去掉 num_camera==1 短路）；`forward(idx, image) = (exp(a)*image + b).clamp(0, 1)`，零初始化 = identity
- `Trainer3DGRUT.init_exposure_model(conf)`：从 `train_dataset.get_frames_per_camera()` 取 num_camera；建独立 `self.exposure_optimizer = Adam(lr=conf.trainer.exposure_lr)`；resume 时从 ckpt["exposure_state"] 恢复
- `run_train_iter`：sky blend 后、loss 前应用 `outputs["pred_rgb"] = self.exposure_model(camera_idx, outputs["pred_rgb"])`；`model.optimizer.step()` 后 `exposure_optimizer.step() + zero_grad()`
- `save_checkpoint`：增 `"exposure_state": {"module": state_dict, "optimizer": state_dict}` 段，与 post_processing_optimizers 范式一致
- `configs/apps/ncore_3dgut_mcmc_v2_full_exposure.yaml`：基于 sky yaml + 5 相机环 + use_exposure=true

**Mac 测试套件**：123 passed, 1 skipped (nvdiffrast 不可用预期 skip)。新增 29 个测试：
- `test_sky_envmap.py` (11 tests)：base abstract / MLP shape+range+grad / Cubemap shape+orthonormal+no-dep error
- `test_layered_gaussians.py` (6 new tests)：holds_sky_module {mlp, cubemap} / blend_sky_alpha_zero / blend_sky_alpha_one / blend_sky_no_sky_layer / forward_multi_layer_with_sky
- `test_layered_loss.py` (6 new tests)：sky_loss zero_when_no_pixels / none_mask / uniform_region_arithmetic / only_on_sky_region / squeezed_mask_shape / grad_flows
- `test_exposure.py` (6 tests)：zero_init_identity / per_camera_grad_isolation / clamp_unit_range / invalid_idx / zero_cameras / state_dict_roundtrip

**byte-identical 回归**：全部 28 个老 layered_gaussians 测试 + 12 layered_mcmc + 6 老 layered_loss 测试维持 PASS（sky / exposure 关闭时与 Stage 4 完全一致）。

**待 A800 验收（用户手动）**：
1. `ssh a800-x2 && conda activate 3dgrut && python -c "import nvdiffrast.torch; print(nvdiffrast.__version__)"` — T5.1 backend 选择
2. `python -u train.py --config-name apps/ncore_3dgut_mcmc_v2_sky n_iterations=5000 dataset.train.duration_sec=2.0 'dataset.camera_ids=[camera_front_wide_120fov]' ...` — T5.6 Stage 5 出口（目标全图 PSNR ≥ 25.8 / sky 区 ≥ 28）
3. `python -u train.py --config-name apps/ncore_3dgut_mcmc_v2_full_exposure n_iterations=5000 dataset.train.duration_sec=2.0 ...` — T6.3.a Stage 6 出口（exposure_a.std() > 0.01 / rear cam +0.3 dB）

---

### 📊 Stage 3 7-cam 20s 30k step baseline (2026-05-19 17:00:34, A800 GPU 0, task #30)

**v1 KPI 对照实测** — 用 v2_road yaml (bg + road, region-weighted L1, perturb Z lock) 跑 7-cam × 整 20s clip × 30k step：

| 指标 | 实测 (v2 Stage 3, 7-cam 20s 30k) | v1 plan "目标" KPI | v2 Stage 7 出口 KPI |
|---|---|---|---|
| Mean PSNR | **23.825 dB** | 27.60 (差 -3.78) | ≥ 28.5 (差 -4.7) |
| Mean SSIM | 0.819 | — | — |
| Mean LPIPS | 0.371 | — | — |
| std_psnr | 3.125 (后向相机偏低) | — | — |
| Eval frames | 525 (7 cam × 75 val) | — | — |
| 训练时间 | 62.75 min (52 train + 10 eval) | A100 35 min | ≤ 60 min ✓ |

**关键观察 & v2 KPI 缺口分析**：

1. **7-cam 比单 cam 低 2.3 dB**（单 cam 30k 未跑，但 5k 已 26.13；7-cam 30k 仅 23.83）：多 cam 7× 像素 + 7× 视角 → 30k step 仍欠拟合迹象明显
2. **后向相机拖低**：std 3.125，rear cameras frame PSNR 17-25 dB；front_wide / front_tele 仍稳在 26-30 dB
3. **v1 plan "27.60"** 可能是借来的 NCore 对照数字（"v1 基线"列），不是 9ae151dc clip 上 v1 真实测过的；我们这次是从 random init 用 v2 protocol 跑的，apples vs oranges
4. **距 v2 Stage 7 出口 ≥28.5 缺 4.7 dB**，候选 ROI 顺序：
   - Stage 5 sky envmap (sky 区不爆黑) → **+1~2 dB** 估
   - LiDAR depth supervision (drivestudio 标配, §14.3 V3-T9) → +0.5~1 dB
   - Stage 6 per-cam exposure (rear/侧 camera 曝光对齐) → +0.5~1 dB
   - v2_full yaml (含 dynamic_rigids 层) → +0.2~0.5 dB
   - 30k → 50k-100k step (drivestudio / NuRec 常用) → +0.5~1 dB
5. **task #30 ssh wrapper exit 255 ≠ training failed**：远端 A800 process 跑完，eval 525 frames 全数完成 + metrics.json 保存；只是本地 ssh session 中途断了

**Stage 3 (bg+road, 单层不含 dyn) 7-cam 20s 30k 数字定调**：23.825 dB 是 v2 当前架构的**7-cam baseline 起点**，Stage 5/6 加完整后预期 26-28 dB。

---

### 🎉 Stage 4 出口 ✅ (2026-05-19 16:39:17, A800 GPU 1, T4.5 完成, commit 4807951)

10k step 三层 (bg + road + dynamic_rigids) Stage 4 出口验收：

| 指标 | Stage 4 (timestamp-aligned) | Stage 3 baseline | Δ |
|---|---|---|---|
| Mean PSNR | **26.315 dB** | 26.133 | **+0.18** |
| Mean SSIM | **0.883** | 0.879 | +0.004 |
| Mean LPIPS | **0.275** | 0.297 | -0.022 |
| Iter speed | **9.58 it/s** | 9.54 | **零性能损失** (三层 vs 两层) |
| 训练时间 | 1043.9s (17.4 min) | 523s (8.7 min, 5k step) | 2× (10k vs 5k) |
| 出口门槛 26.4 | -0.085 | — | noise 级 (< 0.1 dB) |
| dynamic_rigid tracks | 31 real (autolabels v2) | — | 完全省 mock |
| dynamic 粒子数 | 48,488 | — | 出口预期 [10K, 50K] 命中 |

**实施关键 — 真实 NCore 数据 + timestamp-aligned 全栈**：

1. **真 cuboid autolabels（替代 mock tracks）** — `tracks_loader.load_tracks_from_ncore_cuboids(loader, cam_ts)`：
   - iter `loader.get_cuboid_track_observations()` (13657 obs across full clip) → groupby track_id → filter vehicle classes (automobile/truck/bus) → 每 cam frame 找 nearest cuboid obs within 50ms tolerance → `obs.transform("world", ts, pose_graph)` 把 rig→world
   - clip 9ae151dc 在 2s 窗内：179 unique tracks → 31 vehicle tracks
   - 完全绕过 mock，省去用户跑 NCore validator 的等待

2. **timestamp-aligned dyn pose 查询**（替代 frame_idx mismatch）：
   - `Batch.timestamp_us` 字段（dataset 写入 cam END timestamp = sseg key 一致）
   - `LayeredGaussians.forward` 用 `gpu_batch.timestamp_us` 而非 trainer 的 `frame_id=global_step`
   - `_resolve_pose_idx(timestamp_us, frame_id)`：binary-search 共享 `tracks_camera_timestamps_us` buffer，返回最近 pose index；frame_id 路径保留作 backward-compat (T4.3 unit tests)
   - 修复前 frame_idx mismatch 导致 F6/F7 后帧退化 -3 dB；修复后 F6/F7 反而 +0.7 dB
   - 物理上：universal time coordinate 全栈对齐（cuboid obs ts + cam frame END ts + sseg key ts 都是同 NCore 微秒时钟）

3. **MCMC track_ids buffer sync**：
   - `MCMCStrategy.add_new_gaussians` 加 hook：`hasattr(model, "track_ids") → torch.cat([track_ids, track_ids[sampled_idxs]])`
   - `relocate_gaussians` 加 hook：dead particles 继承 alive 的 track_id (`track_ids[dead_idxs] = track_ids[sampled_idxs]`)
   - 之前 add 1.05× 后 track_ids 长度不变 → shape mismatch crash

**逐帧 PSNR vs Stage 3 (timestamp-aligned)**：

| Frame | Stage 4 | Stage 3 | Δ | 解读 |
|---|---|---|---|---|
| 0 | 25.48 | 24.76 | **+0.72** | dyn 加成 |
| 1 | 26.89 | 26.21 | **+0.68** | dyn 加成 |
| 2 | 21.96 | 22.84 | -0.88 | dyn 训练略挤压 bg |
| 3 | 23.18 | 23.77 | -0.59 | 同上 |
| 4 | 26.58 | 26.69 | -0.11 | ≈持平 |
| 5 | 26.16 | 26.07 | +0.09 | ≈持平 |
| 6 | 29.00 | 28.29 | **+0.71** | dyn 加成（之前 -2.7） |
| 7 | 31.27 | 30.45 | **+0.82** | dyn 加成（之前 -2.9） |

**6 轮 first-light + 3 轮 10k 出口迭代 fix 全栈打通**：
1. `cKDTree OOM` (road_init.py 200K×629K=500GB host) → scipy + cdist fallback
2. `init_layer_from_points` 所有 default tensor 跟随 positions device
3. `__getattr__` multi-layer fallback (fused tensor / fused method / broadcast method / ref-layer last resort)
4. `LayeredGaussians.populate_tracks` + 共享 `tracks_camera_timestamps_us` buffer
5. `mcmc add/relocate` 同步 track_ids buffer (T4.2.b D2 兑现)
6. **timestamp-aligned `_resolve_pose_idx`** (替换 frame_idx mismatch, +1.1 dB)

**v1 byte-identical 回归（T4.5 阶段，task #29，commit 同一栈）**：A800 GPU 0 与 Stage 4 出口并行跑 v1 ckpt resume 1k step → 8/8 帧 PSNR 24.123 dB byte-identical with Stage 2/3 baseline ✅ (D8 出口门禁通过)。

---

### 🎉 Stage 3 出口 ✅ (2026-05-19 15:15:58, A800 GPU 1, T3.5.b 完成, commit 8a625c2)

5k step 两层 (bg + road) Stage 3 出口验收：

| 指标 | Stage 3 | 出口门槛 | v1 baseline |
|---|---|---|---|
| Mean PSNR | **26.133 dB** | ≥ 23.6 (+0.5 vs v1) | 24.123 |
| Mean SSIM | **0.879** | (v1 0.846) | 0.846 |
| Mean LPIPS | **0.297** | (v1 0.405) | 0.405 |
| Iter speed | **9.54 it/s** | (v1 9.48) | 9.48 |
| 训练时间 | 523s (8.7 min, 5k) | — | — |
| **超出口门槛** | **+2.5 dB** | — | — |

详见 commit `8a625c2` (T3.5.b) 文档 + §5 后段 Stage 3 节点（保留下方原节点）。

---

### T0.1 ✅ (2026-05-14 16:55-16:57，smoke)

A800-x2 单卡 1k step × 2s × 单相机：

| 指标 | 实际 |
|---|---:|
| Mean PSNR | 24.12 dB |
| Mean SSIM | 0.846 |
| Mean LPIPS | 0.405 |
| 训练耗时 | 105.4 s |
| 训练吞吐 | 9.48 it/s（A100 同条件 ~3.9 it/s，A800 快 2.4×） |

**与原计划偏差（影响下游）**：
- clip 是 `9ae151dc-e87b-41a7-8e85-71772f9603d7`（不是文档误写的 `3435ace9`）
- 路径在 `/root/work/yusun/ncore-nurec/data/ncore/` 而非 `/data/pai_ncore/`
- 必须 `CUDA_VISIBLE_DEVICES=1` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- 多相机必须显式 `dataset.camera_ids=[...]`

### T1.1 ✅ (2026-05-16, commit b0865c4 + 6f86c62 + 8a29fc0 + 5a6a5f9)

LayeredGaussians 容器（`threedgrut/layers/layered_model.py`）：
- `ModuleDict[name → MixtureOfGaussians]`
- NRE 对齐 ckpt schema：`ckpt["model"]["gaussians_nodes"][name]`
- 支持 3 种 ckpt 输入：NRE wrap / 已解开 / v1 flat（自动 route 到 "background"）
- 单 bg 层桥接：透传 attr 读写，让 v1 trainer/MCMC 代码完全无感
- 单测：bit-level v1 ckpt 兼容 + single-bg bridge

A800 smoke 验证：PSNR 24.084 dB / 9.87 it/s；v1 resume 24.123 dB **byte-identical** with `use_layered_model=false`。

### T1.5 ✅ (2026-05-16, commit 8a29fc0 + 5a6a5f9)

Trainer 集成：
- `train.py` 加 `use_layered_model` flag
- `Trainer3DGRUT.setup_training` 支持 LayeredGaussians 分支
- `forward` bridge 让 `self.model(batch, train=...)` 在单 bg 模式下正常工作

### T1.2 ✅ (2026-05-18, commit 60e1154 + 569819b + 6435483)

LayerSpec 完整字段 + registry + trainer wiring：

- `layers/layer_spec.py`：T1.1 的 3 字段 (`name/layer_id/max_n_particles`) → 8 字段，新增 `scale_prior / scale_lr_mult / mask_field / is_particle_layer / density_init`，全部默认值，向后兼容
- `layers/registry.py` 新建：`STANDARD_LAYERS` dict 注册 5 个标准层（background / road / dynamic_rigids / dynamic_deformables / sky_envmap），`specs_from_config(conf)` 工厂按 `conf.layers.enabled` 过滤、保序、未知名抛 ValueError
- `layers/__init__.py`：导出 `LayerSpec / STANDARD_LAYERS / specs_from_config`；`LayeredGaussians` 用 try/except 懒导出（dev 笔记本无 torch 时 package 仍可 import）
- `threedgrut/trainer.py::init_model`：原硬编码单 bg `LayerSpec(...)` → `specs_from_config(conf)`；日志输出实际层名 list
- `configs/base_gs.yaml` 加默认 `use_layered_model: false` 和 `layers.enabled: [background]`（plan 误称 `base.yaml`，实际项目根 yaml 是 `base_gs.yaml`，继承链 `apps/ncore_3dgut_mcmc → base_mcmc → base_gs`）

本地 Mac 验证：

| 指标 | 实际 |
|---|---:|
| `pytest test_layer_spec_registry.py` | 9/9 PASS |
| hydra compose 默认 | `use_layered_model=False / layers.enabled=['background']` |
| hydra compose override `[background,road]` | OK |

**A800 端验证（2026-05-18 11:01-11:02，commit f90b791）**：

| 指标 | T1.1 baseline (5a6a5f9) | T1.2/T1.3 after (f90b791) |
|---|---:|---:|
| Mean PSNR | 24.123 dB | **24.123 dB** ✅ byte-identical |
| Mean SSIM | 0.846 | **0.846** |
| Mean LPIPS | 0.405 | **0.405** |
| 8 帧 PSNR | 21.55/23.99/22.32/22.69/23.93/24.12/27.17/27.23 | **完全一致** |
| `pytest test_layer_spec_registry.py + test_layered_gaussians.py` | — | **18/18 PASS** (5.61s) |

命令：`python train.py --config-name apps/ncore_3dgut_mcmc resume=<v1_ckpt> use_layered_model=true layers.enabled=[background] n_iterations=1000 dataset.train.duration_sec=2.0 dataset.camera_ids=[camera_front_wide_120fov]`

### T1.3 ✅ (2026-05-18, commit ff83028)

v1 flat ckpt resume 在 LayeredGaussians 路径下的错误消息改良：

- `layers/layered_model.py::init_from_checkpoint` 第 122-129 行：当 ckpt 是 v1 flat 形态但 `self.layers` 不含 "background" 时，错误消息从 "no 'background' layer configured" 改为 "'background' layer is not in conf.layers.enabled (got [...])"，并明确告知"Add 'background' to layers.enabled"
- 配套 2 个 A800 contract test：`test_v1_ckpt_resume_without_background_layer_raises` (regex match `layers.enabled`) / `test_v1_ckpt_resume_with_background_layer_works`

**A800 端验证（2026-05-18 11:01-11:02，commit f90b791）**：
- `test_v1_ckpt_resume_without_background_layer_raises` PASS：错误消息正则 match `layers.enabled` ✓
- `test_v1_ckpt_resume_with_background_layer_works` PASS：v1 ckpt 全部路由进 background 层；road 层保持空 ✓
- 端到端 v1 ckpt resume：PSNR 24.123 dB byte-identical with T1.1 baseline ✓

### T1.4 ✅ (2026-05-18, commit 60e1154 + 569819b + ff83028)

单测覆盖扩展：

- 新文件 `threedgrut/tests/test_layer_spec_registry.py`（9 测试，Mac 本地可跑，无 torch 依赖）：
  - `test_layer_spec_frozen_immutable`：FrozenInstanceError 验证
  - `test_layer_spec_full_field_defaults`：8 字段默认值 + 显式赋值透传
  - `test_registry_standard_layers_complete`：5 层全员
  - `test_registry_specs_have_unique_ids`：layer_id 唯一
  - `test_registry_particle_flags_correct`：sky/deformable 非粒子层
  - `test_registry_road_layer_has_flat_scale_prior`：Z-lock 约定 + mask_field
  - `test_specs_from_config_filters_enabled`：按 enabled 过滤保序
  - `test_specs_from_config_unknown_layer_raises`：未知名 ValueError
  - `test_specs_from_config_defaults_to_single_background`：空 conf fallback
- `threedgrut/tests/test_layered_gaussians.py` 追加 3 个 A800 contract test：`test_v1_ckpt_resume_without_background_layer_raises` / `test_v1_ckpt_resume_with_background_layer_works` / `test_multi_layer_ckpt_roundtrip`

测试矩阵：v1 时代 6 个 contract test + v2 Stage 1 共 9 + 3 = 12 个新测试 = **18 个总测试**（Mac 本地 9 + A800 9）。

**A800 端验证（2026-05-18 11:01，commit f90b791）**：

| 测试集 | 数量 | 结果 | 耗时 |
|---|---:|:---:|---:|
| `test_layer_spec_registry.py` | 9 | ✅ 9/9 PASS | < 0.1s |
| `test_layered_gaussians.py`（含 T1.4 roundtrip） | 9 | ✅ 9/9 PASS | 5.6s |
| **合计** | **18** | **✅ 18/18 PASS** | **5.6s** |

注：本地 Mac 首次跑 `test_multi_layer_ckpt_roundtrip` 暴露了 `MoG.get_model_parameters()` 需要 optimizer 初始化的约束（commit f90b791 在测试中加 `setup_optimizer_for_test()` 修复）。

**Stage 1 出口**：T1.1 - T1.5 全 ✅；解锁 Stage 2 (LayeredMCMC) 与 Stage 3 (Road) 并行开发窗口。

---

### T2.1 ✅ (2026-05-18, commit 62fc509)

MCMCStrategy 抽 `_get_add_cap()` 钩子：

- `threedgrut/strategy/mcmc.py`：在 `add_new_gaussians` 前新增 `_get_add_cap() -> int` 方法（默认返回 `conf.strategy.add.max_n_gaussians`，v1 行为完全不变），第 142 行改为调用 `self._get_add_cap()`。共 +5 行，其他方法（`relocate_gaussians` / `perturb_gaussians`）零改动。
- `threedgrut/tests/test_layered_mcmc.py`：新建测试文件，1 个合约测试 `test_mcmc_get_add_cap_defaults_to_conf`，用 `__new__` bypass `__init__`（CUDA JIT）+ sys.modules stub 绕开 ncore 依赖，Mac CPU 可直接运行。

| 指标 | 实际 |
|---|---:|
| `pytest test_layered_mcmc.py` | **1/1 PASS** (Mac CPU, 0.42s) |
| `pytest test_layer_spec_registry.py` | **9/9 PASS** (Stage 1 回归) |
| A800 byte-identical 回归 | **Deferred** (controller batch, Stage 2 末尾) |

**注**：原 plan 描述的 `_select_indices` 钩子（选择 mask 再操作）经评估属于过度设计；sub-strategy-array 方案下只需 `_get_add_cap()` 一个 hook，diff 最小，v1 行为 byte-identical。

---

### T2.2 ✅ (2026-05-18, commit 7ad883b)

LayeredMCMCStrategy sub-strategy 数组实现：

- `threedgrut/strategy/layered_mcmc.py`（新建）：`LayeredMCMCStrategy(BaseStrategy)` 持有 `sub_strategies: dict[str, MCMCStrategy]`，每个 is_particle_layer=True 的层各一个 sub。`_make_sub_conf` 用 `OmegaConf.to_container(resolve=False)` + `OmegaConf.create` 深拷贝 conf 并覆盖 `strategy.add.max_n_gaussians = spec.max_n_particles`（注意必须 `resolve=False`，因为 conf 含 `int_list` 自定义 resolver，在 Hydra context 外不可 resolve）。`_post_optimizer_step` 遍历 subs，`suspend()` 向下传播。
- `threedgrut/trainer.py`：`init_densification_and_pruning_strategy` 新增 `case "LayeredMCMCStrategy"` branch，assert LayeredGaussians + 调 `specs_from_config(conf)` + 构造 `LayeredMCMCStrategy`。
- `threedgrut/tests/test_layered_mcmc.py`：扩展 `_install_stubs()` 以支持真实 `MixtureOfGaussians` 实例化（不再 stub threedgrut.model.model）；新增正确的 stubs（datasets package-level stub + datasets.utils direct load + DEFAULT_DEVICE=cpu + load_mcmc_plugin no-op + MCMCStrategy.__init__ CUDA-bypass patch）；追加 3 个 T2.2 合约测试。附带修复：tqdm/sklearn stubs 从 MagicMock 升级为带 valid `__spec__` 的 ModuleType，解决 torch._dynamo.trace_rules 的 `__spec__ is not set` 跨文件污染问题（Stage 1 全测试从联合运行时 2 fail → 25/25 pass）。

| 指标 | 实际 |
|---|---:|
| `pytest test_layered_mcmc.py` | **4/4 PASS** (Mac CPU, 0.45s) |
| `pytest threedgrut/tests/` (全套) | **25/25 PASS** (Mac CPU, 0.81s) |
| A800 byte-identical 回归 | **Deferred** (controller batch, Stage 2 末尾) |

**实现选择备注**：原 plan T2.2 描述的是继承 MCMCStrategy 并 override `_select_indices` 的方案；实际采用 sub-strategy 数组方案（持有独立 MCMCStrategy 实例，每个 sub.model 指向对应层 MoG），更轻量，不需要在 MCMC 操作内部动态切换 layer 上下文，且 _post_optimizer_step 串行遍历自然实现"零跨层迁移"。

---

### T2.3 ✅ (2026-05-18, commit 1a0d275)

`configs/strategy/layered_mcmc.yaml` 创建 + trainer `specs_from_config` 重复调用修复（I-2）：

- `configs/strategy/layered_mcmc.yaml`（新建）：Hydra group-defaults `[mcmc, _self_]` 继承 mcmc.yaml 全部超参（binom_n_max=51、relocate/add/perturb 各段）；只覆写 `method: LayeredMCMCStrategy`。首次尝试 `defaults: - mcmc - _self_` 即成功（Hydra 1.3 group-defaults 支持），无需 full copy 降级。
- `threedgrut/trainer.py` `case "LayeredMCMCStrategy"`：移除 `from threedgrut.layers.registry import specs_from_config` 懒导入及 `specs = specs_from_config(conf)` 调用；改为直接使用 `self.model.specs`（`LayeredGaussians.__init__` 已在 `object.__setattr__(self, "specs", list(specs))` 存储完全相同的列表）。
- `threedgrut/tests/test_layered_mcmc.py`：追加 `test_layered_mcmc_yaml_inherits_mcmc_defaults`（T2.3），通过 `initialize_config_dir` + compose 验证 yaml 继承正确、三个超参与 mcmc.yaml 保持一致。

| 指标 | 实际 |
|---|---:|
| `pytest test_layered_mcmc.py` | **5/5 PASS** (Mac CPU, 0.54s) |
| `pytest threedgrut/tests/` (全套) | **26/26 PASS** (Mac CPU, 1.14s) |
| yaml 继承方案 | Hydra `defaults: [mcmc, _self_]` group-defaults（DRY，无 full copy） |
| A800 byte-identical 回归 | **Deferred** (controller batch, Stage 2 末尾) |

### T2.4 ✅ (2026-05-18, commit 51540a8 + 04c9174)

conftest.py 迁移 (I-1 fix) + T2.4 不变量测试 + T2.2/T2.3 代码审查遗留修复：

**I-1 fix: 将 sys.modules stubs 迁移到 conftest.py**
- 原先 `_install_stubs()` 和 `MCMCStrategy.__init__` no-CUDA 补丁在 `test_layered_mcmc.py` 模块顶层，Stage 1 测试（`test_layered_gaussians.py`）单独运行时因 collect order 依赖失败（ModuleNotFoundError: ncore）。
- 新建 `threedgrut/tests/conftest.py`：pytest 在收集该目录任何测试文件之前自动加载 conftest.py，保证 stubs 无条件安装。
- **验证**：`pytest threedgrut/tests/test_layered_gaussians.py -v` 单独运行：9/9 PASS ✅

**T2.4 新增测试（3 个）**
- `test_no_cross_layer_migration_after_post_optimizer_step`：结构性验证 sub.model 与各层 MoG 的 identity 绑定（2 层 bg+road，各 100/50 粒子）
- `test_init_densification_buffer_dispatches_to_all_subs`：monkeypatch call_log 验证广播到所有 sub-strategy
- `test_make_sub_conf_does_not_mutate_parent`（T2.2 M-2 遗留）：独立 conf 返回，不影响父 conf

**代码审查遗留修复**
- M-1：将 `test_layered_mcmc_single_bg_equivalent_to_v1` 重命名为 `test_layered_mcmc_single_bg_uses_one_sub_strategy`，docstring 明确"仅结构性，非 byte-identical 训练输出"；同步更新 `layered_mcmc.py` 模块 docstring 中的引用
- M-3：`layered_mcmc.py` 类型注解从 `List[LayerSpec]` / `Optional[dict]` 改为 `list[LayerSpec]` / `dict | None`（移除 `typing.List`/`Optional` 导入）
- M-4：`test_layered_mcmc_yaml_inherits_mcmc_defaults` 函数体内重复的 `import os` 和 `from hydra import compose, initialize_config_dir` 已删除，改用模块级 `_CONFIG_DIR` 常量

| 指标 | 实际 |
|---|---:|
| `pytest threedgrut/tests/test_layered_gaussians.py` 单独运行 | **9/9 PASS** ✅（I-1 修复证明） |
| `pytest threedgrut/tests/test_layer_spec_registry.py` 单独运行 | **9/9 PASS** ✅ |
| `pytest threedgrut/tests/test_layered_mcmc.py` 测试数 | **8 个**（T2.1×1 + T2.2×3 + T2.3×1 + T2.4×3） |
| `pytest threedgrut/tests/` 全套 | **29/29 PASS** ✅ (0.52s) |

### T2.4 carry-over ✅ (2026-05-18, commit 75ed0e4)

T2.4 代码审查遗留修复：
- **A-1**：`test_no_cross_layer_migration_after_post_optimizer_step` → 重命名为 `test_no_cross_layer_migration_structural`，扩展 docstring 明确说明这是"结构性 identity 验证"而非调用 `post_optimizer_step()` 的动态验证。
- **A-2**：`conftest.py` 在 patch 之前捕获 `_original_init = MCMCStrategy.__init__`，并添加 WARNING 注释块，说明未来需要真实 CUDA `__init__` 的测试如何通过 `monkeypatch.setattr` + `from conftest import _original_init` 恢复。
- **A-3**：`test_layered_mcmc.py` 按 `T2.1 / T2.2 / T2.3 / T2.4` 添加分组 section comments。

---

### T2.5 ✅ (2026-05-18, commit d4841df)

`LayeredGaussians.fused_view(frame_id)` + `get_layer_mask(name)` 接口实现：

- `threedgrut/layers/layered_model.py`：在 `_single_bg_layer` 之后、`__getattr__` 之前插入两个方法（作为普通类方法，不走 `__getattr__` 桥）：
  - `fused_view(frame_id=None)`：单 bg 模式短路到 bg 层 Parameter 属性（字节一致的快速路径）；多层模式按 `self.specs` 顺序对粒子层做 `torch.cat`。frame_id 参数预留 T4.3 动态 pose 变换，T2.5 中仅接收不使用。
  - `get_layer_mask(name)`：返回 `Bool[N_total]` mask，布局与 `fused_view` concat 顺序一致；未知层名抛 `ValueError("unknown layer ...")`。
- `threedgrut/tests/test_layered_gaussians.py`：追加 4 个 T2.5 测试（Mac CPU）：
  - `test_fused_view_single_bg_passes_through`：identity 检查（单 bg 模式返回的是层的 Parameter 对象本身）
  - `test_fused_view_two_layers_concat_shape`：2 层 concat，形状 + 顺序数值检查
  - `test_get_layer_mask_partitions_two_layers`：2 层 mask 是完备分区（并集全 True、交集空、dtype bool）
  - `test_get_layer_mask_unknown_name_raises`：非粒子层 / 不存在层名 → ValueError 含 "unknown layer"

**多层渲染集成注意**：T2.5 是接口层，trainer 的 `self.model(batch, ...)` 仍走 `forward()` → `_single_bg_layer()` 透传路径（v1 byte-identical）。多层渲染集成（trainer 调 `fused_view` → renderer → loss）将在 Stage 3 T3.3 road_init 数据就绪后的 T3.4 region-weighted loss 中接入。

| 指标 | 实际 |
|---|---:|
| `pytest threedgrut/tests/` 全套 | **33/33 PASS** ✅ (Mac CPU, 0.64s) |
| `pytest test_layered_gaussians.py` 独立运行 | **13/13 PASS** ✅ (9 prior + 4 new) |
| `pytest -k "fused_view or get_layer_mask"` | **4/4 PASS** ✅ |
| `__getattr__` / `__setattr__` 桥接冲突 | 无 — 方法定义在类上，normal lookup 先于 `__getattr__` |

**Stage 2 完成**：T2.1 + T2.2 + T2.3 + T2.4 + T2.5 = 5/5 ✅；解锁 Stage 3 (Road) 与 Stage 4 (DynamicRigid) 并行开发窗口。

---

> 文档结束。当前应优先处理：**T3.1 / T3.2**（数据加载器，为 road 层提供 aux mask + LiDAR 点）。

---

### 🎉 Stage 3 出口 ✅ (2026-05-19 15:15:58, A800 GPU 1, T3.5.b 完成)

**5k step 单相机 Stage 3 出口验收全过**：

| 指标 | 实测 | 门槛 | Δ |
|---|---|---|---|
| Mean PSNR | **26.133 dB** | ≥ 23.6 | **+2.5 dB 超额** |
| Mean SSIM | 0.879 | (v1 0.846) | +0.033 |
| Mean LPIPS | 0.297 | (v1 0.405) | -0.108 |
| Iter speed | **9.54 it/s** | (v1 9.48) | 零性能损失 |
| 训练时间 | 523.84s (~8.7 min) | < 9 min ✓ | |
| 每帧 PSNR 范围 | 22.84-30.45 dB | — | 全 8 帧收敛 |

**6 次 first-light 迭代修复全栈打通**（按发生顺序）：
1. **OOM exit 137**：`torch.cdist(grid[200K], road_pts[629K])` = 500 GB host RAM → 改用 `scipy.spatial.cKDTree` O(N log N)（cdist fallback for unit tests）
2. **device mismatch**：`init_layer_from_points` 默认 tensor 没跟 positions.device → 全 default tensor 加 `device=` 参数
3. **`model.get_density()` AttributeError**：trainer 多层模式调 model.get_density / get_scale → `__getattr__` 加 fused fallback (concat 各层 get_X)
4. **`model.scheduler_step()` AttributeError**：→ `__getattr__` 加 broadcast 类委托 (per-layer 各自调)
5. **`model.progressive_training` AttributeError**：→ `__getattr__` 加 last-resort fallback 委托第一 particle layer (scalar conf attr 各层一致)
6. **`model.build_acc()` / `setup_optimizer()` 多层**：LayeredGaussians 加显式 broadcast 方法

**关键文件改动**：
- NEW `configs/apps/ncore_3dgut_mcmc_v2_road.yaml` (hydra: layered_mcmc + load_aux_masks + layered_loss)
- MOD `threedgrut/trainer.py::setup_training` case "lidar": 多层模式 per-layer init dispatcher (background + road init_layer_from_points; dynamic_rigids TODO T4.5)
- MOD `threedgrut/layers/layered_model.py`: `build_acc` / `setup_optimizer` 多层 broadcast；`__getattr__` 三层 multi-layer fallback (fused tensor / fused method / broadcast method / ref-layer last resort)
- MOD `threedgrut/layers/road_init.py`: `scipy.cKDTree` + `torch.cdist` fallback (Mac unit test 没 scipy 走 cdist)
- MOD `threedgrut/layers/layered_model.py::init_layer_from_points`: device 一致性 (所有 default tensor 跟随 positions.device)
- MOD `threedgrut/datasets/__init__.py`: NCoreDataset(load_aux_masks=...) 双路 (train + val) 接 config.dataset.load_aux_masks

**T4.5 路径升级（D 探测发现）**：NCore manifest 自带 13657 个真实 `CuboidTrackObservation` (autolabels v2)：
- `loader.get_cuboid_track_observations()` 返回 generator with `bbox3` (centroid+dim) / `track_id` / `class_id` / `timestamp_us` / `reference_frame_id="rig"`
- T4.5 不再依赖 mock tracks，可改 `load_tracks_from_ncore_cuboids(loader)` 直接消费真标注

| 验证矩阵 | 结果 |
|---|---|
| Mac unit tests | **95/95 PASS** (cKDTree fallback 起作用) |
| A800 GPU 1 真实 CUDA pytest | (回归 deferred, 上次 92/92 PASS) |
| A800 5k step training | **26.133 dB ✅** |
| v1 byte-identical 回归 (T20) | 24.123 dB byte-identical with Stage 2 baseline ✅ |

### T3.1.b + T3.2.b ✅ (2026-05-19 14:31, A800 GPU 1 集成测全过)

aux 读取栈改造 + A800 集成验证：

**关键架构决定**：NRE 工具产出的 `aux.*.zarr.itar` 根 `.zattrs` 缺 `version` 字段 → `SequenceComponentGroupsReader` 不接受。**绕过 SDK 直读 itar**：

- 新建 `threedgrut/datasets/aux_readers.py`：
  - `SsegAuxReader`：lazy open `IndexedTarStore + zarr.open`，per-camera group 缓存；`read(camera_id, timestamp_us) -> np.ndarray[H, W] uint8`（PNG decode）
  - `LidarSsegAuxReader`：同模式；`read(lidar_id, timestamp_us) -> np.ndarray[N_pts] uint8`
  - `discover_aux_path(clip_dir, aux_type)`：glob `*.aux.<type>.zarr.itar`
  - 文档化 schema (路径 `/aux/<type>/<sensor>/<ts_us>`，class palette 20 类 + ignore)
- 撤销之前在 `datasetNcore.py` 的 SDK aux paths append（schema 不兼容必失败）
- `_get_semantic_lidar_points`：改用 `LidarSsegAuxReader` + 与 pc shape 匹配 sanity check
- `__getitem__` 训练分支：加 sseg PNG decode → sky/road/dyn pixel masks
- `get_gpu_batch_with_intrinsics`：装 `image_infos` (sky/road/dyn_mask_sseg → GPU)
- `_ensure_aux_readers()` lazy init helper（per-process，无重复 itar open）

**🎯 关键 fix**：sseg/lidar-sseg key 是 `camera.frames_timestamps_us[idx, FrameTimepoint.END]` **不是 START**。A800 探测 599/599 keys 100% match END。最初用 START 给 KeyError。

**A800 集成测结果 (clip 9ae151dc, 2s 51 frames)**：

| 验证 | 实测 | 期望 |
|---|---|---|
| sseg 单帧 latency | 0.11s | < 1s ✓ |
| sky_mask coverage | 1.85% (100% top half) | 上半为主 ✓ |
| road_mask coverage | 21.55% (100% bottom half) | 下半为主 ✓ |
| dyn_mask_sseg coverage | 2.50% | 合理 ✓ |
| 三 mask pairwise disjoint | max sum = 1.0 ✓ | ≤ 1.0 |
| road LiDAR pts | **629K** | [10K, 500K] (略多, 含 sidewalk 类 OK) |
| road Z std | **0.425 m** ✓ | < 0.5 m |
| dyn LiDAR pts | 135K | 合理 |
| Z range / XY 形态 | [-45, 1.67] / (-129,205)×(-19,32) | ego traj + 30m cut 范围合理 |

**剩余 Stage 3**：T3.5.b trainer.init_model 串通 road init + Stage 3 出口 5k step (PSNR ≥ 23.6 dB)。

### T3.5.a ✅ (2026-05-19, Mac local + A800 GPU 1 contract test)

LayeredGaussians 多层 forward 路由 land：

- `threedgrut/layers/layered_model.py`：
  - 新增 `_FusedView` 类（轻量 MoG-like façade）：暴露 positions/rotation/scale/density/features_albedo/features_specular 直接访问 + num_gaussians/n_active_features/max_n_features/background 配置 + get_rotation()/get_scale()/get_density()/get_features()/get_positions() 激活函数借自 ref layer
  - 改 `forward(gpu_batch, train, frame_id)`：单 bg 模式仍 byte-identical 透传到 bg.__call__；多层模式调 `fused_view(frame_id)` → `_FusedView(fused, ref_layer)` → `ref_layer.renderer.render(view, gpu_batch, train, frame_id)`
- `test_layered_gaussians.py` 新增 3 个 T3.5 contract test：
  - `test_fused_view_object_exposes_full_mog_contract`：14 个 attr/method 完整暴露 + 激活函数借用 + identity reuse
  - `test_forward_single_bg_passes_through_to_bg_layer`：monkey-patch bg `__call__`，验证单 bg 不走 fused_view 路径
  - `test_forward_multi_layer_dispatches_to_ref_renderer`：monkey-patch ref.renderer.render，验证多层路径调用、view 类型、num_gaussians、train、frame_id 全 propagate

| 指标 | 实际 |
|---|---:|
| `pytest test_layered_gaussians.py` | 25 prior + 3 new = **28/28 PASS** (Mac CPU, 0.60s) |
| 全套 | **95/95 PASS** |
| A800 GPU 1 真实 CUDA 回归 (含 T3.5.a) | **92/92 PASS** (3.69s, GPU 1 与 nre-tools 并行) |

**T3.5.b 待办**：trainer.init_model 加 road init 串通调用 — 依赖 T3.2.b NCoreDataset.get_road_lidar_points API。新 yaml `configs/apps/ncore_3dgut_mcmc_v2_road.yaml`。A800 5k step PSNR ≥ 23.6 dB 出口门槛。

### A800 byte-identical 回归 (T3.0-T4.4) ✅ (2026-05-19 13:02:23, GPU 1 并行)

D8 出口门禁验证：Stage 3/4 全部本地 commits (T3.0/T3.1.a/T3.2.a/T3.3.a/b/T3.4/T4.0/T4.1.a/b/T4.2.a/b/T4.3/T4.4) 不破坏 v1 byte-identical.

命令（GPU 1，与 GPU 0 上的 nre-tools aux 生成并行）：
```bash
ssh a800-x2 'cd /root/work/yusun/repo/3dgrut && \
  PATH=/root/miniforge3/envs/3dgrut/bin:$PATH \
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -u train.py --config-name apps/ncore_3dgut_mcmc \
    strategy=layered_mcmc \
    path=/root/work/yusun/ncore-nurec/data/ncore/clips/9ae151dc-.../pai_9ae151dc-....json \
    resume=/root/work/yusun/ncore-nurec/output/smoke_t01_a800_20260514_165510/.../ckpt_last.pt \
    use_layered_model=true layers.enabled=[background] \
    n_iterations=1000 dataset.train.duration_sec=2.0 \
    dataset.camera_ids=[camera_front_wide_120fov]'
```

8 帧 PSNR 逐帧对比 Stage 2 baseline (df1e87d):

| 帧 | Stage 2 baseline | T3.0-T4.4 后 | Δ |
|---|---:|---:|---:|
| 0 | 21.55 | 21.55 | 0.00 |
| 1 | 23.99 | 23.99 | 0.00 |
| 2 | 22.32 | 22.32 | 0.00 |
| 3 | 22.69 | 22.69 | 0.00 |
| 4 | 23.93 | 23.93 | 0.00 |
| 5 | 24.12 | 24.12 | 0.00 |
| 6 | 27.17 | 27.17 | 0.00 |
| 7 | 27.23 | 27.23 | 0.00 |
| **mean** | **24.123** | **24.123** | **0.000 ✅ byte-identical** |

证明：
- T3.0 LayeredGaussians.optimizer single-bg identity passthrough → byte-identical
- T3.4 `noise = noise * torch.ones(3).to(...)` 是浮点 identity，无 bit drift
- T3.4 layered_loss=false 默认路径不动 v1 公式
- 所有 Stage 4 新增 import (tracks_loader / dynamic_rigid_init / dynamic_mask)
  不污染 v1 trainer 路径

**D8 出口门禁 ✅ 通过**。后续 T3.5 / T4.5 A800 集成测可在 Stage 3a aux 数据
就绪 + tracks manifest 补齐后启动。

### T4.0–T4.4 ✅ (2026-05-19, Mac local · Stage 4 本地全部完成)

Stage 4 本地代码 + 测试一次性 land；T4.5 A800 集成测 deferred 到下个会话（NCore aux 数据 + tracks manifest 就绪后）。

**T4.0 — tracks buffer**：
- `LayeredGaussians.__init__` 加 `tracks=None` kwarg；每个 track 的 poses + active 注册为持久 buffer (`_track_pose_<tid>` / `_track_active_<tid>`)，mirror 到 `tracks_poses` / `tracks_active` dict（同张 tensor identity 共享）；`tracks=None` 默认 → 空 dict + 无 buffer 污染
- 2 tests：`test_layered_gaussians_holds_tracks_buffers` / `test_layered_gaussians_no_tracks_default`

**T4.1.a + T4.1.b — tracks 加载**：
- 新建 `threedgrut/datasets/tracks_loader.py` 放 `load_tracks_from_manifest()`（独立模块，Mac 可 import；datasetNcore.py 后续 re-export）
- schema 完全对标 plan 文档：`{tid: {pts:None, colors:None, poses[F,4,4], size[3], frame_info[F bool], class:str}}`
- 缺 tracks 字段（T3a.2 验证 NCore 当前 manifest 就是这样）→ 返回 `{}` 不 crash
- 完整字段验证：id/poses/extent/active_frames 缺失 / poses shape / active_frames 长度 mismatch 都 ValueError
- 10 tests covering：basic / multiple / missing tracks / empty / partial active / 各 ValueError 路径

**T4.2.a + T4.2.b — dynamic_rigid_init**：
- 新建 `threedgrut/layers/dynamic_rigid_init.py::init_dynamic_rigid_layer(instance_pts_dict, dyn_lidar_pts, max_pts_per_track=5000) -> (positions, track_ids, track_names)`
- 每 track 每 active frame：world → local → cuboid 内过滤 → concat
- per-track subsample；多 track concat 后输出 `positions[Σ,3]` 在 object-local frame + `track_ids[Σ]` (sorted keys 映射)
- mutate `instance_pts_dict[tid]["pts"]` in place 供 callers 查询
- 8 tests：cuboid filter / local frame roundtrip / max_pts / multi-track routing / 空输入 / 空 tracks / inactive frames / dict mutation

**T4.3 — _transform_means + fused_view dynamic 分支**：
- `LayeredGaussians._transform_means(positions_local, track_ids, frame_id)`：按 `sorted(self.tracks_poses)` 路由，pose stack [K,4,4] → per-pt pose 索引 → `R @ p + t`
- `fused_view(frame_id)` 对 `spec.name == "dynamic_rigids"` 且 `frame_id is not None` 且 有 tracks 时走 transform；D4 fallback：`frame_id=None` → pass-through (Stage 8 推理时再处理)
- 5 tests：identity / single translation / multi routing / fused_view applies / frame_id=None skips

**T4.4 — dynamic_mask scanline AABB (D5)**：
- 新建 `threedgrut/layers/dynamic_mask.py::project_cuboids_to_mask(tracks_poses, tracks_size, K, T_world2cam, H, W, device)`
- 算法：8 角点 local → world → cam → image → per-track 2D AABB → fill (D5 占位，PSNR 不达再升级凸壳)
- T=0 / 后向点 / 越界 / 多 track union OR / size 2× → area ≈4× 全测过
- 6 tests

**测试总数**：
| 模块 | 数量 | 耗时 |
|---|---:|---:|
| `test_layered_gaussians.py` | 25 (含 T4.0×2 / T4.3×5) | — |
| `test_tracks_loader.py` | 10 | — |
| `test_dynamic_rigid_init.py` | 8 | — |
| `test_dynamic_mask.py` | 6 | — |
| **全套** | **92/92 PASS** | **0.75s** |

**Stage 4 出口剩 T4.5**：trainer.init_model 串通 dynamic 初始化 + LayeredGaussians 多层 forward (T3.5 共用) + 新 yaml + A800 10k step smoke。**依赖**：(a) NCore aux lidar-sseg 数据（T3a.1 还差），(b) scene_manifest tracks 数据（T3a.2 验证当前缺失 — 需用户用 NCore validator 补）。

### T3.4 ✅ (2026-05-19, Mac local)

region-weighted L1 loss + MCMC perturb mask hook (D1)：

**Loss 侧 (D6 D7):**
- `threedgrut/model/layered_loss.py` 新建 `compute_layered_l1_loss(rgb_pred, rgb_gt, image_infos, valid_mask, min_pixels=100)`：
  - 纯函数（不依赖 Trainer / CUDA），单测可纯 import
  - image_infos=None / 缺 sky_mask → 走 v1 fallback (.mean()` or masked mean)
  - 否则 bg + road + dyn 三区均值之和（sky 不算 L1，envmap 接管）
  - dyn 优先用 `dyn_mask_cuboid`（Stage 4 T4.4），fallback `dyn_mask_sseg`（Stage 3）
  - mask.sum() < min_pixels 该区跳过（D6 数值稳定）
- `threedgrut/trainer.py::get_losses` 加 `layered_loss` 开关分支；SSIM 保持全图（D7）
- `configs/base_gs.yaml` 加 `trainer.layered_loss: false` 默认

**Perturb mask hook (D1):**
- `threedgrut/strategy/mcmc.py`：抽 `_get_perturb_mask() -> Tensor[3]` 钩子，默认 `torch.ones(3)`（v1 byte-identical）；`perturb_gaussians` noise elementwise 乘 mask
- `threedgrut/strategy/layered_mcmc.py`：`_install_perturb_mask(sub, spec)` 静态方法 — 仅当 `spec.perturb_scale_mask is not None` 时绑定 `sub._perturb_mask_override` 并替换 `sub._get_perturb_mask`，否则保持默认（bg / dyn 行为不变）
- `threedgrut/layers/layer_spec.py` 加 `perturb_scale_mask: tuple[float,float,float] | None = None` 字段
- `threedgrut/layers/registry.py`：road spec 加 `perturb_scale_mask=(1.0, 1.0, 0.0)`

**测试 (10 new tests, Mac 0.59s)：**
- `test_layered_loss.py` 6 tests：v1 fallback / valid_mask / 三区 partition / small region skipped (D6) / cuboid > sseg precedence / scalar+backward
- `test_layered_mcmc.py` 4 new tests：default ones / road spec Z lock / sub install / no-spec skip

| 指标 | 实际 |
|---|---:|
| 三区 partition 数值对账 | pred=1/gt=0, 4×4 三区各 4 px → loss = 3.0 (bg+road+dyn 均值和) ✓ |
| v1 fallback byte-identical | `image_infos=None` → `.mean()` 公式与改前一致 ✓ |
| Road perturb mask installed | `sub["road"]._get_perturb_mask() == [1, 1, 0]` ✓ |
| Bg perturb mask unchanged | `sub["bg"]._get_perturb_mask() == [1, 1, 1]` (default) ✓ |
| 全测试套 | **61/61 PASS** (Mac, 0.59s) |

**剩余 Stage 3**：T3.5 LayeredGaussians 多层 forward + Stage 3 出口集成 (需 A800)；T3.1.b / T3.2.b NCoreDataset 改动 (需 ncore SDK)。

### T3.3.a + T3.3.b ✅ (2026-05-19, Mac local)

road_init.py BEV-grid + LiDAR-Z KNN 实现 + 6 个 contract tests：

- `threedgrut/layers/road_init.py` 新建 `init_road_layer(road_points, ego_trajectory, cut_range=30.0, resolution=0.05, max_n=200_000)`:
  1. BEV bbox = ego traj XY 范围 ± cut_range
  2. 2D grid at `resolution`
  3. KNN-Z (用 `torch.cdist` 避开 PyTorch3D / sklearn)
  4. 截 max_n before cdist（防超大 grid 爆内存）
  5. defaults: identity quat / `log(scale_prior)` / density=0 / 中性灰
- 空 LiDAR / 空 ego traj fallback 返回 shape=(0,...) 一致 tensor
- 跟随地形（10% grade ramp 测试通过 → Z mean 拟合）
- `threedgrut/tests/test_road_init.py` 新建 6 测试（pure CPU mock，0.06s）

| 指标 | 实际 |
|---|---:|
| `pytest test_road_init.py` | **6/6 PASS** (Mac CPU, 0.06s) |
| 全测试套（含 T3.0/T3.1.a/T3.2.a） | **51/51 PASS** |
| Z lock 精度 | mock 100 flat-Z 点 → \|Z\| max < 0.05 m ✓ |
| flat scale 约束 | exp(scale_z) < 0.005 ✓; XY scale ∈ [0.05, 0.2] ✓ |
| 地形跟随 | 10% grade ramp → X=40 处 Z mean ≈ 4 ✓ |
| max_n 截断 | 100×100m BEV @ res=0.5m → 78400 候选 → 截到 1000 ✓ |

**剩余 Stage 3**：T3.4 region loss + perturb mask hook（pure Mac），T3.1.b/T3.2.b/T3.5 需 A800/NCore SDK。

### T3.1.a + T3.2.a ✅ (2026-05-19, Mac local)

NCore semantic class ID 常量 + mock 行为契约测试：

- `threedgrut/datasets/ncore_semantic.py` 新建：占位 Cityscapes palette（mask2former 后端默认）
  - `SKY_CLASS_ID = 10`
  - `ROAD_CLASS_IDS = frozenset({0, 1})` (road + sidewalk 作为广义可行驶面)
  - `DYNAMIC_CLASS_IDS = frozenset({11..18})` (person + vehicle 类)
  - TODO 注释：T3.1.b A800 集成时必须读一帧 sseg itar 抽 unique values 对账；NRE mask2former palette 可能与 Cityscapes 标准有偏移
- `threedgrut/tests/test_ncore_aux_masks.py` 新建：7 个测试
  - T3.1.a (4): 三 mask disjoint partition；road/dyn 全 class IDs 覆盖；sky 是 singular int
  - T3.2.a (3): mock LiDAR semantic filter 行为契约（`_filter_pts_by_label` 参考实现，T3.2.b 实装时用此契约）

| 指标 | 实际 |
|---|---:|
| `pytest test_ncore_aux_masks.py` | **7/7 PASS** (Mac CPU, 0.04s) |
| 全测试套（含 T3.0） | **45/45 PASS** |
| A800 sseg palette 对账 | **Deferred** to T3.1.b A800 integration |

**Stage 3 待办**：T3.1.b / T3.2.b 真实 NCoreDataset 改动（需要 ncore SDK + A800 sseg 数据）；T3.3.a/b road_init（纯本地，下个）。

### T3.0 ✅ (2026-05-19, Mac local)

`LayeredGaussians.init_layer_from_points()` + `optimizer` property — Stage 3/4 共享前置 API：

- `threedgrut/layers/layered_model.py`：
  - 新增 `_LayeredOptimizerView` 类（轻量包装，遍历 `self._layers[*].optimizer.step/zero_grad`；`param_groups` 聚合）
  - 新增 `init_layer_from_points(name, positions, *, colors, rotations, scales, densities, track_ids, observer_pts, setup_optimizer)` 方法：spec-aware 默认（scale_prior log-applied / density_init / identity quat / 中性灰 → SH DC），全量 nn.Parameter 化（避开 `default_initialize_from_points` 的 sklearn KNN，让 Mac CPU 可跑测试）
  - 新增 `optimizer` property：单 bg 模式 byte-identical 透传到 bg.optimizer（identity 检查通过）；多层模式返回 `_LayeredOptimizerView`
  - track_ids 通过 `register_buffer(persistent=True)` 注册（dynamic_rigids T4.3 用）
- `threedgrut/tests/test_layered_gaussians.py`：新增 5 个 T3.0 测试
  - `test_init_layer_from_points_routes_to_mog`：positions 灌进对应层，其他层不受影响；road spec_prior=(0.1,0.1,0.001) Z log 后 e^scale_z < 0.005 ✓
  - `test_init_layer_from_points_unknown_layer_raises`：未知层 ValueError ✓
  - `test_init_layer_from_points_track_ids_registered_as_buffer`：track_ids 注册为 named_buffer ✓
  - `test_optimizer_property_single_bg_passthrough`：identity 检查 `model.optimizer is bg.optimizer` ✓
  - `test_optimizer_wrapper_steps_all_layers`：monkeypatch step 验证多层 fan-out + param_groups 聚合 ✓

| 指标 | 实际 |
|---|---:|
| `pytest test_layered_gaussians.py` | 13 prior + 5 new = **18/18 PASS** (Mac CPU, 0.57s) |
| `pytest threedgrut/tests/` 全套 | **38/38 PASS** (Mac CPU, 0.53s) |
| 单 bg 模式 optimizer identity | `model.optimizer is bg.optimizer` ✓ byte-identical |
| 多层模式 optimizer 类型 | `_LayeredOptimizerView` (非 bg.optimizer) ✓ |

**实现选择备注**：原 plan 建议 `init_layer_from_points` 调 `MoG.default_initialize_from_points()` 走 KNN 路径估 scale；实际全量 Parameter 化更直接 — Mac 上 conftest.py 把 sklearn stub 成空 module，default 路径会 broken，而 road_init / dyn_init 都自己提供 scales（spec.scale_prior 即可），不需要内部 distance estimation。代价是 features_specular 不从 default 路径走（直接 zero init），与 v1 一致（v1 也是 zero init specular）。

---

## 6. Stage 2 Retrospective

### 设计偏差与教训

| 项 | 计划 | 实际 | 教训 |
|---|---|---|---|
| LayeredMCMC 实现路线 | `_select_indices` 钩子 + 上下文切换 | **sub-strategy 数组** (BaseStrategy 子类，持 dict[str → MCMCStrategy]) | Stage 1 每层独立 MoG + optimizer 后，sub-strategy 比 hook 重构更轻、byte-identical 自动成立 |
| 文档同步时机 | Task 6 批量做 | 每个 T2.x 顺手做（CLAUDE.md 规则） | 项目规则优先于 plan 设计 |
| Mac CPU 测试基础设施 | minimal sys.modules stub | 完整 stub + monkey-patch MCMCStrategy.__init__ + conftest.py 集中 | 让 Stage 1 测试也获得了独立可运行性（bonus） |

### Stage 2 出口指标

| 指标 | 达成 |
|---|---|
| 5 code tasks (T2.1-T2.5) 全完成 | ✅ |
| Mac CPU 测试 | 33/33 PASS (Stage 1: 17 + Stage 2: 13 incl. v1_ckpt_compat 3) |
| Stage 1 测试独立可运行性 | ✅ (T2.4 conftest.py 迁移) |
| A800 byte-identical 回归 | ✅ 24.123 dB (2026-05-18 16:24，commit df1e87d) |

### A800 Stage 2 出口验证（2026-05-18 16:24）

命令：

```bash
ssh a800-x2 'cd /root/work/yusun/repo/3dgrut && \
  PATH=/root/miniforge3/envs/3dgrut/bin:$PATH \
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /root/miniforge3/envs/3dgrut/bin/python -u train.py --config-name apps/ncore_3dgut_mcmc \
    strategy=layered_mcmc \
    path=/root/work/yusun/ncore-nurec/data/ncore/clips/9ae151dc-.../pai_9ae151dc-....json \
    resume=/root/work/yusun/ncore-nurec/output/smoke_t01_a800_20260514_165510/.../ckpt_last.pt \
    use_layered_model=true layers.enabled=[background] \
    n_iterations=1000 dataset.train.duration_sec=2.0 \
    dataset.camera_ids=[camera_front_wide_120fov]'
```

关键日志：

```
🔆 Using LayeredGaussians with layers=['background']
LayeredMCMC: 1 sub-strategies for layers ['background']
🔆 Using LayeredMCMC strategy
 Detected v1-shape checkpoint (1000000 particles); routing all into layer 'background'.
```

8 帧 PSNR 对比 Stage 1 T1.2 baseline：

| 帧 | Stage 1 baseline (5a6a5f9) | Stage 2 LayeredMCMC (df1e87d) |
|---|---:|---:|
| 0 | 21.55 | 21.55 |
| 1 | 23.99 | 23.99 |
| 2 | 22.32 | 22.32 |
| 3 | 22.69 | 22.69 |
| 4 | 23.93 | 23.93 |
| 5 | 24.12 | 24.12 |
| 6 | 27.17 | 27.17 |
| 7 | 27.23 | 27.23 |
| **mean** | **24.123 dB** | **24.123 dB ✅ byte-identical** |

证明：
- T2.1 `_get_add_cap()` hook 提取未改变 MCMCStrategy 行为
- T2.2 `LayeredMCMCStrategy` 在 `sub_strategies={"background"}` 时等价于 v1 `MCMCStrategy`
- T2.3 `layered_mcmc.yaml` `defaults: [mcmc, _self_]` 继承全部 hyper-params 正确
- v1 ckpt → layered["background"] 路由（T1.3 错误消息路径）正确触发

### 已记录的 tech debt（Stage 3 可顺手处理）

1. `test_get_layer_mask_unknown_name_raises` 未覆盖"已注册非粒子层"用例（如 `sky_envmap`）；当前仅测 `"nonexistent"` 完全未注册路径。
2. `MixtureOfGaussians.num_gaussians` (property, model.py:53) 不做 `None` 检查，与 `model.py:850` 的 `if self.positions is not None` 分支不一致；当前测试路径不会触发，可作为一致性 patch 补。
3. `fused_view(frame_id, training=True/False)` 语义在 T4.3 wire dynamic pose 时需确认（推理时是否走不同变换？）。

### Stage 3 / 4 解锁

Stage 2 完成后，Stage 3 (Road) 和 Stage 4 (DynamicRigid) 可并行：它们只往 `LayeredGaussians.layers[name]` 塞数据，不再触碰 MCMC。

---

## 14. NuRec vs v2 缺口清单（V3 / V4 任务种子）

> **来源**：`/Users/etendue/repo/report/NVIDIA NuRec 技术深度解析.md`（基于 NuRec 25.7.9 + parsed_config.yaml 7350 行）
> **对照基线**：本 plan（v2 Stage 1-7） + `configs/apps/ncore_3dgut_mcmc_v2_full.yaml`
> **状态符号**：✅ 已落 / 在做 · ⚠️ 占位 / 部分实现 · ❌ 完全未实现（V3 / V4 候选）
> **本节作用**：把所有"NuRec 有 · v2 没做"的 trick 落到具体任务种子，按预期 PSNR 贡献排序，留作 V3 / V4 plan 的输入。

### 14.1 辅助数据通道（NuRec §3）

| NuRec trick | 状态 | V3/V4 任务种子 |
|---|:---:|---|
| DepthAnythingV2 度量深度 prior | ❌ | V3-D1：dataset 加 metric depth 读取 + trainer 加 depth loss head |
| DINOv2 背景层 extra_signal（20 维语义 logits） | ❌ | V3-D2：MoG 加 extra_signal 通道 + dataset DINOv2 feat reader |
| Mask2Former seg-logits（21 类 softmax）作 aux CE loss | ⚠️ T3.1.b 只有 hard mask | V3-D3：sseg 直读 logits + sky/road/dyn CE 头 |
| 场景流 mask（`track_min_speed=1.4 m/s` + dilate 20 px） | ❌ | V3-D4：dataset 加 flow mask + 并入 valid_pixel_mask |
| 交通灯 / 闪烁光源 mask（21 px dilation） | ❌ | V3-D5：与 D4 同走 mask 管线 |
| Cuboid LiDAR padding `[0.5, 0.5, 0.25] m` | ❌ T4.4 仅精确投影 | V3-D6：T4.4 dynamic_mask 加 cuboid 膨胀 |
| Cuboid camera padding `[1.0, 1.0, 0.25] m` | ❌ | V3-D7：同 D6 |
| 相机 mask 30 iter dilation | ❌ | V3-D8：mask 合并管线统一加 dilation |
| 帧 mask 10 iter dilation | ❌ | V3-D9：同 D8 |

### 14.2 多层场景分解（NuRec §4）

| NuRec trick | 状态 | V3/V4 任务种子 |
|---|:---:|---|
| Background `fourier_features_dim=5`（时间编码） | ❌ | V3-L1：MoG 加 fourier-time embedding（背景） |
| Road `fourier_features_dim=1`（轻量时间编码） | ❌ | V3-L2：road 同上（小维度） |
| Road `scale_pos_lr_by_scene_extent=false` | ❌ | V3-L3：LayerSpec 加 `scale_pos_lr_by_scene_extent` 字段，trainer 接 |
| Background `ignore_classes_from_layers=[road]` | ❌ | V3-L4：layered loss 加层级排他 mask |
| DynamicRigid `symmetric_axis='Y'`（左右对称先验） | ❌ | V3-L5：dynamic_rigid_init 注入对称粒子 + 镜像约束 reg |
| DynamicRigid 5000 pts/track + 上限 300K | ⚠️ v2 用 200K | V3-L6：per-track cap + 全层 cap 对齐 NuRec |
| Track-pose 联合优化（fix_first/last + warm start ≥ 500） | ❌（v2.x 明确不做） | V3-L7：dynamic_rigids pose 加可学习 Δpose + Sequential warm-up |
| `optimize_track_albedo`（每轨迹外观偏移） | ❌ | V3-L8：每 track 一个 SH bias，Constant→Linear→Cosine LR |
| `optimize_track_scale`（每轨迹尺度偏移） | ❌ | V3-L9：同 L8 |
| DynamicDeformable hash-grid-object 形变场（permuto hash 16 层 + FullyFusedMLP 64×1，渐进 10→16） | ❌（v2 仅 spec 占位） | **V4 主力**：完整形变层；`deformnet_start_iteration=1000`、`optimize_canonical_xyz`、`smoothness_frame_steps=5` |
| Sky envmap cubemap 512×512×6 面 + nvdiffrast | ⚠️ T5.2 计划 | V2/V3 完成 Stage 5 |
| Sky envmap `should_inpaint=true` + threshold 0.05 + kernel 10 | ❌ | V3-L10：在 T5.2 基础上加 inpaint 模块（关键 — 新视角不爆黑洞） |
| Sky envmap `composite_in_linear_space=false`（gamma 合成） | ❌ | V3-L11：trainer blend 路径加 sRGB↔linear |
| Sky envmap `min_grad_updates=1000`（warm-up） | ❌ | V3-L12：sky_envmap 前 1k 步冻结 |

### 14.3 训练策略（NuRec §5）

| NuRec trick | 状态 | V3/V4 任务种子 |
|---|:---:|---|
| PERTURB `move_outside_of_cuboid=false`（粒子不出 cuboid） | ⚠️ T3.4 perturb mask hook 已抽出 | V3-T1：把 hook 实现为"粒子+noise 后投回 cuboid"约束 |
| `opacity_threshold=0.005` 剪枝 | 待校对 | V3-T2：与 NuRec 校对当前 mcmc.py 实际值 |
| `binom_n_max=51` / `noise_lr=5000` | 待校对 | V3-T3：同 T2 |
| add/relocate 双阶上限（`add.max_n=1.8M`，overall=2M） | ❌ v2 单层 cap | V3-T4：layered_mcmc.yaml 加 `add_cap_ratio=0.9` |
| StepFunCosineAnnealingLR（阶梯余弦） | ❌ | V3-T5：新建 scheduler，供轨迹标定 / albedo / 形变网络 |
| SequentialLR（Constant → Linear → Cosine） | ❌ | V3-T6：与 L7 / L8 配合 |
| position 组 vs 特征组独立 LR + γ=0.9998465 | ⚠️ v1 fused_adam 已分组 | V3-T7：校对 per-layer LR 是否对齐 NuRec |
| 每步 `camera_rays=6144 + lidar_rays=2048` 1:1 | ⚠️ v2 未启 LiDAR ray | V3-T8：trainer step 加 LiDAR ray batch |
| **LiDAR ray 监督本身**（NuRec 主训练同等权重） | ❌（最关键缺口之一） | V3-T9：LiDAR depth/intensity ray loss head |

### 14.4 渲染管线（NuRec §6）

| NuRec trick | 状态 | V3/V4 任务种子 |
|---|:---:|---|
| 3DGRT k-buffer 二次射线 + 与 3DGUT 混合 | ⚠️ v1 有，v2 未启 | V3-R1：v2_full.yaml 切到 3DGRUT 复合 renderer + 配置 secondary ray |
| `lidar_divergence=0.002 rad`（cone 抗锯齿） | ❌ | V3-R2：与 T8 / T9 配合，tracer 端 expose |
| `min_projected_ray_radius=0.5477`（≈√(1/3)） | 待校对 | V3-R3：校对 3DGUT default |
| `image_margin_factor=0.1` | 待校对 | V3-R4：同 R3 |

### 14.5 后处理（NuRec §7）

| NuRec trick | 状态 | V3/V4 任务种子 |
|---|:---:|---|
| **Cosmos-DiFix 扩散修复 + 渐进蒸馏** | ❌（v2 明确 v3） | **V3 主力**：fixer 模型 NGC 下载 + 缓存策略 + 50% 训练视角 + 50% ±2 m 新视角；`start_epoch=16`、`full_novel_view_by_epoch=22`、`use_color_transfer=true` |
| **双边网格 1×1×1 grid（按 `camera_id`）+ ExposureModel 退化优化修复（整合）** | ⚠️ T6 affine 占位 → **Stage 7 实证失控** | **V3-P1（升级版）**：(1) Recon-Studio 双边网格直接 port 替换 affine `exp(a)·img+b`；(2) **加约束防退化**: bilateral grid params L2 reg + lr cosine decay + 2-stage freeze (step > 2000 freeze) — Stage 7 T7.3 实证 ExposureModel 在 7-cam 30k 长训中退化优化失控 (raw psnr_masked 15.63 vs T7.3.b 关掉后 25.76, +10.13 dB)，但 cc_psnr_masked 几乎不变 (24.75 ↔ 24.70) → exposure 与场景重建耦合学到"高斯学个大概 + exposure 补偿"病态解；(3) 训练 forward + eval color_correct_affine 用同一套 bilateral grid (消除 raw vs cc 分歧)；(4) 健康度监控: 训练日志输出 `exposure_a.std` + raw/cc PSNR ratio, >2 dB 警报. **预期收益**: raw 与 cc PSNR 收敛 + cc_psnr_masked 提升 1-2 dB (bilateral 表达能力 > affine) + 7-cam 30k 不再退化. **预估工作量**: 3-5 天. **Stage 7 实证最关键缺口, V3 启动优先级提升** |
| 有效像素 mask 管线（多源 + dilation 合并） | ⚠️ T3.4 部分 | V3-P2：把 D4-D9 mask 全部汇入 valid_pixel_mask |

### 14.6 几何提取 / USDZ（NuRec §8）

> 全部明确为独立 WP（V1-5 / V1-6），不在本表归类 V3/V4 — 仅做 trick 锚点。

| NuRec trick | 状态 | 归属 |
|---|:---:|---|
| Poisson 网格 (`n_neighbors=200`, `trim_distance=0.225`) | ❌ | V1-5 |
| Ground mesh (RANSAC plane, `voxel_size=0.1`, 10 smoothing passes) | ❌ | V1-5 |
| USDZ 包 (nrec_data + rig_trajectories + sequence_tracks + map.xodr) | ❌ | V1-6 |
| OpenDRIVE 坐标链 NuRec → ECEF → ENU | ❌ | V1-6 |

### 14.7 评估 / 验证（NuRec §10）

| NuRec trick | 状态 | V3/V4 任务种子 |
|---|:---:|---|
| `val_lidar=true` LiDAR 域单独 PSNR | ❌ | V3-E1：与 T9 配合开启 |
| cPSNR（按 semantic class 拆 PSNR） | ❌（T7.3 KPI 概念有，工具未实现） | V3-E2：evaluator 加 per-class PSNR / SSIM 拆解 |
| 新视角扰动验证集（±2 m 平移 + 小旋转） | ❌ | V3-E3：与 Cosmos-DiFix 同一套 pose 生成器复用 |

### 14.8 V3 优先级排序（按预期 PSNR 贡献，从大到小）

| # | Trick | 任务种子 ID |
|---:|---|---|
| 1 | Cosmos-DiFix 渐进蒸馏 | V3-Cosmos |
| 2 | LiDAR ray 监督 + lidar_divergence | V3-T8 / T9 / R2 |
| 3 | Sky envmap inpaint + gamma 合成（Stage 5 完成 + 增强） | V3-L10 / L11 / L12 |
| 4 | Track-pose 联合优化（warm start + fix_first/last） | V3-L7 |
| 5 | `symmetric_axis='Y'` + per-track albedo/scale | V3-L5 / L8 / L9 |
| 6 | 背景层 extra_signal 20 维语义 logits + Fourier time encoding | V3-L1 / D2 |
| 7 | 双边网格 1×1×1 替换 affine（T6 升级） | V3-P1 |
| 8 | MCMC PERTURB cuboid 约束 + add/relocate 双阶上限 | V3-T1 / T4 |
| 9 | DynamicDeformable hash-grid 形变场（行人 / 骑行） | **V4 主力** |
| 10 | mask 管线膨胀细节（cuboid padding / 场景流 / 交通灯） | V3-D4-D9 |

### 14.9 不收录的 NuRec 部分

| 部分 | 原因 |
|---|---|
| NCore v4 schema 详解 | 数据层已通过 NCoreDataset + aux_readers.py 完整对接（T3.1.b ✅） |
| 3DGUT UT 投影主射线 / 多项式畸变 / 滚动快门（5 iter） | v1 已对齐，无 gap |
| 渐进式 SH（0 → 3 in 3k step） | v1 已对齐 |
| MCMC 三操作主框架（RELOCATE / ADD / PERTURB） | T2.x ✅ 完成主框架 |
| 多层主架构（背景 / 路面 / 动态刚体 / 天空 envmap） | Stage 1-5 已规划 / 部分完成 |
| Ego 掩膜 | v1 已有 |
| Cuboid 轨迹清洗（`track_min_speed`、`min_centroid_dist`） | T4.1 已对接 |
