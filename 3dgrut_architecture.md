# 3DGRUT 训练架构文档

> **版本**: 基于 3DGRUT 代码库 master 分支  
> **最后更新**: 2026年  
> **语言**: 简体中文

---

## 目录

1. [顶层入口点](#1-顶层入口点)
2. [配置系统](#2-配置系统)
3. [核心 Trainer3DGRUT 类](#3-核心-trainer3dgrut-类)
4. [MixtureOfGaussians 模型](#4-mixtureofgaussians-模型)
5. [双渲染后端](#5-双渲染后端)
6. [策略系统](#6-策略系统)
7. [训练循环](#7-训练循环)
8. [损失函数](#8-损失函数)
9. [优化器与调度器](#9-优化器与调度器)
10. [数据集系统](#10-数据集系统)
11. [导出系统](#11-导出系统)
12. [后处理 PPISP](#12-后处理-ppisp)
13. [ASCII 架构图](#13-ascii-架构图)
14. [关键文件路径参考表](#14-关键文件路径参考表)
15. [技术栈附录](#15-技术栈附录)

---

## 1. 顶层入口点

### 文件

`train.py`（项目根目录）

### 流程

```
train.py
  ├── hydra.main(config_path="configs", version_base=None)  ← Hydra 装饰器
  │   └── main(conf: DictConfig)
  │       ├── logger.info("Git hash: ...")
  │       ├── from threedgrut.trainer import Trainer3DGRUT
  │       ├── trainer = Trainer3DGRUT(conf)
  │       └── trainer.run_training()
  └── if __name__ == "__main__": main()
```

### 关键细节

- **Hydra + OmegaConf** 作为配置框架。`@hydra.main` 装饰器自动解析 `configs/` 目录下的 YAML 文件，组装成 `DictConfig` 对象。
- `OmegaConf.register_new_resolver("int_list", ...)` 注册了一个自定义解析器，用于将 YAML 中的字符串列表（如 `"[7000, 30000]"`) 转换为 Python int 列表。这在配置中用于 `checkpoint.iterations` 和 `writer.log_image_views`。
- 入口点延迟导入 `Trainer3DGRUT`，以便在导入过程中触发 JIT 编译（CUDA/Slang 代码的即时编译）。
- 支持从 checkpoint、INGP、PLY 文件恢复训练的工厂方法（见注释中的 `create_from_ckpt` / `create_from_ingp` / `create_from_ply`）。
- `KeyboardInterrupt` 被捕获以优雅退出。

---

## 2. 配置系统

### YAML 层次结构

Hydra 使用 `defaults` 列表实现配置的组合与覆盖。配置从以下文件组装：

```
configs/
├── base_gs.yaml          ← 基础 GS 配置（默认 strategy=gs, render=3dgrt）
├── base_mcmc.yaml        ← 基础 MCMC 配置（override strategy=mcmc, 额外正则化）
├── apps/                 ← 完整流水线预设
│   ├── nerf_synthetic_3dgut.yaml
│   ├── nerf_synthetic_3dgrt.yaml
│   ├── colmap_3dgut.yaml
│   ├── colmap_3dgrt.yaml
│   ├── colmap_3dgut_mcmc.yaml
│   ├── colmap_3dgrt_mcmc.yaml
│   ├── ncore_3dgut.yaml
│   ├── ncore_3dgrt.yaml
│   ├── ncore_3dgut_mcmc.yaml
│   ├── ncore_3dgrt_mcmc.yaml
│   ├── scannetpp_3dgut.yaml
│   ├── scannetpp_3dgrt.yaml
│   └── cusfm_3dgut.yaml / cusfm_3dgut_mcmc.yaml
├── dataset/
│   ├── colmap.yaml
│   ├── nerf.yaml
│   ├── scannetpp.yaml
│   └── ncore.yaml
├── initialization/
│   ├── colmap.yaml
│   ├── random.yaml
│   ├── point_cloud.yaml
│   ├── checkpoint.yaml
│   ├── lidar.yaml
│   └── fused_point_cloud.yaml
├── render/
│   ├── 3dgrt.yaml
│   └── 3dgut.yaml
└── strategy/
    ├── gs.yaml
    └── mcmc.yaml
```

### 覆盖顺序

以 `apps/nerf_synthetic_3dgut.yaml` 为例：

```yaml
defaults:
  - /base_gs           # 加载 base_gs.yaml 的全部默认值
  - /dataset: nerf     # 覆盖 dataset 为 nerf 配置
  - /initialization: random  # 覆盖初始化方法为 random
  - /render: 3dgut     # 覆盖渲染方法为 3dgut
  - _self_             # 当前文件的键最后覆盖
```

**覆盖优先级（从低到高）**：
1. `base_gs.yaml` 或 `base_mcmc.yaml`
2. `dataset/*.yaml`
3. `initialization/*.yaml`
4. `render/*.yaml`
5. `strategy/*.yaml`
6. `apps/*.yaml` 中 `_self_` 之后的键
7. CLI 参数（最高优先级）

### base_gs.yaml 核心默认值

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_iterations` | 30000 | 总训练迭代数 |
| `val_frequency` | 5000 | 验证频率（步） |
| `num_workers` | 24 | DataLoader 工作进程数 |
| `seed_initialization` | 42 | 初始化随机种子 |
| `model.density_activation` | sigmoid | 密度激活函数 |
| `model.scale_activation` | exp | 缩放激活函数 |
| `model.default_density` | 0.1 | 默认密度值 |
| `model.default_scale_factor` | 1.0 | 默认缩放因子 |
| `model.bvh_update_frequency` | 1 | BVH 更新频率 |
| `model.progressive_training.feature_type` | sh | 渐进式训练特征类型 |
| `model.progressive_training.init_n_features` | 0 | 初始 SH 阶数 |
| `model.progressive_training.max_n_features` | 3 | 最大 SH 阶数 |
| `model.progressive_training.increase_frequency` | 1000 | 每 N 步增加特征维度 |
| `model.background.name` | background-color | 背景类型 |
| `model.background.color` | black | 背景颜色 |
| `loss.use_l1` | true | 启用 L1 损失 |
| `loss.lambda_l1` | 0.8 | L1 损失权重 |
| `loss.use_ssim` | true | 启用 SSIM 损失 |
| `loss.lambda_ssim` | 0.2 | SSIM 损失权重 |
| `loss.use_opacity` | false | 启用不透明度正则 |
| `loss.use_scale` | false | 启用缩放正则 |
| `optimizer.type` | adam | 优化器类型 |
| `optimizer.params.positions.lr` | 0.00016 | 位置学习率 |
| `optimizer.params.density.lr` | 0.05 | 密度学习率 |
| `optimizer.params.features_albedo.lr` | 0.0025 | 反照率特征学习率 |
| `optimizer.params.features_specular.lr` | features_albedo.lr / 20 | 镜面特征学习率 |
| `optimizer.params.rotation.lr` | 0.001 | 旋转学习率 |
| `optimizer.params.scale.lr` | 0.005 | 缩放学习率 |
| `scheduler.positions.type` | exp | 位置学习率调度器类型 |
| `scheduler.positions.lr_final` | 0.0000016 | 位置最终学习率 |
| `scheduler.positions.max_steps` | 30000 | 调度器最大步数 |
| `export_ply.enabled` | false | 训练结束导出 PLY |
| `export_usd.enabled` | false | 训练结束导出 USD |
| `export_usd.format` | standard | USD 格式 (standard/nurec) |
| `post_processing.method` | null | 后处理方法 |
| `post_processing.n_distillation_steps` | 5000 | 蒸馏步数 |

### base_mcmc.yaml 差异

与 `base_gs.yaml` 相比：

```yaml
model.default_density: 0.5       # GS: 0.1
model.default_scale_factor: 0.1  # GS: 1.0
loss.use_opacity: true
loss.lambda_opacity: 0.01        # GS: false/0.0
loss.use_scale: true
loss.lambda_scale: 0.01          # GS: false/0.0
```

---

## 3. 核心 Trainer3DGRUT 类

### 文件

`threedgrut/trainer.py`

### 类成员

| 成员 | 类型 | 说明 |
|------|------|------|
| `model` | `MixtureOfGaussians` | 高斯模型 |
| `train_dataset` | `BoundedMultiViewDataset` | 训练数据集 |
| `val_dataset` | `BoundedMultiViewDataset` | 验证数据集 |
| `train_dataloader` | `MultiEpochsDataLoader` | 训练数据加载器 |
| `val_dataloader` | `DataLoader` | 验证数据加载器 |
| `scene_extent` | `float` | 场景范围（默认 1.0） |
| `scene_bbox` | `tuple[Tensor, Tensor]` | 场景包围盒 (min, max) |
| `strategy` | `BaseStrategy` | 高斯优化策略（GS 或 MCMC） |
| `criterions` | `Dict` | 评估指标：psnr, ssim, lpips |
| `tracking` | `Dict` | 训练追踪：writer, run_name, output_dir |
| `post_processing` | `Optional[nn.Module]` | 后处理模块（PPISP） |
| `post_processing_optimizers` | `Optional[list]` | 后处理优化器 |
| `post_processing_schedulers` | `Optional[list]` | 后处理调度器 |
| `_distillation_start_step` | `int` | 蒸馏开始步（-1 = 禁用） |
| `gui` | `GUI` 或 `ViserGUI` | 交互式可视化界面 |
| `global_step` | `int` | 当前全局迭代步数 |
| `n_iterations` | `int` | 总训练迭代数 |
| `n_epochs` | `int` | 总训练轮数 |

### `__init__` 初始化流程

```
__init__(conf, device)
  ├── 1. 存储 conf, device, global_step, n_iterations, n_epochs, val_frequency
  ├── 2. init_dataloaders(conf)
  │     └── datasets.make(name=conf.dataset.type) → train_dataset, val_dataset
  │     └── configure_dataloader_for_platform() → DataLoader kwargs
  ├── 3. init_scene_extents(train_dataset)
  │     └── train_dataset.get_scene_extent() / get_scene_bbox()
  ├── 4. init_model(conf, scene_extent)
  │     └── MixtureOfGaussians(conf, scene_extent)
  │         ├── 初始化所有参数为空 Tensor
  │         ├── 设置激活函数 (sigmoid/exp/normalize)
  │         ├── 创建背景模型
  │         └── 创建渲染器 (threedgrt_tracer.Tracer 或 threedgut_tracer.Tracer)
  ├── 5. init_densification_and_pruning_strategy(conf)
  │     └── match conf.strategy.method:
  │         case "GSStrategy"    → GSStrategy(conf, model)
  │         case "MCMCStrategy"  → MCMCStrategy(conf, model)
  ├── 6. init_metrics()
  │     └── PSNR, SSIM, LPIPS (torchmetrics)
  ├── 7. setup_training(conf, model, train_dataset)  ← 核心初始化
  ├── 8. init_experiments_tracking(conf)
  │     └── TensorBoard / WandB SummaryWriter, 保存 parsed.yaml
  ├── 9. init_post_processing(conf)
  │     └── 如果 method == "ppisp" → 初始化 PPISP 模块
  └── 10. init_gui(conf, ...)
        └── 如果 with_gui → polyscope GUI
        └── 如果 with_viser_gui → viser 浏览器 GUI
```

### `setup_training` 方法

这是最关键的方法，处理模型初始化的所有分支：

```
setup_training(conf, model, train_dataset)
  │
  ├── 路径 A: conf.resume 非空 → 从 checkpoint 恢复
  │   ├── model.init_from_checkpoint(checkpoint)
  │   ├── strategy.init_densification_buffer(checkpoint)
  │   ├── 恢复 post_processing 状态（如有）
  │   └── global_step = checkpoint["global_step"]
  │
  ├── 路径 B: conf.import_ply.enabled → 从 PLY 文件导入
  │   ├── model.init_from_ply(ply_path)
  │   ├── strategy.init_densification_buffer()
  │   ├── model.build_acc()
  │   └── global_step = conf.import_ply.init_global_step
  │
  └── 路径 C: 全新初始化 → match conf.initialization.method
      ├── "random"           → model.init_from_random_point_cloud(N=100k, xyz∈[-1.5, 1.5])
      ├── "colmap"           → model.init_from_colmap(path, observer_points)
      ├── "point_cloud"      → model.init_from_pretrained_point_cloud(ply_path)
      ├── "fused_point_cloud"→ model.init_from_fused_point_cloud(ply_path, observer_pts)
      ├── "checkpoint"       → model.init_from_checkpoint(ckpt, setup_optimizer=False)
      ├── "lidar"            → model.init_from_lidar(point_cloud, observer_pts)
      └── 然后: strategy.init_densification_buffer(), model.build_acc(), model.setup_optimizer()
```

---

## 4. MixtureOfGaussians 模型

### 文件

`threedgrut/model/model.py`

### 参数表

| 参数名 | 形状 | 说明 | 激活函数 |
|--------|------|------|----------|
| `positions` | `[N, 3]` | 3D 高斯中心位置 (x, y, z) | 无 |
| `rotation` | `[N, 4]` | 单位四元数旋转 | normalize (dim=1) |
| `scale` | `[N, 3]` | 各向异性缩放 | exp |
| `density` | `[N, 1]` | 密度/不透明度（预激活） | sigmoid |
| `features_albedo` | `[N, 3]` | 0 阶 SH 系数（RGB DC 分量） | 无 |
| `features_specular` | `[N, specular_dim]` | 高阶 SH 系数（1 阶及以上） | 无 |

其中 `specular_dim = 3 * ((degree + 1)² - 1)`，degree=3 时为 `3 * (16 - 1) = 45`。

### 激活函数

| 参数 | 激活函数 | 配置键 | 反函数 |
|------|----------|--------|--------|
| density | sigmoid | `model.density_activation` | inverse_sigmoid (log(x/(1-x))) |
| scale | exp | `model.scale_activation` | log |
| rotation | normalize | 硬编码 | — |

### 渐进式 SH 训练

```python
# 配置驱动
model.progressive_training.feature_type = "sh"     # 目前仅支持 SH
model.progressive_training.init_n_features = 0      # 初始 SH 度数
model.progressive_training.max_n_features = 3       # 最大 SH 度数
model.progressive_training.increase_frequency = 1000 # 每 N 步增加
model.progressive_training.increase_step = 1         # 每次增加的度数

# 运行时
n_active_features = min(n_active_features, max_n_features)
# 如果 n_active_features < max_n_features，启用渐进式训练

# 每 increase_frequency 步:
model.increase_num_active_features()
# → n_active_features = min(max_n_features, n_active_features + increase_step)

# 渲染时传递 n_active_features，渲染器只使用前 n_active_features 个 SH 系数
# 未激活的高阶系数在渲染中被忽略（但不为零——它们在反向传播时仍接收梯度）
```

### 协方差矩阵

```python
def get_covariance(self) -> Tensor:
    S = diag(exp(scale))           # [N, 3, 3] 对角缩放矩阵
    R = quaternion_to_so3(normalize(rotation))  # [N, 3, 3] 旋转矩阵
    return R @ S @ S.T @ R.T       # [N, 3, 3] 协方差 = R S S^T R^T
```

### 背景模型

```python
# threedgrut/model/background.py
background = make(conf.model.background.name, conf.model.background)

# 支持的背景:
# - "background-color": 纯色背景（black/white/random）
# - "skip-background": 无背景（透传渲染结果）
```

### BVH 构建

```python
def build_acc(self, rebuild=True):
    self.renderer.build_acc(self, rebuild)

# 3DGRT: 构建/更新 OptiX BVH
# 3DGUT: no-op（光栅化不需要 BVH）

# 更新频率由 model.bvh_update_frequency 控制
# 当策略修改了高斯（增删改）时 scene_updated=True，触发重建
```

---

## 5. 双渲染后端

### 5.1 3DGRT — OptiX 光线追踪

**文件**: `threedgrt_tracer/tracer.py`  
**原生库**: `lib3dgrt_cc`（通过 `setup_3dgrt.py` JIT 编译）  
**技术**: NVIDIA OptiX + CUDA + Slang 着色器

#### 渲染配置 (`configs/render/3dgrt.yaml`)

```yaml
method: 3dgrt
pipeline_type: reference
backward_pipeline_type: ${render.pipeline_type}Bwd
particle_kernel_degree: 4
particle_kernel_density_clamping: true
particle_kernel_min_response: 0.0113    # 1/255 ≈ 0.0039 的 ~3x
particle_kernel_min_alpha: 0.0039       # 1/255
particle_kernel_max_alpha: 0.99
particle_radiance_sph_degree: 3
primitive_type: instances
min_transmittance: 0.001                # 提前终止阈值
max_consecutive_bvh_update: 15
enable_normals: false
enable_hitcounts: true
enable_kernel_timings: false
```

#### 核心流程

```
Tracer.__init__(conf)
  ├── load_3dgrt_plugin(conf)       ← JIT 编译 lib3dgrt_cc
  └── 创建 OptixTracer 实例，传入:
      ├── pipeline_type / backward_pipeline_type
      ├── primitive_type ("instances")
      ├── particle_kernel_degree (Gaussian kernel 多项式阶数)
      ├── particle_kernel_min_response
      ├── particle_kernel_density_clamping
      ├── particle_radiance_sph_degree
      ├── enable_normals
      └── enable_hitcounts

Tracer.build_acc(gaussians, rebuild)
  ├── 提取 positions, rotation(激活后), scale(激活后), density(激活后)
  ├── tracer_wrapper.build_bvh(...)
  └── 更新 num_update_bvh 计数器

Tracer.render(gaussians, gpu_batch, train, frame_id)
  └── _Autograd.apply(forward + backward)
      ├── forward: tracer_wrapper.trace(...) →
      │   ray_radiance  [B, H, W, 3]   ← 渲染 RGB
      │   ray_density   [B, H, W, 1]   ← 累积不透明度
      │   ray_hit_distance [B, H, W, 1] ← 首次命中距离
      │   ray_normals   [B, H, W, 3]   ← 渲染法线
      │   hits_count    [B, H, W, 1]   ← 每条光线的命中数
      │   mog_visibility [N, 1]         ← 每个高斯的可见性
      └── backward: tracer_wrapper.trace_bwd(...) →
          particle_density_grd [N, 13]  ← positions(3)+density(1)+rotation(4)+scale(3)+padding(2)
          mog_sph_grd [N, F]            ← SH 特征梯度
```

### 5.2 3DGUT — CUDA 光栅化（Unscented Transform）

**文件**: `threedgut_tracer/tracer.py`  
**原生库**: `lib3dgut_cc`（通过 `setup_3dgut.py` JIT 编译）  
**技术**: CUDA 光栅化 + Unscented Transform (UT) + 相机模型支持

#### 渲染配置 (`configs/render/3dgut.yaml`)

```yaml
method: 3dgut
particle_kernel_degree: 2
particle_kernel_min_response: 0.0113
min_transmittance: 0.0001

splat:                           # 3DGUT 特有设置
  rect_bounding: true            # 矩形包围盒
  tight_opacity_bounding: true   # 紧密不透明度包围
  tile_based_culling: true       # Tile 裁剪
  n_rolling_shutter_iterations: 5  # 卷帘快门迭代次数
  ut_alpha: 1.0                  # UT alpha 参数
  ut_beta: 2.0                   # UT beta 参数
  ut_kappa: 0.0                  # UT kappa 参数
  ut_in_image_margin_factor: 0.1  # UT 图像边距因子
  ut_require_all_sigma_points_valid: false
  k_buffer_size: 0               # K-buffer 大小 (0=不排序)
  global_z_order: true           # 全局 Z 排序
  fine_grained_load_balancing: false
```

#### 支持的相机模型

3DGUT 通过 `__create_camera_parameters` 支持多种相机模型：

| 相机模型 | 对应 NCore 类型 | 参数 |
|----------|----------------|------|
| OpenCVPinhole | `PinholeCameraModelParameters` | focal_length, principal_point, radial_coeffs(6), tangential_coeffs(2), thin_prism_coeffs(4) |
| OpenCVFisheye | `OpenCVFisheyeCameraModelParameters` | focal_length, principal_point, radial_coeffs(4), max_angle |
| FTheta | `FThetaCameraModelParameters` | principal_point, pixeldist_to_angle_poly, angle_to_pixeldist_poly, max_angle, linear_cde |

#### 卷帘快门支持

```python
# 每帧提供两个位姿：START 和 END
# T_to_world      → 快门开始时的相机到世界变换
# T_to_world_end  → 快门结束时的相机到世界变换
# 光栅化器在 START 和 END 之间插值每条扫描线的位姿

# 快门类型（从 NCore ShutterType 枚举映射）:
# - GLOBAL, ROLLING_TOP_TO_BOTTOM, ROLLING_LEFT_TO_RIGHT,
#   ROLLING_BOTTOM_TO_TOP, ROLLING_RIGHT_TO_LEFT

# n_rolling_shutter_iterations: 迭代细化投影的迭代次数
```

#### Unscented Transform 参数

```
ut_alpha: 控制 sigma 点散布（默认 1.0）
ut_beta:  高阶分布信息合并（默认 2.0，高斯最优）
ut_kappa: 二级缩放参数（默认 0.0）
```

#### 渲染流程

```
Tracer.render(gaussians, gpu_batch, train, frame_id)
  ├── 从 gpu_batch 提取 rays_ori, rays_dir
  ├── __create_camera_parameters(gpu_batch) → sensor, poses
  │   ├── 检测射线空间（全局/世界空间）
  │   ├── 构建相机内参模型
  │   └── 构建 SensorPose3D（含 START/END 位姿和时间戳）
  └── _Autograd.apply(forward + backward)
      ├── forward: tracer_wrapper.trace(...) →
      │   ray_radiance_density [..., 4]  ← RGB + Alpha
      │   ray_hit_distance     [..., 1]
      │   ray_hit_count        [..., 1]
      │   mog_visibility       [N, 1]
      └── backward: tracer_wrapper.trace_bwd(...)
```

### 5.3 公共输出字典

两个后端的 `render()` 方法返回相同结构的字典：

```python
{
    "pred_rgb":      Tensor[B, H, W, 3],   # 渲染 RGB（已与背景 alpha 合成）
    "pred_opacity":  Tensor[B, H, W, 1],   # 累积 alpha
    "pred_dist":     Tensor[B, H, W, 1],   # 首次命中距离
    "pred_normals":  Tensor[B, H, W, 3],   # 法线（3DGUT 返回占位符，已归一化）
    "hits_count":    Tensor[B, H, W, 1],   # 每条光线命中的高斯数
    "frame_time_ms": float,                # 帧渲染时间（毫秒）
    "mog_visibility": Tensor[N, 1],        # 每个高斯的可见性掩码（用于 SelectiveAdam）
}
```

---

## 6. 策略系统

### 6.1 BaseStrategy 基类

**文件**: `threedgrut/strategy/base.py`

```python
class BaseStrategy:
    _suspended: bool = False   # 蒸馏期间挂起策略

    # 钩子方法（每个都有 suspend 检查）:
    pre_backward(step, ...)      → bool  # loss.backward() 之前
    post_backward(step, ...)     → bool  # loss.backward() 之后
    post_optimizer_step(step, ...) → bool  # optimizer.step() 之后

    # 挂起/恢复:
    suspend()                    # 设置 _suspended=True，所有钩子变为 no-op

    # 工具方法:
    _update_param_with_optimizer(update_param_fn, update_optimizer_fn, names)
    # 原子地更新参数和优化器状态，保持 optimizer.state 一致性
```

### 6.2 GSStrategy — 基于梯度的增删

**文件**: `threedgrut/strategy/gs.py`  
**配置**: `configs/strategy/gs.yaml`

#### 配置参数

```yaml
method: GSStrategy
print_stats: true

densify:
  frequency: 300                   # 增密间隔（步）
  start_iteration: 500             # 开始增密的步数
  end_iteration: 15000             # 停止增密的步数
  clone_grad_threshold: 0.0002     # 克隆梯度阈值
  split_grad_threshold: 0.0002     # 分裂梯度阈值
  relative_size_threshold: 0.01    # 大高斯分裂而不是克隆的阈值
  split:
    n_gaussians: 2                 # 每次分裂成的高斯数

prune:
  frequency: 100
  start_iteration: 500
  end_iteration: 15000
  density_threshold: 0.005         # 密度低于此值的高斯被剪枝

reset_density:
  frequency: 3000
  new_max_density: 0.01            # 重置后的最大密度

density_decay:
  gamma: 0.99                      # 密度衰减系数
  start_iteration: -1              # -1 = 禁用
  end_iteration: -1
  frequency: 50

prune_weight:     # start=-1 → 禁用
prune_scale:      # start=-1 → 禁用
```

#### 操作流程

```
每个训练步:
  ├── post_backward:
  │   ├── update_gradient_buffer(sensor_position)
  │   │   └── densify_grad_norm_accum += ||grad_pos * dist_to_camera|| / 2
  │   │   └── densify_grad_norm_denom += 1
  │   └── clamp_density() (如果 activation == "none")
  │
  └── post_optimizer_step:
      ├── [条件满足] densify_gaussians()
      │   ├── densify_grad_norm = accum / denom
      │   ├── clone_gaussians(grad_norm, scene_extent)
      │   │   └── mask = grad_norm >= clone_grad_threshold AND scale <= relative_size * scene_extent
      │   │   └── 克隆满足条件的高斯，新梯度状态初始化为 0
      │   └── split_gaussians(grad_norm, scene_extent)
      │       └── mask = grad_norm >= split_grad_threshold AND scale > relative_size * scene_extent
      │       └── 采样 n_gaussians 个偏移，缩放缩小 0.8/n_gaussians
      ├── [条件满足] prune_gaussians_opacity()
      │   └── 移除 density < density_threshold (0.005) 的高斯
      ├── [条件满足] prune_gaussians_scale()
      │   └── 移除 min_scale / cam_dist * focal_max < threshold 的高斯
      ├── [条件满足] decay_density()
      │   └── 所有密度乘以 gamma (0.99)
      └── [条件满足] reset_density()
          └── 将所有密度钳制到 max(密度, new_max_density)
```

### 6.3 MCMCStrategy — 马尔可夫链蒙特卡洛

**文件**: `threedgrut/strategy/mcmc.py`  
**配置**: `configs/strategy/mcmc.yaml`  
**参考**: "3D Gaussian Splatting as Markov Chain Monte Carlo" (Kheradmand et al.)

#### 配置参数

```yaml
method: MCMCStrategy
print_stats: true
binom_n_max: 51          # 二项式查找表大小
opacity_threshold: 0.005 # 死/活高斯阈值

relocate:
  start_iteration: 500
  end_iteration: 25000
  frequency: 100          # 每 100 步重定位死高斯

perturb:
  start_iteration: 0
  end_iteration: 27500
  frequency: 1            # 每步扰动
  noise_lr: 500000.0      # 噪声学习率

add:
  start_iteration: 500
  end_iteration: 25000
  frequency: 100
  max_n_gaussians: 1000000
```

#### 操作详解

```
post_optimizer_step:
  ├── [条件满足] relocate_gaussians()
  │   ├── dead_idxs = density <= opacity_threshold
  │   ├── alive_idxs = density > opacity_threshold
  │   └── sample_new_gaussians(n_dead, alive_idxs)
  │       ├── 按不透明度加权采样（multinomial，支持 >2^24 元素的 numpy fallback）
  │       ├── compute_relocation_tensor() [CUDA kernel]
  │       │   └── 据 Eq.9 计算新的 density 和 scale（含二项式系数）
  │       ├── 反激活新 density/scale
  │       └── 复制采样位置的 positions, rotation, features
  │
  ├── [条件满足] add_new_gaussians()
  │   ├── target_num = min(max_n_gaussians, 1.05 * current)
  │   ├── num_to_add = target_num - current
  │   └── sample_new_gaussians(num_to_add) → 追加新高斯
  │
  └── [条件满足] perturb_gaussians()
      ├── noise = randn() * sigmoid_like(1-density) * noise_lr * current_lr
      ├── noise = covariance @ noise  ← 相关性扰动
      └── positions.add_(noise)
```

---

## 7. 训练循环

### 详细逐步序列（19 步）

`run_train_iter()` 中每个训练迭代的完整流程：

```
Step 0:  检查蒸馏开始 → 如果需要，冻结高斯和挂起策略
Step 1:  get_gpu_batch_with_intrinsics(batch)         ← 从 CPU batch 获取 GPU 数据
Step 2:  [条件] run_validation_pass(conf)              ← 如果 global_step % val_frequency == 0
Step 3:  model.forward(gpu_batch, train=True)          ← 前向渲染 (3DGRT/3DGUT)
Step 4:  [条件] apply_post_processing(...)             ← PPISP 后处理
Step 5:  get_losses(gpu_batch, outputs)                ← 计算损失
Step 6:  [条件] 添加 post_processing_reg_loss           ← PPISP 正则化损失
Step 7:  strategy.pre_backward(...)                    ← 策略前置钩子 (GS: no-op)
Step 8:  loss.backward()                               ← 反向传播
Step 9:  strategy.post_backward(...)                   ← 梯度累积 (GS: update_gradient_buffer)
Step 10: optimizer.step(visibility) 或 optimizer.step() ← 参数更新
Step 11: optimizer.zero_grad()                          ← 清零梯度
Step 12: model.scheduler_step(global_step)              ← 学习率调度
Step 13: [条件] post_processing_optimizers.step()       ← PPISP 优化器步进
Step 14: strategy.post_optimizer_step(...)              ← 增删剪枝 (clone/split/prune/relocate/perturb)
Step 15: [条件] model.increase_num_active_features()    ← 渐进式 SH
Step 16: [条件] model.build_acc(rebuild=True)           ← BVH 重建
Step 17: global_step += 1                              ← 递增全局步数
Step 18: get_metrics() + log_training_iter()           ← 记录指标
Step 19: [条件] save_checkpoint() + render_gui()        ← 保存和 GUI
```

### 验证流程

```
run_validation_pass()
  ├── @torch.no_grad() 装饰器
  ├── 遍历 val_dataloader:
  │   ├── get_gpu_batch_with_intrinsics(batch_idx)
  │   ├── model(gpu_batch, train=False)
  │   ├── [条件] apply_post_processing(..., training=False)
  │   ├── get_losses()
  │   └── get_metrics(... split="validation")
  └── log_validation_pass(metrics) → TensorBoard/WandB
```

---

## 8. 损失函数

### 文件

`threedgrut/trainer.py` → `get_losses()` 方法  
`threedgrut/model/losses.py` → `ssim()` 使用 `fused_ssim`

### 公式和默认权重

```python
# L1 损失
loss_l1 = |pred_rgb - rgb_gt|.mean()
lambda_l1 = 0.8

# L2 损失（默认禁用）
loss_l2 = MSE(pred_rgb, rgb_gt)
lambda_l2 = 1.0  # use_l2: false

# SSIM 损失
loss_ssim = 1.0 - fused_ssim(pred, gt)
lambda_ssim = 0.2

# 不透明度正则化（仅 MCMC 启用）
loss_opacity = |get_density()|.mean()
lambda_opacity = 0.01  # use_opacity: false (GS), true (MCMC)

# 缩放正则化（仅 MCMC 启用）
loss_scale = |get_scale()|.mean()
lambda_scale = 0.01  # use_scale: false (GS), true (MCMC)

# 总损失
total_loss = λ_l1*l1 + λ_ssim*ssim + λ_opacity*opacity + λ_scale*scale
#          + post_processing_reg_loss  (如果启用 PPISP)
```

### 评估指标

```python
criterions = {
    "psnr":  PeakSignalNoiseRatio(data_range=1),
    "ssim":  StructuralSimilarityIndexMeasure(data_range=1.0),
    "lpips": LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True),
}
```

---

## 9. 优化器与调度器

### 9.1 优化器类型

**配置**: `configs/base_gs.yaml` → `optimizer.type`

| 类型 | 类 | 文件 | 说明 |
|------|-----|------|------|
| `adam` | `torch.optim.Adam` | PyTorch 内置 | 标准 Adam |
| `selective_adam` | `SelectiveAdam` | `threedgrut/optimizers/__init__.py` | 基于可见性的选择性更新 |

### 9.2 SelectiveAdam

```python
class SelectiveAdam(torch.optim.Adam):
    def step(self, visibility: Tensor):
        # visibility: [N, 1] bool 张量
        # 仅对 visibility=True 的参数应用 Adam 更新
        # 通过 CUDA kernel (lib_optimizers_cc) 实现融合更新

# 在训练循环中:
if isinstance(self.model.optimizer, SelectiveAdam):
    self.model.optimizer.step(outputs["mog_visibility"])
else:
    self.model.optimizer.step()
```

**参考**: "Taming 3DGS" (Mallick et al.)，通过 gSplat 库适配。

### 9.3 参数组学习率

```yaml
optimizer.params:
  positions:        { lr: 0.00016 }   # 乘以 scene_extent
  density:          { lr: 0.05 }
  features_albedo:  { lr: 0.0025 }
  features_specular: { lr: 0.000125 } # = features_albedo.lr / 20
  rotation:         { lr: 0.001 }
  scale:            { lr: 0.005 }
```

`setup_optimizer()` 中位置学习率乘以 `scene_extent`：
```python
for param_group in self.optimizer.param_groups:
    if param_group["name"] == "positions":
        param_group["lr"] *= self.scene_extent
```

### 9.4 调度器

```yaml
scheduler:
  positions:
    type: exp                                # 指数衰减
    lr_init: 0.00016
    lr_final: 0.0000016                      # 100x 衰减
    max_steps: 30000
  density:
    type: skip                               # 不调度
```

**调度函数** (`threedgrut/utils/misc.py`):

```python
def exponential_scheduler(lr_init, lr_final, max_steps):
    def helper(step):
        t = clip(step / max_steps, 0, 1)
        return exp(log(lr_init) * (1-t) + log(lr_final) * t)
    return helper
```

**调度步进**（在 `scheduler_step()` 中）:
```python
for param_group in self.optimizer.param_groups:
    if param_group["name"] in self.schedulers:
        lr = self.schedulers[param_group["name"]](step)
        if lr is not None:
            param_group["lr"] = lr
```

---

## 10. 数据集系统

### 文件

| 文件 | 数据集 |
|------|--------|
| `threedgrut/datasets/dataset_colmap.py` | COLMAP / MipNeRF360 |
| `threedgrut/datasets/dataset_nerf.py` | NeRF Synthetic |
| `threedgrut/datasets/dataset_scannetpp.py` | ScanNet++ |
| `threedgrut/datasets/datasetNcore.py` | NCore v4 |
| `threedgrut/datasets/protocols.py` | `BoundedMultiViewDataset` 协议, `Batch` 数据类 |
| `threedgrut/datasets/__init__.py` | `make()` / `make_test()` 工厂函数 |
| `threedgrut/datasets/utils.py` | `MultiEpochsDataLoader`, `PointCloud`, `read_colmap_points3D_text` |

### Batch 数据结构

```python
@dataclass
class Batch:
    rays_ori: Tensor                     # [B, H, W, 3] 射线原点
    rays_dir: Tensor                     # [B, H, W, 3] 射线方向
    T_to_world: Tensor                   # [B, 4, 4] 射线空间→世界空间变换 (START)
    T_to_world_end: Optional[Tensor]     # [B, 4, 4] END 位姿（卷帘快门）
    rays_in_world_space: bool            # 射线是否已在世界空间
    rgb_gt: Optional[Tensor]             # [B, H, W, 3]
    mask: Optional[Tensor]               # [B, H, W, 1]
    intrinsics: Optional[list]           # [fx, fy, cx, cy]
    intrinsics_OpenCVPinholeCameraModelParameters: Optional[dict]
    intrinsics_OpenCVFisheyeCameraModelParameters: Optional[dict]
    intrinsics_FThetaCameraModelParameters: Optional[dict]
    camera_idx: int                      # 相机索引（后处理用）
    frame_idx: int                       # 帧索引（后处理用）
    pixel_coords: Optional[Tensor]       # [B, H, W, 2] (x+0.5, y+0.5)
    exposure: Optional[Tensor]           # 均值归一化的 log2 曝光
```

### NCore v4 数据集

**关键特性**:
- 多相机传感器支持（`camera_ids` 参数）
- LiDAR 点云初始化（`lidar_ids` 参数）
- 卷帘快门位姿插值（每帧 START/END 双位姿）
- GPU 射线缓存（`device="cuda"`）
- JPEG 解码后端可选：`simplejpeg`（libjpeg-turbo）或 `PIL`
- 帧级 train/val 分割（`val_frame_interval=8`，即每 8 帧取 1 帧验证）
- 全图训练（`sample_full_image=true`）
- 时间窗口控制（`seek_offset_sec`, `duration_sec`）

**初始化的 LiDAR 点云**:
```python
pc = PointCloud.from_sequence(
    train_dataset.get_point_clouds(step_frame=1, non_dynamic_points_only=True),
    device="cpu"
)
# 可随机下采样到 num_points
model.init_from_lidar(pc, observer_points)
```

### BoundedMultiViewDataset 协议

```python
class BoundedMultiViewDataset(Protocol):
    def get_scene_bbox() -> tuple[Tensor, Tensor]     # (min, max)
    def get_scene_extent() -> float
    def get_observer_points() -> np.ndarray             # [M, 3]
    def get_poses() -> np.ndarray                       # [N, 4, 4] C2W
    def get_gpu_batch_with_intrinsics(batch) -> Batch
    def get_camera_idx(frame_idx) -> int
    def get_frames_per_camera() -> list[int]
    def __getitem__(index) -> dict
    def __len__() -> int
```

---

## 11. 导出系统

### 文件

```
threedgrut/export/
├── __init__.py           ← 导出所有公共接口
├── base.py               ← ExportableModel, ModelExporter 基类
├── accessor.py           ← GaussianExportAccessor, GaussianAttributes
├── adapter.py            ← AttributesExportAdapter（转码用）
├── transforms.py         ← estimate_normalizing_transform
├── formats/
│   └── ply.py            ← PLYExporter (标准 Gaussian Splatting PLY)
├── importers/
│   ├── base.py           ← FormatImporter 基类
│   ├── ply.py            ← PLYImporter
│   └── usd.py            ← USDImporter
├── usd/
│   ├── exporter.py       ← USDExporter (ParticleField3DGaussianSplat schema)
│   ├── writers/          ← USD stage 写入器
│   ├── nurec/
│   │   └── exporter.py   ← NuRecExporter (Omniverse 兼容格式)
│   └── stage_utils.py
├── scripts/
│   ├── transcode.py      ← 格式转换脚本
│   └── filter_visibility.py ← 可见性过滤
└── tests/
```

### 导出格式

| 格式 | 导出器 | Schema/标准 |
|------|--------|-------------|
| **PLY** | `PLYExporter` | 标准 Gaussian Splatting PLY（含 x,y,z, opacity, scale_*, rot_*, f_dc_*, f_rest_*） |
| **USD (Standard)** | `USDExporter` | OpenUSD `ParticleField3DGaussianSplat` 标准 schema |
| **USD (NuRec)** | `NuRecExporter` | NVIDIA Omniverse USDVol 内部格式（传统） |

### 导出配置

```yaml
export_ply:
  enabled: false
  path: ""                # 空 = 使用默认路径 {out_dir}/export_last.ply

export_usd:
  enabled: false
  path: ""
  apply_normalizing_transform: true
  format: standard        # "standard" 或 "nurec"
  half_precision: false   # fp16 导出
  export_cameras: true
  export_background: true
  sorting_mode_hint: cameraDistance
```

### PLY 导出属性

```
vertex: x, y, z           ← positions [N,3]
vertex: opacity           ← sigmoid(density) [N,1]
vertex: scale_0..scale_2  ← exp(scale) [N,3]
vertex: rot_0..rot_3      ← normalize(rotation) [N,4]
vertex: f_dc_0..f_dc_2    ← features_albedo [N,3]
vertex: f_rest_0..f_rest_N ← features_specular [N, F]（展平为 C 语言序）
```

### 转码

```bash
python -m threedgrut.export.scripts.transcode input.ply -o output.usdz
```

---

## 12. 后处理 PPISP

### 概述

PPISP (Per-Pixel Illumination and Sensor Processing) 是一个可选的后处理模块，用于学习：
- **CRF**（相机响应函数）：每个相机的逐像素颜色校正
- **曝光**：每个帧的对数曝光补偿
- **Controller**：一个神经网络，预测新视角的逐帧校正

### 配置

```yaml
post_processing:
  method: null             # null 或 "ppisp"
  use_controller: true     # 启用 controller 网络
  n_distillation_steps: 5000  # 蒸馏步数
```

### 蒸馏模式

当 `n_distillation_steps > 0` 时启用：

```
训练阶段 1 (main_training_steps = n_iterations - n_distillation_steps):
  ├── 正常训练：高斯参数 + PPISP 参数联合优化
  └── Controller 不活跃（controller_activation_ratio 计算使 controller 在此阶段结束前才开始激活）

训练阶段 2 (最后 n_distillation_steps):
  ├── model.freeze_gaussians()           ← 冻结所有高斯参数
  ├── strategy.suspend()                 ← 挂起增删剪枝操作
  └── 仅训练 Controller（PPISP 参数也冻结）
```

### 推理流程

```python
def apply_post_processing(post_processing, outputs, gpu_batch, training):
    pred_rgb_flat = outputs["pred_rgb"].view(-1, 3)       # [H*W, 3]
    pixel_coords_flat = gpu_batch.pixel_coords.view(-1, 2) # [H*W, 2]
    
    pred_rgb_pp = post_processing(
        pred_rgb_flat, pixel_coords_flat,
        resolution=(W, H),
        camera_idx=camera_idx,
        frame_idx=frame_idx,           # 训练时用实际帧索引，推理时用 -1
        exposure_prior=exposure
    )
    
    outputs["pred_rgb"] = pred_rgb_pp.view(H, W, 3)
    return outputs
```

---

## 13. ASCII 架构图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              train.py                                        │
│                    @hydra.main(config_path="configs")                         │
│                    Trainer3DGRUT(conf).run_training()                         │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     Trainer3DGRUT       │
                    │   (threedgrut/trainer)  │
                    └──────┬──────┬──────┬────┘
                           │      │      │
          ┌────────────────▼┐     │      └──────────────────┐
          │   Config (Hydra) │     │                         │
          │  base_gs/mcmc    │     │                         │
          │  + dataset       │     │                         │
          │  + initialization│     │                         │
          │  + render        │     │                         │
          │  + strategy      │     │                         │
          │  + apps          │     │                         │
          └──────────────────┘     │                         │
                                   │                         │
┌──────────────────────────────────▼─────────────────────┐   │
│              MixtureOfGaussians (model)                 │   │
│  ┌──────────┐ ┌──────────┐ ┌────────┐ ┌─────────────┐  │   │
│  │positions │ │ rotation │ │ scale  │ │   density   │  │   │
│  │  [N,3]   │ │  [N,4]   │ │ [N,3]  │ │    [N,1]    │  │   │
│  └──────────┘ └──────────┘ └────────┘ └─────────────┘  │   │
│  ┌──────────────────────┐ ┌────────────────────────┐   │   │
│  │  features_albedo     │ │  features_specular     │   │   │
│  │     [N,3] (DC)       │ │  [N,45] (SH deg 1-3)  │   │   │
│  └──────────────────────┘ └────────────────────────┘   │   │
│  ┌──────────────────────────────────────────────────┐  │   │
│  │            background (color / skip)              │  │   │
│  └──────────────────────────────────────────────────┘  │   │
└──────────────────────┬─────────────────────────────────┘   │
                       │ forward()                            │
                       ▼                                      │
      ┌────────────────────────────────────┐                  │
      │         Renderer Backend           │                  │
      ├────────────────┬───────────────────┤                  │
      │     3DGRT      │      3DGUT        │                  │
      │  (Ray Tracing) │  (Rasterization)  │                  │
      │  OptiX + CUDA  │  CUDA + UT        │                  │
      │  ┌──────────┐  │  ┌─────────────┐  │                  │
      │  │  BVH     │  │  │Tile Culling │  │                  │
      │  │ 构建/更新│  │  │Camera Models│  │                  │
      │  └──────────┘  │  │Roll.Shutter │  │                  │
      │                │  └─────────────┘  │                  │
      └───────┬────────┴────────┬──────────┘                  │
              │                 │                              │
              └────────┬────────┘                              │
                       ▼                                       │
      ┌────────────────────────────────────┐                  │
      │         Output Dict                │                  │
      │  pred_rgb, pred_opacity,           │                  │
      │  pred_dist, pred_normals,          │                  │
      │  hits_count, mog_visibility        │                  │
      └────────────────┬───────────────────┘                  │
                       │                                       │
                       ▼                                       │
      ┌────────────────────────────────────┐                  │
      │         Loss Computation           │                  │
      │  L1(λ=0.8) + SSIM(λ=0.2)          │                  │
      │  + opacity_reg + scale_reg         │                  │
      └────────────────┬───────────────────┘                  │
                       │                                       │
                       ▼                                       │
      ┌────────────────────────────────────┐                  │
      │    Strategy Hooks                  │                  │
      │  ┌──────────────────────────────┐  │                  │
      │  │  GSStrategy                  │  │                  │
      │  │  clone / split / prune_α     │  │                  │
      │  │  prune_scale / decay / reset │  │                  │
      │  ├──────────────────────────────┤  │                  │
      │  │  MCMCStrategy               │  │                  │
      │  │  relocate / add / perturb    │  │                  │
      │  └──────────────────────────────┘  │                  │
      └────────────────┬───────────────────┘                  │
                       │                                       │
                       ▼                                       │
      ┌────────────────────────────────────┐                  │
      │  Optimizer (Adam/SelectiveAdam)    │◄─────────────────┘
      │  Scheduler (exp for positions)     │
      └────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            Export Pipeline                                   │
│  ┌──────┐    ┌─────────────────┐    ┌──────────────────────────────────┐   │
│  │ PLY  │    │  USD Standard   │    │  USD NuRec (Omniverse legacy)    │   │
│  │Exporter│   │  ParticleField  │    │  USDVol internal format          │   │
│  │      │    │  3DGaussianSplat│    │                                  │   │
│  └──────┘    └─────────────────┘    └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          Post-Processing (PPISP)                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐                          │
│  │   CRF    │    │ Exposure │    │  Controller  │                          │
│  │ per-cam  │    │ per-frame│    │ (distillation)│                          │
│  └──────────┘    └──────────┘    └──────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 14. 关键文件路径参考表

| 路径 | 说明 |
|------|------|
| `train.py` | 训练入口点 |
| `render.py` | 推理/评估脚本 |
| `playground.py` | 交互式可视化启动器 |
| `threedgrut/trainer.py` | `Trainer3DGRUT` 类（训练循环核心） |
| `threedgrut/model/model.py` | `MixtureOfGaussians` 模型定义 |
| `threedgrut/model/losses.py` | SSIM/L1/L2 损失函数 |
| `threedgrut/model/background.py` | 背景模型 |
| `threedgrut/model/geometry.py` | KNN、最近邻距离工具 |
| `threedgrut/strategy/base.py` | `BaseStrategy` 抽象基类 |
| `threedgrut/strategy/gs.py` | `GSStrategy`（基于梯度的增删） |
| `threedgrut/strategy/mcmc.py` | `MCMCStrategy`（MCMC 采样策略） |
| `threedgrut/strategy/src/setup_mcmc.py` | MCMC CUDA kernel JIT 编译 |
| `threedgrut/optimizers/__init__.py` | `SelectiveAdam` 优化器 |
| `threedgrut/optimizers/optimizers.cu` | SelectiveAdam CUDA kernel |
| `threedgrut/render.py` | `Renderer` 类（测试集渲染+评估） |
| `threedgrut/datasets/__init__.py` | `make()` / `make_test()` 工厂 |
| `threedgrut/datasets/protocols.py` | `Batch` 数据结构, `BoundedMultiViewDataset` 协议 |
| `threedgrut/datasets/dataset_colmap.py` | COLMAP 数据集 |
| `threedgrut/datasets/dataset_nerf.py` | NeRF Synthetic 数据集 |
| `threedgrut/datasets/dataset_scannetpp.py` | ScanNet++ 数据集 |
| `threedgrut/datasets/datasetNcore.py` | NCore v4 数据集 |
| `threedgrut/datasets/utils.py` | `MultiEpochsDataLoader`, `PointCloud` |
| `threedgrut/export/__init__.py` | 导出模块公共接口 |
| `threedgrut/export/base.py` | `ExportableModel`, `ModelExporter` |
| `threedgrut/export/formats/ply.py` | `PLYExporter` |
| `threedgrut/export/usd/exporter.py` | `USDExporter` (标准 schema) |
| `threedgrut/export/usd/nurec/exporter.py` | `NuRecExporter` (传统格式) |
| `threedgrut/export/importers/ply.py` | `PLYImporter` |
| `threedgrut/export/importers/usd.py` | `USDImporter` |
| `threedgrut/export/accessor.py` | `GaussianExportAccessor` |
| `threedgrut/export/adapter.py` | `AttributesExportAdapter`（转码适配器） |
| `threedgrut/utils/misc.py` | 激活函数、调度器、SH 维度、jet_map、multinomial_sample |
| `threedgrut/utils/render.py` | `RGB2SH`, `SH2RGB`, `apply_post_processing` |
| `threedgrut/utils/logger.py` | Rich 日志记录器 |
| `threedgrut/utils/timer.py` | `CudaTimer`（CUDA 事件计时） |
| `threedgrt_tracer/tracer.py` | 3DGRT OptiX 渲染器 Python 绑定 |
| `threedgrt_tracer/setup_3dgrt.py` | 3DGRT JIT 编译器 |
| `threedgrt_tracer/bindings.cpp` | 3DGRT PyTorch C++ 绑定 |
| `threedgrt_tracer/src/` | 3DGRT CUDA/Slang 源代码 |
| `threedgut_tracer/tracer.py` | 3DGUT CUDA 光栅化器 Python 绑定 |
| `threedgut_tracer/setup_3dgut.py` | 3DGUT JIT 编译器 |
| `threedgut_tracer/bindings.cpp` | 3DGUT PyTorch C++ 绑定 |
| `threedgut_tracer/src/` | 3DGUT CUDA 源代码 |
| `configs/base_gs.yaml` | 基础 GS 配置 |
| `configs/base_mcmc.yaml` | 基础 MCMC 配置 |
| `configs/apps/*.yaml` | 完整流水线预设（14 个） |
| `configs/render/3dgrt.yaml` | 3DGRT 渲染配置 |
| `configs/render/3dgut.yaml` | 3DGUT 渲染配置 |
| `configs/strategy/gs.yaml` | GS 策略配置 |
| `configs/strategy/mcmc.yaml` | MCMC 策略配置 |
| `configs/dataset/*.yaml` | 数据集配置（4 个） |
| `configs/initialization/*.yaml` | 初始化方法配置（6 个） |
| `thirdparty/optix-dev/` | OptiX SDK（3DGRT 依赖） |
| `thirdparty/tiny-cuda-nn/` | tiny-cuda-nn（3DGUT 依赖） |

---

## 15. 技术栈附录

### Python 依赖

| 包 | 用途 |
|----|------|
| `torch` (≥2.0) | 深度学习框架 |
| `hydra-core` + `omegaconf` | 配置管理 |
| `addict` | `Dict`（属性访问字典） |
| `torchmetrics` | PSNR, SSIM, LPIPS 评估 |
| `fused-ssim` | 可微分 SSIM（CUDA 融合 kernel） |
| `plyfile` | PLY 文件读写 |
| `numpy` | 数值计算 |
| `rich` | 终端日志美化 |
| `tensorboard` (torch.utils.tensorboard) | 训练日志 |
| `wandb` (可选) | Weights & Biases 集成 |
| `ncore` | NCore v4 数据集 SDK |
| `ppisp` (可选) | 后处理模块 |
| `polyscope` (可选) | 桌面 GUI |
| `viser` (可选) | 浏览器 GUI |
| `msgpack` | checkpoint 序列化 |

### CUDA/C++ 组件

| 组件 | 语言 | 编译方式 | 用途 |
|------|------|----------|------|
| `lib3dgrt_cc` | CUDA / Slang / C++ | JIT (`setup_3dgrt.py`) | OptiX 光线追踪 |
| `lib3dgut_cc` | CUDA / C++ | JIT (`setup_3dgut.py`) | CUDA 光栅化 + UT |
| `lib_mcmc_cc` | CUDA / C++ | JIT (`setup_mcmc.py`) | MCMC relocation kernel |
| `lib_optimizers_cc` | CUDA / C++ | JIT (`setup_optimizers.py`) | SelectiveAdam fused kernel |
| `tiny-cuda-nn` | CUDA / C++ | CMake | 神经网络原语（3DGUT 依赖） |
| OptiX SDK | C++ | 预编译 | 光线追踪引擎（3DGRT 依赖） |

### CUDA 版本支持

- CUDA 11.8
- CUDA 12.4
- CUDA 12.6
- CUDA 12.8（默认）
- CUDA 13.0（实验性）

### GPU 要求

- 最低计算能力: 7.0（V100, A100）
- 推荐 RTX 系列（具备 RT Cores，用于 3DGRT 光线追踪加速）
- 3DGRT 需要 OptiX SDK（捆绑在 `thirdparty/optix-dev/`）

### 环境安装

```bash
# UV 包管理器（推荐）
./install_env_uv.sh            # Linux
.\install_env_uv.ps1           # Windows

# Conda（传统）
./install_env.sh 3dgrut

# Docker
docker build --build-arg CUDA_VERSION=12.8.1 -t 3dgrut:cuda128 .
```

---

> 文档结束。本文档从 3DGRUT 代码库 (`/Users/etendue/repo/3dgrut2`) 中提取，覆盖训练架构、配置系统、模型、渲染后端、策略、优化器、数据集、导出及后处理的全链路。
