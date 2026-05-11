# Reconstruction Studio 系统架构文档

## 1. 项目概述

Reconstruction Studio 是一个基于 **3D Gaussian Splatting** 的自动驾驶场景三维重建系统。它支持多数据集（Waymo/KITTI/NuScenes/ArgoVerse/PandaSet/NuPlan/Custom），能够对包含动态物体（车辆、行人、骑行者）的驾驶场景进行高质量新视角合成和三维重建。

## 2. 技术栈

| 类别 | 技术 |
|---|---|
| 语言 | Python |
| 深度学习框架 | PyTorch 2.0.0 + CUDA 11.7 |
| 3D 高斯渲染 | gsplat (3DGS), diff-gaussian-rasterization (2DGS) |
| 3D 几何 | pytorch3d, open3d, trimesh, plyfile |
| 人体模型 | SMPL/SMPL-X |
| 配置管理 | OmegaConf (YAML + CLI override) |
| 实验追踪 | wandb, SwanLab, TensorBoard |
| 3D 可视化 | viser, nerfview |
| 评估指标 | torchmetrics (PSNR/SSIM/LPIPS) |

## 3. 目录结构

```
reconstruction-studio/
├── datasets/                  # 数据加载与预处理
│   ├── dataset_meta.py        # 7种数据集的相机元数据定义
│   ├── driving_dataset.py     # 核心数据集类 DrivingDataset
│   ├── preprocess.py          # 统一数据预处理脚本
│   ├── base/                  # 数据抽象基类
│   │   ├── scene_dataset.py   # SceneDataset (顶层抽象)
│   │   ├── pixel_source.py    # ScenePixelSource + CameraData (图像/相机数据)
│   │   ├── lidar_source.py    # SceneLidarSource (LiDAR 点云数据)
│   │   ├── split_wrapper.py   # SplitWrapper (训练/测试集划分)
│   │   └── depth2normal/      # 深度转法线估计
│   ├── customer/              # 自定义数据集支持
│   └── tools/                 # 数据处理工具 (语义分割/SMPL/人体姿态)
├── models/                    # 模型实现
│   ├── gaussians/             # 高斯表示变体
│   ├── nodes/                 # 动态物体节点
│   ├── trainers/              # 训练编排器
│   ├── modules.py             # 神经网络基础模块
│   ├── human_body.py          # SMPL 人体模型
│   ├── losses.py              # 损失函数
│   ├── mesh_utils.py          # Mesh 提取
│   ├── video_utils.py         # 渲染与视频导出
│   └── luxury/                # 可选模块 (曝光校正)
├── tools/                     # CLI 入口
│   ├── train.py               # 训练入口
│   ├── eval.py                # 评估入口
│   ├── recon.py               # 重建/导出入口
│   └── train_road_surface.py  # 路面重建训练入口
└── utils/                     # 工具函数
    ├── camera.py              # 相机插值/轨迹生成
    ├── visualization.py       # 可视化 (多相机合成/深度/误差图)
    └── ...                    # 其他工具
```

## 4. 核心架构

### 4.1 整体数据流

```
原始数据 → 预处理(preprocess) → DrivingDataset → Trainer → 渲染/导出
                                     ↓
                              ScenePixelSource (图像+相机)
                              SceneLidarSource  (点云)
                              SplitWrapper      (数据划分)
```

训练流程：

```
YAML 配置 → DrivingDataset 构建 → Trainer 实例化
→ 初始化高斯 (LiDAR 点云) → 训练循环:
    重要性采样图像 → 前向传播 → 损失计算 → 反向传播 → 自适应密度控制
→ 定期可视化/检查点/评估 → 最终评估
```

### 4.2 训练器层次（Trainer Hierarchy）

```
nn.Module
  └── BasicTrainer                    # 抽象基类
        ├── SingleTrainer             # 单一背景高斯 (静态/动态场景)
        ├── MultiTrainer              # 场景图 (背景+刚体+可变形+SMPL)
        └── RoadSurfaceTrainer        # 路面专用重建
```

**BasicTrainer** (`models/trainers/base.py`)：

- 定义 `GSModelType` 枚举: Background=0, RigidNodes=1, SMPLNodes=2, DeformableNodes=3, Ground=4
- 核心方法: `forward()` → `process_camera()` → `collect_gaussians()` → `render_gaussians()` → `add_sky()` → `affine_transform()`
- 损失函数 (`compute_losses()`): RGB L1+SSIM, 天空透明度 BCE, 深度监督, 法线监督, 动态区域加权, 正则化 (不透明度熵/逆深度平滑/仿射)
- 支持 pinhole 和 Scaramuzza (鱼眼) 两种相机模型
- 集成 wandb/SwanLab/TensorBoard 实验追踪和 viser/nerfview 在线查看器

**SingleTrainer** (`models/trainers/single.py`)：

- 仅使用 Background 高斯类，支持 Vanilla/Deformable/PVG 三种高斯变体
- 从 LiDAR 点云 + 随机球采样初始化

**MultiTrainer** (`models/trainers/scene_graph.py`)：

- 场景图重建，支持最多 5 种高斯类 (Background + Rigid + Deformable + SMPL + Ground)
- 每类动态物体独立初始化：RigidNodes/DeformableNodes 从 bbox 内 LiDAR 点云初始化，SMPLNodes 从 SMPL 模型初始化
- 评估时支持逐类渲染可视化

**RoadSurfaceTrainer** (`models/trainers/road_surface.py`)：

- 功能与 SingleTrainer 类似，专用路面重建
- 使用 2D 扁平高斯沿路面法线方向初始化

### 4.3 高斯模型层次（Gaussian Model Hierarchy）

```
nn.Module
  ├── VanillaGaussians               # 标准 3DGS
  │     ├── DeformableGaussians       # + 时序变形网络
  │     └── PeriodicVibrationGaussians # + 周期振动 (PVG)
  ├── ScaffoldGaussians               # 锚点式高斯 (Scaffold-GS)
  └── SurfaceGaussians                # 2D 高斯路面重建
```

**VanillaGaussians** (`models/gaussians/vanilla.py`)：

- 核心参数: `_means`, `_scales`, `_quats`, `_features_dc/rest` (SH颜色), `_opacities`, `_labels`
- 支持球高斯 (1D scale)、2D 高斯 (2D scale)、标准 3D 高斯 (3D scale)
- 自适应密度控制: 分裂大高斯、复制小高梯度高斯、剔除透明/过大高斯
- 地面点特殊处理: 不参与剔除，z 方向 scale 和旋转梯度置零
- 渐进式 SH 度数提升

**DeformableGaussians** (`models/gaussians/deformgs.py`)：

- 扩展 VanillaGaussians，添加 `DeformNetwork` (8 层 MLP + skip connection)
- 输入: 正弦编码的位置 + 时间 → 输出: delta_xyz, delta_quat, delta_scale
- 使用 Mip-NeRF 360 风格的坐标收缩将无界空间映射到 [0,1]
- 粗训练阶段后启用变形

**PVG** (`models/gaussians/pvg.py`)：

- 每个高斯增加时序参数: `_taus` (生命峰值), `_betas` (生命跨度), `_velocity` (振动方向)
- 时序位置: `means + velocity * sin(2π/T * (t - τ))`
- 时序透明度: `opacity * exp(-0.5 * (t - τ)² / β²)` — 高斯时间包络
- 密度化同时考虑空间和时间梯度

**ScaffoldGaussians** (`models/gaussians/scaffold.py`)：

- 锚点式范式：存储 `_anchor` 位置 + `_offset` 偏移 + `_anchor_feat` 神经特征
- MLP 解码: anchor 特征 → 预测 offset/gaussian 参数
- 支持多级细节 (coarse/fine anchor)

**SurfaceGaussians** (`models/gaussians/surface.py`)：

- 使用 2D 高斯 (扁椭圆) 进行路面重建
- 包含 `Road` 类，沿路面法线方向初始化
- 支持基于 2DGS 的可微渲染

### 4.4 动态物体节点（Node System）

```
nn.Module
  ├── RigidNodes                      # 刚体物体 (车辆)
  │     └── DeformableNodes           # 可变形物体 (骑行者)
  └── SMPLNodes                       # SMPL 人体
```

**RigidNodes** (`models/nodes/rigid.py`)：

- 每个物体实例拥有独立的 VanillaGaussians 参数
- 6DoF 姿态追踪: 物体坐标系 → 世界坐标系
- 支持时间依赖的物体可见性 (出现/消失)
- 正则化: 体积保持、不透明度稀疏性

**DeformableNodes** (`models/nodes/deformable.py`)：

- 扩展 RigidNodes，添加 `ConditionalDeformNetwork`
- 输入: 物体局部坐标 + 时间 → 局部变形
- 允许物体在刚体运动基础上发生非刚性变形

**SMPLNodes** (`models/nodes/smpl.py`)：

- SMPL 人体模型驱动的动态人体重建
- 优化 SMPL 形状参数 (betas) 和姿态参数 (body_pose)
- 高斯点分布在 SMPL 网格表面
- 正则化: SMPL 参数约束、体积保持、Laplacian 平滑

### 4.5 数据层架构

```
SceneDataset (ABC)
  ├── pixel_source: ScenePixelSource
  │     └── camera_data: Dict[int, CameraData]  # 逐相机数据
  │           ├── cam_to_worlds, intrinsics, distortions
  │           ├── images, sky/dynamic/human/vehicle/road_surface masks
  │           ├── lidar_depth_maps, lidar_normal_maps
  │           └── image_error_maps  # 重要性采样用
  ├── lidar_source: SceneLidarSource
  │     ├── origins, directions, ranges
  │     ├── colors, labels, timesteps
  │     └── visible_masks
  └── SplitWrapper (train/test/full)
        └── propose_training_image()  # 基于误差的重要性采样
```

**CameraData** 核心字段：

- 相机参数: `cam_to_worlds`, `intrinsics`, `distortions`, `cam_model_type` (pinhole/scaramuzza)
- 图像数据: `images` (归一化 [0,1])
- 语义遮罩: `egocar_mask`, `dynamic_masks`, `human_masks`, `vehicle_masks`, `sky_masks`, `road_surface_masks`
- 监督信号: `lidar_depth_maps`, `lidar_normal_maps`, `image_error_maps`

**DrivingDataset** 初始化流程：

1. 动态导入像素/激光雷达数据源类
2. 将 LiDAR 点云投影到图像上 (深度监督 + 可见性过滤)
3. 计算 AABB 场景包围盒
4. 划分训练/测试集
5. 构建 SplitWrapper

**支持的数据集**:

| 数据集 | 相机数 | 图像尺寸 | 自车可见相机 |
|---|---|---|---|
| Waymo | 5 | 1280x1920 / 866x1920 | 无 |
| KITTI | 2 | 375x1242 | 无 |
| NuScenes | 6 | 900x1600 | CAM_BACK |
| ArgoVerse | 7 | 2048x1550 / 1550x2048 | rear_left, rear_right |
| PandaSet | 6 | 1080x1920 | back_camera |
| NuPlan | 8 | 1080x1920 | L0, R0, L2, R2 |
| Customer | 3 | 1080x1920 | CAMERA_2_FRONT_WIDE |

### 4.6 神经网络模块（`models/modules.py`）

| 模块 | 用途 |
|---|---|
| `SinusoidalEncoder` | 位置/时间正弦编码 |
| `MLP` | 带残差连接的多层感知机 |
| `DeformNetwork` | 8层MLP + skip，输出 delta_xyz/quat/scale |
| `ConditionalDeformNetwork` | 条件变形网络 (用于 DeformableNodes) |
| `ExposureModel` | 逐相机曝光校正 (仿射变换) |

### 4.7 渲染管线

```
process_camera() → collect_gaussians() → render_gaussians() → add_sky() → affine_transform()
```

- `process_camera()`: 应用相机姿态优化 (`CamPose`, `CamPosePerturb`)，封装为 `dataclass_camera`
- `collect_gaussians()`: 遍历所有高斯类，调用 `get_gaussians(cam)` 获取参数，拼接为 `dataclass_gs`
- `render_gaussians()`: 使用 gsplat 光栅化，支持 pinhole/Scaramuzza 相机模型，渲染 RGB + Depth (+ 可选法线)
- `add_sky()`: 学习式天空颜色 + 天空遮罩合成
- `affine_transform()`: 逐相机颜色仿射校正

最终 RGB = `affine(gaussian_rgb + sky_rgb * (1 - opacity))`

### 4.8 损失函数体系

| 损失 | 描述 |
|---|---|
| RGB Loss | L1 + SSIM (动态/自车区域加权) |
| Sky Opacity Loss | BCE (天空区域透明度) |
| Depth Loss | L1/L2/Smooth-L1 (LiDAR 深度监督，带衰减调度) |
| Normal Loss | 可选法线监督 |
| Dynamic Region Loss | 动态区域加权 L1 (渐进式启用) |
| Opacity Entropy | 不透明度稀疏正则 |
| Inverse Depth Smooth | 逆深度平滑正则 |
| Affine Regularization | 仿射变换正则 |
| Per-class Reg Loss | 各高斯类自定义正则 (scale ratio/flatten/sparse/max scale 等) |

## 5. CLI 入口点

| 命令 | 文件 | 功能 |
|---|---|---|
| `train.py` | `tools/train.py` | 主训练入口，支持所有高斯变体和训练器 |
| `eval.py` | `tools/eval.py` | 渲染测试集/全量图像，计算 PSNR/SSIM/LPIPS |
| `recon.py` | `tools/recon.py` | 导出高斯点云为 PLY 文件 |
| `train_road_surface.py` | `tools/train_road_surface.py` | 路面专用重建训练 |
| `preprocess.py` | `datasets/preprocess.py` | 统一数据集预处理 |

配置通过 `--config_file` 指定 YAML 文件，支持 CLI override。配置文件存储在 `configs/` 目录（已 gitignore）。

## 6. 关键设计决策

1. **场景图分解**: MultiTrainer 将场景分解为 Background + RigidNodes + DeformableNodes + SMPLNodes，每类独立建模，实现动态物体与静态背景的分离重建
2. **地面约束**: 地面高斯点 z-scale 梯度置零，仅允许 z 轴旋转，保证路面平坦性
3. **重要性采样**: 基于图像重建误差缓冲区实现训练图像的重要性采样，提高困难视角的训练频率
4. **渐进训练**: SH 度数渐进提升、粗训练后启用变形、深度损失衰减调度
5. **多相机模型**: 同时支持 pinhole 和 Scaramuzza 鱼眼相机模型
6. **2D/3D 高斯混合**: 路面使用 2D 扁平高斯，其他使用 3D 高斯，兼顾精度与效率
7. **动态密度控制**: 自适应分裂/复制/剔除策略，PVG 额外考虑时间维度梯度
8. **模块化设计**: 高斯模型、节点类型、训练器均通过配置动态导入，支持灵活组合
