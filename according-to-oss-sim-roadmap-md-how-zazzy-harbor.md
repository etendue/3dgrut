# 评估：实现 oss-sim 路线图所需的 3DGRUT 工作量

## Context

`/Users/etendue/repo/report/oss-sim-roadmap.md` 描述了一个分三阶段（v1/v2/v3）的开源 AV USDZ 生产路线图，使用 3DGRUT 作为 v1 的"重建主干"。本评估的目的是：在不动手实现的前提下，回答用户的问题——**"在 3dgrut 上需要做多少工作？请列出工作包。"** 并为每个工作包补充**可执行的验收方案**，使后续按 WP 推进时有清晰可验证的退出条件。

> 评估对照对象：`/Users/etendue/repo/3dgrut2`。
> 默认输入数据：NCore v4 clip。下文 `<clip>` 表示一段已知的 NCore v4 clip 路径，`<out>` 表示输出目录。

---

## 1. 现状摘要：3DGRUT 已具备的能力

| 能力 | 状态 | 关键路径 |
|---|---|---|
| NCore v4 数据集加载（多相机/LiDAR/rolling shutter） | ✅ 已具备 | `threedgrut/datasets/datasetNcore.py` |
| NCore Hydra 训练预设（GS + MCMC） | ✅ 已具备 | `configs/apps/ncore_3dgut.yaml`, `configs/apps/ncore_3dgut_mcmc.yaml` |
| 相机模型（pinhole / OpenCVFisheye / FTheta） | ✅ 已具备 | `datasetNcore.py:1154-1230`, `export/usd/writers/camera.py` |
| 单一 ego 掩码加载与 loss 加权 | ✅ 已具备 | `datasetNcore.py:351-361`, `trainer.py:514-517` |
| MCMC 致密化（全局预算） | ✅ 已具备 | `threedgrut/strategy/mcmc.py`, `configs/strategy/mcmc.yaml` |
| 训练指标（PSNR/SSIM/LPIPS） | ✅ 已具备 | `trainer.py:344-348` |
| OpenUSD `ParticleField3DGaussianSplat` 导出 | ✅ 已具备 | `threedgrut/export/usd/exporter.py` |
| NuRec/Omniverse 格式 USDZ 导出（旧式） | ✅ 已具备 | `threedgrut/export/usd/nurec/` |
| 网格 → USDZ 注入脚本 | ✅ 已具备 | `threedgrut/export/scripts/add_mesh_to_usdz.py` |
| PPISP 全局后处理（单一色调映射） | ✅ 已具备 | `render.py:51-92`, `trainer.py:943` |

---

## 2. 工作包总览

### v1：开源 baseline（绝大多数在训练器之外）

| WP-ID | 工作包 | 对 3dgrut 改动 | 量级 |
|---|---|---|---|
| **V1-1** | NCore v4 校验器 + `scene_manifest.json` 生成 | 新增 `threedgrut/datasets/ncore_validator.py` | M |
| **V1-2** | 3DGRUT 训练 wrapper（CLI、checkpoint+metrics） | `train.py` 外薄壳 + 复用 `ncore_3dgut_mcmc.yaml` | S |
| **V1-3** | 辅助掩码预处理 + 训练器 loss 扩展 | `datasetNcore.get_mask_images()` 扩展, `trainer.py:514-517` | L |
| **V1-4** | `rig_trajectories.json` / `sequence_tracks.json` / `map.xodr` 导出 | 新增 `export/scripts/export_*.py` | M |
| **V1-5** | 静态网格 + 地面网格抽取 | 新增 `export/mesh/extract_*.py` | L |
| **V1-6** | 完整 USDZ 打包 + 包清单(hash) + QA | 扩展 `export/usd/exporter.py`, 新增 QA 脚本 | M |

### v2：NuRec 训练循环改造（**fork 主干**）

| WP-ID | 工作包 | 量级 |
|---|---|---|
| **V2-0** | 多层架构基座：`MixtureOfGaussians` 引入 per-Gaussian `layer_id` | L |
| **V2-1** | Road Layer：扁平化高斯尺度先验 + 道路掩码初始化 | M |
| **V2-2** | Layer-aware MCMC：每层独立 `cap_max` + kernel 加 layer 索引 | M |
| **V2-3** | 动态刚体局部高斯集（按 track 局部帧）—— v2 中**最重**的项 | XL |
| **V2-4** | Track pose 校准（端点固定残差） | L |
| **V2-5** | 每相机 bilateral grid 颜色校正 | M |

### v3：研究性扩展（高风险）

| WP-ID | 工作包 | 量级 |
|---|---|---|
| **V3-1** | 动态可形变 actor 层（行人/骑手） | XL |
| **V3-2** | 天空 envmap（独立可训练背景层） | L |
| **V3-3** | 开源 DiFix 替代器（diffusion fixer） | XL |
| **V3-4** | Progressive distillation | L |

---

## 3. 各工作包验收方案（可执行）

> 命名约定：每个 WP 的"验收准则"摘自路线图第 5/6 节；"可执行验证"给出具体的命令/脚本/期望输出，使任何评审人都可以复现验证。

### V1-1 · NCore v4 校验器 + scene_manifest.json

**验收准则（路线图）**
- 校验相机/LiDAR 组件可用性、时间戳单调性、帧覆盖、`T_sensor_rig`、ego pose、坐标系元数据、动态 track、map 元数据。
- 必填项缺失时阻断训练。

**可执行验证**
```bash
# 1. 正例：已知良好 clip 应通过
python -m threedgrut.tools.ncore_validate --clip <clip> --out <out>/manifest.json
test $? -eq 0
python -c "import json,sys; m=json.load(open('<out>/manifest.json')); \
  assert {'cameras','lidar','poses','tracks','map_meta','frame_range'} <= set(m), m.keys()"

# 2. 负例：删掉 LiDAR 组件应阻断
python -m threedgrut.tools.ncore_validate --clip <clip-no-lidar> --out /tmp/m.json && exit 1 || true

# 3. 时间戳单调性回归测试
pytest threedgrut/tests/test_ncore_validator.py -k "monotonic or coverage"

# 4. JSON Schema 验证
jsonschema -i <out>/manifest.json schemas/scene_manifest.schema.json
```
**期望**：1 退出 0 且 manifest 包含全部 6 个段；2 退出非 0 且打印缺失项；3 测试通过；4 schema 校验通过。

---

### V1-2 · 3DGRUT 训练 wrapper

**验收准则（路线图）**
- 不依赖 NVIDIA NuRec/NRE 容器即可运行 upstream 3DGRUT。
- 记录 tool commit, config, GPU 类型, runtime, Gaussian 数, 最终指标。
- 产出 held-out 渲染。
- 不主张 NuRec 特性对等。

**可执行验证**
```bash
# 1. 端到端短跑（小 clip + 少量迭代）
python train.py --config-name apps/ncore_3dgut_mcmc \
  dataset.path=<clip> output_path=<out> training.iterations=1000

# 2. checkpoint 产出与字段
python -c "import torch; ck=torch.load('<out>/ours/checkpoint_last.pt', map_location='cpu'); \
  print('keys=', list(ck.keys())[:8]); assert 'positions' in ck or 'state_dict' in ck"

# 3. metrics 报告字段完整性
python -c "import json; r=json.load(open('<out>/training_report.json')); \
  for k in ['git_commit','config_hash','gpu_name','runtime_sec','num_gaussians','psnr_holdout','ssim_holdout','lpips_holdout']: assert k in r, k"

# 4. 容器无关性
docker run --rm -v <out>:/out python:3.11 ls /out/training_report.json   # 不依赖 NVIDIA 镜像
```
**期望**：训练完成、checkpoint 与 metrics 字段齐备、报告文件不来自任何闭源镜像。

---

### V1-3 · 辅助掩码预处理 + 训练器 loss 扩展

**验收准则（路线图）**
- 掩码与相机时间戳/分辨率对齐；动态对象掩码覆盖 tracked 车辆与 VRU 并支持 padding；road mask 可被地面网格与未来 road-layer 训练消费；同输入同 config 时确定性。

**可执行验证**
```bash
# 1. 分辨率/时间戳对齐
python -m threedgrut.tools.aux_masks --clip <clip> --out <out>/masks --types ego,dynamic,road,valid
python -c "
import os, json, cv2
mani = json.load(open('<out>/manifest.json'))
for cam in mani['cameras']:
    rgb = cv2.imread(cam['frames'][0]['path']); h,w = rgb.shape[:2]
    for t in ['ego','dynamic','road','valid']:
        m = cv2.imread(f\"<out>/masks/{cam['id']}/{t}/0000.png\", 0)
        assert m.shape == (h,w), (t, m.shape, (h,w))
"

# 2. 确定性（哈希一致）
sha256sum <out>/masks/**/*.png | sort > /tmp/h1
rm -rf <out>/masks && python -m threedgrut.tools.aux_masks --clip <clip> --out <out>/masks --types ego,dynamic,road,valid
sha256sum <out>/masks/**/*.png | sort > /tmp/h2
diff /tmp/h1 /tmp/h2

# 3. 动态对象覆盖率
python -m threedgrut.tools.mask_qa --masks <out>/masks --tracks <out>/sequence_tracks.json --min-iou 0.95

# 4. 训练器消费回归（带掩码 vs 无掩码 PSNR 不应回退超过 5%）
python train.py --config-name apps/ncore_3dgut_mcmc dataset.path=<clip> dataset.aux_masks=<out>/masks output_path=<out>/with
python train.py --config-name apps/ncore_3dgut_mcmc dataset.path=<clip>                                output_path=<out>/no
python -m threedgrut.tools.compare_metrics <out>/with <out>/no --max-psnr-drop 0.5
```
**期望**：1 全通过；2 哈希一致；3 IoU≥0.95；4 PSNR 回退在阈值内。

---

### V1-4 · rig / track / map 导出

**验收准则（路线图）**
- `rig_trajectories.json` 含 ego trajectory + 传感器外参 + world-base transform。
- `sequence_tracks.json` 含 track id, class, extent, timestamped poses, visibility。
- `map.xodr` 保留 georeference。
- 抽样 pose 与 NCore 源差异在配置容差内。

**可执行验证**
```bash
python -m threedgrut.tools.export_rig    --clip <clip> --out <out>/rig_trajectories.json
python -m threedgrut.tools.export_tracks --clip <clip> --out <out>/sequence_tracks.json
python -m threedgrut.tools.copy_xodr     --clip <clip> --out <out>/map.xodr

# 1. JSON Schema
jsonschema -i <out>/rig_trajectories.json schemas/rig_trajectories.schema.json
jsonschema -i <out>/sequence_tracks.json  schemas/sequence_tracks.schema.json

# 2. Pose round-trip：随机抽 100 帧，残差 < 1e-3 m / 1e-4 rad
python -m threedgrut.tools.pose_residual_check \
  --rig <out>/rig_trajectories.json --clip <clip> \
  --num-samples 100 --max-trans 1e-3 --max-rot 1e-4

# 3. xodr georeference 保留
python -c "
import xml.etree.ElementTree as ET
root = ET.parse('<out>/map.xodr').getroot()
geo = root.find('header/geoReference')
assert geo is not None and 'proj' in (geo.text or ''), 'geoReference missing'
"

# 4. tracks 可视性字段抽样
python -m threedgrut.tools.tracks_qa --tracks <out>/sequence_tracks.json --require-fields id,class,extent,poses,visibility
```
**期望**：所有 4 项均通过；pose 残差在容差内。

---

### V1-5 · 静态网格 + 地面网格

**验收准则（路线图）**
- 静态网格默认排除动态 actors。
- 地面网格作为独立碰撞面。
- 地面与 LiDAR 路面回波残差低于阈值。
- 网格在 OpenUSD 工具中可加载，且可标记为 visible/invisible/collision-only。

**可执行验证**
```bash
# 1. 抽取
python -m threedgrut.export.mesh.extract_static --ckpt <out>/ours/checkpoint_last.pt \
  --tracks <out>/sequence_tracks.json --out <out>/mesh_static.usd
python -m threedgrut.export.mesh.extract_ground --lidar <clip>/lidar --road-mask <out>/masks --out <out>/mesh_ground.usd

# 2. usdchecker 必须无 blocking error
usdchecker <out>/mesh_static.usd && usdchecker <out>/mesh_ground.usd

# 3. 静态网格排除动态 actor：动态 bbox 内的顶点占比 < 0.5%
python -m threedgrut.tools.mesh_dynamic_leak --mesh <out>/mesh_static.usd --tracks <out>/sequence_tracks.json --max-leak 0.005

# 4. 地面残差：LiDAR 路面回波到地面网格的中位距离 < 5cm
python -m threedgrut.tools.ground_residual --mesh <out>/mesh_ground.usd --lidar <clip>/lidar --road-mask <out>/masks --max-median 0.05

# 5. visibility/collision toggling 可读
python -m threedgrut.tools.usd_inspect --usd <out>/mesh_static.usd --check-attrs visibility,physics:collisionEnabled
```
**期望**：1–5 全部退出 0；阈值参数可在配置中调整但默认值与路线图一致。

---

### V1-6 · 完整 USDZ 打包 + QA

**验收准则（路线图）**
- USDZ 包含 Gaussian、static mesh、ground mesh、`rig_trajectories.json`、`sequence_tracks.json`、`map.xodr`、`data_info.json`、包清单（含 hash 与工具版本）。
- 根 stage 使用米单位与文档化的 NCore→USD 坐标变换。
- `usdchecker` 无 blocking error。
- 仿真 loader 可加载并 replay ego trajectory。

**可执行验证**
```bash
# 1. 打包
python -m threedgrut.export.usd.exporter --ckpt <out>/ours/checkpoint_last.pt \
  --rig <out>/rig_trajectories.json --tracks <out>/sequence_tracks.json --xodr <out>/map.xodr \
  --static <out>/mesh_static.usd --ground <out>/mesh_ground.usd \
  --out <out>/scene.usdz

# 2. 包成员清单
unzip -l <out>/scene.usdz | tee /tmp/usdz_list
for f in scene.usda gaussians.ply data_info.json rig_trajectories.json sequence_tracks.json map.xodr mesh_static.usd mesh_ground.usd manifest.json; do
  grep -q "$f" /tmp/usdz_list || { echo "missing $f"; exit 1; }
done

# 3. usdchecker（路线图刚性要求）
usdchecker --strict <out>/scene.usdz

# 4. 单位与坐标变换文档化
python -c "
from pxr import Usd, UsdGeom
s = Usd.Stage.Open('<out>/scene.usdz')
assert UsdGeom.GetStageMetersPerUnit(s) == 1.0, 'must be meter scale'
"
python -c "import json; d=json.load(open('<out>/data_info.json')); assert 'ncore_to_usd_transform' in d"

# 5. 包清单 hash 与实际内容一致
python -m threedgrut.tools.verify_manifest --usdz <out>/scene.usdz

# 6. 仿真 smoke test（使用目标仿真器或最低限度 usdview）
python -m threedgrut.tools.sim_smoke --usdz <out>/scene.usdz --replay-ego --check-extrinsics --check-collision

# 7. 渲染指标聚合（按相机 + clip）
python render.py --config-path <out>/ours --config-name config dataset.path=<clip>
python -m threedgrut.tools.metrics_aggregate --in <out>/ours/eval --out <out>/metrics_clip.json
```
**期望**：1–7 全部退出 0；`usdchecker --strict` 无 error；smoke test 报告显示 ego trajectory 已 replay、外参/时间戳一致、地面碰撞被读取。

---

### V2-0 · 多层架构基座

**验收准则（自定义；为后续 v2 全部 WP 的前置条件）**
- `MixtureOfGaussians` 增加 per-Gaussian `layer_id` 张量；保存/加载/PLY/USD 往返保持。
- 现有 baseline（无 layer）所有测试仍通过。
- 优化器组按 layer 聚合不破坏 selective Adam 收敛。

**可执行验证**
```bash
# 1. 现有测试零回归
pytest threedgrut/tests -x

# 2. 往返保持
python -m threedgrut.tools.layer_roundtrip --ckpt <ckpt> --format ply
python -m threedgrut.tools.layer_roundtrip --ckpt <ckpt> --format usd

# 3. 单 layer baseline 收敛对照（同 seed 同 iter，PSNR 差距 < 0.1dB）
python train.py --config-name apps/ncore_3dgut_mcmc ... model.layered=false output_path=<out>/flat
python train.py --config-name apps/ncore_3dgut_mcmc ... model.layered=true  model.num_layers=1 output_path=<out>/single_layer
python -m threedgrut.tools.compare_metrics <out>/flat <out>/single_layer --max-psnr-drop 0.1
```

---

### V2-1 · Road Layer

**验收准则**
- 训练开始时 road_mask=1 像素覆盖到的 Gaussians 数 ≥ 配置最小值。
- road layer Gaussians 的 `scale` 沿法向被压扁（最短轴/最长轴比 < 阈值 e.g. 0.2）。
- 含 road layer vs 无 road layer：路面区域 PSNR 提升 ≥ 0.5dB。

**可执行验证**
```bash
python train.py --config-name apps/ncore_3dgut_mcmc_v2_road \
  dataset.path=<clip> dataset.aux_masks=<out>/masks output_path=<out>/road

python -m threedgrut.tools.layer_stats --ckpt <out>/road/ours/checkpoint_last.pt --layer road \
  --min-count 5000 --max-flatness 0.2

python -m threedgrut.tools.region_psnr --ckpt-a <out>/no_road/ours --ckpt-b <out>/road/ours \
  --mask <out>/masks --region road --min-improvement 0.5
```

---

### V2-2 · Layer-aware MCMC

**验收准则**
- 每个 layer 的 Gaussian 数始终 ≤ 配置上限。
- 全局总数与各层和一致。
- vs 全局 cap 单 layer 基线：在固定预算下整体 PSNR 不回退。

**可执行验证**
```bash
python train.py --config-name apps/ncore_3dgut_mcmc_v2 \
  strategy.add.per_layer_caps='{background:500000,road:300000,dynamic:200000}' \
  output_path=<out>/lmc

python -m threedgrut.tools.layer_budget_check --ckpt <out>/lmc/ours/checkpoint_last.pt \
  --caps background=500000,road=300000,dynamic=200000

python -m threedgrut.tools.compare_metrics <out>/lmc <out>/baseline_global_cap --max-psnr-drop 0.0
```

---

### V2-3 · 动态刚体局部高斯集

**验收准则**
- 给定 `sequence_tracks.json` 中某 track 的位姿轨迹，模型重渲染时其 Gaussians 随帧 SE3 移动。
- 合成测试（已知 actor 位姿）下，actor 像素位置误差中位 < 1px。
- 双渲染后端（3dgrt OptiX、3dgut splatting）输出在合成场景上 PSNR 差距 < 0.3dB。
- 训练梯度回流到 SE3 参数（数值一致性测试）。

**可执行验证**
```bash
# 1. 合成 actor 像素回归
python -m threedgrut.tests.synthetic_dynamic_actor --backend 3dgut --max-px-err 1.0
python -m threedgrut.tests.synthetic_dynamic_actor --backend 3dgrt --max-px-err 1.0

# 2. 双后端一致性
python -m threedgrut.tools.compare_backends --ckpt <out>/dyn/ours/checkpoint_last.pt --max-psnr-gap 0.3

# 3. SE3 梯度有限差分
pytest threedgrut/tests/test_dynamic_se3_grad.py -x

# 4. 端到端：在带动态 actor 的真实 clip 上整体 PSNR 提升 ≥ 1.0dB
python -m threedgrut.tools.compare_metrics <out>/dyn <out>/baseline_static --min-psnr-gain 1.0 --region dynamic
```

---

### V2-4 · Track pose 校准

**验收准则**
- 端点姿态与源对齐：`‖pose_optim[0] − pose_src[0]‖ ≤ ε` 且末端同理。
- 对 track 全段，与 ground-truth 残差中位 < 校准前。
- 训练 loss 收敛单调（无 explosion）。

**可执行验证**
```bash
python train.py --config-name apps/ncore_3dgut_mcmc_v2 model.calibrate_track_poses=true output_path=<out>/cal

python -m threedgrut.tools.track_pose_check --ckpt <out>/cal/ours/checkpoint_last.pt \
  --tracks <out>/sequence_tracks.json --endpoint-eps 1e-4 --median-residual-improve 0.2

python -m threedgrut.tools.loss_monotonic --log <out>/cal/ours/train.log --window 500
```

---

### V2-5 · Per-camera bilateral grid

**验收准则**
- 网格 0 初始化时整体渲染与无 grid baseline PSNR 差 < 0.05dB（恒等近似）。
- cam_i 的 grid 改动不影响 cam_j 渲染。
- 多相机曝光差异场景下 PSNR 提升 ≥ 0.5dB。

**可执行验证**
```bash
# 1. 恒等回归
python train.py ... model.bilateral_grid.identity_init=true training.iterations=0 output_path=<out>/bg_id
python -m threedgrut.tools.compare_metrics <out>/bg_id <out>/no_bg --max-psnr-drop 0.05

# 2. 相机独立性
pytest threedgrut/tests/test_bilateral_grid_independence.py -x

# 3. 曝光差异 clip 上的提升
python -m threedgrut.tools.compare_metrics <out>/bg <out>/no_bg --min-psnr-gain 0.5 --clip <multi-exposure-clip>
```

---

### v3 (V3-1..V3-4) · 验收方案占位

由于 v3 工作以研究为主，详细验收方案应在每项启动时单独立项；最低限度的退出条件如下：

- **V3-1 动态可形变 actor**：行人 clip 上肢体伪影像素率较 v2 静态/刚体方案降低 ≥ 30%；同时不破坏背景静态区域 PSNR（回退 < 0.2dB）。
- **V3-2 sky envmap**：天空区域（有效像素掩码 + 上视）PSNR 提升 ≥ 1.0dB；天空与前景分离不引入接缝（边缘伪影 < 1%）。
- **V3-3 DiFix 替代器**：在 held-out 新视图上 LPIPS 较未蒸馏基线降低 ≥ 10%；与训练数据时长无显著正相关污染（防过拟合）。
- **V3-4 progressive distillation**：分阶段引入 fixer 输出后训练曲线无 collapse；最终 PSNR/SSIM/LPIPS 不回退于 v2。

---

## 4. 总体工作量画像

```
对 3dgrut 仓库代码的侵入度
v1  ▮▮░░░░░░░░  低（主要新增导出脚本与掩码扩展）
v2  ▮▮▮▮▮▮▮▮░░  高（fork 主干：模型/MCMC/双渲染后端）
v3  ▮▮▮▮▮▮▮▮▮▮  研究级（含独立子项目）
```

- **v1**：3dgrut 本身改动很小，约 6 个工作包，1 人 6–10 周。
- **v2**：fork 主干并改造 5 个核心子系统（多层基座、Road、Layer-aware MCMC、动态刚体+pose 校准、bilateral grid），约 6 个工作包，1 人 ≥3 个月。
- **v3**：研究性，至少 4 个工作包，多数依赖外部组件。

**最关键的单一架构改动**：将 `MixtureOfGaussians` 由扁平张量重构为带 layer_id / 可选 transform_id 的多组结构（V2-0），是 v2 全部任务的隐性前置依赖。

---

## 5. 关键文件清单（按改动优先级）

- `threedgrut/model/model.py`（参数张量、优化器组）
- `threedgrut/strategy/mcmc.py` + `threedgrut/strategy/src/`（致密化与 CUDA plugin）
- `threedgrut/strategy/base.py:77-107`（参数更新工具）
- `threedgrut/datasets/datasetNcore.py`（掩码、track、pose、map 元数据）
- `threedgrut/trainer.py`（loss、render 调用、validation）
- `threedgrt_tracer/`（OptiX RT 内核）
- `threedgut_tracer/`（rasterizer / Slang 着色器）
- `threedgrut/export/usd/exporter.py` 与 `add_mesh_to_usdz.py`（导出与打包）
- `configs/apps/ncore_3dgut_mcmc.yaml` 及相关 strategy/render 配置

---

## 6. 评估结论的可重复检查

任何评审人在仓库内运行下列命令即可复核本评估对"已具备/缺失"的判断：

```bash
# A. 确认 v1 JSON 导出尚未实现：仅命中 writers/camera.py 的注释引用
grep -rn "rig_traject\|sequence_track\|data_info\|map.xodr\|scene_manifest" threedgrut/

# B. 确认 USDZ 导出已具备（OpenUSD + NuRec 两套）
ls threedgrut/export/usd/ threedgrut/export/usd/nurec/

# C. 确认 NCore 训练预设已具备
cat configs/apps/ncore_3dgut_mcmc.yaml

# D. 确认模型为扁平张量（v2 重构对象）
grep -n "self\.positions\|self\.rotation\|self\.scale\|self\.density" threedgrut/model/model.py | head

# E. 确认 trainer 现状只支持单类掩码加权
grep -n "mask" threedgrut/trainer.py | head

# F. 量级估算与路线图甘特图对比（v1 ≈ 55d, v2 ≈ 59d）
sed -n '/section v1/,/section v3/p' /Users/etendue/repo/report/oss-sim-roadmap.md
```

如评估结果与上述任一命令的输出冲突，应立即修正本计划文件。后续若用户希望对任一 WP 启动实际实施，应在本评估基础上单独写实施计划（包括 TDD 测试列表、PR 拆分、回滚策略），不要直接在本评估文档中实施。
