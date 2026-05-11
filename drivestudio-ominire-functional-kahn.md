# 干线卡车仿真重建方案选型：3DGRUT × OmniRe × PVG

## Context

**目标场景**：自动驾驶卡车的仿真测试，ODD 限定为**干线主道**（不含 urban）。
**已有资产**：
- `/Users/etendue/repo/3dgrut2`（3DGRUT 主干，含 NCore v4 loader、3DGUT/3DGRT 双后端、MCMC、OpenUSD/NuRec USDZ exporter）
- `/Users/etendue/repo/drivestudio`（OmniRe 实现：urban-oriented，多层场景图 + SMPL/Deformable + EnvLight + Per-cam Affine）
- `/Users/etendue/repo/PVG`（PVG 实现：self-supervised SHM dynamic，无 cuboid 标签）
- `/Users/etendue/repo/report/oss-sim-roadmap.md`（已落地的开源 USDZ 生产路线，分 v1/v2/v3）
- `/Users/etendue/repo/3dgrut2/according-to-oss-sim-roadmap-md-how-zazzy-harbor.md`（已写好的 oss-sim 工作包评估）

**用户已确认的边界**：
1. 输入数据形态：**NCore v4 是统一接入层**；自有卡车数据先转成 NCore；cuboid（动态 actor 3D 框 + 轨迹）**完整可得**。
2. 动态目标轨迹：**有完整 cuboid 轨迹** → rigid-track 路线优先，PVG 自监督仅作为补充。
3. 下游消费者：**Omniverse / DriveSim USDZ 优先**，同时保留 mesh.ply / tracks.json 导出能力以对接其它仿真器。

**为什么需要选型**：oss-sim-roadmap 是面向"通用 AV USDZ"的，没有针对干线卡车 ODD 做裁剪；OmniRe 和 PVG 是两个外部参考实现，但都有 urban / 通用倾向。本计划要回答："针对干线卡车 ODD，**最优组合**是什么？"，并给出 3 个不同投入档位的可执行选项。

---

## 1. 干线卡车 ODD 的特殊性（与通用 AV / urban 的差异）

| 维度 | urban / 通用 AV | 干线主道卡车 |
|---|---|---|
| 背景 | 建筑、招牌、店面，丰富中近景 | 远场公路、护栏、远山、大片天空 |
| 动态目标 | 行人 / 骑手 / 复杂 VRU | 主要是车辆（轿车、卡车、商用车），刚体 |
| 速度 | 中低速 | 高速（80–120 km/h），rolling shutter 与运动模糊敏感 |
| 序列长度 | 短（~10 s） | 长（30–120 s），单段 clip 可达数公里 |
| 相机 | pinhole 多 | 卡车驾驶舱常配 120° FOV 广角 / 鱼眼 / FTheta |
| 路面 | 复杂（井盖、补丁、人行道） | 标线密集且关键（车道线、合流线、施工锥） |
| 天空占比 | 中 | 大（地平线低，常占帧 30–50%） |
| 反光 | 玻璃幕墙复杂 | 后车尾灯、护栏反光、潮湿路面镜面 |

**直接结论（决定方案取舍）**：
- ✅ 必须重视：MCMC 长序列预算控制、远场天空建模、广角/鱼眼/RS、跨相机色调一致、Road layer 扁平化（车道线）。
- ❌ 价值低：SMPL 行人 / 复杂 deformable 行人 / 自行车骨架。
- ⚠️ 需要：rigid 动态车辆的 track 绑定 + pose 校准（cuboid 已知 → 不需要 self-supervised）。

---

## 2. 三套技术对比矩阵（按干线卡车 ODD 评分）

| 能力 | 3DGRUT (现状) | OmniRe (drivestudio) | PVG | 干线需求 |
|---|---|---|---|---|
| NCore v4 / fisheye / FTheta / RS | ✅ 已具备 | ❌ 仅 pinhole | ❌ 仅 pinhole | **必须** |
| 双后端：UT 光栅 + OptiX RT | ✅ 已具备 | ❌ gsplat 单后端 | ❌ vanilla 3DGS | 反光/镜面 **重要** |
| MCMC 致密化（硬上限） | ✅ 已具备 | ❌ gradient-based | ❌ gradient-based | 长序列 **必须** |
| Layer-aware MCMC（per-layer cap） | ❌ 待 v2 | ❌ 无 | ❌ 无 | **重要** |
| 多层场景图（BG/Road/Rigid/Sky） | ❌ 单层 | ✅ 已具备（含 SMPL/Deform，干线不需要） | ❌ 单层 | **必须**（裁剪后） |
| Road layer（扁平 Gaussian） | ❌ 待 v2 | ❌ 无 | ❌ 无 | 车道线 **必须** |
| Dynamic rigid local Gaussians + cuboid 绑定 | ❌ 待 v2 | ✅ `RigidNodes` 直接可用 | ⚠️ 隐式 SHM | **必须** |
| Track pose 校准（端点锚定） | ❌ 待 v2 | ✅ `CameraOptModule` + 实例 SE3 优化 | ❌ 无 | **重要** |
| Sky envmap（cubemap） | ❌ 无 | ✅ `EnvLight` (6×1024² cubemap) | ✅ 简单 cubemap | 干线 **必须** |
| Per-camera 颜色校正 | ❌ 全局 PPISP | ✅ `AffineTransform` per-cam | ❌ 无 | **重要** |
| Self-supervised dynamic（无标签） | ❌ 无 | ❌ 无 | ✅ 周期振动 SHM | 标签缺失 fallback |
| USDZ + tracks.json + map.xodr 导出 | ✅ NuRec/OpenUSD 双导出 | ❌ 仅 .pth | ❌ 仅 PLY | **必须** |
| 静态 mesh + ground mesh（碰撞面） | ⚠️ 待 v1 工作包 | ❌ 无 | ❌ 无 | **必须** |
| DiFix / 后处理 fixer | ❌ 无（开源缺失） | ❌ 无 | ❌ 无 | nice-to-have |

**核心结论**：
- **3DGRUT 是唯一具备 USDZ + 双后端 + NCore + MCMC 的主干**，必须留作骨架。
- **OmniRe 的 EnvLight / AffineTransform / RigidNodes / CameraOptModule 是可移植的现成模块**，能补 3DGRUT 在 Sky/色调/dynamic-rigid/pose 校准上的空白；其 SMPL/Deformable 直接舍弃。
- **PVG 的核心价值（self-supervised dynamic）在 cuboid 已知时不必要**；只在自采数据 cuboid 缺失时作为 fallback。

---

## 3. 三个候选方案

### 方案 A · 保守 / MVP（沿用 oss-sim v1，6–8 周）

**思路**：不动训练主干，最快建立干线 baseline + 数据闭环。
- 直接用 `apps/ncore_3dgut_mcmc` 训练，沿用 oss-sim v1 的全部工作包（V1-1 校验器 → V1-6 USDZ 打包）。
- 仅做最小化裁剪：在 dataset 端加 sky_mask（去重训练数据）+ road_mask（用于 ground mesh 抽取）。

**适用**：先验证数据闭环 / 数据采集 / 标注 / 仿真消费链路；不在乎重建质量天花板。

**已知问题**（干线场景下显式回归）：
- 远场天空被大量浪费 Gaussian（无 envmap 层）。
- 动态车辆是单一 Gaussian cloud 跟随帧渲染，会产生残影 / 拖尾。
- 多相机色调跳变（无 per-cam 校正）。
- 长 clip（>30 s）在硬上限附近会出现致密化抖动。

**工作量**：≈ 6 个工作包（V1-1..V1-6），1 人 6–10 周。

---

### 方案 B · **推荐**：3DGRUT v2-fork + OmniRe 关键模块移植（4–5 个月）

**思路**：把 oss-sim v2 工作包**裁剪**为干线卡车版本，并把 OmniRe 的 4 个高价值模块**移植**进 3DGRUT 主干。

#### B-1 多层基座（V2-0）
- `MixtureOfGaussians` 增加 per-Gaussian `layer_id`：`{background, road, dynamic_rigid, sky}`（**砍掉 deformable / smpl**）。
- 关键文件：`threedgrut/model/model.py`、`threedgrut/strategy/base.py`。
- 参考实现：`drivestudio/models/trainers/scene_graph.py:12-71`（多 trainer 协调）。

#### B-2 Road layer（V2-1）
- 用 road_mask 初始化扁平 Gaussian（scale [a, b, ε]，最短轴/最长轴 < 0.2）。
- 验收：路面区域 PSNR 提升 ≥ 0.5 dB；车道线锐度提升（手测 + LPIPS 局部）。

#### B-3 Layer-aware MCMC（V2-2）
- 各 layer 独立 `cap_max`：建议干线初值 `{background:600k, road:300k, dynamic_rigid:200k, sky:0}`（sky 走 envmap 不占 Gaussian）。
- 关键文件：`threedgrut/strategy/mcmc.py` + `threedgrut/strategy/src/`（CUDA plugin 加 layer 索引）。

#### B-4 Dynamic rigid local Gaussians（V2-3）
- **直接采用 cuboid 轨迹作为已知 SE3**，不需要 PVG 自监督。
- 移植 OmniRe `RigidNodes`（`drivestudio/models/nodes/rigid.py:14-40`）：每个 track 一个本地 Gaussian 集合 + 实例 SE3。
- 渲染时按 cuboid 轨迹 SE3 变换 → 复用 3DGRUT 现有 splatting / RT 内核。

#### B-5 Track pose 校准（V2-4，简化）
- 干线 cuboid 标注质量较好，仅做端点锚定 + 中段残差优化。
- 借鉴 OmniRe `CameraOptModule`（`drivestudio/models/modules.py:266-300`）的 9-dim SE3 delta 方法，但应用对象是 actor track 而非相机。

#### B-6 Sky envmap（**新增，对干线极重要**）
- 移植 OmniRe `EnvLight`（`drivestudio/models/modules.py:174-208`，6×1024² cubemap，nvdiffrast 采样）。
- 与 sky_mask 配合：sky 区域 α 推 0、loss 由 envmap 承担。
- 验收：天空 PSNR 提升 ≥ 1.0 dB；天空 Gaussian 数下降 ≥ 80%。

#### B-7 Per-camera 颜色校正（V2-5，用 OmniRe Affine）
- 直接移植 `AffineTransform`（`drivestudio/models/modules.py:210-264`）：每相机一个 affine，恒等初始化。
- 比 oss-sim 路线图原计划的 bilateral grid 更轻量；干线 6 相机够用。
- 验收：跨相机拼接区 PSNR 提升 ≥ 0.5 dB；`identity_init=true` 时 0 回归 < 0.05 dB。

#### B-8 Camera pose refinement（**新增，干线长基线特别有用**）
- 移植 OmniRe `CameraOptModule`（per-frame SE3 deltas）。
- 干线长 clip 中累积 IMU drift / 标定误差会被吃掉，对远场 PSNR 帮助大。
- 默认关闭，长 clip 启用。

#### 输出/打包
- 沿用 3DGRUT `threedgrut/export/usd/nurec/exporter.py`，扩展为多层 Gaussian + envmap 序列化。
- mesh 与 ground mesh 走 oss-sim V1-5 的 Poisson 路径。
- USDZ 打包走 V1-6 + 多层注入。

#### 不做（明确舍弃）
- ❌ SMPL / Deformable layer（干线无 VRU）
- ❌ DiFix / 扩散后处理（v3 再考虑）
- ❌ PVG SHM（cuboid 已知）

**工作量**：≈ 8 个工作包，1 人 4–5 个月（与 oss-sim v2 量级相当，但裁剪掉 v2 的 V2-3 重型 self-supervised 部分，新增 B-6/B-8）。

---

### 方案 C · 激进 / 研究路径（B + PVG 残差 + DiFix 替代器，6–8 个月）

**思路**：在 B 完成后，针对两类干线痛点做研究性增强。

#### C-1 PVG SHM 作为 dynamic-rigid 残差精修
- cuboid 标注端点 / 短期偶尔丢帧时，用 PVG 的 `_t / _scaling_t / _velocity` 三参数（`PVG/scene/gaussian_model.py:142-148`）建模残差振动。
- 不取代 cuboid 主干，仅作 fallback。
- 价值：标注成本下降 / 鲁棒性上升。

#### C-2 开源 DiFix 替代器
- 用轻量 diffusion / U-Net fixer 蒸馏 held-out 视图的伪影（眩光、尾灯拖尾、潮湿路面 ghosting）。
- v3 的 LPIPS 提升 ≥ 10%。

#### C-3 Progressive distillation
- 用 fixer 输出反向蒸馏到 Gaussian 表示，抑制夜间 / 夕阳眩光。

**工作量**：B 之上额外 2–3 个月，研究风险较高。

---

## 4. 推荐方案：**B**

理由：
1. **对齐 ODD**：B 的 7 个模块（多层基座 / Road / Layer-MCMC / Dynamic-rigid / Pose-cal / Sky envmap / Per-cam affine）刚好覆盖干线卡车的所有"必须"项；舍弃 SMPL/Deformable 节省 ≥ 30% 工作量。
2. **复用最大化**：3DGRUT 的 NCore + 双后端 + MCMC + USDZ 是唯一现成主干；OmniRe 的 4 个模块以纯函数形式移植，不需要 fork OmniRe 训练循环；PVG 暂不引入。
3. **下游兼容**：USDZ 优先 + mesh/tracks 副输出 → 同时满足 Omniverse 与 CARLA / 内部仿真器。
4. **风险可控**：所有模块都有现成参考实现（OmniRe 已开源 + 3DGRUT 已落地），仅"集成"工作而非"研发"。

A 适合做 first-light demo（先打通数据→USDZ→Omniverse 闭环）；C 等 B 上线半年并积累真实卡车数据后再启动。

---

## 5. 关键文件清单（按改动优先级）

### 5.1 主干扩展（3DGRUT）
- `threedgrut/model/model.py` — `MixtureOfGaussians` 引入 `layer_id`
- `threedgrut/strategy/mcmc.py` + `threedgrut/strategy/src/` — Layer-aware MCMC
- `threedgrut/strategy/base.py` — 参数更新工具
- `threedgrut/datasets/datasetNcore.py` — 扩展辅助掩码（ego/dynamic/road/sky/valid）
- `threedgrut/trainer.py` — loss 多分支（含 envmap loss + per-cam affine）
- `threedgrut/export/usd/nurec/exporter.py` — 多层 Gaussian + envmap 序列化
- `threedgrt_tracer/` + `threedgut_tracer/` — Slang/CUDA shader 加 layer 索引

### 5.2 新增模块（移植自 OmniRe）
- `threedgrut/model/layers/envlight.py` ← `drivestudio/models/modules.py:174-208`
- `threedgrut/model/layers/affine.py` ← `drivestudio/models/modules.py:210-264`
- `threedgrut/model/layers/rigid_actors.py` ← `drivestudio/models/nodes/rigid.py:14-40`
- `threedgrut/model/layers/cam_pose_opt.py` ← `drivestudio/models/modules.py:266-300`

### 5.3 配置
- `configs/apps/ncore_3dgut_truck_highway.yaml`（**新建**，B 方案专用）
- `configs/strategy/mcmc_layered.yaml`（**新建**，per-layer cap）
- `configs/render/3dgut_truck.yaml`（**新建**）

### 5.4 工具与导出（沿用 oss-sim v1，已评估）
- `threedgrut/tools/ncore_validate.py`（V1-1）
- `threedgrut/tools/aux_masks.py`（V1-3，加 sky）
- `threedgrut/export/scripts/export_rig.py` / `export_tracks.py` / `copy_xodr.py`（V1-4）
- `threedgrut/export/mesh/extract_static.py` / `extract_ground.py`（V1-5）

---

## 6. 端到端验证方案

### 6.1 数据准备 smoke
```bash
# 自有卡车 clip → NCore v4
python -m threedgrut.tools.truck_to_ncore --src <truck-clip> --out <ncore-clip>

# 校验
python -m threedgrut.tools.ncore_validate --clip <ncore-clip> --out <out>/manifest.json

# 辅助掩码（含 sky）
python -m threedgrut.tools.aux_masks --clip <ncore-clip> --out <out>/masks --types ego,dynamic,road,sky,valid
```
**期望**：manifest 含 6 段；sky/road/dynamic mask 与帧时间戳/分辨率对齐。

### 6.2 训练（B 方案）
```bash
python train.py --config-name apps/ncore_3dgut_truck_highway \
  dataset.path=<ncore-clip> dataset.aux_masks=<out>/masks \
  output_path=<out>/run_b
```
**期望**：训练完成；layer 计数符合 cap；envmap 收敛；per-cam affine 接近恒等 + 微调。

### 6.3 单元/回归
```bash
# 多层基座往返（PLY/USD round-trip 保层）
python -m threedgrut.tools.layer_roundtrip --ckpt <out>/run_b/ours/checkpoint_last.pt --format usd

# 各 layer 预算
python -m threedgrut.tools.layer_budget_check --ckpt <out>/run_b/ours/checkpoint_last.pt \
  --caps background=600000,road=300000,dynamic_rigid=200000

# Road 扁平度
python -m threedgrut.tools.layer_stats --ckpt <out>/run_b/ours/checkpoint_last.pt --layer road --max-flatness 0.2

# Sky 提升
python -m threedgrut.tools.region_psnr --ckpt-a <out>/run_a --ckpt-b <out>/run_b --region sky --min-improvement 1.0

# Per-cam affine 恒等回归
pytest threedgrut/tests/test_affine_identity.py -x

# Dynamic-rigid SE3 一致性（已知 cuboid → 像素位置误差）
python -m threedgrut.tests.synthetic_truck_actor --max-px-err 1.5
```

### 6.4 USDZ 打包与仿真 smoke
```bash
python -m threedgrut.export.usd.exporter --ckpt <out>/run_b/ours/checkpoint_last.pt \
  --rig <out>/rig_trajectories.json --tracks <out>/sequence_tracks.json --xodr <out>/map.xodr \
  --static <out>/mesh_static.usd --ground <out>/mesh_ground.usd \
  --envmap <out>/run_b/ours/envmap.exr \
  --out <out>/scene.usdz

usdchecker --strict <out>/scene.usdz
python -m threedgrut.tools.sim_smoke --usdz <out>/scene.usdz --replay-ego --check-collision

# Omniverse 加载（外部脚本，可选）
omniverse-cli load <out>/scene.usdz --replay
```

### 6.5 KPI 阈值（vs 方案 A baseline）
| 指标 | 方案 A | 方案 B 目标 |
|---|---|---|
| 整体 PSNR | baseline | +0.8 dB |
| Sky 区域 PSNR | baseline | +1.0 dB |
| Road 区域 PSNR | baseline | +0.5 dB |
| Dynamic actor 像素位置误差中位 | N/A | < 1.5 px |
| Gaussian 总数（同等质量） | baseline | -20%（envmap 节流） |
| 跨相机拼接区 PSNR | baseline | +0.5 dB |

---

## 7. 实施阶段拆解（推荐方案 B 的执行顺序）

| 阶段 | 工作包 | 量级 | 退出条件 |
|---|---|---|---|
| Stage-0 | oss-sim v1 全套（V1-1..V1-6） | 1.5 月 | 干线 clip 能产 USDZ 并被 Omniverse 加载 |
| Stage-1 | B-1 多层基座（layer_id） | 3 周 | 单 layer baseline 不回归 |
| Stage-2 | B-3 Layer-MCMC + B-2 Road | 3 周 | Road PSNR ≥ +0.5 dB |
| Stage-3 | B-6 Sky envmap | 2 周 | Sky PSNR ≥ +1.0 dB |
| Stage-4 | B-7 Per-cam affine | 1 周 | 恒等 0 回归 + 多曝光 ≥ +0.5 dB |
| Stage-5 | B-4 Dynamic-rigid + B-5 Track 校准 | 5 周 | actor 像素误差中位 < 1.5 px |
| Stage-6 | B-8 Camera pose refine（可选） | 2 周 | 长 clip 整体 PSNR 不回退 |
| Stage-7 | USDZ 打包扩展 + 端到端 QA | 2 周 | 全部 KPI 达标 + Omniverse smoke 通过 |

---

## 8. 风险与回退

| 风险 | 触发条件 | 回退 |
|---|---|---|
| 移植 OmniRe 模块时与 3DGRUT 双后端不兼容 | RT 路径 envmap / affine 数学不匹配 | 降级为仅 UT 后端启用，RT 路径 fallback 到全局色调 |
| Layer-aware MCMC 在 CUDA 内核里改动量大 | OptiX/Slang shader 重编译失败 | 先在 Python 侧做 layer mask 后处理（性能下降但功能可达） |
| Cuboid 标注质量低导致 dynamic-rigid 有残影 | actor 像素误差 > 5 px | 启用 C-1 PVG SHM 残差精修 |
| 自有卡车数据 → NCore 转换失真 | NCore 校验器报错 | 先做"半 NCore" loader（仅必填字段），逐步补全 |

---

## 9. 与 oss-sim-roadmap 的对齐

本计划是 oss-sim-roadmap 的**干线卡车 ODD 特化版本**：
- **沿用**：v1 全部 6 个工作包（V1-1..V1-6）
- **裁剪**：v2 中的 V2-3（动态刚体）从"自监督 + pose 校准"简化为"已知 cuboid + 端点锚定"
- **替换**：v2 V2-5 bilateral grid → 更轻量的 OmniRe AffineTransform
- **新增**：v2 增补 B-6 Sky envmap、B-8 Camera pose refine（移植自 OmniRe）
- **删除**：v3 的 V3-1（Deformable）从干线 ODD 中**永久移除**；V3-2 Sky envmap 提前到 v2；V3-3/V3-4 维持 v3
- **保留外部** `official-nre-compat` profile：B 方案输出可对照 NVIDIA NRE 官方 USDZ 做 QA，但不依赖闭源容器
