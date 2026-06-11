# E0.5 — NVIDIA NuRec 官方训练配方 vs 本项目 multilayer 配方逐项 diff 清单

> v4_plan.md E0.5 交付物。喂 E3（表示侧强化）与 E2（生成修复链）借鉴清单用。

## 0. 两侧来源与对比口径

| 侧 | 来源 | 说明 |
|---|---|---|
| **官方 NuRec** | `nre 26.4.146-c63f08a4`（`version.version_string`，commit 2026-05-28），配方 `Hyperion-8.1 car2sim_6cam` + pai overlay，在 PAI clip `9ae151dc` 上训练时的 **Hydra resolved 全量配置**（`/tmp/e05_diff/nurec_parsed.yaml`，8438 行）；层级结构参考顶层配方 `/tmp/e05_diff/car2sim.yaml`、`/tmp/e05_diff/car2sim_6cam.yaml` | resolved 配置 = 实际生效值，含所有 default 链合并结果 |
| **本项目** | [`configs/apps/ncore_3dgut_mcmc_multilayer.yaml`](../../../configs/apps/ncore_3dgut_mcmc_multilayer.yaml) @ git HEAD `f8b5b70`，含 defaults 链：`base_mcmc → base_gs`、`dataset/ncore`、`initialization/lidar`、`render/3dgut → 3dgrt`、`strategy/layered_mcmc → mcmc`；**层级默认值在代码 registry**（`threedgrut/layers/registry.py` STANDARD_LAYERS） | 本项目大量层配置不在 yaml 而在代码 dataclass，对比时一并引用 |

**读法约定**：
- 官方 key 路径以 resolved yaml 为准（如 `model.layers.road.initialization.name`）；本项目以 yaml key 或代码位置标注。
- 官方配置里 `lambda_: 0.0` 的 loss 标为 **OFF（配置存在）**；resolved 配置无法区分「key 存在但模块未实例化」（如 `bilateral_grid_*` 四个 loss key 存在，但 `model.post_processing` 只有 `b: ppisp`，无 bilateral grid 模块），此类标注「**疑未生效**」。
- 找不到对应项写「**未发现**」，不臆测。

---

## 1. 层结构

官方 4 个高斯层 + 1 个非粒子 sky env map 模块（`model.background`）+ PPISP 后处理；本项目 4 个层（其中 sky_envmap 为非粒子层）+ BilateralGrid 曝光。注意本项目的「7 层」指 yaml 递归链层数（dynfix→…→ncore_3dgut_mcmc），不是模型层数——模型层是 `layers.enabled: [background, road, dynamic_rigids, sky_envmap]`。

| 条目 | 官方（resolved） | 本项目 multilayer | 差异含义 | 外推相关性 | 借鉴建议 |
|---|---|---|---|---|---|
| 层清单 | `model.layers`: `background`（sh-gaussians）/ `road`（sh-gaussians）/ `dynamic_rigids`（rigid-gaussians）/ `dynamic_deformables`（deformable-gaussians）；另有 `model.background`（sky-env-map 模块） | `layers.enabled: [background, road, dynamic_rigids, sky_envmap]`；`dynamic_deformables` 在 registry 中是 stub（`max_n_particles=0, is_particle_layer=False`） | 官方多一个**可变形层**（行人/骑行者用 hash-grid deform 网络）；本项目行人散在 background 层里 | 低（行人不是 lateral 外推主要伪影源） | 暂不补 deformables；行人质量成为瓶颈时再立项 |
| 背景层时变外观 | `model.layers.background.fourier_features_dim: 5` + `time_embed: holistic-remap-time-input-embedding`（car2sim.yaml 注释：为兼容 Pacsim 给 background 开 temporal appearance） | 无（background 层 albedo 静态；P1.3b 的 `n_fourier_albedo_terms` extra 仅 dynamic_rigids 可用，默认 1=DC） | 官方把全局光照/曝光的时间变化吸收进背景外观，几何/SH 不必为它扭曲 | 中 | 与 PPISP 配合看（§6）；若 BilateralGrid 已吸收每相机差异，剩余时变残差可考虑 bg Fourier dim 小值实验 |
| dyn 层时变外观 | `model.layers.dynamic_rigids.fourier_features_dim: 20`（defaults: `temporal_appearance@model.layers.dynamic_rigids: fourier_features_dim_20`）+ `time_embed: individual-remap` | 无（V3-L8 per-track albedo bias 实验性，默认 OFF） | 官方车辆外观随时间变（车灯/反光），20 维 Fourier | 低-中 | 车辆闪烁问题再启用；本项目 30k 实测 per-track offset 有副作用，需要 warmup/lr 重调 |
| 每层 SH | 所有层 `particle.radiance_sph_degree: 3` + `radiance_sph_O0: true`，progressive 0→3 每 1000 步 +1（`progressive_training`） | 同：`particle_radiance_sph_degree: 3`（render/3dgrt.yaml），progressive 0→3 每 1000 步（base_gs.yaml `model.progressive_training`）；road 层无降阶（`LayerSpec.sh_degree` 保留未用） | SH 阶与渐进解锁**一致**；官方 road 也没有降 SH 阶（road 的「不过拟合」靠几何冻结+正则，见 §8） | 中（E3.2 的反证据：官方未对 road 砍 SH，而是冻几何） | E3.2 做 road DC-only freeze 时把「官方不降阶但冻几何」作为对照解释 |
| 语义 head | 每层 `particle.camera_extra_signal_dim: 20` + `extra_signal.semantic_logits`（20 类 camera 信号，activation none）；配套 `loss.semantic`（CE λ0.001, start 1000）与 `loss.node_semantic_gaussians`（见 §8） | 无（sseg 仅作 region mask 进 layered L1，不渲染语义） | 官方每个高斯带 20 维语义 logits 可被渲染监督 → 高斯有「类别归属」，支撑层间互斥正则 | 中 | E3.1 空气区 penalty 的官方等价物是「语义互斥 + background_in_track」；若做层归属约束可借此思路而不必上完整语义渲染 |
| 参数预算（init→cap） | init ≈ 2.0M：bg 800k 点云 +100k near +100k far、road 400k、rigids 300k、deformables 300k（car2sim.yaml 注释 + 各层 `initialization.num_*`）；`model.strategy.add.max_n_gaussians: 2_500_000` | init：bg ≤1M（`initialization.num_points: 1_000_000` 全局 LiDAR 子采样）、road ≤200k（BEV grid）、dyn ≤5k/track；add cap per-layer：bg 600k / road 200k / dyn 300k（registry + yaml override）≈ 总 1.1M | 官方总预算 **≈2.3 倍**于本项目；尤其 road 400k vs 200k、bg(含远场) 1.0M vs 0.6M cap | 中 | E0.4 对锚时把参数量对齐（或至少记录差值），否则 PSNR gap 里有纯容量项 |
| 背景层排除 road 类点 | `model.layers.background.ignore_classes_from_layers: ['road']` + `class_labels: null`；road 层 `class_labels: ['road']` | 无等价物：bg `init_from_lidar` 用全部非动态点（**含 road 点**，trainer.py L368），road/bg 初始即重叠；靠 V3-R2 `bg_road_penalty` 事后赶出 | **官方从初始化就做 road/bg 所有权切分**；本项目 P3.3 诊断的 road/bg 耦合（bg 替补率高）根源之一 | **高** | 见 §3/§8 与 Top-5 第 2 条：bg init 时按 lidar-sseg 剔除 road 类点，一行级改动 |

---

## 2. LiDAR 监督

| 条目 | 官方（resolved） | 本项目 multilayer | 差异含义 | 外推相关性 | 借鉴建议 |
|---|---|---|---|---|---|
| LiDAR ray 渲染监督 | **有**：每步采 `dataset.n_train_sample_lidar_rays: 2048` 条真实 LiDAR ray，走 3dgut-nrend lidar 渲染路径（`renderer.tiling.lidar`、`antialiasing.lidar_divergence: 0.002`、`culling.far_clip_distance_lidar: 200.0`），depth loss `loss.lidar`: fn l1, **λ0.005**（car2sim.yaml 0.03 → car2sim_6cam.yaml 覆盖为 0.005），`use_z_depth: false`（ray 距离口径） | **无 ray 渲染**；替代：预 dump 的图像空间 LiDAR 深度图监督（`dataset.load_lidar_depth_map: true` + `trainer.use_lidar_depth: true`，`lambda_lidar_depth: 0.1`，`lidar_w_decay: 0` 全程，L1，`depth_max: 80.0`；sky/dyn/invalid 像素剔除） | 官方深度监督进的是**渲染器原生 lidar ray**（含 200m 远场），λ 小但全程、且与相机像素同 batch 联合优化；本项目是 camera 投影后的稀疏深度图（≤80m、受投影/遮挡损失） | **中-高**（几何被真 ray pin 住 → 视角外推时表面位置更稳，路面/立面少「可动自由度」） | 短期：保留 image-space 方案但放宽 `depth_max`（80→150/200）对齐官方远场约束做 A/B；长期：E4（LiDAR 可选主线）评估接 ray 级监督 |
| LiDAR intensity 监督 | **未启用**：`loss.intensity`: fn mse, **λ0.0**（OFF，配置存在）；各层 `particle.lidar_extra_signal_dim: 0`（无 intensity 通道）。任务输入提到 pai.yaml 注释掉的 intensity extra_signal——resolved 证实该 run 没开 | 无（完全没有 intensity 概念） | 官方该 run **同样没用** intensity；它是个预留通道（`lidar_extra_signal_sph_degree: 0`） | 低 | 不跟进；官方自己都没开 |
| raydrop 监督 | `loss.raydrop`: mse **λ0.0**（OFF，配置存在） | 无 | 同上，预留未启用 | 低 | 不跟进 |
| sky 区域 LiDAR 远场 anchor | `loss.background_lidar`: fn mse, λ0.05（sky env map 对 lidar 路径的背景监督） | `trainer.lambda_bg_lidar: 0.005`（sky 区域 pred_dist 推向 `depth_max=80` 的 MSE，`compute_bg_lidar_loss`） | 思路同源（防 sky 区高斯坍缩近场）；官方权重 10×、且 anchor 在 lidar ray 口径上 | 中 | A/B `lambda_bg_lidar 0.005→0.05`，便宜实验 |
| LiDAR 动态点处理 | `dataset.lidar_dynamic_points.method: dynamic_tracks`；`valid_lidarpoints_cuboid_track_params.track_padding_m: [0.5,0.5,0.25]` | 等价：init 时 `non_dynamic_points_only=True` + cuboid 过滤（`get_dynamic_lidar_points`） | 双方都按 cuboid track 切静/动 LiDAR | 低 | 无动作 |

---

## 3. 初始化

| 条目 | 官方（resolved） | 本项目 multilayer | 差异含义 | 外推相关性 | 借鉴建议 |
|---|---|---|---|---|---|
| background init | `model.layers.background.initialization.name: lidar-rig-trajectory`：`num_point_cloud_points: 800000`、`num_near_points: 100000`、`num_far_points: 100000`、`far_radius_factor: 20`、`scale_multiplier: 0.2`、`observation_scale_factor: 0.01`、`default_density: 0.5`、`default_scale: 0.1`、`non_dynamic_points_only: true`、`step_frame: 1`（任务输入提到 3.67M→800k + sparsity compensation；resolved 仅见上述 key，sparsity compensation 在 resolved 配置中**未发现**对应 key，应为实现内行为） | `initialization`: `method: lidar`、`num_points: 1_000_000`（全局子采样）、`observation_scale_factor: 0.01`、`use_observation_points: false`（multilayer 覆盖 lidar.yaml 的 true）；bg 层吃整包非动态点 | 官方额外造 **100k near + 100k far（半径=场景 20 倍）随机点**兜远场/天际线；本项目无 far points（远场全靠 sky envmap 接） | 中（lateral 平移时远场视差小，但中远建筑/树带断层会暴露；far points 给 MCMC 提供远场 donor） | 在 bg init 加 near/far 随机点带（各 ~100k）是低风险移植；优先级中 |
| road init | `model.layers.road.initialization.name: lidar-ground-mesh-road`：LiDAR 地面**网格化**（`voxel_size: 0.1`、`smoothing_passes: 10`、平面 RANSAC：`num_plane_hypotheses: 100`、`plane_max_distance: 0.3`、`plane_max_angle_deg: 30`、`ground_compat_max_distance: 0.5`、`ground_compat_max_angle_deg: 60`、`min_ray_length: 0.1`）→ `num_point_cloud_points: 400000`、`num_random_points: 0`（6cam 覆盖掉 car2sim 的 100k 随机点）、`default_density: 0.99`、`default_scale: [0.1, 0.1, 0.001]`（薄片）、`scale_multiplier: 1.0`。颜色来源 resolved 中未明示（任务输入称 colored ground mesh；`checkpoint.artifact.mesh.ground.colored: false` 是导出选项非 init） | `threedgrut/layers/road_init.py`：BEV 2D grid（5cm）+ road-sseg LiDAR 点 XY 最近邻取 Z（cdist KNN），≤200k，identity 旋转，scale log(0.1,0.1,0.001)，density logit 0.0，**颜色 neutral gray 0.5**（TODO 注释：未做图像投影上色） | 两边都是「贴地薄片」，但官方初始 **density 0.99（几乎实心）+ 平滑网格表面 + 法向来自 mesh**；本项目 density init 低、Z 来自单点 KNN（噪声直接进 Z）、灰色起步靠 photometric 学色 | **高**（路面是 3m/6m lateral 外推的主要近场区域；初始几何质量决定后续要不要靠「可动自由度」去补） | E3.3 BEV 纹理平面化的输入：①road init Z 改成局部平面拟合/平滑（对齐 smoothing_passes 思想）②init density 提到 ~0.99 ③init 颜色做 LiDAR/图像投影上色 |
| dynamic_rigids init | `initialization.name: lidar-dynamic-tracks`：`num_point_cloud_points_per_track: 5000`、`num_point_cloud_points_in_layer: 300000`、`symmetric_axis: 'Y'`（车辆左右对称镜像补点）、`default_density: 0.5`、`scale_multiplier: 0.2`、`keep_all_track_poses: false` | `init_dynamic_rigid_layer`: `max_pts_per_track=5_000`（trainer.py），层 cap 300k（yaml override）；`symmetric_axis: null`（**默认 OFF**，V3-L5 已实现可 CLI 开 `++layers.overrides.dynamic_rigids.symmetric_axis=Y`） | per-track 5k + 层 300k **完全对齐**（V3-L5 就是抄的官方）；唯一差异：官方 symmetric_axis=Y 默认开，本项目默认关（30k 实测有反转，单独 ablation 未跑） | 低-中（actor 侧） | V3-L5b ablation 补跑后决定默认值 |
| dynamic_deformables init | 同 lidar-dynamic-tracks，300k，`symmetric_axis: null` | 层为 stub，不初始化 | 见 §1 | 低 | 同 §1 |
| init 总规模 | ≈2.0M（见 §1 参数预算行） | ≈1.0M+0.2M+dyn | 同 §1 | 中 | 同 §1 |

---

## 4. densification / pruning / MCMC

双方同为 MCMC 系（官方 `model.strategy.name: mcmc`，gsplat 风格；本项目 `strategy/layered_mcmc`→MCMCStrategy 子类）。

| 条目 | 官方（resolved） | 本项目 multilayer | 差异含义 | 外推相关性 | 借鉴建议 |
|---|---|---|---|---|---|
| add（增密） | `start_iteration: 20000`、`end_iteration: 35000`、`frequency: 100`、`max_n_gaussians: 2_500_000`（6cam 覆盖；car2sim 原值 start 默认/max 2M） | `strategy.add`: start 500、end 25000、freq 100、`max_n_gaussians: 1_000_000`（全局；实际 per-layer cap 走 registry） | **官方前 20k 步完全不增密**——先让 ~2M LiDAR init 几何收敛，后 15k 步才补容量；本项目 500 步就开始 add | 中-高（早期增密的高斯倾向沿训练视线方向铺，几何欠约束时易种下视角依赖伪影） | A/B：`strategy.add.start_iteration 500→{10000, 20000}`（30k 训练等比 15000）；便宜且直接 |
| relocate | start 500、end 35000、freq 100、`max_invisible_steps: 1000`（按不可见步数判死） | start 500、**end 25000**、freq 100、`max_relocation_fraction: 0.4`（dynfix 防瞬间团聚）；死亡判定走 `opacity_threshold` | 官方多一个「连续 1000 步不可见即可搬」的判据；本项目按 opacity | 低-中 | 暂不动；本项目 0.4 cap 是 dyn 层稳定性补丁，保留 |
| perturb（MCMC 噪声） | start 1、end 40000（=全程）、freq 1、`noise_lr: 5000.0`（default 与 per-layer 同值：`noise_lr.layers.{dynamic_rigids,dynamic_deformables,road}: 5000.0`）、`move_outside_of_cuboid: false` | start 0、**end 27500**、freq 1、`noise_lr: 500000.0`（MCMC 论文/gsplat 默认 5e5）；road 层 perturb Z 分量置零（registry `perturb_scale_mask: (1,1,0)`） | 官方 noise_lr 数值是本项目的 **1/100**（本项目公式 `randn * gate(density) * noise_lr * pos_lr`，threedgrut/strategy/mcmc.py L223；官方公式未见源码，若同源则官方扰动小两个量级）；官方扰动跑满全程 | 中（更小扰动 = 收敛后期表面更稳；但跨实现数值不可直接比，需实验确认） | 做一次 `noise_lr 5e5→5e3` 的 5k smoke 看 train PSNR/表面噪点；若官方公式确实同源，这是免费的稳定性 |
| 层豁免 | `model.strategy.exclude_layer_ids: ['road']`（6cam：road 不 add/不 relocate/不 perturb/不 prune） | 无 road 豁免（road 参与 add/relocate/perturb，仅 perturb Z 被 mask、opacity reg 豁免 `loss.exempt_layers_opacity_reg: [road]`） | **官方 road 粒子集合自始至终不变**，几何完全由 init mesh 决定；本项目 road 粒子被 MCMC 动态搬运（P2A 修的 BEV 洞就是这条链的副作用） | **高** | E3 候选：LayeredMCMCStrategy 加 `exclude_layers` 支持（road 全豁免），与 §8 冻结 lr 配套成「官方 road 处理」完整移植 |
| 死亡阈值 | `opacity_threshold: 0.005`、`binom_n_max: 51` | 同：0.005 / 51（strategy/mcmc.yaml） | 一致 | — | — |
| 不透明度衰减正则 | 无全局 opacity L1；用 `loss.gaussian_density`（abs λ0.005，`visibility_filter: true`，road 层 `layer_lambdas.road: 1.0`）见 §5 | `loss.lambda_opacity: 0.01`（L1 全局，road 豁免） | 官方密度正则带**可见性过滤**（只罚当前视锥可见粒子）且默认权重减半 | 中 | 若复刻官方 road 冻结，则 P2A 的 road 豁免可被「road 全豁免 MCMC」替代 |

---

## 5. 正则项与 loss lambdas 总表

官方 resolved `loss.*` 全集 vs 本项目（base_gs.yaml `loss.*` + trainer 内联 loss）。

| loss | 官方 fn / λ | 本项目 fn / λ | 差异含义 | 外推相关性 | 借鉴建议 |
|---|---|---|---|---|---|
| rgb | l1, **0.8**（`loss.rgb`） | l1, **0.8**（`lambda_l1`，region-weighted layered L1） | 一致；本项目多 region 加权（sky 区交给 envmap） | — | — |
| ssim | **0.2**, window 11, mask mode GT（`loss.ssim`） | **0.2** 全图（mask 乘进 rgb） | 一致 | — | — |
| lpips | **未发现**（训练 loss 无 lpips） | 无 | 双方都不用 lpips 训练 | — | — |
| sky envmap 颜色 | `loss.background`: mse **0.05** | `trainer.lambda_sky: 0.1`（L1, sky 区域, pre-exposure） | 量级近似，fn 不同 | 低 | — |
| sky lidar | `loss.background_lidar`: mse **0.05** | `lambda_bg_lidar 0.005` | 见 §2 | 中 | A/B 提权 |
| sky TV | `loss.sky_env_map_background`: total_variation_spatial **0.01** | **无**（MLP/cubemap 无 TV 正则） | 官方 sky 纹理有平滑正则 | 低 | sky 出噪点再加 |
| lidar depth | l1 **0.005**（ray 口径） | l1 **0.1**（图像深度图口径，全程） | 见 §2；注意双方口径不同，λ 不可直接比 | 中-高 | 见 §2 |
| PPISP 正则 | `loss.ppisp`: λ1.0，子项 `lambdas`: exposure_mean 0.001 / exposure_smooth 1.0 / vig_center 0.02 / vig_channel 0.01 / vig_non_pos 0.01 / color_mean 0.01 / color_smooth 1.0 / crf_range 0.01 / crf_gamma 0.001 / crf_channel 0.01 | BilateralGrid：AdamW `weight_decay 1e-4`（向 identity 收缩） | 官方 ISP 模型带均值/平滑/合法性正则族；本项目用 weight decay 一刀流 | 低 | — |
| bilateral grid TV | `loss.bilateral_grid_*` 4 项（λ 0.001/1.0/0.001/0.001）——**疑未生效**（post_processing 只挂了 ppisp） | 不适用（1×1×1 grid 无空间 TV 必要） | — | — | — |
| bg-in-track（cuboid 内 bg 压制） | `loss.background_in_track_gaussian`: **bce_with_logits λ0.1**，`layer_names: [background]`，`layer_labels_to_use.background: [person, rider, car, truck, bus, train, motorcycle, bicycle]`（car2sim.yaml 注释：只罚动态类，提升 cuboid 附近静态区） | `trainer.bg_dyn_cuboid_penalty`: enabled, **λ0.15**, warmup 1000, cuboid mask + dyn 位置 clamp | 思路同源（T8/B3 即对标此项）；官方按**语义类别**限定罚谁，本项目按几何 cuboid 罚全部 | 中 | 已有等价物；官方的「只罚动态类」可避免误伤 cuboid 内静态结构（树干/护栏），可小改 |
| out_of_bound | l1 **λ1.0**（动态层粒子越出 cuboid 的软惩罚） | `dyn_clamp_to_cuboid: true`（硬 clamp 回 |local|≤size/2） | 官方软约束 vs 本项目硬投影 | 低 | 无动作（硬 clamp 已工作） |
| semantic 渲染 | CE **λ0.001**, start 1000（`loss.semantic`） | 无 | 见 §1 语义 head | 中 | 见 §1 |
| node_semantic（层语义互斥） | `loss.node_semantic_gaussians`: **bce λ1.0**, start 1000，`layer_labels_to_use.road: [road]`、`layer_labels_to_exclude.background: [road]` | 近似物：V3-R2 `bg_road_penalty`（λ0.1, warmup 1000, BEV cell 1.0, z_band 0.4，几何判据） | **官方用渲染语义证据驱动「road 像素归 road 层、bg 层在 road 区压 density」**，λ 高达 1.0；本项目用 BEV 高度场几何判据 | **高**（road/bg 耦合 = P3.3 诊断主病灶，官方在 loss 级做所有权分配） | E3.1 演进方向：bg_road_penalty 的判据从纯几何带升级为「sseg road 区域 + 几何带」联合，λ 对齐官方量级试 1.0 |
| road 专项 | `loss.road_gaussians`: abs **λ1.0**, `layer_name: road`, `n_samples: 10`, `grid_len: 0.2`, `min: 2.0`, `range: 4.0`, `rotation_lambda: 10.0`（road 局部网格采样平整性 + 旋转对齐，参数语义源码侧，resolved 仅见数值） | 无等价 loss（靠 init 薄片 + clamp） | 见 §8 | **高** | 见 §8 |
| road z-scale | `loss.gaussian_z_scale`: abs **λ1.0**, `layer_name: road`, `road_z_scale: 0.001`（Z scale 钉死 1mm 量级） | registry `scale_z_max: 0.05`（硬 clamp 5cm）+ init z=0.001 | 官方**软 loss 持续把 road Z scale 拉回 0.001**；本项目只设上限 5cm，0.001→0.05 之间无约束 | **高** | 便宜移植：road 层加 `abs(scale_z - 0.001)` 正则或把 `scale_z_max` 收紧到 0.005 做 A/B |
| scale 正则 | `loss.gaussian_scale`: abs **λ0.005**, `visibility_filter: true`, `layer_lambdas.road: 0.01`（road 2×） | `lambda_scale: 0.01` 全局、无可见性过滤 | 官方带可见性过滤 + road 加倍 | 低-中 | — |
| density 正则 | `loss.gaussian_density`: abs **λ0.005**, `visibility_filter: true`, `layer_lambdas.road: 1.0`（road 200×—— 与 road density init 0.99 + density lr 1e-4 配合，把 road 钉在实心态） | `lambda_opacity: 0.01` 全局 L1，road **豁免**（P2A） | 方向相反的同目的：官方「重罚 road density 漂移」（注：fn=abs 的具体目标值语义在 resolved 中不可见），本项目「road 不罚」——都是防 road 粒子死亡/穿洞 | 中 | 维持 P2A；若做 road 冻结移植则统一到官方方案 |
| gaussian_flatten | abs **λ0.0**（OFF；`max_to_median_ratio_threshold: 1.0, axes_type: fixed` 预留） | 近似物 `anisotropy_ratio_max: 8.0`（road 硬 clamp，registry） | 官方针叶抑制是预留未开；本项目已有 road 各向异性 clamp（V3-R1.2） | 低 | 不动 |
| normal | l1 **λ0.0**（OFF） | 无 | 双方未用 | 低 | — |
| deform 平滑 | `loss.deform_smoothness`: abs **λ0.01**（deformables 专用，`smoothness_frame_steps: 5`） | 不适用（无 deformables） | — | 低 | — |
| opacity/scale 全局 | （见 gaussian_density/gaussian_scale 行） | `use_opacity: true λ0.01（road 豁免）`、`use_scale: true λ0.01` | — | — | — |
| 本项目独有 | — | `bg_road_penalty λ0.1`、`lambda_road_eff_rank 0.0（OFF）`、pose smooth/boundary/prior（poseopt OFF）、`depth_prior λ0.01（OFF）` | 官方无 eff-rank / 无图像深度先验 | — | — |

---

## 6. 相机处理

| 条目 | 官方（resolved） | 本项目 multilayer | 差异含义 | 外推相关性 | 借鉴建议 |
|---|---|---|---|---|---|
| 相机数 | 6-cam：`dataset.camera_ids` = front_wide_120 + **front_tele_30** + cross_left/right_120 + rear_left/right_70（hyperion8.1_6cam_1lidar） | 5-cam 对称环（无 front_tele_30）；V3-E4 实测 7-cam 反而 -0.42 dB | 官方比本项目多前向长焦；远处前向纹理监督更密 | 低-中（front_tele 对 lateral 外推帮助有限，对前向远场有) | E0.3/E0.4 对锚跑官方配方时保持其 6-cam 原样，本项目对比时注明相机集不同 |
| 相机模型 | 数据侧 FTheta（PAI/Hyperion 标配，resolved 配置不含内参；`dataset.camera_max_fov_deg: 190.0`） | 同数据源 FTheta（NCore loader） | 一致 | — | — |
| rolling shutter | `model.renderer.projection.n_rolling_shutter_iterations: 5` | `render.splat.n_rolling_shutter_iterations: 5` | **一致**（同源 3dgut） | — | — |
| UT 投影参数 | `ut_dim 3 / ut_alpha 1.0 / ut_beta 2.0 / ut_kappa 0.0 / ut_require_all_sigma_points: true / image_margin_factor 0.1 / min_projected_ray_radius: 0.5477` | `ut_alpha 1.0 / ut_beta 2.0 / ut_kappa 0.0 / ut_require_all_sigma_points_valid: false / ut_in_image_margin_factor 0.1`；min_projected_ray_radius **未发现** | 仅 `require_all_sigma_points` 不同（官方更严格：任一 sigma point 投影失败即剔除该高斯——高畸变边缘更干净） | 低-中（图像边缘=lateral 外推后变中心的区域） | 把 `ut_require_all_sigma_points_valid` 翻 true 做 5k smoke，看边缘伪影与 PSNR |
| 渲染排序 | `render.mode: kbuffer, k_buffer_size: 0`、`global_z_order: false` | `k_buffer_size: 0`、`global_z_order: true` | 排序口径不同（resolved 值如实记录；性能/瑕疵权衡需实验） | 低 | 不优先 |
| ISP / 曝光补偿 | **PPISP**（`model.post_processing.b: ppisp-post-processing`）：per-camera exposure（lr 0.02）+ vignetting（0.01）+ color matrix（5e-5）+ CRF（0.01），`start_global_step: 250`，warmup→cosine；`per_frame_ppisp_enabled: false`（car2sim.yaml 注释：无 controller + 与 bg Fourier 冲突，所以关 per-frame） | **BilateralGrid**（V3-P1）：per-camera 1×1×1 = 12 参数 3×4 色彩仿射，`exposure_lr 1e-3`，freeze 2000 步后 cosine，AdamW wd 1e-4 | 官方 ISP 物理分解（含 vignetting + CRF 非线性），本项目单一色彩仿射；官方明确**不做 per-frame**（与本项目 per-camera 同口径） | 中（外推视角的颜色一致性影响 difix 链输入质量） | raw/cc gap 已降到 1.21dB，短期够用；若 E0.4 对锚发现官方 cc 口径优势大，再评估 vignetting/CRF 项 |
| 像素采样 | `dataset.samplers.batch_sampler.camera_pixel_sampler`: image-crop, `crop_type: full_image`, **`subsample: 2`**（defaults: `subsampling/3dgut: batch_sampler_4`）；另有 `n_train_sample_camera_rays: 6144`（与 full_image 采样并存，生效性存疑） | 全图全分辨率（`dataset.downsample: 1.0`, `sample_full_image: true`） | **官方训练像素网格做 1/2 子采样**（等效半分辨率监督）→ 单步像素量 ~1/4，高频纹理过拟合压力更小、吞吐更高 | 中 | 值得 A/B：训练时 downsample 0.5 + 步数 ×1.33（对齐官方 40k），看 novel-view LPIPS 是否反升 |
| 有效像素 mask | `camera_mask_sources: [dataset, aux]`、`n_camera_mask_dilation_iterations: 30`、scene-flow 动态 mask（`flow_min_speed_ms: 1.4, flow_dilate_radius: 20`）、traffic light 膨胀 21、cuboid padding [1,1,0.25]m | dataset ego mask + sseg/cuboid mask（dyn mask padding 走 cuboid 投影） | 官方 mask 工程更重：**scene flow 兜未标注动态物** + 大膨胀核 | 中（漏标动态物会被烤进静态层 → 外推鬼影） | E2/E3 之外的卫生项：若 novel-view 出现「拖影车」，优先补 scene-flow mask 而不是调正则 |
| 逐帧 pose 校准 | `model.calib`: free-pose-calib **enabled: false**（camera 子开关 true 但总开关 false；注：car2sim.yaml defaults 写 `calib: enabled`，resolved 为 false，应是 pai overlay/外部覆盖关掉） | 无逐帧相机 pose 优化（pose_adjustment 是 actor track 级，且默认 OFF） | 该 run 双方都没做相机 pose 精调 | 低 | 不跟进 |

---

## 7. sky 处理

| 条目 | 官方（resolved） | 本项目 multilayer | 差异含义 | 外推相关性 | 借鉴建议 |
|---|---|---|---|---|---|
| 模型 | `model.background`: name **sky-env-map**, `envmap_type: cubemap`, **512×512**, `composite_in_linear_space: false`, `saturate_radiance: true` | sky_envmap 层：cubemap **128**（registry extra 默认）或 **MLP fallback**（A800/inceptio 实际用 `trainer.sky_backend=mlp`） | 官方 cubemap 分辨率 4×，且生产路径始终是纹理 cubemap；本项目生产路径常退化到 MLP | 低-中（sky 在 lateral 外推中变化小，但 sky/远景交界破绽会进 difix 输入） | 若 sky 边缘糊：cubemap 分辨率 128→256/512 便宜实验（nvdiffrast 可用的机器） |
| 填洞 | `should_inpaint: true`, `inpaint_threshold: 0.05`, `inpaint_kernel_size: 10`, `min_grad_updates: 1000`（少观测 texel 用邻域 inpaint） | 无 | 官方对**训练视角没覆盖到的天空区域做 inpaint**——novel view 拉出新天空时不出黑斑 | 中（外推视角必然暴露未观测 sky texel） | 移植候选：MLP backend 天然平滑可不做；cubemap backend 补 inpaint |
| 监督 | `loss.background` mse 0.05 + TV 0.01 + `background_lidar` 0.05；sky 优化器独立（`system.optimizers.params.background.lr: 0.001`, cosine T_max 40000） | `lambda_sky: 0.1` L1 + `sky_lr 0.01`（base_gs trainer.sky_lr） | 本项目 sky lr 高 10×、无 TV | 低 | — |
| sky 与几何耦合 | sky 是渲染合成背景（`(1-T)` 之外的 escape ray 落 envmap），高斯层不含 sky 类点（`camera_point_cloud_ignore_classes: [egocar, sky]`） | 同思路：`_blend_sky` 以 `(1 - pred_opacity)` 混合 | 架构一致 | — | — |

---

## 8. road 专项（重点 — 直接喂 E3.1/E3.2/E3.3）

官方对 road 的完整处理是一个**五件套闭环**，单独列出：

1. **几何来源**：`lidar-ground-mesh-road` init —— LiDAR 地面点 → voxel 0.1m → RANSAC 平面族 → **10 轮平滑**的 ground mesh → 采 400k 薄片（`default_scale [0.1,0.1,0.001]`、`default_density 0.99`）。几何是「网格表面」不是「点」。
2. **几何冻结**：road 专属优化器组（car2sim_6cam.yaml 注释原话："Road geometry is initialized from the ground mesh and kept near-frozen during training. Only appearance (features_albedo) learns at a normal rate"）：`positions lr 1e-6`（默认 1.6e-4 的 **1/160**）、`densities 1e-4`（默认 0.05 的 1/500）、`rotations 1e-4`、`scales 1e-4`，**只有 `features_albedo 0.0025` 保持正常**；且 `scale_pos_lr_by_scene_extent: false`（road positions lr 不被场景尺度放大）。
3. **MCMC 全豁免**：`model.strategy.exclude_layer_ids: ['road']` —— 不增密、不搬运、不扰动、不剪枝；粒子集合恒定。
4. **软正则钉死残余自由度**：`road_gaussians`（abs λ1.0，局部网格平整性 + `rotation_lambda 10.0` 旋转对齐）+ `gaussian_z_scale`（λ1.0 把 Z scale 拉回 0.001）+ `gaussian_scale` road 2× + `gaussian_density` road 200×。
5. **所有权切分**：bg init 剔 road 类点（`ignore_classes_from_layers: ['road']`）+ `node_semantic_gaussians`（bce λ1.0：road 层只渲染 road 语义、bg 层在 road 语义区被压）。
6. （外观）`fourier_features_dim: 1`（接近静态外观）；验证时还能单渲 road（`system.test.val_render_selected_nodes.road_only`）。

| 条目 | 官方（resolved） | 本项目 multilayer | 差异含义 | 外推相关性 | 借鉴建议 |
|---|---|---|---|---|---|
| 几何自由度总量 | ≈0（init mesh + lr 冻结 + MCMC 豁免 + 软正则四重锁） | 大：positions lr 1.6e-4×scene_extent（与 bg 同）、scales lr 0.005、参与 MCMC add/relocate/perturb（仅 Z-perturb mask + clamp 上限 + opacity 豁免兜底）。registry 的 `scale_lr_mult: 0.2` **定义了但全代码未消费**（grep 仅 registry.py/layer_spec.py 两处，未接线——文档性 bug） | 官方路面=「带纹理的固定网格」，photometric 误差只能改颜色；本项目路面高斯可以**移动/缩放/重生来迎合训练视角**——这正是 lateral 外推时路面糊/重影的自由度来源（v3 P3.3/P3.4 诊断链） | **高（本清单第一优先）** | E3 主菜：把五件套按性价比拆解移植——(a) road MCMC 豁免 +(b) road positions/scales/densities lr 冻结（每层 optimizer override，代码量小）→ 先做；(c) z-scale 软正则；(d) bg init 剔 road 点 + 语义互斥；(e) E3.3 BEV 纹理化是它的更激进等价物（颜色也不给 per-gaussian 学） |
| SH 阶 | road 仍 sph_degree 3（**没有 DC-only**），靠几何冻结防过拟合 | E3.2 计划 DC-only freeze | 官方证明「冻几何 + 全 SH」可行；view-dependent 颜色本身不是官方眼里的敌人 | 高（修正 E3.2 预期：DC-only 若有效，机理是间接减少逃逸通道，不是对齐官方） | E3.2 照跑（便宜），但解读时与 (a)(b) 对照——若冻结几何后 DC-only 无增益即可放弃 |
| road 外观时变 | `fourier_features_dim: 1` | 无时变 | 近似一致 | 低 | — |
| 验证可观察性 | `val_render_selected_nodes`: road_only / background_only 单层渲染 | render.py 无单层渲染输出（viz_4d 可视化有层信息） | 官方把「road 层自身长什么样」做成例行验证产物 | 中（E1 测量门的诊断手段） | E1 补一个 `--render-layers road` 的 eval 选项，对 R10（路面出洞）监控极有用 |

---

## 9. 训练长度与调度

| 条目 | 官方（resolved） | 本项目 multilayer | 差异含义 | 外推相关性 | 借鉴建议 |
|---|---|---|---|---|---|
| 总步数 | **40000**（`trainer.max_epochs: 1` × `dataset.n_samples_per_epoch: 40000`，6cam 覆盖 car2sim 的 30000）；`checkpoint.every_n_train_steps: 40000` | **30000**（base_gs `n_iterations`，CLI 可覆盖；v3 KPI 跑 30k） | 官方多 33% 步数（但单步像素 ~1/4，见 §6 像素采样） | 低-中 | E0.4 对锚记录双方 wall-clock 与像素吞吐，换算同预算对比 |
| 优化器 | `fused_adam`，`betas: [0.9, 0.99]`、`eps 1e-15`，按 param 分组 lr（positions 1.6e-4 / densities 0.05 / albedo 0.0025 / specular 1.25e-4 / rotations 1e-3 / scales 5e-3——bg 层；road 见 §8；rigids albedo 6.25e-4 即 1/4） | `torch.optim.Adam`，**betas 默认 (0.9, 0.999)**、eps 1e-15，全层共享同一组 lr（base_gs `optimizer.params`，数值与官方 bg 层一致） | ① 二阶动量 0.99 vs 0.999（官方自适应更快）；② 官方 **per-layer lr 分化**（road 冻结、rigids albedo 减 4×），本项目所有层同 lr | 中（per-layer lr 是 §8 的载体） | 跟随 §8 做 per-layer optimizer override 时顺手支持 betas 配置；单独改 betas 优先级低 |
| positions 调度 | ExponentialLR `gamma 0.9998848773724686`/step → 40k 步衰减 ×0.01（1.6e-4→1.6e-6） | exp 调度 lr 1.6e-4→1.6e-6，`max_steps 30000`（base_gs scheduler） | **相对衰减完全一致（×0.01）**，仅时长不同 | 低 | — |
| 其他参数调度 | densities/albedo/scales 恒定；sky bg cosine（T_max 40000, min_factor 0.0333）；PPISP/tracks_calib：250/500 步冻结 → Linear → StepFunCosine（`SequentialLR`） | densities 恒定（`scheduler.density.type: skip`）；exposure：冻 2000 步 → CosineAnnealing | 官方所有辅助模块统一「先冻 → 线性升 → cosine 降」模板 | 低 | — |
| 全局精度 | `trainer.precision: 32` + `matmul_precision: medium` | fp32（默认） | 近似一致 | — | — |
| relative 缩放 | `trainer.relative_lr / relative_schedule / relative_num_workers: true`（按 world size 缩放，单卡无效果） | 无 | 单卡场景无差异 | — | — |

---

## 10. 难例 / novel-view / difix 蒸馏钩子

| 条目 | 官方（resolved） | 本项目 multilayer | 差异含义 | 外推相关性 | 借鉴建议 |
|---|---|---|---|---|---|
| difix 蒸馏（训练中） | `difix` 顶层段存在：name **cosmos-difix**，权重 `nurec-fixer/cosmos_3dgut.pt`（NGC URL），模型分辨率 576×1024。**本 run `difix.training.enabled: false`**（car2sim 系默认关；兄弟配方 sqa_difix_distill 开启——resolved 中未含其值，未发现其具体参数）。已固化的钩子参数：`start_step: 20000`、`p_scheduler`: p_init 0.5、milestones [25000, 28000]、gamma 0.5、`use_color_transfer: true`、`novel_view_poses: translation [0, ±3.0, 0]`（**正负 3m 平移的 novel 视角**）、`shuffle_novel_views: false` | 训练中无任何 difix/novel-view 钩子；difix 仅 eval 后处理（render/3dgrt.yaml `use_difix: false` 默认关，T15.2）+ 独立 nurec-fixer 链 | **官方训练框架原生支持「±3m 平移 novel view 渲染 → difix 修复 → 蒸馏回模型」**，概率 p 从 0.5 按里程碑衰减、20k 步后启动。这就是 lateral 3m 外推的直接训练时增强；本 run 虽关，但配方族里它是一等公民 | **高（E2 主线对应物）** | E2 生成修复链的目标形态即此：①先用 E0.2 官方链验证 difix-distill 开启配方（sqa_difix_distill）在 3m/6m 上的增益上限；②本项目侧把 render-time difix（T15.2 已有）升级为 train-time 蒸馏时，直接抄 p_scheduler/start_step/±3m pose 这组超参 |
| novel-view 合成位姿 | 上行 `novel_view_poses`：横向（rig Y）±3m，无旋转 | 无 | ±3m 恰与 v4 E1 的 3m 外推测量门同量级——官方拿它做训练增强，本项目拿它做测量 | 高 | E1 测量协议沿用 ±3m/±6m，与官方增强口径可比 |
| 颜色一致性 | `use_color_transfer: true`（difix 输出向原图色彩对齐再蒸馏） | nurec-fixer 链路侧有 harmonize 概念（项目外） | 防 difix 把全局色调带跑 | 中 | train-time 蒸馏实现时保留 color transfer |
| 难例采样 | 未发现其他难例加权/采样机制（`reduce.quantile: null` 全 loss） | 无 | 双方都没有 hard-example mining | 低 | — |

---

## 11. 其他显著差异（val 协议 / mask / actor / 杂项）

| 条目 | 官方（resolved） | 本项目 multilayer | 差异含义 | 外推相关性 | 借鉴建议 |
|---|---|---|---|---|---|
| train/val 切分 | `dataset.val_camera_frame_start: 0, val_camera_frame_step: 3`（**每 3 帧 1 val**，33%）；lidar 同步 step 3 | `dataset/ncore.yaml val_frame_interval: 8`（每 8 帧 1 val，12.5%） | 官方 val 密度 2.7×，训练帧更少 | 中（对锚时 PSNR 口径差异源） | E0.4 双向对照必须统一切分（要么都 step 3 要么都 8），否则数字不可比 |
| val 渲染分辨率 | `n_val_image_subsample: 4`（1/4 子采样验证） | `n_val_image_subsample: 1`（全分辨率） | 官方 val PSNR 是低分辨率口径，**直接对比会系统性偏高** | **高（对锚陷阱）** | E0.4 复算官方 metric 时强制 full-res 重渲，或把本项目 eval 也跑一份 1/4 口径 |
| val 指标 | `system.test.metrics`: **cpsnr enabled（per-class PSNR, classes null=全类）**，ssim/lpips **disabled** | render.py 全套（psnr/ssim/lpips + masked + per-class，T6F） | 官方验证主指标即 class-PSNR——与 v3 actor-centric 路线同口径 | 中 | 对锚时以 per-class PSNR 为主轴正合适 |
| 单层渲染验证 | `val_render_selected_nodes: road_only / background_only` | 无 | 见 §8 | 中 | 见 §8 |
| actor track 工程 | `cuboid_tracks_params`: `track_extrapolate: true`（轨迹外插出观测窗）、min_distance/displacement 1m、median 速度≥0.1、`track_min_centroid_rig_dist_m 3.0`、AUTOLABEL 源；`tracks_calib`（direct-tracks-calib：track pose 残差优化，lr 1e-5，500 步冻结→cosine，`fix_first_pose/fix_last_pose: true`）；`track_timestamp_est_precision: null` | tracks 来自 NCore cuboids（同 AUTOLABEL）；pose_adjustment（P1.x）默认 **OFF**（fix_first_last 实现了但 baseline 不开） | 官方**默认开启 track pose 校准且首尾帧锚定**——与 P1.2 结论（fix_first/last 修 drift）一致；`track_extrapolate` 让 actor 在出窗帧仍有 pose | 低-中（actor 侧） | P1.2 复评时引官方默认值（lr 1e-5 + 首尾锚定 + 500 冻结）作为推荐配置依据 |
| ego/静态 cuboid | `generate_static_rigid_cuboid_tracks.enabled: false`（交通灯等静态刚体专层，预留未开） | 无 | 双方未启用 | 低 | — |
| 数据 batch | `datamodule.train_batch_size: 1`、`train_num_workers: 24` | batch 1 全图、`num_workers: 24`（机器相关下调） | 一致 | — | — |
| 显存治理 | `system.collect_garbage_mem_usage: 0.7`（70% 触发 GC，每 10 步查） | 无（PYTORCH_CUDA_ALLOC_CONF 环境变量手段） | 工程卫生项 | 低 | 长训稳定性需要时再加 |
| 产物导出 | `checkpoint.artifact`: ground mesh（ply/usd）+ rig_trajectories + sequence_tracks 默认导出 | `export_ply/usd` 默认关；viz_4d 元数据入 ckpt | 官方训练完直接产出下游（仿真/编辑）资产 | 低（与 E0.6 编辑体验相关） | E0.6 验编辑链时直接取官方 artifact，不必自己导 |
| 随机种子 | `seed: 42` | `seed_initialization: 42` | 一致 | — | — |

---

## 12. 按外推相关性排序的借鉴清单（喂 v4 E3 / E2 / E1）

> 排序依据：该差异**解释官方 lateral 3m/6m 外推优势**的可能性 ×移植性价比。引用各节 yaml key 为准。

### Top-5（高相关，建议按序执行）

1. **road 几何彻底冻结五件套**（§8，外推相关性：高）
   官方路面 = ground-mesh 初始化（400k 薄片、density 0.99、10 轮平滑）+ lr 冻结（positions 1e-6 / scales·rotations·densities 1e-4，仅 albedo 正常）+ `strategy.exclude_layer_ids: ['road']` MCMC 全豁免 + z-scale/平整性软正则（λ1.0）。本项目 road 与 bg 共享全量 lr 且全程参与 MCMC——路面高斯有完整自由度去「迎合训练视角」，这是 lateral 移动后路面糊/重影/洞的第一嫌疑。**最可能单独解释官方路面外推优势的一条**。
   → 行动：E3 新增「road 冻结」实验（MCMC 豁免 + per-layer lr override 两个改动点），先于/并行于 E3.2 DC-only；E3.3 BEV 纹理化是其参数化级终局形态。

2. **road/bg 所有权从 init 即切分**（§1/§5，高）
   官方 `background.ignore_classes_from_layers: ['road']`（bg 初始化剔除 road 语义 LiDAR 点）+ `loss.node_semantic_gaussians`（bce λ1.0 层语义互斥）。本项目 bg init 含 road 点，靠 V3-R2 `bg_road_penalty`（λ0.1 几何带）事后驱赶——正是 P3.3 诊断的 bg 替补率病灶。
   → 行动：①bg init 按 lidar-sseg 剔 road 点（trainer.py init 路径小改）；②E3.1 空气区 penalty 判据升级为 sseg∧几何带，λ 向官方 1.0 量级试探。

3. **train-time difix 蒸馏钩子 = ±3m lateral novel-view 增强**（§10，高）
   官方 `difix.training` 把「渲 ±3m 平移视角 → cosmos-difix 修复 → 当监督蒸馏回模型」做成内建训练特性（start 20000、p 0.5→衰减、color transfer），本 run 关但兄弟配方 sqa_difix_distill 开。它直接在训练时优化我们 E1 要测的那个分布。
   → 行动：E2 主线按此形态实现/复用；先用官方链（E0.2/E0.3）实测 difix-distill 开关在 3m/6m 门指标上的增益上限再决定自研深度。

4. **对锚口径陷阱：官方 val = 每 3 帧 + 1/4 分辨率 + cpsnr**（§11，高—对 E0.4/E1 而非模型本身）
   `val_camera_frame_step: 3`、`n_val_image_subsample: 4`、metrics 只开 cpsnr。直接拿官方日志数字（如传说的 ~36 dB）对本项目 full-res/每 8 帧口径比较会系统性高估官方。
   → 行动：E0.4 双向对照锚必须统一：同切分、同分辨率、同 per-class 口径重算双方。

5. **LiDAR ray 级深度监督 + 200m 远场 + 晚增密**（§2/§4，中-高）
   官方 2048 lidar ray/step 原生渲染监督（λ0.005 全程、far clip 200m）+ `background_lidar` λ0.05 + **add 从 20000 步才开始**（前半程纯靠 2M LiDAR init 收敛）。几何被真实距离测量 pin 死后才允许增密——外推视角下表面位置稳。本项目深度监督是 ≤80m 图像空间深度图、add 500 步即开。
   → 行动：便宜 A/B 三连：`depth_max 80→150`、`lambda_bg_lidar 0.005→0.05`、`strategy.add.start_iteration 500→15000(30k 等比)`；E4 再评 ray 级监督。

### 次优先（中相关）

6. **背景层 Fourier 时变外观（dim 5）+ PPISP 物理 ISP 分解**（§1/§6）：时变光照/ISP 残差吸收进外观模型，几何不背锅。本项目 BilateralGrid 已覆盖 per-camera 静态项；剩余时变项若在 E0.4 对锚中表现为 cc 口径 gap，再评估。
7. **bg init near/far 随机点（100k+100k, far_radius_factor 20）**（§3）：远场/天际线兜底，MCMC 远场 donor 池。
8. **MCMC perturb noise_lr 5000 vs 5e5**（§4）：若公式同源则官方扰动小 100×，收敛后期表面更稳——5k smoke 验证即可定。
9. **训练像素 1/2 子采样 + 40k 步**（§6/§9）：同算力下更多步数、更低高频过拟合压力；A/B downsample 0.5。
10. **`ut_require_all_sigma_points: true`**（§6）+ **sky cubemap 512 + inpaint**（§7）+ **road_only/background_only 单层渲染验证**（§8/§11，E1 诊断手段）。

### 明确不跟进

- LiDAR intensity / raydrop 监督（官方 λ0.0 未启用，§2）；
- dynamic_deformables 层（行人不是当前瓶颈，§1）；
- lpips 训练 loss（双方都没有）；
- 逐帧相机 pose calib（官方该 run enabled false，§6）。

---

### 附：本次对比中发现的本项目文档性问题（顺手记录）

- `LayerSpec.scale_lr_mult: 0.2`（road）在 registry/docstring 中存在，但全代码未消费（无任何 optimizer 读取它）——若按 Top-5 第 1 条做 per-layer lr override，应一并接线或删除该字段，避免误导。
- multilayer yaml 注释自称「7 层递归链扁平版」易与「模型 4 层」混淆，本文档 §1 已注明口径。
