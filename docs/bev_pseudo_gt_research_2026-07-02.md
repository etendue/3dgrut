# 调研报告：环视相机合成 BEV 作「虚拟俯视相机 GT」监督 3DGUT 训练的可行性

- **日期**: 2026-07-02
- **研究问题**: 用多个环视相机（surround cameras）通过 IPM 拼接合成 BEV 鸟瞰图像，把它当作一个「虚拟俯视相机」的 ground truth 图片加入训练 photometric 监督，**目的是提高 road 层的几何一致性**（z 向噪声/不平整、跨视角外推路面破碎）。方案是否可行、是否有效？相机选择对 BEV 质量（从而对监督效果）的影响如何？
- **调研方法**: 三路证据合成——① 本仓库工程摸底（相机模型/监督入口/road 层/BEV 工具链）；② 聚焦调研（14 次搜索 + 深读 8 篇：ParkGaussian / RoGS / RoMe / EMIE-MAP / IPM 误差分析等）；③ deep-research 多 agent 验证（60 agents，关键 claim 经 3 票对抗验证，本文标注投票结果）。

---

## 0. TL;DR

**工程上可行，但作为「提高 road 几何一致性」的手段，证据不支持直接采用。**

| 维度 | 结论 |
|---|---|
| 工程可行性 | ✅ 仓库基础设施大半就位（BEVStitcher / layered mask loss / novel_view 框架 / LiDAR 高度场）；正交相机在 splatting 管线改动极小，3DGUT UT 路径需自行验证 |
| 几何有效性 | ❌ IPM BEV 不含新视差，本质是「photometric 化的平面先验」——压 z 噪声有正则收益，但把路面拉向含 bias 的假设平面，对真实高程零信息增益 |
| 领域实践 | ❌ RoGS / RoMe / EMIE-MAP / BEV-GS / RoadBEV **无一**把 IPM BEV 当训练监督；BEV 一律只作输出/评估域 |
| 最接近先例 | ⚠️ ParkGaussian（环视鱼眼 3DGS + 可微 IPM + BEV L1）实测**裸 IPM L1 直接降质**（PSNR 24.94 → slot-aware 加权后 30.09） |
| 相机选择影响 | ✅ 用户直觉被文献定量证实：径向地面采样分辨率按 d² 退化，有效纹理半径仅 ~10–15 m；侧向广角最优、tele 无 BEV 价值；头号误差源是外参/车身 pitch（错位按 d²·δθ/h 放大） |
| 建议 | BEV 当**评估域**；几何一致性走 LiDAR 高度场 + 轨迹先验 + 平滑/erank/平面化正则（证据链硬）。若坚持 BEV 监督，只做「近场 + LiDAR 高度场 warp + mask/降权 + 低权重正则」退化版，并自跑 A/B（文献空白） |

---

## 1. 机制分析：BEV 伪 GT 到底约束了什么几何

同一时刻环视图 IPM 到平面的 BEV，是**现有观测在平面假设下的重投影，不含新视差**。把它当虚拟俯视相机 GT 时：

- 正对下视（nadir）时，路面高斯的 Δz 误差几乎不产生像素位移（∂u/∂z ≈ 0），photometric loss 对 z 的**直接梯度 ≈ 0**；
- z 的约束是**间接**的：偏离假设面 Δz 的高斯，其渲染纹理与 IPM GT 的纹理位置错开 ≈ Δz·d/h（d = 到源相机地面距离，h = 相机高），loss 通过纹理错位把高斯**推回 IPM 假设的那个平面**；
- 因此它等价于一个「photometric 化的平面先验」+ 跨相机颜色一致性约束：
  - ✅ 对「z 向噪声 / 路面破碎」有正则化收益（压平噪声）；
  - ❌ 拉向的是含 bias 的假设平面（外参误差、车身 pitch、路拱横坡全部烙进几何），**与「提高几何一致性」的目标在有坡度/超高路段恰好相反**；对真实高程无信息增益。

旁证：driving 场景伪视角监督有效的工作（DriveDreamer4D、SGD、DriveX、FreeVS）全部靠**注入外部新信息**（世界模型/扩散先验/LiDAR）；几何 warp 的 pseudo-image 在 FreeVS 里只作为生成模型的输入而非直接 GT。IPM BEV 属「无新信息」类，文献中没有它单独改善几何的正面证据。

## 2. 相机选择 → BEV 质量（定量）

### 2.1 地面采样分辨率随距离平方退化

遥感/摄影测量标准结论：斜视下径向 GSD 乘 sec² 倾角因子。换成车载几何（相机高 h、地面距离 d、焦距 f px、像元 p）：

- **径向 footprint ≈ p·d²/(f·h)**（平方增长）；横向 footprint ≈ p·d/f（线性增长）。
- 按本项目典型配置推算（1920px/120° 广角 f≈554px、h≈2m，推算值非文献原文）：

| 距离 d | 径向地面分辨率 |
|---|---|
| 5 m | ~2.3 cm/px |
| 10 m | ~9 cm/px |
| 20 m | ~36 cm/px |
| 30 m | ~81 cm/px |

- **IPM 纹理有效半径 ~10–15 m**；HD-map 级 IPM 工作明确限制反投影不外推、>20 m 是公认劣化区；量产 AVM 只做车周近场。现有 `BEVStitcher` 的 ±30 m 范围外圈大半不可用。

### 2.2 相机贡献排序

- **cross_left / cross_right 侧向广角**：车侧 0–10 m 掠射角最陡、footprint 最小，BEV 近场质量最好；
- **front_wide**：车前 5–15 m 有用；rear 同理；
- **tele 类（30fov）**：掠射角极浅，长焦 f 只线性补偿、敌不过 d² 退化，**对 BEV 纹理基本无贡献**（只适合远处车道线检测类任务）。
- 现有 [bev_stitcher.py](../threedgrut_playground/utils/bev_stitcher.py) 按**方位角楔形**分配相机（cell 给方位角最近的相机、无 blending）——若用于监督应改为**按每 cell 掠射角/地面采样分辨率选最优相机** + 接缝 blending。

### 2.3 误差源排序（近场 BEV，按量级）

1. **外参 / 车身 pitch（头号）**：由 x = h/tanθ 微分，地面位置误差 ≈ d²·δθ/h——d=20m、h=2m 时**仅 0.5° pitch 误差 → ~1.7 m 纹理错位**（随距离平方放大）。载荷/悬架变化使 pitch 与相机高度时变，文献用在线外参校正（EKF + 车道线观测）应对；
2. **时间不同步**：30 m/s 下 10 ms = 0.3 m 位移即可见拼接伪影；硬件同步 rig 可忽略；
3. **Rolling shutter**：与自运动耦合，近场低速下量级小；
4. **跨相机 AE/AWB**：不伤几何但直接污染 photometric 监督（deep-research 验证 6-0：光度接缝普遍存在；「接缝不可缓解」的强 claim 被 0-3 否决——photometric alignment 可显著减轻）。路面重建文献的正解是**逐相机独立 color decoder**（EMIE-MAP）；
5. **平面假设本身**：路拱（典型 1–2% 横坡）与坡度是系统性违背；RoMe 重庆坡道场景高程 -0.8 m → 7 m+（验证 3-0）。

## 3. 文献证据：领域实践与最接近先例

### 3.1 路面重建专门工作全部不用 IPM BEV 当监督（验证 9-0）

| 工作 | 几何监督 | photometric 监督域 | BEV 的角色 |
|---|---|---|---|
| **RoGS** (arXiv 2405.14342) | 车辆轨迹高程初始化（路面在轮下）+ 可选 LiDAR 高程 + 相邻高程平滑正则 | 原始 perspective 环视图 L1 + Mask2Former 语义 CE | 仅评估：BEV 基准用**拼接 LiDAR 点云渲染**构建（全文零次 IPM）；nuScenes elevation RMSE 0.154 m |
| **RoMe** (arXiv 2306.11368) | z = MLP(PE(x,y)) 显式高程 + 外参优化模块 | 多帧 perspective photometric + semantic | 仅可视化；明确批评 IPM 类方法忽略高程 |
| **EMIE-MAP** (arXiv 2403.11789) | LiDAR 地面点高程监督 + 轨迹初始化 | perspective + **逐相机独立 color decoder** | 仅输出；city street PSNR 26.75 vs RoMe 17.31 |
| **BEV-GS** (arXiv 2504.13207) / **RoadBEV** (arXiv 2404.06605) | LiDAR 累积高程图作 GT（smoothed-L1） | — | BEV 是**几何 GT 域**（LiDAR 来的），不是 photometric 伪 GT |

多帧 vs 单帧：主流是「全轨迹多帧 perspective photometric 优化一个显式路面模型」——等价于多帧累积但绕开 IPM 平面假设；动态物体一律语义 mask 剔除，pose 漂移靠外参/位姿联合优化。

### 3.2 最接近的先例：ParkGaussian（arXiv 2601.01386）

环视鱼眼 3DGS + 可微 IPM 模块 + BEV 空间 L1 + 冻结泊车位检测器蒸馏。消融实测：**不加 slot-aware 加权的裸 IPM L1 → PSNR 24.94；加权后 30.09**——作者明确指出多视角投影在视野边界注入冲突噪声。且其收益体现在 BEV 感知任务，**没有把路面几何单独消融**。

### 3.3 IPM 固有误差（验证 6-0，Bruls et al. IEEE IV 2019 等）

- 高于路面的物体（车辆、骑行者、安全岛）在 IPM 下**严重形变**（径向拉花），该区域 BEV 像素系统性错误；
- 远距离**非均匀映射产生不自然模糊与拉伸**；
- 姿态微小误差与坡度 → 远横向距离**接缝错位**（几何接缝）+ 各相机独立 AE/AWB → **可见光度边界**（光度接缝）。

这些区域若不 mask/降权，photometric loss 会把错误纹理与错误几何反灌进高斯，并与原始环视监督直接冲突（灰色平均、假纹理梯度、floater）。

## 4. 本仓库工程现状与缺口

**已就位**：

- [threedgrut_playground/utils/bev_stitcher.py](../threedgrut_playground/utils/bev_stitcher.py)：5-cam IPM 拼接（`z=ego_z` 单平面、方位角楔形分配、coverage mask 已有、FTheta 畸变与训练路径一致）——注意它是 V3-VIZ 诊断工具，直接当 GT 生成器有 §2/§3 所列全部系统误差；
- [threedgrut/layers/road_init.py](../threedgrut/layers/road_init.py)：road 层 BEV 网格 + LiDAR-Z KNN 高度场（5 cm，处理坡道/超高）——**仓库里已有比平面假设好得多的路面高度场**；
- [threedgrut/model/layered_loss.py](../threedgrut/model/layered_loss.py)：per-pixel mask 加权 photometric loss（sky/road/dyn/valid），BEV 拼接盲区/动态挖洞可走 `valid_pixel_mask`；
- [threedgrut/utils/novel_view.py](../threedgrut/utils/novel_view.py) + render.py novel-view 框架：可扩展 BEV 俯视评估模式；
- effective-rank flat 正则、road perturb_mask D1（防 MCMC z 噪声）已在训练管线中。

**缺口**：

1. 渲染管线无正交投影 raygen（仅 pinhole/fisheye）。EWA splatting 下正交相机改动极小（RoGS：投影 Jacobian J 设单位阵即可，验证 3-0；Tortho-Gaussian 独立演示），**但 3DGUT 的 UT 投影与 3DGRT ray-tracing 路径文献未覆盖**——UT 只消费 rays_ori/rays_dir，平行射线原理上可行，数值稳定性需自行冒烟验证；备选为高空长焦 pinhole 近似；
2. `Batch.rgb_gt_bev` 字段 + BEV loss term 未实现（若做监督版）。

最小可行版（监督版）估计 1–2 周；仅评估版（正交/高空俯视渲染 + LiDAR-BEV 基准）约 1 周。

## 5. 建议路线

### 5.1 首选（证据链硬）：BEV 当评估域，不当监督域

1. **评估侧**：加正交（或高空 pinhole）俯视 novel-view 渲染模式；用累积 road LiDAR 点 + 投影颜色构建 BEV 评估基准（RoGS 式），量化 elevation RMSE / road masked PSNR / lane grad_corr；
2. **几何一致性本身**走已被验证的路：LiDAR 高度场 init（已有）+ effective-rank / 平面化正则（已有，可加强）+ **高程平滑正则**（RoGS 式相邻高斯 z 平滑，未做）+ **轨迹高程先验**（「路面在轮下」，零成本，未用）。

### 5.2 若坚持 BEV 监督：退化安全版设计要点

- warp 目标从 `z=ego_z` 平面换成 **road_init 的 LiDAR 高度场**——伪 GT 的几何来源改为 LiDAR，性质从「平面先验」变为「LiDAR 几何的 photometric 化」（这是方案能否为正收益的分水岭）；
- 只取近场 ≲10–15 m；每 cell 按掠射角/地面采样分辨率选相机，不用纯方位角；
- dyn mask 挖动态物体 + coverage/接缝 down-weight + 距离衰减权重；
- 逐相机曝光解耦（复用现有 exposure 机制，EMIE-MAP 式）；
- **低权重、当正则不当 GT**（ParkGaussian 教训：裸 L1 降质）。

### 5.3 验证实验（文献空白，需自跑 A/B）

inceptio 5k smoke（depth-off 配方，`num_workers=10`）三臂对照：

| 臂 | 配置 | 看什么 |
|---|---|---|
| A（baseline） | 现行 multilayer | road masked PSNR、lane grad_corr、BEV-LiDAR elevation RMSE（新增指标） |
| B（正则路线） | + 高程平滑正则 + 轨迹先验 | 同上，预期几何指标改善且无 photometric 代价 |
| C（BEV 弱监督） | + §5.2 退化版 BEV 正则（低权重） | 同上 + 检查 floater/接缝伪影是否反灌 |

判定：若 C 相对 B 无几何指标增益（elevation RMSE / grad_corr），则 BEV 伪 GT 路线关闭，保留 BEV 仅作评估域。

## 6. Open questions

1. 「BEV 伪 GT vs 无 BEV」的直接 A/B 无已发表工作——§5.3 实验是一手证据机会；
2. mask + 距离衰减 + 低权重下，残余误差能否降到「弱监督可用」水平（唯一可能的挽救路径，无发表证据）；
3. 3DGUT UT 投影对正交/平行射线相机的数值稳定性（RoGS 的 J=I 结论不能直接迁移）；
4. 本次 deep-research 为精简配置（10 claims 上限），StreetSurf、SparseGS/DNGaussian 等 pseudo-view 支线未进入对抗核实清单（聚焦调研已单独覆盖其主结论）。

## 7. 主要引用

**IPM/BEV 误差**：[Bruls et al., IEEE IV 2019](https://arxiv.org/pdf/1812.00913) · [IPM 综述 (Emergent Mind)](https://www.emergentmind.com/topics/inverse-perspective-mapping-ipm) · [IPM Correction, MDPI EngProc 2024](https://doi.org/10.3390/engproc2024079067) · [Adaptive IPM](https://www.sciencedirect.com/science/article/abs/pii/S1051200423000398) · [Photometric alignment for surround view, ICIP 2014](https://ieeexplore.ieee.org/document/7025366/) · [Online Extrinsic Calib for Temporally Consistent IPM](https://deepai.org/publication/online-extrinsic-camera-calibration-for-temporally-consistent-ipm-using-lane-boundary-observations-with-a-lane-width-prior) · [GSD (Wikipedia)](https://en.wikipedia.org/wiki/Ground_sample_distance)

**路面重建**：[RoGS](https://arxiv.org/abs/2405.14342) · [RoMe](https://arxiv.org/abs/2306.11368) · [EMIE-MAP](https://arxiv.org/html/2403.11789v1) · [BEV-GS](https://arxiv.org/html/2504.13207v1) · [RoadBEV](https://arxiv.org/abs/2404.06605) · [StreetSurfGS](https://arxiv.org/html/2410.04354)

**伪视角监督 / BEV 监督先例**：[ParkGaussian](https://arxiv.org/html/2601.01386) · [DriveDreamer4D](https://arxiv.org/abs/2410.13571) · [DriveX](https://arxiv.org/html/2412.01717) · [SGD](https://arxiv.org/abs/2403.20079) · [FreeVS](https://arxiv.org/html/2410.18079)

**正则替代 / 正交渲染**：[Effective Rank Regularization](https://arxiv.org/html/2406.11672v3) · [DN-Splatter, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Turkulainen_DN-Splatter_Depth_and_Normal_Priors_for_Gaussian_Splatting_and_Meshing_WACV_2025_paper.pdf) · [AutoSplat](https://www.researchgate.net/publication/395208562_AutoSplat_Constrained_Gaussian_Splatting_for_Autonomous_Driving_Scene_Reconstruction) · [Tortho-Gaussian](https://arxiv.org/abs/2411.19594)
