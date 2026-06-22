# E3.2.5 设计依据 — reconstruction-studio 路面几何硬退化的交叉分析

> **日期**：2026-06-22
> **来源**：reconstruction-studio（DriveStudio 系）产物的交叉分析 session（分析对象 = inceptio `/home/inceptio/recon/test_set_30000_gs_Background.ply`）
> **关联**：[`v4_plan.md`](../../../v4_plan.md) § 2.3 Phase E3 — **E3.2.5**（几何侧硬退化路面）
> **一句话**：reconstruction-studio 用「几何硬约束」而非「颜色参数化」把路面横移/旋转稳定性做到了媲美/超过 NuRec；这条路比 E3.3 BEV 纹理轻得多，且能解释 roadoff `road-freeze` 实验为何失败。

---

## 1. 动机：aperture problem 的两条根治路径

v4 § 0.1 已确诊外推退化根因 = **aperture problem**：训练相机全在 ego 轨迹一条线 + 路面掠射角观测 → road 几何在「垂直路面方向（厚度）」欠约束，横移/旋转时该自由度暴露成车道线漂移/退化，并与 road/bg 耦合互为表里。

根治可从两侧入手：
- **颜色参数化侧（E3.3 BEV 纹理）**：road 颜色绑死到平面 grid，颜色不随单个高斯漂。effort=3，需改渲染路径。
- **几何侧（本 E3.2.5）**：把 road 高斯几何自由度删掉——零厚度水平 disk + 法线锁 + 冻结。纯参数/mask 级，不改渲染路径。

reconstruction-studio 是几何侧路径的**活体实证**。

---

## 2. reconstruction-studio 路面机制（代码证据）

路面 = Background 层内 `label==1` 的 ground 子集（非独立层），走主流程 `VanillaGaussians` 的 ground-label 硬退化：

| 机制 | 代码 | 作用 |
|---|---|---|
| z 厚度强制 0 | `models/gaussians/vanilla.py:162` `scaling[ground,2]=0` | 椭球退化为水平 disk（无厚度→无掠射歧义） |
| 初始 z=1e-6 + 单位旋转 | `vanilla.py:107`（`log(1e-6)` + `_quats=[1,0,0,0]`） | disk 平躺、法线竖直朝上 |
| 几何梯度冻结 | `vanilla.py:201` `zero_ground_gradients`（position / z-scale / 面内旋转 grad=0） | 路面不会被优化推离平面 |
| 永不裁剪 | `vanilla.py:482` `culls = culls & (~ground_mask)` | 路面恒在，不被 cull |
| 颜色 = per-gaussian DC | SH degree 0（无 f_rest） | **没有用 BEV 纹理** |
| init = LiDAR 累积体素点 | `datasets/driving_dataset.py` `sample_by_voxel`（多帧累积 + ground 语义 label） | 真实测量点，非规则网格 |

---

## 3. 数据证据（产物 `test_set_30000_gs_Background.ply`，3,386,613 高斯）

按指纹「单位旋转 `[1,0,0,0]` + 最薄轴 < 1mm」筛出 ground 高斯，量化如下：

| 指标 | 数值 | 说明 |
|---|---|---|
| ground 高斯数 | **91,008（2.7%）** | 最薄轴 100% 是世界 z 轴 |
| 最薄轴厚度（中位） | **1.00e-06 m** | `scale_2` raw 1% 分位 = −13.82 = ln(1e-6)，精确命中代码初始化 |
| 水平轴尺寸（中位） | 0.12 m | 12cm 的水平片 |
| 点间距（中位） | **9.7 cm**（90% < 21cm） | 97% 成片，致密 |
| 局部 z 起伏（中位） | **8 mm**（90% < 1.9cm） | 极平整 |
| 覆盖 | 313m × 222m | 整条行驶路线（有转弯/路口） |
| 孤立点（>1m） | 179（0.2%），opacity 0.12 | 非路面散落极少且半透明无害 |

**结论**：路面是一张点距 10cm、起伏 8mm、零厚度、法线竖直的致密水平毯——这就是它横移/旋转稳的物理底子。`致密 disk 阵列 + per-gaussian DC ≈ 一张 10cm 分辨率的隐式 BEV 纹理`（disk 当纹素），与 E3.3 显式 grid 殊途同归。

---

## 4. 与 3dgrut2 现状 road 的对比

| 维度 | 3dgrut2 现状（registry.py / road_init.py） | reconstruction-studio（成功） |
|---|---|---|
| 厚度 | `scale_z_max=0.05`（Z≤5cm 软 clamp，仍可浮动） | **z=1µm 硬压**（真零厚度） |
| 法线 | 未锁（任意旋转，只 clamp scale） | **单位旋转锁死，法线竖直** |
| init | 规则 BEV 网格 + 每格**最近单点** Z 吸附（`road_init.py:97`） | **LiDAR 累积测量点**（体素下采样），起伏 8mm |
| 颜色 | per-gaussian SH（E3.2 拟降 DC） | per-gaussian DC |
| 表示 | 标准 3D 高斯（无 2DGS） | 标准 3D 高斯（无 2DGS；专项 `train_road_surface.py` 才用真 2DGS，与本产物无关） |

---

## 5. roadoff `road-freeze` 失败的解释

roadoff worktree（`claude/hopeful-mirzakhani-56467d`）commit `be6742f` 结论：**「road-freeze 控制变量反直觉结果 — 冻结全面变差，真瓶颈是 init 质量」**。

reconstruction-studio 是反例：**同样冻结却成功**。差异在三个 3dgrut2 当时没补齐的前提：
1. 真零厚度（1µm）而非 5cm clamp —— 5cm 在掠射角下仍有歧义；
2. 强制单位旋转锁法线 —— 否则 disk 会歪；
3. 高质量致密 init（真实 LiDAR 测量点，起伏 8mm）而非规则网格单点吸附。

即 **freeze 思路是对的，roadoff 冻结的是「不够薄 + 会歪 + init 带噪」的 road**。

---

## 6. E3.2.5 方案

| 步 | 改动 | 文件 |
|---|---|---|
| ① init 提质（**必须先做**） | Z 吸附从「最近单点」改局部 KNN 中值 / 平面拟合，降噪贴合 | `threedgrut/layers/road_init.py:97` |
| ② 真薄盘 + 法线锁 | road z-scale 硬压 ~1mm floor（非 5cm clamp）+ 强制单位旋转 | `threedgrut/layers/registry.py:54`、`layer_spec.py` |
| ③ 几何冻结 | position + z-scale + 面内旋转梯度冻结（复用 `scale_lr_mult` override 扩到 position/rotation） | `registry.py` / `layered_model.py` |
| ④ 颜色保持 DC | E3.2 已做 | — |

**与现有 E3 的关系**：E3.1（空气区 penalty）互补（压 road 上方 bg 粒子，解 road/bg 耦合）；E3.3（BEV 纹理）降为后备——若致密 disk + DC 仍不够清晰再上，且可用 disk 的 DC 颜色 bake 成 BEV 纹理初值。

**风险**：
- **3DGUT 数值**：零厚度协方差（一个 eigenvalue≈0）在 UT sigma 点下可能不稳——用 **1mm floor，不用 1µm**（比 5cm 薄一个量级，够杀掉掠射歧义又数值安全）。
- **顺序**：freeze 前提是 ① init 先达标，否则重蹈 roadoff「冻结变差」覆辙。

---

## 7. 验收

同 E3 口径（E1 指标 + 守护线）。专项 A/B：**①+②③ on vs off，比 lane grad_corr / band_lpips @ lateral_3m/6m**；守护线 cc ≥ 24.7、interpolated grad_corr/class_psnr 不退。**先 spike 小范围验证 3DGUT 下零厚度 disk 训练稳定，再全量。**
