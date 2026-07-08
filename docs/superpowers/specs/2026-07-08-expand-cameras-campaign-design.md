# 扩相机作战方案（v5 Phase C）—— ego-mask 修复前置 + 阶梯设计

- 日期：2026-07-08
- 状态：**设计已获批**（大g 2026-07-08 拍板：「倾向于扩相机路线，ego-mask fix 作为前置项」；设计稿逐节确认通过）
- 定位：v5 Phase C 的**执行级设计 spec**。上承 [`2026-07-03-offtrack-campaign-design.md`](2026-07-03-offtrack-campaign-design.md)（战役 spec）与 2026-07-07 E2.2 蒸馏终止裁决（v4_plan §5 战略决策条）；下接 per-task 实现 plan（writing-plans 产物）。

## 0. 术语澄清

大g 口中的「路线 C」=** v5_plan Phase C 扩相机阶梯** = 战役 spec 的「思路 B 数据轴」。战役 spec 的「思路 C 官方底座」不在本方案范围。本文统一称 **Phase C 扩相机阶梯**。

## 1. 背景与决策链

- **E2.2 蒸馏方向已终止**（2026-07-07 大g 拍板，三红灯裁决）：主线转表示侧 + 数据侧。本方案 = **数据侧主刀**。
- **扩相机收益已有实测**：B4（2026-07-06）单变量证据——同三台侧后相机从「未参训」到「参训」，held-out cc_psnr 9.66 → 18.54（**+8.88 dB**）。
- **ego mask 无效是扩相机前必修的质量地基**（I8，大g 指定前置）：自车结构（卡车车头/后视镜/支架）被当场景重建、抢监督预算、污染 off-track 视角；6 相机全中招，扩到 10+ 台会放大。
- **两个口径拍板**（大g 2026-07-08）：① 阶梯期间保留 **rear_right 为永久 eval-only held-out**（真 GT off-track 锚，牺牲 1/12 数据换每步真 GT 读数；10-cam 配方定型后再决定是否纳入）；② **B5 novel FID 链路纳入 P0 前置**（无悔棋三件套最后一件）。

## 2. 目标与验收主轴

把 b6a9 参训相机 6 → 9–11 台（rear_right 除外），主 KPI = **off-track 质量收敛**。每阶梯步四读数：

| # | 读数 | 工具 | 判据 |
|---|---|---|---|
| 1 | rear_right held-out cc_psnr（真 GT off-track） | B4 协议脚本化（P0.6） | 随覆盖增加持续改善 |
| 2 | novel 档 FID/KID（lateral 1/3/6m） | B5 链路（P0.5） | 不恶化，期待改善 |
| 3 | 已参训相机 per-cam psnr | metrics.json 现成 | **守护线：不退** |
| 4 | automobile class_psnr | class_psnr.py 现成 | 监控 ≥18.5 量级不退 |

⚠️ 口径纪律：ego-mask 生效后所有 masked 指标口径改变，**R3p 旧锚不可比** → P0.4 重立 R4e 锚，阶梯一切对比以 R4e 为基线。

## 3. P0 前置项（ego-mask 修复 + 评估基建）

### P0.1 诊断 ✅（2026-07-08 本设计会话完成）

诊断脚本（scratchpad `diag_egomask.py`，inceptio 实跑）结论——**双层故障**：

1. **「生成了但没接上」**：`aux.egomask.zarr.itar`（主 clip 目录）里 **4/6 相机有真内容**——cross_left 177,748 / cross_right 163,851 / left_wide 150,528 / right_wide 25,138 nonzero 像素（1080×1920，值域 {0,255}，每相机 4 帧）。但训练侧 `datasetNcore` 读的是 **ncore SDK sequence 内嵌 mask**（`camera_sensor.get_mask_images().get("ego")`），b6a9 sequence 未嵌 ego mask → I8 实测全零。**aux itar 与 SDK mask 是两个存储源，从未接通**——与 sseg 当年 SDK 读不了 nre-tools itar 的处境相同（`aux_readers.py` 直读模式即为此而生）。
2. **front_wide / back_rear_wide 真全黑**：itar 里这两台 nonzero=0。但 sseg **egocar(19)** 类在两台均有 4k–13k 像素/帧（mask2former 认得出卡车 ego，egomask 派生环节把小区域滤掉）→ 可从 sseg 派生补齐。
3. 旁证：aux-meta.json 仅有 `cli` 键（无 ego_mask 注册字段）；四个历史 aux 目录内容一致，主 clip 目录为权威。

### P0.2 主修复：EgomaskAuxReader + datasetNcore 接线（代码，TDD）

- 目标：把 itar 里已有的 ego mask 接进训练/eval 的 masked 口径。
- 要改的文件：
  - [`threedgrut/datasets/aux_readers.py`](../../../threedgrut/datasets/aux_readers.py)：新增 `EgomaskAuxReader`（模式同 `SsegAuxReader`：`_open_itar_zarr` 直读 + per-camera 子组缓存）；读出每相机全部帧（现状 4 帧）取**并集**作为静态 mask（保守：任一帧标 ego 即 ego）。
  - [`threedgrut/datasets/datasetNcore.py`](../../../threedgrut/datasets/datasetNcore.py)（L429–440 一带）：SDK `get_mask_images().get("ego")` 缺失或全零时，fallback 到 `discover_aux_path(clip_dir, "egomask")` + `EgomaskAuxReader`；后续 dilate/valid 取反逻辑复用现有路径不动。
- 关键不变量：**无 egomask itar 的 clip（PAI 线）字节等价**——fallback 仅在 SDK mask 缺失且 itar 存在时激活。
- 测试要点（Mac）：合成小 itar 或 stub reader——① fallback 激活条件（SDK 有 mask 时不碰 itar / SDK 空 + itar 有则用 itar）；② 多帧并集语义；③ 无 itar 时行为与现状逐字节一致；④ valid = logical_not(mask) 语义与现有 dilate 链路兼容。公差：mask 为二值，断言精确相等。

### P0.3 补齐 front_wide / back_rear_wide（数据脚本 + 目检）

> ⚠️ **2026-07-08 supersede**：原「sseg egocar 派生自动补齐」路线**被替换**为「大g 手工多边形标注 → 视觉多边形栅格化 + write-once 替换 itar」。详见 [`2026-07-08-visual-polygon-egomask-design.md`](2026-07-08-visual-polygon-egomask-design.md) + [`../plans/2026-07-08-visual-polygon-egomask.md`](../plans/2026-07-08-visual-polygon-egomask.md)。生成源变（视觉多边形，含浏览器 canvas 手工标注器），存储目标（clip 目录唯一 egomask itar，10 台+2 台跳过）与下游 P0.2 接线不变。**实跑 2026-07-08 完成**（commit `40277d2`，10 台 mask 精细贴合，back_rear_wide 5251 px/0.25% 最干净，back_rear_fisheye 车顶弧 20.9% 最大，前后 vignette 由多边形/圆环覆盖）。

- 目标：两台全黑相机补上静态 ego mask。
- 路径：脚本从 sseg itar 逐帧统计 egocar(19) 像素 → **跨帧并集**（自车相对相机静止，真 ego 高度稳定；抽帧目检排除「把邻车误标 egocar」的污染帧）→ 形态学清理（去孤立小连通域 + dilate 缓冲）→ 写回 egomask itar（复用 `merge_lidar_aux.py` 的 itar 写经验，`create_dataset(shape=src.shape)` 通用处理）。
- 兜底：若 sseg egocar 派生结果目检不干净（误检多/漏车头），改**手工 ROI 多边形**（每相机一张，半天内）。
- 验收：12 相机（含未来扩相机步的新相机）egomask 全非空 + 叠图目检自车结构被覆盖；itar 重读回归（P0.2 的 reader 能读）。

### P0.4 R4e 重锚（一次 30k，约 1h 机时）

- R3p 同配方（`ncore_3dgut_mcmc_multilayer_inceptio.yaml`）+ 修复后 ego-mask，30k 重训立 **R4e 锚**。单变量 = 仅 ego-mask。
- 顺带产出 ego-mask on/off 干净 A/B：预期自车像素不再抢监督预算，automobile / road 侧受益；masked psnr 口径变化如实入档（与 R3p 并排标注口径差异，防未来误比）。
- 训练启动 sanity：日志确认 valid 像素数下降幅度 ≈ mask 占比（cross 系 ~8%）。

### P0.5 B5 novel FID 链路移植（render-only 零训练）

- v4 E1.1/E1.4 工具链（novel 6 档含 lateral 3m/6m + `--render-only`/`--novel-fid`）在 b6a9 config 打通；metrics.json 出 `mean_novel_fid_*` 字段。
- 与 B4 真 GT 数字互证入档（v5_plan B5 卡验收原文）。

### P0.6 held-out 评估脚本化

- B4 的 `--dataset-cameras` render-only 流程（底稿 `.superpowers/sdd/b4_summary.md`）封装为每阶梯步一键驱动：输入 ckpt → 输出 rear_right held-out cc_psnr/lpips/FID + train 侧同口径对照。

P0 估时：P0.2+P0.3 约 1 天（Mac 代码 + inceptio 脚本）；P0.4 约 1h 机时；P0.5+P0.6 约 0.75 天。合计 ~2 天。

## 4. 扩相机阶梯（每步：6k proxy 决策 → 达标晋级 30k）

| 步 | 相机集 | 说明 | gate |
|---|---|---|---|
| **C1** | —（纯 Mac 代码） | telew per-camera loss weight 重实现：`trainer.py` 光度项（L1/SSIM）乘 `_camera_loss_weight(camera_id)`、正则项不动；`configs/base_gs.yaml` 加 `loss.camera_loss_weights: {}`（默认空 = 字节等价）；CLI `++loss.camera_loss_weights.<camera_id>=w` 覆盖。Mac 单测：weight=1 恒等 / weight=2 光度翻倍正则不变。**完成定义 = 代码 + 测试 merge 进 main**（2026-06-25 丢码教训） | P0 不阻塞，可并行 |
| **C2** | 6 → 8（+rear_left、+front_standard） | rear_right 留 eval-only；新相机先补 aux（sseg + egomask + lidar camvis，遮挡 bug 已修 0.7s/帧；egomask 缺则走 P0.3 派生路径）；弱相机用 telew 调权 | P0 + C1 |
| **C3** | 8 → 9（+front_tele） | 4cab 证据：无权重 18.04 → telew 加权 26.24 | C1 |
| **C4** | 9 → 11（+2 台 FTheta 鱼眼） | **可弃项**：上游 [issue #238](https://github.com/nv-tlabs/3dgrut/issues/238) 鱼眼尖刺不可控则 9-cam 收口；FTheta 数据路径 PAI 已证 | C2/C3 守护线不破 |
| **收尾** | 配方定型 | ① rear_right 去留：对比其 held-out 曲线全程数据决定是否纳入最终配方；② **60k 容量校准**（吸收 I1 欠训效应：相机数增加 → 每相机有效步数摊薄；阶梯期间统一 30k 保单变量，定型后一次 60k 校准）；③ 锚数字入 v5_plan §4 Done Log + §0.2 KPI 表 | 阶梯完成 |

## 5. 实验纪律（沿战役 spec §5，全文有效）

- **6k proxy 做一切 A/B 决策**，晋级配方才跑 30k；快测可用 5s 数据窗（数字不与全量锚比）。
- 每 run 启动前登记 kill-criterion（run 名 / proxy 步数 / 读数指标 / 砍单阈值 / 砍后动作）；**发射后验证进程存活**（7/3 driver cd 事故、B1 臂 2 OOM 教训）。
- 数字入档必须 rich log × metrics.json 双源交叉（反伪造，R3 长期风险）。
- inceptio 铁律：depth-off + `num_workers=10`；相机数增加数据量上升，**首个 8-cam run 盯 `free -g`**，内存吃紧则 nw 降 8/6。
- 单变量纪律：每阶梯步只动相机集；telew 权重调整算配方内伴随项，但如权重变化引发争议读数，回退单独 A/B。

## 6. 风险登记

| # | 风险 | 缓解 |
|---|---|---|
| 1 | sseg egocar 派生 mask 含误检（邻车/护栏误标） | 跨帧并集前抽帧目检 + 形态学去小连通域；兜底手工 ROI |
| 2 | ego-mask 口径变化使历史锚不可比 | P0.4 R4e 重锚一次性付清；文档并排标注口径 |
| 3 | 扩相机内存/数据管线（62GB + 单卡） | nw=10 起步按 `free -g` 降档；GPU 饿属已知权衡不影响正确性 |
| 4 | 鱼眼尖刺（上游 #238） | C4 放最后、单变量、可弃 |
| 5 | Mac↔inceptio 网络（wifi IP 漂移 + BindAddress） | `Host inceptio` 已指内网 10.8.28.130（无绑定）；`inceptioExt`（10.8.31.113）须随 Mac IP 更新 BindAddress——连不上先查 `ifconfig` 当前 IP 与 `~/.ssh/config` 是否一致 |
| 6 | 新 4090 到货迁移 | onboarding checklist（战役 spec §4）现成，到货即迁为扩相机专属工位 |

## 7. 出界项（本方案不做）

B2 lane 锚（战役门后复排）；D1 poseopt；表示侧 E3 首刀（独立线，另行拍板）；官方底座 spike（战役思路 C）；行人建模；跨 clip 联训。

## 8. 文档关系

- 本 spec = Phase C 执行级设计；任务编号沿 v5_plan（P0 新增，C1–C4 已有）。
- 完成后回填：[`v5_plan.md`](../../../v5_plan.md) §1 看板（P0 系列入 Phase B/C 之间或作 C0 组）+ §4 Done Log；[`v2_architecture.md`](../../../v2_architecture.md) §6 文件清单（EgomaskAuxReader）+ §7 不变量（fallback 字节等价条目）。
- 实现计划：writing-plans 产物落 `docs/superpowers/plans/`（每任务：目标 / 文件 / 签名 / 测试要点 / 验收命令意图；**不贴代码块**）。
