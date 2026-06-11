# 3DGRUT v4 — Extrapolation 性能 · 可执行计划

> **本文档定位**：v4 **主线 plan**。v4 唯一主题 = **extrapolation（外推视角）性能**：让 v3 已做实的 interpolated 重建质量（车辆 / 道路车道线）在**偏离录制轨迹的视角**（横移 3m/6m、变 yaw、变高度、held-out 相机）下仍然可用。
> **方法论（2026-06-10 NuRec 调研定稿）**：对标 NVIDIA NuRec 的**双层架构**——表示侧忠实重建 + 生成修复器补洞（渐进蒸馏 + 在线增强）。**首要任务（大g 指定）= 先用 NuRec 官方工具链复现场景立锚，再参考其思路在 3dgrut2 仓库实现性能提升。**
> **决策依据（decision of record）**：
> - NuRec 工具链调研 [`~/repo/report/nvidia-nurec-extrapolation-analysis.md`](../report/nvidia-nurec-extrapolation-analysis.md)（修复器三代演进 / DiFix3D+ 量化证据 / held-out 协议 / license）
> - 领域综述 [`~/repo/report/3d-4d-state-of-the-art-2025-2026.md`](../report/3d-4d-state-of-the-art-2025-2026.md)（外推 = 生成先验主战场；NTA-IoU/FID/KID 评测共识）
> - 2026-06-11 外推诊断（aperture problem 根因 + 测量双盲区）：[`v3_plan_revised.md`](v3_plan_revised.md) § 2.3 / § 6 Done Log
> **执行约定**：沿用 [`CLAUDE.md`](CLAUDE.md)（inceptio 首选 / depth-off+nw=10 铁律 / 文档同步纪律 / Mermaid 全角括号）；具体任务开工时按 superpowers 流程在 `docs/superpowers/plans/` 起 TDD 执行 plan。
> **官方工具就绪**：NVIDIA nurec-skills 已装入本环境（`nre` / `ncore` / `asset-harvester` / `nurec-fixer` / `physical-ai-datasets` / `nurec-index`），E0 直接调用。

---

## 0. 目标与 KPI

### 0.1 v4 核心方向（extrapolation，2026-06-11 定稿）

> **真实成功指标 = 外推视角下道路/车道线 + 车辆的可用质量**：lateral 3m/6m 下车道线不糊、车辆不散、路面无悬浮鬼影；并有一套能证明它的测量体系。

三条事实链支撑这个方向：

1. **根因已确诊**（2026-06-11 诊断，v3 § 2.3）：训练相机全在 ego 轨迹一条线上、路面掠射角观测 → **aperture problem 欠约束**——road/bg 耦合（bg 悬浮粒子"帮忙"渲染）与 3m/6m 外推退化是**同根因两面**。训练视角内无解，必须引入新约束或新监督。
2. **领域共识**（SOTA 综述 Key Finding #4）：外推是 2025–2026 最大未解问题，**生成先验是共识解法**；评测用 NTA-IoU / NTL-IoU / FID / KID，不用无 GT 的 PSNR/LPIPS。
3. **NuRec 实证**（调研报告 § 2）：NuRec 外推可用性的支柱是**修复器链**（DiFix3D+ → Fixer → DiffusionHarmonizer）——驾驶外推实测 **+1.8 dB PSNR / FID −20%**；其 ablation 证明**纯后处理只提感知（FID 134.65→49.87），蒸馏回 3D 才提几何（PSNR +1.03），叠加最优**。

**v3 → v4 的关系**：v3 把 interpolated per-class 质量做到位（车 class_psnr 25.07 / lane grad_corr 0.744 / cc 26.06）；v4 把这些质量**带出训练轨迹**。v3 遗留的 Phase 2（行人）与 P3.2 仍归 v3 主线，由大g按资源另行排期；P3.3–P3.5 因属外推主题**移交 v4**（见 § 2.5）。

### 0.2 KPI — 外推指标为主（绝对数 E0/E1 回填）

> ⚠️ 沿 v3 纪律：**不设虚构绝对阶梯**。新 KPI = 每轴相对 E0（NuRec 锚）/ E1（自有锚）的 gap 闭合；绝对目标数锚点测完才定。下表「量级参考」来自他人论文、不同数据/协议，**只作方向感，不可直接对标**。

| 轴（主 KPI） | 现状 | 测量工具 | v4 目标 / 量级参考 |
|---|---|---|---|
| **车道线外推** lane grad_corr / band_lpips @ lateral_3m/6m | **未测（测量盲区）**：现 `NOVEL_VIEW_MODES` 仅 lateral_1m/2m + yaw_5/10deg | E1.1（=P3.3）：[`novel_view.py`](threedgrut/utils/novel_view.py) 扩档 + lane 区域 novel 指标 | E1 立锚 → E2/E3 闭合；参考 DiFix3D+ FID −20% 量级 |
| **车辆外推** NTA-IoU（原轨迹 + 3m/6m 档） | 未测（plan 已备未执行） | E1.2：[`2026-06-10-nta-iou-eval-metric.md`](docs/superpowers/plans/2026-06-10-nta-iou-eval-metric.md) | E1 立锚 → 闭合；参考 ReconDreamer 系 3m 0.498→0.572 |
| **真 GT 外推** held-out camera per-class | 未测（协议不存在） | E1.3：留出侧相机做真 GT 外推集（DiFix3D+ RDS 协议反用） | E1 立锚 → 闭合 |
| **感知质量** FID/KID @ 3m/6m | 未测 | E1.4：FID/KID 接入 render eval | 立锚 → 不升 |
| **NuRec 同 clip 对照锚** | 传闻 ~36 dB 未实测（v3_plan.md L33 理论对标值） | **E0.4**：官方 nre 配方在自有 clip 训练 + 双向全指标对照 | 把"与 NuRec 差距"从估计变实测，量化 v4 天花板 |
| 守护线（interpolated 不退化） | class_psnr 25.07 / cc 26.06 / lane grad_corr 0.744 / novel(≤2m) 0.5962 | 现成（v3 全套） | cc ≥ 24.7（沿 v3）；grad_corr / class_psnr 不退 |

### 0.3 v4 不做（明确出界）

- **行人**（SMPL/rigid 垫脚石）—— 留 v3 Phase 2，v4 外推评测亦不含 person 轴（无模型可推）
- **编辑/仿真完整产品化**（删/插 actor 不留痕的质量达标）—— 留 v5；但 v4 **纳入两个窄域 spike**（E0.6 官方链编辑体验 + E2.5 3dgrut2 侧插入协调，2026-06-11 大g 拍板）：asset-harvester 与 Harmonizer 同为 NuRec 官方「优化与资产」两大件、Harmonizer 训练管道③（asset re-insertion，正是用 3DGUT+Asset Harvester 构造）④（PBR 阴影）专为插入协调而设，且插入物在新位姿/光照下渲染本质是**内容外推**——与 v4 主题同根
- closed-loop 仿真器集成（CARLA / AlpaSim gRPC 接入）、跨 clip 联训、feed-forward、relighting、LiDAR intensity/ray-drop 内核改造（E4-A3 backlog）
- NuRec 专有数据复现（1M–1B 张 post-train 数据不可得；开源 Harmonizer 权重已含其收益）

### 0.4 v4 起点 baseline（继承 v3，不重训）

| 维度 | 数值 | 来源 |
|---|---:|---|
| 车辆 class_psnr（interpolated） | **25.07** | v3 P1.2 fix（poseopt+boundary+prior，inceptio 30k） |
| cc_psnr_masked | **26.06** | 同上 |
| lane grad_corr（interpolated） | **0.744** | v3 P3.1-A（门锚 0.693 → +0.051；代码在 PR #24） |
| novel LPIPS avg（lateral≤2m+yaw 4 档） | 0.5962 | P3.1-A 实测（**仅 ≤2m 有效**，3m/6m 盲区） |
| baseline 对照配方 | inceptio depth-off + nw=10，从头训 | CLAUDE.md 铁律；v3 R7（resume 不可靠） |
| NuRec 对照锚 | **待 E0.4 回填** | — |

---

## 1. 项目看板（Kanban）

> 状态：⬜ Todo · 🟡 In Progress · 🔵 Review · ✅ Done · ⏸ 降级 · ⏭ Skip

### 1.1 顶层看板（Mermaid Kanban）

```mermaid
%%{init: {'theme':'base'}}%%
kanban
    Backlog
        [E0.4 同 clip 双向对照锚（NuRec vs multilayer 全指标）]
        [E0.6 官方链 actor 插入取代体验（nre 编辑 + asset-harvester + Harmonizer 协调）]
        [E1.2 NTA-IoU 接入（plan 已备）+ 外推档联动]
        [E1.3 held-out camera 真 GT 外推协议]
        [E1.4 FID KID 感知指标接入]
        [E2.1 Harmonizer 升级集成 + 域差 spike]
        [E2.2 渐进外推蒸馏（核心移植）]
        [E2.3 actor 弱观测面修复蒸馏]
        [E2.5 编辑协调 spike（AH 注入 + Harmonizer 协调 + NTA-IoU FID 验收）]
        [E3.3 BEV 纹理平面化（gate E1 锚）]
        [E4.1 LiDAR 点云推理（A0 gate，可选）]

    "In Progress"

    "Blocked"
        [E1.1 ＝v3 P3.3 移交：3m6m 档 + lane 区域 novel 指标 + 三方 ckpt 立锚]
        [E3.1 ＝v3 P3.4 移交：空气区 penalty（gate E1.1 + R9 PR24）]
        [E3.2 ＝v3 P3.5 移交：road SH DC-only freeze（gate E1.1 + R9 PR24）]

    "Done"
        [E0.1 容器冒烟 ✅ 2026-06-11：nre 26.4.146，无 key 全链通]
        [E0.2 USDZ 渲染+修复链 ✅：FID 锚 7.4/57/92，Harmonizer 两档跑通，目视显著]
        [E0.3 自有 clip 官方训练 ✅：40k 步 2h07m，test psnr 30.30（官方口径）]
        [E0.5 配方 diff 清单 ✅：docs/superpowers/specs/ 入档，Top-5 借鉴点]
        [继承: v3 P1.2 track-pose（class 25.07 cc 26.06）]
        [继承: v3 P3.0 lane 门 + P3.1-A lane loss（grad_corr 0.744，代码 PR #24）]
        [继承: 2026-06-11 外推诊断（aperture problem + 测量双盲区）]
        [继承: NuRec 调研 + SOTA 综述两份报告]
        [继承: T15.2 difix.py（Fixer 一代集成，E2.1 升级对象）]
```

> 注：E1.1/E3.1/E3.2 初始在 Blocked 列——E1.1 等 E0.1 环境完成后即解锁（纯 eval 可与 E0 并行）；E3.1/E3.2 gate = E1.1 锚点 + PR #24 去留（R9）。

### 1.2 任务级看板（E*.* 编号）

| ID | Phase | 主题 | 继承来源 | 估时(d) | 状态 | gate / 备注 |
|---|---|---|---|---:|:---:|---|
| **E0.1** ★ | E0 | **容器环境 gate**：镜像拉取 ✅ + 运行时冒烟 ✅（2026-06-11 全部完成）。实测：`nre-ga:latest` = **26.4.146-c63f08a4**（2026-05-28 build）；validate_setup.py 仅 2 预期 FAIL（24GB 边界 + key 空）；容器内 nvidia-smi/CLI 全子命令面正常；**无 NGC key 跑通 train/render 全链**（R-v4.1 答案：key 仅 difix-distill 下载 `cosmos_3dgut.pt` 时需要） | 新 | 0.5 | ✅ | 无 key 训练+渲染实测通过；唯一例外＝官方 train-time difix 蒸馏权重在 NGC |
| **E0.2** ★ | E0 | **官方 USDZ 场景渲染 + 修复链体验** ✅（2026-06-11）：场景 0fd06bc3（1.92GB 4K）+ 048b974e 已下；`nre render` 三档各 595 帧（gt / lat3m / lat6m，rig offset 法，17.8ms/帧 @4090 1080p）；Harmonizer 时间模式修复两档跑通。**FID（vs 真实参考帧分布）：原轨迹渲染 7.37（忠实度极高）/ lat3m 57.3→修复后 65.6（↑）/ lat6m 91.8→86.6（↓5.2）**；目视修复显著（锥桶修直 / 涂抹消除 / 黄线连续）但 FID 几乎不动——**判别产出：官方表示侧伪影轻，FID 大头＝视角内容差，修复链 FID 收益∝伪影占比**（论文 134→50 是重伪影场景；3dgrut2 伪影重于官方→E2 收益预期更大）；FID 单指标评修复会误判（E1.4 须配区域化指标，R-v4.5 实证） | 新 | 1 | ✅ | 帧档：inceptio `~/work/nurec_e0/renders/0fd06bc3/` + `~/repo/harmonizer/input_frames_cosmos_temporal_lat{3,6}m/`；真帧 `~/work/nurec_e0/real_frames/`；Harmonizer 权重全 HF 化（token 须开 gated 权限 + Cosmos-Predict2 模型页接受 license——两步均需 HF 网页手动操作，已留档） |
| **E0.3** ★ | E0 | **自有 clip 官方配方训练复现** ✅（2026-06-11）：clip 9ae151dc 用 Hyperion-8.1 `car2sim_6cam`+pai overlay（PAI 配方，非 Waymo 3dgut_dynamic）40k 步一次训完——**4090 24GB 无降配**（峰值 16GB / 稳态 ~9GB，2.62M gaussians），2h07m（7.45 it/s）。**官方口径锚：test/psnr 30.30 / cpsnr road 38.27 · car 34.59 · person 32.65 / chamfer 0.295**；产物 `artifacts/last.usdz`（1.1GB）+ 20 类 cpsnr 全套。兼容性结论：官方分片全兼容 nre 26.4，唯一不兼容＝v3 自产 lane aux（缺 consolidated 元数据 + sequence 标识不符）→ 移出即过（R-v4.3 实际未命中） | 新 | 1.5 | ✅ | ⚠️ 官方 val 口径＝每 3 帧+1/4 分辨率+cpsnr，**不可直接对比 multilayer 数字**（E0.4 统一口径重算） |
| **E0.4** ★ | E0 | **同 clip 双向对照锚**：NuRec ckpt 与 multilayer baseline 互渲 — interpolated（PSNR/LPIPS/per-class）+ 外推（lateral 3m/6m lane/NTA-IoU/FID，用 E1 工具）双向跑全指标 → **v4 gap 表首行**（量化"差距在哪一层"：表示/配方/修复器） | 新 | 1 | ⬜ | gate=E0.3 + E1.1/E1.2 工具就绪 |
| **E0.5** | E0 | **配方 diff 清单** ✅（2026-06-11）：官方 resolved 全量（8438 行 parsed.yaml）vs [`ncore_3dgut_mcmc_multilayer.yaml`](configs/apps/ncore_3dgut_mcmc_multilayer.yaml) 11 维度逐项 diff → [`2026-06-11-e05-nurec-vs-multilayer-recipe-diff.md`](docs/superpowers/specs/2026-06-11-e05-nurec-vs-multilayer-recipe-diff.md)。**Top-5**：① road 几何冻结五件套（ground-mesh init + lr 1e-6 冻结 + MCMC 豁免 + z-scale/平整正则，喂 E3.1-E3.3）② road/bg 所有权 init 切分（bg init 剔 road 类点）③ 官方 train-time difix 蒸馏钩子原生＝±3m lateral 增强（本 run 关，sqa_difix_distill 开，喂 E2.2）④ 对锚口径陷阱（官方 val 每 3 帧+1/4 res+cpsnr）⑤ LiDAR ray 级监督 2048 ray/step + 200m 远场 | 新 | 0.5 | ✅ | 顺手发现：`LayerSpec.scale_lr_mult` 死配置（已 spawn 后台任务）；官方 `noise_lr 5000` vs 本项目 5e5 待验证 |
| **E0.6** | E0 | **官方链 actor 插入/取代体验**：在 E0.2 USDZ 场景或 E0.3 自有 clip 重建上走官方编辑工作流——nre actor 编辑（gRPC/CLI 删/移/替）+ `asset-harvester` 资产插入（项目已收割 3 车+3 人直接可用）+ Harmonizer 协调（训练管道③④正为此设）→ 渲染对照 + **官方编辑能力/限制清单**（删除后路面是否出洞 / 插入物阴影来源 / 协调前后 FID） | 新（大g 2026-06-11 提议纳入） | 1 | ⬜ | 直接回答「官方如何解决 P1.4 撞到的 spiky/光照失配/悬浮」；清单喂 E2.5 |
| **E1.1** ★ | E1 | **外推测量门扩展** = **v3 P3.3 移交**：lateral_3m/6m 新档（4 档 avg 口径不变保历史可比）+ lane 区域 novel 指标（路面平面诱导 warp 重投影）+ 三方 ckpt（baseline/B3/aniso20）立锚，顺答 B3 细长高斯外推张力 | v3 P3.3（2026-06-11 立项原文 [`v3_plan_revised.md`](v3_plan_revised.md) §1.2） | 1.5 | ⬜ | 纯 eval 无训练；**E1 之门** |
| **E1.2** ★ | E1 | **NTA-IoU 接入**：按 [`2026-06-10-nta-iou-eval-metric.md`](docs/superpowers/plans/2026-06-10-nta-iou-eval-metric.md) 执行（Task 0–5 全 TDD 已写好）+ **增量**：novel 外推档下也跑 NTA-IoU（渲 lateral_3m/6m 帧→检测→与投影 GT box IoU） | docs/superpowers plan（未执行） | 1.5 | ⬜ | 外推档增量是对原 plan 的小扩展 |
| **E1.3** ★ | E1 | **held-out camera 真 GT 外推协议**：训练排除 1–2 台侧相机（`dataset.camera_ids` 覆盖），eval 在被排除相机跑 per-class 全套 → 唯一**有真 GT** 的外推轴（DiFix3D+ RDS cross-reference 协议反用）；需从头训 1 个对照 ckpt | NuRec 调研 § 5.1 | 1.5 | ⬜ | 1 次 30k 训练成本；与 E1.1 档位互补 |
| **E1.4** | E1 | **FID/KID 接入**：novel 外推档渲染帧 vs 训练视角真图分布的 FID/KID（torchmetrics/clean-fid），写 metrics.json `mean_novel_fid_{mode}` | SOTA 综述（无 GT 外推共识指标） | 1 | ⬜ | 帧数少时 KID 优先（无偏） |
| **E1.5** | E1 | **v4 gap 表回填**：E0.4 NuRec 锚 + E1.1–E1.4 自有锚汇总入 § 1.3，**据实重排 E2/E3 优先级**（对标 v3 R1 纪律） | — | 0.5 | ⬜ | E1 出口 |
| **E2.1** ★ | E2 | **Harmonizer 升级集成 + 域差 spike**：[`third_party/Fixer`](third_party/Fixer)（一代）→ [NVIDIA/harmonizer](https://github.com/NVIDIA/harmonizer)（Cosmos Predict2 0.6B，时间条件，Apache-2.0）；HF `nvidia/Harmonizer` 权重 → 对 baseline 渲染的 3m/6m 帧离线修复 → E1 指标前后对比（**纯后处理预期：FID/感知大改善、几何指标不动**——正确预期，勿误判失败） | NuRec 调研 § 2.3/5.2 + v3 T15.2 | 1 | ⬜ | inceptio 4090 推理（0.6B 单步 OK）；`nurec-fixer` skill 辅助 |
| **E2.2** ★★ | E2 | **渐进外推蒸馏（v4 核心）**：DiFix3D+ progressive update 移植——外推位姿从 1m→2m→3m→6m 逐步推进，每步「渲染→Harmonizer 修复→修复帧按低权重蒸馏回 3D（road/lane 区域加权）→下一步」；区别 v3 Stage 15 教训：不打全图 repro 轴，蒸馏目标=外推档 + road/lane 病灶区 | NuRec 调研 § 2.4（ablation 证据）+ v3 Stage 15 复活改轴 | 2.5 | ⬜ | gate=E2.1 spike + E1 锚；验收=E1 全指标 |
| **E2.3** | E2 | **actor 弱观测面修复蒸馏**：对车辆 track object-centric 环绕渲染弱观测面 → Harmonizer 修复 → cuboid×sseg mask 内低权蒸馏；攻 P1.4 验尸根因（未观测面缺约束）的 2D 监督解法 | v3 P1.4 否定结论 + SOTA 共识（2D 监督非 3D 注入） | 2 | ⬜ | gate=E2.1；验收=class_psnr + NTA-IoU + 守护线 |
| **E2.4** | E2 | （备选）**Harmonizer 域内微调**：若 E2.1 spike 显示 NCore 域差大——按 DiFix3D+ 降质构造法（cycle reconstruction 横移 1–6m / model underfitting / cross reference）在自有 clip 造配对，LoRA 级微调 | DiFix3D+ 论文 § 训练数据构造 | 2 | ⬜ | 仅 E2.1 域差坐实才投 |
| **E2.5** | E2 | **编辑协调 spike（3dgrut2 侧）**：复用 AH 注入引擎（PR #18 plumbing / frozen 离线手术）在自有 ckpt 插入/取代 1–2 辆 asset-harvester 车 → Harmonizer 时间模式协调 → NTA-IoU/FID + 目视验收；**不训练或轻训练**——区别 P1.4 warm-start（重建轴）：本卡是编辑场景 + 生成协调，正是 NuRec 官方编辑形态 | 新（E0.6 的 3dgrut2 对应） | 1.5 | ⬜ | gate=E2.1 + E0.6 清单；v5 编辑轴第一块带指标基石 |
| **E3.1** | E3 | **空气区 penalty** = **v3 P3.4 移交**：路面上方 0.4m~上界悬浮 bg opacity penalty（cuboid actor 豁免），复用 V3-R2 基建 | v3 P3.4 | 1.5 | ⬜ | gate=E1.1 锚 + **R9（PR #24 去留先决）** |
| **E3.2** | E3 | **road SH 降阶 DC-only（freeze 法）** = **v3 P3.5 移交**：砍 view-dependent 过拟合逃逸通道（路面近似 Lambertian） | v3 P3.5 | 1 | ⬜ | gate 同 E3.1 |
| **E3.3** ★ | E3 | **BEV 纹理平面化**（v4 backlog 转正）：road 颜色不再 per-gaussian SH，改 BEV feature grid/纹理图采样、真正贴在高度场平面 → **外推天然正确**（参数化级根治 aperture problem；ExtraGS Road Surface Gaussians 同思路）；复用 [`road_region.py`](threedgrut/model/road_region.py) BEV 网格基建 | v3 § 5 backlog「外推终极方向」 | 3 | ⬜ | gate=E1 锚 + E3.1/E3.2 结果（短刀够用则缓） |
| **E3.4** | E3 | （备选）**平面诱导 warp 伪横移一致性 loss**：训练时按路面平面 homography warp 伪横移视角做一致性约束 | v3 § 5 backlog 备选 | 1.5 | ⬜ | E3.3 的轻量替代/前菜 |
| **E4.1** | E4 | （可选）**LiDAR 点云推理**：按 [`2026-06-10-lidar-pointcloud-from-gs.md`](docs/superpowers/plans/2026-06-10-lidar-pointcloud-from-gs.md) 执行（A0 gate：3DGUT ckpt 能否被 3DGRT 渲 → A1 射线表 → A2 range-L1/出 .ply）；外推的传感器维度（novel 轨迹渲 LiDAR），对标 NuRec LiDAR re-sim | docs/superpowers plan（未执行） | 2.5 | ⬜ | A0 NO-GO 则整线作废（plan 内置判据） |

### 1.3 Phase 状态汇总 + v4 gap 表（E0/E1 回填）

| Phase | 主题 | 任务数 (Done/Total) | 主验收 | 守护线 | 状态 |
|---:|---|---:|---|:---:|:---:|
| **E0** ★ | NuRec 工具链复现立锚（**首要**） | 4/6 | ≥2 场景跑通（1 USDZ 渲染+修复链 ✅、1 自有 clip 训练 ✅）+ NuRec 锚入档 ✅ + 配方 diff 清单 ✅ + 官方编辑能力清单（E0.6 待开） | — | 🟡 |
| **E1** ★ | 外推测量门（gate 后续一切） | 0/5 | 3m/6m + NTA-IoU + held-out + FID/KID 全部立锚入 gap 表 | interpolated 全指标不退 | ⬜ |
| **E2** | 生成修复链（NuRec 思路移植）+ 编辑协调 spike | 0/5（含 1 备选） | E1 外推指标相对锚改善（量级参考 DiFix3D+：蒸馏+后处理 +1.8dB/FID −20%）；E2.5 插入协调立带指标基石 | cc ≥ 24.7 / grad_corr 0.744 不退 | ⬜ |
| **E3** | 表示侧外推强化（与 E2 互补） | 0/4（含 1 备选） | 同 E2 验收口径；E3 减伪影产生、E2 修残余 | 同上 | ⬜ |
| **E4** | LiDAR 外推（可选） | 0/1 | A0 GO + range-L1 入档 | — | ⬜ |
| **总计** | — | **0/21** | — | — | — |

> **v4 gap 表（E0.4 + E1.5 回填，格式预置）**：
> | 轴 | 3dgrut2 锚（E1） | NuRec 锚（E0.4） | 差距 | E2/E3 后 |
> |---|---|---|---|---|
> | lane grad_corr @ 3m / 6m | 待测 | 待测 | — | — |
> | NTA-IoU @ 原轨迹 / 3m / 6m | 待测 | 待测 | — | — |
> | held-out cam per-class（车/lane） | 待测 | 待测 | — | — |
> | FID/KID @ 3m / 6m | 待测 | 官方 USDZ 场景参考：FID 57.3 @3m / 93.0 @6m（vs 原轨迹分布，0fd06bc3，修复前） | — | — |
> | interpolated（守护线） | class 25.07 / cc 26.06 / grad_corr 0.744 | **官方口径**：psnr 30.30 / cpsnr car 34.59 · road 38.27 · person 32.65 / chamfer 0.295（E0.3，**口径未统一不可直接比**，E0.4 重算） | 待 E0.4 | 不退化 |

### 1.4 任务依赖图

```mermaid
flowchart TD
  classDef gate fill:#e6f4ff,stroke:#0070f3,color:#000,font-weight:bold
  classDef todo fill:#f5f5f5,stroke:#999,color:#333
  classDef opt fill:#fbe9e7,stroke:#d33,color:#900

  v3["v3 成果（继承）<br/>class 25.07 / lane 0.744 / 外推诊断 / NTA-IoU plan"]:::gate
  E0["E0 NuRec 复现立锚 ★首要<br/>E0.1 环境 → E0.2 USDZ渲染+修复链 → E0.3 自有clip训练<br/>→ E0.4 双向对照锚 + E0.5 配方diff + E0.6 编辑体验"]:::gate
  E1["E1 外推测量门<br/>E1.1 三米六米档（＝P3.3） / E1.2 NTA-IoU / E1.3 held-out<br/>/ E1.4 FID-KID → E1.5 gap表+重排"]:::gate
  E2["E2 生成修复链<br/>E2.1 Harmonizer升级spike → E2.2 渐进外推蒸馏 ★★<br/>→ E2.3 actor弱面蒸馏 / E2.5 编辑协调spike（E2.4 微调备选）"]:::todo
  E3["E3 表示侧强化<br/>E3.1 空气区penalty（＝P3.4） / E3.2 road DC-only（＝P3.5）<br/>→ E3.3 BEV纹理平面化（E3.4 warp备选）"]:::todo
  E4["E4 LiDAR外推（可选）<br/>A0 gate → 射线表 → range-L1"]:::opt
  PR24["R9：PR #24 去留先决"]:::opt

  v3 --> E0
  v3 --> E1
  E0 -->|NuRec 锚 + diff 清单| E1
  E1 -->|锚点 gate| E2
  E1 -->|锚点 gate| E3
  E0 -.配方借鉴.-> E3
  PR24 -.先决.-> E3
  E1 -.可选.-> E4
```

> 并行性：E1.1/E1.2 纯 eval，可与 E0 并行开工（E0.1 环境就绪即可）；E0.4 需要 E1 工具，故 E0 与 E1 交替推进。E2 与 E3 在 E1 锚后可并行（不同文件域），但**单变量 A/B 纪律**：同一对照实验只动一边。

---

## 2. Phase 详细任务卡

> 只描述目标 / 改动文件 / 验收准则，不放代码（CLAUDE.md 约束）。开工时每个任务按 superpowers 流程起 `docs/superpowers/plans/` TDD 执行 plan。

### 2.0 Phase E0 — NuRec 工具链复现立锚（首要任务）★

**触发**：立即。
**核心**：在自己动手改进之前，先把"对标对象"真实地跑起来——① 跑通官方**渲染+修复链**直观感受外推天花板（E0.2）；② 在**同一份自有 clip** 上跑官方配方，把流传的"~36 dB"对标值变成同数据实测锚（E0.3/E0.4）；③ 把官方配方与 multilayer 配方的差异变成可执行的借鉴清单（E0.5）；④ 顺手走一遍官方 actor 编辑工作流，拿到「插入/取代」的能力/限制清单（E0.6）。**全程用已装的 nurec-skills**（`nre` / `ncore` / `asset-harvester` / `nurec-fixer` / `physical-ai-datasets`），不重复造轮子。

| Task | 关键动作 | 锚点 / 备注 |
|---|---|---|
| E0.1 | ✅ 镜像已就位（`nre-ga:latest` 42.6GB / `nre-tools-ga:latest` 59.7GB，2026-06-11 实测，拉取未用 key）→ 剩：nre skill `scripts/validate_setup.py` → 容器内 `nvidia-smi` + `nre --help` 冒烟 → **不带 NGC key 先试**，遇 auth 错再补（R-v4.1）→ 显存/内存评估 + `:latest` 实际版本号记录 | inceptio 已有 docker + nvidia runtime（难点已清）；**4090 24GB 处官方推荐（24–48GB+）下限** |
| E0.2 | `physical-ai-datasets` skill 选下 1–2 个 PhysicalAI-AV-NuRec USDZ（~每场景数 GB，HF token）→ `nre render --artifact-path <usdz>` 原轨迹 + 自定义 `--custom-rig-trajectory`（横移 3m/6m）渲帧 → `nurec-fixer` skill 过 Harmonizer（时间/非时间两模式）→ 目视对照 + 修复前后 FID | **不训练、成本最低**，先验证"官方场景外推+修复后长什么样"；产出对照帧存档供 E2 设计参考 |
| E0.3 | 自有 clip（`pai_9ae151dc`，inceptio `~/work/data/9ae151dc/`）先 `ncore` skill validate 格式版本 → `nre` 官方 AV 配方（`3dgut_dynamic` 系）训练；OOM 则降分辨率/相机数，仍不行 → `vast-train` skill 起 48GB 卡 | **NCore v4 本就是 NuRec 原生格式**，预期低摩擦；R-v4.3 版本兼容风险见 § 3 |
| E0.4 | 双向对照：NuRec ckpt 与 multilayer baseline ckpt 在**同一评测协议**下跑全指标——interpolated（PSNR/LPIPS/class_psnr/lane grad_corr）+ 外推（E1.1 的 3m/6m lane 指标、E1.2 NTA-IoU、E1.4 FID/KID）。NuRec 侧用 `nre render` 出帧后喂项目 eval 工具（指标代码统一用项目侧，保口径一致） | **v4 gap 表首行**；若 NuRec 外推也糊 → 修复器才是主差距，E2 权重↑；若 NuRec 表示侧就稳 → 配方/E3 权重↑。**这个判别本身就是 E0 最大价值** |
| E0.5 | diff 官方 yaml vs [`ncore_3dgut_mcmc_multilayer.yaml`](configs/apps/ncore_3dgut_mcmc_multilayer.yaml)：LiDAR intensity 监督、densification/正则参数、相机/rolling shutter 处理、sky/road 专项、训练长度与 lr 调度 → 按「外推相关性」标记优先级，喂 E3 | 产出 markdown 清单入 `docs/superpowers/specs/` |
| E0.6 | nre actor 编辑（gRPC `serve-grpc` 演员编辑 / CLI）做「删一辆 / 插一辆 AH 收割车 / 取代一辆」→ Harmonizer 协调 → 渲染对照存档 + 能力/限制清单（删除后路面洞？插入物阴影来源？协调前后 FID/目视） | **P1.4 官方解法对照**：官方不把 asset 做完美，而是插入后生成协调器擦屁股；清单直接喂 E2.5 设计 |

**验收**：≥2 场景跑通（1 官方 USDZ 渲染+修复链、1 自有 clip 官方训练）；NuRec 锚数字写入 § 1.3 gap 表 + § 5 Done Log（commit hash + 实测数）；配方 diff 清单 + 官方编辑能力/限制清单入档。**E0 不改 3dgrut2 任何训练代码。**

### 2.1 Phase E1 — 外推测量门 ★

**触发**：E0.1 完成即可与 E0 并行。
**核心**：v3 教训（测量先行）在外推轴重演——现 novel 指标存在**幅度盲区**（仅 ≤2m）与**区域盲区**（全图 LPIPS 沥青/bg 主导）。E1 不立锚，E2/E3 任何"改善"不可证。

| Task | 改动文件 / 锚点 |
|---|---|
| E1.1（=P3.3 移交，任务卡原文见 [`v3_plan_revised.md`](v3_plan_revised.md) § 1.2 P3.3） | [`novel_view.py`](threedgrut/utils/novel_view.py)（加 lateral_3m/6m 档，4 档 avg 字段口径不变）、[`render.py`](threedgrut/render.py)、[`per_class_eval.py`](threedgrut/model/per_class_eval.py)（lane 区域 novel 指标：路面平面诱导 warp 重投影 lane band，FTheta 兼容）；三方 ckpt（baseline/B3/aniso20，inceptio 现存）立锚 |
| E1.2 | 按 [`2026-06-10-nta-iou-eval-metric.md`](docs/superpowers/plans/2026-06-10-nta-iou-eval-metric.md) Task 0–5 执行（新建 `nta_iou.py` / `vehicle_detector.py`）；**增量**：novel 档渲染帧上同样跑 NTA-IoU（GT cuboid 投影到 novel 相机位姿——投影函数复用 `project_cuboids_to_mask`，位姿来自 novel_view 档位变换） |
| E1.3 | 训练配置：`'dataset.camera_ids=[...]'` 排除 1–2 台 cross 相机从头训对照 ckpt（inceptio depth-off+nw=10 配方）；eval：被排除相机上跑 per-class 全套 + NTA-IoU。新增 eval 开关（held-out 相机集合可配） |
| E1.4 | render eval 新增 FID/KID：novel 档渲染帧集 vs 训练视角真图集（torchmetrics image.fid / kid；帧数 <500 用 KID）。metrics.json：`mean_novel_fid_{mode}` / `mean_novel_kid_{mode}` |
| E1.5 | 汇总 E0.4 + E1.1–1.4 → § 1.3 gap 表回填 + **据实重排 E2/E3**（写明判别逻辑结果：差距主因=修复器/表示/配方） |

**验收**：gap 表全行立锚；现有 interpolated 指标零回归（纯增量改动）；metrics.json 新字段齐全（CLAUDE.md § B6：没见到新 key 不许标 ✅）。

### 2.2 Phase E2 — 生成修复链（NuRec 核心思路移植）

**触发**：E1 锚点入档 + E2.1 spike 正向。
**核心**：把 NuRec 已验证的「修复器双模式」移植到 3dgrut2——**蒸馏回 3D 提几何，在线后处理提感知**（DiFix3D+ ablation 实证）。与 v3 Stage 15 的本质区别：旧实验打"全图 repro 轴"（+0.30 而废），E2 打"外推档 + road/lane/actor 病灶区"。

| Task | 改动文件 / 锚点 |
|---|---|
| E2.1 | `third_party/` 新增 harmonizer（[NVIDIA/harmonizer](https://github.com/NVIDIA/harmonizer)，HF `nvidia/Harmonizer` 权重：`diffusion_harmonizer.pkl` 时间版 / `harmonizer_nontemporal.pt` 单帧版）；升级 [`correction/difix.py`](threedgrut/correction/difix.py) 为后端可切换（fixer/harmonizer）；spike：baseline 渲 3m/6m 帧 → 两模式修复 → E1 指标前后对比 + 目视存档。**预期正确性提醒：纯后处理 FID/感知应大改善、grad_corr/NTA-IoU 等几何敏感指标基本不动——这是 ablation 的预期行为，不是失败** |
| E2.2 | [`trainer.py`](threedgrut/trainer.py) 新增渐进蒸馏调度（opt-in，默认关）：每轮从 [`novel_view.py`](threedgrut/utils/novel_view.py) 取当前外推档位姿 → 渲染 → Harmonizer 修复（时间模式，批量离线）→ 修复帧入蒸馏 buffer（低权重 λ_distill，road/lane 区域加权 mask）→ 档位按 schedule 推进 1m→2m→3m→6m。ckpt/恢复兼容；A/B：E2.2 on vs off 从头训对照 |
| E2.3 | object-centric 弱面渲染（复用 playground 环绕相机基建 [`threedgrut_playground/`](threedgrut_playground/)）→ Harmonizer 修复 → cuboid×sseg mask 内蒸馏（与 E2.2 共用蒸馏 buffer 机制，目标区域不同）；攻 P1.4 根因——**约束式 2D 监督**替代被否定的 freeze 式 3D 注入 |
| E2.4 | （备选）域内微调：DiFix3D+ 降质构造（cycle reconstruction：训练 ckpt 沿横移轨迹渲帧再反向重建造退化对；underfit：25–75% epoch ckpt 渲帧）造自有 clip 配对 → LoRA 微调 Harmonizer。**仅 E2.1 域差坐实才投** |
| E2.5 | 编辑协调 spike：AH 注入引擎（PR #18 plumbing / frozen 离线手术）在自有 ckpt 插入/取代 1–2 辆收割车 → [`correction/difix.py`](threedgrut/correction/difix.py)（E2.1 升级后）Harmonizer 时间模式协调 → **三验收**：NTA-IoU（插入车被检出且框齐）/ FID（协调前后）/ 目视存档；E0.6 官方能力清单作设计输入；产出即 v5 编辑轴立项依据 |

**验收**：E1 外推指标（lane@3m/6m + NTA-IoU + FID/KID + held-out per-class）相对锚改善；守护线不破（cc ≥ 24.7 / interpolated grad_corr、class_psnr 不退）；**双协议验收防幻觉**（R-v4.5）：held-out 真 GT 指标与无 GT 感知指标须同向改善。

### 2.3 Phase E3 — 表示侧外推强化（与 E2 互补）

**触发**：E1 锚点 + R9（PR #24 去留）落定。
**核心**：E2 修"已产生的伪影"，E3 减少"伪影的产生"——对应 NuRec 的表示侧（LiDAR 强监督、mesh、配方）。E3.1/E3.2 是 2026-06-11 诊断的"第 1 层短刀"（原 v3 P3.4/P3.5 移交），E3.3 是参数化级根治。

| Task | 改动文件 / 锚点 |
|---|---|
| E3.1（=P3.4 移交） | [`road_region.py`](threedgrut/model/road_region.py)、[`trainer.py`](threedgrut/trainer.py)、multilayer yaml：路面上方 0.4m~上界（A/B 定）空气区 bg opacity penalty，cuboid 内 actor 豁免；复用 V3-R2 height field / `query_ground_z` / cuboid mask 全套 |
| E3.2（=P3.5 移交） | [`layered_model.py`](threedgrut/layers/layered_model.py)、[`registry.py`](threedgrut/layers/registry.py)：road 层高阶 SH zero+freeze（保 45 维宽度一致，绕 V3-R1.1 fused SH 坑）。注：2026-06-11 已把死配置 `scale_lr_mult` 接线（`_apply_scale_lr_mult`，registry 默认 1.0 保锚点等价）——E3 做官方式 road scales lr 冻结（5e-3→1e-4）直接 `++layers.overrides.road.scale_lr_mult=0.02`，无需新代码；positions 1e-6 冻结仍需另做绝对值 override 机制 |
| E3.3 | road 颜色改 BEV feature grid/纹理图采样（贴高度场平面）：[`road_region.py`](threedgrut/model/road_region.py) BEV 网格基建扩展 + 渲染路径改造；E0.5 配方 diff 中 NuRec 对路面的处理方式作输入。**先 spike 小网格验证训练稳定，再全量** |
| E3.4 | （备选）平面诱导 warp 一致性 loss：训练 batch 内按路面 homography warp 伪横移视角与渲染一致性约束；E3.3 的轻量前菜，若 E3.1/E3.2 后 3m/6m 指标已达标则跳过 |

**验收**：同 E2 口径（E1 指标 + 守护线）；E3.1 另验 road/bg 耦合改善（路面区 bg 粒子占比、bg 替补率 <24% 现状）；R10（路面出洞）监控——E3.1 只动空气区、贴地带机制不变。

### 2.4 Phase E4 —（可选）LiDAR 外推

按 [`2026-06-10-lidar-pointcloud-from-gs.md`](docs/superpowers/plans/2026-06-10-lidar-pointcloud-from-gs.md) 执行，plan 已含 A0 stop/go gate（3DGUT ckpt ↔ 3DGRT tracer 兼容性）、A1 纯几何 TDD、A2 GPU 集成与 range-L1 验收。v4 语境下的定位：**外推的传感器维度**（novel 轨迹下渲 LiDAR 点云），对标 NuRec 的 camera+LiDAR re-sim 能力；非主线，资源富余或 E2/E3 阻塞时插空。A3（intensity/ray-drop 内核）保持 backlog。

### 2.5 与 v3 的边界与移交（避免重复实现）

| 项 | 归属 | 说明 |
|---|---|---|
| **P3.3 / P3.4 / P3.5** | **移交 v4**（E1.1 / E3.1 / E3.2） | 2026-06-11 立项时即为外推主题；执行与回填以本 plan 为准。**待办**：v3_plan_revised.md 对应三行加「→ 移交 v4_plan.md（E1.1/E3.1/E3.2）」标注（一行级改动，防双 plan 漂移） |
| Phase 2 行人（P2.1–2.3）、P3.2 遮挡式 bg、P1.1 sseg 边界、P1.3、P-CAP、AH-* | **留 v3 主线** | per-class interpolated 轴；与 v4 并行与否由大g按资源排期。P3.2 与 E3.1 共用 road 基建，若同期开工须协调（R-v4.6） |
| per-class evaluator（P0）、lane 门（P3.0）、novel_view.py、track-pose、dynamic_mask 投影、USDZ 导出、asset-harvester、difix.py | **v4 直接复用** | 不重做；E1/E2 在其上做增量 |
| NTA-IoU plan、LiDAR plan（docs/superpowers/plans/ 2026-06-10 两份） | **v4 吸收执行**（E1.2 / E4.1） | plan 文档已写好 TDD 步骤，直接按卡执行，不重写 |
| BEV 纹理平面化、平面诱导 warp、Cosmos-DiFix synthesized GT（v3 § 5 backlog） | **v4 转正**（E3.3 / E3.4 / E2.2 思路） | v3 backlog 中的外推项全部进 v4 主线或备选 |
| PR #24（P3.1-A lane loss 代码） | **R9 先决**（继承 v3） | E3.1/E3.2 动同一批 road spec/yaml，开工前必须先定 PR #24 去留，保 road 几何参数单一来源 |

---

## 3. 风险登记表（Risk Log）

| ID | 风险 | 触发 | 影响 | 缓解 | 关联 |
|---|---|---|---|---|---|
| R-v4.1 | ~~nre 容器运行时 NGC key 需求未验证~~ **已关闭（2026-06-11 实测）**：无 NGC key 跑通 validate→train（40k 步）→render→USDZ 导出全链；唯一确认需要 key 的点＝官方 train-time difix 蒸馏权重 `cosmos_3dgut.pt`（NGC API URL、HF 无副本，仅 `sqa_difix_distill` 配方用到——E2 用开源 Harmonizer 自研蒸馏绕开） | — | — | 若将来要跑官方 `sqa_difix_distill` 对照 run，再向大g 要 key | E0 ✅ |
| R-v4.2 | inceptio 4090 24GB 低于官方推荐显存（24–48GB+） | E0.3 官方配方 OOM | 训练复现受阻 | 降分辨率/相机数先跑通；`vast-train` skill 起 48GB 卡（成本 ~$0.5/h，5k smoke 级） | E0.3 |
| R-v4.3 | 自有 clip 与 nre 26.x 的 NCore 版本兼容性（项目 clip 较旧 vs NCore 2026.04） | E0.3 数据加载报错 | 同上 | 先 `ncore` skill validate；不兼容则用官方 PhysicalAI 场景完成 E0.3/E0.4（锚点换数据，对照价值略降但仍立得住） | E0.3 |
| R-v4.4 | Harmonizer 域差（NVIDIA 车队数据 post-train vs NCore clip 相机/ISP） | E2.1 spike 修复质量差/引入异物 | E2 全线打折 | E2.1 先 spike 实测再投 E2.2/2.3；域差坐实走 E2.4 微调（DiFix3D+ 降质构造法已备） | E2 |
| R-v4.5 | **幻觉污染评测**：蒸馏的是扩散模型生成像素，指标升 ≠ 更真实 | E2.2/E2.3 验收 | 自欺 | **双协议验收**：held-out 真 GT 指标（E1.3）与感知指标（E1.4）须同向；目视存档强制；蒸馏权重 λ 从小起 | E2 |
| R-v4.6 | PR #24 / road 基建多任务交叠（继承 v3 R9 + P3.2 同域） | E3 开工 | road spec/yaml 双源漂移、合并冲突 | **先定 PR #24 去留再开 E3**；E3 与 v3 P3.2 不同期动 road 文件 | E3 |
| R-v4.7 | 渐进蒸馏破坏 interpolated 质量（修复帧与真图监督打架） | E2.2 训练 | 守护线破 | λ_distill 低权重起步 + 外推帧只在病灶区域加权 + 守护线全程监控（v3 全套 interpolated 指标在同一 metrics.json） | E2.2 |
| R-v4.8 | E1.3 held-out 协议与 5-cam ring 配方互斥（少相机训练本身改变 baseline） | E1.3 立锚 | 锚点口径混乱 | held-out 对照独立成线（自己 vs 自己），不与全相机 baseline 跨协议比较；文档写死口径 | E1.3 |
| R-v4.9 | Harmonizer license 合规（权重 NVIDIA Open Model License；PhysicalAI 数据集限制性专有） | 商用/再分发场景 | 合规风险 | 代码 Apache-2.0 无虞；权重许可允许商用但需保留条款；PhysicalAI 数据**仅限内部开发评测**、不入训练数据、不再分发 | E0/E2 |

---

## 4. v5 / backlog 转出

- **编辑/仿真轴产品化**（删/插不留痕质量达标、批量编辑、inpaint 遮挡地面、学习式软分割）——v4 已纳入 E0.6/E2.5 两个 spike 立基石（官方工作流能力/限制清单 + 带指标插入协调）；产品化与留痕清零留 v5，E2.5 产出即其立项依据
- closed-loop 仿真集成（gRPC 渲染服务化、CARLA/AlpaSim 对接——NuRec 调研 § 1.4 形态）
- LiDAR intensity / ray-drop（E4-A3，需改 OptiX/CUDA 内核）
- Harmonizer 时间模式在线化（训练内实时增强；v4 仅离线批量用）
- 跨 clip 联训、行人外推（等 v3 Phase 2 行人模型落地后自然并入 E1 评测轴）

---

## 5. Done Log（继承锚点 + 新条目）

**继承锚点（作 v4 对照基础）**：
- **2026-06-11 外推诊断**：aperture problem 根因 + 测量双盲区（幅度 ≤2m / 区域沥青主导）+ road/bg 耦合量化（road 自盖 68%、bg 替补 24%）——P3.3–P3.5 立项依据，本 plan E1.1/E3.1/E3.2 之源（[`v3_plan_revised.md`](v3_plan_revised.md) § 6）。
- **2026-06-10 NuRec 工具链调研**：修复器三代演进（DiFix3D+ → Fixer → DiffusionHarmonizer）、ablation 证据（后处理提感知/蒸馏提几何）、驾驶外推 +1.8dB/FID −20%、held-out 协议、nurec-skills——E0/E2 之源（[`~/repo/report/nvidia-nurec-extrapolation-analysis.md`](../report/nvidia-nurec-extrapolation-analysis.md)）。
- **v3 P1.2 track-pose**：class 25.07 / cc 26.06（interpolated 守护线锚）。
- **v3 P3.0 + P3.1-A**：lane grad_corr 门锚 0.693 → 0.744（+0.051；代码 PR #24，R9）。
- **v3 P1.4 protected warm-start 否定**：freeze 式 3D 注入被否，真瓶颈 = 未观测面缺约束 → E2.3 改 2D 修复蒸馏的立项依据。
- **v3 T15.2**：Fixer 一代已集成（[`correction/difix.py`](threedgrut/correction/difix.py)），E2.1 升级起点；Stage 15 全图蒸馏 +0.30 教训 → E2.2 改打外推档+病灶区。

**新条目（v4 启动后填充，格式：日期 + commit + 实测数）**：
- **2026-06-11 E0.1 + E0.2 + E0.3 + E0.5 完成**（commit `8b2fcbe`，worktree 分支 claude/amazing-tereshkova-5c7b30）——E0 首日四卡落地（剩 E0.4 gate E1 工具、E0.6 材料已备），全程无 NGC key：
  - **E0.1 容器冒烟**：`nre-ga:latest` 实测 = **26.4.146-c63f08a4**（2026-05-28 build，entrypoint `/app/run`）。容器内 nvidia-smi（CUDA 13.1）/ CLI 全子命令面正常。**R-v4.1 答案：无 NGC key 跑通 validate→train→render 全链**；key 唯一已知需求点 = 官方 train-time difix 蒸馏权重 `cosmos_3dgut.pt`（NGC API URL，HF 无副本）。
  - **E0.3 官方配方训练**（inceptio 4090）：clip 9ae151dc + Hyperion-8.1 `car2sim_6cam` + `references/configs/pai.yaml` overlay（`--config-name=external_overrides`），40k 步 **2h07m**（7.45 it/s），峰值显存 16GB（**24GB 无需任何降配**，R-v4.2 未触发），2.62M gaussians。**官方口径锚（每 3 帧 + 1/4 分辨率 + cpsnr）：test/psnr 30.30 / cpsnr road 38.27 · car 34.59 · person 32.65 · sky 38.81 / chamfer_distance 0.295**；产物 `~/work/nurec_e0/train_out/PVG7YYV72YKPLumogi7F7U/`（`artifacts/last.usdz` 1.1GB + val mp4 + metrics.yaml 20 类全套）。“传闻 ~36dB”在本 clip 官方口径下不存在（30.30）；与 multilayer 26.06 的对比须待 E0.4 统一口径。
  - **数据兼容实录（R-v4.3 实际未命中）**：官方 13 个 itar 全兼容 nre 26.4；唯一阻塞 = **v3 P3.0 自产 `aux.lane.zarr.itar`**（旧版 ncore 写入、缺 `.zmetadata.cbor.xz` 且 sequence 标识不符），两次启动失败（KeyError / Can't load aux data for different sequences）→ 用容器内官方 `consolidate_compressed_metadata()` 升级全部 itar 至 `9ae151dc_consolidated/` + lane aux 移出 → 三次启动成功。教训入档：**自产 aux 文件会被 nre 按文件名 glob 自动吞掉，喂官方容器的目录必须只留官方产物**。
  - **E0.2 渲染 + 修复链全链完成**（官方 USDZ 场景 0fd06bc3，1.92GB，4K→1080p）：`nre render` 三档各 595 帧（原轨迹 GT / lateral 3m / 6m，**17.8ms/帧** @4090，rig offset 法）→ Harmonizer 时间模式修复两档（~1.07it/s，594 帧/档 ~9min）。**FID（vs 真实参考帧分布，正确口径）：原轨迹渲染 7.37 / lat3m 57.3→修复后 65.6（↑8.4）/ lat6m 91.8→86.6（↓5.2）**；（早期 vs GT 渲染分布口径：57.3/93.0，留档备查）。**目视修复显著**（lat6m 锥桶修直、左缘树木涂抹消除、路面脏斑清除、黄线连续）**但 FID 几乎不动——E0 判别性结论：① 官方表示侧太强（原轨迹 FID 7.4、6m 横移 raw 才 92，伪影占比低），FID 大头是视角内容差，修复链 FID 收益∝伪影占比（DiFix3D+ 论文 134→50 是重伪影场景）；② 推论：3dgrut2 自有 ckpt 伪影远重于官方 → E2 修复收益预期更接近论文场景；③ FID 单指标评修复会系统性误判（修复输出的扩散平滑风格会抵消去伪影收益）→ E1.4 必须搭配区域化/patch 指标，R-v4.5 双协议验收被实测证实必要**。对照帧存档：inceptio `~/work/nurec_e0/renders/0fd06bc3/{gt,lat3m,lat6m}/`、修复帧 `~/repo/harmonizer/input_frames_cosmos_temporal_lat{3,6}m/`、真帧 `~/work/nurec_e0/real_frames/0fd06bc3/`（594 帧 ffmpeg 自参考 mp4）。Harmonizer 链工程留档：镜像 `harmonizer-cosmos-env`（33.1GB，base `pytorch:25.10-py3` 本地、Dockerfile COPY stage 改本地镜像绕 docker.io 拉取）；HF 权重全量（`diffusion_harmonizer.pkl` 5GB + `Cosmos-Predict2-0.6B-Text2Image` 4.1GB）；**HF 侧两道人工解锁：fine-grained token 勾 gated repos read 权限（403 报错文案误导为网络错误）+ Cosmos-Predict2 模型页接受 license（与 token 权限独立）**；推理须 `--entrypoint python` + 每 run 独立输出目录（时间模式会回读历史输出做参考帧）。
  - **E0.5 配方 diff**：见 §1.2 行内 Top-5 与 [`docs/superpowers/specs/2026-06-11-e05-nurec-vs-multilayer-recipe-diff.md`](docs/superpowers/specs/2026-06-11-e05-nurec-vs-multilayer-recipe-diff.md)（11 维度 / 全部引 resolved key 路径）。**最大架构发现：官方 road = 几何冻结的 ground-mesh 薄片层 + MCMC 全豁免**（aperture problem 的官方答案是“路面不让动”，与 v3 诊断完全互证）；官方 difix 蒸馏钩子原生就是 ±3m lateral（默认关）。
  - **大g 目视终评（nre viewer 交互对比，E0 判别预答案）**：官方配方产物 vs multilayer 自训——①整体锐度**不更好**（官方 subsample:2 半分辨率监督所致，我们 full-res 不输）；②**路面车道线明显更好**；③**3m 侧向移动退化明显更小**。且该 run difix 蒸馏关闭——**官方纯表示侧（road 冻结五件套 + LiDAR ray 监督）就赢下外推**，E0.4 判别初步指向：表示侧为主差距 → **E3 优先级 ≥ E2**（待 E1 量化锚正式确认）。
  - nre viewer 工程留档：26.4.146 viewer 两个 bug/坑——①客户端连接竞态（camera on_update 先于 reload action → `Can't unpack empty optional` 渲染线程死），host-side 两行 patch 绕过（`~/work/nurec_e0/patches/av_patched.py` 挂载覆盖）；②nrend 快速路径对自训 USDZ 间歇性 `NRenderer.render failed`（官方 USDZ 正常），`--no-enable-nrend` 走 torch 路径稳定。viewer GUI 自带 Camera Translation Offset（可直接体验横移外推）+ Render Video 导出。
  - 环境备忘：HF 下载一律 `HF_HUB_DISABLE_XET=1` + 代理（xet/直连均会卡死或失败）；harmonizer 容器 entrypoint 是 `/bin/bash`，跑推理须 `--entrypoint python`。

---

## 6. 文档关系速查

| 想找 | 去哪 |
|---|---|
| v3 interpolated 主线（行人 Phase 2 / P3.2 / P1.x 剩余） | [`v3_plan_revised.md`](v3_plan_revised.md)（继续有效，与本 plan 并行） |
| NuRec 工具链全景 / 修复器细节 / license | [`nvidia-nurec-extrapolation-analysis.md`](../report/nvidia-nurec-extrapolation-analysis.md) |
| 领域综述（外推方法谱系 / 指标共识） | [`3d-4d-state-of-the-art-2025-2026.md`](../report/3d-4d-state-of-the-art-2025-2026.md) |
| NTA-IoU / LiDAR 点云 TDD 执行 plan | [`docs/superpowers/plans/2026-06-10-*.md`](docs/superpowers/plans/) |
| 架构图 / 文件清单 / 关键不变量（v4 新模块继续在此登记） | [`v2_architecture.md`](v2_architecture.md) |
| 执行约定（机器 / 配方 / 文档同步） | [`CLAUDE.md`](CLAUDE.md) |
