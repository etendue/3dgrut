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
        [E2.2 渐进外推蒸馏（核心移植）]
        [E2.3 actor 弱观测面修复蒸馏]
        [E2.6 viser_gui_4d temporal 后处理（difixer Fixer→Harmonizer 三代时间模式，回读前帧提 inference 时序一致性）]
        [E2.7-C dyn features_albedo Fourier→SH 转换（E2.7-B 烟雾感 follow-up）]
        [E3.2.6 takeover 调强 spike（E3.2.5 viser follow-up）：车道线白条被 bg 抢→调强 bg_road_penalty λ0.4 z_band1.5 收回 road；driver 就位、gate E3.2.5 ✅]
        [E3.3 BEV 纹理平面化（gate E1 锚）]
        [E4.1 LiDAR 点云推理（A0 gate，可选）]

    "In Progress"
        [E0.6 官方编辑体验：run-book + 资产 + schema 全就绪，待 GPU 空档]

    "Blocked"
        [E3.1 ＝v3 P3.4 移交：空气区 penalty（gate E1.1 锚 ✅ + R9 PR24）]
        [E3.2 ＝v3 P3.5 移交：road SH DC-only freeze（gate 同 E3.1）]

    "Done"
        [E3.2.5 几何硬退化 disk ✅ spike 2026-06-22→23：6k A/B on＞off（守护线 cc +0.41 / lateral lane grad_corr +0.04~0.06 / freeze 铁证 rot 0°·z 1mm·N 恒定）→ 反驳 roadoff「光冻结变差」；viser 视觉 on 斑马线分明、白条被 bg 抢→E3.2.6；30k 全量待排期]
        [E2.8 系统性 rigid 全替流水线 ✅ 2026-06-17→18 6 阶段：USDZ拆→全 active/附近 vehicle replace+insert（20 AH automobile + 1 跨源真 bus recon t405）→drop 非vehicle→MLP 跨源天空→QA 闸；端到端实测 coverage=1.0、opacity_med 0.10、QA passed、viser+harmonizer K=4 RTT~1s 0 OOM；slot-basis 铁律 adc2a5c + 坐标 +translate f0d9aca 两修；41 测全绿（35 e28 + 6 quant）；Task6 定量 raw ✅（NTA 0.085 / novel FID 233，bg 离轴涂抹主导、几何有效，harmonizer before/after ✅ FID 233→208 −25、6m −43）、Task7 bus AH 收割 ✅（asset-harvester 收 bus/405+truck/165 干净 AH 资产入 bank → 重跑 driver AH-match 21 / 跨源 recon 0、bus recon→AH 自动切、packed_ckpt_busah.pt）]
        [E2.5 编辑协调 spike ✅ 2026-06-17：AH 车 frozen 注入取代 recon 车 + viser+harmonizer 实时协调目测——harmonizer 协调有效但有限（违和感降低、优于无 harmonizer、未完全自然）；定量 NTA-IoU FID 跳过留 v5]
        [E2.7-B dynamic_rigids 接线 ✅ 2026-06-15：cuboid wireframes + dyn gaussian 位置贴车随 timeline 动（t405 bus 实测对齐）；color 烟雾感转 E2.7-C]
        [E2.7 viser_gui_4d 加载 NVIDIA usdz ✅ 2026-06-15：world_to_nre +38m 对齐 + camera_front_wide_120fov frustum 修复；视觉对标 NRE 路面横向 3m/6m + 360° 不退化]
        [E2.1 Harmonizer 离线修复 spike ✅ 2026-06-13：FID −30%/KID −60%/NTA +35%，lane_grad_corr 退化（扩散平滑）→ E2.2 GO]
        [E0.4 双向对照锚 ✅ 2026-06-12：3m 档官方 +0.05 corr，6m 两家同崩；interp FID 61 vs 75]
        [E1.5 gap 表收口 ✅：重排结论 E3 先行、E2 补 6m+ 档；E1 阶段全绿]
        [E1.3 held-out 真 GT 外推 ✅ 2026-06-12：差距 7.77 dB（upper 26.93 − heldout 19.16）]
        [E1.1 外推测量门 ✅ 2026-06-12：6 档 + lane warp 指标；三方锚 grad_corr@6m≈0.30，B3 张力否定]
        [E1.2 NTA-IoU ✅ 2026-06-12：interp ≈0.12，@6m ≈0.06，novel 档联动]
        [E1.4 FID KID ✅ 2026-06-12：render 75 → 6m 193 单调，KID 主指标]
        [E0.1 容器冒烟 ✅ 2026-06-11：nre 26.4.146，无 key 全链通]
        [E0.2 USDZ 渲染+修复链 ✅：FID 锚 7.4/57/92，Harmonizer 两档跑通，目视显著]
        [E0.3 自有 clip 官方训练 ✅：40k 步 2h07m，test psnr 30.30（官方口径）]
        [E0.5 配方 diff 清单 ✅：docs/superpowers/specs/ 入档，Top-5 借鉴点]
        [E0.7 β' Harmonizer 取代 Fixer 蒸馏 ✅ 2026-06-15：三方 interpolated psnr 30.30/29.77/29.91（Harmonizer＞Fixer＞baseline−0.4）；外推档第二层待排期]
        [E0.7 IPC difix 对照 ✅ α：B 级 Fixer 蒸馏 40k，C1 interpolated −0.5dB / 目视车道线略好·6m 难接受 / β 定量待 E1]
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
| **E0.4** ★ | E0 | **同 clip 双向对照锚**：NuRec ckpt 与 multilayer baseline 互渲 — interpolated（PSNR/LPIPS/per-class）+ 外推（lateral 3m/6m lane/NTA-IoU/FID，用 E1 工具）双向跑全指标 → **v4 gap 表首行**（量化"差距在哪一层"：表示/配方/修复器） | 新 | 1 | ✅ | **2026-06-12 完成**：时间戳精确对齐（600 映射 max_dt=0µs）+ 反号保险趟实证 offset 语义（grad_corr 0.437 vs 0.045）；**判别数字：lane@3m 官方 +0.05 corr/+3.9dB band_psnr、@6m 两家同崩 ~0.30；interp FID 61 vs 75 官方更干净、interp grad_corr 我方反超（B3 0.739 vs 官方 0.659，full-res+lane loss 近景更锐）**；详见 §5 Done Log |
| **E0.5** | E0 | **配方 diff 清单** ✅（2026-06-11）：官方 resolved 全量（8438 行 parsed.yaml）vs [`ncore_3dgut_mcmc_multilayer.yaml`](configs/apps/ncore_3dgut_mcmc_multilayer.yaml) 11 维度逐项 diff → [`2026-06-11-e05-nurec-vs-multilayer-recipe-diff.md`](docs/superpowers/specs/2026-06-11-e05-nurec-vs-multilayer-recipe-diff.md)。**Top-5**：① road 几何冻结五件套（ground-mesh init + lr 1e-6 冻结 + MCMC 豁免 + z-scale/平整正则，喂 E3.1-E3.3）② road/bg 所有权 init 切分（bg init 剔 road 类点）③ 官方 train-time difix 蒸馏钩子原生＝±3m lateral 增强（本 run 关，sqa_difix_distill 开，喂 E2.2）④ 对锚口径陷阱（官方 val 每 3 帧+1/4 res+cpsnr）⑤ LiDAR ray 级监督 2048 ray/step + 200m 远场 | 新 | 0.5 | ✅ | 顺手发现：`LayerSpec.scale_lr_mult` 死配置（已 spawn 后台任务）；官方 `noise_lr 5000` vs 本项目 5e5 待验证 |
| **E0.6** | E0 | **官方链 actor 插入/取代体验**：在 E0.2 USDZ 场景或 E0.3 自有 clip 重建上走官方编辑工作流——nre actor 编辑（gRPC/CLI 删/移/替）+ `asset-harvester` 资产插入（项目已收割 3 车+3 人直接可用）+ Harmonizer 协调（训练管道③④正为此设）→ 渲染对照 + **官方编辑能力/限制清单**（删除后路面是否出洞 / 插入物阴影来源 / 协调前后 FID） | 新（大g 2026-06-11 提议纳入） | 1 | 🟡 | run-book + 能力清单骨架已入档（[`2026-06-12-e06-official-actor-editing-capability.md`](docs/superpowers/specs/2026-06-12-e06-official-actor-editing-capability.md)）；前置全验证（sequence_tracks ✅ / AH 资产已传 / schema 全文消化）；待 GPU 空档执行 |
| **E0.7** | E0 | **官方 difix-distill 对照 run**（2026-06-12 大g 拍板新增；独立执行 plan [`2026-06-12-e07-official-difix-distill-ab.md`](docs/superpowers/plans/2026-06-12-e07-official-difix-distill-ab.md)）：**E0.3 配方 + `difix.training.enabled=true` 单 key 覆盖**（最小变量——E0.5 实测蒸馏钩子参数已固化在 parsed.yaml：±3m novel poses + p_scheduler + color transfer，唯 enabled=false；`sqa_difix_distill.yaml` 降级为开蒸馏方式的文档性对照）同 clip 9ae151dc 重训 40k → 与 E0.3 锚（difix OFF）单变量对比 → **官方修复器蒸馏增益上限锚**＝E2.2 预期收益的直接校准（Δ@3m vs Δ@6m＝渐进推进收益读数）。权重：① inceptio 已 ngc login，先试官方下 `cosmos_3dgut.pt`（A 级口径）；② 不行走 R-v4.10 hack——B 级 HF [nvidia/Fixer](https://huggingface.co/nvidia/Fixer) / C 级 sd_difix+HF Difix3D，等级标进增益表列头 | 新（E0.5 diff Top-5 ③衍生） | 0.5 | ✅ α+β' | **α 完成（2026-06-12，IPC 方案，权重级 B）**：NGC/hack 权重因 jit.load-vs-state_dict 格式墙不可行 → 大g改 IPC（Fixer server@harmonizer 容器 + nre DifixModel socket 转发，换 server=换修复器）。B 级 HF Fixer 蒸馏 40k（2h19m，单变量审计干净）。**C1 interpolated 全面 −0.5~1.35dB**（psnr 30.30→29.77）；**C3 目视车道线略好·其它持平·6m 难接受** → difix 蒸馏机制有效（车道线正中 v4 lane KPI）但 B 级权重温和局部，显著增益需官方权重+progressive（喂 E2.2）。**β lane/NTA-IoU 定量待 E1.1/E1.2**（USDZ 已落盘）。详见 §5 Done Log。**β'（Harmonizer 取代 Fixer，大g 提议）✅ 完成（2026-06-15，inceptio run `SqPigPNBiA366kkzTMi3EN`，容器 exit 0）**：三方 interpolated 对照 psnr **30.30（baseline）/ 29.77（α-Fixer 二代）/ 29.91（β'-Harmonizer 三代）**——Harmonizer 三代全面略优于 Fixer（+0.14 psnr / car +0.05 / road +0.07 / building +0.15）但都低于无蒸馏 baseline（−0.39，蒸馏优化外推分布、interpolated 付微降代价）；**喂 E2.2 读数**：①代际增益方向 Harmonizer>Fixer（支持 E2.2 用 Harmonizer）②interpolated 代价 −0.4dB 量级 → 蒸馏权重 λ 须小起步。**第二层外推档（两 USDZ 各 `nre render` lateral 3m/6m 出帧 → `eval_frames_dir`）+ β lane/NTA-IoU 定量仍待排期**（USDZ 已落盘，纯渲染回填不重训）。移交方案 [`2026-06-12-e07-harmonizer-as-fixer-handoff.md`](docs/superpowers/specs/2026-06-12-e07-harmonizer-as-fixer-handoff.md)（同 IPC 协议 `harmonizer_server.py` smoke ✅） |
| **E1.1** ★ | E1 | **外推测量门扩展** = **v3 P3.3 移交**：lateral_3m/6m 新档（4 档 avg 口径不变保历史可比）+ lane 区域 novel 指标（路面平面诱导 warp 重投影）+ 三方 ckpt（baseline/B3/aniso20）立锚，顺答 B3 细长高斯外推张力 | v3 P3.3（2026-06-11 立项原文 [`v3_plan_revised.md`](v3_plan_revised.md) §1.2） | 1.5 | ✅ | **2026-06-12 完成**：PR #24 eval 侧移植（R9 中立）+ plane_warp 模块 + 6 档扩档；avg 口径回归 Δ1.5e-5 PASS；三方锚 lane grad_corr@3m≈0.38 / @6m≈0.30 三方打平 → **B3 张力否定、外推退化配方无关**；详见 §5 Done Log |
| **E1.2** ★ | E1 | **NTA-IoU 接入**：按 [`2026-06-10-nta-iou-eval-metric.md`](docs/superpowers/plans/2026-06-10-nta-iou-eval-metric.md) 执行（Task 0–5 全 TDD 已写好）+ **增量**：novel 外推档下也跑 NTA-IoU（渲 lateral_3m/6m 帧→检测→与投影 GT box IoU） | docs/superpowers plan（未执行） | 1.5 | ✅ | **2026-06-12 完成**：interp 0.117/0.120/0.120（三方），@3m 0.076–0.096，@6m 0.054–0.062 单调；口径注记：全 GT 车含远景小目标、YOLOv8m conf 0.3 best-match，绝对值不与论文比 |
| **E1.3** ★ | E1 | **held-out camera 真 GT 外推协议**：训练排除 1–2 台侧相机（`dataset.camera_ids` 覆盖），eval 在被排除相机跑 per-class 全套 → 唯一**有真 GT** 的外推轴（DiFix3D+ RDS cross-reference 协议反用）；需从头训 1 个对照 ckpt | NuRec 调研 § 5.1 | 1.5 | ✅ | **2026-06-12 完成**：4-cam 30k 实际仅 **70 min**（depth-off 轻配方）；三件套（同 exposure-off 口径）——held-out cc 19.16 / guard 25.82 / upper 26.93 → **真 GT 外推差距 7.77 dB**（car class gap 3.5 dB / NTA 0.101→0.071）；guard 25.82 ≈ 5-cam baseline 25.79 → 4-cam 不伤训练相机 |
| **E1.4** | E1 | **FID/KID 接入**：novel 外推档渲染帧 vs 训练视角真图分布的 FID/KID（torchmetrics/clean-fid），写 metrics.json `mean_novel_fid_{mode}` | SOTA 综述（无 GT 外推共识指标） | 1 | ✅ | **2026-06-12 完成**：`--novel-fid` 开关；baseline FID render 75.3 → 1m 124 → 3m 168 → 6m 193 单调（K4 sanity PASS）；KID 主指标（subset 自适应）；**FID render 75 vs 官方场景 7.4 → 自有表示侧伪影重一个量级（E0.2 推论③实证）** |
| **E1.5** | E1 | **v4 gap 表回填**：E0.4 NuRec 锚 + E1.1–E1.4 自有锚汇总入 § 1.3，**据实重排 E2/E3 优先级**（对标 v3 R1 纪律） | — | 0.5 | ✅ | **2026-06-12 重排结论：E3 先行、E2 定位 6m+ 档互补**。证据链：①官方纯表示侧（difix 关）3m lane +0.05/+3.9dB（road 冻结五件套之效）②6m 两家同崩 ~0.30 → 表示侧只能右移退化曲线一档，6m 必须修复链 ③三方锚配方无差 → 差距是结构性配方非调参 ④interp FID 61 vs 75 官方伪影少 ⑤E1.3 真 GT gap 7.77dB。**执行序：E3.1/E3.2 短刀（待 R9，大g暂缓）→ E3.3 BEV；E2.1 spike 低成本并行（域差已被 E0.7 smoke 初步排除）** |
| **E2.1** ★ | E2 | **Harmonizer 升级集成 + 域差 spike**：[`third_party/Fixer`](third_party/Fixer)（一代）→ [NVIDIA/harmonizer](https://github.com/NVIDIA/harmonizer)（Cosmos Predict2 0.6B，时间条件，Apache-2.0）；HF `nvidia/Harmonizer` 权重 → 对 baseline 渲染的 3m/6m 帧离线修复 → E1 指标前后对比（**纯后处理预期：FID/感知大改善、几何指标不动**——正确预期，勿误判失败） | NuRec 调研 § 2.3/5.2 + v3 T15.2 | 1 | ✅ | **2026-06-13 完成**（render-only daefc03/4ac911d · batch-fix cd9fa6b · compare 24b03b0）：render-only 关监督出帧 10.6→4.62s/帧 + harmonizer IPC 批修复 750 帧 + eval_frames_dir 评。**FID −33%/−28% · KID −64%/−56% · NTA +0.033/+0.037 · lane_grad_corr −0.085/−0.095**（raw≡E1锚口径已验）；目视去伪影显著无异物 → **E2.2 GO**（§5 Done Log） |
| **E2.2** ★★ | E2 | **渐进外推蒸馏（v4 核心）**：DiFix3D+ progressive update 移植——外推位姿从 1m→2m→3m→6m 逐步推进，每步「渲染→Harmonizer 修复→修复帧按低权重蒸馏回 3D（road/lane 区域加权）→下一步」；区别 v3 Stage 15 教训：不打全图 repro 轴，蒸馏目标=外推档 + road/lane 病灶区 | NuRec 调研 § 2.4（ablation 证据）+ v3 Stage 15 复活改轴 | 2.5 | ⬜ | gate=E2.1 spike + E1 锚；验收=E1 全指标 |
| **E2.3** | E2 | **actor 弱观测面修复蒸馏**：对车辆 track object-centric 环绕渲染弱观测面 → Harmonizer 修复 → cuboid×sseg mask 内低权蒸馏；攻 P1.4 验尸根因（未观测面缺约束）的 2D 监督解法 | v3 P1.4 否定结论 + SOTA 共识（2D 监督非 3D 注入） | 2 | ⬜ | gate=E2.1；验收=class_psnr + NTA-IoU + 守护线 |
| **E2.4** | E2 | （备选）**Harmonizer 域内微调**：若 E2.1 spike 显示 NCore 域差大——按 DiFix3D+ 降质构造法（cycle reconstruction 横移 1–6m / model underfitting / cross reference）在自有 clip 造配对，LoRA 级微调 | DiFix3D+ 论文 § 训练数据构造 | 2 | ⬜ | 仅 E2.1 域差坐实才投 |
| **E2.5** | E2 | **编辑协调 spike（3dgrut2 侧）**：复用 AH 注入引擎（PR #18 plumbing / frozen 离线手术）在自有 ckpt 插入/取代 1–2 辆 asset-harvester 车 → Harmonizer 时间模式协调 → NTA-IoU/FID + 目视验收；**不训练或轻训练**——区别 P1.4 warm-start（重建轴）：本卡是编辑场景 + 生成协调，正是 NuRec 官方编辑形态 | 新 e25_inject.py + scripts/e25_inject_ah_replace.py（0350e34/21cf1c4） | 1.5 | ✅ 目测 | gate=E2.1 + E0.6 清单；**2026-06-17 目测 spike 完成**：3 AH 车 frozen 注入取代 recon automobile（'316'/'372'/'24'）+ viser+harmonizer 实时协调，目测 harmonizer 协调**有效但有限**（违和感降低、优于无 harmonizer、未完全自然）；定量 NTA-IoU/FID 按大g 决定跳过留 v5。v5 编辑轴第一块基石 |
| **E2.6** ★ | E2 | **viser_gui_4d 交互 difixer 升级（temporal 后处理提 inference 质量）**：[`viser_gui_4d.py`](threedgrut_playground/viser_gui_4d.py) 现 `--difix_server` 接 Fixer 一代（单帧）→ 换 DiffusionHarmonizer 三代**时间模式**（`diffusion_harmonizer.pkl`，回读前 K 帧已修复输出做时序参考）→ 对交互渲染的连续帧序列后处理，用帧间时序一致性提升 inference 视觉质量（去闪烁/运动连贯）。复用 E0.7 IPC server 架构（`fixer_server.py`→harmonizer temporal server）+ viser_gui_4d 已有 `--difix_server` 钩子。**唯有连续渲染序列能发挥 Harmonizer 时序优势**（区别 E2.1 离线单帧 / E2.2 训练蒸馏必 nontemporal）| 新（2026-06-12 大g 提议）| 1 | ✅ | **2026-06-15 完成，inceptio 端到端目测通过**（commits 9de9101/5cf26e7/4058704/cce14ba）。设计：client-side K-deque（history 在 client、server 无状态、seek=clear 自然 reset）；HMN1 协议独立于 DFX1（守护 E1 锚）；`_on_time_change` 的 `source!="play"` 自动 reset 历史。**关键修复（cce14ba）**：Harmonizer temporal Conv3d kernel=3 只接受 V=1 或 V≥3，冷启动逐步增长（V=2,3,4）会 crash → 改为 history 满了 K 才发 V=1+K，否则 V=1（对齐官方 `have_history = len >= min_history`）。实测 V=5 延迟 ~1000ms（5 帧 0.6B），K=2 可降到 ~600ms 备选。详见 §5 Done Log |
| **E2.7-B** | E2 | **viser_gui_4d 加载 NVIDIA usdz dynamic_rigids 接线**（E2.7 跟进；让 NRE 训练的车辆动态对象也能在 viser 里渲染 + 随 timeline 移动）：移植 fervent-knuth-d25fe9 873L 的 Stage 2 helpers（pose7_to_mat / resample_track_to_timeline / parse_volume_usda_track_order with NRE Lightning lax fallback / parse_sequence_tracks）到 amazing-lalande 简版 loader；main() USDZ 模式 + metadata 加载之后注入 dynamic_rigids 三件套（`dyn_layer.track_ids` buffer + `engine.scene_mog.populate_tracks(tracks_dict)` + `metadata.tracks/tracks_camera_timestamps_us`），让 LayeredGaussians.`_transform_means_and_active` 在 render_pass(timestamp_us) 里把 89,285 个 object-local gaussians 用 track_pose@local 变换到 NCore world frame。**关键 frame-of-reference 规则（大g insight）**：NRE state_dict 内 background/road gaussians 在 NRE local frame **要 apply +translate**；sequence_tracks.json 内 cuboid poses 跟 3dgrut2-own viz_4d.tracks 同源在 **NCore world frame 不 translate**（铁证：tid='18' parked 车两边都是 (-51.30, 1.07~1.12, 1.42~1.47) 同 frame） | 新（2026-06-15 大g 任务）| 0.5 | ✅ | **2026-06-15 完成**（commit 7e5edac）。**架构层接通完成**：cuboid wireframes 跟 dyn gaussians 同 source pose 跟随 timeline 显示在车的真实位置（大g 视觉验证 t405 bus 位置 OK + 人行道/十字路口可见物体跟着时间移动）。**遗留：dynamic_rigids gaussian 颜色呈"烟雾"感**（features_albedo Fourier→SH 数学不兼容，NRE Fourier-in-time 取 DC 后等于时间平均 ≈ 灰色，丢失车辆色彩细节）→ E2.7-C follow-up |
| **E2.7-C** | E2 | **dyn features_albedo Fourier→SH 转换 + 渲染色彩修复**（E2.7-B 烟雾感跟进）：根因 cross-source 对比锁定——3dgrut2-own dyn features_albedo (N, 3) 是 SH 空间 DC band 系数 (std=1.5, range=[-2.9, +13.7])，NRE dyn features_albedo (N, **20**, 3) 是 Fourier-in-time 时间系数 (DC 占 622% 能量但 std=0.42 range=[-1.9, +2.2])，**两套数学不兼容**。amazing-lalande loader 直接取 NRE DC[0] 当 3dgrut2 SH DC，bg/road 巧合数值范围接近所以 OK，但 dyn 车辆色彩随光照/夜间/阴影变化大 → DC = 时间平均 ≈ 灰色 = 烟雾。可选路径：(a) NRE Fourier 取 sequence 中点 frame_id evaluate 拿单帧色彩（最简单，仍 static 但比 DC 鲜艳）(b) 跨 ckpt 借 3dgrut2-own dyn features by tid match（89k vs 300k 不同数量 KNN/cluster 工作量大）(c) LayeredGaussians 内部加 Fourier-in-time albedo 渲染路径每帧 evaluate（最正确但侵入 render path）| 新（2026-06-15 E2.7-B 衍生）| 1.5 | 🔴 | **2026-06-16 尝试失败，方向需重新分析**（见 §5 Done Log 2026-06-16）。path A/C 实现了 Fourier→per-frame 颜色（分支 `e27c-dynamic-color-path-a`，未合并），但**真 blocker 是形状不是颜色**：dynamic gaussians opacity 0.11 + 极端各向异性 + 稀疏 → 在 3dgut renderer 渲成 smeary 半透明条状雾，不成形。top suspect=旋转约定（片状 gaussian 朝向）。颜色/亮度(CRF ISP)是下游问题 |
| **E2.7** ★ | E2 | **viser_gui_4d 加载 NVIDIA usdz checkpoint（同 clip 三方视觉对标工具）**：[`viser_gui_4d.py`](threedgrut_playground/viser_gui_4d.py) 加 `--usdz <path>` 入口，启动时自动 USDZ→3dgrut2-native `.pt` 转换并 apply USDZ 容器内 `rig_trajectories.world_to_nre` 坐标变换（NRE→NCore world 平移 +38m x for 9ae151dc），透明走现有 `--gs_object` 加载路径。**用途**：浏览器同时开两个 viser tab，用同一套 UI/相机/timeline 把 NVIDIA NRE 训练产物（E0.3 last.usdz）和 3dgrut2 自家产物并排做视觉对标 — 直接看 NuRec 路面建模、横向外推退化、actor 渲染等差异，比指标定量更直观。复用 amazing-lalande 简版 loader（472L+169L tests，14 tests Mac+inceptio 全绿）。**关键技术发现**：USDZ 容器内 `rig_trajectories.json:world_to_nre.matrix[:3, 3]` 是 NRE 训练时主动 apply 的坐标变换；正确 align = NRE 位置 + `-world_to_nre.translation`。Plan: [viser-gui-4d-py-nvidia-virtual-toucan.md](../../.claude/plans/viser-gui-4d-py-nvidia-virtual-toucan.md) | 新（2026-06-15 大g 任务）| 0.5 | ✅ | **2026-06-15 完成，inceptio 端到端视觉对标通过**（commits 19ffd3d/00c8b8c）。**视觉观察（大g 360° + 3m/6m 横向漂移测试）**：NRE 路面建模质量明显优于 3dgrut2 自家训练 — 横向 3m/6m 漂移视图都不出现 lane_grad_corr aperture problem 的明显退化，360° 视角转动路面纹理稳定 → 直观证实 E2.2（progressive distillation）+ E3.3（BEV 纹理平面化）走 NRE 配方的方向是对的。**6 处子修复**：① world_to_nre 平移修复（量级 18× 方向也错）② loader 强制 `conf.use_layered_model=True` ③ `_load_metadata` NCoreDataset signature 修复（T8.6 dormant bug）④ `metadata.n_frames` callable 兼容 ⑤ Reset View + frustum 用 `--initial_cam_id` cam 不用 NCore primary（cross_left 下视鱼眼）⑥ stale JIT FileBaton lock 自动清理（PyTorch upstream issue #9711，pkill -9 留下的 lock 文件让所有后续进程 polling 死循环，连重启都修不掉）。详见 §5 Done Log |
| **E2.8** ★ | E2 | **系统性 dynamic rigid 全替流水线（单 clip scene factory）**：E2.5 手术式换 3 车 → 系统化全 vehicle track 成批换 AH 资产库（class+size 最近 + fallback ladder，一资产可服务多 track）；USDZ拆（deformable 丢弃）→ 全替 → harmonizer 协调 → QA 闸（sanity + NTA-IoU/FID）。新增 `asset_bank.py`（bank 查询）/`e28_replace.py`（全 track 枚举+批 align+守护）/`nre_usdz_viz4d.py`（USDZ→可渲染 ckpt+viz_4d，移植 fervent-knuth rig/viz4d 接线到 checkpoint.ckpt flavor）/`scripts/e28_systematic_replace_pipeline.py`（driver）；复用 E2.5 e25_inject/warmstart_*、E2.7 nre_usdz_loader、E2.1/E2.6 harmonizer、E1.2 NTA、E1.4 FID | spec [`2026-06-17-e28-...-design.md`](docs/superpowers/specs/2026-06-17-e28-systematic-rigid-replacement-pipeline-design.md) + plan（E2.5 升级）| 2.5 | ✅ | **2026-06-17→18 6 阶段端到端**（6144f23 vendor E2.5 / 946147c bank / 60103e6 assign / 8dc4d0a replace / cd02856+05e8474 qa / 321926f driver / **f0d9aca 坐标 +translate** / **adc2a5c slot-basis 铁律** / 1a03ceb viser vehicle 过滤 / 215a50e replace→insert / e6d2e4e 跨源 recon bus / e27452d out_name / 09311b8 drop 非vehicle / b55ba3b 跨源 MLP 天空 / 2436a0e handoff）：**inceptio 实测 coverage=1.000、20 AH automobile 全替 + 1 跨源真 bus recon t405（21006 gaussians，朝向 0.99）、opacity_med 0.10、QA passed、viser+harmonizer K=4 RTT~1s 0 OOM 目测干净**；**35 E2.8 测 + 6 Task6 quant + 2 render 回归（_override_conf_path）全绿**。**viser 实测抓+修真 bug**：viz_4d tracks numpy→torch；slot-basis（dyn track_ids slot→tid 走 sorted(viz_4d.tracks.keys())，修最初「cuboid≠asset 90°」）；坐标（bg/road +translate，ego/track 不变换）。**Task6 定量 raw ✅**（scripts/e28_quant_qa.py 编排 render→eval_frames_dir→qa_report.json；inceptio n=75/档：NTA-IoU 0.085 / novel FID 233 / KID 0.259；肉眼验证非坏帧，lateral bg/road 离轴涂抹主导、几何有效；**harmonizer before/after ✅ mean FID 233→208（−25，6m −43），「FID after<before」验收 ✅**）；**Task7 bus AH 收割 ✅**（asset-harvester Workflow N：收 bus/405 干净 AH 公交 5.2M + truck/165 入 bank 6→8；重跑 driver **AH-match 21 / 跨源 recon 0**——bus+truck size ratio≤1.5 自动 recon→AH、driver 零改码、QA passed、packed_ckpt_busah.pt 1.1G） | gate=E2.1+E2.5+E2.6+E1.2/E1.4 ✅；验收=coverage 100% + NTA≥recon + FID after<before + viser 目测干净 |
| **E3.1** | E3 | **空气区 penalty** = **v3 P3.4 移交**：路面上方 0.4m~上界悬浮 bg opacity penalty（cuboid actor 豁免），复用 V3-R2 基建 | v3 P3.4 | 1.5 | ⬜ | gate=E1.1 锚 + **R9（PR #24 去留先决）** |
| **E3.2** | E3 | **road SH 降阶 DC-only（freeze 法）** = **v3 P3.5 移交**：砍 view-dependent 过拟合逃逸通道（路面近似 Lambertian） | v3 P3.5 | 1 | ⬜ | gate 同 E3.1 |
| **E3.2.5** ★ | E3 | **几何侧硬退化路面（reconstruction-studio 实证路径）**：road 高斯压成真·零厚度水平 disk（z-scale→~1mm floor，非现状 5cm clamp）+ **强制单位旋转锁法线竖直** + position/z-scale/面内旋转梯度冻结 + 颜色保持 DC；前提=先把 road init 提质（LiDAR 累积测量点 / 局部 KNN 中值-平面拟合 Z，替代规则网格单点吸附）。**几何侧根治 aperture problem**，比 E3.3 BEV 纹理轻（纯参数/mask 级，不改渲染路径）。依据：reconstruction-studio 产物 91,008 ground disk（厚 1µm / 法线 100% 竖直 / 点距 10cm / 局部起伏 8mm）横移旋转稳达/超 NuRec；并解释 roadoff freeze 失败=init 不够+未锁法线+未真薄。spec [`2026-06-22-e325-recon-studio-ground-disk-geometry.md`](docs/superpowers/specs/2026-06-22-e325-recon-studio-ground-disk-geometry.md) | reconstruction-studio 交叉分析（2026-06-22 session）| 1.5 | ✅ spike | **6k A/B ✅（代码 commit `2374161`）**：on（roaddisk）vs off（multilayer）inceptio depth-off 单变量 — 守护线 cc 24.01 vs 23.60（**+0.41 不退反升**）/ lateral lane grad_corr +0.057@3m +0.039@6m / band_lpips 降；freeze 铁证 rotation tilt 0°·z 1mm·N 200000 恒定（recon-studio 对齐）；viser 视觉确认 on 路面锐利斑马线分明；**反驳 roadoff「光冻结变差」**（补齐 init+薄盘+法线锁后冻结转改善）。30k 全量待排期；发现白条被 bg 抢 → E3.2.6 follow-up |
| **E3.2.6** | E3 | **road-takeover 调强 spike**（E3.2.5 viser 视觉 follow-up）：roaddisk viser 对比发现车道线白条被 bg「抢走」（road-only 视图缺白条、开 bg 才补上 → 白条渲染权在 bg 不在 road）。调强现有 `bg_road_penalty`（λ 0.1→0.4 / z_band 0.4→1.5）跑 A/B，验证 takeover 够强能否把白条从 bg 收回 road。driver [`scripts/e325_takeover_ab.sh`](scripts/e325_takeover_ab.sh) 已就位；对照 `e325_g2_on`（λ0.1）。**E3.1（完整空气区 penalty）的轻量前菜** | E3.2.5 viser 视觉 follow-up（2026-06-23 大g） | 0.5 | ⬜ | gate=E3.2.5 ✅；验收=road-only 白条收回 road + lateral lane 不退 + 守护线 cc 别过压 |
| **E3.3** ★ | E3 | **BEV 纹理平面化**（v4 backlog 转正）：road 颜色不再 per-gaussian SH，改 BEV feature grid/纹理图采样、真正贴在高度场平面 → **外推天然正确**（参数化级根治 aperture problem；ExtraGS Road Surface Gaussians 同思路）；复用 [`road_region.py`](threedgrut/model/road_region.py) BEV 网格基建 | v3 § 5 backlog「外推终极方向」 | 3 | ⬜ | gate=E1 锚 + E3.1/E3.2 结果（短刀够用则缓）。**【2026-06-22 out-of-plan /loop 反哺】road-freeze 控制变量实测证伪「光冻结几何」——冻结 noisy BEV-KNN init 反令 off-track grad_corr 全档变差 −0.05~0.10 → 坐实 ground-mesh init 是冻结生效的前提（高质量 init 才是第一性瓶颈）；per-layer lr 冻结 + MCMC 豁免机制已就位（PR #34），等本任务好 init 即可配套。详见 §5 Done Log** |
| **E3.4** | E3 | （备选）**平面诱导 warp 伪横移一致性 loss**：训练时按路面平面 homography warp 伪横移视角做一致性约束 | v3 § 5 backlog 备选 | 1.5 | ⬜ | E3.3 的轻量替代/前菜 |
| **E4.1** | E4 | （可选）**LiDAR 点云推理**：按 [`2026-06-10-lidar-pointcloud-from-gs.md`](docs/superpowers/plans/2026-06-10-lidar-pointcloud-from-gs.md) 执行（A0 gate：3DGUT ckpt 能否被 3DGRT 渲 → A1 射线表 → A2 range-L1/出 .ply）；外推的传感器维度（novel 轨迹渲 LiDAR），对标 NuRec LiDAR re-sim | docs/superpowers plan（未执行） | 2.5 | ⬜ | A0 NO-GO 则整线作废（plan 内置判据） |

### 1.3 Phase 状态汇总 + v4 gap 表（E0/E1 回填）

| Phase | 主题 | 任务数 (Done/Total) | 主验收 | 守护线 | 状态 |
|---:|---|---:|---|:---:|:---:|
| **E0** ★ | NuRec 工具链复现立锚（**首要**） | 5/7 | ≥2 场景跑通 ✅ + NuRec 锚 ✅ + 配方 diff ✅ + **双向对照 ✅（E0.4 判别数字入 gap 表）** + difix-distill 增益锚 ✅α（E0.7：B 级 Fixer 蒸馏，车道线略好 / interpolated −0.5dB / β 定量待回填）+ 官方编辑能力清单（E0.6 🟡）+ 修复器代际 β' ✅（**2026-06-15 完成**：interpolated 三方 30.30/29.77/29.91，Harmonizer>Fixer 均低于 baseline−0.4；外推档第二层待排期） | — | 🟡 |
| **E1** ★ | 外推测量门（gate 后续一切） | **5/5 ✅** | 3m/6m ✅ + NTA-IoU ✅ + FID/KID ✅ + held-out ✅（真 GT 差距 7.77 dB）+ gap 表收口 ✅（E1.5 重排：E3 先行） | interpolated 全指标不退（已验：avg Δ1.5e-5 / cc 25.79 / grad_corr 0.6931 三点零回归） | ✅ |
| **E2** | 生成修复链（NuRec 思路移植）+ 编辑协调 spike + viser temporal 后处理 + viser USDZ 视觉对标 + dyn rigids 接线 + 系统性全替流水线 | 7/10（含 1 备选；E2.8 ✅ 全替+定量+建库） | 同左；**E2.8 ✅ 系统性全替（6 阶段）：USDZ拆→20 AH automobile 全替 + 1 跨源真 bus recon t405→drop 非vehicle→MLP 跨源天空→QA 闸 coverage=1.0 opacity_med 0.10 passed→viser+harmonizer K=4 目测干净（41 测绿：35 e28 + 6 quant；slot-basis/坐标两修）；定量 ✅（NTA 0.085 / novel FID 233；harmonizer before/after FID 233→208 −25、6m −43、「after<before」✅；bg 离轴退化主导、E3.5 floater-prune 续）、bus AH 收割 ✅（重跑 driver AH-match 21 / 跨源 recon 0）**；**E2.1 ✅ 离线 Harmonizer：FID −30%/KID −60%/NTA +35%**；**E2.5 ✅ 目测 spike：harmonizer 协调有效但有限（违和感降低、未完全自然）**；**E2.6 ✅ inceptio 目测通过（V=5 temporal）**；**E2.7 ✅ viser_gui_4d 加载 NVIDIA usdz：路面横向 3m/6m + 360° 不退化**；**E2.7-B ✅ dynamic_rigids 接线（cuboid wireframes + dyn gaussian 位置贴车随 timeline 动，commit 7e5edac；颜色烟雾感转 E2.7-C）**；E2.7-C ⬜ dyn features_albedo Fourier→SH | cc ≥ 24.7 / grad_corr 0.744 不退 | 🟡 |
| **E3** | 表示侧外推强化（与 E2 互补） | 1/5（含 1 备选） | 同 E2 验收口径；E3 减伪影产生、E2 修残余；**E3.2.5 ✅ spike 几何硬退化 disk：6k A/B on>off（cc +0.41 / lateral lane grad_corr +0.04~0.06 / freeze 铁证 rot 0°·z 1mm·N 恒定）→ 反驳 roadoff「光冻结变差」；viser 视觉确认 on 斑马线分明，发现白条被 bg 抢 → E3.2.6 takeover 调强 follow-up** | 同上 | 🟡 |
| **E4** | LiDAR 外推（可选） | 0/1 | A0 GO + range-L1 入档 | — | ⬜ |
| **总计** | — | **11/24** | — | — | — |

> **v4 gap 表（E0.4 + E1.5 回填，格式预置）**：
> | 轴 | 3dgrut2 锚（E1） | NuRec 锚（E0.4） | 差距 | E2/E3 后 |
> |---|---|---|---|---|
> | lane grad_corr @ 3m / 6m（warped 口径） | **B3 0.381 / 0.307**（baseline 0.384/0.303，aniso20 0.389/0.298——三方打平；对照 interp 0.74 → 外推腰斩再腰斩） | **0.437 / 0.297**（band_psnr 16.34/13.45 vs 我方 12.47/11.33——**3m 档官方 +0.05 corr/+3.9dB，6m 档两家同崩**） | 3m：表示侧差距实证；6m：无差 | **E2.1 后 fixed 0.299 / 0.208（Δ−0.085/−0.095）**：扩散平滑伤高频车道线，E2.2 蒸馏待修 |
> | NTA-IoU @ 原轨迹 / 3m / 6m | **B3 0.120 / 0.076 / 0.054**（baseline 0.117/0.096/0.062；口径＝全 GT 车含远景，绝对值不与论文比） | **0.126 / 0.087 / 0.044**（interp 略高；外推档与我方交错——样本小，仅观察不下结论） | ≈持平 | **E2.1 后 fixed 0.128 / 0.100（Δ+0.033/+0.037 ✅）**：修复改善车辆检出 |
> | held-out cam 真 GT 外推（cross_left，exposure-off 口径） | **gap 7.77 dB**（heldout cc 19.16 vs upper 26.93）；car class gap 3.5 dB；NTA 0.101→0.071；guard 25.82 ≈ baseline 25.79 | 待测（E0.4 可选：NuRec 同协议 4-cam 重训成本高，暂不对称） | — | — |
> | FID/KID @ 3m / 6m | **B3 FID 165/192 · KID 0.081/0.098**（baseline 168/193 · 0.082/0.102；render 75.3/0.021；**口径＝5 相机混合分布**） | lateral：FID 217/237 · KID 0.230/0.265（**口径＝前视单相机 75 帧，与我方 5 相机口径不可直接比**）；**interp FID 61.3 vs 我方 75.3（同口径可比 → 官方伪影更少）** | interp 感知：官方更干净 | **E2.1 后 fixed FID 113/139（−33%/−28%）· KID 0.029/0.045（−64%/−56%）✅** |
> | interpolated（守护线） | class 25.07 / cc 26.06 / grad_corr 0.744 | **官方口径**：psnr 30.30 / cpsnr car 34.59 · road 38.27 · person 32.65 / chamfer 0.295（E0.3，**口径未统一不可直接比**，E0.4 重算） | 待 E0.4 | 不退化 |

### 1.4 任务依赖图

```mermaid
flowchart TD
  classDef gate fill:#e6f4ff,stroke:#0070f3,color:#000,font-weight:bold
  classDef todo fill:#f5f5f5,stroke:#999,color:#333
  classDef opt fill:#fbe9e7,stroke:#d33,color:#900

  v3["v3 成果（继承）<br/>class 25.07 / lane 0.744 / 外推诊断 / NTA-IoU plan"]:::gate
  E0["E0 NuRec 复现立锚 ★首要<br/>E0.1 环境 → E0.2 USDZ渲染+修复链 → E0.3 自有clip训练<br/>→ E0.4 双向对照锚 + E0.5 配方diff + E0.6 编辑体验<br/>+ E0.7 difix蒸馏对照 ✅α+β'（IPC方案／Fixer+Harmonizer两代／Harmonizer略优·均低于baseline）"]:::gate
  E1["E1 外推测量门<br/>E1.1 三米六米档（＝P3.3） / E1.2 NTA-IoU / E1.3 held-out<br/>/ E1.4 FID-KID → E1.5 gap表+重排"]:::gate
  E2["E2 生成修复链<br/>E2.1 Harmonizer升级spike → E2.2 渐进外推蒸馏 ★★<br/>→ E2.3 actor弱面蒸馏 / E2.5 编辑协调spike（E2.4 微调备选）"]:::todo
  E3["E3 表示侧强化<br/>E3.1 空气区penalty（＝P3.4） / E3.2 road DC-only（＝P3.5）<br/>→ E3.2.5 几何硬退化disk（recon-studio实证）→ E3.3 BEV纹理平面化（E3.4 warp备选）"]:::todo
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
| E0.7 | 按独立 plan [`2026-06-12-e07-official-difix-distill-ab.md`](docs/superpowers/plans/2026-06-12-e07-official-difix-distill-ab.md) 执行（Task 0–5）：E0.3 配方 + `difix.training.enabled=true` 单 key 覆盖同 clip 重训 40k（蒸馏钩子参数已固化于 parsed.yaml：±3m / p_scheduler / color transfer）→ vs E0.3（OFF）单变量对比（parsed.yaml diff 审计入档）→ **官方蒸馏增益上限锚**（E2.2 校准；Δ@3m vs Δ@6m＝渐进推进收益读数）。权重路径：先试官方 ngc 下载（A 级），不行 hack——B 级 HF Fixer 预放 cache / C 级 `difix=sd_difix`+HF Difix3D；state_dict 探针先行 + 300 步 smoke（start_step=50 强迫早 fire）再烧全量，等级标进增益表列头 | **✅ α 完成（2026-06-12，IPC 方案，权重级 B）**：NGC/hack 权重因 jit.load-vs-state_dict 格式墙走不通 → 大g改 **IPC**（Fixer server@harmonizer 容器 + nre `DifixModel` socket 转发，换 server=换修复器）。C1 interpolated −0.5dB（psnr 30.30→29.77）/ 目视车道线略好·6m 难接受 → difix 蒸馏机制有效（正中 v4 lane KPI）·B 级权重温和局部·显著增益需官方权重+progressive（喂 E2.2）。β lane/NTA-IoU 定量待 E1.1/E1.2。详见 §5 Done Log |

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
| E2.6 ★ | **viser_gui_4d 交互 difixer 升级 = temporal 后处理提 inference 质量（2026-06-12 大g 新增）**：[`viser_gui_4d.py`](threedgrut_playground/viser_gui_4d.py) 现 `--difix_server` 接的 Fixer 一代（单帧，via [`correction/difix.py`](threedgrut/correction/difix.py)）→ 换 DiffusionHarmonizer 三代 **时间模式**（`diffusion_harmonizer.pkl`，回读前 K 帧已修复输出做时序参考）→ 对交互渲染的连续帧序列后处理，用帧间时序一致性提升 inference 视觉质量（去闪烁 / 运动连贯）。复用 E0.7 IPC server 架构（`fixer_server.py` → harmonizer temporal server，回读历史输出做参考帧）+ viser_gui_4d 已有 `--difix_server` 钩子。**为何独立于 E2.1/E2.2：E2.1 离线单帧修复、E2.2 训练蒸馏必 nontemporal（随机单 novel view 无时序可言）——唯有交互 viewer 的连续渲染序列能发挥 Harmonizer 的时序优势，这正是 Harmonizer 三代相对 Fixer 二代的核心代际升级（视频级一致性）的用武之地**。验收：viser_gui_4d 下 temporal Harmonizer vs Fixer 单帧的帧间一致性（去闪烁目视存档 + 可选时序抖动指标）；gate=E2.1（Harmonizer 集成）；纯 inference 后处理、不依赖训练 |

**验收**：E1 外推指标（lane@3m/6m + NTA-IoU + FID/KID + held-out per-class）相对锚改善；守护线不破（cc ≥ 24.7 / interpolated grad_corr、class_psnr 不退）；**双协议验收防幻觉**（R-v4.5）：held-out 真 GT 指标与无 GT 感知指标须同向改善。

### 2.3 Phase E3 — 表示侧外推强化（与 E2 互补）

**触发**：E1 锚点 + R9（PR #24 去留）落定。
**核心**：E2 修"已产生的伪影"，E3 减少"伪影的产生"——对应 NuRec 的表示侧（LiDAR 强监督、mesh、配方）。E3.1/E3.2 是 2026-06-11 诊断的"第 1 层短刀"（原 v3 P3.4/P3.5 移交），**E3.2.5 是几何侧根治（reconstruction-studio 实证、轻量，2026-06-22 加入）**，E3.3 是颜色参数化级根治（重，作后备）。

| Task | 改动文件 / 锚点 |
|---|---|
| E3.1（=P3.4 移交） | [`road_region.py`](threedgrut/model/road_region.py)、[`trainer.py`](threedgrut/trainer.py)、multilayer yaml：路面上方 0.4m~上界（A/B 定）空气区 bg opacity penalty，cuboid 内 actor 豁免；复用 V3-R2 height field / `query_ground_z` / cuboid mask 全套 |
| E3.2（=P3.5 移交） | [`layered_model.py`](threedgrut/layers/layered_model.py)、[`registry.py`](threedgrut/layers/registry.py)：road 层高阶 SH zero+freeze（保 45 维宽度一致，绕 V3-R1.1 fused SH 坑）。注：2026-06-11 已把死配置 `scale_lr_mult` 接线（`_apply_scale_lr_mult`，registry 默认 1.0 保锚点等价）——E3 做官方式 road scales lr 冻结（5e-3→1e-4）直接 `++layers.overrides.road.scale_lr_mult=0.02`，无需新代码；positions 1e-6 冻结仍需另做绝对值 override 机制 |
| E3.2.5 | road 几何硬退化（reconstruction-studio 路径）：① **init 提质** [`road_init.py`](threedgrut/layers/road_init.py) Z 吸附从「最近单点」改局部 KNN 中值/平面拟合（攻 roadoff「init 质量」瓶颈）；② [`registry.py`](threedgrut/layers/registry.py)/[`layer_spec.py`](threedgrut/layers/layer_spec.py) road z-scale 硬压 ~1mm floor（非 5cm clamp）+ **强制单位旋转锁法线竖直**；③ position+z-scale+面内旋转梯度冻结（复用 `scale_lr_mult` override 机制扩到 position/rotation 绝对冻结——正好接上行 E3.2 尾「positions 1e-6 冻结仍需另做 override」）；④ 颜色保持 DC（E3.2 已做）。**3DGUT 数值风险**：零厚度协方差用 1mm floor 不用 1µm；**freeze 前提=① 先达标**（否则重蹈 roadoff「冻结变差」覆辙）。**先 spike ①+②③ 的 A/B（横移 3m/6m）再全量** |
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
| R-v4.1 | ~~nre 容器运行时 NGC key 需求未验证~~ **已关闭（2026-06-11 实测）**：无 NGC key 跑通 validate→train（40k 步）→render→USDZ 导出全链；唯一确认需要 key 的点＝官方 train-time difix 蒸馏权重 `cosmos_3dgut.pt`（NGC API URL、HF 无副本，仅 `sqa_difix_distill` 配方用到——E2 用开源 Harmonizer 自研蒸馏绕开） | — | — | **2026-06-12 补充（大g 实证）**：官方 difix-distill 的无 key 路径已走通——`fixer_server.py`（harmonizer-cosmos-env 容器，socket IPC :59487）挂 HF 开源 `nvidia/Fixer` 权重替代 NGC `cosmos_3dgut.pt`，官方训练 `difix.training.enabled=true` 正常蒸馏；E0.7 在同协议上换 Harmonizer 做代际 A/B | E0 ✅ |
| R-v4.2 | inceptio 4090 24GB 低于官方推荐显存（24–48GB+） | E0.3 官方配方 OOM | 训练复现受阻 | 降分辨率/相机数先跑通；`vast-train` skill 起 48GB 卡（成本 ~$0.5/h，5k smoke 级） | E0.3 |
| R-v4.3 | 自有 clip 与 nre 26.x 的 NCore 版本兼容性（项目 clip 较旧 vs NCore 2026.04） | E0.3 数据加载报错 | 同上 | 先 `ncore` skill validate；不兼容则用官方 PhysicalAI 场景完成 E0.3/E0.4（锚点换数据，对照价值略降但仍立得住） | E0.3 |
| R-v4.4 | Harmonizer 域差（NVIDIA 车队数据 post-train vs NCore clip 相机/ISP） | E2.1 spike 修复质量差/引入异物 | E2 全线打折 | E2.1 先 spike 实测再投 E2.2/2.3；域差坐实走 E2.4 微调（DiFix3D+ 降质构造法已备） | E2 |
| R-v4.5 | **幻觉污染评测**：蒸馏的是扩散模型生成像素，指标升 ≠ 更真实 | E2.2/E2.3 验收 | 自欺 | **双协议验收**：held-out 真 GT 指标（E1.3）与感知指标（E1.4）须同向；目视存档强制；蒸馏权重 λ 从小起 | E2 |
| R-v4.6 | PR #24 / road 基建多任务交叠（继承 v3 R9 + P3.2 同域） | E3 开工 | road spec/yaml 双源漂移、合并冲突 | **先定 PR #24 去留再开 E3**；E3 与 v3 P3.2 不同期动 road 文件 | E3 |
| R-v4.7 | 渐进蒸馏破坏 interpolated 质量（修复帧与真图监督打架） | E2.2 训练 | 守护线破 | λ_distill 低权重起步 + 外推帧只在病灶区域加权 + 守护线全程监控（v3 全套 interpolated 指标在同一 metrics.json） | E2.2 |
| R-v4.8 | E1.3 held-out 协议与 5-cam ring 配方互斥（少相机训练本身改变 baseline） | E1.3 立锚 | 锚点口径混乱 | held-out 对照独立成线（自己 vs 自己），不与全相机 baseline 跨协议比较；文档写死口径 | E1.3 |
| R-v4.9 | Harmonizer license 合规（权重 NVIDIA Open Model License；PhysicalAI 数据集限制性专有） | 商用/再分发场景 | 合规风险 | 代码 Apache-2.0 无虞；权重许可允许商用但需保留条款；PhysicalAI 数据**仅限内部开发评测**、不入训练数据、不再分发 | E0/E2 |
| R-v4.10 | **E0.7 的 cosmos-difix 权重不可得**：`cosmos_3dgut.pt` 仅在 NGC（`nurec-fixer/versions/cosmos_3dgut`，HF 无副本，E0.5 实测确认），NGC key 拿不到或权限不足则官方蒸馏配方跑不起来 | E0.7 启动 | E0.7 阻塞 | **hack 缓解（大g 2026-06-12 指定）**：用开源 [HF nvidia/Fixer](https://huggingface.co/nvidia/Fixer) 权重塞进 nre pipeline——① 先试容器自带 legacy 变体 `difix=sd_difix`（SD 架构加载器，与 HF Fixer 同源概率高）；② HF 权重预放 `~/.cache/nre/difix/cosmos_3dgut.pt` 绕过 NGC 下载（`difix.model_url` 仅在 cache miss 时拉取）；③ 加载不兼容则对齐 state_dict / 改 `difix` 配置段指向本地。⚠️ hack 版增益锚口径与官方 cosmos-difix 不完全等价（架构/post-train 数据不同），入档时须标注权重来源。**2026-06-12 更新**：大g确认 inceptio 已 ngc login——先试官方下载一次（403/404 证据入档），预期权限不足再走 hack；α/β 拆分已拍板（硬 gate＝权重，E1.1/E1.2 仅 gate β 段指标回填）；权重等级 A/B/C 口径与全套步骤见独立 plan [`2026-06-12-e07-official-difix-distill-ab.md`](docs/superpowers/plans/2026-06-12-e07-official-difix-distill-ab.md)；全部不通 → E0.7 ⏸ 不阻塞主线。**2026-06-12 结案（IPC 方案绕开，非 ⏸）**：权重确证不可得（A 级 NGC 无特权 key；B 级 HF Fixer = state_dict、C 级 HF Difix3D = diffusers，而 nre difix loader = `torch.jit.load`（自包含 TorchScript），格式墙实测失败；JIT 自导出 trace 卡 cosmos 模型 cpu inlined 常量 device 无底洞）→ 大g改 **IPC 方案**：不把权重塞 nre，而是 Fixer 做独立 server（harmonizer 容器原生 Python 推理）+ nre `DifixModel` socket 转发，零 trace/device 坑、换 server=换修复器。E0.7 α 已完成（权重级 B，详见 §5 Done Log） | E0.7 ✅(IPC) |

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

- **2026-06-18 E2.8 流水线完成（6 阶段）+ Task 6 定量 QA（分支 `claude/sweet-engelbart-56e3da`，续 06-17 core）**。core（Tasks 0-5）后 9 个 follow-up commit 把流水线从「全替」推到最终 **6 阶段 single-clip scene factory** 并补定量验收。
  - **两条核心正确性修正**（最贵的两个 bug，已登记 v2_architecture §7 不变量）：① **坐标 +translate**（`f0d9aca`）：bg/road gaussians 在 NRE local frame 要 `+(-world_to_nre.translation)`≈+38m；ego(rig c2w)/track poses/dynamic_rigids **不变换**（实测 ego vs baseline NCore 相机旋转差 0.0°、world_to_nre 旋转块=单位阵）。② **slot-basis 铁律**（`adc2a5c`）：dyn `track_ids` 的 slot→tid 必须走 `sorted(viz_4d.tracks.keys())`（layered_model.py:378），不是 `sorted(set(track_order))`；viz_4d 带全部 179 tid、track_order 只是 cuboid 子集 → 两 basis 不一致让每辆车套**错 tid 的 pose**（=最初「cuboid≠asset 90°」真凶）。
  - **流水线扩成 6 阶段**：replace→**replace+insert**（`215a50e`：给 active/附近无 gaussian 的 vehicle cuboid 也放 AH 车）；**跨源 recon fallback**（`e6d2e4e`：size 配不上 AH 的 bus/truck 从 baseline ckpt 抽真 recon——同 clip baseline vs USDZ track pose 旋转差仅 0.4°，注入无 90°）；**drop 非vehicle**（`09311b8`：每簇被各自 cuboid 框住）；**MLP 跨源天空**（`b55ba3b`：从 sibling baseline ckpt 借 sky_envmap_state）；viser active-cuboid 只显 vehicle 类（`1a03ceb`）。
  - **最终产物**（inceptio `~/work/output/e28_run/packed_ckpt.pt`）：20 AH automobile 全替 + 1 跨源真 bus recon t405（21006 gaussians、12.5m、朝向 0.99，非 AH pickup）+ MLP 天空；**qa_sanity coverage=1.000 / n_replaced 20 / opacity_median 0.101 / passed**；viser+harmonizer K=4（cosmos temporal 需 V=1+K=5 帧）RTT~1s 0 OOM 目测干净。
  - **Task 6 定量 QA**（新 `scripts/e28_quant_qa.py` 编排 + 6 测；复用 render.py novel dump→`eval_frames_dir.py` NTA-IoU+FID/KID→可选 harmonizer before/after→`qa_report.json`）。**附带修真 bug**（`_override_conf_path` + 2 回归测）：`render.py from_checkpoint` 在 `--path` override 前就读 `conf.path` 求 object_name，packed_ckpt 的 conf.path=MISSING → 崩；修为载入 conf 后即应用 path override。
    - **实测定量数**（inceptio front-cam novel，n=75/档，`qa_report.json`）：**NTA-IoU lateral_3m 0.110 / 6m 0.060（mean 0.085）；novel FID lateral_3m 198 / 6m 269（mean 233）、KID 0.259**。
    - **关键发现（肉眼验证 4 帧，非坏帧）**：几何有效（建筑/路面/斑马线/车在位、coverage 1.0），但 **lateral 视角 bg/road gaussian 重涂抹 + floaters（6m 远比 3m 重，经典 3DGS 离轴外推退化）**，FID 大头是 **background 涂抹**、非车替换。frankenstein 拼装（跨源 bg/road + AH 车 + 重组天空）novel-view 真实感比原生训练 recon 差（对照 E0.2 官方 lateral_6m raw FID 92 → 本编辑场景 269）。**n=75 小样本 FID 绝对值偏高（FID 需 ~2048 样本），宜作 harmonizer before/after 相对比较、非绝对锚**。
    - **harmonizer before/after ✅**（non-temporal DiffusionHarmonizer server :59489 + `e28_quant_qa.py --skip_render --harmonizer_*`，复用 packed_ckpt 已渲帧）：**所有指标都朝正确方向改善** —— FID lateral_3m 197.6→190.2（−7.4）/ lateral_6m 269.2→226.2（**−43.0 / −16%**），mean FID **233.4→208.2（−25.2，improved=True）**；NTA-IoU 0.110→0.119 / 0.060→0.068；KID 0.191→0.182 / 0.327→0.270。harmonizer 对退化更重的 6m 救得更多，与 E2.1 anchor（FID −30%）方向一致。**「FID after<before」验收项 ✅**。
    - **结论**：流水线产出**几何可编辑场景**成立、harmonizer 能改善 novel-view 真实感（FID −25），但绝对 FID 仍高（208，bg/road 离轴退化主导）→ **E3.5 floater-prune 是进一步改善方向**。AH 资产前馈 lifting 比逐实例优化 recon 略糙（97k 高斯但软）是固有权衡，非降采样。
  - **测试**：35 e28（asset_bank 5 / replace 13 / qa_sanity 5 / usdz_viz4d 12）+ 6 quant + 2 render 回归 = **43 全绿**（Mac CPU；inceptio 35 e28 绿，render 回归测因 env transformers↔sklearn 冲突无法 collect、Mac 已证）。
  - **Task 7 bus AH 收割 ✅**（asset-harvester Workflow N，HF gate 已接受）：inceptio 装 asset-harvester 独立 env（gsplat 1.5.3 源码编译过）+ 下 4 checkpoints（mv 6.7G/seg 4.7G/tokengs 1.3G/cam 24M）+ 2 辅助 HF 模型（dc-ae VAE 1.3G + nvidia/C-RADIO 2.5G，diffusion 运行时依赖、不在 AH bundle 内——之前 hang 在拉 VAE/C-RADIO 上，预下载后解决）。`run_ncore_parser.sh --track-ids 405,165`（track-id 与 viz_4d 同源 clip cuboid 一致：bus 405 留 11/11 帧、truck 165 留 4 帧）→ SparseViewDiT 16 视图 + TokenGS lifting（3.36s/sample）→ orient（Y 轴 90°）→ metadata。产物 **bus/405 gaussians.ply 5.2M（label_class=bus, dims=[12.45,3.08,3.52]）+ heavy_truck/165 5.6M**；**肉眼验证多视图 = 干净可辨城市公交（深红车身/车窗/车轮，远胜偏淡跨源 recon）**。并入 bank（6→8 assets，+bus +heavy_truck class，备份 bundle.bak）→ **重跑 driver：bank assets=8 → AH-match 21 + cross-source recon 0（bus+truck size ratio≤1.5 自动 recon→AH，driver 零改码）、coverage=1.000、opacity_med 0.098、QA passed → packed_ckpt_busah.pt 1.1G**。验证了「bank 加资产即自动改路由」这一 E2.8 设计意图。

- **2026-06-17 E2.8 系统性 rigid 全替流水线 — core 完成（Tasks 0-5，分支 `claude/sweet-engelbart-56e3da`）**。E2.5 手术式换 3 车系统化为「单 clip 全 vehicle track 成批换 AH 资产库」可复现流水线。**大g 决策**：替换入口走 USDZ 完整路径（非 baseline-ckpt 捷径）。
  - **Task 0**（6144f23）：E2.5 模块（e25_inject/warmstart_metadata/warmstart_ply/e25_inject_ah_replace）只在 `claude/practical-mcnulty-94bb49`，未合 main/执行分支 → surgical 拷入（已验证 import 自洽，layered_model 符号齐全）。E2.5 自带 **17 passed**。
  - **Task 1-4 纯函数 TDD**（946147c/60103e6/8dc4d0a/cd02856）：`asset_bank.query_bank`（class 过滤+L2 最近+fallback ladder，**bank 不消耗资产**——区别 e25 bijection）/`e28_replace.assign_assets_to_tracks`+`replace_all_vehicle_tracks`（复用 `replace_tracks_in_dyn_node` 守护 bg/road/非 vehicle 字节不变）/`qa_sanity`。**13 测**（后增至 22）。
  - **Task 5 USDZ拆+driver**（321926f）：新写 `nre_usdz_viz4d.py`——当前分支 loader 默认只载 bg+road、dynamic_rigids 接线（track_ids+viz_4d）缺失（只在 fervent-knuth，且其读 volume.nurec 与我们 checkpoint.ckpt flavor 不兼容）→ 复用本分支 `build_native_ckpt`(读 checkpoint.ckpt) + 移植 fervent-knuth 已验证的 `parse_rig_trajectories`/`build_viz4d_dict`/`build_ftheta_dict`（timeline+ftheta+world_to_nre 全来自 USDZ 自带 rig_trajectories.json；坐标 offset 实测 no-op）。
  - **inceptio 端到端实测**（`last.usdz` 9ae151dc，与 baseline 同 clip）：拆出 3 层（bg 2.34M/road/dyn 95792，**deformable 天然丢弃**）+ viz_4d（179 tracks）+ recon（27 present）；**8/8 vehicle track 全替（含 heavy_truck，is_vehicle 实数据修正为子串匹配），coverage=1.000、QA passed**；一资产服务多 track（2 资产覆盖 8 车，E2.8 相对 E2.5 bijection 的新意）。
  - **实测驱动的两处修正**：① `qa_sanity` opacity floor 0.15→0.02——实测 NRE 重建 dynamic per-gaussian opacity 中位数本就 ~0.08（recon 0.081/AH 替换 0.103/bg 0.056），plan 的 0.15「防烟雾」误标定且无法区分 0.11 烟雾与正常；改 anti-degenerate + 只测替换粒子（replaced_slots 掩码），真烟雾交 Task6 FID+目视；AH 0.103>recon 0.081 → 替换无烟雾。② **viser 实测抓真 bug**（05e8474）：`build_viz4d_dict` tracks 原存 numpy → engine.load_3dgrt_object/render.py 的 populate_tracks auto-hook 调 `.to()` 崩 → 改存 torch（repro 179 tracks 全 torch；viser 加载 `loaded schema_v2 (179 tracks)` + `listening *:8090` 通过）。**此修复同时让 Task6 render.py 路径受益**。
  - **待 GPU 续作**：Task6 定量 QA（render-only→harmonizer IPC :59489 temporal→NTA-IoU(vehicle_detector)→FID(--novel-fid)，E2.1 配方 ~4.62s/帧+1.8s/帧）；Task7 asset-harvester 建 sedan/SUV/van/bus/truck 5 类资产库（现 3 车 bundle 为起点）。

- **2026-06-11 E0.1 + E0.2 + E0.3 + E0.5 完成**（commit `8b2fcbe`，worktree 分支 claude/amazing-tereshkova-5c7b30）——E0 首日四卡落地（剩 E0.4 gate E1 工具、E0.6 材料已备），全程无 NGC key：
  - **E0.1 容器冒烟**：`nre-ga:latest` 实测 = **26.4.146-c63f08a4**（2026-05-28 build，entrypoint `/app/run`）。容器内 nvidia-smi（CUDA 13.1）/ CLI 全子命令面正常。**R-v4.1 答案：无 NGC key 跑通 validate→train→render 全链**；key 唯一已知需求点 = 官方 train-time difix 蒸馏权重 `cosmos_3dgut.pt`（NGC API URL，HF 无副本）。
  - **E0.3 官方配方训练**（inceptio 4090）：clip 9ae151dc + Hyperion-8.1 `car2sim_6cam` + `references/configs/pai.yaml` overlay（`--config-name=external_overrides`），40k 步 **2h07m**（7.45 it/s），峰值显存 16GB（**24GB 无需任何降配**，R-v4.2 未触发），2.62M gaussians。**官方口径锚（每 3 帧 + 1/4 分辨率 + cpsnr）：test/psnr 30.30 / cpsnr road 38.27 · car 34.59 · person 32.65 · sky 38.81 / chamfer_distance 0.295**；产物 `~/work/nurec_e0/train_out/PVG7YYV72YKPLumogi7F7U/`（`artifacts/last.usdz` 1.1GB + val mp4 + metrics.yaml 20 类全套）。“传闻 ~36dB”在本 clip 官方口径下不存在（30.30）；与 multilayer 26.06 的对比须待 E0.4 统一口径。
  - **数据兼容实录（R-v4.3 实际未命中）**：官方 13 个 itar 全兼容 nre 26.4；唯一阻塞 = **v3 P3.0 自产 `aux.lane.zarr.itar`**（旧版 ncore 写入、缺 `.zmetadata.cbor.xz` 且 sequence 标识不符），两次启动失败（KeyError / Can't load aux data for different sequences）→ 用容器内官方 `consolidate_compressed_metadata()` 升级全部 itar 至 `9ae151dc_consolidated/` + lane aux 移出 → 三次启动成功。教训入档：**自产 aux 文件会被 nre 按文件名 glob 自动吞掉，喂官方容器的目录必须只留官方产物**。
  - **E0.2 渲染 + 修复链全链完成**（官方 USDZ 场景 0fd06bc3，1.92GB，4K→1080p）：`nre render` 三档各 595 帧（原轨迹 GT / lateral 3m / 6m，**17.8ms/帧** @4090，rig offset 法）→ Harmonizer 时间模式修复两档（~1.07it/s，594 帧/档 ~9min）。**FID（vs 真实参考帧分布，正确口径）：原轨迹渲染 7.37 / lat3m 57.3→修复后 65.6（↑8.4）/ lat6m 91.8→86.6（↓5.2）**；（早期 vs GT 渲染分布口径：57.3/93.0，留档备查）。**目视修复显著**（lat6m 锥桶修直、左缘树木涂抹消除、路面脏斑清除、黄线连续）**但 FID 几乎不动——E0 判别性结论：① 官方表示侧太强（原轨迹 FID 7.4、6m 横移 raw 才 92，伪影占比低），FID 大头是视角内容差，修复链 FID 收益∝伪影占比（DiFix3D+ 论文 134→50 是重伪影场景）；② 推论：3dgrut2 自有 ckpt 伪影远重于官方 → E2 修复收益预期更接近论文场景；③ FID 单指标评修复会系统性误判（修复输出的扩散平滑风格会抵消去伪影收益）→ E1.4 必须搭配区域化/patch 指标，R-v4.5 双协议验收被实测证实必要**。对照帧存档：inceptio `~/work/nurec_e0/renders/0fd06bc3/{gt,lat3m,lat6m}/`、修复帧 `~/repo/harmonizer/input_frames_cosmos_temporal_lat{3,6}m/`、真帧 `~/work/nurec_e0/real_frames/0fd06bc3/`（594 帧 ffmpeg 自参考 mp4）。Harmonizer 链工程留档：镜像 `harmonizer-cosmos-env`（33.1GB，base `pytorch:25.10-py3` 本地、Dockerfile COPY stage 改本地镜像绕 docker.io 拉取）；HF 权重全量（`diffusion_harmonizer.pkl` 5GB + `Cosmos-Predict2-0.6B-Text2Image` 4.1GB）；**HF 侧两道人工解锁：fine-grained token 勾 gated repos read 权限（403 报错文案误导为网络错误）+ Cosmos-Predict2 模型页接受 license（与 token 权限独立）**；推理须 `--entrypoint python` + 每 run 独立输出目录（时间模式会回读历史输出做参考帧）。
  - **E0.5 配方 diff**：见 §1.2 行内 Top-5 与 [`docs/superpowers/specs/2026-06-11-e05-nurec-vs-multilayer-recipe-diff.md`](docs/superpowers/specs/2026-06-11-e05-nurec-vs-multilayer-recipe-diff.md)（11 维度 / 全部引 resolved key 路径）。**最大架构发现：官方 road = 几何冻结的 ground-mesh 薄片层 + MCMC 全豁免**（aperture problem 的官方答案是“路面不让动”，与 v3 诊断完全互证）；官方 difix 蒸馏钩子原生就是 ±3m lateral（默认关）。
  - **大g 目视终评（nre viewer 交互对比，E0 判别预答案）**：官方配方产物 vs multilayer 自训——①整体锐度**不更好**（官方 subsample:2 半分辨率监督所致，我们 full-res 不输）；②**路面车道线明显更好**；③**3m 侧向移动退化明显更小**。且该 run difix 蒸馏关闭——**官方纯表示侧（road 冻结五件套 + LiDAR ray 监督）就赢下外推**，E0.4 判别初步指向：表示侧为主差距 → **E3 优先级 ≥ E2**（待 E1 量化锚正式确认）。
  - nre viewer 工程留档：26.4.146 viewer 两个 bug/坑——①客户端连接竞态（camera on_update 先于 reload action → `Can't unpack empty optional` 渲染线程死），host-side 两行 patch 绕过（`~/work/nurec_e0/patches/av_patched.py` 挂载覆盖）；②nrend 快速路径对自训 USDZ 间歇性 `NRenderer.render failed`（官方 USDZ 正常），`--no-enable-nrend` 走 torch 路径稳定。viewer GUI 自带 Camera Translation Offset（可直接体验横移外推）+ Render Video 导出。
  - 环境备忘：HF 下载一律 `HF_HUB_DISABLE_XET=1` + 代理（xet/直连均会卡死或失败）；harmonizer 容器 entrypoint 是 `/bin/bash`，跑推理须 `--entrypoint python`。
- **2026-06-12 E1.1 + E1.2 + E1.4 完成，E1.3 训练中，E0.4 工具就绪，E0.6 run-book 入档**（worktree 分支 claude/dreamy-raman-8264ca，commit 链 `4d27bf6`→`5e61064`，Mac 测试 71 个全绿）——E1 测量门首日三卡落地 + 三方 canonical 锚（每 ckpt 全 5 相机 375 val 帧 × 7 渲染 × 全指标，26 min/ckpt @4090）：
  - **E1.1 外推测量门**（`d53e4d9` 移植 / `c3779c2` 扩档 / `a3a6d26` 口径保护 / `02d458c` plane_warp / `6ef63ac` 接线）：PR #24 eval 侧 path 限定移植（9 文件 +634 行，lane loss/trainer/yaml 未拉——**R9 中立**，决议后 reconcile）；`NOVEL_VIEW_MODES` 扩 lateral_3m/6m，`LEGACY_NOVEL_AVG_MODES` 冻结 4 档历史口径，新增 `mean_novel_lpips_avg6`；新建 [`plane_warp.py`](threedgrut/model/plane_warp.py)（novel 射线↔road 高度场定点求交→FTheta 投回原相机采伪 GT；复用 dataset `rays_dir` 缓存 + `query_ground_z` + dynamic_mask 同源投影，BUG-1 隔离）。**口径回归三点全中**：baseline `mean_novel_lpips_avg` 0.598718 vs 历史 0.598733（Δ1.5e-5）、cc_psnr_masked 25.789 ≡ v3 锚、interp `mean_lane_grad_corr` 0.6931 ≡ P3.0 门锚 0.6932。
  - **三方锚 + F3 判别（lane grad_corr，warped 口径，front cam 75 帧）**：baseline 1m 0.551 / 2m 0.467 / 3m 0.384 / 6m 0.303；B3 0.581/0.473/**0.381/0.307**；aniso20 0.573/0.465/0.389/0.298。**结论①（B3 张力否定）**：aniso 8→30 未放大 3m/6m 退化（三方 ±0.01 打平），B3 可继续当主配方；**结论②（更重要）**：lane loss 的 interpolated 优势（+0.05）到 2m 即耗尽、3m/6m 三方无差——**外推退化与表示侧短刀配方无关，必须 E2 修复链 / E3 参数化级方案**，与 E0 判别（官方 road 冻结五件套赢外推）同向互证。
  - **E1.2 NTA-IoU**（`add0aa7` 核心 / `ea22e2c` 接线）：VEHICLE_TRACK_CLASSES 实测＝automobile 68/heavy_truck 1/bus 1（ckpt viz_4d census）；投影含 behind-camera 前置剔除（pinhole 分支 z-clamp 坑）；interp 0.117/0.120/0.120（baseline/B3/aniso20），novel 档单调降至 @6m 0.054–0.062，375/375 帧有车。**口径注记**：全 GT 车含远景小目标（检不出按 0 计入 best-match）→ 绝对值偏低是口径效应，作相对标尺；novel 档 B3/aniso20 略低于 baseline（@3m 0.076 vs 0.096）——**仅记录观察，不下结论**（噪声 or lane loss 配方对车辆外推的副作用，未判；大g 2026-06-12 决议保留待察）。
  - **E1.4 FID/KID**（`b20ff48` + `5e61064` key 名修正为 `mean_novel_fid_<mode>`）：baseline FID render 75.3 → 1m 124 → 2m 152 → 3m 168 → 6m 193 全程单调（K4 sanity PASS）；KID 0.021 → 0.102 @6m（subset 自适应 37）。**FID render 75 vs 官方场景同口径 7.4 → 自有表示侧伪影重一个量级，E0.2 推论③（E2 修复收益更接近论文场景）获定量支撑**；三方 FID/KID 互差 <3（配方间外推无感知差，与 lane 结论一致）。
  - **E1.3 进行中**（`9503242`）：`--dataset-cameras` 开关 ✅（ckpt 嵌入 conf 的 camera_ids 替换 + **BilateralGrid 强制关断**——camera_idx 错套陷阱，held-out 口径以 cc_* 为准，R-v4.8）；4-cam（排除 cross_left）30k depth-off 训练 14:17 自动接力开训（排队脚本），~7h。
  - **E0.4 工具就绪**（`49f27b2`/`7096b55`）：[`dump_test_split_manifest.py`](scripts/dump_test_split_manifest.py) 实测 375=75×5 ✓；[`eval_frames_dir.py`](scripts/eval_frames_dir.py)（render_all 去模型版：项目数据集供 GT/sseg/lane/cuboid/FTheta，NuRec 只供像素帧；interpolated 全指标 + lateral 档 plane-warp/NTA/KID；缺帧硬报错保对齐完整性；tracks 从 ckpt viz_4d 免建模查找——validity 字段实名 `frame_info`）。
  - **E0.6 run-book 入档**（`21e64f8`）：[`2026-06-12-e06-official-actor-editing-capability.md`](docs/superpowers/specs/2026-06-12-e06-official-actor-editing-capability.md)——前置五项全验证（last.usdz 含 sequence_tracks.json 可编辑、AH 3 车+3 人已传 inceptio、镜像/patch 在位）；上游 asset-editing schema 全文消化：remove/replace/insert 三操作 JSON 语法、**`--renderer default` 取代弃用的 `--no-enable-nrend`**、编辑是会话级内存态（render-grpc 完成自动 `restore_model_parameters` 回滚）、AH bundle 需重排嵌套目录。
  - 工程留档：anchor 跑批耗时几乎全在指标栈（渲染仅 ~2%：每帧 7×LPIPS + 6×plane-warp + 7×YOLO + 14×Inception）；权重预下（yolov8m 52MB CWD 解析 + Inception 91MB torch hub）经 mihomo；`pkill -f` 自匹配坑（ssh bash -c 命令行含目标串会自杀，用 `[.]` 正则规避）；三份 anchor metrics.json 已补规范 FID key 别名。
  - **E1.3 held-out 真 GT 锚（当日补完）**：4-cam（排除 cross_left）30k 实际 **70 min**（depth-off 轻配方，远低于 7h 预估）。三件套（统一 exposure-off / cc_* 口径，R-v4.8 协议内比）：**heldout cc_psnr_masked 19.16 / guard 25.82 / upper（5-cam baseline 看 cross_left）26.93 → 真 GT 外推差距 7.77 dB**；car class 17.94 vs 21.48（gap 3.5 dB）、road_crop 15.71 vs 18.23、NTA 0.071 vs 0.101、cc_lpips 0.543 vs 0.380。guard 25.82 ≈ 5-cam baseline 标准口径 25.79 → **少一台相机不伤共有训练相机质量**（信息量集中在 held-out 视角缺失本身）。产物 `~/work/e13/{heldout,guard,upper}/`。
- **2026-06-12 E0.7 立项（大g 提议）：Harmonizer 取代 Fixer 的官方蒸馏训练 A/B**——大g 已用 `fixer_server.py`（64 行 socket IPC，harmonizer-cosmos-env 容器复用 Fixer 的 `pix2pix_turbo_nocond_cosmos_base_faster_tokenizer` + HF `pretrained_fixer.pkl`，复刻 nre DifixModel 前后处理 576×1024、color_transfer 留 client 端 kornia）打通官方 difix-distill 无 key 路径（R-v4.1 补充关闭），对照组训练运行中（`pycena ... difix.training.enabled=true` + fixer_server，`~/work/nurec_e0/e07/`）。实验组设计：同协议 `harmonizer_server.py` 换 **DiffusionHarmonizer 非时间单帧变体**（train-time 蒸馏逐 step 随机 novel view、无帧序 → temporal 模式不适用）→ 同 cmd 重训 → **三方对照**（E0.3 无蒸馏 30.30 / e07-Fixer / e07-Harmonizer）E1 工具同口径评 → 修复器代际增益官方侧标定，直接喂 E2.2 预期。
- **2026-06-12 E0.7 difix 蒸馏对照 run 完成（α 段，IPC 方案）**（commit `2d974de`，worktree 分支 claude/sleepy-einstein-bf3a95）——回答「difix 蒸馏在本 clip 值多少」：**B 级权重温和局部增益（车道线外推略好），interpolated 微降代价，6m 无改善**。
  - **方案转向（大g 拍板）**：原 plan 的 NGC/hack 权重路径不可行——nre difix loader 是 `torch.jit.load`（NVIDIA 内部 `torch.jit.save` 导出的自包含 TorchScript，图里嵌 transformer_engine 算子），而 HF 公开权重全是 state_dict/diffusers 格式，**交付物类型不同、非 state_dict remap 可桥接**（A 级 NGC `cosmos_3dgut.pt`：inceptio 无 ngc CLI / 特权 key 拿不到；B 级 HF Fixer = state_dict、C 级 HF Difix3D = diffusers：实测 `torch.jit.load` 均失败）。JIT 导出工程（harmonizer 容器把 `Pix2Pix_Turbo` trace 成 TorchScript）实测攻克了跨 torch 2.9→2.7 / TE 2.8→1.11 兼容 + checkpoint/RoPE/flash_attn 三大 fused-op patch + trace+save+跨容器 nre-ga load，但**卡在 cosmos 模型内 cpu inlined 标量常量的 device 无底洞**（逐个 patch 不收敛）。→ **大g 改定 IPC 方案**：Fixer 推理做成独立 server（harmonizer 容器跑 `Pix2Pix_Turbo` 原生 Python 推理，**零 trace/device 坑**）+ nre 容器 patch `DifixModel`（`training_controller.py:335` 唯一调用点）经 socket（127.0.0.1:59487）转发降质渲染帧、收回修复帧当蒸馏目标；color_transfer 留 client（nre 有 kornia，harmonizer 无）。**换 server 进程 = 换修复器，训练侧零改动**（Harmonizer 三代 A/B 已据此备好）。
  - **权重级 = B**（HF `nvidia/Fixer`，Cosmos-Predict2-0.6B 架构族，非官方 cosmos-difix；增益方向可信、幅度打折号）。
  - **训练**（inceptio 4090，run id `Qm52hePw64ydcMF3cz3vdA`）：E0.3 配方 + 恰好两处 override（`difix.training.enabled=true` + ckpt 频率 40000→5000）+ IPC difix，difix 钩子全官方默认（start_step=20000 / p_init=0.5 → milestones[25k,28k]×0.5 / ±3m novel poses / color_transfer），40k 步 **2h19m**（vs E0.3 2h07m，difix 蒸馏仅 +12min，IPC 开销小）。**parsed.yaml diff 单变量审计**：仅 `difix.training.enabled` + ckpt 频率两处差（外加 camera_cross_right 列表顺序差一位，同 6 相机集合不影响训练/val 聚合）。蒸馏实证 fire：step 20000 后密集 novel-view 蒸馏步（`train/nrays` 1.3e5 vs 普通步 9.79e5，p_scheduler 期望 ~4750 次）+ server GPU 49% util；smoke（start_step=50/p_init=1.0 强迫早 fire）已先验通路。
  - **C1 interpolated（官方 val 口径，与 E0.3 同协议直接可比）**：test/psnr **30.30→29.77（−0.53）**；cpsnr car 34.59→33.92 / road 38.27→37.79 / person 32.65→31.71 / building 39.50→38.33 / vegetation 38.07→36.72；chamfer 0.295→0.317。**全面略降 ~0.48–1.35 dB**（蒸馏微调代价，符合 R-v4.5/R-v4.7 预期管理：蒸馏优化外推分布、interpolated 可能微降）。
  - **C3 目视（大g viser 交互 + `render` 截图对照，frame300 同轨迹同帧）**：**外推 3m 车道线 E0.7-Fixer 略好**（路面标线外推改善，大g viser + 截图双确认），其它区域差不多；**6m 仍难以接受**（超出 ±3m 蒸馏增强分布，扩散平滑甚至让细节略糊）。
  - **判别结论**：hack **B 级** Fixer 蒸馏在本 clip 是**温和、局部**增益——difix 蒸馏机制有效（车道线/路面标线外推改善被目视证实，正中 v4 lane KPI 方向），但 ① interpolated 微降代价 ② 增益不外推到 6m（目视 3m 改善 > 6m，分布外泛化不足；定量 Δ@3m/Δ@6m 待 β/E1 lane grad_corr）③ B 级权重幅度有限。**两个读数喂 v4**：(a) difix 蒸馏方向对 → 支持 E2.2 渐进外推蒸馏；(b) 显著增益需官方 cosmos-difix 权重（NGC）+ progressive 蒸馏（1m→3m→6m 逐步扩大增强范围），±3m 单档蒸馏不能直接泛化到 6m。
  - **β 段待 E1**（lane grad_corr / band_lpips @3m/6m + NTA-IoU 定量）：大g 目视的「车道线略好」须 E1.1/E1.2 工具定量坐实；两个 USDZ 已落盘（E0.3 `PVG7…` + E0.7-Fixer `Qm52…` 的 last.usdz），届时纯渲染回填，不重训。
  - **工程留档**：IPC server `~/work/nurec_e0/e07/ipc/fixer_server.py` + client `model_ipc.py`（mount 覆盖 nre `nre/difix/model.py` 真身路径，`run.runfiles` 是其 symlink）+ launch `launch_full.sh`；完整记录 `~/work/nurec_e0/e07/ipc_solution_log.md` + 权重决策 `weight_decision.md`。**关键教训：把 Fixer「塞进 nre」两条路——JIT trace 卡在序列化（device 无底洞）、in-process Python 卡在依赖缺失（nre-ga 无 cosmos_predict2/imaginaire）；IPC 用跨容器 socket 绕开两者**。
- **2026-06-12 E0.4 + E1.5 完成，E1 阶段全绿（5/5）**——双向对照锚落地 + gap 表收口 + E2/E3 重排拍板：
  - **E0.4 执行实录**：nre 出帧 8 趟（gt×5 相机 + 前视 ±3m/6m + 反号保险，18.7ms/帧，offset 用标定矩阵精确换算 `3×cam_right_rig=(-0.015,-3.000,0.043)`）→ `eval_frames_dir` 四评测。三层 bug 各修一刀：① scripts/ 直跑 sys.path 落到 env editable 安装（主仓库旧码）→ repo-root bootstrap；② 本 dataset batch 无 frame_idx（全 -1）→ **时间戳精确连接**（manifest `timestamp_us` ≡ nre `frame_end_timestamp_us`，600 映射 max_dt=0µs，对齐由假设变证明）；③ tracks provider numpy 真值判断。**反号保险趟兑现**：正号 grad_corr 0.437 vs 反号 0.045 → offset 语义实证。产物 `~/work/e04/{renders,evals}/` + 四份 metrics json。
  - **判别数字（同口径，项目工具）**：interpolated——NuRec psnr 25.12 / FID **61.3 vs 我方 75.3**（官方伪影更少）/ lane grad_corr **0.659 vs 我方 B3 0.739**（full-res + lane loss 近景反超官方半分辨率监督）/ NTA 0.126≈0.120；lateral——**@3m 官方 grad_corr 0.437 vs 0.381（+0.05）、band_psnr 16.34 vs 12.47（+3.9dB）；@6m 0.297 vs 0.307 两家同崩**；NTA 外推档交错（0.087/0.044 vs 0.076/0.054，小样本仅观察）；lateral FID 217/237（前视单相机口径，与我方 5 相机口径不可直接比，已注记）。
  - **E1.5 重排结论（E1 出口）**：**E3 先行，E2 定位 6m+ 档与残余伪影的互补**。证据链五条：①官方纯表示侧（difix 关）3m lane 全面领先 → road 冻结五件套之效，表示侧差距实证；②6m 两家同崩 ~0.30 → 表示侧只能把退化曲线右移一档，6m 档修复链是共同必需；③三方锚配方无差 → 差距是结构性配方非调参；④interp FID 61 vs 75；⑤E1.3 真 GT gap 7.77dB 待咬。执行序：E3.1/E3.2 短刀（**待 R9 决议，大g 暂缓**）→ E3.3 BEV 纹理平面化；E2.1 Harmonizer spike 低成本并行（域差已被 E0.7 smoke 初步排除：零微调修复我方重伪影帧目视显著）。
  - 工程留档：nre render 输出嵌套 `<cam>/<cam>/` + `timestamps.json`（file_name/render_frame_idx/frame_start・end_timestamp_us）——时间戳连接是对外部渲染器帧对齐的标准做法，假设性序号映射不可靠（nre 599-600 帧 vs dataset 595 帧）。
- **2026-06-13 E2.1 离线 Harmonizer 修复 spike 完成（Task 0–5）**（branch `e21-harmonizer-spike`；commit 链 render-only `daefc03`/`4ac911d` · batch-fix `cd9fa6b` · compare/montage `24b03b0`；pytest frame_align 3 + ipc_client 1 + compare 3 全绿）——量化 DiffusionHarmonizer 三代对 baseline 3m/6m 渲染帧的纯离线修复增益，出 E2.2 go/no-go 判别：
  - **核心对比（raw=修复前≡E1锚 / fixed=修复后，全 5 相机 375 帧/档）**：FID@3m 168→113（−33%）/@6m 193→139（−28%）；KID@3m 0.082→0.029（−64%）/@6m 0.102→0.045（−56%）；NTA-IoU@3m 0.095→0.128（+0.033）/@6m 0.062→0.100（+0.037）；lane_band_psnr +0.57/+0.25；**lane_grad_corr@3m 0.384→0.299（−0.085）/@6m 0.303→0.208（−0.095）**。产物 `~/work/e21/{raw,fixed,evals,montage}`。
  - **raw 交叉验证 ≡ E1 锚（口径可信）**：raw lane_grad_corr 0.384/0.303、NTA 0.095/0.062、FID 168/193 与 E1.1/E1.2/E1.4 锚逐项吻合 → eval_frames_dir 口径 ≡ render_all。
  - **目视（lateral_6m 拼图 raw\|fixed）**：银色悬浮涂抹消除、路面平整、建筑连贯、车辆清晰，**无引入异物** → R-v4.4 域差排除（Harmonizer 在 NCore 域零微调有效）。
  - **E2.2 go/no-go = GO**：教科书级 DiFix3D+ ablation——纯后处理大幅提感知（FID −30%/KID −60%）+ 车辆检出（NTA +35%），但扩散平滑伤车道线高频（grad_corr 退 ~22%/31%）。**lane_grad_corr 退化正是 E2.2「蒸馏回 3D 提几何 + road/lane 区域加权保护」要解决的**（后处理提感知、蒸馏提几何、叠加最优）。修复链对我方重伪影场景显著有效，值得投 E2.2。
  - **工程副产出（归因「render 出帧为何慢 10s/帧」，大g 提问驱动）**：① 主因是**每帧加载全套 GT 监督数据（LiDAR 深度图 8MB + sseg + lane mask）**——非指标栈（微基准 LPIPS/SSIM/FID/KID 仅 0.9s/帧）、非 GPU 争用（关 harmonizer server 仍 10.6s/帧）；新增 render.py `--render-only`（conf override 关 4 监督加载开关 + NTA）+ `--novel-only`（只渲 3m/6m）→ **10.6→4.62s/帧**，~29min 出齐 750 帧。② 修复用 E0.7 IPC 架构（harmonizer_server socket :59489 + `scripts/e21_harmonizer_batch_fix.py` client + kornia Reinhard color_transfer），harmonizer ~1.8s/帧 × 750。③ render_only 顺手加 `if compute_extra_metrics` guard，修了 render_all 无条件访问 `criterions["ssim"/"lpips"]` 的潜在 bug。
  - 新增代码：`threedgrut/render.py`（novel-save-n + frames_map + render-only/novel-only）、`scripts/{e21_harmonizer_batch_fix,e21_compare_metrics,e21_visual_montage}.py`、`tests/test_e21_{frame_align,ipc_client,compare}.py`（7 测全绿）；零训练代码改动（trainer/difix.py/yaml 未碰，符合 spec §8）。
- **2026-06-15 E0.7 β' 启动**（inceptio）——核验发现 handoff spec 标"已移交另一 session 执行"但实际**未启动**（Fixer 对照组 `train_out_e07/Qm52…` 已 done @ `fixer_done.flag`，但无 `harmonizer_done.flag`、无 harmonizer 训练日志、`launch_harmonizer_train.sh` 未跑过）→ 执行 `bash ~/work/nurec_e0/e07/launch_harmonizer_train.sh`：harmonizer_server 容器 READY@:59487 + e07_harmonizer_train 训练容器起（40k 步，~2.5h，~9 it/s）。
- **2026-06-15 E0.7 β' 完成，三方对照表回填**（inceptio，run-id `SqPigPNBiA366kkzTMi3EN`，容器 exit 0）——handoff §5 第一层 interpolated 口径（官方 val：每 3 帧 + 1/4 res + cpsnr，与 α 单变量可比）：

  | run | difix 蒸馏 | test/psnr | cpsnr car | cpsnr road | cpsnr building |
  |---|---|---|---|---|---|
  | E0.3（无蒸馏） | — | 30.30 | — | — | — |
  | α-Fixer（二代） | Fixer | 29.77 | 33.92 | 37.79 | 38.33 |
  | **β'-Harmonizer（三代）** | Harmonizer | **29.91** | **33.97** | **37.86** | **38.48** |

  **判别结论**：Harmonizer 三代全面略优于 Fixer 二代（psnr +0.14 / car +0.05 / road +0.07 / building +0.15），但**仍低于无蒸馏 baseline**（−0.39）—— 与 α 的"interpolated 微降"趋势一致（蒸馏优化外推分布，interpolated 付微降代价）。增益大头应在**外推档**（需第二层评测：两 USDZ 各 `nre render` lateral 3m/6m 出帧喂 `eval_frames_dir`，handoff §5 第二层，待排期）。**喂 E2.2 的读数**：①修复器代际增益方向 = Harmonizer > Fixer（支持 E2.2 用 Harmonizer）；②interpolated 代价 −0.4dB 量级，E2.2 蒸馏权重 λ 须小起步。
- **2026-06-15 E2.6 代码完成，端到端验证 pending（等 β' 释放 GPU）**（branch `e26-harmonizer-temporal`；commit 链 `bff0ef9`(worktree)→`9de9101`(IPC 链路)→`5cf26e7`(viser 接线)→`4058704`(demo 脚本)）——把 viser_gui_4d 的单帧 DiFix 后处理扩展为 Harmonizer temporal 模式（V=1+K，回读前 K 帧已修复输出做时序参考），用于 Play 连续序列去闪烁：
  - **设计决策（client-side K-deque）**：历史在 client（`HarmonizerTemporalClient`），server 无状态（每连接读 1+K_in 帧一次 forward）。理由：① Mac 可用 echo stand-in 直接测协议 + reset 语义，不需 GPU；② seek/scrub = `history.clear()` 自然 reset，无需协议信号；③ 与 difix_server 同构（一连接多帧 + gpu_lock 并发）。K=4（论文默认）。
  - **新增文件（独立 HMN1 协议，不动 DFX1/DifixClient/difix.py，守护 E1 锚）**：`utils/harmonizer_protocol.py`（HMN1，20B 头 magic+H+W+C+K + (1+K)×uint8 帧，DFX1 reply 复用为单帧返回）、`utils/harmonizer_client.py`（K-帧自引用 deque + `fix(img, reset=)` + 失败不污染历史 + graceful raw fallback）、`harmonizer_temporal_server.py`（`make_harmonizer_temporal_transform` 做 V-stacking→5D (1,C,1+K,H,W) forward→V=0 output 选择；纯 torch 无 einops；保留 E0.7 两坑 chdir/HARMONIZER_PORT，默认 59490）、`tests/test_harmonizer_temporal_ipc.py`（**20 测全绿**）、`scripts/e26_temporal_demo.py`（raw/nontemporal/temporal 三列对照）。
  - **viser_gui_4d.py 接线（6 处最小侵入）**：`--harmonizer_temporal_server`/`--harmonizer_temporal_K` CLI（与 `--difix_server` 互斥，main() reject）+ `HarmonizerTemporalClient` 构造 + `_postproc_reset_flag`（`_on_time_change` 中 `source!="play"` 即 set——seek/scrub/loop-wrap/init 自动 reset 历史）+ `_postproc_last_wh` 分辨率锁 + `_maybe_difix` 改分发（temporal 优先带 reset，否则原 DifixClient 路径不变 + None-guard 防 AttributeError）+ "Harmonizer (temporal, de-flicker)" checkbox。
  - **零回归验证**：`test_difix_ipc.py` + `test_e21_ipc_client.py` 16 测 + `test_harmonizer_temporal_ipc.py` 20 测 = **36/36 全绿**（HMN1 magic 与 DFX1 隔离，DiFix 单帧路径逐字不变）。
  - **端到端验证完成（2026-06-15，inceptio 目测通过）**：β' 完成释放 GPU 后起 temporal server(:59490) + viser_gui_4d（baseline ckpt `p1_2_runB_fix_30k`）目测。**功能正常**：勾选 "Harmonizer (temporal, de-flicker)" → 前序 Play 连续帧走 temporal（V=5，K=4），seek/拖动自动 reset 历史；frame buffer 存的是 Harmonizer 修复后数据（`_maybe_difix` 覆盖 img 后才 `set_background_image`），history deque 自引用修复帧（符合 Harmonizer 设计）。**实测 V=5 延迟 ~1000ms**（5 帧过 0.6B），交互 Play 会降到 ~1fps；K=2（V=3）可降到 ~600ms 备选。
  - **Conv3d warmup 关键修复（commit `cce14ba`）**：首次 inceptio 联调 server 端报 `RuntimeError: Kernel size (3,1,1) can't be greater than actual input size (2 x 144 x 256)`——Harmonizer temporal CausalConv3d（kernel=3 on V 轴）**只接受 V=1 或 V≥3**，我的冷启动逐步增长（V=1→2→3→4→5）撞进 V=2..K-1 禁区。查官方 `inference_pix2pix_turbo_harmonizer.py` L133 `have_history = len >= min_history`（min_history=-min(offset_list=[-1,-2,-3,-4])=4）→ 凑满 4 帧才走 temporal，否则 V=1。**修复**：client 侧 `fix()` 改为 `len(history)>=K` 才发 V=1+K，否则 V=1（payload_history=[]），对齐官方语义；20/20 Mac 测更新通过。**教训**：model 类注释说 "V can be 1" 是指 nontemporal 路径，不是任意 V 都行——temporal 路径有 kernel=3 硬约束。

- **2026-06-15 E2.7 完成（viser_gui_4d 加载 NVIDIA usdz checkpoint，commit `19ffd3d` + `00c8b8c`）** — 让 viser_gui_4d.py 直接加载 NVIDIA NRE/NuRec 训练产物（E0.3 last.usdz），与 3dgrut2 自家产物**用同一套 UI/相机/timeline 并排视觉对标**。Plan: [viser-gui-4d-py-nvidia-virtual-toucan.md](../../.claude/plans/viser-gui-4d-py-nvidia-virtual-toucan.md)
  - **视觉对标关键发现（大g 360° + 横向漂移测试）**：**NuRec 路面建模质量明显优于 3dgrut2 自家训练** — 横向 3m / 6m 漂移视图都不出现 lane_grad_corr aperture problem 的明显退化（与 3dgrut2 自家 ckpt 横向退化形成强烈反差），360° 视角转动路面纹理稳定。**直观证实 v4_plan 主线方向**：E2.2（progressive distillation 渐进外推蒸馏）+ E3.3（BEV 纹理平面化）走 NRE 配方的方向是对的，比指标定量更直观。
  - **代码移植**：从 `claude/amazing-lalande-34ba46` worktree 拿 `nre_usdz_loader.py` (472L, tolerant pickle + Fourier albedo DC + clip_floater_gaussians 砍 ±1500m / scale>20m 远场 + 95-percentile scene_extent) + `test_nre_usdz_loader.py` (169L, 14 tests 全绿) 两个新增文件；**不**带那边对 viser_gui_4d.py 的反向 diff（base 早于 E2.6 会删 Harmonizer）。silly-germain 在 main+E2.6 之上独立写 viser 集成（commit 19ffd3d）+ 6 个修复 (commit 00c8b8c)。
  - **核心技术发现：USDZ 容器 `rig_trajectories.json:world_to_nre.matrix` 坐标变换**——大g 提议"对比 3dgrut2 ckpt 和 USDZ 同源相机 pose 数值应该一致"诊断锁定。NRE 训练时主动 apply 了 NCore world → NRE 坐标系变换（典型纯平移 ~38m，把 ego trajectory 中段当 NRE 训练原点保 float32 数值稳定）。9ae151dc clip 的 `world_to_nre.matrix[:3, 3] = (-38.00, 2.155, 0.278)`，反过来 NRE→world translate = `(+38.00, -2.155, -0.278)`。**之前错误猜测 = NCore SDK ego_pose[0]（~2m，方向相反、量级差 18×）**。修后 background median (3.28, -4.16, 7.83) → (41.28, -6.32, 7.55) 落在 ego trajectory (2.78~48.59) 中段；road median z 从 +7.55 → -0.37（≈ NCore world ground）；视觉对齐完美。
  - **viser_gui_4d.py 改动 (+299 行 / -3 行，main+Viser4DViewer)**：新增 4 CLI (`--usdz` / `--usdz_cache_dir` / `--usdz_layers` / `--initial_cam_id`)；`--gs_object` 改 `required=False`（与 `--usdz` 互斥）；USDZ→.pt 转换 + world_to_nre align 块（mtime hash idempotent cache + 独立 aligned cache）；`Viser4DViewer.__init__` 接 `initial_cam_id` + 存 `self._initial_cam_id`；`_on_client_connect` 用 `_initial_cam_id` cam c2w snap 客户端（H1，**不再落 NCore primary cam=cross_left 朝下视角**）；`_update_ego_frustum` + `Reset View` button handler 都用 `_initial_cam_id` cam c2w（P3/P4 修复，**frustum 不再朝下**）；sanity-check 打印 positions median/range；metadata 必填校验 + USDZ 模式必填 `--dataset_path`；coord 诊断打印。**不动** E2.6 Harmonizer / DiFix / FTheta / no_gaussian_render 任何路径。
  - **6 个修复打包（commit `00c8b8c`）**：① **world_to_nre 平移修复**（量级 18× 方向也错的根因）② **loader 强制 `conf.use_layered_model=True`**（不然 Engine 走 v1 MoG 分支报 `KeyError: 'positions'`；`# @package _global_` 在 compose 时不传 conf）③ **_load_metadata NCoreDataset signature 修复**（T8.6 / 2026-05-20 dormant bug 自 USDZ 首次触发该 fallback 暴露：原 `NCoreDataset(conf, ...)` 把 conf 当 datapath 传，改 `datapath=str(dataset_path)` + 多 sensor ValueError fallback）④ **`metadata.n_frames` 是 method 不是 property**（属性名错改 callable 兼容）⑤ **Reset View P3 + ego frustum P4 修复**（NCore primary cam=`camera_cross_left_120fov` 字母序首位 A 柱盲区下视鱼眼，改用 `--initial_cam_id` 的 c2w）⑥ **stale JIT FileBaton lock 自动清理**（PyTorch upstream issue #9711/#41511，`torch.utils.file_baton.wait` 是文件存在性 polling 不是 fcntl 锁，pkill -9 时 lock 残留→后续进程死循环；新加 `_cleanup_stale_jit_baton_locks()` 在 main() 入口扫 `~/.cache/torch_extensions/py*/*/lock` age>60s 即删）。
  - **本次诊断最深一层坑（重启都修不掉的）**：file_baton stale lock 让 inceptio 上**所有** viser 进程（不只我的，大g 主仓库下跑也复现）在 `Engine3DGRUT → Tracer → load_3dgut_plugin → jit.load → _jit_compile → file_baton.wait` 死循环 polling（WCHAN=hrtimer_nanosleep / 0% CPU / 端口不监听）。重启 inceptio 不修因为 lock 在磁盘上（`~/.cache/torch_extensions/py311_cu128/lib3dgut_cc/lock` 0 bytes）。faulthandler `dump_traceback_later(60, repeat=True)` 拿到 stack 才定位到 `file_baton.py:51`。教训：**任何依赖 torch.utils.cpp_extension.load 的入口都要在 main 之前主动清 stale lock**。
  - **inceptio 端到端验证**：commit 后 worktree 通过 `git stash` + `git push inceptio` + `git stash drop` 成功更新；aligned cache + JIT 编译产物都 persistent 在 inceptio 上；浏览器 http://127.0.0.1:8090（Mac SSH tunnel）看到 USDZ + camera_front_wide_120fov 视角下完美渲染。Engine init 115.9s（首次 BVH build for 2.47M gaussians）。
  - **未来 follow-up（不阻塞 E2.7 ✅）**：① 三方 viser 并排对比（NVIDIA usdz + 3dgrut2 自家 + Harmonizer post-process）—— 大g 已在另一进程跑 `p1_2_runB_fix_30k` baseline + Harmonizer，缺一个统一脚本起三个 viser；② dynamic_rigids 层渲染（amazing-lalande 简版 Phase A 范围外，需要 fervent-knuth 873L 完整版的 `gaussian_cuboid_ids → sorted-tid remap + populate_tracks` 接线，作为 E2.7-B 跟进任务）；③ NRE 路面"横向 3m/6m + 360° 不退化"这个观察值得作为 E2.2/E3.3 的 baseline 视觉标定（看 E2.2 蒸馏后 3dgrut2 自家 ckpt 能不能逼近 NRE 这个水平）。

- **2026-06-15 E2.7-B 完成（dynamic_rigids 接线，commit `7e5edac`）** — 在 E2.7 之上把 NVIDIA E0.3 USDZ 的 dynamic_rigids 层（车辆动态对象）也接上 viser 渲染 + 跟着 timeline 移动。
  - **架构层接通**：移植 fervent-knuth-d25fe9 873L 完整版 Stage 2 helpers（`pose7_to_mat` / `_slerp_wxyz` / `resample_track_to_timeline` / `parse_volume_usda_track_order` 含 **NRE Lightning lax regex fallback** / `parse_sequence_tracks` / `build_dynamic_tracks_for_viz4d` 主入口）到 amazing-lalande 简版 loader (+361 行 in `nre_usdz_loader.py`)。viser_gui_4d.py main() (+233 行) 在 USDZ 模式 + metadata 加载之后注入三件套：`dyn_layer.register_buffer("track_ids", ...)` (gaussian → sorted-tid slot remap) + `engine.scene_mog.populate_tracks(tracks_dict)` (注册 `_track_pose_<tid>` + `_track_active_<tid>` buffers 触发 `_transform_means_and_active` 路径) + `metadata.tracks/tracks_camera_timestamps_us` (让 viser cuboid wireframes 也 timeline-aware，NCore SDK fallback 不提供 tracks)。
  - **USDZ 容器两种变体 fallback**：fervent-knuth 测过的 NVIDIA demo (0fd06bc3) 用 `volume.usda` + 严格 `bounding_boxes/track_NNNNN_TID/cuboid` prim 路径；E0.3 NRE Lightning 训练 USDZ 用 `sequence_tracks.usda` 不同 scope。`parse_volume_usda_track_order` 先试严格 regex 再 fallback 到 lax `track_(\d+)_(\d+)` + 按 seq_idx 排序去重，覆盖两种格式。
  - **两条 frame-of-reference 规则（大g 关键 insight: "相机初始位置要加 translate，cuboid 不需要"）**：E0.3 NRE Lightning USDZ 容器里两个数据源在两个不同 frame，cross-source diff 实测铁证 clip 9ae151dc tid='18' parked 车：
    - 3dgrut2-own `viz_4d.tracks["18"][0].translation` = **(-51.30, 1.07, 1.42)** (NCore world frame)
    - NRE `sequence_tracks ["18"] raw 7-vec[0:3]` = **(-51.30, 1.12, 1.47)** — 差 0.05m
    两者**同 frame, cuboid pose 不需平移**；而 NRE state_dict 内 background/road gaussians 位于 NRE local frame = NCore world − (38.00, -2.155, -0.278)，**static 层需 +translate**。dynamic_rigids gaussians 是 **object-local frame**，render_pass 用 `track_pose @ object_local + t` 变换到 NCore world frame（**dyn 不加 translate**，跟 fervent-knuth 873L 注释 `node_offset = offset if spec.name != 'dynamic_rigids' else None` 一致）。
  - **视觉验证状态**：✅ background+road 静态层（继承 E2.7-A，对齐 ego trajectory 中段）；✅ dynamic_rigids gaussian 位置正确（大g 在人行道和十字路口看到"一坨一坨在移动"的物体）；✅ cuboid wireframes (大g 凭印象 t405 bus 位置正确)。⚠️ dynamic_rigids gaussian **颜色呈"烟雾"感**（半透明灰）→ E2.7-C follow-up。
  - **未完成根因（E2.7-C）**：cross-source 对比锁定到 features_albedo 数学不兼容。3dgrut2-own dyn features_albedo (N, 3) 是 **SH 空间** DC band 系数 (std=1.5, range=[-2.9, +13.7])；NRE dyn features_albedo (N, **20**, 3) 是 **Fourier-in-time** 时间系数 (DC 占 622% 能量但 std=0.42 range=[-1.9, +2.2])。amazing-lalande loader 直接取 NRE DC[0] 当 3dgrut2 SH DC 用 — bg/road 巧合数值范围接近所以 OK，但 dyn 车辆色彩随光照/阴影/夜间变化大 → DC = 时间平均 ≈ 灰色 → 烟雾感。
  - **架构选择教训（amazing-lalande 简版 vs fervent-knuth 873L 完整版）**：amazing-lalande 简版 (472L loader) 实现 Phase A 静态层后已能让 viser 渲染 background+road，但 dyn 接线和 features 转换都被简化；fervent-knuth 873L 完整版 Stage 2 包含 `parse_rig_trajectories` / `_assemble_dynamic_track_assets` / `build_viz4d_dict` / `build_layered_model_from_usdz` 一整套完整接线，移植 helpers 时学到的：USDZ 容器格式跟具体 NRE 训练 pipeline 强耦合（demo scene vs Lightning checkpoint），写 parser 必须考虑多容器变体。
  - **不阻塞 E2.7 主 KPI**：E2.7 视觉对标的主要价值（NRE 路面横向 3m/6m + 360° 不退化）E2.7-A 已经达成；E2.7-B 让 dynamic 车辆能看到位置 + cuboid 框 + 跟着 timeline 动，**够做 4D 对比**；颜色精细度交给 E2.7-C 跟进。

- **2026-06-16 E2.7-C 尝试 + 失败复盘（path A/C 实现但未解决问题；代码留分支 `e27c-dynamic-color-path-a`，未合 main、未 push GitHub，不要直接续用）** — 目标修 E2.7-B 的 dynamic 车辆"烟雾"感。**结论：问题定性错了（当成颜色问题，实为形状/几何问题），方向需重新分析。**

  **A. 实测确认的数据事实（这些是对的，可复用）：**
  - **(N,20,3) features_albedo 确为 Fourier-in-time**：checkpoint `hyper_parameters` 实锤 `model.layers.<layer>.fourier_features_dim` = dynamic **20** / background **5** / road **1**；per-k 能量 k=0(DC) 主导(0.855)、k≥1 ~0.1 平 = 典型 Fourier 签名。
  - **features_specular (N,45) = SH rest（degree 1-3，静态 view-dependent）**：`radiance_sph_degree=3` / `radiance_sph_O0=True`。即 albedo=SH DC band（被 Fourier 时变扩展）、specular=高阶 SH（静态）。两者是 SH 的两半，**45 与 20 无直接关系**（之前误以为要把 20 转成 45 的某种 SH，错）。
  - **time_embed 类型（path-A/C-v1 在此踩坑）**：dynamic = `individual-remap-time-input-embedding`（每 track 在自己活跃窗 `time_embed.timestamps_us_ranges[cuboid_id]` 内 remap 到 [0,1]）；bg/road = `holistic-remap`（全局 clip remap）；`remap_min=0/remap_max=1`。path A/C-v1 误用全局 `t/N_t` → 只对 bg(holistic) 蒙对、dynamic(individual) 错。ce63395 改 per-gaussian individual-remap 修正了时间维。
  - **SH→RGB 约定正确（不是 bug）**：renderer `threedgut_tracer/.../gaussianParticles.cuh:94 rad += 0.5f` + `SpHCoeff0=0.28209479177387814`，即 `rgb_01 = 0.5 + C0·sh`，RGB 在 **(0,1)** 非 (0,255)；`model.py:591 features_albedo=RGB2SH(colors/255)` 印证。dynamic SH-DC 均值 −0.81 → rgb≈0.27（暗灰）。
  - **ISP/ppisp（亮度差距来源）**：`model.post_processings.0.ppisp` 里 **exposure_params 全 0（无效）+ color_params 单位阵 + vignetting 极小**；唯一活跃亮度项是 **CRF tone curve（`crf_params` 6×3×7）**，闭源 slang shader，实测**非简单多项式**（求值非单调/出界）→ 无法可靠反推。即 loader docstring 早标注的 "appearance gap vs NVIDIA's renderer"。

  **B. 实现了什么（分支 `e27c-dynamic-color-path-a`，Mac 43 测试绿，NOT merged）：**
  `track_albedo_fourier.fourier_albedo_full` + `fourier_albedo_at_tnorm`（per-gaussian 归一化时间 cosine-Fourier eval）；loader `albedo_mode="fourier"`（保全系数 + ride `timestamps_us_ranges`）；`layered_model.fused_view` render hook（per-gaussian individual-remap：`t_norm=clamp((t−start)/(end−start),0,1)` 按 cuboid 时间窗）；viser buffer 注册 + CLI。commits：`ac3175e`(path-a) / `dab87a4`(path-c) / `ce63395`(individual-remap)。

  **C. 为什么仍失败（大g 一针见血：「即使颜色不对，外形至少要靠谱」）：**
  - individual-remap 让 dynamic 颜色**时变正确 + 有 hue**（不再纯灰），但**根本问题不是颜色，是形状**——dynamic gaussians 渲染成 **smeary 半透明条状雾团**，不成车/人形。
  - **几何实测根因（三层对比）**：dynamic opacity median **0.11**（48% <0.1，wispy）+ **极端各向异性**（max/min 轴比 p90 **107040**，针/片状 streaky）+ 每物体仅 **~3000 gaussian**（89k/28），在 3dgut renderer 里 alpha 合成**不出实心表面**。对照：road opacity **0.99** → 渲染实心干净；background opacity 0.08 但靠 **2.3M 密集堆叠**勉强成形。
  - **NVIDIA 自己的 renderer 能把同一批 gaussian 渲实心** → 问题在**我们这边加载/渲染**，非数据本身。**top suspect（未验证）**：各向异性片状 gaussian 要靠**正确旋转**才能贴成车体表面（如 road 扁盘贴路面）；若 E2.7-B `_transform_means_and_active` 的四元数约定错（NRE `rotations` wxyz vs xyzw / `q_world=q_pose⊗q_local` 组合方向），片状 gaussian 朝向乱 → streak。**位置对 ≠ 朝向对**（大g 验过位置，没验朝向）。

  **D. 教训 + 重新分析方向：**
  1. **定性教训**：E2.7-B「位置正确 + 随 timeline 动 = 够做 4D 对比」低估了形状；E2.7-C 把问题当"颜色/Fourier→SH"，方向错——真 blocker 是 **dynamic 几何在 3dgut renderer 的保真度（opacity / anisotropy / rotation / 跨 renderer 合成）**。albedo（颜色/亮度/CRF）是这些都解决后才轮到的**下游**问题。
  2. **流程教训（大g 反复纠正）**：① 交付前自己用 Claude in Chrome 先看效果，别裸交付；② 别只盯一个维度（颜色）忽略更基础的（形状）；③ viser 缓存（`~/.cache/3dgrut2/nre_usdz_pt/*.pt`）改 .pt 结构后须手动清，cache key 不含代码版本；④ ssh inceptio 抖动 + `pkill -f viser_gui_4d.py` 会误杀含该串的 launch shell（pkill 与 nohup 不要同条 ssh）。
  3. **下一步候选（不要直接续 path C，先重新分析）**：(A) 查 dynamic gaussian **旋转约定**是否让片状 gaussian 正确成面（最可能单点 bug，验证便宜）；(B) 评估 NVIDIA dynamic gaussian 在 3dgut renderer 的根本保真差距是否可弥合；(C) 重新审视"viser 加载 NVIDIA USDZ 看 dynamic actor"这条路本身是否合适。

- **2026-06-17 E2.5 目测 spike 完成**（commit `0350e34` 注入引擎+TDD / `21cf1c4` yaw-flip 修复；inceptio worktree e25，注入产物 `~/work/output/e25_ah_frozen/ckpt_e25_frozen.pt`）——AH 车 frozen 注入取代 recon 车 + viser+harmonizer 实时协调目测：
  - **工程**：新 `threedgrut/layers/e25_inject.py`（纯 CPU 注入核心：`build_name_to_int_id` 命门 / `match_assets_by_size` 尺寸贪心 / `aligned_to_node_tensors` 特征转换+specular 零填充 / `replace_tracks_in_dyn_node` per-track 子集替换 / `flip_forward_180` 朝向修正）+ `scripts/e25_inject_ah_replace.py` CLI（--dry_run/--mapping/--ensure_viz_4d）+ 从 PR#18 引入 `warmstart_{metadata,ply}.py` 对齐引擎（**不引** warmstart_inject——依赖不存在的 merge_warmstart_with_lidar）；**17 单测 Mac CPU 全绿**。
  - **注入**：3 AH consumer_vehicles 尺寸贪心配 recon automobile '316'/'372'/'24'（前两对 Δdims<0.1m 极贴，第三对最大 AH 车 H 1.88 填 recon '24' H 2.5 被拉高）frozen 替换；dynamic_rigids **300000→506146** 粒子；其它层/viz_4d/轨迹 byte-identical。
  - **命门验证**：注入车整数 track_id 复用 recon buffer id（= `sorted(viz_4d.tracks str keys)` enumerate，与 `_transform_means_and_active` 一致；实证 viz_4d 70 str key == layered_track_state pose tids，buffer id↔name↔size 三套标识齐对 → AH 车跟对轨迹）。
  - **目测结论（大g）**：① 首版车头车尾颠倒，根因 = PR#18 `_VEHICLE_AXIS_MAP` 标定 NuRec demo USDZ 朝向、NCore cuboid forward 差 180° → `flip_forward_180`（绕 object-local up 180°，det+1 非镜像）修复；② 修复后 **harmonizer 协调有效但有限**——开 harmonizer 注入 asset 与场景违和感明显低于不开、但未完全自然 → 教科书式 DiFix3D+ ablation（纯 2D 后处理提感知/降违和、消不掉几何/域差，须 E2.2/E2.3 蒸馏回 3D 补）。
  - **决策**：E2.5 目测验收通过 = **v5 编辑轴立项依据**；spec 三验收剩 NTA-IoU/FID 两项定量按大g 决定本次跳过留 v5；E2.7 系统性替换暂不升级。
  - **踩坑（呼应 E2.7-C 流程教训④）**：`pkill -f viser_gui_4d.py` 的 pattern self-match 杀掉执行它的 launch shell（命令行含该串）→ 改用 `pkill -f "[v]iser_gui_4d.py"` 规避；inceptio ssh 抖动多次（exit 255/无回显）→ 用独立简单 ssh + until-loop poll 验证规避。
- **2026-06-17 NRE USDZ 渲染开销对标（E0.3 `last.usdz`，inceptio 4090，nre-ga:latest，1080p，camera_front_wide_120fov，200 帧 frame-step3）** — 大g 任务：怀疑 deformable / dynamic_rigid 吃大算力 → **实测证伪**。Plan: [`nurec-usdz-4090-nurec-atomic-dewdrop.md`](../../.claude/plans/nurec-usdz-4090-nurec-atomic-dewdrop.md)；产物 inceptio `~/work/nurec_e0/profile/`（计时 log + `FINDINGS.md` + `structure.json` + 脚本 `drop_node.py`/`make_edit_json.py`/`analyze.py`）。
  - **Phase 1（ckpt vs usda）**：usdz 12 成员里 `checkpoint.ckpt` = 1.06GB（占 96%），**无 `.nurec`、无 nrend dict**；编辑内嵌 ckpt 删层重打包后渲染随之改变（render_cam 7.15→2.10ms）→ 坐实 **NRE render 读 usdz 内嵌 ckpt**。per-node splat（`export-artifact-structure`）：background 2,349,423（89.7%）/ road 118,930 / dynamic_rigids 89,285 / dynamic_deformables 61,292，TOTAL 2,618,930（=2.62M ✓）。
  - **Phase 2（ckpt-node drop ablation，按每帧 wall median Δ，权威口径）**：baseline **15.96ms/帧**（p95 17.30）。**background −5.11ms（32%，纯光栅化）最贵**；**dynamic_deformables −1.15ms（7.2%，每帧变形 MLP；keep=1 漏测 MLP 仅 −0.15，keep=0 才 −1.15）**；road −0.19ms（1.2%）；dynamic_rigids −0.19ms（1.2%）。固定开销 ~9.3ms（58%，raygen/FTheta 相机/sky/ISP，与场景无关）；冷启动 ~72.8s（每个新 render 容器一次性 JIT/TRT warmup）。
  - **结论：动态 actor 不吃算力（rigid 0.19 / deformable 1.15ms/帧），background 才是渲染大头；整帧 ~16ms ≈ 60fps。原始怀疑证伪。** 方法学：keep=1 漏固定 MLP→须 keep=0；`--enable-timing` stage 计时因 GPU 异步重叠 Σ>wall→以 per-frame wall median Δ 为准。
  - **附带（NRE viewer 交互卡顿排查）**：viser viewer 转视角 ~2-3fps 卡顿 = viewer render 线程**固有节流**（`render_trigger.wait(0.2)` + high/low 分辨率状态机），**与 actor 算力 / GPU / 网络 / 分辨率 / av_patch 均无关**——证据：GPU util 峰值仅 22%（没满载）、RTT 1.4ms 同 LAN、缩窗无效、客户端 WebGL 60fps、原版未打 patch 也卡。3dgrut2 viser 无此节流故顺（但渲染不了 deformable）。

- **2026-06-22 ⚠[OUT-OF-PLAN /loop 探索] road 层 off-track KPI 攻坚 — evaluation 改造 + AutoSplat/eff_rank 单变量 ablation**（**计划外探索：不在 §1.2 E*.* 任务看板；结论反哺 E1 测量门 / E3 表示侧 / E0.4 对标，非某 E 任务执行；可复用工程产物见 PR #34**。commits `b1ca3e8` eval改造 / `e1184aa` AutoSplat / `52a6106` driver / `21ecb31` 30k runner；branch `claude/hopeful-mirzakhani-56467d`；inceptio depth-off + num_workers=10；clip 9ae151dc）——大g /loop 任务：调研+应用 road-focused 3DGS SOTA，专攻 **road-only off-track KPI**（translation 3m/6m + rotation 10/30/60°，质量评估只渲 road、排除 bg/sky/dynamic-rigid）。
  - **评估改造（3 项新能力，21 单测 Mac 全绿）**：① `threedgrut/utils/novel_view.py` 补 `yaw_30deg`/`yaw_60deg`（保留 `LEGACY_NOVEL_AVG_MODES` 不动护历史 avg 锚，`_yaw_deg_from_mode` 角度解析）——补齐 2026-06-21 识别的 rotation gate 缺口（原仅到 yaw_10）；② `threedgrut/render.py` **road-only eval**（render_all 开头 `enabled_layer_names={"road"}` 单赋值关 bg/sky/dynamic-rigid，使 road_crop/lane/novel 指标度量 road 层自身而非 bg 代渲；`from_checkpoint(road_only=)` 走 dict-style conf 注入，旧 ckpt 兼容）；③ `scripts/eval_road_offtrack.py` 统一 from-ckpt off-track launcher（baseline 与 SOTA 同路径 → A/B 可比）。
  - **3k 单变量 ablation 结果**（前视 `camera_front_wide_120fov` · road-only · baseline=e36 5k ckpt 参考锚 / SOTA=3k 重训）—— off-track `lane_grad_corr`↑：**eff_rank(λ=0.01)** lat3m **0.279→0.301** · lat6m 0.238→0.245 · yaw10/30/60° **0.302/0.278/0.194→0.342/0.313/0.232**，`band_psnr` 全 5 档 **+0.2~+1.0dB**，on-track road_crop_psnr 18.40→**18.77**，**全档一致提升且 3k 即超 baseline 5k**；**AutoSplat(var_up 平面约束 λ=0.05)** 反而 band_psnr 全档降 −0.5~−0.7dB、grad_corr 多档微降、road_crop_psnr 18.40→18.03，**意外有害**（平面约束在 road-only lane 锐度口径下过压扁/扭转路面高斯）。配置经 parsed.yaml 实测确认单变量生效（autosplat 仅 planar=0.05、effrank 仅 eff_rank=0.01，均 depth-off 3k）。
  - **关键结论**：① **eff_rank（谱熵正则，V3-R1.3 早有但 v3 baseline 关为 0.0）是明确赢家**；② **新写的 AutoSplat var_up planar 反而退化** → 与文献预期（横移 FID +4.8~18.9）相反，口径/强度敏感；③ **单变量 ablation 兑现价值**——叠加全开会让 AutoSplat 负作用抵消 eff_rank 增益、得错误"无效"结论；④ **反直觉**：最有效的是翻开仓库已有开关而非新写 loss——eff_rank yaml 注释原标"⚠ A/B 实测 null 剔除"是**全图 lidar-on 口径**，本任务 **road-only off-track + rotation 口径**下它有效（口径决定结论）。
  - **caveat + 续跑**：baseline 5k vs SOTA 3k iter 不齐；但 autosplat/effrank 同 3k 严格可比，eff_rank 明确胜。**大g 拍板「3k 探路 + 赢家上 30k」→ `loop_effrank_30k`（eff_rank λ=0.01 depth-off 30k，55min @9.05it/s，2026-06-22 11:00 完成）— 混合信号**：on-track road_crop_psnr 18.40→**19.01**（+0.61）/ lpips 0.135→**0.125**（最佳）；off-track `grad_corr` lat3m 0.301→**0.309**（vs baseline **+0.030**）/ yaw10/30/60° 0.345/0.311/0.223 → **30k≈3k 持平·微升（lane 结构锐度收敛保持）**；**但 `band_psnr` 全 5 档大降 −1.7~−3.3dB**（3k 16.4 甜点 → 30k 13.5）。解读：**eff_rank 30k 让 on-track 拟合更实 + grad_corr（NuRec 对标主指标·阈值无关）保持，但 off-track 绝对亮度/颜色泛化退化**（band_psnr 绝对像素口径惩罚之）→ **iter 非越多越好，band_psnr 角度 3k 是甜点、grad_corr+on-track 角度 30k 略优**。**NuRec 同口径对标（2026-06-22 补，订正上面误导的全图 0.437）**：NuRec E0.3 baseline USDZ（`train_out/PVG7YYV72YKPLumogi7F7U/artifacts/last.usdz`，40k）经新写 `scripts/convert_align_nurec.py`（USDZ→.pt + `world_to_nre` 实测 translation `[-38.00,2.16,0.28]` → +38m 对齐回 NCore 帧，双重确认 = 9ae151dc）跑**同一个 `eval_road_offtrack`**（road-only 前视）→ 真正可比。**off-track lane_grad_corr：NuRec 全面领先且 gap 随幅度单调增大**——lat3m 我方 eff_30k 0.309 vs NuRec **0.352**（−0.043，最接近）/ lat6m 0.245 vs 0.324 / yaw10 0.345 vs 0.425 / yaw30 0.311 vs 0.417 / **yaw60 0.223 vs 0.441（−0.217，NuRec≈2×，我方最大短板）**；band_psnr NuRec 同样领先（除 eff_3k 小角度接近）。on-track road_crop_psnr 我方 19.01 vs NuRec 12.67 是**域差伪影**（NRE ISP/曝光 vs NCore GT 绝对颜色口径不同），非质量超越——公平看结构指标 grad_corr，NuRec 全胜。**结论：未达 NuRec**；eff_rank 把我方推近 translation 近档（3m gap −0.04），但 **off-track 幅度越大差距越大 = aperture problem 本质，纯 loss 单招补不满。 **[2026-06-22 大g 质疑后订正：原写的"两条腿(ground-mesh+Cosmos difix)"无实证依据、且与本次证据矛盾——本次对比用的 NuRec E0.3 baseline `difix.training.enabled:false`(E0.5 spec L169)，difix 后处理根本没参与，那条腿不成立。]** 据 E0.5 recipe-diff（对 NuRec `car2sim_6cam.yaml` resolved 实测）真正依据是**重建侧几何约束**（非后处理）：① **road 几何冻结四重锁**(lidar-ground-mesh init + road positions lr 1e-6 + `strategy.exclude_layer_ids:[road]` MCMC 全豁免 + z-scale 软正则)＝spec §9 标"**最可能单独解释官方路面外推优势的一条**"——我方 road 与 bg 共享全 lr + 全程参与 MCMC，路面高斯有完整自由度去迎合训练视角 → lateral/yaw 外推糊；② **LiDAR depth 监督**(NuRec ray 级全程 pin 住几何，我方本次 **depth-OFF** = inceptio 内存妥协)。**重大 caveat：本次非同配方对比**(我方 depth-OFF + road 全自由 vs NuRec depth-ON + road 冻结 + 40k)，gap 无法拆分到底来自哪个变量。**下一步控制变量＝移植 road 几何冻结(per-layer lr 冻结 + MCMC 豁免)同配方再比**(E3 主菜，spec L209)。 **[2026-06-22 road-freeze 控制变量实测 ＝ 反直觉结论]**：实现 per-layer absolute lr override(`layer_spec` 5 字段 + `layered_model._apply_layer_lr_overrides` 设 absolute lr + 删该组 scheduler) + MCMC 豁免(`layered_mcmc` `strategy.exclude_layer_ids`)，4 单测过；跑 ctrl(3k depth-off road 自由) vs road-freeze(同 3k depth-off + road positions lr 1e-6 / density·rotation·scale 1e-4 + `exclude_layer_ids:[road]`)单变量。**`road_pts=200000` 锁死(豁免生效)，但 road-freeze 全面变差**：off-track grad_corr 每档 **−0.045~−0.096**(lat3m 0.260→**0.188** / yaw10 0.286→0.190 / yaw60 0.185→0.140)、band_psnr **−3~4dB**、on-track road_crop_psnr 18.48→**13.53**(−4.96)。**根因：冻结只在 init 几何质量好时才有用**——NuRec 冻结的是高质量 lidar-ground-mesh init(RANSAC 平面 + 10 轮平滑 + density 0.99)，我方冻结的是 **noisy BEV-KNN 单点取 Z init**(spec L52 自承"噪声直接进 Z")→ 把路面焊死在错误几何、训练再无法修正 → 全崩。**再次纠正 E0.5 spec L200 判断**(原说"road 自由度是主因、冻结能改善")：控制变量证伪——**NuRec 优势 = ground-mesh init 质量 × 冻结的乘积，缺 init 质量光移植冻结适得其反**。我方现状(noisy init)排序：**eff_rank 软正则+自由(0.309) > 纯自由(ctrl 0.260) >> 硬冻结(0.188)**。**真正瓶颈 = road init 质量(E3.3 lidar-ground-mesh / BEV 平面化)，非几何自由度**；冻结须配合高质量 init 才有意义。inceptio ssh 踩坑记 § 关键不变量（高频连接限流 / `pkill -9` 占显存进程断 ssh / nohup 需 setsid+</dev/null detach / base64 长命令易 255 改走 git）。

---

- **2026-06-22→23 E3.2.5 完成（几何侧硬退化路面，reconstruction-studio 实证路径；代码 commit `2374161` + driver/verify/takeover 脚本 + 本回填同 PR；inceptio worktree e325，depth-off + nw=10，clip 9ae151dc）** — road 高斯压成真零厚度水平 disk，几何侧根治 aperture problem（路面厚度方向欠约束 → 横移/旋转车道线漂移）。spec [`2026-06-22-e325-recon-studio-ground-disk-geometry.md`](docs/superpowers/specs/2026-06-22-e325-recon-studio-ground-disk-geometry.md)。
  - **四步实现**：① `road_init.py` `knn_k`（默认 1=legacy 字节等价 / 5=k 近邻 LiDAR-Z 中值滤离群尖刺）+ LayerSpec `road_init_knn_k` + trainer 接线；② `scale_z_max` 5cm→1mm（preset，z 厚度 clamp 每步锁死，不用 1µm 防 3DGUT 零厚度协方差数值风险）；③ position `positions_lr=1e-6` 软冻 + rotation 新 `freeze_rotation_grad`（`layered_mcmc._post_backward` grad-zero，杀梯度+Adam 动量）+ `exclude_layer_ids=[road]` MCMC 豁免；④ 颜色 DC（multilayer 已 DC-only）。新 preset `ncore_3dgut_mcmc_multilayer_roaddisk.yaml`。决策细化：不设 `scale_lr`（z 靠 clamp 锁、xy 盘半径自由学习，对齐 recon-studio「冻 z-scale 不冻 xy-scale」）；保留 registry `anisotropy=8`（test 证 z 末尾 re-cap 回 1mm）。
  - **Mac TDD**：168 测试零回归（test_road_init 11 / test_road_freeze 8 / 新 test_road_disk_clamp 3 钉死 1mm×anisotropy 交互）。
  - **G1 数值 gate**：500 step / 10.29 it/s 无 NaN → 1mm 零厚度 disk 在 3DGUT UT sigma 点数值稳定（spec §6 风险证否）。
  - **G2 A/B（6k spike，on=roaddisk vs off=multilayer，inceptio depth-off 单变量）**：守护线 cc_psnr_masked on **24.01** vs off 23.60（**+0.41 不退反升**）；lateral lane grad_corr on/off **0.422 / 0.366 @3m（+0.057）· 0.309 / 0.270 @6m（+0.039）**；band_lpips on ≤ off。**on 全面优于 off**。
  - **freeze/clamp 端到端铁证**（on ckpt 200000 road，`scripts/verify_road_freeze.py`）：rotation tilt-from-identity **mean/p95/max = 0.0°**（法线竖直锁死，Adam 零漂移 → 证 grad-zero 强于 lr 压）/ z-scale **max/p95/median = 1.00mm**（clamp）/ road **N=200000 恒定**（exclude 无 densify）→ 对齐 recon-studio 法线 100% 竖直 + 零厚度 + 永不裁剪。
  - **viser 视觉对标（大g 2026-06-23）**：on 路面锐利、斑马线分明（定性印证 grad_corr +0.057）；road-only vs +bg 对比发现**车道线白条部分被 bg「抢走」**（白条渲染权落 bg 层不在 road）→ road/bg 所有权耦合，是 E3.2.5 几何管不到的另一半 → 衍生 **E3.2.6**（takeover 调强 spike）/ E3.1（空气区 penalty）。
  - **核心结论**：正面反驳 roadoff「光冻结几何变差」（§5 2026-06-22 out-of-plan 条目：冻结 noisy init → off-track grad_corr 全档 −0.05~0.10）——补齐 init 提质（KNN 中值）+ 真薄盘（1mm）+ 法线锁（rotation 硬冻结）三前提后，road 冻结翻转为 lateral lane grad_corr +0.04~0.06。**几何侧硬退化方向坐实，6k spike 通过，30k 全量待排期**。

## 6. 文档关系速查

| 想找 | 去哪 |
|---|---|
| v3 interpolated 主线（行人 Phase 2 / P3.2 / P1.x 剩余） | [`v3_plan_revised.md`](v3_plan_revised.md)（继续有效，与本 plan 并行） |
| NuRec 工具链全景 / 修复器细节 / license | [`nvidia-nurec-extrapolation-analysis.md`](../report/nvidia-nurec-extrapolation-analysis.md) |
| 领域综述（外推方法谱系 / 指标共识） | [`3d-4d-state-of-the-art-2025-2026.md`](../report/3d-4d-state-of-the-art-2025-2026.md) |
| NTA-IoU / LiDAR 点云 TDD 执行 plan | [`docs/superpowers/plans/2026-06-10-*.md`](docs/superpowers/plans/) |
| 架构图 / 文件清单 / 关键不变量（v4 新模块继续在此登记） | [`v2_architecture.md`](v2_architecture.md) |
| 执行约定（机器 / 配方 / 文档同步） | [`CLAUDE.md`](CLAUDE.md) |
