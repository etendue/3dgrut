# 3DGRUT v5 — inceptio 数据线提优（多相机 + cuboids + 360° LiDAR）· 可执行计划

> **本文档定位**：v5 **inceptio 数据线主 plan**。v5 唯一主题 = **让 inceptio 自有数据（inc_b6a9ed61 为主战 clip）的重建质量达到 NVIDIA pipeline（NRE）水平**：road/车道线可用、动态车辆清晰、多相机全量纳入训练，并有同 clip NRE 锚证明差距在收敛。
> **与 v4 的关系**：[`v4_plan.md`](v4_plan.md) 仍是 **外推（extrapolation）主线**（PAI 9ae clip 线，E2.2 渐进蒸馏等继续归 v4）；v4 的 **E5.1 / E5.2 移交本文档**（改编号 A1 / A2，执行与回填以 v5 为准）。两线并行不冲突：v4 = 方法轴（外推），v5 = 数据轴（inceptio 落地）。
> **决策依据（decision of record）**：
> - inc_b6a9 onboarding + road/dynamic 诊断：[PR #42](https://github.com/etendue/3dgrut/pull/42)（E5.0，v4 §5 Done Log 2026-07-02）
> - 4cab NRE 对照方法论：[`inceptio_4cabad44_3dgrut_vs_nre.md`](inceptio_4cabad44_3dgrut_vs_nre.md)（⚠️ 其 20.2 multi-cam 崩溃结论已被 2026-06-25 调查证伪，见 B3 勘误任务）
> - multi-cam 真相 + telew 实验：2026-06-25 angry-heisenberg 调查（6cam 真实 24.02、front_tele 18.04 根因 = 无 per-camera loss 权重；telew 实验 tele 18.04→26.24 有效但**代码未 commit 已丢失**，见 C1）
> - cuboid autogen 终局：[PR #40](https://github.com/etendue/3dgrut/pull/40)（纯几何天花板结论，2026-06-30 commit `2d32ea6`；收尾决策见 D2）
> - 业界调研（2026-07-02 三路）：外推蒸馏配方（FaithFusion 置信度加权 / FixingGS 连续小步）留 v4 E2.2 吸收；3D auto-label 开源链（CenterPoint+MOT）作 D2 备选记录
> - **2026-07-03 off-track 战役收敛（大g 拍板）**：v5 KPI 主轴升级为 **off-track 质量**（road + dynamic rigids，FID/lane 口径）；执行序 A（生成先验蒸馏，v4 E2.2）→ B（数据轴扩相机）→ C（官方底座，仅测量不作产品路径）；战役设计、算力调度与决策门见 [`docs/superpowers/specs/2026-07-03-offtrack-campaign-design.md`](docs/superpowers/specs/2026-07-03-offtrack-campaign-design.md)
> **执行约定**：沿用 [`CLAUDE.md`](CLAUDE.md)（inceptio 首选 / **depth-off + num_workers=10 铁律** / worktree 工作流 / 文档同步纪律 / Mermaid 全角括号）；**单变量 A/B 纪律**（同一对照只动一个变量）；**反伪造纪律**——本仓库已两次踩注入伪造数字坑（2026-06-17 / 06-25），一切训练数字必须 rich log × metrics.json 交叉验证后才可入档。

---

## 0. 目标与 KPI

### 0.1 v5 核心方向

三条事实链支撑（2026-07-02 定稿）：

1. **cuboid 缺口已实质打通**：inc_b6a9 带真 ppn_fusion cuboids（50 tracks / 1504 obs，36 动态 track 已进 dynamic_rigids 训练）——inceptio 数据第一次具备完整四层训练条件；4cab 时代"纯几何 autogen 红团"阶段结束。
2. **当前质量离 NVIDIA 水平有明确、可修的差距**：3-cam 30k 锚 mean_psnr 21.04 / lpips 0.645；road 层稀疏无色（根因 = 相机选择偏前向，lidar-sseg 92.9% ignore）、动态车模糊（根因 = 少视角欠约束 + cuboid ts 漂移 100ms）——全部有诊断、有解法（A1/A2）。
3. **数据独特优势未兑现**：12 相机 360° 覆盖（对外推是数据侧解法）+ 朝地面 multi-lidar（对车道线是天然监督）——扩相机阶梯（C）与 lane 锚（B2）负责兑现。

### 0.2 KPI — 以 b6a9 锚为起点，NRE 锚回填后定绝对目标

> ⚠️ 沿 v3/v4 纪律：**不设虚构绝对阶梯**。绝对目标数等 B1（NRE 同 clip 锚）测完才定；此前一切任务以「相对 2026-07-02 锚的 gap 闭合 + 守护线不退」验收。

| 轴（主 KPI） | 现状（2026-07-02 锚，3-cam 30k depth-off） | 测量工具 | v5 目标 |
|---|---|---|---|
| 全图质量 | mean_psnr **21.04** / cc 19.70 / ssim 0.578 / **lpips 0.645** | `render.py` metrics.json | 对 NRE 锚 gap 收敛（B1 回填） |
| road / 车道线 | road_crop_psnr 25.99 / lpips 0.254；**road 层稀疏无色**；lane 指标**未测** | per_class_eval + `compute_lane_metrics` | A1 后 road 有色非稀疏；B2 立 lane grad_corr 锚 |
| 车辆 class_psnr（动态区） | **17.53**（A3 已立锚 2026-07-02：automobile，299 records，72 条 <15dB） | [`class_psnr.py`](threedgrut/model/class_psnr.py) | A2/D1 闭合 |
| 行人 per-class（监控） | person 15.40（13 rec，无专属模型，同 v3 结论） | `per_class_eval.py` | 仅监控，不主动做（v5 不做行人建模） |
| NRE 同 clip gap | **未测**（4cab 锚 28.99 不可跨 clip 比） | B1（双臂）：nre-ga 官方配方 ± difix 蒸馏 | 把"落后多少"变实测数字 + C 路线天花板 |
| **off-track 质量（2026-07-03 新主轴）** | **B4 实测 2026-07-06**：held-out 侧相机 cc_psnr 旧锚 7.0 / R0c 9.7（vs train 19.5，gap **−10~−12 dB**）；扩相机收益 **+8.88 dB**（R0c held-out 9.66 → R1p 参训 18.54） | B4 held-out 真 GT ✅ + B5 novel 档 FID/KID | 门 1 判据；B5 novel 档补 → 门 1/2 后定目标 |
| 纳入训练的相机数 | 3 / 12 | per-camera psnr 表 | C 阶梯：5 → 10 → 11（+tele）→ 12（+鱼眼） |
| 守护线 | 现有 3 相机 per-cam psnr（22.07 / 20.71 / 20.31） | 现成 | 每步扩相机/改动后原相机不退 |

### 0.3 v5 不做（明确出界）

- **行人建模**（SMPL / rigid 垫脚石）——高速卡车场景行人稀疏，ROI 低（业界结论一致：deformable 轻量即可，且非本线主题）
- **外推蒸馏 E2.2 / Harmonizer 链**——留 v4 主线（PAI 线验证后再迁移 b6a9）
- **纯几何 cuboid autogen 继续调优**（L-shape / dimension prior）——4cab 已证天花板（recall 0.049 / init 错到 poseopt 救不回），正式关闭（D2 入档）
- 跨 clip 联训、closed-loop 仿真集成、relighting

### 0.4 v5 起点 baseline（不重训）

| 维度 | 数值 | 来源 |
|---|---:|---|
| b6a9 3-cam 30k 锚 | psnr 21.04 / cc 19.70 / lpips 0.645 / road_crop 25.99 | PR #42 E5.0（inceptio depth-off + nw=10） |
| per-camera | front_wide 22.07 / cross_left 20.71 / cross_right 20.31 | 同上 metrics.json |
| cuboids | 50 tracks / 1504 obs（动态 36 track / 50213 粒子进训练） | SDK 直读 + viser 诊断日志 |
| ckpt | `inceptio:~/work/output/inc_b6a9_3cam_multilayer_30k/…/ckpt_last.pt` | 2026-07-01 |
| telew 实验证据 | 4cab 6cam：front_tele 18.04→26.24、mean 23.93（**代码已丢，须 C1 重实现**） | 2026-06-25 调查（run 产物在 `~/work/output/inc4cab_multicam/`） |
| NRE 方法论锚 | 4cab：NRE 28.99 vs 3dgrut 单 cam 28.44（流程 runbook 现成） | [`inceptio_4cabad44_3dgrut_vs_nre.md`](inceptio_4cabad44_3dgrut_vs_nre.md) §8 |

---

## 1. 项目看板（Kanban）

> 状态：⬜ Todo · 🟡 In Progress · 🔵 Review · ✅ Done · ⏸ 降级 · ⏭ Skip

### 1.1 顶层看板（Mermaid Kanban）

```mermaid
%%{init: {'theme':'base'}}%%
kanban
    Backlog
        [B1 NRE 同 clip 对照锚（升级双臂）：官方 baseline 臂 + difix 蒸馏臂（Harmonizer IPC），gap 实测化 + C 路线天花板]
        [B2 lane 指标立锚：gen_lane_sseg 自跑 + compute_lane_metrics，兑现朝地 LiDAR 优势（冻结至战役门后）]
        [B3 文档勘误：vs_nre 报告 20.2 伪造结论撤回 + inceptio_5cam_task 结果回填]
        [B5 E1 外推度量移植 b6a9：novel 6 档 + FID/KID render-only·无悔棋]
        [C1 telew per-camera loss weight 重实现（上轮丢码）：光度项乘权 + 默认字节等价 + 必须 commit 进 main]
        [C2 扩相机 5→10 pinhole：单变量逐组加，per-cam psnr 全表 + telew 调权]
        [C3 加 front_tele（gate C1）：telew 加权纳入长焦]
        [C4 加 2 台 FTheta 鱼眼（最后，单变量）：上游 issue 238 尖刺风险，可弃]
        [D1 poseopt 迁移 b6a9（gate A2+A3）：P1.2 配方（boundary+prior+smooth）上 36 tracks]
        [D2 cuboid autogen 收尾决策：PR 40 去留 + eval yaw 约定定案 + 纯几何路线关闭入档]

    "In Progress"

    "Review"

    "Done"
        [B4 held-out off-track 锚 ✅ 2026-07-06：旧锚 held-out cc_psnr 7.0 / R0c 9.7 vs train 19.5，gap −10~−12dB；扩相机收益 +8.88dB（思路 B 首个实测证据）]
        [A1 road 修复 ✅ 2026-07-04（aux 遮挡 bug 2117× ＋ opacity 正则根治 ＋ lattice v2 收官；baseline 锚 R3p 20.25，inceptio 配方 yaml 入库）]
        [A5 pinhole cuboid mask 修复簇 ✅ 297a0bc（2026-07-02 新增：三处 FTheta-only gate ＋ behind-camera 过滤）]
        [A2 cuboid ts 插值 ✅ 6983018（per-camera END ts ＋ lerp、slerp 插值；wireframe 目检 cross 相机套准）]
        [A3 车辆锚 ✅（automobile class_psnr 17.53，299 records；person 15.40）]
        [A4 lidar 监督定论 ✅（30k 锚 parsed.yaml 实测 depth 全关＝铁律 CLI 覆盖，未生效属预期）]
        [继承: E5.0 inc_b6a9 onboarding ✅ 2026-07-02（PR 42：30k 锚 21.04 + viser exposure 修复 + aux 并行 6×）]
        [继承: cuboids ppn_fusion 数据打通（50 tracks 进 dynamic_rigids）]
        [继承: multi-cam 假崩溃勘误 + telew 实验证据（2026-06-25 调查）]
        [继承: 4cab NRE 对照 runbook（28.99 锚方法论）]
```

### 1.2 任务级看板

| ID | Phase | 主题 | 继承来源 | 估时(d) | 状态 | gate / 备注 |
|---|---|---|---|---:|:---:|---|
| **A1** ★ | A | **road 修复（根因改写）** — 步骤0 诊断推翻 E5.1 假设：非相机覆盖问题，系 **nre-tools lidar-seg 遮挡检查 bug**（1mm 容差×掠射路面×聚合 lidar spin mesh 误杀）；`NRE_LIDARSEG_OCCLUSION=off` 容器补丁后 road+sidewalk 21460（0.019%）→ **45.4M（40.08%），2117×**；6-cam aux 入 clip 目录；lattice v2 收官 → **baseline 配方 [`ncore_3dgut_mcmc_multilayer_inceptio.yaml`](configs/apps/ncore_3dgut_mcmc_multilayer_inceptio.yaml) 入库**（锚 R3p 20.25 / road_crop 24.47 / automobile 18.53） | **v4 E5.1 移交** | 1 | ✅ | 诊断脚本 `diag_lidar_sseg_vs_proj.py`（2f55017）；opacity 正则根治（`loss.use_opacity=false`，A800 双卡单变量坐实）；另修三类训练 NaN（ray 极点 52224b3 / 死层守卫 5f62cb0 / relocation 消毒 0b960eb）；PAI 线 multilayer 配方不动 |
| **A2** ★ | A | **cuboid 时间戳对齐** — [`tracks_loader.py`](threedgrut/datasets/tracks_loader.py) 按 per-camera END 时间戳精化 cuboid pose，消 cross 相机 ~100ms 漂移 | **v4 E5.2 移交** | 1 | ✅ | `6983018`：interp_pose_to_ts（lerp+slerp）+ `dataset.cuboid_ts_mode` 键（默认 ref_nearest 字节等价）；wireframe 目检 cross 相机 Δt=100ms 拖后 ~1m → 套准；训练收益走 R3 |
| **A3** ★ | A | **per-class eval 立锚** — 现成 [`class_psnr.py`](threedgrut/model/class_psnr.py)（cuboid-based 车辆）+ `per_class_eval.py`（person/rider）在 b6a9 ckpt 跑 eval → inceptio 首个车辆 per-class 锚 | v3 P0 工具复用 | 0.5 | ✅ | **automobile 17.53**（299 records，72 条 <15dB）/ person 15.40；依赖 A5 修 render.py FTheta-only gate 后才出字段 |
| **A4** | A | **LiDAR ray 监督确认** — b6a9 metrics 无 lidar_psnr 字段；查 multilayer resolved config + 训练 log 定生效与否 | E0.5 借鉴点⑤ | 0.5 | ✅ | 定论：**未生效**——parsed.yaml 实测 `use_lidar_depth=false`/`load_lidar_depth_map=false`（inceptio depth-off 铁律 CLI 覆盖），metrics 无字段属预期；A800 lidar-on 配方不受影响 |
| **A5** ★ | A | **pinhole cuboid mask 修复簇**（2026-07-02 大g 发现新增）— 三处 FTheta-only gate 在 pinhole clip 静默失效：trainer `_maybe_fill_cuboid_mask`（训练 dyn_mask_cuboid 从未生成）、render.py class_psnr eval gate（A3 缺字段根因）、共享 `project_cuboids_to_mask` pinhole 分支缺 behind-camera 过滤 | 大g 代码审查 | 0.5 | ✅ | `297a0bc`：z>0 corner 过滤两分支共用 + `resolve_batch_cuboid_intrinsics` 双模型 dispatch（FTheta 字节等价）；3D 路径（bg_cuboid_penalty/clamp）不受影响；旧行为可 `++trainer.bg_dyn_cuboid_penalty.use_cuboid_mask=false` 复现 |
| **B1** ★ | B | **NRE 同 clip 对照锚（双臂）** — 臂 1＝nre-ga car2sim 官方配方 baseline；臂 2＝同配方 + `difix.training.enabled=true`（Harmonizer IPC，单变量）→ 官方口径 + `nre render` lateral 3m/6m 帧 FID → **v5 gap 表首行实测化 + C 路线 off-track 天花板（门 1 输入）** | 4cab runbook + E0.7 IPC 架构 | 1.5 | ⬜ | 夜间 docker 挂机；**前置＝IPC 实物验证**（inceptio `~/work/nurec_e0/e07/` flags/日志/USDZ）；⚠️ 口径陷阱（官方 val 每 3 帧 + 1/4 res），对锚须统一口径或显式标注 |
| **B2** | B | **lane 指标立锚** — [`gen_lane_sseg.py`](scripts/gen_lane_sseg.py) 自跑 Mapillary lane sseg → `compute_lane_metrics`（grad_corr / band_psnr）前视立锚，验证朝地 LiDAR 车道线优势 | v3 P3.0 工具复用 | 1 | ⬜ | gate＝A1（侧相机进来 road 覆盖才够意义）；不跨 clip 比 PAI 锚 0.693 |
| **B3** | B | **文档勘误** — [`inceptio_4cabad44_3dgrut_vs_nre.md`](inceptio_4cabad44_3dgrut_vs_nre.md) 加勘误段（20.2 崩溃 + rational×MCMC 假设撤回，真实 6cam 23.2-24.0）；[`inceptio_5cam_task.md`](inceptio_5cam_task.md) 状态回填（已执行，5cam ~24.9@7k） | 2026-06-25 调查结论 | 0.5 | ⬜ | 防伪造数字再误导下游判断（本轮分析已被误导一次） |
| **B4** ★ | B | **held-out 侧相机真 GT off-track 锚** — 现有 3-cam 30k ckpt 在未参训侧相机位姿渲染 vs 真图（真 GT 外推，v4 E1.3 协议反用）→ held-out per-cam psnr/lpips + FID 与训练相机同口径对照 | v4 E1.3 协议 + E5.0 ckpt | 0.5 | ✅ | **2026-07-06 实测**（render-only，worktree @ a5083c8）：held-out cc_psnr 旧锚 7.0 / R0c 9.7 vs train 19.5（gap −10~−12 dB）；扩相机收益 +8.88 dB（思路 B 证据）；双源交叉一致；详见 §4 Done Log |
| **B5** | B | **E1 外推度量移植 b6a9** — novel 6 档（含 lateral 3m/6m）+ FID/KID（`--render-only` / `--novel-fid` 链路）在 b6a9 config 打通 | v4 E1.1/E1.4 工具 | 0.5 | ⬜ | render-only；**无悔棋三件套之一**；metrics.json 出 novel 档字段，与 B4 真 GT 互证 |
| **C1** ★ | C | **telew per-camera loss weight 重实现** — `trainer.py` 加 `_camera_loss_weight(camera_id)` + 光度项（L1/SSIM）乘权、正则项不动；`configs/base_gs.yaml` 加 `loss.camera_loss_weights: {}`（默认空 = 字节等价）；**必须 commit 进 main**（上轮实现验证有效但 worktree reset 丢码的教训） | 2026-06-25 调查 #6882 方案 | 0.5 | ⬜ | Mac 单测：weight=1 恒等 / weight=2 光度翻倍正则不变 |
| **C2** | C | **扩相机 5→10 pinhole** — 单变量逐组加 rear×2 / back_rear_wide / front_standard，telew 按 per-cam psnr 调权 | 新 | 1 | ⬜ | gate＝A1 + C1；守护线＝已有相机 psnr 不退 |
| **C3** | C | **加 front_tele** — telew 加权纳入（4cab 经验：无权重 18.04、加权 26.24） | 4cab telew 证据 | 0.5 | ⬜ | gate＝C1 |
| **C4** | C | **加 2 台 FTheta 鱼眼** — 最后单变量纳入 `camera_front_fisheye` / `camera_back_rear_fisheye`；FTheta 路径 PAI 已证（6cam 26.31），但留意上游 [issue #238](https://github.com/nv-tlabs/3dgrut/issues/238) 鱼眼尖刺 | 新 | 1 | ⬜ | 可弃：尖刺不可控则 10+tele 收口 |
| **D1** | D | **poseopt 迁移 b6a9** — v3 P1.2 配方（boundary anchor + prior + temporal smooth）上 36 真 tracks，对照 A3 锚看动态车清晰度 | v3 P1.2（class +1.03 证据） | 1.5 | ⬜ | gate＝A2 + A3 锚；4cab 教训不适用（那是 init 错，b6a9 是真标注） |
| **D2** | D | **cuboid autogen 收尾决策** — ① PR #40 去留（建议：merge 作离线工具保留，autogen 仅限无标注 clip 的 demo 用途）② eval `_yaw_of()` 约定定案（对已知角 box 直查 pose 矩阵，半小时）③ 纯几何 L-shape 路线正式关闭入档；备选记录：无标注 clip 未来走 CenterPoint+MOT 开源链换前端、复用 PR #40 V4 shard 基建 | PR #40 + 2026-06-26 L-shape session | 0.5 | ⬜ | 决策级任务，大g 拍板 |

### 1.3 Phase 状态汇总

| Phase | 主题 | 任务数 (Done/Total) | 主验收 | 守护线 | 状态 |
|---:|---|---:|---|:---:|:---:|
| **A** ★ | b6a9 质量解锁（短刀，全部有诊断有解法；+A5 新增） | **5/5** | road 有色 + cuboid 对齐 + 车辆锚入档 + lidar 监督定论 + pinhole gate 修复 | 3-cam per-cam psnr 不退 | ✅（[PR #44](https://github.com/etendue/3dgrut/pull/44)） |
| **B** ★ | 对标定锚 + off-track 评估（战役无悔棋） | 1/5 | NRE gap 双臂实测化 + held-out off-track 锚 + novel FID 链路 + lane 锚 + 伪数字勘误 | — | 🟡 |
| **C** | 扩相机阶梯 3→12 | 0/4 | 12 相机全量纳入或明确收口点 | 每步原相机不退 | ⬜ |
| **D** | 动态质量 + 收尾 | 0/2 | poseopt 增益入档 + autogen 去留定案 | class_psnr 不退 | ⬜ |
| **总计** | — | **6/16** | — | — | — |

### 1.4 任务依赖图

```mermaid
flowchart TD
  classDef gate fill:#e6f4ff,stroke:#0070f3,color:#000,font-weight:bold
  classDef todo fill:#f5f5f5,stroke:#999,color:#333
  classDef opt fill:#fbe9e7,stroke:#d33,color:#900

  E50["E5.0 onboarding ✅（继承 v4）<br/>30k 锚 21.04 / cuboids 打通 / 诊断入档"]:::gate
  A["Phase A 质量解锁<br/>A1 road 相机 ＋ A2 cuboid ts（并行）<br/>→ A3 per-class 锚 ＋ A4 lidar 确认"]:::gate
  B["Phase B 对标定锚<br/>B1 NRE 锚（可与 A 并行·GPU 排队）<br/>B2 lane 锚（gate A1）＋ B3 勘误"]:::gate
  C["Phase C 扩相机阶梯<br/>C1 telew 重实现 → C2 十 pinhole<br/>→ C3 加 tele → C4 加鱼眼（可弃）"]:::todo
  D["Phase D 动态质量<br/>D1 poseopt（gate A2＋A3）<br/>D2 autogen 收尾决策"]:::opt

  E50 --> A
  E50 --> B
  A -->|A1 定相机集| B
  A -->|锚就绪| C
  A -->|A2 A3| D
  B -.gap 数字重排优先级.-> C
```

> 并行性：A1 与 A2 不同文件域可并行；B1（docker 挂机）可与 A 并行排 GPU；B3/C1/D2 为纯 Mac/文档任务可穿插。
> **执行序（2026-07-03 战役版，覆盖旧建议）**：无悔棋三件套（B4 → B1 双臂 → B5）先行 3-4 天 → A1/A2（A1 重训排新 4090，到货前 aux 先备）→ 门 1 后按数字排 C/D；**B2/C3/C4/D2 冻结至战役门后**。E2.2 主线在 v4 执行（inceptio 白天 + A800 蒸馏臂），算力调度详见战役 spec §4。

---

## 2. Phase 详细任务卡

### 2.1 Phase A — b6a9 质量解锁

**A1 road 相机选择修复**
- 目标：lidar-sseg road+sidewalk 点 21460（0.3%）→ 数量级提升，road 层（roaddisk 冻结前提 = 好 init）变密集有色。
- 步骤意图：① 按 CLAUDE.md「nre-tools aux 多容器并行」runbook 给 2-3 台侧/后相机补 sseg + lidar-sseg/camvis（注意 itar write-once、并发容器不共享目录、`merge_lidar_aux.py` 合并）；② `dataset.camera_ids` 扩 5-cam 重训 30k（multilayer，inceptio 铁律 depth-off + nw=10）；③ 无代码改动，纯 config/CLI。
- 验收：诊断脚本输出 road 点数对比；viser road-only 视图有色连续（对照 E5.0 无色截图）；mean_psnr / road_crop 对 21.04 / 25.99 不退且 road 侧改善；新相机 per-cam psnr 入档。

**A2 cuboid 时间戳对齐**
- 目标：cross 相机 cuboid pose 漂移 ~100ms → ≈0。
- 改动：[`tracks_loader.py`](threedgrut/datasets/tracks_loader.py) 的 cuboid ts ↔ 相机帧匹配逻辑，改为按 per-camera END 时间戳精化（插值 track pose 到各相机实际曝光时刻）。
- 测试要点：Mac 纯函数单测——合成匀速 track + 已知相机 ts 偏移，断言精化后 pose 位置残差小于厘米级公差；默认路径与旧行为字节等价开关。
- 验收：`scripts/validate_cuboid_pretrain.py` cross 相机 wireframe 目检套准（对照 dt=100ms 旧图）；重训后动态区清晰度以 A3 class_psnr + viser 目检双读数。

**A3 per-class eval 立锚**
- 目标：b6a9 首个车辆 class_psnr（cuboid-based，36 tracks）+ person/rider 锚入档。
- 步骤意图：在现有 30k ckpt（及 A1/A2 后的新 ckpt）上跑 `render.py` eval，确认 metrics.json 出 by_class 字段（v3 P0 链路已通，如缺字段按 CLAUDE.md 把关清单核查 trainer/render 双路径）。
- 验收：车辆 by_class + person 数字写入本文档 §4 Done Log 与 §0.2 KPI 表。

**A4 LiDAR ray 监督确认**
- 目标：定论 b6a9 训练中 LiDAR ray 级监督是否生效（metrics 无 lidar_psnr 字段的疑点）。
- 步骤意图：查 run 的 parsed.yaml lidar 相关键 + 训练 log；若未生效，单变量 A/B（on vs off，6k 短跑即可）。
- 验收：结论 + 原因入档；若开启有益则进 b6a9 baseline 配方。

### 2.2 Phase B — 对标定锚

**B1 NRE 同 clip 对照锚（双臂，2026-07-03 升级）**
- 目标：把「b6a9 21.04 落后 NVIDIA 多少」从推测变实测（区分 pipeline 差距 vs 场景难度——36 动态车的 urban 场景 PSNR 天然低于 4cab 单卡车高速）；臂 2 同时给出 **C 路线 off-track 天花板**（战役门 1 输入）。
- 步骤意图：臂 1 复用 4cab runbook（nre-ga car2sim 配方 docker 一条命令）；臂 2 同配方单变量开 `difix.training.enabled=true`，修复器走 Harmonizer IPC（`fixer_server.py`/`harmonizer_server.py` 架构，**前置＝IPC 实物验证** `~/work/nurec_e0/e07/`）；两臂各出官方指标 + `nre render` lateral 3m/6m 帧 → FID 对比。
- 验收：v5 gap 表首行回填 + 两臂 off-track FID 差入档（门 1 判据）；口径统一或显式标注官方 val 口径陷阱；据 gap 数字重排 C/D 优先级（对标 v4 E1.5 纪律）。

**B4 held-out 侧相机真 GT off-track 锚（零训练，无悔棋）**
- 目标：b6a9 第一个真 GT 离轴数字——现有 3-cam 30k ckpt 从未见过侧相机，在侧相机位姿渲染 vs 真图即真外推测量（v4 E1.3 协议反用）。
- 步骤意图：eval 侧相机集注入（`dataset.camera_ids` eval-only 覆盖或 render.py 位姿加载路径）；侧相机帧只需图像+位姿，不需 sseg aux；render-only 出帧后与真图算 per-cam psnr/lpips + FID。
- 验收：held-out 数字与 3 台训练相机同口径对照入档（§4 Done Log + §0.2 KPI 表 off-track 行）；回答「b6a9 离轴差多少」。

**B5 E1 外推度量移植 b6a9（无悔棋）**
- 目标：v4 E1.1/E1.4 工具链（novel 6 档含 lateral 3m/6m + FID/KID）在 b6a9 config 打通，补齐「inceptio 数据无 off-track 评估」的洞。
- 步骤意图：`--render-only` / `--novel-only` / `--novel-fid` 链路对 b6a9 manifest 跑通；配置差异（相机数/分辨率）按需适配。
- 验收：b6a9 metrics.json 出 `mean_novel_fid_*` 等 novel 档字段；与 B4 真 GT 数字互证入档。

**B2 lane 指标立锚**
- 目标：兑现朝地 multi-lidar 的车道线优势，建立 b6a9 lane grad_corr / band_psnr 锚。
- 步骤意图：`gen_lane_sseg.py` 自跑（b6a9 无 lane aux）→ `datasetNcore` 加载 → `render.py` 前视 eval 出 mean_lane_* 字段（v3 P3.0 全链路现成）。
- 验收：lane 锚数字入档；与 road_crop/road 层视觉互证。

**B3 文档勘误**
- 目标：清除两处会误导后续判断的过期结论。
- 改动：[`inceptio_4cabad44_3dgrut_vs_nre.md`](inceptio_4cabad44_3dgrut_vs_nre.md) §5 加勘误段（20.2 崩溃数字系注入伪造已撤回；真实 6cam@7k 23.2-24.0；rational×MCMC 失稳假设不成立，真根因 = per-camera loss 权重缺失 + 多视角稀释）；[`inceptio_5cam_task.md`](inceptio_5cam_task.md) 状态"待执行"→ 已执行 + 结果回填。
- 验收：两文档更新，引用该结论的下游文档无残留。

### 2.3 Phase C — 扩相机阶梯

**C1 telew 重实现**：见 §1.2 行内描述；关键约束——只乘光度项、默认空 dict 字节等价、CLI 以 `++loss.camera_loss_weights.<camera_id>=w` 覆盖；完成定义 = **代码 + 测试 merge 进 main**。
**C2 → C3 → C4**：每步单变量、守护线 = 已纳入相机 per-cam psnr 不退；C4 鱼眼为可弃项（尖刺不可控则在 11 相机收口，FTheta 数据侧无阻碍）。

### 2.4 Phase D — 动态质量 + 收尾

**D1 poseopt 迁移**：`trainer.pose_adjustment.enabled=true`（lambda_t 1e-2 / lambda_r 1e-1，v3 P1.2 已证配方）30k A/B，验收 = A3 车辆 class_psnr 提升 + viser 动态车抖动目检。
**D2 cuboid autogen 收尾**：决策任务——PR #40 merge 与否、eval yaw 约定半小时定案、L-shape 路线关闭结论入档、无标注 clip 的 CenterPoint+MOT 备选路线记录（详见 §1.2）。

---

## 3. 风险登记表（Risk Log）

| # | 风险 | 触发条件 | 缓解 | 状态 |
|---|---|---|---|---|
| R1 | 加侧相机后 road 覆盖仍不足 | A1 验收不过 | **已解除（2026-07-02）**：根因根本不是相机覆盖——nre-tools lidar-seg 遮挡检查 bug 修复后 road 点 40.08%，无需备选路线 | ✅ 解除 |
| R2 | 跨相机曝光/白平衡差异随相机数放大 | C2-C4 cc 与 raw psnr 差扩大 | C1 telew + BilateralGrid exposure（已默认开）；per-cam 光度监控 | ⬜ |
| R3 | 注入伪造数字（已踩两次） | 任何训练数字入档前 | rich log × metrics.json 交叉验证；只认两源一致的数字 | 长期 |
| R4 | 单卡 4090 排队 | A1/B1/C2 训练冲突 | 长任务 setsid 驱动脚本串行；B1 docker 可夜间挂机 | ⬜ |
| R5 | NRE 官方口径陷阱 | B1 对锚 | E0.3 教训：官方 val 每 3 帧 + 1/4 res + cpsnr，须统一口径互渲或显式标注 | ⬜ |
| R6 | 鱼眼尖刺（上游 issue #238） | C4 | 放最后、单变量、可弃；必要时向上游报 issue | ⬜ |
| R7 | itar 损坏（write-once + 中途 stop） | A1 aux 并行 | PR #42 runbook：硬链隔离目录 + 完整跑完再合并 | ⬜ |

---

## 4. Done Log（继承锚点 + 新条目）

**继承锚点（已验证，作 baseline / 方法论基础）**：
- **2026-07-02 E5.0 inc_b6a9 onboarding**（PR #42）：3-cam 30k 锚 psnr 21.04 / cc 19.70 / lpips 0.645 / road_crop 25.99；cuboids 50 tracks 打通（36 动态进训练）；viser exposure 修复；aux 并行 6× runbook；road/dynamic 根因诊断（→A1/A2）。
- **2026-06-25 multi-cam 真相调查**：「6cam 崩 20.2」系注入伪造已撤回；真实 6cam@7k 24.02（refix 23.24）、5cam ~24.9、2cam 26.69；front_tele 18.04 根因 = per-pixel mean 无相机权重；telew 实验 tele→26.24 有效（代码丢失 → C1）。
- **2026-06-30 cuboid autogen 终局**（PR #40 + `2d32ea6`）：纯几何天花板坐实（recall 0.049 / yaw 65° / poseopt 救不回错 init）；V4 shard 写读基建可复用（→D2）。
- **2026-06-24 4cab NRE 锚方法论**：NRE 28.99 / 3dgrut 单 cam 28.44，runbook 现成（→B1）。

**新条目**（任务完成后按 CLAUDE.md 纪律追加：日期 + commit + 实测数字）：

- **2026-07-06 ★ B4 held-out 侧相机真 GT off-track 锚**（render-only，inceptio worktree @ a5083c8；门 1 判据之一）：
  - **四组 render**（旧锚/R0c × train/held-out，`--dataset-cameras` 替换 camera_ids + `--novel-fid`，exposure 自动禁用→cc 口径）：run1 旧锚×train cc_psnr **19.46**/FID 330 · run2 旧锚×held-out **7.00**/408 · run3 R0c×train **19.60**/114 · run4 R0c×held-out **9.66**/384。双源交叉（rich log × metrics.json）全一致；run1 gate cc19.46 pass 钉死 held-out 低分＝**真实离轴崩溃**（非 pinhole×--dataset-cameras bug）。
  - **结论① held-out−train gap（off-track 离轴差距，KPI off-track 行）**：旧锚 **−12.46 dB**、R0c **−9.95 dB**——b6a9 侧后相机（left/right/back_rear_wide）离轴质量从 ~19.5 崩到 7-10 dB cc_psnr。
  - **结论② 扩相机收益（单变量 R1p=R0c+6cam，同三台侧后相机 参训 vs 未参训）**：R0c held-out 9.66 → R1p 参训 18.54 = **+8.88 dB**（left +6.73 / right +10.96 / back +6.52）——**思路 B（扩相机改善离轴）首个实测证据**：扩相机把侧后相机从「离轴崩溃」救到「参训正常」；R3p 参训 18.72 旁证。
  - 口径注记：FID/KID 72 帧/组（<2048 有偏，仅四组内部 A/B）；cc_psnr per-frame 仿射拟合曝光鲁棒，故 R1p/R3p（带 exposure）与四组（exposure 禁用）可比；旧锚/R0c 均 3-cam 前向训练，held-out=未训侧后。底稿 `.superpowers/sdd/b4_summary.md`；metrics 在 `inceptio:~/work/output/b4_{old,r0c}_{train,heldout}/`。
  - 执行注记：版本口径 worktree @ a5083c8（大g 拍板，避 inceptio 主仓库 daa8ce5 分叉 + 未提交 E2.6 工作）；Task 3 subagent stalled + Mac 断网各遇一次，均 controller 接管（batch setsid 后台不受影响），数据零丢失。

- **2026-07-04 ★ baseline 固化（大g 决策：先立 baseline 再逐个解 issue）+ R2p/A800 depth A/B 出数**：
  - **baseline = `configs/apps/ncore_3dgut_mcmc_multilayer_inceptio.yaml`**（compose multilayer + 6-cam camera_ids + `loss.use_opacity=false`；`# @package _global_` 头必需，Mac compose 验证 6 相机/正则off/mask on）。**锚 = R2p 30k：mean 20.11 / cc 18.54 / ssim 0.640 / lpips 0.629 / road_crop 24.51 / automobile 18.54（<15dB 38/346）**；per-cam front 21.16 / cross 20.18/19.90 / left 18.44 / right 21.24 / back 19.58。
  - **A5 单变量增益坐实（R2p vs R1p）**：automobile **+1.15**、劣质记录 73→38（−48%）、road_crop +0.63，mean/ssim/lpips 持平，front −0.43（监督预算向 dyn 重分配）——cuboid mask 留在 baseline（默认 on）。
  - **A800 depth A/B v2（正则off 干净版）**：lidaron 20.22/road 24.38/auto 18.19/lidar_psnr 25.22 vs depthoff 20.17/24.24/18.41 → **depth 监督对 b6a9 6-cam@30k 中性**（差异噪声级）；A800 20.2 与 inceptio 20.1-20.2 跨机咬合（健康互证，不作跨机锚）。
  - 事故记录：7/3 夜 R2p/R3p 未跑——/tmp sed 裁剪驱动继承 `cd $(dirname $0)/..` 落到根目录（`python //train.py` 秒死）；教训＝发射后必须验证进程存活；正式驱动 `fc93bd3` 入库。
  - **PAI 数据回归 A/B 通过（PR #44 合并前证据，大g 提问触发）**：inceptio 双 worktree（main `5819316` vs branch `a3a4ecf`），PAI 9ae clip、multilayer 原配方、5s 窗 5k 步（快测约定首秀）、depth-off+nw=10 双侧同覆盖。**main 22.499/lpips 0.4462 vs branch 22.587/lpips 0.4496 → Δpsnr +0.09（<0.3 判据，run-to-run 噪声级）**，branch 侧 dead=0/nonfinite=0（新守卫在 FTheta 路径全程未触发 = no-op 实证）。配置层：resolved config diff 仅一个默认等价新键 `cuboid_ts_mode: ref_nearest`。结论：**分支对 PAI 线无回归**；inceptio 配方改名 `_inceptio.yaml` 与 PAI multilayer 完全分离。
  - **lattice v2 收官（12:20 ALL COMPLETE）**：R3p（+A2 interp）**20.25 / cc 18.72 / lpips 0.627 / road_crop 24.47 / automobile 18.53**，front +0.47 / back +0.54（vs R2p），无回退项（唯 <15dB 记录 38→48），告警仅 left_wide 极点修复预期 3 次。**判定：`cuboid_ts_mode=per_camera_interp` 升级进 baseline，锚改记 R3p**（yaml 头注释同步）。lattice v2 单变量全链完整：aux 修复 +0.26（R0b）→ 正则权衡（R0c）→ 相机集（R1p）→ A5 +1.15 auto（R2p）→ A2 +0.15 mean（R3p）。
  - **Issues 清单（baseline 之上逐个解，带证据与优先级）**：
    | # | issue | 证据 | 候选解 | 优先级 |
    |---|---|---|---|---|
    | I1 | 6-cam 守护线对 3-cam 锚退 0.4~0.9（欠训） | 6-cam 数据×2、30k=每相机有效步数减半 | **R2p 配方 60k 等比重训** | ★高 |
    | I2 | 车辆仍糊（18.54，虽 +1.15） | dyn 层 30k + 36 tracks | R3p（A2 interp，跑数中）→ D1 poseopt | ★高 |
    | I3 | road 层颜色贡献度不足（road-only 偏暗） | road-only 渲染图 2026-07-03 | E3.6 takeover 完整版（bg init 剔 road 点 + road 补密） | 中 |
    | I4 | opacity 正则修剪红利丢失（3-cam mean −0.46） | R0c vs R0b | 分层 λ / dyn 豁免 / 可见性加权 | 中 |
    | I5 | viser OPCV 初始 fov 未同步（涂抹） | 2026-07-03 终审 | 一行修（芯片 task_aed6e22a） | 低（有使用姿势绕过） |
    | I6 | left_wide 极点像素 + 最弱相机（18.44） | ray 极点已防护；per-cam 表 | 去畸变 shader 修（上游）+ C1 telew 权重 | 低 |
    | I7 | B 阶段：NRE 同 clip gap 未实测 | B1 未跑（A800 无 nre-ga 容器） | 拉容器或 inceptio 夜间挂机 | 中 |
    | I8 | **ego mask 全空**（6 相机 0.00% 覆盖，自车完全未排除；2026-07-07 大g 目测 R3p → 实测坐实） | **实测**：6 相机 manifest `get_mask_images()['ego']` nonzero=0/2073600（全黑）→ 车头+后视镜+支架全被当场景重建；aux-meta `ego_mask=false`/`ego_mask_camera_ids=[]` 佐证 `--ego-mask` 未生效或 converter 写空占位（同"converter 没接 cuboids"类问题）。原始 R3p f048 render/GT 左下卡车支架被清晰重建 | ① 查 `aux.egomask.itar`(165KB) 有无真内容（区分"从没生成"vs"生成没接训练"）② 重生成 ego mask（`--ego-mask` 正确跑）或手动补卡车自车 ROI filter ③ 大g 推测 sseg 对卡车 ego OOD 可能是生成失败主因之一 | **中高**（自车像素占监督预算+被 sseg 误标污染层+off-track 视角错位；6 相机全中招） |

- **2026-07-02/03 任务A 主体执行（A2/A3/A4/A5 ✅ + A1 根因改写与 aux 修复 + 三类训练 NaN 排障；branch `claude/laughing-lichterman-a52733`）**：
  - **A1 步骤0 诊断（`2f55017` diag_lidar_sseg_vs_proj.py）推翻 E5.0 结论**：thinkpad lidar-cam 投影图（大g 提供，已存 `inceptio:~/work/data/inc_b6a9ed61_20s/proj_ref/`）× camvis × 几何重投影三方对照——前向 3-cam 明明看得到大量路面点（front_wide 每 sweep ~7 万点落 road 像素却 100% 被标 ignore），`P(ignore|camvis>0)=0.875`。真根因＝**nre-tools lidar-seg 遮挡检查**（`estimators.py` spin-mesh 射线求交 1mm 容差 × 掠射路面几何 × 聚合 multi-lidar 破坏 spin 有序性 → 误杀）；vegetation/building 近垂直入射不受害，故 E5.0 误判为相机覆盖问题。
  - **aux 修复（容器 bind-mount 补丁 `NRE_LIDARSEG_OCCLUSION=off`，补丁文件 `inceptio:/tmp/estimators_patched.py`）**：road+sidewalk **21460（0.019%）→ 45,423,991（40.08%），2117×**；`P(ignore|camvis>0)` 0.875→**0.0069**；附带解锁速度 105s/帧→0.7s/帧（60×，慢的元凶就是求交）——多容器并行 runbook 对 lidar-seg 不再必要。6-cam aux（原生重跑 sseg+egomask 13min + lidar-seg 200 帧 2min）已入 clip 目录，旧 3-cam aux 备份 `aux_3cam_backup/`；merge_lidar_aux 修跨相机集并集 bug（`2c18848`）；⚠️ 自写 merge itar 缺 `.zmetadata.cbor.xz`（consolidated metadata）会让 nre-tools 容器 KeyError——容器输入一律用原生 itar，训练侧读非 consolidated 路径不受影响。
  - **A5（`297a0bc`，大g 发现）**：pinhole clip 三处 FTheta-only gate 静默失效（trainer cuboid mask / render class_psnr / 共享投影 behind-camera 过滤）全部修复；FTheta 字节等价（既有测试全绿证明）；sanity 日志 `[A5] dyn_mask_cuboid filled via OpenCVPinhole K` 确认训练接线。
  - **A3 车辆锚**：旧 3-cam 30k ckpt 重跑 eval（A5 修复后字段才出现）——**automobile mean_class_psnr 17.53**（299 records，72 条 <15dB）、person 15.40；mean/cc/lpips 21.04/19.70/0.645 与锚双源一致。
  - **A4 定论**：30k 锚 parsed.yaml 实测 `use_lidar_depth=false / load_lidar_depth_map=false`（inceptio depth-off 铁律 CLI 覆盖）→ lidar ray 监督未生效、metrics 无 lidar_psnr 属预期；A800 lidar-on 配方不受影响。
  - **A2（`6983018`）**：`interp_pose_to_ts`（平移 lerp + 四元数 slerp 短弧）+ `dataset.cuboid_ts_mode`（默认 `ref_nearest` 字节等价 / `per_camera_interp` union 时间轴）+ trainer/viz inject/validate 三处接线；单测 11 例（40ms 偏移：nearest 0.8m 漂移 vs interp <1e-4m）；wireframe 目检 f185 cross_right Δt=100ms 线框拖后 ~1m → interp 套准。
  - **三类训练 NaN 排障（6-cam 首训连环撞雷，全部修复+回归测试）**：① **ray 生成极点**（`52224b3`+哨兵 `8699b30`/`6ea5097`/`794e269`）——left_wide rational 畸变分母极点落图像角内，固定像素 (1917,1042) 每帧 1 个 NaN ray（三次复现同坐标坐实）→ `repair_nonfinite_rays` 预计算修复+像素永久 invalid；sanitize 实验证明 0·NaN=NaN 梯度穿透（1px→下帧全图 NaN），loss 侧掩蔽不可行。② **MCMC 全死层守卫**（`5f62cb0`）——空存活集 multinomial CUDA invalid-configuration。③ **relocation kernel 输出消毒**（`0b960eb`）——Eq.9 binomial 在 opacity→1 边界产非有限值经 log 逆激活直写参数（绕过 loss，bg 层 60% density NaN、死层告警 153 次、终致 illegal memory access）→ 输入 clamp + 坏行回退 donor。
  - **lattice（单变量链，driver `4fc0f5d`/`8cf5e5b`）**：sanity ✅（ckpt 零 NaN + [A5]/road-init 接线证据）；**R0b（3-cam+新aux）✅：21.30 / cc 20.04 / ssim 0.585 / lpips 0.631 / road_crop 26.34 / automobile 17.59**——对锚 21.04/19.70/0.578/0.645/25.99 全线改善，per-cam 22.34/21.06/20.46 守护线零退（aux 修复单变量增益 +0.26 dB）。
- **2026-07-03（下午）第四类训练故障根治：全局 opacity 正则杀死非豁免层（6-cam 拦路虎 + R0b 车辆糊的深层共因）**：
  - **现象**：6-cam 30k 三个独立 run（inceptio R1 + A800 双卡）bg 层 100 万粒子在 iter 1000-3000 集体死亡（opacity≤0.005，死层守卫防崩但层不复活）→ 全图 18 dB 量级；TB `loss/opacity` 曲线 iter 2200-2600 悬崖（3-cam 同点位无事）。⚠️ 死层告警里 `400000/1000000` 是 max_relocation_fraction=0.4 截断显示（`80808cf` 已修文案）——实为全死，无 NaN。
  - **A800 双卡二轮单变量（agent 托管）坐实**：S-b（bg_road_penalty off）死亡仅推迟 6k→7.9k（penalty 只是加速器，双向排除）；**S-c（`loss.use_opacity=false`）bg 0%→99.84%、dyn 15.8%→99.94%、8k 全指标最优（mean +1.84 / automobile +1.65）**；S-d（λ/10）bg 苟活但 **dyn 照死 87%**（降档救不了稀疏光度层）。机制＝`base_mcmc.yaml` 默认 λ=0.01 的无可见性加权 L1 每步全量下压 + 6-cam 采样稀释翻转力量对比 + road 豁免不对称（road 独活 opacity 0.91 takeover）。**尸检副产物：R0b/锚时代 dynamic 层其实 99.8% 死——车辆 class_psnr 只有 17.5 的深层原因，一直存在**。
  - **修正配方 = `loss.use_opacity=false`**（multilayer 三粒子层语境等价全豁免；`3f081ab` 进 lattice v2）。3-cam 域 R0c 实测有权衡：mean 21.30→20.84 / road_crop −0.26（正则的修剪功能有真实价值）但 lpips 0.631→**0.613**、automobile→**17.92**（dyn 复活）；后续精调方向（不阻塞）＝分层 λ / dyn 豁免 / 可见性加权。
  - **lattice v2 实测（进行中）**：R0c ✅ 20.84/0.613/17.92；**R1p（6-cam 首个健康 run）✅ 20.16 / ssim 0.640 / lpips 0.630 / automobile 17.39，死层告警 0，训练 51 min 恢复正常速度**；⚠️ 口径注记：mean/road_crop 系全相机平均（新相机 left_wide 18.1 天然难），跨相机集不可直接比；同 3 台守护线 21.59/20.34/20.05（对锚 −0.3~0.5，容量摊薄效应）；**公平对比需 per-camera 等比步数（6-cam 数据翻倍 → 30k 等于欠训一半），R1p-60k 列为达成 KPI 的第一候选**。R2p/R3p 过夜跑。A800 双卡并行重做 depth A/B（正则off 干净版，agent 托管）。
  - **viser 涂抹一案终审（半天调查，结论入档）**：渲染管线全链无罪（曝光/渲染器/ckpt/代码版本/驱动/JIT/坐标约定逐层排除，PAI 对照 + 昨日 ckpt 对照 + Follow Camera 实验闭环）。真凶＝**OPCV 模式漏了 connect 时 `client.camera.fov = 训练相机 fov_y` 的一行同步**——服务端按训练内参渲 120° 广角图，浏览器端按默认 ~75° 贴合 → 全屏拉伸涂抹；FTheta 模式接线完整故 PAI 正常。**使用姿势（修复前）**：启动带 `--initial_cam_id <cam>`（激活 rational 光线路径）+ UI 勾一次 Follow Camera（校准 fov，之后自由飞行正常）。一行修复+分辨率锁已挂任务芯片（含验证协议）。历史教训：6/24"验证"只查非全黑、6/26 只查日志 active——**画面正确性从未被验证过**，伪完成识别再添一例。

---

## 5. 文档关系速查

| 内容 | 文档 |
|---|---|
| v5 主线（本文档）：inceptio 数据线提优 | `v5_plan.md` |
| v4 主线：外推性能（E2.2 蒸馏等，PAI 线）；E5.1/E5.2 已移交本文档 A1/A2 | [`v4_plan.md`](v4_plan.md) |
| v3 主线（收敛）：per-class 重建质量（行人 Phase 2 遗留归 v3） | [`v3_plan_revised.md`](v3_plan_revised.md) |
| 架构差异图 + 关键不变量 | [`v2_architecture.md`](v2_architecture.md) |
| 4cab NRE 对照 + multi-cam 报告（⚠️ 待 B3 勘误） | [`inceptio_4cabad44_3dgrut_vs_nre.md`](inceptio_4cabad44_3dgrut_vs_nre.md) |
| A800/inceptio/vast 执行环境与铁律 | [`CLAUDE.md`](CLAUDE.md) |
| **off-track 战役设计（A→B→C 收敛 + 算力调度 + 决策门）** | [`docs/superpowers/specs/2026-07-03-offtrack-campaign-design.md`](docs/superpowers/specs/2026-07-03-offtrack-campaign-design.md) |
