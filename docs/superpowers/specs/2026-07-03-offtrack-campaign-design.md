# off-track 质量战役设计（A→B→C 收敛 + 算力调度）

- 日期：2026-07-03
- 状态：**已批准**（大g 2026-07-03 拍板：「按思路 A、B、C 走」+「先写入文档，后续开展时再按任务细化」）
- 定位：**战役级 decision of record**。本文档定义目标、路线优先序、算力调度与决策门；**不替代任务级 plan**——每个任务开工时按 superpowers 流程另起 per-task 设计/TDD plan（遵守「plan 不贴代码块」约定）。

---

## 0. 目标重定义

- 全项目（v3/v4/v5 各线）的真实靶心统一为：**off-track（离轨/外推视角）质量**，口径 = FID/KID + lane 区域指标；两大病灶 = **road 路面** 与 **dynamic rigids**。
- on-track PSNR（官方口径 30.30、b6a9 21.04 等）只作守护线与参照，**不再是追求目标**。
- inceptio 数据线（inc_b6a9）当前**没有任何 off-track 评估**——评估先行是硬前置（§3 无悔棋三件套）。

## 1. 证据链（为什么这么收敛）

1. **根因已确诊**（2026-06-11 诊断）：训练相机共线 + 路面掠射角 → aperture problem 欠约束；训练视角内无解，必须引入新信息。三条路线 = 三种新信息源：A 生成先验 / B 真实数据（12 相机环视）/ C 官方底座（两者官方机制打包）。
2. **表示侧小修三连负**：E3.3 BEV 纹理、E3.2.6 takeover 三档、cuboid 纯几何 autogen（另有 BEV 伪 GT 调研判「不采用」）。继续表示侧 trick 大概率第四次判负——战役期冻结此类新 spike。
3. **E1.5 收口结论**：表示侧只能右移退化曲线一档；6m 档修复链（生成先验）是官方与自研的共同必需。
4. **修复→蒸馏链每环节已单独验证**：E2.1 离线修复（FID −33%/KID −64%）；E0.7 α/β'（IPC 蒸馏链路通、Harmonizer>Fixer、interpolated 代价 −0.4dB → λ 须小起步）；NuRec ablation（纯后处理只提感知，蒸馏回 3D 才提几何）。
5. **dynamic rigids 的 off-track 已有工程答案**：E2.8 资产全替流水线（coverage 1.0、harmonizer 协调 FID −25），且其残余 FID 大头 = bg/road 离轴涂抹 → **两病灶合流为 road off-track 一个主攻点**。

## 2. 三条路线与优先序（A → B → C）

### 2.1 思路 A（主线）——生成先验蒸馏（v4 E2.2）

- **机制**：现有 ckpt → 渲染离轨位姿帧 → Harmonizer 修复 → 修复帧作伪 GT 以低权重 λ 蒸馏回 3D（road/lane 区域加权）→ 档位渐进 1m→2m→3m→6m（每档模型变好，下一档渲染伪影更少、修复更可靠）。
- **执行场地**：PAI 9ae clip（E1 度量与 baseline 锚全现成：FID 原轨迹 75 → 3m 168 → 6m 193，lane grad_corr@3m 0.38 / @6m 0.30）；方法验证后再迁移 b6a9。
- **每档验收三读数（缺一不可）**：① FID/KID@档位改善；② lane grad_corr / band_psnr@档位不塌（防扩散抹平车道线，E2.1 已见 −0.09 风险）；③ interpolated 守护线不退（cc ≥ 24.7 等，v4 §0.2）。
- **dynamic rigids 侧**：E2.8 资产替换为主答案；重建侧仅留 A2（cuboid 时间戳对齐）+ D1（poseopt）两把短刀，不再深挖未观测面重建（P1.4 已判负）。
- **fallback**：3m 档 lane 被扩散抹平且 λ/区域加权调不回 → 触发 E2.4（域内降质配对 LoRA 微调），带 kill-criterion。

### 2.2 思路 B（跟进）——数据轴扩相机

- **依据**：b6a9 12 相机环视 + 朝地 multi-lidar 提供真实离轴观测（aperture problem 的数据侧缓解）；上限风险 = 官方多视角配方 6m 照样崩 → 预期主要救 ≤3m 档。
- **序**：A1（侧相机 aux + 5-cam 重训，兼修 road 层稀疏无色）→ C1（per-camera loss 权重，telew 证据 18.04→26.24）→ C2/C3/C4 阶梯。**每步除 per-cam psnr 守护线外必须加 off-track 读数**（B4/B5 产物）。
- **场地**：新 4090 到货后作为扩相机重训专属工位。

### 2.3 思路 C（仅测量，不作产品路径）——官方底座

- **B1 双臂**产出「官方 + 蒸馏」在 b6a9 的 off-track 天花板数字（门 1 输入）。
- **已核对的残余限制（2026-07-03 会话对 blocker 的核查结论）**：
  1. 官方蒸馏权重 `cosmos_3dgut.pt` 不可得 + 格式墙（torch.jit.load vs state_dict）为实——但 **IPC 方案已解**（`fixer_server.py` / `harmonizer_server.py` socket 转发，E0.7 α/β' 两次 40k 全量跑通）；待实物验证（防伪造纪律，见 B1 前置）。
  2. 官方蒸馏课程表固化（±3m novel poses + p_scheduler），**渐进课程不可控**——这是 C 相对 A 的本质劣势。
  3. USDZ 产物三用途拆解：评估（`nre render` 17.8ms/帧，**不 block**）；编辑替换（E2.8 drop-and-replace AH 资产，**不 block**）；忠实渲染 NRE dynamic rigids（**未解**——E2.7-C 复盘：真 blocker 是几何/旋转约定非 Fourier 颜色，per-frame Fourier eval 代码在未合并分支 `e27c-dynamic-color-path-a`）。
  4. viser 加载 NRE ckpt 延迟大，未归因（可诊断问题，非原理性死结）。
- **复活条件**：门 1 显示官方+蒸馏臂显著好于 A 路线可达水平 → 立「dyn rigids 旋转约定 debug」有界 spike（预算 ≤2d，kill-criterion 先行）；否则 C 正式出局。

## 3. 无悔棋三件套（先行 3-4 天；任务编号落 v5_plan Phase B）

三件互相独立、与 A/B/C 选择无关、产出三条路线的定价数字。

### B4 held-out 侧相机真 GT off-track 锚（零训练）

- 目标：b6a9 第一个**真 GT** 离轴数字。现有 3-cam 30k ckpt 从未见过侧相机 → 在侧相机位姿渲染 vs 侧相机真图 = 真外推测量（v4 E1.3 协议反用；无需重训）。
- 要动的面：eval 侧相机集注入（`dataset.camera_ids` eval-only 覆盖或 render.py 位姿加载路径）；侧相机帧只需图像+位姿，**不需要 sseg aux**。
- 验收意图：held-out 侧相机 per-cam psnr/lpips + FID（与 3 台训练相机同口径对照）写入 v5 §4 Done Log；结论回答「b6a9 现在离轴差多少」。

### B1 双臂（升级自原 B1 单臂）

- 臂 1 = nre-ga 官方配方 b6a9 baseline（4cab runbook）；臂 2 = 同配方 + `difix.training.enabled=true`，修复器走 Harmonizer IPC（单变量）。
- **前置：IPC 实物验证**（半小时）——inceptio `~/work/nurec_e0/e07/` 应有 `fixer_done.flag`、harmonizer 训练日志、两个 USDZ、`launch_harmonizer_train.sh`；实物不齐则先补 300 步 smoke 再挂全量。
- 验收意图：两臂官方口径指标 + `nre render` lateral 3m/6m 出帧 FID 对比入档；官方 val 口径陷阱（每 3 帧 + 1/4 res + cpsnr）显式标注。

### B5 E1 外推度量移植 b6a9

- 目标：novel 6 档（含 lateral 3m/6m）+ FID/KID（`--render-only` / `--novel-fid` 链路）在 b6a9 config 上打通。
- 验收意图：b6a9 metrics.json 出 novel 档 FID/KID 字段；与 B4 真 GT 数字互证（FID 代理 vs 真 GT 的一致性本身是有价值读数）。

## 4. 算力布局与调度

按**数据 locality** 分工，不搬大数据：

| 资源 | 角色 | 理由 / 约束 |
|---|---|---|
| inceptio 4090（白天） | E2.2 迭代主战场（渲染→修复→蒸馏循环） | 9ae 数据、harmonizer 容器、我方仓库全本地，链路已验证 |
| inceptio 4090（夜间） | B1 双臂 docker 挂机 + 大批量 render | b6a9 数据与 nre 容器只在 inceptio；**GPU 夜间不许空转** |
| A800 ×2 卡 | E2.2 并行蒸馏臂（λ sweep / 档长 A/B）+ PAI 线晋级 30k | 9ae 数据 A800 已有；**消费模式**：修复帧在 inceptio 生产 → 打包 rsync（750 帧 ≈1GB，分钟级）→ A800 只跑蒸馏训练，绕开 harmonizer 容器不在 A800 的问题。铁律照 CLAUDE.md：grep 验证代码同步、env `3dgrut2`、不带 set -u |
| 新 4090（T+1~2 周） | 思路 B 扩相机重训专属工位 | 到货前备好 onboarding checklist（见下），到货当天可产出 |
| vast.ai | 突发溢出臂 | 不进常规排程；~$0.5/hr，用 vast-train skill 流程 |

**新 4090 onboarding checklist**：conda env 3dgrut2（含 gxx / c++ 软链）、VGG16 权重 cache、JIT 预热编译、b6a9 + 9ae 数据 rsync、git remote + worktree 工作流接线、num_workers 按机器内存表定（CLAUDE.md 经验表）、跑一次 1k smoke 验收。

## 5. 实验纪律（战役期强制）

1. **6k/7k proxy 做一切 A/B 决策**；只有晋级 baseline 的配方跑 30k（E3.2.5 先例正式化）。
2. **E2.2 各档为增量微调**（2-4k 步/档，从现有 ckpt 续），不从头训；单档成本 ≈1-1.5h。
3. **每个 run 启动前登记 kill-criterion**：run 名 / proxy 步数 / 读数指标 / 砍单阈值 / 砍后动作。失败 run（OOM 静默退出、代码没同步、伪造数字）是历史最大时间黑洞，三类事故 checklist 均已有，严格执行。
4. **训练数字入档必须 rich log × metrics.json 交叉验证**（反伪造，v5 R3 长期风险）。

## 6. 两周作战表

**第 1 周（现有 3 卡）**

- D1：IPC 实物验证（半小时）｜B4 held-out 评估实现 + 跑（Mac 写码 + inceptio render-only）→ 拿到 b6a9 真 GT off-track 锚。
- D1-D2 夜：B1 臂 1（官方 baseline docker）→ 臂 2（+Harmonizer IPC 蒸馏）。
- D2-D3：B5 度量移植（Mac + render-only）。
- D2 起：E2.2 第 1 档（lateral 1m）inceptio 开跑；修复帧包 rsync 至 A800 起 2 条并行蒸馏臂（λ 两档）。
- 周末收口：三个决策数字（held-out gap / B1 臂差 / 官方 vs 自研 FID）+ E2.2 首档三读数全部入档 → **门 1**。

**第 2 周（新卡到货即并入）**

- E2.2 推 2m→3m→6m，每档三读数 → 3m 档后过**门 2**。
- 新 4090 onboarding → A1 侧相机 aux（并行容器 runbook 现成）→ 5-cam 30k 重训（思路 B 第一步）。
- A800 跑晋级配方 30k 全量。

## 7. 决策门

| 门 | 时点 | 输入 | 分支 |
|---|---|---|---|
| **门 1** | 第 1 周末 | held-out gap + B1 臂差 + 官方 vs 自研 FID | 蒸馏臂显著好 → C 保留有界 spike 选项（旋转约定 debug ≤2d）；差距不大 → C 出局，资源全给 A/B |
| **门 2** | E2.2@3m 读数后 | 3m 档 FID + lane 双读数 | 双改善 → 推 6m + 启动迁移 b6a9 准备；lane 塌且 λ/区域加权调不回 → 触发 E2.4 备选（带 kill-criterion） |

## 8. 冻结清单（战役期出界）

- v3 行人遗留（Phase 2）；v5 的 B2 / C3 / C4 / D2（C3/C4 待门后按数字复排，D2 纯决策任务可穿插但不占 GPU）；一切新的表示侧 spike（几何/纹理/所有权类 trick）。

## 9. 文档关系

- 本 spec = 战役级决策记录；[v4_plan.md](../../../v4_plan.md) 持有 E2.2 / E2.4 执行与回填；[v5_plan.md](../../../v5_plan.md) 持有 B1 双臂 / B4 / B5 / A1 / A2 / C1 阶梯；任务开工时另起 per-task plan。
- 决策来源：2026-07-03 大g × Claude 战略会话（三思路展开 → blocker 核查 → A→B→C 拍板 + 算力约束）。
