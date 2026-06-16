# E2.5（+E2.7）— 生成式 asset 取代 recon dynamic rigid（actor 替换 spike）

- **编号**：E2.5（已立「编辑协调 spike」，本文把它的「取代」分支系统化）；若单 actor 验证成功 → 升级 **E2.7（建议新编号）系统性替换 dynamic layer**
- **状态**：⬜ Spike 提纲（待排期）
- **一句话目标**：用 asset-harvester 生成的**完整 3DGS asset** 取代 recon 的 dynamic rigid actor（车/人），Harmonizer 协调融合 → 攻 actor 弱观测面退化。

---

## 1. 背景与证据

- **弱观测面问题**：dynamic rigid（车/人）在 clip 里只被有限帧/角度看到，背面/底部从没观测过 → recon gaussian 在这些面空/瞎编 → 换视角穿帮（P1.4 验尸根因）。
- **官方做法（E0.6 能力清单）**：NuRec **不把 asset 做完美**，而是插入后用生成协调器擦屁股；asset-harvester 与 Harmonizer 同为官方「优化与资产」两大件，**Harmonizer 训练管道③（asset re-insertion）④（PBR 阴影）专为此设**。
- **资产就绪**：asset-harvester skill 已收割 **3 车 + 3 人**（`gaussians.ply` + `metadata.yaml`）。
- **两条路对照**：
  - **E2.3 蒸馏式**（保留 recon，2D 监督修弱面，复用 E2.2 蒸馏 buffer）—— 温和；
  - **本卡 E2.5 生成式替换**（直接换掉 recon dynamic gaussian）—— 激进，正是官方编辑形态。

### 核心洞察

弱观测面是**观测空洞**，2D 蒸馏只能「猜」空洞、生成式 asset 直接用**生成先验填满所有未观测面**。代价是把问题从「重建」转成「**对齐 + 协调**」——这正是 Harmonizer 的用武之地。

---

## 2. 方法（三层对齐）

复用 AH 注入引擎（PR #18 plumbing / frozen 离线手术）+ [`correction/difix.py`](threedgrut/correction/difix.py)（E2.1 升级后 Harmonizer）+ [`vehicle_detector.py`](threedgrut/model/vehicle_detector.py)（NTA-IoU）。

| 层 | 方法 | 验证 |
|---|---|---|
| **几何 / pose / scale** | AH asset canonical frame → 场景 cuboid 的 `size/center/yaw` 归一化缩放 + 摆位 | 投影框 vs cuboid 框 IoU |
| **外观 / 光照** | 替换 actor 的帧 render → Harmonizer 协调（temporal 模式，复用 E2.6 IPC）；可选 PBR 阴影（管道④）| FID（协调前后）/ 目视 |
| **时序** | cuboid 轨迹驱动 rigid 变换，asset 静态几何跟轨迹；appearance 跨帧由 Harmonizer temporal 稳 | 跨帧 flicker 目视 |

**路线**：先 **E2.5 单 actor spike**（取代 1 辆，离线 frozen 手术，**不训练或轻训练**）→ 验三层对齐 + 三验收 → 成功则升级 **E2.7 系统性替换**（全部 dynamic rigid 换 AH asset）。

---

## 3. 验收指标（三验收）

- **NTA-IoU**：替换 actor 被检出 + 框齐（对比 recon 原 actor，[`vehicle_detector.py`](threedgrut/model/vehicle_detector.py) yolov8m）。
- **FID**：协调前后。
- **弱观测面目视**：zoom-out / 环绕渲染，背面/底部不再穿帮（这是替换相对 recon 的核心增量）。
- **守护线**：场景其它部分不变（替换是局部手术）。

---

## 4. 成本 / GPU / gate

- **成本**：E2.5 单 actor **~1.5d**（不训练或轻训练）；E2.7 全量 **+2d**。
- **GPU**：inceptio（渲染 + Harmonizer IPC）。
- **gate**：E2.1 ✅（Harmonizer 集成）+ E0.6 能力清单 + asset-harvester 资产 ✅。
- **风险**：① canonical pose/scale 对齐误差（AH 朝向定义 vs cuboid yaw）② 光照 mismatch，Harmonizer 协调不够（需 PBR）③ 动态 actor 跨帧 appearance flicker ④ 全替换丢 recon 真实细节 vs 混合更复杂 ⑤ 类别覆盖（AH 对车好，人/异形物?）。

---

## 5. TDD 任务拆解骨架

- **Task 0**：AH asset → cuboid pose 对齐函数（scale / yaw / center 归一化）单测。
- **Task 1**：离线注入手术（frozen ckpt 删 recon dynamic + 插 AH asset）单测（粒子层正确、其它层不动）。
- **Task 2**：替换帧渲染 + Harmonizer 协调（复用 E2.1/E2.6 IPC）。
- **Task 3**：三验收（NTA-IoU / FID / 弱面目视）inceptio。
- **Task 4**：决策**全替换 vs 混合** + 是否升级 E2.7 + 文档回填。

---

## 6. 与其它方向的关系

- 与 **① road** / **③ 全局 floater** 正交（本卡治 actor 弱观测面）。
- 与 **E2.3 蒸馏式**互补：E2.3 是「修 recon」，本卡是「换 recon」；spike 结果决定 actor 轴主路线。
- 复用 E2.1（Harmonizer 集成）/ E2.6（temporal IPC）已落地基建——**端到端对齐链最长但想象力最大**，建议单 actor 先验可行性。
