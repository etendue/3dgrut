# E2.8 — 生成式 dynamic rigid 系统性全替流水线（单 clip scene factory）

- **建议编号**：**E2.8**（E2.5「编辑协调 spike」的系统化升级；E2.7 号已被「viser USDZ 对标 / dyn rigids 接线 / Fourier 颜色」A/B/C 占用，故另起新号，待大g 回填看板）
- **状态**：✅ Design（本文）→ 转 writing-plans 出 TDD 执行 plan
- **一句话目标**：把 USDZ(NuRec) 重建场景**拆**成「静态底（bg+road）+ per-track dynamic rigid」，把**所有** vehicle track **成批换**成 asset-harvester（AH）资产库里的干净 3DGS asset，deformable 丢弃，harmonizer 离线协调，QA 闸（sanity + NTA-IoU/FID）把关 → 产出在 3dgrut2 viser 可编辑/可仿真的干净场景。这是一条**可复现、可规模化跑 N 个 actor** 的流水线，不是交互式手术。

---

## 1. 决策依据（本文之前的 brainstorming 已拍定）

| 决策 | 选择 | 理由 |
|---|---|---|
| 成功标准 | **可编辑/可仿真场景**（非忠实重建） | 不要求 actor identity → 用通用 AH 资产 size-match 复用即可 |
| 渲染端 | **3dgrut2 自家 viser** | 已有 USDZ loader + viser_gui_4d（E2.7）；AH ply 在 3dgut renderer 渲染保真度已被 E2.5 验证 |
| 批量维度 | **单 clip 系统性全替**（= spec 原「升级 E2.7」） | 把 E2.5 手术式换 3 车扩成全 vehicle track；链天然对多 clip 可复用 |
| deformable | **USDZ→ckpt 导出时丢弃** | `nre_usdz_loader.py:59` 白名单天然排除（注释「大g decision」，viser 不支持 neural deform field）；保留 = 保留烟雾行人，不如删 |
| 替换范围 | **rigid vehicle 全替**（car/bus/truck），非 vehicle 类不替 | AH 对 vehicle 收割好且已验证；`_CAR_CLASS` class filter 已在引擎内 |
| 资产供给 | **建 AH 资产库**（size+class match 复用） | 资产库复用才是「批量」的真杠杆；per-actor 现割是反批量 |
| QA | **sanity + NTA-IoU/FID 定量**（大g 升级要求） | E2.5 只目测；流水线化要防 silent 坏档 + 给定量证据 |

**关键前提（已验证，非赌）**：
- AH `gaussians.ply` 在 3dgut renderer 渲染干净（不烟雾）—— P1.4 注入引擎（`23e7e57`）+ E2.5（3 车 `0350e34/21cf1c4`）已坐实；convention 映射（rotation wxyz / scale_log / SH / `AxisMap` 保 det=+1）已实现。
- deformable 丢弃零新增（loader 白名单）。
- 三层对齐（cuboid size/center + 180° yaw-flip + convention）已实现于 `e25_inject.py` / `warmstart_ply.py`。

---

## 2. 范围

**In（v4 本流水线交付）**：
- 单 clip USDZ → 干净可编辑 ckpt（静态底 + 全 AH vehicle，无 deformable）+ harmonizer 协调帧 + QA 报告。
- AH 资产库（覆盖 clip vehicle 尺寸谱）+ 库 manifest 约定。
- 编排 driver（一条龙 nohup）。

**Out（明确不做，留 v5 或他轴）**：
- deformable / 行人的真步态（AH 无步态能力；丢弃即可）。
- 跨 clip 工厂 / SDG 场景变体生成（链可复用但本卡不做编排）。
- 删/插不留痕的产品级质量、inpaint 遮挡地面（v5 编辑轴）。
- 训练 / 蒸馏（本卡 frozen 离线手术，不训练）。

---

## 3. 架构：6 阶段流水线

```
┌──────────┐  ①拆        ┌─────────────────────────┐  ②配        ┌──────────────┐
│ USDZ(NRE)│ ──────────▶ │ ckpt: bg+road(静) +     │ ──────────▶ │ AH 资产库     │
│ last.usdz│  loader     │ dyn_rigids(per-track)    │  size+class │ metadata.yaml│
│+seq_track│  白名单丢   │ + tracks_dict(timeline)  │  match      │ + *.ply      │
└──────────┘  deformable └─────────────────────────┘             └──────┬───────┘
                                                                         │
┌──────────┐  ⑥消费     ┌──────────────┐  ④协调      ┌──────────────┐  ③批注入    │
│ viser_gui│ ◀───────── │ 协调后 ckpt + │ ◀────────── │ 全替后 ckpt   │ ◀──────────┘
│ _4d 编辑 │            │ harmonized帧  │  离线batch  │ (全 vehicle  │  逐 track
└──────────┘            └──────┬───────┘  +temporal   │  track 换)   │  三层对齐
                               │                      └──────────────┘  frozen手术
                          ⑤ QA 闸（sanity + NTA-IoU/FID，pass/fail + 报告）
```

| 阶段 | 输入 → 输出 | 复用基建 | 新增 |
|---|---|---|---|
| **① 拆** | USDZ → ckpt{bg,road,dyn_rigids} + tracks_dict | `nre_usdz_loader.py` (`NRE_PARTICLE_LAYERS`) | — |
| **② 配** | ckpt 全 vehicle track ↔ 库资产（class+size 最近） | `match_assets_by_size`/`_CAR_CLASS`/`load_bundle_metadata` | bijection→nearest-from-bank + fallback |
| **③ 批注入** | 逐 track 三层对齐 + frozen 替换 → 全替 ckpt | `e25_inject_ah_replace.py`/`replace_tracks_in_dyn_node` | 全 track 枚举循环 |
| **④ 协调** | 渲全替帧 → harmonizer 离线固化 | `e21_harmonizer_batch_fix.py`/`harmonizer_server.py`(E2.6 temporal) | 接在批注入后的 batch 节点 |
| **⑤ QA** | sanity + NTA-IoU + FID → pass/fail + report | `vehicle_detector.py`(E1.2 NTA)/E1.4 FID | QA 编排脚本 + 阈值闸 |
| **⑥ 消费** | 协调后 ckpt → viser 交互编辑 | `viser_gui_4d.py` | — |

---

## 4. 各阶段详细设计

### ① 拆（现成，零改动）
- `load_nre_usdz(...)` 翻译白名单 `("background","road","dynamic_rigids")` → deformable 天然丢弃；`clip_floater_gaussians` 裁 background sky 尾巴。
- 产 `tracks_dict`（per-tid pose over timeline，来自 `sequence_tracks.json` + `gaussian_cuboid_ids` remap）+ `dyn["track_ids"]` buffer（gaussian→tid slot）。
- **流水线只需调用、不改 loader**。

### ② 配（AH 资产库 — 核心新增）

**库结构**（沿用 `warmstart_metadata.py` 已支持的两种布局）：
```
asset_bank/
  metadata.yaml          # {asset_hash: {label_class, cuboids_dims:[L,W,H], ply_file}}
  <class>/<hash>/gaussians.ply   (nested)   或   <class>__<hash>.ply (flat)
```
- **库内容目标**：覆盖一个 clip 的 vehicle 尺寸谱——至少 sedan / SUV / van / bus / truck 各 1–2 尺寸代表（现有 3 车为起点，AH 补割）。
- **建库来源**：`asset-harvester` skill 从 NCore clip per-object 收割（每 actor 有 cuboid+mask）；产 `gaussians.ply` + `metadata.yaml`。
- **匹配函数升级**：`match_assets_by_size` 现为 N↔N bijection；改/包一层 `query_bank(class, dims) -> asset_hash`（**允许一个库资产被多 track 复用**，按 class 过滤后取 L2(dims) 最近）。
- **fallback ladder**（库不覆盖时）：
  1. 同 class 最近尺寸（默认）；
  2. 跨 class 全局最近 + WARN（如 truck 缺退而求 van）；
  3. `--on_miss=skip` → 保留该 track recon + 记入 report（不 silent）。

### ③ 批注入（枚举 + 复用对齐引擎）
- 从 `dyn["track_ids"].unique()` 枚举全 track → 查 class（`sequence_tracks` autolabel）→ vehicle 集。
- 逐 track：`query_bank` 选资产 → `align`（填 recon **live cuboid** size/center + `flip_forward_180` + `AxisMap` convention）→ `replace_tracks_in_dyn_node` frozen 手术（删 recon track 粒子 + 插对齐后 AH 粒子，其它层/非 vehicle track 不动）。
- 纯函数 `replace_all_vehicle_tracks(ckpt, bank, policy) -> (ckpt, ReplaceReport)`；`ReplaceReport` 记每 track {tid, class, chosen_asset, fallback_level, n_before, n_after, proj_iou}。

### ④ 离线协调固化
- 渲全替 ckpt 帧（`--render-only` 关监督，~4.62s/帧，E2.1 已知）→ harmonizer IPC batch（E2.1 `e21_harmonizer_batch_fix.py`）+ **temporal 模式**（E2.6，帧间一致去 flicker）。
- 产 `harmonized_frames/`（协调后帧序列，存档 + 喂 ⑤ FID）。
- nohup 节点，不依赖 viser 交互。

### ⑤ QA 闸（sanity + 定量，大g 升级要求）
**Sanity（廉价，自动 pass/fail，先跑）**：
- 替换覆盖率 = 已替 vehicle track / 总 vehicle track（应 100%，skip 的列报告）；
- 投影框-cuboid IoU 抽样（对齐没歪，阈值如 ≥0.5）；
- 注入粒子 opacity 中位数在正常区（防「烟雾」回归，远离 0.11）；
- 粒子数 / scene_extent sanity。

**定量（贵，inceptio，大g 要求）**：
- **NTA-IoU**（`vehicle_detector.py` yolov8m，E1.2 口径）：全替后渲染帧检出车 + 与投影 cuboid IoU；对照 recon 原 actor（替换不掉检出率）。
- **FID**（E1.4 口径）：协调前 vs 协调后、及 vs 训练视角分布（协调有效、伪影不升）。
- 产 `qa_report.json`（sanity 全字段 + NTA-IoU/FID 数字 + pass/fail）。

### ⑥ viser 消费（现成）
- 协调后 ckpt → `viser_gui_4d.py`（USDZ/`--gs_object` 路径）交互编辑/展示。

### 编排 driver
- `scripts/e28_systematic_replace_pipeline.{sh,py}`：读配置 `(usdz, asset_bank, dataset_path, harmonizer_server, qa_thresholds, fallback_policy)` → 串 ①→⑤ → 产物 `out/<clip>/{ckpt_replaced.pt, harmonized_frames/, qa_report.json, replace_report.json}`。
- 单 clip 一条龙 nohup；多 clip = 外层循环（本卡不做，但接口预留）。

---

## 5. 验收标准

| 类 | 准则 | 工具 |
|---|---|---|
| **功能** | 单 clip 全 vehicle track 被替换（覆盖率 100% 或 skip 全部有据）；deformable 不在产物 | ReplaceReport / ckpt node 检查 |
| **几何对齐** | 投影框-cuboid IoU 抽样 ≥ 阈值；无烟雾（opacity 正常） | QA sanity |
| **检出（NTA-IoU）** | 全替后 NTA-IoU 不低于 recon 基线（替换不掉检出） | E1.2 |
| **感知（FID）** | harmonizer 协调后 FID 较协调前下降；不引入异物（目视） | E1.4 + 目视 |
| **守护线** | bg/road/非 vehicle 部分与拆出 ckpt 逐字节不变（局部手术） | 单测 |
| **可复现** | driver nohup 一条龙跑通，产物齐全 | 端到端 |

---

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| 库不覆盖某尺寸/类 → 配错 | fallback ladder + ReplaceReport 显式记录，`--on_miss=skip` 不 silent |
| 一资产复用多 track 显重复感 | 编辑场景可接受；库多样性 + 可选 per-track 随机扰动（YAGNI，先不做） |
| harmonizer 协调「有效但有限」（E2.5 已知天花板） | 本卡不追求完全自然；FID 下降 + 目视无异物即过；进一步留 v5 |
| NTA-IoU/FID 跑在 inceptio 成本 | render-only + 抽样帧；sanity 先 gate，过了才跑贵的定量 |
| 全 track 枚举漏 / class 误判 | class filter 单测 + 覆盖率闸；非 vehicle 默认不替（安全侧） |
| AH 资产坐标/朝向新案例（bus/truck 与 car canonical 不同） | `AxisMap` 按 class 校准；建库时每类先单资产 viser 目测过一遍再入库 |

---

## 7. 工作分解（喂 writing-plans，TDD）

> 纯函数 + Mac 可测优先；inceptio 仅渲染/检测/harmonizer 重活。

- **T0 资产库 manifest + query**：`AssetBank` 抽象（`load_bundle_metadata` 包装）+ `query_bank(class, dims, policy)` 最近匹配 + fallback ladder。**单测**（Mac，合成 metadata）。
- **T1 全 track 枚举 + 批注入纯函数**：`replace_all_vehicle_tracks(ckpt, bank, policy) -> (ckpt, ReplaceReport)`；复用 `align`/`replace_tracks_in_dyn_node`。**单测**：全 vehicle 替换、非 vehicle 不动、bg/road 字节不变、ReplaceReport 字段、skip fallback。
- **T2 QA sanity 脚本**：覆盖率/proj-IoU/opacity/粒子数 → pass/fail。**单测**（合成 ckpt）。
- **T3 协调节点接线**：批注入 ckpt → render-only → harmonizer batch（temporal）→ `harmonized_frames/`。inceptio smoke。
- **T4 QA 定量**：NTA-IoU（E1.2）+ FID（E1.4）跑协调帧 → `qa_report.json`。inceptio。
- **T5 编排 driver**：配置 → ①–⑤ 一条龙 → 产物布局。端到端 inceptio（**资产库已建为前提**）。
- **T6 建库执行**（并行可先行）：AH 收割补 sedan/SUV/van/bus/truck，每类 viser 目测入库 + manifest。
- **T7 文档回填**：v4_plan.md 看板 + Done Log；v2_architecture.md 节点（若新模块）。

**gate**：E2.1 ✅ + E2.5 ✅ + E2.6 ✅ + E1.2/E1.4 ✅（QA 定量依赖）。
**成本估**：代码侧 ~2d（多为现有零件编排 + 单测）；建库 AH 收割 ~0.5–1d；inceptio 端到端 + 定量 ~0.5d。

---

## 8. 与看板/其它任务关系
- **E2.5 升级**：E2.5 验单 actor 可行 → 本卡系统化全替 + 流水线化（spec 原「升级 E2.7」诉求，改号 E2.8）。
- **复用** E2.1（harmonizer batch）/ E2.6（temporal IPC）/ E2.7（USDZ loader + viser）/ E1.2（NTA-IoU）/ E1.4（FID）/ P1.4+E2.5（注入对齐引擎）。
- **正交**于 road 轴（E3.x）、外推蒸馏（E2.2）—— 本卡治「编辑场景的 dynamic 干净度」，非外推质量。
- **喂 v5**：本卡是 v5 编辑/SDG 轴的第一条端到端产线（链对多 clip / 资产组合天然可复用）。
