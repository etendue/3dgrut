# E2.1 设计：Harmonizer 离线修复 spike

> **任务**：v4_plan.md E2.1（Harmonizer 升级集成 + 域差 spike）的**纯离线 spike 段**。
> **定位**：E1.5 重排把 E2.1 定为「低成本并行 spike」（估时 1d）。本 spike 出 **E2.2（渐进外推蒸馏）go/no-go 判据**，不做 difix.py 后端重构（那是 E2.2/E2.6 真正调用时才需要的基建，按 YAGNI 推迟）。
> **范围决策（2026-06-13 大g 拍板）**：① ckpt = baseline only；② 修复器 = 仅 Harmonizer 三代；③ 修复前基线数字 = 直接引 E1 锚 metrics.json（+ 一道 raw 交叉验证保口径）。
> **前置**：E1 测量门 5/5 ✅（lane warp / NTA-IoU / FID-KID / held-out 全就绪）；E0.7 `harmonizer_server.py`（socket IPC，smoke 通过）+ `diffusion_harmonizer.pkl`（5GB）已在 inceptio 在位。

## 1. 目标与成功判据

**目标**：量化 **DiffusionHarmonizer（修复器三代）对我方 baseline 3m/6m 渲染帧**的修复增益——用 E1 全套指标（lane grad_corr / band_lpips + NTA-IoU + FID/KID）做**修复前 / 修复后**对比，配目视存档。

**成功 = 一张「修复前 vs 修复后」E1 指标对比表 + 目视拼图，据此判 E2.2 go/no-go。**

**预期管理（R-v4.5 写死，防误判）**：纯后处理修复链——
- **FID / KID（感知）应大改善**（Harmonizer 去伪影、补连贯）；
- **几何敏感指标（lane grad_corr / NTA-IoU）基本不动、甚至略降**（扩散平滑会抹掉一点高频）；
- 这是 DiFix3D+ ablation 的**预期行为，不是失败**（论文：纯后处理提感知 134→50，几何要靠蒸馏回 3D 才提 +1.03）。

**E2.2 go 的判据**：FID/目视**显著**改善 **且** 几何指标不大幅退化 → 修复链对我方重伪影场景有效 → 投 E2.2 把修复帧蒸馏回 3D 提几何。
**E2.2 no-go 信号**：Harmonizer 在 NCore 域引入异物 / FID 不降 / 几何大幅退化（R-v4.4 域差坐实）→ 转 E2.4 域内微调，或重排。

## 2. 数据流（全部复用现成件，新代码仅编排脚本）

```
baseline ckpt（inceptio 现存）
  └─渲 lateral_3m / lateral_6m 帧（全 5 相机，375 帧/档）→ raw/ 落盘
       ├─ eval_frames_dir(raw)        → metrics_raw.json     （交叉验证：应 ≈ E1 锚）
       └─ harmonizer_server 逐帧修复  → fixed/ 落盘
            └─ eval_frames_dir(fixed) → metrics_fixed.json
                                          └─ 前后对比表（修复前列引 E1 锚）+ 目视拼图
```

**复用现成件（零新模型代码）**：
- **渲帧**：现有 render / `novel_view.py` 路径（`NOVEL_VIEW_MODES` 已含 lateral_3m/6m，E1.1 落地）。
- **修复**：E0.7 现成 `~/work/nurec_e0/e07/ipc/harmonizer_server.py`（socket IPC :端口，length-prefixed `>Q` + torch.save `{input:(h·w,3), img_size:[h,w]}`，跑在 `harmonizer-cosmos-env` 容器，nontemporal V=1）。
- **评测**：E0.4 现成 [`scripts/eval_frames_dir.py`](../../../scripts/eval_frames_dir.py)（项目侧供 GT/sseg/lane/cuboid/FTheta，外部只供像素帧；interp 全指标 + lateral 档 plane-warp / NTA / FID-KID 同口径）。

**E2.1 唯一新代码**：
1. 一个**编排脚本**（`scripts/` 下）：渲 raw → 经 **socket client**（协议复刻 E0.7 `model_ipc.py` / viser [`utils/difix_client.py`](../../../threedgrut_playground/utils/difix_client.py)：length-prefixed `>Q` + torch.save dict）调 harmonizer_server 批修复 → eval_frames_dir 评 fixed + raw → 汇出对比表。
2. 一个**目视拼图小工具**（raw | harmonized 并排，重伪影区抽帧）。

**零 3dgrut2 训练代码改动**（`trainer.py` / `correction/difix.py` / yaml 一律不碰）。

## 3. Harmonizer 模式 = nontemporal（单帧）

被 E1 eval 口径**逼定**，不是可选项：
- E1 指标算在 **val 抽样横移帧**上（官方 val 每 3 帧抽样 × 5 相机 = 375 帧），**非连续视频序**；Harmonizer 时间模式需回读自己前 K 帧输出做参考，这批帧无连续前序 → 口径上只能 nontemporal（V=1）。
- nontemporal 正是 **E2.2 训练蒸馏的形态**（蒸馏每 step 随机单 novel view，无帧序）→ 本 spike 数字直接喂 E2.2 校准。
- temporal 去闪烁是**连续渲染序列**的优势 → 留 **E2.6（viser temporal 后处理）**，E2.1 不碰。

## 4. 覆盖范围（与 E1 锚同口径，前后直接可比）

| 维度 | 取值 | 理由 |
|---|---|---|
| ckpt | **baseline 一个** | E1 三方锚之一；出 E2.2 判据足够（B3/aniso20 在 E1.1 已证三方外推打平，配方无差）。富余再加，非 spike 必须 |
| 档 | **lateral_3m + lateral_6m** | 外推主战场；1m/2m 退化轻（lane loss 优势到 2m 即耗尽，E1.1 实测），spike 不必 |
| 相机 | **全 5 相机 375 帧/档** | lane=front 75 / NTA·FID=全相机混合，与 E1 锚口径一致，前后可直接减 |

## 5. 修复前基线口径（大g 选「直接引 E1 锚」+ 一道保险）

- **主对比表「修复前」列 = 直接引 E1 锚 baseline metrics.json**（E1.1/E1.2/E1.4 已落：lane grad_corr@3m 0.384 / @6m 0.303、NTA@3m 0.096 / @6m 0.062、FID@3m 168 / @6m 193 等）。
- **一道轻量交叉验证（保口径纯净）**：raw 帧反正要渲出来喂 harmonizer，落盘后顺手对它跑一次 `eval_frames_dir` → 得 `metrics_raw.json`，确认 **raw ≈ E1 锚**（验证 `render_all` ↔ `eval_frames_dir` 两路径口径一致）。
  - 若 raw ≈ 锚（差 < 噪声）→ 主表放心引 E1 锚数字。
  - 若 raw 与锚偏差大 → 暴露路径口径暗差，改用 `metrics_raw.json` 作「修复前」（自洽优先），并记录差异来源。

## 6. 验收 / 产出

- **前后对比表** → v4_plan.md §1.3 gap 表新列（「E2.1 后」）+ §5 Done Log（commit hash + 实测数 + raw sanity 结论）。
- **目视拼图**（raw vs harmonized，3m/6m 各抽重伪影帧；对照 E0.7 `smoke_{input,harmonized}.png` 已存的 lateral_6m 银色涂抹修复基准）。
- **E2.2 go/no-go 结论 + 域差判别**（R-v4.4：Harmonizer 在 NCore 域是否引入异物 / 修复质量是否够）。
- 不改训练代码；新增仅 `scripts/` 编排脚本 + 拼图工具。

## 7. 执行环境与已知风险

- **机器**：inceptio（4090 跑 0.6B 单帧 OK，~0.5–1s/帧 → 750 帧/档 × 2 档 ~20–30 min 修复）。eval_frames_dir 跑 3dgrut2 env；harmonizer_server 跑 `harmonizer-cosmos-env` 容器。
- **环境核实（2026-06-13 ✅）**：GPU 全空（28 MiB / 24.5 GB，0% util，无 β' 竞争）；`~/work/nurec_e0/e07/ipc/` 含 `harmonizer_server.py` + `model_ipc.py` + `e2e_client.py`（client 协议可直接复用）；`diffusion_harmonizer.pkl` 5.04 GB 在位；镜像 `harmonizer-cosmos-env`(33.1G) / `nre-ga`(42.6G) 在位；e04/e13/e07 产物目录在。
- **代码 branch（执行前置，2026-06-13 核实）**：E1 五件套（`eval_frames_dir.py` / `plane_warp.py` / 扩档 `novel_view.py` / NTA / FID）已随 PR #26 合入 **Mac main（HEAD `8b24357`）**，但 **inceptio main 落后（`aa4db61`），未含**；E1 代码现在 inceptio `~/repo/3dgrut2-wt/e1` worktree。→ E2.1 执行：从 main@`8b24357` `git push inceptio` 建独立 worktree `~/repo/3dgrut2-wt/e21`（按 CLAUDE.md worktree 工作流 + 补 submodule）。
- **baseline ckpt（Task 0 已锁定）**：`/home/inceptio/work/output/v3_base_scratch30k_lam01/ckpt_30000.pt`（1.0 GB，2026-06-03 生成）。锁定证据：`mean_cc_psnr_masked = 25.7891`（目录 `v3_base_scratch30k_lam01/metrics.json`，≈ 25.79 ✓）；lane eval 重跑（`p3_lane_anchor/v3_base_scratch30k_lam01`，前视 75 帧）`mean_lane_grad_corr = 0.6931`（= 门锚 ✓）；depth-off 30k 无 lane loss 配方，非 B3（B3 的 grad_corr = 0.7386）✓。E1.1 三方锚另两条：B3 = `~/work/output/p31b3_depthoff_30k/`（grad_corr 0.7386）/ aniso20 = `~/work/output/p31aniso20_depthoff_30k/`（grad_corr 0.7325）。
- **R-v4.4 域差**：E0.7 smoke 已初步排除（Harmonizer 零微调修复我方最重伪影帧目视显著）；本 spike 用 E1 定量坐实。
- **口径陷阱（R-v4.5 / E0.2 实证）**：FID 单指标评修复会系统性误判（扩散平滑风格抵消去伪影收益）→ 必须 FID/KID + 区域化 lane/NTA 指标 + 目视三者合看，不靠单 FID 下结论。
- **GPU 共存**：harmonizer_server ~8GB + 渲染/eval 峰值，单卡 24GB 内需留余量；修复与渲染分阶段跑（非并行），避免 OOM。

## 8. 不做（出界，防 scope creep）

- 不重构 `correction/difix.py`（后端可切换基建留 E2.2/E2.6）。
- 不跑 temporal 模式（留 E2.6）。
- 不跑 B3/aniso20、不跑 Fixer 二代代际对比（spike 聚焦 baseline × Harmonizer）。
- 不训练、不蒸馏（E2.2 才做）。
