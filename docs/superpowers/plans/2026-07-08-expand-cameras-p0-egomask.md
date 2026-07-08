# 扩相机作战（v5 Phase C）实现计划 —— P0 ego-mask 修复 + C1 telew + 阶梯 runbook

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修通 b6a9 ego-mask（接线 + 补齐）、立 R4e 新锚与 off-track 评估基建，然后按单变量阶梯把参训相机 6 → 9–11 台（rear_right 永久 eval-only）。

**Architecture:** 主修复 = 代码接线（`EgomaskAuxReader` 直读 nre-tools aux itar，`datasetNcore` 在 SDK sequence mask 缺失/全零时 fallback）；front/back 全黑相机从 sseg egocar(19) 派生补齐；评估三件套（R4e 锚 / B5 novel FID / held-out 一键驱动）先行，阶梯每步四读数验收。

**Tech Stack:** Python + zarr/IndexedTarStore（ncore SDK）+ scipy.ndimage；pytest（Mac CPU venv + conftest stubs）；inceptio 4090 训练（conda env `3dgrut2`）。

**Spec:** [`docs/superpowers/specs/2026-07-08-expand-cameras-campaign-design.md`](../specs/2026-07-08-expand-cameras-campaign-design.md)

## Global Constraints

- **plan 格式（大g 约定）**：本 plan 不贴代码块；每步给签名 / 断言要点 / 命令意图。
- **inceptio 铁律**：depth-off + `num_workers=10`；`source ~/miniforge3/etc/profile.d/conda.sh && conda activate 3dgrut2`；长任务 setsid 驱动 + 发射后验证进程存活；worktree 工作流（分支 push inceptio remote + worktree + 补 submodule）。
- **字节等价不变量**：无 egomask itar 的 clip（PAI 9ae 线）行为逐字节不变；`loss.camera_loss_weights` 默认空 dict 恒等。
- **数字入档**：rich log × metrics.json 双源交叉；6k proxy 做 A/B 决策、晋级配方才 30k；每 run 登记 kill-criterion。
- **文档同步**：任务完成 = 代码 commit + v5_plan/v2_architecture 同步（mermaid 全角括号铁律）。
- Mac 测试命令统一：`.venv/bin/python -m pytest threedgrut/tests/<file> -v`。

---

### Task 1: EgomaskAuxReader + resolve_ego_valid_mask 纯函数

**Files:**
- Modify: `threedgrut/datasets/aux_readers.py`（`SsegAuxReader` 之后追加）
- Test: `threedgrut/tests/test_egomask_aux_reader.py`（新建）

**Interfaces:**
- Produces: `EgomaskAuxReader(itar_path)`，方法 `camera_ids() -> list[str]`、`has_camera(camera_id: str) -> bool`、`read_static_mask(camera_id: str) -> np.ndarray`（`(H, W)` bool，**该相机全部帧的并集**——任一帧标 ego 即 True；帧为 `(H,W)` uint8 {0,255} 数组，非 0-D PNG bytes，`decode` 需兼容两种）。
- Produces: `resolve_ego_valid_mask(sdk_mask_image, clip_dir, camera_id, resolution_hw, dilation_iters) -> np.ndarray`（`(H, W)` bool **valid** 图）：① SDK mask 存在且非全零 → 沿现逻辑 convert("L")→dilate→取反；② 否则若 `discover_aux_path(clip_dir, "egomask")` 命中且 reader `has_camera` → itar mask→dilate→取反；③ 都无 → 全 True。`sdk_mask_image` 为 PIL Image 或 None；`clip_dir` 为 None 时跳过 ②（兼容非 NCore 调用方）。
- Consumes: 现有 `_open_itar_zarr` / `discover_aux_path`（同文件）。

- [ ] **Step 1: 写失败测试。** 用 fake zarr root（嵌套 dict 风格 stub，monkeypatch `_open_itar_zarr`）构造两相机数据：camA 两帧不同区域非零、camB 全零。断言要点：① `read_static_mask("camA")` 为两帧**并集**（两块区域都 True，精确相等）；② camB 全 False；③ `has_camera("camX")` False、`read_static_mask("camX")` 抛 KeyError；④ `resolve_ego_valid_mask` 三分支——SDK 非零 mask 时**不触碰** itar（传入会炸的 sentinel clip_dir 证明未调用）、SDK None/全零 + itar 有 → valid = not(dilate(itar_mask))、两者皆无 → 全 True 且 shape=resolution_hw；⑤ dilation_iters=0 时 mask 不膨胀（精确相等，二值无公差）。
- [ ] **Step 2: 跑测试确认失败**（ImportError / AttributeError 级失败，非 collection error）。
- [ ] **Step 3: 实现 `EgomaskAuxReader` + `resolve_ego_valid_mask`**，风格对齐 `SsegAuxReader`（lazy open + per-camera 子组缓存）；scipy `ndimage.binary_dilation` 与 datasetNcore 现用法一致。
- [ ] **Step 4: 跑新测试全绿 + 既有 aux 测试回归**（`test_aux_discover_lane.py` 等同目录全套）。
- [ ] **Step 5: Commit** `feat(P0.2): EgomaskAuxReader + resolve_ego_valid_mask 纯函数（aux itar 直读 ego mask）`。

### Task 2: datasetNcore ego-mask fallback 接线

**Files:**
- Modify: `threedgrut/datasets/datasetNcore.py:429-440`（「Statically unmasked pixels (ego mask)」块）
- Test: `threedgrut/tests/test_egomask_aux_reader.py`（追加接线语义测试，仍测纯函数层）

**Interfaces:**
- Consumes: Task 1 的 `resolve_ego_valid_mask`。
- Produces: `datasetNcore` 该块改为单行委托——SDK `camera_sensor.get_mask_images().get("ego")` 与 clip 目录、`camera_model.resolution`、`self.n_camera_mask_dilation_iterations` 传入纯函数；返回值继续赋 `camera_valid_pixels_ego_mask`（下游 L466 `repair_nonfinite_rays` 及缓存路径**不动**）。

- [ ] **Step 1: 写失败测试。** 断言要点：① fallback 激活时（SDK 全零 + itar 有 4 台相机数据）各相机 valid 像素数 = 总数 − dilate 后 mask 数；② PAI 语义回归——SDK 无 'ego' 键 + 无 itar → valid 全 True（与现状逐字节一致）。
- [ ] **Step 2: 跑测试确认失败。**
- [ ] **Step 3: 改 datasetNcore 该块**为委托调用 + 一行 sanity 日志（风格对齐 A5 的 `[A5] dyn_mask_cuboid filled via ...`）：fallback 激活时打 `[P0.2] ego mask via aux itar fallback: <camera_id> coverage=<pct>%`。clip 目录取值沿 `discover_aux_path` 在本文件既有用法（L1297 一带的 clip_dir 来源）。
- [ ] **Step 4: Mac 全套回归**：`.venv/bin/python -m pytest threedgrut/tests/ -x -q` 零失败。
- [ ] **Step 5: Commit** `feat(P0.2): datasetNcore ego mask SDK缺失/全零时 fallback 读 aux itar（PAI 线字节等价）`。
- [ ] **Step 6: inceptio 实证**（分支 push → worktree）：跑一个 500 步 smoke（R3p 配方 + `n_iterations=500`），grep 日志确认 6 相机中 4 台出现 `[P0.2]` fallback 行、coverage 与诊断数字量级一致（cross ~8%、right_wide ~1%）。

### Task 3: front/back ego mask 派生脚本 + 实跑补齐

**Files:**
- Create: `scripts/derive_egomask_from_sseg.py`
- Consumes: `scripts/diag_egomask_itar.py`（诊断先例）、`scripts/merge_lidar_aux.py`（itar 写模式）、Task 1 reader。

**Interfaces:**
- Produces: CLI `python scripts/derive_egomask_from_sseg.py --clip-dir <D> --cameras <a,b> --occurrence-thresh 0.5 --min-blob-px 200 --dilate-px 5 --preview-dir <P>`。逻辑意图：sseg itar 全帧 `egocar(19)` → 每像素**出现率图**（静态自车出现率≈1，邻车误检为暂态低出现率）→ 阈值 → 去小连通域（`scipy.ndimage.label`）→ dilate → 生成新完整 egomask itar：目标相机写派生 mask、其余相机原样拷贝。
- ⚠️ **itar 唯一性**：`discover_aux_path` 同目录多个 `*.aux.egomask.zarr.itar` 会 ValueError → 新 itar 先写临时名，旧 itar `mv` 到 `aux_backup/`，再改回正名（write-once，不可 in-place）。
- Produces: `--preview-dir` 每相机输出「真图 × 派生 mask 叠图」png 供目检。

- [ ] **Step 1: 写脚本**（纯 numpy/scipy 逻辑 + itar 读写；Mac 可 dry-run 语法检查，实跑在 inceptio）。
- [ ] **Step 2: inceptio 实跑** front_wide + back_rear_wide；scp 回 preview 叠图**目检**：自车结构（车头边缘/后视镜/天线）被覆盖、无大块误检（护栏/邻车）。不干净则调 `--occurrence-thresh`/`--min-blob-px` 重跑；仍不干净 → 兜底改手工 ROI 多边形（每相机一张，写回同一 itar 流程）。
- [ ] **Step 3: 回归验证**：`diag_egomask_itar.py` 重跑 → 6/6 相机 nonzero>0；Task 2 的 500 步 smoke 重跑 → 6 台全部出 `[P0.2]` 行。
- [ ] **Step 4: Commit 脚本** `feat(P0.3): sseg egocar 出现率派生静态 ego mask 脚本 + b6a9 front/back 补齐实跑`（itar 是数据不进 git；preview 关键截图入 commit message 描述或 v5_plan Done Log 引用路径）。

### Task 4: R4e 重锚（30k，ego-mask 单变量）

**Files:**
- Create: `scripts/drivers/r4e_rebaseline.sh`（沿 `fc93bd3` 正式驱动模式）
- Modify（跑完后）: `v5_plan.md` §4 Done Log + §0.2 KPI 表、`configs/apps/ncore_3dgut_mcmc_multilayer_inceptio.yaml` 头注释锚更新

**Interfaces:**
- Consumes: Task 2/3 完成后的代码 + 补齐 aux；R3p 配方（`ncore_3dgut_mcmc_multilayer_inceptio.yaml`）零改动——单变量 = 仅 ego-mask 生效。
- Produces: **R4e 锚**（mean/cc/ssim/lpips/road_crop/automobile + per-cam 全套 + iter-6000 val 读数存档作后续 proxy 参照）。

- [ ] **Step 1: 登记 kill-criterion**（run 名 r4e_30k / 观察点 iter 2k：无 NaN、无死层告警、loss 曲线正常 / 砍单动作：停 run 回 Task 2/3 查）。
- [ ] **Step 2: setsid 驱动启动 30k** + 发射后验证（`pgrep -f '[p]ython.*train.py'` + log 前 50 行含 6 条 `[P0.2]`/派生 mask 行 + valid 像素占比 sanity）。
- [ ] **Step 3: 完成后双源交叉**（`🎊/⭐` 两表 + metrics.json 一致）；与 R3p 并排入档，**显式标注口径差异**（masked 指标含义变化，不作同口径比较）；ego-mask on/off 定性对比（automobile/road 侧预期受益）写入 Done Log。
- [ ] **Step 4: Commit 文档同步** `docs(plan): R4e ego-mask 锚入档（口径注记 + iter6k proxy 参照）`。

### Task 5: B5 novel FID 链路移植 b6a9

**Files:**
- Create: `scripts/drivers/b5_novel_fid_b6a9.sh`（render-only 驱动）
- Modify（跑完后）: `v5_plan.md` B5 卡 ✅ + Done Log

**Interfaces:**
- Consumes: `render.py` 既有 `--novel-fid` / `--render-only` / novel 6 档链路（v4 E1.1/E1.4 工具，本仓库已合 main）；R4e ckpt（Task 4）。
- Produces: b6a9 metrics.json 出 `mean_novel_fid_*` / `mean_novel_kid_*` 全档字段；与 B4 真 GT 数字互证结论一行。

- [ ] **Step 1: 对 R4e ckpt 跑 novel FID eval**（参数用法对照 v4 E1.4 Done Log 用例；b6a9 config 差异——相机数/分辨率——按报错最小适配，若需代码改动先写回归测试再改）。
- [ ] **Step 2: 验收字段齐全**（lateral 1/3/6m 各档 FID/KID 单调性 sanity——离轴越远越差为健康信号）+ 双源交叉。
- [ ] **Step 3: 互证入档**：novel FID 各档 vs B4 held-out gap 方向一致性结论写 Done Log；Commit `feat(B5)+docs(plan)`。

### Task 6: held-out 评估一键驱动

**Files:**
- Create: `scripts/drivers/eval_heldout_b6a9.sh`
- Consumes: B4 流程底稿（`.superpowers/sdd/b4_summary.md`）：`render.py --dataset-cameras` 替换 camera_ids + `--novel-fid`，exposure 自动禁用 → cc 口径。

**Interfaces:**
- Produces: 输入 `<ckpt> <out_tag>` → 两组 render（train-cam 组 / rear_right held-out 组）→ 汇总一行输出：train cc_psnr / held-out cc_psnr / gap / FID，供阶梯每步直接调用。

- [ ] **Step 1: 封装脚本**（B4 四组流程裁成两组；输出目录 `~/work/output/heldout_<tag>/`）。
- [ ] **Step 2: 用 R4e ckpt 实跑一次**作基线读数（rear_right 此时未参训 = 真 held-out）；双源交叉后入档 Done Log（这就是阶梯的「读数 1」起点）。
- [ ] **Step 3: Commit** `feat(P0.6): held-out 一键评估驱动 + R4e 基线读数`。

### Task 7: C1 telew per-camera loss weight（必须 merge main）

**Files:**
- Modify: `threedgrut/trainer.py`（`get_losses` L1129–1198 光度项区 + 新私有方法）
- Modify: `configs/base_gs.yaml`（`loss` 节新增 key）
- Test: `threedgrut/tests/test_camera_loss_weight.py`（新建）

**Interfaces:**
- Produces: `Trainer._camera_loss_weight(camera_id) -> float`——读 `self.conf.loss.get("camera_loss_weights", {})`，命中返回权重、未命中/None 返回 1.0。
- Produces: `get_losses` 中 `loss_l1`、`loss_ssim` 在进入 L1344 加权汇总**前**各乘 `w = self._camera_loss_weight(getattr(gpu_batch, "camera_id", None))`；正则项（opacity/scale/sky/road 等）一律不乘。
- Produces: config key `loss.camera_loss_weights: {}`（默认空 = 字节等价）；CLI 用法 `++loss.camera_loss_weights.camera_front_tele_30fov=4.0`。

- [ ] **Step 1: 写失败测试。** 断言要点（mock conf + 最小 trainer 构造，沿 conftest stub 模式）：① 默认空 dict → w=1.0、loss 值与未改代码路径**精确相等**；② `{camX: 2.0}` + batch.camera_id=camX → 返回的 `l1_loss`/`ssim_loss` 恰为 baseline 2 倍（浮点 rtol 1e-6）、其余 loss 项逐项不变；③ camera_id 不在 dict / batch 无 camera_id → 1.0；④ weight=0.0 合法（光度完全屏蔽该相机）。
- [ ] **Step 2: 跑测试确认失败。**
- [ ] **Step 3: 实现**（方法 + 两处乘权 + yaml key）。
- [ ] **Step 4: 全套回归**零失败（重点：既有 loss 相关测试不动）。
- [ ] **Step 5: Commit 并 merge 进 main** `feat(C1): per-camera photometric loss weight（telew 重实现，默认字节等价）`——**完成定义含 main 合入**（2026-06-25 丢码教训）；v5_plan C1 卡 ✅ 同 commit。

### Task 8: C2 前置——新相机 aux 生成（rear_left + front_standard）

**Interfaces:**
- Consumes: CLAUDE.md nre-tools runbook + A1 遮挡补丁（`NRE_LIDARSEG_OCCLUSION=off`，`/tmp/estimators_patched.py` bind-mount）；Task 3 派生脚本。
- Produces: 8 相机全套 aux（sseg + egomask + lidar-camvis 覆盖新相机；lidar-sseg 重跑或合并）落主 clip 目录，旧 aux 备份。

- [ ] **Step 1: 容器 run A**（sseg + egomask，8 相机 id 列表，`--parallel-mode --workers-per-gpu=3`）+ run B（lidar-seg camvis，遮挡补丁下 0.7s/帧）；itar write-once 纪律——完整跑完再替换，绝不中途 stop。
- [ ] **Step 2: 验收**：`diag_egomask_itar.py` 扫 8 相机——egomask 全黑者用 Task 3 脚本派生补齐 + 目检；lidar-sseg road 占比量级复核（A1 修复后 ~40% 参照）。
- [ ] **Step 3: 结果与命令记录入 v5_plan Done Log**（无代码 commit，文档 commit）。

### Task 9: C2 阶梯 run（6 → 8 cam）

**Interfaces:**
- Consumes: R4e 锚 + Task 6/5 评估驱动 + Task 7 telew + Task 8 aux。
- Produces: 8-cam 配方判定 + 四读数入档；晋级则 `_inceptio.yaml` camera_ids 更新为 8-cam、锚改记。

- [ ] **Step 1: 6k proxy**（`dataset.camera_ids` 8 台 CLI 覆盖，rear_right 不进；kill-criterion 登记；首个 8-cam run 盯 `free -g`，内存吃紧 nw 10→8）。判据：已参训 6 台 iter-6k val 对 R4e iter-6k 参照不退 >0.3 dB、无 NaN/死层告警。新相机 psnr 明显偏弱（>2 dB 落差）→ telew 调权（0.5 档步进）重跑 proxy，**调权 run 单独命名**记录。
- [ ] **Step 2: 晋级 30k** + 发射后存活验证。
- [ ] **Step 3: 四读数**：① `eval_heldout_b6a9.sh`（rear_right held-out cc_psnr——预期对 R4e 基线改善，B4 +8.88 dB 效应方向）② novel FID 各档 ③ per-cam 守护线 ④ automobile class_psnr。双源交叉入档。
- [ ] **Step 4: 判定与文档**：守护线不破 + 读数 ①② 至少一项改善 → 晋级（yaml + Done Log + 看板）；破线 → 按单变量回退定位（先 telew 权重、再相机逐台二分），结论入档。

### Task 10: C3 阶梯 run（+front_tele → 9 cam）

- Consumes: Task 9 晋级配方；4cab 证据（tele 无权重 18.04 → telew 26.24）。
- [ ] **Step 1: 6k proxy**——front_tele 初始权重按 4cab 经验直接给非 1 值（起点 2.0，视 proxy per-cam psnr 调）；判据同 Task 9。
- [ ] **Step 2: 晋级 30k + 四读数 + 判定入档**（流程同 Task 9 Step 2–4）。

### Task 11: C4 阶梯 run（+2 台 FTheta 鱼眼 → 11 cam，可弃）

- Consumes: Task 10 晋级配方；上游 [issue #238](https://github.com/nv-tlabs/3dgrut/issues/238) 鱼眼尖刺风险；PAI 线 FTheta 路径已证。
- [ ] **Step 1: 登记 kill-criterion（本任务整体可弃）**：proxy 出现不可控尖刺伪影（目检）或守护线破 >0.5 dB 且 telew 调不回 → **弃**，9-cam 收口，弃因入档。
- [ ] **Step 2: 鱼眼 aux 补齐**（Task 8 流程，FTheta 相机 id）→ 6k proxy → 判定。
- [ ] **Step 3: 存活则 30k + 四读数 + 判定入档**（同 Task 9 流程）。

### Task 12: 收尾——rear_right 判定 + 60k 校准 + 文档回填

- Consumes: Task 9–11 全部读数。
- [ ] **Step 1: rear_right 去留材料**：其 held-out cc_psnr 全程曲线（R4e→C2→C3→C4）汇总表 → 交大g 拍板（纳入最终配方 vs 永久 eval-only）。
- [ ] **Step 2: 60k 容量校准**（定型配方一次 60k，吸收 I1 欠训效应；对照同配方 30k——预期每相机有效步数翻倍的增益读数）。
- [ ] **Step 3: 文档回填**：v5_plan §1 看板（P0 组 + C2–C4 状态 + 锚数字）、§0.2 KPI 表、§4 Done Log；v2_architecture §6 文件清单（`EgomaskAuxReader` / `derive_egomask_from_sseg.py` / telew）+ §7 不变量（fallback 字节等价 + camera_loss_weights 默认恒等）两行；mermaid 全角括号自查零输出。
- [ ] **Step 4: Commit** `docs(plan)+docs(arch): Phase C 阶梯收尾回填`。

---

## Self-Review 记录

- **Spec 覆盖**：P0.1 ✅（诊断已完成入 spec）；P0.2→Task 1+2；P0.3→Task 3；P0.4→Task 4；P0.5→Task 5；P0.6→Task 6；C1→Task 7；C2→Task 8+9；C3→Task 10；C4→Task 11；收尾→Task 12。无缺口。
- **签名一致性**：`resolve_ego_valid_mask` 在 Task 1 定义、Task 2 消费同名同参；`EgomaskAuxReader.read_static_mask` Task 1/3 一致；`_camera_loss_weight` 仅 Task 7。
- **占位符扫描**：无 TBD/TODO；训练类任务（4/8–12）为 runbook 型，验收判据与命令意图明确，不虚构结果数字。
