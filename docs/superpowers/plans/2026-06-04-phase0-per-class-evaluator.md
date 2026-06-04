# Phase 0 测量门 — per-class actor evaluator 执行 plan

> **任务来源**：[`v3_plan_revised.md`](../../../v3_plan_revised.md) § 2.0 Phase 0（P0.1–P0.4），plan 自定义的 **gate（门）**——把「前景 actor per-class 质量」从含糊的 30 dB 变成**可证的真实数字**，否则后续 Phase 1/2/3 任何「改善」都不可证（违反 [`CLAUDE.md`](../../../CLAUDE.md) 把关清单 C「metric 数字达标才算完成」）。
>
> **本 plan 性质**：建评测工具 + 立 per-class baseline 锚点，**纯 eval、无新训练**。代码 + Mac 单测可全部离线完成；只有「在 baseline ckpt 上跑出真实数字」需要 GPU（A800/vast），按用户决策**待排期**（Phase D 单独 gate）。
>
> **决策依据**：[v3 战略诊断](../specs/2026-06-04-v3-actor-centric-perf-diagnosis.md)、[v3_plan_revised.md](../../../v3_plan_revised.md) § 0–2、§ 2.4 降级清单（T17.2 V3-E2 per-class evaluator 提前到 P0.4）。

---

## 0. 范围与非目标

| 做 | 不做 |
|---|---|
| P0.1 车辆 class_psnr 立锚（跑现成工具，零新代码） | 任何新训练 / 重训 baseline（§0.4 ckpt 固定） |
| P0.2 行人/骑行 sseg-based per-class PSNR/LPIPS（新建） | 改善 actor 质量（那是 Phase 1/2/3） |
| P0.3 车道线 road-crop LPIPS（新建，lane mask 不存在已坐实） | 追求全局 novel-view PSNR ≥ 30（已判错轴） |
| P0.4 整合进 render.py eval + metrics.json 字段规范 | trainer 训练侧 per-class（可选，见 Phase C 说明） |
| 回填 §1.3 per-class gap 表 + Done Log + arch doc | A800 调度本身（用户决策待排期） |

---

## Phase 0 — Documentation Discovery 结论（Allowed APIs）

> 以下全部来自**实际读码**（4 个 Explore agent 报告，2026-06-04），非假设。每条带文件:行号出处。**实现时只用此清单内的 API；不在表内 = 先去读码确认，不要发明。**

### A. 现成可直接复用的张量评测函数

| API | 位置 | 签名 / 用途 | 复用于 |
|---|---|---|---|
| `compute_psnr_in_mask(rgb_pred, rgb_gt, mask, min_pixels=50) -> Optional[float]` | [`class_psnr.py:33`](../../../threedgrut/model/class_psnr.py) | 输入 `[H,W,3]`(0-1) + `[H,W]` mask；`PSNR=-10·log10(MSE)` 仅 mask 内；像素<min_pixels 返回 None | P0.2 / P0.3 PSNR |
| `compute_class_psnr(...) -> Dict` | [`class_psnr.py:60`](../../../threedgrut/model/class_psnr.py) | cuboid-based 车辆 per-class PSNR；**仅处理 batch[0]**；返回 `per_track`/`mean`/`by_class` | P0.1（已接通） |
| `collect_active_tracks_for_frame(...) -> List[Dict]` | [`class_psnr.py:160`](../../../threedgrut/model/class_psnr.py) | 从 `model.tracks_poses/tracks_active/tracks_metadata` 抽某帧 active tracks | P0.1（已接通） |
| `project_cuboids_to_mask(...) -> bool[H,W]` | [`dynamic_mask.py:111`](../../../threedgrut/layers/dynamic_mask.py) | cuboid 8 角 → 相机 → 2D AABB → bool mask；pinhole + FTheta 两路 | P0.1 内部 |

### B. 语义分割数据（sseg）— P0.2 / P0.3 数据源

| 事实 | 出处 |
|---|---|
| 类表确认：`person=11` / `rider=12` / `bicycle=18`；`road=0,sidewalk=1`(`ROAD_CLASS_IDS`)；`sky=10`；`DYNAMIC_CLASS_IDS={11..18}` | [`ncore_semantic.py:7-43`](../../../threedgrut/datasets/ncore_semantic.py) |
| **无 lane/marking/line 类**（Cityscapes 19 类标准；road 是粗类） | [`ncore_semantic.py:1-44`](../../../threedgrut/datasets/ncore_semantic.py) |
| sseg 磁盘读取：`SsegAuxReader(itar).read(camera_id, timestamp_us) -> [H,W] uint8`，文件 `*.aux.sseg.zarr.itar`，**START 时间戳** | [`aux_readers.py:59-119`](../../../threedgrut/datasets/aux_readers.py) |
| eval 路径**已加载 sseg** 并生成 `sky_mask`/`road_mask`/`dyn_mask_sseg`（仅 val/test 分支，`load_aux_masks=True`） | [`datasetNcore.py:930-956`](../../../threedgrut/datasets/datasetNcore.py) |
| ⚠️ batch 里只有**聚合** `dyn_mask_sseg`（11-18 合并），**没有 per-class（person/rider/bicycle 分开）也没有 raw sseg** | [`datasetNcore.py:953-955`](../../../threedgrut/datasets/datasetNcore.py) |
| GPU batch 入口：`gpu_batch.image_infos["dyn_mask_sseg"]` `[B,H,W]` float32 | [`protocols.py:51`](../../../threedgrut/datasets/protocols.py) / [`datasetNcore.py:1558-1570`](../../../threedgrut/datasets/datasetNcore.py) |

### C. 现成 metric 工具

| API | 位置 |
|---|---|
| LPIPS：`LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)` (torchmetrics) | [`trainer.py:568`](../../../threedgrut/trainer.py) / [`render.py:363`](../../../threedgrut/render.py) |
| **masked LPIPS/SSIM 正确模式 = GT-fill**（torchmetrics LPIPS 无像素 mask）：`pred_filled = pred*m + gt*(1-m)` 再 `lpips(pred_filled, gt)` | [`trainer.py:1009-1018`](../../../threedgrut/trainer.py) / [`render.py:608-622`](../../../threedgrut/render.py) |
| masked PSNR 公式：`diff_sq=(pred-gt)²*mask; mse=diff_sq.sum()/(mask.sum()*3); -10·log10(mse)` | [`trainer.py:1000-1008`](../../../threedgrut/trainer.py) |
| color-correct：`color_correct_affine(pred, gt)` | [`render.py:573`](../../../threedgrut/render.py) |

### D. road region / BEV 基建 — P0.3 复用

| API | 位置 | 用途 |
|---|---|---|
| `build_road_height_field(road_positions[M,3], cell_size=1.0) -> Dict` | [`road_region.py:22`](../../../threedgrut/model/road_region.py) | road 层粒子 → BEV 高度场 |
| `query_ground_z(positions_xy[N,2], hf) -> (ground_z[N], valid[N])` | [`road_region.py:93`](../../../threedgrut/model/road_region.py) | 任意 XY 查地面 Z + valid（无梯度） |
| road 粒子 XY scale max **0.3m ≈ 2×车道线宽**，Z 0.05m 薄盘 → 线条锐度在此尺度，**LPIPS 敏感、PSNR 被沥青淹没** | [`layer_spec.py:66-72`](../../../threedgrut/layers/layer_spec.py) |
| BEV warp 参考实现（可选 stretch 方案） | [`threedgrut_playground/utils/bev_stitcher.py`](../../../threedgrut_playground/utils/bev_stitcher.py) |

### E. metrics 两路径 + 集成点 — P0.4

| 事实 | 出处 |
|---|---|
| **metrics.json 真源 = render.py `render_all()`**（eval 独立路径） | [`render.py:354`](../../../threedgrut/render.py) |
| metrics.json 组装 + 写盘 | [`render.py:790-940`](../../../threedgrut/render.py)（write 在 `:938-940`） |
| **class_psnr 已接通**：调用 `:691-720`，写 `mean_class_psnr`/`class_psnr_by_class`/`class_psnr_n_records`/`class_psnr_n_low_15db` `:881-899` | [`render.py:691-899`](../../../threedgrut/render.py) |
| novel-view 4 档：`NOVEL_VIEW_MODES=("lateral_1m","lateral_2m","yaw_5deg","yaw_10deg")`，写 `mean_novel_lpips_<mode>`+`mean_novel_lpips_avg` | [`novel_view.py:42`](../../../threedgrut/utils/novel_view.py) / [`render.py:805-828`](../../../threedgrut/render.py) |
| 逐帧主循环（render→metric→累积），per-class 新指标插这里 | [`render.py:466-744`](../../../threedgrut/render.py) |
| 训练侧 `Trainer3DGRUT.get_metrics()`（**class_psnr/novel_lpips/per_camera 均 render-only，不在此**） | [`trainer.py:936-1069`](../../../threedgrut/trainer.py) |
| ego mask 逻辑：`mask=gpu_batch.mask` 只屏蔽 ego 车身、**不屏蔽行人** | [`trainer.py:999`](../../../threedgrut/trainer.py) |
| ckpt 加载：`Renderer.from_checkpoint(checkpoint_path, ...)`；`torch.load(weights_only=False)` | [`render.py:119-219`](../../../threedgrut/render.py) |

---

## 反模式清单（Anti-patterns — 实现时 grep 自查）

| ❌ 反模式 | ✅ 正确做法 | 出处 |
|---|---|---|
| `lpips(pred*mask, gt*mask)`（把背景置黑当 mask） | GT-fill：`pred_filled=pred*m+gt*(1-m)` 再 `lpips(pred_filled,gt)` | [`trainer.py:1009-1018`](../../../threedgrut/trainer.py) |
| 假设 batch 里有 raw sseg 或 person/rider/bicycle mask | 现状只有聚合 `dyn_mask_sseg`，**必须新增透传** | [`datasetNcore.py:953-955`](../../../threedgrut/datasets/datasetNcore.py) |
| 找/造 `lane`/`lane_marking` 语义类 | **不存在**；P0.3 走 road-crop LPIPS | [`ncore_semantic.py`](../../../threedgrut/datasets/ncore_semantic.py) |
| per-class 新指标只插 render.py 就以为「双路径」漏了 trainer | class_psnr/novel_lpips 本就 render-only；trainer 侧**可选**（见 Phase C） | [`render.py:691`](../../../threedgrut/render.py) |
| 训练 exit 0 / ckpt 写出就标 ✅ | metrics.json **实测新 key 出现**才标 ✅ | [`CLAUDE.md`](../../../CLAUDE.md) C.6/C.9 |
| 直接 GPU 跑评测前不验 sseg palette | 先读 1 帧 `np.unique(sseg)` 确认 0-19（NRE palette 可能偏移） | [`ncore_semantic.py:29-32`](../../../threedgrut/datasets/ncore_semantic.py) TODO |
| compute_class_psnr 当多图函数用 | 它**只处理 batch[0]** | [`class_psnr.py:102-103`](../../../threedgrut/model/class_psnr.py) |

---

## Phase 结构总览

| Phase | 内容 | 需 GPU? | 产出 | 估时 |
|---|---|:--:|---|--:|
| **A** | P0.2 行人/骑行 per-class evaluator + sseg 透传 + Mac 单测 | ❌ 离线 | `per_class_eval.py` + test | 1.5d |
| **B** | P0.3 车道线 road-crop LPIPS evaluator + Mac 单测 | ❌ 离线 | road-crop 函数 + test | 1d |
| **C** | P0.4 整合进 render.py `render_all()` + metrics.json 字段 + Mac dry-run | ❌ 离线 | render.py 改动 | 1d |
| **D** ⛔gate | sseg palette 验证 + P0.1 跑现成 + 全 eval → 真实数字 → 回填 gap 表/Done Log | ✅ A800/vast **待排期** | per-class baseline 数字 | 0.5d GPU |
| **E** | 最终验证（grep 反模式 + metrics.json key 核对 + 文档同步） | 部分 | 验收闭环 | 0.5d |

> **离线先行**：A/B/C 全部不碰 GPU，可立即做。D 是唯一 GPU gate，用户决策待排期；A/B/C 完成后 D 只需一次 eval run 即可一次性产出全部 per-class 数字。

---

## Phase A — P0.2 行人/骑行 sseg-based per-class evaluator（离线）

**目标**：person(11)/rider(12)/bicycle(18) 三类的 per-class PSNR + LPIPS，对齐到 render eval 逐帧路径。

### A.1 sseg per-class mask 透传（新增 batch 字段）
**复制来源**：[`datasetNcore.py:930-955`](../../../threedgrut/datasets/datasetNcore.py)（现有 sky/road/dyn 三 mask 生成块）。
**做什么**：在该块内，紧挨 `dyn_mask_sseg` 之后，**新增 raw sseg 透传**（推荐，最通用）：把 `sseg`（resize 后的 `[H,W] uint8`）原样存进 `batch_dict["semantic_sseg"]`，再在 `get_gpu_batch_with_intrinsics`（[`datasetNcore.py:1558-1570`](../../../threedgrut/datasets/datasetNcore.py)）把它加进 `image_infos`。
- **为何 raw 而非三个 per-class mask**：evaluator 端 `(sseg==11)`/`(==12)`/`(==18)` 现取，P0.3 road-crop 也能 `(sseg∈{0,1})` 复用，一处改动服务两 Phase。
- **anti-pattern 守卫**：不要假设它已存在——现状只有聚合 `dyn_mask_sseg`。

### A.2 per-class evaluator 模块（新文件）
**新建** `threedgrut/model/per_class_eval.py`，**直接 import 复用** [`compute_psnr_in_mask`](../../../threedgrut/model/class_psnr.py#L33)（不重造 PSNR）。
- `compute_per_class_psnr(rgb_pred[H,W,3], rgb_gt[H,W,3], sseg[H,W], class_ids: dict) -> dict`：对每类 `mask=(sseg==id)`，调 `compute_psnr_in_mask`，<min_pixels 返回 None（帧内该类不存在很常见，需可空跳过）。
- `compute_per_class_lpips(...)`：**GT-fill 模式**（照抄 [`render.py:608-622`](../../../threedgrut/render.py)），不可用 `pred*mask`。
- 默认 `class_ids={"person":11,"rider":12,"bicycle":18}`，从 [`ncore_semantic.py`](../../../threedgrut/datasets/ncore_semantic.py) 常量引入，不硬编码数字两份。

### A.3 Mac 单测（先写，pin 行为）
**模板**：[`tests/test_class_psnr.py:98-242`](../../../threedgrut/tests/test_class_psnr.py)（已有「已知 PSNR」合成张量测试范式）。
- 构造合成 `rgb_pred/rgb_gt` + 合成 `sseg`（某矩形块=11），断言该类 PSNR ≈ 解析值。
- 测「某类 0 像素 → 返回 None / 跳过」；测 GT-fill LPIPS 在 mask 全 1 时 == 全图 LPIPS。
- 跑：`python -m pytest threedgrut/tests/test_per_class_eval.py -v`（Mac 本地，按 [`CLAUDE.md`](../../../CLAUDE.md) Python `.venv` 约定）。

### A 验收
- [ ] `pytest test_per_class_eval.py` 全绿（Mac）
- [ ] grep 确认无 `lpips(.*\*.*mask` 裸 mask 反模式
- [ ] evaluator 不依赖 GPU，可纯 CPU 张量跑通

---

## Phase B — P0.3 车道线 road-crop LPIPS evaluator（离线）

**前提结论（已坐实）**：lane mask 不存在 → 走 road-crop。**LPIPS 选型理由**：road PSNR 被大片平整沥青低频淹没，LPIPS 感知对 0.1–1m 车道线高频边缘敏感（road 粒子 0.3m XY scale 印证该尺度，[`layer_spec.py:66`](../../../threedgrut/layers/layer_spec.py)）。

### B.1 road-crop LPIPS（主方案，最低风险）
**复用**：A.1 已透传的 `sseg` → `road_mask=(sseg∈ROAD_CLASS_IDS)`（[`ncore_semantic.py`](../../../threedgrut/datasets/ncore_semantic.py)），在 `per_class_eval.py` 加：
- `compute_road_crop_lpips(rgb_pred, rgb_gt, road_mask) -> Optional[float]`：GT-fill LPIPS 限定 road 像素；像素<阈值返回 None。
- 同时输出 `road_crop_psnr`（作对照，预期被沥青淹没——记录以证明「PSNR 测不出、LPIPS 才行」）。

### B.2（可选 stretch）BEV warp road-crop LPIPS
更忠实「俯视锐度」，但重。**不阻塞 P0**——先交 B.1，BEV 作 backlog。
- 参考 [`bev_stitcher.py`](../../../threedgrut_playground/utils/bev_stitcher.py) IPM + [`road_region.query_ground_z`](../../../threedgrut/model/road_region.py#L93)。
- ⚠️ gap：`query_ground_z` 按世界 XY 查询，需每像素世界坐标（depth/raycast）——非平凡；故 B.1 透视 road-crop 才是确定可交付路径。

### B 验收
- [ ] `compute_road_crop_lpips` 合成测试通过（road=全图时 == 全图 LPIPS）
- [ ] road_crop_psnr vs road_crop_lpips 双值都产出（佐证选型）

---

## Phase C — P0.4 整合进 render.py eval + metrics.json 字段（离线）

**集成点**：[`render.py` `render_all()`](../../../threedgrut/render.py#L354)，**照搬现有 class_psnr 的接线范式**（它已是 render-only per-class 指标的活样板）。

### C.1 逐帧累积
在主循环 [`render.py:466-744`](../../../threedgrut/render.py)（class_psnr 调用 `:691-720` 旁），新增：从 `gpu_batch.image_infos["semantic_sseg"]` 取 sseg → 调 `compute_per_class_psnr/lpips` + `compute_road_crop_lpips` → append 到新累积 list（仿 `psnr=[]` 等 `:385-413`）。

### C.2 metrics.json 字段
在组装段 [`render.py:790-940`](../../../threedgrut/render.py) 仿 class_psnr 块（`:881-899`）追加：
```
mean_person_psnr / mean_person_lpips
mean_rider_psnr  / mean_rider_lpips
mean_bicycle_psnr/ mean_bicycle_lpips
person_n_records / rider_n_records / bicycle_n_records   # 该类有效帧数（行人很可能=地板）
mean_road_crop_lpips / mean_road_crop_psnr
```
- **空值处理**：行人极可能多帧 0 像素 → mean 仅对 non-None 求；`*_n_records=0` 时显式写 0（证明「测了、确实没有」，不是漏测）。

### C.3 4 档 novel pose 拆解（P0.4 第二半）
现有 `mean_novel_lpips_<mode>`（[`render.py:805-828`](../../../threedgrut/render.py)）是**全图**。P0.4 要「车/人/路/bg 拆解」：在 novel 重渲分支 [`render.py:651-689`](../../../threedgrut/render.py) 内，对每 mode 的 `pred_novel` 也跑 per-class/road-crop（GT 用原帧 sseg 近似）。**降风险**：novel 无真 GT，先只交「全图 novel 不退化监控」+ per-class novel 标 🟡 stretch，不阻塞 gate。

### C.4 trainer 侧（可选，默认不做）
class_psnr/novel_lpips/per_camera 均 render-only；P0.2/P0.3 同性质（eval-time）。**默认只改 render.py**。若要训练中监控 per-class，再在 [`trainer.py:977`](../../../threedgrut/trainer.py) `get_metrics` 加——但**非 gate 必需**。在 plan/commit 注明此决策，避免被 CLAUDE.md「两路径」误判为漏改。

### C.5 离线 dry-run（无 GPU）
写一个最小脚本/测试，喂合成 `image_infos["semantic_sseg"]` + 假 outputs，断言 metrics dict 出现全部新 key（不需真 ckpt/渲染）。

### C 验收
- [ ] 离线 dry-run：新 key 全部出现在组装出的 metrics dict
- [ ] `python -c "import threedgrut.render"` 无 import error
- [ ] grep `render_all` 改动只动累积+组装两处，未破坏现有 class_psnr/novel 接线

---

## Phase D — GPU gate：实测立锚 + 回填（⛔ 待用户排期 A800/vast）

> 离线 A/B/C 完成后，**一次 eval run** 同时产出 P0.1 车辆数字 + P0.2 行人 + P0.3 车道线。GPU 调度按用户决策待排期。

### D.1 远程代码就绪验证（按 [`CLAUDE.md`](../../../CLAUDE.md) 清单 A）
- rsync A/B/C 改动 → `grep -n` 关键字符串确认同步到 a800。
- `head -25 render.py` 确认是 argparse 脚本风格（历史踩坑防呆）。

### D.2 sseg palette 健全性（先于 eval，便宜防呆）
读 baseline clip 的 1 帧 sseg：`SsegAuxReader(...).read(cam, ts)` → `np.unique` 应 ⊆ 0-19 且含 11/12/18 之一（[`ncore_semantic.py:29-32`](../../../threedgrut/datasets/ncore_semantic.py) TODO）。palette 偏移则先修映射再 eval。

### D.3 P0.1（零新代码）+ 全 eval
在 §0.4 baseline ckpt（`a800:.../v3_base_scratch30k_lam01/...-0406_204815/ours_30000/ckpt_30000.pt`）上跑 render.py eval（exact flag 见 [`render.py:22-80`](../../../threedgrut/render.py)；带 `--novel-view`，`load_aux_masks=true`）。
- Monitor 只 grep 关键节点（**不 grep "PSNR"**，会被逐帧刷爆 rate limit，[`CLAUDE.md`](../../../CLAUDE.md)）。

### D.4 回填（证据闭环）
`cat metrics.json` 确认全部新 key 存在且 `*_n_records` 合理，把实测数字填入：
- [`v3_plan_revised.md`](../../../v3_plan_revised.md) § 1.3 **per-class gap 表**三行（车辆/行人/车道线）
- § 6 Done Log 新条目（日期+commit+实测数）
- § 1.1/1.2 看板：P0.1–P0.4 → ✅
- [`v2_architecture.md`](../../../v2_architecture.md) 文件清单 + 不变量表（新 evaluator 模块）

### D 验收（CLAUDE.md C.6/C.8/C.9）
- [ ] metrics.json 含 `mean_class_psnr`(车) + `mean_person_psnr`(+rider/bicycle) + `mean_road_crop_lpips` 全部新 key
- [ ] gap 表三行真实数字回填，commit hash 入档
- [ ] **若实测改变 Phase 1/2 优先级**（如车辆已不差、行人是唯一真缺口）→ 据实重排看板（§ 2.0 验收明文要求）

---

## Phase E — 最终验证

1. **反模式 grep 全仓自查**：`grep -rn "lpips(.*\* *mask"`（裸 mask）、`grep -rn "lane_marking\|lane_mask"`（造不存在类）应无新增命中。
2. **双路径核对**：确认 metrics.json 真源（render.py）已含新 key；trainer 侧未改的决策已在 commit message 注明（C.4）。
3. **Mac 全测**：`pytest threedgrut/tests/test_per_class_eval.py` + 现有 `test_class_psnr.py`/`test_ncore_aux_masks.py` 回归全绿。
4. **文档同步**（[`CLAUDE.md`](../../../CLAUDE.md) v2 工作流纪律）：plan 看板 + Done Log + arch 文件清单三处同步，commit message 含 `docs(plan)` / `docs(arch)` 行。
5. **gate 出口**：§1.3 per-class gap 表「Phase 0 实测」列三行不再是「待 P0.x」。

---

## 风险登记（本 plan 局部，补 [`v3_plan_revised.md`](../../../v3_plan_revised.md) § 4）

| ID | 风险 | 缓解 |
|---|---|---|
| E1 | sseg NRE palette 与 Cityscapes 偏移 → person/road mask 错类 | D.2 先 `np.unique` 1 帧验证再 eval |
| E2 | 行人帧多数 0 像素 → 指标几乎空 | 这正是结论（「地板」），用 `*_n_records` 显式记录非缺测；少量有像素帧即可立锚 |
| E3 | road-crop LPIPS 仍被沥青稀释 | 同时输出 road_crop_psnr 作对照；BEV warp(B.2) 作 backlog stretch |
| E4 | torchmetrics LPIPS 裸 mask 误用 | 强制 GT-fill，E.1 grep 守卫 |
| E5 | 只改 render.py 被误判漏 trainer 双路径 | C.4 显式决策 + commit 注明（class_psnr 同为 render-only 先例） |
| E6 | GPU 排期拖延 | A/B/C 全离线先交，D 解耦为单次 run；不阻塞代码评审/合并 |

---

## 执行顺序速记

```
A (sseg透传+行人evaluator+Mac测) ─┐
B (road-crop LPIPS+Mac测) ───────┼─► C (render.py整合+metrics字段+dry-run) ─► [⛔gate] D (GPU eval+回填) ─► E (验收+文档同步)
                                  ┘   离线可全部先行 ───────────────────────►  待用户排期
```
