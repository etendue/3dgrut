# v4 E1 外推测量门 + E0.4 双向对照锚 + E0.6 官方编辑体验 — 执行 Plan

> **For agentic workers:** REQUIRED SUB-SKILL: 用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 按卡执行。步骤用 checkbox（`- [ ]`）跟踪。
>
> **Goal:** 把 v4 的外推性能从不可证变为可证——立全 4 类外推锚（3m/6m lane、NTA-IoU、held-out 真 GT、FID/KID），用同一套工具完成 NuRec vs multilayer 双向对照（E0.4），并在自有 clip 重建上产出官方 actor 编辑能力/限制清单（E0.6）。
> **Architecture:** 全部为 eval 侧增量（零训练代码改动，除 E1.3 的一次配置级从头训练）；新指标沿 render.py 既有 accumulate-then-finalize 模式挂入；NuRec 侧只产帧、指标代码统一用项目侧保口径。
> **Tech Stack:** PyTorch / torchmetrics（+torch-fidelity）/ ultralytics YOLOv8m / nre 26.4 容器 / Harmonizer（harmonizer-cosmos-env）。
> **机器纪律（CLAUDE.md）:** GPU 任务一律 inceptio（depth-off + num_workers=10 铁律）；Mac 做 TDD 单测；inceptio 用 git worktree 工作流（push 分支 → worktree add → 跑完删）。
> **Step 0（开工第一步）:** 把本 plan 落盘到仓库 `docs/superpowers/plans/2026-06-12-e1-extrapolation-gate-and-e06.md` 并 commit（plan-mode 文件在仓库外，按项目惯例入库）。

---

## 1. 目的

当前外推测量体系有两个盲区（2026-06-11 诊断，v3_plan_revised.md § 2.3）：

- **幅度盲区**：`NOVEL_VIEW_MODES` 只有 lateral_1m/2m + yaw_5/10deg，退化主战场 3m/6m 在测量范围外；
- **区域盲区**：全图 LPIPS 被沥青/bg 大面积主导，车道线（几像素宽条纹）糊掉指标无反应。

E1 立 4 类锚补盲区；E0.4 用同口径量化"与 NuRec 差距在哪一层"（表示/配方/修复器）——这是 E2/E3 优先级的判别依据；E0.6 产出官方编辑工作流的能力/限制清单（喂 E2.5，对照 P1.4 撞过的 spiky/光照失配/悬浮问题官方怎么解）。**E1 不立锚，E2/E3 任何"改善"不可证**——v3"测量先行"教训在外推轴的重演。

## 2. 原理

1. **aperture problem 根因**：训练相机全在 ego 轨迹一条线、路面掠射角观测 → road/bg 欠约束耦合；外推退化与 bg 悬浮粒子同根因两面 → 测量必须按「幅度档位 × 病灶区域」分解，不能用全图单数。
2. **lane novel 指标 = 平面诱导 warp 造伪 GT**：3m/6m 外推无像素 GT，但车道线油漆贴在路面平面上——novel 像素射线与路面高度场求交、投回原相机采样原 GT，对平面上内容（lane band）该 warp 数学严格成立。指标内核复用 PR #24 的 `compute_lane_metrics`（Sobel grad_corr + band LPIPS/PSNR）。
3. **NTA-IoU / held-out**：车辆外推用「渲染帧检测框 vs 投影 GT cuboid」IoU（ReconDreamer 系共识协议）；held-out 侧相机是唯一有真像素 GT 的外推轴（DiFix3D+ RDS cross-reference 反用）。
4. **E0.2 FID 教训（R-v4.5 实证）**：官方场景修复后目视显著但 FID 几乎不动（lat3m 57.3→65.6 反升）——FID 大头是视角内容差非伪影 → E1.4 必须 KID 优先（小样本无偏）且只作辅助：任何验收要求真 GT/区域化指标与感知指标**同向**（双协议）。
5. **历史口径不可破**：`mean_novel_lpips_avg`（4 档平均，B3 锚 0.5962）字段语义永不变；新档进新字段。
6. **R9 边界**：E1.1 只移植 PR #24 的 **eval 侧**（lane 指标 + 数据通路 + lane GT 脚本），lane loss / trainer / road yaml 一概不拉 → 不预决 PR #24 去留。

## 3. 任务列表

| 卡 | 一行目标 | 验收 | 估时 |
|---|---|---|---:|
| **E1.1** | novel 扩 lateral_3m/6m（历史 avg 不变）+ PR #24 eval 侧移植 + 平面 warp lane novel 指标 + 三方 ckpt 立锚 | Mac 单测全绿；三方 ckpt metrics.json 含全部新 key；baseline `mean_novel_lpips_avg` 回归 ±0.001 | 1.5d |
| **E1.2** | NTA-IoU 按既有 plan Task 0–5 + 增量 novel 档 NTA-IoU | `mean_nta_iou` ∈ (0,1] 且 `mean_novel_nta_iou_lateral_3m/6m` 齐全 | 1.5d |
| **E1.3** | held-out 协议：排除 cross_left 4-cam 从头训 30k + `--dataset-cameras` eval 开关 + 三件套立锚 | held-out 相机 per-class 全套 + NTA 入档；三件套（held-out/守护/上界）齐 | 1.5d |
| **E1.4** | FID/KID 接入 render eval（accumulate-then-finalize，`--novel-fid` 开关） | `mean_novel_kid_<mode>`（主）+ `mean_novel_fid_<mode>` 新 key 齐；6m>3m 方向 sanity | 1d |
| **E0.4** | NuRec last.usdz 同位姿出帧 → `scripts/eval_frames_dir.py` 双向同口径全指标 | gap 表 NuRec 列回填 + 判别结论入档 | 1d |
| **E1.5** | gap 表回填 + 据实重排 E2/E3 | §1.3 gap 表无"待测"残留；E2/E3 顺序有依据句 | 0.5d |
| **E0.6** | 自有 clip last.usdz 官方编辑 run-book（删/插/替 + Harmonizer 协调） | `docs/superpowers/specs/2026-06-12-e06-official-actor-editing-capability.md` 清单入档 + 对照帧存档 | 1d |

## 4. 依赖关系与排程

```mermaid
flowchart TD
  classDef gate fill:#e6f4ff,stroke:#0070f3,color:#000
  classDef night fill:#fff3e0,stroke:#e65100,color:#000
  classDef indep fill:#f0f9eb,stroke:#67c23a,color:#000

  PR24["E1.1-A PR24 eval 侧移植（path 限定，不预决 R9）"]:::gate
  E11["E1.1 扩档 + lane warp 指标 + 三方立锚"]:::gate
  E12["E1.2 NTA-IoU + novel 档联动"]
  E13T["E1.3-B 4-cam 30k 训练（夜间档 ~7h）"]:::night
  E13E["E1.3-C held-out 三件套 eval"]
  E14["E1.4 FID KID 接入"]
  E04["E0.4 双向对照锚（NuRec 帧 → eval_frames_dir）"]
  E15["E1.5 gap 表回填 + E2 E3 重排"]
  E06["E0.6 官方编辑 run-book（独立穿插）"]:::indep

  PR24 --> E11
  E11 --> E12
  E11 --> E14
  E12 --> E13E
  E13T --> E13E
  E11 --> E04
  E12 --> E04
  E14 --> E04
  E13E --> E15
  E04 --> E15
  E06 -. 能力清单喂 E2.5（截止 E1.5 前即可）.-> E15
```

- **GPU 错峰（inceptio 4090 单卡）**：E1.3-B 训练是唯一长任务（~7h），**E1.1-A/B 完成当晚即启动**（训练只需配置覆盖，与 eval 工具解耦）；白天跑 eval 短任务（三方立锚每 ckpt 6 档 novel ~40–60min、smoke ~15min、nre render 17.8ms/帧级、E0.6 serve-grpc）。serve-grpc 常驻占显存，**与训练/eval 互斥**，用完即收。
- **E0.6 完全独立**（不碰项目代码），作训练期间白天的并行穿插任务。
- **代码全在本 worktree 分支** `claude/dreamy-raman-8264ca` 开发；GPU 跑用 `git push inceptio <branch>` + inceptio worktree（CLAUDE.md ⭐ 工作流，记得补 submodule rsync）。

## 5. 具体 subtask

### 5.1 E1.1 — 外推测量门扩展（=v3 P3.3 移交）

**Files:**
- Port（来自 `origin/claude/crazy-wing-a16f19`，仅 eval 侧）: `threedgrut/model/per_class_eval.py`、`threedgrut/render.py`、`render.py`、`threedgrut/datasets/__init__.py`、`threedgrut/datasets/datasetNcore.py`、`threedgrut/datasets/aux_readers.py`、`scripts/gen_lane_sseg.py`、`threedgrut/tests/test_per_class_eval.py`、`threedgrut/tests/test_aux_discover_lane.py`
- Modify: `threedgrut/utils/novel_view.py:42-100`、`threedgrut/render.py`（novel 分支 ~L674-704 / 聚合 ~L845-863）
- Create: `threedgrut/model/plane_warp.py`、`threedgrut/tests/test_plane_warp.py`
- Test: `threedgrut/tests/test_novel_view.py`

#### E1.1-A：PR #24 eval 侧移植

- [ ] **A1** path 限定 apply（**禁拉**：`lane_loss.py` / `trainer.py` / `configs/*` / `test_lane_loss.py` / `test_road_scale_override.py`）：

```bash
git fetch origin claude/crazy-wing-a16f19
git diff origin/main...origin/claude/crazy-wing-a16f19 -- \
  threedgrut/model/per_class_eval.py threedgrut/render.py render.py \
  threedgrut/datasets/__init__.py threedgrut/datasets/datasetNcore.py \
  threedgrut/datasets/aux_readers.py threedgrut/tests/test_per_class_eval.py \
  threedgrut/tests/test_aux_discover_lane.py scripts/gen_lane_sseg.py \
  | git apply --3way
```

拉入：`compute_lane_metrics` / `LANE_CLASS_IDS=(23,24)` / `DEFAULT_LANE_BAND_PX=8`（per_class_eval）；`load_lane_masks` 数据通路 + `semantic_lane_sseg` 透传（datasets 三件）；`--load-lane-masks` / `--lane-band-px` CLI + `mean_lane_*` 聚合（render 两层）；lane GT 生成脚本 + 两个测试。
- [ ] **A2** 验证：`pytest threedgrut/tests/test_per_class_eval.py threedgrut/tests/test_aux_discover_lane.py -v` 全绿（Mac）；`git diff main --stat` 确认无 trainer/configs/lane_loss。
- [ ] **A3** commit（message 写明：eval-only port from PR #24 commit 区间，R9 边界声明——loss/trainer/yaml 未拉，R9 决议后 reconcile）。
- [ ] **A4** lane GT 数据盘点（并行）：`ssh inceptio 'ls ~/work/data/9ae151dc/ | grep -i lane; ls ~/work/data/9ae151dc_consolidated/ 2>/dev/null | grep -i lane'`——E0.3 时 lane aux itar 被移出 consolidated 目录，确认原目录仍在；缺则用 `scripts/gen_lane_sseg.py` 重产（mask2former 权重走 mihomo 代理）。

#### E1.1-B：novel_view.py 扩档（TDD）

- [ ] **B1** 改 `threedgrut/tests/test_novel_view.py`：4 档常量断言改为「`LEGACY_NOVEL_AVG_MODES` == 4 档 ∧ `NOVEL_VIEW_MODES` == 6 档 ∧ 前 4 元素一致」；新增 `test_lateral_3m_is_triple_of_1m`（位移 = 3×lateral_1m 位移）、`test_lateral_6m_shutter_pair_rigid`（start/end 同 delta）。跑 `pytest threedgrut/tests/test_novel_view.py -v` 确认 FAIL。
- [ ] **B2** 实现 `threedgrut/utils/novel_view.py`：

```python
LEGACY_NOVEL_AVG_MODES: Tuple[str, ...] = (
    "lateral_1m", "lateral_2m", "yaw_5deg", "yaw_10deg")  # 历史 avg 口径，永不扩
NOVEL_VIEW_MODES: Tuple[str, ...] = LEGACY_NOVEL_AVG_MODES + (
    "lateral_3m", "lateral_6m")                           # E1.1 新档
```

`perturb_c2w` 加两个 elif（`+3.0 / +6.0 * m[:3, 0]`）；`perturb_shutter_pair` lateral 分支 `startswith("lateral")` 已通用，零改动。
- [ ] **B3** `pytest threedgrut/tests/test_novel_view.py -v` PASS → commit。

#### E1.1-C：聚合口径保护（render.py）

- [ ] **C1** `threedgrut/render.py` 聚合段（~L845-863）改为双字段：

```python
legacy_vals = [float(np.mean(novel_lpips[m])) for m in LEGACY_NOVEL_AVG_MODES if novel_lpips.get(m)]
if legacy_vals:
    novel_json["mean_novel_lpips_avg"] = float(np.mean(legacy_vals))   # 口径不变（仅 4 档）
all_vals = [float(np.mean(novel_lpips[m])) for m in NOVEL_VIEW_MODES if novel_lpips.get(m)]
if all_vals:
    novel_json["mean_novel_lpips_avg6"] = float(np.mean(all_vals))     # 新 6 档聚合
```

（变量名以现场代码为准；per-mode `mean_novel_lpips_<mode>` 循环自动覆盖新档。）
- [ ] **C2** 口径回归（inceptio，三方立锚时合并跑）：baseline ckpt `--novel-view` 渲染后比对 `mean_novel_lpips_avg` 与其历史 metrics.json，差 ≤0.001。

#### E1.1-D：`threedgrut/model/plane_warp.py`（纯函数，Mac TDD）

核心算法（三步）：① novel 像素射线——`gpu_batch.rays_dir`（[1,H,W,3] 相机系，dataset 按 FTheta 多项式预生成缓存，datasetNcore.py:1459-1496）→ 世界系 `d_w = R_novel @ d_cam`，`o_w = t_novel`（**无需自写 FTheta 反投影**；离线场景用 `threedgrut_playground/utils/ftheta_intrinsics.py:23 ftheta_pixels_to_camera_rays` 同数学）；② 路面求交——复用 `threedgrut/model/road_region.py:93 query_ground_z`（V3-R2 高度场），定点迭代 2 轮：`t=(z_ground−o_z)/d_z → xy=o+t·d → 重查 z_ground`，`valid = 命中格 ∧ t∈(0,120m) ∧ d_z<0`；③ 投回原相机——`P_cam = inv(c2w_orig) @ P_w`，FTheta 正投影（与 `threedgrut/layers/dynamic_mask.py:84` `_corners_to_pixels_ftheta` 同数学，抽公共点级函数）→ `F.grid_sample` 采原 GT（bilinear）/ lane mask（nearest）。

- [ ] **D1** 写 `threedgrut/tests/test_plane_warp.py`（纯 CPU 合成数据）：
  - `test_ftheta_project_matches_dynamic_mask`：`ftheta_project_points` 与 `_corners_to_pixels_ftheta` 同输入同输出（钉公共数学）；
  - `test_flat_plane_identity_warp`：z=0 平面 + 近似恒等 FTheta 多项式下，横移相机 warp 后棋盘 GT 与解析投影一致（atol 1px）；
  - `test_sky_rays_invalid`：`d_z ≥ 0` 像素 valid=False；
  - `test_grid_normalization_corners`：u=0→−1、u=W−1→+1（align_corners=True）。
  跑确认 FAIL。
- [ ] **D2** 实现，签名：

```python
def ftheta_project_points(points_cam: torch.Tensor, ftheta_params: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """[N,3] 相机系点 → (uv [N,2] 像素, valid [N]＝z>0)。Horner(angle_to_pixeldist_poly)。"""

def build_plane_warp(rays_dir_cam, c2w_novel, c2w_orig, ftheta_params,
                     height_field: dict | None, z0_fallback: float | None = None,
                     n_iters: int = 2, t_max: float = 120.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """→ (grid [1,H,W,2] 归一化供 grid_sample, valid [H,W] bool)。
    valid = 命中高度场 ∧ 投回原相机 z>0 ∧ uv 在图内。"""

def warp_image(img_hwc, grid, valid, mode="bilinear") -> torch.Tensor:
    """warp 后 [H,W,C]；invalid 像素置 0，调用方以 valid 作 restrict_mask。"""
```

高度场来源（render_all 内建一次）：road 层 positions（layered ckpt）→ `build_road_height_field(road_pos, cell_size=1.0)`；非 layered ckpt 且无 `--ground-z` → 跳过 lane novel 指标 + warn（软失败）。
- [ ] **D3** `pytest threedgrut/tests/test_plane_warp.py -v` PASS → commit。

#### E1.1-E：render.py novel 分支接 lane warp 指标

- [ ] **E1** novel 循环内（每 mode 渲完 `pred_novel` 后，仅当 lane mask 存在且 warp 可用）：

```python
grid, valid = build_plane_warp(gpu_batch.rays_dir[0], nT[0], orig_T[0],
                               ftheta_params, height_field)
gt_warp   = warp_image(rgb_gt_full[0], grid, valid)
lane_warp = warp_image(lane_one.float(), grid, valid, mode="nearest").long()
lm = compute_lane_metrics(pred_novel[0], gt_warp, lane_warp, LANE_CLASS_IDS,
                          band_px=lane_band_px, restrict_mask=valid,
                          lpips_fn=criterions.get("lpips"))
```

6 档全算（yaw 档 warp 同样成立，白送信号）；聚合段写 `mean_novel_lane_{grad_corr|band_lpips|band_psnr}_<mode>` + `novel_lane_n_records_<mode>` + `novel_lane_warp_valid_ratio_<mode>`（valid∩band 像素比，质量哨兵）。
- [ ] **E2** 已知近似写注释 + 文档：warp 用 shutter-start 位姿（忽略 rolling shutter）；bilinear 轻微平滑 → **warped 指标只在同 warp 版本内可比**（跨模型可比，不与 interpolated `mean_lane_grad_corr` 比绝对值）。
- [ ] **E3** inceptio smoke：`python render.py --checkpoint <baseline ckpt_30000> --out-dir ~/work/e11/smoke --novel-view --load-lane-masks` → metrics.json 新 key 齐 + 旧 key 集合不变（CLAUDE.md B6：没见到新 key 不许 ✅）。

#### E1.1-F：三方 ckpt 立锚 + 文档同步

- [ ] **F1** ckpt 盘点：inceptio 确认 baseline（`v3_base_scratch30k_lam01` ckpt_30000）与 aniso20 路径；`ssh a800-x2` 找 B3 ckpt_30000（A800 不稳，**丢则降级两方立锚**并在 Done Log 声明）。B3 找到则 rsync 至 inceptio。
- [ ] **F2** 三次 eval（同命令模板、白天错峰）：`--novel-view --load-lane-masks`（+ E1.2/E1.4 完成后补跑增量指标）。结果表**标注配方来源**（A800 lidar-on vs inceptio depth-off 分组呈现，不混比）。
- [ ] **F3** 顺答 P3.3 遗留问题：B3（aniso 8→30）3m/6m lane 指标是否相对 baseline 放大退化 → 一句话结论。
- [ ] **F4** 文档同步（§8 模板）→ commit。

### 5.2 E1.2 — NTA-IoU 接入 + novel 档联动

**Files:** Create `threedgrut/model/nta_iou.py`、`threedgrut/model/vehicle_detector.py`、`threedgrut/tests/test_nta_iou.py`；Modify `threedgrut/render.py`、`render.py`、`requirements.txt`。

- [ ] **N1** 按 [`docs/superpowers/plans/2026-06-10-nta-iou-eval-metric.md`](docs/superpowers/plans/2026-06-10-nta-iou-eval-metric.md) Task 0–4 执行（Task 0 manifest 勘探在 inceptio 跑），带 4 处实现期修正：
  1. 测试路径 `tests/` → `threedgrut/tests/`（仓库唯一测试目录）；
  2. plan 里 `device=str(self.device)` → 固定 `"cuda"`（Renderer 无 self.device）；
  3. `intrinsics_pinhole_K` key 不存在 → 统一 `K=None, ftheta_params=ftheta_params`（本 clip 全 FTheta）；
  4. `requirements.txt` 追加 `ultralytics>=8.2.0`（yolov8m.pt 首跑自动下载，inceptio 走 mihomo 代理）。
- [ ] **N2** 增量 subtask（插原 plan Task 4/5 之间）——novel 档 NTA-IoU：novel 循环内 `T_w2c_novel = torch.linalg.inv(nT[0])`（GT cuboid 留世界系 GT 位姿，timestamp 不变 → `active` tracks 复用锚帧）→ `compute_frame_nta_iou(pred_novel[0], active, detector, K=None, ftheta_params=..., T_w2c=T_w2c_novel, H=H_, W=W_)` → 聚合 `mean_novel_nta_iou_<mode>` + `novel_nta_iou_n_frames_<mode>`。
- [ ] **N3** Mac 测试：合成 cuboid + fake detector，断言相机横移 3m 后 GT box 移动方向/量级正确、IoU 匹配逻辑不变。`pytest threedgrut/tests/test_nta_iou.py -v` PASS。
- [ ] **N4** inceptio smoke：`mean_nta_iou` ∈ (0,1] 且 `mean_novel_nta_iou_lateral_3m ≤ mean_nta_iou`（外推更难，方向 sanity）→ commit + 文档同步（原 plan Task 5 改为同步 v4_plan.md）。

### 5.3 E1.3 — held-out camera 真 GT 外推协议

**held-out 选择**：`camera_cross_left_120fov` 单台（侧视 120°，最贴近 lateral 外推轴；保留 cross_right 维持训练侧向覆盖）。训练 4-cam = front_wide + cross_right + rear_left + rear_right。

**Files:** Modify `render.py`（顶层 CLI）、`threedgrut/render.py`（from_checkpoint conf 注入）；Create `threedgrut/tests/test_render_dataset_cameras.py`。

- [ ] **H1**（TDD，10 行级）`--dataset-cameras` 开关：现有 `--eval-cameras` 只是 batch 级过滤救不了（4-cam ckpt 嵌入 conf 里没有 cross_left，`make_test` 走 `config.dataset.camera_ids`，datasets/__init__.py:248）。在 `from_checkpoint` 沿 V3-E4 模式注入 `conf["dataset"]["camera_ids"] = list(dataset_cameras)`。
  **exposure 陷阱（关键正确性）**：BilateralGrid 按训练 camera_idx 索引（threedgrut/render.py:505-517），相机集变更后索引错套 → `--dataset-cameras` 激活时**强制 `exposure_model=None`** + warning；held-out 口径以 `cc_psnr_masked` / `cc_lpips_masked`（逐帧 affine 自校）为主指标，文档写死。
  测试：mock conf dict 断言注入与 exposure 关断逻辑（不跑真渲染）。`pytest threedgrut/tests/test_render_dataset_cameras.py -v` PASS → commit。
- [ ] **H2** 夜间档训练（**E1.1-A/B 完成当晚即可发**，不等 eval 工具）：

```bash
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH \
 && export CUDA_VISIBLE_DEVICES=0 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
 && cd ~/repo/3dgrut2-wt/e1 \
 && nohup python train.py --config-name apps/ncore_3dgut_mcmc_multilayer \
    n_iterations=30000 num_workers=10 \
    trainer.use_lidar_depth=false trainer.use_depth_prior=false dataset.load_lidar_depth_map=false \
    "dataset.camera_ids=[camera_front_wide_120fov,camera_cross_right_120fov,camera_rear_left_70fov,camera_rear_right_70fov]" \
    path=~/work/data/9ae151dc/pai_9ae151dc-e87b-41a7-8e85-71772f9603d7.json \
    trainer.sky_backend=mlp out_dir=~/work/output experiment_name=e13_heldout_xleft_30k \
    > /tmp/e13_train.log 2>&1 & echo PID $!'
```

（depth-off + nw=10 铁律；具体 override key 路径以 multilayer yaml 现场为准；proven inline nohup 模式；~7h。）
- [ ] **H3** 三件套 eval（训完白天跑，R-v4.8 口径纪律：三个数只在本协议内部比，gap 表单列）：
  1. **held-out 主测**：`python render.py --checkpoint <e13 ckpt_30000> --out-dir ~/work/e13/heldout --dataset-cameras camera_cross_left_120fov`（per-class 全套 + NTA-IoU；lane 产物若 front-only 则 cross_left 无 lane 指标，预期并记录；eval 集 = 该相机 val 帧 ~74）；
  2. **守护测**：同 ckpt 无覆盖（4 训练相机标准 eval）→ 4-cam 训练相对 5-cam baseline 公共相机成本；
  3. **上界参照**：5-cam baseline ckpt `--dataset-cameras camera_cross_left_120fov`（它见过该相机）→ **gap = 上界 − held-out = 真外推差距**。
- [ ] **H4** 立锚入档 + 文档同步 → commit。

### 5.4 E1.4 — FID/KID 接入

**Files:** Modify `threedgrut/render.py`、`render.py`（`--novel-fid` 开关，默认 off 保字节等价）、`requirements.txt`（+`torch-fidelity`）；Create `threedgrut/tests/test_novel_fid.py`。

- [ ] **K1**（TDD）测试：`_kid_subset_size(n) = max(10, min(50, n//2))` 纯函数 + uint8 转换形状/值域 + torchmetrics 实例可构造（不跑 Inception 前向）。FAIL → 实现 → PASS。
- [ ] **K2** 实现（accumulate-then-finalize，与 per_class 聚合同级）：每 novel mode 一对 `FrechetInceptionDistance(feature=2048)` + `KernelInceptionDistance(subset_size=自适应)`；另一对给 interpolated（`mean_render_fid/kid`，对齐 E0.2 原轨迹 FID 7.37 口径概念）。循环内 `update((img*255).to(torch.uint8).permute(0,3,1,2), real=True/False)`；finalize 写 `mean_novel_fid_<mode>` / `mean_novel_kid_<mode>` / `mean_novel_kid_std_<mode>` / `fid_n_real` / `fid_n_fake_<mode>`。**KID 为主**（val ~74 帧/相机）。
- [ ] **K3** Inception 权重离线预案：inceptio mihomo 代理首跑下载，或 Mac 下好 scp 进 `~/.cache/torch/hub/checkpoints/`。
- [ ] **K4** inceptio smoke：`--novel-view --novel-fid` → 新 key 齐 + `mean_novel_fid_lateral_6m > mean_novel_fid_lateral_3m`（方向 sanity，对齐 E0.2 官方 57→92 单调性）→ commit + 文档同步。
- [ ] **K5** 文档红线写入 plan 卡：FID/KID 不得单独作修复收益判据（E0.2 教训），必须与 lane/NTA/真 GT 同向（R-v4.5）。

### 5.5 E0.4 — 同 clip 双向对照锚

**口径统一原则：指标代码全在项目侧；NuRec 只产帧。**

**Files:** Create `scripts/dump_test_split_manifest.py`、`scripts/eval_frames_dir.py`、`threedgrut/tests/test_eval_frames_dir.py`（`scripts/eval_difix_pngs.py` 与 DiFix 强耦合，不动）。

- [ ] **O1** `scripts/dump_test_split_manifest.py`：加载 ckpt conf → `make_test` → 输出 JSON（每条 `{iteration, camera_id, frame_idx, timestamp, c2w_start, c2w_end, HxW}` + 头部 split 参数）。验收：条目数 = Σ 各相机 val 帧（~370）。
- [ ] **O2** `scripts/eval_frames_dir.py`（= render_all 的 eval loop 去掉 model）：`--checkpoint` 只取 conf 建 test dataset（不 build 模型），pred 帧从 `--frames-dir` 读：

```
python scripts/eval_frames_dir.py --checkpoint <multilayer ckpt> \
  --frames-dir <dir> [--frames-map map.json] \
  --mode interpolated|lateral_3m|lateral_6m \
  --lane --nta-iou --kid --output metrics_nurec_<mode>.json
```

`interpolated`：PSNR/SSIM/LPIPS（masked）+ per-class + lane + NTA + FID/KID，key 与 render.py 完全一致；`lateral_*`：对每条位姿 `perturb_c2w` 得 novel 位姿 → plane warp lane 指标（height_field 由 ckpt road 层建或 `--height-field` 注入）+ NTA（`T_w2c=inv(perturbed)`）+ KID/FID；无像素 GT 指标一律不算。
  测试：帧映射解析 + 「interpolated 喂 GT 自身 → PSNR=inf / LPIPS≈0」恒等性（合成小图）。
- [ ] **O3** NuRec 侧出帧（inceptio nre 容器，`last.usdz` = `~/work/nurec_e0/train_out/PVG7YYV72YKPLumogi7F7U/artifacts/last.usdz`）：
  1. **轴向 sanity（1 帧级，先做）**：nre render rig offset +3m（E0.2 方法）渲 front_wide 第 0 val 帧，与项目侧 `perturb_c2w(c2w,"lateral_3m")` 位姿差比对（平移差仅在相机右轴 ±3m，容差 <0.05m）；不符 → fallback `export-custom-rig-trajectory` 按 manifest 位姿逐帧渲染；
  2. 产帧三档（原轨迹全 5 相机 + front_wide ±3m/6m），按 manifest 挑 val 帧；自训 USDZ 遇 nrend fail 走 `--no-enable-nrend`（E0.2 结论）；
  3. 跑 O2 三个 mode → `metrics_nurec_*.json`。
- [ ] **O4** multilayer 侧数字直接复用 E1.1/E1.2/E1.4 立锚产物（同 split 同协议）；汇 gap 表 NuRec 列 + 判别段（与大g目视终评「官方纯表示侧赢下外推」互证：lane@3m/6m、NTA@3m 是否量化领先）→ 文档同步 + commit。注明「项目口径，与 E0.3 官方口径 30.30 不混比」。

### 5.6 E1.5 — gap 表回填与重排（纯文档）

- [ ] **G1** 收数汇总：三方 ckpt × 6 档 + held-out 三件套 + NuRec 双列 → v4_plan.md §1.3 gap 表回填（无"待测"残留）。
- [ ] **G2** 写判别结论：差距主因 = 表示/配方/修复器哪层；确认或推翻 E0 初判「表示侧为主 → E3 ≥ E2」；E2/E3 顺序拍板句 + §1.2 gate 改写。
- [ ] **G3** §1.1 看板移列 + §5 Done Log + R-v4.8 口径备注落表 → commit（`docs(plan):`）。

### 5.7 E0.6 — 官方链 actor 编辑体验（run-book，零项目代码）

**前置（一次性）**
- [ ] **R1** 资产上传：`scp -r /Users/etendue/repo/asset-harvester-verify/verify_assets/bundle/ inceptio:~/work/nurec_e0/assets/bundle/`（3 车 + 3 人 PLY + metadata.yaml，~99k 高斯/个，asset-harvester 产物即官方插入格式）；确认 Harmonizer 镜像 `harmonizer-cosmos-env` 与 viewer patch `~/work/nurec_e0/patches/av_patched.py` 在位。

**操作序列（inceptio，与训练错峰；serve-grpc 用完即收）**
- [ ] **R2** 起服（编辑模式，自训 USDZ 必加 no-nrend）：serve-grpc `--enable-editing-actors --no-enable-nrend`，挂 `last.usdz`（具体脚本/参数面开工时按 nre skill 文档现查，记录实际命令入清单）。
- [ ] **R3** `export-external-assets` → `edit-assets.json`；记录可编辑 actor id 列表与 schema 字段面（参照容器内 `references/asset-editing.md`）。
- [ ] **R4** 四档渲染（同一段 val 帧）：`frames_base/`（无编辑）→ **场景 A 删一辆**（标记移除）`frames_del/` → **场景 B 插一辆收割车**（追加 asset 条目引用 `bundle/<car>/`，位姿放空闲车道）`frames_ins/` → **场景 C 替换**（A+B，收割车套被删 actor 轨迹）`frames_rep/`，均经 `render-grpc --edit-assets`。
- [ ] **R5** Harmonizer 协调：每档时间模式 `--entrypoint python ... inference_pix2pix_turbo_harmonizer.py --timestep 250 --resolution 1024 --use_sched`（**每 run 独立输出目录**——时间模式回读历史 4 帧）；抽 3 帧 `--nontemporal` 单帧对照。
- [ ] **R6** 量化 + 目视：FID/KID 双口径（编辑档 vs `frames_base` 残留口径；vs GT 帧分布）；插入车可选 `scripts/eval_frames_dir.py --nta-iou`（GT box = 插入位姿手工 cuboid）验证「被检出且框齐」。**FID 解读带 E0.2 教训注**（编辑残留是局部伪影，FID 钝感 → 目视为主、FID 为辅）。
- [ ] **R7** 交付：`docs/superpowers/specs/2026-06-12-e06-official-actor-editing-capability.md`，能力/限制清单模板：

| 维度 | 观察项 |
|---|---|
| 删除 | 路面/人行道是否出洞、阴影残影、被遮挡区如何补全 |
| 插入 | 收割资产兼容性（PLY+metadata 直接吃下？）、阴影来源（无/烘焙/PBR）、光照失配程度、位姿参数化（单 pose vs 轨迹） |
| 替换 | 新旧尺寸不一致的露馅模式 |
| 协调 | Harmonizer 前后 FID/KID 双口径 + 目视（修掉什么/引入什么）、时间模式闪烁 |
| 工程 | gRPC 接口面（能否逐帧改位姿）、nrend/torch 稳定性、时延、与 P1.4 spiky/悬浮对照 → 结论喂 E2.5 |

- [ ] **R8** 对照帧存档 inceptio + 文档同步（v4_plan.md E0.6 行 ✅ + Done Log）→ commit。

## 6. metrics.json 新 key 总表（验收 checklist，CLAUDE.md B6）

| 卡 | 新 key |
|---|---|
| E1.1 | `mean_novel_lpips_lateral_3m/_6m`（per-mode 循环自动）；`mean_novel_lpips_avg` **不变**；`mean_novel_lpips_avg6`；`mean_novel_lane_{grad_corr,band_lpips,band_psnr}_<mode>`；`novel_lane_n_records_<mode>`；`novel_lane_warp_valid_ratio_<mode>` |
| E1.2 | `mean_nta_iou` / `nta_iou_n_frames`；`mean_novel_nta_iou_<mode>` / `novel_nta_iou_n_frames_<mode>` |
| E1.3 | 无新 key（口径靠 out_dir 命名 + 文档写死） |
| E1.4 | `mean_render_{fid,kid,kid_std}`；`mean_novel_{fid,kid,kid_std}_<mode>`；`fid_n_real` / `fid_n_fake_<mode>` |

## 7. 风险与 fallback

| 风险 | fallback |
|---|---|
| B3 ckpt（A800 训）已丢 | 两方立锚（baseline vs aniso20），降级声明入 Done Log |
| 高度场不可得（非 layered ckpt / 离线） | `--ground-z` 常数平面兜底；再不行跳过 lane novel 指标（软失败） |
| lane GT itar 在 E0.3 时被移出、原件丢失 | `scripts/gen_lane_sseg.py` 重产（mask2former 权重走 mihomo 代理） |
| torch-fidelity / Inception / yolov8m / mask2former 权重离线下不动 | mihomo 代理；或 Mac 下好 scp 进 `~/.cache/torch/hub/checkpoints/` |
| nre rig offset 轴向 ≠ 相机右轴 | E0.4-O3 第 1 步 1 帧 sanity 先验证；不符走 custom-rig-trajectory |
| 自训 USDZ nrend 间歇 fail | `--no-enable-nrend`（E0.2 已验证） |
| PR #24 后续 merge/关闭与移植冲突 | path 限定移植 + commit message 写来源区间；R9 决议后一次 reconcile |
| held-out exposure 错套 | `--dataset-cameras` 强制 exposure off + cc_* 为主指标（H1 内建） |
| KID subset_size > 样本数 | 自适应 `max(10, min(50, n//2))` |
| 4090 单卡训练/eval/serve-grpc 抢卡 | 排程：训练夜间档、eval 白天、serve-grpc 即用即收 |

## 8. 文档同步（每卡最后一步，同 commit）

1. **v4_plan.md**：§1.1 看板移 Done（卡片文案带实测数，**括号全角（）**，提交前跑 CLAUDE.md awk 自查）；§1.2 任务行 ⬜→✅ + 实测锚；§1.3 计数 + gap 表回填；§5 Done Log（日期 + commit hash + 实测数）。
2. **v2_architecture.md**：§6 登记新模块（`plane_warp.py` / `nta_iou.py` / `vehicle_detector.py` / `eval_frames_dir.py` / `dump_test_split_manifest.py`）；§7 不变量加三条——「`mean_novel_lpips_avg` 永远只聚合 LEGACY 4 档」「lane novel 指标 = warped 伪 GT 口径，仅同 warp 版本内可比」「NTA-IoU 投影只走 `dynamic_mask.project_cuboids_to_mask`，不走 viser 投影（BUG-1 隔离）」。
3. commit message 格式沿项目惯例：`feat(E1.x): ...` + `docs(plan)/docs(arch)` 段。

## 9. 端到端验证

- **Mac**：`pytest threedgrut/tests/ -v -k "novel or plane_warp or nta or fid or frames_dir or per_class or aux_discover or dataset_cameras"` 全绿。
- **inceptio 回归**：baseline ckpt `--novel-view --load-lane-masks --novel-fid` 一次跑齐 → ① §6 全部新 key 在 metrics.json 出现；② `mean_novel_lpips_avg` 与历史差 ≤0.001；③ 旧 key 集合不变（interpolated 守护零回归）。
- **方向 sanity**：`lane grad_corr@6m < @3m < @1m`、`mean_novel_nta_iou_lateral_3m ≤ mean_nta_iou`、`FID@6m > FID@3m`。
- **协议完整**：E1.3 三件套三个数齐；E0.4 gap 表 NuRec/multilayer 双列齐 + 判别结论句；E0.6 清单五维度填满 + 帧档案路径可访问。
- **伪完成排查**（CLAUDE.md C9）：训练 exit 0 ≠ ✅；每卡以 metrics 数字 + commit hash 双证据入 Done Log 为准。
