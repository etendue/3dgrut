# Phase 3 车道线 GT — 方案 A（车道线分割 mask → per-class lane 指标）设计草稿

> **状态**：设计草稿（brainstorming 产物），供新 session 起 plan。
> **定位**：这是 **Phase 3 的「测量门」**，对标 Phase 0——先把"车道线质量"测出来，**才能**让后续 P3.1（road 定向加密/平面 grid）、P3.2（遮挡式 bg）的改善可证。本方案**只做 GT + 指标**，不含 P3.1/P3.2 的实际改动。
> **决策依据**：2026-06-09 session 评估 + Explore 测绘（见下"现状锚点"）。

---

## 0. 为什么需要这个门（动机）

旧坑复述：v3 的核心教训是「优化错了轴 / 指标测不准」。Phase 3 的目标是"车道线锐度↑"，但当前**根本没有车道线专属信号**：

- sseg 是 `nre-tools ncore-aux-data` 出的 **mask2former Cityscapes-19**，只有 `road(0)`/`sidewalk(1)`，**无 lane-marking 类**（[`ncore_semantic.py`](../../../threedgrut/datasets/ncore_semantic.py)，注释还标注 palette 可能有偏移、未对账）。
- P0.3 的 `road_crop`（road+sidewalk 整片）**沥青主导**：PSNR 29.20 虚高，LPIPS 0.154 基本测的是沥青纹理，不是线条。
- 数据链路里**无 HD map / 车道矢量**；LiDAR 只取点/深度，**intensity 通道未加载**。

**结论**：不先补一个车道线专属指标，P3.1/P3.2 任何"改善"都不可证 → 必须先立这个门。

---

## 1. 方案 A 总览

**一句话**：用一个带「车道线 / road-marking」类的分割模型逐帧产出 lane mask，接进现成 [`per_class_eval.py`](../../../threedgrut/model/per_class_eval.py) 框架，得到 per-class lane 指标，并在现 baseline ckpt 上立锚（无新训练）。

数据流（复用 P0 全套基建）：
```
NCore RGB 帧
   → [A.1] lane-seg 模型  → 逐帧 lane mask（zarr/itar，与现 sseg 同款产物）
   → [A.2] datasetNcore val 分支加载 + per_class_eval.class_mask_from_sseg
   → [A.3] dilated-lane-band 指标（LPIPS + 辅助 PSNR/edge）
   → [A.4] render.py eval loop → metrics.json 新字段 mean_lane_*
   → 在 baseline ckpt 上跑一次 → 立 Phase 3 lane 锚点
```

---

## 2. 子决策与组件

### A.1 — lane mask 来源（核心子决策，2 条路）

| 路径 | 做法 | 优 | 劣 / 风险 |
|---|---|---|---|
| **A-seg-swap（首选）** | 换/加一个 palette 含 lane-marking 的分割模型：**Mapillary Vistas**（有 `Lane Marking - General` / crosswalk 等）或 BDD100K。两种落法：① 若 `nre-tools ncore-aux-data` 支持换模型/palette → 直接产 lane sseg；② 否则我们自己用 mask2former/segformer + Mapillary 权重在 NCore RGB 上离线跑。 | 最贴近现有 sseg 管线，**一个 mask 产物即喂进现成 per_class_eval**；与 person/road_crop 同款 | 线细（亚像素级）、域差（NCore 相机畸变 vs Mapillary 分布）、palette 需抽帧对账 |
| **A-lane-detector（备选）** | 跑专用车道线检测器（CLRNet / LaneATT / BEV lane-seg）→ 逐帧 lane mask | 对细线更专、可能更干净 | 又一个模型要集成、域差、输出是 line instance 需转 mask |

> **建议**：先试 **A-seg-swap（Mapillary 类）**。若 seg lane mask 太噪 → 退 A-lane-detector。
> **待你/新 session 确认的关键事实**：`nre-tools` 能否换 Mapillary palette？（决定 A-seg-swap 是"白嫖现管线"还是"自己跑模型"）

### A.2 — 接进评测器（复用 P0 基建，避免双路径坑）

- [`ncore_semantic.py`](../../../threedgrut/datasets/ncore_semantic.py)：加 `LANE_CLASS_IDS`（Mapillary lane-marking 的类 id，对账后填）。
- [`per_class_eval.py`](../../../threedgrut/model/per_class_eval.py)：`class_mask_from_sseg` 支持 lane 类，产 lane mask（与 person/road_crop 同路径）。
- [`datasetNcore.py`](../../../threedgrut/datasets/datasetNcore.py)：**val/test 分支**也要加载 lane sseg（**P0.4 同款双路径教训**：sseg 一度只在 train 分支加载，val 路径漏掉 → metrics.json 缺字段；lane sseg 必须在 eval 真路径透传到 `image_infos`，render-res NEAREST resize）。
- [`render.py`](../../../threedgrut/render.py) `render_all()`：逐帧累积 + metrics.json 新字段 `mean_lane_{psnr,lpips}` / `*_n_records` / `*_total_pixels`。

### A.3 — 指标设计（关键：细 mask 会让 naive PSNR/LPIPS 失效）

P0.3/P0.4 已坐实：**小/细 mask 的 LPIPS 受面积主导（≈0 假象）、PSNR 被背景填充主导**。所以 lane 指标不能照搬：

1. **dilated lane band**：把 lane mask 膨胀 N 像素（线 + 紧邻沥青），让区域大到 LPIPS 有意义、且聚焦"线锐不锐 / 位置对不对"。
2. **edge / 锐度感知指标**（更直指目标）：rendered vs GT 的梯度幅值相关，或 lane edge 的 edge-IoU——直接量"线是否清晰且在对位置"。
3. **同时报多个**：dilated-region LPIPS（锐度代理）+ lane-PSNR（存在/对比度）+（可选）edge-IoU；**先在 baseline 上测全部候选，挑有信号/有方差的那个**当 Phase 3 主指标。

> ⚠️ **A.3 本身要先在 baseline ckpt 上做经验验证**：若所有候选指标在 baseline 上都没可用信号/方差 → 红旗，回头改指标（别急着投 P3.1/P3.2）。

### A.4 — 测量门语义（对标 Phase 0）

- **不训练**：方案 A 落地后，先在现 baseline ckpt（`v3_base_scratch30k_lam01`）上跑一遍 → 立 Phase 3 lane 锚点，写进 [`v3_plan_revised.md`](../../../v3_plan_revised.md) §1.3 per-class gap 表 + §6 Done Log。
- 守护线零回归核对（cc 25.79 / novel 0.5987 / lidar 22.69 不变，纯增量）。

---

## 3. 风险登记

| ID | 风险 | 缓解 |
|---|---|---|
| L1 | Mapillary palette 与实际输出偏移 | 抽 1 帧 lane sseg 读 unique values 对账（同 T3.1.b sseg 教训） |
| L2 | 域差：NCore 畸变/外围相机上 lane seg 噪 | lane 指标限**前视相机 / 中心 crop**（lane 最清处）；confidence 阈值 |
| L3 | 细 mask 指标失效 | A.3 dilated band + edge-IoU；baseline 上先验证有信号 |
| L4 | `nre-tools` 不支持换 palette | 退路：自己离线跑 mask2former+Mapillary，或 A-lane-detector |
| L5 | 图像空间近大远小（远处 lane 几像素） | 可选 BEV 变体（用 [`road_region.py`](../../../threedgrut/model/road_region.py) 高度场把 road crop 正交投影到鸟瞰），lane 分辨率均匀——列为 stretch |

---

## 4. 现状锚点（Explore 测绘，2026-06-09）

| 维度 | 现状 | 文件 |
|---|---|---|
| road 层 | layer_id=1，200k cap，薄盘先验 (0.1,0.1,0.001)，Z 锁扰动，LiDAR(类0/1)→BEV 5cm 网格 KNN init | `layers/registry.py`、`layers/road_init.py` |
| BEV 基建 | 高度场 `build_road_height_field` / `query_ground_z` / bg-road opacity penalty（**可复用** A.3 BEV 变体 / P3.x） | `model/road_region.py` |
| 评测框架 | `per_class_eval.py`（compute_per_class_metrics / compute_lpips_in_mask GT-fill / class_mask_from_sseg）+ render.py eval loop（独立写 metrics.json） | `model/per_class_eval.py`、`render.py` |
| sseg | mask2former Cityscapes-19，无 lane 类 | `datasets/ncore_semantic.py` |

---

## 5. 范围边界（YAGNI）

- ✅ **本方案 = 仅测量门**：lane GT 来源（A.1）+ 评测接通（A.2）+ 指标设计（A.3）+ baseline 立锚（A.4）。
- ❌ **不含**：P3.1（road 定向加密 / 平面 feature grid）、P3.2（遮挡式 bg mask-loss）——那是这个门要去**测量**的改善，单独立项。
- ❌ **不含**：换 sseg 模型后对**其他类**（person/road_crop）指标的回填/重测（除非顺手，避免 scope 膨胀）。

---

## 6. 给新 session 做 plan 的开口问题

1. `nre-tools ncore-aux-data` 能否换 Mapillary/BDD palette？（定 A.1 走 swap 还是自跑模型）
2. lane 指标主指标定哪个（dilated-LPIPS / lane-PSNR / edge-IoU）——还是 baseline 实测后再定？
3. 图像空间 vs BEV 变体——v3 先只做图像空间（便宜），BEV 留 stretch？
4. 跑 baseline 立锚用哪台机（A800 / inceptio），约 ~3 min eval（同 P0 实测体量）。
