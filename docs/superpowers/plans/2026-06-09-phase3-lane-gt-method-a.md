# Phase 3 车道线 GT — 方案 A（lane sseg → per-class lane 指标）执行 plan

> **For agentic workers:** REQUIRED SUB-SKILL: 用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐 task 执行本 plan。步骤用 checkbox（`- [ ]`）跟踪。

> **任务来源**：[`docs/superpowers/specs/2026-06-09-phase3-lane-gt-method-a-design.md`](../specs/2026-06-09-phase3-lane-gt-method-a-design.md)（brainstorming 产物）。对标 Phase 0：先把"车道线质量"测出来，**才能**让后续 P3.1（road 定向加密）、P3.2（遮挡式 bg）的改善可证。
>
> **本 plan 性质**：建 **lane GT 来源 + per-class lane 指标**，**纯 eval、无新训练**。Mac 侧纯张量代码 + 单测可全部离线完成（Task 1–6）；只有「生成 lane 产物 + 在 baseline ckpt 上跑出真实数字」需要 GPU（Task 7–8，A800/inceptio）。
>
> **决策依据**：2026-06-09 session 评估 + 3 个 Explore + 1 个 Plan agent 测绘（全部带 file:line 出处，见 § 1）。

**Goal:** 给 NCore eval 链路加一个「车道线专属」per-class 指标（dilated-band LPIPS + lane-PSNR + 梯度锐度），在现 baseline ckpt 上立 Phase 3 lane 锚点，且对现有 metrics.json 零回归（纯增量）。

**Architecture:** lane 类来自**独立的分割产物**（Mapillary palette 含 lane-marking），不动现有 Cityscapes `semantic_sseg`（避免扰动 person/road_crop，spec 明令禁止换 palette）。lane 走自己的产物文件（`*.aux.lane.zarr.itar`）→ 自己的 reader → 自己的 image_infos key（`semantic_lane_sseg`）→ 自己的 eval 调用，最终汇进现成 metrics.json 机制。细线 mask 用 **torch-pure 膨胀（max_pool2d）**变 dilated band，让 LPIPS 有意义。

**Tech Stack:** PyTorch（纯张量，`F.max_pool2d` / `F.conv2d` 做膨胀+Sobel）、torchmetrics LPIPS（依赖注入，不直接 import）、NCore aux zarr.itar（复用现成 `SsegAuxReader`）、Hydra config、pytest（Mac CPU 单测）。

---

## 0. 范围与非目标（YAGNI）

| 做 | 不做 |
|---|---|
| A.1 lane mask 来源（独立 sseg 产物 + 类 id 对账） | P3.1（road 定向加密 / 平面 feature grid）——本门要去测量的改善 |
| A.2 接进 `per_class_eval` + datasetNcore val/test + render.py | P3.2（遮挡式 bg mask-loss）——同上 |
| A.3 dilated-band LPIPS + lane-PSNR + 梯度锐度（先全报，baseline 后挑主指标） | 换 sseg 模型后回填 person/road_crop 其他类指标 |
| A.4 baseline ckpt 立锚（无训练）+ 守护线零回归核对 | 任何新训练 / 重训 baseline |
| 回填 `v3_plan_revised.md` §1.3 + §6 + `v2_architecture.md` | trainer 训练侧 lane（lane 无 loss，纯 eval） |
| 图像空间指标（便宜） | BEV 正交投影变体（spec L81 列为 stretch，本 plan 不做） |
| edge-IoU → 用**阈值无关的梯度幅值相关**替代（见 Task 2 决策） | edge-IoU（需标定阈值、细线脆弱，P0 已踩坑） |

---

## 1. Documentation Discovery 结论（Allowed APIs，全部实读码）

> 实现时**只用此清单内的 API**；不在表内 = 先读码确认，不要发明。

### A. 现成可复用的纯张量评测函数（`threedgrut/model/per_class_eval.py`，纯 torch、Mac 可测）

| API | 位置 | 签名 / 用途 |
|---|---|---|
| `class_mask_from_sseg(sseg, ids) -> bool[H,W]` | [`per_class_eval.py:55`](../../../threedgrut/model/per_class_eval.py) | `[H,W]` 类 id 图 → 命中 ids 的 bool mask（纯集合，无形态学） |
| `compute_lpips_in_mask(rgb_pred, rgb_gt, mask, lpips_fn, min_pixels=50) -> Optional[float]` | [`per_class_eval.py:64`](../../../threedgrut/model/per_class_eval.py) | GT-fill：`pred*m + gt*(1-m)`，mask 外感知距离→0；像素<min 返回 None |
| `compute_per_class_metrics(...) -> {name:{psnr,lpips,n_pixels}}` | [`per_class_eval.py:92`](../../../threedgrut/model/per_class_eval.py) | 现有 person/rider/bicycle/road_crop 入口 |
| `compute_psnr_in_mask(rgb_pred, rgb_gt, mask, min_pixels=50) -> Optional[float]` | [`class_psnr.py:33`](../../../threedgrut/model/class_psnr.py) | mask 内 PSNR；lane-PSNR 直接复用 |
| 常量 `DEFAULT_ACTOR_CLASS_SPECS` / `ROAD_CLASS_IDS=(0,1)` | [`per_class_eval.py:42-49`](../../../threedgrut/model/per_class_eval.py) | 类 id 本地声明（**不 import** ncore_semantic，保 Mac 测纯净，见模块 docstring L25-31） |

### B. sseg 磁盘读取（`threedgrut/datasets/aux_readers.py`）

| 事实 | 出处 |
|---|---|
| `SsegAuxReader(itar).read(camera_id, ts_us) -> [H,W] uint8`；内部读 `/aux/semantic_segmentation/<cam>/<ts>`（PNG bytes，0-D zarr） | [`aux_readers.py:59-119`](../../../threedgrut/datasets/aux_readers.py) |
| `reader.class_palette -> list`（读 group attrs `stuff_classes`，用于对账 lane id） | [`aux_readers.py:77-87`](../../../threedgrut/datasets/aux_readers.py) |
| `discover_aux_path(clip_dir, aux_type) -> Optional[Path]`（glob `*.aux.<aux_type>.zarr.itar`，**通用、已支持任意 aux_type 字符串**） | [`aux_readers.py:169-188`](../../../threedgrut/datasets/aux_readers.py) |
| ⚠️ **时间戳键 = END**（datasetNcore 实际用 END，aux_readers docstring 写 START 是旧注释；datasetNcore L933 注明 599/599 对上 END） | [`datasetNcore.py:930-942`](../../../threedgrut/datasets/datasetNcore.py) |

### C. sseg 在 datasetNcore 的加载（train + val/test 双路径）

| 事实 | 出处 |
|---|---|
| `__init__` flag `load_aux_masks: bool=False`（L105），3 个 reader 字段 + `_aux_readers_initialized`（L178-180） | [`datasetNcore.py:105,170-180`](../../../threedgrut/datasets/datasetNcore.py) |
| `_ensure_aux_readers()`：`discover_aux_path(clip_dir,"sseg")` → `SsegAuxReader`；**sseg 无文件 = 硬 `FileNotFoundError`** | [`datasetNcore.py:1260-1287`](../../../threedgrut/datasets/datasetNcore.py) |
| **train 分支** sseg 加载（END ts + NEAREST resize → `semantic_sseg`） | [`datasetNcore.py:935-960`](../../../threedgrut/datasets/datasetNcore.py) |
| **val/test 分支** sseg 加载（P0.4 双路径教训注释明示：曾只在 train、漏 val → metrics 缺字段） | [`datasetNcore.py:1108-1136`](../../../threedgrut/datasets/datasetNcore.py) |
| `image_infos` 透传：`if "sky_mask" in batch: batch_dict["image_infos"]={...}; if "semantic_sseg" in batch: 转 GPU` | [`datasetNcore.py:1600-1609`](../../../threedgrut/datasets/datasetNcore.py) |
| 3 个构造点都从 config 读 flag：`load_aux_masks=config.dataset.get("load_aux_masks",False)`（train L143 / val L183 / test L264） | [`datasets/__init__.py:143,183,264`](../../../threedgrut/datasets/__init__.py) |

### D. render.py eval loop（`render_all`，**唯一** per-class 指标路径）

| 事实 | 出处 |
|---|---|
| `per_class_eval_specs = {**DEFAULT_ACTOR_CLASS_SPECS, "road_crop": ROAD_CLASS_IDS}` + 3 个累加 dict | [`render.py:425-428`](../../../threedgrut/render.py) |
| `_cam_id = getattr(gpu_batch,"camera_id",None)`（L496，eval site 在域内） | [`render.py:496`](../../../threedgrut/render.py) |
| per-class 调用点：读 `image_infos["semantic_sseg"]` → `compute_per_class_metrics(...)` → 累加 | [`render.py:743-755`](../../../threedgrut/render.py) |
| metrics.json 写入：`if per_class_npix:` for 每类写 `mean_{name}_{psnr,lpips}` / `{name}_n_records` / `{name}_total_pixels` | [`render.py:941-949`](../../../threedgrut/render.py) |
| conf 读取范式：`self.conf.render.get("key", default)` | [`render.py:99-101,373`](../../../threedgrut/render.py) |
| **trainer.py 不做 per-class**（单路径）→ lane 指标**只改 render.py，无需改 trainer**（项目 trainer/render 双路径坑此处不适用） | Explore 报告（grep `compute_per_class_metrics` 仅 render.py 命中） |

### E. 前视相机 id（lane 最清处，risk L2 用来限定）

| 事实 | 出处 |
|---|---|
| 前视相机逻辑 id = `"camera_front_wide_120fov"` | [`datasetNcore.py`](../../../threedgrut/datasets/datasetNcore.py) / [`aux_readers.py:99`](../../../threedgrut/datasets/aux_readers.py) / 多处测试 |

---

## 2. lane 产物契约（解耦 Mac 代码 与 GPU 产物生成）

> **关键解耦**：Task 1–6（Mac 纯张量代码 + datasetNcore/render 接线 + 单测）可**立即落地、CI 全绿**，不阻塞于"用 nre-tools 还是自跑模型"这个 open question——因为接口由下面这份契约钉死，`LANE_CLASS_IDS` 是唯一一个待 GPU 对账的常量。

**`*.aux.lane.zarr.itar` 契约（钉死，Task 7 产物必须满足）：**

- **文件名**：clip 目录下 `*.aux.lane.zarr.itar`，由 `discover_aux_path(clip_dir, "lane")` 发现（glob 已通用支持）。
- **内部布局**：与 `aux.sseg.zarr.itar` 字节同构，**复用内部 group 名 `aux/semantic_segmentation`** → `SsegAuxReader` 原样复用，**aux_readers.py 零改动**。即 `/aux/semantic_segmentation/<camera_id>/<timestamp_us>` 存 0-D `|S<n>` PNG bytes；per-camera `.zattrs` 带 `stuff_classes`（Mapillary palette）+ `resolution:[W,H]`。
- **每帧载荷**：`[H,W] uint8` 类 id 图，H/W 与该相机 sseg 同分辨率。
- **时间戳键**：**END** 时间戳（`frames_timestamps_us[idx, FrameTimepoint.END]`），与 datasetNcore 实际查询一致。
- **lane 类 id**：`LANE_CLASS_IDS` 占位 `(24,)`；**Task 7 在 GPU 上读一帧 `unique()` 对 `reader.class_palette` 对账后定值**，同 commit 更新 guard test（risk L1）。

> 若 Task 7 选 nre-tools 路且其内部 group 名非 `semantic_segmentation` → 退路是给 `SsegAuxReader` 加 `group_name="..."` 形参（默认不变，一行改动）；本 plan 默认自跑模型写 `semantic_segmentation` group，免改 aux_readers.py。

---

## 3. 文件结构 / 改动总览

| 文件 | 改动 | Task |
|---|---|---|
| `threedgrut/model/per_class_eval.py` | 加 `DEFAULT_LANE_BAND_PX` / `LANE_CLASS_IDS` / `dilate_mask` / `_luma` / `_grad_mag` / `_grad_mag_corr_in_mask` / `compute_lane_metrics` | 1,2,3 |
| `threedgrut/tests/test_per_class_eval.py` | 加 dilation / lane-metric / 锐度 / guard 单测 | 1,2,3 |
| `threedgrut/datasets/datasetNcore.py` | 加 `load_lane_masks` flag + `_lane_reader` + val/test 加载块 + image_infos 透传 | 4 |
| `threedgrut/tests/test_aux_discover_lane.py`（新建） | `discover_aux_path(clip_dir,"lane")` glob 单测（纯 pathlib，Mac 可测） | 4 |
| `threedgrut/datasets/__init__.py` | 3 个构造点加 `load_lane_masks=config.dataset.get(...)` | 5 |
| `threedgrut/render.py` | lane eval 调用（前视 gate）+ metrics.json 写入 | 5 |
| `configs/apps/ncore_3dgut_mcmc_multilayer.yaml` | 加 `load_lane_masks: false`（默认关，eval 时 CLI 开）+ 可选 `render.lane_band_px` | 5 |
| `scripts/gen_lane_sseg.py`（新建，GPU） | 自跑 mask2former/segformer + Mapillary → `*.aux.lane.zarr.itar`（仅自跑分支用） | 7 |
| `v3_plan_revised.md` / `v2_architecture.md` | 看板 + Done Log + 文件清单 + 不变量回填 | 9 |

---

## Task 1：torch-pure 膨胀工具 `dilate_mask`

**Files:**
- Modify: `threedgrut/model/per_class_eval.py`（顶部 import 区 + 常量区 + 新函数）
- Test: `threedgrut/tests/test_per_class_eval.py`

- [ ] **Step 1: 写失败的测试**（追加到 `test_per_class_eval.py` 末尾）

```python
# -----------------------------------------------------------------------------
# dilate_mask — torch-pure 膨胀（细 lane mask → dilated band）
# -----------------------------------------------------------------------------
from threedgrut.model.per_class_eval import dilate_mask  # noqa: E402


def test_dilate_mask_grows_by_radius():
    m = torch.zeros(11, 11, dtype=torch.bool)
    m[5, 5] = True
    d = dilate_mask(m, 2)  # 5x5 方形结构元
    assert d.dtype == torch.bool
    assert d.shape == (11, 11)
    assert int(d.sum().item()) == 25


def test_dilate_mask_radius_zero_is_identity():
    m = torch.zeros(8, 8, dtype=torch.bool)
    m[3, 4] = True
    d = dilate_mask(m, 0)
    assert d.dtype == torch.bool
    assert torch.equal(d, m)


def test_dilate_mask_clamps_at_border():
    m = torch.zeros(6, 6, dtype=torch.bool)
    m[0, 0] = True  # 角点，radius=2 只有界内 3x3=9 存活
    d = dilate_mask(m, 2)
    assert int(d.sum().item()) == 9
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /Users/etendue/repo/3dgrut2 && python -m pytest threedgrut/tests/test_per_class_eval.py -k dilate -v`
Expected: FAIL — `ImportError: cannot import name 'dilate_mask'`

- [ ] **Step 3: 写最小实现**

在 `per_class_eval.py` 顶部 import 区（L37 `import torch` 之后）加：

```python
import torch.nn.functional as F
```

在 `ROAD_CLASS_IDS`（L49）之后加常量：

```python
# Phase 3 lane band 半宽（px）。lane 条纹亚像素~几像素；N=8 → band ~17px 宽，
# 让 LPIPS 有足够空间上下文判断"条纹锐不锐 / 位置对不对"，又不被背景淹没。
# render.py 调用点可经 conf.render.lane_band_px 覆盖。
DEFAULT_LANE_BAND_PX: int = 8
```

在 `class_mask_from_sseg`（L61 之后）之前/之后加：

```python
def dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """``[H, W]`` bool/float mask → 方形结构元（边长 2*radius+1）膨胀后的 bool mask。

    纯 torch（``F.max_pool2d``）→ 本模块保持 scipy-free / Mac 可测。
    ``radius <= 0`` 原样返回（转 bool）。形状不变（padding=radius, stride=1）。
    """
    if radius <= 0:
        return mask.to(torch.bool)
    m = mask.to(torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    k = 2 * radius + 1
    dil = F.max_pool2d(m, kernel_size=k, stride=1, padding=radius)
    return dil[0, 0] > 0.5
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /Users/etendue/repo/3dgrut2 && python -m pytest threedgrut/tests/test_per_class_eval.py -k dilate -v`
Expected: PASS（3 个）

- [ ] **Step 5: commit**

```bash
git add threedgrut/model/per_class_eval.py threedgrut/tests/test_per_class_eval.py
git commit -m "feat(P3-lane): torch-pure dilate_mask + DEFAULT_LANE_BAND_PX (Task1)"
```

---

## Task 2：梯度锐度标量 `_grad_mag_corr_in_mask`（替代 edge-IoU）

> **决策（偏离 spec）**：spec A.3 列 edge-IoU **或** 梯度相关。本 plan 选**梯度幅值 Pearson 相关**——阈值无关、纯 torch、直接奖励"边缘出现在对的位置且强度对"。edge-IoU 需标定梯度阈值（一个还得 sweep 的自由参数）且对细条纹脆弱，正是 P0 已踩的坑，**不做**。

**Files:**
- Modify: `threedgrut/model/per_class_eval.py`
- Test: `threedgrut/tests/test_per_class_eval.py`

- [ ] **Step 1: 写失败的测试**

```python
# -----------------------------------------------------------------------------
# _grad_mag_corr_in_mask — 梯度锐度标量
# -----------------------------------------------------------------------------
from threedgrut.model.per_class_eval import _grad_mag_corr_in_mask  # noqa: E402


def test_grad_corr_identical_is_one():
    H = W = 32
    gt = torch.rand(H, W, 3)
    mask = torch.ones(H, W, dtype=torch.bool)
    c = _grad_mag_corr_in_mask(gt.clone(), gt, mask)
    assert c is not None
    assert math.isclose(c, 1.0, abs_tol=1e-4)


def test_grad_corr_flat_pred_returns_none():
    """pred 无边缘（常数）→ 梯度方差 0 → 相关无定义 → None。"""
    H = W = 32
    gt = torch.rand(H, W, 3)
    pred = torch.full((H, W, 3), 0.5)
    mask = torch.ones(H, W, dtype=torch.bool)
    assert _grad_mag_corr_in_mask(pred, gt, mask) is None


def test_grad_corr_too_few_pixels_returns_none():
    H = W = 32
    gt = torch.rand(H, W, 3)
    mask = torch.zeros(H, W, dtype=torch.bool)
    mask[0, :3] = True  # 3 < min_pixels(50)
    assert _grad_mag_corr_in_mask(gt.clone(), gt, mask) is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /Users/etendue/repo/3dgrut2 && python -m pytest threedgrut/tests/test_per_class_eval.py -k grad_corr -v`
Expected: FAIL — `ImportError: cannot import name '_grad_mag_corr_in_mask'`

- [ ] **Step 3: 写最小实现**（加在 `compute_lpips_in_mask` 之后）

```python
def _luma(rgb: torch.Tensor) -> torch.Tensor:
    """``[H, W, 3]`` → ``[H, W]`` 亮度（Rec.601）。"""
    w = torch.tensor([0.299, 0.587, 0.114], dtype=rgb.dtype, device=rgb.device)
    return (rgb * w).sum(-1)


def _grad_mag(img2d: torch.Tensor) -> torch.Tensor:
    """``[H, W]`` → ``[H, W]`` Sobel 梯度幅值。"""
    x = img2d.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                      dtype=img2d.dtype, device=img2d.device)
    ky = kx.t().contiguous()
    gx = F.conv2d(x, kx.view(1, 1, 3, 3), padding=1)
    gy = F.conv2d(x, ky.view(1, 1, 3, 3), padding=1)
    return torch.sqrt(gx * gx + gy * gy)[0, 0]


def _grad_mag_corr_in_mask(
    rgb_pred: torch.Tensor,   # [H, W, 3] in [0, 1]
    rgb_gt: torch.Tensor,     # [H, W, 3]
    mask: torch.Tensor,       # [H, W] bool/float
    min_pixels: int = 50,
) -> Optional[float]:
    """mask 内 Sobel 梯度幅值（亮度）的 Pearson 相关。

    直指"线是否锐且在对位置"：阈值无关。像素 < min_pixels 或任一侧梯度无方差
    → None。
    """
    m = mask.to(torch.bool)
    if int(m.sum().item()) < min_pixels:
        return None
    gp = _grad_mag(_luma(rgb_pred.clip(0, 1)))[m]
    gg = _grad_mag(_luma(rgb_gt))[m]
    gp = gp - gp.mean()
    gg = gg - gg.mean()
    denom = gp.norm() * gg.norm()
    if float(denom.item()) <= 0.0:
        return None
    return float((gp @ gg / denom).item())
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /Users/etendue/repo/3dgrut2 && python -m pytest threedgrut/tests/test_per_class_eval.py -k grad_corr -v`
Expected: PASS（3 个）

- [ ] **Step 5: commit**

```bash
git add threedgrut/model/per_class_eval.py threedgrut/tests/test_per_class_eval.py
git commit -m "feat(P3-lane): Sobel grad-magnitude correlation sharpness scalar (Task2)"
```

---

## Task 3：lane 指标主函数 `compute_lane_metrics` + `LANE_CLASS_IDS`

**Files:**
- Modify: `threedgrut/model/per_class_eval.py`
- Test: `threedgrut/tests/test_per_class_eval.py`

- [ ] **Step 1: 写失败的测试**

```python
# -----------------------------------------------------------------------------
# compute_lane_metrics — dilated-band LPIPS + lane-PSNR + 梯度锐度
# -----------------------------------------------------------------------------
from threedgrut.model.per_class_eval import (  # noqa: E402
    LANE_CLASS_IDS,
    compute_lane_metrics,
)

_LANE_KEYS = {
    "lane_band_lpips", "lane_band_psnr", "lane_raw_psnr",
    "lane_grad_corr", "lane_n_pixels", "lane_band_n_pixels",
}


def test_lane_metrics_dict_keys_exact():
    H = W = 64
    gt = torch.zeros(H, W, 3)
    pred = torch.full((H, W, 3), 0.1)
    lane = torch.zeros(H, W, dtype=torch.long)
    lane[30, :] = LANE_CLASS_IDS[0]  # 一条 1px 横线
    out = compute_lane_metrics(pred, gt, lane, LANE_CLASS_IDS,
                               band_px=8, lpips_fn=_fake_lpips)
    assert set(out.keys()) == _LANE_KEYS


def test_lane_thin_mask_band_is_meaningful():
    """1px lane 线本身像素 < min_pixels（raw LPIPS 会 None），但膨胀后 band 够大
    → band LPIPS 有值。编码 P0 教训：细 mask 必须膨胀才有 LPIPS 信号。"""
    H = W = 64
    gt = torch.zeros(H, W, 3)
    pred = torch.full((H, W, 3), 0.3)
    lane = torch.zeros(H, W, dtype=torch.long)
    lane[30, :] = LANE_CLASS_IDS[0]  # 64 px raw
    out = compute_lane_metrics(pred, gt, lane, LANE_CLASS_IDS,
                               band_px=8, lpips_fn=_fake_lpips)
    assert out["lane_n_pixels"] == 64
    assert out["lane_band_n_pixels"] > 64 * 5  # 膨胀显著放大
    assert out["lane_band_lpips"] is not None
    assert out["lane_band_psnr"] is not None


def test_lane_absent_returns_none_metrics():
    H = W = 64
    gt = torch.zeros(H, W, 3)
    pred = torch.full((H, W, 3), 0.1)
    lane = torch.zeros(H, W, dtype=torch.long)  # 无 lane 像素
    out = compute_lane_metrics(pred, gt, lane, LANE_CLASS_IDS,
                               band_px=8, lpips_fn=_fake_lpips)
    assert out["lane_n_pixels"] == 0
    assert out["lane_band_n_pixels"] == 0
    assert out["lane_band_lpips"] is None
    assert out["lane_band_psnr"] is None
    assert out["lane_raw_psnr"] is None
    assert out["lane_grad_corr"] is None


def test_lane_restrict_mask_limits_region():
    """restrict_mask（如中心 crop / 前视）只保留左半 → raw 像素减半。"""
    H = W = 64
    gt = torch.zeros(H, W, 3)
    pred = torch.full((H, W, 3), 0.1)
    lane = torch.zeros(H, W, dtype=torch.long)
    lane[30, :] = LANE_CLASS_IDS[0]  # 满宽 64 px
    restrict = torch.zeros(H, W, dtype=torch.bool)
    restrict[:, :32] = True  # 左半
    out = compute_lane_metrics(pred, gt, lane, LANE_CLASS_IDS, band_px=0,
                               restrict_mask=restrict, lpips_fn=_fake_lpips)
    assert out["lane_n_pixels"] == 32


def test_lane_class_ids_guard():
    """钉死 lane 类 id（Mapillary palette 对账后改这里 + 本断言，同 commit）。"""
    assert LANE_CLASS_IDS == (24,)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /Users/etendue/repo/3dgrut2 && python -m pytest threedgrut/tests/test_per_class_eval.py -k lane -v`
Expected: FAIL — `ImportError: cannot import name 'LANE_CLASS_IDS'`

- [ ] **Step 3: 写最小实现**

在 `per_class_eval.py` 常量区（`DEFAULT_LANE_BAND_PX` 之后）加：

```python
# Mapillary "Lane Marking" 类 id。占位 (24,)，待 Task 7 在 GPU 上读一帧
# unique() 对 reader.class_palette 对账后定值；guard test 同 commit 更新。
LANE_CLASS_IDS: Tuple[int, ...] = (24,)
```

在 `compute_per_class_metrics`（L129 之后）末尾加：

```python
def compute_lane_metrics(
    rgb_pred: torch.Tensor,    # [H, W, 3] in [0, 1]
    rgb_gt: torch.Tensor,      # [H, W, 3]
    lane_sseg: torch.Tensor,   # [H, W] lane 产物类 id（**不是** semantic_sseg）
    lane_ids: Iterable[int] = LANE_CLASS_IDS,
    *,
    band_px: int = DEFAULT_LANE_BAND_PX,
    restrict_mask: Optional[torch.Tensor] = None,  # [H,W] bool/float，如前视中心 crop
    lpips_fn: Optional[LpipsFn] = None,
    min_pixels: int = 50,
) -> Dict[str, object]:
    """车道线重建指标（dilated band）。一次全报所有候选，baseline 后挑主指标：

    * ``lane_band_lpips`` — 锐度/定位代理（raw 细 mask 受面积主导≈0，膨胀给 LPIPS 上下文）。
    * ``lane_band_psnr``  — band 内存在/对比度。
    * ``lane_raw_psnr``   — 未膨胀条纹本身的对比度。
    * ``lane_grad_corr``  — band 内 Sobel 梯度幅值相关（最直指"线锐且在对位置"）。

    返回 6 key（不足 min_pixels / band 空 → 对应 None）：
    ``lane_band_lpips, lane_band_psnr, lane_raw_psnr, lane_grad_corr,
    lane_n_pixels（raw 条纹）, lane_band_n_pixels（膨胀 band）``。
    """
    raw_mask = class_mask_from_sseg(lane_sseg, lane_ids)   # [H,W] bool
    band_mask = dilate_mask(raw_mask, band_px)             # [H,W] bool
    if restrict_mask is not None:
        rm = restrict_mask.to(torch.bool)
        raw_mask = raw_mask & rm
        band_mask = band_mask & rm

    n_raw = int(raw_mask.sum().item())
    n_band = int(band_mask.sum().item())

    band_lpips = (
        compute_lpips_in_mask(rgb_pred, rgb_gt, band_mask, lpips_fn, min_pixels=min_pixels)
        if lpips_fn is not None else None
    )
    band_psnr = compute_psnr_in_mask(rgb_pred, rgb_gt, band_mask, min_pixels=min_pixels)
    raw_psnr = compute_psnr_in_mask(rgb_pred, rgb_gt, raw_mask, min_pixels=min_pixels)
    grad_corr = _grad_mag_corr_in_mask(rgb_pred, rgb_gt, band_mask, min_pixels=min_pixels)

    return {
        "lane_band_lpips": band_lpips,
        "lane_band_psnr": band_psnr,
        "lane_raw_psnr": raw_psnr,
        "lane_grad_corr": grad_corr,
        "lane_n_pixels": n_raw,
        "lane_band_n_pixels": n_band,
    }
```

- [ ] **Step 4: 跑测试确认通过 + 全模块回归**

Run: `cd /Users/etendue/repo/3dgrut2 && python -m pytest threedgrut/tests/test_per_class_eval.py -v`
Expected: PASS（原 11 + 新增 ~14 全绿）

- [ ] **Step 5: commit**

```bash
git add threedgrut/model/per_class_eval.py threedgrut/tests/test_per_class_eval.py
git commit -m "feat(P3-lane): compute_lane_metrics (band LPIPS+PSNR+grad-corr) + LANE_CLASS_IDS (Task3)"
```

---

## Task 4：datasetNcore 加载 lane 产物（val/test-only + image_infos 透传）

> **决策（偏离 spec 的"双路径"）**：lane 是**纯 eval 指标、无 loss**，只有 render.py 离线 eval（make_test → val/test 分支）消费它。现有 sseg 在 train+val 双路径是因为 `sky/road/dyn_mask` 是**训练 loss 输入**；lane 没有。train 侧加载 lane = 纯 host I/O + 解码浪费 + 多一份 bug 面。**只在消费处（val/test）加载**。P0.4 教训是"train 有 val 漏"，这里我们直接只放 val——恰好对。

**Files:**
- Modify: `threedgrut/datasets/datasetNcore.py:105`（`__init__` 形参）, `:178-180`（reader 字段）, `:1260-1287`（`_ensure_aux_readers`）, `:1116-1136`（val/test 加载块）, `:1600-1609`（image_infos 透传）
- Test: `threedgrut/tests/test_aux_discover_lane.py`（新建，纯 pathlib 可 Mac 测）

- [ ] **Step 1: 写失败的测试**（新建 `threedgrut/tests/test_aux_discover_lane.py`）

```python
# SPDX-License-Identifier: Apache-2.0
"""discover_aux_path 对 lane 产物的 glob 行为（纯 pathlib，Mac 可测）。"""
from __future__ import annotations

import pytest

from threedgrut.datasets.aux_readers import discover_aux_path


def test_discover_lane_finds_single(tmp_path):
    (tmp_path / "clip.aux.lane.zarr.itar").touch()
    (tmp_path / "clip.aux.sseg.zarr.itar").touch()  # 不应被 lane 命中
    p = discover_aux_path(tmp_path, "lane")
    assert p is not None
    assert p.name == "clip.aux.lane.zarr.itar"


def test_discover_lane_absent_returns_none(tmp_path):
    (tmp_path / "clip.aux.sseg.zarr.itar").touch()
    assert discover_aux_path(tmp_path, "lane") is None


def test_discover_lane_ambiguous_raises(tmp_path):
    (tmp_path / "a.aux.lane.zarr.itar").touch()
    (tmp_path / "b.aux.lane.zarr.itar").touch()
    with pytest.raises(ValueError):
        discover_aux_path(tmp_path, "lane")
```

- [ ] **Step 2: 跑测试确认通过**（`discover_aux_path` 已通用，这步应**直接 PASS** —— 证明产物发现无需改 aux_readers.py）

Run: `cd /Users/etendue/repo/3dgrut2 && python -m pytest threedgrut/tests/test_aux_discover_lane.py -v`
Expected: PASS（3 个）。若 FAIL 说明 glob 有意外，需先查 `discover_aux_path`。

- [ ] **Step 3: datasetNcore 接线（5 处）**

(a) `__init__` 形参（`load_aux_masks: bool = False,` L105 之后加一行）：
```python
        # Phase 3 lane GT：独立的 lane 分割产物（*.aux.lane.zarr.itar，Mapillary
        # palette 含 lane-marking）。纯 eval 指标，只在 val/test 分支加载 →
        # render.py per-class 评测器读 image_infos["semantic_lane_sseg"]。
        # 默认 False → 现有路径字节等价。
        load_lane_masks: bool = False,
```

(b) reader 字段（`self._aux_readers_initialized: bool = False` L180 之后加）：
```python
        self.load_lane_masks: bool = load_lane_masks
        # 复用 SsegAuxReader（lane 产物内部 group 名同为 semantic_segmentation）。
        self._lane_reader: Optional[SsegAuxReader] = None
```
> ⚠️ `load_lane_masks=True` 时 `__init__` 里也要保证 `self.load_aux_masks=True`（lane 加载块复用 `_ensure_aux_readers` 的 `_camera_resolutions`/clip_dir 上下文）。在 (b) 之后加一行守护：
```python
        if self.load_lane_masks:
            self.load_aux_masks = True  # lane 加载复用 aux reader 基建
```

(c) `_ensure_aux_readers()`（L1283 `self._aux_readers_initialized = True` **之前**加 lane 发现块，**软失败**——lane 缺文件不该让现有 eval 崩）：
```python
        if self.load_lane_masks:
            lane_path = discover_aux_path(clip_dir, "lane")
            if lane_path is not None:
                self._lane_reader = SsegAuxReader(lane_path)
                logger.info(f"NCoreDataset[{self.split}] lane reader ready: {lane_path.name}")
            else:
                logger.warning(
                    f"NCoreDataset: load_lane_masks=True but no *.aux.lane.zarr.itar "
                    f"in {clip_dir}; lane metrics skipped (软失败，不影响其他类)."
                )
```

(d) val/test 加载块（L1136 `val_batch["semantic_sseg"] = ...` **之后**、L1138 `return val_batch` 之前加）：
```python
                # Phase 3: lane 产物（独立 itar）。复用 sseg 的 END-ts + 渲染分辨率
                # NEAREST resize（类 id 不可插值）。软失败：reader None → 跳过。
                if self.load_lane_masks and self._lane_reader is not None:
                    lane = self._lane_reader.read(camera_id, val_ts_us)  # [H_full,W_full] uint8
                    if lane.shape[0] != h_render or lane.shape[1] != w_render:
                        lane = cv2.resize(
                            lane, (w_render, h_render), interpolation=cv2.INTER_NEAREST
                        )
                    val_batch["semantic_lane_sseg"] = to_torch(lane, device="cpu")
```
> 注：`h_render, w_render` 在 L1118 已由 `load_aux_masks` 块算出（lane 块在其后、同作用域），直接复用。

(e) image_infos 透传（L1609 `semantic_sseg` 透传之后加）：
```python
            # Phase 3 lane：与 semantic_sseg 并列透传给 render.py 评测器。
            if "semantic_lane_sseg" in batch:
                batch_dict["image_infos"]["semantic_lane_sseg"] = _to_gpu(batch["semantic_lane_sseg"])
```
> 注：该块在 `if "sky_mask" in batch:`（L1600）内。lane 与 sseg 同受 `load_aux_masks` 基建保障（守护 (b) 已确保 sky_mask 会被加载），故嵌在内层安全。

- [ ] **Step 4: import smoke + 全单测回归**（datasetNcore 需 NCore/cv2，Mac 仅做 import smoke；真实加载在 Task 8 GPU 集成验证）

Run:
```bash
cd /Users/etendue/repo/3dgrut2 && python -m pytest threedgrut/tests/test_aux_discover_lane.py -v \
  && python -c "import ast; ast.parse(open('threedgrut/datasets/datasetNcore.py').read()); print('datasetNcore syntax OK')"
```
Expected: pytest PASS + `datasetNcore syntax OK`

- [ ] **Step 5: commit**

```bash
git add threedgrut/datasets/datasetNcore.py threedgrut/tests/test_aux_discover_lane.py
git commit -m "feat(P3-lane): load *.aux.lane.zarr.itar in val/test branch → semantic_lane_sseg (Task4)"
```

---

## Task 5：render.py lane eval 调用 + metrics.json 写入 + config

**Files:**
- Modify: `threedgrut/render.py:35`（import）, `:425-428`（常量+累加器）, `:755`（eval 调用）, `:949`（metrics 写入）
- Modify: `threedgrut/datasets/__init__.py:143,183,264`（3 个构造点）
- Modify: `configs/apps/ncore_3dgut_mcmc_multilayer.yaml`（`load_lane_masks` + 可选 `render.lane_band_px`）

- [ ] **Step 1: render.py import（L35 `compute_per_class_metrics,` 所在 import 块内加）**

```python
    compute_lane_metrics,
    LANE_CLASS_IDS,
    DEFAULT_LANE_BAND_PX,
```

- [ ] **Step 2: 常量 + 累加器（L428 `per_class_npix` 之后加）**

```python
        # Phase 3 lane — 独立 lane 产物（semantic_lane_sseg）的候选指标累加器。
        # 限前视相机（risk L2：lane 最清处）。无 lane 产物 → 累加器空 →
        # metrics.json 字节等价（与 per_class 同样的"缺即不写"语义）。
        LANE_EVAL_CAMERAS = ("camera_front_wide_120fov",)
        lane_band_px = int(self.conf.render.get("lane_band_px", DEFAULT_LANE_BAND_PX))
        lane_metric_keys = ("lane_band_lpips", "lane_band_psnr", "lane_raw_psnr", "lane_grad_corr")
        lane_acc: dict[str, list] = {}
        lane_npix_acc: list = []
        lane_band_npix_acc: list = []
```

- [ ] **Step 3: eval 调用（L755 per-class 块 `per_class_npix.setdefault(...)` 之后加）**

```python
            # Phase 3 lane：独立 lane 产物 + 膨胀 band 指标。限前视相机。
            # 缺 semantic_lane_sseg（未开 load_lane_masks / 非前视）→ 跳过 → 无新字段。
            _lane = (getattr(gpu_batch, "image_infos", None) or {}).get("semantic_lane_sseg")
            if _lane is not None and _cam_id in LANE_EVAL_CAMERAS:
                lane_one = _lane[0] if _lane.dim() == 3 else _lane  # [H, W]
                lm = compute_lane_metrics(
                    pred_rgb_full[0], rgb_gt_full[0], lane_one, LANE_CLASS_IDS,
                    band_px=lane_band_px, lpips_fn=criterions.get("lpips"),
                )
                for _k in lane_metric_keys:
                    if lm[_k] is not None:
                        lane_acc.setdefault(_k, []).append(lm[_k])
                lane_npix_acc.append(lm["lane_n_pixels"])
                lane_band_npix_acc.append(lm["lane_band_n_pixels"])
```

- [ ] **Step 4: metrics.json 写入（L949 per_class 写入块之后加）**

```python
        # Phase 3 lane 指标聚合。仅当 lane 产物被加载（lane_npix_acc 非空）→
        # 否则整块缺省，metrics.json 字节等价（守护线零回归）。
        if lane_npix_acc:
            for _k in lane_metric_keys:
                _v = lane_acc.get(_k, [])
                metrics_json[f"mean_{_k}"] = float(np.mean(_v)) if _v else None
            metrics_json["lane_n_records"] = int(len(lane_npix_acc))
            metrics_json["lane_total_pixels"] = int(np.sum(lane_npix_acc))
            metrics_json["lane_band_total_pixels"] = int(np.sum(lane_band_npix_acc))
```
> **新增 metrics.json key**：`mean_lane_band_lpips` / `mean_lane_band_psnr` / `mean_lane_raw_psnr` / `mean_lane_grad_corr` / `lane_n_records` / `lane_total_pixels` / `lane_band_total_pixels`。

- [ ] **Step 5: config 3 个构造点（`threedgrut/datasets/__init__.py`）**

每处 `load_aux_masks=config.dataset.get("load_aux_masks", False),` 之后加同款一行（train L143 / val L183 / test L264）：
```python
                load_lane_masks=config.dataset.get("load_lane_masks", False),  # Phase 3 lane
```

- [ ] **Step 6: yaml（`configs/apps/ncore_3dgut_mcmc_multilayer.yaml`，`load_aux_masks: true`（L93）附近加）**

```yaml
  load_lane_masks: false   # Phase 3 lane GT：eval 时 CLI 覆盖 dataset.load_lane_masks=true
```
并在 `render:` 段（若无则不强加，render.py 已有 `.get` 默认）可选加：
```yaml
  # lane_band_px: 8   # Phase 3 lane dilated-band 半宽（默认 8）
```

- [ ] **Step 7: syntax smoke（render.py 需 GPU，Mac 仅 parse；真实验证在 Task 8）**

Run:
```bash
cd /Users/etendue/repo/3dgrut2 \
  && python -c "import ast; ast.parse(open('threedgrut/render.py').read()); print('render.py syntax OK')" \
  && python -c "import ast; ast.parse(open('threedgrut/datasets/__init__.py').read()); print('datasets/__init__ OK')" \
  && python -c "import yaml; yaml.safe_load(open('configs/apps/ncore_3dgut_mcmc_multilayer.yaml')); print('yaml OK')" \
  && python -m pytest threedgrut/tests/test_per_class_eval.py -q
```
Expected: 3 行 OK + per_class_eval 全绿

- [ ] **Step 8: commit**

```bash
git add threedgrut/render.py threedgrut/datasets/__init__.py configs/apps/ncore_3dgut_mcmc_multilayer.yaml
git commit -m "feat(P3-lane): render.py lane eval (front-cam) + metrics.json mean_lane_* + config wiring (Task5)"
```

---

## Task 6：Mac 全门回归（提交前关）

> 对标 CLAUDE.md 把关清单：rsync 上 GPU 前先把便宜的 Mac 单测跑绿。

- [ ] **Step 1: 跑受影响模块全测**

Run:
```bash
cd /Users/etendue/repo/3dgrut2 && python -m pytest \
  threedgrut/tests/test_per_class_eval.py \
  threedgrut/tests/test_aux_discover_lane.py \
  threedgrut/tests/test_class_psnr.py -v
```
Expected: 全 PASS（per_class_eval 原 11 + 新 ~14；discover_lane 3；class_psnr 原样不破）

- [ ] **Step 2: 确认无 import 副作用**（per_class_eval 仍不引 cv2/NCore）

Run: `cd /Users/etendue/repo/3dgrut2 && python -c "import threedgrut.model.per_class_eval as m; print([n for n in dir(m) if 'lane' in n.lower()])"`
Expected: 打印 `['LANE_CLASS_IDS', 'compute_lane_metrics']`（无 ImportError、无 cv2 报错）

- [ ] **Step 3:（无新 commit，纯验证关）**——若全绿，Mac 侧（Task 1–5）就绪可 rsync。

---

## Task 7（GPU / inceptio）：自跑 mask2former+Mapillary 生成 lane 产物 + 对账 `LANE_CLASS_IDS`

> **决策已定（用户 2026-06-09）**：走**分支 B 自跑模型**（不依赖 nre-tools），机器 **inceptio（RTX 4090）**。本 Task 不训练、仅离线逐帧推理（`num_workers` 约束不适用）。clip 用 baseline 同 clip（确认 baseline ckpt `v3_base_scratch30k_lam01` 对应的 clip，见 Task 8 step 1）。
>
> ssh 范式（CLAUDE.md）：`ssh inceptio 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate 3dgrut2 && cd ~/repo/3dgrut2 && ...'`；长任务用 inline nohup + 末尾 `echo PID $!`。

- [ ] **Step 1: rsync Mac 代码到 inceptio**

```bash
rsync -az --exclude='.claude/worktrees' --exclude='.venv' --exclude='__pycache__' \
  /Users/etendue/repo/3dgrut2/ inceptio:~/repo/3dgrut2/
```

- [ ] **Step 2: 写 `scripts/gen_lane_sseg.py`（自跑 mask2former + Mapillary → 写 zarr.itar）**

新建 `scripts/gen_lane_sseg.py`（GPU 离线，逐帧推理 + PNG 编码 + 写 zarr.itar，反向复用 `SsegAuxReader` 的读格式）。**契约（§2）必须满足**：`/aux/semantic_segmentation/<cam>/<END_ts_us>` 存 `[H,W] uint8` PNG bytes、`.zattrs.stuff_classes` = Mapillary palette、`.zattrs.resolution=[W,H]`；产物落 `<clip>/<clip>.aux.lane.zarr.itar`。骨架：

```python
# scripts/gen_lane_sseg.py（inceptio GPU）
# 1. 读 clip manifest（pai_<clip>.json），遍历 camera_front_wide_120fov 各帧:
#    取 RGB（解畸变后送模型即可）+ END timestamp（frames_timestamps_us[idx, END]）
# 2. mask2former(Mapillary Vistas 权重) 推理 → [H,W] uint8 类 id（保留 lane-marking 类）
#    - HF: facebook/mask2former-swin-* 的 Mapillary checkpoint，经 ~/.cache/huggingface/token
#    - inceptio 已有 mask2former JIT 缓存基建（CLAUDE.md 首次运行记录）
# 3. PNG 编码每帧 → zarr.open(IndexedTarStore(<out>.itar,"w")) 写
#    /aux/semantic_segmentation/<cam>/<END_ts_us>（0-D |S<n> bytes, attrs.format="png"）
# 4. group .zattrs 写 stuff_classes（Mapillary palette list）+ resolution [W,H]
# 产物：<clip_dir>/<clip>.aux.lane.zarr.itar  （discover_aux_path(clip_dir,"lane") 可发现）
```
> **只对前视相机 `camera_front_wide_120fov` 出帧**（risk L2 + 省算力），与 render eval 的 `LANE_EVAL_CAMERAS` 一致。写完用 `discover_aux_path(clip_dir,"lane")` + `SsegAuxReader.read` 自测能读回一帧再进 Step 3。

- [ ] **Step 3: 跑生成 + 验证可读回**

```bash
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH \
  && export CUDA_VISIBLE_DEVICES=0 && cd ~/repo/3dgrut2 \
  && nohup python scripts/gen_lane_sseg.py \
       --clip <clip_dir> --camera camera_front_wide_120fov \
       > /tmp/gen_lane.log 2>&1 & echo PID $!'
# 跑完看日志 + 确认产物存在
ssh inceptio 'ls -la <clip_dir>/*.aux.lane.zarr.itar && tail -5 /tmp/gen_lane.log'
```

- [ ] **Step 4: 对账 lane 类 id（risk L1，硬门）**

```bash
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH && cd ~/repo/3dgrut2 && python - <<PY
from threedgrut.datasets.aux_readers import SsegAuxReader, discover_aux_path
import numpy as np
p = discover_aux_path("<clip_dir>", "lane"); r = SsegAuxReader(p)
print("palette:", r.class_palette)
cam = "camera_front_wide_120fov"
ts = next(iter(r._cam_group(cam).array_keys()))
m = r.read(cam, int(ts))
u, c = np.unique(m, return_counts=True)
print("unique ids:", dict(zip(u.tolist(), c.tolist())))
PY'
```
读出 lane-marking 对应的真实 id（对 `class_palette` 里 `"lane marking"`/`"general lane marking"` 之类的索引），**更新 `per_class_eval.py:LANE_CLASS_IDS` + `test_per_class_eval.py:test_lane_class_ids_guard` 断言**（注意：Mapillary lane id 大概率**不是**占位的 24），Mac 重跑 `pytest -k lane_class_ids_guard` 绿。

- [ ] **Step 5: commit（生成脚本 + 对账结果）**

```bash
git add scripts/gen_lane_sseg.py threedgrut/model/per_class_eval.py threedgrut/tests/test_per_class_eval.py
git commit -m "feat(P3-lane): self-run mask2former+Mapillary lane product gen + reconcile LANE_CLASS_IDS (Task7)"
```

---

## Task 8（GPU / inceptio）：baseline ckpt 立锚 + 守护线零回归

> 纯 eval、**无训练**。机器 inceptio（与 Task 7 同机，lane 产物已在 clip 目录）。对标 CLAUDE.md 把关清单 B/C：metrics.json 必须出现所有 `mean_lane_*` key 才算 ✅。

- [ ] **Step 1: 定位 baseline ckpt + clip**

```bash
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH \
  && find ~/work/output -maxdepth 3 -iname "*v3_base_scratch30k_lam01*" -name "*.pt" 2>/dev/null | head'
```
记下 ckpt 路径 + 对应 clip 的 `pai_<clip>.json`（lane 产物须在同 clip 目录，Task 7 已生成）。若 inceptio 上无该 baseline ckpt → 先从 A800 rsync ckpt 过来，或在 inceptio 重 eval 前确认 ckpt 可加载。

- [ ] **Step 2: 确认代码已同步（Task 7 Step 1 已 rsync；这里只 verify）**

```bash
ssh inceptio "grep -n 'compute_lane_metrics' ~/repo/3dgrut2/threedgrut/render.py"  # 确认同步到位
```

- [ ] **Step 3: 跑 render eval（开 load_lane_masks，限前视）**

```bash
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH \
  && export CUDA_VISIBLE_DEVICES=0 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  && cd ~/repo/3dgrut2 \
  && python render.py --config-name apps/ncore_3dgut_mcmc_multilayer \
       path=<clip>/pai_<clip>.json \
       dataset.load_lane_masks=true \
       render.eval_cameras="[camera_front_wide_120fov]" \
       <renderer/ckpt 加载参数：复用 P0.4 立锚同款 render 调用> \
       out_dir=~/work/output experiment_name=p3_lane_anchor 2>&1 | tee /tmp/p3_lane_anchor.log'
```
> render.py 的 ckpt 加载 / eval 入口参数沿用 P0.4 baseline 立锚那次的调用（见 `v3_plan_revised.md` §6 P0.4 Done Log 的实际命令）。约 ~3 min（同 P0 体量）。

- [ ] **Step 4: 验 metrics.json（硬门）**

```bash
ssh inceptio 'cat ~/work/output/p3_lane_anchor/metrics.json' | python -m json.tool
```
**必须看到**：`mean_lane_band_lpips` / `mean_lane_band_psnr` / `mean_lane_raw_psnr` / `mean_lane_grad_corr` / `lane_n_records`（> 0）/ `lane_total_pixels` / `lane_band_total_pixels`。
- `lane_n_records == 0` → lane 产物没接通（查 step 1 clip 是否含 `*.aux.lane.zarr.itar` + 前视相机对账 id）。task 保持 🟡。

- [ ] **Step 5: 守护线零回归核对（纯增量证明）**

同一 metrics.json 里核对未变：`mean_cc_psnr ≈ 25.79`、novel `mean_novel_lpips_avg ≈ 0.5987`（若该 eval 开 novel）、`mean_lidar_psnr ≈ 22.69`、`mean_road_crop_*` 与 P0.3 一致。任何漂移 → lane 改动非纯增量，回头查（多半是 image_infos 透传误改了既有 key）。

- [ ] **Step 6: A.3 红旗自检（信号/方差）**

看 4 个候选 `mean_lane_*` 跨帧是否**有信号 + 有方差**（`lane_n_records` 足够、`grad_corr` 不恒为 1/None、`band_lpips` 不恒 0）。
- **挑主指标**：信号最强、方差最合理的那个（预期 `lane_band_lpips` 或 `lane_grad_corr`）记为 Phase 3 主 KPI。
- **全 None/无方差 = 红旗**：回 Task 3/7 改指标或 band_px（`render.lane_band_px` CLI sweep 8→4/16），**别急着投 P3.1/P3.2**。

- [ ] **Step 7: 回填实测数字（进 Task 9 文档）**——把 4 个 `mean_lane_*` + `lane_n_records` + 选定主指标 + ckpt/commit hash 记下。

---

## Task 9：文档同步（CLAUDE.md 强制：不更新 = 任务未完成）

**Files:** `v3_plan_revised.md`, `v2_architecture.md`

- [ ] **Step 1: `v3_plan_revised.md`**
  - §1.1 Kanban：把 Phase 3 lane 测量门任务从 Backlog → Done，标 ✅（卡片标签括号一律**全角（）**，见 CLAUDE.md mermaid 铁律）。
  - §1.2 任务表：状态 ✅ + 填本 plan 各 Task 的 commit 短 hash。
  - §1.3 per-class gap 表：**回填 Phase 3 lane 三/四行真实数字**（Task 8 step 7 的 `mean_lane_*` + 选定主指标）。
  - §6 Done Log：追加一条 —— 日期 2026-06-XX + commit hashes + 改动摘要 +「baseline lane 锚点：band_lpips=X / band_psnr=Y / grad_corr=Z / n_records=N，主指标=<choice>，守护线零回归确认」。

- [ ] **Step 2: `v2_architecture.md`**
  - 对应 mermaid 图：lane 评测节点 `:::todo → :::done` + 追加 commit 短 hash（括号全角）。
  - §6.1/6.2 文件清单：`per_class_eval.py`（+compute_lane_metrics）/`datasetNcore.py`（+lane 加载）/`render.py`（+mean_lane_*）/`scripts/gen_lane_sseg.py` 标 ✅。
  - §7 关键不变量：加一行「lane 指标只在 render.py（单路径），trainer 不算；lane 产物缺省时 metrics.json 字节等价」+ 验证锚点（test_per_class_eval lane 用例）。

- [ ] **Step 3: mermaid 提交前自查（应零输出）**

Run: `cd /Users/etendue/repo/3dgrut2 && awk '/\`\`\`mermaid/{i=1;next} /\`\`\`/&&i{i=0;next} i&&/\(/{print FILENAME":"NR": "$0}' v3_plan_revised.md`
Expected: 零输出（无半角 `(` 漏网）

- [ ] **Step 4: commit**

```bash
git add v3_plan_revised.md v2_architecture.md
git commit -m "docs(plan): mark Phase3 lane gate done + lane anchor numbers in v3_plan_revised
docs(arch): flip P3-lane nodes to done in v2_architecture.md"
```

---

## 验证总览（end-to-end）

| 层 | 验证 | 命令 / 期望 |
|---|---|---|
| 纯张量单元 | dilation / lane 指标 / 锐度 / guard | `pytest threedgrut/tests/test_per_class_eval.py threedgrut/tests/test_aux_discover_lane.py -v` 全绿 |
| Mac 集成 | 无 cv2/NCore 副作用 | `python -c "import threedgrut.model.per_class_eval"` 无报错 |
| 语法 | render/dataset/yaml | 三个 `ast.parse`/`yaml.safe_load` OK |
| GPU 产物 | lane itar + 类 id 对账 | `SsegAuxReader.read` 出非空 + unique 含 lane id |
| GPU eval | metrics.json 出 `mean_lane_*` | `lane_n_records > 0` + 4 个 mean key 存在 |
| 零回归 | 守护线不变 | `mean_cc_psnr/novel/lidar/road_crop` 与 P0 一致 |
| 文档 | 看板/Done Log/arch 回填 | mermaid 自查零输出 + §1.3 有真实数字 |

## 关键风险 → 缓解（承自 spec § 3）

| ID | 风险 | 本 plan 缓解 |
|---|---|---|
| L1 | Mapillary palette 偏移 | Task 7 step 3 读 unique 对账 + guard test 钉死 |
| L2 | 外围相机 lane seg 噪 | render `LANE_EVAL_CAMERAS` 限前视；`restrict_mask` 留中心 crop 接口 |
| L3 | 细 mask 指标失效 | dilated band（Task 1）+ baseline 红旗自检（Task 8 step 6） |
| L4 | nre-tools 不支持换 palette | Task 7 分支 B 自跑模型（契约 § 2 解耦，代码不变） |
| L5 | 近大远小 / 远处 lane 几像素 | 图像空间先做；BEV 变体明确**不在本 plan**（spec stretch） |

## 自检（Self-Review）结论

- **spec 覆盖**：A.1（Task 7）/A.2（Task 4+5）/A.3（Task 1-3 + Task 8 红旗）/A.4（Task 8 立锚 + 零回归）全部有对应 Task；§5 非目标（P3.1/P3.2/BEV/回填他类）均在 § 0 排除。
- **类型一致**：`compute_lane_metrics` 返回 6 key（Task 3 定义）= render 累加器读的 4 metric key + 2 npix（Task 5）= metrics.json 7 字段，三处对齐。
- **无占位**：每个 code step 给完整代码 + 真实 file:line 锚点；唯一待定值 `LANE_CLASS_IDS=(24,)` 是显式占位 + Task 7 对账门 + guard test，非"TODO 后补"。
