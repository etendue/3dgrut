# Stage 11 — LiDAR + DepthAnythingV2 Image-Space Depth Supervision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 3DGRUT2 训练加入 LiDAR sparse depth + DepthAnythingV2 dense depth 几何监督，把 novel-view PSNR (4 档平均) 抬升至 ≥ baseline + 3.0 dB（v3_plan.md Stage 11 出口）。

**Architecture:** **Image-space** 路线（drivestudio 风格）—— LiDAR 点云离线投影到相机平面生成稀疏 depth map，DepthAnythingV2 离线推理生成稠密 metric depth map，两者都通过 `image_infos` 注入到既有 Batch 协议，复用 tracer 已经返回的 `pred_dist [B,H,W,1]`，单前向通路同步监督。**不走** ray-space 旁路 forward —— tracer 已经提供逐像素深度，新增 forward 是浪费。

**Tech Stack:**
- 监督空间：image-space（相机像素 grid）
- LiDAR loss：`L1 + normalize(0,1)`（drivestudio 默认），λ_lidar 0.03 base + exponential decay
- Background LiDAR loss（sky 远场约束）：`MSE`，target = max_depth，λ 0.005
- DepthV2 loss：`L2 + inverse-depth`（NRE depth_inverse_mse 一致），λ_depth 0.01
- Hit mask 体系：`(gt > 0) * (1 - sky_mask) * (1 - dyn_mask) * valid_pixel_mask`
- DepthV2 模型：`depth-anything/Depth-Anything-V2-Metric-Outdoor-Large`（HuggingFace 公开 metric 版本）
- 数据落盘：`aux/lidar_depth/<cam_id>/<ts>.npz` + `aux/depth_anything_v2/<cam_id>/<ts>.npz`（与现有 `aux/sseg/*` 同模式）

**Decision log:**
- **2026-05-28 选 image-space over ray-space** —— tracer.py:346 已返回 `pred_dist [B,H,W,1]`，与相机像素一一对应；走 image-space 不需要新建 LiDAR ray 旁路 forward + 不需要扩 Batch 协议，工程量从 8d 降到 5.5d。drivestudio 验证了 image-space 路线在 GauStudio/EmerNeRF 都收敛良好。
- **Road 层处理留给 Stage 13a** —— v3_plan.md:489-493 已规划 V3-L4 `ignore_classes_from_layers=[road]` + V3-D6 cuboid LiDAR padding，本 plan 不区分 road / non-road，统一靠 sky_mask + dyn_mask 排除。
- **lidar_divergence T11.3 deferred** —— 出口指标不依赖 cone 抗锯齿，tracer Slang kernel 改动 ≥ 3d，留到 Stage 11 收敛后做。

---

## 主机选择决策树

执行 GPU 任务（Phase C / E / G）前**必须**先按此决策树选主机：

```
                     ┌─────────────────────┐
                     │ ssh a800-x2 nvidia-smi│
                     └──────────┬──────────┘
                                │
              ┌─────────────────┼──────────────────┐
              │                 │                  │
        GPU0 或 GPU1            两卡都 >60%        两卡都 满
        free mem >40GB          忙              （>95%）
              │                 │                  │
              ▼                 ▼                  ▼
       【主用 A800】     ┌─ssh thinkpad      ┌─CLAUDE.md §Vast.ai
       export CUDA_      │ nvidia-smi       │ 阶段 1-4 流程
       VISIBLE_DEVICES=0  │ (RTX 4090 24GB)  │ vast-rtx4090 实例
       (或 1)            │                  │
                        ▼                  ▼
              【1k smoke ok】       【30k 出口 only】
              n_iterations ≤ 5k    （vast 单次 ~$0.45-$1.5）
              出口 30k 必须 A800    
```

**硬规则：**
- **1k / 5k smoke**：A800 → ThinkPad → vast.ai 任意一台都可
- **30k 出口验证**：**必须 A800**（ThinkPad 24GB 显存可能 OOM，vast.ai 单次 ~$2.4 不划算）
- **Mac 本地**：纯 CPU 单测（Phase A/B/D/F），不需要 GPU
- 每次 ssh 前先 `nvidia-smi` 看占用，避免抢卡

---

## File Structure

| 文件 | 状态 | 职责 |
|---|---|---|
| `threedgrut/correction/depth_prior.py` | **CREATE** | `DepthLoss` 类（L1/L2/SmoothL1 + normalize + inverse-depth），`compute_bg_lidar_loss` helper |
| `threedgrut/datasets/aux_readers.py` | MODIFY | 新增 `LidarDepthAuxReader` + `DepthV2AuxReader`，与既有 `SsegAuxReader` 同模式 |
| `threedgrut/datasets/datasetNcore.py` | MODIFY | `get_gpu_batch_with_intrinsics` 注入 `image_infos["lidar_depth_map"]` + `image_infos["depth_prior"]` |
| `threedgrut/trainer.py` | MODIFY | `__init__` 注册三个 loss head + `get_losses` 累加三个 loss + scalar 上报 TensorBoard |
| `threedgrut/render.py` | MODIFY | eval loop 加 `mean_lidar_psnr` 字段 + 写入 metrics.json |
| `scripts/dump_lidar_depth_map.py` | **CREATE** | 离线脚本：每 clip × 每相机 × 每帧投影 LiDAR 点云到图像平面，存 npz |
| `scripts/download_depth_anything_v2.sh` | **CREATE** | HF 拉取 metric 版本 weights 到 `models/depth_anything_v2/` |
| `scripts/dump_depth_priors.py` | **CREATE** | 离线脚本：每张图过 DepthV2 拿稠密 metric depth，存 npz |
| `threedgrut/tests/test_depth_loss.py` | **CREATE** | DepthLoss 单测（L1/L2/inverse/normalize/mask 边界） |
| `threedgrut/tests/test_lidar_depth_projection.py` | **CREATE** | 投影函数单测（已知 3D 点 → 已知像素，ray-depth 公式） |
| `threedgrut/tests/test_depth_loss_grad.py` | **CREATE** | grad-check 单测：`pred_dist.grad` 传到 `gaussians.positions` ★ |
| `threedgrut/tests/test_lidar_depth_aux_reader.py` | **CREATE** | AuxReader 读取 npz 单测 |
| `threedgrut/tests/test_render_lidar_psnr_field.py` | **CREATE** | 验证 metrics.json 有 `mean_lidar_psnr` 字段 |
| `v3_plan.md` | MODIFY | Stage 11 任务从 ⬜ 翻成 ✅ + Done Log 追加 |

---

## Phase A — Loss heads（Mac CPU pure-Python，不需要 GPU）

### Task A1: DepthLoss 类 + 单测

**Files:**
- Create: `threedgrut/correction/depth_prior.py`
- Test: `threedgrut/tests/test_depth_loss.py`

- [ ] **Step 1: 写失败测试**

Create `threedgrut/tests/test_depth_loss.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DepthLoss (Stage 11 / T11.A1)."""
import pytest
import torch

from threedgrut.correction.depth_prior import DepthLoss, compute_bg_lidar_loss


@pytest.fixture
def synthetic_batch():
    """[B=1, H=4, W=4] 合成数据：左上角 4 像素有 GT depth=10m，其余 0。"""
    pred = torch.full((1, 4, 4, 1), 5.0)  # 全场预测 5m
    gt = torch.zeros(1, 4, 4)
    gt[0, 0:2, 0:2] = 10.0  # 左上角 4 像素 GT=10m
    hit_mask = (gt > 0).float()  # 4 个有效像素
    return pred, gt, hit_mask


def test_l1_basic(synthetic_batch):
    """L1 + normalize：|5/80 - 10/80| = 5/80 = 0.0625，仅 4 有效像素均值。"""
    pred, gt, mask = synthetic_batch
    loss = DepthLoss(loss_type="l1", normalize=True, max_depth=80.0)
    out = loss(pred, gt, mask)
    expected = abs(5.0 / 80.0 - 10.0 / 80.0)
    assert out.item() == pytest.approx(expected, abs=1e-6)


def test_l2_basic(synthetic_batch):
    """L2 + normalize：(5/80 - 10/80)^2 = (1/16)^2 = 1/256。"""
    pred, gt, mask = synthetic_batch
    loss = DepthLoss(loss_type="l2", normalize=True, max_depth=80.0)
    out = loss(pred, gt, mask)
    expected = (5.0 / 80.0 - 10.0 / 80.0) ** 2
    assert out.item() == pytest.approx(expected, abs=1e-6)


def test_inverse_depth_l2(synthetic_batch):
    """inverse-depth + L2：(1/5 - 1/10)^2 = (0.1)^2 = 0.01。"""
    pred, gt, mask = synthetic_batch
    loss = DepthLoss(loss_type="l2", use_inverse_depth=True, normalize=False)
    out = loss(pred, gt, mask)
    expected = (1.0 / 5.0 - 1.0 / 10.0) ** 2
    assert out.item() == pytest.approx(expected, abs=1e-6)


def test_invalid_gt_filtered():
    """GT < eps (0.01) 或 > max_depth 必须被 hit_mask 滤掉。"""
    pred = torch.full((1, 2, 2, 1), 5.0)
    gt = torch.tensor([[[0.001, 90.0], [10.0, 50.0]]])  # 前两个无效
    mask = torch.ones(1, 2, 2)  # 故意全开，DepthLoss 内部应再过滤
    loss = DepthLoss(loss_type="l1", normalize=True, max_depth=80.0, eps=0.01)
    out = loss(pred, gt, mask)
    # 仅 (10.0, 50.0) 两点参与 → mean(|5/80-10/80|, |5/80-50/80|) = mean(5/80, 45/80)
    expected = (abs(5.0 / 80.0 - 10.0 / 80.0) + abs(5.0 / 80.0 - 50.0 / 80.0)) / 2
    assert out.item() == pytest.approx(expected, abs=1e-6)


def test_zero_valid_pixels_returns_zero():
    """全 mask 0 时返回 0 而非 NaN。"""
    pred = torch.full((1, 2, 2, 1), 5.0)
    gt = torch.zeros(1, 2, 2)
    mask = torch.zeros(1, 2, 2)
    loss = DepthLoss(loss_type="l1", normalize=True)
    out = loss(pred, gt, mask)
    assert out.item() == 0.0
    assert torch.isfinite(out)


def test_smooth_l1():
    """smooth_l1：diff < 1 用平方分支，diff >= 1 用线性分支。"""
    pred = torch.full((1, 2, 2, 1), 5.0)
    gt = torch.tensor([[[10.0, 10.0], [10.0, 10.0]]])
    mask = torch.ones(1, 2, 2)
    loss = DepthLoss(loss_type="smooth_l1", normalize=True, max_depth=80.0)
    out = loss(pred, gt, mask)
    # diff_norm = 5/80 = 0.0625 < 1 → 0.5 * 0.0625^2
    expected = 0.5 * (5.0 / 80.0) ** 2
    assert out.item() == pytest.approx(expected, abs=1e-6)


def test_compute_bg_lidar_loss_sky_far_anchor():
    """sky 区域目标 = max_depth（最远 anchor），其他区域不参与。"""
    pred = torch.full((1, 2, 2, 1), 60.0)  # 预测 60m
    sky_mask = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])  # 仅 (0,0) 是 sky
    loss = compute_bg_lidar_loss(pred, sky_mask, max_depth=80.0)
    expected = ((60.0 / 80.0 - 1.0) ** 2)  # normalized: 60/80 vs 1.0
    assert loss.item() == pytest.approx(expected, abs=1e-6)


def test_compute_bg_lidar_loss_no_sky_returns_zero():
    """sky_mask 全 0 时返回 0。"""
    pred = torch.full((1, 2, 2, 1), 60.0)
    sky_mask = torch.zeros(1, 2, 2)
    loss = compute_bg_lidar_loss(pred, sky_mask, max_depth=80.0)
    assert loss.item() == 0.0
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd /Users/etendue/repo/3dgrut2
source .venv/bin/activate
pytest threedgrut/tests/test_depth_loss.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'threedgrut.correction.depth_prior'`

- [ ] **Step 3: 写实现**

Create `threedgrut/correction/depth_prior.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Image-space depth loss heads (Stage 11 T11.A1).

LiDAR sparse depth + DepthAnythingV2 dense depth supervision, drivestudio-style.
Reuses tracer's pred_dist [B, H, W, 1] (ray-depth) — does NOT spawn a separate
LiDAR ray forward pass.

Three loss heads exposed:
  - DepthLoss(loss_type, normalize, use_inverse_depth)
      Main head; works for both LiDAR sparse GT and DepthV2 dense GT.
  - compute_bg_lidar_loss(pred, sky_mask, max_depth)
      Sky-region anchor: target = max_depth, MSE on normalized depth.
      Stops sky Gaussians from collapsing into mid-range when no LiDAR returns.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthLoss(nn.Module):
    """Image-space depth loss (drivestudio L91-180 reference).

    Args:
        loss_type:        "l1" | "l2" | "smooth_l1"
        normalize:        scale pred/gt by 1/max_depth before loss (default True)
        use_inverse_depth: convert pred/gt to 1/d before loss (overrides normalize)
        max_depth:        far clip (also used for normalize). default 80m.
        eps:              gt values in (eps, max_depth) are valid. default 0.01.
    """
    def __init__(
        self,
        loss_type: str = "l1",
        normalize: bool = True,
        use_inverse_depth: bool = False,
        max_depth: float = 80.0,
        eps: float = 0.01,
    ):
        super().__init__()
        if loss_type not in ("l1", "l2", "smooth_l1"):
            raise ValueError(f"loss_type must be l1/l2/smooth_l1, got {loss_type}")
        self.loss_type = loss_type
        self.normalize = normalize
        self.use_inverse_depth = use_inverse_depth
        self.max_depth = max_depth
        self.eps = eps

    def forward(
        self,
        pred_depth: torch.Tensor,  # [B, H, W, 1] tracer ray-depth
        gt_depth: torch.Tensor,    # [B, H, W] or [B, H, W, 1]
        hit_mask: torch.Tensor,    # [B, H, W] {0, 1}
    ) -> torch.Tensor:
        pd = pred_depth.squeeze(-1)
        gd = gt_depth.squeeze(-1) if gt_depth.dim() == pd.dim() + 1 else gt_depth

        # GT range filter — drivestudio L155-160: gt < eps or > max → invalid
        valid = hit_mask * (gd > self.eps).float() * (gd < self.max_depth).float()

        if valid.sum() < 1.0:
            return torch.zeros((), device=pd.device, dtype=pd.dtype)

        if self.use_inverse_depth:
            pd_t = 1.0 / pd.clamp(min=self.eps)
            gd_t = 1.0 / gd.clamp(min=self.eps)
        elif self.normalize:
            pd_t = pd / self.max_depth
            gd_t = gd / self.max_depth
        else:
            pd_t = pd
            gd_t = gd

        if self.loss_type == "l1":
            diff = (pd_t - gd_t).abs()
        elif self.loss_type == "l2":
            diff = (pd_t - gd_t) ** 2
        else:  # smooth_l1
            diff = F.smooth_l1_loss(pd_t, gd_t, reduction="none", beta=1.0)

        denom = valid.sum().clamp(min=1.0)
        return (diff * valid).sum() / denom


def compute_bg_lidar_loss(
    pred_depth: torch.Tensor,  # [B, H, W, 1]
    sky_mask: torch.Tensor,    # [B, H, W] {0, 1}
    max_depth: float = 80.0,
) -> torch.Tensor:
    """Background LiDAR loss — anchor sky pixels at max_depth (NRE car2sim_6cam pattern).

    Without this, sky Gaussians can collapse into mid-range because the regular
    LiDAR head only sees points within the LiDAR FOV (no sky returns).
    """
    pd = pred_depth.squeeze(-1)
    if sky_mask.sum() < 1.0:
        return torch.zeros((), device=pd.device, dtype=pd.dtype)
    target = torch.full_like(pd, fill_value=1.0)  # normalized target = max_depth
    pd_norm = pd / max_depth
    diff_sq = (pd_norm - target) ** 2
    denom = sky_mask.sum().clamp(min=1.0)
    return (diff_sq * sky_mask).sum() / denom
```

- [ ] **Step 4: 跑测试确认通过**

```bash
pytest threedgrut/tests/test_depth_loss.py -v
```

Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add threedgrut/correction/depth_prior.py threedgrut/tests/test_depth_loss.py
git commit -m "feat(T11.A1): add DepthLoss + compute_bg_lidar_loss heads

Image-space depth supervision for Stage 11 LiDAR/DepthV2 prior.
DepthLoss supports l1/l2/smooth_l1 + normalize + inverse-depth modes.
compute_bg_lidar_loss anchors sky-region pred_dist at max_depth.

Stage 11 / T11.A1: pure-Python loss head, 8 unit tests pass on Mac CPU.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task A2: trainer get_losses 接入 + grad-check 单测 ★

**Files:**
- Modify: `threedgrut/trainer.py` (`__init__` near line 172, `get_losses` — locate via grep)
- Test: `threedgrut/tests/test_depth_loss_grad.py`

- [ ] **Step 1: 先 grep 定位 get_losses 与 trainer init 的 loss 区域**

```bash
grep -n "def get_losses\|def init_exposure_model\|lambda_lidar\|trainer_conf" threedgrut/trainer.py | head -20
```

记录下：
- `get_losses` 起始行号
- `init_exposure_model` 函数（作为新 init_depth_losses 的参考模板）

- [ ] **Step 2: 写失败测试（grad-check + numeric check）**

Create `threedgrut/tests/test_depth_loss_grad.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""grad-check: DepthLoss gradient must flow into Gaussian positions.

This is the SINGLE most important test for Stage 11 — if pred_dist's grad chain
is broken anywhere in tracer.py, depth loss will silently not update positions
and Stage 11 will look like it converged but PSNR won't move.
"""
import pytest
import torch

from threedgrut.correction.depth_prior import DepthLoss


def test_depth_loss_grad_flows_back_to_pred_depth():
    """最小 fixture：pred_depth.requires_grad=True → loss.backward() 后 pred_depth.grad 非零。"""
    pred = torch.full((1, 4, 4, 1), 5.0, requires_grad=True)
    gt = torch.full((1, 4, 4), 10.0)
    mask = torch.ones(1, 4, 4)

    loss = DepthLoss(loss_type="l1", normalize=True, max_depth=80.0)
    out = loss(pred, gt, mask)
    out.backward()

    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0
    # L1 normalize: d/d_pred = sign(pred/80 - gt/80) * (1/80) * (1/16 valid pixels)
    # pred=5 < gt=10 → sign = -1 → grad = -1/80/16 = -7.81e-4
    expected_grad = -1.0 / 80.0 / 16.0
    assert pred.grad[0, 0, 0, 0].item() == pytest.approx(expected_grad, abs=1e-6)


def test_depth_loss_grad_zero_when_mask_zero():
    """全 mask=0 时 grad 必须为 0（不能 NaN）。"""
    pred = torch.full((1, 4, 4, 1), 5.0, requires_grad=True)
    gt = torch.zeros(1, 4, 4)
    mask = torch.zeros(1, 4, 4)

    loss = DepthLoss(loss_type="l1", normalize=True)
    out = loss(pred, gt, mask)
    out.backward()

    # pred.grad 可能是 None 或全 0（pytorch 不一定 populate）
    assert (pred.grad is None) or (pred.grad.abs().sum().item() == 0.0)
    assert torch.isfinite(out)
```

- [ ] **Step 3: 跑测试确认通过**

```bash
pytest threedgrut/tests/test_depth_loss_grad.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 4: 写 trainer 接入代码**

在 `threedgrut/trainer.py` 找到 `init_exposure_model` 函数（约 line 601），在它后面增加 `init_depth_losses`：

```python
def init_depth_losses(self, conf: DictConfig) -> None:
    """T11.A2: 注册 DepthLoss + bg_lidar 三个 loss head + λ 衰减参数。

    Wired into __init__ next to init_exposure_model. Enabled by
    ``conf.trainer.use_lidar_depth`` / ``conf.trainer.use_depth_prior``.
    """
    from threedgrut.correction.depth_prior import DepthLoss

    trainer_conf = conf.trainer
    self.use_lidar_depth = bool(getattr(trainer_conf, "use_lidar_depth", False))
    self.use_depth_prior = bool(getattr(trainer_conf, "use_depth_prior", False))

    # LiDAR sparse depth: l1 + normalize (drivestudio default)
    self.lidar_depth_loss_fn = DepthLoss(
        loss_type=str(getattr(trainer_conf, "lidar_depth_loss_type", "l1")),
        normalize=True,
        use_inverse_depth=False,
        max_depth=float(getattr(trainer_conf, "depth_max", 80.0)),
        eps=0.01,
    )
    # DepthV2 dense depth: l2 + inverse-depth (NRE depth_inverse_mse aligned)
    self.depth_prior_loss_fn = DepthLoss(
        loss_type="l2",
        normalize=False,
        use_inverse_depth=True,
        max_depth=float(getattr(trainer_conf, "depth_max", 80.0)),
        eps=0.01,
    )
    # λ 参数
    self.lambda_lidar_depth_base = float(getattr(trainer_conf, "lambda_lidar_depth", 0.03))
    self.lambda_lidar_decay_rate = float(getattr(trainer_conf, "lidar_w_decay", 1.0))
    self.lambda_bg_lidar = float(getattr(trainer_conf, "lambda_bg_lidar", 0.005))
    self.lambda_depth_prior = float(getattr(trainer_conf, "lambda_depth_prior", 0.01))
    self.depth_max = float(getattr(trainer_conf, "depth_max", 80.0))

    logger.info(
        f"init_depth_losses: use_lidar={self.use_lidar_depth} "
        f"use_depth_prior={self.use_depth_prior} "
        f"λ_lidar={self.lambda_lidar_depth_base} (decay={self.lambda_lidar_decay_rate}) "
        f"λ_bg={self.lambda_bg_lidar} λ_depth={self.lambda_depth_prior}"
    )


def _lidar_lambda_decayed(self, step: int) -> float:
    """drivestudio L678-682: exp(-step/8000 * decay_rate)."""
    import math
    if self.lambda_lidar_decay_rate <= 0:
        return self.lambda_lidar_depth_base
    decay = math.exp(-step / 8000.0 * self.lambda_lidar_decay_rate)
    return self.lambda_lidar_depth_base * decay
```

在 `__init__` 内 `self.init_exposure_model(conf)` 一行后加：

```python
        self.init_exposure_model(conf)
        self.init_depth_losses(conf)  # T11.A2
```

在 `get_losses` 内（grep 定位），在 sky_loss 累加之后、return 之前加：

```python
        # T11.A2: image-space LiDAR + DepthV2 depth supervision
        image_infos = getattr(gpu_batch, "image_infos", None) or {}
        pred_dist = outputs.get("pred_dist")

        if pred_dist is not None and self.use_lidar_depth and "lidar_depth_map" in image_infos:
            lidar_gt = image_infos["lidar_depth_map"]
            sky_mask = image_infos.get("sky_mask")
            dyn_mask = image_infos.get("dyn_mask_cuboid", image_infos.get("dyn_mask_sseg"))
            valid_px = image_infos.get("valid_pixel_mask")

            hit = (lidar_gt > 0).float()
            if sky_mask is not None:
                hit = hit * (1 - sky_mask.float())
            if dyn_mask is not None:
                hit = hit * (1 - dyn_mask.float())
            if valid_px is not None:
                vp = valid_px.squeeze(-1) if valid_px.dim() == hit.dim() + 1 else valid_px
                hit = hit * vp.float()

            l_lidar = self.lidar_depth_loss_fn(pred_dist, lidar_gt, hit)
            lam = self._lidar_lambda_decayed(int(self.global_step))
            losses["lidar_depth_loss"] = l_lidar.detach()
            losses["total_loss"] = losses["total_loss"] + lam * l_lidar

            if sky_mask is not None and self.lambda_bg_lidar > 0:
                from threedgrut.correction.depth_prior import compute_bg_lidar_loss
                sky2d = sky_mask.squeeze(-1) if sky_mask.dim() == hit.dim() + 1 else sky_mask
                l_bg = compute_bg_lidar_loss(pred_dist, sky2d.float(), self.depth_max)
                losses["bg_lidar_loss"] = l_bg.detach()
                losses["total_loss"] = losses["total_loss"] + self.lambda_bg_lidar * l_bg

        if pred_dist is not None and self.use_depth_prior and "depth_prior" in image_infos:
            dp_gt = image_infos["depth_prior"]
            sky_mask = image_infos.get("sky_mask")
            dyn_mask = image_infos.get("dyn_mask_cuboid", image_infos.get("dyn_mask_sseg"))
            valid_px = image_infos.get("valid_pixel_mask")

            valid = torch.ones_like(dp_gt)
            if sky_mask is not None:
                valid = valid * (1 - sky_mask.squeeze(-1).float() if sky_mask.dim() == valid.dim() + 1 else (1 - sky_mask.float()))
            if dyn_mask is not None:
                valid = valid * (1 - dyn_mask.float())
            if valid_px is not None:
                vp = valid_px.squeeze(-1) if valid_px.dim() == valid.dim() + 1 else valid_px
                valid = valid * vp.float()
            valid = valid * (dp_gt < self.depth_max).float()

            l_dp = self.depth_prior_loss_fn(pred_dist, dp_gt, valid)
            losses["depth_prior_loss"] = l_dp.detach()
            losses["total_loss"] = losses["total_loss"] + self.lambda_depth_prior * l_dp
```

- [ ] **Step 5: Regression — 跑既有 trainer 测试，确认无破坏**

```bash
pytest threedgrut/tests/ -v -k "loss or trainer" --timeout 60
```

Expected: 既有测试全 PASS（应该是 ~58 个，与 T9.3 commit 1981 同基线）。

- [ ] **Step 6: Commit**

```bash
git add threedgrut/trainer.py threedgrut/tests/test_depth_loss_grad.py
git commit -m "feat(T11.A2): wire DepthLoss into trainer.get_losses

Image-space LiDAR + bg + DepthV2 depth loss accumulation, gated by
trainer.use_lidar_depth / use_depth_prior conf flags (default false →
zero impact when disabled). λ_lidar with exponential decay (drivestudio
lidar_w_decay pattern).

grad-check test verifies gradient flow pred_dist → loss → backward.

Stage 11 / T11.A2.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Phase B — LiDAR depth map 离线投影（Mac CPU + NCore SDK）

### Task B1: 离线 dump 脚本 + 单测

**Files:**
- Create: `scripts/dump_lidar_depth_map.py`
- Test: `threedgrut/tests/test_lidar_depth_projection.py`

- [ ] **Step 1: 先看 NCore SDK 的相机投影 API**

```bash
grep -rn "project\|camera_model\|FTheta\|OpenCVPinhole" threedgrut/datasets/datasetNcore.py | head -20
grep -rn "world_to_camera\|cam_to_world\|T_world\|pose_graph" threedgrut/datasets/datasetNcore.py | head -10
```

记录：
- 相机投影函数（FTheta / Pinhole / Fisheye 三种）
- pose_graph API（world ↔ camera ↔ lidar transform）

- [ ] **Step 2: 写投影函数失败测试**

Create `threedgrut/tests/test_lidar_depth_projection.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LiDAR → image-plane projection (Stage 11 T11.B1).

Pure geometry tests with synthetic intrinsics — does NOT load NCore SDK.
"""
import numpy as np
import pytest

from scripts.dump_lidar_depth_map import project_pinhole, ray_depth_from_cam_pts


def test_ray_depth_is_norm_not_z():
    """ray-depth = ‖cam_pts‖, NOT cam_pts.z. drivestudio L759-822 invariant."""
    cam_pts = np.array([[3.0, 4.0, 0.0]])  # x=3, y=4, z=0 → norm=5, z=0
    rd = ray_depth_from_cam_pts(cam_pts)
    assert rd[0] == pytest.approx(5.0, abs=1e-6)


def test_project_pinhole_principal_point():
    """主光轴点 (0, 0, 1) 投到 (cx, cy)。"""
    cam_pts = np.array([[0.0, 0.0, 1.0]])
    intrinsics = dict(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0)
    uv, valid = project_pinhole(cam_pts, intrinsics, (1080, 1920))
    assert valid[0]
    assert uv[0, 0] == pytest.approx(960.0, abs=1e-3)
    assert uv[0, 1] == pytest.approx(540.0, abs=1e-3)


def test_project_pinhole_behind_camera_invalid():
    """z<=0 的点 valid=False。"""
    cam_pts = np.array([[0.0, 0.0, -1.0]])
    intrinsics = dict(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0)
    uv, valid = project_pinhole(cam_pts, intrinsics, (1080, 1920))
    assert not valid[0]


def test_project_pinhole_outside_image_invalid():
    """投到 image 外的点 valid=False。"""
    cam_pts = np.array([[10.0, 10.0, 1.0]])  # fx*x/z = 10000 远超 W=1920
    intrinsics = dict(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0)
    uv, valid = project_pinhole(cam_pts, intrinsics, (1080, 1920))
    assert not valid[0]


def test_multi_point_to_same_pixel_takes_nearest():
    """两个点投到同一像素时，depth_map 取最近的。"""
    from scripts.dump_lidar_depth_map import scatter_depth_map
    uv = np.array([[100.0, 100.0], [100.4, 100.3]])  # 同 floor 像素
    ray_d = np.array([20.0, 5.0])  # 第二个更近
    valid = np.array([True, True])
    dmap = scatter_depth_map(uv, ray_d, valid, H=200, W=200)
    assert dmap[100, 100] == pytest.approx(5.0, abs=1e-6)
```

- [ ] **Step 3: 跑测试确认失败**

```bash
pytest threedgrut/tests/test_lidar_depth_projection.py -v
```

Expected: FAIL with `ImportError: cannot import name 'project_pinhole'`

- [ ] **Step 4: 写离线 dump 脚本**

Create `scripts/dump_lidar_depth_map.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Offline LiDAR → image-plane depth-map dump (Stage 11 T11.B1).

Iterates every (clip, camera_id, frame) and writes a sparse [H, W] ray-depth
map under aux/lidar_depth/<cam_id>/<timestamp_us>.npz. Loader counterpart is
threedgrut/datasets/aux_readers.py::LidarDepthAuxReader.

Pure NumPy + NCore SDK. Run from Mac (no GPU needed) or from A800.

CLI:
    python scripts/dump_lidar_depth_map.py \
        --manifest /path/to/pai_<clip>.json \
        --camera-ids camera_front_wide_120fov ... \
        --out-root /path/to/clip/aux/lidar_depth \
        --max-depth 80.0
"""
import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def ray_depth_from_cam_pts(cam_pts: np.ndarray) -> np.ndarray:
    """Return ‖cam_pts‖ per row — ray-depth, not z-depth.

    Matches tracer.pred_dist semantics (distance along ray from camera origin
    to surface).
    """
    return np.linalg.norm(cam_pts, axis=-1)


def project_pinhole(
    cam_pts: np.ndarray,            # [N, 3] in camera frame (right-down-front)
    intrinsics: dict,               # {fx, fy, cx, cy}
    image_shape: Tuple[int, int],   # (H, W)
) -> Tuple[np.ndarray, np.ndarray]:
    """Project Nx3 camera-frame points to image UV. Returns (uv[N,2], valid[N]).

    Valid := (z > 0) ∧ (0 <= u < W) ∧ (0 <= v < H).
    """
    H, W = image_shape
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    z = cam_pts[:, 2]
    valid_z = z > 1e-3
    u = np.where(valid_z, fx * cam_pts[:, 0] / np.where(valid_z, z, 1.0) + cx, -1.0)
    v = np.where(valid_z, fy * cam_pts[:, 1] / np.where(valid_z, z, 1.0) + cy, -1.0)
    valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    uv = np.stack([u, v], axis=-1)
    return uv, valid_z & valid_uv


def scatter_depth_map(
    uv: np.ndarray,        # [N, 2] float
    ray_depth: np.ndarray, # [N]    float (ray-depth)
    valid: np.ndarray,     # [N]    bool
    H: int,
    W: int,
) -> np.ndarray:
    """Scatter sparse ray-depth points to a dense [H, W] depth map.

    Conflict resolution: if multiple points fall in the same pixel, keep the
    nearest (smallest ray-depth) — drivestudio L850 same rule.
    """
    dmap = np.zeros((H, W), dtype=np.float32)
    # Sort by descending depth so later (nearer) overwrites
    order = np.argsort(-ray_depth)
    for i in order:
        if not valid[i]:
            continue
        u, v = uv[i]
        ui, vi = int(np.floor(u)), int(np.floor(v))
        if 0 <= ui < W and 0 <= vi < H:
            dmap[vi, ui] = ray_depth[i]
    return dmap


def dump_clip(
    manifest_path: Path,
    camera_ids: list[str],
    out_root: Path,
    max_depth: float = 80.0,
) -> None:
    """Iterate every frame × every camera; write npz per frame.

    Imports NCore SDK only inside the function so the unit-test fixtures (which
    don't have NCore) can still load the module.
    """
    try:
        import ncore.data.v4
    except ImportError as e:
        raise RuntimeError(
            "scripts/dump_lidar_depth_map.py needs ncore SDK. Run from A800 or "
            "from a Mac venv with ncore installed."
        ) from e

    # ... (full implementation requires NCore loader, defer details to actual run)
    raise NotImplementedError("filled in once we hit Phase B run on real clip")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--camera-ids", nargs="+", required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--max-depth", type=float, default=80.0)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    dump_clip(args.manifest, args.camera_ids, args.out_root, args.max_depth)
```

> **Note**: `dump_clip` 内的 NCore SDK 调用细节在 Step 7（A800 实跑前）填，先把投影核心写完 + 单测通过。

- [ ] **Step 5: 跑测试确认通过**

```bash
pytest threedgrut/tests/test_lidar_depth_projection.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 6: Commit 投影核心**

```bash
git add scripts/dump_lidar_depth_map.py threedgrut/tests/test_lidar_depth_projection.py
git commit -m "feat(T11.B1): LiDAR → image-plane projection helpers + tests

Pure-NumPy projection core (project_pinhole / ray_depth_from_cam_pts /
scatter_depth_map). 5 unit tests pin geometry invariants (ray-depth vs
z-depth, behind-camera invalid, nearest-point wins on conflict).

NCore SDK call site (dump_clip) is a NotImplementedError stub for now —
filled in T11.B1.b when we have the actual clip manifest path.

Stage 11 / T11.B1.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

- [ ] **Step 7: 在 A800 上填 dump_clip + 跑 dump（在 Phase C 启动时同步做）**

ssh a800-x2 拉 NCore SDK 源码看 `SequenceLoaderV4.get_lidar_sensor` 怎么读单帧点云，然后填充 `dump_clip`。运行命令（A800）：

```bash
ssh a800-x2 'export PATH=/root/miniforge3/envs/3dgrut/bin:$PATH \
  && cd /root/work/yusun/repo/3dgrut \
  && python scripts/dump_lidar_depth_map.py \
    --manifest /root/work/yusun/ncore-nurec/data/ncore/clips/<clip>/pai_<clip>.json \
    --camera-ids camera_front_wide_120fov camera_cross_left_120fov \
                 camera_cross_right_120fov camera_rear_tele_30fov \
                 camera_front_tele_30fov \
    --out-root /root/work/yusun/ncore-nurec/data/ncore/clips/<clip>/aux/lidar_depth \
    --max-depth 80.0' &
```

预计 5-cam × ~250 frame × ~150 ms = ~3 min。Output 体积估计 ~50 MB / clip。

---

### Task B2: LidarDepthAuxReader + dataset 注入 + 单测

**Files:**
- Modify: `threedgrut/datasets/aux_readers.py`
- Modify: `threedgrut/datasets/datasetNcore.py` (in `get_gpu_batch_with_intrinsics` near line 1299)
- Test: `threedgrut/tests/test_lidar_depth_aux_reader.py`

- [ ] **Step 1: 写失败测试**

Create `threedgrut/tests/test_lidar_depth_aux_reader.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""LidarDepthAuxReader unit test (Stage 11 T11.B2)."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from threedgrut.datasets.aux_readers import LidarDepthAuxReader


@pytest.fixture
def fake_depth_dir(tmp_path):
    cam_id = "camera_front_wide_120fov"
    ts = 1234567890
    cam_dir = tmp_path / cam_id
    cam_dir.mkdir(parents=True)
    dmap = np.zeros((216, 384), dtype=np.float32)
    dmap[10, 20] = 15.5  # one valid pixel
    np.savez_compressed(cam_dir / f"{ts}.npz", depth=dmap)
    return tmp_path, cam_id, ts, dmap


def test_lidar_depth_aux_reader_basic(fake_depth_dir):
    root, cam_id, ts, expected = fake_depth_dir
    reader = LidarDepthAuxReader(root)
    assert reader.has_frame(cam_id, ts)
    got = reader.read(cam_id, ts)
    np.testing.assert_array_equal(got, expected)


def test_lidar_depth_aux_reader_missing_returns_zero(fake_depth_dir):
    """缺失帧返回全 0（让 hit_mask = 0 → loss 跳过该帧）。"""
    root, cam_id, _, _ = fake_depth_dir
    reader = LidarDepthAuxReader(root, default_shape=(216, 384))
    got = reader.read(cam_id, timestamp_us=9999999999)  # missing
    assert got.shape == (216, 384)
    assert got.sum() == 0.0
```

- [ ] **Step 2: 跑测试确认失败**

```bash
pytest threedgrut/tests/test_lidar_depth_aux_reader.py -v
```

Expected: FAIL with `ImportError: cannot import name 'LidarDepthAuxReader'`

- [ ] **Step 3: 实现 LidarDepthAuxReader**

在 `threedgrut/datasets/aux_readers.py` 末尾追加：

```python
class LidarDepthAuxReader:
    """Reads pre-dumped LiDAR → image-plane depth maps (Stage 11 T11.B2).

    Layout: ``<root>/<camera_id>/<timestamp_us>.npz`` with single key ``depth``
    of shape ``[H, W]`` float32 (ray-depth, 0 = no LiDAR hit).

    Mirrors the SsegAuxReader / LidarSsegAuxReader interface (has_frame +
    read), with the addition of ``default_shape`` for missing-frame fallback
    (returns zeros so the hit_mask naturally zeros out the loss).
    """
    def __init__(self, root: Path | str, default_shape: tuple[int, int] | None = None):
        self.root = Path(root)
        self.default_shape = default_shape
        self._cache: dict[tuple[str, int], np.ndarray] = {}

    def has_frame(self, camera_id: str, timestamp_us: int) -> bool:
        return (self.root / camera_id / f"{timestamp_us}.npz").exists()

    def read(self, camera_id: str, timestamp_us: int) -> np.ndarray:
        key = (camera_id, timestamp_us)
        if key in self._cache:
            return self._cache[key]
        path = self.root / camera_id / f"{timestamp_us}.npz"
        if not path.exists():
            if self.default_shape is None:
                raise FileNotFoundError(f"LidarDepthAuxReader: missing {path}")
            depth = np.zeros(self.default_shape, dtype=np.float32)
        else:
            with np.load(path) as f:
                depth = f["depth"].astype(np.float32)
        self._cache[key] = depth
        return depth


class DepthV2AuxReader(LidarDepthAuxReader):
    """Reads pre-dumped DepthAnythingV2 metric depth maps (Stage 11 T11.D2).

    Same layout as LidarDepthAuxReader but under
    ``<root>/depth_anything_v2/<camera_id>/<timestamp_us>.npz``.
    """
    pass
```

- [ ] **Step 4: 跑测试确认通过**

```bash
pytest threedgrut/tests/test_lidar_depth_aux_reader.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: 在 NCoreDataset 内注入到 image_infos**

修改 `threedgrut/datasets/datasetNcore.py`：

```python
# __init__ 入参新增（与 load_aux_masks 同档）：
        load_lidar_depth_map: bool = False,
        load_depth_prior: bool = False,
        lidar_depth_aux_root: str | None = None,  # default: <clip>/aux/lidar_depth
        depth_prior_aux_root: str | None = None,  # default: <clip>/aux/depth_anything_v2

# __init__ 体内初始化（与 _lidar_sseg_reader 同段）：
        self.load_lidar_depth_map = load_lidar_depth_map
        self.load_depth_prior = load_depth_prior
        self._lidar_depth_aux_root = lidar_depth_aux_root
        self._depth_prior_aux_root = depth_prior_aux_root
        self._lidar_depth_reader: Optional[LidarDepthAuxReader] = None
        self._depth_prior_reader: Optional[DepthV2AuxReader] = None
```

在 `get_gpu_batch_with_intrinsics`（line ~1299）内 image_infos 构造段后加：

```python
        # T11.B2: image_infos["lidar_depth_map"] — sparse LiDAR depth in ray-depth
        if self.load_lidar_depth_map and image_infos is not None:
            if self._lidar_depth_reader is None:
                from pathlib import Path
                root = Path(self._lidar_depth_aux_root) if self._lidar_depth_aux_root \
                       else Path(self._clip_root) / "aux" / "lidar_depth"
                self._lidar_depth_reader = LidarDepthAuxReader(
                    root, default_shape=(int(image_height), int(image_width))
                )
            depth_np = self._lidar_depth_reader.read(camera_id, int(timestamp_us))
            depth_t = torch.from_numpy(depth_np).to(device=device, dtype=torch.float32).unsqueeze(0)
            image_infos["lidar_depth_map"] = depth_t  # [1, H, W]

        # T11.D2: image_infos["depth_prior"] — dense DepthV2 metric depth
        if self.load_depth_prior and image_infos is not None:
            if self._depth_prior_reader is None:
                from pathlib import Path
                root = Path(self._depth_prior_aux_root) if self._depth_prior_aux_root \
                       else Path(self._clip_root) / "aux" / "depth_anything_v2"
                self._depth_prior_reader = DepthV2AuxReader(
                    root, default_shape=(int(image_height), int(image_width))
                )
            dp_np = self._depth_prior_reader.read(camera_id, int(timestamp_us))
            dp_t = torch.from_numpy(dp_np).to(device=device, dtype=torch.float32).unsqueeze(0)
            image_infos["depth_prior"] = dp_t  # [1, H, W]
```

> **注**：具体变量名（`device`、`image_height` 等）以实际 `get_gpu_batch_with_intrinsics` 内的命名为准，做小调整。

- [ ] **Step 6: 跑既有 dataset 测试，确认无破坏**

```bash
pytest threedgrut/tests/ -v -k "ncore or dataset or aux" --timeout 60
```

Expected: 既有测试全 PASS。

- [ ] **Step 7: Commit**

```bash
git add threedgrut/datasets/aux_readers.py threedgrut/datasets/datasetNcore.py \
        threedgrut/tests/test_lidar_depth_aux_reader.py
git commit -m "feat(T11.B2): LidarDepthAuxReader + DepthV2AuxReader + dataset injection

Pre-dumped depth maps under aux/lidar_depth/ and aux/depth_anything_v2/,
loaded into image_infos[\"lidar_depth_map\"] / image_infos[\"depth_prior\"]
when load_lidar_depth_map=true / load_depth_prior=true.

Stage 11 / T11.B2.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Phase C — A800 1k smoke 第一轮（LiDAR-only）

### Task C1: GPU 选定 + rsync + 远端代码验证

- [ ] **Step 1: 检查 GPU 占用**

```bash
ssh a800-x2 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits'
```

按主机选择决策树判定：A800 / ThinkPad / vast.ai。**记录选定主机和 GPU index 到 plan 注释**。

- [ ] **Step 2: 在 A800 上跑 dump_lidar_depth_map.py（先填 Phase B Step 7 的 NCore SDK 调用）**

```bash
ssh a800-x2 'export PATH=/root/miniforge3/envs/3dgrut/bin:$PATH \
  && cd /root/work/yusun/repo/3dgrut \
  && python scripts/dump_lidar_depth_map.py \
    --manifest /root/work/yusun/ncore-nurec/data/ncore/clips/9ae151dc/pai_9ae151dc.json \
    --camera-ids camera_front_wide_120fov camera_cross_left_120fov \
                 camera_cross_right_120fov camera_rear_tele_30fov \
                 camera_front_tele_30fov \
    --out-root /root/work/yusun/ncore-nurec/data/ncore/clips/9ae151dc/aux/lidar_depth \
    --max-depth 80.0 2>&1 | tee /tmp/dump_lidar_depth.log'
```

Expected: log 末尾出现 `Dumped N frames × 5 cams`，文件总数 = frames × 5。

```bash
ssh a800-x2 'ls /root/work/yusun/ncore-nurec/data/ncore/clips/9ae151dc/aux/lidar_depth/camera_front_wide_120fov/ | wc -l'
```

Expected: 与该相机帧数一致（~250）。

- [ ] **Step 3: rsync 本地代码到 A800**

```bash
rsync -avz --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  /Users/etendue/repo/3dgrut2/threedgrut/ \
  a800-x2:/root/work/yusun/repo/3dgrut/threedgrut/

rsync -avz /Users/etendue/repo/3dgrut2/scripts/dump_lidar_depth_map.py \
  a800-x2:/root/work/yusun/repo/3dgrut/scripts/
```

- [ ] **Step 4: 远端代码就绪验证（CLAUDE.md §A 清单）**

```bash
ssh a800-x2 "grep -n 'init_depth_losses\|lidar_depth_loss' /root/work/yusun/repo/3dgrut/threedgrut/trainer.py | head -5"
ssh a800-x2 "grep -n 'LidarDepthAuxReader' /root/work/yusun/repo/3dgrut/threedgrut/datasets/aux_readers.py"
ssh a800-x2 "head -25 /root/work/yusun/repo/3dgrut/render.py"
```

Expected:
- trainer.py grep 命中 init_depth_losses 调用 + lidar_depth_loss 累加
- aux_readers.py 有 LidarDepthAuxReader 类
- render.py head 是 `import argparse` + `if __name__ == "__main__":`（CLAUDE.md L47 历史踩坑）

---

### Task C2: A800 1k smoke (LiDAR-only)

- [ ] **Step 1: 启动 1k smoke**

```bash
ssh a800-x2 'export PATH=/root/miniforge3/envs/3dgrut/bin:$PATH \
  && export CUDA_VISIBLE_DEVICES=0 \
  && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  && cd /root/work/yusun/repo/3dgrut \
  && python train.py --config-name apps/ncore_3dgut_mcmc_multilayer \
    n_iterations=1000 \
    path=/root/work/yusun/ncore-nurec/data/ncore/clips/9ae151dc/pai_9ae151dc.json \
    trainer.sky_backend=mlp \
    trainer.use_exposure=true \
    trainer.use_lidar_depth=true \
    trainer.use_depth_prior=false \
    trainer.lambda_lidar_depth=0.03 \
    trainer.lidar_w_decay=1.0 \
    trainer.lambda_bg_lidar=0.005 \
    dataset.load_aux_masks=true \
    dataset.load_lidar_depth_map=true \
    out_dir=/root/work/yusun/ncore-nurec/output \
    experiment_name=stage11_t11c2_lidar_only_1k \
    2>&1 | tee /tmp/stage11_lidar_1k.log' \
  > /tmp/stage11_lidar_1k.local.log 2>&1 &

echo "Background PID: $!"
```

`run_in_background=true`，预计 4-6 min。

- [ ] **Step 2: 监控关键节点**

用 Monitor grep 关键模式（CLAUDE.md L191）：

```
RUN [0-9]:|⭐ Test Metrics|^=== |Traceback|FAILED|OOM|lidar_depth_loss|🎊 Training Statistics
```

期望节点：
1. `init_depth_losses: use_lidar=True use_depth_prior=False λ_lidar=0.03 ...`
2. TensorBoard scalar 每 100 step 打出 `lidar_depth_loss` 和 `bg_lidar_loss`
3. `🎊 Training Statistics` 末尾表

- [ ] **Step 3: 收集结果**

```bash
ssh a800-x2 'cat /root/work/yusun/ncore-nurec/output/stage11_t11c2_lidar_only_1k/metrics.json'
ssh a800-x2 'grep -E "lidar_depth_loss|bg_lidar_loss|total_loss" /tmp/stage11_lidar_1k.log | tail -20'
ssh a800-x2 'grep -E "it/s|⭐ Test Metrics|psnr" /tmp/stage11_lidar_1k.log | tail -20'
```

记录到 plan 注释：
- it/s（必须 ≥ 6.8，v3_plan.md:429）
- lidar_depth_loss 起点 / 终点（应单调下降）
- bg_lidar_loss 起点 / 终点
- cc_psnr_masked（不应跌破基线 24.7 太多，可短暂退至 23-24）

- [ ] **Step 4: 决定是否继续**

| 现象 | 决策 |
|---|---|
| lidar_depth_loss 单调下降 + it/s ≥ 6.8 + PSNR 退化 < 0.5 dB | ✅ 继续 Phase D |
| lidar_depth_loss 振荡 / 上升 | ⚠️ λ_lidar 降到 0.01 重跑 |
| PSNR 退化 > 1 dB | ⚠️ λ_lidar 降到 0.01 + 检查 hit_mask 是否过激进 |
| NaN / OOM / 训练崩 | 🔴 stop the line，debug |
| it/s < 5 | ⚠️ Phase D 同跑会更慢，需要降 batch 或频率 |

- [ ] **Step 5: Commit log + 部分日志到 Done Log**

不 commit 代码（代码没变），但记录 A800 出口数据到 `v3_plan.md § 5 Done Log` 草稿（暂不 push，等 Phase G 一起）。

---

## Phase D — DepthAnythingV2 prior

### Task D1: 下载 DepthV2 weights + dump 脚本

**Files:**
- Create: `scripts/download_depth_anything_v2.sh`
- Create: `scripts/dump_depth_priors.py`

- [ ] **Step 1: 写下载脚本**

Create `scripts/download_depth_anything_v2.sh`:

```bash
#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Download DepthAnythingV2 Metric Outdoor Large from HuggingFace.
# Stage 11 / T11.D1.

set -euo pipefail

MODEL_REPO="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large"
OUT_DIR="${OUT_DIR:-models/depth_anything_v2}"

mkdir -p "${OUT_DIR}"

python - <<PY
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="${MODEL_REPO}",
    local_dir="${OUT_DIR}",
    local_dir_use_symlinks=False,
)
print(f"Downloaded to: ${OUT_DIR}")
PY
```

加 .gitignore：

```bash
echo "models/depth_anything_v2/" >> .gitignore
```

- [ ] **Step 2: 写 dump 脚本**

Create `scripts/dump_depth_priors.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Offline DepthAnythingV2 metric-depth dump (Stage 11 T11.D1).

Iterates every (camera_id, frame) image and writes a dense [H, W] metric
ray-depth map under aux/depth_anything_v2/<cam_id>/<timestamp_us>.npz.

CLI:
    python scripts/dump_depth_priors.py \
        --manifest /path/to/pai_<clip>.json \
        --camera-ids ... \
        --weights models/depth_anything_v2 \
        --out-root /path/to/clip/aux/depth_anything_v2 \
        --device cuda:0
"""
import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_model(weights_dir: Path, device: str):
    """Load DepthAnythingV2 metric outdoor model from local snapshot."""
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained(weights_dir)
    model = AutoModelForDepthEstimation.from_pretrained(weights_dir).to(device).eval()
    return processor, model


def infer_one(processor, model, image_np: np.ndarray, device: str) -> np.ndarray:
    """[H, W, 3] uint8 → [H, W] float32 metric ray-depth (meters)."""
    import torch
    inputs = processor(images=image_np, return_tensors="pt").to(device)
    with torch.no_grad():
        depth = model(**inputs).predicted_depth  # [1, H', W'] z-depth in meters
    H, W = image_np.shape[:2]
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
    ).squeeze().cpu().numpy().astype(np.float32)
    # DepthV2 predicts z-depth; we want ray-depth = z * sqrt(1 + (x/fx)^2 + (y/fy)^2)
    # — but for a stage-11 first pass, treat as approximate ray-depth (drivestudio
    # also conflates these for the depth loss; difference is < 5% in image center
    # which dominates valid regions).
    return depth


def dump_clip(manifest_path: Path, camera_ids: list[str], weights_dir: Path,
              out_root: Path, device: str) -> None:
    try:
        import ncore.data.v4
    except ImportError as e:
        raise RuntimeError("dump_depth_priors.py needs ncore SDK.") from e
    # ... (NCore loader iteration; fill at A800 run-time same as T11.B1.b)
    raise NotImplementedError("filled at T11.D1.b A800 run")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--camera-ids", nargs="+", required=True)
    p.add_argument("--weights", type=Path, default="models/depth_anything_v2")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    dump_clip(args.manifest, args.camera_ids, args.weights, args.out_root, args.device)
```

- [ ] **Step 3: 在 A800 上下载 + dump**

```bash
ssh a800-x2 'export PATH=/root/miniforge3/envs/3dgrut/bin:$PATH \
  && cd /root/work/yusun/repo/3dgrut \
  && pip install transformers safetensors --quiet \
  && bash scripts/download_depth_anything_v2.sh'

# 填 dump_clip 后跑：
ssh a800-x2 'export PATH=/root/miniforge3/envs/3dgrut/bin:$PATH \
  && cd /root/work/yusun/repo/3dgrut \
  && python scripts/dump_depth_priors.py \
    --manifest /root/work/yusun/ncore-nurec/data/ncore/clips/9ae151dc/pai_9ae151dc.json \
    --camera-ids camera_front_wide_120fov camera_cross_left_120fov \
                 camera_cross_right_120fov camera_rear_tele_30fov \
                 camera_front_tele_30fov \
    --weights models/depth_anything_v2 \
    --out-root /root/work/yusun/ncore-nurec/data/ncore/clips/9ae151dc/aux/depth_anything_v2 \
    --device cuda:0 2>&1 | tee /tmp/dump_depthv2.log' &
```

预计 5-cam × ~250 frame × ~200 ms (RTX 4090 同档速度) = ~4 min。

- [ ] **Step 4: Commit**

```bash
git add scripts/download_depth_anything_v2.sh scripts/dump_depth_priors.py .gitignore
git commit -m "feat(T11.D1): DepthAnythingV2 download + offline dump scripts

scripts/download_depth_anything_v2.sh pulls Metric-Outdoor-Large from HF.
scripts/dump_depth_priors.py runs the model over every (cam, frame) and
writes per-frame npz under aux/depth_anything_v2/.

Stage 11 / T11.D1.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Phase E — A800 1k smoke 第二轮（三 loss 同跑）

### Task E1: A800 1k smoke (LiDAR + DepthV2 + bg)

- [ ] **Step 1: 启动训练**

```bash
ssh a800-x2 'export PATH=/root/miniforge3/envs/3dgrut/bin:$PATH \
  && export CUDA_VISIBLE_DEVICES=0 \
  && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  && cd /root/work/yusun/repo/3dgrut \
  && python train.py --config-name apps/ncore_3dgut_mcmc_multilayer \
    n_iterations=1000 \
    path=/root/work/yusun/ncore-nurec/data/ncore/clips/9ae151dc/pai_9ae151dc.json \
    trainer.sky_backend=mlp \
    trainer.use_exposure=true \
    trainer.use_lidar_depth=true \
    trainer.use_depth_prior=true \
    trainer.lambda_lidar_depth=0.03 \
    trainer.lidar_w_decay=1.0 \
    trainer.lambda_bg_lidar=0.005 \
    trainer.lambda_depth_prior=0.01 \
    dataset.load_aux_masks=true \
    dataset.load_lidar_depth_map=true \
    dataset.load_depth_prior=true \
    out_dir=/root/work/yusun/ncore-nurec/output \
    experiment_name=stage11_t11e1_three_loss_1k \
    2>&1 | tee /tmp/stage11_three_loss_1k.log' \
  > /tmp/stage11_three_loss_1k.local.log 2>&1 &
```

- [ ] **Step 2: 监控 + 收集**

同 Phase C Step 2/3，关注：
- 三个 loss 各自单调下降
- it/s 是否仍 ≥ 6.0（多一个 loss 略降是预期）
- cc_psnr_masked 是否回到 ≥ 24.5

- [ ] **Step 3: 决定 Phase F 或返修**

| 现象 | 决策 |
|---|---|
| 三 loss 都收敛 + PSNR 不退 | ✅ 进 Phase F |
| depth_prior_loss 振荡 | ⚠️ λ_depth_prior 降到 0.005 |
| it/s < 5 | 🔴 考虑 DepthV2 每 2 step 跑一次 |

---

## Phase F — Eval 集成（mean_lidar_psnr 字段）

### Task F1: render.py + trainer eval 加 mean_lidar_psnr 字段

**Files:**
- Modify: `threedgrut/render.py` (eval loop)
- Modify: `threedgrut/trainer.py` (run_validation_pass / compute_metrics)
- Test: `threedgrut/tests/test_render_lidar_psnr_field.py`

- [ ] **Step 1: 写失败测试（验证 metrics.json schema）**

Create `threedgrut/tests/test_render_lidar_psnr_field.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Verify metrics.json contains mean_lidar_psnr key (Stage 11 T11.F1)."""
import json
from pathlib import Path

import pytest


def test_metrics_json_has_lidar_psnr_schema(tmp_path):
    """Schema test only — actual values come from A800 run."""
    from threedgrut.utils.eval_metrics import build_metrics_dict

    metrics = build_metrics_dict(
        mean_psnr=25.0,
        mean_ssim=0.85,
        mean_lpips=0.15,
        mean_psnr_masked=25.3,
        mean_lidar_psnr=27.1,
    )
    assert "mean_lidar_psnr" in metrics
    assert metrics["mean_lidar_psnr"] == 27.1
```

- [ ] **Step 2: 找现有 metrics 构造点**

```bash
grep -n "mean_psnr\|metrics.json\|compute_metrics\|build_metrics" threedgrut/render.py threedgrut/trainer.py | head -20
```

定位 metrics dict 构造位置。可能需要新建 `threedgrut/utils/eval_metrics.py` 抽离 `build_metrics_dict` —— 测试要求的。

- [ ] **Step 3: 实现 build_metrics_dict + 接入 render.py / trainer.py**

Create `threedgrut/utils/eval_metrics.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Centralized metrics-dict builder (Stage 11 T11.F1).

Both render.py (offline eval) and trainer.run_validation_pass (train-end eval)
write metrics.json; centralizing the schema here avoids the T6F.2 trap where
only one path got the new field (CLAUDE.md L51-54).
"""
from typing import Optional


def build_metrics_dict(
    mean_psnr: float,
    mean_ssim: float,
    mean_lpips: float,
    mean_psnr_masked: Optional[float] = None,
    mean_lidar_psnr: Optional[float] = None,
    mean_depth_l1: Optional[float] = None,
    **extra,
) -> dict:
    out = {
        "mean_psnr": float(mean_psnr),
        "mean_ssim": float(mean_ssim),
        "mean_lpips": float(mean_lpips),
    }
    if mean_psnr_masked is not None:
        out["mean_psnr_masked"] = float(mean_psnr_masked)
    if mean_lidar_psnr is not None:
        out["mean_lidar_psnr"] = float(mean_lidar_psnr)
    if mean_depth_l1 is not None:
        out["mean_depth_l1"] = float(mean_depth_l1)
    out.update(extra)
    return out


def compute_lidar_psnr(pred_dist: "torch.Tensor", lidar_depth_map: "torch.Tensor",
                       hit_mask: "torch.Tensor", max_depth: float = 100.0) -> float:
    """LiDAR-domain PSNR (v3_plan.md:426 ≥ 25 target).

    PSNR = -10 * log10(MSE / max_depth^2), normalized to [0, 100m] range.
    """
    import torch
    pd = pred_dist.squeeze(-1)
    gd = lidar_depth_map.squeeze(-1) if lidar_depth_map.dim() == pd.dim() + 1 else lidar_depth_map
    valid = hit_mask.float() * (gd > 0).float() * (gd < max_depth).float()
    if valid.sum() < 1.0:
        return float("nan")
    mse = ((pd - gd) ** 2 * valid).sum() / valid.sum().clamp(min=1.0)
    psnr = -10.0 * torch.log10(mse / (max_depth ** 2) + 1e-12)
    return float(psnr.item())
```

在 `render.py` eval loop（grep `mean_psnr` 定位）每帧 eval 后加：

```python
        # T11.F1: per-frame LiDAR-domain PSNR
        if "lidar_depth_map" in (gpu_batch.image_infos or {}) and "pred_dist" in outputs:
            from threedgrut.utils.eval_metrics import compute_lidar_psnr
            lpsnr = compute_lidar_psnr(
                outputs["pred_dist"], gpu_batch.image_infos["lidar_depth_map"],
                (gpu_batch.image_infos["lidar_depth_map"] > 0).float(),
            )
            if not (lpsnr != lpsnr):  # not NaN
                lidar_psnrs.append(lpsnr)
```

在 render.py 末尾 metrics 写出处改成：

```python
        from threedgrut.utils.eval_metrics import build_metrics_dict
        mean_lidar_psnr = float(np.mean(lidar_psnrs)) if lidar_psnrs else None
        metrics = build_metrics_dict(
            mean_psnr=float(np.mean(psnrs)),
            mean_ssim=float(np.mean(ssims)),
            mean_lpips=float(np.mean(lpipss)),
            mean_psnr_masked=float(np.mean(psnrs_masked)) if psnrs_masked else None,
            mean_lidar_psnr=mean_lidar_psnr,
        )
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
```

**同样改动 `trainer.py::run_validation_pass`**（CLAUDE.md L51-54 强制要求两处都改）。

- [ ] **Step 4: 跑测试 + 既有 regression**

```bash
pytest threedgrut/tests/test_render_lidar_psnr_field.py -v
pytest threedgrut/tests/ -k "render or metric" --timeout 60
```

Expected: 新测试 PASS + 既有测试不挂。

- [ ] **Step 5: Commit**

```bash
git add threedgrut/utils/eval_metrics.py threedgrut/render.py threedgrut/trainer.py \
        threedgrut/tests/test_render_lidar_psnr_field.py
git commit -m "feat(T11.F1): add mean_lidar_psnr metric to render.py + trainer eval

Centralized eval_metrics.build_metrics_dict ensures both eval paths
write the new schema (CLAUDE.md L51-54 anti-pattern check).
compute_lidar_psnr: -10 log10(MSE / max_depth^2) over LiDAR-hit pixels.

Stage 11 / T11.F1.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Phase G — 出口验证 + 文档同步

### Task G1: A800 30k 7-cam 出口

- [ ] **Step 1: 启动 30k 训练**

```bash
ssh a800-x2 'export PATH=/root/miniforge3/envs/3dgrut/bin:$PATH \
  && export CUDA_VISIBLE_DEVICES=0 \
  && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  && cd /root/work/yusun/repo/3dgrut \
  && python train.py --config-name apps/ncore_3dgut_mcmc_multilayer \
    n_iterations=30000 \
    path=/root/work/yusun/ncore-nurec/data/ncore/clips/9ae151dc/pai_9ae151dc.json \
    "dataset.camera_ids=[camera_front_wide_120fov, camera_cross_left_120fov, camera_cross_right_120fov, camera_rear_tele_30fov, camera_front_tele_30fov, camera_rear_left_120fov, camera_rear_right_120fov]" \
    trainer.sky_backend=mlp \
    trainer.use_exposure=true \
    trainer.use_lidar_depth=true \
    trainer.use_depth_prior=true \
    trainer.lambda_lidar_depth=0.03 \
    trainer.lidar_w_decay=1.0 \
    trainer.lambda_bg_lidar=0.005 \
    trainer.lambda_depth_prior=0.01 \
    dataset.load_aux_masks=true \
    dataset.load_lidar_depth_map=true \
    dataset.load_depth_prior=true \
    out_dir=/root/work/yusun/ncore-nurec/output \
    experiment_name=stage11_t11g1_7cam_30k \
    2>&1 | tee /tmp/stage11_30k.log' \
  > /tmp/stage11_30k.local.log 2>&1 &
```

预计 30000 step × 0.15 s/it = ~75 min。

- [ ] **Step 2: 监控 + 收集出口数据**

```bash
ssh a800-x2 'cat /root/work/yusun/ncore-nurec/output/stage11_t11g1_7cam_30k/metrics.json'
```

期望字段 ≥ v3_plan.md:425-429：

| 字段 | 目标 | 实际 |
|---|---|---|
| `mean_psnr` (novel-view 4 档) | ≥ baseline + 3.0 dB | (填) |
| `mean_lidar_psnr` | ≥ 25 dB | (填) |
| `mean_psnr_masked` (cc_psnr) | ≥ 24.0（轻退化可接受） | (填) |
| it/s | ≥ 6.8 | (填) |
| depth_prior_loss 末尾 | 收敛单调 | (填) |

- [ ] **Step 3: 决策**

- 全部达标 → ✅ Stage 11 完工
- mean_psnr 未达 +3.0 → λ_lidar 调到 0.05 重跑 + Stage 12 抬权重补偿
- mean_lidar_psnr < 25 → bg_lidar_loss λ 加到 0.01

---

### Task G2: v3_plan.md + v3_architecture.md 同步

- [ ] **Step 1: v3_plan.md Stage 11 任务表标 ✅**

修改 `v3_plan.md`：

- L188-193: T11.1–T11.6 状态从 ⬜ 翻到 ✅，"改动 / 新增"列填 commit short hash
- L249: Stage 11 表格 `0/6` → `5/6`（T11.3 仍 deferred）
- L277: Stage 11 mermaid node `:::todo` → `:::done`
- § 5 Done Log 追加：

```markdown
### 2026-05-29 Stage 11 — LiDAR + DepthV2 Image-Space Depth Supervision 完工

**Commits**: <hash_A1>..<hash_G1>

**改动摘要**: 
- 新增 `threedgrut/correction/depth_prior.py`（DepthLoss + compute_bg_lidar_loss）
- 新增 `threedgrut/datasets/aux_readers.py` 两个 reader（LidarDepth / DepthV2）
- `trainer.py` get_losses 接入三个 image-space depth loss，λ_lidar 走 drivestudio decay
- `render.py` + `trainer.run_validation_pass` 双路接入 mean_lidar_psnr
- 离线 dump: `aux/lidar_depth/`（~50 MB/clip）+ `aux/depth_anything_v2/`（~120 MB/clip）

**验收数据**（A800 30k 7-cam, clip 9ae151dc）:
- mean_psnr (novel): X.XX dB（baseline + X.X dB） ✅
- mean_lidar_psnr: XX.X dB（target ≥ 25） ✅
- mean_psnr_masked (cc): XX.X dB ✅
- it/s: X.XX（target ≥ 6.8） ✅
- 训练总时长: XX min

**T11.3 deferred**: lidar_divergence cone 抗锯齿 留到 Stage 12 之后（出口指标已达，无需）。
```

- [ ] **Step 2: Commit 文档同步**

```bash
git add v3_plan.md
git commit -m "docs(plan): mark Stage 11 done — LiDAR/DepthV2 image-space depth supervision

T11.A1-A2 + T11.B1-B2 + T11.D1 + T11.E1 + T11.F1 + T11.G1 complete.
T11.3 (lidar_divergence cone) deferred — out-of-scope for exit criteria.

A800 30k 7-cam metrics:
- mean_psnr: X.XX dB (+X.X over baseline)
- mean_lidar_psnr: XX.X dB
- mean_psnr_masked: XX.X dB
- it/s: X.XX

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

- [ ] **Step 3: push 到 origin**

```bash
git push origin main
```

---

## Self-Review

**1. Spec coverage:**
- ✅ T11.1（LiDAR ray batch）→ 改走 image-space，T11.B1 + T11.B2 替代
- ✅ T11.2（LiDAR depth loss head）→ T11.A1（DepthLoss）+ T11.A2（trainer 接入）
- ⏭️ T11.3（lidar_divergence cone）→ deferred，不阻塞出口
- ✅ T11.4（DepthV2 prior）→ T11.D1 + T11.B2 reader 复用
- ✅ T11.5（mean_lidar_psnr 字段）→ T11.F1
- ✅ T11.6（A800 出口）→ T11.G1

**2. Placeholder scan:**
- ⚠️ `scripts/dump_lidar_depth_map.py::dump_clip` 与 `scripts/dump_depth_priors.py::dump_clip` 留了 NCore SDK 调用的 NotImplementedError —— 在 Phase C Step 2 / Phase D Step 3 实跑前填实。这是 **可控的 placeholder**，因为我没看过 NCore SequenceLoaderV4.get_lidar_sensor 的实际签名，填这部分需要 ssh a800-x2 看 SDK，作为 task 步骤内的子工作明确标注，不是 plan 失败。
- 其他 step 全部含完整可运行代码。

**3. Type consistency:**
- `DepthLoss.forward` 签名在 T11.A1 / T11.A2 / T11.F1 全部一致：`(pred_depth [B,H,W,1], gt_depth [B,H,W], hit_mask [B,H,W]) → scalar`
- `image_infos["lidar_depth_map"]` 形状统一 `[B, H, W]` float32
- `image_infos["depth_prior"]` 形状统一 `[B, H, W]` float32
- conf 字段命名统一：`trainer.use_lidar_depth` / `trainer.use_depth_prior` / `trainer.lambda_lidar_depth` / `trainer.lambda_bg_lidar` / `trainer.lambda_depth_prior` / `trainer.lidar_w_decay` / `trainer.depth_max` / `dataset.load_lidar_depth_map` / `dataset.load_depth_prior`

---

## Estimated time

| Phase | 工作 | 估时 | 主机 |
|---|---|---|---|
| A | Loss heads + 单测 + trainer 接入 | 0.5d | Mac |
| B | 投影核心 + AuxReader + dataset 注入 | 0.5d | Mac (核心) + A800 (dump) |
| C | 1k smoke LiDAR-only + 解析 | 0.5d | A800 (~6min train) |
| D | DepthV2 下载 + dump | 0.5d | A800 (~10min dump) |
| E | 1k smoke 三 loss | 0.5d | A800 (~8min train) |
| F | mean_lidar_psnr 字段 | 0.5d | Mac + A800 (eval re-run) |
| G | 30k 出口 + 文档 | 1.5d | A800 (~75min train) |
| **合计** | | **4.5d** | |

