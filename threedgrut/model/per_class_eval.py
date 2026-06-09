# SPDX-License-Identifier: Apache-2.0
"""P0.2 / P0.3 — per-class (sseg-based) PSNR / LPIPS evaluator.

``mean_psnr`` / ``cc_psnr_masked`` aggregate over the whole frame (minus the
ego mask), so they cannot say how well the foreground *actors* — pedestrians,
riders, cyclists — or the *lane markings* are reconstructed. ``class_psnr.py``
answers that for tracked vehicles via cuboid projection, but pedestrians have
no cuboids and lane markings have no class. This module fills both gaps using
the semantic-segmentation mask that the eval dataset already loads:

* **P0.2** per-class PSNR/LPIPS over ``person`` (11), ``rider`` (12),
  ``bicycle`` (18) sseg pixels — turns "completely unmodelled" into a number.
* **P0.3** road-crop LPIPS over ``road`` (0) + ``sidewalk`` (1) pixels — PSNR
  is drowned by flat asphalt, but LPIPS is perceptually sensitive to the
  0.1–1 m lane-stripe scale (road particles are capped at 0.3 m XY ≈ 2 stripe
  widths, see layers/layer_spec.py:66), so we report both and let LPIPS carry
  the lane-sharpness signal.

Why a separate pure-tensor module (mirrors class_psnr.py): both render.py's
eval loop and the Mac unit suite need it; keeping it free of model / trainer
state and of cv2 / NCore lets the test suite exercise it without a renderer.
LPIPS is dependency-injected (``lpips_fn``) for the same reason — the module
never imports torchmetrics, and tests pass a fake.

Class IDs mirror the Cityscapes-19 table documented in
``threedgrut/datasets/ncore_semantic.py``. They are *redeclared* here rather
than imported because ``import threedgrut.datasets.ncore_semantic`` executes
``threedgrut/datasets/__init__.py`` → ``datasetNcore`` → cv2 / NCore, which is
unavailable on a dev laptop and would break the Mac test purity this module is
designed to preserve. ``test_per_class_eval.py`` pins the two tables together
so drift is caught.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F

from threedgrut.model.class_psnr import compute_psnr_in_mask

# Actor classes for P0.2 (id tuple per name; mirror ncore_semantic.py table).
DEFAULT_ACTOR_CLASS_SPECS: Dict[str, Tuple[int, ...]] = {
    "person": (11,),
    "rider": (12,),
    "bicycle": (18,),
}

# Road + sidewalk — the road-crop region for P0.3 (mirror ROAD_CLASS_IDS).
ROAD_CLASS_IDS: Tuple[int, ...] = (0, 1)

# Phase 3 lane band 半宽（px）。lane 条纹亚像素~几像素；N=8 → band ~17px 宽，
# 让 LPIPS 有足够空间上下文判断"条纹锐不锐 / 位置对不对"，又不被背景淹没。
# render.py 调用点可经 conf.render.lane_band_px 覆盖。
DEFAULT_LANE_BAND_PX: int = 8

# (a, b) are [1, 3, H, W] in [0, 1]; returns a scalar (tensor or float).
LpipsFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def class_mask_from_sseg(sseg: torch.Tensor, ids: Iterable[int]) -> torch.Tensor:
    """``[H, W]`` semantic-id map → ``[H, W]`` bool mask of pixels in ``ids``."""
    sseg_long = sseg.to(torch.long)
    mask = torch.zeros_like(sseg_long, dtype=torch.bool)
    for i in ids:
        mask |= sseg_long == int(i)
    return mask


def dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """``[H, W]`` bool/float mask → 方形结构元（边长 2*radius+1）膨胀后的 bool mask。

    纯 torch（``F.max_pool2d``）→ 本模块保持 scipy-free / Mac 可测。
    ``radius <= 0`` 原样返回（转 bool）。形状不变（padding=radius, stride=1）。
    """
    if radius <= 0:
        return mask.to(torch.bool)
    m = mask.to(torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    k = 2 * radius + 1
    # max_pool2d 用 0 填充边缘（非 -inf），但 mask 是 {0,1}，0 不会超过真实
    # True 像素 → 边界不会凭空产生 True（角点膨胀只长界内邻域）。
    dil = F.max_pool2d(m, kernel_size=k, stride=1, padding=radius)
    return dil[0, 0] > 0.5


def compute_lpips_in_mask(
    rgb_pred: torch.Tensor,   # [H, W, 3] in [0, 1]
    rgb_gt: torch.Tensor,     # [H, W, 3]
    mask: torch.Tensor,       # [H, W] float or bool — 1 inside the region
    lpips_fn: LpipsFn,
    min_pixels: int = 50,
) -> Optional[float]:
    """Masked LPIPS via GT-fill.

    torchmetrics LPIPS has no pixel-mask support, so we fill non-mask pixels
    with GT (their perceptual contribution → 0) and run LPIPS on the full
    frame — the exact pattern used for ``lpips_masked`` at
    threedgrut/render.py:608-622. Returns ``None`` when the mask covers fewer
    than ``min_pixels`` (too few pixels for a stable perceptual score).
    """
    mask_f = mask.to(rgb_pred.dtype)
    n_pix = float(mask_f.sum().item())
    if n_pix < min_pixels:
        return None
    m = mask_f.unsqueeze(-1)  # [H, W, 1]
    pred_filled = rgb_pred.clip(0, 1) * m + rgb_gt * (1.0 - m)
    # → [1, 3, H, W] to match the torchmetrics LPIPS call convention.
    pred_in = pred_filled.permute(2, 0, 1).unsqueeze(0)
    gt_in = rgb_gt.permute(2, 0, 1).unsqueeze(0)
    val = lpips_fn(pred_in, gt_in)
    return float(val.item() if torch.is_tensor(val) else val)


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    """``[H, W, 3]`` → ``[H, W]`` 亮度（Rec.601）。"""
    w = torch.tensor([0.299, 0.587, 0.114], dtype=rgb.dtype, device=rgb.device)
    return (rgb * w).sum(-1)


def _grad_mag(img2d: torch.Tensor) -> torch.Tensor:
    """``[H, W]`` → ``[H, W]`` Sobel 梯度幅值。

    使用 replicate padding，确保常数图梯度为全 0（边界不产生伪边缘）。
    """
    x = img2d.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                      dtype=img2d.dtype, device=img2d.device)
    ky = kx.t().contiguous()  # 转置即垂直 Sobel 核 [[-1,-2,-1],[0,0,0],[1,2,1]]
    x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(x_pad, kx.view(1, 1, 3, 3), padding=0)
    gy = F.conv2d(x_pad, ky.view(1, 1, 3, 3), padding=0)
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
    norm_p = gp.norm()
    norm_g = gg.norm()
    if float(norm_p.item()) <= 0.0 or float(norm_g.item()) <= 0.0:
        return None
    return float((gp @ gg / (norm_p * norm_g)).item())


def compute_per_class_metrics(
    rgb_pred: torch.Tensor,   # [H, W, 3] in [0, 1]
    rgb_gt: torch.Tensor,     # [H, W, 3]
    sseg: torch.Tensor,       # [H, W] semantic class ids
    class_specs: Dict[str, Iterable[int]],
    *,
    lpips_fn: Optional[LpipsFn] = None,
    min_pixels: int = 50,
) -> Dict[str, Dict[str, object]]:
    """Per-class PSNR (+ optional LPIPS) over sseg-derived masks.

    Args:
        rgb_pred, rgb_gt: one rendered + GT image, ``[H, W, 3]`` in ``[0, 1]``.
        sseg: ``[H, W]`` per-pixel semantic class ids (same resolution).
        class_specs: ``{name: ids}`` — e.g. ``DEFAULT_ACTOR_CLASS_SPECS`` plus
            ``{"road_crop": ROAD_CLASS_IDS}``.
        lpips_fn: optional injected LPIPS callable; ``None`` → lpips skipped.
        min_pixels: per-class stability floor (below → metrics ``None``).

    Returns:
        ``{name: {"psnr": float | None, "lpips": float | None,
                  "n_pixels": int}}``. Classes absent from the frame are still
        reported (``n_pixels`` recorded, metrics ``None``) so "measured, not
        present" is distinguishable from "not measured" — critical for the
        pedestrian floor case where most frames have zero person pixels.
    """
    out: Dict[str, Dict[str, object]] = {}
    for name, ids in class_specs.items():
        mask = class_mask_from_sseg(sseg, ids)
        n_pix = int(mask.sum().item())
        psnr = compute_psnr_in_mask(rgb_pred, rgb_gt, mask, min_pixels=min_pixels)
        lpips_v = (
            compute_lpips_in_mask(rgb_pred, rgb_gt, mask, lpips_fn, min_pixels=min_pixels)
            if lpips_fn is not None
            else None
        )
        out[name] = {"psnr": psnr, "lpips": lpips_v, "n_pixels": n_pix}
    return out
