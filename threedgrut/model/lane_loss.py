# SPDX-License-Identifier: Apache-2.0
"""Phase 3 P3.1 — lane-band sharpness loss（可微、二阶稳定）。

直接推 pred 在车道线 band 内出和 GT 一样强的边缘（Sobel 梯度幅值 L1），
是 eval KPI ``lane_grad_corr``（Pearson，只读）的**可微稳定代理**——不直接
优化 Pearson（二阶不稳 + Goodhart 自己的 KPI）。复用 per_class_eval 的纯
张量 helper（无 cv2 / NCore → Mac 可测）。契约同 ``compute_sky_loss``：无
lane / band < min_pixels → 返回 0（device/dtype 对齐、无 grad）→ trainer
可无条件调，前视-only 样本不均时不产生 NaN。

⚠️ 梯度幅值用**本模块自己的 eps-safe 版** ``_grad_mag_safe``，不复用
``per_class_eval._grad_mag``：后者是 eval-only（``no_grad``）用的，``sqrt(gx²+gy²)``
在平坦区（gx=gy=0，训练初期 road 区常见）**反向梯度 = 1/(2·0) = NaN**。
``sqrt(·+eps)`` 让平坦区梯度退化为有限的 0，不污染训练。P3.0 立锚的
``_grad_mag`` 保持不动（eval grad_corr 0.693 锚点不受影响）。
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F

from threedgrut.model.per_class_eval import (
    DEFAULT_LANE_BAND_PX,
    LANE_CLASS_IDS,
    _luma,
    class_mask_from_sseg,
    dilate_mask,
)


def _grad_mag_safe(img2d: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """``[H, W]`` → ``[H, W]`` Sobel 梯度幅值，``sqrt(·+eps)`` 防平坦区反向 NaN。

    与 ``per_class_eval._grad_mag`` 同核（replicate pad，常数图前向≈0），但
    加 ``eps`` 使平坦区（gx=gy=0）的反向梯度退化为有限 0 而非 NaN。
    """
    x = img2d.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                      dtype=img2d.dtype, device=img2d.device)
    ky = kx.t().contiguous()
    x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(x_pad, kx.view(1, 1, 3, 3), padding=0)
    gy = F.conv2d(x_pad, ky.view(1, 1, 3, 3), padding=0)
    return torch.sqrt(gx * gx + gy * gy + eps)[0, 0]


def compute_lane_sharpness_loss(
    rgb_pred: torch.Tensor,    # [H, W, 3] in [0, 1]（须可微）
    rgb_gt: torch.Tensor,      # [H, W, 3]
    lane_sseg: torch.Tensor,   # [H, W] lane 产物类 id
    lane_ids: Iterable[int] = LANE_CLASS_IDS,
    *,
    band_px: int = DEFAULT_LANE_BAND_PX,
    min_pixels: int = 50,
) -> torch.Tensor:
    """lane band 内亮度 Sobel 梯度幅值的 L1 匹配损失（标量）。"""
    raw = class_mask_from_sseg(lane_sseg, lane_ids)   # [H,W] bool
    band = dilate_mask(raw, band_px)                  # [H,W] bool
    bm = band.to(rgb_pred.dtype)
    n = bm.sum()
    if float(n.item()) < min_pixels:
        return torch.zeros((), device=rgb_pred.device, dtype=rgb_pred.dtype)
    gp = _grad_mag_safe(_luma(rgb_pred.clip(0, 1)))   # 梯度流过 pred
    gg = _grad_mag_safe(_luma(rgb_gt))
    return (torch.abs(gp - gg) * bm).sum() / (n + 1e-6)
