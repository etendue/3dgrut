# SPDX-License-Identifier: Apache-2.0
"""T6F.2 unit tests for masked PSNR / SSIM / LPIPS in trainer.compute_metrics.

Stage 6-fix bug: trainer.compute_metrics 始终在含 ego 车身像素的全图上算
PSNR/SSIM/LPIPS → Stage 3/4/5/6 baseline PSNR 都包含 ego 区水分.

T6F.2 修复：
  - 保留全图三指标（与历史 baseline 可比）
  - 当 Batch.mask 不为 None 时追加 psnr_masked / ssim_masked / lpips_masked
  - PSNR_masked 用解析公式 -10·log10(sum((p-g)² · m) / (sum(m) · 3))
  - SSIM / LPIPS 不支持像素级掩膜 → GT-fill：mask=False 区填 GT（差=0），
    该区 SSIM≈1 / LPIPS≈0，按面积稀释，driving-3DGS 文献广泛采用
  - mask=None → 三 masked 指标 ≡ 全图指标（byte-identical 回归）

实际 trainer.compute_metrics 依赖 torchmetrics（CPU venv 未安装）+ Hydra conf +
CUDA tracer，故本测试以 **纯函数公式复刻** + **mock criterion** 验证：
  (a) PSNR_masked 解析公式正确性（不依赖任何外部库）
  (b) GT-fill 路径走通（mock SSIM/LPIPS 验证 input rgb 被 GT 替换）
  (c) mask=None 走 byte-identical 路径（masked 三指标直接复制全图值）

trainer.py L671-704 的 T6F.2 段代码与下方 `_compute_masked_metrics` 一字一句对齐.
"""

from __future__ import annotations

from typing import Callable, Optional

import pytest
import torch

# --- 复刻 trainer.compute_metrics 中的 T6F.2 段（保持一一对应） ----------


def _compute_masked_metrics(
    rgb_pred: torch.Tensor,
    rgb_gt: torch.Tensor,
    mask: Optional[torch.Tensor],
    psnr_full: float,
    ssim_full: float,
    lpips_full: float,
    ssim_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lpips_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> dict[str, float]:
    """Mirror of trainer.compute_metrics T6F.2 段 (trainer.py:671-704)."""
    rgb_gt_full = rgb_gt.permute(0, 3, 1, 2)
    pred_rgb_full = rgb_pred.permute(0, 3, 1, 2)
    pred_rgb_full_clipped = rgb_pred.clip(0, 1).permute(0, 3, 1, 2)

    if mask is not None:
        mask = mask.to(rgb_pred.dtype)
        diff_sq = (rgb_pred - rgb_gt).pow(2) * mask
        denom = mask.sum().clamp(min=1.0) * 3
        mse_masked = diff_sq.sum() / denom
        psnr_masked = (-10.0 * torch.log10(mse_masked.clamp(min=1e-10))).item()
        m4d = mask.permute(0, 3, 1, 2)
        rgb_pred_filled = pred_rgb_full * m4d + rgb_gt_full * (1.0 - m4d)
        rgb_pred_filled_clipped = pred_rgb_full_clipped * m4d + rgb_gt_full * (1.0 - m4d)
        ssim_masked = ssim_fn(rgb_pred_filled, rgb_gt_full).item()
        lpips_masked = lpips_fn(rgb_pred_filled_clipped, rgb_gt_full).item()
    else:
        psnr_masked = psnr_full
        ssim_masked = ssim_full
        lpips_masked = lpips_full

    return {
        "psnr_masked": psnr_masked,
        "ssim_masked": ssim_masked,
        "lpips_masked": lpips_masked,
    }


def _mse_ssim_fn(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mock SSIM = 1 - MSE (sentinel for testing GT-fill path).

    真实 SSIM 是 [-1, 1]；我们用 1 - MSE 作为单调降映射验证：
      - pred == gt → MSE=0 → "SSIM"=1.0 (perfect)
      - pred 在 mask=False 区被填成 gt → 该区贡献 MSE=0
    """
    return 1.0 - (pred - gt).pow(2).mean()


def _mse_lpips_fn(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mock LPIPS = MSE (sentinel for testing GT-fill path).

    真实 LPIPS 越小越好；MSE 同方向单调（越小越好）.
    """
    return (pred - gt).pow(2).mean()


# --- 测试 ----------------------------------------------------------------


def test_masked_metrics_equal_full_when_mask_none():
    """T6F.2 byte-identical 回归：mask=None → 三 masked 指标 ≡ 全图指标.

    NeRF / Colmap 等无 ego mask 的 dataset 不引入回归.
    """
    rng = torch.Generator().manual_seed(0)
    rgb_pred = torch.rand(1, 4, 4, 3, generator=rng)
    rgb_gt = torch.rand(1, 4, 4, 3, generator=rng)
    psnr_full, ssim_full, lpips_full = 26.3, 0.88, 0.27

    out = _compute_masked_metrics(
        rgb_pred,
        rgb_gt,
        mask=None,
        psnr_full=psnr_full,
        ssim_full=ssim_full,
        lpips_full=lpips_full,
        ssim_fn=_mse_ssim_fn,
        lpips_fn=_mse_lpips_fn,
    )
    assert out["psnr_masked"] == psnr_full
    assert out["ssim_masked"] == ssim_full
    assert out["lpips_masked"] == lpips_full


def test_masked_metrics_equal_full_when_mask_all_ones():
    """T6F.2 退化：mask=全 1 → masked 三指标 = 用 (pred, gt) 直接算的全图三指标.

    PSNR_masked 解析公式与原 PSNR 完全等价 (sum_sq / (numel * 3) → -10·log10);
    SSIM/LPIPS 走 GT-fill 但 m4d=1 → rgb_filled = pred_full（无 GT 混入）→
    数值与无 mask 调用 ssim_fn(pred_full, gt_full) 完全相等.
    """
    rng = torch.Generator().manual_seed(1)
    rgb_pred = torch.rand(1, 4, 4, 3, generator=rng)
    rgb_gt = torch.rand(1, 4, 4, 3, generator=rng)
    mask = torch.ones(1, 4, 4, 1, dtype=torch.float32)

    # 直接算"全图"参考值（与 trainer.compute_metrics 全图分支一致的输入）
    rgb_gt_full = rgb_gt.permute(0, 3, 1, 2)
    pred_full = rgb_pred.permute(0, 3, 1, 2)
    pred_full_clipped = rgb_pred.clip(0, 1).permute(0, 3, 1, 2)
    psnr_ref = (-10.0 * torch.log10(((rgb_pred - rgb_gt).pow(2)).mean().clamp(min=1e-10))).item()
    ssim_ref = _mse_ssim_fn(pred_full, rgb_gt_full).item()
    lpips_ref = _mse_lpips_fn(pred_full_clipped, rgb_gt_full).item()

    out = _compute_masked_metrics(
        rgb_pred,
        rgb_gt,
        mask=mask,
        psnr_full=psnr_ref,
        ssim_full=ssim_ref,
        lpips_full=lpips_ref,
        ssim_fn=_mse_ssim_fn,
        lpips_fn=_mse_lpips_fn,
    )

    assert out["psnr_masked"] == pytest.approx(psnr_ref, abs=1e-4)
    assert out["ssim_masked"] == pytest.approx(ssim_ref, abs=1e-6)
    assert out["lpips_masked"] == pytest.approx(lpips_ref, abs=1e-6)


def test_psnr_masked_uniform_error_matches_analytic_formula():
    """T6F.2 数值正确性：mask=True 区 uniform 误差 δ → psnr_masked = -10·log10(δ²).

    构造 δ=0.1 → psnr_masked = -10·log10(0.01) = 20.0 dB.
    """
    H, W = 8, 8
    delta = 0.1
    rgb_gt = torch.zeros(1, H, W, 3)
    rgb_pred = torch.full((1, H, W, 3), delta)
    mask = torch.ones(1, H, W, 1, dtype=torch.float32)

    out = _compute_masked_metrics(
        rgb_pred,
        rgb_gt,
        mask=mask,
        psnr_full=99.0,
        ssim_full=1.0,
        lpips_full=0.0,
        ssim_fn=_mse_ssim_fn,
        lpips_fn=_mse_lpips_fn,
    )
    assert out["psnr_masked"] == pytest.approx(20.0, abs=1e-4)


def test_psnr_masked_ignores_error_in_masked_region():
    """T6F.2 关键不变量：mask=False 区造巨大误差，masked 区误差为 0 → psnr_masked 很高.

    构造：mask=True 区 pred=gt（无误差），mask=False 区 pred=1 / gt=0（最大误差）.
    psnr_masked 应该极高（→ +∞，受 1e-10 clamp 限制为 ~100 dB），
    而全图 PSNR ≈ -10·log10(0.5 * 1.0 + 0.5 * 0) = 3.01 dB（半图最大误差）.
    """
    H, W = 8, 8
    rgb_gt = torch.zeros(1, H, W, 3)
    rgb_pred = torch.zeros(1, H, W, 3)
    rgb_pred[:, H // 2 :, :, :] = 1.0  # 下半图最大误差
    mask = torch.ones(1, H, W, 1, dtype=torch.float32)
    mask[:, H // 2 :, :, :] = 0.0  # 下半图被排除

    out = _compute_masked_metrics(
        rgb_pred,
        rgb_gt,
        mask=mask,
        psnr_full=3.01,
        ssim_full=0.5,
        lpips_full=0.5,
        ssim_fn=_mse_ssim_fn,
        lpips_fn=_mse_lpips_fn,
    )
    # mask=True 区域误差 = 0 → mse → 1e-10 clamp → -10·log10(1e-10) = 100 dB
    assert out["psnr_masked"] >= 99.0, (
        f"psnr_masked={out['psnr_masked']} 应 ≥ 99 dB（mask 区无误差），" f"实际 < 99 表明 mask 没把误差排除掉"
    )


def test_ssim_masked_via_gt_fill_better_than_full_ssim():
    """T6F.2 GT-fill 单调性：mask=False 区填 GT 后 SSIM 单调改善（更接近 perfect）.

    使用 mock _mse_ssim_fn = 1 - MSE：
      - 全图 SSIM = 1 - MSE_full（mask=False 区有大误差，MSE_full 大）
      - GT-fill SSIM = 1 - MSE_filled（mask=False 区填成 GT → MSE_filled 小）
      → ssim_masked > ssim_full.
    """
    H, W = 8, 8
    rgb_gt = torch.zeros(1, H, W, 3)
    rgb_pred = torch.full((1, H, W, 3), 0.5)
    rgb_pred[:, H // 2 :, :, :] = 0.9  # ego 区大误差
    mask = torch.ones(1, H, W, 1, dtype=torch.float32)
    mask[:, H // 2 :, :] = 0.0

    rgb_gt_full = rgb_gt.permute(0, 3, 1, 2)
    pred_full = rgb_pred.permute(0, 3, 1, 2)
    ssim_full = _mse_ssim_fn(pred_full, rgb_gt_full).item()

    out = _compute_masked_metrics(
        rgb_pred,
        rgb_gt,
        mask=mask,
        psnr_full=10.0,
        ssim_full=ssim_full,
        lpips_full=0.5,
        ssim_fn=_mse_ssim_fn,
        lpips_fn=_mse_lpips_fn,
    )
    assert out["ssim_masked"] > ssim_full, (
        f"ssim_masked ({out['ssim_masked']}) 应 > ssim_full ({ssim_full})；"
        f"GT-fill 在 ego 区把 pred 替成 gt → 该区贡献 SSIM=1"
    )
    assert out["ssim_masked"] <= 1.0 + 1e-6, "SSIM 上限"


def test_lpips_masked_via_gt_fill_better_than_full_lpips():
    """T6F.2 GT-fill 单调性（LPIPS 越小越好）：

    使用 mock _mse_lpips_fn = MSE：
      - 全图 LPIPS = MSE_full (大)
      - GT-fill LPIPS = MSE_filled (小)
      → lpips_masked < lpips_full.
    """
    H, W = 8, 8
    rgb_gt = torch.zeros(1, H, W, 3)
    rgb_pred = torch.full((1, H, W, 3), 0.5)
    rgb_pred[:, H // 2 :, :, :] = 0.9
    mask = torch.ones(1, H, W, 1, dtype=torch.float32)
    mask[:, H // 2 :, :] = 0.0

    rgb_gt_full = rgb_gt.permute(0, 3, 1, 2)
    pred_full_clipped = rgb_pred.clip(0, 1).permute(0, 3, 1, 2)
    lpips_full = _mse_lpips_fn(pred_full_clipped, rgb_gt_full).item()

    out = _compute_masked_metrics(
        rgb_pred,
        rgb_gt,
        mask=mask,
        psnr_full=10.0,
        ssim_full=0.5,
        lpips_full=lpips_full,
        ssim_fn=_mse_ssim_fn,
        lpips_fn=_mse_lpips_fn,
    )
    assert out["lpips_masked"] < lpips_full, (
        f"lpips_masked ({out['lpips_masked']}) 应 < lpips_full ({lpips_full})；"
        f"GT-fill 在 ego 区把 pred 替成 gt → 该区贡献 LPIPS=0"
    )
    assert out["lpips_masked"] >= -1e-6, "LPIPS 下限"


def test_psnr_masked_mask_shape_broadcast_to_rgb():
    """T6F.2 形状契约：mask [B, H, W, 1] broadcast 到 rgb [B, H, W, 3] 不爆.

    PyTorch 自动 broadcast last dim 1→3；这也是为何 trainer 不需 .repeat.
    """
    H, W = 4, 4
    rgb_gt = torch.zeros(1, H, W, 3)
    rgb_pred = torch.full((1, H, W, 3), 0.1)
    mask = torch.ones(1, H, W, 1, dtype=torch.float32)

    out = _compute_masked_metrics(
        rgb_pred,
        rgb_gt,
        mask=mask,
        psnr_full=20.0,
        ssim_full=0.9,
        lpips_full=0.1,
        ssim_fn=_mse_ssim_fn,
        lpips_fn=_mse_lpips_fn,
    )
    # 全 mask + uniform δ=0.1 → 同 test_psnr_masked_uniform_error
    assert out["psnr_masked"] == pytest.approx(20.0, abs=1e-4)
