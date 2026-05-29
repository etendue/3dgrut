# SPDX-License-Identifier: Apache-2.0
"""LiDAR-domain depth PSNR for Stage 11 eval (T11.F1).

Pure-torch helper so it unit-tests on Mac CPU without the CUDA tracer. Used by
BOTH eval paths (render.py offline eval + trainer.get_metrics validation pass)
— the two-path requirement from CLAUDE.md §B L51-54 (a metric added to only one
path silently misses from metrics.json).
"""
from __future__ import annotations

import torch


def compute_lidar_psnr(
    pred_dist: torch.Tensor,        # [B,H,W,1] or [H,W,1] rendered ray-depth
    lidar_depth_map: torch.Tensor,  # [B,H,W] or [H,W] sparse GT ray-depth (0 = no hit)
    hit_mask: torch.Tensor,         # [B,H,W] or [H,W] {0,1}
    max_depth: float = 100.0,
) -> float:
    """LiDAR-domain PSNR over hit pixels: -10*log10(MSE / max_depth^2).

    Returns float('nan') when no valid hit pixels (caller skips NaN so frames /
    cameras without LiDAR coverage don't poison the mean). max_depth=100 is the
    NuRec reference normalization (v3_plan.md:426).
    """
    pd = pred_dist.squeeze(-1)
    gd = lidar_depth_map
    if gd.dim() == pd.dim() + 1:
        gd = gd.squeeze(-1)
    valid = hit_mask.float() * (gd > 0).float() * (gd < max_depth).float()
    n = valid.sum()
    if n < 1.0:
        return float("nan")
    mse = ((pd - gd) ** 2 * valid).sum() / n.clamp(min=1.0)
    psnr = -10.0 * torch.log10(mse / (max_depth ** 2) + 1e-12)
    return float(psnr.item())
