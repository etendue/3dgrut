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
