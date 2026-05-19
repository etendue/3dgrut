# SPDX-License-Identifier: Apache-2.0
"""Region-weighted L1 loss for layered training (T3.4).

Pure function module so unit tests can verify the partition arithmetic without
instantiating Trainer (which pulls in CUDA tracers / NCore SDK).

Semantics (D6 D7):
  - When `image_infos` is missing or lacks sky_mask, falls back to plain
    .mean() L1 (v1 byte-identical), optionally masked by `valid_mask`.
  - Otherwise partitions L1 across {bg, road, dyn} regions and SUMS the per-
    region means. Sky region is excluded (envmap takes over in Stage 5).
  - A region whose mask.sum() < `min_pixels` is dropped (numerical stability:
    edge frames where a region barely appears would otherwise inflate noise).
  - SSIM is intentionally NOT region-weighted (D7); the caller keeps SSIM
    full-image.
"""
from __future__ import annotations

from typing import Optional

import torch


def compute_layered_l1_loss(
    rgb_pred: torch.Tensor,
    rgb_gt: torch.Tensor,
    image_infos: Optional[dict] = None,
    valid_mask: Optional[torch.Tensor] = None,
    min_pixels: int = 100,
) -> torch.Tensor:
    """Region-weighted L1 (bg + road + dyn) or plain L1 fallback.

    Args:
        rgb_pred / rgb_gt: ``[..., 3]`` matching shapes, typically ``[B,H,W,3]``.
        image_infos: optional dict with keys ``"sky_mask" / "road_mask"`` and
            either ``"dyn_mask_cuboid"`` (preferred, T4.4) or ``"dyn_mask_sseg"``
            (T3.4 placeholder). Each ``[H,W]`` or broadcastable to ``rgb_pred[...,0]``.
        valid_mask: optional valid-pixel mask used in the v1-fallback path.
        min_pixels: regions with fewer pixels are dropped (D6).

    Returns:
        Scalar loss tensor on the same device/dtype as rgb_pred.
    """
    l1_chan = (rgb_pred - rgb_gt).abs().mean(dim=-1)  # mean over channels → [...,]

    if image_infos is None or "sky_mask" not in image_infos:
        # v1 byte-identical fallback
        if valid_mask is not None:
            denom = valid_mask.sum().clamp(min=1)
            return (l1_chan * valid_mask).sum() / denom
        return l1_chan.mean()

    sky = image_infos["sky_mask"].to(l1_chan.dtype)
    road = image_infos["road_mask"].to(l1_chan.dtype)
    # T4.4 will inject "dyn_mask_cuboid" (cuboid projection, tighter); T3.4
    # initial wire uses sseg fallback. If neither present, dyn region is 0.
    dyn = image_infos.get("dyn_mask_cuboid", image_infos.get("dyn_mask_sseg"))
    if dyn is None:
        dyn = torch.zeros_like(sky)
    else:
        dyn = dyn.to(l1_chan.dtype)

    if "valid_pixel_mask" in image_infos:
        valid = image_infos["valid_pixel_mask"].to(l1_chan.dtype)
    elif valid_mask is not None:
        valid = valid_mask.to(l1_chan.dtype)
    else:
        valid = torch.ones_like(sky)

    bg = valid * (1 - road) * (1 - dyn) * (1 - sky)

    def _w(mask: torch.Tensor) -> torch.Tensor:
        s = mask.sum()
        # min_pixels guard: skip noisy edge-frame regions
        if s.item() < min_pixels:
            return torch.zeros((), device=mask.device, dtype=l1_chan.dtype)
        return (l1_chan * mask).sum() / (s + 1e-6)

    # Sky region intentionally NOT included: envmap (Stage 5) takes over.
    return _w(bg) + _w(road) + _w(dyn)
