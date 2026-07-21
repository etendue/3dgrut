#!/usr/bin/env python3
"""Layer ownership metric kernel for MCRO read-only checkpoint analysis."""
from __future__ import annotations
import torch
import torch.nn.functional as F

def compute_ownership_metrics(bg_alpha, road_alpha, road_rgb, sky_contrib, road_mask, sky_mask=None, erosion_px=1, band_px=1):
    mask = road_mask.bool()
    if mask.dim() == 2: mask = mask[None, ..., None]
    if mask.dim() == 3: mask = mask.unsqueeze(-1)
    interior = mask
    if erosion_px:
        x = mask.permute(0,3,1,2).float()
        interior = (F.avg_pool2d(x, 2*erosion_px+1, stride=1, padding=erosion_px, count_include_pad=True)==1).permute(0,2,3,1)
    def stat(x, q=None):
        values=x[interior.expand_as(x)]
        if not values.numel(): return float('nan')
        return float(values.mean() if q is None else torch.quantile(values, q))
    outside = ~mask
    return {"n_valid_px": int(interior.sum()), "bg_on_road_alpha_mean":stat(bg_alpha), "bg_on_road_alpha_p50":stat(bg_alpha,.5), "bg_on_road_alpha_p90":stat(bg_alpha,.9), "road_coverage_p10":stat(road_alpha,.1), "road_coverage_p50":stat(road_alpha,.5), "road_coverage_mean":stat(road_alpha), "road_outside_alpha_mean": float(road_alpha[outside.expand_as(road_alpha)].mean()) if outside.any() else float('nan'), "sky_on_road_energy": stat(sky_contrib.abs().mean(dim=-1,keepdim=True))}
