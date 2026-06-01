# SPDX-License-Identifier: Apache-2.0
"""V3-R1 road-layer regularizers (pure functions).

Three orthogonal regularizers used by V3-R1 Phase 1 to suppress
lane-marking blur and deformation under novel-view perturbation:

1. ``clamp_layer_scales`` — in-place scale upper bounds + anisotropy
   ratio cap applied AFTER MCMC post-optimizer step (V3-R1.2).
2. ``compute_effective_rank_loss`` — entropy-of-scale-spectrum penalty
   that softens needle-shaped Gaussians (V3-R1.3).
3. ``compute_depth_tv_loss`` — total-variation smoothness on a rendered
   depth map restricted to a road mask, used at virtual viewpoints
   (V3-R1.4).

Pure functions / no Trainer / no CUDA — safe to unit-test on Mac CPU.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from threedgrut.layers.layer_spec import LayerSpec


def clamp_layer_scales(scale_log: torch.Tensor, spec: LayerSpec) -> torch.Tensor:
    """Clamp a per-particle log-space scale tensor by the per-layer bounds.

    Args:
        scale_log: ``[N, 3]`` log-space scale parameter (model.scale).
        spec: layer descriptor; reads ``scale_xy_max`` / ``scale_z_max`` /
            ``anisotropy_ratio_max``. Any field that is None disables that
            clamp.

    Returns:
        Returns scale_log unchanged (same object) when no clamps are
        configured; otherwise a fresh clamped tensor (input never mutated).

    Notes:
        - XY/Z clamps are absolute upper bounds in physical units (exp(log)).
        - Anisotropy clamp raises the smallest eigenvalue if max/min >
          ratio_max, leaving the largest untouched. This biases toward
          larger Gaussians rather than shrinking the in-plane extent.
        - Hard XY/Z caps always hold; the anisotropy ratio is best-effort
          and is subordinate to the caps (may be exceeded when
          ratio < scale_xy_max/scale_z_max). Re-applying caps after the
          anisotropy step ensures the physical thin-disc bound always wins.
        - All three clamps are applied in order: XY/Z caps -> ratio -> XY/Z caps.
    """
    if (
        spec.scale_xy_max is None
        and spec.scale_z_max is None
        and spec.anisotropy_ratio_max is None
    ):
        return scale_log

    out = scale_log.clone()
    xy_cap = math.log(spec.scale_xy_max) if spec.scale_xy_max is not None else None
    z_cap = math.log(spec.scale_z_max) if spec.scale_z_max is not None else None

    def _apply_xyz_caps(t: torch.Tensor) -> None:
        if xy_cap is not None:
            t[:, 0].clamp_(max=xy_cap)
            t[:, 1].clamp_(max=xy_cap)
        if z_cap is not None:
            t[:, 2].clamp_(max=z_cap)

    _apply_xyz_caps(out)

    if spec.anisotropy_ratio_max is not None:
        s = torch.exp(out)
        s_max, _ = s.max(dim=-1, keepdim=True)
        floor = s_max / float(spec.anisotropy_ratio_max)
        s = torch.maximum(s, floor)
        out = torch.log(s.clamp_min(1e-12))
        # Hard XY/Z caps take precedence over the (softer) anisotropy ratio:
        # raising the min axis can push an axis back over its cap when
        # ratio < scale_xy_max/scale_z_max. Re-apply the caps so the physical
        # thin-disc bound always wins (ratio may then be slightly exceeded for
        # very tight ratios — acceptable; the caps are the stronger prior).
        _apply_xyz_caps(out)

    return out


def compute_effective_rank_loss(
    scale_log: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Spectral-entropy regularizer encouraging isotropic-ish Gaussians.

    For each particle, normalize its 3 scale eigenvalues to a probability
    simplex and compute Shannon entropy. Loss = -entropy.mean() so that
    minimizing -> push entropy up -> push Gaussians toward isotropy.

    Args:
        scale_log: ``[N, 3]`` log-scale parameter.
        mask: optional ``[N]`` bool/float mask selecting which particles
            contribute. None = all particles.

    Returns:
        Scalar tensor on the same device/dtype as scale_log.
        Values are in [-log(3), 0]; near -1.099 means near-isotropic
        (regularizer working); a training monitor will show a negative value.
    """
    s = torch.exp(scale_log)
    s = s / s.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    entropy = -(s * (s.clamp_min(1e-12).log())).sum(dim=-1)
    if mask is not None:
        m = mask.to(entropy.dtype)
        denom = m.sum().clamp_min(1.0)
        return -(entropy * m).sum() / denom
    return -entropy.mean()


def compute_depth_tv_loss(
    depth: torch.Tensor,
    road_mask: torch.Tensor,
    min_pixels: int = 100,
) -> torch.Tensor:
    """Total-variation smoothness on a rendered depth map, road-region only.

    Args:
        depth: ``[B, H, W]`` or ``[H, W]`` rendered depth (metres).
        road_mask: same spatial shape, binary {0, 1}.
        min_pixels: when road_mask.sum() < min_pixels, returns 0
            (graceful no-op for edge frames with no road).

    Returns:
        Scalar TV loss in metres, normalized by the number of road
        boundary pairs.
    """
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)
        road_mask = road_mask.unsqueeze(0)
    rm = road_mask.to(depth.dtype)
    if rm.sum().item() < min_pixels:
        return torch.zeros((), device=depth.device, dtype=depth.dtype)

    pair_h = rm[:, :, :-1] * rm[:, :, 1:]
    pair_v = rm[:, :-1, :] * rm[:, 1:, :]

    diff_h = (depth[:, :, :-1] - depth[:, :, 1:]).abs() * pair_h
    diff_v = (depth[:, :-1, :] - depth[:, 1:, :]).abs() * pair_v

    num = diff_h.sum() + diff_v.sum()
    den = pair_h.sum() + pair_v.sum() + 1e-6
    return num / den
