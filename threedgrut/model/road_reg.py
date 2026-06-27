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
from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# A1: road-slab background exclusion (hard, gradient-free)
# ---------------------------------------------------------------------------


@dataclass
class RoadBev:
    """Cached BEV road-surface height field built from the (frozen) road layer.

    ``height`` holds the mean road z per cell; ``mask`` marks cells with road
    support. The road layer is ~frozen, so the caller builds this once and
    reuses it every optimizer step.
    """

    origin: torch.Tensor  # [2] (min_x, min_y)
    cell: float
    height: torch.Tensor  # [H, W] mean road z per cell (0 where unsupported)
    mask: torch.Tensor  # [H, W] bool, True where road support exists


def build_road_bev_height(road_xyz: torch.Tensor, cell: float = 0.20) -> RoadBev:
    """Build a BEV height grid from road-layer gaussian centers.

    Each occupied cell stores the mean road z of the centers that fall in it.
    """
    xy = road_xyz[:, :2]
    z = road_xyz[:, 2]
    origin = xy.min(dim=0).values  # [2]
    ij = torch.floor((xy - origin) / cell).long()  # [N, 2]
    H = int(ij[:, 0].max().item()) + 1
    W = int(ij[:, 1].max().item()) + 1
    n_cells = H * W
    if n_cells > 8_000_000:
        # Defensive: a far road-position outlier would explode the grid and OOM.
        # The road layer is frozen + LiDAR-init-bounded, so this should never
        # fire; raise a clear error instead of a silent CUDA OOM if it does.
        raise ValueError(
            f"build_road_bev_height: BEV grid {H}x{W}={n_cells} cells too large "
            f"(cell={cell} m); road positions likely contain an outlier."
        )
    flat = ij[:, 0] * W + ij[:, 1]
    sum_z = torch.zeros(n_cells, device=z.device, dtype=z.dtype).scatter_add_(0, flat, z)
    cnt = torch.zeros(n_cells, device=z.device, dtype=z.dtype).scatter_add_(
        0, flat, torch.ones_like(z)
    )
    height = torch.where(
        cnt > 0, sum_z / cnt.clamp(min=1.0), torch.zeros_like(sum_z)
    ).reshape(H, W)
    mask = (cnt > 0).reshape(H, W)
    return RoadBev(origin=origin, cell=float(cell), height=height, mask=mask)


def bg_in_road_slab_mask(
    bg_xyz: torch.Tensor, bev: RoadBev, band_z: float = 0.15
) -> torch.Tensor:
    """Bool[M]: background centers inside the road footprint AND within +/-band_z
    of that cell's road height. Centers outside the grid return False."""
    H, W = bev.height.shape
    xy = bg_xyz[:, :2]
    z = bg_xyz[:, 2]
    ij = torch.floor((xy - bev.origin) / bev.cell).long()
    ix, iy = ij[:, 0], ij[:, 1]
    in_bounds = (ix >= 0) & (ix < H) & (iy >= 0) & (iy < W)
    ixc = ix.clamp(0, H - 1)
    iyc = iy.clamp(0, W - 1)
    cell_supported = bev.mask[ixc, iyc] & in_bounds
    cell_z = bev.height[ixc, iyc]
    return cell_supported & (torch.abs(z - cell_z) < band_z)


# ---------------------------------------------------------------------------
# A2: image-space road-mask projection test (catches floating bg the 3D slab
# misses) — project bg centers into a training camera, test the road sseg-mask
# at the projected pixel. numpy / camera-projector based (Mac/CPU testable).
# ---------------------------------------------------------------------------


def project_bg_road_hits(bg_xyz, T_c2w, intr_dict, model_type, road_mask):
    """Bool[N]: background centers that, projected into this training camera,
    are in front + on-image AND land on a road-mask pixel.

    Args:
        bg_xyz:     [N, 3] world-frame centers (np or torch -> np).
        T_c2w:      [4, 4] camera->world (OpenCV/NCore convention).
        intr_dict:  camera intrinsics dict (FTheta or OpenCVPinhole).
        model_type: "ftheta" | "pinhole".
        road_mask:  [H, W] float mask in {0,1} at render resolution.

    Catches bg by WHERE IT PROJECTS, not its 3D height — so floating bg above
    the road that paints the road from the training view is caught. ``visible``
    from the projector already rejects behind-camera / beyond-FoV points.
    """
    import numpy as np

    bg_xyz = np.asarray(bg_xyz, dtype=np.float64)
    T_c2w = np.asarray(T_c2w, dtype=np.float64)
    rm = np.asarray(road_mask)
    while rm.ndim > 2:  # [B, H, W] (B=1) -> [H, W]
        rm = rm[0]
    Hm, Wm = int(rm.shape[0]), int(rm.shape[1])
    N = bg_xyz.shape[0]
    if N == 0:
        return np.zeros((0,), dtype=bool)

    if str(model_type) == "ftheta":
        from threedgrut_playground.utils.ftheta_projector import FthetaForwardProjector
        # NCore T_to_world is already OpenCV convention -> identity flip (NOT the
        # projector's viser default FLIP_VISER_TO_OPENCV).
        proj = FthetaForwardProjector(intr_dict, world_to_camera_flip=np.eye(4))
    else:
        from threedgrut_playground.utils.pinhole_projector import PinholeForwardProjector
        proj = PinholeForwardProjector(intr_dict)  # default flip = I

    uv, vis = proj.project_points(bg_xyz, T_c2w)  # uv [N,2], vis [N] bool
    vis = np.asarray(vis, dtype=bool)
    if uv.shape[0] != N:  # defensive: projector should return aligned [N,2]
        return np.zeros((N,), dtype=bool)

    # Scale projector pixel coords (intrinsics resolution) to road_mask res.
    res = np.asarray(intr_dict["resolution"], dtype=np.float64).ravel()
    Wi, Hi = float(res[0]), float(res[1])
    u = uv[:, 0] * (Wm / Wi)
    v = uv[:, 1] * (Hm / Hi)
    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)
    inb = vis & (ui >= 0) & (ui < Wm) & (vi >= 0) & (vi < Hm)
    uic = np.clip(ui, 0, Wm - 1)
    vic = np.clip(vi, 0, Hm - 1)
    on_road = rm[vic, uic] > 0.5
    return inb & on_road
