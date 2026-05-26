# SPDX-License-Identifier: Apache-2.0
"""T8/B3 — Background-layer opacity penalty for particles inside active
dynamic-rigid cuboids.

Why: T8/B3 baseline diagnostic shows ~10 % of `background` particles
(101,740 of 1,000,000 in the 30k ckpt) physically fall inside active vehicle
cuboids. MCMC densification is per-layer-scoped, so these particles never
migrate to `dynamic_rigids`; the only way to push them out is to depress
their opacity so MCMC `relocate_gaussians` marks them dead
(``sigmoid(density) < opacity_threshold = 0.005``, see ``mcmc.py:110-141``)
and reassigns them to alive donors elsewhere in the bg layer.

The penalty term is:
    L_bg = λ(step) * mean(sigmoid(bg.density) * inside_any_active_cuboid)
where ``inside_any_active_cuboid`` is a [N_bg] bool computed in world frame.

Lambda warmup: ``0 → λ_max`` linearly over ``warmup_iters`` (Plan-agent
recommendation, ramps slow enough that bg layer doesn't collapse while
dynamic_rigids MCMC has time to densify into the freed cuboid interiors).
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch


def lambda_schedule(step: int, lambda_max: float, warmup_iters: int) -> float:
    """Linear warmup ``0 → lambda_max`` over ``warmup_iters`` then constant.

    >>> lambda_schedule(0, 0.05, 5000)
    0.0
    >>> lambda_schedule(2500, 0.05, 5000)
    0.025
    >>> lambda_schedule(5000, 0.05, 5000)
    0.05
    >>> lambda_schedule(10000, 0.05, 5000)
    0.05
    """
    if step <= 0 or warmup_iters <= 0:
        return 0.0 if step <= 0 else float(lambda_max)
    if step >= warmup_iters:
        return float(lambda_max)
    return float(lambda_max) * (float(step) / float(warmup_iters))


def collect_active_cuboids_for_frame(
    tracks_poses: Dict[str, torch.Tensor],
    tracks_active: Dict[str, torch.Tensor],
    tracks_size: Dict[str, torch.Tensor],
    frame_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(active_poses [T, 4, 4], active_sizes [T, 3])`` for the tracks
    that are active at ``frame_idx``.

    Tracks are visited in ``sorted(tracks_poses.keys())`` order for
    determinism. Tracks whose pose tensor is too short for the requested
    frame are silently skipped (NCore loader pads correctly but defensive).
    """
    active_poses_list = []
    active_sizes_list = []
    for tid in sorted(tracks_poses.keys()):
        poses = tracks_poses[tid]
        active = tracks_active.get(tid)
        if active is None or frame_idx < 0 or frame_idx >= int(active.shape[0]):
            continue
        if not bool(active[frame_idx]):
            continue
        size = tracks_size.get(tid)
        if size is None:
            continue
        active_poses_list.append(poses[frame_idx])
        active_sizes_list.append(size)
    if not active_poses_list:
        return torch.zeros(0, 4, 4), torch.zeros(0, 3)
    return torch.stack(active_poses_list), torch.stack(active_sizes_list)


def particles_inside_any_cuboid_mask(
    positions: torch.Tensor,         # [N, 3] world frame
    active_poses: torch.Tensor,      # [T, 4, 4] obj→world (active subset)
    active_sizes: torch.Tensor,      # [T, 3] full extent
) -> torch.Tensor:
    """Return ``[N]`` BoolTensor: True iff position falls inside any of the
    T cuboids (axis-aligned in each object frame).

    Implementation: per-track loop over T to bound peak memory at
    ``O(N × 3 floats)`` rather than ``O(T × N × 3)``. T is small (≤ ~30
    active per frame in NCore), python overhead is negligible.
    """
    N = int(positions.shape[0])
    device = positions.device
    dtype = positions.dtype
    inside_any = torch.zeros(N, dtype=torch.bool, device=device)
    if N == 0 or active_poses.shape[0] == 0:
        return inside_any

    poses = active_poses.to(device=device, dtype=dtype)
    sizes = active_sizes.to(device=device, dtype=dtype)
    size_half = sizes * 0.5                                          # [T, 3]
    ones = torch.ones(N, 1, dtype=dtype, device=device)
    pts_h = torch.cat([positions, ones], dim=-1)                     # [N, 4]
    for t in range(int(poses.shape[0])):
        pose_inv = torch.linalg.inv(poses[t])
        local = (pose_inv @ pts_h.T).T[:, :3]                        # [N, 3]
        inside_t = (local.abs() <= size_half[t]).all(dim=-1)         # [N]
        inside_any |= inside_t
    return inside_any


def compute_bg_cuboid_opacity_penalty(
    bg_positions: torch.Tensor,      # [N, 3] world frame
    bg_density_raw: torch.Tensor,    # [N] or [N, 1] pre-sigmoid (the nn.Parameter)
    active_poses: torch.Tensor,      # [T, 4, 4] obj→world (active subset)
    active_sizes: torch.Tensor,      # [T, 3]
    lambda_val: float,
) -> torch.Tensor:
    """Scalar loss penalising background particles inside active cuboids.

    Loss = ``λ * mean(sigmoid(bg_density_raw) * inside_any_cuboid)``

    Returns ``torch.zeros(())`` (on bg_positions.device) when ``lambda_val == 0``
    or no active cuboids — keeps the call site uncluttered (no None handling).
    Gradient flows through ``bg_density_raw`` only; ``bg_positions`` and the
    cuboid masks are treated as constants (no_grad branch on cuboid checks).
    """
    device = bg_positions.device
    if (
        lambda_val == 0.0
        or active_poses.shape[0] == 0
        or bg_positions.shape[0] == 0
    ):
        return torch.zeros((), device=device, dtype=bg_density_raw.dtype)

    # Cuboid containment is a piecewise-constant function of bg_positions; no
    # gradient flows through it (we don't want positions to be pulled into/out
    # of the mask, only the opacity of whichever particles happen to be in).
    with torch.no_grad():
        inside_any = particles_inside_any_cuboid_mask(
            bg_positions, active_poses, active_sizes,
        )
    mask_f = inside_any.to(dtype=bg_density_raw.dtype)

    opacity = torch.sigmoid(bg_density_raw.view(-1))                  # [N]
    loss = (opacity * mask_f).mean()
    return float(lambda_val) * loss


def clamp_layer_positions_to_cuboids(
    positions: torch.Tensor,         # [N, 3] object-local frame (in-place mutated)
    track_ids: torch.Tensor,         # [N] long
    track_keys_sorted: Iterable[str],
    tracks_size: Dict[str, torch.Tensor],
) -> int:
    """In-place clamp ``positions[i]`` to ``[-size_half, +size_half]`` of its
    owning track. Returns the number of particles that were clamped (i.e.
    were outside before).

    Plan-agent Q3 recommendation: dynamic_rigids MCMC perturbation +
    add_new_gaussians can drift particles past the cuboid boundary; a
    per-step clamp keeps them physically attached to the actor. Cheap
    (one ``clamp_`` per track, ≤ 70 tracks).
    """
    if positions.numel() == 0:
        return 0
    name_to_id = {name: i for i, name in enumerate(track_keys_sorted)}
    n_clamped = 0
    for tid, size in tracks_size.items():
        if tid not in name_to_id:
            continue
        mask = (track_ids == name_to_id[tid])
        if not bool(mask.any()):
            continue
        size_half = size.to(dtype=positions.dtype, device=positions.device) / 2.0
        owned = positions[mask]
        outside_before = int((owned.abs() > size_half).any(dim=-1).sum().item())
        n_clamped += outside_before
        owned.clamp_(min=-size_half, max=size_half)
        positions[mask] = owned
    return n_clamped
