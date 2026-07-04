# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V3 Stage B — temporal smoothness regularizer for learnable per-track pose.

Stage A (commits 67d11e6 / 8c48815) made ``LayeredGaussians.dynamic_rigids``'s
per-track per-frame cuboid SE(3) pose a learnable ``nn.Parameter`` pair
(``_track_quat_<tid>`` ``[F, 4]`` wxyz quat + ``_track_trans_<tid>`` ``[F, 3]``).
With Adam optimizing each frame independently, the photometric loss can push
the cuboid pose to a per-frame local optimum that disagrees with neighbours
→ visible jitter on dynamic objects in 4D playback ("顿挫感").

This module computes the second-order finite-difference penalty (the
DriveStudio ``RigidNodes.temporal_smooth_reg`` formulation):

    Δ²t[f] = t[f+1] − 2·t[f] + t[f−1]                       (trans)
    Δ²q[f] = q'[f+1] − 2·q[f] + q'[f−1]   (quats sign-aligned to q[f])

summed over all triple-active frames ``a[f-1] = a[f] = a[f+1] = 1``,
averaged across the batch of (track, frame) pairs, and weighted by
``λ_trans`` / ``λ_rot``. Returns a scalar tensor compatible with the other
loss contributors in :meth:`Trainer.get_losses`.

Active mask reasoning: a track entering / leaving the scene exposes the
GT pose at the boundary; smoothing across an active boundary would pull
the GT pose toward 0 and break the (already correct) prior. Stick to
strictly-interior triples.

Quat sign alignment: each rotation has two equivalent unit-quat
representations differing by sign. Without alignment a bit-flip between
adjacent frames produces a chord distance of ``‖2q‖² = 4`` even when the
underlying rotation is identical. We align ``q[f±1]`` to ``q[f]`` via
``sign(⟨q[f], q[f±1]⟩)`` before subtraction.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch


def compute_pose_smoothness_loss(
    model,
    lambda_trans: float,
    lambda_rot: float,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Compute the Stage B temporal smoothness loss.

    Parameters
    ----------
    model
        A ``LayeredGaussians`` (or any module exposing ``tracks_active``
        plus ``_track_quat_<tid>`` / ``_track_trans_<tid>`` /
        ``_track_active_<tid>`` attributes). When the module has no
        learnable pose state (legacy buffer mode, no tracks, etc.), the
        function returns a zero scalar.
    lambda_trans
        Weight on the translation smoothness term. ``<= 0`` disables it.
    lambda_rot
        Weight on the rotation (quat chord) smoothness term. ``<= 0``
        disables it.
    device
        Device for the returned scalar. Defaults to the device of the
        first ``_track_trans_<tid>`` Parameter encountered.

    Returns
    -------
    torch.Tensor
        Scalar tensor of shape ``[1]``. Differentiable wrt the
        ``_track_quat_*`` / ``_track_trans_*`` Parameters when the
        relevant ``λ > 0`` and at least one track has ``F ≥ 3`` with a
        triple-active frame.
    """
    # Cheap-out path 1: both terms disabled — nothing to compute.
    if lambda_trans <= 0.0 and lambda_rot <= 0.0:
        return _zero(device)

    tracks_active = getattr(model, "tracks_active", None)
    if not tracks_active:
        return _zero(device)
    tids: Iterable[str] = sorted(tracks_active.keys())

    # Resolve device from first available Parameter if not given.
    if device is None:
        for tid in tids:
            t = getattr(model, f"_track_trans_{tid}", None)
            if t is not None:
                device = t.device
                break
        else:
            return _zero(None)

    # Init as scalar zero on device — keeps autograd happy and avoids
    # losing the gradient connection through Python-level `if`.
    sum_t = torch.zeros((), device=device)
    sum_r = torch.zeros((), device=device)
    n_t = 0
    n_r = 0

    for tid in tids:
        q = getattr(model, f"_track_quat_{tid}", None)
        t = getattr(model, f"_track_trans_{tid}", None)
        a = getattr(model, f"_track_active_{tid}", None)
        # Buffer-mode tracks lack quat/trans Parameters; skip silently.
        if q is None or t is None or a is None:
            continue
        F = t.shape[0]
        if F < 3:
            continue

        # Mask must live on t's device — even when the function-level
        # ``device`` arg targets cuda, individual ``_track_active_<tid>``
        # buffers may have stayed on cpu (LayeredGaussians populates them
        # via register_buffer at construction time, before ``.to(device)``
        # in some test stubs / partial-init paths). Pin to t.device so
        # ``sq * m`` doesn't trip the cross-device guard.
        a_bool = a.to(dtype=torch.bool, device=t.device)
        # Triple-active mask covering interior frames f ∈ [1, F-1).
        mask = a_bool[:-2] & a_bool[1:-1] & a_bool[2:]  # [F-2]
        if not bool(mask.any()):
            continue
        n_valid = int(mask.sum().item())
        m = mask.to(dtype=t.dtype)

        if lambda_trans > 0.0:
            d2t = t[2:] - 2.0 * t[1:-1] + t[:-2]  # [F-2, 3]
            sq = (d2t * d2t).sum(dim=-1)  # [F-2]
            # `.to(device)` reconciles the per-track contribution back
            # onto the accumulator's device (Parameters may live on a
            # different cuda index than self.device in DDP scenarios).
            sum_t = sum_t + (sq * m).sum().to(device)
            n_t += n_valid

        if lambda_rot > 0.0:
            q_center = q[1:-1]
            q_prev = q[:-2]
            q_next = q[2:]
            # Sign-align q_prev / q_next to q_center (q ≡ -q under SO(3)).
            dot_prev = (q_center * q_prev).sum(dim=-1, keepdim=True)
            dot_next = (q_center * q_next).sum(dim=-1, keepdim=True)
            q_prev_a = torch.where(dot_prev >= 0, q_prev, -q_prev)
            q_next_a = torch.where(dot_next >= 0, q_next, -q_next)
            d2q = q_next_a - 2.0 * q_center + q_prev_a  # [F-2, 4]
            sq_r = (d2q * d2q).sum(dim=-1)
            sum_r = sum_r + (sq_r * m).sum().to(device)
            n_r += n_valid

    out = torch.zeros((), device=device)
    if n_t > 0 and lambda_trans > 0.0:
        out = out + lambda_trans * sum_t / float(n_t)
    if n_r > 0 and lambda_rot > 0.0:
        out = out + lambda_rot * sum_r / float(n_r)
    # Match the [1]-shape convention used by other loss contributors
    # (compute_bg_cuboid_opacity_penalty, etc.) for clean broadcasting in
    # Trainer.get_losses.
    return out.reshape(1)


def _zero(device) -> torch.Tensor:
    """Shape-``[1]`` zero on the requested device (or CPU when None)."""
    if device is None:
        return torch.zeros(1)
    return torch.zeros(1, device=device)
