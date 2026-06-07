# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""P1.2 Stage B/C — pose boundary anchor + pose prior regularizers.

Stage A (commits 67d11e6 / 8c48815) made ``LayeredGaussians.dynamic_rigids``'s
per-track per-frame cuboid SE(3) pose a learnable ``nn.Parameter`` pair
(``_track_quat_<tid>`` ``[F, 4]`` wxyz + ``_track_trans_<tid>`` ``[F, 3]``),
keeping a frozen ``_track_pose_gt_<tid>`` ``[F, 4, 4]`` buffer as the GT
reference. With Adam optimising each frame against photometric loss, the
learned trajectory can drift *globally* (a constant translation/rotation that
the affine color-correction can't undo → the −0.61 cc regression observed in
the Stage A A/B). ``pose_smoothness`` only penalises *curvature*, not drift.

Two anchors pull the learned pose back toward GT:

  - :func:`compute_pose_boundary_loss` — penalise ONLY the first + last
    *active* frame of each track. Pinning the endpoints removes the global
    drift degree of freedom while leaving interior frames free to refine
    (this is the ``fix_first/last`` knob).
  - :func:`compute_pose_prior_loss` — a soft L2 over *all* active frames
    toward GT (the ``lambda_pose_prior_*`` knob). Use a small λ relative to
    the boundary term; it keeps the whole trajectory near GT.

Translation is compared directly (``‖t − t_gt‖²``). Rotation is compared in
rotation-matrix space (Frobenius²) so the quaternion double-cover
(``q ≡ −q`` under SO(3)) never inflates the loss — no sign alignment needed.

Pure-function module (only ``getattr`` on the model + plain torch), so the
unit tests in ``test_pose_anchor.py`` run on Mac without the CUDA/NCore stack,
mirroring ``pose_smoothness.py``.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch


def _quat_wxyz_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """``[..., 4]`` wxyz quat → ``[..., 3, 3]`` rotation matrix.

    Local replica of ``layered_model._quat_wxyz_to_rotmat`` (imported there
    would drag the CUDA/NCore stack into this Mac-testable module). Normalises
    ``q`` first — Adam steps break unit norm.
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    row0 = torch.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
    row1 = torch.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)], dim=-1)
    row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _anchor_loss(
    model,
    lambda_trans: float,
    lambda_rot: float,
    device: Optional[torch.device | str],
    mode: str,
) -> torch.Tensor:
    """Shared core for boundary / prior anchors.

    ``mode="boundary"`` selects the first + last active frame per track;
    ``mode="prior"`` selects all active frames. Both compare the learned
    ``_track_quat_`` / ``_track_trans_`` to the frozen ``_track_pose_gt_``
    SE(3) buffer over the selected frames.
    """
    # Cheap-out: both terms disabled.
    if lambda_trans <= 0.0 and lambda_rot <= 0.0:
        return _zero(device)

    tracks_active = getattr(model, "tracks_active", None)
    if not tracks_active:
        return _zero(device)
    tids: Iterable[str] = sorted(tracks_active.keys())

    if device is None:
        for tid in tids:
            t = getattr(model, f"_track_trans_{tid}", None)
            if t is not None:
                device = t.device
                break
        else:
            return _zero(None)

    sum_t = torch.zeros((), device=device)
    sum_r = torch.zeros((), device=device)
    n_t = 0
    n_r = 0

    for tid in tids:
        q = getattr(model, f"_track_quat_{tid}", None)
        t = getattr(model, f"_track_trans_{tid}", None)
        a = getattr(model, f"_track_active_{tid}", None)
        gt = getattr(model, f"_track_pose_gt_{tid}", None)
        # Buffer-mode / legacy tracks lack learnable params or the GT
        # reference; nothing to anchor → skip silently.
        if q is None or t is None or a is None or gt is None:
            continue

        a_bool = a.to(dtype=torch.bool, device=t.device)
        active_idx = a_bool.nonzero(as_tuple=False).flatten()
        if active_idx.numel() == 0:
            continue

        if mode == "boundary":
            first = int(active_idx[0].item())
            last = int(active_idx[-1].item())
            frames = [first] if first == last else [first, last]
            sel = torch.tensor(frames, device=t.device, dtype=torch.long)
        else:  # prior
            sel = active_idx.to(t.device)

        gt_t = gt[:, :3, 3].to(device=t.device, dtype=t.dtype)
        gt_R = gt[:, :3, :3].to(device=t.device, dtype=t.dtype)
        n_sel = int(sel.numel())

        if lambda_trans > 0.0:
            dt = t[sel] - gt_t[sel]                  # [n_sel, 3]
            sum_t = sum_t + (dt * dt).sum().to(device)
            n_t += n_sel

        if lambda_rot > 0.0:
            R = _quat_wxyz_to_rotmat(q[sel])         # [n_sel, 3, 3]
            dR = R - gt_R[sel]
            sum_r = sum_r + (dR * dR).sum().to(device)
            n_r += n_sel

    out = torch.zeros((), device=device)
    if n_t > 0 and lambda_trans > 0.0:
        out = out + lambda_trans * sum_t / float(n_t)
    if n_r > 0 and lambda_rot > 0.0:
        out = out + lambda_rot * sum_r / float(n_r)
    return out.reshape(1)


def compute_pose_boundary_loss(
    model,
    lambda_trans: float,
    lambda_rot: float,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Anchor the first + last active frame of each track to GT (``fix_first/last``).

    Returns a shape-``[1]`` scalar, differentiable wrt ``_track_quat_*`` /
    ``_track_trans_*`` when the relevant ``λ > 0``. Zero when no learnable
    pose / GT state exists or both λ ≤ 0.
    """
    return _anchor_loss(model, lambda_trans, lambda_rot, device, mode="boundary")


def compute_pose_prior_loss(
    model,
    lambda_trans: float,
    lambda_rot: float,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Soft L2 of every active frame's learned pose toward GT (pose prior).

    Same contract as :func:`compute_pose_boundary_loss` but averaged over all
    active frames. Intended with a small λ relative to the boundary term.
    """
    return _anchor_loss(model, lambda_trans, lambda_rot, device, mode="prior")


def _zero(device) -> torch.Tensor:
    """Shape-``[1]`` zero on the requested device (or CPU when None)."""
    if device is None:
        return torch.zeros(1)
    return torch.zeros(1, device=device)
