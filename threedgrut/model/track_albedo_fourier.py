# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""P1.3b — Fourier (4D-SH) time-varying per-track albedo bias.

P1.3 (already on ``main``) added a *constant* per-track RGB bias on the
``features_albedo`` DC band of dynamic_rigids particles, stored in
``LayeredGaussians._track_albedo_table`` of shape ``[K, 3]`` and gathered by
``track_ids``. That captures a single per-vehicle colour offset but cannot
model how a moving car's apparent albedo changes across the clip (entering /
leaving shadow, sun angle, headlight glow, ...).

P1.3b generalises the constant to a truncated cosine Fourier series over the
camera-frame time axis — the StreetGaussian (ECCV 2024) "4D-SH" trick applied
to the DC SH coefficient:

    albedo_bias(t) = Σ_{i=0}^{k-1}  f_i · cos(i · π · t / N_t)

where

    f_i ∈ R^3   is the i-th Fourier coefficient per track  (k = number of terms)
    t           is the current camera-frame index ∈ [0, N_t-1]
    N_t         is the total number of frames (track pose schedule length F)

The table therefore grows from ``[K, 3]`` to ``[K, 3, k]``. The DEGENERATE
case ``k == 1`` is byte-identical to P1.3: ``cos(0 · π · t / N_t) = cos(0) = 1``
for every ``t``, so the bias is the frame-independent ``f_0`` — exactly the old
DC-only gather. This guarantees a fresh ``k=1`` run reproduces the P1.3
baseline.

This module is deliberately **pure torch** (no ``threedgrt_tracer`` / ``ncore``
imports) so the Fourier math can be unit-tested on a CUDA-less Mac, mirroring
``threedgrut.model.pose_smoothness``.
"""

from __future__ import annotations

import math

import torch


def fourier_cos_basis(
    frame_id: int,
    n_frames: int,
    n_terms: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the cosine basis row ``[cos(i·π·t/N)]_{i=0}^{k-1}`` of shape ``[k]``.

    Parameters
    ----------
    frame_id
        Current camera-frame index ``t``. Clamped defensively into
        ``[0, n_frames-1]`` so an out-of-range / negative caller index never
        wraps or indexes out of bounds.
    n_frames
        Total number of frames ``N_t`` (track pose schedule length ``F``).
        Values ``<= 1`` collapse the time axis: every harmonic argument
        becomes ``i·π·0`` so the basis is all-ones (``k=1`` semantics, no
        division by zero).
    n_terms
        Number of Fourier terms ``k`` (``>= 1``).
    """
    k = int(n_terms)
    if k < 1:
        raise ValueError(f"n_terms must be >= 1, got {k}")
    N = int(n_frames)
    t = max(0, min(int(frame_id), N - 1)) if N >= 1 else 0
    # N <= 1 → single-frame track: collapse time so cos arg = i·π·0 = 0.
    denom = float(N) if N > 1 else 1.0
    t_norm = (t / denom) if N > 1 else 0.0
    i = torch.arange(k, device=device, dtype=dtype)
    return torch.cos(i * math.pi * t_norm)  # [k]


def fourier_albedo_bias(
    fourier_table: torch.Tensor,
    track_ids: torch.Tensor,
    frame_id: int,
    n_frames: int,
) -> torch.Tensor:
    """Per-particle time-varying albedo bias gathered by ``track_ids``.

    Parameters
    ----------
    fourier_table
        ``[K, 3, k]`` Fourier coefficients per track (``K`` tracks, RGB,
        ``k`` cosine terms). ``k == 1`` reproduces the P1.3 DC-only gather.
    track_ids
        ``[N]`` long tensor mapping each particle to its track row.
    frame_id
        Current camera-frame index ``t`` (clamped to ``[0, n_frames-1]``).
    n_frames
        Total frames ``N_t``.

    Returns
    -------
    torch.Tensor
        ``[N, 3]`` RGB bias to ADD onto the ``features_albedo`` DC band.
        Differentiable wrt ``fourier_table``.
    """
    if fourier_table.dim() != 3:
        raise ValueError(f"fourier_table must be [K, 3, k]; got shape {tuple(fourier_table.shape)}")
    k = fourier_table.shape[-1]
    basis = fourier_cos_basis(
        frame_id,
        n_frames,
        k,
        device=fourier_table.device,
        dtype=fourier_table.dtype,
    )  # [k]
    # Contract the k axis: [K, 3, k] · [k] -> [K, 3] per-track bias at time t.
    per_track = torch.matmul(fourier_table, basis)  # [K, 3]
    ids = track_ids.to(device=per_track.device, dtype=torch.long)
    return per_track[ids]  # [N, 3]


def upgrade_albedo_table(table: torch.Tensor, k: int) -> torch.Tensor:
    """Resize an albedo table to ``[K, 3, k]`` (ckpt back-compat).

    Handles three input shapes:
      * ``[K, 3]``      — a P1.3 (DC-only) table → DC kept in term 0, higher
                          harmonics zero.
      * ``[K, 3, k']``  — an existing Fourier table → low harmonics kept;
                          zero-pad if ``k > k'``; truncate if ``k < k'``.

    The returned tensor is a plain (non-Parameter) tensor on the same device
    and dtype as the input; callers wrap it in ``nn.Parameter`` as needed.
    """
    if int(k) < 1:
        raise ValueError(f"target k must be >= 1, got {k}")
    k = int(k)
    if table.dim() == 2:
        # [K, 3] DC-only → [K, 3, k] with DC in slot 0.
        K, C = table.shape
        out = table.new_zeros((K, C, k))
        out[..., 0] = table
        return out
    if table.dim() == 3:
        K, C, k_old = table.shape
        if k_old == k:
            return table.clone()
        out = table.new_zeros((K, C, k))
        n_copy = min(k_old, k)
        out[..., :n_copy] = table[..., :n_copy]
        return out
    raise ValueError(f"albedo table must be [K, 3] or [K, 3, k]; got shape {tuple(table.shape)}")
