# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""P1.3b unit tests — Fourier (4D-SH) time-varying per-track albedo bias.

P1.3 (already on ``main``) added a DC-only per-track albedo table
``_track_albedo_table[K, 3]``: a constant RGB bias added to the
``features_albedo`` DC band of every dynamic_rigids particle, gathered by
``track_ids``. P1.3b generalises that constant to a truncated Fourier series
over the camera-frame time axis (StreetGaussian ECCV-2024 "4D-SH"):

    albedo_bias(t) = Σ_{i=0}^{k-1}  f_i · cos(i · π · t / N_t)

where ``f_i ∈ R^3`` is the i-th Fourier coefficient per track, ``t`` is the
current camera-frame index ``∈ [0, N_t-1]`` and ``N_t`` is the total number of
frames (track pose schedule length ``F``).

The math lives in the pure-torch module ``threedgrut.model.track_albedo_fourier``
so it can be unit-tested on a CUDA-less Mac (importing ``layered_model`` pulls
in the ``threedgrt_tracer`` CUDA extension + ``ncore`` and fails on Mac).

Covered:
    1. k=1 degenerates EXACTLY to the old DC-only behaviour: cos(0)=1 → for
       any ``frame_id`` the bias == f_0. This is the byte-identical guarantee.
    2. Fourier evaluation is numerically correct vs a hand-computed reference.
    3. ckpt back-compat: an old ``[K, 3]`` table "upgrades" to ``[K, 3, k]``
       with the DC coefficient preserved and all higher harmonics zero.
    4. frame_id boundaries (t=0 and t=N_t-1) stay in-bounds and correct;
       clamping handles out-of-range frame ids defensively.
"""

from __future__ import annotations

import math

import torch

from threedgrut.model.track_albedo_fourier import (
    fourier_albedo_bias,
    upgrade_albedo_table,
)


# ----------------------------------------------------------------------------
# (1) k=1 degenerates exactly to DC-only behaviour
# ----------------------------------------------------------------------------
def test_k1_degenerates_to_dc_only():
    """n_fourier=1 → cos(0·π·t/N)=1 for every t → bias == f_0 always."""
    K, N_t = 2, 7
    table = torch.tensor(
        [
            [[0.10], [0.20], [0.30]],  # track 0: f_0 = (0.1, 0.2, 0.3)
            [[-0.40], [-0.50], [-0.60]],  # track 1
        ],
        dtype=torch.float32,
    )  # [K, 3, 1]
    track_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    dc_expected = torch.tensor(
        [
            [0.10, 0.20, 0.30],
            [0.10, 0.20, 0.30],
            [-0.40, -0.50, -0.60],
            [-0.40, -0.50, -0.60],
        ]
    )
    # Output must equal the DC bias for EVERY frame_id in [0, N_t-1].
    for t in range(N_t):
        bias = fourier_albedo_bias(table, track_ids, frame_id=t, n_frames=N_t)
        assert bias.shape == (4, 3)
        assert torch.allclose(bias, dc_expected, atol=1e-7), f"k=1 must be DC-only (frame-independent); failed at t={t}"


def test_k1_matches_plain_gather():
    """k=1 fourier bias == plain ``old_table[track_ids]`` (the P1.3 path)."""
    old_table = torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]])  # [K, 3]
    table = old_table.unsqueeze(-1)  # [K, 3, 1]
    track_ids = torch.tensor([1, 0, 1], dtype=torch.long)
    plain = old_table[track_ids]
    bias = fourier_albedo_bias(table, track_ids, frame_id=3, n_frames=10)
    assert torch.allclose(bias, plain, atol=1e-7)


# ----------------------------------------------------------------------------
# (2) Fourier evaluation correctness vs hand-computed reference
# ----------------------------------------------------------------------------
def test_fourier_sum_correct():
    """Σ_i f_i·cos(i·π·t/N) matches an explicit Python reference."""
    K, N_t, k = 1, 8, 4
    # f_0..f_3 for the single track, distinct per channel.
    f = torch.tensor(
        [
            [
                [1.0, 0.5, 2.0, 0.25],  # R harmonics
                [0.0, 1.0, 0.0, -1.0],  # G
                [-1.0, 0.0, 0.5, 0.5],  # B
            ]
        ]
    )  # [1, 3, 4]
    track_ids = torch.tensor([0], dtype=torch.long)
    t = 3
    # Reference: per channel sum_i f[c,i] * cos(i*pi*t/N)
    cos = [math.cos(i * math.pi * t / N_t) for i in range(k)]
    ref = torch.tensor(
        [
            [
                sum(f[0, 0, i].item() * cos[i] for i in range(k)),
                sum(f[0, 1, i].item() * cos[i] for i in range(k)),
                sum(f[0, 2, i].item() * cos[i] for i in range(k)),
            ]
        ]
    )
    bias = fourier_albedo_bias(f, track_ids, frame_id=t, n_frames=N_t)
    assert bias.shape == (1, 3)
    assert torch.allclose(bias, ref, atol=1e-6), f"got {bias}, want {ref}"


def test_fourier_per_track_distinct():
    """Two tracks with different coefficients produce different biases."""
    f = torch.zeros(2, 3, 3)
    f[0, :, 0] = torch.tensor([0.1, 0.2, 0.3])  # track 0: only DC
    f[1, :, 1] = torch.tensor([1.0, 1.0, 1.0])  # track 1: only first harmonic
    track_ids = torch.tensor([0, 1], dtype=torch.long)
    N_t = 4
    t = 1  # cos(1*pi*1/4) = cos(pi/4) = sqrt(2)/2
    bias = fourier_albedo_bias(f, track_ids, frame_id=t, n_frames=N_t)
    expected0 = torch.tensor([0.1, 0.2, 0.3])
    expected1 = torch.full((3,), math.cos(math.pi / 4))
    assert torch.allclose(bias[0], expected0, atol=1e-6)
    assert torch.allclose(bias[1], expected1, atol=1e-6)


# ----------------------------------------------------------------------------
# (3) ckpt back-compat: [K,3] -> [K,3,k] upgrade
# ----------------------------------------------------------------------------
def test_upgrade_old_table_preserves_dc_zeros_harmonics():
    """Old [K,3] table upgrades to [K,3,k]: DC kept, higher harmonics zero."""
    old = torch.tensor([[0.1, 0.2, 0.3], [-0.4, -0.5, -0.6]])  # [K, 3]
    k = 4
    up = upgrade_albedo_table(old, k)
    assert up.shape == (2, 3, k)
    # DC slice == old values
    assert torch.allclose(up[..., 0], old, atol=1e-7)
    # all higher harmonics are zero
    assert torch.allclose(up[..., 1:], torch.zeros(2, 3, k - 1), atol=1e-7)


def test_upgrade_then_evaluate_equals_old_dc():
    """An upgraded old table evaluated by Fourier == old DC gather (any t)."""
    old = torch.tensor([[0.7, -0.1, 0.0], [0.2, 0.2, 0.2]])
    up = upgrade_albedo_table(old, k=5)
    track_ids = torch.tensor([0, 1, 0], dtype=torch.long)
    N_t = 12
    for t in (0, 5, 11):
        bias = fourier_albedo_bias(up, track_ids, frame_id=t, n_frames=N_t)
        assert torch.allclose(bias, old[track_ids], atol=1e-7), f"upgraded old table must reproduce DC gather at t={t}"


def test_upgrade_k1_is_noop_unsqueeze():
    """Upgrading to k=1 is just adding a trailing dim (no info loss)."""
    old = torch.tensor([[0.1, 0.2, 0.3]])
    up = upgrade_albedo_table(old, k=1)
    assert up.shape == (1, 3, 1)
    assert torch.allclose(up[..., 0], old)


def test_upgrade_truncates_when_target_smaller():
    """Downgrading a [K,3,k_old] table to a smaller k keeps the low harmonics."""
    big = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    small = upgrade_albedo_table(big, k=2)
    assert small.shape == (2, 3, 2)
    assert torch.allclose(small, big[..., :2])


def test_upgrade_pads_when_target_larger():
    """Upgrading a [K,3,k_old] table to a larger k zero-pads new harmonics."""
    small = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    big = upgrade_albedo_table(small, k=5)
    assert big.shape == (2, 3, 5)
    assert torch.allclose(big[..., :2], small)
    assert torch.allclose(big[..., 2:], torch.zeros(2, 3, 3))


# ----------------------------------------------------------------------------
# (4) frame_id boundaries
# ----------------------------------------------------------------------------
def test_frame_boundaries_in_bounds_and_correct():
    """t=0 and t=N_t-1 are valid and numerically correct."""
    f = torch.tensor(
        [
            [
                [0.0, 1.0],  # R: f0=0, f1=1
                [0.0, 1.0],  # G
                [0.0, 1.0],  # B
            ]
        ]
    )  # [1, 3, 2]
    track_ids = torch.tensor([0], dtype=torch.long)
    N_t = 6
    # t=0: cos(0)=1, cos(0)=1 → bias = f0 + f1 = (1,1,1)
    b0 = fourier_albedo_bias(f, track_ids, frame_id=0, n_frames=N_t)
    assert torch.allclose(b0, torch.ones(1, 3), atol=1e-6)
    # t=N_t-1=5: cos(0·...)=1 for i=0; cos(1·pi·5/6) for i=1.
    t = N_t - 1
    expected_h1 = math.cos(math.pi * t / N_t)
    b_last = fourier_albedo_bias(f, track_ids, frame_id=t, n_frames=N_t)
    assert torch.allclose(b_last, torch.full((1, 3), expected_h1), atol=1e-6)


def test_frame_id_clamped_when_out_of_range():
    """frame_id beyond [0, N_t-1] is clamped, not crashing / wrapping."""
    f = torch.zeros(1, 3, 2)
    f[0, :, 1] = 1.0  # only first harmonic
    track_ids = torch.tensor([0], dtype=torch.long)
    N_t = 4
    # frame_id=99 should clamp to N_t-1=3.
    b_clamped = fourier_albedo_bias(f, track_ids, frame_id=99, n_frames=N_t)
    b_ref = fourier_albedo_bias(f, track_ids, frame_id=N_t - 1, n_frames=N_t)
    assert torch.allclose(b_clamped, b_ref, atol=1e-7)
    # negative frame_id clamps to 0.
    b_neg = fourier_albedo_bias(f, track_ids, frame_id=-5, n_frames=N_t)
    b_zero = fourier_albedo_bias(f, track_ids, frame_id=0, n_frames=N_t)
    assert torch.allclose(b_neg, b_zero, atol=1e-7)


def test_single_frame_schedule_no_div_by_zero():
    """N_t=1 (single-frame track) must not divide by zero; reduces to DC sum."""
    f = torch.tensor([[[0.1, 0.5], [0.2, 0.5], [0.3, 0.5]]])  # [1,3,2]
    track_ids = torch.tensor([0], dtype=torch.long)
    # N_t=1 → t can only be 0 → cos(i*pi*0/1)=cos(0)=1 → sum of all coeffs.
    bias = fourier_albedo_bias(f, track_ids, frame_id=0, n_frames=1)
    expected = torch.tensor([[0.1 + 0.5, 0.2 + 0.5, 0.3 + 0.5]])
    assert torch.allclose(bias, expected, atol=1e-6)


def test_gradient_flows_to_coefficients():
    """The bias is differentiable wrt the Fourier coefficient table."""
    f = torch.zeros(1, 3, 3, requires_grad=True)
    track_ids = torch.tensor([0, 0], dtype=torch.long)
    bias = fourier_albedo_bias(f, track_ids, frame_id=2, n_frames=8)
    bias.sum().backward()
    assert f.grad is not None
    assert not torch.allclose(f.grad, torch.zeros_like(f.grad)), "gradient must reach the coefficient table"
