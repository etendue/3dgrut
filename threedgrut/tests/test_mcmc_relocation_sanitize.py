# SPDX-License-Identifier: Apache-2.0
"""A1 — _sanitize_relocation: contain non-finite MCMC relocation outputs.

inc_b6a9 6-cam R1 (2026-07-02/03): compute_relocation_tensor (CUDA, MCMC
Eq.9 binomial math) emitted non-finite/non-positive new densities/scales at
opacity→1 boundaries; scale_activation_inv (log) turned them into NaN/-inf
parameters — 60% of the background layer's densities were NaN within ~3k
steps (bypassing the pred-based drop guard entirely, since the poison never
crosses the loss). Sanitize: any bad row falls back to the donor's original
values (relocation degenerates to a plain copy for that row).
"""

from __future__ import annotations

import torch

from threedgrut.strategy.mcmc import _sanitize_relocation


def _mk(n=4):
    d_new = torch.rand(n, 1) * 0.5 + 0.25
    s_new = torch.rand(n, 3) * 0.5 + 0.25
    d_don = torch.rand(n, 1) * 0.5 + 0.25
    s_don = torch.rand(n, 3) * 0.5 + 0.25
    return d_new, s_new, d_don, s_don


def test_clean_passthrough_zero_count():
    d_new, s_new, d_don, s_don = _mk()
    d, s, n_bad = _sanitize_relocation(d_new, s_new, d_don, s_don)
    assert n_bad == 0
    assert torch.equal(d, d_new) and torch.equal(s, s_new)


def test_nan_density_row_falls_back_to_donor():
    d_new, s_new, d_don, s_don = _mk()
    d_new[1, 0] = float("nan")
    d, s, n_bad = _sanitize_relocation(d_new, s_new, d_don, s_don)
    assert n_bad == 1
    assert torch.equal(d[1], d_don[1])
    assert torch.equal(s[1], s_don[1])  # whole row reverts together
    assert torch.isfinite(d).all() and torch.isfinite(s).all()


def test_zero_or_negative_scale_row_falls_back():
    d_new, s_new, d_don, s_don = _mk()
    s_new[0, 2] = 0.0
    s_new[3, 0] = -1e-3
    d, s, n_bad = _sanitize_relocation(d_new, s_new, d_don, s_don)
    assert n_bad == 2
    assert torch.equal(s[0], s_don[0]) and torch.equal(s[3], s_don[3])
    assert (s > 0).all()


def test_inf_scale_and_nonpositive_density_covered():
    d_new, s_new, d_don, s_don = _mk()
    s_new[2, 1] = float("inf")
    d_new[0, 0] = 0.0  # non-positive density would break log-inverse too
    d, s, n_bad = _sanitize_relocation(d_new, s_new, d_don, s_don)
    assert n_bad == 2
    assert torch.isfinite(s).all() and (d > 0).all()
