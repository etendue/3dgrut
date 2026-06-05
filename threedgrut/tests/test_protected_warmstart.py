# SPDX-License-Identifier: Apache-2.0
"""Protected warm-start (C2): MCMC must not relocate/perturb asset-injected
(unobserved-face) particles. CPU-only unit tests — the dynamic relocate path
itself is CUDA (_mcmc_plugin) and is validated by the A800 A/B; here we pin the
pure index/mask logic + the buffer plumbing + the warm-track-id source.
"""
from __future__ import annotations

import torch

from threedgrut.strategy.mcmc import (
    _filter_protected_indices,
    _protected_particle_mask,
)


def test_filter_protected_indices_drops_matching():
    track_ids = torch.tensor([10, 20, 10, 30, 20])
    protected = torch.tensor([20])
    idxs = torch.tensor([0, 1, 2, 3, 4])
    out = _filter_protected_indices(idxs, track_ids, protected)
    assert out.tolist() == [0, 2, 3]  # particles 1 and 4 (track 20) removed


def test_filter_protected_indices_subset_input():
    track_ids = torch.tensor([10, 20, 10, 30, 20])
    protected = torch.tensor([20, 30])
    idxs = torch.tensor([1, 3, 4])  # all protected
    out = _filter_protected_indices(idxs, track_ids, protected)
    assert out.numel() == 0


def test_filter_protected_indices_noop_when_no_protected():
    track_ids = torch.tensor([10, 20, 10])
    idxs = torch.tensor([0, 1, 2])
    assert torch.equal(_filter_protected_indices(idxs, track_ids, None), idxs)
    empty = torch.tensor([], dtype=torch.long)
    assert torch.equal(_filter_protected_indices(idxs, track_ids, empty), idxs)


def test_filter_protected_indices_noop_when_no_track_ids():
    idxs = torch.tensor([0, 1, 2])
    out = _filter_protected_indices(idxs, None, torch.tensor([20]))
    assert torch.equal(out, idxs)


def test_protected_particle_mask_marks_protected():
    track_ids = torch.tensor([10, 20, 10, 30, 20])
    protected = torch.tensor([20])
    mask = _protected_particle_mask(track_ids, protected, n=5, device=torch.device("cpu"))
    assert mask is not None
    assert mask.tolist() == [False, True, False, False, True]


def test_protected_particle_mask_none_when_unprotected():
    track_ids = torch.tensor([10, 20, 10])
    assert _protected_particle_mask(track_ids, None, 3, torch.device("cpu")) is None
    empty = torch.tensor([], dtype=torch.long)
    assert _protected_particle_mask(track_ids, empty, 3, torch.device("cpu")) is None
    assert _protected_particle_mask(None, torch.tensor([20]), 3, torch.device("cpu")) is None
