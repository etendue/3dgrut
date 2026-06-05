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


# --- Task 3: perturb_gaussians freezes protected particles (CPU integration) ---

from types import SimpleNamespace

import torch.nn as nn

from threedgrut.strategy.mcmc import MCMCStrategy


class _StubMoG:
    """Minimal duck-typed model exposing exactly what perturb_gaussians reads."""

    def __init__(self, positions, densities, track_ids, protected):
        self.positions = nn.Parameter(positions.clone())
        self._density = densities
        self.track_ids = track_ids
        self._warmstart_protected_track_ids = protected
        n = positions.shape[0]
        # identity covariance per particle → bmm leaves the noise vector intact
        self._cov = torch.eye(3).unsqueeze(0).expand(n, 3, 3).contiguous()
        self.optimizer = SimpleNamespace(
            param_groups=[{"name": "positions", "lr": 1.0}]
        )

    def get_covariance(self):
        return self._cov

    def get_positions(self):
        return self.positions.detach()

    def get_density(self):
        return self._density


def test_perturb_freezes_protected_particles():
    torch.manual_seed(0)
    n = 6
    positions = torch.zeros(n, 3)
    # low density everywhere → op_sigmoid(1-d) is non-trivial → real noise
    densities = torch.full((n, 1), 0.02)
    track_ids = torch.tensor([10, 20, 10, 30, 20, 10])
    protected = torch.tensor([20])  # particles 1 and 4 must not move
    model = _StubMoG(positions, densities, track_ids, protected)

    strat = MCMCStrategy.__new__(MCMCStrategy)  # bypass CUDA __init__
    strat.model = model
    strat.conf = SimpleNamespace(
        strategy=SimpleNamespace(perturb=SimpleNamespace(noise_lr=1.0))
    )

    before = model.positions.detach().clone()
    strat.perturb_gaussians()
    after = model.positions.detach()

    protected_rows = torch.tensor([1, 4])
    moved_rows = torch.tensor([0, 2, 3, 5])
    assert torch.equal(after[protected_rows], before[protected_rows])  # frozen
    assert not torch.allclose(after[moved_rows], before[moved_rows])   # perturbed


def test_perturb_byte_identical_without_protected():
    torch.manual_seed(0)
    n = 4
    model = _StubMoG(
        torch.zeros(n, 3), torch.full((n, 1), 0.02),
        torch.tensor([10, 20, 10, 30]), None,  # no protected buffer
    )
    strat = MCMCStrategy.__new__(MCMCStrategy)
    strat.model = model
    strat.conf = SimpleNamespace(
        strategy=SimpleNamespace(perturb=SimpleNamespace(noise_lr=1.0))
    )
    before = model.positions.detach().clone()
    strat.perturb_gaussians()
    # every particle moved (no freezing path taken)
    assert not torch.allclose(model.positions.detach(), before)
