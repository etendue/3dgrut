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


# --- Task 4: init_layer_from_points writes a persistent protected buffer ---

import os

import pytest
from hydra import compose, initialize_config_dir

from threedgrut.layers.layer_spec import LayerSpec

_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs")
)


@pytest.fixture(scope="module")
def dyn_conf():
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="apps/ncore_3dgut_mcmc")


def _dyn_model(conf):
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=100),
        LayerSpec(name="dynamic_rigids", layer_id=1, max_n_particles=100),
    ]
    return LayeredGaussians(conf, specs=specs, scene_extent=1.0)


def _inject_dyn(model, n_per_track):
    """Inject n_per_track particles for tracks 0,1,2 into dynamic_rigids."""
    positions, track_ids = [], []
    for tid in (0, 1, 2):
        positions.append(torch.zeros(n_per_track, 3))
        track_ids.append(torch.full((n_per_track,), tid, dtype=torch.long))
    positions = torch.cat(positions)
    track_ids = torch.cat(track_ids)
    model.init_layer_from_points(
        "dynamic_rigids",
        positions,
        scales=torch.full((positions.shape[0], 3), -3.0),
        densities=torch.zeros(positions.shape[0], 1),
        colors=torch.full((positions.shape[0], 3), 0.5),
        rotations=torch.tensor([1.0, 0, 0, 0]).expand(positions.shape[0], 4).clone(),
        track_ids=track_ids,
        protected_track_ids=torch.tensor([0, 2], dtype=torch.long),
        setup_optimizer=False,
    )
    return model


def test_init_layer_writes_protected_buffer(dyn_conf):
    model = _inject_dyn(_dyn_model(dyn_conf), n_per_track=4)
    layer = model.layers["dynamic_rigids"]
    assert hasattr(layer, "_warmstart_protected_track_ids")
    assert layer._warmstart_protected_track_ids.tolist() == [0, 2]
    assert layer._warmstart_protected_track_ids.dtype == torch.long


def test_protected_buffer_persists_in_state_dict(dyn_conf):
    model = _inject_dyn(_dyn_model(dyn_conf), n_per_track=4)
    sd = model.state_dict()
    key = "layers.dynamic_rigids._warmstart_protected_track_ids"
    assert key in sd, f"{key} missing from state_dict (not persistent?)"
    assert sd[key].tolist() == [0, 2]


def test_init_layer_no_protected_buffer_when_omitted(dyn_conf):
    """Backward-compat: omitting protected_track_ids registers no buffer."""
    model = _dyn_model(dyn_conf)
    model.init_layer_from_points(
        "dynamic_rigids",
        torch.zeros(6, 3),
        scales=torch.full((6, 3), -3.0),
        densities=torch.zeros(6, 1),
        colors=torch.full((6, 3), 0.5),
        rotations=torch.tensor([1.0, 0, 0, 0]).expand(6, 4).clone(),
        track_ids=torch.tensor([0, 0, 1, 1, 2, 2]),
        setup_optimizer=False,
    )
    layer = model.layers["dynamic_rigids"]
    assert not hasattr(layer, "_warmstart_protected_track_ids")


# --- Task 5: build_warmstart_layer_inputs surfaces the warm (protected) ids ---

from pathlib import Path

from threedgrut.layers.warmstart_inject import build_warmstart_layer_inputs
from threedgrut.layers.warmstart_metadata import load_bundle_metadata

# Same bundle convention as test_warmstart_ply_engine.py (absent on Mac → skip).
_BUNDLE = Path(
    os.environ.get(
        "WARMSTART_BUNDLE",
        "/Users/etendue/repo/asset-harvester-verify/verify_assets/bundle",
    )
)
_needs_bundle = pytest.mark.skipif(
    not (_BUNDLE / "metadata.yaml").is_file(),
    reason=f"warm-start bundle not present at {_BUNDLE}",
)


@_needs_bundle
def test_build_warmstart_returns_warm_track_ids():
    """warm_track_ids == the integer ids of the asset-mapped tracks, derived
    from track_names order (name_to_id)."""
    bundle = load_bundle_metadata(_BUNDLE / "metadata.yaml")
    asset_hash = next(iter(bundle.keys()))
    track_key = "carA"
    track_names = ["bg_ignore", track_key]  # name_to_id[track_key] == 1
    tracks = {track_key: {"size": torch.tensor([4.0, 2.0, 1.6])}}
    mapping = {track_key: asset_hash}

    out = build_warmstart_layer_inputs(
        bundle_path=_BUNDLE,
        mapping=mapping,
        tracks=tracks,
        track_names=track_names,
        lidar_positions=torch.zeros(0, 3),
        lidar_track_ids=torch.zeros(0, dtype=torch.long),
        scale_prior=(0.1, 0.1, 0.1),
        density_init=0.1,
        mode="replace",
        max_pts_per_track=500,
        seed=0,
    )
    assert out is not None
    assert "warm_track_ids" in out
    assert out["warm_track_ids"].tolist() == [1]   # name_to_id["carA"]
    assert out["warm_track_ids"].dtype == torch.long
    # Every protected id must actually appear among the merged particles.
    merged_ids = set(out["track_ids"].unique().tolist())
    assert set(out["warm_track_ids"].tolist()) <= merged_ids


# --- Task 7: opacity-reg exemption includes dynamic_rigids ---


def test_multilayer_exempts_dynamic_rigids_from_opacity_reg():
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        conf = compose(config_name="apps/ncore_3dgut_mcmc_multilayer")
    exempt = list(conf.loss.exempt_layers_opacity_reg)
    assert "road" in exempt
    assert "dynamic_rigids" in exempt
