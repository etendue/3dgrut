# SPDX-License-Identifier: Apache-2.0
"""Training-time contracts for MCRO B12 projection ownership exclusion."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from threedgrut.strategy.layered_mcmc import LayeredMCMCStrategy


class _Background:
    def __init__(self, n: int = 4):
        self.positions = torch.nn.Parameter(torch.zeros(n, 3))
        self.scale = torch.nn.Parameter(torch.zeros(n, 3))
        self.density = torch.nn.Parameter(torch.full((n, 1), 2.0))

    def get_scale(self):
        return self.scale.exp()

    def get_density(self):
        return self.density.sigmoid()


class _Model:
    def __init__(self, n: int = 4):
        self.layers = {"background": _Background(n)}

    def get_layer_mask(self, name: str):
        assert name == "background"
        return torch.tensor([True] * len(self.layers["background"].positions) + [False, False])


def _cfg(**overrides):
    values = {
        "enabled": True,
        "sample_every_steps": 1,
        "window_samples": 2,
        "warmup_steps": 0,
        "road_mask_erosion_px": 0,
        "protection_margin_px": 0,
        "footprint_sigma": 2.0,
        "max_footprint_px": 48.0,
        "chunk_size": 100,
        "min_road_hits": 1,
        "max_protected_hits": 0,
        "min_visible_hits": 1,
        "opacity_threshold": 0.005,
        "action": "recycle",
        "dead_density_raw": -50.0,
        "density_decay": 5.0,
        "footprint_shrink_factor": 0.5,
        "min_footprint_scale": 1e-4,
    }
    values.update(overrides)
    return OmegaConf.create({"layers": {"bg_road_duplicate_exclusion": values}})


def _strategy(**cfg_overrides):
    strategy = LayeredMCMCStrategy.__new__(LayeredMCMCStrategy)
    strategy.conf = _cfg(**cfg_overrides)
    strategy.model = _Model()
    strategy._bg_road_duplicate_visibility = None
    strategy._bg_road_duplicate_counts = None
    strategy._bg_road_duplicate_n_samples = 0
    strategy.last_bg_road_duplicate_stats = None
    return strategy


def _batch():
    intrinsics = {
        "focal_length": torch.tensor([10.0, 10.0]),
        "principal_point": torch.tensor([1.5, 1.5]),
        "resolution": torch.tensor([4.0, 4.0]),
    }
    return SimpleNamespace(
        image_infos={"road_mask": torch.ones(4, 4, dtype=torch.bool)},
        T_to_world=torch.eye(4),
        intrinsics_OpenCVPinholeCameraModelParameters=intrinsics,
    )


def _fake_accumulator(counts, **kwargs):
    # Rows 0/1 touch road. Row 1 is protected and must survive. Row 2 is
    # visible non-road. Row 3 is invisible.
    counts["visible_hits"] += torch.tensor([1, 1, 1, 0], dtype=torch.int32)
    counts["road_footprint_hits"] += torch.tensor([1, 1, 0, 0], dtype=torch.int32)
    counts["protected_center_hits"] += torch.tensor([0, 1, 0, 0], dtype=torch.int32)


def test_default_off_does_not_retain_renderer_output():
    strategy = _strategy(enabled=False)
    strategy.set_step_outputs({"mog_visibility": torch.ones(6)})
    assert strategy._bg_road_duplicate_visibility is None
    before = strategy.model.layers["background"].density.detach().clone()
    assert strategy._maybe_apply_bg_road_duplicate_exclusion(0, _batch()) is False
    assert torch.equal(strategy.model.layers["background"].density, before)


def test_visibility_is_sliced_to_background_rows():
    strategy = _strategy()
    strategy.set_step_outputs({"mog_visibility": torch.tensor([1, 0, 1, 0, 1, 1], dtype=torch.bool)})
    assert torch.equal(
        strategy._bg_road_duplicate_visibility,
        torch.tensor([1, 0, 1, 0], dtype=torch.bool),
    )


def test_window_recycles_road_candidate_but_protects_nonroad(monkeypatch):
    monkeypatch.setattr(
        "threedgrut.strategy.layered_mcmc.accumulate_projection_counts",
        _fake_accumulator,
    )
    strategy = _strategy()
    initial = strategy.model.layers["background"].density.detach().clone()
    for step in (0, 1):
        strategy.set_step_outputs({"mog_visibility": torch.ones(4)})
        changed = strategy._maybe_apply_bg_road_duplicate_exclusion(step, _batch())
    assert changed is True
    density = strategy.model.layers["background"].density.detach().reshape(-1)
    assert density[0] == -50.0
    assert density[1] == initial[1]
    assert strategy.last_bg_road_duplicate_stats == {
        "n_window_samples": 2,
        "n_visible": 3,
        "n_road_candidates": 2,
        "n_protected": 1,
        "n_recycled": 1,
        "n_decayed": 0,
        "n_footprint_shrunk": 0,
    }
    assert strategy._bg_road_duplicate_counts is None
    assert strategy._bg_road_duplicate_n_samples == 0


@pytest.mark.parametrize(
    ("action", "expected_density", "expected_scale"),
    [
        ("density_decay", -3.0, 0.0),
        ("footprint_shrink", 2.0, pytest.approx(-0.69314718)),
    ],
)
def test_soft_actions(monkeypatch, action, expected_density, expected_scale):
    monkeypatch.setattr(
        "threedgrut.strategy.layered_mcmc.accumulate_projection_counts",
        _fake_accumulator,
    )
    strategy = _strategy(action=action, window_samples=1)
    strategy.set_step_outputs({"mog_visibility": torch.ones(4)})
    assert strategy._maybe_apply_bg_road_duplicate_exclusion(0, _batch()) is True
    bg = strategy.model.layers["background"]
    assert float(bg.density.detach()[0]) == pytest.approx(expected_density)
    assert float(bg.scale.detach()[0, 0]) == expected_scale
    # Protected row never receives the action.
    assert float(bg.density.detach()[1]) == pytest.approx(2.0)
    assert float(bg.scale.detach()[1, 0]) == pytest.approx(0.0)


def test_nonfinite_input_skips_sample_and_mutation(monkeypatch):
    monkeypatch.setattr(
        "threedgrut.strategy.layered_mcmc.accumulate_projection_counts",
        lambda *args, **kwargs: pytest.fail("accumulator must not run"),
    )
    strategy = _strategy(window_samples=1)
    strategy.model.layers["background"].positions.data[0, 0] = float("nan")
    before = strategy.model.layers["background"].density.detach().clone()
    strategy.set_step_outputs({"mog_visibility": torch.ones(4)})
    assert strategy._maybe_apply_bg_road_duplicate_exclusion(0, _batch()) is False
    assert torch.equal(strategy.model.layers["background"].density, before)
    assert strategy._bg_road_duplicate_n_samples == 0


def test_resume_initialization_drops_coordinate_indexed_evidence():
    strategy = _strategy()
    strategy._bg_road_duplicate_counts = {"visible_hits": torch.ones(4)}
    strategy._bg_road_duplicate_n_samples = 7
    strategy.sub_strategies = {}
    strategy.init_densification_buffer(checkpoint={"ignored": True})
    assert strategy._bg_road_duplicate_counts is None
    assert strategy._bg_road_duplicate_n_samples == 0
