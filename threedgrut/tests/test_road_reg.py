# SPDX-License-Identifier: Apache-2.0
"""V3-R1 unit tests for road_reg pure functions."""
from __future__ import annotations

import math

import pytest
import torch

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.model.road_reg import clamp_layer_scales


def _make_road_spec(**kwargs):
    base = dict(
        name="road", layer_id=1, max_n_particles=100,
        scale_xy_max=0.3, scale_z_max=0.05, anisotropy_ratio_max=8.0,
    )
    base.update(kwargs)
    return LayerSpec(**base)


def test_clamp_no_op_when_spec_has_no_clamps():
    spec = LayerSpec(name="x", layer_id=0, max_n_particles=10)
    scale_log = torch.randn(10, 3)
    out = clamp_layer_scales(scale_log, spec)
    assert torch.equal(out, scale_log)


def test_clamp_xy_upper_bound():
    spec = LayerSpec(name="r", layer_id=1, max_n_particles=10, scale_xy_max=0.3)
    scale_log = torch.zeros(4, 3)  # exp = 1.0m, above 0.3
    out = clamp_layer_scales(scale_log, spec)
    out_exp = torch.exp(out)
    assert torch.all(out_exp[:, 0] <= 0.3 + 1e-6)
    assert torch.all(out_exp[:, 1] <= 0.3 + 1e-6)
    assert torch.allclose(out_exp[:, 2], torch.tensor(1.0))  # Z untouched


def test_clamp_z_upper_bound():
    spec = LayerSpec(name="r", layer_id=1, max_n_particles=10, scale_z_max=0.05)
    scale_log = torch.zeros(4, 3)
    out = clamp_layer_scales(scale_log, spec)
    assert torch.all(torch.exp(out)[:, 2] <= 0.05 + 1e-6)


def test_clamp_anisotropy_ratio():
    spec = _make_road_spec(scale_xy_max=None, scale_z_max=None,
                            anisotropy_ratio_max=4.0)
    scale_log = torch.tensor([[0.0, 0.0, math.log(0.05)]])  # ratio 20x
    out = clamp_layer_scales(scale_log, spec)
    out_exp = torch.exp(out)
    ratio = out_exp.max(dim=-1).values / out_exp.min(dim=-1).values
    assert torch.all(ratio <= 4.0 + 1e-5)


def test_clamp_combined_xy_z_anisotropy_road():
    spec = _make_road_spec()
    scale_log = torch.log(torch.tensor([
        [0.5, 0.5, 0.001],
        [0.2, 0.2, 0.04],
        [0.1, 0.1, 0.5],
    ]))
    out = clamp_layer_scales(scale_log, spec)
    out_exp = torch.exp(out)
    assert torch.all(out_exp[:, :2] <= 0.3 + 1e-6)
    assert torch.all(out_exp[:, 2] <= 0.05 + 1e-6)
    ratio = out_exp.max(dim=-1).values / out_exp.min(dim=-1).values
    assert torch.all(ratio <= 8.0 + 1e-5)


def test_clamp_returns_same_dtype_device():
    spec = _make_road_spec()
    scale_log = torch.zeros(4, 3, dtype=torch.float32)
    out = clamp_layer_scales(scale_log, spec)
    assert out.dtype == torch.float32
    assert out.device == scale_log.device


def test_clamp_does_not_mutate_input():
    spec = _make_road_spec()
    scale_log = torch.zeros(4, 3)
    _ = clamp_layer_scales(scale_log, spec)
    assert torch.all(scale_log == 0.0)


def test_clamp_hard_caps_win_over_tight_ratio():
    """When ratio < xy_max/z_max, the Z cap still holds (caps beat ratio)."""
    spec = _make_road_spec(scale_xy_max=0.3, scale_z_max=0.05,
                            anisotropy_ratio_max=4.0)  # 4 < 0.3/0.05=6 → tight
    scale_log = torch.log(torch.tensor([[0.3, 0.3, 0.001]]))  # thin needle
    out = clamp_layer_scales(scale_log, spec)
    out_exp = torch.exp(out)
    assert torch.all(out_exp[:, :2] <= 0.3 + 1e-6)
    assert torch.all(out_exp[:, 2] <= 0.05 + 1e-6), "Z cap must hold even for tight ratio"
