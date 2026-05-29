# SPDX-License-Identifier: Apache-2.0
"""V3-R1.2 unit tests for per-layer scale clamp + anisotropy via LayerSpec."""
from __future__ import annotations

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.layers.registry import STANDARD_LAYERS


def test_layerspec_default_scale_clamps_are_none():
    spec = LayerSpec(name="x", layer_id=0, max_n_particles=100)
    assert spec.scale_xy_max is None
    assert spec.scale_z_max is None
    assert spec.anisotropy_ratio_max is None


def test_road_layer_clamps_set():
    """V3-R1.2 acceptance: road layer caps scale (XY <= 0.3m, Z <= 0.05m)
    and anisotropy ratio (max/min eigenvalue <= 8x)."""
    s = STANDARD_LAYERS["road"]
    assert s.scale_xy_max == 0.3
    assert s.scale_z_max == 0.05
    assert s.anisotropy_ratio_max == 8.0


def test_background_layer_clamps_not_set():
    s = STANDARD_LAYERS["background"]
    assert s.scale_xy_max is None
    assert s.scale_z_max is None
    assert s.anisotropy_ratio_max is None
