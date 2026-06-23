# SPDX-License-Identifier: Apache-2.0
"""E3.3 Task 1 — road colour rendering path samples the BEV feature grid.

Mirrors test_track_albedo_scale_params.py (real Hydra conf + bg/road
LayeredGaussians, CPU). Verifies the pre-bake injection in ``fused_view``:
road ``features_albedo`` is overridden by the BEV grid sample (→ SH DC), road
``features_specular`` is zeroed (DC-only), non-road layers and the underlying
Parameters are untouched, gradients flow back to the grid, and the disabled
path is byte-identical (regression pin).
"""
from __future__ import annotations

import os

import pytest
import torch
from hydra import compose, initialize_config_dir

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.layers.layered_model import LayeredGaussians
from threedgrut.model.bev_texture import sample_bev_feature, bev_feature_to_sh_dc

_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs")
)
_C0 = 0.28209479177387814


@pytest.fixture(scope="module")
def real_conf():
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="apps/ncore_3dgut_mcmc")


def _flat_road_pts(n: int = 3) -> torch.Tensor:
    xs = torch.linspace(0.0, 4.0, n)
    xy = torch.stack(torch.meshgrid(xs, xs, indexing="ij"), dim=-1).reshape(-1, 2)
    return torch.cat([xy, torch.zeros(xy.shape[0], 1)], dim=-1)


def _build_model(real_conf, *, bev: bool, cell_size: float = 1.0,
                 n_channels: int = 3, road_colors=None, n: int = 3
                 ) -> LayeredGaussians:
    extra: dict = {}
    if bev:
        extra["bev_road_texture"] = True
        extra["bev_cell_size"] = cell_size
        extra["bev_channels"] = n_channels
    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="road", layer_id=1, max_n_particles=200_000,
                  scale_prior=(0.1, 0.1, 0.001), extra=extra),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    model.init_layer_from_points("background", torch.randn(5, 3),
                                 setup_optimizer=False)
    model.init_layer_from_points("road", _flat_road_pts(n),
                                 colors=road_colors, setup_optimizer=False)
    return model


# --- (a) registration gate (regression pin: OFF adds nothing) ---------------
def test_bev_grid_registered_when_enabled(real_conf):
    model = _build_model(real_conf, bev=True)
    assert hasattr(model, "_road_bev_grid")
    assert isinstance(model._road_bev_grid, torch.nn.Parameter)
    assert "_road_bev_grid" in dict(model.named_parameters())


def test_bev_grid_not_registered_when_disabled(real_conf):
    model = _build_model(real_conf, bev=False)
    assert not hasattr(model, "_road_bev_grid")
    assert all("_road_bev_grid" not in k for k in dict(model.named_parameters()))


# --- (b) road albedo overridden by the grid sample --------------------------
def test_fused_view_road_albedo_from_grid(real_conf):
    model = _build_model(real_conf, bev=True)
    model._road_bev_grid.data.uniform_(0.1, 0.9)  # non-trivial pattern
    fv = model.fused_view()
    road_mask = model.get_layer_mask("road")
    road_alb = fv["features_albedo"][road_mask]
    road_xy = model.layers["road"].positions[:, :2].detach()
    expected = bev_feature_to_sh_dc(
        sample_bev_feature(road_xy, model._road_bev_struct()))
    assert torch.allclose(road_alb, expected, atol=1e-5)


# --- (c) road specular zeroed (DC-only) + uniform width ---------------------
def test_fused_view_road_specular_zeroed(real_conf):
    model = _build_model(real_conf, bev=True)
    # non-zero specular proves the zeroing actually fires (not a default 0)
    model.layers["road"].features_specular.data.fill_(1.0)
    fv = model.fused_view()
    road_mask = model.get_layer_mask("road")
    road_spec = fv["features_specular"][road_mask]
    assert torch.equal(road_spec, torch.zeros_like(road_spec))
    # uniform specular width across layers (fused-renderer invariant)
    assert (fv["features_specular"].shape[1]
            == model.layers["background"].features_specular.shape[1])


# --- (d) non-road layers untouched ------------------------------------------
def test_fused_view_non_road_unaffected(real_conf):
    model = _build_model(real_conf, bev=True)
    fv = model.fused_view()
    bg_mask = model.get_layer_mask("background")
    bg = model.layers["background"]
    assert torch.equal(fv["features_albedo"][bg_mask], bg.features_albedo)
    assert torch.equal(fv["features_specular"][bg_mask], bg.features_specular)


# --- (e) does not mutate the underlying road Parameter ----------------------
def test_fused_view_does_not_mutate_road_parameter(real_conf):
    model = _build_model(real_conf, bev=True)
    before = model.layers["road"].features_albedo.detach().clone()
    _ = model.fused_view()
    assert torch.equal(model.layers["road"].features_albedo.detach(), before)


# --- (f) gradient flows render-colour → grid (kernel-free full chain) -------
def test_grad_flows_from_road_albedo_to_grid(real_conf):
    model = _build_model(real_conf, bev=True)
    fv = model.fused_view()
    road_mask = model.get_layer_mask("road")
    fv["features_albedo"][road_mask].sum().backward()
    g = model._road_bev_grid.grad
    assert g is not None and g.abs().sum() > 0


# --- (g) grid init at road colour mean → smooth takeover --------------------
def test_grid_init_matches_road_color_mean(real_conf):
    n = 3
    rgb = torch.tensor([0.3, 0.6, 0.9])
    road_colors = rgb.expand(n * n, 3).contiguous()
    model = _build_model(real_conf, bev=True, road_colors=road_colors, n=n)
    fv = model.fused_view()
    road_mask = model.get_layer_mask("road")
    road_alb = fv["features_albedo"][road_mask]
    expected_dc = bev_feature_to_sh_dc(rgb.view(1, 3))
    assert torch.allclose(road_alb, expected_dc.expand_as(road_alb), atol=1e-4)


# --- (h) disabled is byte-identical (regression pin) ------------------------
def test_disabled_is_identity(real_conf):
    model = _build_model(real_conf, bev=False)
    road = model.layers["road"]
    fv = model.fused_view()
    road_mask = model.get_layer_mask("road")
    assert torch.equal(fv["features_albedo"][road_mask], road.features_albedo)
    assert torch.equal(fv["features_specular"][road_mask], road.features_specular)
