# SPDX-License-Identifier: Apache-2.0
"""V3-L5/L8/L9 metrics.json diagnostic field unit tests.

The render.py eval loop block computes 4 fields from the model state:
    * symmetric_axis (echo of LayerSpec.extra value)
    * track_albedo_l2_mean (None when OFF)
    * track_log_scale_mean (None when OFF)
    * track_log_scale_std (None when OFF or |table| < 2)

This test exercises the same arithmetic on a synthetic LayeredGaussians so
we don't need to run a full A800 eval to know the formulas are correct.
"""

from __future__ import annotations

import os

import pytest
import torch
from hydra import compose, initialize_config_dir

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.layers.layered_model import LayeredGaussians

_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs"))


@pytest.fixture(scope="module")
def real_conf():
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="apps/ncore_3dgut_mcmc")


def _make_model(real_conf, *, albedo: bool, scale: bool, symmetric_axis=None) -> LayeredGaussians:
    extra = {}
    if symmetric_axis is not None:
        extra["symmetric_axis"] = symmetric_axis
    if albedo:
        extra["optimize_track_albedo"] = True
    if scale:
        extra["optimize_track_scale"] = True
    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(
            name="dynamic_rigids", layer_id=2, max_n_particles=200_000, scale_prior=(0.05, 0.05, 0.05), extra=extra
        ),
    ]
    eye = torch.eye(4).expand(5, 4, 4).clone()
    tracks = {
        "alice": {"poses": eye.clone(), "active": torch.ones(5, dtype=torch.bool)},
        "bob": {"poses": eye.clone(), "active": torch.ones(5, dtype=torch.bool)},
    }
    return LayeredGaussians(real_conf, specs=specs, scene_extent=10.0, tracks=tracks)


def _compute_v3_metrics(model) -> dict:
    """Mirror of the render.py block (so this unit test exercises the same
    arithmetic without running the real eval loop)."""
    albedo_t = getattr(model, "_track_albedo_table", None)
    log_scale_t = getattr(model, "_track_log_scale_table", None)
    sym_axis_val = None
    specs = getattr(model, "specs", None)
    if specs is not None:
        dyn = next((s for s in specs if s.name == "dynamic_rigids"), None)
        if dyn is not None:
            sym_axis_val = (getattr(dyn, "extra", {}) or {}).get("symmetric_axis")
    return {
        "symmetric_axis": sym_axis_val,
        "track_albedo_l2_mean": (float(albedo_t.detach().norm(dim=-1).mean().cpu()) if albedo_t is not None else None),
        "track_log_scale_mean": (float(log_scale_t.detach().mean().cpu()) if log_scale_t is not None else None),
        "track_log_scale_std": (
            float(log_scale_t.detach().std().cpu()) if log_scale_t is not None and log_scale_t.numel() > 1 else None
        ),
    }


def test_v3_metrics_all_null_when_off(real_conf):
    model = _make_model(real_conf, albedo=False, scale=False, symmetric_axis=None)
    m = _compute_v3_metrics(model)
    assert m == {
        "symmetric_axis": None,
        "track_albedo_l2_mean": None,
        "track_log_scale_mean": None,
        "track_log_scale_std": None,
    }


def test_v3_metrics_symmetric_axis_echo_when_set(real_conf):
    """symmetric_axis is reported even when the param tables are OFF
    (it's a yaml-time setting, no runtime tensor)."""
    model = _make_model(real_conf, albedo=False, scale=False, symmetric_axis="Y")
    m = _compute_v3_metrics(model)
    assert m["symmetric_axis"] == "Y"
    assert m["track_albedo_l2_mean"] is None
    assert m["track_log_scale_mean"] is None


def test_v3_metrics_zero_init_yields_zero(real_conf):
    """Tables exist but zero-initialised → l2_mean = 0, scale_mean = 0,
    scale_std = 0. None is reserved for the OFF code path."""
    model = _make_model(real_conf, albedo=True, scale=True, symmetric_axis="Y")
    m = _compute_v3_metrics(model)
    assert m["symmetric_axis"] == "Y"
    assert m["track_albedo_l2_mean"] == pytest.approx(0.0, abs=1e-6)
    assert m["track_log_scale_mean"] == pytest.approx(0.0, abs=1e-6)
    assert m["track_log_scale_std"] == pytest.approx(0.0, abs=1e-6)


def test_v3_metrics_non_zero_values(real_conf):
    """Inject known table values; verify the diagnostic formulas.

    P1.3b: the albedo table is [K, 3, k] Fourier coefficients (k=1 default).
    render.py's diagnostic (render.py:1496) is ``norm(dim=-1).mean()`` — norm
    over the Fourier axis, then mean over K×3 — so for k=1 each entry's norm
    is just |coef|.
    """
    model = _make_model(real_conf, albedo=True, scale=True)
    with torch.no_grad():
        # [K=2, 3, k=1]; norm over last(k) dim → |val| per (track, channel);
        # mean over 2×3 = (0.4+0.3+0.0 + 0.0+0.5+0.0) / 6 = 0.2
        model._track_albedo_table.copy_(
            torch.tensor(
                [
                    [[0.4], [0.3], [0.0]],
                    [[0.0], [0.5], [0.0]],
                ]
            )
        )
        # log_scale = [0.2, -0.4] → mean = -0.1, std (unbiased) ≈ 0.4243
        model._track_log_scale_table.copy_(torch.tensor([[0.2], [-0.4]]))

    m = _compute_v3_metrics(model)
    assert m["track_albedo_l2_mean"] == pytest.approx(0.2, abs=1e-5)
    assert m["track_log_scale_mean"] == pytest.approx(-0.1, abs=1e-5)
    # std unbiased: sqrt(((0.2-(-0.1))^2 + (-0.4-(-0.1))^2) / (2-1)) = sqrt(0.18) ≈ 0.4243
    assert m["track_log_scale_std"] == pytest.approx(0.4243, abs=1e-3)


def test_v3_metrics_single_track_log_scale_std_is_none(real_conf):
    """numel() < 2 → std is undefined (torch.std with N=1 returns NaN);
    diagnostic guards against that with the `numel > 1` check."""
    # Override _make_model to give a single track.
    extra = {"optimize_track_scale": True}
    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(
            name="dynamic_rigids", layer_id=2, max_n_particles=200_000, scale_prior=(0.05, 0.05, 0.05), extra=extra
        ),
    ]
    eye = torch.eye(4).expand(3, 4, 4).clone()
    tracks = {
        "solo": {"poses": eye.clone(), "active": torch.ones(3, dtype=torch.bool)},
    }
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0, tracks=tracks)
    m = _compute_v3_metrics(model)
    assert m["track_log_scale_std"] is None
    assert m["track_log_scale_mean"] == pytest.approx(0.0)
