# SPDX-License-Identifier: Apache-2.0
"""V3-L8 + V3-L9 unit tests for per-track albedo / log-scale tables.

Covers:
    (a) ``optimize_track_albedo=False`` + ``optimize_track_scale=False``
        → no tables registered (regression pin, byte-identical ckpt schema)
    (b) ``optimize_track_albedo=True`` registers ``_track_albedo_table``
        as Parameter[K, 3], zero-init, ``requires_grad=False`` (warmup gate)
    (c) ``optimize_track_scale=True`` registers ``_track_log_scale_table``
        as Parameter[K, 1], zero-init, ``requires_grad=False``
    (d) ``fused_view`` with zero-init tables is identity for features_albedo
        and scale (so a freshly-enabled flag doesn't break a smoke run)
    (e) Non-zero albedo table → fused_view ``features_albedo`` shifted by
        per-track bias correctly indexed by ``track_ids``
    (f) Non-zero log-scale table → fused_view ``scale`` shifted in log-space
        (broadcast across all 3 axes, so ``exp(scale_new)`` = ``exp(off)`` ×
        ``exp(scale_orig)``)
    (g) Order independence with MCMC: the table application is a *post*
        gather operation on ``scale``, so any earlier perturb mutation on
        ``layer.scale`` (the Parameter) is preserved untouched.

The conftest pattern matches test_layered_gaussians.py (real Hydra conf).
"""
from __future__ import annotations

import os

import pytest
import torch
from hydra import compose, initialize_config_dir

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.layers.layered_model import LayeredGaussians

_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs")
)


@pytest.fixture(scope="module")
def real_conf():
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="apps/ncore_3dgut_mcmc")


def _two_track_dict(F: int = 5) -> dict:
    eye = torch.eye(4).expand(F, 4, 4).clone()
    return {
        "alice": {"poses": eye.clone(), "active": torch.ones(F, dtype=torch.bool),
                  "class": "automobile", "size": torch.tensor([4.0, 2.0, 1.5])},
        "bob":   {"poses": eye.clone(), "active": torch.ones(F, dtype=torch.bool),
                  "class": "automobile", "size": torch.tensor([4.0, 2.0, 1.5])},
    }


def _build_model(real_conf, *, albedo: bool, scale: bool,
                 n_pts_per_track: int = 4,
                 n_fourier: int | None = None) -> tuple[LayeredGaussians, list[str]]:
    """bg + dyn LayeredGaussians with V3-L8/L9 toggles, 2 tracks, n_pts each.

    Returns (model, sorted_track_names).
    """
    extra = {}
    if albedo:
        extra["optimize_track_albedo"] = True
    if n_fourier is not None:
        extra["n_fourier_albedo_terms"] = int(n_fourier)
    if scale:
        extra["optimize_track_scale"] = True
    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="dynamic_rigids", layer_id=2, max_n_particles=200_000,
                  scale_prior=(0.05, 0.05, 0.05), extra=extra),
    ]
    tracks = _two_track_dict()
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0,
                             tracks=tracks)
    # bg init (small)
    model.init_layer_from_points("background", torch.randn(5, 3),
                                 setup_optimizer=False)
    # dyn init: n_pts_per_track local-origin points per track
    track_names = sorted(tracks.keys())
    all_pts = []
    all_ids = []
    for tid in track_names:
        all_pts.append(torch.zeros(n_pts_per_track, 3))
        all_ids.append(torch.full((n_pts_per_track,),
                                  track_names.index(tid), dtype=torch.long))
    model.init_layer_from_points("dynamic_rigids",
                                 torch.cat(all_pts),
                                 track_ids=torch.cat(all_ids),
                                 setup_optimizer=False)
    return model, track_names


# --- (a) OFF: no tables registered (regression pin) -------------------------
def test_tables_not_registered_when_both_off(real_conf):
    model, _ = _build_model(real_conf, albedo=False, scale=False)
    assert not hasattr(model, "_track_albedo_table"), \
        "OFF mode must NOT register _track_albedo_table"
    assert not hasattr(model, "_track_log_scale_table"), \
        "OFF mode must NOT register _track_log_scale_table"
    # state_dict roundtrip-equivalence: keys shouldn't contain new params.
    param_names = list(dict(model.named_parameters()).keys())
    assert all("_track_albedo_table" not in n for n in param_names)
    assert all("_track_log_scale_table" not in n for n in param_names)


# --- (b) ON albedo: registered + zero-init + requires_grad=False ------------
def test_albedo_table_shape_and_init(real_conf):
    model, names = _build_model(real_conf, albedo=True, scale=False)
    assert hasattr(model, "_track_albedo_table")
    t = model._track_albedo_table
    assert isinstance(t, torch.nn.Parameter)
    # P1.3b: albedo table is [K, 3, k] Fourier coefficients; default k=1.
    assert t.shape == (len(names), 3, 1)
    assert torch.equal(t, torch.zeros_like(t)), "albedo table must zero-init"
    assert t.requires_grad is False, \
        "warmup gate: requires_grad starts False; trainer flips at step 500"


# --- (c) ON scale: registered + zero-init + requires_grad=False -------------
def test_log_scale_table_shape_and_init(real_conf):
    model, names = _build_model(real_conf, albedo=False, scale=True)
    assert hasattr(model, "_track_log_scale_table")
    t = model._track_log_scale_table
    assert isinstance(t, torch.nn.Parameter)
    assert t.shape == (len(names), 1)
    assert torch.equal(t, torch.zeros_like(t))
    assert t.requires_grad is False


# --- (d) zero-init = identity on fused_view ---------------------------------
def test_fused_view_identity_under_zero_init(real_conf):
    """A freshly-enabled run (tables exist, all zeros) must produce
    fused_view tensors byte-equal to the OFF baseline for the two
    relevant fields."""
    model_off, _ = _build_model(real_conf, albedo=False, scale=False)
    model_on, _ = _build_model(real_conf, albedo=True, scale=True)

    # Force the dyn layers to have the SAME raw features/scale so the only
    # difference under inspection is the bias application.
    layer_off = model_off.layers["dynamic_rigids"]
    layer_on = model_on.layers["dynamic_rigids"]
    with torch.no_grad():
        layer_on.features_albedo.copy_(layer_off.features_albedo)
        layer_on.scale.copy_(layer_off.scale)
        layer_on.rotation.copy_(layer_off.rotation)
        layer_on.density.copy_(layer_off.density)
        layer_on.positions.copy_(layer_off.positions)
        if hasattr(layer_on, "features_specular"):
            layer_on.features_specular.copy_(layer_off.features_specular)

    fv_off = model_off.fused_view()
    fv_on = model_on.fused_view()
    assert torch.allclose(fv_off["features_albedo"], fv_on["features_albedo"]), \
        "zero albedo bias must not change DC SH"
    assert torch.allclose(fv_off["scale"], fv_on["scale"]), \
        "zero log-scale offset must not change scale"


# --- (e) non-zero albedo → per-track shift ----------------------------------
def test_albedo_table_shifts_features_per_track(real_conf):
    model, names = _build_model(real_conf, albedo=True, scale=False,
                                n_pts_per_track=3)
    with torch.no_grad():
        # Distinct per-track bias so we can identify them in the output.
        # P1.3b: table is [K, 3, k]; default k=1 so the DC slot is [..., 0].
        # A constant (frame-independent) bias = only the DC Fourier term.
        model._track_albedo_table.copy_(torch.tensor([
            [0.10, 0.20, 0.30],   # track 0 = alice
            [-0.10, -0.20, -0.30],  # track 1 = bob
        ]).unsqueeze(-1))
    layer = model.layers["dynamic_rigids"]
    # Zero out the layer's own features_albedo so only the bias appears.
    with torch.no_grad():
        layer.features_albedo.zero_()
    fv = model.fused_view()
    # bg layer is at the head; dyn layer at the tail. bg has 5 pts (zero-init
    # features_albedo from init_layer_from_points + MoG defaults). Just slice
    # the last 2*3=6 dyn rows.
    dyn_feat = fv["features_albedo"][-6:]
    # alice rows = first 3 (track_id 0); bob rows = last 3.
    assert torch.allclose(dyn_feat[:3], torch.tensor([0.10, 0.20, 0.30]).expand(3, 3),
                          atol=1e-6)
    assert torch.allclose(dyn_feat[3:], torch.tensor([-0.10, -0.20, -0.30]).expand(3, 3),
                          atol=1e-6)


# --- (f) non-zero log-scale → per-track log-space add -----------------------
def test_log_scale_table_shifts_scale_per_track(real_conf):
    model, names = _build_model(real_conf, albedo=False, scale=True,
                                n_pts_per_track=3)
    with torch.no_grad():
        model._track_log_scale_table.copy_(torch.tensor([[0.5], [-0.5]]))
        # Zero out the layer's own log-scale so only the offset appears.
        model.layers["dynamic_rigids"].scale.zero_()
    fv = model.fused_view()
    dyn_scale = fv["scale"][-6:]
    # alice rows (first 3): all 3 axes shifted by +0.5; bob rows: -0.5.
    assert torch.allclose(dyn_scale[:3], torch.full((3, 3), 0.5), atol=1e-6)
    assert torch.allclose(dyn_scale[3:], torch.full((3, 3), -0.5), atol=1e-6)


# --- (g) preserves underlying Parameter (only fused view shifts) ------------
def test_table_application_is_post_gather_not_mutation(real_conf):
    """V3 design §D: per-track scale offset is a fused-view *post* shift;
    it must NOT mutate ``layer.scale`` (the nn.Parameter). MCMC perturb
    on ``layer.scale`` then runs against the un-shifted parameter."""
    model, _ = _build_model(real_conf, albedo=True, scale=True,
                            n_pts_per_track=3)
    with torch.no_grad():
        # P1.3b: albedo table is [K, 3, k] (default k=1).
        model._track_albedo_table.copy_(torch.full((2, 3, 1), 0.7))
        model._track_log_scale_table.copy_(torch.full((2, 1), 0.3))
    layer = model.layers["dynamic_rigids"]
    feat_before = layer.features_albedo.detach().clone()
    scale_before = layer.scale.detach().clone()
    _ = model.fused_view()  # discard; we only care about side effects.
    assert torch.equal(layer.features_albedo.detach(), feat_before), \
        "fused_view must not mutate the underlying features_albedo Parameter"
    assert torch.equal(layer.scale.detach(), scale_before), \
        "fused_view must not mutate the underlying scale Parameter"


# --- (h) ckpt save/load: tables ride along when enabled ---------------------
def test_tables_present_in_state_dict_when_enabled(real_conf):
    """V3-L8/L9 tables must show up in state_dict so save/load roundtrips."""
    model, _ = _build_model(real_conf, albedo=True, scale=True)
    sd = model.state_dict()
    assert "_track_albedo_table" in sd
    assert "_track_log_scale_table" in sd
    # P1.3b: albedo table is [K, 3, k]; default k=1 → [K, 3, 1].
    assert sd["_track_albedo_table"].shape == (2, 3, 1)
    assert sd["_track_log_scale_table"].shape == (2, 1)


def test_tables_absent_in_state_dict_when_off(real_conf):
    """Symmetric to (a): OFF mode keeps state_dict shape identical to v2."""
    model, _ = _build_model(real_conf, albedo=False, scale=False)
    sd = model.state_dict()
    assert "_track_albedo_table" not in sd
    assert "_track_log_scale_table" not in sd


# ===========================================================================
# P1.3b — Fourier (4D-SH) time-varying albedo INTEGRATION (real model + fused_view)
# ===========================================================================

# --- (i) n_fourier=4 registers a [K, 3, 4] table ----------------------------
def test_fourier_table_shape_k4(real_conf):
    model, names = _build_model(real_conf, albedo=True, scale=False, n_fourier=4)
    t = model._track_albedo_table
    assert t.shape == (len(names), 3, 4)
    assert torch.equal(t, torch.zeros_like(t))
    assert t.requires_grad is False


# --- (j) k=1 fused_view is frame-independent (DC-only degeneracy) ------------
def test_fourier_k1_frame_independent(real_conf):
    """k=1 → fused_view features_albedo identical across all frame_ids."""
    model, _ = _build_model(real_conf, albedo=True, scale=False,
                            n_pts_per_track=3, n_fourier=1)
    with torch.no_grad():
        model._track_albedo_table.copy_(
            torch.tensor([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]).unsqueeze(-1)
        )
        model.layers["dynamic_rigids"].features_albedo.zero_()
    # F=5 in _two_track_dict; sweep all frames.
    fv0 = model.fused_view(frame_id=0)["features_albedo"][-6:]
    for t in range(1, 5):
        fvt = model.fused_view(frame_id=t)["features_albedo"][-6:]
        assert torch.allclose(fv0, fvt, atol=1e-6), \
            f"k=1 must be frame-independent; differs at frame {t}"


# --- (k) k>1 fused_view varies with frame_id --------------------------------
def test_fourier_k4_varies_with_frame(real_conf):
    """k=4 with a non-zero first harmonic → bias changes across frames."""
    model, _ = _build_model(real_conf, albedo=True, scale=False,
                            n_pts_per_track=3, n_fourier=4)
    with torch.no_grad():
        # track 0: DC=0.1 on R + first harmonic=0.5 on R.
        tab = torch.zeros(2, 3, 4)
        tab[0, 0, 0] = 0.1   # DC
        tab[0, 0, 1] = 0.5   # 1st harmonic, channel R
        model._track_albedo_table.copy_(tab)
        model.layers["dynamic_rigids"].features_albedo.zero_()
    import math as _math
    N_t = 5
    # alice rows are the first 3 dyn rows (track 0).
    for t in range(N_t):
        fv = model.fused_view(frame_id=t)["features_albedo"][-6:]
        expected_r = 0.1 + 0.5 * _math.cos(_math.pi * t / N_t)
        assert torch.allclose(
            fv[:3, 0], torch.full((3,), expected_r), atol=1e-5
        ), f"frame {t}: R channel mismatch"
        # G/B channels untouched → 0.
        assert torch.allclose(fv[:3, 1:], torch.zeros(3, 2), atol=1e-6)


# --- (l) k>1 zero-init is still identity (smoke-safe) -----------------------
def test_fourier_zero_init_identity(real_conf):
    """A freshly-enabled k=4 run (all-zero table) must not change features."""
    model_off, _ = _build_model(real_conf, albedo=False, scale=False)
    model_on, _ = _build_model(real_conf, albedo=True, scale=False, n_fourier=4)
    layer_off = model_off.layers["dynamic_rigids"]
    layer_on = model_on.layers["dynamic_rigids"]
    with torch.no_grad():
        layer_on.features_albedo.copy_(layer_off.features_albedo)
        layer_on.positions.copy_(layer_off.positions)
        layer_on.rotation.copy_(layer_off.rotation)
        layer_on.density.copy_(layer_off.density)
        layer_on.scale.copy_(layer_off.scale)
        if hasattr(layer_on, "features_specular"):
            layer_on.features_specular.copy_(layer_off.features_specular)
    fv_off = model_off.fused_view(frame_id=2)
    fv_on = model_on.fused_view(frame_id=2)
    assert torch.allclose(
        fv_off["features_albedo"], fv_on["features_albedo"], atol=1e-7
    )


# --- (m) ckpt back-compat: old [K,3] DC table loads into a k=4 model --------
def test_ckpt_backcompat_old_dc_into_fourier_model(real_conf):
    """A P1.3-era ckpt saved its albedo table as [K, 3]. Loading it into a
    P1.3b k=4 model must place the DC values in term 0 and zero the rest."""
    model, _ = _build_model(real_conf, albedo=True, scale=False, n_fourier=4)
    dc_vals = torch.tensor([[0.7, -0.1, 0.2], [0.0, 0.3, -0.5]])  # [K, 3]
    # Simulate an old-style ckpt: track_optim_state.tables.albedo = [K, 3].
    old_ckpt = {
        "gaussians_nodes": {},  # per-layer init handled separately; empty OK
        "track_optim_state": {
            "tables": {"albedo": dc_vals.clone()},
        },
    }
    model.init_from_checkpoint(old_ckpt, setup_optimizer=False)
    t = model._track_albedo_table.detach().cpu()
    assert t.shape == (2, 3, 4)
    assert torch.allclose(t[..., 0], dc_vals, atol=1e-6), \
        "old DC values must land in Fourier term 0"
    assert torch.allclose(t[..., 1:], torch.zeros(2, 3, 3), atol=1e-7), \
        "higher harmonics must be zero after upgrade"


# --- (n) warmup gate still flips the [K,3,k] table --------------------------
def test_warmup_gate_flips_fourier_table(real_conf):
    model, _ = _build_model(real_conf, albedo=True, scale=False, n_fourier=4)
    assert model._track_albedo_table.requires_grad is False
    # Below warmup (default 500) → no flip.
    assert model.maybe_activate_track_params(0) is False
    assert model._track_albedo_table.requires_grad is False
    # At/after warmup → flip once.
    flipped = model.maybe_activate_track_params(500)
    assert flipped is True
    assert model._track_albedo_table.requires_grad is True
