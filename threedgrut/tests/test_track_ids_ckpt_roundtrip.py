# SPDX-License-Identifier: Apache-2.0
"""T8/B3 Phase E.4 — ``track_ids`` buffer survives ckpt save/load.

Without this, viewer/playground loads of v2 ckpts lose the per-particle owner
mapping; ``_transform_means`` indexes pose_stack with whatever stale buffer
existed → particles render at wrong cuboid → bg layer ends up explaining
vehicle pixels and the "勾掉 dynamic_rigids 视觉无变化" bug surfaces.

Tests pin three properties:
  1. ``LayeredGaussians.get_model_parameters`` includes ``track_ids`` under
     each particle layer that has the buffer registered.
  2. ``LayeredGaussians.init_from_checkpoint`` restores it as a persistent
     int64 buffer on the corresponding MoG layer.
  3. The roundtrip is shape-/dtype-stable (torch.save + torch.load).
"""
from __future__ import annotations

import os

import pytest
import torch
from hydra import compose, initialize_config_dir

from threedgrut.layers.layered_model import LayeredGaussians
from threedgrut.layers.registry import specs_from_config

_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs")
)


@pytest.fixture(scope="module")
def real_conf():
    """Hydra-composed full conf from apps/ncore_3dgut_mcmc (matches the
    fixture in test_layered_gaussians.py so config schemas align)."""
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="apps/ncore_3dgut_mcmc")


def _with_dyn_layer(conf):
    """Return a conf identical to base but with ['background','dynamic_rigids']
    enabled — registry.specs_from_config reads this list."""
    from copy import deepcopy
    c = deepcopy(conf)
    c.layers = {"enabled": ["background", "dynamic_rigids"]}
    return c


def _seed_dyn_layer(model: LayeredGaussians, n: int = 20) -> torch.Tensor:
    """Initialize the dynamic_rigids layer with N particles and known
    track_ids (4 distinct owners, repeating). Also seeds a background-layer
    placeholder so the bg sub-MoG has an optimizer to satisfy
    ``MoG.get_model_parameters`` invariants."""
    bg_pos = torch.randn(4, 3) * 0.1
    model.init_layer_from_points("background", bg_pos, setup_optimizer=False)
    positions = torch.randn(n, 3) * 0.1
    track_ids = torch.tensor([i % 4 for i in range(n)], dtype=torch.int64)
    model.init_layer_from_points(
        "dynamic_rigids", positions, track_ids=track_ids, setup_optimizer=False,
    )
    # Attach test-only optimizers across all layers so get_model_parameters
    # passes the "optimizer is not None" assertion in MoG.
    model.setup_optimizer_for_test()
    return track_ids


# -----------------------------------------------------------------------------
# save side
# -----------------------------------------------------------------------------

def test_get_model_parameters_includes_track_ids_for_dyn_layer(real_conf):
    conf = _with_dyn_layer(real_conf)
    model = LayeredGaussians(conf, specs=specs_from_config(conf), scene_extent=10.0)
    expected_ids = _seed_dyn_layer(model, n=12)

    params = model.get_model_parameters()
    nodes = params["gaussians_nodes"]
    assert "dynamic_rigids" in nodes
    assert "track_ids" in nodes["dynamic_rigids"], \
        "dynamic_rigids node must carry track_ids on save"
    saved = nodes["dynamic_rigids"]["track_ids"]
    assert saved.dtype == torch.int64
    assert torch.equal(saved.cpu(), expected_ids.cpu())


def test_get_model_parameters_omits_track_ids_when_layer_has_none(real_conf):
    """The background layer never gets track_ids; the save dict must not
    inject a stray key on it."""
    # default real_conf has layers.enabled = ['background'] only
    model = LayeredGaussians(real_conf, specs=specs_from_config(real_conf), scene_extent=10.0)
    bg_pos = torch.randn(4, 3) * 0.1
    model.init_layer_from_points("background", bg_pos, setup_optimizer=False)
    model.setup_optimizer_for_test()
    params = model.get_model_parameters()
    nodes = params["gaussians_nodes"]
    assert "background" in nodes
    assert "track_ids" not in nodes["background"]


# -----------------------------------------------------------------------------
# load side
# -----------------------------------------------------------------------------

def test_init_from_checkpoint_restores_track_ids(real_conf):
    conf = _with_dyn_layer(real_conf)
    model_a = LayeredGaussians(conf, specs=specs_from_config(conf), scene_extent=10.0)
    expected_ids = _seed_dyn_layer(model_a, n=20)
    ckpt = {"model": model_a.get_model_parameters()}

    model_b = LayeredGaussians(conf, specs=specs_from_config(conf), scene_extent=10.0)
    model_b.init_from_checkpoint(ckpt, setup_optimizer=False)

    assert hasattr(model_b.layers["dynamic_rigids"], "track_ids")
    restored = model_b.layers["dynamic_rigids"].track_ids
    assert restored.dtype == torch.int64
    assert torch.equal(restored.cpu(), expected_ids.cpu())


def test_roundtrip_via_torch_save_load(real_conf, tmp_path):
    """Full pickled-disk roundtrip — emulates the production save_checkpoint /
    load path that the playground engine uses."""
    conf = _with_dyn_layer(real_conf)
    model_a = LayeredGaussians(conf, specs=specs_from_config(conf), scene_extent=10.0)
    expected_ids = _seed_dyn_layer(model_a, n=15)

    ckpt_path = tmp_path / "test_ckpt.pt"
    torch.save({"model": model_a.get_model_parameters()}, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_b = LayeredGaussians(conf, specs=specs_from_config(conf), scene_extent=10.0)
    model_b.init_from_checkpoint(ckpt, setup_optimizer=False)

    restored = model_b.layers["dynamic_rigids"].track_ids
    assert torch.equal(restored.cpu(), expected_ids.cpu())


def test_init_from_checkpoint_no_track_ids_key_does_not_attach_buffer(real_conf):
    """v1 ckpts / pre-E4 ckpts don't have track_ids; loading must not fail
    nor inject a bogus buffer."""
    conf = _with_dyn_layer(real_conf)
    model = LayeredGaussians(conf, specs=specs_from_config(conf), scene_extent=10.0)
    # Build both layers WITHOUT a track_ids buffer (v1-shaped state).
    model.init_layer_from_points("background", torch.randn(4, 3) * 0.1, setup_optimizer=False)
    model.init_layer_from_points(
        "dynamic_rigids", torch.randn(8, 3) * 0.1, setup_optimizer=False,
    )
    model.setup_optimizer_for_test()
    assert not hasattr(model.layers["dynamic_rigids"], "track_ids")

    ckpt_nodes = model.get_model_parameters()
    assert "track_ids" not in ckpt_nodes["gaussians_nodes"]["dynamic_rigids"]

    model_b = LayeredGaussians(conf, specs=specs_from_config(conf), scene_extent=10.0)
    model_b.init_from_checkpoint({"model": ckpt_nodes}, setup_optimizer=False)
    assert not hasattr(model_b.layers["dynamic_rigids"], "track_ids")


def test_idempotent_reload_replaces_buffer(real_conf):
    """Loading twice must replace the buffer, not duplicate-register it."""
    conf = _with_dyn_layer(real_conf)
    model = LayeredGaussians(conf, specs=specs_from_config(conf), scene_extent=10.0)
    _seed_dyn_layer(model, n=10)
    ckpt = {"model": model.get_model_parameters()}

    model_b = LayeredGaussians(conf, specs=specs_from_config(conf), scene_extent=10.0)
    model_b.init_from_checkpoint(ckpt, setup_optimizer=False)
    model_b.init_from_checkpoint(ckpt, setup_optimizer=False)  # second time
    assert hasattr(model_b.layers["dynamic_rigids"], "track_ids")


def test_state_dict_persistence_drives_full_module_save(real_conf):
    """register_buffer(..., persistent=True) puts track_ids into
    LayeredGaussians.state_dict() under ``layers.dynamic_rigids.track_ids``."""
    conf = _with_dyn_layer(real_conf)
    model = LayeredGaussians(conf, specs=specs_from_config(conf), scene_extent=10.0)
    expected_ids = _seed_dyn_layer(model, n=12)
    sd = model.state_dict()
    assert "layers.dynamic_rigids.track_ids" in sd
    assert torch.equal(sd["layers.dynamic_rigids.track_ids"].cpu(),
                       expected_ids.cpu())


# -----------------------------------------------------------------------------
# V3-L8/L9: per-track albedo / log-scale ckpt roundtrip
# -----------------------------------------------------------------------------

def _with_dyn_v3_layer(conf, *, albedo: bool = True, scale: bool = True):
    """Same as ``_with_dyn_layer`` but also flips V3-L8/L9 extras on."""
    from copy import deepcopy
    c = deepcopy(conf)
    overrides = {}
    if albedo:
        overrides["optimize_track_albedo"] = True
    if scale:
        overrides["optimize_track_scale"] = True
    overrides["track_warmup_steps"] = 500
    c.layers = {
        "enabled": ["background", "dynamic_rigids"],
        "overrides": {"dynamic_rigids": overrides},
    }
    return c


def _seed_dyn_v3_layer(model, *, n: int = 20):
    """``_seed_dyn_layer`` + populate_tracks (so V3 tables are registered)."""
    bg_pos = torch.randn(4, 3) * 0.1
    model.init_layer_from_points("background", bg_pos, setup_optimizer=False)
    # populate_tracks → registers _track_albedo_table and/or _track_log_scale_table.
    eye = torch.eye(4).expand(5, 4, 4).clone()
    tracks = {
        "alice": {"poses": eye.clone(), "active": torch.ones(5, dtype=torch.bool)},
        "bob": {"poses": eye.clone(), "active": torch.ones(5, dtype=torch.bool)},
    }
    model.populate_tracks(tracks)
    track_ids = torch.tensor([i % 2 for i in range(n)], dtype=torch.int64)
    positions = torch.randn(n, 3) * 0.1
    model.init_layer_from_points(
        "dynamic_rigids", positions, track_ids=track_ids, setup_optimizer=False,
    )
    model.setup_optimizer_for_test()
    return tracks


def test_v3_tables_present_in_get_model_parameters(real_conf):
    """When V3-L8/L9 toggles are ON, ``get_model_parameters`` emits a
    ``track_optim_state`` sibling key with both tables."""
    conf = _with_dyn_v3_layer(real_conf)
    model = LayeredGaussians(conf, specs=specs_from_config(conf), scene_extent=10.0)
    _seed_dyn_v3_layer(model, n=12)
    params = model.get_model_parameters()
    assert "track_optim_state" in params
    tables = params["track_optim_state"]["tables"]
    # P1.3b: albedo table is [K, 3, k] Fourier coefficients; default
    # n_fourier_albedo_terms=1 → [K, 3, 1] (DC-only, byte-identical behaviour
    # to the old [K, 3] gather since cos(0)=1).
    assert tables["albedo"].shape == (2, 3, 1)
    assert tables["log_scale"].shape == (2, 1)


def test_v3_tables_absent_in_get_model_parameters_when_off(real_conf):
    """OFF-toggle ckpts must NOT include the V3 sibling key (byte-identical
    schema to v2 baseline ckpts)."""
    conf = _with_dyn_layer(real_conf)  # no V3 overrides
    model = LayeredGaussians(conf, specs=specs_from_config(conf), scene_extent=10.0)
    _seed_dyn_layer(model, n=10)  # baseline path (no populate_tracks)
    params = model.get_model_parameters()
    assert "track_optim_state" not in params


def test_v3_tables_roundtrip_via_torch_save_load(real_conf, tmp_path):
    """Full pickled-disk roundtrip for V3-L8/L9 tables."""
    conf = _with_dyn_v3_layer(real_conf)
    model_a = LayeredGaussians(conf, specs=specs_from_config(conf), scene_extent=10.0)
    _seed_dyn_v3_layer(model_a, n=12)
    # Write non-zero values so the assertion catches identity-init false positives.
    with torch.no_grad():
        # P1.3b: albedo table is [K, 3, k]; default k=1 → unsqueeze the DC term.
        model_a._track_albedo_table.copy_(torch.tensor([[0.1, 0.2, 0.3],
                                                        [-0.1, -0.2, -0.3]]).unsqueeze(-1))
        model_a._track_log_scale_table.copy_(torch.tensor([[0.4], [-0.4]]))
    expected_albedo = model_a._track_albedo_table.detach().clone()
    expected_log_scale = model_a._track_log_scale_table.detach().clone()

    ckpt_path = tmp_path / "v3_ckpt.pt"
    torch.save({"model": model_a.get_model_parameters()}, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_b = LayeredGaussians(conf, specs=specs_from_config(conf), scene_extent=10.0)
    _seed_dyn_v3_layer(model_b, n=12)  # registers fresh zero-init tables
    model_b.init_from_checkpoint(ckpt, setup_optimizer=False)

    assert torch.allclose(model_b._track_albedo_table.detach().cpu(),
                          expected_albedo.cpu())
    assert torch.allclose(model_b._track_log_scale_table.detach().cpu(),
                          expected_log_scale.cpu())


def test_v3_ckpt_load_into_off_config_is_safe(real_conf):
    """Loading an ON-trained ckpt into an OFF-config model must not crash
    (the unused sibling key is silently skipped)."""
    conf_on = _with_dyn_v3_layer(real_conf)
    conf_off = _with_dyn_layer(real_conf)
    model_on = LayeredGaussians(conf_on, specs=specs_from_config(conf_on), scene_extent=10.0)
    _seed_dyn_v3_layer(model_on, n=8)
    ckpt = {"model": model_on.get_model_parameters()}

    model_off = LayeredGaussians(conf_off, specs=specs_from_config(conf_off), scene_extent=10.0)
    # No populate_tracks → no V3 tables. Loading shouldn't attach them either.
    model_off.init_layer_from_points("background", torch.randn(2, 3),
                                      setup_optimizer=False)
    model_off.init_layer_from_points(
        "dynamic_rigids", torch.randn(8, 3), setup_optimizer=False,
    )
    model_off.setup_optimizer_for_test()
    model_off.init_from_checkpoint(ckpt, setup_optimizer=False)
    assert not hasattr(model_off, "_track_albedo_table")
    assert not hasattr(model_off, "_track_log_scale_table")
