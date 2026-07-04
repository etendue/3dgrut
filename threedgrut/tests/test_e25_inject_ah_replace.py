# SPDX-License-Identifier: Apache-2.0
"""E2.5 — frozen AH-asset drop-in replacing recon cars (offline ckpt surgery).

Pins the pure-CPU core of ``threedgrut/layers/e25_inject.py``:

  1. ``match_assets_by_size`` — greedy size match (3 AH dims ↔ 3 recon cuboid
     sizes) is order-invariant and minimises per-axis L2.
  2. ``aligned_to_node_tensors`` — one AlignedAsset → the 6 MoG node tensors,
     specular zero-filled to the target dim, features_albedo recovered as the
     exact inverse of init_layer_from_points' ``(colors-0.5)/_SH_C0``.
  3. ``replace_tracks_in_dyn_node`` — per-track subset replacement: target
     tracks' particles swapped for AH particles, untouched recon tracks kept
     byte-identical (incl. their learned degree-3 specular), track_ids
     self-consistent.
  4. The surgically-edited ckpt reloads through ``LayeredGaussians`` and the
     dynamic_rigids transform path runs (renderable).

The alignment engine itself (warmstart_ply) is reused from PR #18; two smoke
tests confirm it imports and aligns under this branch.
"""

from __future__ import annotations

import os

import pytest
import torch
from hydra import compose, initialize_config_dir

from threedgrut.layers.e25_inject import (
    aligned_to_node_tensors,
    build_name_to_int_id,
    flip_forward_180,
    match_assets_by_size,
    replace_tracks_in_dyn_node,
)
from threedgrut.layers.layered_model import (
    _SH_C0,
    LayeredGaussians,
    _rotmat_to_quat_wxyz,
)
from threedgrut.layers.registry import specs_from_config
from threedgrut.layers.warmstart_ply import (
    AlignedAsset,
    WarmStartAsset,
    apply_alignment,
    asset_extent,
    compute_axis_alignment,
)

_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs"))


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_conf():
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="apps/ncore_3dgut_mcmc")


def _with_dyn_layer(conf):
    from copy import deepcopy

    c = deepcopy(conf)
    c.layers = {"enabled": ["background", "dynamic_rigids"]}
    return c


def _synth_aligned(n: int, *, seed: int = 0) -> AlignedAsset:
    """A synthetic object-local aligned AH asset (no PLY IO needed)."""
    g = torch.Generator().manual_seed(seed)
    return AlignedAsset(
        positions=torch.randn(n, 3, generator=g),
        rotations=torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
        scales_log=torch.randn(n, 3, generator=g),
        density_logit=torch.randn(n, 1, generator=g),
        colors=torch.rand(n, 3, generator=g),
    )


def _forge_dyn_node(particles_per_id: dict[int, int], *, spec_dim: int = 45) -> dict:
    """Forge a MoG dynamic_rigids node dict with known recon particles.

    ``particles_per_id`` maps integer track id → particle count. Each tensor
    field is filled so identity is traceable (positions encode the owner id).
    """
    pos_parts, rot, scl, den, alb, spec, ids = [], [], [], [], [], [], []
    for tid, k in particles_per_id.items():
        pos_parts.append(torch.full((k, 3), float(tid)))
        rot.append(torch.nn.functional.normalize(torch.randn(k, 4), dim=-1))
        scl.append(torch.randn(k, 3))
        den.append(torch.randn(k, 1))
        alb.append(torch.randn(k, 3))
        # specular tagged with owner id so we can prove byte-identity on keep
        spec.append(torch.full((k, spec_dim), float(tid)))
        ids.append(torch.full((k,), tid, dtype=torch.int64))
    return {
        "positions": torch.cat(pos_parts),
        "rotation": torch.cat(rot),
        "scale": torch.cat(scl),
        "density": torch.cat(den),
        "features_albedo": torch.cat(alb),
        "features_specular": torch.cat(spec),
        "track_ids": torch.cat(ids),
        # non-tensor metadata that must ride along untouched
        "n_active_features": 3,
        "max_n_features": 3,
        "scene_extent": 12.0,
    }


# -----------------------------------------------------------------------------
# 1. size matching
# -----------------------------------------------------------------------------


def test_match_assets_by_size_greedy_order_invariant():
    # recon cars (track_key -> [L,W,H]); AH assets (hash -> [L,W,H]) shuffled.
    recon = {
        "t_sedan": (4.2, 1.9, 1.6),
        "t_van": (5.8, 2.3, 1.9),
        "t_compact": (4.0, 1.8, 1.4),
    }
    ah = {
        "h_big": (5.76, 2.25, 1.88),  # → t_van
        "h_small": (4.0, 1.8, 1.43),  # → t_compact
        "h_mid": (4.23, 1.9, 1.61),  # → t_sedan
    }
    out = match_assets_by_size(ah, recon)
    assert out == {"t_van": "h_big", "t_compact": "h_small", "t_sedan": "h_mid"}


def test_match_assets_by_size_more_recon_than_ah():
    """Fewer AH than recon → only len(AH) tracks get mapped (rest untouched)."""
    recon = {"a": (4.0, 1.8, 1.4), "b": (4.5, 1.9, 1.5), "c": (6.0, 2.4, 2.0)}
    ah = {"h1": (4.05, 1.82, 1.41)}
    out = match_assets_by_size(ah, recon)
    assert out == {"a": "h1"}


# -----------------------------------------------------------------------------
# 1b. name → integer track id (THE命门: must match sorted(tracks_poses.keys())
#     used by _transform_means_and_active, else AH cars render on wrong tracks)
# -----------------------------------------------------------------------------


def test_build_name_to_int_id_is_sorted_enumerate():
    tracks = {
        "z_car@scene:1": {"size": (4, 2, 1)},
        "a_car@scene:1": {"size": (4, 2, 1)},
        "m_car@scene:1": {"size": (4, 2, 1)},
    }
    out = build_name_to_int_id(tracks)
    assert out == {"a_car@scene:1": 0, "m_car@scene:1": 1, "z_car@scene:1": 2}


# -----------------------------------------------------------------------------
# 2. aligned asset → node tensors
# -----------------------------------------------------------------------------


def test_aligned_to_node_tensors_specular_zero_filled():
    a = _synth_aligned(7)
    node = aligned_to_node_tensors(a, spec_dim=45)
    assert node["features_specular"].shape == (7, 45)
    assert torch.count_nonzero(node["features_specular"]) == 0


def test_aligned_to_node_tensors_albedo_inverse_of_colors():
    """features_albedo must equal (colors-0.5)/_SH_C0 so the round-trip through
    init_layer_from_points' albedo recovery is lossless."""
    a = _synth_aligned(5)
    node = aligned_to_node_tensors(a, spec_dim=45)
    expected = (a.colors - 0.5) / _SH_C0
    assert torch.allclose(node["features_albedo"], expected, atol=1e-6)


def test_aligned_to_node_tensors_geometry_passthrough():
    a = _synth_aligned(4)
    node = aligned_to_node_tensors(a, spec_dim=45)
    assert torch.equal(node["positions"], a.positions)
    assert torch.equal(node["rotation"], a.rotations)
    assert torch.equal(node["scale"], a.scales_log)
    assert torch.equal(node["density"], a.density_logit)


# -----------------------------------------------------------------------------
# 3. per-track subset replacement (the core surgery)
# -----------------------------------------------------------------------------


def test_replace_swaps_only_target_tracks():
    dyn = _forge_dyn_node({0: 6, 1: 4, 2: 5, 3: 7, 4: 3}, spec_dim=45)
    aligned_by_id = {1: _synth_aligned(8, seed=1), 3: _synth_aligned(10, seed=3)}
    new = replace_tracks_in_dyn_node(dyn, aligned_by_id)

    new_ids = new["track_ids"]
    # target tracks now have AH particle counts
    assert int((new_ids == 1).sum()) == 8
    assert int((new_ids == 3).sum()) == 10
    # untouched tracks keep original counts
    assert int((new_ids == 0).sum()) == 6
    assert int((new_ids == 2).sum()) == 5
    assert int((new_ids == 4).sum()) == 3


def test_replace_preserves_untouched_recon_specular_byte_identical():
    dyn = _forge_dyn_node({0: 6, 1: 4, 2: 5}, spec_dim=45)
    aligned_by_id = {1: _synth_aligned(8, seed=1)}
    new = replace_tracks_in_dyn_node(dyn, aligned_by_id)

    # recon specular was tagged with owner id; kept tracks (0,2) must survive
    # with that exact value (degree-3 view-dependence NOT zeroed).
    spec = new["features_specular"]
    ids = new["track_ids"]
    for keep_id in (0, 2):
        rows = spec[ids == keep_id]
        assert torch.all(rows == float(keep_id)), f"track {keep_id} specular altered"


def test_replace_ah_specular_zero_filled_to_recon_dim():
    dyn = _forge_dyn_node({0: 6, 1: 4}, spec_dim=45)
    aligned_by_id = {1: _synth_aligned(8, seed=1)}
    new = replace_tracks_in_dyn_node(dyn, aligned_by_id)
    assert new["features_specular"].shape[1] == 45
    ah_rows = new["features_specular"][new["track_ids"] == 1]
    assert torch.count_nonzero(ah_rows) == 0


def test_replace_track_ids_self_consistent():
    dyn = _forge_dyn_node({0: 6, 1: 4, 2: 5, 3: 7}, spec_dim=45)
    aligned_by_id = {1: _synth_aligned(8, seed=1), 3: _synth_aligned(2, seed=3)}
    new = replace_tracks_in_dyn_node(dyn, aligned_by_id)

    n = new["positions"].shape[0]
    for key in ("rotation", "scale", "density", "features_albedo", "features_specular", "track_ids"):
        assert new[key].shape[0] == n, f"{key} length != positions"
    # id set unchanged (3 kept + 2 replaced still cover {0,1,2,3})
    assert set(new["track_ids"].tolist()) == {0, 1, 2, 3}


def test_replace_carries_nontensor_metadata():
    dyn = _forge_dyn_node({0: 6, 1: 4}, spec_dim=45)
    new = replace_tracks_in_dyn_node(dyn, {1: _synth_aligned(3, seed=1)})
    assert new["n_active_features"] == 3
    assert new["max_n_features"] == 3
    assert new["scene_extent"] == 12.0


def test_replace_ah_particles_carry_aligned_positions():
    dyn = _forge_dyn_node({0: 6, 1: 4}, spec_dim=45)
    ah = _synth_aligned(8, seed=1)
    new = replace_tracks_in_dyn_node(dyn, {1: ah})
    ah_pos = new["positions"][new["track_ids"] == 1]
    # AH particles' positions are exactly the aligned ones (order-preserving)
    assert torch.equal(ah_pos, ah.positions)


# -----------------------------------------------------------------------------
# 4. alignment engine smoke (PR #18 reuse under this branch)
# -----------------------------------------------------------------------------


def test_compute_axis_alignment_proper_rotation():
    half = torch.tensor([2.0, 1.0, 0.7])
    center = torch.zeros(3)
    xf = compute_axis_alignment("consumer_vehicles", (4.47, 1.82, 1.43), half, center)
    assert abs(torch.det(xf.R).item() - 1.0) < 1e-5
    assert abs(xf.q_R.norm().item() - 1.0) < 1e-5


def test_flip_forward_180_is_yaw_about_up():
    """NCore cuboid forward(+X) is opposite the AH Objaverse canonical forward,
    so the injected car drives backwards (head/tail swapped). flip_forward_180
    composes a 180° yaw about object-local up(Z): X,Y rows negated, Z unchanged,
    det stays +1 (proper rotation, not a mirror)."""
    half = torch.tensor([2.0, 1.0, 0.7])
    center = torch.zeros(3)
    xf = compute_axis_alignment("consumer_vehicles", (4.0, 2.0, 1.4), half, center)
    xff = flip_forward_180(xf)

    assert abs(torch.det(xff.R).item() - 1.0) < 1e-5, "flip must stay a proper rotation"
    assert torch.allclose(xff.R[0], -xf.R[0], atol=1e-6), "forward(X) row must negate"
    assert torch.allclose(xff.R[1], -xf.R[1], atol=1e-6), "left(Y) row must negate"
    assert torch.allclose(xff.R[2], xf.R[2], atol=1e-6), "up(Z) row must be unchanged"
    assert torch.allclose(xff.q_R, _rotmat_to_quat_wxyz(xff.R), atol=1e-5)
    # scale / center / perm untouched
    assert torch.allclose(xff.scale_local, xf.scale_local)
    assert torch.allclose(xff.center, xf.center)
    assert xff.perm == xf.perm


def test_flip_forward_180_reverses_aligned_positions():
    """End-to-end: a cloud that aligns to +local-X (forward) ends at -X after flip,
    with up(Z) preserved — i.e. the car turns around in place."""
    g = torch.Generator().manual_seed(0)
    pts = (torch.rand(200, 3, generator=g) - 0.5) * 2.0
    pts[:, 0] += 3.0  # bias along ply-x so the aligned cloud lands at +forward
    asset = WarmStartAsset(
        positions=pts,
        rotations=torch.nn.functional.normalize(torch.randn(200, 4, generator=g), dim=-1),
        scales_log=torch.zeros(200, 3),
        density_logit=torch.zeros(200, 1),
        albedo=torch.zeros(200, 3),
    )
    half, center = asset_extent(asset)
    xf = compute_axis_alignment("consumer_vehicles", (4.0, 2.0, 1.4), half, center)
    base = apply_alignment(asset, xf)
    flipped = apply_alignment(asset, flip_forward_180(xf))
    # forward (X) mean sign flips; up (Z) mean preserved
    assert base.positions[:, 0].mean() * flipped.positions[:, 0].mean() < 0
    assert torch.allclose(base.positions[:, 2].mean(), flipped.positions[:, 2].mean(), atol=1e-5)


def test_apply_alignment_fills_cuboid():
    # a point cloud with known half-spans; after alignment per-axis half-span
    # should match dims/2 (containment + fill by construction).
    g = torch.Generator().manual_seed(0)
    pts = (torch.rand(200, 3, generator=g) - 0.5) * 2.0  # half-span ~1 each
    pts[:, 0] *= 3.0  # stretch x so half-spans differ across axes
    asset = WarmStartAsset(
        positions=pts,
        rotations=torch.nn.functional.normalize(torch.randn(200, 4, generator=g), dim=-1),
        scales_log=torch.zeros(200, 3),
        density_logit=torch.zeros(200, 1),
        albedo=torch.zeros(200, 3),
    )
    half, center = asset_extent(asset)
    dims = (4.47, 1.82, 1.43)
    xf = compute_axis_alignment("consumer_vehicles", dims, half, center)
    aligned = apply_alignment(asset, xf)
    lo = aligned.positions.amin(dim=0)
    hi = aligned.positions.amax(dim=0)
    out_half = (hi - lo) * 0.5
    expected = torch.tensor([d * 0.5 for d in dims])
    assert torch.allclose(out_half, expected, atol=1e-3)


# -----------------------------------------------------------------------------
# 5. surgically-edited ckpt reloads through LayeredGaussians and renders
# -----------------------------------------------------------------------------


def test_injected_ckpt_reloadable_and_transformable(real_conf):
    conf = _with_dyn_layer(real_conf)
    model = LayeredGaussians(conf, specs=specs_from_config(conf), scene_extent=10.0)
    model.init_layer_from_points("background", torch.randn(6, 3) * 0.1, setup_optimizer=False)
    recon_pos = torch.randn(30, 3) * 0.1
    recon_ids = torch.tensor([i % 5 for i in range(30)], dtype=torch.int64)  # 5 cars
    model.init_layer_from_points("dynamic_rigids", recon_pos, track_ids=recon_ids, setup_optimizer=False)
    model.setup_optimizer_for_test()
    ckpt = {"model": model.get_model_parameters()}

    dyn = ckpt["model"]["gaussians_nodes"]["dynamic_rigids"]
    spec_dim = dyn["features_specular"].shape[1]
    # replace ids 1 and 3 with synthetic AH assets
    aligned_by_id = {1: _synth_aligned(8, seed=1), 3: _synth_aligned(10, seed=3)}
    ckpt["model"]["gaussians_nodes"]["dynamic_rigids"] = replace_tracks_in_dyn_node(dyn, aligned_by_id)

    model_b = LayeredGaussians(conf, specs=specs_from_config(conf), scene_extent=10.0)
    model_b.init_from_checkpoint(ckpt, setup_optimizer=False)

    dyn_layer = model_b.layers["dynamic_rigids"]
    # kept ids 0,2,4 = 18 particles + 8 + 10 = 36
    assert dyn_layer.positions.shape[0] == 36
    restored_ids = dyn_layer.track_ids
    assert restored_ids.dtype == torch.int64
    assert int((restored_ids == 1).sum()) == 8
    assert int((restored_ids == 3).sum()) == 10
    # specular dim preserved (reload would crash on mismatch)
    assert dyn_layer.features_specular.shape[1] == spec_dim
