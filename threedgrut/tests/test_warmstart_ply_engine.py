# SPDX-License-Identifier: Apache-2.0
"""P1.4 asset-harvester warm-start injection engine.

Pins the load → coordinate-align → inject → ckpt-roundtrip path (AH-0/AH-1/AH-2)
entirely on CPU so the engine gate runs on Mac with zero GPU. The 6 demo PLYs
under ``asset-harvester-verify/verify_assets/bundle`` (3 cars + 3 peds, Objaverse
Y-up canonical) serve as fixtures; alignment-math tests build synthetic assets
in-memory so they never depend on the external bundle.
"""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from threedgrut.layers.layered_model import _SH_C0
from threedgrut.layers.warmstart_metadata import (
    AssetSpec,
    load_bundle_metadata,
    resolve_ply_path,
)
from threedgrut.layers.warmstart_ply import (
    _CANONICAL_AXIS_MAP,
    AlignedAsset,
    WarmStartAsset,
    albedo_to_colors,
    apply_alignment,
    asset_extent,
    assets_to_layer_inputs,
    compute_axis_alignment,
    load_warmstart_ply,
    subsample_asset,
)

# External demo bundle (not in-repo). Real-PLY tests skip when absent.
_BUNDLE = Path("/Users/etendue/repo/asset-harvester-verify/verify_assets/bundle")
_HAS_BUNDLE = (_BUNDLE / "metadata.yaml").is_file()
_needs_bundle = pytest.mark.skipif(not _HAS_BUNDLE, reason="demo bundle not present")

_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs")
)


@pytest.fixture(scope="module")
def real_conf():
    """Hydra-composed conf matching test_track_ids_ckpt_roundtrip's fixture."""
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="apps/ncore_3dgut_mcmc")


def _with_dyn_layer(conf):
    c = deepcopy(conf)
    c.layers = {"enabled": ["background", "dynamic_rigids"]}
    return c


# -----------------------------------------------------------------------------
# Group A — bundle metadata parse + PLY path resolve
# -----------------------------------------------------------------------------

@_needs_bundle
def test_metadata_parse_six_assets():
    bundle = load_bundle_metadata(_BUNDLE / "metadata.yaml")
    assert len(bundle) == 6
    spec = bundle["382fee4aea8819ce"]
    assert spec.label_class == "consumer_vehicles"
    assert spec.cuboids_dims == pytest.approx((4.4675, 1.8167, 1.4311), abs=1e-3)
    classes = sorted(s.label_class for s in bundle.values())
    assert classes.count("consumer_vehicles") == 3
    assert classes.count("VRU_pedestrians") == 3


@_needs_bundle
def test_resolve_ply_path_flat_layout():
    bundle = load_bundle_metadata(_BUNDLE / "metadata.yaml")
    spec = bundle["382fee4aea8819ce"]
    p = resolve_ply_path(_BUNDLE, spec)
    assert p.name == "consumer_vehicles__382fee4aea8819ce.ply"
    assert p.is_file()


def test_resolve_ply_path_missing_raises(tmp_path):
    spec = AssetSpec(
        asset_hash="deadbeef", ply_file="x/y/gaussians.ply",
        label_class="consumer_vehicles", cuboids_dims=(1.0, 1.0, 1.0),
    )
    with pytest.raises(FileNotFoundError):
        resolve_ply_path(tmp_path, spec)


# -----------------------------------------------------------------------------
# Group B — PLY load (wrap PLYImporter) + lossless albedo→color conversion
# -----------------------------------------------------------------------------

@_needs_bundle
def test_load_warmstart_ply_shapes():
    bundle = load_bundle_metadata(_BUNDLE / "metadata.yaml")
    spec = bundle["382fee4aea8819ce"]
    asset = load_warmstart_ply(resolve_ply_path(_BUNDLE, spec))
    n = asset.positions.shape[0]
    assert n > 50_000  # demo car ~103k
    assert asset.positions.shape == (n, 3)
    assert asset.rotations.shape == (n, 4)
    assert asset.scales_log.shape == (n, 3)
    assert asset.density_logit.shape == (n, 1)
    assert asset.albedo.shape == (n, 3)
    for t in (asset.positions, asset.rotations, asset.scales_log,
              asset.density_logit, asset.albedo):
        assert t.dtype == torch.float32
        assert torch.isfinite(t).all()


def test_albedo_color_roundtrip_lossless():
    """colors = albedo*SH_C0 + 0.5 inverts init_layer_from_points' albedo
    recovery (colors-0.5)/SH_C0 exactly — so warm-start albedo survives the
    round-trip through the existing injection entrypoint."""
    albedo = torch.randn(64, 3)
    colors = albedo_to_colors(albedo)
    recovered = (colors - 0.5) / _SH_C0
    assert torch.allclose(recovered, albedo, atol=1e-6)


# -----------------------------------------------------------------------------
# Group C — AH-1 canonical→object-local alignment (head-of-risk)
# -----------------------------------------------------------------------------

def _synthetic_asset(halfspans, n=4000, seed=0) -> WarmStartAsset:
    """Random canonical cloud filling box [-h, h] per axis; identity-ish random
    unit quats. pos[0]/pos[1] pinned to ±h so (max-min)/2 == halfspan exactly,
    making the fill invariant tight."""
    g = torch.Generator().manual_seed(seed)
    h = torch.tensor(halfspans, dtype=torch.float32)
    pos = (torch.rand(n, 3, generator=g) * 2 - 1) * h
    pos[0] = h
    pos[1] = -h
    rot = torch.randn(n, 4, generator=g)
    rot = rot / rot.norm(dim=-1, keepdim=True)
    scales_log = torch.randn(n, 3, generator=g) * 0.2 - 3.0
    density = torch.randn(n, 1, generator=g)
    albedo = torch.rand(n, 3, generator=g)
    return WarmStartAsset(pos, rot, scales_log, density, albedo)


def _align(asset, label_class, cuboids_dims):
    half, center = asset_extent(asset)
    xf = compute_axis_alignment(label_class, cuboids_dims, half, center)
    return apply_alignment(asset, xf), xf


@pytest.mark.parametrize("label_class", ["consumer_vehicles", "VRU_pedestrians"])
def test_alignment_R_is_proper_rotation(label_class):
    asset = _synthetic_asset((0.5, 1.0, 0.3))
    half, center = asset_extent(asset)
    xf = compute_axis_alignment(label_class, (4.0, 2.0, 1.5), half, center)
    assert torch.allclose(xf.R @ xf.R.T, torch.eye(3), atol=1e-5)
    assert torch.det(xf.R).item() == pytest.approx(1.0, abs=1e-5)


def test_alignment_containment():
    """Head-of-risk invariant: every aligned point sits inside the metric cuboid
    (same predicate the LiDAR-cuboid filter uses)."""
    dims = (4.0, 2.0, 1.5)
    asset = _synthetic_asset((0.49, 0.20, 0.30), seed=3)  # car-like canonical
    aligned, _ = _align(asset, "consumer_vehicles", dims)
    half_dims = torch.tensor(dims) * 0.5
    assert (aligned.positions.abs() <= half_dims + 1e-4).all()


def test_alignment_fill():
    dims = (4.0, 2.0, 1.5)
    asset = _synthetic_asset((0.49, 0.20, 0.30), seed=3)
    aligned, _ = _align(asset, "consumer_vehicles", dims)
    half_dims = torch.tensor(dims) * 0.5
    for i in range(3):
        assert aligned.positions[:, i].abs().max().item() == pytest.approx(
            half_dims[i].item(), abs=1e-4)


def test_alignment_quaternion_unit_norm():
    asset = _synthetic_asset((0.5, 1.0, 0.3), seed=7)
    aligned, _ = _align(asset, "consumer_vehicles", (4.0, 2.0, 1.5))
    norms = aligned.rotations.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_logscale_metric_shift_exact():
    asset = _synthetic_asset((0.5, 1.0, 0.3), seed=11)
    aligned, xf = _align(asset, "consumer_vehicles", (4.0, 2.0, 1.5))
    expected = asset.scales_log[:, list(xf.perm)] + torch.log(xf.scale_local)
    assert torch.allclose(aligned.scales_log, expected, atol=1e-6)


def test_canonical_axis_map_is_yup():
    for cls in ("consumer_vehicles", "VRU_pedestrians"):
        amap = _CANONICAL_AXIS_MAP[cls]
        assert amap.perm == (0, 2, 1)  # local-Z(up) <- ply-y(up)


def test_align_preserves_density_and_colors():
    asset = _synthetic_asset((0.5, 1.0, 0.3), seed=5)
    aligned, _ = _align(asset, "consumer_vehicles", (4.0, 2.0, 1.5))
    assert torch.equal(aligned.density_logit, asset.density_logit)
    assert torch.allclose(aligned.colors, albedo_to_colors(asset.albedo), atol=1e-6)


@_needs_bundle
@pytest.mark.parametrize("asset_hash", ["382fee4aea8819ce", "0d7b602f2da8c364"])
def test_real_bundle_alignment_containment(asset_hash):
    bundle = load_bundle_metadata(_BUNDLE / "metadata.yaml")
    spec = bundle[asset_hash]
    asset = load_warmstart_ply(resolve_ply_path(_BUNDLE, spec))
    aligned, _ = _align(asset, spec.label_class, spec.cuboids_dims)
    half_dims = torch.tensor(spec.cuboids_dims) * 0.5
    assert (aligned.positions.abs() <= half_dims + 1e-3).all()


@_needs_bundle
def test_real_bundle_up_axis_per_class():
    """car: local-X (length) is the largest extent; ped: local-Z (height)."""
    bundle = load_bundle_metadata(_BUNDLE / "metadata.yaml")
    car = bundle["382fee4aea8819ce"]
    ped = bundle["0d7b602f2da8c364"]
    car_al, _ = _align(load_warmstart_ply(resolve_ply_path(_BUNDLE, car)),
                       car.label_class, car.cuboids_dims)
    ped_al, _ = _align(load_warmstart_ply(resolve_ply_path(_BUNDLE, ped)),
                       ped.label_class, ped.cuboids_dims)
    car_ext = car_al.positions.abs().amax(0)
    ped_ext = ped_al.positions.abs().amax(0)
    assert car_ext.argmax().item() == 0  # local-X largest for vehicle
    assert ped_ext.argmax().item() == 2  # local-Z largest for pedestrian


# -----------------------------------------------------------------------------
# Group D — subsample + assets→layer-input concat
# -----------------------------------------------------------------------------

def _aligned(n, fill=0.1, seed=0) -> AlignedAsset:
    g = torch.Generator().manual_seed(seed)
    return AlignedAsset(
        positions=torch.rand(n, 3, generator=g) * fill,
        rotations=torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 4).clone(),
        scales_log=torch.full((n, 3), -3.0),
        density_logit=torch.zeros(n, 1),
        colors=torch.rand(n, 3, generator=g),
    )


def test_subsample_respects_budget_deterministic():
    asset = _aligned(100, seed=1)
    g1 = torch.Generator().manual_seed(42)
    out1 = subsample_asset(asset, 30, generator=g1)
    g2 = torch.Generator().manual_seed(42)
    out2 = subsample_asset(asset, 30, generator=g2)
    assert out1.positions.shape == (30, 3)
    assert torch.equal(out1.positions, out2.positions)  # same seed → same pick


def test_subsample_noop_when_under_budget():
    asset = _aligned(20, seed=2)
    out = subsample_asset(asset, 50)
    assert out.positions.shape == (20, 3)
    assert torch.equal(out.positions, asset.positions)


def test_assets_to_layer_inputs_concat_and_track_ids():
    a0 = _aligned(5, seed=3)
    a1 = _aligned(8, seed=4)
    kw = assets_to_layer_inputs([(0, a0), (2, a1)])
    assert kw["positions"].shape == (13, 3)
    assert kw["colors"].shape == (13, 3)
    assert kw["rotations"].shape == (13, 4)
    assert kw["scales"].shape == (13, 3)
    assert kw["densities"].shape == (13, 1)
    assert kw["track_ids"].dtype == torch.int64
    assert kw["track_ids"].tolist() == [0] * 5 + [2] * 8


# -----------------------------------------------------------------------------
# Group E — merge warm-start with LiDAR (replace / augment / budget)
# -----------------------------------------------------------------------------

_SCALE_PRIOR = (0.05, 0.05, 0.05)
_DENSITY_INIT = -2.0


def _lidar(counts: dict[int, int]):
    """Build LiDAR positions+track_ids with `counts` points per track id."""
    pos, ids = [], []
    for tid, k in counts.items():
        pos.append(torch.full((k, 3), float(tid)))
        ids.append(torch.full((k,), tid, dtype=torch.int64))
    return torch.cat(pos), torch.cat(ids)


def _warm_kwargs(tid: int, n: int, seed=0):
    return assets_to_layer_inputs([(tid, _aligned(n, seed=seed))])


def test_merge_replace_drops_lidar_for_mapped_tracks():
    from threedgrut.layers.dynamic_rigid_init import merge_warmstart_with_lidar
    lpos, lids = _lidar({0: 3, 1: 4})
    warm = _warm_kwargs(0, 5, seed=9)
    out = merge_warmstart_with_lidar(
        lpos, lids, warm, max_pts_per_track=100,
        scale_prior=_SCALE_PRIOR, density_init=_DENSITY_INIT, mode="replace",
    )
    ids = out["track_ids"]
    assert (ids == 0).sum().item() == 5   # track 0: warm only (LiDAR dropped)
    assert (ids == 1).sum().item() == 4   # track 1: LiDAR kept (no asset)
    # track-0 colors are the warm colors, not the neutral-0.5 LiDAR default
    assert torch.allclose(out["colors"][ids == 0], warm["colors"], atol=1e-6)


def test_merge_augment_concats_lidar_and_warm():
    from threedgrut.layers.dynamic_rigid_init import merge_warmstart_with_lidar
    lpos, lids = _lidar({0: 3, 1: 4})
    warm = _warm_kwargs(0, 5, seed=9)
    out = merge_warmstart_with_lidar(
        lpos, lids, warm, max_pts_per_track=100,
        scale_prior=_SCALE_PRIOR, density_init=_DENSITY_INIT, mode="augment",
    )
    ids = out["track_ids"]
    assert (ids == 0).sum().item() == 8   # 3 LiDAR + 5 warm
    assert (ids == 1).sum().item() == 4


def test_merge_augment_respects_per_track_budget():
    from threedgrut.layers.dynamic_rigid_init import merge_warmstart_with_lidar
    lpos, lids = _lidar({0: 3, 1: 4})
    warm = _warm_kwargs(0, 5, seed=9)
    g = torch.Generator().manual_seed(0)
    out = merge_warmstart_with_lidar(
        lpos, lids, warm, max_pts_per_track=6,
        scale_prior=_SCALE_PRIOR, density_init=_DENSITY_INIT, mode="augment",
        generator=g,
    )
    ids = out["track_ids"]
    assert (ids == 0).sum().item() == 6   # 8 capped to 6
    assert (ids == 1).sum().item() == 4


def test_merge_invalid_mode_raises():
    from threedgrut.layers.dynamic_rigid_init import merge_warmstart_with_lidar
    lpos, lids = _lidar({0: 2})
    warm = _warm_kwargs(0, 2)
    with pytest.raises(ValueError):
        merge_warmstart_with_lidar(
            lpos, lids, warm, max_pts_per_track=10,
            scale_prior=_SCALE_PRIOR, density_init=_DENSITY_INIT, mode="bogus",
        )


# -----------------------------------------------------------------------------
# Group F — inject into LayeredGaussians + ckpt roundtrip (AH-0 head gate)
# -----------------------------------------------------------------------------

def _build_model(conf):
    from threedgrut.layers.layered_model import LayeredGaussians
    from threedgrut.layers.registry import specs_from_config
    return LayeredGaussians(conf, specs=specs_from_config(conf), scene_extent=10.0)


def _inject(model, aligned_list):
    """Seed bg + inject warm-start kwargs into dynamic_rigids (CPU, no optimizer)."""
    model.init_layer_from_points("background", torch.randn(4, 3) * 0.1,
                                 setup_optimizer=False)
    kw = assets_to_layer_inputs(aligned_list)
    pos = kw["positions"]
    model.init_layer_from_points(
        "dynamic_rigids", pos,
        colors=kw["colors"], rotations=kw["rotations"], scales=kw["scales"],
        densities=kw["densities"], track_ids=kw["track_ids"],
        setup_optimizer=False,
    )
    model.setup_optimizer_for_test()
    return kw


def test_track_ids_correct_after_inject(real_conf):
    model = _build_model(_with_dyn_layer(real_conf))
    kw = _inject(model, [(0, _aligned(5, seed=1)), (3, _aligned(7, seed=2))])
    tids = model.layers["dynamic_rigids"].track_ids
    assert tids.dtype == torch.int64
    assert tids.tolist() == [0] * 5 + [3] * 7


def test_inject_warmstart_roundtrip(real_conf, tmp_path):
    """Full pickled-disk roundtrip: aligned warm-start particles survive
    get_model_parameters → torch.save/load → init_from_checkpoint byte-stable."""
    model_a = _build_model(_with_dyn_layer(real_conf))
    kw = _inject(model_a, [(0, _aligned(6, seed=4)), (1, _aligned(9, seed=5))])
    dyn_a = model_a.layers["dynamic_rigids"]
    exp_pos = dyn_a.positions.detach().clone()
    exp_scale = dyn_a.scale.detach().clone()
    exp_ids = dyn_a.track_ids.detach().clone()

    ckpt_path = tmp_path / "warm_ckpt.pt"
    torch.save({"model": model_a.get_model_parameters()}, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_b = _build_model(_with_dyn_layer(real_conf))
    model_b.init_from_checkpoint(ckpt, setup_optimizer=False)
    dyn_b = model_b.layers["dynamic_rigids"]
    assert torch.equal(dyn_b.positions.detach().cpu(), exp_pos)
    assert torch.equal(dyn_b.scale.detach().cpu(), exp_scale)
    assert torch.equal(dyn_b.track_ids.detach().cpu(), exp_ids)


# -----------------------------------------------------------------------------
# Group G — warm-start config keys route into LayerSpec.extra
# -----------------------------------------------------------------------------

def test_warmstart_keys_route_to_extra(real_conf):
    from threedgrut.layers.registry import specs_from_config
    c = deepcopy(real_conf)
    c.layers = {
        "enabled": ["background", "dynamic_rigids"],
        "overrides": {"dynamic_rigids": {
            "warmstart_ply_bundle": "/tmp/bundle/metadata.yaml",
            "warmstart_ply_mapping": {"trackA": "382fee4aea8819ce"},
            "warmstart_mode": "replace",
            "warmstart_max_pts_per_track": 5000,
            "warmstart_seed": 7,
        }},
    }
    dyn = next(s for s in specs_from_config(c) if s.name == "dynamic_rigids")
    assert dyn.extra["warmstart_ply_bundle"] == "/tmp/bundle/metadata.yaml"
    assert dyn.extra["warmstart_mode"] == "replace"
    assert dyn.extra["warmstart_max_pts_per_track"] == 5000
    assert dyn.extra["warmstart_seed"] == 7
    assert dict(dyn.extra["warmstart_ply_mapping"]) == {"trackA": "382fee4aea8819ce"}


# -----------------------------------------------------------------------------
# Group H — asset↔track mapping + end-to-end orchestration (trainer seam logic)
# -----------------------------------------------------------------------------

def _fake_bundle():
    return {
        "382fee4aea8819ce": AssetSpec(
            asset_hash="382fee4aea8819ce", ply_file="x.ply",
            label_class="consumer_vehicles", cuboids_dims=(4.4, 1.8, 1.4)),
    }


def test_map_assets_to_tracks_explicit():
    from threedgrut.layers.warmstart_metadata import map_assets_to_tracks
    bundle = _fake_bundle()
    tracks = {"trk7": {"size": [4.4, 1.8, 1.4]}}
    m = map_assets_to_tracks(bundle, tracks, {"trk7": "382fee4aea8819ce"})
    assert m["trk7"].label_class == "consumer_vehicles"


def test_map_assets_to_tracks_none_raises():
    from threedgrut.layers.warmstart_metadata import map_assets_to_tracks
    with pytest.raises(ValueError):
        map_assets_to_tracks(_fake_bundle(), {"trk7": {}}, None)


def test_map_assets_to_tracks_unknown_track_raises():
    from threedgrut.layers.warmstart_metadata import map_assets_to_tracks
    with pytest.raises(KeyError):
        map_assets_to_tracks(_fake_bundle(), {"trk7": {}},
                             {"ghost": "382fee4aea8819ce"})


def test_map_assets_to_tracks_unknown_asset_raises():
    from threedgrut.layers.warmstart_metadata import map_assets_to_tracks
    with pytest.raises(KeyError):
        map_assets_to_tracks(_fake_bundle(), {"trk7": {}},
                             {"trk7": "deadbeef"})


def test_map_assets_to_tracks_strips_at_suffix():
    """3dgrut track keys keep the raw '<id>@scene:...' suffix while harvested
    asset ids are cleaned ('24'); mapping by cleaned id must resolve to the raw
    track key (so downstream name_to_id indexing matches)."""
    from threedgrut.layers.warmstart_metadata import map_assets_to_tracks
    bundle = _fake_bundle()
    raw = "24@scene:obstacles:autolabels:v2"
    tracks = {raw: {"size": [4.4, 1.8, 1.4]}}
    m = map_assets_to_tracks(bundle, tracks, {"24": "382fee4aea8819ce"})
    assert raw in m
    assert m[raw].asset_hash == "382fee4aea8819ce"


def test_map_assets_to_tracks_json_path(tmp_path):
    import json
    from threedgrut.layers.warmstart_metadata import map_assets_to_tracks
    p = tmp_path / "map.json"
    p.write_text(json.dumps({"trk7": "382fee4aea8819ce"}))
    m = map_assets_to_tracks(_fake_bundle(), {"trk7": {}}, str(p))
    assert m["trk7"].asset_hash == "382fee4aea8819ce"


@_needs_bundle
def test_build_warmstart_layer_inputs_replace_end_to_end():
    from threedgrut.layers.warmstart_inject import build_warmstart_layer_inputs
    tracks = {"t0": {"size": [4.4, 1.8, 1.4]}, "t1": {"size": [4.2, 1.9, 1.6]}}
    track_names = sorted(tracks)            # ["t0","t1"] → ids 0,1
    lpos, lids = _lidar({0: 10, 1: 12})
    merged = build_warmstart_layer_inputs(
        bundle_path=_BUNDLE / "metadata.yaml",
        mapping={"t0": "382fee4aea8819ce"},
        tracks=tracks, track_names=track_names,
        lidar_positions=lpos, lidar_track_ids=lids,
        scale_prior=_SCALE_PRIOR, density_init=_DENSITY_INIT,
        mode="replace", max_pts_per_track=2000, seed=0,
    )
    ids = merged["track_ids"]
    assert (ids == 0).sum().item() == 2000   # car warm-start, capped (LiDAR dropped)
    assert (ids == 1).sum().item() == 12     # track 1 keeps LiDAR (no asset)
    # warm-start track-0 particles sit inside the LIVE track-size cuboid
    half = torch.tensor([4.4, 1.8, 1.4]) / 2
    assert (merged["positions"][ids == 0].abs() <= half + 1e-2).all()


def test_build_warmstart_layer_inputs_no_mapping_returns_none():
    from threedgrut.layers.warmstart_inject import build_warmstart_layer_inputs
    lpos, lids = _lidar({0: 3})
    out = build_warmstart_layer_inputs(
        bundle_path="/nonexistent/metadata.yaml",
        mapping={},                          # empty → nothing to inject
        tracks={"t0": {}}, track_names=["t0"],
        lidar_positions=lpos, lidar_track_ids=lids,
        scale_prior=_SCALE_PRIOR, density_init=_DENSITY_INIT,
    )
    assert out is None                       # caller falls back to LiDAR-only path
