# SPDX-License-Identifier: Apache-2.0
"""T1.1 contract tests for LayeredGaussians container.

These tests rely on the real Hydra config (`apps/ncore_3dgut_mcmc`) because
`MixtureOfGaussians.__init__` constructs a real Tracer that reads many config
keys. The Tracer's CUDA extension is JIT-compiled once and cached.

To skip optimizer setup in init_from_checkpoint (which would need the full
optimizer/scheduler conf trees), tests pass `setup_optimizer=False`.

The 6 per-particle Parameters use `max_n_features=3` from the real config,
so `features_specular` has dim `sh_degree_to_specular_dim(3) = 45`.
"""

import os

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.utils.misc import sh_degree_to_specular_dim


# ----------------------------------------------------------------------- conf
_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs")
)


@pytest.fixture(scope="module")
def real_conf():
    """Hydra-composed full conf from apps/ncore_3dgut_mcmc.

    Module-scoped so Tracer CUDA compile happens at most once per test session.
    """
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="apps/ncore_3dgut_mcmc")


def _v1_shape_dict(N: int, conf) -> dict:
    """Build a v1-shape flat ckpt sub-dict for one layer.

    Matches `MixtureOfGaussians.get_model_parameters()` output (minus optimizer
    state) so `init_from_checkpoint(setup_optimizer=False)` accepts it.
    """
    sh_deg = conf.model.progressive_training.max_n_features
    specular_dim = sh_degree_to_specular_dim(sh_deg)
    return {
        "positions":             torch.nn.Parameter(torch.randn(N, 3)),
        "rotation":              torch.nn.Parameter(torch.randn(N, 4)),
        "scale":                 torch.nn.Parameter(torch.randn(N, 3)),
        "density":               torch.nn.Parameter(torch.randn(N, 1)),
        "features_albedo":       torch.nn.Parameter(torch.randn(N, 3)),
        "features_specular":     torch.nn.Parameter(torch.randn(N, specular_dim)),
        "background":            {},
        "n_active_features":     0,
        "max_n_features":        sh_deg,
        "scene_extent":          10.0,
        # progressive_training=True when n_active < max_n_features, so these are required
        "feature_dim_increase_interval": conf.model.progressive_training.increase_frequency,
        "feature_dim_increase_step":     conf.model.progressive_training.increase_step,
    }


# ----------------------------------------------------------------------- tests
def test_layered_gaussians_init_with_single_background_layer(real_conf):
    """空构造 → 仅 background 层 → 行为等价 v1 单 model"""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1_000_000)]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    assert "background" in model.layers
    assert model.layers["background"].num_gaussians == 0
    assert len(model.layers) == 1


def test_layered_gaussians_init_with_multiple_layers(real_conf):
    """多层 spec → 每层各一个 MixtureOfGaussians"""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="road",       layer_id=1, max_n_particles=200_000),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    assert set(model.layers.keys()) == {"background", "road"}


def test_layered_gaussians_get_model_parameters_nested(real_conf):
    """get_model_parameters 返回 NRE schema: gaussians_nodes.<name>"""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1_000_000)]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    model.setup_optimizer_for_test()

    params = model.get_model_parameters()
    assert "gaussians_nodes" in params
    assert "background" in params["gaussians_nodes"]
    assert "positions" in params["gaussians_nodes"]["background"]
    assert "scene_extent" in params


def test_load_v1_checkpoint_routes_to_background(real_conf):
    """v1 ckpt (无 'gaussians_nodes' 嵌套) → 全部塞 background layer"""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1_000_000)]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)

    v1_ckpt = _v1_shape_dict(N=100, conf=real_conf)
    model.init_from_checkpoint(v1_ckpt, setup_optimizer=False)

    assert model.layers["background"].positions.shape == (100, 3)


def test_load_v2_checkpoint_dispatches_per_layer(real_conf):
    """v2 ckpt (NRE schema: 'model.gaussians_nodes.<name>') → 分发到对应 layer"""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=1_000_000),
        LayerSpec(name="road",       layer_id=1, max_n_particles=200_000),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    v2_ckpt = {
        "global_step": 30000,
        "model": {
            "gaussians_nodes": {
                "background": _v1_shape_dict(N=600, conf=real_conf),
                "road":       _v1_shape_dict(N=200, conf=real_conf),
            },
        },
    }
    model.init_from_checkpoint(v2_ckpt, setup_optimizer=False)
    assert model.layers["background"].positions.shape == (600, 3)
    assert model.layers["road"].positions.shape == (200, 3)


def test_load_v2_checkpoint_warns_on_missing_layer(real_conf):
    """ckpt 缺某层 → warn + skip，不 raise"""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=1_000_000),
        LayerSpec(name="road",       layer_id=1, max_n_particles=200_000),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    v2_ckpt = {
        "global_step": 30000,
        "model": {
            "gaussians_nodes": {
                "background": _v1_shape_dict(N=600, conf=real_conf),
                # road missing
            },
        },
    }
    model.init_from_checkpoint(v2_ckpt, setup_optimizer=False)
    assert model.layers["background"].positions.shape == (600, 3)
    assert model.layers["road"].positions.shape == (0, 3)


# ----------------------------------------------------------- T1.3 / T1.4 additions
def test_v1_ckpt_resume_without_background_layer_raises(real_conf):
    """T1.3: layers.enabled=['road'] + v1 ckpt → 明确报错指向 layers.enabled"""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="road", layer_id=1, max_n_particles=200_000)]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)

    v1_ckpt = _v1_shape_dict(N=100, conf=real_conf)
    with pytest.raises(ValueError, match="layers.enabled"):
        model.init_from_checkpoint(v1_ckpt, setup_optimizer=False)


def test_v1_ckpt_resume_with_background_layer_works(real_conf):
    """T1.3: layers.enabled=['background','road'] + v1 ckpt → 全部塞 background"""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="road",       layer_id=1, max_n_particles=200_000),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)

    v1_ckpt = _v1_shape_dict(N=100, conf=real_conf)
    model.init_from_checkpoint(v1_ckpt, setup_optimizer=False)
    assert model.layers["background"].positions.shape == (100, 3)
    assert model.layers["road"].positions.shape == (0, 3)


def test_multi_layer_ckpt_roundtrip(real_conf):
    """T1.4: 2 层 LayeredGaussians: save → load 后各层 tensor 字节级一致"""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="road", layer_id=1, max_n_particles=200_000),
    ]
    model_a = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)

    src_ckpt = {
        "global_step": 30000,
        "model": {
            "gaussians_nodes": {
                "background": _v1_shape_dict(N=300, conf=real_conf),
                "road":       _v1_shape_dict(N=150, conf=real_conf),
            },
        },
    }
    model_a.init_from_checkpoint(src_ckpt, setup_optimizer=False)

    # MoG.get_model_parameters() asserts optimizer is not None (checkpoint
    # writes per-param-group state). The test-only helper attaches a minimal
    # Adam so we can exercise the save path without the full conf-driven
    # scheduler tree.
    model_a.setup_optimizer_for_test()

    saved = model_a.get_model_parameters()
    assert "gaussians_nodes" in saved
    assert set(saved["gaussians_nodes"].keys()) == {"background", "road"}

    model_b = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    model_b.init_from_checkpoint(
        {"gaussians_nodes": saved["gaussians_nodes"]},
        setup_optimizer=False,
    )

    for layer_name in ["background", "road"]:
        for attr in [
            "positions", "rotation", "scale", "density",
            "features_albedo", "features_specular",
        ]:
            t_a = getattr(model_a.layers[layer_name], attr)
            t_b = getattr(model_b.layers[layer_name], attr)
            assert torch.equal(t_a.data, t_b.data), (
                f"Roundtrip mismatch at {layer_name}.{attr}: "
                f"shapes {t_a.shape} vs {t_b.shape}"
            )


# --- T2.5: fused_view / get_layer_mask ---
def test_fused_view_single_bg_passes_through(real_conf):
    """T2.5: single-bg mode → fused_view returns the bg layer's Parameters
    (identity check, not a new concat tensor — this is the byte-identical fast path).
    """
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=600_000)]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    v1_ckpt = _v1_shape_dict(N=50, conf=real_conf)
    model.init_from_checkpoint(v1_ckpt, setup_optimizer=False)

    fused = model.fused_view()
    for attr in ["positions", "rotation", "scale", "density",
                 "features_albedo", "features_specular"]:
        assert fused[attr] is getattr(model.layers["background"], attr), (
            f"single-bg fused_view must short-circuit to bg layer's {attr}"
        )


def test_fused_view_two_layers_concat_shape(real_conf):
    """T2.5: two-layer fused_view → concat in specs order; shape and values check."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="road",       layer_id=1, max_n_particles=200_000),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    model.init_from_checkpoint(
        {"gaussians_nodes": {
            "background": _v1_shape_dict(N=100, conf=real_conf),
            "road":       _v1_shape_dict(N=50,  conf=real_conf),
        }},
        setup_optimizer=False,
    )

    fused = model.fused_view()
    assert fused["positions"].shape == (150, 3)
    assert fused["density"].shape == (150, 1)
    # specs order: first 100 = background, next 50 = road
    assert torch.equal(fused["positions"][:100], model.layers["background"].positions)
    assert torch.equal(fused["positions"][100:], model.layers["road"].positions)


def test_get_layer_mask_partitions_two_layers(real_conf):
    """T2.5: get_layer_mask returns a Bool[N_total] mask partitioning the two layers."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="road",       layer_id=1, max_n_particles=200_000),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    model.init_from_checkpoint(
        {"gaussians_nodes": {
            "background": _v1_shape_dict(N=100, conf=real_conf),
            "road":       _v1_shape_dict(N=50,  conf=real_conf),
        }},
        setup_optimizer=False,
    )

    bg_mask = model.get_layer_mask("background")
    road_mask = model.get_layer_mask("road")
    assert bg_mask.shape == (150,)
    assert road_mask.shape == (150,)
    assert bg_mask.dtype == torch.bool
    assert bg_mask[:100].all().item() and not bg_mask[100:].any().item()
    assert (not road_mask[:100].any().item()) and road_mask[100:].all().item()
    # partition: union covers all; intersection empty
    assert (bg_mask | road_mask).all().item()
    assert not (bg_mask & road_mask).any().item()


def test_get_layer_mask_unknown_name_raises(real_conf):
    """T2.5: unknown / non-particle layer name → ValueError with helpful message."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=600_000)]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    with pytest.raises(ValueError, match="unknown layer"):
        model.get_layer_mask("nonexistent")


# --- T3.0: init_layer_from_points + optimizer property ---
def _two_layer_model(real_conf):
    """Helper: build a bg+road LayeredGaussians for T3.0 tests."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="road",       layer_id=1, max_n_particles=200_000,
                  scale_prior=(0.1, 0.1, 0.001), mask_field="road_mask"),
    ]
    return LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)


def test_init_layer_from_points_routes_to_mog(real_conf):
    """T3.0: init_layer_from_points places positions into the named layer's MoG
    without touching other layers; defaults from spec.scale_prior / density_init.
    """
    model = _two_layer_model(real_conf)
    assert model.layers["road"].num_gaussians == 0
    assert model.layers["background"].num_gaussians == 0

    pts = torch.randn(100, 3)
    model.init_layer_from_points("road", pts, setup_optimizer=False)

    assert model.layers["road"].positions.shape == (100, 3)
    # spec.scale_prior=(0.1, 0.1, 0.001) → log-applied; Z-axis log(0.001) ≈ -6.907
    sz = model.layers["road"].scale[:, 2].exp().max().item()
    assert sz < 0.005, f"road Z scale prior leaked: max={sz}"
    # background stays empty
    assert model.layers["background"].num_gaussians == 0


def test_init_layer_from_points_unknown_layer_raises(real_conf):
    """T3.0: unknown layer name raises ValueError listing enabled layers."""
    model = _two_layer_model(real_conf)
    with pytest.raises(ValueError, match="unknown layer"):
        model.init_layer_from_points("sky_envmap", torch.randn(10, 3),
                                     setup_optimizer=False)


def test_init_layer_from_points_track_ids_registered_as_buffer(real_conf):
    """T3.0: track_ids kwarg registers a persistent buffer on the layer (for T4.3)."""
    model = _two_layer_model(real_conf)
    pts = torch.randn(20, 3)
    tids = torch.arange(20, dtype=torch.long)
    model.init_layer_from_points("road", pts, track_ids=tids, setup_optimizer=False)
    layer = model.layers["road"]
    assert hasattr(layer, "track_ids")
    assert layer.track_ids.dtype == torch.int64
    assert layer.track_ids.shape == (20,)
    # registered as a named buffer (so .to(device) / state_dict() carry it)
    assert "track_ids" in dict(layer.named_buffers())


def test_optimizer_property_single_bg_passthrough(real_conf):
    """T3.0: single-bg mode short-circuits to the bg layer's optimizer
    (byte-identical with v1 — no wrapper allocation)."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=600_000)]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    model.init_layer_from_points("background", torch.randn(30, 3),
                                 setup_optimizer=False)
    model.setup_optimizer_for_test()

    bg_opt = model.layers["background"].optimizer
    assert model.optimizer is bg_opt, (
        "single-bg mode must return the bg layer's optimizer identity"
    )


def test_optimizer_wrapper_steps_all_layers(real_conf, monkeypatch):
    """T3.0: multi-layer mode: model.optimizer.step() fans out to every layer's
    sub-optimizer; zero_grad / param_groups also aggregated."""
    model = _two_layer_model(real_conf)
    model.init_layer_from_points("background", torch.randn(50, 3),
                                 setup_optimizer=False)
    model.init_layer_from_points("road",       torch.randn(50, 3),
                                 setup_optimizer=False)
    model.setup_optimizer_for_test()

    calls: list[str] = []
    monkeypatch.setattr(
        model.layers["background"].optimizer, "step",
        lambda *a, **kw: calls.append("bg"),
    )
    monkeypatch.setattr(
        model.layers["road"].optimizer, "step",
        lambda *a, **kw: calls.append("road"),
    )

    view = model.optimizer
    # Multi-layer: NOT the same object as bg.optimizer; it's a fan-out view.
    assert view is not model.layers["background"].optimizer
    view.step()
    assert set(calls) == {"bg", "road"}, f"got {calls}"

    # param_groups aggregation: sum of per-layer groups
    expected = sum(len(l.optimizer.param_groups) for l in model.layers.values())
    assert len(view.param_groups) == expected


# --- T4.0: tracks buffer ---
def test_layered_gaussians_holds_tracks_buffers(real_conf):
    """T4.0: tracks kwarg → tracks_poses / tracks_active dicts populated AND
    PyTorch buffers registered for each track (so .to(device) / state_dict work)."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="dynamic_rigids", layer_id=2, max_n_particles=200_000),
    ]
    tracks = {
        "alice": {"poses": torch.eye(4).expand(20, 4, 4).clone(),
                  "active": torch.ones(20, dtype=torch.bool)},
        "bob":   {"poses": torch.eye(4).expand(20, 4, 4).clone(),
                  "active": torch.cat([torch.zeros(5, dtype=torch.bool),
                                       torch.ones(15, dtype=torch.bool)])},
    }
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0,
                             tracks=tracks)

    # Mirror dicts populated
    assert set(model.tracks_poses.keys()) == {"alice", "bob"}
    assert model.tracks_poses["alice"].shape == (20, 4, 4)
    assert model.tracks_active["bob"].sum().item() == 15

    # Buffers registered (named_buffers includes them)
    buf_names = dict(model.named_buffers())
    assert "_track_pose_alice" in buf_names
    assert "_track_pose_bob" in buf_names
    assert "_track_active_alice" in buf_names
    assert buf_names["_track_pose_alice"].shape == (20, 4, 4)
    # mirror dict and buffer point to the same tensor (no copy)
    assert model.tracks_poses["alice"] is buf_names["_track_pose_alice"]


def test_layered_gaussians_no_tracks_default(real_conf):
    """T4.0: when tracks=None (default), tracks_poses/active are empty dicts;
    no buffer pollution."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=600_000)]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    assert model.tracks_poses == {}
    assert model.tracks_active == {}
    # No _track_pose_* / _track_active_* buffers
    buf_names = list(dict(model.named_buffers()).keys())
    assert not any(n.startswith("_track_pose_") for n in buf_names)
    assert not any(n.startswith("_track_active_") for n in buf_names)


# --- T4.3: _transform_means + fused_view dynamic 分支 ---
def _make_dyn_model(real_conf, tracks: dict, n_pts_per_track=10):
    """Helper: build bg+dyn LayeredGaussians, init dyn with given tracks."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="dynamic_rigids", layer_id=2, max_n_particles=200_000,
                  scale_prior=(0.05, 0.05, 0.05)),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0,
                             tracks=tracks)
    # bg init
    model.init_layer_from_points("background", torch.randn(5, 3),
                                 setup_optimizer=False)
    # dyn init: concat per-track points + track_ids
    track_names = sorted(tracks.keys())
    all_pts = []
    all_ids = []
    for tid in track_names:
        pts = torch.zeros(n_pts_per_track, 3)  # all at local origin for simplicity
        all_pts.append(pts)
        all_ids.append(torch.full((n_pts_per_track,),
                                  track_names.index(tid), dtype=torch.long))
    model.init_layer_from_points("dynamic_rigids",
                                 torch.cat(all_pts),
                                 track_ids=torch.cat(all_ids),
                                 setup_optimizer=False)
    return model


def test_transform_means_identity_pose(real_conf):
    """T4.3: identity pose → world == local (no change)."""
    tracks = {"v0": {
        "poses": torch.eye(4).expand(5, 4, 4).clone(),
        "active": torch.ones(5, dtype=torch.bool),
    }}
    model = _make_dyn_model(real_conf, tracks, n_pts_per_track=3)
    local = model.layers["dynamic_rigids"].positions
    world = model._transform_means(
        local, model.layers["dynamic_rigids"].track_ids, frame_id=2,
    )
    assert torch.allclose(world, local.to(world.dtype))


def test_transform_means_single_track_translation(real_conf):
    """T4.3: pose = identity rot + (1, 2, 3) translation → world = local + t."""
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    tracks = {"v0": {
        "poses": pose.expand(5, 4, 4).clone(),
        "active": torch.ones(5, dtype=torch.bool),
    }}
    model = _make_dyn_model(real_conf, tracks, n_pts_per_track=4)
    local = model.layers["dynamic_rigids"].positions  # all zeros
    world = model._transform_means(
        local, model.layers["dynamic_rigids"].track_ids, frame_id=2,
    )
    expected = torch.tensor([1.0, 2.0, 3.0]).expand(4, 3)
    assert torch.allclose(world, expected.to(world.dtype))


def test_transform_means_multi_track_routing(real_conf):
    """T4.3: 2 tracks with different poses → particles routed correctly."""
    pose_a = torch.eye(4); pose_a[:3, 3] = torch.tensor([10.0, 0.0, 0.0])
    pose_b = torch.eye(4); pose_b[:3, 3] = torch.tensor([0.0, 20.0, 0.0])
    tracks = {
        "alice": {"poses": pose_a.expand(5, 4, 4).clone(),
                  "active": torch.ones(5, dtype=torch.bool)},
        "bob":   {"poses": pose_b.expand(5, 4, 4).clone(),
                  "active": torch.ones(5, dtype=torch.bool)},
    }
    model = _make_dyn_model(real_conf, tracks, n_pts_per_track=3)
    # _make_dyn_model assigns track_ids 0 to alice (sorted), 1 to bob
    local = model.layers["dynamic_rigids"].positions  # 6 zero pts (3 per track)
    world = model._transform_means(
        local, model.layers["dynamic_rigids"].track_ids, frame_id=0,
    )
    # First 3 → alice translation, last 3 → bob translation
    assert torch.allclose(world[:3], torch.tensor([10.0, 0.0, 0.0]).expand(3, 3).to(world.dtype))
    assert torch.allclose(world[3:], torch.tensor([0.0, 20.0, 0.0]).expand(3, 3).to(world.dtype))


def test_fused_view_dynamic_layer_applies_transform(real_conf):
    """T4.3: fused_view(frame_id=N) on bg+dyn → dyn positions transformed to world."""
    pose = torch.eye(4); pose[:3, 3] = torch.tensor([7.0, 8.0, 9.0])
    tracks = {"v0": {
        "poses": pose.expand(5, 4, 4).clone(),
        "active": torch.ones(5, dtype=torch.bool),
    }}
    model = _make_dyn_model(real_conf, tracks, n_pts_per_track=2)
    fused = model.fused_view(frame_id=0)
    # bg: 5 pts (random); dyn: 2 pts (origin → transformed to (7,8,9))
    assert fused["positions"].shape == (7, 3)
    # Last 2 rows = transformed dyn pts
    dyn_world = fused["positions"][5:]
    assert torch.allclose(dyn_world, torch.tensor([7.0, 8.0, 9.0]).expand(2, 3).to(dyn_world.dtype))


def test_fused_view_dynamic_layer_frame_id_none_uses_first_active_fallback(real_conf):
    """E.2.c (replaces the original "skip transform" behaviour): when no
    timestamp_us or frame_id is given, each track falls back to its first
    active frame so inference free cameras don't dump dyn particles to
    world origin.

    With every frame active and identity rotations + translation (7,8,9),
    the first-active fallback picks frame 0 → world position = local + t.
    """
    pose = torch.eye(4); pose[:3, 3] = torch.tensor([7.0, 8.0, 9.0])
    tracks = {"v0": {
        "poses": pose.expand(5, 4, 4).clone(),
        "active": torch.ones(5, dtype=torch.bool),
    }}
    model = _make_dyn_model(real_conf, tracks, n_pts_per_track=2)
    fused = model.fused_view(frame_id=None)
    # dyn local positions are zeros; world = R · 0 + t = (7, 8, 9)
    dyn_world = fused["positions"][5:]
    expected = torch.tensor([7.0, 8.0, 9.0]).expand_as(dyn_world)
    assert torch.allclose(dyn_world, expected.to(dyn_world))


# --- T3.5: multi-layer forward routing + _FusedView contract ---
def test_fused_view_object_exposes_full_mog_contract(real_conf):
    """T3.5: _FusedView exposes all attrs/methods the renderer accesses on
    a MoG: positions / rotation / scale / density / features_albedo /
    features_specular / num_gaussians / n_active_features / background /
    get_rotation() / get_scale() / get_density() / get_features() /
    get_positions(). Borrows activations + background from ref layer."""
    from threedgrut.layers.layered_model import LayeredGaussians, _FusedView

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="road",       layer_id=1, max_n_particles=200_000,
                  scale_prior=(0.1, 0.1, 0.001)),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    model.init_layer_from_points("background", torch.randn(30, 3),
                                 setup_optimizer=False)
    model.init_layer_from_points("road", torch.randn(20, 3),
                                 setup_optimizer=False)

    fused = model.fused_view(frame_id=0)
    ref = model.layers["background"]
    view = _FusedView(fused, ref)

    # Direct tensor access
    assert view.positions.shape == (50, 3)
    assert view.rotation.shape == (50, 4)
    assert view.scale.shape == (50, 3)
    assert view.density.shape == (50, 1)
    assert view.features_albedo.shape[0] == 50
    assert view.features_specular.shape[0] == 50

    # Shape / config
    assert view.num_gaussians == 50
    assert view.n_active_features == ref.n_active_features
    assert view.max_n_features == ref.max_n_features
    assert view.background is ref.background  # identity reuse, no copy

    # Activated accessors run through ref's activation fns
    rot = view.get_rotation()
    assert torch.allclose(rot, ref.rotation_activation(fused["rotation"]))
    scl = view.get_scale()
    assert torch.allclose(scl, ref.scale_activation(fused["scale"]))
    dns = view.get_density()
    assert torch.allclose(dns, ref.density_activation(fused["density"]))

    # Pre-activation passthrough
    assert view.get_rotation(preactivation=True) is fused["rotation"]
    assert view.get_scale(preactivation=True) is fused["scale"]
    assert view.get_density(preactivation=True) is fused["density"]

    # Features concat
    feat = view.get_features()
    assert feat.shape[0] == 50
    assert feat.shape[1] == (view.features_albedo.shape[1]
                              + view.features_specular.shape[1])

    # get_positions = positions (no activation on positions)
    assert view.get_positions() is fused["positions"]


def test_forward_single_bg_passes_through_to_bg_layer(real_conf, monkeypatch):
    """T3.5: single-bg mode → forward() bypasses fused_view path entirely
    and delegates to bg.__call__ (byte-identical with v1)."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=600_000)]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)

    # Spy on bg layer __call__; intercept to avoid invoking real renderer.
    bg = model.layers["background"]
    calls: list = []
    def fake_call(self, gpu_batch, train=False, frame_id=0):
        calls.append((id(gpu_batch), train, frame_id))
        return {"pred_rgb": torch.zeros(1, 4, 4, 3)}
    monkeypatch.setattr(type(bg), "__call__", fake_call)

    out = model(object(), train=True, frame_id=42)  # gpu_batch sentinel
    assert len(calls) == 1
    assert calls[0][1] is True
    assert calls[0][2] == 42
    assert "pred_rgb" in out


def test_forward_multi_layer_dispatches_to_ref_renderer(real_conf, monkeypatch):
    """T3.5: multi-layer mode → forward() builds fused_view + _FusedView
    + calls ref_layer.renderer.render(view, batch, train, frame_id).

    We don't run a real renderer (no CUDA tracer on Mac); instead we
    monkey-patch ref.renderer.render to capture (view_type, train, frame_id)
    and verify fused tensor shapes propagate correctly."""
    from threedgrut.layers.layered_model import LayeredGaussians, _FusedView

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="road",       layer_id=1, max_n_particles=200_000,
                  scale_prior=(0.1, 0.1, 0.001)),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    model.init_layer_from_points("background", torch.randn(30, 3),
                                 setup_optimizer=False)
    model.init_layer_from_points("road", torch.randn(20, 3),
                                 setup_optimizer=False)

    ref = next(iter(model.layers.values()))
    captured: dict = {}
    def fake_render(view, gpu_batch, train, frame_id):
        captured["view_type"] = type(view).__name__
        captured["num_gaussians"] = view.num_gaussians
        captured["train"] = train
        captured["frame_id"] = frame_id
        return {"pred_rgb": torch.zeros(1, 4, 4, 3)}
    monkeypatch.setattr(ref.renderer, "render", fake_render)

    out = model(object(), train=False, frame_id=7)
    assert captured["view_type"] == "_FusedView"
    assert captured["num_gaussians"] == 50  # 30 bg + 20 road
    assert captured["train"] is False
    assert captured["frame_id"] == 7
    assert "pred_rgb" in out


# ============================================================================
# T5.4: sky envmap layer integration tests.
# ============================================================================
def _make_fake_batch(H: int = 4, W: int = 4):
    """Minimal Batch-like object exposing rays_dir / T_to_world / world flag.

    LayeredGaussians._blend_sky only touches these three attributes.
    """
    class _Batch:
        pass
    b = _Batch()
    # Camera-frame rays roughly forward; world-frame conversion goes through
    # T_to_world's rotation.
    b.rays_dir = torch.randn(1, H, W, 3)
    b.T_to_world = torch.eye(4).unsqueeze(0)  # identity → rays unchanged
    b.rays_in_world_space = False
    b.timestamp_us = -1
    return b


def test_layered_gaussians_holds_sky_module_mlp(real_conf):
    """T5.4: sky_envmap layer with backend='mlp' creates SkyEnvmapMLP module."""
    from threedgrut.layers.layered_model import LayeredGaussians
    from threedgrut.correction.sky_envmap import SkyEnvmapMLP

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="sky_envmap", layer_id=4, max_n_particles=0,
                  scale_prior=(0.0, 0.0, 0.0), is_particle_layer=False,
                  extra={"backend": "mlp"}),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    assert "sky_envmap" in model.layers
    assert isinstance(model.layers["sky_envmap"], SkyEnvmapMLP)


def test_layered_gaussians_holds_sky_module_cubemap(real_conf):
    """T5.4: backend='cubemap' creates SkyEnvmapCubemap with custom resolution."""
    from threedgrut.layers.layered_model import LayeredGaussians
    from threedgrut.correction.sky_envmap import SkyEnvmapCubemap

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="sky_envmap", layer_id=4, max_n_particles=0,
                  scale_prior=(0.0, 0.0, 0.0), is_particle_layer=False,
                  extra={"backend": "cubemap", "resolution": 32}),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    sky = model.layers["sky_envmap"]
    assert isinstance(sky, SkyEnvmapCubemap)
    assert sky.base.shape == (6, 32, 32, 3)


def test_blend_sky_alpha_zero_returns_sky_only(real_conf):
    """alpha=0 (no Gaussians hit) → pred_rgb == rgb_sky (Gauss contributes 0)."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="sky_envmap", layer_id=4, max_n_particles=0,
                  scale_prior=(0.0, 0.0, 0.0), is_particle_layer=False,
                  extra={"backend": "mlp"}),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    batch = _make_fake_batch(H=4, W=4)
    outputs = {
        "pred_rgb": torch.zeros(1, 4, 4, 3),
        "pred_opacity": torch.zeros(1, 4, 4, 1),
    }
    out = model._blend_sky(outputs, batch)
    assert "rgb_sky" in out and "rgb_gaussians" in out
    # Gauss is 0, alpha is 0 → pred_rgb == rgb_sky.
    assert torch.allclose(out["pred_rgb"], out["rgb_sky"])


def test_blend_sky_alpha_one_returns_gauss_only(real_conf):
    """alpha=1 (fully opaque) → pred_rgb == rgb_gauss, sky contributes 0."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="sky_envmap", layer_id=4, max_n_particles=0,
                  scale_prior=(0.0, 0.0, 0.0), is_particle_layer=False,
                  extra={"backend": "mlp"}),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    batch = _make_fake_batch(H=4, W=4)
    rgb_gauss = torch.rand(1, 4, 4, 3)
    outputs = {
        "pred_rgb": rgb_gauss.clone(),
        "pred_opacity": torch.ones(1, 4, 4, 1),
    }
    out = model._blend_sky(outputs, batch)
    assert torch.allclose(out["pred_rgb"], rgb_gauss, atol=1e-6)
    assert torch.allclose(out["rgb_gaussians"], rgb_gauss, atol=1e-6)


def test_blend_sky_passes_through_when_no_sky_layer(real_conf):
    """No sky layer in specs → outputs unmodified, no rgb_sky / rgb_gaussians key."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="road",       layer_id=1, max_n_particles=200_000,
                  scale_prior=(0.1, 0.1, 0.001)),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    batch = _make_fake_batch()
    outputs = {
        "pred_rgb": torch.rand(1, 4, 4, 3),
        "pred_opacity": torch.rand(1, 4, 4, 1),
    }
    out = model._blend_sky(outputs, batch)
    # When sky is absent, _blend_sky must be a no-op.
    assert out is outputs
    assert "rgb_sky" not in out
    assert "rgb_gaussians" not in out


def test_forward_multi_layer_with_sky_attaches_sky_outputs(real_conf, monkeypatch):
    """Multi-layer forward with sky_envmap → outputs gains rgb_sky + rgb_gaussians.

    Mocks the renderer to return a constant rgb / opacity so we can verify the
    blend keys appear and the math is correct end-to-end through forward.
    """
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="sky_envmap", layer_id=4, max_n_particles=0,
                  scale_prior=(0.0, 0.0, 0.0), is_particle_layer=False,
                  extra={"backend": "mlp"}),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    model.init_layer_from_points("background", torch.randn(10, 3),
                                 setup_optimizer=False)

    ref = model.layers["background"]
    def fake_render(view, gpu_batch, train, frame_id):
        return {
            "pred_rgb":     torch.zeros(1, 4, 4, 3),  # no gaussian contribution
            "pred_opacity": torch.zeros(1, 4, 4, 1),  # transparent → sky only
        }
    monkeypatch.setattr(ref.renderer, "render", fake_render)

    out = model(_make_fake_batch(H=4, W=4), train=False, frame_id=0)
    assert "rgb_sky" in out
    assert "rgb_gaussians" in out
    # alpha=0 + gauss=0 → pred_rgb == rgb_sky elementwise.
    assert torch.allclose(out["pred_rgb"], out["rgb_sky"])


def test_sky_envmap_state_roundtrip_in_checkpoint(real_conf):
    """T5.4: sky layer state_dict round-trips through get_model_parameters /
    init_from_checkpoint without touching gaussians_nodes.

    Bug found on A800 dry-run: LayeredGaussians.get_model_parameters tried to
    call ``layer.get_model_parameters()`` on the SkyEnvmapMLP module (which is
    not a MoG). The fix routes sky via state_dict() under a sibling key.
    """
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="sky_envmap", layer_id=4, max_n_particles=0,
                  scale_prior=(0.0, 0.0, 0.0), is_particle_layer=False,
                  extra={"backend": "mlp"}),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    # Init the particle layer so get_model_parameters has something to emit.
    # setup_optimizer_for_test attaches a minimal Adam so each particle layer's
    # own get_model_parameters() (which asserts on self.optimizer) passes.
    model.init_layer_from_points("background", torch.randn(8, 3),
                                 setup_optimizer=False)
    model.setup_optimizer_for_test()

    # Mutate sky weights so the round-trip can detect them.
    sky_layer0 = model.layers["sky_envmap"].layer0
    with torch.no_grad():
        sky_layer0.weight.fill_(0.42)
        sky_layer0.bias.fill_(-0.13)
    saved_weight = sky_layer0.weight.detach().clone()
    saved_bias = sky_layer0.bias.detach().clone()

    params = model.get_model_parameters()
    # Save shape contract.
    assert "gaussians_nodes" in params
    assert "background" in params["gaussians_nodes"]
    assert "sky_envmap" not in params["gaussians_nodes"]   # NOT under gaussians_nodes
    assert "sky_envmap_state" in params                     # sibling key

    # Build a fresh container and round-trip through init_from_checkpoint.
    model2 = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    # Wrap under "model.gaussians_nodes" + "model.sky_envmap_state" to match
    # Trainer.save_checkpoint on-disk schema for LayeredGaussians.
    ckpt = {"model": params}
    model2.init_from_checkpoint(ckpt, setup_optimizer=False)

    # Sky weights restored bit-for-bit.
    assert torch.equal(model2.layers["sky_envmap"].layer0.weight, saved_weight)
    assert torch.equal(model2.layers["sky_envmap"].layer0.bias, saved_bias)


def test_get_model_parameters_skips_non_particle_layers(real_conf):
    """T5.4: dynamic_deformables (is_particle_layer=False stub) must not be
    iterated as a MoG — the gaussians_nodes dict only contains particle layers.
    """
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="sky_envmap", layer_id=4, max_n_particles=0,
                  scale_prior=(0.0, 0.0, 0.0), is_particle_layer=False,
                  extra={"backend": "mlp"}),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    model.init_layer_from_points("background", torch.randn(5, 3),
                                 setup_optimizer=False)
    model.setup_optimizer_for_test()
    params = model.get_model_parameters()
    assert list(params["gaussians_nodes"].keys()) == ["background"]


# ============================================================================
# viser_gui_4d "Gaussian Layers" toggle — runtime layer enable/disable.
# Backs the GUI control in viser_gui_4d.py:_build_static_gui; GUI mutates
# self.enabled_layer_names via wholesale set replacement.
# ============================================================================
def test_enabled_layer_names_default_includes_all_contributing(real_conf):
    """Default enabled set covers every layer that actually contributes to
    the rendered image: all particle layers + sky_envmap (if present).
    dynamic_deformables stub is excluded (is_particle_layer=False, no module)."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background",    layer_id=0, max_n_particles=600_000),
        LayerSpec(name="road",          layer_id=1, max_n_particles=200_000),
        LayerSpec(name="dynamic_rigids", layer_id=2, max_n_particles=200_000),
        LayerSpec(name="sky_envmap",    layer_id=4, max_n_particles=0,
                  scale_prior=(0.0, 0.0, 0.0), is_particle_layer=False,
                  extra={"backend": "mlp"}),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    assert model.enabled_layer_names == {
        "background", "road", "dynamic_rigids", "sky_envmap",
    }


def test_fused_view_skips_disabled_layer(real_conf):
    """Disabling a particle layer drops its particles from the concat. With
    bg(100) + road(50) enabled → 150 rows; disable road → 100 rows from bg only."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="road",       layer_id=1, max_n_particles=200_000),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    model.init_from_checkpoint(
        {"gaussians_nodes": {
            "background": _v1_shape_dict(N=100, conf=real_conf),
            "road":       _v1_shape_dict(N=50,  conf=real_conf),
        }},
        setup_optimizer=False,
    )

    assert model.fused_view()["positions"].shape == (150, 3)
    # Wholesale set replacement — mirrors the viser_gui_4d callback pattern.
    object.__setattr__(
        model, "enabled_layer_names",
        model.enabled_layer_names - {"road"},
    )
    fused = model.fused_view()
    assert fused["positions"].shape == (100, 3)
    assert torch.equal(fused["positions"], model.layers["background"].positions)


def test_fused_view_all_disabled_returns_zero_particle(real_conf):
    """All particle layers disabled → fused_view returns 0-row tensors with
    correct trailing dims so consumers never trip on torch.cat([])."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="road",       layer_id=1, max_n_particles=200_000),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    model.init_from_checkpoint(
        {"gaussians_nodes": {
            "background": _v1_shape_dict(N=100, conf=real_conf),
            "road":       _v1_shape_dict(N=50,  conf=real_conf),
        }},
        setup_optimizer=False,
    )
    object.__setattr__(model, "enabled_layer_names", set())
    fused = model.fused_view()
    assert fused["positions"].shape == (0, 3)
    assert fused["rotation"].shape == (0, 4)
    assert fused["scale"].shape == (0, 3)
    assert fused["density"].shape == (0, 1)
    # features_specular trailing dim borrowed from a real layer (sh dependent).
    ref = model.layers["background"]
    assert fused["features_specular"].shape == (0, ref.features_specular.shape[1])


def test_forward_all_disabled_returns_empty_render(real_conf, monkeypatch):
    """forward() with every particle layer disabled must NOT call ref renderer.
    Returns _empty_render (zero RGB / opacity) and then blends sky on top
    (no-op when sky is also disabled, which is the default for this spec set)."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="road",       layer_id=1, max_n_particles=200_000,
                  scale_prior=(0.1, 0.1, 0.001)),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    model.init_layer_from_points("background", torch.randn(10, 3),
                                 setup_optimizer=False)
    model.init_layer_from_points("road", torch.randn(10, 3),
                                 setup_optimizer=False)

    # Trip-wire on both layers' renderers — neither should be called when
    # everything is off.
    for layer in model.layers.values():
        monkeypatch.setattr(
            layer.renderer, "render",
            lambda *a, **kw: pytest.fail(
                "renderer.render must not run when all particle layers disabled"
            ),
        )
    object.__setattr__(model, "enabled_layer_names", set())

    out = model(_make_fake_batch(H=4, W=4), train=False, frame_id=0)
    assert "pred_rgb" in out
    assert out["pred_rgb"].shape == (1, 4, 4, 3)
    assert torch.all(out["pred_rgb"] == 0)
    assert torch.all(out["pred_opacity"] == 0)


def test_blend_sky_skipped_when_sky_disabled(real_conf):
    """sky_envmap disabled via enabled_layer_names → _blend_sky is a no-op
    (no rgb_sky / rgb_gaussians keys added). Mirrors the viser checkbox
    flipping sky off while keeping particle layers on."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="sky_envmap", layer_id=4, max_n_particles=0,
                  scale_prior=(0.0, 0.0, 0.0), is_particle_layer=False,
                  extra={"backend": "mlp"}),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    object.__setattr__(
        model, "enabled_layer_names",
        model.enabled_layer_names - {"sky_envmap"},
    )
    batch = _make_fake_batch(H=4, W=4)
    rgb_gauss = torch.rand(1, 4, 4, 3)
    outputs = {
        "pred_rgb":     rgb_gauss.clone(),
        "pred_opacity": torch.zeros(1, 4, 4, 1),  # alpha=0 would normally pull sky in
    }
    out = model._blend_sky(outputs, batch)
    assert out is outputs
    assert "rgb_sky" not in out
    assert "rgb_gaussians" not in out
    assert torch.equal(out["pred_rgb"], rgb_gauss)
