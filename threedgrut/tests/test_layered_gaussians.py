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


def test_fused_view_dynamic_layer_frame_id_none_skips_transform(real_conf):
    """T4.3 D4: frame_id=None → dynamic positions passed through unchanged
    (TODO Stage 8 inference fallback)."""
    pose = torch.eye(4); pose[:3, 3] = torch.tensor([7.0, 8.0, 9.0])
    tracks = {"v0": {
        "poses": pose.expand(5, 4, 4).clone(),
        "active": torch.ones(5, dtype=torch.bool),
    }}
    model = _make_dyn_model(real_conf, tracks, n_pts_per_track=2)
    fused = model.fused_view(frame_id=None)
    # dyn pts should still be at local origin (zeros), not transformed
    dyn_local = fused["positions"][5:]
    assert torch.allclose(dyn_local, torch.zeros_like(dyn_local))
