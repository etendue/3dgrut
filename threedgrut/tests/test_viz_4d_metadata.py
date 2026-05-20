"""Task B contract tests — extract_4d_metadata + ckpt viz_4d block.

These verify the pure-CPU metadata extraction surface that
``Trainer.save_checkpoint`` calls when ``conf.viz_4d.enabled``. The trainer
hook itself is exercised separately on A800 (manual smoke); here we keep
everything synthesisable on a Mac:

  - extract_smoke:       schema_version + sub-dict presence
  - subsample_respected: lidar subsample caps both xyz and rgb
  - tracks_metadata_*:   class/size land in tracks_metadata via populate_tracks
  - extract_no_dataset_lidar: dataset without get_*_lidar_points keeps None
  - extract_with_no_tracks:    empty tracks dict + None shared ts
  - schema_version_constant:   stay at 1 until a deliberate bump
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.viz.metadata import SCHEMA_VERSION, extract_4d_metadata


_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs")
)


# ----------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def real_conf():
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="apps/ncore_3dgut_mcmc")


def _mock_dataset(*, n_frames: int = 5, road_pts: int = 1000,
                  dyn_pts: int = 500, with_lidar: bool = True):
    """Build a duck-typed dataset matching NCoreDataset's viz_4d surface."""
    poses = np.eye(4, dtype=np.float32)[None].repeat(n_frames, axis=0)
    poses[:, 0, 3] = np.arange(n_frames, dtype=np.float32)
    frame_indices = np.arange(n_frames)
    timestamps_us = (np.arange(n_frames, dtype=np.int64) + 1) * 1000  # 1ms steps
    # NCore sensor.frames_timestamps_us has [N, 2] (START / END columns).
    raw_ts = np.stack([timestamps_us - 100, timestamps_us], axis=1)

    camera_model = SimpleNamespace(
        resolution=torch.tensor([1600.0, 900.0]),
        focal_length=torch.tensor([1200.0, 1200.0]),
    )
    camera_sensor = SimpleNamespace(frames_timestamps_us=raw_ts)
    seq_id = "test_seq"
    ds = SimpleNamespace(
        sequence_id=seq_id,
        camera_ids=["front_long"],
        camera_train_frame_indices={"front_long": frame_indices},
        sequence_camera_sensors={seq_id: {"front_long": camera_sensor}},
        sequence_camera_models={seq_id: {"front_long": camera_model}},
        get_poses=lambda: poses,
    )
    if with_lidar:
        ds.get_road_lidar_points = lambda: (
            torch.randn(road_pts, 3), torch.rand(road_pts, 3))
        ds.get_dynamic_lidar_points = lambda: (
            torch.randn(dyn_pts, 3), None)
    return ds


def _model_with_tracks(real_conf, *, with_metadata: bool = True):
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1000)]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=1.0)
    F = 5
    tracks = {
        "t0": {
            "poses":      torch.eye(4).repeat(F, 1, 1),
            "frame_info": torch.ones(F, dtype=torch.bool),
            "cam_timestamps_us": torch.tensor([1000, 2000, 3000, 4000, 5000],
                                              dtype=torch.int64),
        },
        "t1": {
            "poses":      torch.eye(4).repeat(F, 1, 1),
            "frame_info": torch.tensor([1, 0, 1, 1, 0], dtype=torch.bool),
        },
    }
    if with_metadata:
        tracks["t0"]["class"] = "automobile"
        tracks["t0"]["size"]  = torch.tensor([2.0, 1.5, 4.5])
        tracks["t1"]["class"] = "heavy_truck"
        tracks["t1"]["size"]  = torch.tensor([3.0, 2.5, 12.0])
    model.populate_tracks(tracks)
    return model


# ----------------------------------------------------------------- tests
def test_extract_smoke(real_conf):
    model = _model_with_tracks(real_conf)
    dataset = _mock_dataset(n_frames=5)
    md = extract_4d_metadata(model, dataset, real_conf)

    assert md["schema_version"] == SCHEMA_VERSION == 1
    assert md["dataset_type"] == "ncore"
    assert md["sequence_id"] == "test_seq"
    # ego
    assert md["ego"]["poses_c2w"].shape == (5, 4, 4)
    assert md["ego"]["frame_timestamps_us"].shape == (5,)
    assert md["ego"]["primary_camera_id"] == "front_long"
    assert md["ego"]["primary_camera_aspect"] == pytest.approx(1600.0 / 900.0)
    # tracks
    assert set(md["tracks"].keys()) == {"t0", "t1"}
    assert md["tracks"]["t0"]["poses"].shape == (5, 4, 4)
    assert md["tracks"]["t0"]["class"] == "automobile"
    assert md["tracks"]["t1"]["class"] == "heavy_truck"
    # tracks shared timestamps
    assert md["tracks_camera_timestamps_us"].shape == (5,)
    # lidar
    assert md["lidar"]["road_xyz"].shape[0] <= 1000
    assert md["lidar"]["road_subsample"] == md["lidar"]["road_xyz"].shape[0]
    # viewer_defaults
    assert md["viewer_defaults"]["initial_c2w"].shape == (4, 4)
    assert md["viewer_defaults"]["t_us_first"] == 1000
    assert md["viewer_defaults"]["t_us_last"] == 5000


def test_subsample_respected(real_conf):
    """Large lidar gets sliced to the subsample cap (xyz + rgb in lockstep)."""
    model = _model_with_tracks(real_conf)
    dataset = _mock_dataset(n_frames=3, road_pts=5000, dyn_pts=3000)

    # Override defaults via a minimal patch of real_conf using OmegaConf.
    from omegaconf import OmegaConf
    conf = OmegaConf.merge(real_conf, OmegaConf.create({
        "viz_4d": {"lidar_road_subsample": 200, "lidar_dynamic_subsample": 100,
                   "include_lidar": True}
    }))
    md = extract_4d_metadata(model, dataset, conf)
    assert md["lidar"]["road_xyz"].shape == (200, 3)
    assert md["lidar"]["road_rgb"].shape == (200, 3)
    assert md["lidar"]["road_n_total"] == 5000
    assert md["lidar"]["road_subsample"] == 200
    assert md["lidar"]["dynamic_xyz"].shape == (100, 3)
    assert md["lidar"]["dynamic_n_total"] == 3000


def test_include_lidar_false_skips_clouds(real_conf):
    model = _model_with_tracks(real_conf)
    dataset = _mock_dataset()
    from omegaconf import OmegaConf
    conf = OmegaConf.merge(real_conf, OmegaConf.create({
        "viz_4d": {"include_lidar": False}
    }))
    md = extract_4d_metadata(model, dataset, conf)
    assert md["lidar"]["road_xyz"] is None
    assert md["lidar"]["dynamic_xyz"] is None


def test_tracks_metadata_populated_via_populate_tracks(real_conf):
    """populate_tracks captures class/size into model.tracks_metadata so
    extract_4d_metadata can surface them."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1000)]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=1.0)
    model.populate_tracks({
        "ta": {
            "poses": torch.eye(4).repeat(3, 1, 1),
            "frame_info": torch.ones(3, dtype=torch.bool),
            "cam_timestamps_us": torch.tensor([10, 20, 30], dtype=torch.int64),
            "class": "bus",
            "size": torch.tensor([3.0, 2.5, 12.0]),
        },
    })
    assert "ta" in model.tracks_metadata
    assert model.tracks_metadata["ta"]["class"] == "bus"
    assert torch.allclose(model.tracks_metadata["ta"]["size"],
                          torch.tensor([3.0, 2.5, 12.0]))


def test_extract_without_tracks(real_conf):
    """Single-bg LayeredGaussians (no tracks populated) yields empty tracks
    and None shared timestamps — viewer gracefully degrades."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1000)]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=1.0)
    dataset = _mock_dataset(with_lidar=False)
    md = extract_4d_metadata(model, dataset, real_conf)
    assert md["tracks"] == {}
    assert md["tracks_camera_timestamps_us"] is None


def test_dataset_without_lidar_methods(real_conf):
    """No-LiDAR datasets (ColmapDataset) don't crash; lidar block stays None."""
    model = _model_with_tracks(real_conf)
    dataset = _mock_dataset(with_lidar=False)
    md = extract_4d_metadata(model, dataset, real_conf)
    assert md["lidar"]["road_xyz"] is None
    assert md["lidar"]["dynamic_xyz"] is None


def test_unknown_class_default(real_conf):
    """populate_tracks without class info → extract yields 'unknown' (not crash)."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1000)]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=1.0)
    model.populate_tracks({
        "tx": {
            "poses": torch.eye(4).repeat(2, 1, 1),
            "frame_info": torch.ones(2, dtype=torch.bool),
            "cam_timestamps_us": torch.tensor([1, 2], dtype=torch.int64),
        },
    })
    dataset = _mock_dataset(n_frames=2, with_lidar=False)
    md = extract_4d_metadata(model, dataset, real_conf)
    assert md["tracks"]["tx"]["class"] == "unknown"


def test_ckpt_roundtrip(tmp_path, real_conf):
    """torch.save / torch.load preserves the viz_4d block intact."""
    model = _model_with_tracks(real_conf)
    dataset = _mock_dataset(n_frames=4, road_pts=300)
    md = extract_4d_metadata(model, dataset, real_conf)

    ckpt = {"model": {"gaussians_nodes": {}}, "viz_4d": md}
    path = tmp_path / "smoke.pt"
    torch.save(ckpt, path)
    reloaded = torch.load(path, weights_only=False)
    assert reloaded["viz_4d"]["schema_version"] == 1
    assert set(reloaded["viz_4d"]["tracks"].keys()) == set(md["tracks"].keys())
    assert reloaded["viz_4d"]["lidar"]["road_xyz"].shape == md["lidar"]["road_xyz"].shape
