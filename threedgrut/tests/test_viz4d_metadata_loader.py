"""Task C contract tests — FourDMetadata loader + timeline helpers.

Tests the pure-CPU container that ``viser_gui_4d.py`` uses to parse and look
up entries in ``ckpt['viz_4d']``. No viser / kaolin / engine imports.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from threedgrut_playground.utils.viz4d_metadata import FourDMetadata


# ---------------------------------------------------------- helpers
def _make_viz_block(F: int = 5, N_ego: int = 6, road_pts: int = 100,
                    dyn_pts: int = 50) -> dict:
    return {
        "schema_version": 1,
        "dataset_type":   "ncore",
        "sequence_id":    "seq_x",
        "ego": {
            "poses_c2w": torch.stack(
                [torch.eye(4) for _ in range(N_ego)]
            ),
            "frame_timestamps_us": torch.tensor(
                [1000 * (i + 1) for i in range(N_ego)], dtype=torch.int64
            ),
            "primary_camera_id":        "front_long",
            "primary_camera_fov_y_rad": 0.65,
            "primary_camera_aspect":    1.78,
        },
        "tracks": {
            "t0": {
                "poses":      torch.eye(4).repeat(F, 1, 1),
                "size":       torch.tensor([2.0, 1.5, 4.5]),
                "frame_info": torch.tensor([1, 1, 0, 1, 1], dtype=torch.bool)[:F],
                "class":      "automobile",
            },
            "t1": {
                "poses":      torch.eye(4).repeat(F, 1, 1),
                "size":       torch.tensor([3.0, 2.5, 12.0]),
                "frame_info": torch.zeros(F, dtype=torch.bool),
                "class":      "heavy_truck",
            },
        },
        "tracks_camera_timestamps_us": torch.tensor(
            [1000 * (i + 1) for i in range(F)], dtype=torch.int64),
        "lidar": {
            "road_xyz":          torch.randn(road_pts, 3) if road_pts > 0 else None,
            "road_rgb":          torch.rand(road_pts, 3) if road_pts > 0 else None,
            "road_n_total":      road_pts * 5,
            "road_subsample":    road_pts,
            "dynamic_xyz":       torch.randn(dyn_pts, 3),
            "dynamic_rgb":       None,
            "dynamic_n_total":   dyn_pts * 5,
            "dynamic_subsample": dyn_pts,
        },
        "viewer_defaults": {
            "initial_c2w":  torch.eye(4),
            "near":         0.1,
            "far":          500.0,
            "resolution":   1024,
            "t_us_first":   1000,
            "t_us_last":    5000,
        },
    }


# ---------------------------------------------------------- tests
def test_from_ckpt_none_when_missing():
    assert FourDMetadata.from_ckpt({}) is None
    assert FourDMetadata.from_ckpt({"model": {}}) is None
    assert FourDMetadata.from_ckpt({"viz_4d": None}) is None


def test_from_ckpt_basic_fields():
    md = FourDMetadata.from_ckpt({"viz_4d": _make_viz_block()})
    assert md is not None
    assert md.schema_version == 1
    assert md.sequence_id == "seq_x"
    assert md.ego_primary_camera_id == "front_long"
    assert md.ego_poses_c2w.shape == (6, 4, 4)
    assert md.ego_poses_c2w.dtype == np.float32
    assert md.ego_frame_timestamps_us.dtype == np.int64
    assert md.n_tracks() == 2
    assert md.n_frames() == 5
    assert md.has_lidar() is True
    assert md.road_xyz.shape == (100, 3)
    assert md.dyn_xyz.shape == (50, 3)


def test_lookup_frame_idx_binary_search():
    md = FourDMetadata.from_ckpt({"viz_4d": _make_viz_block(F=5)})
    # ts buffer = [1000, 2000, 3000, 4000, 5000]
    assert md.lookup_frame_idx(1000) == 0
    assert md.lookup_frame_idx(3000) == 2
    assert md.lookup_frame_idx(2100) == 1  # closer to 2000
    assert md.lookup_frame_idx(2900) == 2  # closer to 3000
    assert md.lookup_frame_idx(0) == 0
    assert md.lookup_frame_idx(9999) == 4


def test_active_tracks_at():
    md = FourDMetadata.from_ckpt({"viz_4d": _make_viz_block(F=5)})
    # t0 mask = [1,1,0,1,1]; t1 mask = [0,0,0,0,0]
    assert md.active_tracks_at(0) == ["t0"]
    assert md.active_tracks_at(2) == []  # t0 inactive at idx 2
    assert md.active_tracks_at(3) == ["t0"]
    # Out-of-range idx → no actives (defensive)
    assert md.active_tracks_at(-1) == []
    assert md.active_tracks_at(99) == []


def test_ego_pose_at_nearest():
    block = _make_viz_block(N_ego=4)
    # ego ts = [1000, 2000, 3000, 4000]; poses all identity but tag translations
    poses = block["ego"]["poses_c2w"]
    for i in range(4):
        poses[i, 0, 3] = float(i + 1) * 10.0  # x = 10, 20, 30, 40
    md = FourDMetadata.from_ckpt({"viz_4d": block})
    assert md.ego_pose_at(1000)[0, 3] == 10.0
    assert md.ego_pose_at(2100)[0, 3] == 20.0  # closer to 2000
    assert md.ego_pose_at(5000)[0, 3] == 40.0  # past end → clamp


def test_empty_tracks_block():
    block = _make_viz_block()
    block["tracks"] = {}
    block["tracks_camera_timestamps_us"] = None
    md = FourDMetadata.from_ckpt({"viz_4d": block})
    assert md.n_tracks() == 0
    assert md.n_frames() == 0
    assert md.lookup_frame_idx(1234) == 0
    assert md.active_tracks_at(0) == []


def test_no_lidar_block():
    block = _make_viz_block()
    block["lidar"] = {
        "road_xyz": None, "road_rgb": None, "road_n_total": None, "road_subsample": None,
        "dynamic_xyz": None, "dynamic_rgb": None, "dynamic_n_total": None, "dynamic_subsample": None,
    }
    md = FourDMetadata.from_ckpt({"viz_4d": block})
    assert md.has_lidar() is False
    assert md.road_xyz is None
    assert md.dyn_xyz is None


def test_ckpt_roundtrip_via_torch_save(tmp_path):
    """Mirror the real save_checkpoint roundtrip: torch.save → torch.load → from_ckpt."""
    ckpt = {"viz_4d": _make_viz_block(F=4)}
    p = tmp_path / "smoke.pt"
    torch.save(ckpt, p)
    ckpt2 = torch.load(p, weights_only=False)
    md = FourDMetadata.from_ckpt(ckpt2)
    assert md is not None
    assert md.n_tracks() == 2
    assert md.lookup_frame_idx(int(md.tracks_camera_timestamps_us[1])) == 1
