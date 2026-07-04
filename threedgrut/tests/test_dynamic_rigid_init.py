# SPDX-License-Identifier: Apache-2.0
"""T4.2.a / T4.3.a unit tests for dynamic_rigid_init + _transform_means.

Pure CPU mock tests.
"""

from __future__ import annotations

import torch

from threedgrut.layers.dynamic_rigid_init import init_dynamic_rigid_layer


def _identity_track(F: int = 5, size=(4.0, 2.0, 1.5), active=None) -> dict:
    eye = torch.eye(4).expand(F, 4, 4).clone()
    active = torch.ones(F, dtype=torch.bool) if active is None else active
    return {
        "pts": None,
        "colors": None,
        "poses": eye,
        "size": torch.tensor(list(size)),
        "frame_info": active,
        "class": "vehicle",
    }


def _translated_track(F: int = 5, t=(10.0, 5.0, 0.0), size=(4.0, 2.0, 1.5)) -> dict:
    eye = torch.eye(4)
    eye[:3, 3] = torch.tensor(list(t))
    poses = eye.expand(F, 4, 4).clone()
    return {
        "pts": None,
        "colors": None,
        "poses": poses,
        "size": torch.tensor(list(size)),
        "frame_info": torch.ones(F, dtype=torch.bool),
        "class": "vehicle",
    }


# --- T4.2.a ---
def test_init_dyn_rigid_cuboid_filter_keeps_inside_points():
    """T4.2.a: identity-pose track (4×2×1.5 box at origin) → LiDAR points
    inside box are kept (in object-local frame == world frame for identity)."""
    tracks = {"v0": _identity_track(F=1, size=(4.0, 2.0, 1.5))}
    # 8 LiDAR points: 4 inside cuboid (|x|<2, |y|<1, |z|<0.75), 4 outside
    pts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.5, 0.5],
            [-1.0, -0.5, -0.3],
            [1.9, 0.9, 0.7],  # inside (4)
            [3.0, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 0.0, 1.0],
            [-3.0, -3.0, -3.0],  # outside (4)
        ]
    )
    positions, track_ids, names = init_dynamic_rigid_layer(tracks, pts, max_pts_per_track=100)
    assert positions.shape == (4, 3)
    assert torch.all(track_ids == 0)
    assert names == ["v0"]


def test_init_dyn_rigid_local_frame_roundtrip():
    """T4.2.a: translated track → LiDAR points in world but cuboid filter
    works in local frame; local pts NEAR origin (not at world translation)."""
    tracks = {"v0": _translated_track(F=1, t=(10.0, 5.0, 0.0), size=(4.0, 2.0, 1.5))}
    # World pts near (10, 5, 0) → in local frame near (0, 0, 0)
    pts = torch.tensor(
        [
            [10.0, 5.0, 0.0],  # local (0,0,0)
            [11.5, 5.5, 0.5],  # local (1.5, 0.5, 0.5) inside
            [11.9, 5.9, 0.7],  # local (1.9, 0.9, 0.7) inside
            [13.0, 5.0, 0.0],  # local (3, 0, 0) outside (|x|>2)
        ]
    )
    positions, _, _ = init_dynamic_rigid_layer(tracks, pts, max_pts_per_track=100)
    # 3 inside; positions in LOCAL frame (near origin, not (10,5,0))
    assert positions.shape == (3, 3)
    assert positions[:, 0].abs().max().item() < 2.0  # local |x| within half-extent
    # No point near world (10,5,0)
    world_dist = (positions - torch.tensor([10.0, 5.0, 0.0])).norm(dim=-1)
    assert world_dist.min().item() > 5.0


def test_init_dyn_rigid_subsample_respects_max_pts():
    """T4.2.a: > max_pts_per_track 后输出截到 max_pts_per_track."""
    tracks = {"v0": _identity_track(F=1, size=(20.0, 20.0, 20.0))}  # huge box
    # 1000 random pts all inside (size half=10)
    pts = (torch.rand(1000, 3) - 0.5) * 10
    positions, track_ids, _ = init_dynamic_rigid_layer(tracks, pts, max_pts_per_track=100)
    assert positions.shape == (100, 3)
    assert track_ids.shape == (100,)


def test_init_dyn_rigid_multi_track_routing():
    """T4.2.a: 2 tracks → concat positions + correct track_ids 路由."""
    tracks = {
        "v0": _identity_track(F=1, size=(2.0, 2.0, 2.0)),
        "v1": _translated_track(F=1, t=(50.0, 0.0, 0.0), size=(2.0, 2.0, 2.0)),
    }
    pts = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # v0 inside (local 0,0,0)
            [0.5, 0.5, 0.5],  # v0 inside
            [50.0, 0.0, 0.0],  # v1 inside (local 0,0,0)
            [50.3, 0.0, 0.0],  # v1 inside
            [50.4, 0.0, 0.0],  # v1 inside
            [100.0, 0.0, 0.0],  # neither
        ]
    )
    positions, track_ids, names = init_dynamic_rigid_layer(tracks, pts, max_pts_per_track=100)
    # Determinism: sorted names → v0=0, v1=1
    assert names == ["v0", "v1"]
    # 2 pts for v0, 3 pts for v1, total 5
    assert positions.shape == (5, 3)
    assert (track_ids == 0).sum().item() == 2
    assert (track_ids == 1).sum().item() == 3


def test_init_dyn_rigid_empty_lidar_returns_empty():
    """T4.2.a: 空 dyn_lidar_pts → shape (0,3) / (0,) 返回不 crash."""
    tracks = {"v0": _identity_track(F=5)}
    positions, track_ids, names = init_dynamic_rigid_layer(tracks, torch.zeros(0, 3), max_pts_per_track=100)
    assert positions.shape == (0, 3)
    assert track_ids.shape == (0,)
    assert names == ["v0"]


def test_init_dyn_rigid_empty_tracks_returns_empty():
    """T4.2.a: 空 instance_pts_dict → 空输出."""
    positions, track_ids, names = init_dynamic_rigid_layer({}, torch.randn(100, 3), max_pts_per_track=50)
    assert positions.shape == (0, 3)
    assert names == []


def test_init_dyn_rigid_inactive_frames_skipped():
    """T4.2.a: 只有部分 frame active → 只用 active frame 的 pose 过滤."""
    active = torch.tensor([False, True, False, False, False])
    tracks = {"v0": _identity_track(F=5, size=(4.0, 2.0, 1.5), active=active)}
    pts = torch.tensor([[0.0, 0.0, 0.0]])  # 1 point at origin (in box)
    positions, _, _ = init_dynamic_rigid_layer(tracks, pts, max_pts_per_track=100)
    # 1 active frame * 1 inside pt = 1 collected
    assert positions.shape == (1, 3)


def test_init_dyn_rigid_mutates_instance_pts_dict():
    """T4.2.a: 函数 mutate instance_pts_dict[tid]['pts'] 填入 local-frame 点."""
    tracks = {"v0": _identity_track(F=1, size=(4.0, 2.0, 1.5))}
    assert tracks["v0"]["pts"] is None  # before
    pts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    init_dynamic_rigid_layer(tracks, pts, max_pts_per_track=100)
    assert tracks["v0"]["pts"].shape == (2, 3)  # after
