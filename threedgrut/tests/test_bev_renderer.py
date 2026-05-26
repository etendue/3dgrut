# SPDX-License-Identifier: Apache-2.0
"""Unit tests for threedgrut_playground.utils.bev_renderer (V3-VIZ.1).

Pure CPU / Mac-runnable. No torch, no GPU, no viser.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from threedgrut_playground.utils.bev_renderer import (
    BEVRenderInputs,
    build_inputs_from_metadata,
    render_bev_frame,
)


@dataclass
class _FakeMeta:
    """Minimal FourDMetadata stand-in for tests."""
    ego_poses_c2w: np.ndarray
    ego_frame_timestamps_us: np.ndarray
    sequence_id: str = "synthetic"
    tracks: dict = None
    tracks_camera_timestamps_us: np.ndarray = None

    def __post_init__(self):
        if self.tracks is None:
            self.tracks = {}
        if self.tracks_camera_timestamps_us is None:
            self.tracks_camera_timestamps_us = self.ego_frame_timestamps_us

    def active_tracks_at(self, fi: int):
        out = []
        for tid, t in self.tracks.items():
            m = t["frame_info"]
            if 0 <= fi < m.shape[0] and bool(m[fi]):
                out.append(tid)
        return out


def _make_ego_line(n: int = 20, step: float = 1.0) -> np.ndarray:
    """Straight-line ego trajectory along +X at z=0."""
    poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    poses[:, 0, 3] = np.arange(n, dtype=np.float32) * step
    # Camera looks down +Z (NCore convention): leave R = I → heading = (0,0,1)
    # which projects to (0,0) — fallback unit (0,1). Tweak so heading is +X by
    # setting Z-col-of-R to point +X (rotate -90deg about Y).
    R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
    poses[:, :3, :3] = R
    return poses


def test_build_inputs_basic_no_cuboids():
    """With empty tracks, build_inputs should still produce a usable BEVRenderInputs."""
    n = 10
    poses = _make_ego_line(n=n)
    ts = (np.arange(n) * 100_000).astype(np.int64)
    meta = _FakeMeta(ego_poses_c2w=poses, ego_frame_timestamps_us=ts)

    layer_positions = {
        "background": np.array([[5.0, 1.0, 0.5], [5.0, -1.0, 0.5]], dtype=np.float32),
    }
    inputs = build_inputs_from_metadata(meta, layer_positions, frame_idx=5)

    assert inputs.ego_xy_trajectory.shape == (n, 2)
    assert np.allclose(inputs.ego_current_xy, [5.0, 0.0])
    assert np.allclose(inputs.ego_current_heading_xy, [1.0, 0.0], atol=1e-5)
    assert inputs.cuboids == []
    assert "background" in inputs.layer_positions_xy
    assert inputs.layer_positions_xy["background"].shape == (2, 2)


def test_build_inputs_z_filter_drops_high_points():
    poses = _make_ego_line(n=3)
    ts = (np.arange(3) * 100_000).astype(np.int64)
    meta = _FakeMeta(ego_poses_c2w=poses, ego_frame_timestamps_us=ts)

    layer_positions = {
        "background": np.array([
            [0.0, 0.0, 0.0],          # in-window
            [0.0, 0.0, 50.0],         # sky-high, should drop
            [0.0, 0.0, -50.0],        # sub-ground, should drop
        ], dtype=np.float32),
    }
    inputs = build_inputs_from_metadata(meta, layer_positions, frame_idx=1, z_window_m=10.0)
    assert inputs.layer_positions_xy["background"].shape == (1, 2)


def test_build_inputs_active_cuboid_at_frame():
    n = 5
    poses = _make_ego_line(n=n)
    ts = (np.arange(n) * 100_000).astype(np.int64)
    track_poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    track_poses[:, :2, 3] = np.array([[2.0, 1.0]] * n, dtype=np.float32)
    frame_info = np.array([False, True, True, False, False], dtype=bool)
    meta = _FakeMeta(
        ego_poses_c2w=poses, ego_frame_timestamps_us=ts,
        tracks={"42": {
            "poses": track_poses,
            "size": np.array([4.5, 1.8, 1.5], dtype=np.float32),
            "frame_info": frame_info,
            "class": "automobile",
        }},
    )

    inputs_f0 = build_inputs_from_metadata(meta, {}, frame_idx=0)
    inputs_f1 = build_inputs_from_metadata(meta, {}, frame_idx=1)
    assert inputs_f0.cuboids == []
    assert len(inputs_f1.cuboids) == 1
    cu = inputs_f1.cuboids[0]
    assert cu["tid"] == "42"
    assert cu["class"] == "automobile"
    assert cu["footprint_xy"].shape == (4, 2)
    # Footprint is centered at (2, 1) with extents (4.5/2, 1.8/2).
    fp = cu["footprint_xy"]
    assert np.isclose(fp[:, 0].mean(), 2.0, atol=1e-5)
    assert np.isclose(fp[:, 1].mean(), 1.0, atol=1e-5)


def test_render_bev_frame_shape_and_nonempty():
    """Render returns RGB image of expected dimensions, non-empty (variance > 0)."""
    n = 3
    poses = _make_ego_line(n=n)
    ts = (np.arange(n) * 100_000).astype(np.int64)
    meta = _FakeMeta(ego_poses_c2w=poses, ego_frame_timestamps_us=ts)
    layer_positions = {
        "background": np.random.RandomState(0).normal(0, 5, (200, 3)).astype(np.float32),
        "road": np.random.RandomState(1).normal(0, 3, (100, 3)).astype(np.float32),
    }
    inputs = build_inputs_from_metadata(meta, layer_positions, frame_idx=1)
    img = render_bev_frame(inputs, xy_range_m=20.0, dpi=80, title="test")
    assert img.ndim == 3 and img.shape[-1] == 3
    # 10×10 inches @ 80 dpi → 800×800 (matplotlib rounds; allow ±5 px slack).
    assert 750 <= img.shape[0] <= 850, img.shape
    assert 750 <= img.shape[1] <= 850, img.shape
    assert img.dtype == np.uint8
    # Variance > 0 means we actually drew something (not solid white).
    assert img.var() > 100.0


def test_render_bev_frame_empty_inputs():
    """Renderer should not crash with all-empty inputs (zero ego frames)."""
    inputs = BEVRenderInputs(
        ego_xy_trajectory=np.empty((0, 2), dtype=np.float32),
        ego_current_xy=np.zeros(2, dtype=np.float32),
        ego_current_heading_xy=np.array([0.0, 1.0], dtype=np.float32),
        cuboids=[],
        layer_positions_xy={},
    )
    img = render_bev_frame(inputs, xy_range_m=10.0, dpi=60)
    assert img.shape[-1] == 3
