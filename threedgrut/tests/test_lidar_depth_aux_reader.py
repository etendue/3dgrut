# SPDX-License-Identifier: Apache-2.0
"""LidarDepthAuxReader + DepthV2AuxReader unit tests (Stage 11 T11.B2).

These readers load per-frame npz depth maps dumped by
scripts/dump_lidar_depth_map.py (and the DepthV2 dump). npz layout:
    <root>/<camera_id>/<timestamp_us>.npz   with single key "depth" [H,W] float32
"""

import numpy as np
import pytest

from threedgrut.datasets.aux_readers import DepthV2AuxReader, LidarDepthAuxReader


@pytest.fixture
def fake_depth_dir(tmp_path):
    cam_id = "camera_front_wide_120fov"
    ts = 1234567890
    cam_dir = tmp_path / cam_id
    cam_dir.mkdir(parents=True)
    dmap = np.zeros((216, 384), dtype=np.float32)
    dmap[10, 20] = 15.5  # one valid pixel
    np.savez_compressed(cam_dir / f"{ts}.npz", depth=dmap)
    return tmp_path, cam_id, ts, dmap


def test_lidar_depth_aux_reader_basic(fake_depth_dir):
    root, cam_id, ts, expected = fake_depth_dir
    reader = LidarDepthAuxReader(root)
    assert reader.has_frame(cam_id, ts)
    got = reader.read(cam_id, ts)
    np.testing.assert_array_equal(got, expected)
    assert got.dtype == np.float32


def test_lidar_depth_aux_reader_missing_returns_zero(fake_depth_dir):
    """Missing frame returns all-zeros [default_shape] so hit_mask=0 → loss skips it.

    A camera/frame with no LiDAR coverage is a legal situation, not an error.
    """
    root, cam_id, _, _ = fake_depth_dir
    reader = LidarDepthAuxReader(root, default_shape=(216, 384))
    got = reader.read(cam_id, timestamp_us=9999999999)  # missing
    assert got.shape == (216, 384)
    assert got.sum() == 0.0
    assert got.dtype == np.float32


def test_lidar_depth_aux_reader_missing_no_default_raises(fake_depth_dir):
    """Without default_shape, a missing frame is a hard error (caller must opt in)."""
    root, cam_id, _, _ = fake_depth_dir
    reader = LidarDepthAuxReader(root)  # no default_shape
    with pytest.raises(FileNotFoundError):
        reader.read(cam_id, timestamp_us=9999999999)


def test_lidar_depth_aux_reader_caches(fake_depth_dir):
    """Second read of the same frame returns the cached array (same object)."""
    root, cam_id, ts, _ = fake_depth_dir
    reader = LidarDepthAuxReader(root)
    a = reader.read(cam_id, ts)
    b = reader.read(cam_id, ts)
    assert a is b  # cache hit returns identical object


def test_depthv2_aux_reader_is_subclass(fake_depth_dir):
    """DepthV2AuxReader shares LidarDepthAuxReader's npz-loading behavior."""
    root, cam_id, ts, expected = fake_depth_dir
    reader = DepthV2AuxReader(root)
    assert isinstance(reader, LidarDepthAuxReader)
    got = reader.read(cam_id, ts)
    np.testing.assert_array_equal(got, expected)


def test_has_frame_missing_returns_false(fake_depth_dir):
    """has_frame must report False for a frame that was never dumped."""
    root, cam_id, _, _ = fake_depth_dir
    reader = LidarDepthAuxReader(root)
    assert reader.has_frame(cam_id, timestamp_us=9999999999) is False


def test_wrong_npz_key_raises_clear_error(tmp_path):
    """An npz missing the 'depth' key raises a KeyError that names the path."""
    cam_id = "camera_front_wide_120fov"
    ts = 555
    cam_dir = tmp_path / cam_id
    cam_dir.mkdir(parents=True)
    # Wrong key: 'd' instead of 'depth'
    np.savez_compressed(cam_dir / f"{ts}.npz", d=np.zeros((4, 4), dtype=np.float32))
    reader = LidarDepthAuxReader(tmp_path)
    with pytest.raises(KeyError, match="no 'depth' key"):
        reader.read(cam_id, ts)


def test_cache_eviction_bounds_memory(tmp_path):
    """With cache_maxsize=2, reading 3 distinct frames evicts the oldest."""
    cam_id = "cam"
    cam_dir = tmp_path / cam_id
    cam_dir.mkdir(parents=True)
    for ts in (1, 2, 3):
        np.savez_compressed(cam_dir / f"{ts}.npz", depth=np.full((2, 2), float(ts), dtype=np.float32))
    reader = LidarDepthAuxReader(tmp_path, cache_maxsize=2)
    reader.read(cam_id, 1)
    reader.read(cam_id, 2)
    reader.read(cam_id, 3)  # evicts ts=1
    assert (cam_id, 1) not in reader._cache
    assert (cam_id, 2) in reader._cache
    assert (cam_id, 3) in reader._cache
