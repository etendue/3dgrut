"""Bug 1 fix — Viser4DViewer Follow Ego checkbox + camera snap on time change.

The Mac-side contract test: __init__ takes ``_follow_ego_enabled`` default False;
``_snap_clients_to_ego`` iterates server.get_clients() and writes
wxyz/position/look_at/up_direction from ``meta.ego_pose_at(t_us)``; the
existing ``_on_time_change`` end-of-method conditionally calls it.

We stub heavy deps (viser/kaolin/engine) so the test runs on Mac CPU.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest


@pytest.fixture
def viewer_class(monkeypatch):
    """Stub heavy deps and return Viser4DViewer for direct construction.

    Mirrors test_viser_gui_4d_fov.py's viewer_class fixture so the two tests
    share the same import strategy. We do NOT spin up a real ViserServer or
    GUI — the test only exercises the Follow Ego state field and the
    _snap_clients_to_ego helper.
    """
    fake_viser = SimpleNamespace(
        ViserServer=mock.MagicMock,
        ClientHandle=object,
        CameraHandle=object,
        transforms=SimpleNamespace(SO3=mock.MagicMock),
    )
    monkeypatch.setitem(sys.modules, "viser", fake_viser)
    monkeypatch.setitem(sys.modules, "viser.transforms", fake_viser.transforms)
    fake_kaolin = SimpleNamespace(render=SimpleNamespace(
        camera=SimpleNamespace(Camera=object)))
    monkeypatch.setitem(sys.modules, "kaolin", fake_kaolin)
    monkeypatch.setitem(sys.modules, "kaolin.render", fake_kaolin.render)
    monkeypatch.setitem(sys.modules, "kaolin.render.camera",
                        fake_kaolin.render.camera)
    fake_engine_mod = SimpleNamespace(Engine3DGRUT=type("Engine3DGRUT", (), {}))
    monkeypatch.setitem(sys.modules, "threedgrut_playground.engine",
                        fake_engine_mod)
    from threedgrut_playground import viser_gui_4d
    return viser_gui_4d.Viser4DViewer


def _make_fake_metadata():
    """FourDMetadata with 3 ego frames at known timestamps + poses.

    Frame 0: position (0, 0, 0)     identity rotation
    Frame 1: position (10, 0, 0)    identity rotation
    Frame 2: position (20, 0, 0)    identity rotation
    """
    from threedgrut_playground.utils.viz4d_metadata import FourDMetadata
    poses = np.tile(np.eye(4, dtype=np.float32)[None, ...], (3, 1, 1))
    poses[0, :3, 3] = [0.0, 0.0, 0.0]
    poses[1, :3, 3] = [10.0, 0.0, 0.0]
    poses[2, :3, 3] = [20.0, 0.0, 0.0]
    ts = np.array([0, 1_000_000, 2_000_000], dtype=np.int64)
    return FourDMetadata(
        schema_version=2, sequence_id="test",
        ego_poses_c2w=poses,
        ego_frame_timestamps_us=ts,
        ego_primary_camera_id="primary",
        ego_primary_fov_y_rad=1.0, ego_primary_aspect=1.78,
        ego_primary_intrinsics_ftheta=None,
        ego_primary_resolution=None,
        tracks={}, tracks_camera_timestamps_us=ts,
        road_xyz=None, road_rgb=None, dyn_xyz=None, dyn_rgb=None,
        road_n_total=None, dyn_n_total=None,
        dyn_local_xyz=None, dyn_track_ids=None, dyn_track_names=None,
        initial_c2w=poses[0], t_us_first=0, t_us_last=2_000_000,
    )


def _bypass_init_viewer(viewer_cls, *, meta):
    """Construct a Viser4DViewer without running __init__ side effects.

    We bypass the real __init__ (which would build a ViserServer + GUI) and
    manually set just the fields under test. Then we attach a stub server
    whose get_clients() returns a controllable dict of fake clients.
    """
    with mock.patch.object(viewer_cls, "__init__", autospec=True) as init_mock:
        init_mock.return_value = None
        viewer = viewer_cls(port=8080, engine=None, metadata=meta)
    viewer.meta = meta
    viewer._t_us_current = meta.t_us_first
    viewer._follow_ego_enabled = False
    viewer.h_ego_frustum = None
    # Fake server with mutable clients dict.
    fake_client = SimpleNamespace(camera=SimpleNamespace(
        wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        position=np.array([99.0, 99.0, 99.0]),  # initial "wrong" position
        look_at=np.array([0.0, 0.0, 0.0]),
        up_direction=np.array([0.0, 1.0, 0.0]),
    ))
    fake_server = SimpleNamespace(get_clients=lambda: {"c1": fake_client})
    viewer.server = fake_server
    return viewer, fake_client


# ============================================================================
def test_follow_ego_default_off(viewer_class):
    """Viser4DViewer should default _follow_ego_enabled to False so existing
    free-orbit behavior is preserved (zero-regression contract)."""
    meta = _make_fake_metadata()
    viewer, _ = _bypass_init_viewer(viewer_class, meta=meta)
    assert viewer._follow_ego_enabled is False


def test_snap_clients_to_ego_writes_camera_pose(viewer_class):
    """_snap_clients_to_ego must write wxyz + position + look_at + up_direction
    on every connected client, derived from meta.ego_pose_at(t_us)."""
    meta = _make_fake_metadata()
    viewer, client = _bypass_init_viewer(viewer_class, meta=meta)
    # Snap to frame 1 (position (10, 0, 0))
    viewer._snap_clients_to_ego(1_000_000)
    # Position must match ego pose translation at t=1e6 us
    np.testing.assert_allclose(
        np.asarray(client.camera.position, dtype=np.float32),
        np.array([10.0, 0.0, 0.0], dtype=np.float32),
        atol=1e-5,
    )
    # wxyz must be identity quaternion (poses are identity rotations)
    np.testing.assert_allclose(
        np.asarray(client.camera.wxyz, dtype=np.float32),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        atol=1e-5,
    )


def test_snap_clients_no_op_when_meta_none(viewer_class):
    """Guard: viewer with meta=None (v1 ckpt static-3D mode) must not crash
    on _snap_clients_to_ego — the method is a no-op."""
    meta = _make_fake_metadata()
    viewer, client = _bypass_init_viewer(viewer_class, meta=meta)
    viewer.meta = None
    initial_pos = client.camera.position.copy()
    viewer._snap_clients_to_ego(1_000_000)  # should not raise, not write
    np.testing.assert_array_equal(client.camera.position, initial_pos)


def test_on_time_change_writes_camera_only_when_enabled(viewer_class):
    """_on_time_change → _snap_clients_to_ego gated by _follow_ego_enabled.

    follow_ego=False: client camera position untouched.
    follow_ego=True:  client camera position equals ego_pose translation.
    """
    meta = _make_fake_metadata()
    viewer, client = _bypass_init_viewer(viewer_class, meta=meta)
    # Stub the other side-effect methods called from _on_time_change so the
    # test doesn't need cuboid/frustum/lidar machinery.
    viewer._update_ego_frustum = lambda t: None
    viewer._update_active_cuboids = lambda idx: None
    viewer._update_dynamic_lidar = lambda idx: None
    viewer._mirror_ui = lambda t, idx: None

    initial_pos = client.camera.position.copy()
    viewer._on_time_change(1_000_000, source="test")
    # Follow OFF: camera should NOT move.
    np.testing.assert_array_equal(client.camera.position, initial_pos)
    # Flip Follow ON and advance to frame 2 (pos=(20,0,0)).
    viewer._follow_ego_enabled = True
    viewer._on_time_change(2_000_000, source="test")
    np.testing.assert_allclose(
        np.asarray(client.camera.position, dtype=np.float32),
        np.array([20.0, 0.0, 0.0], dtype=np.float32),
        atol=1e-5,
    )
