from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from threedgrut_playground.utils.camera_render_state import CameraModelKind


@pytest.fixture
def viewer_module(monkeypatch):
    fake_viser = SimpleNamespace(
        ViserServer=mock.MagicMock,
        ClientHandle=object,
        CameraHandle=object,
        transforms=SimpleNamespace(SO3=mock.MagicMock),
    )
    monkeypatch.setitem(sys.modules, "viser", fake_viser)
    monkeypatch.setitem(sys.modules, "viser.transforms", fake_viser.transforms)
    fake_kaolin = SimpleNamespace(render=SimpleNamespace(camera=SimpleNamespace(Camera=object)))
    monkeypatch.setitem(sys.modules, "kaolin", fake_kaolin)
    monkeypatch.setitem(sys.modules, "kaolin.render", fake_kaolin.render)
    monkeypatch.setitem(sys.modules, "kaolin.render.camera", fake_kaolin.render.camera)
    fake_engine_mod = SimpleNamespace(Engine3DGRUT=type("Engine3DGRUT", (), {}))
    monkeypatch.setitem(sys.modules, "threedgrut_playground.engine", fake_engine_mod)
    from threedgrut_playground import viser_gui_4d

    return viser_gui_4d


def _poses(offset: float = 0.0) -> np.ndarray:
    poses = np.tile(np.eye(4, dtype=np.float32)[None], (2, 1, 1))
    poses[0, 0, 3] = offset
    poses[1, 0, 3] = offset + 10.0
    return poses


def _ftheta_dict() -> dict:
    return {
        "resolution": np.array([6, 4], dtype=np.int64),
        "shutter_type": "GLOBAL",
        "principal_point": np.array([3.0, 2.0], dtype=np.float32),
        "reference_poly": "ANGLE_TO_PIXELDIST",
        "pixeldist_to_angle_poly": np.array([0.0, 0.01], dtype=np.float32),
        "angle_to_pixeldist_poly": np.array([0.0, 100.0], dtype=np.float32),
        "max_angle": 1.5,
        "linear_cde": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }


def _opencv_dict(tag: float) -> dict:
    return {
        "resolution": np.array([6, 4], dtype=np.int64),
        "shutter_type": "GLOBAL",
        "principal_point": np.array([3.0, 2.0], dtype=np.float32),
        "focal_length": np.array([tag, tag], dtype=np.float32),
        "radial_coeffs": np.zeros(6, dtype=np.float32),
        "tangential_coeffs": np.zeros(2, dtype=np.float32),
        "thin_prism_coeffs": np.zeros(4, dtype=np.float32),
    }


def _entry(*, offset=0.0, ftheta=False, opencv_tag=None) -> dict:
    rays = None if opencv_tag is None else np.full((4, 6, 3), opencv_tag, dtype=np.float32)
    return {
        "c2w": _poses(offset),
        "timestamps_us": np.array([0, 1_000_000], dtype=np.int64),
        "resolution": (6, 4),
        "fov_y_rad": 1.0 + offset * 0.01,
        "ftheta_dict": _ftheta_dict() if ftheta else None,
        "opencv_pinhole_dict": _opencv_dict(opencv_tag) if opencv_tag is not None else None,
        "opencv_pinhole_rays": rays,
    }


def _bypass_viewer(mod):
    cls = mod.Viser4DViewer
    with mock.patch.object(cls, "__init__", autospec=True) as init_mock:
        init_mock.return_value = None
        viewer = cls(port=8080, engine=None, metadata=None)
    viewer._multi_cam_poses = {
        "wide": _entry(offset=0.0, opencv_tag=4.0),
        "fish_front": _entry(offset=20.0, ftheta=True),
        "tele": _entry(offset=40.0, opencv_tag=8.0),
        "fish_rear": _entry(offset=60.0, ftheta=True),
        "legacy": _entry(offset=80.0),
    }
    viewer._active_camera_state = None
    viewer._active_render_wh = None
    viewer.ftheta_intrinsics = None
    viewer.ftheta_render_wh = None
    viewer.opencv_pinhole_intrinsics = None
    viewer.opencv_pinhole_rays = None
    viewer.opencv_pinhole_render_wh = None
    viewer._overlay_compositor = None
    viewer._overlay_static_ego_polylines = []
    viewer._overlay_static_track_polylines = []
    viewer.h_cuboid_lines = None
    viewer.h_ego_traj = None
    viewer.h_track_trajectories = None
    viewer._cuboid_label_handles = {}
    viewer._current_dropdown_cam = None
    viewer._cam_dropdown = None
    viewer.need_update = False
    viewer.server = SimpleNamespace(get_clients=lambda: {})
    return viewer


def test_initial_cam_id_wins_over_metadata_primary(viewer_module):
    cam = viewer_module.Viser4DViewer._choose_initial_camera_id(
        ["wide", "fish_front"], initial_cam_id="fish_front", metadata_primary="wide"
    )
    assert cam == "fish_front"


def test_metadata_primary_is_fallback_when_initial_missing(viewer_module):
    cam = viewer_module.Viser4DViewer._choose_initial_camera_id(
        ["wide", "fish_front"], initial_cam_id="missing", metadata_primary="wide"
    )
    assert cam == "wide"


def test_mixed_switch_sequence_has_no_stale_projection_fields(viewer_module):
    viewer = _bypass_viewer(viewer_module)
    sequence = ["wide", "fish_front", "tele", "fish_rear", "wide"]
    expected = [
        (CameraModelKind.OPENCV_PINHOLE, False, True),
        (CameraModelKind.FTHETA, True, False),
        (CameraModelKind.OPENCV_PINHOLE, False, True),
        (CameraModelKind.FTHETA, True, False),
        (CameraModelKind.OPENCV_PINHOLE, False, True),
    ]
    for cam_id, (kind, has_ft, has_cv) in zip(sequence, expected):
        state = viewer._set_active_camera(cam_id, 500_000, snap_clients=False)
        assert state.model_kind is kind
        assert (viewer.ftheta_intrinsics is not None) is has_ft
        assert (viewer.opencv_pinhole_intrinsics is not None) is has_cv
        assert (viewer.opencv_pinhole_rays is not None) is has_cv
        assert viewer._current_dropdown_cam == cam_id

    np.testing.assert_array_equal(
        viewer.opencv_pinhole_rays, viewer._multi_cam_poses["wide"]["opencv_pinhole_rays"]
    )
    assert viewer.opencv_pinhole_intrinsics is viewer._multi_cam_poses["wide"]["opencv_pinhole_dict"]


def test_calibrated_camera_uses_native_render_resolution(viewer_module):
    viewer = _bypass_viewer(viewer_module)
    viewer._set_active_camera("wide", 500_000, snap_clients=False)
    assert viewer._active_render_wh == (6, 4)
    viewer._set_active_camera("fish_front", 500_000, snap_clients=False)
    assert viewer._active_render_wh == (6, 4)
    viewer._set_active_camera("legacy", 500_000, snap_clients=False)
    assert viewer._active_render_wh is None


def test_calibrated_camera_switch_builds_matching_overlay_projector(viewer_module):
    from threedgrut_playground.utils.ftheta_projector import FthetaForwardProjector
    from threedgrut_playground.utils.pinhole_projector import PinholeForwardProjector

    viewer = _bypass_viewer(viewer_module)
    viewer._set_active_camera("wide", 0, snap_clients=False)
    assert isinstance(viewer._overlay_compositor.projector, PinholeForwardProjector)
    viewer._set_active_camera("fish_front", 0, snap_clients=False)
    assert isinstance(viewer._overlay_compositor.projector, FthetaForwardProjector)
    viewer._set_active_camera("legacy", 0, snap_clients=False)
    assert viewer._overlay_compositor is None


def test_status_text_identifies_projection_and_pose_source(viewer_module):
    viewer = _bypass_viewer(viewer_module)
    state = viewer._set_active_camera("fish_front", 500_000, snap_clients=False)
    text = viewer._active_camera_status_text(state)
    assert "camera: fish_front" in text
    assert "model: FTheta" in text
    assert "render: 6×4" in text
    assert "interpolated" in text
    assert "overlay: FTheta image-space" in text


def test_status_text_warns_on_large_pose_gap(viewer_module):
    viewer = _bypass_viewer(viewer_module)
    viewer._multi_cam_poses["gappy"] = _entry(opencv_tag=4.0)
    viewer._multi_cam_poses["gappy"]["timestamps_us"] = np.array([0, 600_000], dtype=np.int64)
    state = viewer._set_active_camera("gappy", 300_000, snap_clients=False)
    text = viewer._active_camera_status_text(state)
    assert "WARNING" in text
    assert "600.0 ms" in text


def test_ego_frustum_uses_current_camera_state(viewer_module):
    viewer = _bypass_viewer(viewer_module)
    viewer.meta = SimpleNamespace(ego_pose_at=lambda _: np.eye(4, dtype=np.float32))
    viewer.h_ego_frustum = SimpleNamespace(wxyz=None, position=None)
    viewer._set_active_camera("fish_rear", 500_000, snap_clients=False)
    viewer._update_ego_frustum(500_000)
    np.testing.assert_allclose(viewer.h_ego_frustum.position, [65.0, 0.0, 0.0])


def test_follow_camera_midpoint_uses_interpolated_pose(viewer_module):
    viewer = _bypass_viewer(viewer_module)
    camera = SimpleNamespace(
        wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        position=np.array([-1.0, 0.0, 0.0]),
        look_at=np.zeros(3),
        up_direction=np.array([0.0, -1.0, 0.0]),
        fov=0.0,
    )
    viewer.server = SimpleNamespace(get_clients=lambda: {"c": SimpleNamespace(camera=camera)})
    viewer._set_active_camera("wide", 500_000, snap_clients=True)
    np.testing.assert_allclose(camera.position, [5.0, 0.0, 0.0], atol=1e-6)
    assert viewer._active_camera_state.pose_sample.interpolated is True


def test_hydrate_ego_rig_poses_uses_primary_camera_manifest_entry(viewer_module):
    meta = SimpleNamespace(
        ego_primary_camera_id="wide",
        ego_poses_c2w=np.zeros((2, 4, 4), dtype=np.float32),
        ego_rig_poses_c2w=None,
    )
    rig = np.tile(np.eye(4, dtype=np.float32)[None], (2, 1, 1))
    rig[:, 2, 3] = -2.3
    entries = {"wide": {"rig_poses_c2w": rig}}

    assert viewer_module._hydrate_ego_rig_poses(meta, entries) is True
    np.testing.assert_allclose(meta.ego_rig_poses_c2w, rig)


def test_hydrate_ego_rig_poses_rejects_wrong_length(viewer_module):
    meta = SimpleNamespace(
        ego_primary_camera_id="wide",
        ego_poses_c2w=np.zeros((2, 4, 4), dtype=np.float32),
        ego_rig_poses_c2w=None,
    )
    entries = {"wide": {"rig_poses_c2w": np.zeros((3, 4, 4), dtype=np.float32)}}
    assert viewer_module._hydrate_ego_rig_poses(meta, entries) is False
    assert meta.ego_rig_poses_c2w is None


def test_set_active_camera_rejects_unknown_camera(viewer_module):
    viewer = _bypass_viewer(viewer_module)
    with pytest.raises(KeyError, match="unknown camera"):
        viewer._set_active_camera("nope", 0, snap_clients=False)
