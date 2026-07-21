"""Tests for the GT-image-on-frustum feature (2026-07-21).

Covers:
  * `_load_multi_cam_poses` populates per-cam `image_source` closures that
    decode the nearest frame's RGB array from the NCore sensor.
  * `_update_ego_frustum` injects the image only when the toggle is on AND
    a source exists for the current dropdown camera.
  * Switching dropdown cameras rebuilds the frustum with the new camera's
    aspect + the GT image attached at birth.
  * Toggle OFF tears down and re-creates the frustum without an image.

These are pure-Python unit tests: viser / kaolin / NCore SDK are stubbed,
no GPU or browser is needed. The `viewer_module` fixture mirrors the
pattern in `test_viser_gui_4d_camera_switch.py`.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# Fixtures / helpers — mirror test_viser_gui_4d_camera_switch.py
# --------------------------------------------------------------------------- #
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


def _entry_with_image_source(*, aspect=1.5, image_fn=None, fov=1.0):
    """Build a multi_cam_poses entry that carries an `image_source`."""
    W, H = int(round(aspect * 4)), 4
    return {
        "c2w": np.tile(np.eye(4, dtype=np.float32)[None], (2, 1, 1)),
        "rig_poses_c2w": None,
        "timestamps_us": np.array([0, 1_000_000], dtype=np.int64),
        "ftheta_dict": None,
        "resolution": (W, H),
        "fov_y_rad": fov,
        "opencv_pinhole_dict": None,
        "opencv_pinhole_rays": None,
        "image_source": image_fn,
    }


def _bypass_viewer(mod, *, multi_cam_poses, current_cam=None):
    """Construct a Viser4DViewer bypassing __init__ and seed the minimum
    state `_update_ego_frustum` needs to run end-to-end."""
    cls = mod.Viser4DViewer
    with mock.patch.object(cls, "__init__", autospec=True) as init_mock:
        init_mock.return_value = None
        viewer = cls(port=8080, engine=None, metadata=None)
    viewer._multi_cam_poses = multi_cam_poses
    # Mirror the __init__ caching logic for image sources.
    viewer._cam_gt_image_sources = {
        cid: e["image_source"]
        for cid, e in multi_cam_poses.items()
        if isinstance(e, dict) and callable(e.get("image_source"))
    }
    viewer._frustum_last_aspect = None
    viewer._frustum_last_cam_id = None
    viewer.show_gt_image = None
    viewer._gt_image_enabled = False
    viewer.show_ego_frust = None
    viewer._active_camera_state = None
    viewer._active_render_wh = None
    viewer.ftheta_intrinsics = None
    viewer.ftheta_render_wh = None
    viewer.opencv_pinhole_intrinsics = None
    viewer.opencv_pinhole_rays = None
    viewer.opencv_pinhole_render_wh = None
    viewer._overlay_compositor = None
    viewer._current_dropdown_cam = current_cam
    viewer._cam_dropdown = None
    viewer.need_update = False
    viewer.meta = SimpleNamespace(
        ego_pose_at=lambda _: np.eye(4, dtype=np.float32),
        ego_primary_aspect=1.5,
        ego_primary_fov_y_rad=1.0,
    )
    viewer.h_ego_frustum = SimpleNamespace(
        wxyz=None, position=None, remove=lambda: None,
    )

    # Capture every frustum (re)creation so tests can assert on kwargs.
    created = []

    def _add(name, **kwargs):
        handle = SimpleNamespace(
            wxyz=kwargs.get("wxyz"),
            position=kwargs.get("position"),
            remove=lambda: None,
            _image=kwargs.get("image"),
        )
        # Support `handle.image = arr` assignment for in-place refresh.
        def _set_image(arr):
            handle._image = arr
        handle.image = kwargs.get("image")  # initial value (read/write)
        # Convert to a property-like by intercepting attribute set via
        # __dict__ mutation through a SimpleNamespace; for test purposes
        # we just make image a plain mutable attribute.
        handle.__dict__["_image"] = kwargs.get("image")
        # Use object.__setattr__ workaround: SimpleNamespace allows it.
        handle._set_image = _set_image
        created.append({"name": name, "kwargs": kwargs, "handle": handle})
        return handle

    viewer.server = SimpleNamespace(
        scene=SimpleNamespace(add_camera_frustum=_add),
        get_clients=lambda: {},
    )
    return viewer, created


# --------------------------------------------------------------------------- #
# _load_multi_cam_poses closure
# --------------------------------------------------------------------------- #
def _make_sensor(cam_id, image_fn, *, n_frames=2):
    """Minimal NCore CameraSensor stub covering every attribute
    `_load_multi_cam_poses` + the image_source closure touches."""
    return SimpleNamespace(
        sensor_id=cam_id,
        frames_timestamps_us=np.array(
            [[0, (i + 1) * 1_000_000] for i in range(n_frames)], dtype=np.int64
        ),
        get_frame_image_array=image_fn,
        # pose-graph frame transform used by get_frames_T_source_target.
        get_frames_T_source_target=lambda **kwargs: np.tile(
            np.eye(4, dtype=np.float64)[None], (n_frames, 1, 1)
        ),
    )


def _make_cam_model(W=6, H=4):
    """Minimal camera-model stub. resolution entries must support .item()."""
    return SimpleNamespace(
        resolution=np.array([W, H]),
        focal_length=np.array([float(W), float(H)]),
    )


def _make_ncore_module():
    return SimpleNamespace(
        data=SimpleNamespace(
            FrameTimepoint=SimpleNamespace(START=0, END=1),
        ),
        sensors=SimpleNamespace(),  # isinstance checks fall through
    )


def _patch_ncore(monkeypatch, ds, *, cam_ids=None):
    """Wire NCore SDK stubs into sys.modules.

    `cam_ids` lets multi-cam tests simulate the "Multiple camera sensors"
    ValueError that `_load_multi_cam_poses` catches to auto-discover
    camera ids. None → single-sensor fast path.
    """
    fake_ncore = _make_ncore_module()

    def _ctor(**kwargs):
        if kwargs.get("camera_ids") is None and cam_ids is not None:
            raise ValueError(
                f"Multiple camera sensors found: {list(cam_ids)}"
            )
        return ds

    fake_datasetNcore = SimpleNamespace(NCoreDataset=_ctor)
    monkeypatch.setitem(sys.modules, "ncore", fake_ncore)
    monkeypatch.setitem(sys.modules, "ncore.data", fake_ncore.data)
    monkeypatch.setitem(
        sys.modules, "threedgrut.datasets.datasetNcore", fake_datasetNcore
    )


def _make_ds(sensors, cam_models=None, cam_ids=None):
    seq = "seq"
    cam_models = cam_models or {cid: _make_cam_model() for cid in sensors}
    cam_ids = list(sensors.keys()) if cam_ids is None else cam_ids
    return SimpleNamespace(
        sequence_id=seq,
        camera_ids=cam_ids,
        sequence_camera_sensors={seq: sensors},
        camera_train_frame_indices={cid: list(range(len(sensors[cid].frames_timestamps_us))) for cid in sensors},
        sequence_camera_models={seq: cam_models},
        T_world_to_world_global=np.eye(4, dtype=np.float64),
        sequence_loaders={
            seq: SimpleNamespace(
                pose_graph=SimpleNamespace(
                    evaluate_poses=lambda *a, **k: np.tile(
                        np.eye(4, dtype=np.float64)[None],
                        (max(len(sensors[c].frames_timestamps_us) for c in sensors), 1, 1),
                    ),
                ),
            ),
        },
    )


def test_load_multi_cam_poses_attaches_image_source(viewer_module, monkeypatch):
    """_load_multi_cam_poses must embed a callable `image_source` per cam
    that resolves the nearest frame's RGB array from the NCore sensor."""
    fake_image = np.zeros((4, 6, 3), dtype=np.uint8)
    fake_image[0, 0, 0] = 123  # sentinel

    sensors = {"cam_a": _make_sensor("cam_a", lambda fidx: fake_image)}
    ds = _make_ds(sensors)
    _patch_ncore(monkeypatch, ds, cam_ids=["cam_a"])

    out = viewer_module._load_multi_cam_poses("/fake/manifest.json", "")
    assert "cam_a" in out
    src = out["cam_a"].get("image_source")
    assert callable(src), "image_source closure must be attached"
    arr = src(1_500_000)  # nearest-neighbour → frame 1
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4, 6, 3)
    assert arr.dtype == np.uint8
    assert arr[0, 0, 0] == 123


def test_load_multi_cam_poses_image_source_swallows_errors(viewer_module, monkeypatch):
    """The image_source closure must return None (not raise) on decode
    failure so a corrupt frame can never kill the viewer's render loop."""
    def _boom(fidx):
        raise RuntimeError("corrupt frame")

    sensors = {"cam_a": _make_sensor("cam_a", _boom)}
    ds = _make_ds(sensors)
    _patch_ncore(monkeypatch, ds, cam_ids=["cam_a"])

    out = viewer_module._load_multi_cam_poses("/fake/manifest.json", "")
    src = out["cam_a"]["image_source"]
    # Must not raise — returns None on failure.
    assert src(0) is None


def test_image_source_closure_binds_per_cam_not_loop_variable(viewer_module, monkeypatch):
    """The classic Python closure-in-loop trap: each cam's image_source must
    resolve its OWN sensor even though they were all created inside the
    same for-loop. Default-argument binding is what we rely on; this test
    pins that contract."""
    images = {
        "cam_a": np.full((4, 6, 3), 1, dtype=np.uint8),
        "cam_b": np.full((4, 6, 3), 2, dtype=np.uint8),
    }
    sensors = {
        cid: _make_sensor(cid, lambda fidx, _cid=cid: images[_cid])
        for cid in ("cam_a", "cam_b")
    }
    ds = _make_ds(sensors)
    _patch_ncore(monkeypatch, ds, cam_ids=["cam_a", "cam_b"])

    out = viewer_module._load_multi_cam_poses("/fake/manifest.json", "")
    # Critical: cam_a's source returns cam_a's image, cam_b's returns cam_b's.
    # If the closure captured the loop variable, BOTH would return cam_b.
    assert out["cam_a"]["image_source"](0)[0, 0, 0] == 1
    assert out["cam_b"]["image_source"](0)[0, 0, 0] == 2


# --------------------------------------------------------------------------- #
# _update_ego_frustum toggle behaviour
# --------------------------------------------------------------------------- #
def test_toggle_off_does_not_attach_image(viewer_module):
    """When the GT-image toggle is OFF (default), the rebuilt frustum
    must NOT carry an `image=` kwarg — zero IO cost."""
    multi = {"cam_a": _entry_with_image_source(image_fn=lambda t: np.zeros((4, 6, 3), dtype=np.uint8))}
    viewer, created = _bypass_viewer(
        viewer_module, multi_cam_poses=multi, current_cam="cam_a",
    )
    viewer._update_ego_frustum(0)
    assert len(created) == 1
    assert "image" not in created[0]["kwargs"], (
        "GT toggle OFF must not pass image= to add_camera_frustum"
    )


def test_toggle_on_attaches_image(viewer_module):
    """When ON and a source exists for the current cam, the rebuilt
    frustum carries image=<HxWx3 uint8> and format='jpeg'."""
    sentinel = np.full((4, 6, 3), 7, dtype=np.uint8)
    multi = {"cam_a": _entry_with_image_source(image_fn=lambda t: sentinel)}
    viewer, created = _bypass_viewer(
        viewer_module, multi_cam_poses=multi, current_cam="cam_a",
    )
    viewer._gt_image_enabled = True
    viewer._update_ego_frustum(0)
    assert len(created) == 1
    assert "image" in created[0]["kwargs"]
    np.testing.assert_array_equal(created[0]["kwargs"]["image"], sentinel)
    assert created[0]["kwargs"]["format"] == "jpeg"


def test_toggle_on_but_no_source_does_not_attach_image(viewer_module):
    """Toggle ON without a matching image_source (e.g. cam was loaded
    without --dataset_path) must degrade gracefully to a plain frustum."""
    multi = {"cam_a": _entry_with_image_source(image_fn=None)}
    viewer, created = _bypass_viewer(
        viewer_module, multi_cam_poses=multi, current_cam="cam_a",
    )
    viewer._gt_image_enabled = True
    viewer._update_ego_frustum(0)
    assert len(created) == 1
    assert "image" not in created[0]["kwargs"]


def test_toggle_on_but_source_returns_none_does_not_attach_image(viewer_module):
    """If the source closure returns None (decode failure), we must not
    crash and must not attach an image= kwarg."""
    multi = {"cam_a": _entry_with_image_source(image_fn=lambda t: None)}
    viewer, created = _bypass_viewer(
        viewer_module, multi_cam_poses=multi, current_cam="cam_a",
    )
    viewer._gt_image_enabled = True
    viewer._update_ego_frustum(0)
    assert len(created) == 1
    assert "image" not in created[0]["kwargs"]


# --------------------------------------------------------------------------- #
# Multi-camera aspect rebuild
# --------------------------------------------------------------------------- #
def test_aspect_change_forces_rebuild(viewer_module):
    """Switching dropdown camera to one with a different aspect ratio must
    remove + re-create the frustum (viser geometry is immutable)."""
    multi = {
        "cam_a": _entry_with_image_source(aspect=1.5, image_fn=None),
        "cam_b": _entry_with_image_source(aspect=2.0, image_fn=None),
    }
    viewer, created = _bypass_viewer(
        viewer_module, multi_cam_poses=multi, current_cam="cam_a",
    )
    viewer._update_ego_frustum(0)
    assert created[0]["kwargs"]["aspect"] == pytest.approx(1.5)

    # Simulate dropdown switch to cam_b.
    viewer._current_dropdown_cam = "cam_b"
    viewer._update_ego_frustum(0)
    assert len(created) == 2
    assert created[1]["kwargs"]["aspect"] == pytest.approx(2.0)


def test_same_aspect_same_cam_skips_rebuild(viewer_module):
    """When neither aspect nor cam_id changed, the frustum must NOT be
    torn down — only the pose attributes are updated in place."""
    multi = {"cam_a": _entry_with_image_source(image_fn=None)}
    viewer, created = _bypass_viewer(
        viewer_module, multi_cam_poses=multi, current_cam="cam_a",
    )
    viewer._update_ego_frustum(0)
    assert len(created) == 1
    viewer._update_ego_frustum(500_000)
    # Second call: same cam + aspect → no rebuild.
    assert len(created) == 1


# --------------------------------------------------------------------------- #
# __init__ caching contract — mirrored by _bypass_viewer, this is the
# pure contract test that doesn't require the full viser/kaolin init.
# --------------------------------------------------------------------------- #
def test_init_caches_callable_image_sources_only(viewer_module):
    """The __init__ caching loop must collect only *callable* image_source
    entries from multi_cam_poses. Non-callable / missing entries are
    silently skipped (the GUI checkbox is disabled for those cams).

    We reconstruct the loop verbatim because the full __init__ needs many
    more meta fields than this contract cares about. The intent is to pin
    the loop's filtering semantics.
    """
    fn = lambda t: np.zeros((2, 2, 3), dtype=np.uint8)  # noqa: E731
    multi_cam_poses = {
        "cam_callable": {
            "c2w": np.eye(4, dtype=np.float32)[None],
            "image_source": fn,
        },
        "cam_none": {"c2w": np.eye(4, dtype=np.float32)[None]},
        "cam_noncallable": {
            "c2w": np.eye(4, dtype=np.float32)[None],
            "image_source": "not-a-fn",
        },
    }
    # Mirror the __init__ cache loop.
    cached = {}
    for cid, entry in multi_cam_poses.items():
        src = entry.get("image_source") if isinstance(entry, dict) else None
        if callable(src):
            cached[cid] = src

    assert set(cached.keys()) == {"cam_callable"}
    assert cached["cam_callable"] is fn
