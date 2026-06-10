# SPDX-License-Identifier: Apache-2.0
"""BUG-1 regression — active cuboid wireframes must ride the FTheta overlay
path so they stay pixel-aligned with the Gaussian backdrop.

Root cause (v3_plan_revised.md § 2.5): the backdrop is rendered through the
FTheta fisheye polynomial (engine.render_pass(fisheye_intrinsics=...)) while
``add_line_segments`` cuboids are projected by the browser's pinhole camera.
The two projections only agree at the optical axis; at NCore wide-FoV the
wireframe visibly detaches from the Gaussian vehicle it should hug. Commit
bea5508 (V3-VIZ.2) moved cuboids from the pixel-aligned B2 overlay onto 3D
scene primitives and under-estimated that drift.

Contract pinned here:
  - FTheta mode + overlay ON  → cuboids come back as overlay PolylineLayerSpec
    (per-track instance color), and the pinhole line_segments node is skipped.
  - FTheta mode + overlay OFF → legacy line_segments fallback (A/B compare).
  - Pinhole ckpts (no compositor) → line_segments path untouched.

Heavy deps (viser/kaolin/engine) are stubbed; mirrors the bypass-__init__
strategy of test_viser_gui_4d_follow_ego.py.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from threedgrut_playground.utils.cuboid import cuboid_world_edges, instance_color


@pytest.fixture
def viser_gui_4d(monkeypatch):
    """Stub heavy deps and import the module under test."""
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
    from threedgrut_playground import viser_gui_4d as mod
    return mod


TRACK_ID = "7"
TRACK_SIZE = np.array([4.0, 2.0, 1.5], dtype=np.float32)


def _make_meta_with_track():
    """3 ego frames; one automobile track active at frames 0 and 1."""
    from threedgrut_playground.utils.viz4d_metadata import FourDMetadata
    ego = np.tile(np.eye(4, dtype=np.float32)[None, ...], (3, 1, 1))
    ego[0, :3, 3] = [0.0, 0.0, 0.0]
    ego[1, :3, 3] = [10.0, 0.0, 0.0]
    ego[2, :3, 3] = [20.0, 0.0, 0.0]
    ts = np.array([0, 1_000_000, 2_000_000], dtype=np.int64)

    track_poses = np.tile(np.eye(4, dtype=np.float32)[None, ...], (3, 1, 1))
    track_poses[0, :3, 3] = [5.0, 0.0, 0.0]
    track_poses[1, :3, 3] = [6.0, 0.0, 0.0]
    track_poses[2, :3, 3] = [7.0, 0.0, 0.0]
    tracks = {
        TRACK_ID: {
            "poses": track_poses,
            "size": TRACK_SIZE.copy(),
            "frame_info": np.array([True, True, False]),
            "class": "automobile",
        }
    }
    return FourDMetadata(
        schema_version=2, sequence_id="test",
        ego_poses_c2w=ego,
        ego_frame_timestamps_us=ts,
        ego_primary_camera_id="primary",
        ego_primary_fov_y_rad=1.0, ego_primary_aspect=1.78,
        ego_primary_intrinsics_ftheta=None,
        ego_primary_resolution=None,
        tracks=tracks, tracks_camera_timestamps_us=ts,
        road_xyz=None, road_rgb=None, dyn_xyz=None, dyn_rgb=None,
        road_n_total=None, dyn_n_total=None,
        dyn_local_xyz=None, dyn_track_ids=None, dyn_track_names=None,
        initial_c2w=ego[0], t_us_first=0, t_us_last=2_000_000,
    )


def _fake_handle():
    return SimpleNamespace(visible=True, remove=mock.MagicMock())


def _bypass_viewer(mod, *, ftheta: bool, show_cuboids: bool = True,
                   show_ego_traj: bool = True, show_tracks: bool = True):
    """Viser4DViewer without __init__ side effects; only fields the cuboid
    and trajectory paths touch are populated."""
    meta = _make_meta_with_track()
    cls = mod.Viser4DViewer
    with mock.patch.object(cls, "__init__", autospec=True) as init_mock:
        init_mock.return_value = None
        viewer = cls(port=8080, engine=None, metadata=meta)
    viewer.meta = meta
    viewer._overlay_compositor = object() if ftheta else None
    viewer.show_cuboids = SimpleNamespace(value=show_cuboids)
    viewer.show_ego_traj = SimpleNamespace(value=show_ego_traj)
    viewer.show_tracks = SimpleNamespace(value=show_tracks)
    viewer.h_cuboid_lines = None
    viewer.h_ego_traj = None
    viewer.h_ego_frustum = None
    viewer.h_track_trajectories = None
    viewer._cuboid_label_handles = {}
    # Static overlay caches normally built in __init__ (BUG-1c): ego polyline
    # + per-track (class, centers) — only populated in FTheta mode.
    if ftheta:
        viewer._overlay_static_ego_polylines = [
            meta.ego_poses_c2w[:, :3, 3].astype(np.float64)]
        t = meta.tracks[TRACK_ID]
        centers = t["poses"][t["frame_info"], :3, 3].astype(np.float64)
        viewer._overlay_static_track_polylines = [("automobile", centers)]
    else:
        viewer._overlay_static_ego_polylines = []
        viewer._overlay_static_track_polylines = []
    scene = SimpleNamespace(
        add_line_segments=mock.MagicMock(side_effect=lambda *a, **k: _fake_handle()),
        add_label=mock.MagicMock(side_effect=lambda *a, **k: _fake_handle()),
        add_camera_frustum=mock.MagicMock(side_effect=lambda *a, **k: _fake_handle()),
    )
    viewer.server = SimpleNamespace(scene=scene)
    return viewer


# ============================================================ overlay specs
def test_overlay_specs_include_active_cuboids_in_ftheta_mode(viser_gui_4d):
    """FTheta mode + Active-cuboids ON → _collect_overlay_layer_specs must
    emit the active cuboid edges so the compositor draws them through the
    SAME FTheta polynomial as the backdrop (the actual BUG-1 fix)."""
    viewer = _bypass_viewer(viser_gui_4d, ftheta=True)
    specs = viewer._collect_overlay_layer_specs(t_us=0)
    cuboid_specs = [s for s in specs if "active_cuboids" in s.name]
    assert cuboid_specs, (
        "no active_cuboids overlay layer emitted — cuboid wireframes are "
        "left to the pinhole line_segments path and drift off the FTheta "
        "backdrop (BUG-1)"
    )

    # Exactly one active track at frame 0 → 12 edges in world frame, matching
    # cuboid_world_edges(pose@frame0, size) bit-for-bit.
    all_polylines = [pl for s in cuboid_specs for pl in s.polylines_world]
    assert len(all_polylines) == 12
    expected = cuboid_world_edges(
        viewer.meta.tracks[TRACK_ID]["poses"][0], TRACK_SIZE)
    got = np.stack([np.asarray(pl) for pl in all_polylines])  # (12, 2, 3)
    np.testing.assert_allclose(got, expected.astype(np.float64), atol=1e-6)


def test_overlay_specs_use_per_track_instance_color(viser_gui_4d):
    """Wireframe color must match the 3D-primitive path's instance_color so
    the overlay looks identical to what users saw before (not plain green)."""
    viewer = _bypass_viewer(viser_gui_4d, ftheta=True)
    specs = viewer._collect_overlay_layer_specs(t_us=0)
    cuboid_specs = [s for s in specs if "active_cuboids" in s.name]
    assert cuboid_specs
    expected_rgb = tuple(
        int(round(c * 255)) for c in instance_color(TRACK_ID))
    for s in cuboid_specs:
        assert tuple(s.color[:3]) == expected_rgb
        assert s.color[3] > 0  # opaque-ish alpha


def test_overlay_specs_respect_show_cuboids_off(viser_gui_4d):
    """Unchecking Active cuboids must also hide the overlay wireframes."""
    viewer = _bypass_viewer(viser_gui_4d, ftheta=True, show_cuboids=False)
    specs = viewer._collect_overlay_layer_specs(t_us=0)
    assert not [s for s in specs if "active_cuboids" in s.name]


def test_overlay_specs_empty_when_no_active_tracks(viser_gui_4d):
    """Frame 2 has no active tracks → no cuboid layer, no crash."""
    viewer = _bypass_viewer(viser_gui_4d, ftheta=True)
    specs = viewer._collect_overlay_layer_specs(t_us=2_000_000)
    assert not [s for s in specs if "active_cuboids" in s.name]


# ======================================================= BUG-1b: labels
def test_overlay_specs_include_track_labels(viser_gui_4d):
    """Each cuboid overlay layer must carry its 't<tid> | <class>' label,
    anchored at the cuboid's top-back-right corner (vertex 7) — the same
    anchor the 3D label path used — so the compositor projects text through
    the SAME FTheta polynomial as the wireframe (BUG-1b)."""
    viewer = _bypass_viewer(viser_gui_4d, ftheta=True)
    specs = viewer._collect_overlay_layer_specs(t_us=0)
    cuboid_specs = [s for s in specs if "active_cuboids" in s.name]
    assert cuboid_specs
    labels = [lab for s in cuboid_specs for lab in s.labels_world]
    assert len(labels) == 1
    anchor, text = labels[0]
    assert text == f"t{TRACK_ID} | automobile"
    pose = viewer.meta.tracks[TRACK_ID]["poses"][0]
    expected = pose[:3, :3] @ (TRACK_SIZE * 0.5) + pose[:3, 3]
    np.testing.assert_allclose(np.asarray(anchor), expected, atol=1e-5)


def test_update_active_cuboids_removes_3d_labels_when_overlay_active(viser_gui_4d):
    """FTheta mode → browser-side 3D labels must be REMOVED (the overlay text
    path replaces them); otherwise the pinhole-projected 3D label drifts away
    from the FTheta-aligned wireframe it belongs to."""
    viewer = _bypass_viewer(viser_gui_4d, ftheta=True)
    stale_label = _fake_handle()
    viewer._cuboid_label_handles[TRACK_ID] = stale_label

    viewer._update_active_cuboids(frame_idx=0)

    stale_label.remove.assert_called_once()
    assert viewer._cuboid_label_handles == {}
    viewer.server.scene.add_label.assert_not_called()


def test_pinhole_mode_keeps_3d_labels(viser_gui_4d):
    """Pinhole ckpts (no compositor) → legacy 3D label path keeps working."""
    viewer = _bypass_viewer(viser_gui_4d, ftheta=False)
    viewer._update_active_cuboids(frame_idx=0)
    assert viewer.server.scene.add_label.called


# ================================================ BUG-1c: trajectories
def test_overlay_specs_include_ego_trajectory(viser_gui_4d):
    """FTheta mode → ego trajectory rides the overlay (same FTheta projection
    as the backdrop) instead of the pinhole line_segments primitive."""
    viewer = _bypass_viewer(viser_gui_4d, ftheta=True)
    specs = viewer._collect_overlay_layer_specs(t_us=0)
    ego = [s for s in specs if s.name == "ego_trajectory"]
    assert ego, "ego trajectory missing from overlay specs (BUG-1c)"
    np.testing.assert_allclose(
        np.asarray(ego[0].polylines_world[0]),
        viewer.meta.ego_poses_c2w[:, :3, 3].astype(np.float64))
    # Dense multi-vertex polyline → low subdivide (B2 perf rationale).
    assert ego[0].subdivide_n <= 5


def test_overlay_specs_include_track_trajectories(viser_gui_4d):
    """FTheta mode → per-class track trajectory layers, colored like the 3D
    path (class_color) so the visuals don't change, only the projection."""
    from threedgrut_playground.utils.cuboid import class_color
    viewer = _bypass_viewer(viser_gui_4d, ftheta=True)
    specs = viewer._collect_overlay_layer_specs(t_us=0)
    tr = [s for s in specs if s.name.startswith("track_trajectories")]
    assert tr, "track trajectories missing from overlay specs (BUG-1c)"
    expected_rgb = tuple(int(round(c * 255)) for c in class_color("automobile"))
    assert tuple(tr[0].color[:3]) == expected_rgb


def test_overlay_specs_respect_trajectory_toggles(viser_gui_4d):
    """Unchecking Ego trajectory / Track trajectories hides their overlay
    layers (content toggles keep working with the overlay path)."""
    viewer = _bypass_viewer(viser_gui_4d, ftheta=True,
                            show_ego_traj=False, show_tracks=False)
    specs = viewer._collect_overlay_layer_specs(t_us=0)
    assert not [s for s in specs if s.name == "ego_trajectory"]
    assert not [s for s in specs if s.name.startswith("track_trajectories")]


def test_add_ego_trajectory_skips_polyline_in_ftheta(viser_gui_4d):
    """FTheta mode → the pinhole trajectory line_segments must NOT be created
    (overlay draws it); the ego frustum (a 3D position widget, not a
    backdrop annotation) is still created."""
    viewer = _bypass_viewer(viser_gui_4d, ftheta=True)
    viewer._add_ego_trajectory()
    viewer.server.scene.add_line_segments.assert_not_called()
    assert viewer.h_ego_traj is None
    assert viewer.server.scene.add_camera_frustum.called


def test_add_track_trajectories_skips_in_ftheta(viser_gui_4d):
    viewer = _bypass_viewer(viser_gui_4d, ftheta=True)
    viewer._add_track_trajectories()
    viewer.server.scene.add_line_segments.assert_not_called()
    assert viewer.h_track_trajectories is None


def test_add_trajectories_pinhole_unchanged(viser_gui_4d):
    """Pinhole ckpts keep the 3D primitive trajectories (zero regression)."""
    viewer = _bypass_viewer(viser_gui_4d, ftheta=False)
    viewer._add_ego_trajectory()
    viewer._add_track_trajectories()
    assert viewer.h_ego_traj is not None
    assert viewer.h_track_trajectories is not None


# ===================================================== line_segments gating
def test_update_active_cuboids_skips_line_segments_when_overlay_active(viser_gui_4d):
    """FTheta mode → the pinhole-projected line_segments node must NOT be
    (re)created, and any stale node must be removed, so the browser never
    double-draws a drifting wireframe on top of the aligned overlay. The
    overlay is the ONLY cuboid path in FTheta mode (the legacy toggle that
    re-enabled the misaligned pinhole path was removed)."""
    viewer = _bypass_viewer(viser_gui_4d, ftheta=True)
    stale = _fake_handle()
    viewer.h_cuboid_lines = stale

    viewer._update_active_cuboids(frame_idx=0)

    viewer.server.scene.add_line_segments.assert_not_called()
    stale.remove.assert_called_once()
    assert viewer.h_cuboid_lines is None
    # BUG-1b: labels ride the overlay text path now — no 3D labels created.
    viewer.server.scene.add_label.assert_not_called()


def test_update_active_cuboids_pinhole_mode_unchanged(viser_gui_4d):
    """No FTheta compositor (pinhole ckpt) → line_segments path untouched
    (zero-regression contract for non-NCore checkpoints)."""
    viewer = _bypass_viewer(viser_gui_4d, ftheta=False)
    viewer._update_active_cuboids(frame_idx=0)
    assert viewer.server.scene.add_line_segments.called
    assert viewer.h_cuboid_lines is not None


# ============================================== camera-consistency contract
def _synthetic_ftheta():
    """Linear angle→pixel polynomial, principal point at image center."""
    return {
        "resolution":              np.array([1920, 1080], dtype=np.int64),
        "shutter_type":            "ROLLING_TOP_TO_BOTTOM",
        "principal_point":         np.array([960.0, 540.0], dtype=np.float32),
        "reference_poly":          "ANGLE_TO_PIXELDIST",
        "angle_to_pixeldist_poly": np.array(
            [0.0, 800.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "pixeldist_to_angle_poly": np.array(
            [0.0, 1.0 / 800.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "max_angle":               np.pi / 2 - 0.01,
        "linear_cde":              np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }


def test_viewer_overlay_camera_matches_renderer_camera():
    """BUG-1 root-cause pin: the viewer-config compositor (flip=identity)
    must project a point on the BACKDROP's viewing axis to the principal
    point. The backdrop's viewing direction is the c2w +Z column — FTheta
    raygen produces rz=cos(theta)>0 rays (ftheta_pixels_to_camera_rays), so
    the image center looks down +Z. The legacy FLIP_VISER_TO_OPENCV default
    aimed the overlay at -Z (180° away): the on-axis point was INVISIBLE and
    wireframes were mirror-projections of the tracks behind the ego (the
    user-reported misalignment, mis-calibrated as 'aligned' in the B2 probe
    because streets are fore-aft symmetric)."""
    from threedgrut_playground.utils.viser_overlay_compositor import (
        Viser4DOverlayCompositor,
    )
    ft = _synthetic_ftheta()
    cmp_viewer = Viser4DOverlayCompositor(
        ft, height=1080, width=1920, world_to_camera_flip=np.eye(4))
    on_axis = np.array([[0.0, 0.0, 10.0]])          # 10 m down +Z (forward)
    uv, vis = cmp_viewer.projector.project_points(on_axis, np.eye(4))
    assert bool(vis[0]), "forward on-axis point must be visible"
    np.testing.assert_allclose(uv[0], [960.0, 540.0], atol=1.0)

    # And the legacy default must NOT see it (regression tripwire: if someone
    # reverts the viewer to the default flip, the contract above breaks).
    cmp_legacy = Viser4DOverlayCompositor(ft, height=1080, width=1920)
    _, vis_legacy = cmp_legacy.projector.project_points(on_axis, np.eye(4))
    assert not bool(vis_legacy[0])
