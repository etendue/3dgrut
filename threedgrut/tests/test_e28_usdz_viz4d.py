# SPDX-License-Identifier: Apache-2.0
"""E2.8 Task 5 — USDZ→viz4d wiring pure functions (synthetic, no GPU/USDZ).

Covers the ported fervent-knuth rig/viz4d logic + the cuboid→track_id remap that
``convert_usdz_to_ckpt_with_tracks`` relies on. The GPU orchestrator itself is
validated end-to-end on inceptio with the real USDZ (driver run).
"""
import numpy as np
import pytest

from threedgrut_playground.utils.nre_usdz_loader import TrackRaw
from threedgrut_playground.utils.nre_usdz_viz4d import (
    RigInfo,
    build_viz4d_dict,
    cuboid_ids_to_track_ids,
    parse_rig_trajectories,
    resolve_primary_cam,
    short_cam_id,
)


def _synth_rt():
    """Two-frame rig with one pinhole cam; (F,2,4,4) exposure pairs + (F,2) ts."""
    eye = np.eye(4).tolist()

    def pose(tx):
        m = np.eye(4)
        m[0, 3] = tx
        return m.tolist()

    # frame 0: start tx=0.0, end tx=1.0 ; frame 1: start tx=2.0, end tx=3.0
    rig_pairs = [[pose(0.0), pose(1.0)], [pose(2.0), pose(3.0)]]
    key = "camera_front_wide_120fov@clipgt-uuid"
    return {
        "world_to_nre": {"matrix": eye},
        "rig_trajectories": [{
            "sequence_id": "seqZ",
            "cameras_frame_T_rig_worlds": {key: rig_pairs},
            "cameras_frame_timestamps_us": {key: [[10, 20], [30, 40]]},
            "T_rig_worlds": [],
            "T_rig_world_timestamps_us": [],
        }],
        "camera_calibrations": {key: {
            "T_sensor_rig": eye,
            "logical_sensor_name": "camera_front_wide_120fov",
            "camera_model": {"type": "pinhole",
                             "parameters": {"resolution": [1920, 1080]}},
        }},
    }


def test_short_cam_id_strips_suffix():
    assert short_cam_id("camera_front_wide_120fov@clipgt-uuid") == "camera_front_wide_120fov"
    assert short_cam_id("plain") == "plain"


def test_parse_rig_takes_exposure_end_pose_and_ts():
    rig = parse_rig_trajectories(_synth_rt())
    assert "camera_front_wide_120fov" in rig.cams        # keyed by short logical name
    cam = rig.cams["camera_front_wide_120fov"]
    assert cam["c2w"].shape == (2, 4, 4)
    # END poses: frame0 tx=1.0, frame1 tx=3.0 (start poses 0.0/2.0 discarded)
    assert cam["c2w"][0][0, 3] == pytest.approx(1.0)
    assert cam["c2w"][1][0, 3] == pytest.approx(3.0)
    # END timestamps
    assert cam["timestamps_us"].tolist() == [20, 40]
    assert rig.world_to_nre is not None


def test_resolve_primary_cam_fallback():
    rig = RigInfo("s", {"camA": {}, "camB": {}}, np.zeros((0, 4, 4)), np.zeros(0))
    assert resolve_primary_cam(rig, "camA") == "camA"
    assert resolve_primary_cam(rig, "missing") == "camA"  # first sorted


def test_cuboid_ids_to_track_ids_remap():
    # track_order declares cuboid idx -> tid; sorted_tids = ["3","9"]
    track_ids, sorted_tids = cuboid_ids_to_track_ids(
        np.array([0, 1, 2, 0]), track_order=["9", "3", "9"]
    )
    assert sorted_tids == ["3", "9"]
    # cid_to_sorted: idx0 "9"->1, idx1 "3"->0, idx2 "9"->1
    assert track_ids.tolist() == [1, 0, 1, 1]


def test_cuboid_ids_out_of_range_raises():
    with pytest.raises(IndexError):
        cuboid_ids_to_track_ids(np.array([0, 5]), track_order=["9", "3"])


def test_build_viz4d_dict_timeline_and_tracks():
    rig = parse_rig_trajectories(_synth_rt())
    tracks = [TrackRaw(
        tid="7",
        poses7=np.array([[0, 0, 0, 0, 0, 0, 1.0],
                         [5, 0, 0, 0, 0, 0, 1.0]], dtype=np.float64),
        ts_us=np.array([20, 40], dtype=np.int64),
        label_class="automobile",
        dims=np.array([4.5, 1.8, 1.5], dtype=np.float32),
    )]
    viz = build_viz4d_dict(rig, tracks, primary_cam="camera_front_wide_120fov")
    assert viz["schema_version"] == 2
    assert viz["tracks_camera_timestamps_us"].tolist() == [20, 40]
    assert set(viz["tracks"].keys()) == {"7"}
    t = viz["tracks"]["7"]
    assert t["poses"].shape == (2, 4, 4)          # resampled onto 2-frame timeline
    assert t["class"] == "automobile"
    assert tuple(t["size"]) == pytest.approx((4.5, 1.8, 1.5))
    assert viz["ego"]["poses_c2w"].shape == (2, 4, 4)
    assert viz["ego"]["primary_camera_id"] == "camera_front_wide_120fov"
