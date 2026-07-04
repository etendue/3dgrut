"""E2.1 frame-alignment key / path pure functions — unit tests.

These functions must match eval_frames_dir.resolve_pred_path's join-key
format (ts:<camera_id>:<timestamp_us>) and the per-camera subdir layout
(<camera_id>/<frame_idx:06d>.png) exactly so that Harmonizer offline fix
can locate rendered frames by timestamp.
"""

from threedgrut.utils.novel_view import novel_frame_key, novel_frame_relpath


def test_frame_key_matches_eval_frames_dir_format():
    # eval_frames_dir.resolve_pred_path builds: ts:<camera_id>:<timestamp_us>
    assert novel_frame_key("camera_front_wide", 1717000000123456) == "ts:camera_front_wide:1717000000123456"


def test_frame_key_casts_timestamp_to_int():
    assert novel_frame_key("cam_x", 100.0) == "ts:cam_x:100"


def test_frame_relpath_is_camera_subdir_zero_padded():
    assert novel_frame_relpath("camera_front_wide", 7) == "camera_front_wide/000007.png"
