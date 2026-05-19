# SPDX-License-Identifier: Apache-2.0
"""T4.1.a unit tests for tracks_loader.load_tracks_from_manifest.

Pure stdlib + torch; runs on Mac CPU without NCore SDK.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from threedgrut.datasets.tracks_loader import load_tracks_from_manifest


def _write_manifest(payload: dict) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(payload, f)
    f.close()
    return f.name


def _make_track(id_: str, F: int = 20, extent: list = None,
                class_: str = "vehicle") -> dict:
    eye = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    return {
        "id": id_,
        "poses": [eye] * F,
        "extent": extent or [4.5, 2.0, 1.7],
        "active_frames": [True] * F,
        "class": class_,
    }


def test_load_tracks_basic_schema():
    """T4.1.a: load 1 vehicle track → instance_pts_dict schema correct."""
    manifest = {"tracks": [_make_track("veh_01", F=20)]}
    path = _write_manifest(manifest)
    d = load_tracks_from_manifest(path)

    assert set(d.keys()) == {"veh_01"}
    trk = d["veh_01"]
    assert trk["pts"] is None
    assert trk["colors"] is None
    assert trk["poses"].shape == (20, 4, 4)
    assert trk["poses"].dtype == torch.float32
    assert trk["size"].tolist() == pytest.approx([4.5, 2.0, 1.7], abs=1e-5)
    assert trk["frame_info"].dtype == torch.bool
    assert trk["frame_info"].shape == (20,)
    assert trk["class"] == "vehicle"


def test_load_tracks_multiple():
    """T4.1.a: 3 tracks (vehicle + pedestrian + truck) all loaded preserving keys."""
    manifest = {"tracks": [
        _make_track("veh_01", F=20, class_="vehicle"),
        _make_track("ped_07", F=20, extent=[0.5, 0.5, 1.8], class_="pedestrian"),
        _make_track("trk_03", F=20, extent=[8.0, 2.5, 3.5], class_="truck"),
    ]}
    d = load_tracks_from_manifest(_write_manifest(manifest))
    assert set(d.keys()) == {"veh_01", "ped_07", "trk_03"}
    assert d["ped_07"]["size"].tolist() == pytest.approx([0.5, 0.5, 1.8], abs=1e-5)
    assert d["trk_03"]["class"] == "truck"


def test_load_tracks_missing_tracks_key_returns_empty():
    """T4.1.a: manifest without 'tracks' key (T3a.2 verified NCore current state)
    → empty dict, NOT a crash. trainer.init_model handles by skipping dyn layer."""
    manifest = {"sequence_id": "abc", "component_stores": []}
    d = load_tracks_from_manifest(_write_manifest(manifest))
    assert d == {}


def test_load_tracks_empty_array():
    """T4.1.a: 'tracks': [] → empty dict."""
    d = load_tracks_from_manifest(_write_manifest({"tracks": []}))
    assert d == {}


def test_load_tracks_active_frames_subset():
    """T4.1.a: active_frames can be partial (track inactive in some frames)."""
    track = _make_track("veh_01", F=10)
    track["active_frames"] = [False, False, True, True, True, True, True, False, False, False]
    d = load_tracks_from_manifest(_write_manifest({"tracks": [track]}))
    assert d["veh_01"]["frame_info"].tolist() == track["active_frames"]
    assert d["veh_01"]["frame_info"].sum().item() == 5


def test_load_tracks_missing_id_raises():
    """T4.1.a: 缺 id 字段 → ValueError 含 'missing'."""
    bad = _make_track("placeholder")
    del bad["id"]
    with pytest.raises(ValueError, match="missing 'id'"):
        load_tracks_from_manifest(_write_manifest({"tracks": [bad]}))


def test_load_tracks_missing_required_field_raises():
    """T4.1.a: 缺 poses / extent / active_frames → ValueError pin 字段名."""
    for missing in ("poses", "extent", "active_frames"):
        bad = _make_track("veh_01")
        del bad[missing]
        with pytest.raises(ValueError, match=f"missing required field '{missing}'"):
            load_tracks_from_manifest(_write_manifest({"tracks": [bad]}))


def test_load_tracks_poses_shape_validation():
    """T4.1.a: poses 不是 [F, 4, 4] 报清晰错误."""
    bad = _make_track("veh_01")
    bad["poses"] = [[1, 2, 3]]  # not 4x4
    with pytest.raises(ValueError, match="poses shape invalid"):
        load_tracks_from_manifest(_write_manifest({"tracks": [bad]}))


def test_load_tracks_active_frames_length_mismatch_raises():
    """T4.1.a: active_frames 长度 != F → ValueError."""
    bad = _make_track("veh_01", F=20)
    bad["active_frames"] = [True] * 10  # 应该 20
    with pytest.raises(ValueError, match="active_frames len 10 != poses F 20"):
        load_tracks_from_manifest(_write_manifest({"tracks": [bad]}))


def test_load_tracks_file_not_found():
    """T4.1.a: 不存在路径 → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_tracks_from_manifest("/nonexistent/path/manifest.json")
