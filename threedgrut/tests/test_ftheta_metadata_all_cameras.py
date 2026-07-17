"""Phase 2 Task 3: all-active-camera FTheta checkpoint/viewer contract."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from threedgrut.viz.metadata import _extract_camera_models
from threedgrut_playground.utils.camera_render_state import (
    CameraModelKind,
    merge_checkpoint_camera_models,
    resolve_camera_render_state,
)
from threedgrut_playground.utils.viz4d_metadata import FourDMetadata

ACTIVE_7_CAMERAS = (
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_left_wide_90fov",
    "camera_right_wide_90fov",
    "camera_back_rear_wide_90fov",
    "camera_rear_left_70fov",
)


def _ftheta_params(index: int, resolution=(1920, 1080)) -> SimpleNamespace:
    return SimpleNamespace(
        resolution=np.asarray(resolution, dtype=np.uint64),
        shutter_type=SimpleNamespace(name="GLOBAL"),
        principal_point=np.asarray([960.0 + index, 540.0], dtype=np.float32),
        reference_poly=SimpleNamespace(name="PIXELDIST_TO_ANGLE"),
        pixeldist_to_angle_poly=np.asarray([0.0, 0.002 + index * 1e-5], dtype=np.float32),
        angle_to_pixeldist_poly=np.asarray([0.0, 500.0 + index], dtype=np.float32),
        max_angle=1.0 + index * 0.01,
        linear_cde=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    )


def _ftheta_model(index: int, resolution=(1920, 1080)) -> SimpleNamespace:
    params = _ftheta_params(index, resolution)
    return SimpleNamespace(
        resolution=np.asarray(resolution, dtype=np.uint64),
        max_angle=params.max_angle,
        get_parameters=lambda: params,
    )


def _strict_dataset() -> SimpleNamespace:
    sequence_id = "b6a9"
    models = {
        cam_id: _ftheta_model(index, (1920 - index * 8, 1080 - index * 4))
        for index, cam_id in enumerate(ACTIVE_7_CAMERAS)
    }
    return SimpleNamespace(
        sequence_id=sequence_id,
        camera_ids=list(ACTIVE_7_CAMERAS),
        sequence_camera_models={sequence_id: models},
        ftheta_override_enabled=True,
        ftheta_parameter_fingerprints={cam_id: f"sha256:{index:064x}" for index, cam_id in enumerate(ACTIVE_7_CAMERAS)},
    )


def _minimal_viz(camera_models: dict) -> dict:
    primary = ACTIVE_7_CAMERAS[0]
    primary_contract = camera_models[primary]
    return {
        "schema_version": 3,
        "dataset_type": "ncore",
        "sequence_id": "b6a9",
        "camera_models": camera_models,
        "ego": {
            "poses_c2w": np.eye(4, dtype=np.float32)[None],
            "frame_timestamps_us": np.asarray([1_000], dtype=np.int64),
            "primary_camera_id": primary,
            "primary_camera_fov_y_rad": 2.0,
            "primary_camera_aspect": 16.0 / 9.0,
            # Keep the v2 aliases populated for old viewers.
            "primary_camera_intrinsics_FTheta": primary_contract["intrinsics_FTheta"],
            "primary_camera_resolution": primary_contract["native_resolution"],
        },
        "tracks": {},
        "tracks_camera_timestamps_us": None,
        "lidar": {},
        "viewer_defaults": {
            "initial_c2w": np.eye(4, dtype=np.float32),
            "t_us_first": 1_000,
            "t_us_last": 1_000,
        },
    }


def _pose_entries(camera_models: dict) -> dict:
    entries = {}
    for cam_id in camera_models:
        entries[cam_id] = {
            "c2w": np.eye(4, dtype=np.float32)[None],
            "timestamps_us": np.asarray([1_000], dtype=np.int64),
            "resolution": (640, 360),
            "fov_y_rad": 0.5,
            "ftheta_dict": None,
            # Raw manifest state may still be OpenCV pinhole. The checkpoint
            # FTheta contract must replace it atomically.
            "opencv_pinhole_dict": {"focal_length": [100.0, 100.0]},
            "opencv_pinhole_rays": np.zeros((360, 640, 3), dtype=np.float32),
        }
    return entries


def test_extract_persists_all_seven_active_ftheta_cameras():
    contracts = _extract_camera_models(_strict_dataset())

    assert tuple(contracts) == ACTIVE_7_CAMERAS
    assert len(contracts) == 7
    for index, cam_id in enumerate(ACTIVE_7_CAMERAS):
        entry = contracts[cam_id]
        assert entry["model_type"] == "FTheta"
        assert entry["native_resolution"] == (1920 - index * 8, 1080 - index * 4)
        assert len(entry["intrinsics_FTheta"]) == 8
        assert entry["parameter_fingerprint"] == f"sha256:{index:064x}"


def test_strict_extract_rejects_one_non_ftheta_active_camera():
    dataset = _strict_dataset()
    dataset.sequence_camera_models[dataset.sequence_id][ACTIVE_7_CAMERAS[-1]] = SimpleNamespace(
        resolution=np.asarray([1920, 1080]),
        focal_length=np.asarray([900.0, 900.0]),
    )

    with pytest.raises(ValueError, match=ACTIVE_7_CAMERAS[-1]):
        _extract_camera_models(dataset)


def test_legacy_non_ftheta_fisheye_is_not_misclassified():
    """OpenCVFisheye may expose max_angle/get_parameters but no FTheta poly."""
    camera_id = "legacy_fisheye"
    fisheye_params = SimpleNamespace(
        resolution=np.asarray([1920, 1080], dtype=np.uint64),
        shutter_type=SimpleNamespace(name="GLOBAL"),
        principal_point=np.asarray([960.0, 540.0], dtype=np.float32),
        focal_length=np.asarray([700.0, 700.0], dtype=np.float32),
        radial_coeffs=np.zeros(4, dtype=np.float32),
    )
    model = SimpleNamespace(
        resolution=fisheye_params.resolution,
        max_angle=1.2,
        get_parameters=lambda: fisheye_params,
    )
    dataset = SimpleNamespace(
        sequence_id="legacy",
        camera_ids=[camera_id],
        sequence_camera_models={"legacy": {camera_id: model}},
        ftheta_override_enabled=False,
    )

    contracts = _extract_camera_models(dataset)

    assert contracts[camera_id]["model_type"] == "IdealPinhole"
    assert "intrinsics_FTheta" not in contracts[camera_id]


def test_schema_v3_switch_uses_each_camera_ftheta_and_native_resolution():
    camera_models = _extract_camera_models(_strict_dataset())
    metadata = FourDMetadata.from_ckpt({"viz_4d": _minimal_viz(camera_models)})

    merged = merge_checkpoint_camera_models(_pose_entries(camera_models), metadata.camera_models)

    assert tuple(merged) == ACTIVE_7_CAMERAS
    for index, cam_id in enumerate(ACTIVE_7_CAMERAS):
        state = resolve_camera_render_state(cam_id, merged[cam_id], 1_000)
        assert state.model_kind is CameraModelKind.FTHETA
        assert state.resolution == (1920 - index * 8, 1080 - index * 4)
        assert state.ftheta_dict["principal_point"][0] == pytest.approx(960.0 + index)
        assert state.opencv_pinhole_dict is None
        assert state.opencv_pinhole_rays is None


def test_schema_v3_hydrates_primary_alias_without_ideal_pinhole_fallback():
    camera_models = _extract_camera_models(_strict_dataset())
    viz = _minimal_viz(camera_models)
    del viz["ego"]["primary_camera_intrinsics_FTheta"]
    del viz["ego"]["primary_camera_resolution"]

    metadata = FourDMetadata.from_ckpt({"viz_4d": viz})

    assert metadata.has_ftheta() is True
    np.testing.assert_allclose(
        metadata.ego_primary_intrinsics_ftheta["principal_point"],
        camera_models[ACTIVE_7_CAMERAS[0]]["intrinsics_FTheta"]["principal_point"],
    )
    assert metadata.ego_primary_resolution == (1920, 1080)


def test_schema_v3_ftheta_entry_missing_required_key_fails_fast():
    camera_models = _extract_camera_models(_strict_dataset())
    del camera_models[ACTIVE_7_CAMERAS[3]]["intrinsics_FTheta"]["max_angle"]

    with pytest.raises(ValueError, match=f"{ACTIVE_7_CAMERAS[3]}.*max_angle"):
        FourDMetadata.from_ckpt({"viz_4d": _minimal_viz(camera_models)})


@pytest.mark.parametrize("missing_value", [pytest.param("absent", id="absent"), pytest.param(None, id="none")])
def test_schema_v3_missing_camera_models_fails_fast(missing_value):
    viz = _minimal_viz(_extract_camera_models(_strict_dataset()))
    if missing_value == "absent":
        del viz["camera_models"]
    else:
        viz["camera_models"] = missing_value

    with pytest.raises(ValueError, match="schema_v3.*camera_models"):
        FourDMetadata.from_ckpt({"viz_4d": viz})


def test_schema_v3_resolution_mismatch_fails_fast():
    camera_models = _extract_camera_models(_strict_dataset())
    camera_models[ACTIVE_7_CAMERAS[2]]["native_resolution"] = (1280, 720)

    with pytest.raises(ValueError, match=f"{ACTIVE_7_CAMERAS[2]}.*resolution"):
        FourDMetadata.from_ckpt({"viz_4d": _minimal_viz(camera_models)})


def test_merge_fails_when_active_camera_pose_is_missing():
    camera_models = _extract_camera_models(_strict_dataset())
    pose_entries = _pose_entries(camera_models)
    del pose_entries[ACTIVE_7_CAMERAS[4]]

    with pytest.raises(KeyError, match=ACTIVE_7_CAMERAS[4]):
        merge_checkpoint_camera_models(pose_entries, camera_models)


def test_schema_v2_checkpoint_keeps_primary_only_compatibility():
    camera_models = _extract_camera_models(_strict_dataset())
    viz = _minimal_viz(camera_models)
    viz["schema_version"] = 2
    del viz["camera_models"]

    metadata = FourDMetadata.from_ckpt({"viz_4d": viz})

    assert metadata.camera_models == {}
    assert metadata.has_ftheta() is True
