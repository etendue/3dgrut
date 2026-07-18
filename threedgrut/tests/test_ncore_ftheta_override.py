# SPDX-License-Identifier: Apache-2.0
"""PIN-FTHETA Task 2: explicit, strict NCore FTheta camera override."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from threedgrut.datasets.ftheta_override import (
    FTHETA_PARAMETER_KEYS,
    add_intrinsics_to_batch_dict,
    assert_ftheta_max_angle_preserved,
    build_ftheta_camera_model,
    build_ftheta_camera_model_parameters,
    extract_ftheta_camera_model_parameters,
    load_ftheta_override_parameters,
    transform_camera_model_parameters,
    validate_ftheta_fov_cap,
)
from threedgrut.datasets.utils import (
    CameraRayDomainStats,
    apply_ftheta_own_domain_mask,
    compute_ftheta_own_domain_mask,
    format_camera_ray_domain_telemetry,
)
from threedgrut_playground.utils.ftheta_intrinsics import (
    ftheta_pixels_to_camera_rays,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET_SOURCE = ROOT / "threedgrut" / "datasets" / "datasetNcore.py"
FACTORY_SOURCE = ROOT / "threedgrut" / "datasets" / "__init__.py"
BASE_CONFIG = ROOT / "configs" / "dataset" / "ncore.yaml"
EXPERIMENT_CONFIG = ROOT / "configs" / "apps" / "ncore_3dgut_mcmc_multilayer_inceptio_ftheta_7cam.yaml"
PINHOLE_EXPERIMENT_CONFIG = ROOT / "configs" / "apps" / "ncore_3dgut_mcmc_multilayer_inceptio_7cam.yaml"
V4_EXPERIMENT_CONFIG = (
    ROOT / "configs" / "apps" / "ncore_3dgut_mcmc_multilayer_inceptio_7cam_v4.yaml"
)
PARAMS_ARTIFACT = ROOT / "scripts" / "pin_ftheta_b6a9_7cam_params.json"
SURVEY_ARTIFACT = ROOT / "scripts" / "pin_ftheta_b6a9_9cam_params.json"
V4_PARAMS_ARTIFACT = (
    ROOT / "scripts" / "pin_ftheta_b6a9_7cam_params_v4_full_domain.json"
)
CONFIG_DIR = str(ROOT / "configs")

SEVEN_CAMERAS = [
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_left_wide_90fov",
    "camera_right_wide_90fov",
    "camera_back_rear_wide_90fov",
    "camera_rear_left_70fov",
]


def _params() -> dict:
    return {
        "resolution": [1920, 1080],
        "shutter_type": "ROLLING_TOP_TO_BOTTOM",
        "principal_point": [959.5, 539.5],
        "reference_poly": "PIXELDIST_TO_ANGLE",
        "pixeldist_to_angle_poly": [0.0, 0.001, 0.0, 0.0, 0.0, 0.0],
        "angle_to_pixeldist_poly": [0.0, 1000.0, 0.0, 0.0, 0.0, 0.0],
        "max_angle": 1.0,
        "linear_cde": [1.0, 0.0, 0.0],
    }


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "ftheta.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _compose(config_name: str):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name=config_name)


def _function_call_keywords(path: Path, function_name: str, callee_name: str) -> list[set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    calls = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
        if name == callee_name:
            calls.append({kw.arg for kw in node.keywords})
    return calls


def test_default_off_constructor_and_base_config() -> None:
    tree = ast.parse(DATASET_SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NCoreDataset")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    positional = [arg.arg for arg in init.args.args]
    defaults = dict(zip(positional[-len(init.args.defaults) :], init.args.defaults, strict=True))
    assert isinstance(defaults["ftheta_params_path"], ast.Constant)
    assert defaults["ftheta_params_path"].value is None
    assert "ftheta_params_path: null" in BASE_CONFIG.read_text(encoding="utf-8")


def test_default_base_config_keeps_mask_off_and_validity_domain_on() -> None:
    config = _compose("apps/ncore_3dgut_mcmc_multilayer_inceptio")

    assert config.dataset.mask_forward_invalid_pixels is False
    assert config.dataset.opencv_pinhole_use_validity_domain is True


@pytest.mark.parametrize(
    "config_name",
    [
        "apps/ncore_3dgut_mcmc_multilayer_inceptio_7cam",
        "apps/ncore_3dgut_mcmc_multilayer_inceptio_ftheta_7cam",
    ],
)
def test_matched_seven_camera_configs_freeze_shared_forward_mask(config_name: str) -> None:
    config = _compose(config_name)

    assert config.dataset.mask_forward_invalid_pixels is True
    assert config.dataset.opencv_pinhole_use_validity_domain is False


def test_loads_exact_camera_mapping_and_stable_fingerprints(tmp_path: Path) -> None:
    path = _write(tmp_path, {"cam_a": _params(), "cam_b": _params()})
    loaded, fingerprints = load_ftheta_override_parameters(path, ["cam_a", "cam_b"])
    assert set(loaded) == {"cam_a", "cam_b"}
    assert all(set(value) == FTHETA_PARAMETER_KEYS for value in loaded.values())
    assert fingerprints["cam_a"] == fingerprints["cam_b"]
    assert len(fingerprints["cam_a"]) == 64


@pytest.mark.parametrize(
    ("payload", "camera_ids", "message"),
    [
        ({"cam_a": _params()}, ["cam_a", "cam_b"], "camera set mismatch"),
        ({"cam_a": _params(), "cam_b": _params()}, ["cam_a"], "camera set mismatch"),
        ({"cam_a": _params()}, ["cam_a", "cam_a"], "duplicate selected camera ID"),
    ],
)
def test_rejects_camera_set_errors(tmp_path: Path, payload: dict, camera_ids: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_ftheta_override_parameters(_write(tmp_path, payload), camera_ids)


def test_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    params_json = json.dumps(_params())
    path = tmp_path / "duplicate.json"
    path.write_text(f'{{"cam_a": {params_json}, "cam_a": {params_json}}}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key 'cam_a'"):
        load_ftheta_override_parameters(path, ["cam_a"])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("max_angle"), "missing"),
        (lambda p: p.update({"unexpected": 1}), "unexpected"),
        (lambda p: p.update({"resolution": [1920.0, 1080]}), "resolution"),
        (lambda p: p.update({"pixeldist_to_angle_poly": [0.0]}), "pixeldist_to_angle_poly"),
        (lambda p: p.update({"max_angle": float("nan")}), "max_angle"),
    ],
)
def test_rejects_missing_extra_and_invalid_parameter_types(tmp_path: Path, mutate, message: str) -> None:
    params = _params()
    mutate(params)
    with pytest.raises((TypeError, ValueError), match=message):
        load_ftheta_override_parameters(_write(tmp_path, {"cam_a": params}), ["cam_a"])


class _FakeFThetaParameters:
    class PolynomialType:
        PIXELDIST_TO_ANGLE = "enum-p2a"
        ANGLE_TO_PIXELDIST = "enum-a2p"

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeShutterType:
    ROLLING_TOP_TO_BOTTOM = "enum-rolling"


def test_builds_public_ncore_ftheta_parameter_type_without_pinhole_fallback() -> None:
    fake_ncore_data = SimpleNamespace(
        FThetaCameraModelParameters=_FakeFThetaParameters,
        ShutterType=_FakeShutterType,
    )
    built = build_ftheta_camera_model_parameters(_params(), ncore_data=fake_ncore_data)
    assert isinstance(built, _FakeFThetaParameters)
    assert built.reference_poly == "enum-p2a"
    assert built.shutter_type == "enum-rolling"
    assert built.resolution.dtype == np.uint64
    assert built.principal_point.dtype == np.float32
    assert built.pixeldist_to_angle_poly.shape == (6,)
    assert built.angle_to_pixeldist_poly.shape == (6,)


class _FakeFThetaCameraModel:
    pass


class _FakeCameraModelFactory:
    calls = []
    result = _FakeFThetaCameraModel()

    @classmethod
    def from_parameters(cls, parameters, *, device, dtype):
        cls.calls.append((parameters, device, dtype))
        return cls.result


def test_conversion_builds_ftheta_model_on_cpu_and_rejects_any_fallback() -> None:
    fake_ncore_data = SimpleNamespace(
        FThetaCameraModelParameters=_FakeFThetaParameters,
        ShutterType=_FakeShutterType,
    )
    fake_ncore_sensors = SimpleNamespace(
        CameraModel=_FakeCameraModelFactory,
        FThetaCameraModel=_FakeFThetaCameraModel,
    )
    _FakeCameraModelFactory.calls = []
    _FakeCameraModelFactory.result = _FakeFThetaCameraModel()

    model = build_ftheta_camera_model(
        _params(),
        camera_id="cam_a",
        ncore_data=fake_ncore_data,
        ncore_sensors=fake_ncore_sensors,
    )

    assert isinstance(model, _FakeFThetaCameraModel)
    assert len(_FakeCameraModelFactory.calls) == 1
    parameters, device, dtype = _FakeCameraModelFactory.calls[0]
    assert isinstance(parameters, _FakeFThetaParameters)
    assert device == "cpu"
    assert dtype is torch.float32

    _FakeCameraModelFactory.result = object()
    with pytest.raises(TypeError, match="cam_a.*refusing native/ideal-pinhole fallback"):
        build_ftheta_camera_model(
            _params(),
            camera_id="cam_a",
            ncore_data=fake_ncore_data,
            ncore_sensors=fake_ncore_sensors,
        )


def test_resolution_scaling_uses_transform_with_scale_and_new_resolution() -> None:
    class SpyParameters:
        resolution = np.array([1920, 1080])

        def __init__(self):
            self.calls = []

        def transform(self, **kwargs):
            self.calls.append(kwargs)
            return "scaled"

    params = SpyParameters()
    assert transform_camera_model_parameters(params, (960, 540)) == "scaled"
    assert params.calls == [{"image_domain_scale": (0.5, 0.5), "new_resolution": (960, 540)}]


def test_ftheta_intrinsics_transform_and_exact_batch_field() -> None:
    transform_calls = []

    class FThetaCameraModelParameters:
        def __init__(self, resolution=(1920, 1080)):
            self.resolution = np.asarray(resolution, dtype=np.uint64)
            self.shutter_type = SimpleNamespace(name="ROLLING_TOP_TO_BOTTOM")
            self.principal_point = np.asarray([959.5, 539.5], dtype=np.float32)
            self.reference_poly = SimpleNamespace(name="PIXELDIST_TO_ANGLE")
            self.pixeldist_to_angle_poly = np.arange(6, dtype=np.float32)
            self.angle_to_pixeldist_poly = np.arange(6, dtype=np.float32) + 10
            self.max_angle = 1.1
            self.linear_cde = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

        def transform(self, *, image_domain_scale, new_resolution):
            transform_calls.append(
                {
                    "image_domain_scale": image_domain_scale,
                    "new_resolution": new_resolution,
                }
            )
            return FThetaCameraModelParameters(new_resolution)

    class FThetaCameraModel:
        def __init__(self, parameters):
            self.parameters = parameters

        def get_parameters(self):
            return self.parameters

    fake_sensors = SimpleNamespace(FThetaCameraModel=FThetaCameraModel)
    params_dict, model_type_name = extract_ftheta_camera_model_parameters(
        FThetaCameraModel(FThetaCameraModelParameters()),
        (960, 540),
        ncore_sensors=fake_sensors,
    )

    assert transform_calls == [
        {
            "image_domain_scale": (0.5, 0.5),
            "new_resolution": (960, 540),
        }
    ]
    assert model_type_name == "FThetaCameraModelParameters"
    assert set(params_dict) == FTHETA_PARAMETER_KEYS
    assert tuple(params_dict["resolution"]) == (960, 540)

    batch_dict = {"rays_in_world_space": False}
    add_intrinsics_to_batch_dict(batch_dict, (params_dict, model_type_name))
    assert batch_dict == {
        "rays_in_world_space": False,
        "intrinsics_FThetaCameraModelParameters": params_dict,
    }


def test_ftheta_domain_red_green_and_strict_boundary() -> None:
    max_angle = 1.2
    angles = np.array(
        [
            np.nextafter(max_angle, 0.0),
            max_angle,
            np.nextafter(max_angle, np.inf),
        ],
        dtype=np.float64,
    )
    finite_rays = np.stack(
        [np.sin(angles), np.zeros_like(angles), np.cos(angles)],
        axis=-1,
    )
    rays = np.concatenate(
        [finite_rays, np.array([[np.nan, 0.0, 1.0]])],
        axis=0,
    ).reshape(1, 4, 3)

    # Historical finite-only supervision accepted the equal/outside rays.
    finite_only = np.isfinite(rays).all(axis=-1)
    np.testing.assert_array_equal(finite_only, [[True, True, True, False]])

    valid = np.ones((1, 4), dtype=bool)
    stats = apply_ftheta_own_domain_mask(rays, valid, max_angle)
    np.testing.assert_array_equal(valid, [[True, False, False, False]])
    np.testing.assert_array_equal(
        compute_ftheta_own_domain_mask(rays, max_angle),
        valid,
    )
    assert stats == CameraRayDomainStats(
        total_pixels=4,
        excluded_by_max_angle=2,
        nonfinite=1,
    )


def test_ftheta_domain_rejects_zero_length_without_rejecting_optical_axis() -> None:
    tiny = np.nextafter(np.float64(0.0), np.float64(1.0))
    huge = np.finfo(np.float64).max
    rays = np.array(
        [
            [0.0, 0.0, 0.0],        # no direction: always invalid
            [0.0, 0.0, 2.0],        # ordinary +Z optical axis: theta=0
            [0.0, 0.0, -2.0],       # -Z optical axis: theta=pi
            [tiny, 0.0, tiny],       # finite subnormal ray: theta=pi/4
            [huge, huge, huge],      # finite huge ray: stable scaled angle
        ],
        dtype=np.float64,
    )
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        mask = compute_ftheta_own_domain_mask(rays, max_angle=1.2)
    np.testing.assert_array_equal(mask, [False, True, False, True, True])


def test_ftheta_domain_mask_has_no_pinhole_trust_math() -> None:
    source = (
        inspect.getsource(compute_ftheta_own_domain_mask)
        + inspect.getsource(apply_ftheta_own_domain_mask)
    ).lower()
    for forbidden in ("icd", "denominator", "jacobian", "opencv", "pinhole"):
        assert forbidden not in source


def test_v4_native_resolution_ftheta_excluded_count_oracles() -> None:
    parameters = json.loads(V4_PARAMS_ARTIFACT.read_text(encoding="utf-8"))
    expected_excluded = {
        "camera_front_wide_120fov": 148,
        "camera_cross_left_120fov": 138,
        "camera_cross_right_120fov": 133,
        "camera_left_wide_90fov": 26_355,
        "camera_right_wide_90fov": 44_292,
        "camera_back_rear_wide_90fov": 120,
        "camera_rear_left_70fov": 101,
    }
    assert list(parameters) == SEVEN_CAMERAS
    for camera_id, expected in expected_excluded.items():
        rays = ftheta_pixels_to_camera_rays(parameters[camera_id])
        mask = compute_ftheta_own_domain_mask(
            rays,
            parameters[camera_id]["max_angle"],
        )
        assert mask.size - int(mask.sum()) == expected, camera_id


def test_ftheta_override_rejects_clipping_and_max_angle_drift() -> None:
    validate_ftheta_fov_cap(1.2, 190.0, camera_id="cam_a")
    with pytest.raises(ValueError, match="silently clipped"):
        validate_ftheta_fov_cap(1.2, 100.0, camera_id="cam_a")

    assert_ftheta_max_angle_preserved(
        1.2,
        np.float32(1.2),
        camera_id="cam_a",
        context="test transform",
    )
    with pytest.raises(ValueError, match="changed max_angle"):
        assert_ftheta_max_angle_preserved(
            1.2,
            1.19,
            camera_id="cam_a",
            context="test transform",
        )


def test_camera_domain_telemetry_has_stable_required_fields() -> None:
    message = format_camera_ray_domain_telemetry(
        split="val",
        camera_id="camera_front_wide_120fov",
        model_type="FThetaCameraModel",
        artifact_fingerprint="abc123",
        stats=CameraRayDomainStats(
            total_pixels=2_073_600,
            excluded_by_max_angle=148,
            nonfinite=0,
        ),
    )
    assert message == (
        "[CAMERA-RAY-DOMAIN] split=val camera=camera_front_wide_120fov "
        "model_type=FThetaCameraModel artifact_fingerprint=abc123 "
        "total=2073600 excluded_by_max_angle=148 nonfinite=0"
    )


def test_train_val_test_all_forward_ftheta_override() -> None:
    make_calls = _function_call_keywords(FACTORY_SOURCE, "make", "NCoreDataset")
    test_calls = _function_call_keywords(FACTORY_SOURCE, "make_test", "NCoreDataset")
    assert len(make_calls) == 2
    assert len(test_calls) == 1
    assert all("ftheta_params_path" in keywords for keywords in make_calls + test_calls)
    assert all("camera_max_fov_deg" in keywords for keywords in make_calls + test_calls)

    factory_tree = ast.parse(FACTORY_SOURCE.read_text(encoding="utf-8"))
    split_literals = []
    for node in ast.walk(factory_tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
        if name != "NCoreDataset":
            continue
        split_kw = next(keyword for keyword in node.keywords if keyword.arg == "split")
        split_literals.append(split_kw.value.value)
    assert split_literals == ["train", "val", "test"]


def test_dataset_source_rejects_runtime_injection_and_keeps_native_mask_guards() -> None:
    source = DATASET_SOURCE.read_text(encoding="utf-8")
    assert "ftheta_params_path is deprecated and unsupported at runtime" in source
    assert "camera_model = build_ftheta_camera_model(" not in source
    assert "result = extract_ftheta_camera_model_parameters(" in source
    assert "add_intrinsics_to_batch_dict(batch_dict, intrinsics_result)" in source
    assert "maybe_apply_forward_valid_mask" in source  # native path remains available
    assert "mask_forward_invalid_pixels and isinstance(camera_model, ncore.sensors.OpenCVPinholeCameraModel)" in source
    assert "_domain_stats = apply_ftheta_own_domain_mask(" in source
    assert "if isinstance(camera_model, ncore.sensors.FThetaCameraModel):" in source
    assert "format_camera_ray_domain_telemetry(" in source

    tree = ast.parse(source)
    dataset_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NCoreDataset"
    )
    method_calls: dict[str, list[str]] = {}
    for method in dataset_class.body:
        if not isinstance(method, ast.FunctionDef):
            continue
        method_calls[method.name] = [
            node.func.id
            if isinstance(node.func, ast.Name)
            else getattr(node.func, "attr", "")
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
        ]
    assert method_calls["_init_worker"].count(
        "format_camera_ray_domain_telemetry"
    ) == 1
    assert "format_camera_ray_domain_telemetry" not in method_calls["__getitem__"]


def test_v4_shared_config_freezes_seven_cameras_and_fov_cap() -> None:
    config = _compose("apps/ncore_3dgut_mcmc_multilayer_inceptio_7cam_v4")
    assert list(config.dataset.camera_ids) == SEVEN_CAMERAS
    assert config.dataset.ftheta_params_path is None
    assert config.dataset.camera_max_fov_deg == pytest.approx(190.0)
    assert config.dataset.mask_forward_invalid_pixels is True
    assert config.dataset.opencv_pinhole_use_validity_domain is False
    source = V4_EXPERIMENT_CONFIG.read_text(encoding="utf-8")
    assert "camera_max_fov_deg: 190.0" in source
    assert "camera_front_standard_55fov" not in source
    assert "camera_front_tele_30fov" not in source


def test_seven_camera_experiment_artifact_and_config_are_matched() -> None:
    artifact = json.loads(PARAMS_ARTIFACT.read_text(encoding="utf-8"))
    survey = json.loads(SURVEY_ARTIFACT.read_text(encoding="utf-8"))
    assert list(artifact) == SEVEN_CAMERAS
    assert all(set(params) == FTHETA_PARAMETER_KEYS for params in artifact.values())
    assert artifact == {camera_id: survey["cameras"][camera_id]["ftheta_parameters"] for camera_id in SEVEN_CAMERAS}

    config = EXPERIMENT_CONFIG.read_text(encoding="utf-8")
    assert "ftheta_params_path: scripts/pin_ftheta_b6a9_7cam_params.json" in config
    for camera_id in SEVEN_CAMERAS:
        assert f"- {camera_id}" in config
    assert "- camera_front_standard_55fov" not in config
    assert "- camera_front_tele_30fov" not in config
    assert "camera_loss_weights" not in config

    pinhole_config = PINHOLE_EXPERIMENT_CONFIG.read_text(encoding="utf-8")
    assert "ftheta_params_path: null" in pinhole_config
    for camera_id in SEVEN_CAMERAS:
        assert f"- {camera_id}" in pinhole_config
    assert "- camera_front_standard_55fov" not in pinhole_config
    assert "- camera_front_tele_30fov" not in pinhole_config
    assert "camera_loss_weights" not in pinhole_config
