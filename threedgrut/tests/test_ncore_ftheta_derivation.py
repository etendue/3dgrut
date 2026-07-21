# SPDX-License-Identifier: Apache-2.0
"""Native-NCore FTheta derivation and manifest-training contracts."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from threedgrut.datasets.ftheta_derivation import (
    copy_nonintrinsics_components,
    fit_source_opencv_camera_parameters,
    opencv_pinhole_parameters_to_dict,
    prepare_ftheta_conversion_parameters,
    validate_and_expand_raster_max_angle,
    validate_native_ftheta_model_raster,
)
from threedgrut.datasets.ftheta_override import (
    FTHETA_PARAMETER_KEYS,
    add_intrinsics_to_batch_dict,
    build_ftheta_camera_model_parameters,
    extract_ftheta_camera_model_parameters,
    load_ftheta_conversion_parameters,
    transform_camera_model_parameters,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET_SOURCE = ROOT / "threedgrut" / "datasets" / "datasetNcore.py"
FACTORY_SOURCE = ROOT / "threedgrut" / "datasets" / "__init__.py"
BASE_CONFIG = ROOT / "configs" / "dataset" / "ncore.yaml"
NATIVE_6CAM_CONFIG = (
    ROOT / "configs" / "apps" / "ncore_3dgut_mcmc_multilayer_inceptio_6cam_native_ab.yaml"
)
V4_ARTIFACT = ROOT / "scripts" / "pin_ftheta_b6a9_7cam_params_v4_full_domain.json"
CONFIG_DIR = str(ROOT / "configs")

SIX_CAMERAS = [
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
    "camera_back_rear_wide_90fov",
]

HISTORICAL_SEVEN_CAMERAS = [
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_left_wide_90fov",
    "camera_right_wide_90fov",
    "camera_back_rear_wide_90fov",
    "camera_rear_left_70fov",
]


def _parameters(*, resolution=(5, 3), max_angle=0.05) -> dict:
    return {
        "resolution": list(resolution),
        "shutter_type": "ROLLING_TOP_TO_BOTTOM",
        "principal_point": [(resolution[0] - 1) / 2, (resolution[1] - 1) / 2],
        "reference_poly": "PIXELDIST_TO_ANGLE",
        "pixeldist_to_angle_poly": [0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
        "angle_to_pixeldist_poly": [0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
        "max_angle": max_angle,
        "linear_cde": [1.0, 0.0, 0.0],
    }


def _compose(config_name: str):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name=config_name)


def test_training_dataset_rejects_runtime_ftheta_override() -> None:
    source = DATASET_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    dataset_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NCoreDataset"
    )
    constructor = next(
        node for node in dataset_class.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    argument_names = {argument.arg for argument in constructor.args.args}

    assert "ftheta_params_path" in argument_names
    assert "ftheta_params_path is deprecated and unsupported at runtime" in source
    assert "ftheta_override_enabled" not in source
    assert "load_ftheta_override_parameters" not in source
    assert "build_ftheta_camera_model(" not in source
    assert "apply_ftheta_own_domain_mask" not in source
    assert "camera_sensors[camera_id].model_parameters" in source


def test_derivation_keeps_its_lazy_zarr_store_dependency() -> None:
    source = DATASET_SOURCE.parent / "ftheta_derivation.py"
    text = source.read_text(encoding="utf-8")
    assert "import zarr" in text
    assert "zarr.open_group(store=source_store, mode=\"r\")" in text


def test_shared_intrinsics_store_preserves_nonintrinsics_components() -> None:
    class FakeZarr:
        copies: list[tuple[object, object, str, str]] = []

        @classmethod
        def copy(cls, source, target, *, name, if_exists):
            cls.copies.append((source, target, name, if_exists))

    source = {"intrinsics": "replace", "poses": "keep-poses", "masks": "keep-masks", "cuboids": "keep-cuboids"}
    target = object()
    copy_nonintrinsics_components(source, target, zarr_module=FakeZarr)

    assert FakeZarr.copies == [
        ("keep-poses", target, "poses", "raise"),
        ("keep-masks", target, "masks", "raise"),
        ("keep-cuboids", target, "cuboids", "raise"),
    ]


def test_factory_keeps_only_a_fail_closed_legacy_field_and_native_config_omits_it() -> None:
    assert "ftheta_params_path" in FACTORY_SOURCE.read_text(encoding="utf-8")
    assert "ftheta_params_path: null" in BASE_CONFIG.read_text(encoding="utf-8")
    assert "ftheta_params_path" not in NATIVE_6CAM_CONFIG.read_text(encoding="utf-8")

    config = _compose("apps/ncore_3dgut_mcmc_multilayer_inceptio_6cam_native_ab")
    assert list(config.dataset.camera_ids) == SIX_CAMERAS
    # The base config retains a fail-closed migration key for historical
    # configurations. The native config does not declare or populate it.
    assert config.dataset.ftheta_params_path is None


def test_native_ftheta_does_not_receive_a_generic_supervision_mask() -> None:
    source = DATASET_SOURCE.read_text(encoding="utf-8")
    assert (
        "self.mask_forward_invalid_pixels and isinstance(camera_model, "
        "ncore.sensors.OpenCVPinholeCameraModel)"
    ) in source
    assert "apply_ftheta_own_domain_mask" not in source
    assert "compute_ftheta_own_domain_mask" not in source


def test_full_raster_coverage_expands_only_max_angle() -> None:
    parameters = _parameters()
    prepared, coverage = validate_and_expand_raster_max_angle(parameters, margin_deg=0.1)

    expected_raster_angle = 0.1 * math.hypot(2.0, 1.0)
    assert coverage.total_pixels == 15
    assert coverage.finite_pixels == 15
    assert coverage.raster_max_angle == pytest.approx(expected_raster_angle)
    assert coverage.output_max_angle > coverage.raster_max_angle
    assert coverage.adjusted is True
    assert prepared["max_angle"] == coverage.output_max_angle
    for key in FTHETA_PARAMETER_KEYS - {"max_angle"}:
        assert prepared[key] == parameters[key]


def test_existing_larger_max_angle_is_preserved_exactly() -> None:
    parameters = _parameters(max_angle=1.0)
    prepared, coverage = validate_and_expand_raster_max_angle(parameters)

    assert coverage.adjusted is False
    assert prepared["max_angle"] == parameters["max_angle"]


def test_invalid_linear_term_fails_conversion_instead_of_masking_pixels() -> None:
    parameters = _parameters()
    parameters["linear_cde"] = [1.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="singular"):
        validate_and_expand_raster_max_angle(parameters)


def test_native_raster_validation_checks_all_pixels_and_roundtrip() -> None:
    class NativeModel:
        def pixels_to_camera_rays(self, pixels):
            pixels = np.asarray(pixels)
            return np.column_stack((pixels, np.ones(len(pixels), dtype=np.float32)))

        def camera_rays_to_pixels(self, rays):
            pixels = np.asarray(rays)[:, :2].astype(np.int32)
            return SimpleNamespace(
                pixels=pixels,
                valid_flag=np.ones(len(pixels), dtype=bool),
            )

    validate_native_ftheta_model_raster(
        NativeModel(), camera_id="cam", width=5, height=3, rows_per_chunk=2
    )


def test_native_raster_validation_rejects_any_forward_invalid_pixel() -> None:
    class NativeModel:
        def pixels_to_camera_rays(self, pixels):
            pixels = np.asarray(pixels)
            return np.column_stack((pixels, np.ones(len(pixels), dtype=np.float32)))

        def camera_rays_to_pixels(self, rays):
            pixels = np.asarray(rays)[:, :2].astype(np.int32)
            valid = np.ones(len(pixels), dtype=bool)
            valid[-1] = False
            return SimpleNamespace(pixels=pixels, valid_flag=valid)

    with pytest.raises(ValueError, match="forward path rejects"):
        validate_native_ftheta_model_raster(
            NativeModel(), camera_id="cam", width=5, height=3, rows_per_chunk=2
        )


def test_parameters_are_expanded_to_cover_all_native_pixels() -> None:
    prepared, coverage = validate_and_expand_raster_max_angle(_parameters())

    assert coverage.finite_pixels == coverage.total_pixels
    assert coverage.output_max_angle > coverage.raster_max_angle
    assert prepared["max_angle"] == coverage.output_max_angle


def test_conversion_loader_keeps_exact_camera_mapping(tmp_path: Path) -> None:
    artifact = tmp_path / "parameters.json"
    artifact.write_text(json.dumps({"cam": _parameters()}), encoding="utf-8")
    loaded, fingerprints = load_ftheta_conversion_parameters(artifact, ["cam"])
    assert set(loaded["cam"]) == FTHETA_PARAMETER_KEYS
    assert len(fingerprints["cam"]) == 64


def test_source_opencv_fit_uses_each_selected_camera_calibration() -> None:
    class Shutter:
        name = "ROLLING_TOP_TO_BOTTOM"

    def source(cx: float):
        return SimpleNamespace(
            resolution=np.asarray([64, 48], dtype=np.uint64),
            shutter_type=Shutter(),
            principal_point=np.asarray([cx, 23.5], dtype=np.float32),
            focal_length=np.asarray([50.0, 50.0], dtype=np.float32),
            radial_coeffs=np.zeros(6, dtype=np.float32),
            tangential_coeffs=np.zeros(2, dtype=np.float32),
            thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        )

    sources = {"left": source(31.0), "right": source(32.0)}
    prepared, coverage, fingerprints, calibrations = fit_source_opencv_camera_parameters(
        sources,
        ["left", "right"],
    )

    assert list(prepared) == ["left", "right"]
    assert prepared["left"]["principal_point"] == [31.0, 23.5]
    assert prepared["right"]["principal_point"] == [32.0, 23.5]
    assert fingerprints["left"] != fingerprints["right"]
    assert calibrations["left"] == opencv_pinhole_parameters_to_dict(sources["left"])
    assert all(item.finite_pixels == item.total_pixels for item in coverage.values())


class _FakeFThetaParameters:
    class PolynomialType:
        PIXELDIST_TO_ANGLE = "pixel-to-angle"
        ANGLE_TO_PIXELDIST = "angle-to-pixel"

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeShutterType:
    ROLLING_TOP_TO_BOTTOM = "rolling"


def test_conversion_builds_native_ncore_parameter_type() -> None:
    fake_ncore_data = SimpleNamespace(
        FThetaCameraModelParameters=_FakeFThetaParameters,
        ShutterType=_FakeShutterType,
    )
    result = build_ftheta_camera_model_parameters(_parameters(), ncore_data=fake_ncore_data)

    assert isinstance(result, _FakeFThetaParameters)
    assert result.reference_poly == "pixel-to-angle"
    assert result.shutter_type == "rolling"
    assert result.resolution.dtype == np.uint64
    assert result.pixeldist_to_angle_poly.dtype == np.float32


def test_native_pai_ftheta_intrinsics_still_flow_to_renderer_batch() -> None:
    transform_calls = []

    class FThetaCameraModelParameters:
        def __init__(self, resolution=(1920, 1080)):
            self.resolution = np.asarray(resolution, dtype=np.uint64)
            self.shutter_type = SimpleNamespace(name="ROLLING_TOP_TO_BOTTOM")
            self.principal_point = np.asarray([959.5, 539.5], dtype=np.float32)
            self.reference_poly = SimpleNamespace(name="PIXELDIST_TO_ANGLE")
            self.pixeldist_to_angle_poly = np.arange(6, dtype=np.float32)
            self.angle_to_pixeldist_poly = np.arange(6, dtype=np.float32) + 10
            self.max_angle = 1.3
            self.linear_cde = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

        def transform(self, *, image_domain_scale, new_resolution):
            transform_calls.append((image_domain_scale, new_resolution))
            return FThetaCameraModelParameters(new_resolution)

    class FThetaCameraModel:
        def get_parameters(self):
            return FThetaCameraModelParameters()

    params, model_type = extract_ftheta_camera_model_parameters(
        FThetaCameraModel(),
        (960, 540),
        ncore_sensors=SimpleNamespace(FThetaCameraModel=FThetaCameraModel),
    )
    batch = {}
    add_intrinsics_to_batch_dict(batch, (params, model_type))

    assert transform_calls == [((0.5, 0.5), (960, 540))]
    assert set(params) == FTHETA_PARAMETER_KEYS
    assert "intrinsics_FThetaCameraModelParameters" in batch


def test_camera_parameter_downsample_uses_native_transform() -> None:
    class Parameters:
        resolution = np.asarray([1920, 1080])

        def __init__(self):
            self.calls = []

        def transform(self, **kwargs):
            self.calls.append(kwargs)
            return self

    parameters = Parameters()
    transform_camera_model_parameters(parameters, (960, 540))
    assert parameters.calls == [
        {"image_domain_scale": (0.5, 0.5), "new_resolution": (960, 540)}
    ]
