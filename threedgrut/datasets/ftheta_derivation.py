# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a native-FTheta NCore V4 dataset as an offline derivation step.

This module deliberately has no dependency on :class:`NCoreDataset`.  FTheta
parameters are validated and, when necessary, their ``max_angle`` is expanded
to cover the complete target raster before they are serialized as native NCore
intrinsics. Training subsequently follows the ordinary manifest-native NCore
path; it does not inject FTheta parameters into a PAI dataset at runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from threedgrut.datasets.ftheta_override import (
    build_ftheta_camera_model_parameters,
    load_ftheta_conversion_parameters,
)
from threedgrut_playground.utils.ftheta_fitter import (
    compute_fullimage_angular_error,
    fit_ftheta_from_opencv_rational,
)


@dataclass(frozen=True)
class FThetaRasterCoverage:
    """Coverage result for one native FTheta calibration."""

    width: int
    height: int
    original_max_angle: float
    raster_max_angle: float
    output_max_angle: float
    margin_rad: float
    adjusted: bool
    original_pixeldist_zero_intercept: float
    output_pixeldist_zero_intercept: float
    zero_intercept_adjusted: bool
    finite_pixels: int
    total_pixels: int


def _eval_poly(coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate ascending-order coefficients with Horner's method."""

    result = np.zeros_like(x, dtype=np.float64)
    for coefficient in coefficients[::-1]:
        result = result * x + coefficient
    return result


def _validate_monotonic_polynomial(
    coefficients: Sequence[float],
    *,
    domain_max: float,
    label: str,
    samples: int = 4097,
) -> None:
    """Fail conversion when either native FTheta mapping folds in its domain."""

    coefficients_array = np.asarray(coefficients, dtype=np.float64)
    if coefficients_array.shape != (6,) or not np.isfinite(coefficients_array).all():
        raise ValueError(f"{label} must contain six finite polynomial coefficients")
    if not math.isfinite(domain_max) or domain_max <= 0.0:
        raise ValueError(f"{label} has invalid monotonicity domain")
    values = np.linspace(0.0, domain_max, samples, dtype=np.float64)
    derivative = np.array(
        [(index + 1) * coefficients_array[index + 1] for index in range(5)],
        dtype=np.float64,
    )
    slope = _eval_poly(derivative, values)
    if not np.isfinite(slope).all() or float(np.min(slope)) <= 0.0:
        raise ValueError(
            f"{label} is not strictly monotonic over its native FTheta domain "
            f"(minimum derivative={float(np.nanmin(slope)):.9g})"
        )


def validate_and_expand_raster_max_angle(
    parameters: Mapping[str, Any],
    *,
    margin_deg: float = 0.1,
    rows_per_chunk: int = 128,
) -> tuple[dict[str, Any], FThetaRasterCoverage]:
    """Return parameters whose strict FTheta cone covers every target pixel.

    NCore stores the FTheta principal point at pixel-centre coordinates.  Its
    runtime model adds 0.5 to that value and ``pixels_to_camera_rays`` adds the
    same 0.5 to integer pixel indices, so the offsets cancel here.  This is the
    exact native inverse-ray geometry, evaluated in bounded row chunks.

    Two image-domain repairs are allowed: expanding ``max_angle`` and enforcing
    the physical central-ray constraint ``theta(r=0)=0`` when an unconstrained
    fit has a small negative constant term. Other non-finite/negative angles,
    a singular linear term, or a raster reaching/pi crossing fail conversion.
    """

    if not math.isfinite(margin_deg) or margin_deg <= 0.0:
        raise ValueError("margin_deg must be finite and positive")
    if rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")

    width, height = (int(v) for v in parameters["resolution"])
    principal_point = np.asarray(parameters["principal_point"], dtype=np.float64)
    bw_poly = np.asarray(parameters["pixeldist_to_angle_poly"], dtype=np.float64)
    original_zero_intercept = float(bw_poly[0])
    c, d, e = (float(v) for v in parameters["linear_cde"])
    determinant = c - e * d
    if not math.isfinite(determinant) or abs(determinant) < 1e-12:
        raise ValueError("FTheta linear_cde is singular")
    inverse_linear = np.asarray([[1.0, -d], [-e, c]], dtype=np.float64) / determinant

    corners = np.asarray(
        [
            [-principal_point[0], -principal_point[1]],
            [width - 1 - principal_point[0], -principal_point[1]],
            [-principal_point[0], height - 1 - principal_point[1]],
            [width - 1 - principal_point[0], height - 1 - principal_point[1]],
        ],
        dtype=np.float64,
    )
    raster_radius_max = float(np.max(np.linalg.norm(corners @ inverse_linear.T, axis=1)))

    x = np.arange(width, dtype=np.float64) - principal_point[0]

    def scan_raster(poly: np.ndarray) -> tuple[float, float, int]:
        raster_max = -math.inf
        raster_min = math.inf
        finite_count = 0
        for y0 in range(0, height, rows_per_chunk):
            y = np.arange(y0, min(y0 + rows_per_chunk, height), dtype=np.float64)
            y = y[:, None] - principal_point[1]
            dx = np.broadcast_to(x[None, :], (len(y), width))
            dy = np.broadcast_to(y, (len(y), width))
            undistorted_x = inverse_linear[0, 0] * dx + inverse_linear[0, 1] * dy
            undistorted_y = inverse_linear[1, 0] * dx + inverse_linear[1, 1] * dy
            radius = np.hypot(undistorted_x, undistorted_y)
            theta = _eval_poly(poly, radius)
            finite = np.isfinite(theta)
            finite_count += int(finite.sum())
            if not finite.all():
                continue
            raster_max = max(raster_max, float(theta.max()))
            raster_min = min(raster_min, float(theta.min()))
        return raster_min, raster_max, finite_count

    min_angle, raster_max_angle, finite_pixels = scan_raster(bw_poly)

    total_pixels = width * height
    if finite_pixels != total_pixels:
        raise ValueError(
            f"FTheta inverse polynomial is non-finite for {total_pixels - finite_pixels}/{total_pixels} target pixels"
        )
    zero_intercept_adjusted = min_angle < -1e-7 and original_zero_intercept < 0.0
    if zero_intercept_adjusted:
        bw_poly = bw_poly.copy()
        bw_poly[0] = 0.0
        min_angle, raster_max_angle, finite_pixels = scan_raster(bw_poly)
    if min_angle < -1e-7:
        raise ValueError(f"FTheta inverse polynomial produced negative angle {min_angle:.9g} rad")
    if not 0.0 < raster_max_angle < math.pi:
        raise ValueError(
            f"FTheta target raster requires invalid maximum angle {raster_max_angle:.9g} rad"
        )

    margin_rad = math.radians(margin_deg)
    required = raster_max_angle + margin_rad
    if required >= math.pi:
        raise ValueError("FTheta raster plus safety margin reaches or exceeds pi radians")
    original = float(parameters["max_angle"])
    if original >= required:
        output = original
    else:
        # The NCore forward-valid check is strict (theta < max_angle). Preserve
        # a strict margin after float32 serialization when expanding the cone.
        output = float(np.nextafter(np.float32(required), np.float32(np.inf)))

    adjusted_parameters = dict(parameters)
    if zero_intercept_adjusted:
        adjusted_parameters["pixeldist_to_angle_poly"] = bw_poly.tolist()
    adjusted_parameters["max_angle"] = output
    # Both forward and inverse models must remain one-to-one across the full
    # target raster.  A finite fit that folds is a data-conversion failure,
    # not a training-time masking decision.
    _validate_monotonic_polynomial(
        adjusted_parameters["pixeldist_to_angle_poly"],
        domain_max=raster_radius_max,
        label="pixeldist_to_angle_poly",
    )
    _validate_monotonic_polynomial(
        adjusted_parameters["angle_to_pixeldist_poly"],
        domain_max=output,
        label="angle_to_pixeldist_poly",
    )
    return adjusted_parameters, FThetaRasterCoverage(
        width=width,
        height=height,
        original_max_angle=original,
        raster_max_angle=raster_max_angle,
        output_max_angle=output,
        margin_rad=margin_rad,
        adjusted=(
            zero_intercept_adjusted
            or not math.isclose(output, original, rel_tol=0.0, abs_tol=1e-8)
        ),
        original_pixeldist_zero_intercept=original_zero_intercept,
        output_pixeldist_zero_intercept=float(bw_poly[0]),
        zero_intercept_adjusted=zero_intercept_adjusted,
        finite_pixels=finite_pixels,
        total_pixels=total_pixels,
    )


def prepare_ftheta_conversion_parameters(
    artifact_path: str | Path,
    *,
    margin_deg: float = 0.1,
) -> tuple[dict[str, dict[str, Any]], dict[str, FThetaRasterCoverage], dict[str, str]]:
    """Load the exact artifact and prepare full-raster native parameters."""

    artifact_path = Path(artifact_path).expanduser().resolve()
    with artifact_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError("FTheta conversion artifact root must be a camera mapping")
    parameters, fingerprints = load_ftheta_conversion_parameters(
        artifact_path, payload.keys()
    )
    prepared: dict[str, dict[str, Any]] = {}
    coverage: dict[str, FThetaRasterCoverage] = {}
    for camera_id, camera_parameters in parameters.items():
        prepared[camera_id], coverage[camera_id] = validate_and_expand_raster_max_angle(
            camera_parameters,
            margin_deg=margin_deg,
        )
    return prepared, coverage, fingerprints


def _jsonable(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "name"):
        return str(value.name)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _fit_report(
    source_calibration: Mapping[str, Any],
    ftheta_parameters: Mapping[str, Any],
    coverage: FThetaRasterCoverage,
) -> dict[str, Any]:
    """Report approximation error without using it as a conversion gate."""

    metrics = _jsonable(compute_fullimage_angular_error(dict(source_calibration), dict(ftheta_parameters)))
    warnings: list[str] = []
    for key in ("nonradial_floor_mean_deg", "nonradial_floor_max_deg", "max_deg", "forward_poly_max_px"):
        value = metrics.get(key)
        if value is not None and float(value) > 0.0:
            warnings.append(f"{key}={float(value):.6g} is recorded for review, not used as a conversion gate")
    return {
        "opencv_to_ftheta_metrics": metrics,
        "full_raster_max_angle_rad": coverage.raster_max_angle,
        "output_max_angle_rad": coverage.output_max_angle,
        "full_raster_coverage": coverage.finite_pixels / coverage.total_pixels,
        "warnings": warnings,
    }


def opencv_pinhole_parameters_to_dict(parameters: Any) -> dict[str, Any]:
    """Extract the OpenCV fields consumed by the deterministic FTheta fitter."""

    required = (
        "resolution",
        "shutter_type",
        "principal_point",
        "focal_length",
        "radial_coeffs",
        "tangential_coeffs",
        "thin_prism_coeffs",
    )
    missing = [name for name in required if not hasattr(parameters, name)]
    if missing:
        raise TypeError(f"source OpenCV pinhole parameters are missing fields: {missing}")
    result = {name: _jsonable(getattr(parameters, name)) for name in required}
    result["resolution"] = [int(value) for value in result["resolution"]]
    for name in (
        "principal_point",
        "focal_length",
        "radial_coeffs",
        "tangential_coeffs",
        "thin_prism_coeffs",
    ):
        result[name] = [float(value) for value in result[name]]
    return result


def fit_source_opencv_camera_parameters(
    source_parameters_by_camera: Mapping[str, Any],
    camera_ids: Sequence[str],
    *,
    margin_deg: float = 0.1,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, FThetaRasterCoverage],
    dict[str, str],
    dict[str, dict[str, Any]],
]:
    """Fit selected source OpenCV cameras and prepare native FTheta fields.

    The source calibration is read from the same NCore manifest that will be
    derived.  This prevents a calibration from another clip (or the opposite
    rear camera) from being silently substituted.
    """

    ordered_ids = [str(camera_id) for camera_id in camera_ids]
    if not ordered_ids or len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("camera_ids must contain a non-empty unique ordered list")
    missing = [camera_id for camera_id in ordered_ids if camera_id not in source_parameters_by_camera]
    if missing:
        raise ValueError(f"selected cameras missing from source intrinsics: {missing}")

    prepared: dict[str, dict[str, Any]] = {}
    coverage: dict[str, FThetaRasterCoverage] = {}
    source_fingerprints: dict[str, str] = {}
    source_calibrations: dict[str, dict[str, Any]] = {}
    for camera_id in ordered_ids:
        source = opencv_pinhole_parameters_to_dict(source_parameters_by_camera[camera_id])
        canonical = json.dumps(source, sort_keys=True, separators=(",", ":"), allow_nan=False)
        source_fingerprints[camera_id] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        source_calibrations[camera_id] = source
        fitted = _jsonable(fit_ftheta_from_opencv_rational(source))
        prepared[camera_id], coverage[camera_id] = validate_and_expand_raster_max_angle(
            fitted,
            margin_deg=margin_deg,
        )
        prepared[camera_id] = _jsonable(prepared[camera_id])
    return prepared, coverage, source_fingerprints, source_calibrations


def prepare_ftheta_conversion_parameters_from_manifest(
    source_manifest: str | Path,
    camera_ids: Sequence[str],
    *,
    margin_deg: float = 0.1,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, FThetaRasterCoverage],
    dict[str, str],
    dict[str, dict[str, Any]],
]:
    """Read selected OpenCV calibrations from NCore and fit them offline."""

    import ncore.data as ncore_data
    import ncore.data.v4 as ncore_v4

    manifest = Path(source_manifest).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"source NCore manifest not found: {manifest}")
    reader = ncore_v4.SequenceComponentGroupsReader([manifest])
    intrinsics_readers = reader.open_component_readers(ncore_v4.IntrinsicsComponent.Reader)
    if "default" not in intrinsics_readers:
        raise ValueError("source NCore manifest has no default intrinsics component")
    intrinsics = intrinsics_readers["default"]
    source_parameters: dict[str, Any] = {}
    for camera_id in camera_ids:
        parameters = intrinsics.get_camera_model_parameters(camera_id)
        if not isinstance(parameters, ncore_data.OpenCVPinholeCameraModelParameters):
            raise TypeError(
                f"camera '{camera_id}' source model must be OpenCVPinholeCameraModelParameters, "
                f"got {type(parameters).__name__}"
            )
        source_parameters[camera_id] = parameters
    return fit_source_opencv_camera_parameters(
        source_parameters,
        camera_ids,
        margin_deg=margin_deg,
    )


def validate_native_ftheta_model_raster(
    camera_model: Any,
    *,
    camera_id: str,
    width: int,
    height: int,
    rows_per_chunk: int = 64,
) -> dict[str, Any]:
    """Exercise the exact native NCore inverse/forward path for every pixel."""

    if rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")
    x = np.arange(width, dtype=np.int32)
    validated_pixels = 0
    for y0 in range(0, height, rows_per_chunk):
        y = np.arange(y0, min(y0 + rows_per_chunk, height), dtype=np.int32)
        xx, yy = np.meshgrid(x, y)
        pixels = np.stack((xx.ravel(), yy.ravel()), axis=-1)
        rays = camera_model.pixels_to_camera_rays(pixels)
        rays_array = rays.detach().cpu().numpy() if hasattr(rays, "detach") else np.asarray(rays)
        if not np.isfinite(rays_array).all():
            raise ValueError(f"camera '{camera_id}' native FTheta inverse produced non-finite rays")
        projected = camera_model.camera_rays_to_pixels(rays)
        valid = projected.valid_flag
        valid_array = valid.detach().cpu().numpy() if hasattr(valid, "detach") else np.asarray(valid)
        if not np.asarray(valid_array, dtype=bool).all():
            invalid = int(np.count_nonzero(~np.asarray(valid_array, dtype=bool)))
            raise ValueError(
                f"camera '{camera_id}' native FTheta forward path rejects {invalid} target pixels "
                f"in rows [{y0}, {min(y0 + rows_per_chunk, height)})"
            )
        roundtrip = projected.pixels
        roundtrip_array = (
            roundtrip.detach().cpu().numpy()
            if hasattr(roundtrip, "detach")
            else np.asarray(roundtrip)
        )
        if not np.array_equal(roundtrip_array, pixels):
            mismatch = int(np.count_nonzero(np.any(roundtrip_array != pixels, axis=-1)))
            raise ValueError(
                f"camera '{camera_id}' native FTheta pixel round-trip mismatches {mismatch} target pixels "
                f"in rows [{y0}, {min(y0 + rows_per_chunk, height)})"
            )
        validated_pixels += len(pixels)
    return {
        "total_pixels": width * height,
        "validated_pixels": validated_pixels,
        "coverage": validated_pixels / (width * height),
        "roundtrip_mismatches": 0,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: Path, target: Path, mode: Literal["hardlink", "symlink", "copy"]) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite derived-data target: {target}")
    if mode == "hardlink":
        try:
            os.link(source, target)
        except OSError as exc:
            raise OSError(
                f"cannot hardlink {source} -> {target}; source/output may be on "
                "different filesystems. Re-run in an empty output directory with "
                "--link-mode symlink or --link-mode copy"
            ) from exc
    elif mode == "symlink":
        target.symlink_to(os.path.relpath(source, target.parent))
    elif mode == "copy":
        shutil.copy2(source, target)
    else:  # pragma: no cover - Literal plus CLI choices make this defensive
        raise ValueError(f"unsupported link mode: {mode}")


def copy_nonintrinsics_components(source_root: Any, target_root: Any, *, zarr_module: Any) -> None:
    """Preserve every non-intrinsics component when a V4 store is shared.

    NCore may group poses, masks, and cuboids with intrinsics in one itar.
    That one rewritten store is the declared intrinsics-store exception; its
    non-intrinsics zarr groups are copied byte-for-byte at the array level.
    All other raw and aux store files remain linked and SHA-256-identical.
    """

    for component_name in source_root.keys():
        if component_name != "intrinsics":
            zarr_module.copy(
                source_root[component_name],
                target_root,
                name=component_name,
                if_exists="raise",
            )


def derive_native_ftheta_ncore_v4(
    *,
    source_manifest: str | Path,
    ftheta_artifact: str | Path | None = None,
    camera_ids: Sequence[str] | None = None,
    output_dir: str | Path,
    margin_deg: float = 0.1,
    link_mode: Literal["hardlink", "symlink", "copy"] = "hardlink",
) -> Path:
    """Create a non-mutating native-FTheta NCore V4 dataset.

    The store containing the original intrinsics is rewritten because NCore V4
    does not support per-component manifest overlays. Other manifest stores and
    adjacent aux stores are linked/copied unchanged.
    """

    # Keep NCore/zarr imports out of training and dependency-light unit tests.
    import ncore.data as ncore_data
    import ncore.data.v4 as ncore_v4
    import ncore.sensors as ncore_sensors
    import torch
    import zarr
    from ncore.impl.data import stores as ncore_stores
    from upath import UPath

    source_manifest = Path(source_manifest).expanduser().resolve()
    artifact_path = (
        Path(ftheta_artifact).expanduser().resolve()
        if ftheta_artifact is not None
        else None
    )
    output_dir = Path(output_dir).expanduser().resolve()
    if not source_manifest.is_file():
        raise FileNotFoundError(f"source NCore manifest not found: {source_manifest}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must not exist or must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if (artifact_path is None) == (camera_ids is None):
        raise ValueError("provide exactly one of ftheta_artifact or camera_ids")
    source_calibrations: dict[str, dict[str, Any]] | None = None
    if artifact_path is not None:
        prepared, coverage, input_fingerprints = prepare_ftheta_conversion_parameters(
            artifact_path,
            margin_deg=margin_deg,
        )
        parameter_source = "versioned-artifact"
    else:
        assert camera_ids is not None
        (
            prepared,
            coverage,
            input_fingerprints,
            source_calibrations,
        ) = prepare_ftheta_conversion_parameters_from_manifest(
            source_manifest,
            camera_ids,
            margin_deg=margin_deg,
        )
        parameter_source = "source-ncore-opencv-fit"
    manifest_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    if manifest_payload.get("version") != "v4":
        raise ValueError("only NCore V4 manifests are supported")
    component_stores = manifest_payload.get("component_stores")
    if not isinstance(component_stores, list) or not component_stores:
        raise ValueError("NCore manifest has no component stores")

    intrinsics_entries = [
        entry
        for entry in component_stores
        if "intrinsics" in (entry.get("components") or {})
        and "default" in entry["components"]["intrinsics"]
    ]
    if len(intrinsics_entries) != 1:
        raise ValueError(
            f"expected exactly one default intrinsics store, found {len(intrinsics_entries)}"
        )
    intrinsics_source = (source_manifest.parent / intrinsics_entries[0]["path"]).resolve()
    if not intrinsics_source.is_file():
        raise FileNotFoundError(f"intrinsics component store not found: {intrinsics_source}")

    source_store = ncore_stores.IndexedTarStore(UPath(intrinsics_source), mode="r")
    source_root = zarr.open_group(store=source_store, mode="r")
    source_group_name = str(source_root.attrs["component_group_name"])
    source_intrinsics_group = source_root["intrinsics"]["default"]
    source_camera_ids = set(source_intrinsics_group["cameras"].keys())
    missing_cameras = sorted(set(prepared) - source_camera_ids)
    if missing_cameras:
        source_store.close()
        raise ValueError(f"FTheta cameras missing from source intrinsics: {missing_cameras}")

    source_reader = ncore_v4.SequenceComponentGroupsReader([source_manifest])
    source_intrinsics_reader = source_reader.open_component_readers(
        ncore_v4.IntrinsicsComponent.Reader
    )["default"]
    native_roundtrip_reports: dict[str, dict[str, Any]] = {}
    for camera_id, parameters in prepared.items():
        source_parameters = source_intrinsics_reader.get_camera_model_parameters(camera_id)
        if not isinstance(source_parameters, ncore_data.OpenCVPinholeCameraModelParameters):
            source_store.close()
            raise TypeError(
                f"camera '{camera_id}' source model must be OpenCVPinholeCameraModelParameters, "
                f"got {type(source_parameters).__name__}"
            )
        source_resolution = [int(v) for v in source_parameters.resolution]
        if source_resolution != parameters["resolution"]:
            source_store.close()
            raise ValueError(
                f"camera '{camera_id}' resolution mismatch: source={source_resolution}, "
                f"artifact={parameters['resolution']}"
            )
        native_parameters = build_ftheta_camera_model_parameters(
            parameters, ncore_data=ncore_data
        )
        native_model = ncore_sensors.CameraModel.from_parameters(
            native_parameters,
            device="cpu",
            dtype=torch.float32,
        )
        if not isinstance(native_model, ncore_sensors.FThetaCameraModel):
            source_store.close()
            raise TypeError(
                f"camera '{camera_id}' conversion did not construct native FThetaCameraModel"
            )
        native_roundtrip_reports[camera_id] = validate_native_ftheta_model_raster(
            native_model,
            camera_id=camera_id,
            width=source_resolution[0],
            height=source_resolution[1],
        )

    interval = source_reader.sequence_timestamp_interval_us
    store_base_name = intrinsics_source.name.split(".ncore4", maxsplit=1)[0]
    writer = ncore_v4.SequenceComponentGroupsWriter(
        output_dir_path=UPath(output_dir),
        store_base_name=store_base_name,
        sequence_id=source_reader.sequence_id,
        sequence_timestamp_interval_us=interval,
        generic_meta_data=source_reader.generic_meta_data,
        store_type="itar",
    )
    target_root = writer.get_base_group(source_group_name or None)
    copy_nonintrinsics_components(source_root, target_root, zarr_module=zarr)
    intrinsics_writer = writer.register_component_writer(
        ncore_v4.IntrinsicsComponent.Writer,
        component_instance_name="default",
        group_name=source_group_name or None,
        generic_meta_data=dict(source_intrinsics_group.attrs.get("generic_meta_data", {})),
    )
    for camera_id in sorted(source_camera_ids):
        if camera_id in prepared:
            output_parameters = build_ftheta_camera_model_parameters(
                prepared[camera_id], ncore_data=ncore_data
            )
        else:
            output_parameters = source_intrinsics_reader.get_camera_model_parameters(camera_id)
        intrinsics_writer.store_camera_intrinsics(camera_id, output_parameters)
    for lidar_id in sorted(source_intrinsics_group["lidars"].keys()):
        lidar_parameters = source_intrinsics_reader.get_lidar_model_parameters(lidar_id)
        if lidar_parameters is not None:
            intrinsics_writer.store_lidar_intrinsics(lidar_id, lidar_parameters)
    rewritten_paths = [Path(str(path)) for path in writer.finalize()]
    source_store.close()
    if len(rewritten_paths) != 1:
        raise RuntimeError(f"expected one rewritten component store, got {rewritten_paths}")

    manifest_store_paths: list[Path] = [rewritten_paths[0]]
    unchanged_file_hashes: dict[str, dict[str, str]] = {}

    def link_unchanged(source_path: Path) -> None:
        target_path = output_dir / source_path.name
        if source_path.name in unchanged_file_hashes:
            raise ValueError(f"duplicate unchanged store filename: {source_path.name}")
        _link_or_copy(source_path, target_path, link_mode)
        source_hash = _sha256(source_path)
        derived_hash = _sha256(target_path)
        if source_hash != derived_hash:
            raise ValueError(f"unchanged store hash mismatch after derivation: {source_path.name}")
        unchanged_file_hashes[source_path.name] = {
            "source_sha256": source_hash,
            "derived_sha256": derived_hash,
        }

    manifest_source_paths = {
        (source_manifest.parent / entry["path"]).resolve() for entry in component_stores
    }
    for source_path in sorted(manifest_source_paths):
        if source_path == intrinsics_source:
            continue
        if not source_path.is_file():
            raise FileNotFoundError(f"component store not found: {source_path}")
        link_unchanged(source_path)
        manifest_store_paths.append(output_dir / source_path.name)

    # Aux components are intentionally not listed in the main manifest; the
    # dataset discovers them next to it. Mirror those sidecars as well.
    for source_path in sorted(source_manifest.parent.glob("*.zarr.itar")):
        resolved = source_path.resolve()
        if resolved in manifest_source_paths:
            continue
        link_unchanged(resolved)

    derived_reader = ncore_v4.SequenceComponentGroupsReader(manifest_store_paths)
    derived_intrinsics = derived_reader.open_component_readers(
        ncore_v4.IntrinsicsComponent.Reader
    )["default"]
    for camera_id, expected in prepared.items():
        actual = derived_intrinsics.get_camera_model_parameters(camera_id)
        if not isinstance(actual, ncore_data.FThetaCameraModelParameters):
            raise TypeError(
                f"derived camera '{camera_id}' is not native FTheta: {type(actual).__name__}"
            )
        if float(actual.max_angle) + 1e-7 < coverage[camera_id].raster_max_angle:
            raise ValueError(f"derived camera '{camera_id}' does not cover its target raster")
        derived_model = ncore_sensors.CameraModel.from_parameters(
            actual,
            device="cpu",
            dtype=torch.float32,
        )
        native_roundtrip_reports[camera_id] = validate_native_ftheta_model_raster(
            derived_model,
            camera_id=camera_id,
            width=int(actual.resolution[0]),
            height=int(actual.resolution[1]),
        )

    manifest_path = output_dir / source_manifest.name
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    derived_meta = derived_reader.get_sequence_meta().to_dict()
    derived_meta["component_stores"] = sorted(
        derived_meta["component_stores"], key=lambda entry: entry["path"]
    )
    temporary_manifest.write_text(
        json.dumps(derived_meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)

    parameters_path = output_dir / f"{source_manifest.stem}.ftheta-parameters.json"
    parameters_path.write_text(
        json.dumps(prepared, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    provenance_path = output_dir / f"{source_manifest.stem}.ftheta-derivation.json"
    fit_reports = (
        {
            camera_id: _fit_report(source_calibrations[camera_id], prepared[camera_id], coverage[camera_id])
            for camera_id in prepared
        }
        if source_calibrations is not None
        else {}
    )
    provenance = {
        "schema_version": 3,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "parameter_source": parameter_source,
        "camera_order": list(prepared),
        "ftheta_parameters": parameters_path.name,
        "ftheta_parameters_sha256": _sha256(parameters_path),
        "link_mode": link_mode,
        "source_calibration_fingerprints": input_fingerprints,
        "coverage": {camera_id: asdict(value) for camera_id, value in coverage.items()},
        "native_raster_roundtrip": native_roundtrip_reports,
        "fit_reports": fit_reports,
        "unchanged_file_hashes": unchanged_file_hashes,
    }
    if artifact_path is not None:
        provenance["input_ftheta_artifact"] = str(artifact_path)
        provenance["input_ftheta_artifact_sha256"] = _sha256(artifact_path)
    if source_calibrations is not None:
        provenance["source_calibrations"] = source_calibrations
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path
