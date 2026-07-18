#!/usr/bin/env python3
"""Fail-closed, read-only readiness checks for an NCore V4 clip.

The production path opens every manifest component and aux ``zarr.itar`` via
the same IndexedTarStore helper used by the dataset.  The opener and sequence
loader factory are injectable so the contract remains testable without the
internal NCore SDK.
"""

from __future__ import annotations

import io
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


StoreOpener = Callable[[Path], Any]
SequenceLoaderFactory = Callable[[Path], Any]

_AUX_GROUPS = {
    "sseg": "aux/semantic_segmentation",
    "lidar-sseg": "aux/lidar_semantic_segmentation",
    "lidar-camvis": "aux/lidar_camera_visibility",
}
V4_REQUIRED_AUX_TYPES = ("sseg", "egomask", "lidar-sseg", "lidar-camvis")
V4_MULTILAYER_READINESS_PROFILE = "v4-multilayer"
_KNOWN_AUX_TYPES = frozenset(V4_REQUIRED_AUX_TYPES)

V4_MULTILAYER_PROFILE_CONTRACT = {
    "config": {
        "path": "configs/apps/ncore_3dgut_mcmc_multilayer_inceptio_7cam_v4.yaml",
        "sha256": "92434503692ce0298fe08502c58dfcdafe5972b74515a991c0be731e614b647e",
    },
    "runtime_artifact": {
        "path": "scripts/pin_ftheta_b6a9_7cam_params_v4_full_domain.json",
        "sha256": "e637b5845302edaa940b10671b31d4b7d29a727eeb358f98249ac5334d459fbd",
    },
    "provenance_sidecar": {
        "path": "scripts/pin_ftheta_b6a9_7cam_params_v4_full_domain.provenance.json",
        "sha256": "df3d51f371b59f6c7b30e99bd909dc678eea9f3df19e088e1f3245a4bee5a981",
    },
    "survey_artifact": {
        "path": "scripts/pin_ftheta_b6a9_9cam_survey_v4_full_domain.json",
        "sha256": "08087b1fee6f1bb5a9935c509d493bebfb57be53560891f526a087f1552ac00c",
    },
    "clip_id": "inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9",
    "manifest_sha256": "df2021203cfe318cfa8da3462e38c5b7fbf6bf3963d3a8149d145f98f6036e31",
    "fitter_version": "pin-ftheta-numpy-v4-full-calibration-domain-2026-07-18",
    "generation_command_prefix": (
        ".venv/bin/python scripts/export_9cam_ftheta_params.py "
        "--calibrations scripts/pin_ftheta_b6a9_calibs.json "
        "--survey-output scripts/pin_ftheta_b6a9_9cam_survey_v4_full_domain.json "
        "--runtime-output scripts/pin_ftheta_b6a9_7cam_params_v4_full_domain.json "
        "--provenance-output scripts/pin_ftheta_b6a9_7cam_params_v4_full_domain.provenance.json "
        "--generated-at "
    ),
    "camera_order": (
        "camera_front_wide_120fov",
        "camera_cross_left_120fov",
        "camera_cross_right_120fov",
        "camera_left_wide_90fov",
        "camera_right_wide_90fov",
        "camera_back_rear_wide_90fov",
        "camera_rear_left_70fov",
    ),
    "sources": {
        "scripts/export_9cam_ftheta_params.py": "ae955ad754de1630fd6d53a035c5a0cfbe0c6d4ed80d9bc972c3c18d76c4b19d",
        "scripts/pin_ftheta_b6a9_calibs.json": "80be88487dc34253dd14ffeaffb2aa9a0962469faf4087b56fd0a4af1f78d62d",
        "scripts/pin_ftheta_camera_survey.py": "ef11796801c3b0db18ca2859f9d65a0fefee2d71a5c347c145335ca5b9c09a37",
        "threedgrut/ftheta_override_contract.py": "48b23146d8aa53f9e9af2efa88ed33d5d186400f56a7a911703ea8cc5b162f6d",
        "threedgrut_playground/utils/ftheta_fitter.py": "558a9e55be082b63045f22e35479daeb11360ff70b279676ff8664deca8b2aba",
        "threedgrut_playground/utils/opencv_inverse.py": "56a74a975447ca0e4178adfde88b21411a0f18c1342219c82927bf450acba8e9",
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"v4-multilayer profile file is missing or unreadable: {path}: {exc}") from exc
    return digest.hexdigest()


def validate_v4_multilayer_profile_contract(
    repo_root: str | Path,
    config_path: str | Path,
    artifact_path: str | Path,
) -> dict[str, Path]:
    """Bind the v4 readiness profile to its immutable config/artifacts.

    This is intentionally independent of the NCore store scan so callers can
    reject stale provenance and legacy v3 inputs in the cheap preflight stage.
    """

    root = Path(repo_root).expanduser().resolve()
    contract = V4_MULTILAYER_PROFILE_CONTRACT
    resolved: dict[str, Path] = {}
    for name in ("config", "runtime_artifact", "provenance_sidecar", "survey_artifact"):
        record = contract[name]
        expected = (root / record["path"]).resolve()
        resolved[name] = expected
        if name == "config":
            supplied = Path(config_path).expanduser().resolve()
            if supplied != expected:
                raise ValueError(
                    f"v4-multilayer config path mismatch: expected={expected} actual={supplied}"
                )
        elif name == "runtime_artifact":
            supplied = Path(artifact_path).expanduser().resolve()
            if supplied != expected:
                raise ValueError(
                    f"v4-multilayer runtime artifact path mismatch: expected={expected} actual={supplied}"
                )
        actual_sha256 = _sha256_file(expected)
        if actual_sha256 != record["sha256"]:
            raise ValueError(
                f"v4-multilayer {name} SHA-256 mismatch: "
                f"expected={record['sha256']} actual={actual_sha256} path={expected}"
            )

    sidecar_path = resolved["provenance_sidecar"]
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"v4-multilayer provenance sidecar is unreadable: {sidecar_path}: {exc}") from exc
    expected_scalars = {
        "schema_version": 1,
        "clip_id": contract["clip_id"],
        "manifest_sha256": contract["manifest_sha256"],
        "fitter_version": contract["fitter_version"],
    }
    for key, expected in expected_scalars.items():
        if sidecar.get(key) != expected:
            raise ValueError(
                f"v4-multilayer provenance sidecar {key} mismatch: "
                f"expected={expected!r} actual={sidecar.get(key)!r}"
            )
    generated_at = sidecar.get("generated_at")
    try:
        generated_time = datetime.fromisoformat(generated_at)
    except (TypeError, ValueError) as exc:
        raise ValueError("v4-multilayer provenance sidecar generated_at is not valid ISO-8601") from exc
    if generated_time.tzinfo is None:
        raise ValueError("v4-multilayer provenance sidecar generated_at must include a timezone")
    expected_command = contract["generation_command_prefix"] + generated_at
    if sidecar.get("generation_command") != expected_command:
        raise ValueError("v4-multilayer provenance sidecar generation_command mismatch")
    if tuple(sidecar.get("camera_order", ())) != tuple(contract["camera_order"]):
        raise ValueError("v4-multilayer provenance sidecar camera_order mismatch")
    expected_artifacts = {
        contract["runtime_artifact"]["path"]: contract["runtime_artifact"]["sha256"],
        contract["survey_artifact"]["path"]: contract["survey_artifact"]["sha256"],
    }
    if sidecar.get("artifacts") != expected_artifacts:
        raise ValueError("v4-multilayer provenance sidecar artifacts mapping mismatch")
    if sidecar.get("sources") != contract["sources"]:
        raise ValueError("v4-multilayer provenance sidecar sources mapping mismatch")
    for relative_path, expected_sha256 in contract["sources"].items():
        source_path = (root / relative_path).resolve()
        actual_sha256 = _sha256_file(source_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"v4-multilayer provenance source SHA-256 mismatch for {relative_path}: "
                f"expected={expected_sha256} actual={actual_sha256}"
            )
    return resolved


def validate_v4_multilayer_dataset_contract(manifest_path: str | Path) -> dict[str, Any]:
    """Freeze the canonical b6a9 manifest before the expensive store scan."""

    path = Path(manifest_path).expanduser().resolve()
    expected_sha256 = V4_MULTILAYER_PROFILE_CONTRACT["manifest_sha256"]
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"v4-multilayer canonical manifest SHA-256 mismatch: "
            f"expected={expected_sha256} actual={actual_sha256} path={path}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"v4-multilayer canonical manifest is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("v4-multilayer canonical manifest root must be a mapping")
    if value.get("sequence_id") != V4_MULTILAYER_PROFILE_CONTRACT["clip_id"]:
        raise ValueError("v4-multilayer canonical manifest sequence_id mismatch")
    component_stores = value.get("component_stores")
    if not isinstance(component_stores, (list, dict)) or len(component_stores) != 14:
        actual_count = len(component_stores) if isinstance(component_stores, (list, dict)) else None
        raise ValueError(
            f"v4-multilayer canonical manifest must reference exactly 14 component stores; got {actual_count}"
        )
    return value


def _default_store_opener(path: Path) -> Any:
    from threedgrut.datasets.aux_readers import _open_itar_zarr

    return _open_itar_zarr(path)


def _default_sequence_loader_factory(manifest_path: Path) -> Any:
    import ncore.data.v4 as ncore_v4

    reader = ncore_v4.SequenceComponentGroupsReader([str(manifest_path)], open_consolidated=False)
    return ncore_v4.SequenceLoaderV4(reader)


def _component_store_paths(manifest: dict, manifest_path: Path) -> list[Path]:
    stores = manifest.get("component_stores")
    entries: list[tuple[str, Any]]
    if isinstance(stores, list) and stores:
        entries = [(str(index), entry) for index, entry in enumerate(stores)]
    elif isinstance(stores, dict) and stores:
        entries = [(str(name), entry) for name, entry in stores.items()]
    else:
        raise ValueError("NCore readiness: manifest component_stores must be a non-empty list or mapping")

    paths: list[Path] = []
    for label, entry in entries:
        raw_path = entry.get("path") if isinstance(entry, dict) else entry
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"NCore readiness: component_store[{label}] has no valid path")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = manifest_path.parent / path
        paths.append(path.resolve())
    if len(set(paths)) != len(paths):
        raise ValueError("NCore readiness: manifest references duplicate component store paths")
    return paths


def _group_keys(group: Any, *, context: str) -> list[str]:
    try:
        return sorted(str(key) for key in group.group_keys())
    except Exception as exc:
        raise ValueError(f"NCore readiness: {context} cannot enumerate groups: {exc}") from exc


def _array_keys(group: Any, *, context: str) -> list[str]:
    try:
        return sorted(str(key) for key in group.array_keys())
    except Exception as exc:
        raise ValueError(f"NCore readiness: {context} cannot enumerate arrays: {exc}") from exc


def _read_array(array: Any, *, context: str) -> Any:
    try:
        shape = tuple(array.shape)
        value = array[()] if not shape else array[tuple(0 for _ in shape)]
    except Exception as exc:
        raise ValueError(f"NCore readiness: {context} representative array is unreadable: {exc}") from exc
    if value is None or isinstance(value, (bytes, bytearray)) and not value:
        raise ValueError(f"NCore readiness: {context} representative array is empty")
    return value


def _first_array(root: Any, *, context: str) -> tuple[str, Any]:
    stack: list[tuple[str, Any]] = [("", root)]
    while stack:
        prefix, group = stack.pop()
        arrays = _array_keys(group, context=f"{context}/{prefix}".rstrip("/"))
        if arrays:
            key = arrays[0]
            try:
                return f"{prefix}{key}", group[key]
            except Exception as exc:
                raise ValueError(f"NCore readiness: {context}/{prefix}{key} cannot be opened: {exc}") from exc
        for key in reversed(_group_keys(group, context=f"{context}/{prefix}".rstrip("/"))):
            try:
                child = group[key]
            except Exception as exc:
                raise ValueError(f"NCore readiness: {context}/{prefix}{key} cannot be opened: {exc}") from exc
            stack.append((f"{prefix}{key}/", child))
    raise ValueError(f"NCore readiness: {context} contains no readable arrays")


def _open_store(path: Path, opener: StoreOpener, *, context: str) -> Any:
    if not path.is_file():
        raise ValueError(f"NCore readiness: {context} missing: {path}")
    try:
        root = opener(path)
    except Exception as exc:
        raise ValueError(f"NCore readiness: {context} cannot open {path}: {exc}") from exc
    array_path, array = _first_array(root, context=context)
    _read_array(array, context=f"{context}/{array_path}")
    return root


def _get_group(root: Any, path: str, *, context: str) -> Any:
    try:
        return root[path]
    except Exception as exc:
        raise ValueError(f"NCore readiness: {context} missing group {path!r}: {exc}") from exc


def _camera_group_keys(group: Any, camera_ids: Iterable[str], *, context: str) -> dict[str, tuple[str, ...]]:
    """Return exact per-camera array keys after reading one payload per camera."""

    available = set(_group_keys(group, context=context))
    inventories: dict[str, tuple[str, ...]] = {}
    for camera_id in camera_ids:
        if camera_id not in available:
            raise ValueError(f"NCore readiness: {context} missing active camera {camera_id}")
        try:
            camera_group = group[camera_id]
        except Exception as exc:
            raise ValueError(f"NCore readiness: {context}/{camera_id} cannot open: {exc}") from exc
        keys = tuple(_array_keys(camera_group, context=f"{context}/{camera_id}"))
        if not keys:
            raise ValueError(f"NCore readiness: {context}/{camera_id} has zero frames")
        try:
            representative = camera_group[keys[0]]
        except Exception as exc:
            raise ValueError(f"NCore readiness: {context}/{camera_id}/{keys[0]} cannot open: {exc}") from exc
        _read_array(representative, context=f"{context}/{camera_id}/{keys[0]}")
        inventories[camera_id] = keys
    return inventories


def _sensor_group_keys(group: Any, *, context: str) -> dict[str, tuple[str, ...]]:
    """Read one payload from every sensor subgroup and return exact keys."""

    sensor_ids = _group_keys(group, context=context)
    if not sensor_ids:
        raise ValueError(f"NCore readiness: {context} has no lidar groups")
    inventories: dict[str, tuple[str, ...]] = {}
    for sensor_id in sensor_ids:
        try:
            sensor_group = group[sensor_id]
        except Exception as exc:
            raise ValueError(f"NCore readiness: {context}/{sensor_id} cannot open: {exc}") from exc
        keys = _array_keys(sensor_group, context=f"{context}/{sensor_id}")
        if not keys:
            raise ValueError(f"NCore readiness: {context}/{sensor_id} has zero frames")
        try:
            array = sensor_group[keys[0]]
        except Exception as exc:
            raise ValueError(f"NCore readiness: {context}/{sensor_id}/{keys[0]} cannot open: {exc}") from exc
        _read_array(array, context=f"{context}/{sensor_id}/{keys[0]}")
        inventories[sensor_id] = tuple(keys)
    return inventories


def _raw_point_cloud_inventory(loader: Any) -> dict[str, dict[str, Any]]:
    """Read every raw point cloud and return exact timestamps/point counts."""

    try:
        source_ids = list(loader.lidar_ids) + list(loader.point_clouds_ids)
    except Exception as exc:
        raise ValueError(f"NCore readiness: cannot enumerate raw point-cloud sources: {exc}") from exc
    source_ids = list(dict.fromkeys(str(source_id) for source_id in source_ids))
    if not source_ids:
        raise ValueError("NCore readiness: raw sequence has no point-cloud sources")
    inventories: dict[str, dict[str, Any]] = {}
    for source_id in source_ids:
        try:
            source = loader.get_point_clouds_source(source_id)
            timestamps = np.asarray(source.pc_timestamps_us).reshape(-1)
        except Exception as exc:
            raise ValueError(f"NCore readiness: raw point-cloud source {source_id} is unavailable: {exc}") from exc
        keys = tuple(str(int(value)) for value in timestamps)
        if not keys:
            raise ValueError(f"NCore readiness: raw point-cloud source {source_id} has zero frames")
        if len(set(keys)) != len(keys):
            raise ValueError(f"NCore readiness: raw point-cloud source {source_id} has duplicate timestamps")
        point_counts: dict[str, int] = {}
        for index, key in enumerate(keys):
            try:
                point_cloud = source.get_pc(index)
                xyz = np.asarray(point_cloud.xyz)
            except Exception as exc:
                raise ValueError(
                    f"NCore readiness: raw point cloud {source_id}/{key} is unreadable: {exc}"
                ) from exc
            if xyz.ndim != 2 or xyz.shape[0] <= 0 or xyz.shape[1] < 3:
                raise ValueError(
                    f"NCore readiness: raw point cloud {source_id}/{key} has invalid xyz shape {xyz.shape}"
                )
            point_counts[key] = int(xyz.shape[0])
        inventories[source_id] = {"keys": keys, "point_counts": point_counts}
    return inventories


def _scalar_png_array(array: Any, *, context: str) -> np.ndarray:
    """Decode one scalar PNG payload into a nonempty numpy image."""

    try:
        shape = tuple(int(value) for value in array.shape)
    except Exception as exc:
        raise ValueError(f"NCore readiness: {context} has no readable shape: {exc}") from exc
    if shape:
        raise ValueError(f"NCore readiness: {context} must be a scalar PNG payload, got shape={shape}")
    payload = _read_array(array, context=context)
    if isinstance(payload, np.ndarray) and payload.shape == ():
        payload = payload.item()
    if isinstance(payload, np.bytes_):
        payload = bytes(payload)
    if not isinstance(payload, (bytes, bytearray)):
        raise ValueError(f"NCore readiness: {context} scalar payload is not PNG bytes")
    try:
        from PIL import Image

        with Image.open(io.BytesIO(bytes(payload))) as image:
            image.load()
            decoded = np.asarray(image)
    except (OSError, ValueError) as exc:
        raise ValueError(f"NCore readiness: {context} PNG payload is unreadable: {exc}") from exc
    if decoded.size <= 0:
        raise ValueError(f"NCore readiness: {context} decoded PNG has zero points")
    return decoded


def _camvis_point_count(array: Any, *, context: str) -> int:
    """Read a complete camvis payload and return its point-axis length."""

    try:
        shape = tuple(int(value) for value in array.shape)
    except Exception as exc:
        raise ValueError(f"NCore readiness: {context} has no readable shape: {exc}") from exc
    if not shape or any(value <= 0 for value in shape):
        raise ValueError(f"NCore readiness: {context} has invalid/nonempty-required shape {shape}")
    try:
        value = np.asarray(array[...])
    except Exception as exc:
        raise ValueError(f"NCore readiness: {context} payload is unreadable: {exc}") from exc
    if value.shape != shape:
        raise ValueError(
            f"NCore readiness: {context} payload shape mismatch: metadata={shape} decoded={value.shape}"
        )
    return int(value.shape[0])


def _decode_image_resolution(array: Any, *, context: str) -> tuple[int, int]:
    """Decode a representative aux image and return ``(width, height)``.

    NRE normally stores sseg/egomask images as scalar PNG byte arrays.  Older
    egomask stores may contain a plain HxW/HxWC array, which is also accepted
    after a representative pixel read. Pillow is imported lazily so this gate
    remains lightweight for callers that do not request image aux components.
    """

    try:
        shape = tuple(int(value) for value in array.shape)
    except Exception as exc:
        raise ValueError(f"NCore readiness: {context} has no readable shape: {exc}") from exc
    if len(shape) in (2, 3):
        _read_array(array, context=context)
        height, width = shape[0], shape[1]
        if width <= 0 or height <= 0:
            raise ValueError(f"NCore readiness: {context} image shape is empty: {shape}")
        if len(shape) == 3 and shape[2] not in (1, 3, 4):
            raise ValueError(f"NCore readiness: {context} image channel dimension is invalid: {shape}")
        return width, height
    if shape:
        raise ValueError(f"NCore readiness: {context} image array must be scalar PNG bytes, HxW, or HxWC, got {shape}")

    payload = _read_array(array, context=context)
    if isinstance(payload, np.ndarray) and payload.shape == ():
        payload = payload.item()
    if isinstance(payload, np.bytes_):
        payload = bytes(payload)
    if not isinstance(payload, (bytes, bytearray)):
        raise ValueError(f"NCore readiness: {context} scalar image payload is not PNG bytes")
    try:
        from PIL import Image

        with Image.open(io.BytesIO(bytes(payload))) as image:
            image.load()
            width, height = image.size
    except (OSError, ValueError) as exc:
        raise ValueError(f"NCore readiness: {context} PNG payload is unreadable: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"NCore readiness: {context} decoded PNG has empty resolution")
    return int(width), int(height)


def _raw_camera_resolution(sensor: Any, *, camera_id: str) -> tuple[int, int] | None:
    """Decode one raw frame on CPU when the sensor exposes that safe API."""

    get_image = getattr(sensor, "get_frame_image_array", None)
    if callable(get_image):
        try:
            image = np.asarray(get_image(0))
        except Exception as exc:
            raise ValueError(
                f"NCore readiness: active raw camera {camera_id} representative frame is unreadable: {exc}"
            ) from exc
        if image.ndim < 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError(
                f"NCore readiness: active raw camera {camera_id} representative frame has invalid shape {image.shape}"
            )
        return int(image.shape[1]), int(image.shape[0])

    model_parameters = getattr(sensor, "model_parameters", None)
    resolution = getattr(model_parameters, "resolution", None)
    if resolution is None:
        raise ValueError(
            f"NCore readiness: active raw camera {camera_id} exposes neither a representative image nor resolution"
        )
    try:
        values = np.asarray(resolution).reshape(-1)
        width, height = int(values[0]), int(values[1])
    except Exception as exc:
        raise ValueError(f"NCore readiness: active raw camera {camera_id} resolution is unreadable: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"NCore readiness: active raw camera {camera_id} resolution is invalid: {(width, height)}")
    return width, height


def _validate_camera_image_resolutions(
    group: Any,
    camera_keys: dict[str, tuple[str, ...]],
    raw_resolutions: dict[str, tuple[int, int] | None],
    *,
    context: str,
) -> dict[str, tuple[int, int]]:
    decoded: dict[str, tuple[int, int]] = {}
    for camera_id, keys in camera_keys.items():
        array_context = f"{context}/{camera_id}/{keys[0]}"
        try:
            array = group[camera_id][keys[0]]
        except Exception as exc:
            raise ValueError(f"NCore readiness: {array_context} cannot open: {exc}") from exc
        resolution = _decode_image_resolution(array, context=array_context)
        expected = raw_resolutions[camera_id]
        if expected is not None and resolution != expected:
            raise ValueError(
                f"NCore readiness: {context} resolution mismatch for {camera_id}: "
                f"raw={expected[0]}x{expected[1]} aux={resolution[0]}x{resolution[1]}"
            )
        decoded[camera_id] = resolution
    return decoded


def _egomask_camera_group(root: Any, *, context: str) -> Any:
    """Return the project-format static egomask camera parent.

    Current NRE output uses ``aux/egomask/<camera>/0``.  A small generic
    fallback mirrors :class:`EgomaskAuxReader` for older group-name variants.
    """

    try:
        return root["aux/egomask"]
    except KeyError:
        pass
    except Exception as exc:
        raise ValueError(f"NCore readiness: {context} canonical group 'aux/egomask' is unreadable: {exc}") from exc
    stack = [root]
    while stack:
        group = stack.pop()
        children = _group_keys(group, context=context)
        child_groups = []
        for key in children:
            try:
                child_groups.append((key, group[key]))
            except Exception as exc:
                raise ValueError(f"NCore readiness: {context}/{key} cannot open: {exc}") from exc
        if child_groups and all(_array_keys(child, context=f"{context}/{key}") for key, child in child_groups):
            return group
        stack.extend(child for _, child in child_groups)
    raise ValueError(f"NCore readiness: {context} has no per-camera static-mask groups")


def _discover_aux_path(clip_dir: Path, aux_type: str) -> Path:
    matches = sorted(clip_dir.glob(f"*.aux.{aux_type}.zarr.itar"))
    if not matches:
        raise ValueError(f"NCore readiness: required aux.{aux_type} store missing in {clip_dir}")
    if len(matches) != 1:
        raise ValueError(f"NCore readiness: required aux.{aux_type} store is ambiguous: {matches}")
    return matches[0].resolve()


def validate_ncore_data_readiness(
    manifest_path: str | Path,
    active_camera_ids: Iterable[str],
    *,
    required_aux: Iterable[str],
    store_opener: StoreOpener | None = None,
    sequence_loader_factory: SequenceLoaderFactory | None = None,
) -> dict[str, Any]:
    """Validate raw and aux stores without creating output or touching a GPU."""

    path = Path(manifest_path).expanduser().resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"NCore readiness: cannot read manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("NCore readiness: manifest root must be a mapping")
    camera_ids = list(active_camera_ids)
    if not camera_ids or len(set(camera_ids)) != len(camera_ids):
        raise ValueError("NCore readiness: active camera IDs must be non-empty and unique")
    required_aux_types = tuple(required_aux)
    if len(set(required_aux_types)) != len(required_aux_types) or not set(required_aux_types) <= _KNOWN_AUX_TYPES:
        raise ValueError(
            "NCore readiness: required_aux must contain unique known aux types; "
            f"got {required_aux_types}"
        )

    opener = store_opener or _default_store_opener
    component_paths = _component_store_paths(manifest, path)
    for index, component_path in enumerate(component_paths):
        _open_store(component_path, opener, context=f"component store[{index}]")

    loader_factory = sequence_loader_factory or _default_sequence_loader_factory
    try:
        loader = loader_factory(path)
    except Exception as exc:
        raise ValueError(f"NCore readiness: NCore sequence loader cannot open manifest: {exc}") from exc
    raw_counts: dict[str, int] = {}
    raw_timestamp_keys: dict[str, tuple[str, ...]] = {}
    raw_resolutions: dict[str, tuple[int, int] | None] = {}
    for camera_id in camera_ids:
        try:
            sensor = loader.get_camera_sensor(camera_id)
            timestamps = np.asarray(sensor.frames_timestamps_us)
        except Exception as exc:
            raise ValueError(f"NCore readiness: active raw camera {camera_id} is unavailable: {exc}") from exc
        count = int(timestamps.shape[0]) if timestamps.ndim >= 1 else 0
        if count <= 0:
            raise ValueError(f"NCore readiness: active raw camera {camera_id} has zero frames")
        # Dataset and NRE aux readers key camera labels by the frame END
        # timestamp. It is the final timepoint column in the V4 Nx2 matrix.
        timestamp_values = timestamps[:, -1] if timestamps.ndim >= 2 else timestamps
        keys = tuple(str(int(value)) for value in np.asarray(timestamp_values).reshape(-1))
        if len(set(keys)) != len(keys):
            raise ValueError(f"NCore readiness: active raw camera {camera_id} has duplicate frame timestamps")
        raw_counts[camera_id] = count
        raw_timestamp_keys[camera_id] = keys
        raw_resolutions[camera_id] = _raw_camera_resolution(sensor, camera_id=camera_id)

    aux_roots: dict[str, Any] = {}
    aux_paths: dict[str, Path] = {}
    for aux_type in required_aux_types:
        aux_path = _discover_aux_path(path.parent, aux_type)
        aux_paths[aux_type] = aux_path
        aux_roots[aux_type] = _open_store(aux_path, opener, context=f"aux.{aux_type} store")

    sseg_counts: dict[str, int] = {}
    sseg_resolutions: dict[str, tuple[int, int]] = {}
    if "sseg" in aux_roots:
        sseg_group = _get_group(aux_roots["sseg"], _AUX_GROUPS["sseg"], context="aux.sseg store")
        sseg_keys = _camera_group_keys(sseg_group, camera_ids, context="aux.sseg")
        sseg_counts = {camera_id: len(keys) for camera_id, keys in sseg_keys.items()}
        for camera_id in camera_ids:
            if sseg_counts[camera_id] != raw_counts[camera_id]:
                raise ValueError(
                    f"NCore readiness: raw/sseg frame-count mismatch for {camera_id}: "
                    f"raw={raw_counts[camera_id]} sseg={sseg_counts[camera_id]}"
                )
            raw_key_set = set(raw_timestamp_keys[camera_id])
            sseg_key_set = set(sseg_keys[camera_id])
            if sseg_key_set != raw_key_set:
                missing = sorted(raw_key_set - sseg_key_set)[:3]
                unexpected = sorted(sseg_key_set - raw_key_set)[:3]
                raise ValueError(
                    f"NCore readiness: raw/sseg timestamp-key mismatch for {camera_id}: "
                    f"missing={missing} unexpected={unexpected}"
                )
        sseg_resolutions = _validate_camera_image_resolutions(
            sseg_group, sseg_keys, raw_resolutions, context="aux.sseg"
        )

    # Egomasks are static in this project: one or more arrays per camera are
    # valid, and their count intentionally does not have to equal raw frames.
    egomask_counts: dict[str, int] = {}
    egomask_resolutions: dict[str, tuple[int, int]] = {}
    if "egomask" in aux_roots:
        egomask_group = _egomask_camera_group(aux_roots["egomask"], context="aux.egomask")
        egomask_keys = _camera_group_keys(egomask_group, camera_ids, context="aux.egomask")
        egomask_counts = {camera_id: len(keys) for camera_id, keys in egomask_keys.items()}
        egomask_resolutions = _validate_camera_image_resolutions(
            egomask_group, egomask_keys, raw_resolutions, context="aux.egomask"
        )

    lidar_counts: dict[str, dict[str, int]] = {}
    lidar_keys: dict[str, dict[str, tuple[str, ...]]] = {}
    lidar_groups: dict[str, Any] = {}
    for aux_type in ("lidar-sseg", "lidar-camvis"):
        if aux_type not in aux_roots:
            continue
        group = _get_group(aux_roots[aux_type], _AUX_GROUPS[aux_type], context=f"aux.{aux_type} store")
        lidar_groups[aux_type] = group
        lidar_keys[aux_type] = _sensor_group_keys(group, context=f"aux.{aux_type}")
        lidar_counts[aux_type] = {sensor_id: len(keys) for sensor_id, keys in lidar_keys[aux_type].items()}
    if set(lidar_keys) == {"lidar-sseg", "lidar-camvis"}:
        if lidar_counts["lidar-sseg"] != lidar_counts["lidar-camvis"]:
            raise ValueError(
                "NCore readiness: lidar aux frame-count mismatch: "
                f"lidar-sseg={lidar_counts['lidar-sseg']} lidar-camvis={lidar_counts['lidar-camvis']}"
            )
        for sensor_id in lidar_keys["lidar-sseg"]:
            sseg_key_set = set(lidar_keys["lidar-sseg"][sensor_id])
            camvis_key_set = set(lidar_keys["lidar-camvis"][sensor_id])
            if sseg_key_set != camvis_key_set:
                raise ValueError(
                    f"NCore readiness: lidar aux timestamp-key mismatch for {sensor_id}: "
                    f"lidar-sseg-only={sorted(sseg_key_set - camvis_key_set)[:3]} "
                    f"lidar-camvis-only={sorted(camvis_key_set - sseg_key_set)[:3]}"
                )
        raw_lidar_inventory = _raw_point_cloud_inventory(loader)
        for sensor_id, aux_keys in lidar_keys["lidar-sseg"].items():
            if sensor_id not in raw_lidar_inventory:
                raise ValueError(f"NCore readiness: lidar aux sensor {sensor_id} has no matching raw point-cloud source")
            raw_inventory = raw_lidar_inventory[sensor_id]
            raw_key_set = set(raw_inventory["keys"])
            aux_key_set = set(aux_keys)
            if aux_key_set != raw_key_set:
                raise ValueError(
                    f"NCore readiness: raw/lidar aux timestamp-key mismatch for {sensor_id}: "
                    f"missing={sorted(raw_key_set - aux_key_set)[:3]} "
                    f"unexpected={sorted(aux_key_set - raw_key_set)[:3]}"
                )
            for timestamp_key in aux_keys:
                try:
                    sseg_array = lidar_groups["lidar-sseg"][sensor_id][timestamp_key]
                    camvis_array = lidar_groups["lidar-camvis"][sensor_id][timestamp_key]
                except Exception as exc:
                    raise ValueError(
                        f"NCore readiness: lidar payload {sensor_id}/{timestamp_key} cannot open: {exc}"
                    ) from exc
                sseg_count = int(
                    _scalar_png_array(
                        sseg_array,
                        context=f"aux.lidar-sseg/{sensor_id}/{timestamp_key}",
                    ).size
                )
                camvis_count = _camvis_point_count(
                    camvis_array,
                    context=f"aux.lidar-camvis/{sensor_id}/{timestamp_key}",
                )
                raw_count = int(raw_inventory["point_counts"][timestamp_key])
                if raw_count != sseg_count or raw_count != camvis_count:
                    raise ValueError(
                        f"NCore readiness: lidar point-count mismatch for {sensor_id}/{timestamp_key}: "
                        f"raw={raw_count} lidar-sseg={sseg_count} lidar-camvis={camvis_count}"
                    )

    return {
        "manifest": str(path),
        "component_store_count": len(component_paths),
        "raw_camera_frame_counts": raw_counts,
        "raw_camera_resolutions": raw_resolutions,
        "sseg_frame_counts": sseg_counts,
        "sseg_resolutions": sseg_resolutions,
        "egomask_static_array_counts": egomask_counts,
        "egomask_resolutions": egomask_resolutions,
        "lidar_aux_frame_counts": lidar_counts,
        "aux_paths": {name: str(aux_paths[name]) for name in required_aux_types},
    }
