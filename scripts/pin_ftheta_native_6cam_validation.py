#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Lean CPU preflight for the matched six-camera Pinhole/FTheta experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from threedgrut.datasets.ftheta_derivation import (  # noqa: E402
    prepare_ftheta_conversion_parameters,
)


CONFIG_NAME = "apps/ncore_3dgut_mcmc_multilayer_inceptio_6cam_native_ab"
CAMERA_IDS = (
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
    "camera_back_rear_wide_90fov",
)
MODE = {
    "smoke": {"iterations": 5000, "duration_sec": 5.0},
    # The P baseline's source manifest is already a 20-second clip; -1 means
    # consume that complete clip and is therefore the exact resolved setting
    # recorded in its parsed.yaml.
    "full": {"iterations": 30000, "duration_sec": -1},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _derivation_paths(ftheta_manifest: Path) -> tuple[Path, Path, dict[str, Any]]:
    stem = ftheta_manifest.stem
    record_path = ftheta_manifest.parent / f"{stem}.ftheta-derivation.json"
    if not record_path.is_file():
        raise ValueError(f"FTheta derivation record is missing: {record_path}")
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"FTheta derivation record is unreadable: {record_path}: {exc}") from exc
    if record.get("schema_version") != 3:
        raise ValueError("FTheta derivation record must use schema_version=3")
    raw_parameters = record.get("ftheta_parameters")
    if not isinstance(raw_parameters, str) or not raw_parameters:
        raise ValueError("FTheta derivation record has no parameter artifact")
    parameters_path = ftheta_manifest.parent / raw_parameters
    if not parameters_path.is_file():
        raise ValueError(f"FTheta parameter artifact is missing: {parameters_path}")
    if _sha256(parameters_path) != record.get("ftheta_parameters_sha256"):
        raise ValueError("FTheta parameter artifact SHA-256 disagrees with derivation record")
    unchanged = record.get("unchanged_file_hashes")
    if not isinstance(unchanged, dict) or not unchanged:
        raise ValueError("FTheta derivation record is missing unchanged-store hashes")
    for filename, hashes in unchanged.items():
        if not isinstance(filename, str) or not isinstance(hashes, dict):
            raise ValueError("FTheta derivation record has malformed unchanged-store hashes")
        if hashes.get("source_sha256") != hashes.get("derived_sha256"):
            raise ValueError(f"FTheta derivation changed non-intrinsics store: {filename}")
    return record_path, parameters_path, record


def validate_native_pair(pin_manifest: str | Path, ftheta_manifest: str | Path) -> dict[str, Any]:
    """Require matching sequence identity and native model types for all six cameras."""

    import ncore.data as ncore_data
    import ncore.data.v4 as ncore_v4

    pin_path = Path(pin_manifest).expanduser().resolve()
    ftheta_path = Path(ftheta_manifest).expanduser().resolve()
    if pin_path == ftheta_path:
        raise ValueError("Pinhole and FTheta arms must use distinct manifests")
    if not pin_path.is_file() or not ftheta_path.is_file():
        raise ValueError("both Pinhole and FTheta manifests must exist")
    record_path, parameters_path, record = _derivation_paths(ftheta_path)
    if record.get("source_manifest_sha256") != _sha256(pin_path):
        raise ValueError("FTheta derivation source hash does not match the Pinhole manifest")
    if tuple(record.get("camera_order", ())) != CAMERA_IDS:
        raise ValueError("FTheta derivation camera order does not match the six-camera contract")
    expected, coverage, fingerprints = prepare_ftheta_conversion_parameters(parameters_path)
    if tuple(expected) != CAMERA_IDS:
        raise ValueError("FTheta parameter camera order does not match the six-camera contract")

    readers = {}
    for arm, manifest in (("P", pin_path), ("F", ftheta_path)):
        reader = ncore_v4.SequenceComponentGroupsReader([manifest])
        intrinsics = reader.open_component_readers(ncore_v4.IntrinsicsComponent.Reader)
        if "default" not in intrinsics:
            raise ValueError(f"Arm {arm} has no default intrinsics component")
        readers[arm] = (reader, intrinsics["default"])
    pin_reader, pin_intrinsics = readers["P"]
    ftheta_reader, ftheta_intrinsics = readers["F"]
    if pin_reader.sequence_id != ftheta_reader.sequence_id:
        raise ValueError("P/F sequence IDs differ")
    if pin_reader.sequence_timestamp_interval_us != ftheta_reader.sequence_timestamp_interval_us:
        raise ValueError("P/F sequence timestamp intervals differ")

    models: dict[str, dict[str, str]] = {}
    for camera_id in CAMERA_IDS:
        pin_parameters = pin_intrinsics.get_camera_model_parameters(camera_id)
        ftheta_parameters = ftheta_intrinsics.get_camera_model_parameters(camera_id)
        if not isinstance(pin_parameters, ncore_data.OpenCVPinholeCameraModelParameters):
            raise ValueError(
                f"Arm P camera {camera_id!r} is not OpenCV Pinhole: {type(pin_parameters).__name__}"
            )
        if not isinstance(ftheta_parameters, ncore_data.FThetaCameraModelParameters):
            raise ValueError(
                f"Arm F camera {camera_id!r} is not native FTheta: {type(ftheta_parameters).__name__}"
            )
        if tuple(int(value) for value in pin_parameters.resolution) != tuple(
            int(value) for value in ftheta_parameters.resolution
        ):
            raise ValueError(f"P/F resolution mismatch for {camera_id!r}")
        if float(ftheta_parameters.max_angle) <= coverage[camera_id].raster_max_angle:
            raise ValueError(f"FTheta max_angle does not cover the complete raster for {camera_id!r}")
        models[camera_id] = {
            "P": type(pin_parameters).__name__,
            "F": type(ftheta_parameters).__name__,
        }
    return {
        "sequence_id": pin_reader.sequence_id,
        "camera_ids": list(CAMERA_IDS),
        "models": models,
        "parameter_fingerprints": fingerprints,
        "derivation_record": str(record_path),
        "parameters": str(parameters_path),
    }


def _leaf_differences(left: Any, right: Any, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: set[str] = set()
        for key in set(left) | set(right):
            dotted = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                differences.add(dotted)
            else:
                differences.update(_leaf_differences(left[key], right[key], dotted))
        return differences
    return set() if left == right else {prefix}


def validate_resolved_configs(
    mode: str,
    pin_manifest: str | Path,
    ftheta_manifest: str | Path,
) -> str:
    """Compose both arms and require representation-only path differences."""

    if mode not in MODE:
        raise ValueError(f"unsupported mode: {mode!r}")
    with initialize_config_dir(config_dir=str(_REPO_ROOT / "configs"), version_base=None):
        base = compose(config_name=CONFIG_NAME)
    configs = {}
    for arm, manifest in (("P", pin_manifest), ("F", ftheta_manifest)):
        config = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
        updates = {
            "path": str(Path(manifest).expanduser().resolve()),
            "experiment_name": f"pin_ftheta_native_6cam_{mode}_arm{arm}",
            "out_dir": "/preflight/no-output",
            "n_iterations": MODE[mode]["iterations"],
            "seed_initialization": 42,
            "test_last": True,
            "num_workers": 10,
            "dataset.train.seek_offset_sec": 0.0,
            "dataset.train.duration_sec": MODE[mode]["duration_sec"],
            "dataset.val.seek_offset_sec": 0.0,
            "dataset.val.duration_sec": MODE[mode]["duration_sec"],
            "dataset.downsample": 1.0,
            "dataset.n_val_image_subsample": 1,
            "dataset.load_lidar_depth_map": False,
            "dataset.load_depth_prior": False,
            "trainer.use_lidar_depth": False,
            "trainer.use_depth_prior": False,
            "trainer.sky_backend": "mlp",
        }
        for key, value in updates.items():
            OmegaConf.update(config, key, value, merge=False)
        if list(config.dataset.camera_ids) != list(CAMERA_IDS):
            raise ValueError("resolved config camera order does not match the six-camera contract")
        if OmegaConf.select(config, "dataset.ftheta_params_path") is not None:
            raise ValueError("runtime FTheta parameter overrides are forbidden")
        if OmegaConf.select(config, "dataset.mask_forward_invalid_pixels") is not False:
            raise ValueError("matched P/F contract requires mask_forward_invalid_pixels=false")
        if OmegaConf.select(config, "dataset.opencv_pinhole_use_validity_domain") is not True:
            raise ValueError("matched P/F contract requires opencv_pinhole_use_validity_domain=true")
        configs[arm] = OmegaConf.to_container(config, resolve=False)

    differences = _leaf_differences(configs["P"], configs["F"])
    if differences != {"path", "experiment_name"}:
        raise ValueError(f"P/F resolved configs have unexpected differences: {sorted(differences)}")
    normalized = copy.deepcopy(configs["P"])
    normalized.pop("path", None)
    normalized.pop("experiment_name", None)
    normalized.pop("out_dir", None)
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_against_baseline_parsed(
    mode: str,
    baseline_parsed: str | Path,
    ftheta_manifest: str | Path,
) -> str:
    """Require the F arm to differ from the recorded P run only as contracted."""

    baseline_path = Path(baseline_parsed).expanduser().resolve()
    if not baseline_path.is_file():
        raise ValueError(f"P baseline parsed config is missing: {baseline_path}")
    if mode not in MODE:
        raise ValueError(f"unsupported mode: {mode!r}")
    baseline = OmegaConf.to_container(OmegaConf.load(baseline_path), resolve=False)
    with initialize_config_dir(config_dir=str(_REPO_ROOT / "configs"), version_base=None):
        expected = compose(config_name=CONFIG_NAME)
    updates = {
        "path": str(Path(ftheta_manifest).expanduser().resolve()),
        "experiment_name": f"pin_ftheta_native_6cam_{mode}_armF",
        "out_dir": "/preflight/no-output",
        "n_iterations": MODE[mode]["iterations"],
        "seed_initialization": 42,
        "test_last": True,
        "num_workers": 10,
        "dataset.train.seek_offset_sec": 0.0,
        "dataset.train.duration_sec": MODE[mode]["duration_sec"],
        "dataset.val.seek_offset_sec": 0.0,
        "dataset.val.duration_sec": MODE[mode]["duration_sec"],
        "dataset.downsample": 1.0,
        "dataset.n_val_image_subsample": 1,
        "dataset.load_lidar_depth_map": False,
        "dataset.load_depth_prior": False,
        "trainer.use_lidar_depth": False,
        "trainer.use_depth_prior": False,
        "trainer.sky_backend": "mlp",
    }
    for key, value in updates.items():
        OmegaConf.update(expected, key, value, merge=False)
    expected_container = OmegaConf.to_container(expected, resolve=False)
    allowed = {"path", "experiment_name", "out_dir"}
    if mode == "smoke":
        allowed.update(
            {"n_iterations", "dataset.train.duration_sec", "dataset.val.duration_sec"}
        )
    differences = _leaf_differences(baseline, expected_container)
    unexpected = differences - allowed
    if unexpected:
        raise ValueError(
            "FTheta config does not strictly match the recorded P baseline; "
            f"unexpected differences: {sorted(unexpected)}"
        )
    canonical = json.dumps(expected_container, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(MODE), required=True)
    parser.add_argument("--pin-manifest", required=True)
    parser.add_argument("--ftheta-manifest", required=True)
    parser.add_argument("--baseline-parsed", help="recorded P baseline parsed.yaml for strict matching")
    args = parser.parse_args()
    try:
        pair = validate_native_pair(args.pin_manifest, args.ftheta_manifest)
        config_hash = validate_resolved_configs(args.mode, args.pin_manifest, args.ftheta_manifest)
        baseline_hash = (
            validate_against_baseline_parsed(args.mode, args.baseline_parsed, args.ftheta_manifest)
            if args.baseline_parsed
            else None
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(json.dumps({**pair, "mode": args.mode, "config_sha256": config_hash, "baseline_config_sha256": baseline_hash}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
