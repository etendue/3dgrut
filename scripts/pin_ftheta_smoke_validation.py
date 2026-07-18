#!/usr/bin/env python3
"""Fail-fast validation for the matched PIN-FTHETA smoke driver outputs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image, UnidentifiedImageError

from scripts.ncore_data_readiness import (
    V4_MULTILAYER_PROFILE_CONTRACT,
    V4_MULTILAYER_READINESS_PROFILE,
    V4_REQUIRED_AUX_TYPES,
    validate_ncore_data_readiness,
    validate_v4_multilayer_dataset_contract,
    validate_v4_multilayer_profile_contract,
)
from threedgrut.datasets.ftheta_override import (
    FTHETA_PARAMETER_KEYS,
    load_ftheta_override_parameters,
)

RENDER_METRIC_KEYS = (
    "mean_psnr",
    "mean_ssim",
    "mean_lpips",
    "mean_cc_psnr",
    "mean_cc_ssim",
    "mean_cc_lpips",
    "mean_psnr_masked",
    "mean_ssim_masked",
    "mean_lpips_masked",
    "mean_cc_psnr_masked",
    "mean_cc_ssim_masked",
    "mean_cc_lpips_masked",
)

_REPAIRED_CAMERA_RAY = re.compile(
    r"\[A1\]\s+camera_[A-Za-z0-9_]+:\s+repaired\s+\d+\s+"
    r"non-finite\s+camera\s+ray\(s\)\s+\(\+\d+\s+in\s+val\s+subsample\)\s+"
    r"and\s+masked\s+the\s+pixel\(s\)\s+invalid",
    flags=re.IGNORECASE,
)
_NONFINITE_CAMERA_RAY = re.compile(r"non-finite\s+camera\s+ray", flags=re.IGNORECASE)
_RICH_SOURCE_COLUMN = re.compile(r"\s+[A-Za-z0-9_./-]+\.py:\d+\s*$")
_LOGICAL_MESSAGE_START = re.compile(
    r"^(?:(?:\[\d{2}:\d{2}:\d{2}\]\s*)?" r"\[(?:INFO|WARNING|ERROR|DEBUG|CRITICAL)\]\s+|" r"\[[A-Z][A-Z0-9_.-]*\]\s+)"
)
_TRAIN_FRAME_HEADER = re.compile(r"NCoreDataset\s+(?:\[train\]\s+)?frame counts\s+\(after\s+temporal\s+filtering\):")
_TRAIN_FRAME_LINE = re.compile(r"\b(camera_[A-Za-z0-9_]+):\s+(\d+)\s+frames\b")
_CAMERA_RAY_DOMAIN = re.compile(
    r"\[CAMERA-RAY-DOMAIN\]\s+split=(train|val|test)\s+"
    r"camera=(camera_[A-Za-z0-9_]+)\s+model_type=([A-Za-z0-9_]+)\s+"
    r"artifact_fingerprint=([A-Za-z0-9]+)\s+total=(\d+)\s+"
    r"excluded_by_max_angle=(\d+)\s+nonfinite=(\d+)"
)
_V4_CAMERA_RAY_DOMAIN = re.compile(
    r"\[CAMERA-RAY-DOMAIN\]\s+split=(train|val|test)\s+"
    r"camera=(camera_[A-Za-z0-9_]+)\s+model_type=([A-Za-z0-9_]+)\s+"
    r"artifact_fingerprint=([A-Za-z0-9]+)\s+total=(\d+)\s+"
    r"excluded_by_max_angle=(\d+)\s+nonfinite=(\d+)\s+"
    r"raw_nonfinite=(\d+)\s+cached_nonfinite=(\d+)\s+"
    r"supervised_nonfinite=(\d+)"
)
_V4_NONFINITE_PRED_OR_RENDER_DROP = re.compile(
    r"(?:non[- ]finite|nonfinite).{0,120}(?:pred(?:_rgb)?|render|drop(?:ped|ping)?)|"
    r"(?:drop(?:ped|ping)?).{0,120}(?:batch|render).{0,120}(?:non[- ]finite|nonfinite)",
    flags=re.IGNORECASE,
)
# Rich may hard-wrap the quoted basename after ``ckpt_`` in redirected logs.
_FINAL_CHECKPOINT_LINE = re.compile(r'Saved checkpoint to:\s*"(?:[^"]*/)?ckpt_\s*last\.pt"')
_FTHETA_OVERRIDE_ENABLED = re.compile(
    r"\[PIN-FTHETA\]\s+NCoreDataset\s+(?:\[train\]\s+)?"
    r"explicit\s+override\s+enabled:\s+.{0,300}?cameras=7(?=\s|$|[;,])",
    flags=re.IGNORECASE,
)
_FTHETA_OVERRIDE_ATTEMPT = re.compile(r"\[PIN-FTHETA\].{0,300}?\boverride\b", flags=re.IGNORECASE)
_FTHETA_FALLBACK = re.compile(
    r"\[PIN-FTHETA\].{0,1000}?\b(?:fallback|falling\s+back)\b",
    flags=re.IGNORECASE,
)
_REQUIRED_SOURCE_NAMES = frozenset({"dataset_manifest", "config", "artifact", "driver", "validator"})
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_READINESS_SOURCE = Path(__file__).resolve().parent / "ncore_data_readiness.py"
_V4_DRIVER_VALIDATOR_SOURCE = Path(__file__).resolve().parent / "pin_ftheta_v4_driver_validation.py"
_REQUIRED_OUTPUT_NAMES = frozenset({"parsed_yaml", "checkpoint", "metrics", "train_log", "eval_log"})
_EXPECTED_LAYERS = ["background", "road", "dynamic_rigids", "sky_envmap"]
_V4_NATIVE_RENDER_OUTPUT = "native_render_inventory"
_V4_SMOKE_RUN_PROFILE = "pin_ftheta_v4_7cam_smoke_5s_5k_v1"
_V4_NATIVE_RESOLUTION = (1920, 1080)
_V4_TOTAL_PIXELS = 2_073_600
_V4_EXCLUDED_BY_MAX_ANGLE = {
    "camera_front_wide_120fov": 148,
    "camera_cross_left_120fov": 138,
    "camera_cross_right_120fov": 133,
    "camera_left_wide_90fov": 26_355,
    "camera_right_wide_90fov": 44_292,
    "camera_back_rear_wide_90fov": 120,
    "camera_rear_left_70fov": 101,
}
_V4_PINHOLE_RAW_NONFINITE = {
    "camera_front_wide_120fov": 0,
    "camera_cross_left_120fov": 0,
    "camera_cross_right_120fov": 0,
    "camera_left_wide_90fov": 6,
    "camera_right_wide_90fov": 7,
    "camera_back_rear_wide_90fov": 0,
    "camera_rear_left_70fov": 0,
}


def _normalize_rich_log(text: str) -> str:
    """Strip Rich source columns and fold display-wrapped logical messages."""

    without_source_columns = "\n".join(_RICH_SOURCE_COLUMN.sub("", line).rstrip() for line in text.splitlines())
    return re.sub(r"\s+", " ", without_source_columns).strip()


def _rich_logical_messages(text: str) -> list[str]:
    """Rebuild Rich-wrapped messages while retaining their boundaries."""

    messages: list[str] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = _RICH_SOURCE_COLUMN.sub("", raw_line).strip()
        if not line:
            continue
        if current and _LOGICAL_MESSAGE_START.match(line):
            messages.append(re.sub(r"\s+", " ", " ".join(current)).strip())
            current = []
        current.append(line)
    if current:
        messages.append(re.sub(r"\s+", " ", " ".join(current)).strip())
    return messages


def _arm(value: str) -> str:
    result = value.upper()
    if result not in {"P", "F"}:
        raise ValueError(f"arm must be P or F, got {value!r}")
    return result


def _artifact_contract(path: str | Path) -> tuple[list[str], dict[str, dict], dict[str, str]]:
    artifact_path = Path(path)
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read strict FTheta artifact {artifact_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("strict FTheta artifact must be a camera mapping")
    camera_ids = list(payload)
    try:
        parameters, fingerprints = load_ftheta_override_parameters(artifact_path, camera_ids)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"strict FTheta artifact failed validation: {exc}") from exc
    return camera_ids, parameters, fingerprints


def _is_v4_runtime_artifact(path: str | Path) -> bool:
    return Path(path).expanduser().resolve() == (
        _REPO_ROOT / V4_MULTILAYER_PROFILE_CONTRACT["runtime_artifact"]["path"]
    ).resolve()


def _validate_v4_camera_ray_domain_telemetry(
    text: str,
    arm: str,
    artifact_path: str | Path,
) -> None:
    """Require one exact camera-domain record for every split/camera pair."""

    arm = _arm(arm)
    camera_ids, _, fingerprints = _artifact_contract(artifact_path)
    if tuple(camera_ids) != tuple(_V4_EXCLUDED_BY_MAX_ANGLE):
        raise ValueError("v4 telemetry oracle camera order does not match the runtime artifact")
    normalized = _normalize_rich_log(text)
    records: dict[tuple[str, str], tuple[str, str, int, int, int, int, int, int]] = {}
    telemetry_matches = list(_V4_CAMERA_RAY_DOMAIN.finditer(normalized))
    marker_count = normalized.count("[CAMERA-RAY-DOMAIN]")
    if marker_count != len(telemetry_matches):
        raise ValueError(
            "v4 camera-ray-domain telemetry has missing or malformed extended fields: "
            f"markers={marker_count} complete_records={len(telemetry_matches)}"
        )
    for match in telemetry_matches:
        (
            split,
            camera_id,
            model_type,
            fingerprint,
            total,
            excluded,
            nonfinite,
            raw_nonfinite,
            cached_nonfinite,
            supervised_nonfinite,
        ) = match.groups()
        key = (split, camera_id)
        if key in records:
            raise ValueError(f"duplicate v4 camera-ray-domain telemetry record: split={split} camera={camera_id}")
        records[key] = (
            model_type,
            fingerprint,
            int(total),
            int(excluded),
            int(nonfinite),
            int(raw_nonfinite),
            int(cached_nonfinite),
            int(supervised_nonfinite),
        )

    expected_keys = {(split, camera_id) for split in ("train", "val", "test") for camera_id in camera_ids}
    if set(records) != expected_keys:
        missing = sorted(expected_keys - set(records))
        unexpected = sorted(set(records) - expected_keys)
        raise ValueError(
            "v4 camera-ray-domain telemetry split/camera coverage mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )
    expected_model = "FThetaCameraModel" if arm == "F" else "OpenCVPinholeCameraModel"
    for (
        split,
        camera_id,
    ), (
        model_type,
        fingerprint,
        total,
        excluded,
        nonfinite,
        raw_nonfinite,
        cached_nonfinite,
        supervised_nonfinite,
    ) in records.items():
        expected_fingerprint = fingerprints[camera_id] if arm == "F" else "none"
        expected_excluded = _V4_EXCLUDED_BY_MAX_ANGLE[camera_id] if arm == "F" else 0
        expected_raw_nonfinite = 0 if arm == "F" else _V4_PINHOLE_RAW_NONFINITE[camera_id]
        if model_type != expected_model:
            raise ValueError(
                f"v4 {arm} telemetry model mismatch for {split}/{camera_id}: "
                f"expected={expected_model} actual={model_type}"
            )
        if fingerprint != expected_fingerprint:
            raise ValueError(
                f"v4 {arm} telemetry fingerprint mismatch for {split}/{camera_id}: "
                f"expected={expected_fingerprint} actual={fingerprint}"
            )
        if nonfinite != raw_nonfinite:
            raise ValueError(
                f"v4 {arm} telemetry raw alias mismatch for {split}/{camera_id}: "
                f"nonfinite={nonfinite} raw_nonfinite={raw_nonfinite}"
            )
        if (
            total != _V4_TOTAL_PIXELS
            or excluded != expected_excluded
            or raw_nonfinite != expected_raw_nonfinite
            or cached_nonfinite != 0
            or supervised_nonfinite != 0
        ):
            raise ValueError(
                f"v4 {arm} telemetry oracle mismatch for {split}/{camera_id}: "
                f"total={total} excluded_by_max_angle={excluded} "
                f"raw_nonfinite={raw_nonfinite} cached_nonfinite={cached_nonfinite} "
                f"supervised_nonfinite={supervised_nonfinite}"
            )


def _train_frame_counts(text: str, expected_camera_ids: list[str]) -> dict[str, int]:
    text = _normalize_rich_log(text)
    candidates: list[dict[str, int]] = []
    for header in _TRAIN_FRAME_HEADER.finditer(text):
        remainder = text[header.end() :]
        total = re.search(r"\bTotal:\s+\d+\s+frames\b", remainder)
        if total is None:
            continue
        counts: dict[str, int] = {}
        for match in _TRAIN_FRAME_LINE.finditer(remainder[: total.start()]):
            counts[match.group(1)] = int(match.group(2))
        candidates.append(counts)

    expected = set(expected_camera_ids)
    for counts in candidates:
        if set(counts) == expected and all(count > 0 for count in counts.values()):
            return {camera_id: counts[camera_id] for camera_id in expected_camera_ids}
    summaries = [
        {
            "missing": sorted(expected - set(counts)),
            "extra": sorted(set(counts) - expected),
            "nonpositive": sorted(camera_id for camera_id, count in counts.items() if count <= 0),
        }
        for counts in candidates
    ]
    raise ValueError(f"train frame counts are not exact-seven positive: candidates={summaries}")


def validate_training_log(path: str | Path, arm: str, artifact_path: str | Path) -> dict[str, int]:
    """Reject only fatal trainer sentinels and unrepaired camera-ray reports.

    Known containment logs are deliberately allowed: dataset ray repair,
    renderer batch/pixel containment, and MCMC relocation sanitization.
    """

    arm = _arm(arm)
    log_path = Path(path)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "Traceback (most recent call last):" in text:
        raise ValueError(f"{arm} training log contains a Python traceback")
    if re.search(r"Non-finite total_loss at step", text, flags=re.IGNORECASE):
        raise ValueError(f"{arm} training log contains the trainer total-loss sentinel")
    if _is_v4_runtime_artifact(artifact_path) and _V4_NONFINITE_PRED_OR_RENDER_DROP.search(text):
        raise ValueError(f"{arm} v4 training log contains a non-finite prediction/render drop sentinel")
    normalized = _normalize_rich_log(text)
    repaired_spans = [match.span() for match in _REPAIRED_CAMERA_RAY.finditer(normalized)]
    for occurrence in _NONFINITE_CAMERA_RAY.finditer(normalized):
        if not any(start <= occurrence.start() < end for start, end in repaired_spans):
            context_start = max(0, occurrence.start() - 80)
            context_end = min(len(normalized), occurrence.end() + 120)
            raise ValueError(
                f"{arm} training log contains an unrepaired camera-ray invariant: "
                f"{normalized[context_start:context_end]}"
            )
    for summary in ("Training Statistics", "Test Metrics"):
        if summary not in text:
            raise ValueError(f"{arm} training log missing {summary!r}")
    if _FINAL_CHECKPOINT_LINE.search(normalized) is None:
        raise ValueError(f"{arm} training log has no final ckpt_last save signal")

    ftheta_messages = [message for message in _rich_logical_messages(text) if "[PIN-FTHETA]" in message]
    override_enabled = any(_FTHETA_OVERRIDE_ENABLED.search(message) is not None for message in ftheta_messages)
    fallback_logged = any(_FTHETA_FALLBACK.search(message) is not None for message in ftheta_messages)
    override_attempted = any(_FTHETA_OVERRIDE_ATTEMPT.search(message) is not None for message in ftheta_messages)
    if arm == "F" and (not override_enabled or fallback_logged):
        raise ValueError("Arm F did not log the strict FTheta dataset override")
    if arm == "P" and (override_attempted or fallback_logged):
        raise ValueError("Arm P unexpectedly enabled the FTheta dataset override")
    camera_ids, _, _ = _artifact_contract(artifact_path)
    if _is_v4_runtime_artifact(artifact_path):
        _validate_v4_camera_ray_domain_telemetry(text, arm, artifact_path)
    return _train_frame_counts(text, camera_ids)


def _config(value: Any) -> DictConfig:
    if isinstance(value, DictConfig):
        return value
    if isinstance(value, dict):
        return OmegaConf.create(value)
    raise ValueError(f"checkpoint config must be a mapping, got {type(value).__name__}")


def _require_config_value(config: DictConfig, dotted_key: str, expected: Any) -> None:
    actual = OmegaConf.select(config, dotted_key, default=None)
    if isinstance(actual, DictConfig):
        actual = OmegaConf.to_container(actual, resolve=False)
    elif hasattr(actual, "_content"):
        actual = OmegaConf.to_container(actual, resolve=False)
    if actual != expected:
        raise ValueError(f"scientific config {dotted_key}: {actual!r} != {expected!r}")


def validate_scientific_config(
    config_value: Any,
    arm: str,
    artifact_path: str | Path,
    input_manifest_path: str | Path,
) -> None:
    """Pin every declared smoke invariant in embedded/parsed configuration."""

    arm = _arm(arm)
    camera_ids, _, _ = _artifact_contract(artifact_path)
    config = _config(config_value)
    scalar_contract = {
        "n_iterations": 5000,
        "seed_initialization": 42,
        "test_last": True,
        "num_workers": 10,
        "dataset.train.seek_offset_sec": 0.0,
        "dataset.train.duration_sec": 5.0,
        "dataset.val.seek_offset_sec": 0.0,
        "dataset.val.duration_sec": 5.0,
        "dataset.downsample": 1.0,
        "dataset.mask_forward_invalid_pixels": True,
        "dataset.opencv_pinhole_use_validity_domain": False,
        "dataset.load_lidar_depth_map": False,
        "dataset.load_depth_prior": False,
        "trainer.use_lidar_depth": False,
        "trainer.use_depth_prior": False,
        "trainer.sky_backend": "mlp",
        "loss.use_opacity": False,
        "viz_4d.enabled": True,
    }
    for dotted_key, expected in scalar_contract.items():
        _require_config_value(config, dotted_key, expected)
    if Path(artifact_path).expanduser().resolve() == (
        _REPO_ROOT / V4_MULTILAYER_PROFILE_CONTRACT["runtime_artifact"]["path"]
    ).resolve():
        _require_config_value(config, "dataset.camera_max_fov_deg", 190.0)
        _require_config_value(config, "dataset.n_val_image_subsample", 1)
    _require_config_value(config, "dataset.camera_ids", camera_ids)
    _require_config_value(config, "loss.camera_loss_weights", {})
    _require_config_value(config, "layers.enabled", _EXPECTED_LAYERS)

    configured_manifest = OmegaConf.select(config, "path")
    if Path(str(configured_manifest)).expanduser().resolve() != Path(input_manifest_path).expanduser().resolve():
        raise ValueError("scientific config path does not match the hashed dataset manifest")
    configured_ftheta = OmegaConf.select(config, "dataset.ftheta_params_path")
    if arm == "P":
        if configured_ftheta is not None:
            raise ValueError("Arm P dataset.ftheta_params_path must be null")
    elif (
        configured_ftheta is None
        or Path(str(configured_ftheta)).expanduser().resolve() != Path(artifact_path).expanduser().resolve()
    ):
        raise ValueError("Arm F dataset.ftheta_params_path does not match the strict artifact")


def validate_parsed_config(
    path: str | Path,
    arm: str,
    artifact_path: str | Path,
    input_manifest_path: str | Path,
) -> None:
    try:
        config = OmegaConf.load(Path(path))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load parsed config {path}: {exc}") from exc
    validate_scientific_config(config, arm, artifact_path, input_manifest_path)


def _normalized_scientific_config(path: str | Path) -> dict:
    config = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=False)
    if not isinstance(config, dict):
        raise ValueError(f"parsed config {path} must be a mapping")
    result = copy.deepcopy(config)
    result.pop("experiment_name", None)
    result.pop("out_dir", None)
    dataset = result.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError(f"parsed config {path} has no dataset mapping")
    dataset.pop("ftheta_params_path", None)
    return result


def compare_parsed_configs(p_path: str | Path, f_path: str | Path) -> str:
    p_config = _normalized_scientific_config(p_path)
    f_config = _normalized_scientific_config(f_path)
    if p_config != f_config:
        raise ValueError("P/F scientific config mismatch after removing only representation/output bookkeeping")
    canonical = json.dumps(p_config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolution(value: Any, *, context: str) -> tuple[int, int]:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    raw = np.asarray(value).reshape(-1)
    if raw.size != 2:
        raise ValueError(f"{context}: resolution must contain W,H, got {raw.shape}")
    return int(raw[0]), int(raw[1])


def _compare_ftheta_value(camera_id: str, key: str, actual: Any, expected: Any) -> None:
    context = f"{camera_id}.intrinsics_FTheta.{key}"
    if key == "resolution":
        if _resolution(actual, context=context) != tuple(expected):
            raise ValueError(f"{context}: does not match strict artifact")
        return
    if key in {"shutter_type", "reference_poly"}:
        if actual != expected:
            raise ValueError(f"{context}: {actual!r} != {expected!r}")
        return
    # NCore's FTheta parameter dataclass and metadata contract store numeric
    # arrays as float32. Compare against the artifact after the same public
    # serialization boundary; the separately checked SHA-256 remains over the
    # canonical pre-runtime artifact values.
    actual_array = np.asarray(actual, dtype=np.float32)
    expected_array = np.asarray(expected, dtype=np.float32)
    if actual_array.shape != expected_array.shape or not np.array_equal(actual_array, expected_array):
        raise ValueError(f"{context}: does not match strict artifact at float32 serialization")


def validate_checkpoint(
    path: str | Path,
    arm: str,
    artifact_path: str | Path,
    input_manifest_path: str | Path,
) -> None:
    """Validate the public ``ckpt['viz_4d']['camera_models']`` schema."""

    arm = _arm(arm)
    camera_ids, parameters, fingerprints = _artifact_contract(artifact_path)
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{arm} checkpoint root must be a mapping")
    global_step = checkpoint.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step != 5000:
        raise ValueError(f"{arm} checkpoint global_step {global_step!r} != 5000")
    validate_scientific_config(checkpoint.get("config"), arm, artifact_path, input_manifest_path)
    validate_checkpoint_camera_models(checkpoint, arm, artifact_path)


def validate_checkpoint_camera_models(
    checkpoint: dict,
    arm: str,
    artifact_path: str | Path,
) -> None:
    """Validate the camera-model portion shared by smoke and full checkpoints."""

    arm = _arm(arm)
    camera_ids, parameters, fingerprints = _artifact_contract(artifact_path)
    viz_4d = checkpoint.get("viz_4d")
    if not isinstance(viz_4d, dict):
        raise ValueError(f"{arm} checkpoint has no viz_4d mapping")
    camera_models = viz_4d.get("camera_models")
    if not isinstance(camera_models, dict):
        raise ValueError(f"{arm} checkpoint has no viz_4d.camera_models mapping")
    if set(camera_models) != set(camera_ids):
        raise ValueError(
            f"{arm} checkpoint camera set mismatch: "
            f"missing={sorted(set(camera_ids) - set(camera_models))} "
            f"extra={sorted(set(camera_models) - set(camera_ids))}"
        )

    for camera_id in camera_ids:
        entry = camera_models[camera_id]
        if not isinstance(entry, dict):
            raise ValueError(f"{camera_id}: camera-model contract must be a mapping")
        native_resolution = _resolution(entry.get("native_resolution"), context=f"{camera_id}.native_resolution")
        if native_resolution != (1920, 1080):
            raise ValueError(f"{camera_id}: native resolution {native_resolution} != (1920, 1080)")

        if arm == "P":
            if entry.get("model_type") != "OpenCVPinhole":
                raise ValueError(f"{camera_id}: Arm P model must be OpenCVPinhole")
            if entry.get("intrinsics_FTheta") is not None or entry.get("parameter_fingerprint") is not None:
                raise ValueError(f"{camera_id}: Arm P must not contain a FTheta contract")
            continue

        if entry.get("model_type") != "FTheta":
            raise ValueError(f"{camera_id}: Arm F model must be FTheta")
        intrinsics = entry.get("intrinsics_FTheta")
        if not isinstance(intrinsics, dict) or set(intrinsics) != FTHETA_PARAMETER_KEYS:
            actual_keys = set(intrinsics) if isinstance(intrinsics, dict) else set()
            raise ValueError(
                f"{camera_id}: invalid eight-field FTheta contract; "
                f"missing={sorted(FTHETA_PARAMETER_KEYS - actual_keys)} "
                f"extra={sorted(actual_keys - FTHETA_PARAMETER_KEYS)}"
            )
        for key in FTHETA_PARAMETER_KEYS:
            _compare_ftheta_value(camera_id, key, intrinsics[key], parameters[camera_id][key])
        if entry.get("parameter_fingerprint") != fingerprints[camera_id]:
            raise ValueError(f"{camera_id}: FTheta parameter fingerprint does not match strict artifact")


def _finite_metric(mapping: dict, key: str, *, context: str) -> None:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{context}: missing/non-finite {key}={value!r}")


def validate_metrics(path: str | Path, artifact_path: str | Path) -> None:
    """Require all twelve render.py metrics globally and for every camera."""

    camera_ids, _, _ = _artifact_contract(artifact_path)
    metrics_path = Path(path)
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read metrics {metrics_path}: {exc}") from exc
    if not isinstance(metrics, dict):
        raise ValueError("metrics.json root must be a mapping")
    if _is_v4_runtime_artifact(artifact_path):
        # render.py deliberately preserves the legacy sparse schema: this key
        # is emitted only when the count is non-zero. Absence therefore means
        # the renderer-observed count is exactly zero, while any present value
        # must also be the integer zero to pass a v4 evidence gate.
        nonfinite_pred_px = metrics.get("nonfinite_pred_px", 0)
        if (
            isinstance(nonfinite_pred_px, bool)
            or not isinstance(nonfinite_pred_px, int)
            or nonfinite_pred_px != 0
        ):
            raise ValueError(
                f"v4 metrics nonfinite_pred_px must be the explicit integer 0, got {nonfinite_pred_px!r}"
            )
    for key in RENDER_METRIC_KEYS:
        _finite_metric(metrics, key, context="top-level metrics")

    per_camera = metrics.get("per_camera")
    if not isinstance(per_camera, dict) or set(per_camera) != set(camera_ids):
        actual = set(per_camera) if isinstance(per_camera, dict) else set()
        raise ValueError(
            "per_camera mismatch: "
            f"missing={sorted(set(camera_ids) - actual)} extra={sorted(actual - set(camera_ids))}"
        )
    for camera_id in camera_ids:
        camera_metrics = per_camera[camera_id]
        if not isinstance(camera_metrics, dict):
            raise ValueError(f"{camera_id}: per-camera metrics must be a mapping")
        n_frames = camera_metrics.get("n_frames")
        if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames <= 0:
            raise ValueError(f"{camera_id}: n_frames must be a positive integer, got {n_frames!r}")
        for key in RENDER_METRIC_KEYS:
            _finite_metric(camera_metrics, key, context=camera_id)


def _metrics_frame_counts(path: str | Path, artifact_path: str | Path) -> dict[str, int]:
    validate_metrics(path, artifact_path)
    metrics = json.loads(Path(path).read_text(encoding="utf-8"))
    camera_ids, _, _ = _artifact_contract(artifact_path)
    return {camera_id: int(metrics["per_camera"][camera_id]["n_frames"]) for camera_id in camera_ids}


def compare_per_camera_frame_counts(
    p_metrics_path: str | Path,
    f_metrics_path: str | Path,
    artifact_path: str | Path,
) -> dict[str, int]:
    p_counts = _metrics_frame_counts(p_metrics_path, artifact_path)
    f_counts = _metrics_frame_counts(f_metrics_path, artifact_path)
    if p_counts != f_counts:
        raise ValueError(f"P/F per-camera n_frames mismatch: P={p_counts} F={f_counts}")
    return p_counts


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _validate_native_png(path: Path, kind: str) -> None:
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError(f"native {kind} file is not PNG: {path}")
            if image.size != _V4_NATIVE_RESOLUTION:
                raise ValueError(
                    f"native {kind} PNG must be {_V4_NATIVE_RESOLUTION[0]}x{_V4_NATIVE_RESOLUTION[1]}, "
                    f"got {image.size[0]}x{image.size[1]}: {path}"
                )
            image.verify()
    except (OSError, SyntaxError, UnidentifiedImageError) as exc:
        raise ValueError(f"native {kind} file is not a readable PNG: {path}: {exc}") from exc


def validate_smoke_native_render_tree(
    metrics_path: str | Path,
    artifact_path: str | Path,
    inventory_path: str | Path | None = None,
) -> dict:
    """Hash every native smoke render/GT PNG and bind it to metrics counts."""

    metrics_path = Path(metrics_path).expanduser().resolve()
    if not _is_v4_runtime_artifact(artifact_path):
        raise ValueError("native smoke render inventory is only defined for the v4 artifact")
    validate_metrics(metrics_path, artifact_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    camera_ids, _, _ = _artifact_contract(artifact_path)
    per_camera_counts = {
        camera_id: int(metrics["per_camera"][camera_id]["n_frames"])
        for camera_id in camera_ids
    }
    expected_count = sum(per_camera_counts.values())
    step_root = metrics_path.parent / "ours_5000"
    render_dir = step_root / "renders"
    gt_dir = step_root / "gt"
    render_paths = sorted(render_dir.glob("*.png")) if render_dir.is_dir() else []
    gt_paths = sorted(gt_dir.glob("*.png")) if gt_dir.is_dir() else []
    if len(render_paths) != expected_count:
        raise ValueError(f"native smoke render PNG count {len(render_paths)} != evaluated frames {expected_count}")
    if len(gt_paths) != expected_count:
        raise ValueError(f"native smoke GT PNG count {len(gt_paths)} != evaluated frames {expected_count}")
    render_names = [path.name for path in render_paths]
    gt_names = [path.name for path in gt_paths]
    if render_names != gt_names:
        raise ValueError("native smoke render and GT PNG filenames do not match")
    for path in render_paths:
        _validate_native_png(path, "render")
    for path in gt_paths:
        _validate_native_png(path, "GT")
    inventory = {
        "schema_version": 1,
        "profile": _V4_SMOKE_RUN_PROFILE,
        "global_step": 5000,
        "native_resolution": list(_V4_NATIVE_RESOLUTION),
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "render_dir": str(render_dir.resolve()),
        "gt_dir": str(gt_dir.resolve()),
        "render_png_count": len(render_paths),
        "gt_png_count": len(gt_paths),
        "gt_png_names": gt_names,
        "per_camera_n_frames": per_camera_counts,
        "render_tree_sha256": _tree_sha256(render_paths, render_dir),
        "gt_tree_sha256": _tree_sha256(gt_paths, gt_dir),
    }
    if inventory_path is not None:
        _write_json_atomic(Path(inventory_path), inventory)
    return inventory


def _file_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"evidence file does not exist: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _verify_file_record(record: Any, *, context: str, drift_kind: str) -> None:
    if not isinstance(record, dict) or "path" not in record or "sha256" not in record:
        raise ValueError(f"run manifest {context} record is invalid")
    recorded_path = record["path"]
    if not isinstance(recorded_path, str):
        raise ValueError(f"run manifest {context} path is invalid")
    canonical_path = Path(recorded_path).expanduser().resolve()
    if str(canonical_path) != recorded_path:
        raise ValueError(f"run manifest {context} path is not canonical: {recorded_path!r}")
    if not canonical_path.is_file():
        raise ValueError(f"run manifest {context} file does not exist: {canonical_path}")
    actual = sha256_file(canonical_path)
    if actual != record["sha256"]:
        raise ValueError(f"{drift_kind} hash drift for {context}: " f"recorded={record['sha256']} current={actual}")


def _verify_arm_output_records(value: dict, *, require_both: bool) -> None:
    arms = value.get("arms")
    if not isinstance(arms, dict):
        raise ValueError("run manifest arms must be a mapping")
    if require_both and set(arms) != {"P", "F"}:
        raise ValueError("run manifest must contain both Arm P and Arm F evidence")
    required_outputs = set(_REQUIRED_OUTPUT_NAMES)
    if value.get("ncore_readiness_profile") == V4_MULTILAYER_READINESS_PROFILE:
        required_outputs.add(_V4_NATIVE_RENDER_OUTPUT)
    for arm, evidence in arms.items():
        if arm not in {"P", "F"} or not isinstance(evidence, dict):
            raise ValueError(f"run manifest Arm {arm!r} evidence is invalid")
        missing = required_outputs - set(evidence)
        if missing:
            raise ValueError(f"run manifest Arm {arm} outputs missing {sorted(missing)}")
        for output_name in sorted(required_outputs):
            _verify_file_record(
                evidence[output_name],
                context=f"Arm {arm} output {output_name}",
                drift_kind="output",
            )


def _load_run_manifest(path: str | Path) -> dict:
    manifest_path = Path(path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read run manifest {manifest_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("run manifest root must be a mapping")
    return value


def _write_run_manifest(path: str | Path, value: dict, *, exclusive: bool = False) -> None:
    manifest_path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if exclusive:
        with manifest_path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        return
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=manifest_path.parent, prefix=".run_manifest_", delete=False
    ) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    os.replace(temporary, manifest_path)


def _prepare_run_manifest(
    run_id: str,
    git_commit: str,
    sources: dict[str, str | Path],
    *,
    ncore_readiness_profile: str | None = None,
    repo_root: str | Path | None = None,
) -> dict:
    if not run_id or not git_commit:
        raise ValueError("run_id and git_commit must be non-empty")
    if set(sources) != _REQUIRED_SOURCE_NAMES:
        raise ValueError(f"run manifest sources must be {sorted(_REQUIRED_SOURCE_NAMES)}, got {sorted(sources)}")
    if ncore_readiness_profile not in (None, V4_MULTILAYER_READINESS_PROFILE):
        raise ValueError(f"unsupported NCore readiness profile: {ncore_readiness_profile!r}")
    source_records = {name: _file_record(source) for name, source in sorted(sources.items())}
    if ncore_readiness_profile is not None:
        profile_root = _REPO_ROOT if repo_root is None else Path(repo_root).expanduser().resolve()
        supplied_driver = Path(sources["driver"]).expanduser().resolve()
        expected_driver = (profile_root / "scripts/pin_ftheta_7cam_v4_smoke.sh").resolve()
        if supplied_driver != expected_driver:
            raise ValueError(
                f"v4 smoke driver path mismatch: expected={expected_driver} actual={supplied_driver}"
            )
        profile_paths = validate_v4_multilayer_profile_contract(
            profile_root,
            sources["config"],
            sources["artifact"],
        )
        validate_v4_multilayer_dataset_contract(sources["dataset_manifest"])
        source_records["data_readiness_validator"] = _file_record(_DATA_READINESS_SOURCE)
        source_records["v4_driver_validator"] = _file_record(_V4_DRIVER_VALIDATOR_SOURCE)
        source_records["v4_provenance_sidecar"] = _file_record(profile_paths["provenance_sidecar"])
        source_records["v4_survey_artifact"] = _file_record(profile_paths["survey_artifact"])
    value = {
        "schema_version": 2,
        "run_id": run_id,
        "status": "started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "sources": source_records,
        "arms": {},
    }
    if ncore_readiness_profile is not None:
        value["ncore_readiness_profile"] = ncore_readiness_profile
    return value


def create_run_manifest(
    path: str | Path,
    run_id: str,
    git_commit: str,
    sources: dict[str, str | Path],
    *,
    ncore_readiness_profile: str | None = None,
    repo_root: str | Path | None = None,
) -> dict:
    value = _prepare_run_manifest(
        run_id,
        git_commit,
        sources,
        ncore_readiness_profile=ncore_readiness_profile,
        repo_root=repo_root,
    )
    if ncore_readiness_profile == V4_MULTILAYER_READINESS_PROFILE:
        camera_ids, _, _ = _artifact_contract(sources["artifact"])
        validate_ncore_data_readiness(
            sources["dataset_manifest"],
            camera_ids,
            required_aux=V4_REQUIRED_AUX_TYPES,
        )
    _write_run_manifest(path, value, exclusive=True)
    return value


def verify_run_manifest(path: str | Path, current_git_commit: str) -> dict:
    value = _load_run_manifest(path)
    if value.get("schema_version") != 2:
        raise ValueError(f"unsupported run manifest schema_version={value.get('schema_version')!r}")
    if value.get("git_commit") != current_git_commit:
        raise ValueError(f"git commit drift: recorded={value.get('git_commit')!r} current={current_git_commit!r}")
    sources = value.get("sources")
    profile = value.get("ncore_readiness_profile")
    if profile not in (None, V4_MULTILAYER_READINESS_PROFILE):
        raise ValueError(f"unsupported NCore readiness profile: {profile!r}")
    if value.get("status") not in {"started", "failed", "complete"}:
        raise ValueError(f"run manifest status is invalid: {value.get('status')!r}")
    expected_sources = set(_REQUIRED_SOURCE_NAMES)
    if profile is not None:
        expected_sources.update(
            {
                "data_readiness_validator",
                "v4_driver_validator",
                "v4_provenance_sidecar",
                "v4_survey_artifact",
            }
        )
    if not isinstance(sources, dict) or set(sources) != expected_sources:
        raise ValueError("run manifest source set is invalid")
    for name, record in sources.items():
        _verify_file_record(record, context=name, drift_kind="source")
    if value.get("status") == "complete":
        _verify_arm_output_records(value, require_both=True)
        if profile == V4_MULTILAYER_READINESS_PROFILE:
            for arm in ("P", "F"):
                evidence = value["arms"][arm]
                current_inventory = validate_smoke_native_render_tree(
                    evidence["metrics"]["path"], value["sources"]["artifact"]["path"]
                )
                recorded_inventory = json.loads(
                    Path(evidence[_V4_NATIVE_RENDER_OUTPUT]["path"]).read_text(encoding="utf-8")
                )
                if current_inventory != recorded_inventory or current_inventory != evidence.get("native_render"):
                    raise ValueError(f"Arm {arm} v4 smoke native render tree drifted after completion")
    return value


def record_arm_outputs(
    manifest_path: str | Path,
    arm: str,
    parsed_yaml: str | Path,
    checkpoint_path: str | Path,
    metrics_path: str | Path,
    train_log: str | Path,
    eval_log: str | Path,
    artifact_path: str | Path,
    input_manifest_path: str | Path,
    native_render_inventory: str | Path | None = None,
) -> dict:
    arm = _arm(arm)
    value = _load_run_manifest(manifest_path)
    v4_profile = value.get("ncore_readiness_profile") == V4_MULTILAYER_READINESS_PROFILE
    validate_parsed_config(parsed_yaml, arm, artifact_path, input_manifest_path)
    validate_checkpoint(checkpoint_path, arm, artifact_path, input_manifest_path)
    train_counts = validate_training_log(train_log, arm, artifact_path)
    validate_metrics(metrics_path, artifact_path)

    arms = value.setdefault("arms", {})
    if arm in arms:
        raise ValueError(f"Arm {arm} evidence already recorded")
    evidence = {
        "parsed_yaml": _file_record(parsed_yaml),
        "checkpoint": {
            **_file_record(checkpoint_path),
            "global_step": 5000,
        },
        "metrics": _file_record(metrics_path),
        "train_log": _file_record(train_log),
        "eval_log": _file_record(eval_log),
        "train_frames_per_camera": train_counts,
    }
    if v4_profile:
        if native_render_inventory is None:
            raise ValueError("v4 smoke record-arm requires native render inventory")
        actual_inventory = validate_smoke_native_render_tree(metrics_path, artifact_path)
        try:
            recorded_inventory = json.loads(Path(native_render_inventory).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read v4 smoke native render inventory: {exc}") from exc
        if recorded_inventory != actual_inventory:
            raise ValueError("v4 smoke native render inventory does not match the current render tree")
        evidence[_V4_NATIVE_RENDER_OUTPUT] = _file_record(native_render_inventory)
        evidence["native_render"] = actual_inventory
    arms[arm] = evidence
    _write_run_manifest(manifest_path, value)
    return value


def finalize_run_manifest(
    manifest_path: str | Path,
    current_git_commit: str,
) -> dict:
    value = verify_run_manifest(manifest_path, current_git_commit)
    _verify_arm_output_records(value, require_both=True)
    p_train_counts = value["arms"]["P"].get("train_frames_per_camera")
    f_train_counts = value["arms"]["F"].get("train_frames_per_camera")
    if p_train_counts != f_train_counts:
        raise ValueError(f"P/F train frame-count mismatch: P={p_train_counts} F={f_train_counts}")
    p_outputs = value["arms"]["P"]
    f_outputs = value["arms"]["F"]
    scientific_hash = compare_parsed_configs(
        p_outputs["parsed_yaml"]["path"],
        f_outputs["parsed_yaml"]["path"],
    )
    frame_counts = compare_per_camera_frame_counts(
        p_outputs["metrics"]["path"],
        f_outputs["metrics"]["path"],
        value["sources"]["artifact"]["path"],
    )
    comparison_extra: dict[str, Any] = {}
    if value.get("ncore_readiness_profile") == V4_MULTILAYER_READINESS_PROFILE:
        for arm in ("P", "F"):
            evidence = value["arms"][arm]
            current_inventory = validate_smoke_native_render_tree(
                evidence["metrics"]["path"], value["sources"]["artifact"]["path"]
            )
            recorded_inventory = json.loads(
                Path(evidence[_V4_NATIVE_RENDER_OUTPUT]["path"]).read_text(encoding="utf-8")
            )
            if current_inventory != recorded_inventory or current_inventory != evidence.get("native_render"):
                raise ValueError(f"Arm {arm} v4 smoke native render tree drifted after evidence recording")
        p_native = value["arms"]["P"]["native_render"]
        f_native = value["arms"]["F"]["native_render"]
        for key in ("gt_png_count", "gt_png_names", "gt_tree_sha256", "native_resolution"):
            if p_native[key] != f_native[key]:
                raise ValueError(f"P/F v4 smoke native render parity mismatch for {key}")
        if p_native["per_camera_n_frames"] != f_native["per_camera_n_frames"]:
            raise ValueError("P/F v4 smoke native per-camera frame inventory mismatch")
        comparison_extra = {
            "gt_png_count": p_native["gt_png_count"],
            "gt_tree_sha256": p_native["gt_tree_sha256"],
            "native_resolution": p_native["native_resolution"],
        }
    value["comparison"] = {
        "normalized_scientific_config_sha256": scientific_hash,
        "only_representation_path_differs": True,
        "train_frames_per_camera": p_train_counts,
        "per_camera_n_frames": frame_counts,
        **comparison_extra,
    }
    value["status"] = "complete"
    value["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_run_manifest(manifest_path, value)
    return value


def mark_run_failed(manifest_path: str | Path, stage: str, exit_code: int) -> dict:
    value = _load_run_manifest(manifest_path)
    if value.get("status") == "complete":
        raise ValueError("cannot mark a completed smoke run failed")
    value["status"] = "failed"
    value["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
    value["failure"] = {"stage": stage, "exit_code": int(exit_code)}
    _write_run_manifest(manifest_path, value)
    return value


def _git_commit(repo_root: str | Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot resolve git commit in {repo_root}: {exc}") from exc


def ensure_tracked_worktree_clean(repo_root: str | Path) -> None:
    """Reject staged/unstaged tracked edits while preserving untracked user files."""

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot inspect tracked changes in {repo_root}: {exc}") from exc
    changes = result.stdout.strip()
    if changes:
        raise ValueError(
            "refusing launch with staged or unstaged tracked changes; " f"commit/stash them first:\n{changes}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    log = commands.add_parser("log", help="validate a completed training log")
    log.add_argument("--path", required=True)
    log.add_argument("--arm", required=True, choices=("P", "F"))
    log.add_argument("--artifact", required=True)

    checkpoint = commands.add_parser("checkpoint", help="validate checkpoint camera metadata")
    checkpoint.add_argument("--path", required=True)
    checkpoint.add_argument("--arm", required=True, choices=("P", "F"))
    checkpoint.add_argument("--artifact", required=True)
    checkpoint.add_argument("--input-manifest", required=True)

    metrics = commands.add_parser("metrics", help="validate native render metrics")
    metrics.add_argument("--path", required=True)
    metrics.add_argument("--artifact", required=True)

    render_tree = commands.add_parser("render-tree", help="freeze the v4 native smoke render inventory")
    render_tree.add_argument("--metrics", required=True)
    render_tree.add_argument("--artifact", required=True)
    render_tree.add_argument("--inventory", required=True)

    create = commands.add_parser("manifest-create", help="freeze source hashes before launch")
    create.add_argument("--path", required=True)
    create.add_argument("--run-id", required=True)
    create.add_argument("--repo-root", required=True)
    create.add_argument("--dataset-manifest", required=True)
    create.add_argument("--config", required=True)
    create.add_argument("--artifact", required=True)
    create.add_argument("--driver", required=True)
    create.add_argument("--validator", required=True)
    create.add_argument(
        "--ncore-readiness-profile",
        choices=(V4_MULTILAYER_READINESS_PROFILE,),
        default=None,
    )
    create.add_argument("--expected-commit")

    verify = commands.add_parser("manifest-verify", help="reject source or commit drift")
    verify.add_argument("--path", required=True)
    verify.add_argument("--repo-root", required=True)

    record = commands.add_parser("record-arm", help="persist one completed arm's evidence")
    record.add_argument("--manifest", required=True)
    record.add_argument("--arm", required=True, choices=("P", "F"))
    record.add_argument("--parsed-yaml", required=True)
    record.add_argument("--checkpoint", required=True)
    record.add_argument("--metrics", required=True)
    record.add_argument("--train-log", required=True)
    record.add_argument("--eval-log", required=True)
    record.add_argument("--native-render-inventory")
    record.add_argument("--artifact", required=True)
    record.add_argument("--input-manifest", required=True)
    record.add_argument("--repo-root", required=True)

    finalize = commands.add_parser("finalize", help="compare arms and close the run manifest")
    finalize.add_argument("--manifest", required=True)
    finalize.add_argument("--repo-root", required=True)
    fail = commands.add_parser("manifest-fail", help="record a failed v4 smoke stage")
    fail.add_argument("--manifest", required=True)
    fail.add_argument("--stage", required=True)
    fail.add_argument("--exit-code", required=True, type=int)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "log":
        validate_training_log(args.path, args.arm, args.artifact)
    elif args.command == "checkpoint":
        validate_checkpoint(args.path, args.arm, args.artifact, args.input_manifest)
    elif args.command == "metrics":
        validate_metrics(args.path, args.artifact)
    elif args.command == "render-tree":
        validate_smoke_native_render_tree(args.metrics, args.artifact, args.inventory)
    elif args.command == "manifest-create":
        ensure_tracked_worktree_clean(args.repo_root)
        git_commit = _git_commit(args.repo_root)
        if args.ncore_readiness_profile == V4_MULTILAYER_READINESS_PROFILE:
            if not args.expected_commit:
                raise ValueError("v4 manifest-create requires --expected-commit")
            if args.expected_commit != git_commit:
                raise ValueError(
                    f"v4 expected commit mismatch: expected={args.expected_commit} current={git_commit}"
                )
        create_run_manifest(
            args.path,
            args.run_id,
            git_commit,
            {
                "dataset_manifest": args.dataset_manifest,
                "config": args.config,
                "artifact": args.artifact,
                "driver": args.driver,
                "validator": args.validator,
            },
            ncore_readiness_profile=args.ncore_readiness_profile,
            repo_root=args.repo_root,
        )
    elif args.command == "manifest-verify":
        verify_run_manifest(args.path, _git_commit(args.repo_root))
    elif args.command == "record-arm":
        verify_run_manifest(args.manifest, _git_commit(args.repo_root))
        record_arm_outputs(
            args.manifest,
            args.arm,
            args.parsed_yaml,
            args.checkpoint,
            args.metrics,
            args.train_log,
            args.eval_log,
            args.artifact,
            args.input_manifest,
            args.native_render_inventory,
        )
    elif args.command == "finalize":
        finalize_run_manifest(
            args.manifest,
            _git_commit(args.repo_root),
        )
    else:
        mark_run_failed(args.manifest, args.stage, args.exit_code)


if __name__ == "__main__":
    main()
