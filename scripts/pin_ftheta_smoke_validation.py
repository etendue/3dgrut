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
    r"\[A1\].*: repaired \d+ non-finite camera ray\(s\) "
    r"\(\+\d+ in val subsample\) and masked the pixel\(s\) invalid"
)
_TRAIN_FRAME_HEADER = re.compile(r"NCoreDataset\s+\[train\]\s+frame counts\s+\(after temporal\s+filtering\):")
_TRAIN_FRAME_LINE = re.compile(r"\b(camera_[A-Za-z0-9_]+):\s+(\d+)\s+frames\b")
_FINAL_CHECKPOINT_LINE = re.compile(r'Saved checkpoint to:\s*"[^"]*ckpt_last\.pt"')
_REQUIRED_SOURCE_NAMES = frozenset({"dataset_manifest", "config", "artifact", "driver", "validator"})
_REQUIRED_OUTPUT_NAMES = frozenset({"parsed_yaml", "checkpoint", "metrics", "train_log", "eval_log"})
_EXPECTED_LAYERS = ["background", "road", "dynamic_rigids", "sky_envmap"]


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


def _train_frame_counts(text: str, expected_camera_ids: list[str]) -> dict[str, int]:
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
    for line in text.splitlines():
        if "non-finite camera ray" in line.lower() and _REPAIRED_CAMERA_RAY.search(line) is None:
            raise ValueError(f"{arm} training log contains an unrepaired camera-ray invariant: {line}")
    for summary in ("Training Statistics", "Test Metrics"):
        if summary not in text:
            raise ValueError(f"{arm} training log missing {summary!r}")
    if _FINAL_CHECKPOINT_LINE.search(text) is None:
        raise ValueError(f"{arm} training log has no final ckpt_last save signal")

    override_logged = re.search(r"\[PIN-FTHETA\].*explicit override enabled", text) is not None
    if arm == "F" and not override_logged:
        raise ValueError("Arm F did not log the strict FTheta dataset override")
    if arm == "P" and override_logged:
        raise ValueError("Arm P unexpectedly enabled the FTheta dataset override")
    camera_ids, _, _ = _artifact_contract(artifact_path)
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
    for arm, evidence in arms.items():
        if arm not in {"P", "F"} or not isinstance(evidence, dict):
            raise ValueError(f"run manifest Arm {arm!r} evidence is invalid")
        missing = _REQUIRED_OUTPUT_NAMES - set(evidence)
        if missing:
            raise ValueError(f"run manifest Arm {arm} outputs missing {sorted(missing)}")
        for output_name in sorted(_REQUIRED_OUTPUT_NAMES):
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


def create_run_manifest(
    path: str | Path,
    run_id: str,
    git_commit: str,
    sources: dict[str, str | Path],
) -> dict:
    if not run_id or not git_commit:
        raise ValueError("run_id and git_commit must be non-empty")
    if set(sources) != _REQUIRED_SOURCE_NAMES:
        raise ValueError(f"run manifest sources must be {sorted(_REQUIRED_SOURCE_NAMES)}, got {sorted(sources)}")
    value = {
        "schema_version": 2,
        "run_id": run_id,
        "status": "started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "sources": {name: _file_record(source) for name, source in sorted(sources.items())},
        "arms": {},
    }
    _write_run_manifest(path, value, exclusive=True)
    return value


def verify_run_manifest(path: str | Path, current_git_commit: str) -> dict:
    value = _load_run_manifest(path)
    if value.get("schema_version") != 2:
        raise ValueError(f"unsupported run manifest schema_version={value.get('schema_version')!r}")
    if value.get("git_commit") != current_git_commit:
        raise ValueError(f"git commit drift: recorded={value.get('git_commit')!r} current={current_git_commit!r}")
    sources = value.get("sources")
    if not isinstance(sources, dict) or set(sources) != _REQUIRED_SOURCE_NAMES:
        raise ValueError("run manifest source set is invalid")
    for name, record in sources.items():
        _verify_file_record(record, context=name, drift_kind="source")
    if value.get("status") == "complete":
        _verify_arm_output_records(value, require_both=True)
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
) -> dict:
    arm = _arm(arm)
    validate_parsed_config(parsed_yaml, arm, artifact_path, input_manifest_path)
    validate_checkpoint(checkpoint_path, arm, artifact_path, input_manifest_path)
    train_counts = validate_training_log(train_log, arm, artifact_path)
    validate_metrics(metrics_path, artifact_path)

    value = _load_run_manifest(manifest_path)
    arms = value.setdefault("arms", {})
    if arm in arms:
        raise ValueError(f"Arm {arm} evidence already recorded")
    arms[arm] = {
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
    value["comparison"] = {
        "normalized_scientific_config_sha256": scientific_hash,
        "only_representation_path_differs": True,
        "train_frames_per_camera": p_train_counts,
        "per_camera_n_frames": frame_counts,
    }
    value["status"] = "complete"
    value["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
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

    create = commands.add_parser("manifest-create", help="freeze source hashes before launch")
    create.add_argument("--path", required=True)
    create.add_argument("--run-id", required=True)
    create.add_argument("--repo-root", required=True)
    create.add_argument("--dataset-manifest", required=True)
    create.add_argument("--config", required=True)
    create.add_argument("--artifact", required=True)
    create.add_argument("--driver", required=True)
    create.add_argument("--validator", required=True)

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
    record.add_argument("--artifact", required=True)
    record.add_argument("--input-manifest", required=True)
    record.add_argument("--repo-root", required=True)

    finalize = commands.add_parser("finalize", help="compare arms and close the run manifest")
    finalize.add_argument("--manifest", required=True)
    finalize.add_argument("--repo-root", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "log":
        validate_training_log(args.path, args.arm, args.artifact)
    elif args.command == "checkpoint":
        validate_checkpoint(args.path, args.arm, args.artifact, args.input_manifest)
    elif args.command == "metrics":
        validate_metrics(args.path, args.artifact)
    elif args.command == "manifest-create":
        ensure_tracked_worktree_clean(args.repo_root)
        create_run_manifest(
            args.path,
            args.run_id,
            _git_commit(args.repo_root),
            {
                "dataset_manifest": args.dataset_manifest,
                "config": args.config,
                "artifact": args.artifact,
                "driver": args.driver,
                "validator": args.validator,
            },
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
        )
    else:
        finalize_run_manifest(
            args.manifest,
            _git_commit(args.repo_root),
        )


if __name__ == "__main__":
    main()
