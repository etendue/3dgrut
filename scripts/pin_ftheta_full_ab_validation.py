#!/usr/bin/env python3
"""Fail-fast evidence validation for the full-window PIN-FTHETA P/F A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
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
from scripts.pin_ftheta_smoke_validation import (
    _EXPECTED_LAYERS,
    _artifact_contract,
    _arm,
    _config,
    _file_record,
    _git_commit,
    _load_run_manifest,
    _require_config_value,
    _verify_arm_output_records,
    _verify_file_record,
    _write_run_manifest,
    compare_parsed_configs,
    compare_per_camera_frame_counts,
    ensure_tracked_worktree_clean,
    sha256_file,
    validate_checkpoint_camera_models,
    validate_metrics,
    validate_training_log,
)

FULL_RUN_PROFILE = "pin_ftheta_7cam_full_20s_30k_v1"
FULL_ITERATIONS = 30000
V4_FULL_DURATION_SEC = 20.0
NATIVE_RESOLUTION = (1920, 1080)
FROZEN_B6A9_CLIP_ID = "inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9"
FROZEN_B6A9_MANIFEST_SHA256 = "df2021203cfe318cfa8da3462e38c5b7fbf6bf3963d3a8149d145f98f6036e31"
EXPECTED_CAMERA_IDS = (
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_left_wide_90fov",
    "camera_right_wide_90fov",
    "camera_back_rear_wide_90fov",
    "camera_rear_left_70fov",
)
_FULL_REQUIRED_OUTPUT_NAMES = frozenset(
    {"parsed_yaml", "checkpoint", "metrics", "train_log", "eval_log", "native_render_inventory"}
)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FULL_STATIC_SOURCES = {
    "artifact": "scripts/pin_ftheta_b6a9_7cam_params.json",
    "config": "configs/apps/ncore_3dgut_mcmc_multilayer_inceptio_7cam.yaml",
    "driver": "scripts/pin_ftheta_7cam_full_ab.sh",
    "validator": "scripts/pin_ftheta_full_ab_validation.py",
    "smoke_validator": "scripts/pin_ftheta_smoke_validation.py",
    "train_entrypoint": "train.py",
    "render_entrypoint": "render.py",
    "render_implementation": "threedgrut/render.py",
    "dataset_implementation": "threedgrut/datasets/datasetNcore.py",
    "ftheta_override": "threedgrut/datasets/ftheta_override.py",
    "trainer_implementation": "threedgrut/trainer.py",
    "frozen_calibration_provenance": "scripts/pin_ftheta_b6a9_calibs.json",
    "experiment_spec": "docs/T8_artifacts/PIN_FTHETA_9CAM_EXPERIMENT_SPEC.md",
    "config_parent_inceptio": "configs/apps/ncore_3dgut_mcmc_multilayer_inceptio.yaml",
    "config_parent_multilayer": "configs/apps/ncore_3dgut_mcmc_multilayer.yaml",
    "config_base_mcmc": "configs/base_mcmc.yaml",
    "config_base_gs": "configs/base_gs.yaml",
    "config_dataset_ncore": "configs/dataset/ncore.yaml",
    "config_initialization_lidar": "configs/initialization/lidar.yaml",
    "config_render_3dgut": "configs/render/3dgut.yaml",
    "config_render_3dgrt": "configs/render/3dgrt.yaml",
    "config_strategy_layered_mcmc": "configs/strategy/layered_mcmc.yaml",
    "config_strategy_mcmc": "configs/strategy/mcmc.yaml",
    "config_strategy_gs": "configs/strategy/gs.yaml",
}
_V4_FULL_DRIVER_RELATIVE_PATH = "scripts/pin_ftheta_7cam_v4_full_ab.sh"
_V4_DRIVER_VALIDATOR_RELATIVE_PATH = "scripts/pin_ftheta_v4_driver_validation.py"


def _is_v4_runtime_artifact(path: str | Path) -> bool:
    return Path(path).expanduser().resolve() == (
        _REPO_ROOT / V4_MULTILAYER_PROFILE_CONTRACT["runtime_artifact"]["path"]
    ).resolve()


def _full_artifact_contract(path: str | Path) -> tuple[list[str], dict[str, dict], dict[str, str]]:
    camera_ids, parameters, fingerprints = _artifact_contract(path)
    if camera_ids != list(EXPECTED_CAMERA_IDS):
        raise ValueError(
            "strict full artifact camera order/set mismatch: "
            f"expected={list(EXPECTED_CAMERA_IDS)} actual={camera_ids}"
        )
    return camera_ids, parameters, fingerprints


def _frozen_calibration_provenance() -> dict:
    provenance_path = _REPO_ROOT / _FULL_STATIC_SOURCES["frozen_calibration_provenance"]
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8")).get("provenance")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError) as exc:
        raise ValueError(f"cannot read frozen b6a9 calibration provenance: {exc}") from exc
    if not isinstance(provenance, dict):
        raise ValueError("frozen b6a9 calibration provenance is missing")
    return provenance


def validate_frozen_b6a9_manifest(path: str | Path) -> dict:
    """Require the byte-exact NCore V4 manifest frozen in the experiment spec."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"frozen b6a9 dataset manifest missing: {manifest_path}")
    actual_sha256 = sha256_file(manifest_path)
    if actual_sha256 != FROZEN_B6A9_MANIFEST_SHA256:
        raise ValueError(
            "frozen b6a9 dataset manifest SHA-256 mismatch: "
            f"expected={FROZEN_B6A9_MANIFEST_SHA256} actual={actual_sha256}"
        )
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"frozen b6a9 dataset manifest is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("frozen b6a9 dataset manifest root must be a mapping")
    if value.get("sequence_id") != FROZEN_B6A9_CLIP_ID:
        raise ValueError(
            "frozen b6a9 dataset manifest sequence_id mismatch: "
            f"expected={FROZEN_B6A9_CLIP_ID!r} actual={value.get('sequence_id')!r}"
        )
    interval = value.get("sequence_timestamp_interval_us")
    if not isinstance(interval, dict):
        raise ValueError("frozen b6a9 dataset manifest timestamp interval is missing")
    start, stop = interval.get("start"), interval.get("stop")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(stop, bool)
        or not isinstance(stop, int)
        or start >= stop
    ):
        raise ValueError("frozen b6a9 dataset manifest timestamp interval is invalid")
    if "version" not in value:
        raise ValueError("frozen b6a9 dataset manifest version is missing")
    component_stores = value.get("component_stores")
    if not isinstance(component_stores, (dict, list)) or not component_stores:
        raise ValueError("frozen b6a9 dataset manifest component_stores is missing or empty")
    provenance = _frozen_calibration_provenance()
    if provenance.get("clip_id") != FROZEN_B6A9_CLIP_ID:
        raise ValueError("frozen b6a9 calibration artifact clip_id disagrees with validator")
    if provenance.get("manifest_sha256") != FROZEN_B6A9_MANIFEST_SHA256:
        raise ValueError("frozen b6a9 calibration artifact manifest SHA-256 disagrees with validator")
    return value


def _expanded_full_sources(
    sources: dict[str, str | Path],
    *,
    ncore_readiness_profile: str | None = None,
) -> dict[str, str | Path]:
    required_dynamic = {"dataset_manifest", "config", "artifact", "driver", "validator"}
    if set(sources) != required_dynamic:
        raise ValueError(f"full run manifest launch sources must be {sorted(required_dynamic)}, got {sorted(sources)}")
    static_sources = dict(_FULL_STATIC_SOURCES)
    profile_paths: dict[str, Path] | None = None
    if ncore_readiness_profile == V4_MULTILAYER_READINESS_PROFILE:
        static_sources["config"] = V4_MULTILAYER_PROFILE_CONTRACT["config"]["path"]
        static_sources["artifact"] = V4_MULTILAYER_PROFILE_CONTRACT["runtime_artifact"]["path"]
        static_sources["driver"] = _V4_FULL_DRIVER_RELATIVE_PATH
        static_sources["v4_driver_validator"] = _V4_DRIVER_VALIDATOR_RELATIVE_PATH
        profile_paths = validate_v4_multilayer_profile_contract(
            _REPO_ROOT,
            sources["config"],
            sources["artifact"],
        )
        validate_v4_multilayer_dataset_contract(sources["dataset_manifest"])
    for name in ("config", "artifact", "driver", "validator"):
        expected = (_REPO_ROOT / static_sources[name]).resolve()
        actual = Path(sources[name]).expanduser().resolve()
        if actual != expected:
            raise ValueError(f"full run source {name} path {actual} != frozen path {expected}")
    validate_frozen_b6a9_manifest(sources["dataset_manifest"])
    expanded: dict[str, str | Path] = {"dataset_manifest": sources["dataset_manifest"]}
    expanded.update({name: _REPO_ROOT / relative for name, relative in static_sources.items()})
    if ncore_readiness_profile is not None:
        expanded["data_readiness_validator"] = _REPO_ROOT / "scripts/ncore_data_readiness.py"
        assert profile_paths is not None
        expanded["v4_provenance_sidecar"] = profile_paths["provenance_sidecar"]
        expanded["v4_survey_artifact"] = profile_paths["survey_artifact"]
    return expanded


def validate_full_scientific_config(
    config_value: Any,
    arm: str,
    artifact_path: str | Path,
    input_manifest_path: str | Path,
) -> None:
    """Pin the complete full-window scientific contract for either arm."""

    arm = _arm(arm)
    camera_ids, _, _ = _full_artifact_contract(artifact_path)
    config = _config(config_value)
    v4_profile = _is_v4_runtime_artifact(artifact_path)
    duration_sec = V4_FULL_DURATION_SEC if v4_profile else -1
    scalar_contract = {
        "n_iterations": FULL_ITERATIONS,
        "seed_initialization": 42,
        "test_last": True,
        "num_workers": 10,
        "dataset.train.seek_offset_sec": 0.0,
        "dataset.train.duration_sec": duration_sec,
        "dataset.val.seek_offset_sec": 0.0,
        "dataset.val.duration_sec": duration_sec,
        "dataset.downsample": 1.0,
        "dataset.n_val_image_subsample": 1,
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
    if v4_profile:
        _require_config_value(config, "dataset.camera_max_fov_deg", 190.0)
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


def validate_full_parsed_config(
    path: str | Path,
    arm: str,
    artifact_path: str | Path,
    input_manifest_path: str | Path,
) -> None:
    try:
        config = OmegaConf.load(Path(path))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load parsed config {path}: {exc}") from exc
    validate_full_scientific_config(config, arm, artifact_path, input_manifest_path)


def validate_full_checkpoint(
    path: str | Path,
    arm: str,
    artifact_path: str | Path,
    input_manifest_path: str | Path,
) -> None:
    """Require a completed 30k checkpoint plus the strict camera metadata."""

    arm = _arm(arm)
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{arm} checkpoint root must be a mapping")
    global_step = checkpoint.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step != FULL_ITERATIONS:
        raise ValueError(f"{arm} checkpoint global_step {global_step!r} != {FULL_ITERATIONS}")
    validate_full_scientific_config(checkpoint.get("config"), arm, artifact_path, input_manifest_path)
    validate_checkpoint_camera_models(checkpoint, arm, artifact_path)


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
            if image.size != NATIVE_RESOLUTION:
                raise ValueError(
                    f"native {kind} PNG must be {NATIVE_RESOLUTION[0]}x{NATIVE_RESOLUTION[1]}, "
                    f"got {image.size[0]}x{image.size[1]}: {path}"
                )
            image.verify()
    except (OSError, SyntaxError, UnidentifiedImageError) as exc:
        raise ValueError(f"native {kind} file is not a readable PNG: {path}: {exc}") from exc


def validate_native_render_tree(
    metrics_path: str | Path,
    artifact_path: str | Path,
    inventory_path: str | Path | None = None,
) -> dict:
    """Prove native render persisted one render/GT PNG per evaluated frame."""

    metrics_path = Path(metrics_path).expanduser().resolve()
    validate_metrics(metrics_path, artifact_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    camera_ids, _, _ = _full_artifact_contract(artifact_path)
    per_camera_counts = {camera_id: int(metrics["per_camera"][camera_id]["n_frames"]) for camera_id in camera_ids}
    expected_count = sum(per_camera_counts.values())

    step_root = metrics_path.parent / f"ours_{FULL_ITERATIONS}"
    render_dir = step_root / "renders"
    gt_dir = step_root / "gt"
    render_paths = sorted(render_dir.glob("*.png")) if render_dir.is_dir() else []
    gt_paths = sorted(gt_dir.glob("*.png")) if gt_dir.is_dir() else []
    if len(render_paths) != expected_count:
        raise ValueError(f"native render PNG count {len(render_paths)} != evaluated frames {expected_count}")
    if len(gt_paths) != expected_count:
        raise ValueError(f"native GT PNG count {len(gt_paths)} != evaluated frames {expected_count}")
    render_names = [path.name for path in render_paths]
    gt_names = [path.name for path in gt_paths]
    if render_names != gt_names:
        raise ValueError("native render and GT PNG filenames do not match")
    for path in render_paths:
        _validate_native_png(path, "render")
    for path in gt_paths:
        _validate_native_png(path, "GT")

    inventory = {
        "schema_version": 1,
        "profile": FULL_RUN_PROFILE,
        "global_step": FULL_ITERATIONS,
        "native_resolution": list(NATIVE_RESOLUTION),
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


def _require_full_manifest(value: dict) -> None:
    if value.get("profile") != FULL_RUN_PROFILE:
        raise ValueError(f"run manifest profile {value.get('profile')!r} != {FULL_RUN_PROFILE!r}")
    v4_profile = value.get("ncore_readiness_profile") == V4_MULTILAYER_READINESS_PROFILE
    duration_sec = V4_FULL_DURATION_SEC if v4_profile else -1
    expected_contract = {
        "iterations": FULL_ITERATIONS,
        "train_seek_offset_sec": 0.0,
        "train_duration_sec": duration_sec,
        "val_seek_offset_sec": 0.0,
        "val_duration_sec": duration_sec,
        "native_resolution": list(NATIVE_RESOLUTION),
        "arm_order": ["P", "F"],
    }
    if value.get("contract") != expected_contract:
        raise ValueError("run manifest full scientific contract is invalid")
    if value.get("status") not in {"running", "failed", "complete"}:
        raise ValueError(f"run manifest status is invalid: {value.get('status')!r}")


def _prepare_full_run_manifest(
    run_id: str,
    git_commit: str,
    sources: dict[str, str | Path],
    *,
    ncore_readiness_profile: str | None = None,
) -> dict:
    if not run_id or not git_commit:
        raise ValueError("run_id and git_commit must be non-empty")
    if ncore_readiness_profile not in (None, V4_MULTILAYER_READINESS_PROFILE):
        raise ValueError(f"unsupported NCore readiness profile: {ncore_readiness_profile!r}")
    expanded_sources = _expanded_full_sources(
        sources,
        ncore_readiness_profile=ncore_readiness_profile,
    )
    _full_artifact_contract(expanded_sources["artifact"])
    duration_sec = (
        V4_FULL_DURATION_SEC
        if ncore_readiness_profile == V4_MULTILAYER_READINESS_PROFILE
        else -1
    )
    value = {
        "schema_version": 3,
        "run_id": run_id,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "profile": FULL_RUN_PROFILE,
        "contract": {
            "iterations": FULL_ITERATIONS,
            "train_seek_offset_sec": 0.0,
            "train_duration_sec": duration_sec,
            "val_seek_offset_sec": 0.0,
            "val_duration_sec": duration_sec,
            "native_resolution": list(NATIVE_RESOLUTION),
            "arm_order": ["P", "F"],
        },
        "sources": {name: _file_record(source) for name, source in sorted(expanded_sources.items())},
        "arms": {},
    }
    if ncore_readiness_profile is not None:
        value["ncore_readiness_profile"] = ncore_readiness_profile
    return value


def create_full_run_manifest(
    path: str | Path,
    run_id: str,
    git_commit: str,
    sources: dict[str, str | Path],
    *,
    ncore_readiness_profile: str | None = None,
) -> dict:
    value = _prepare_full_run_manifest(
        run_id,
        git_commit,
        sources,
        ncore_readiness_profile=ncore_readiness_profile,
    )
    if ncore_readiness_profile == V4_MULTILAYER_READINESS_PROFILE:
        validate_ncore_data_readiness(
            sources["dataset_manifest"],
            EXPECTED_CAMERA_IDS,
            required_aux=V4_REQUIRED_AUX_TYPES,
        )
    _write_run_manifest(path, value, exclusive=True)
    return value


def verify_full_run_manifest(path: str | Path, current_git_commit: str) -> dict:
    value = _load_run_manifest(path)
    if value.get("schema_version") != 3:
        raise ValueError(f"unsupported full run manifest schema_version={value.get('schema_version')!r}")
    if value.get("git_commit") != current_git_commit:
        raise ValueError(f"git commit drift: recorded={value.get('git_commit')!r} current={current_git_commit!r}")
    _require_full_manifest(value)
    sources = value.get("sources")
    profile = value.get("ncore_readiness_profile")
    if profile not in (None, V4_MULTILAYER_READINESS_PROFILE):
        raise ValueError(f"unsupported NCore readiness profile: {profile!r}")
    expected_source_names = {"dataset_manifest", *_FULL_STATIC_SOURCES}
    if profile is not None:
        expected_source_names.update(
            {
                "data_readiness_validator",
                "v4_driver_validator",
                "v4_provenance_sidecar",
                "v4_survey_artifact",
            }
        )
    if not isinstance(sources, dict) or set(sources) != expected_source_names:
        raise ValueError("full run manifest source set is invalid")
    for name, record in sources.items():
        _verify_file_record(record, context=name, drift_kind="source")
    validate_frozen_b6a9_manifest(sources["dataset_manifest"]["path"])
    arms = value.get("arms", {})
    if not isinstance(arms, dict):
        raise ValueError("run manifest arms must be a mapping")
    for arm, evidence in arms.items():
        if not isinstance(evidence, dict):
            raise ValueError(f"run manifest Arm {arm} evidence is invalid")
        missing = _FULL_REQUIRED_OUTPUT_NAMES - set(evidence)
        if missing:
            raise ValueError(f"run manifest Arm {arm} outputs missing {sorted(missing)}")
        _verify_file_record(
            evidence["native_render_inventory"],
            context=f"Arm {arm} output native_render_inventory",
            drift_kind="output",
        )
        if value.get("status") == "complete":
            current_inventory = validate_native_render_tree(
                evidence["metrics"]["path"], value["sources"]["artifact"]["path"]
            )
            recorded_inventory = json.loads(
                Path(evidence["native_render_inventory"]["path"]).read_text(encoding="utf-8")
            )
            if current_inventory != recorded_inventory or current_inventory != evidence.get("native_render"):
                raise ValueError(f"Arm {arm} native render tree drifted after completion")
    return value


def _require_record_source_argument(value: dict, source_name: str, path: str | Path) -> Path:
    record = value["sources"][source_name]
    actual_path = Path(path).expanduser().resolve()
    if str(actual_path) != record["path"]:
        raise ValueError(
            f"{source_name} path does not match run manifest source: " f"recorded={record['path']} actual={actual_path}"
        )
    actual_sha256 = sha256_file(actual_path)
    if actual_sha256 != record["sha256"]:
        raise ValueError(
            f"{source_name} hash does not match run manifest source: "
            f"recorded={record['sha256']} actual={actual_sha256}"
        )
    return actual_path


def record_full_arm_outputs(
    manifest_path: str | Path,
    arm: str,
    parsed_yaml: str | Path,
    checkpoint_path: str | Path,
    metrics_path: str | Path,
    train_log: str | Path,
    eval_log: str | Path,
    native_render_inventory: str | Path,
    artifact_path: str | Path,
    input_manifest_path: str | Path,
    current_git_commit: str,
) -> dict:
    arm = _arm(arm)
    value = verify_full_run_manifest(manifest_path, current_git_commit)
    if value.get("status") != "running":
        raise ValueError(f"record-arm requires manifest status 'running', got {value.get('status')!r}")
    arms = value["arms"]
    if arm in arms:
        raise ValueError(f"Arm {arm} evidence already recorded")
    if list(arms) not in ([], ["P"]):
        raise ValueError(f"run manifest arm evidence/order is invalid: {list(arms)}")
    expected_next_arm = "P" if not arms else "F"
    if arm != expected_next_arm:
        raise ValueError(f"Arm execution order violation: expected {expected_next_arm}, got {arm}")
    frozen_artifact = _require_record_source_argument(value, "artifact", artifact_path)
    frozen_manifest = _require_record_source_argument(value, "dataset_manifest", input_manifest_path)
    validate_full_parsed_config(parsed_yaml, arm, frozen_artifact, frozen_manifest)
    validate_full_checkpoint(checkpoint_path, arm, frozen_artifact, frozen_manifest)
    train_counts = validate_training_log(train_log, arm, frozen_artifact)
    validate_metrics(metrics_path, frozen_artifact)
    actual_inventory = validate_native_render_tree(metrics_path, frozen_artifact)
    recorded_inventory = json.loads(Path(native_render_inventory).read_text(encoding="utf-8"))
    if recorded_inventory != actual_inventory:
        raise ValueError("native render inventory does not match current render tree")

    arms[arm] = {
        "parsed_yaml": _file_record(parsed_yaml),
        "checkpoint": {**_file_record(checkpoint_path), "global_step": FULL_ITERATIONS},
        "metrics": _file_record(metrics_path),
        "train_log": _file_record(train_log),
        "eval_log": _file_record(eval_log),
        "native_render_inventory": _file_record(native_render_inventory),
        "train_frames_per_camera": train_counts,
        "native_render": actual_inventory,
    }
    _write_run_manifest(manifest_path, value)
    return value


def finalize_full_run_manifest(manifest_path: str | Path, current_git_commit: str) -> dict:
    value = verify_full_run_manifest(manifest_path, current_git_commit)
    if value.get("status") != "running":
        raise ValueError(f"finalize requires manifest status 'running', got {value.get('status')!r}")
    _verify_arm_output_records(value, require_both=True)
    if set(value["arms"]) != {"P", "F"}:
        raise ValueError("run manifest must contain both Arm P and Arm F evidence")
    p_train_counts = value["arms"]["P"].get("train_frames_per_camera")
    f_train_counts = value["arms"]["F"].get("train_frames_per_camera")
    if p_train_counts != f_train_counts:
        raise ValueError(f"P/F train frame-count mismatch: P={p_train_counts} F={f_train_counts}")

    for arm in ("P", "F"):
        evidence = value["arms"][arm]
        current_inventory = validate_native_render_tree(
            evidence["metrics"]["path"], value["sources"]["artifact"]["path"]
        )
        recorded_inventory = json.loads(Path(evidence["native_render_inventory"]["path"]).read_text(encoding="utf-8"))
        if current_inventory != recorded_inventory or current_inventory != evidence.get("native_render"):
            raise ValueError(f"Arm {arm} native render tree drifted after evidence recording")

    p_native = value["arms"]["P"]["native_render"]
    f_native = value["arms"]["F"]["native_render"]
    if p_native["gt_png_count"] != f_native["gt_png_count"]:
        raise ValueError(f"P/F GT PNG count mismatch: P={p_native['gt_png_count']} F={f_native['gt_png_count']}")
    if p_native["gt_png_names"] != f_native["gt_png_names"]:
        raise ValueError("P/F GT PNG frame set mismatch")
    if p_native["gt_tree_sha256"] != f_native["gt_tree_sha256"]:
        raise ValueError(
            "P/F GT tree SHA-256 mismatch: " f"P={p_native['gt_tree_sha256']} F={f_native['gt_tree_sha256']}"
        )

    scientific_hash = compare_parsed_configs(
        value["arms"]["P"]["parsed_yaml"]["path"],
        value["arms"]["F"]["parsed_yaml"]["path"],
    )
    frame_counts = compare_per_camera_frame_counts(
        value["arms"]["P"]["metrics"]["path"],
        value["arms"]["F"]["metrics"]["path"],
        value["sources"]["artifact"]["path"],
    )
    value["comparison"] = {
        "normalized_scientific_config_sha256": scientific_hash,
        "only_representation_path_differs": True,
        "arm_order": ["P", "F"],
        "train_frames_per_camera": p_train_counts,
        "per_camera_n_frames": frame_counts,
        "gt_png_count": p_native["gt_png_count"],
        "gt_tree_sha256": p_native["gt_tree_sha256"],
    }
    value["status"] = "complete"
    value["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_run_manifest(manifest_path, value)
    return value


def mark_full_run_failed(manifest_path: str | Path, stage: str, exit_code: int) -> dict:
    value = _load_run_manifest(manifest_path)
    _require_full_manifest(value)
    if value.get("status") == "complete":
        raise ValueError("cannot mark a completed run failed")
    value["status"] = "failed"
    value["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
    value["failure"] = {"stage": stage, "exit_code": int(exit_code)}
    _write_run_manifest(manifest_path, value)
    return value


def run_preflight(
    repo_root: str | Path,
    config_name: str,
    artifact_path: str | Path,
    input_manifest_path: str | Path,
    ncore_readiness_profile: str | None = None,
) -> None:
    repo_root = Path(repo_root).expanduser().resolve()
    artifact_path = Path(artifact_path).expanduser().resolve()
    input_manifest_path = Path(input_manifest_path).expanduser().resolve()
    if not input_manifest_path.is_file():
        raise ValueError(f"dataset manifest missing: {input_manifest_path}")
    if not artifact_path.is_file():
        raise ValueError(f"FTheta artifact missing: {artifact_path}")
    validate_frozen_b6a9_manifest(input_manifest_path)
    if ncore_readiness_profile not in (None, V4_MULTILAYER_READINESS_PROFILE):
        raise ValueError(f"unsupported NCore readiness profile: {ncore_readiness_profile!r}")
    if ncore_readiness_profile == V4_MULTILAYER_READINESS_PROFILE:
        expected_config_name = str(Path(V4_MULTILAYER_PROFILE_CONTRACT["config"]["path"]).relative_to("configs").with_suffix(""))
        if config_name != expected_config_name:
            raise ValueError(
                f"v4-multilayer config name mismatch: expected={expected_config_name!r} actual={config_name!r}"
            )
        validate_v4_multilayer_profile_contract(
            repo_root,
            repo_root / V4_MULTILAYER_PROFILE_CONTRACT["config"]["path"],
            artifact_path,
        )
        validate_v4_multilayer_dataset_contract(input_manifest_path)
        _full_artifact_contract(artifact_path)
        validate_ncore_data_readiness(
            input_manifest_path,
            EXPECTED_CAMERA_IDS,
            required_aux=V4_REQUIRED_AUX_TYPES,
        )
    else:
        _full_artifact_contract(artifact_path)

    duration_sec = (
        V4_FULL_DURATION_SEC
        if ncore_readiness_profile == V4_MULTILAYER_READINESS_PROFILE
        else -1
    )
    with initialize_config_dir(config_dir=str(repo_root / "configs"), version_base=None):
        base = compose(config_name=config_name)
    for arm in ("P", "F"):
        config: DictConfig = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
        updates = {
            "n_iterations": FULL_ITERATIONS,
            "seed_initialization": 42,
            "test_last": True,
            "num_workers": 10,
            "path": str(input_manifest_path),
            "out_dir": "/preflight/no-output",
            "experiment_name": f"preflight_arm{arm}",
            "dataset.train.seek_offset_sec": 0.0,
            "dataset.train.duration_sec": duration_sec,
            "dataset.val.seek_offset_sec": 0.0,
            "dataset.val.duration_sec": duration_sec,
            "dataset.downsample": 1.0,
            "dataset.n_val_image_subsample": 1,
            "dataset.camera_max_fov_deg": 190.0,
            "dataset.mask_forward_invalid_pixels": True,
            "dataset.opencv_pinhole_use_validity_domain": False,
            "dataset.load_lidar_depth_map": False,
            "dataset.load_depth_prior": False,
            "trainer.use_lidar_depth": False,
            "trainer.use_depth_prior": False,
            "trainer.sky_backend": "mlp",
            "dataset.ftheta_params_path": None if arm == "P" else str(artifact_path),
        }
        for dotted_key, expected in updates.items():
            OmegaConf.update(config, dotted_key, expected, merge=False)
        validate_full_scientific_config(config, arm, artifact_path, input_manifest_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--repo-root", required=True)
    preflight.add_argument("--config-name", required=True)
    preflight.add_argument("--artifact", required=True)
    preflight.add_argument("--input-manifest", required=True)
    preflight.add_argument(
        "--ncore-readiness-profile",
        choices=(V4_MULTILAYER_READINESS_PROFILE,),
        default=None,
    )

    log = commands.add_parser("log")
    log.add_argument("--path", required=True)
    log.add_argument("--arm", required=True, choices=("P", "F"))
    log.add_argument("--artifact", required=True)

    checkpoint = commands.add_parser("checkpoint")
    checkpoint.add_argument("--path", required=True)
    checkpoint.add_argument("--arm", required=True, choices=("P", "F"))
    checkpoint.add_argument("--artifact", required=True)
    checkpoint.add_argument("--input-manifest", required=True)

    metrics = commands.add_parser("metrics")
    metrics.add_argument("--path", required=True)
    metrics.add_argument("--artifact", required=True)

    render_tree = commands.add_parser("render-tree")
    render_tree.add_argument("--metrics", required=True)
    render_tree.add_argument("--artifact", required=True)
    render_tree.add_argument("--inventory", required=True)

    create = commands.add_parser("manifest-create")
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

    verify = commands.add_parser("manifest-verify")
    verify.add_argument("--path", required=True)
    verify.add_argument("--repo-root", required=True)

    record = commands.add_parser("record-arm")
    record.add_argument("--manifest", required=True)
    record.add_argument("--arm", required=True, choices=("P", "F"))
    record.add_argument("--parsed-yaml", required=True)
    record.add_argument("--checkpoint", required=True)
    record.add_argument("--metrics", required=True)
    record.add_argument("--train-log", required=True)
    record.add_argument("--eval-log", required=True)
    record.add_argument("--native-render-inventory", required=True)
    record.add_argument("--artifact", required=True)
    record.add_argument("--input-manifest", required=True)
    record.add_argument("--repo-root", required=True)

    finalize = commands.add_parser("finalize")
    finalize.add_argument("--manifest", required=True)
    finalize.add_argument("--repo-root", required=True)

    fail = commands.add_parser("manifest-fail")
    fail.add_argument("--manifest", required=True)
    fail.add_argument("--stage", required=True)
    fail.add_argument("--exit-code", required=True, type=int)
    return parser


def _current_clean_commit(repo_root: str | Path) -> str:
    ensure_tracked_worktree_clean(repo_root)
    return _git_commit(repo_root)


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "preflight":
        run_preflight(
            args.repo_root,
            args.config_name,
            args.artifact,
            args.input_manifest,
            args.ncore_readiness_profile,
        )
    elif args.command == "log":
        validate_training_log(args.path, args.arm, args.artifact)
    elif args.command == "checkpoint":
        validate_full_checkpoint(args.path, args.arm, args.artifact, args.input_manifest)
    elif args.command == "metrics":
        validate_metrics(args.path, args.artifact)
    elif args.command == "render-tree":
        validate_native_render_tree(args.metrics, args.artifact, args.inventory)
    elif args.command == "manifest-create":
        # Keep cheap cleanliness, frozen-provenance, artifact-contract, and
        # source hashing checks ahead of the expensive NCore opens. Readiness
        # is deliberately the final operation before the exclusive write.
        git_commit = _current_clean_commit(args.repo_root)
        if args.ncore_readiness_profile == V4_MULTILAYER_READINESS_PROFILE:
            if not args.expected_commit:
                raise ValueError("v4 manifest-create requires --expected-commit")
            if args.expected_commit != git_commit:
                raise ValueError(
                    f"v4 expected commit mismatch: expected={args.expected_commit} current={git_commit}"
                )
        create_full_run_manifest(
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
        )
    elif args.command == "manifest-verify":
        verify_full_run_manifest(args.path, _current_clean_commit(args.repo_root))
    elif args.command == "record-arm":
        current_git_commit = _current_clean_commit(args.repo_root)
        record_full_arm_outputs(
            args.manifest,
            args.arm,
            args.parsed_yaml,
            args.checkpoint,
            args.metrics,
            args.train_log,
            args.eval_log,
            args.native_render_inventory,
            args.artifact,
            args.input_manifest,
            current_git_commit,
        )
    elif args.command == "finalize":
        finalize_full_run_manifest(args.manifest, _current_clean_commit(args.repo_root))
    else:
        mark_full_run_failed(args.manifest, args.stage, args.exit_code)


if __name__ == "__main__":
    main()
