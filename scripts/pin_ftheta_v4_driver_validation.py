#!/usr/bin/env python3
"""CPU-only launch gate shared by the dedicated FTheta v4 P/F drivers.

This module intentionally does not import torch, the trainer, or renderer.  It
freezes the committed source, v4 provenance, canonical NCore data, resolved
scientific configuration, and output namespace before a driver may expose a
GPU or create a run directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Callable

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from scripts.ncore_data_readiness import (
    V4_MULTILAYER_PROFILE_CONTRACT,
    V4_REQUIRED_AUX_TYPES,
    validate_ncore_data_readiness,
    validate_v4_multilayer_dataset_contract,
    validate_v4_multilayer_profile_contract,
)
from threedgrut.ftheta_override_contract import (
    FTHETA_PARAMETER_KEYS,
    _validate_ftheta_parameters,
)


EXPECTED_BRANCH = "codex/ftheta-full-domain-v4"
EXPECTED_CONFIG_NAME = "apps/ncore_3dgut_mcmc_multilayer_inceptio_7cam_v4"
EXPECTED_GENERATED_AT = "2026-07-18T12:29:38+08:00"
LEGACY_V3_ARTIFACT_SHA256 = "73965c6d10693b7742c7afb538169990a94d2457890738a7a4f66d13fcd0a450"
LEGACY_V3_MAX_ANGLE = 0.7303101158645611
V4_DRIVER_VALIDATOR_RELATIVE_PATH = "scripts/pin_ftheta_v4_driver_validation.py"

EXPECTED_CAMERA_IDS = tuple(V4_MULTILAYER_PROFILE_CONTRACT["camera_order"])
EXPECTED_PARAMETER_FINGERPRINTS = {
    "camera_front_wide_120fov": "0785f301bb8ee9bc3084d1882b2459d60055afe47633a800ff9be18997c7aa55",
    "camera_cross_left_120fov": "8a4bbf97ccef47c95645f63450b08612a9f383b657ebec6cbb6e2580398e2ac2",
    "camera_cross_right_120fov": "49fd193cbebf9db682ce4f7f8fa41b9a7bc56c5d79d83d72ef2003a0bc662f5c",
    "camera_left_wide_90fov": "683441e06e127ec7dbbedf672fcec176760323e19e03141838609aac2a5d381c",
    "camera_right_wide_90fov": "e5410c753599762326b049e6dfb407d75ea6754ba99b0968717410a8eb9886a5",
    "camera_back_rear_wide_90fov": "7e7a327a84a8f55ef4a246824f1e26eb9956c85d84194a206e5bca04736e87b3",
    "camera_rear_left_70fov": "d92d0cf162a5abe75c572b3211d487f9b368542e6198680b5d89996290046a4a",
}

MODE_SPECS = {
    "smoke": {
        "driver": "scripts/pin_ftheta_7cam_v4_smoke.sh",
        "run_base_name": "pin_ftheta_v4_smoke_runs",
        "iterations": 5000,
        "train_duration_sec": 5.0,
        "val_duration_sec": 5.0,
        "experiment_suffix": "5s_5k",
    },
    "full": {
        "driver": "scripts/pin_ftheta_7cam_v4_full_ab.sh",
        "run_base_name": "pin_ftheta_v4_full_ab_runs",
        "iterations": 30000,
        "train_duration_sec": 20.0,
        "val_duration_sec": 20.0,
        "experiment_suffix": "20s_30k",
    },
}

_LEGACY_OUTPUT_ROOT_NAMES = frozenset(
    {
        "pin_ftheta_smoke_runs",
        "pin_ftheta_7cam_full_ab_runs",
        "pin_ftheta_v3_smoke_runs",
        "pin_ftheta_v3_full_ab_runs",
    }
)
_EXPECTED_LAYERS = ["background", "road", "dynamic_rigids", "sky_envmap"]
_REQUIRED_SUBMODULE_HEADERS = (
    "thirdparty/tiny-cuda-nn/include/tiny-cuda-nn/common.h",
    "threedgrt_tracer/dependencies/optix-dev/include/optix.h",
)
_REQUIRED_SUBMODULE_PATHS = (
    "thirdparty/tiny-cuda-nn",
    "threedgrt_tracer/dependencies/optix-dev",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo_root, text=True, stderr=subprocess.STDOUT
        ).rstrip()
    except (OSError, subprocess.CalledProcessError) as exc:
        output = getattr(exc, "output", "")
        raise ValueError(f"cannot run git {' '.join(args)} in {repo_root}: {output}") from exc


def validate_release_worktree(
    repo_root: str | Path,
    driver_path: str | Path,
    expected_commit: str,
) -> dict[str, str]:
    """Require an exact clean committed branch before expensive data opens."""

    root = Path(repo_root).expanduser().resolve()
    branch = _git(root, "branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise ValueError(f"v4 launch branch mismatch: expected={EXPECTED_BRANCH!r} actual={branch!r}")
    commit = _git(root, "rev-parse", "HEAD")
    if len(expected_commit) != 40 or any(char not in "0123456789abcdef" for char in expected_commit):
        raise ValueError("expected commit must be a full lowercase 40-character SHA-1")
    if commit != expected_commit:
        raise ValueError(f"v4 launch commit mismatch: expected={expected_commit} actual={commit}")

    dirty = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if dirty:
        raise ValueError(f"v4 launch requires a completely clean worktree:\n{dirty}")

    driver = Path(driver_path).expanduser().resolve()
    try:
        driver_relative = driver.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"v4 driver must be inside the repository: {driver}") from exc
    required_relative_paths = (
        driver_relative,
        V4_DRIVER_VALIDATOR_RELATIVE_PATH,
        V4_MULTILAYER_PROFILE_CONTRACT["config"]["path"],
        V4_MULTILAYER_PROFILE_CONTRACT["runtime_artifact"]["path"],
        V4_MULTILAYER_PROFILE_CONTRACT["provenance_sidecar"]["path"],
        V4_MULTILAYER_PROFILE_CONTRACT["survey_artifact"]["path"],
    )
    for relative_path in required_relative_paths:
        _git(root, "ls-files", "--error-unmatch", "--", relative_path)
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative_path}"],
            cwd=root,
            capture_output=True,
            check=False,
        )
        if committed.returncode != 0:
            raise ValueError(f"v4 release source is not readable from HEAD: {relative_path}")
        current_path = root / relative_path
        try:
            current = current_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"v4 release source is missing: {current_path}") from exc
        if committed.stdout != current:
            raise ValueError(f"v4 release source differs from HEAD: {relative_path}")

    submodule_status = _git(root, "submodule", "status", "--recursive")
    status_records: list[tuple[str, str, str, str]] = []
    for line in submodule_status.splitlines():
        if line[:1] in {"-", "+", "U"}:
            raise ValueError(f"v4 release submodule is not initialized at its gitlink commit: {line}")
        if line[:1] != " ":
            raise ValueError(f"v4 release submodule status is malformed: {line}")
        fields = line[1:].split(maxsplit=2)
        if len(fields) < 2:
            raise ValueError(f"v4 release submodule status is malformed: {line}")
        status_commit, relative_path = fields[:2]
        status_records.append((relative_path, status_commit, line, fields[2] if len(fields) > 2 else ""))

    initialized_submodules: dict[str, str] = {}
    for relative_path, status_commit, line, _description in sorted(
        status_records, key=lambda record: (len(Path(record[0]).parts), record[0])
    ):
        parent_candidates = [
            known_path
            for known_path in initialized_submodules
            if relative_path.startswith(f"{known_path}/")
        ]
        owner_relative = max(parent_candidates, key=len, default=None)
        if owner_relative is None:
            owner_root = root
            child_path_in_owner = relative_path
        else:
            owner_root = root / owner_relative
            child_path_in_owner = Path(relative_path).relative_to(owner_relative).as_posix()

        tree_entry = _git(owner_root, "ls-tree", "HEAD", "--", child_path_in_owner)
        try:
            metadata, listed_path = tree_entry.split("\t", maxsplit=1)
            mode, object_type, gitlink_commit = metadata.split()
        except ValueError as exc:
            raise ValueError(
                f"v4 release submodule gitlink is malformed: path={relative_path!r} entry={tree_entry!r}"
            ) from exc
        if listed_path != child_path_in_owner or mode != "160000" or object_type != "commit":
            raise ValueError(
                "v4 release submodule is not a gitlink in its owning repository: "
                f"path={relative_path!r} owner={owner_relative or '.'!r} entry={tree_entry!r}"
            )
        submodule_root = (root / relative_path).resolve()
        try:
            submodule_root.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"v4 release submodule escapes repository: {relative_path!r}") from exc
        actual_commit = _git(submodule_root, "rev-parse", "HEAD")
        if status_commit != gitlink_commit or actual_commit != gitlink_commit:
            raise ValueError(
                "v4 release submodule commit mismatch: "
                f"path={relative_path!r} gitlink={gitlink_commit} "
                f"status={status_commit} actual={actual_commit}"
            )
        submodule_dirty = _git(
            submodule_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        )
        if submodule_dirty:
            raise ValueError(
                f"v4 release submodule must be recursively clean: path={relative_path!r}\n"
                f"{submodule_dirty}"
            )
        initialized_submodules[relative_path] = actual_commit

    missing_submodules = sorted(set(_REQUIRED_SUBMODULE_PATHS) - initialized_submodules.keys())
    if missing_submodules:
        raise ValueError(
            f"v4 release required submodules are absent or uninitialized: {missing_submodules}"
        )
    missing_headers = [relative for relative in _REQUIRED_SUBMODULE_HEADERS if not (root / relative).is_file()]
    if missing_headers:
        raise ValueError(f"v4 release submodule content is incomplete: missing={missing_headers}")
    return {
        "branch": branch,
        "commit": commit,
        "submodule_status_policy": "initialized-clean-exact-gitlink-with-required-headers",
    }


def validate_mode_paths(
    mode: str,
    repo_root: str | Path,
    driver_path: str | Path,
    config_name: str,
    artifact_path: str | Path,
    input_manifest_path: str | Path,
    run_base: str | Path,
) -> dict[str, Path]:
    if mode not in MODE_SPECS:
        raise ValueError(f"unsupported v4 driver mode: {mode!r}")
    spec = MODE_SPECS[mode]
    root = Path(repo_root).expanduser().resolve()
    expected_driver = (root / spec["driver"]).resolve()
    driver = Path(driver_path).expanduser().resolve()
    if driver != expected_driver:
        raise ValueError(f"v4 {mode} driver path mismatch: expected={expected_driver} actual={driver}")
    if config_name != EXPECTED_CONFIG_NAME:
        raise ValueError(
            f"v4 config name mismatch: expected={EXPECTED_CONFIG_NAME!r} actual={config_name!r}"
        )
    expected_artifact = (root / V4_MULTILAYER_PROFILE_CONTRACT["runtime_artifact"]["path"]).resolve()
    artifact = Path(artifact_path).expanduser().resolve()
    if artifact != expected_artifact:
        raise ValueError(f"v4 artifact path mismatch: expected={expected_artifact} actual={artifact}")

    manifest = Path(input_manifest_path).expanduser().resolve()
    output = Path(run_base).expanduser().resolve()
    if output.name != spec["run_base_name"]:
        raise ValueError(
            f"v4 {mode} output root name mismatch: expected={spec['run_base_name']!r} actual={output.name!r}"
        )
    if output.name in _LEGACY_OUTPUT_ROOT_NAMES:
        raise ValueError(f"v4 output root collides with a legacy namespace: {output}")
    for evidence_path, label in ((artifact, "artifact"), (manifest, "dataset manifest")):
        if evidence_path == output or output in evidence_path.parents or evidence_path in output.parents:
            raise ValueError(f"v4 output root and {label} path are not isolated: {output} vs {evidence_path}")
    return {
        "driver": driver,
        "artifact": artifact,
        "manifest": manifest,
        "run_base": output,
    }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in v4 artifact: {key!r}")
        result[key] = value
    return result


def validate_v4_artifact(path: str | Path) -> dict[str, str]:
    artifact_path = Path(path).expanduser().resolve()
    actual_sha256 = _sha256_file(artifact_path)
    if actual_sha256 == LEGACY_V3_ARTIFACT_SHA256:
        raise ValueError("legacy v3 FTheta artifact SHA-256 is forbidden")
    expected_sha256 = V4_MULTILAYER_PROFILE_CONTRACT["runtime_artifact"]["sha256"]
    try:
        value = json.loads(artifact_path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"v4 runtime artifact is unreadable: {artifact_path}: {exc}") from exc
    if not isinstance(value, dict) or tuple(value) != EXPECTED_CAMERA_IDS:
        actual_order = tuple(value) if isinstance(value, dict) else ()
        raise ValueError(
            f"v4 runtime artifact camera order mismatch: expected={EXPECTED_CAMERA_IDS} actual={actual_order}"
        )
    fingerprints: dict[str, str] = {}
    for camera_id in EXPECTED_CAMERA_IDS:
        normalized = _validate_ftheta_parameters(camera_id, value[camera_id])
        if set(normalized) != FTHETA_PARAMETER_KEYS:
            raise ValueError(f"{camera_id}: v4 artifact must contain the exact eight FTheta keys")
        max_angle = float(normalized["max_angle"])
        if math.isclose(max_angle, LEGACY_V3_MAX_ANGLE, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{camera_id}: legacy 0.730310 rad max-angle sentinel is forbidden")
        canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
        fingerprints[camera_id] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"v4 runtime artifact SHA-256 mismatch: expected={expected_sha256} actual={actual_sha256}"
        )
    if fingerprints != EXPECTED_PARAMETER_FINGERPRINTS:
        raise ValueError("v4 per-camera parameter fingerprints do not match the frozen release")
    return fingerprints


def validate_v4_provenance(repo_root: str | Path, artifact_path: str | Path) -> dict[str, str]:
    root = Path(repo_root).expanduser().resolve()
    fingerprints = validate_v4_artifact(artifact_path)
    resolved = validate_v4_multilayer_profile_contract(
        root,
        root / V4_MULTILAYER_PROFILE_CONTRACT["config"]["path"],
        artifact_path,
    )
    sidecar_path = resolved["provenance_sidecar"]
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"v4 provenance sidecar is missing or unreadable: {sidecar_path}: {exc}") from exc
    if sidecar.get("generated_at") != EXPECTED_GENERATED_AT:
        raise ValueError(
            "v4 provenance sidecar generated_at drift: "
            f"expected={EXPECTED_GENERATED_AT!r} actual={sidecar.get('generated_at')!r}"
        )
    return fingerprints


def _require_config_value(config: DictConfig, dotted_key: str, expected: Any) -> None:
    actual = OmegaConf.select(config, dotted_key, default=None)
    if isinstance(actual, DictConfig) or hasattr(actual, "_content"):
        actual = OmegaConf.to_container(actual, resolve=False)
    if actual != expected:
        raise ValueError(f"v4 scientific config {dotted_key}: {actual!r} != {expected!r}")


def _build_arm_config(
    base: DictConfig,
    *,
    mode: str,
    arm: str,
    artifact_path: Path,
    input_manifest_path: Path,
) -> DictConfig:
    spec = MODE_SPECS[mode]
    config = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
    experiment = f"pin_ftheta_v4_7cam_arm{arm}_{spec['experiment_suffix']}"
    updates = {
        "n_iterations": spec["iterations"],
        "seed_initialization": 42,
        "test_last": True,
        "num_workers": 10,
        "path": str(input_manifest_path),
        "out_dir": f"/preflight/no-output/{mode}",
        "experiment_name": experiment,
        "dataset.train.seek_offset_sec": 0.0,
        "dataset.train.duration_sec": spec["train_duration_sec"],
        "dataset.val.seek_offset_sec": 0.0,
        "dataset.val.duration_sec": spec["val_duration_sec"],
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
    for dotted_key, value in updates.items():
        OmegaConf.update(config, dotted_key, value, merge=False)
    return config


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


def validate_resolved_pf_configs(
    mode: str,
    repo_root: str | Path,
    artifact_path: str | Path,
    input_manifest_path: str | Path,
) -> str:
    root = Path(repo_root).expanduser().resolve()
    artifact = Path(artifact_path).expanduser().resolve()
    manifest = Path(input_manifest_path).expanduser().resolve()
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        base = compose(config_name=EXPECTED_CONFIG_NAME)
    configs = {
        arm: _build_arm_config(
            base,
            mode=mode,
            arm=arm,
            artifact_path=artifact,
            input_manifest_path=manifest,
        )
        for arm in ("P", "F")
    }
    spec = MODE_SPECS[mode]
    scalar_contract = {
        "n_iterations": spec["iterations"],
        "seed_initialization": 42,
        "test_last": True,
        "num_workers": 10,
        "dataset.train.seek_offset_sec": 0.0,
        "dataset.train.duration_sec": spec["train_duration_sec"],
        "dataset.val.seek_offset_sec": 0.0,
        "dataset.val.duration_sec": spec["val_duration_sec"],
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
        "loss.use_opacity": False,
        "viz_4d.enabled": True,
    }
    for arm, config in configs.items():
        for dotted_key, expected in scalar_contract.items():
            _require_config_value(config, dotted_key, expected)
        _require_config_value(config, "dataset.camera_ids", list(EXPECTED_CAMERA_IDS))
        _require_config_value(config, "layers.enabled", _EXPECTED_LAYERS)
        _require_config_value(config, "loss.camera_loss_weights", {})
        if any("front_standard" in camera_id or "front_tele" in camera_id for camera_id in EXPECTED_CAMERA_IDS):
            raise ValueError("front_standard/front_tele must not appear in the active v4 cameras")
        expected_ftheta = None if arm == "P" else str(artifact)
        _require_config_value(config, "dataset.ftheta_params_path", expected_ftheta)

    containers = {
        arm: OmegaConf.to_container(config, resolve=False) for arm, config in configs.items()
    }
    differences = _leaf_differences(containers["P"], containers["F"])
    allowed = {"dataset.ftheta_params_path", "experiment_name", "out_dir"}
    required = {"dataset.ftheta_params_path", "experiment_name"}
    if not required.issubset(differences) or not differences.issubset(allowed):
        raise ValueError(
            "P/F resolved config differs outside representation/output bookkeeping: "
            f"required={sorted(required)} allowed={sorted(allowed)} actual={sorted(differences)}"
        )
    normalized = copy.deepcopy(containers["P"])
    normalized.pop("experiment_name", None)
    normalized.pop("out_dir", None)
    dataset = normalized.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("v4 normalized config has no dataset mapping")
    dataset.pop("ftheta_params_path", None)
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_preflight(
    *,
    mode: str,
    repo_root: str | Path,
    driver_path: str | Path,
    config_name: str,
    artifact_path: str | Path,
    input_manifest_path: str | Path,
    run_base: str | Path,
    expected_commit: str,
    readiness_validator: Callable[..., dict[str, Any]] = validate_ncore_data_readiness,
) -> dict[str, Any]:
    release = validate_release_worktree(repo_root, driver_path, expected_commit)
    paths = validate_mode_paths(
        mode,
        repo_root,
        driver_path,
        config_name,
        artifact_path,
        input_manifest_path,
        run_base,
    )
    fingerprints = validate_v4_provenance(repo_root, paths["artifact"])
    validate_v4_multilayer_dataset_contract(paths["manifest"])
    readiness = readiness_validator(
        paths["manifest"],
        EXPECTED_CAMERA_IDS,
        required_aux=V4_REQUIRED_AUX_TYPES,
    )
    normalized_hash = validate_resolved_pf_configs(
        mode,
        repo_root,
        paths["artifact"],
        paths["manifest"],
    )
    return {
        **release,
        "mode": mode,
        "config_name": config_name,
        "artifact_sha256": V4_MULTILAYER_PROFILE_CONTRACT["runtime_artifact"]["sha256"],
        "manifest_sha256": V4_MULTILAYER_PROFILE_CONTRACT["manifest_sha256"],
        "camera_ids": list(EXPECTED_CAMERA_IDS),
        "parameter_fingerprints": fingerprints,
        "normalized_scientific_config_sha256": normalized_hash,
        "run_base": str(paths["run_base"]),
        "readiness": readiness,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--mode", choices=tuple(MODE_SPECS), required=True)
    preflight.add_argument("--repo-root", required=True)
    preflight.add_argument("--driver", required=True)
    preflight.add_argument("--config-name", required=True)
    preflight.add_argument("--artifact", required=True)
    preflight.add_argument("--input-manifest", required=True)
    preflight.add_argument("--run-base", required=True)
    preflight.add_argument("--expected-commit", required=True)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        value = run_preflight(
            mode=args.mode,
            repo_root=args.repo_root,
            driver_path=args.driver,
            config_name=args.config_name,
            artifact_path=args.artifact,
            input_manifest_path=args.input_manifest,
            run_base=args.run_base,
            expected_commit=args.expected_commit,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(json.dumps(value, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
