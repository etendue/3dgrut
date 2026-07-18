#!/usr/bin/env python3
"""Deterministically export the immutable b6a9 FTheta v4 artifact set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_FORBIDDEN_LEGACY_OUTPUTS = frozenset(
    {
        (_REPO_ROOT / "scripts" / "pin_ftheta_b6a9_7cam_params.json").resolve(),
        (_REPO_ROOT / "scripts" / "pin_ftheta_b6a9_9cam_params.json").resolve(),
    }
)

from scripts.pin_ftheta_camera_survey import survey_bundle  # noqa: E402


def _json_bytes(value: Any, *, sort_keys: bool) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=sort_keys, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def build_artifact(calibration_bundle: dict) -> dict:
    """Build the rich nine-camera survey artifact in memory."""
    survey = survey_bundle(calibration_bundle)
    return {
        "schema_version": 2,
        "provenance": survey["provenance"],
        "fitter_version": survey["fitter_version"],
        "quality_warning_thresholds": survey["quality_warning_thresholds"],
        "active_camera_order": survey["active_camera_order"],
        "excluded_from_runtime_camera_order": survey[
            "excluded_from_runtime_camera_order"
        ],
        "scope": survey["scope"],
        "v3_invalidation": survey["v3_invalidation"],
        "camera_order": survey["camera_order"],
        "all_hard_invariants_passed": survey["all_hard_invariants_passed"],
        "hard_failures": survey["hard_failures"],
        "active_subset_hard_invariants_passed": survey[
            "active_subset_hard_invariants_passed"
        ],
        "active_hard_failures": survey["active_hard_failures"],
        "quality_warning_cameras": survey["quality_warning_cameras"],
        "cameras": {
            camera_id: {
                "source_model_type": survey["cameras"][camera_id]["source_model_type"],
                "source_parameters_type": survey["cameras"][camera_id][
                    "source_parameters_type"
                ],
                "source_calibration_sha256": survey["cameras"][camera_id][
                    "source_calibration_sha256"
                ],
                "fitter_version": survey["cameras"][camera_id]["fitter_version"],
                "ftheta_parameters": survey["cameras"][camera_id][
                    "ftheta_parameters"
                ],
                "fit_metrics": survey["cameras"][camera_id]["fit_metrics"],
                "hard_invariants": survey["cameras"][camera_id][
                    "hard_invariants"
                ],
                "quality_warnings": survey["cameras"][camera_id][
                    "quality_warnings"
                ],
            }
            for camera_id in survey["camera_order"]
        },
    }


def build_runtime_artifact(survey: dict) -> dict:
    """Extract the loader-compatible exact active-camera parameter map."""
    active_order = list(survey["active_camera_order"])
    if len(active_order) != 7 or len(set(active_order)) != 7:
        raise ValueError(f"expected exactly seven active camera IDs; got {active_order}")
    missing = [camera_id for camera_id in active_order if camera_id not in survey["cameras"]]
    if missing:
        raise ValueError(f"active cameras missing from survey: {missing}")
    return {
        camera_id: survey["cameras"][camera_id]["ftheta_parameters"]
        for camera_id in active_order
    }


def _hashed_source_paths(calibrations_path: Path) -> list[Path]:
    return [
        calibrations_path,
        _REPO_ROOT / "threedgrut_playground" / "utils" / "ftheta_fitter.py",
        _REPO_ROOT / "threedgrut_playground" / "utils" / "opencv_inverse.py",
        _REPO_ROOT / "threedgrut" / "ftheta_override_contract.py",
        _REPO_ROOT / "scripts" / "pin_ftheta_camera_survey.py",
        Path(__file__).resolve(),
    ]


def _source_hashes(calibrations_path: Path) -> dict[str, str]:
    source_paths = _hashed_source_paths(calibrations_path)
    return {_display_path(path): _sha256_file(path) for path in source_paths}


def _generation_command(
    *,
    calibrations_path: Path,
    survey_output: Path,
    runtime_output: Path,
    provenance_output: Path,
    generated_at: str,
) -> str:
    return shlex.join(
        [
            ".venv/bin/python",
            "scripts/export_9cam_ftheta_params.py",
            "--calibrations",
            _display_path(calibrations_path),
            "--survey-output",
            _display_path(survey_output),
            "--runtime-output",
            _display_path(runtime_output),
            "--provenance-output",
            _display_path(provenance_output),
            "--generated-at",
            generated_at,
        ]
    )


def build_provenance_sidecar(
    survey: dict,
    *,
    calibrations_path: Path,
    survey_output: Path,
    runtime_output: Path,
    provenance_output: Path,
    survey_payload: bytes,
    runtime_payload: bytes,
    generated_at: str,
) -> dict:
    """Bind final source/artifact hashes without recursively hashing itself."""
    provenance = survey["provenance"]
    return {
        "schema_version": 1,
        "fitter_version": survey["fitter_version"],
        "generated_at": generated_at,
        "clip_id": provenance["clip_id"],
        "manifest_sha256": provenance["manifest_sha256"],
        "camera_order": list(survey["active_camera_order"]),
        "generation_command": _generation_command(
            calibrations_path=calibrations_path,
            survey_output=survey_output,
            runtime_output=runtime_output,
            provenance_output=provenance_output,
            generated_at=generated_at,
        ),
        "serialization": {
            "encoding": "UTF-8",
            "indent": 2,
            "survey_sort_keys": True,
            "runtime_sort_keys": False,
            "trailing_newline": True,
        },
        "sources": _source_hashes(calibrations_path),
        "artifacts": {
            _display_path(runtime_output): _sha256_bytes(runtime_payload),
            _display_path(survey_output): _sha256_bytes(survey_payload),
        },
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_many(payloads: dict[Path, bytes]) -> None:
    """Commit all three files or restore every destination byte-for-byte."""
    if len(payloads) != 3:
        raise ValueError("the FTheta artifact set must contain exactly three files")
    resolved = [path.expanduser().resolve() for path in payloads]
    if len(set(resolved)) != len(resolved):
        raise ValueError("artifact output paths must be distinct")

    destinations = [path.expanduser().resolve() for path in payloads]
    directories = sorted({path.parent for path in destinations})
    temporary_paths: dict[Path, Path] = {}
    backup_paths: dict[Path, Path | None] = {}
    try:
        for destination, payload in payloads.items():
            destination = destination.expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
            temporary = Path(temporary_name)
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_paths[destination] = temporary

        # Preserve every pre-transaction target before the first replace.
        for destination in destinations:
            if destination.exists():
                fd, backup_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.",
                    suffix=".backup",
                    dir=destination.parent,
                )
                os.close(fd)
                backup = Path(backup_name)
                shutil.copyfile(destination, backup)
                with backup.open("rb") as handle:
                    os.fsync(handle.fileno())
                backup_paths[destination] = backup
            else:
                backup_paths[destination] = None
        for directory in directories:
            _fsync_directory(directory)

        try:
            for destination in destinations:
                os.replace(temporary_paths[destination], destination)
            for directory in directories:
                _fsync_directory(directory)
        except Exception:
            rollback_errors: list[Exception] = []
            for destination in destinations:
                backup = backup_paths[destination]
                try:
                    if backup is None:
                        destination.unlink(missing_ok=True)
                    else:
                        os.replace(backup, destination)
                except Exception as rollback_error:  # pragma: no cover - fatal I/O
                    rollback_errors.append(rollback_error)
            for directory in directories:
                try:
                    _fsync_directory(directory)
                except Exception as rollback_error:  # pragma: no cover - fatal I/O
                    rollback_errors.append(rollback_error)
            if rollback_errors:
                raise RuntimeError(
                    f"artifact transaction rollback failed: {rollback_errors}"
                )
            raise
    finally:
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)
        for backup in backup_paths.values():
            if backup is not None:
                backup.unlink(missing_ok=True)
        for directory in directories:
            if directory.exists():
                _fsync_directory(directory)


def _validate_output_paths(calibrations_path: Path, *paths: Path) -> None:
    resolved = [path.expanduser().resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("artifact output paths must be distinct")
    forbidden = [path for path in resolved if path in _FORBIDDEN_LEGACY_OUTPUTS]
    if forbidden:
        raise ValueError(
            "refusing to overwrite immutable legacy FTheta artifact(s): "
            + ", ".join(str(path) for path in forbidden)
        )
    hashed_sources = {
        path.expanduser().resolve() for path in _hashed_source_paths(calibrations_path)
    }
    collisions = [path for path in resolved if path in hashed_sources]
    if collisions:
        raise ValueError(
            "refusing artifact output collision with hashed source(s): "
            + ", ".join(str(path) for path in collisions)
        )


def generate_artifact_set(
    calibration_bundle: dict,
    *,
    calibrations_path: Path,
    survey_output: Path,
    runtime_output: Path,
    provenance_output: Path,
    generated_at: str,
) -> dict[str, str]:
    """Generate and atomically write the deterministic three-file set."""
    _validate_output_paths(
        calibrations_path,
        survey_output,
        runtime_output,
        provenance_output,
    )
    survey = build_artifact(calibration_bundle)
    if not survey["active_subset_hard_invariants_passed"]:
        raise RuntimeError(
            f"active-subset hard invariant failures: {survey['active_hard_failures']}"
        )
    runtime = build_runtime_artifact(survey)
    survey_payload = _json_bytes(survey, sort_keys=True)
    runtime_payload = _json_bytes(runtime, sort_keys=False)
    sidecar = build_provenance_sidecar(
        survey,
        calibrations_path=calibrations_path,
        survey_output=survey_output,
        runtime_output=runtime_output,
        provenance_output=provenance_output,
        survey_payload=survey_payload,
        runtime_payload=runtime_payload,
        generated_at=generated_at,
    )
    provenance_payload = _json_bytes(sidecar, sort_keys=True)
    _atomic_write_many(
        {
            survey_output: survey_payload,
            runtime_output: runtime_payload,
            provenance_output: provenance_payload,
        }
    )
    return {
        "survey_sha256": _sha256_bytes(survey_payload),
        "runtime_sha256": _sha256_bytes(runtime_payload),
        "provenance_sha256": _sha256_bytes(provenance_payload),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibrations",
        type=Path,
        default=_REPO_ROOT / "scripts" / "pin_ftheta_b6a9_calibs.json",
    )
    parser.add_argument("--survey-output", type=Path, required=True)
    parser.add_argument("--runtime-output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument(
        "--generated-at",
        required=True,
        help="Frozen ISO-8601 timestamp recorded verbatim for deterministic output.",
    )
    args = parser.parse_args(argv)
    try:
        _validate_output_paths(
            args.calibrations,
            args.survey_output,
            args.runtime_output,
            args.provenance_output,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    with args.calibrations.open(encoding="utf-8") as handle:
        calibrations = json.load(handle)
    try:
        hashes = generate_artifact_set(
            calibrations,
            calibrations_path=args.calibrations,
            survey_output=args.survey_output,
            runtime_output=args.runtime_output,
            provenance_output=args.provenance_output,
            generated_at=args.generated_at,
        )
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    for name, digest in hashes.items():
        print(f"{name}={digest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
