# SPDX-License-Identifier: Apache-2.0
"""Fail-closed contracts for the dedicated FTheta v4 smoke/full drivers."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from PIL import Image

import scripts.pin_ftheta_v4_driver_validation as v4_validation
import scripts.pin_ftheta_smoke_validation as smoke_validation
import scripts.pin_ftheta_full_ab_validation as full_validation
from scripts.ncore_data_readiness import V4_MULTILAYER_PROFILE_CONTRACT
from scripts.pin_ftheta_full_ab_validation import validate_full_scientific_config


ROOT = Path(__file__).resolve().parents[2]
SMOKE_DRIVER = ROOT / "scripts/pin_ftheta_7cam_v4_smoke.sh"
FULL_DRIVER = ROOT / "scripts/pin_ftheta_7cam_v4_full_ab.sh"
V4_ARTIFACT = ROOT / V4_MULTILAYER_PROFILE_CONTRACT["runtime_artifact"]["path"]
V4_CONFIG = ROOT / V4_MULTILAYER_PROFILE_CONTRACT["config"]["path"]
_TEST_NESTED_SUBMODULE_PATHS = ("deps/nested-a", "deps/nested-b")


@pytest.mark.parametrize("driver", [SMOKE_DRIVER, FULL_DRIVER])
def test_v4_driver_is_valid_bash(driver: Path) -> None:
    result = subprocess.run(["bash", "-n", str(driver)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert driver.stat().st_mode & stat.S_IXUSR


@pytest.mark.parametrize(
    ("driver", "mode", "iterations", "duration", "output_name"),
    [
        (SMOKE_DRIVER, "smoke", "5000", "5.0", "pin_ftheta_v4_smoke_runs"),
        (FULL_DRIVER, "full", "30000", "20.0", "pin_ftheta_v4_full_ab_runs"),
    ],
)
def test_v4_driver_freezes_scientific_and_provenance_inputs(
    driver: Path,
    mode: str,
    iterations: str,
    duration: str,
    output_name: str,
) -> None:
    text = driver.read_text(encoding="utf-8")
    assert f"--mode {mode}" in text
    assert "apps/ncore_3dgut_mcmc_multilayer_inceptio_7cam_v4" in text
    assert "pin_ftheta_b6a9_7cam_params_v4_full_domain.json" in text
    assert "--ncore-readiness-profile \"$READINESS_PROFILE\"" in text
    assert "READINESS_PROFILE=\"v4-multilayer\"" in text
    assert f"n_iterations={iterations}" in text
    assert f"dataset.train.duration_sec={duration}" in text
    assert f"dataset.val.duration_sec={duration}" in text
    assert "seed_initialization=42" in text
    assert "num_workers=10" in text
    assert "dataset.camera_max_fov_deg=190.0" in text
    assert "trainer.use_lidar_depth=false" in text
    assert "trainer.use_depth_prior=false" in text
    assert "dataset.load_lidar_depth_map=false" in text
    assert "dataset.load_depth_prior=false" in text
    assert output_name in text
    assert "front_standard" not in text
    assert "front_tele" not in text
    assert "pin_ftheta_b6a9_7cam_params.json" not in text
    assert "PIN_FTHETA_EXPECTED_COMMIT" in text
    assert '--expected-commit "$EXPECTED_COMMIT"' in text
    if mode == "smoke":
        assert "native_render_inventory.json" in text
        assert "render-tree" in text
        assert "manifest-fail" in text
        assert "CURRENT_STAGE=" in text


@pytest.mark.parametrize("driver", [SMOKE_DRIVER, FULL_DRIVER])
def test_v4_preflight_precedes_cuda_output_and_launch(driver: Path) -> None:
    text = driver.read_text(encoding="utf-8")
    preflight = text.index('"$PYTHON_BIN" -m "$PREFLIGHT_VALIDATOR_MODULE" preflight')
    preflight_exit = text.index('if [ "$MODE" = "--preflight" ]')
    cuda = text.index("export CUDA_VISIBLE_DEVICES=0")
    output = text.index('mkdir -p "$RUN_BASE"')
    train = text.index('"$PYTHON_BIN" train.py')
    render = text.index('"$PYTHON_BIN" render.py')
    assert preflight < preflight_exit < cuda < output < train < render
    helper_text = (ROOT / "scripts/pin_ftheta_v4_driver_validation.py").read_text(encoding="utf-8")
    assert "\nimport torch" not in helper_text
    assert "train.py" not in helper_text
    assert "render.py" not in helper_text


@pytest.mark.parametrize(
    ("driver", "output_name"),
    [
        (SMOKE_DRIVER, "pin_ftheta_v4_smoke_runs"),
        (FULL_DRIVER, "pin_ftheta_v4_full_ab_runs"),
    ],
)
def test_shell_preflight_mode_never_creates_outputs_or_launches(
    tmp_path: Path,
    driver: Path,
    output_name: str,
) -> None:
    capture = tmp_path / "python_calls.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$CAPTURE\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    run_base = tmp_path / output_name
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "CAPTURE": str(capture),
            "DATA_PATH": str(tmp_path / "canonical.json"),
            "RUN_BASE": str(run_base),
            "PIN_FTHETA_EXPECTED_COMMIT": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        }
    )
    result = subprocess.run(
        ["bash", str(driver), "--preflight"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    calls = capture.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 1
    assert "scripts.pin_ftheta_v4_driver_validation preflight" in calls[0]
    assert "manifest-create" not in calls[0]
    assert "train.py" not in calls[0]
    assert "render.py" not in calls[0]
    assert not run_base.exists()


def test_v4_artifact_exact_order_keys_fingerprints_and_no_removed_cameras() -> None:
    fingerprints = v4_validation.validate_v4_artifact(V4_ARTIFACT)
    assert fingerprints == v4_validation.EXPECTED_PARAMETER_FINGERPRINTS
    assert tuple(fingerprints) == v4_validation.EXPECTED_CAMERA_IDS
    assert all("front_standard" not in camera_id for camera_id in fingerprints)
    assert all("front_tele" not in camera_id for camera_id in fingerprints)


def test_v4_artifact_rejects_legacy_sha_and_max_angle_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = ROOT / "scripts/pin_ftheta_b6a9_7cam_params.json"
    with pytest.raises(ValueError, match="legacy v3.*SHA-256"):
        v4_validation.validate_v4_artifact(legacy)

    payload = json.loads(V4_ARTIFACT.read_text(encoding="utf-8"))
    payload[v4_validation.EXPECTED_CAMERA_IDS[0]]["max_angle"] = v4_validation.LEGACY_V3_MAX_ANGLE
    sentinel = tmp_path / "sentinel.json"
    sentinel.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        v4_validation,
        "_sha256_file",
        lambda _path: V4_MULTILAYER_PROFILE_CONTRACT["runtime_artifact"]["sha256"],
    )
    with pytest.raises(ValueError, match="0.730310 rad"):
        v4_validation.validate_v4_artifact(sentinel)


def test_v4_artifact_rejects_camera_order_and_fingerprint_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = json.loads(V4_ARTIFACT.read_text(encoding="utf-8"))
    reordered = {key: payload[key] for key in reversed(payload)}
    artifact = tmp_path / "reordered.json"
    artifact.write_text(json.dumps(reordered), encoding="utf-8")
    monkeypatch.setattr(
        v4_validation,
        "_sha256_file",
        lambda _path: V4_MULTILAYER_PROFILE_CONTRACT["runtime_artifact"]["sha256"],
    )
    with pytest.raises(ValueError, match="camera order mismatch"):
        v4_validation.validate_v4_artifact(artifact)

    payload[v4_validation.EXPECTED_CAMERA_IDS[0]]["principal_point"][0] += 1.0
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprints"):
        v4_validation.validate_v4_artifact(artifact)


def test_v4_provenance_rejects_missing_and_stale_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sidecar = tmp_path / "missing.json"
    monkeypatch.setattr(
        v4_validation,
        "validate_v4_multilayer_profile_contract",
        lambda *_args: {"provenance_sidecar": sidecar},
    )
    with pytest.raises(ValueError, match="missing or unreadable"):
        v4_validation.validate_v4_provenance(ROOT, V4_ARTIFACT)

    sidecar.write_text(json.dumps({"generated_at": "2026-07-18T00:00:00+08:00"}), encoding="utf-8")
    with pytest.raises(ValueError, match="generated_at drift"):
        v4_validation.validate_v4_provenance(ROOT, V4_ARTIFACT)


def test_v4_provenance_propagates_stale_source_hash_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def stale(*_args):
        raise ValueError("v4-multilayer provenance source SHA-256 mismatch")

    monkeypatch.setattr(v4_validation, "validate_v4_multilayer_profile_contract", stale)
    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        v4_validation.validate_v4_provenance(ROOT, V4_ARTIFACT)


@pytest.mark.parametrize(
    ("mode", "driver", "output_name"),
    [
        ("smoke", SMOKE_DRIVER, "pin_ftheta_v4_smoke_runs"),
        ("full", FULL_DRIVER, "pin_ftheta_v4_full_ab_runs"),
    ],
)
def test_v4_output_and_checkpoint_namespaces_are_isolated(
    tmp_path: Path,
    mode: str,
    driver: Path,
    output_name: str,
) -> None:
    manifest = tmp_path / "data" / "canonical.json"
    paths = v4_validation.validate_mode_paths(
        mode,
        ROOT,
        driver,
        v4_validation.EXPECTED_CONFIG_NAME,
        V4_ARTIFACT,
        manifest,
        tmp_path / "output" / output_name,
    )
    assert paths["run_base"].name == output_name
    with pytest.raises(ValueError, match="output root name mismatch"):
        v4_validation.validate_mode_paths(
            mode,
            ROOT,
            driver,
            v4_validation.EXPECTED_CONFIG_NAME,
            V4_ARTIFACT,
            manifest,
            tmp_path / "output" / "pin_ftheta_7cam_full_ab_runs",
        )
    with pytest.raises(ValueError, match="artifact path mismatch"):
        v4_validation.validate_mode_paths(
            mode,
            ROOT,
            driver,
            v4_validation.EXPECTED_CONFIG_NAME,
            ROOT / "scripts/pin_ftheta_b6a9_7cam_params.json",
            manifest,
            tmp_path / "output" / output_name,
        )


def test_v4_smoke_and_full_resolved_pf_parity_and_exact_windows(tmp_path: Path) -> None:
    manifest = tmp_path / "canonical.json"
    smoke_hash = v4_validation.validate_resolved_pf_configs("smoke", ROOT, V4_ARTIFACT, manifest)
    full_hash = v4_validation.validate_resolved_pf_configs("full", ROOT, V4_ARTIFACT, manifest)
    assert len(smoke_hash) == 64
    assert len(full_hash) == 64
    assert smoke_hash != full_hash
    assert v4_validation.MODE_SPECS["smoke"]["iterations"] == 5000
    assert v4_validation.MODE_SPECS["smoke"]["train_duration_sec"] == 5.0
    assert v4_validation.MODE_SPECS["smoke"]["val_duration_sec"] == 5.0
    assert v4_validation.MODE_SPECS["full"]["iterations"] == 30000
    assert v4_validation.MODE_SPECS["full"]["train_duration_sec"] == 20.0
    assert v4_validation.MODE_SPECS["full"]["val_duration_sec"] == 20.0


def test_full_output_validator_requires_explicit_v4_20_second_window(tmp_path: Path) -> None:
    manifest = tmp_path / "canonical.json"
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        base = compose(config_name=v4_validation.EXPECTED_CONFIG_NAME)
    config = v4_validation._build_arm_config(
        base,
        mode="full",
        arm="F",
        artifact_path=V4_ARTIFACT.resolve(),
        input_manifest_path=manifest.resolve(),
    )
    validate_full_scientific_config(config, "F", V4_ARTIFACT, manifest)
    config.dataset.train.duration_sec = -1
    with pytest.raises(ValueError, match="dataset.train.duration_sec"):
        validate_full_scientific_config(config, "F", V4_ARTIFACT, manifest)


def _git_test_repo(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _make_initialized_release_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_test_repo(repo, "init", "-q", "-b", v4_validation.EXPECTED_BRANCH)
    _git_test_repo(repo, "config", "user.email", "test@example.invalid")
    _git_test_repo(repo, "config", "user.name", "Test")
    relative_paths = {
        v4_validation.MODE_SPECS["smoke"]["driver"],
        v4_validation.V4_DRIVER_VALIDATOR_RELATIVE_PATH,
        V4_MULTILAYER_PROFILE_CONTRACT["config"]["path"],
        V4_MULTILAYER_PROFILE_CONTRACT["runtime_artifact"]["path"],
        V4_MULTILAYER_PROFILE_CONTRACT["provenance_sidecar"]["path"],
        V4_MULTILAYER_PROFILE_CONTRACT["survey_artifact"]["path"],
    }
    for relative in relative_paths:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")

    for index, (submodule_relative, header_relative) in enumerate(
        zip(
            v4_validation._REQUIRED_SUBMODULE_PATHS,
            v4_validation._REQUIRED_SUBMODULE_HEADERS,
            strict=True,
        )
    ):
        source = tmp_path / f"submodule-source-{index}"
        source.mkdir()
        _git_test_repo(source, "init", "-q", "-b", "main")
        _git_test_repo(source, "config", "user.email", "test@example.invalid")
        _git_test_repo(source, "config", "user.name", "Test")
        header_inside_submodule = Path(header_relative).relative_to(submodule_relative)
        source_header = source / header_inside_submodule
        source_header.parent.mkdir(parents=True, exist_ok=True)
        source_header.write_text(header_relative, encoding="utf-8")
        _git_test_repo(source, "add", ".")
        _git_test_repo(source, "commit", "-qm", "submodule base")
        if index == 0:
            for nested_index, nested_relative in enumerate(_TEST_NESTED_SUBMODULE_PATHS):
                nested_source = tmp_path / f"nested-submodule-source-{nested_index}"
                nested_source.mkdir()
                _git_test_repo(nested_source, "init", "-q", "-b", "main")
                _git_test_repo(nested_source, "config", "user.email", "test@example.invalid")
                _git_test_repo(nested_source, "config", "user.name", "Test")
                (nested_source / "payload.txt").write_text(
                    f"nested payload {nested_index}", encoding="utf-8"
                )
                _git_test_repo(nested_source, "add", ".")
                _git_test_repo(nested_source, "commit", "-qm", "nested release")
                _git_test_repo(
                    source,
                    "-c",
                    "protocol.file.allow=always",
                    "submodule",
                    "add",
                    str(nested_source),
                    nested_relative,
                )
            _git_test_repo(source, "commit", "-qam", "add nested topology")
        _git_test_repo(
            repo,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(source),
            submodule_relative,
        )

    _git_test_repo(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "update",
        "--init",
        "--recursive",
    )
    _git_test_repo(repo, "add", ".")
    _git_test_repo(repo, "commit", "-qm", "release")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    return repo, commit


def test_release_gate_accepts_exact_initialized_clean_submodules(tmp_path: Path) -> None:
    repo, commit = _make_initialized_release_repo(tmp_path)
    recursive_status = subprocess.check_output(
        ["git", "submodule", "status", "--recursive"], cwd=repo, text=True
    )
    assert len(recursive_status.splitlines()) == len(v4_validation._REQUIRED_SUBMODULE_PATHS) + 2
    assert all(line.startswith(" ") for line in recursive_status.splitlines())
    release = v4_validation.validate_release_worktree(
        repo,
        repo / v4_validation.MODE_SPECS["smoke"]["driver"],
        commit,
    )
    assert release == {
        "branch": v4_validation.EXPECTED_BRANCH,
        "commit": commit,
        "submodule_status_policy": "initialized-clean-exact-gitlink-with-required-headers",
    }


def test_release_gate_requires_committed_clean_branch_and_exact_commit(tmp_path: Path) -> None:
    repo, commit = _make_initialized_release_repo(tmp_path)
    for untracked_relative in ("sitecustomize.py", "scripts/shadow_validator.py"):
        untracked = repo / untracked_relative
        untracked.parent.mkdir(parents=True, exist_ok=True)
        untracked.write_text("shadow", encoding="utf-8")
        with pytest.raises(ValueError, match="completely clean"):
            v4_validation.validate_release_worktree(
                repo,
                repo / v4_validation.MODE_SPECS["smoke"]["driver"],
                commit,
            )
        untracked.unlink()
    with pytest.raises(ValueError, match="commit mismatch"):
        v4_validation.validate_release_worktree(
            repo,
            repo / v4_validation.MODE_SPECS["smoke"]["driver"],
            "0" * 40,
        )
    (repo / V4_MULTILAYER_PROFILE_CONTRACT["config"]["path"]).write_text("dirty", encoding="utf-8")
    with pytest.raises(ValueError, match="completely clean"):
        v4_validation.validate_release_worktree(
            repo,
            repo / v4_validation.MODE_SPECS["smoke"]["driver"],
            commit,
        )


def test_release_gate_rejects_partial_restored_submodule_header_tree(tmp_path: Path) -> None:
    repo, commit = _make_initialized_release_repo(tmp_path)
    submodule_relative = v4_validation._REQUIRED_SUBMODULE_PATHS[0]
    _git_test_repo(repo, "submodule", "deinit", "-f", "--", submodule_relative)
    restored_header = repo / v4_validation._REQUIRED_SUBMODULE_HEADERS[0]
    restored_header.parent.mkdir(parents=True, exist_ok=True)
    restored_header.write_text("restored sentinel without initialized git metadata", encoding="utf-8")
    assert restored_header.is_file()
    with pytest.raises(ValueError, match="not initialized|uninitialized"):
        v4_validation.validate_release_worktree(
            repo,
            repo / v4_validation.MODE_SPECS["smoke"]["driver"],
            commit,
        )


def test_release_gate_rejects_submodule_wrong_commit(tmp_path: Path) -> None:
    repo, commit = _make_initialized_release_repo(tmp_path)
    submodule = repo / v4_validation._REQUIRED_SUBMODULE_PATHS[0]
    (submodule / "wrong-commit.txt").write_text("wrong", encoding="utf-8")
    _git_test_repo(submodule, "add", ".")
    _git_test_repo(submodule, "commit", "-qm", "wrong checked-out commit")
    with pytest.raises(ValueError, match="clean worktree|commit mismatch"):
        v4_validation.validate_release_worktree(
            repo,
            repo / v4_validation.MODE_SPECS["smoke"]["driver"],
            commit,
        )


def test_release_gate_rejects_dirty_submodule(tmp_path: Path) -> None:
    repo, commit = _make_initialized_release_repo(tmp_path)
    header = repo / v4_validation._REQUIRED_SUBMODULE_HEADERS[0]
    header.write_text("dirty", encoding="utf-8")
    with pytest.raises(ValueError, match="clean worktree|recursively clean"):
        v4_validation.validate_release_worktree(
            repo,
            repo / v4_validation.MODE_SPECS["smoke"]["driver"],
            commit,
        )


def test_release_gate_rejects_deinitialized_nested_submodule(tmp_path: Path) -> None:
    repo, commit = _make_initialized_release_repo(tmp_path)
    outer = repo / v4_validation._REQUIRED_SUBMODULE_PATHS[0]
    _git_test_repo(outer, "submodule", "deinit", "-f", "--", _TEST_NESTED_SUBMODULE_PATHS[0])
    with pytest.raises(ValueError, match="not initialized"):
        v4_validation.validate_release_worktree(
            repo,
            repo / v4_validation.MODE_SPECS["smoke"]["driver"],
            commit,
        )


def test_release_gate_rejects_nested_submodule_wrong_commit(tmp_path: Path) -> None:
    repo, commit = _make_initialized_release_repo(tmp_path)
    nested = (
        repo / v4_validation._REQUIRED_SUBMODULE_PATHS[0] / _TEST_NESTED_SUBMODULE_PATHS[0]
    )
    (nested / "wrong-commit.txt").write_text("wrong", encoding="utf-8")
    _git_test_repo(nested, "add", ".")
    _git_test_repo(nested, "commit", "-qm", "wrong nested checked-out commit")
    with pytest.raises(ValueError, match="clean worktree|commit mismatch"):
        v4_validation.validate_release_worktree(
            repo,
            repo / v4_validation.MODE_SPECS["smoke"]["driver"],
            commit,
        )


def test_release_gate_rejects_dirty_nested_submodule(tmp_path: Path) -> None:
    repo, commit = _make_initialized_release_repo(tmp_path)
    nested_payload = (
        repo
        / v4_validation._REQUIRED_SUBMODULE_PATHS[0]
        / _TEST_NESTED_SUBMODULE_PATHS[0]
        / "payload.txt"
    )
    nested_payload.write_text("dirty", encoding="utf-8")
    with pytest.raises(ValueError, match="clean worktree|recursively clean"):
        v4_validation.validate_release_worktree(
            repo,
            repo / v4_validation.MODE_SPECS["smoke"]["driver"],
            commit,
        )


def test_preflight_order_finishes_parity_after_dataset_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    manifest = tmp_path / "canonical.json"
    artifact = tmp_path / "artifact.json"
    run_base = tmp_path / "pin_ftheta_v4_smoke_runs"

    monkeypatch.setattr(
        v4_validation,
        "validate_release_worktree",
        lambda *_args: events.append("release") or {"branch": "b", "commit": "c"},
    )
    monkeypatch.setattr(
        v4_validation,
        "validate_mode_paths",
        lambda *_args: events.append("paths")
        or {"driver": SMOKE_DRIVER, "artifact": artifact, "manifest": manifest, "run_base": run_base},
    )
    monkeypatch.setattr(
        v4_validation,
        "validate_v4_provenance",
        lambda *_args: events.append("provenance") or v4_validation.EXPECTED_PARAMETER_FINGERPRINTS,
    )
    monkeypatch.setattr(
        v4_validation,
        "validate_v4_multilayer_dataset_contract",
        lambda *_args: events.append("dataset-contract") or {},
    )
    monkeypatch.setattr(
        v4_validation,
        "validate_resolved_pf_configs",
        lambda *_args: events.append("parity") or "f" * 64,
    )

    def readiness(*_args, **_kwargs):
        events.append("readiness")
        return {"component_store_count": 14}

    result = v4_validation.run_preflight(
        mode="smoke",
        repo_root=ROOT,
        driver_path=SMOKE_DRIVER,
        config_name=v4_validation.EXPECTED_CONFIG_NAME,
        artifact_path=artifact,
        input_manifest_path=manifest,
        run_base=run_base,
        expected_commit="a" * 40,
        readiness_validator=readiness,
    )
    assert events == ["release", "paths", "provenance", "dataset-contract", "readiness", "parity"]
    assert result["readiness"] == {"component_store_count": 14}


@pytest.mark.parametrize("driver", [SMOKE_DRIVER, FULL_DRIVER])
def test_shell_requires_external_frozen_commit_and_rejects_wrong_sha(
    tmp_path: Path,
    driver: Path,
) -> None:
    run_base_name = (
        "pin_ftheta_v4_smoke_runs" if driver == SMOKE_DRIVER else "pin_ftheta_v4_full_ab_runs"
    )
    base_env = os.environ.copy()
    base_env.update(
        {
            "PYTHON_BIN": sys.executable,
            "DATA_PATH": str(tmp_path / "missing-canonical.json"),
            "RUN_BASE": str(tmp_path / run_base_name),
        }
    )
    base_env.pop("PIN_FTHETA_EXPECTED_COMMIT", None)
    missing = subprocess.run(
        ["bash", str(driver), "--preflight"],
        cwd=ROOT,
        env=base_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode != 0
    assert "PIN_FTHETA_EXPECTED_COMMIT must be set" in missing.stdout

    wrong_env = dict(base_env)
    wrong_env["PIN_FTHETA_EXPECTED_COMMIT"] = "0" * 40
    wrong = subprocess.run(
        ["bash", str(driver), "--preflight"],
        cwd=ROOT,
        env=wrong_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert wrong.returncode != 0
    assert "v4 launch commit mismatch" in wrong.stderr

    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    correct_env = dict(base_env)
    correct_env["PIN_FTHETA_EXPECTED_COMMIT"] = current
    correct = subprocess.run(
        ["bash", str(driver), "--preflight"],
        cwd=ROOT,
        env=correct_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert correct.returncode != 0
    assert "v4 launch commit mismatch" not in correct.stderr
    assert "expected commit must" not in correct.stderr
    assert not (tmp_path / run_base_name).exists()


def _v4_training_log(arm: str) -> str:
    camera_ids = v4_validation.EXPECTED_CAMERA_IDS
    lines = []
    if arm == "F":
        lines.append("[PIN-FTHETA] NCoreDataset [train] explicit override enabled: cameras=7")
    lines.append("NCoreDataset [train] frame counts (after temporal filtering):")
    lines.extend(f"{camera_id}: 4 frames" for camera_id in camera_ids)
    lines.append("Total: 28 frames")
    model = "FThetaCameraModel" if arm == "F" else "OpenCVPinholeCameraModel"
    for split in ("train", "val", "test"):
        for camera_id in camera_ids:
            fingerprint = (
                v4_validation.EXPECTED_PARAMETER_FINGERPRINTS[camera_id] if arm == "F" else "none"
            )
            excluded = (
                smoke_validation._V4_EXCLUDED_BY_MAX_ANGLE[camera_id] if arm == "F" else 0
            )
            lines.append(
                f"[CAMERA-RAY-DOMAIN] split={split} camera={camera_id} "
                f"model_type={model} artifact_fingerprint={fingerprint} "
                f"total=2073600 excluded_by_max_angle={excluded} nonfinite=0"
            )
    lines.extend(
        [
            'Saved checkpoint to: "/run/ckpt_last.pt"',
            "Training Statistics",
            "Test Metrics",
        ]
    )
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("arm", ["P", "F"])
def test_v4_training_log_requires_exact_split_camera_telemetry(tmp_path: Path, arm: str) -> None:
    log = tmp_path / f"{arm}.log"
    log.write_text(_v4_training_log(arm), encoding="utf-8")
    counts = smoke_validation.validate_training_log(log, arm, V4_ARTIFACT)
    assert counts == {camera_id: 4 for camera_id in v4_validation.EXPECTED_CAMERA_IDS}


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("fingerprint", "fingerprint mismatch"),
        ("excluded", "telemetry oracle mismatch"),
        ("nonfinite", "telemetry oracle mismatch"),
        ("missing_split", "coverage mismatch"),
        ("missing_camera", "coverage mismatch"),
        ("drop", "prediction/render drop sentinel"),
    ],
)
def test_v4_training_log_rejects_telemetry_and_render_failures(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    text = _v4_training_log("F")
    camera_id = v4_validation.EXPECTED_CAMERA_IDS[0]
    fingerprint = v4_validation.EXPECTED_PARAMETER_FINGERPRINTS[camera_id]
    if mutation == "fingerprint":
        text = text.replace(f"artifact_fingerprint={fingerprint}", "artifact_fingerprint=deadbeef", 1)
    elif mutation == "excluded":
        text = text.replace("excluded_by_max_angle=148", "excluded_by_max_angle=149", 1)
    elif mutation == "nonfinite":
        text = text.replace("nonfinite=0", "nonfinite=1", 1)
    elif mutation == "missing_split":
        text = "\n".join(
            line
            for line in text.splitlines()
            if not ("split=test" in line and f"camera={camera_id}" in line)
        )
    elif mutation == "missing_camera":
        text = "\n".join(line for line in text.splitlines() if f"camera={camera_id}" not in line)
    else:
        text += "[A1] non-finite pred_rgb (2 px) at step 42 — dropping batch before loss/backward\n"
    log = tmp_path / "bad.log"
    log.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        smoke_validation.validate_training_log(log, "F", V4_ARTIFACT)


def test_full_validator_reuses_v4_log_and_metrics_quality_gates(tmp_path: Path) -> None:
    log = tmp_path / "full.log"
    log.write_text(_v4_training_log("F").replace("nonfinite=0", "nonfinite=1", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="telemetry oracle mismatch"):
        full_validation.validate_training_log(log, "F", V4_ARTIFACT)
    metrics = _complete_v4_metrics()
    metrics["nonfinite_pred_px"] = 3
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite_pred_px"):
        full_validation.validate_metrics(metrics_path, V4_ARTIFACT)


def _complete_v4_metrics() -> dict:
    top = {
        key: float(index + 1)
        for index, key in enumerate(smoke_validation.RENDER_METRIC_KEYS)
    }
    top["nonfinite_pred_px"] = 0
    top["per_camera"] = {
        camera_id: {
            **{
                key: float(index + 1)
                for index, key in enumerate(smoke_validation.RENDER_METRIC_KEYS)
            },
            "n_frames": 1,
        }
        for camera_id in v4_validation.EXPECTED_CAMERA_IDS
    }
    return top


def test_v4_metrics_require_effective_zero_nonfinite_and_exact_cameras(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics = _complete_v4_metrics()
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    smoke_validation.validate_metrics(metrics_path, V4_ARTIFACT)

    for bad in (1, True, "0"):
        mutated = json.loads(json.dumps(metrics))
        mutated["nonfinite_pred_px"] = bad
        metrics_path.write_text(json.dumps(mutated), encoding="utf-8")
        with pytest.raises(ValueError, match="nonfinite_pred_px"):
            smoke_validation.validate_metrics(metrics_path, V4_ARTIFACT)
    mutated = json.loads(json.dumps(metrics))
    mutated.pop("nonfinite_pred_px")
    metrics_path.write_text(json.dumps(mutated), encoding="utf-8")
    smoke_validation.validate_metrics(metrics_path, V4_ARTIFACT)
    mutated = json.loads(json.dumps(metrics))
    mutated["per_camera"].pop(v4_validation.EXPECTED_CAMERA_IDS[0])
    metrics_path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="per_camera mismatch"):
        smoke_validation.validate_metrics(metrics_path, V4_ARTIFACT)


def test_v4_smoke_native_render_inventory_hashes_and_revalidates_tree(tmp_path: Path) -> None:
    metrics_path = tmp_path / "eval" / "metrics.json"
    metrics_path.parent.mkdir()
    metrics_path.write_text(json.dumps(_complete_v4_metrics()), encoding="utf-8")
    render_dir = metrics_path.parent / "ours_5000" / "renders"
    gt_dir = metrics_path.parent / "ours_5000" / "gt"
    render_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    for index in range(7):
        name = f"{index:06d}.png"
        Image.new("RGB", (1920, 1080), (index, 0, 0)).save(render_dir / name)
        Image.new("RGB", (1920, 1080), (0, index, 0)).save(gt_dir / name)
    inventory_path = tmp_path / "inventory.json"
    inventory = smoke_validation.validate_smoke_native_render_tree(
        metrics_path, V4_ARTIFACT, inventory_path
    )
    assert inventory["render_png_count"] == 7
    assert inventory["gt_png_count"] == 7
    assert json.loads(inventory_path.read_text(encoding="utf-8")) == inventory
    Image.new("RGB", (1920, 1080), (255, 0, 0)).save(render_dir / "000000.png")
    drifted = smoke_validation.validate_smoke_native_render_tree(metrics_path, V4_ARTIFACT)
    assert drifted["render_tree_sha256"] != inventory["render_tree_sha256"]


def test_v4_smoke_manifest_requires_and_hash_checks_native_inventory(tmp_path: Path) -> None:
    files = {}
    for name in (*smoke_validation._REQUIRED_OUTPUT_NAMES, "native_render_inventory"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        files[name] = smoke_validation._file_record(path)
    value = {
        "ncore_readiness_profile": smoke_validation.V4_MULTILAYER_READINESS_PROFILE,
        "arms": {"P": dict(files), "F": dict(files)},
    }
    smoke_validation._verify_arm_output_records(value, require_both=True)
    del value["arms"]["P"]["native_render_inventory"]
    with pytest.raises(ValueError, match="native_render_inventory"):
        smoke_validation._verify_arm_output_records(value, require_both=True)


def test_completed_v4_smoke_manifest_revalidates_native_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = {
        "gt_png_count": 7,
        "gt_png_names": [f"{index:06d}.png" for index in range(7)],
        "gt_tree_sha256": "a" * 64,
        "render_tree_sha256": "b" * 64,
        "native_resolution": [1920, 1080],
        "per_camera_n_frames": {
            camera_id: 1 for camera_id in v4_validation.EXPECTED_CAMERA_IDS
        },
    }
    arms = {}
    for arm in ("P", "F"):
        inventory_path = tmp_path / f"{arm}_inventory.json"
        inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
        arms[arm] = {
            "metrics": {"path": str(tmp_path / f"{arm}_metrics.json")},
            "native_render_inventory": {"path": str(inventory_path)},
            "native_render": inventory,
        }
    source_names = set(smoke_validation._REQUIRED_SOURCE_NAMES) | {
        "data_readiness_validator",
        "v4_driver_validator",
        "v4_provenance_sidecar",
        "v4_survey_artifact",
    }
    value = {
        "schema_version": 2,
        "git_commit": "abc123",
        "status": "complete",
        "ncore_readiness_profile": smoke_validation.V4_MULTILAYER_READINESS_PROFILE,
        "sources": {
            name: {"path": str(V4_ARTIFACT) if name == "artifact" else str(tmp_path / name)}
            for name in source_names
        },
        "arms": arms,
    }
    calls = []
    monkeypatch.setattr(smoke_validation, "_load_run_manifest", lambda _path: value)
    monkeypatch.setattr(smoke_validation, "_verify_file_record", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(smoke_validation, "_verify_arm_output_records", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        smoke_validation,
        "validate_smoke_native_render_tree",
        lambda metrics, _artifact: calls.append(Path(metrics).name) or inventory,
    )
    smoke_validation.verify_run_manifest(tmp_path / "manifest.json", "abc123")
    assert calls == ["P_metrics.json", "F_metrics.json"]
    Path(arms["F"]["native_render_inventory"]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Arm F.*drifted"):
        smoke_validation.verify_run_manifest(tmp_path / "manifest.json", "abc123")


def test_smoke_manifest_fail_records_stage_and_exit_code(tmp_path: Path) -> None:
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps({"status": "started"}), encoding="utf-8")
    value = smoke_validation.mark_run_failed(manifest, "armF-native-render", 17)
    assert value["status"] == "failed"
    assert value["failure"] == {"stage": "armF-native-render", "exit_code": 17}
