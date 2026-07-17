# SPDX-License-Identifier: Apache-2.0
"""Static contract tests for the matched seven-camera FTheta smoke driver."""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.pin_ftheta_smoke_validation import (
    RENDER_METRIC_KEYS,
    _build_parser,
    compare_parsed_configs,
    compare_per_camera_frame_counts,
    create_run_manifest,
    ensure_tracked_worktree_clean,
    finalize_run_manifest,
    record_arm_outputs,
    validate_checkpoint,
    validate_metrics,
    validate_parsed_config,
    validate_training_log,
    verify_run_manifest,
)
from threedgrut.datasets.ftheta_override import load_ftheta_override_parameters

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "scripts" / "pin_ftheta_9cam_smoke.sh"
CONFIG_DIR = str(ROOT / "configs")
BASE_CONFIG = "apps/ncore_3dgut_mcmc_multilayer_inceptio_7cam"
FTHETA_ARTIFACT = "scripts/pin_ftheta_b6a9_7cam_params.json"
FTHETA_ARTIFACT_PATH = ROOT / FTHETA_ARTIFACT
SEVEN_CAMERAS = [
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_left_wide_90fov",
    "camera_right_wide_90fov",
    "camera_back_rear_wide_90fov",
    "camera_rear_left_70fov",
]


def _compose(overrides: list[str] | None = None):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name=BASE_CONFIG, overrides=overrides or [])


def test_smoke_driver_is_valid_bash() -> None:
    result = subprocess.run(
        ["bash", "-n", str(DRIVER)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _validator_subprocess_env(tmp_path: Path) -> dict[str, str]:
    """Expose the real validator module without importing NCore on the Mac."""

    datasets_path = ROOT / "threedgrut" / "datasets"
    (tmp_path / "sitecustomize.py").write_text(
        "import sys, types\n"
        "package = types.ModuleType('threedgrut.datasets')\n"
        f"package.__path__ = [{str(datasets_path)!r}]\n"
        "package.__package__ = 'threedgrut.datasets'\n"
        "sys.modules['threedgrut.datasets'] = package\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(ROOT), env.get("PYTHONPATH", "")])
    return env


def test_validator_module_cli_imports_from_repo_root(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.pin_ftheta_smoke_validation", "--help"],
        cwd=ROOT,
        env=_validator_subprocess_env(tmp_path),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "manifest-create" in result.stdout


def test_smoke_driver_freezes_matched_common_overrides() -> None:
    source = DRIVER.read_text(encoding="utf-8")

    required_common_overrides = [
        "n_iterations=5000",
        "seed_initialization=42",
        "test_last=true",
        "dataset.train.seek_offset_sec=0.0",
        "dataset.train.duration_sec=5.0",
        "dataset.val.seek_offset_sec=0.0",
        "dataset.val.duration_sec=5.0",
        "dataset.downsample=1.0",
        "dataset.mask_forward_invalid_pixels=true",
        "dataset.opencv_pinhole_use_validity_domain=false",
        "trainer.sky_backend=mlp",
        "trainer.use_lidar_depth=false",
        "trainer.use_depth_prior=false",
        "dataset.load_lidar_depth_map=false",
        "dataset.load_depth_prior=false",
        "num_workers=10",
    ]
    for override in required_common_overrides:
        assert source.count(override) == 1, override

    assert "--config-name apps/ncore_3dgut_mcmc_multilayer_inceptio_7cam" in source
    assert 'run_arm "P" "null"' in source
    assert f'run_arm "F" "{FTHETA_ARTIFACT}"' in source
    assert 'dataset.ftheta_params_path="$ftheta_params_path"' in source
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in source
    assert "CUDA_VISIBLE_DEVICES=0" in source
    assert 'VALIDATOR_MODULE="scripts.pin_ftheta_smoke_validation"' in source
    assert 'python "$VALIDATOR_FILE"' not in source
    for command in (
        "log",
        "checkpoint",
        "metrics",
        "manifest-create",
        "manifest-verify",
        "record-arm",
        "finalize",
    ):
        assert f'-m "$VALIDATOR_MODULE" {command}' in source
    assert "date +%s%N" in source
    assert 'mkdir "$RUN_ROOT"' in source
    assert "tee -a" not in source
    assert 'train_log="$ARM_ROOT/train.log"' in source
    assert 'eval_log="$ARM_ROOT/eval.log"' in source


def test_finalize_cli_rejects_caller_supplied_evidence_paths() -> None:
    parser = _build_parser()
    args = parser.parse_args(["finalize", "--manifest", "/run/manifest.json", "--repo-root", "/repo"])
    assert vars(args) == {
        "command": "finalize",
        "manifest": "/run/manifest.json",
        "repo_root": "/repo",
    }

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "finalize",
                "--manifest",
                "/run/manifest.json",
                "--repo-root",
                "/repo",
                "--p-metrics",
                "/attacker/substitute.json",
            ]
        )


def test_smoke_driver_does_not_reintroduce_excluded_cameras_or_weights() -> None:
    source = DRIVER.read_text(encoding="utf-8")

    assert "camera_front_standard_55fov" not in source
    assert "camera_front_tele_30fov" not in source
    assert "camera_loss_weights" not in source
    assert "TELEW" not in source


def test_smoke_arm_configs_differ_only_by_ftheta_override() -> None:
    pinhole = _compose(["dataset.ftheta_params_path=null"])
    ftheta = _compose([f"dataset.ftheta_params_path={FTHETA_ARTIFACT}"])

    assert list(pinhole.dataset.camera_ids) == SEVEN_CAMERAS
    assert list(ftheta.dataset.camera_ids) == SEVEN_CAMERAS
    assert pinhole.dataset.ftheta_params_path is None
    assert ftheta.dataset.ftheta_params_path == FTHETA_ARTIFACT
    assert pinhole.dataset.mask_forward_invalid_pixels is True
    assert ftheta.dataset.mask_forward_invalid_pixels is True
    assert pinhole.dataset.opencv_pinhole_use_validity_domain is False
    assert ftheta.dataset.opencv_pinhole_use_validity_domain is False
    assert pinhole.loss.camera_loss_weights == {}
    assert ftheta.loss.camera_loss_weights == {}

    pinhole.dataset.ftheta_params_path = FTHETA_ARTIFACT
    assert OmegaConf.to_container(pinhole, resolve=False) == OmegaConf.to_container(ftheta, resolve=False)


def _frame_count_block(*, zero_camera: str | None = None, omit_camera: str | None = None) -> str:
    lines = ["NCoreDataset [train] frame counts (after temporal filtering):"]
    total = 0
    for camera_id in SEVEN_CAMERAS:
        if camera_id == omit_camera:
            continue
        count = 0 if camera_id == zero_camera else 4
        total += count
        lines.append(f"  {camera_id}: {count} frames")
    lines.append(f"  Total: {total} frames")
    return "\n".join(lines)


def _successful_log(*, arm: str = "P") -> str:
    override = "[PIN-FTHETA] NCoreDataset [train] explicit override enabled: cameras=7\n" if arm == "F" else ""
    return (
        override
        + _frame_count_block()
        + '\n💾 Saved checkpoint to: "/run/ckpt_last.pt"\n'
        + "Training Statistics\nTest Metrics\n"
    )


def _real_rich_probe_log() -> str:
    counts = {
        "camera_front_wide_120fov": 38,
        "camera_cross_left_120fov": 42,
        "camera_cross_right_120fov": 37,
        "camera_left_wide_90fov": 39,
        "camera_right_wide_90fov": 43,
        "camera_back_rear_wide_90fov": 42,
        "camera_rear_left_70fov": 43,
    }
    lines = [
        "[13:17:45] [WARNING] [A1] camera_left_wide_90fov: repaired 6        logger.py:71",
        "           non-finite camera ray(s) (+6 in val subsample) and",
        "           masked the pixel(s) invalid — rational-distortion pole,",
        "           see repair_nonfinite_rays",
        "           [INFO] NCoreDataset  frame counts (after temporal        logger.py:68",
        "           filtering):",
    ]
    lines.extend(
        f"           [INFO]   {camera_id}: {count} frames             logger.py:68"
        for camera_id, count in counts.items()
    )
    lines.extend(
        [
            "           [INFO]   Total: 284 frames                    logger.py:68",
            '💾 Saved checkpoint to: "/run/ckpt_last.pt"',
            "Training Statistics",
            "Test Metrics",
        ]
    )
    return "\n".join(lines) + "\n"


def test_log_validation_allows_known_containment_messages(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        _successful_log()
        + "\n".join(
            [
                "[A1] camera_left_wide_90fov: repaired 1 non-finite camera ray(s) "
                "(+0 in val subsample) and masked the pixel(s) invalid — rational-distortion pole",
                "[A1] mcmc relocate: sanitized 4 non-finite/non-positive rows — fell back to donor copy",
                "[A1] non-finite pred_rgb (2 px) at step 42 — dropping batch before loss/backward",
            ]
        ),
        encoding="utf-8",
    )

    assert validate_training_log(log, "P", FTHETA_ARTIFACT_PATH) == {camera_id: 4 for camera_id in SEVEN_CAMERAS}

    # Rich may wrap the long production header when stdout is redirected.
    wrapped = log.read_text(encoding="utf-8").replace(
        "frame counts (after temporal filtering):",
        "frame counts (after temporal\nfiltering):",
    )
    log.write_text(wrapped, encoding="utf-8")
    validate_training_log(log, "P", FTHETA_ARTIFACT_PATH)


def test_log_validation_allows_real_rich_wrapped_repair_warning(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(_real_rich_probe_log(), encoding="utf-8")

    assert validate_training_log(log, "P", FTHETA_ARTIFACT_PATH) == {
        "camera_front_wide_120fov": 38,
        "camera_cross_left_120fov": 42,
        "camera_cross_right_120fov": 37,
        "camera_left_wide_90fov": 39,
        "camera_right_wide_90fov": 43,
        "camera_back_rear_wide_90fov": 42,
        "camera_rear_left_70fov": 43,
    }


def test_real_rich_probe_log_passes_validator_module_cli(tmp_path: Path) -> None:
    log = tmp_path / "armP.log"
    log.write_text(_real_rich_probe_log(), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.pin_ftheta_smoke_validation",
            "log",
            "--path",
            str(log),
            "--arm",
            "P",
            "--artifact",
            str(FTHETA_ARTIFACT_PATH),
        ],
        cwd=ROOT,
        env=_validator_subprocess_env(tmp_path),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_log_validation_requires_ftheta_override_only_for_arm_f(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(_successful_log(arm="F"), encoding="utf-8")

    validate_training_log(log, "F", FTHETA_ARTIFACT_PATH)
    with pytest.raises(ValueError, match="Arm P unexpectedly"):
        validate_training_log(log, "P", FTHETA_ARTIFACT_PATH)

    log.write_text(_successful_log(), encoding="utf-8")
    with pytest.raises(ValueError, match="Arm F did not log"):
        validate_training_log(log, "F", FTHETA_ARTIFACT_PATH)


@pytest.mark.parametrize(
    "bad_line",
    [
        "Non-finite total_loss at step 12: camera_id=cam",
        "Traceback (most recent call last):",
        "camera_x has non-finite camera ray(s) after initialization",
        "[A1] camera_x: repaired six\nnon-finite camera ray(s) "
        "(+0 in val subsample) and masked the pixel(s) invalid",
        "[A1] camera_x: repaired 6\nnon-finite camera ray(s) "
        "(+not-a-number in val subsample) and masked the pixel(s) invalid",
        "[A1] camera_x: repaired 6\nnon-finite camera ray(s) " "(+0 in val subsample) but left the pixel(s) valid",
    ],
)
def test_log_validation_rejects_only_fatal_sentinels_and_unrepaired_rays(tmp_path: Path, bad_line: str) -> None:
    log = tmp_path / "train.log"
    log.write_text(f"{_successful_log()}\n{bad_line}\n", encoding="utf-8")

    with pytest.raises(ValueError):
        validate_training_log(log, "P", FTHETA_ARTIFACT_PATH)


@pytest.mark.parametrize(
    "log_text",
    [
        "Training Statistics\nTest Metrics\n",
        _successful_log().replace('💾 Saved checkpoint to: "/run/ckpt_last.pt"\n', ""),
        _successful_log().replace(_frame_count_block(), _frame_count_block(zero_camera=SEVEN_CAMERAS[0])),
        _successful_log().replace(_frame_count_block(), _frame_count_block(omit_camera=SEVEN_CAMERAS[0])),
    ],
)
def test_log_validation_rejects_header_only_incomplete_or_zero_frame_runs(tmp_path: Path, log_text: str) -> None:
    log = tmp_path / "train.log"
    log.write_text(log_text, encoding="utf-8")

    with pytest.raises(ValueError):
        validate_training_log(log, "P", FTHETA_ARTIFACT_PATH)


def _camera_contracts(arm: str) -> dict:
    params, fingerprints = load_ftheta_override_parameters(FTHETA_ARTIFACT_PATH, SEVEN_CAMERAS)
    if arm == "P":
        return {
            camera_id: {
                "model_type": "OpenCVPinhole",
                "native_resolution": (1920, 1080),
            }
            for camera_id in SEVEN_CAMERAS
        }
    return {
        camera_id: {
            "model_type": "FTheta",
            "native_resolution": (1920, 1080),
            "intrinsics_FTheta": params[camera_id],
            "parameter_fingerprint": fingerprints[camera_id],
        }
        for camera_id in SEVEN_CAMERAS
    }


@pytest.mark.parametrize("arm", ["P", "F"])
def _resolved_config(tmp_path: Path, arm: str):
    data_path = tmp_path / "dataset.json"
    data_path.write_text("{}", encoding="utf-8")
    ftheta_path = "null" if arm == "P" else FTHETA_ARTIFACT
    return _compose(
        [
            "n_iterations=5000",
            "seed_initialization=42",
            "test_last=true",
            f"path={data_path}",
            f"out_dir={tmp_path / 'out'}",
            f"experiment_name=arm{arm}",
            "dataset.train.seek_offset_sec=0.0",
            "dataset.train.duration_sec=5.0",
            "dataset.val.seek_offset_sec=0.0",
            "dataset.val.duration_sec=5.0",
            "dataset.downsample=1.0",
            "dataset.mask_forward_invalid_pixels=true",
            "dataset.opencv_pinhole_use_validity_domain=false",
            "trainer.sky_backend=mlp",
            "trainer.use_lidar_depth=false",
            "trainer.use_depth_prior=false",
            "dataset.load_lidar_depth_map=false",
            "dataset.load_depth_prior=false",
            "num_workers=10",
            f"dataset.ftheta_params_path={ftheta_path}",
        ]
    )


def _checkpoint_payload(tmp_path: Path, arm: str) -> dict:
    return {
        "global_step": 5000,
        "config": _resolved_config(tmp_path, arm),
        "viz_4d": {"camera_models": _camera_contracts(arm)},
    }


@pytest.mark.parametrize("arm", ["P", "F"])
def test_checkpoint_validation_accepts_exact_seven_camera_contract(tmp_path: Path, arm: str) -> None:
    checkpoint = tmp_path / f"{arm}.pt"
    payload = _checkpoint_payload(tmp_path, arm)
    torch.save(payload, checkpoint)

    validate_checkpoint(checkpoint, arm, FTHETA_ARTIFACT_PATH, payload["config"].path)


@pytest.mark.parametrize(
    ("arm", "mutation"),
    [
        ("P", "missing_camera"),
        ("P", "wrong_resolution"),
        ("P", "hidden_ftheta"),
        ("F", "missing_key"),
        ("F", "bad_parameter"),
        ("F", "bad_fingerprint"),
    ],
)
def test_checkpoint_validation_rejects_incomplete_or_cross_arm_contracts(
    tmp_path: Path, arm: str, mutation: str
) -> None:
    contracts = _camera_contracts(arm)
    camera_id = SEVEN_CAMERAS[0]
    if mutation == "missing_camera":
        del contracts[camera_id]
    elif mutation == "wrong_resolution":
        contracts[camera_id]["native_resolution"] = (1280, 720)
    elif mutation == "hidden_ftheta":
        contracts[camera_id]["intrinsics_FTheta"] = {"resolution": [1920, 1080]}
    elif mutation == "missing_key":
        del contracts[camera_id]["intrinsics_FTheta"]["max_angle"]
    elif mutation == "bad_parameter":
        contracts[camera_id]["intrinsics_FTheta"]["max_angle"] += 0.1
    elif mutation == "bad_fingerprint":
        contracts[camera_id]["parameter_fingerprint"] = "0" * 64
    payload = _checkpoint_payload(tmp_path, arm)
    payload["viz_4d"]["camera_models"] = contracts
    checkpoint = tmp_path / f"{arm}_{mutation}.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError):
        validate_checkpoint(checkpoint, arm, FTHETA_ARTIFACT_PATH, payload["config"].path)


@pytest.mark.parametrize("mutation", ["step", "seed", "camera", "depth", "loss", "layers"])
def test_checkpoint_validation_rejects_incomplete_scientific_completion(tmp_path: Path, mutation: str) -> None:
    payload = _checkpoint_payload(tmp_path, "P")
    if mutation == "step":
        payload["global_step"] = 4999
    elif mutation == "seed":
        payload["config"].seed_initialization = 7
    elif mutation == "camera":
        payload["config"].dataset.camera_ids = SEVEN_CAMERAS[:-1]
    elif mutation == "depth":
        payload["config"].trainer.use_lidar_depth = True
    elif mutation == "loss":
        payload["config"].loss.camera_loss_weights = {SEVEN_CAMERAS[0]: 2.0}
    elif mutation == "layers":
        payload["config"].layers.enabled = ["background"]
    checkpoint = tmp_path / f"bad_{mutation}.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError):
        validate_checkpoint(checkpoint, "P", FTHETA_ARTIFACT_PATH, payload["config"].path)


def _complete_metrics() -> dict:
    metrics = {key: float(index + 1) for index, key in enumerate(RENDER_METRIC_KEYS)}
    metrics["per_camera"] = {
        camera_id: {
            "n_frames": 2,
            **{key: float(index + 1) for index, key in enumerate(RENDER_METRIC_KEYS)},
        }
        for camera_id in SEVEN_CAMERAS
    }
    return metrics


def test_metrics_validation_accepts_all_twelve_finite_metrics(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(_complete_metrics()), encoding="utf-8")

    validate_metrics(path, FTHETA_ARTIFACT_PATH)


@pytest.mark.parametrize(
    ("scope", "mutation"),
    [
        ("top", "missing"),
        ("top", "nan"),
        ("camera", "missing"),
        ("camera", "nan"),
        ("camera", "zero_frames"),
        ("per_camera", "missing_camera"),
    ],
)
def test_metrics_validation_rejects_missing_nan_and_empty_camera_results(
    tmp_path: Path, scope: str, mutation: str
) -> None:
    metrics = copy.deepcopy(_complete_metrics())
    target = metrics if scope == "top" else metrics["per_camera"][SEVEN_CAMERAS[0]]
    if mutation == "missing_camera":
        del metrics["per_camera"][SEVEN_CAMERAS[0]]
    elif mutation == "missing":
        del target[RENDER_METRIC_KEYS[-1]]
    elif mutation == "nan":
        target[RENDER_METRIC_KEYS[-1]] = math.nan
    elif mutation == "zero_frames":
        target["n_frames"] = 0
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_metrics(path, FTHETA_ARTIFACT_PATH)


def _write_parsed_config(tmp_path: Path, arm: str) -> Path:
    path = tmp_path / f"parsed_{arm}.yaml"
    OmegaConf.save(_resolved_config(tmp_path, arm), path)
    return path


def test_parsed_configs_are_scientifically_identical_except_ftheta_path(
    tmp_path: Path,
) -> None:
    p_path = _write_parsed_config(tmp_path, "P")
    f_path = _write_parsed_config(tmp_path, "F")
    data_path = tmp_path / "dataset.json"

    validate_parsed_config(p_path, "P", FTHETA_ARTIFACT_PATH, data_path)
    validate_parsed_config(f_path, "F", FTHETA_ARTIFACT_PATH, data_path)
    digest = compare_parsed_configs(p_path, f_path)
    assert len(digest) == 64

    f_config = OmegaConf.load(f_path)
    f_config.loss.lambda_l1 = 0.5
    OmegaConf.save(f_config, f_path)
    with pytest.raises(ValueError, match="scientific config mismatch"):
        compare_parsed_configs(p_path, f_path)


def test_per_camera_frame_counts_must_match_between_arms(tmp_path: Path) -> None:
    p_path = tmp_path / "p_metrics.json"
    f_path = tmp_path / "f_metrics.json"
    p_metrics = _complete_metrics()
    f_metrics = copy.deepcopy(p_metrics)
    p_path.write_text(json.dumps(p_metrics), encoding="utf-8")
    f_path.write_text(json.dumps(f_metrics), encoding="utf-8")

    assert compare_per_camera_frame_counts(p_path, f_path, FTHETA_ARTIFACT_PATH) == {
        camera_id: 2 for camera_id in SEVEN_CAMERAS
    }
    f_metrics["per_camera"][SEVEN_CAMERAS[0]]["n_frames"] = 3
    f_path.write_text(json.dumps(f_metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="n_frames mismatch"):
        compare_per_camera_frame_counts(p_path, f_path, FTHETA_ARTIFACT_PATH)


def _source_files(tmp_path: Path) -> dict[str, Path]:
    sources = {}
    for name in ("dataset_manifest", "config", "artifact", "driver", "validator"):
        path = tmp_path / f"{name}.txt"
        path.write_text(name, encoding="utf-8")
        sources[name] = path
    return sources


def test_run_manifest_is_exclusive_and_detects_source_or_commit_drift(tmp_path: Path) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    sources = _source_files(tmp_path)
    create_run_manifest(manifest_path, "unique_run", "abc123", sources)

    manifest = verify_run_manifest(manifest_path, "abc123")
    assert manifest["run_id"] == "unique_run"
    assert set(manifest["sources"]) == set(sources)
    assert all(len(entry["sha256"]) == 64 for entry in manifest["sources"].values())
    with pytest.raises(FileExistsError):
        create_run_manifest(manifest_path, "collision", "abc123", sources)
    with pytest.raises(ValueError, match="git commit drift"):
        verify_run_manifest(manifest_path, "different")

    sources["artifact"].write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash drift"):
        verify_run_manifest(manifest_path, "abc123")


def test_launch_cleanliness_rejects_tracked_changes_but_allows_untracked(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    tracked = repo / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=repo, check=True)

    (repo / "user-untracked.txt").write_text("preserve me\n", encoding="utf-8")
    ensure_tracked_worktree_clean(repo)

    tracked.write_text("unstaged\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tracked changes"):
        ensure_tracked_worktree_clean(repo)

    subprocess.run(["git", "restore", "tracked.txt"], cwd=repo, check=True)
    tracked.write_text("staged\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    with pytest.raises(ValueError, match="tracked changes"):
        ensure_tracked_worktree_clean(repo)


def test_record_and_finalize_persist_parsed_hashes_and_matched_evidence(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    sources = _source_files(tmp_path)
    data_path = tmp_path / "dataset.json"
    data_path.write_text("{}", encoding="utf-8")
    sources["dataset_manifest"] = data_path
    sources["artifact"] = FTHETA_ARTIFACT_PATH
    create_run_manifest(manifest_path, "unique_run", "abc123", sources)
    for arm in ("P", "F"):
        parsed = _write_parsed_config(tmp_path, arm)
        checkpoint = tmp_path / f"{arm}.pt"
        torch.save(_checkpoint_payload(tmp_path, arm), checkpoint)
        metrics = tmp_path / f"{arm}_metrics.json"
        metrics.write_text(json.dumps(_complete_metrics()), encoding="utf-8")
        train_log = tmp_path / f"{arm}_train.log"
        train_log.write_text(_successful_log(arm=arm), encoding="utf-8")
        eval_log = tmp_path / f"{arm}_eval.log"
        eval_log.write_text("native eval complete", encoding="utf-8")
        record_arm_outputs(
            manifest_path,
            arm,
            parsed,
            checkpoint,
            metrics,
            train_log,
            eval_log,
            FTHETA_ARTIFACT_PATH,
            data_path,
        )
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    for arm in ("P", "F"):
        for output_name in ("parsed_yaml", "checkpoint", "metrics", "train_log", "eval_log"):
            record = recorded["arms"][arm][output_name]
            assert Path(record["path"]).is_absolute()
            assert len(record["sha256"]) == 64

    original_p_eval = Path(recorded["arms"]["P"]["eval_log"]["path"]).read_text(encoding="utf-8")
    Path(recorded["arms"]["P"]["eval_log"]["path"]).write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="output hash drift"):
        finalize_run_manifest(manifest_path, "abc123")
    Path(recorded["arms"]["P"]["eval_log"]["path"]).write_text(original_p_eval, encoding="utf-8")

    finalized = finalize_run_manifest(
        manifest_path,
        "abc123",
    )
    assert finalized["status"] == "complete"
    assert len(finalized["arms"]["P"]["parsed_yaml"]["sha256"]) == 64
    assert len(finalized["arms"]["F"]["parsed_yaml"]["sha256"]) == 64
    assert finalized["comparison"]["train_frames_per_camera"] == {camera_id: 4 for camera_id in SEVEN_CAMERAS}
    assert finalized["comparison"]["per_camera_n_frames"] == {camera_id: 2 for camera_id in SEVEN_CAMERAS}

    Path(finalized["arms"]["F"]["metrics"]["path"]).write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="output hash drift"):
        verify_run_manifest(manifest_path, "abc123")
    Path(finalized["arms"]["F"]["metrics"]["path"]).write_text(json.dumps(_complete_metrics()), encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["arms"]["F"]["train_frames_per_camera"][SEVEN_CAMERAS[0]] = 5
    manifest["status"] = "started"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="train frame-count mismatch"):
        finalize_run_manifest(
            manifest_path,
            "abc123",
        )
