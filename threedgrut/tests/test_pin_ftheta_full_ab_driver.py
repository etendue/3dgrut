# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the full-window matched seven-camera FTheta A/B."""

from __future__ import annotations

import json
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image

import scripts.pin_ftheta_full_ab_validation as full_validation
from scripts.pin_ftheta_full_ab_validation import (
    EXPECTED_CAMERA_IDS,
    FULL_RUN_PROFILE,
    create_full_run_manifest,
    finalize_full_run_manifest,
    mark_full_run_failed,
    record_full_arm_outputs,
    validate_full_checkpoint,
    validate_full_scientific_config,
    validate_native_render_tree,
    verify_full_run_manifest,
)
from scripts.pin_ftheta_smoke_validation import RENDER_METRIC_KEYS, sha256_file
from threedgrut.datasets.ftheta_override import load_ftheta_override_parameters

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "scripts" / "pin_ftheta_7cam_full_ab.sh"
V4_DRIVER = ROOT / "scripts" / "pin_ftheta_7cam_v4_full_ab.sh"
CONFIG_DIR = str(ROOT / "configs")
BASE_CONFIG = "apps/ncore_3dgut_mcmc_multilayer_inceptio_7cam"
FTHETA_ARTIFACT = "scripts/pin_ftheta_b6a9_7cam_params.json"
FTHETA_ARTIFACT_PATH = ROOT / FTHETA_ARTIFACT
V4_CONFIG_PATH = ROOT / "configs/apps/ncore_3dgut_mcmc_multilayer_inceptio_7cam_v4.yaml"
V4_FTHETA_ARTIFACT_PATH = ROOT / "scripts/pin_ftheta_b6a9_7cam_params_v4_full_domain.json"
SEVEN_CAMERAS = [
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_left_wide_90fov",
    "camera_right_wide_90fov",
    "camera_back_rear_wide_90fov",
    "camera_rear_left_70fov",
]
FROZEN_CLIP_ID = "inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9"


def _png_bytes(*, size: tuple[int, int] = (1920, 1080), color: tuple[int, int, int] = (0, 0, 0)) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return stream.getvalue()


def _write_test_manifest(path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path.write_text(
        json.dumps(
            {
                "sequence_id": FROZEN_CLIP_ID,
                "sequence_timestamp_interval_us": {"start": 1_000_000, "stop": 21_000_000},
                "version": 4,
                "component_stores": {"camera": "clip.ncore4-camera.zarr.itar"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        full_validation,
        "FROZEN_B6A9_MANIFEST_SHA256",
        sha256_file(path),
        raising=False,
    )
    monkeypatch.setattr(
        full_validation,
        "_frozen_calibration_provenance",
        lambda: {
            "clip_id": FROZEN_CLIP_ID,
            "manifest_sha256": sha256_file(path),
        },
    )
    return path


def _compose(overrides: list[str] | None = None):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name=BASE_CONFIG, overrides=overrides or [])


def _full_config(tmp_path: Path, arm: str):
    data_path = tmp_path / "dataset.json"
    data_path.write_text("{}", encoding="utf-8")
    params = "null" if arm == "P" else FTHETA_ARTIFACT
    return _compose(
        [
            "n_iterations=30000",
            "seed_initialization=42",
            "test_last=true",
            f"path={data_path}",
            f"out_dir={tmp_path / 'out'}",
            f"experiment_name=full_arm{arm}",
            "dataset.train.seek_offset_sec=0.0",
            "dataset.train.duration_sec=-1",
            "dataset.val.seek_offset_sec=0.0",
            "dataset.val.duration_sec=-1",
            "dataset.downsample=1.0",
            "dataset.n_val_image_subsample=1",
            "dataset.mask_forward_invalid_pixels=true",
            "dataset.opencv_pinhole_use_validity_domain=false",
            "trainer.sky_backend=mlp",
            "trainer.use_lidar_depth=false",
            "trainer.use_depth_prior=false",
            "dataset.load_lidar_depth_map=false",
            "dataset.load_depth_prior=false",
            "num_workers=10",
            f"dataset.ftheta_params_path={params}",
        ]
    )


def _camera_contracts(arm: str) -> dict:
    parameters, fingerprints = load_ftheta_override_parameters(FTHETA_ARTIFACT_PATH, SEVEN_CAMERAS)
    if arm == "P":
        return {
            camera_id: {"model_type": "OpenCVPinhole", "native_resolution": (1920, 1080)} for camera_id in SEVEN_CAMERAS
        }
    return {
        camera_id: {
            "model_type": "FTheta",
            "native_resolution": (1920, 1080),
            "intrinsics_FTheta": parameters[camera_id],
            "parameter_fingerprint": fingerprints[camera_id],
        }
        for camera_id in SEVEN_CAMERAS
    }


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


def _validator_subprocess_env(tmp_path: Path) -> dict[str, str]:
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


def test_full_driver_is_valid_bash_and_freezes_exact_contract() -> None:
    result = subprocess.run(["bash", "-n", str(DRIVER)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    source = DRIVER.read_text(encoding="utf-8")

    for override in (
        "n_iterations=30000",
        "seed_initialization=42",
        "test_last=true",
        "dataset.train.seek_offset_sec=0.0",
        "dataset.train.duration_sec=-1",
        "dataset.val.seek_offset_sec=0.0",
        "dataset.val.duration_sec=-1",
        "dataset.downsample=1.0",
        "dataset.n_val_image_subsample=1",
        "dataset.mask_forward_invalid_pixels=true",
        "dataset.opencv_pinhole_use_validity_domain=false",
        "trainer.sky_backend=mlp",
        "trainer.use_lidar_depth=false",
        "trainer.use_depth_prior=false",
        "dataset.load_lidar_depth_map=false",
        "dataset.load_depth_prior=false",
        "num_workers=10",
    ):
        assert source.count(override) == 1, override

    assert "--config-name apps/ncore_3dgut_mcmc_multilayer_inceptio_7cam" in source
    assert 'run_arm "P" "null"' in source
    assert f'run_arm "F" "{FTHETA_ARTIFACT}"' in source
    assert 'dataset.ftheta_params_path="$ftheta_params_path"' in source
    assert source.index('run_arm "P" "null"') < source.index(f'run_arm "F" "{FTHETA_ARTIFACT}"')
    assert "camera_front_standard_55fov" not in source
    assert "camera_front_tele_30fov" not in source
    assert "camera_loss_weights" not in source
    assert "TELEW" not in source
    assert tuple(SEVEN_CAMERAS) == EXPECTED_CAMERA_IDS
    assert 'MODE="${1:-run}"' in source
    assert 'if [ "$MODE" = "--preflight" ]' in source
    for command in (
        "preflight",
        "manifest-create",
        "manifest-verify",
        "log",
        "checkpoint",
        "metrics",
        "render-tree",
        "record-arm",
        "finalize",
        "manifest-fail",
    ):
        assert f'-m "$VALIDATOR_MODULE" {command}' in source


@pytest.mark.parametrize("arm", ["P", "F"])
def test_full_scientific_config_accepts_only_20s_30k_native_contract(tmp_path: Path, arm: str) -> None:
    config = _full_config(tmp_path, arm)
    validate_full_scientific_config(config, arm, FTHETA_ARTIFACT_PATH, config.path)

    for dotted_key, bad_value in (
        ("n_iterations", 5000),
        ("dataset.train.duration_sec", 5.0),
        ("dataset.val.duration_sec", 5.0),
        ("dataset.downsample", 0.5),
        ("dataset.n_val_image_subsample", 2),
        ("loss.camera_loss_weights", {"camera_front_tele_30fov": 2.0}),
    ):
        broken = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        OmegaConf.update(broken, dotted_key, bad_value, merge=False)
        with pytest.raises(ValueError):
            validate_full_scientific_config(broken, arm, FTHETA_ARTIFACT_PATH, config.path)


@pytest.mark.parametrize("arm", ["P", "F"])
def test_full_checkpoint_requires_step_30000_and_native_seven_camera_metadata(tmp_path: Path, arm: str) -> None:
    config = _full_config(tmp_path, arm)
    checkpoint = tmp_path / f"{arm}.pt"
    payload = {
        "global_step": 30000,
        "config": config,
        "viz_4d": {"camera_models": _camera_contracts(arm)},
    }
    torch.save(payload, checkpoint)
    validate_full_checkpoint(checkpoint, arm, FTHETA_ARTIFACT_PATH, config.path)

    payload["global_step"] = 5000
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="30000"):
        validate_full_checkpoint(checkpoint, arm, FTHETA_ARTIFACT_PATH, config.path)


def test_native_render_tree_requires_full_png_inventory_and_writes_hashed_record(tmp_path: Path) -> None:
    eval_root = tmp_path / "eval"
    metrics_path = eval_root / "metrics.json"
    render_dir = eval_root / "ours_30000" / "renders"
    gt_dir = eval_root / "ours_30000" / "gt"
    render_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    metrics_path.write_text(json.dumps(_complete_metrics()), encoding="utf-8")
    render_png = _png_bytes(color=(1, 2, 3))
    gt_png = _png_bytes(color=(4, 5, 6))
    for index in range(14):
        (render_dir / f"{index:05d}.png").write_bytes(render_png)
        (gt_dir / f"{index:05d}.png").write_bytes(gt_png)

    inventory_path = eval_root / "native_render_inventory.json"
    inventory = validate_native_render_tree(metrics_path, FTHETA_ARTIFACT_PATH, inventory_path)
    assert inventory["profile"] == FULL_RUN_PROFILE
    assert inventory["global_step"] == 30000
    assert inventory["render_png_count"] == 14
    assert inventory["gt_png_count"] == 14
    assert inventory_path.is_file()

    (render_dir / "00000.png").unlink()
    with pytest.raises(ValueError, match="render PNG count"):
        validate_native_render_tree(metrics_path, FTHETA_ARTIFACT_PATH, inventory_path)


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        (b"not-a-png", "readable PNG"),
        (_png_bytes(size=(16, 8)), "1920x1080"),
    ],
)
def test_native_render_tree_rejects_invalid_or_wrong_size_png(tmp_path: Path, payload: bytes, error: str) -> None:
    eval_root = tmp_path / "eval"
    metrics_path = eval_root / "metrics.json"
    render_dir = eval_root / "ours_30000" / "renders"
    gt_dir = eval_root / "ours_30000" / "gt"
    render_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    metrics_path.write_text(json.dumps(_complete_metrics()), encoding="utf-8")
    valid_png = _png_bytes()
    for index in range(14):
        (render_dir / f"{index:05d}.png").write_bytes(valid_png)
        (gt_dir / f"{index:05d}.png").write_bytes(valid_png)
    (render_dir / "00000.png").write_bytes(payload)
    with pytest.raises(ValueError, match=error):
        validate_native_render_tree(metrics_path, FTHETA_ARTIFACT_PATH)


def test_full_driver_preflight_rejects_unfrozen_manifest_without_output_tree(tmp_path: Path) -> None:
    data_path = tmp_path / "dataset.json"
    data_path.write_text("{}", encoding="utf-8")
    run_base = tmp_path / "must_not_exist"
    env = _validator_subprocess_env(tmp_path)
    env.update({"DATA_PATH": str(data_path), "RUN_BASE": str(run_base)})
    result = subprocess.run(
        ["bash", str(DRIVER), "--preflight"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "frozen b6a9" in result.stderr
    assert not run_base.exists()


def _manifest_create_argv(tmp_path: Path, *, readiness: bool = True) -> list[str]:
    config_path = V4_CONFIG_PATH if readiness else ROOT / "configs/apps/ncore_3dgut_mcmc_multilayer_inceptio_7cam.yaml"
    artifact_path = V4_FTHETA_ARTIFACT_PATH if readiness else FTHETA_ARTIFACT_PATH
    argv = [
        "pin_ftheta_full_ab_validation.py",
        "manifest-create",
        "--path",
        str(tmp_path / "run_manifest.json"),
        "--run-id",
        "test-run",
        "--repo-root",
        str(ROOT),
        "--dataset-manifest",
        str(tmp_path / "dataset.json"),
        "--config",
        str(config_path),
        "--artifact",
        str(artifact_path),
        "--driver",
        str(V4_DRIVER if readiness else DRIVER),
        "--validator",
        str(ROOT / "scripts/pin_ftheta_full_ab_validation.py"),
    ]
    if readiness:
        argv.extend(
            [
                "--ncore-readiness-profile",
                full_validation.V4_MULTILAYER_READINESS_PROFILE,
                "--expected-commit",
                "abc123",
            ]
        )
    return argv


def test_full_manifest_create_readiness_failure_prevents_manifest_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", _manifest_create_argv(tmp_path))
    monkeypatch.setattr(full_validation, "_current_clean_commit", lambda _repo: "abc123")
    monkeypatch.setattr(full_validation, "_prepare_full_run_manifest", lambda *_args, **_kwargs: {"prepared": True})

    def reject(_manifest, _camera_ids, *, required_aux):
        assert tuple(required_aux) == full_validation.V4_REQUIRED_AUX_TYPES
        raise ValueError("canonical store is corrupt")

    monkeypatch.setattr(full_validation, "validate_ncore_data_readiness", reject)
    with pytest.raises(ValueError, match="canonical store is corrupt"):
        full_validation.main()
    assert not (tmp_path / "run_manifest.json").exists()


def test_public_full_create_with_v4_profile_cannot_bypass_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "public.json"
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}", encoding="utf-8")
    sources = _source_files(tmp_path, dataset)
    monkeypatch.setattr(full_validation, "_prepare_full_run_manifest", lambda *_args, **_kwargs: {"prepared": True})
    monkeypatch.setattr(
        full_validation,
        "validate_ncore_data_readiness",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("public full readiness gate")),
    )
    with pytest.raises(ValueError, match="public full readiness gate"):
        create_full_run_manifest(
            output,
            "public-v4",
            "abc123",
            sources,
            ncore_readiness_profile=full_validation.V4_MULTILAYER_READINESS_PROFILE,
        )
    assert not output.exists()


def test_full_manifest_create_rejects_frozen_provenance_before_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", _manifest_create_argv(tmp_path))
    monkeypatch.setattr(full_validation, "_current_clean_commit", lambda _repo: "abc123")
    readiness_called = False

    def readiness(*_args, **_kwargs):
        nonlocal readiness_called
        readiness_called = True

    monkeypatch.setattr(full_validation, "validate_ncore_data_readiness", readiness)
    with pytest.raises(ValueError, match=r"canonical manifest SHA-256"):
        full_validation.main()
    assert not readiness_called
    assert not (tmp_path / "run_manifest.json").exists()


def test_full_manifest_create_runs_readiness_before_manifest_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", _manifest_create_argv(tmp_path))
    events = []

    def clean(repo_root):
        events.append(("clean", Path(repo_root)))
        return "abc123"

    def prepare(run_id, commit, sources, **kwargs):
        events.append(("prepare", run_id, commit, sources, kwargs))
        return {"prepared": True}

    def readiness(manifest, camera_ids, *, required_aux):
        events.append(("readiness", Path(manifest), tuple(camera_ids), tuple(required_aux)))

    def write(path, value, *, exclusive):
        events.append(("write", Path(path), value, exclusive))

    monkeypatch.setattr(full_validation, "_current_clean_commit", clean)
    monkeypatch.setattr(full_validation, "_prepare_full_run_manifest", prepare)
    monkeypatch.setattr(full_validation, "validate_ncore_data_readiness", readiness)
    monkeypatch.setattr(full_validation, "_write_run_manifest", write)
    full_validation.main()

    assert [event[0] for event in events] == ["clean", "prepare", "readiness", "write"]
    assert events[1][1:3] == ("test-run", "abc123")
    assert events[2] == (
        "readiness",
        dataset,
        EXPECTED_CAMERA_IDS,
        full_validation.V4_REQUIRED_AUX_TYPES,
    )
    assert events[3] == ("write", tmp_path / "run_manifest.json", {"prepared": True}, True)


def test_full_manifest_create_without_profile_preserves_legacy_no_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", _manifest_create_argv(tmp_path, readiness=False))
    monkeypatch.setattr(full_validation, "_current_clean_commit", lambda _repo: "abc123")
    monkeypatch.setattr(full_validation, "validate_ncore_data_readiness", lambda *_a, **_k: pytest.fail("called"))
    monkeypatch.setattr(full_validation, "validate_frozen_b6a9_manifest", lambda _path: {})
    written = []
    monkeypatch.setattr(
        full_validation,
        "_write_run_manifest",
        lambda path, value, *, exclusive: written.append((Path(path), value, exclusive)),
    )
    full_validation.main()
    assert written and "ncore_readiness_profile" not in written[0][1]
    assert set(written[0][1]["sources"]) == {"dataset_manifest", *full_validation._FULL_STATIC_SOURCES}


def test_full_v4_profile_records_hashed_readiness_validator_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(full_validation, "validate_frozen_b6a9_manifest", lambda _path: {})
    monkeypatch.setattr(full_validation, "validate_v4_multilayer_dataset_contract", lambda _path: {})
    sources = _source_files(tmp_path, dataset)
    sources["config"] = V4_CONFIG_PATH
    sources["artifact"] = V4_FTHETA_ARTIFACT_PATH
    sources["driver"] = V4_DRIVER
    value = full_validation._prepare_full_run_manifest(
        "v4-full",
        "abc123",
        sources,
        ncore_readiness_profile=full_validation.V4_MULTILAYER_READINESS_PROFILE,
    )
    record = value["sources"]["data_readiness_validator"]
    assert value["ncore_readiness_profile"] == full_validation.V4_MULTILAYER_READINESS_PROFILE
    assert record["sha256"] == sha256_file(ROOT / "scripts/ncore_data_readiness.py")
    assert "v4_provenance_sidecar" in value["sources"]
    assert "v4_driver_validator" in value["sources"]
    assert value["contract"]["train_duration_sec"] == 20.0
    assert value["contract"]["val_duration_sec"] == 20.0


def test_manifest_create_rejects_wrong_hash_and_wrong_b6a9_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen b6a9.*SHA-256"):
        create_full_run_manifest(tmp_path / "empty-run.json", "empty", "abc123", _source_files(tmp_path, empty))

    impostor = tmp_path / "impostor.json"
    impostor.write_text(
        json.dumps(
            {
                "sequence_id": "other-clip",
                "sequence_timestamp_interval_us": {"start": 1, "stop": 2},
                "version": 4,
                "component_stores": {"camera": "x"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(full_validation, "FROZEN_B6A9_MANIFEST_SHA256", sha256_file(impostor))
    monkeypatch.setattr(
        full_validation,
        "_frozen_calibration_provenance",
        lambda: {"clip_id": FROZEN_CLIP_ID, "manifest_sha256": sha256_file(impostor)},
    )
    with pytest.raises(ValueError, match="sequence_id mismatch"):
        create_full_run_manifest(
            tmp_path / "impostor-run.json", "impostor", "abc123", _source_files(tmp_path, impostor)
        )


def _source_files(tmp_path: Path, data_path: Path) -> dict[str, Path]:
    return {
        "dataset_manifest": data_path,
        "artifact": FTHETA_ARTIFACT_PATH,
        "config": ROOT / "configs/apps/ncore_3dgut_mcmc_multilayer_inceptio_7cam.yaml",
        "driver": DRIVER,
        "validator": ROOT / "scripts/pin_ftheta_full_ab_validation.py",
    }


def _successful_log(arm: str) -> str:
    lines = []
    if arm == "F":
        lines.append("[PIN-FTHETA] NCoreDataset [train] explicit override enabled: cameras=7")
    lines.append("NCoreDataset [train] frame counts (after temporal filtering):")
    lines.extend(f"  {camera_id}: 4 frames" for camera_id in SEVEN_CAMERAS)
    lines.extend(
        [
            "  Total: 28 frames",
            'Saved checkpoint to: "/run/ckpt_last.pt"',
            "Training Statistics",
            "Test Metrics",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_full_arm_evidence(
    tmp_path: Path,
    arm: str,
    data_path: Path,
    *,
    shared_gt_png: bytes | None = None,
) -> dict[str, Path]:
    arm_root = tmp_path / arm
    arm_root.mkdir()
    config = _full_config(arm_root, arm)
    config.path = str(data_path)
    parsed = arm_root / "parsed.yaml"
    OmegaConf.save(config, parsed)
    checkpoint = arm_root / "ckpt_last.pt"
    torch.save(
        {
            "global_step": 30000,
            "config": config,
            "viz_4d": {"camera_models": _camera_contracts(arm)},
        },
        checkpoint,
    )
    eval_root = arm_root / "eval"
    metrics = eval_root / "metrics.json"
    render_dir = eval_root / "ours_30000" / "renders"
    gt_dir = eval_root / "ours_30000" / "gt"
    render_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    metrics.write_text(json.dumps(_complete_metrics()), encoding="utf-8")
    render_png = _png_bytes(color=(1 if arm == "P" else 2, 3, 4))
    gt_png = shared_gt_png if shared_gt_png is not None else _png_bytes(color=(5, 6, 7))
    for index in range(14):
        (render_dir / f"{index:05d}.png").write_bytes(render_png)
        (gt_dir / f"{index:05d}.png").write_bytes(gt_png)
    inventory = arm_root / "native_render_inventory.json"
    validate_native_render_tree(metrics, FTHETA_ARTIFACT_PATH, inventory)
    train_log = arm_root / "train.log"
    train_log.write_text(_successful_log(arm), encoding="utf-8")
    eval_log = arm_root / "eval.log"
    eval_log.write_text("native render complete\n", encoding="utf-8")
    return {
        "parsed": parsed,
        "checkpoint": checkpoint,
        "metrics": metrics,
        "train_log": train_log,
        "eval_log": eval_log,
        "inventory": inventory,
        "render_dir": render_dir,
    }


def test_full_manifest_records_ordered_arms_completion_failure_and_render_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "dataset.json"
    _write_test_manifest(data_path, monkeypatch)
    manifest = tmp_path / "run_manifest.json"
    create_full_run_manifest(manifest, "full-run", "abc123", _source_files(tmp_path, data_path))
    started = verify_full_run_manifest(manifest, "abc123")
    assert started["status"] == "running"
    assert started["contract"]["arm_order"] == ["P", "F"]

    failure_manifest = tmp_path / "failed_manifest.json"
    create_full_run_manifest(failure_manifest, "failed-run", "abc123", _source_files(tmp_path, data_path))
    failed = mark_full_run_failed(failure_manifest, "armP-train-test_last", 17)
    assert failed["status"] == "failed"
    assert failed["failure"] == {"stage": "armP-train-test_last", "exit_code": 17}

    evidence_by_arm = {arm: _write_full_arm_evidence(tmp_path, arm, data_path) for arm in ("P", "F")}
    for arm in ("P", "F"):
        evidence = evidence_by_arm[arm]
        record_full_arm_outputs(
            manifest,
            arm,
            evidence["parsed"],
            evidence["checkpoint"],
            evidence["metrics"],
            evidence["train_log"],
            evidence["eval_log"],
            evidence["inventory"],
            FTHETA_ARTIFACT_PATH,
            data_path,
            "abc123",
        )
    complete = finalize_full_run_manifest(manifest, "abc123")
    assert complete["status"] == "complete"
    assert complete["comparison"]["only_representation_path_differs"] is True
    assert complete["comparison"]["arm_order"] == ["P", "F"]

    (evidence_by_arm["F"]["render_dir"] / "00000.png").write_bytes(_png_bytes(color=(99, 98, 97)))
    with pytest.raises(ValueError, match="native render tree drifted"):
        verify_full_run_manifest(manifest, "abc123")


def test_finalize_rejects_different_p_f_gt_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_path = _write_test_manifest(tmp_path / "dataset.json", monkeypatch)
    manifest = tmp_path / "run_manifest.json"
    create_full_run_manifest(manifest, "gt-mismatch", "abc123", _source_files(tmp_path, data_path))
    evidence_by_arm = {
        arm: _write_full_arm_evidence(
            tmp_path,
            arm,
            data_path,
            shared_gt_png=_png_bytes(color=(10 if arm == "P" else 20, 30, 40)),
        )
        for arm in ("P", "F")
    }
    for arm in ("P", "F"):
        evidence = evidence_by_arm[arm]
        record_full_arm_outputs(
            manifest,
            arm,
            evidence["parsed"],
            evidence["checkpoint"],
            evidence["metrics"],
            evidence["train_log"],
            evidence["eval_log"],
            evidence["inventory"],
            FTHETA_ARTIFACT_PATH,
            data_path,
            "abc123",
        )
    with pytest.raises(ValueError, match="P/F GT"):
        finalize_full_run_manifest(manifest, "abc123")


def test_finalize_rejects_different_p_f_gt_frame_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_path = _write_test_manifest(tmp_path / "dataset.json", monkeypatch)
    manifest = tmp_path / "run_manifest.json"
    create_full_run_manifest(manifest, "gt-set-mismatch", "abc123", _source_files(tmp_path, data_path))
    evidence_by_arm = {arm: _write_full_arm_evidence(tmp_path, arm, data_path) for arm in ("P", "F")}
    f_evidence = evidence_by_arm["F"]
    f_eval_root = f_evidence["metrics"].parent / "ours_30000"
    (f_eval_root / "renders" / "00013.png").rename(f_eval_root / "renders" / "99999.png")
    (f_eval_root / "gt" / "00013.png").rename(f_eval_root / "gt" / "99999.png")
    validate_native_render_tree(f_evidence["metrics"], FTHETA_ARTIFACT_PATH, f_evidence["inventory"])
    for arm in ("P", "F"):
        evidence = evidence_by_arm[arm]
        record_full_arm_outputs(
            manifest,
            arm,
            evidence["parsed"],
            evidence["checkpoint"],
            evidence["metrics"],
            evidence["train_log"],
            evidence["eval_log"],
            evidence["inventory"],
            FTHETA_ARTIFACT_PATH,
            data_path,
            "abc123",
        )
    with pytest.raises(ValueError, match="P/F GT PNG frame set"):
        finalize_full_run_manifest(manifest, "abc123")


def test_record_arm_requires_running_unique_ordered_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_path = _write_test_manifest(tmp_path / "dataset.json", monkeypatch)
    evidence = _write_full_arm_evidence(tmp_path, "P", data_path)

    def record(path: Path, arm: str) -> None:
        record_full_arm_outputs(
            path,
            arm,
            evidence["parsed"],
            evidence["checkpoint"],
            evidence["metrics"],
            evidence["train_log"],
            evidence["eval_log"],
            evidence["inventory"],
            FTHETA_ARTIFACT_PATH,
            data_path,
            "abc123",
        )

    failed_manifest = tmp_path / "failed.json"
    create_full_run_manifest(failed_manifest, "failed", "abc123", _source_files(tmp_path, data_path))
    mark_full_run_failed(failed_manifest, "test", 1)
    with pytest.raises(ValueError, match="status.*running"):
        record(failed_manifest, "P")

    wrong_order = tmp_path / "wrong-order.json"
    create_full_run_manifest(wrong_order, "wrong", "abc123", _source_files(tmp_path, data_path))
    with pytest.raises(ValueError, match="expected P"):
        record(wrong_order, "F")

    duplicate = tmp_path / "duplicate.json"
    create_full_run_manifest(duplicate, "duplicate", "abc123", _source_files(tmp_path, data_path))
    record(duplicate, "P")
    with pytest.raises(ValueError, match="already recorded"):
        record(duplicate, "P")


def test_completed_manifest_is_immutable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_path = _write_test_manifest(tmp_path / "dataset.json", monkeypatch)
    manifest = tmp_path / "run_manifest.json"
    create_full_run_manifest(manifest, "immutable", "abc123", _source_files(tmp_path, data_path))
    evidence_by_arm = {arm: _write_full_arm_evidence(tmp_path, arm, data_path) for arm in ("P", "F")}
    for arm in ("P", "F"):
        evidence = evidence_by_arm[arm]
        record_full_arm_outputs(
            manifest,
            arm,
            evidence["parsed"],
            evidence["checkpoint"],
            evidence["metrics"],
            evidence["train_log"],
            evidence["eval_log"],
            evidence["inventory"],
            FTHETA_ARTIFACT_PATH,
            data_path,
            "abc123",
        )
    finalize_full_run_manifest(manifest, "abc123")
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match="status.*running"):
        finalize_full_run_manifest(manifest, "abc123")
    p_evidence = evidence_by_arm["P"]
    with pytest.raises(ValueError, match="status.*running"):
        record_full_arm_outputs(
            manifest,
            "P",
            p_evidence["parsed"],
            p_evidence["checkpoint"],
            p_evidence["metrics"],
            p_evidence["train_log"],
            p_evidence["eval_log"],
            p_evidence["inventory"],
            FTHETA_ARTIFACT_PATH,
            data_path,
            "abc123",
        )
    with pytest.raises(ValueError, match="cannot mark a completed"):
        mark_full_run_failed(manifest, "late", 2)
    assert manifest.read_bytes() == before


def test_full_driver_err_trap_preserves_original_failure_status() -> None:
    source = DRIVER.read_text(encoding="utf-8")
    assert "set -Eeuo pipefail" in source
    assert "local rc=$?" in source
    assert "trap - ERR" in source
    assert 'exit "$rc"' in source
    assert "manifest-fail" in source
    assert 'CURRENT_STAGE="arm${arm}-source-verify"' in source


def test_record_arm_rejects_alternate_manifest_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_path = _write_test_manifest(tmp_path / "dataset.json", monkeypatch)
    manifest = tmp_path / "run_manifest.json"
    create_full_run_manifest(manifest, "sources", "abc123", _source_files(tmp_path, data_path))
    alternate_data = tmp_path / "alternate-dataset.json"
    alternate_data.write_bytes(data_path.read_bytes())
    evidence = _write_full_arm_evidence(tmp_path, "P", alternate_data)
    alternate_artifact = tmp_path / "alternate-artifact.json"
    alternate_artifact.write_bytes(FTHETA_ARTIFACT_PATH.read_bytes())
    with pytest.raises(ValueError, match="artifact.*path"):
        record_full_arm_outputs(
            manifest,
            "P",
            evidence["parsed"],
            evidence["checkpoint"],
            evidence["metrics"],
            evidence["train_log"],
            evidence["eval_log"],
            evidence["inventory"],
            alternate_artifact,
            alternate_data,
            "abc123",
        )
    with pytest.raises(ValueError, match="dataset_manifest.*path"):
        record_full_arm_outputs(
            manifest,
            "P",
            evidence["parsed"],
            evidence["checkpoint"],
            evidence["metrics"],
            evidence["train_log"],
            evidence["eval_log"],
            evidence["inventory"],
            FTHETA_ARTIFACT_PATH,
            alternate_data,
            "abc123",
        )


def test_manifest_source_inventory_covers_execution_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = _write_test_manifest(tmp_path / "dataset.json", monkeypatch)
    manifest = tmp_path / "run_manifest.json"
    value = create_full_run_manifest(manifest, "sources", "abc123", _source_files(tmp_path, data_path))
    assert {
        "smoke_validator",
        "train_entrypoint",
        "render_entrypoint",
        "render_implementation",
        "dataset_implementation",
        "ftheta_override",
        "trainer_implementation",
        "frozen_calibration_provenance",
        "experiment_spec",
    } <= set(value["sources"])


def test_manifest_verify_rejects_head_and_comprehensive_source_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = _write_test_manifest(tmp_path / "dataset.json", monkeypatch)
    manifest = tmp_path / "run_manifest.json"
    create_full_run_manifest(manifest, "drift", "abc123", _source_files(tmp_path, data_path))
    with pytest.raises(ValueError, match="git commit drift"):
        verify_full_run_manifest(manifest, "different-head")

    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["sources"]["train_entrypoint"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="source hash drift for train_entrypoint"):
        verify_full_run_manifest(manifest, "abc123")


@pytest.mark.parametrize("command", ["manifest-verify", "record-arm", "finalize"])
def test_manifest_cli_operations_recheck_tracked_clean(monkeypatch: pytest.MonkeyPatch, command: str) -> None:
    calls: list[Path] = []
    head_calls: list[Path] = []
    monkeypatch.setattr(full_validation, "ensure_tracked_worktree_clean", lambda root: calls.append(Path(root)))
    monkeypatch.setattr(
        full_validation,
        "_git_commit",
        lambda root: head_calls.append(Path(root)) or "abc123",
    )
    monkeypatch.setattr(full_validation, "verify_full_run_manifest", lambda *args, **kwargs: {})
    monkeypatch.setattr(full_validation, "record_full_arm_outputs", lambda *args, **kwargs: {})
    monkeypatch.setattr(full_validation, "finalize_full_run_manifest", lambda *args, **kwargs: {})
    arguments = ["validator", command, "--repo-root", str(ROOT)]
    if command == "manifest-verify":
        arguments.extend(["--path", "/tmp/manifest.json"])
    elif command == "finalize":
        arguments.extend(["--manifest", "/tmp/manifest.json"])
    else:
        arguments.extend(
            [
                "--manifest",
                "/tmp/manifest.json",
                "--arm",
                "P",
                "--parsed-yaml",
                "/tmp/parsed.yaml",
                "--checkpoint",
                "/tmp/checkpoint.pt",
                "--metrics",
                "/tmp/metrics.json",
                "--train-log",
                "/tmp/train.log",
                "--eval-log",
                "/tmp/eval.log",
                "--native-render-inventory",
                "/tmp/inventory.json",
                "--artifact",
                "/tmp/artifact.json",
                "--input-manifest",
                "/tmp/dataset.json",
            ]
        )
    monkeypatch.setattr(sys, "argv", arguments)
    full_validation.main()
    assert calls == [ROOT]
    assert head_calls == [ROOT]
