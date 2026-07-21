# SPDX-License-Identifier: Apache-2.0
"""Focused contracts for native-NCore six-camera FTheta launchers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from scripts.pin_ftheta_native_6cam_validation import CAMERA_IDS, validate_resolved_configs

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "apps" / "ncore_3dgut_mcmc_multilayer_inceptio_6cam_native_ab.yaml"
SMOKE = ROOT / "scripts" / "pin_ftheta_native_6cam_smoke.sh"
FULL = ROOT / "scripts" / "pin_ftheta_native_6cam_full.sh"
DERIVE = ROOT / "scripts" / "derive_inceptio_ftheta_6cam.sh"


@pytest.mark.parametrize("driver", [SMOKE, FULL])
def test_v4_driver_uses_separate_native_ncore_manifests(driver: Path) -> None:
    source = driver.read_text(encoding="utf-8")
    assert "PIN_DATA_PATH" in source
    assert "FTHETA_DATA_PATH" in source
    assert 'path="$data_path"' in source
    assert 'run_arm F "$FTHETA_DATA_PATH"' in source
    assert 'ARMS="${ARMS:-F}"' in source
    assert "only runs arm F" in source


@pytest.mark.parametrize("driver", [SMOKE, FULL])
def test_v4_driver_shell_is_syntactically_valid(driver: Path) -> None:
    subprocess.run(["bash", "-n", str(driver)], check=True)


@pytest.mark.parametrize(
    ("mode", "expected_hash_length"),
    [("smoke", 64), ("full", 64)],
)
def test_resolved_pf_configs_differ_only_by_manifest_and_bookkeeping(
    mode: str,
    expected_hash_length: int,
) -> None:
    digest = validate_resolved_configs(
        mode,
        Path("/data/source-pinhole.json"),
        Path("/data/derived-ftheta.json"),
    )
    assert len(digest) == expected_hash_length


def test_runtime_override_is_fail_closed_in_active_dataset_sources() -> None:
    paths = (
        ROOT / "threedgrut" / "datasets" / "datasetNcore.py",
        ROOT / "threedgrut" / "datasets" / "__init__.py",
        ROOT / "configs" / "dataset" / "ncore.yaml",
        CONFIG,
        SMOKE,
        FULL,
    )
    for path in paths:
        source = path.read_text(encoding="utf-8")
        if path in (CONFIG, SMOKE, FULL):
            assert "ftheta_params_path" not in source
        else:
            assert "ftheta_params_path" in source


@pytest.mark.parametrize("driver", [SMOKE, FULL])
def test_native_launchers_refuse_to_start_a_new_pinhole_arm(driver: Path, tmp_path: Path) -> None:
    baseline = tmp_path / "parsed.yaml"
    baseline.write_text("{}\n", encoding="utf-8")
    result = subprocess.run(
        ["bash", str(driver)],
        env={**os.environ, "PIN_DATA_PATH": "/missing-p", "FTHETA_DATA_PATH": "/missing-f", "PIN_BASELINE_PARSED": str(baseline), "ARMS": "PF"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert "only runs arm F" in result.stderr


def test_native_validator_tracks_both_manifests_and_exact_camera_order() -> None:
    source = (ROOT / "scripts" / "pin_ftheta_native_6cam_validation.py").read_text(
        encoding="utf-8"
    )
    assert "pin_manifest" in source
    assert "ftheta_manifest" in source
    assert "runtime FTheta parameter overrides are forbidden" in source
    assert list(CAMERA_IDS) == [
        "camera_front_wide_120fov",
        "camera_cross_left_120fov",
        "camera_cross_right_120fov",
        "camera_rear_left_70fov",
        "camera_rear_right_70fov",
        "camera_back_rear_wide_90fov",
    ]


def test_derivation_launcher_uses_the_same_ordered_six_cameras() -> None:
    source = DERIVE.read_text(encoding="utf-8")
    positions = [source.index(f"--camera-id {camera_id}") for camera_id in CAMERA_IDS]
    assert positions == sorted(positions)
    assert source.count("--camera-id ") == len(CAMERA_IDS)
    subprocess.run(["bash", "-n", str(DERIVE)], check=True)
