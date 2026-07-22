from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from threedgrut.telemetry.per_camera_stats import PerCameraTelemetry


def test_base_config_keeps_per_camera_telemetry_off_by_default():
    config_path = Path(__file__).resolve().parents[2] / "configs" / "base_gs.yaml"
    assert OmegaConf.load(config_path).trainer.per_camera_telemetry is False


def test_per_camera_telemetry_tracks_uneven_sampling_and_relocations(tmp_path):
    telemetry = PerCameraTelemetry()
    telemetry.record_step("front", {"total_loss": 2.0, "rgb": 1.0}, grad_norm=0.5)
    telemetry.record_step("front", {"total_loss": 4.0, "rgb": 3.0}, grad_norm=2.0)
    telemetry.record_step("rear", {"total_loss": 9.0}, grad_norm=0.0)
    telemetry.record_relocation("background", 7)
    telemetry.record_relocation("background", 2)
    telemetry.record_ownership_actions("front", {"n_recycled": 3, "n_footprint_shrunk": 4})
    path = tmp_path / "per_camera_telemetry.json"

    telemetry.dump(path)
    payload = json.loads(path.read_text())

    assert payload["cameras"]["front"]["n_steps"] == 2
    assert payload["cameras"]["front"]["mean_losses"]["total_loss"] == pytest.approx(3.0)
    assert payload["cameras"]["rear"]["mean_grad_norm"] == 0.0
    assert payload["cameras"]["front"]["grad_norm_bins"]["[1e-3,1)"] == 1
    assert payload["relocations_by_layer"] == {"background": 9}
    assert payload["ownership_actions_by_camera"]["front"] == {
        "n_footprint_shrunk": 4,
        "n_recycled": 3,
    }
