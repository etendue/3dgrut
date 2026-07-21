"""Unit tests for the offline MCRO checkpoint layer-statistics driver."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


_DRIVER_PATH = Path(__file__).resolve().parents[2] / "scripts/drivers/mcro_ckpt_layer_stats.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("mcro_ckpt_layer_stats", _DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _node(scale_values: list[list[float]], opacity_values: list[float]) -> dict[str, torch.Tensor]:
    opacity = torch.tensor(opacity_values, dtype=torch.float32).reshape(-1, 1)
    return {
        "positions": torch.zeros((len(scale_values), 3), dtype=torch.float32),
        "scale": torch.log(torch.tensor(scale_values, dtype=torch.float32)),
        "density": torch.logit(opacity),
    }


def _write_checkpoint(path: Path, background_count: int = 3) -> None:
    background_scales = [[1.0, 2.0, 4.0], [2.0, 4.0, 8.0], [4.0, 8.0, 16.0]][:background_count]
    background_opacity = [0.001, 0.01, 0.5][:background_count]
    torch.save(
        {
            "model": {
                "gaussians_nodes": {
                    "background": _node(background_scales, background_opacity),
                    "road": _node([[0.5, 0.5, 0.1], [1.0, 1.0, 0.2]], [0.25, 0.75]),
                }
            }
        },
        path,
    )


def test_compute_layer_stats_reports_activated_scale_and_opacity_percentiles(tmp_path: Path) -> None:
    driver = _load_driver()
    checkpoint = tmp_path / "synthetic.pt"
    _write_checkpoint(checkpoint)

    stats = driver.compute_layer_stats(str(checkpoint), alive_threshold=0.005)
    background = stats["layers"]["background"]

    assert background["n_particles"] == 3
    assert background["alive_ratio"] == pytest.approx(2.0 / 3.0, abs=1e-6)
    assert background["scale_p50"] == pytest.approx(
        np.percentile(np.array([[1.0, 2.0, 4.0], [2.0, 4.0, 8.0], [4.0, 8.0, 16.0]]), 50, axis=0),
        abs=1e-6,
    )
    assert background["opacity_p10"] == pytest.approx(
        float(np.percentile([0.001, 0.01, 0.5], 10)), abs=1e-6
    )
    assert stats["alive_threshold"] == pytest.approx(0.005)


def test_compare_layer_stats_uses_b_minus_a_delta(tmp_path: Path) -> None:
    driver = _load_driver()
    ckpt_a = tmp_path / "a.pt"
    ckpt_b = tmp_path / "b.pt"
    _write_checkpoint(ckpt_a, background_count=2)
    _write_checkpoint(ckpt_b, background_count=3)

    comparison = driver.compare_checkpoints(str(ckpt_a), str(ckpt_b), alive_threshold=0.005)

    background = comparison["layers"]["background"]
    assert background["delta"]["n_particles"] == 1
    assert background["delta"]["alive_ratio"] == pytest.approx(1.0 / 6.0, abs=1e-6)
    assert comparison["delta_convention"] == "b_minus_a"


def test_missing_required_layer_tensor_names_the_layer(tmp_path: Path) -> None:
    driver = _load_driver()
    checkpoint = tmp_path / "broken.pt"
    torch.save(
        {
            "model": {
                "gaussians_nodes": {
                    "road": {"positions": torch.zeros((1, 3)), "density": torch.zeros((1, 1))}
                }
            }
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="road.*scale"):
        driver.compute_layer_stats(str(checkpoint))


def test_compare_rejects_a_missing_particle_layer_by_name(tmp_path: Path) -> None:
    driver = _load_driver()
    complete = tmp_path / "complete.pt"
    missing_road = tmp_path / "missing_road.pt"
    _write_checkpoint(complete)
    torch.save(
        {"model": {"gaussians_nodes": {"background": _node([[1.0, 1.0, 1.0]], [0.5])}}},
        missing_road,
    )

    with pytest.raises(ValueError, match="road"):
        driver.compare_checkpoints(str(complete), str(missing_road))


def test_write_outputs_emits_json_and_markdown(tmp_path: Path) -> None:
    driver = _load_driver()
    checkpoint = tmp_path / "synthetic.pt"
    _write_checkpoint(checkpoint)

    json_path, markdown_path = driver.write_outputs(driver.compute_layer_stats(str(checkpoint)), tmp_path / "report")

    assert json.loads(json_path.read_text(encoding="utf-8"))["layers"]["road"]["n_particles"] == 2
    assert "| background |" in markdown_path.read_text(encoding="utf-8")
