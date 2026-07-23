import importlib.util
from pathlib import Path

import pytest
import torch


P = Path(__file__).resolve().parents[2] / "scripts/drivers/mcro_road_surface_stats.py"


def module():
    spec = importlib.util.spec_from_file_location("surface_stats", P)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_analyze_surface_reports_relative_height_candidates():
    road = torch.tensor(
        [[x + dx, 0.0, 0.2 * x] for x in range(3) for dx in (0.0, 0.1, 0.2)]
    )
    background = torch.tensor(
        [[0.1, 0.0, 0.05], [1.1, 0.0, 0.25], [2.1, 0.0, 3.0], [99.0, 99.0, 0.0]]
    )
    report = module().analyze_surface(
        road,
        background,
        cell_size=1.0,
        min_support=3,
        max_xy_distance=0.1,
        max_z_dispersion=0.01,
        chunk_size=2,
    )
    assert report["n_background_surface_valid"] == 3
    assert report["candidate_-0.25_+0.15_count"] == 2
    assert report["road_particle_z_min"] == pytest.approx(0.0)
    assert report["road_particle_z_max"] == pytest.approx(0.4)


def test_checkpoint_layer_positions_errors_cleanly():
    with pytest.raises(KeyError, match="background"):
        module().checkpoint_layer_positions({"model": {"gaussians_nodes": {}}}, "background")
