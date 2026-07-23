import importlib.util
from pathlib import Path

import torch


P = Path(__file__).resolve().parents[2] / "scripts/drivers/mcro_filter_bg_road_relative.py"


def module():
    spec = importlib.util.spec_from_file_location("relative_filter", P)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def layers():
    road_positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
    )
    background_positions = torch.tensor([[0.1, 0.0, -2.0], [0.1, 0.0, -2.0]])
    return (
        {
            "positions": background_positions,
            "scale": torch.tensor([[0.1, 0.1, 1.1], [0.1, 0.1, 0.2]]).log(),
            "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
            "density": torch.zeros(2, 1),
        },
        {"positions": road_positions},
    )


def test_extent_aware_filter_selects_only_road_reaching_gaussian():
    background, road = layers()
    mask, report = module().build_candidate_mask(
        background,
        road,
        cell_size=1.0,
        min_support=3,
        max_xy_distance=0.1,
        max_z_dispersion=0.25,
        relative_height_min=-0.1,
        relative_height_max=0.1,
        sigma_multiplier=2.0,
        chunk_size=1,
    )
    assert mask.tolist() == [True, False]
    assert report["n_candidates"] == 1


def test_filter_checkpoint_preserves_shapes_and_records_metadata():
    background, road = layers()
    checkpoint = {"model": {"gaussians_nodes": {"background": background, "road": road}}}
    mask = torch.tensor([True, False])
    result = module().filter_checkpoint(checkpoint, mask, {"n_candidates": 1})
    assert result["model"]["gaussians_nodes"]["background"]["density"].shape == (2, 1)
    assert result["model"]["gaussians_nodes"]["background"]["density"][0].item() == -100
    assert result["model"]["gaussians_nodes"]["background"]["density"][1].item() == 0
    assert result["mcro_bg_road_relative_filter"]["n_alive_candidates"] == 1
