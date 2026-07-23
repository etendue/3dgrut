import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


P = Path(__file__).resolve().parents[2] / "scripts/drivers/mcro_filter_bg_projection_candidates.py"


def module():
    spec = importlib.util.spec_from_file_location("projection_filter", P)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def counts():
    return {
        "visible_hits": np.array([4, 4, 4, 0]),
        "road_center_hits": np.array([4, 2, 0, 0]),
        "road_footprint_hits": np.array([4, 3, 1, 0]),
        "protected_center_hits": np.array([0, 2, 4, 0]),
        "protected_footprint_hits": np.array([0, 3, 4, 0]),
    }


def test_projection_mask_combines_road_ratio_and_protection():
    mask = module().projection_candidate_mask(
        counts(),
        road_field="road_footprint_hits",
        min_road_hits=2,
        protect_field="protected_center_hits",
        max_protected_hits=0,
        min_road_visible_ratio=0.8,
    )
    assert mask.tolist() == [True, False, False, False]


def test_projection_mask_rejects_invalid_fields():
    with pytest.raises(ValueError, match="road_field"):
        module().projection_candidate_mask(
            counts(),
            road_field="bad",
            min_road_hits=1,
            protect_field="protected_center_hits",
            max_protected_hits=0,
            min_road_visible_ratio=0,
        )


def test_projection_filter_preserves_shapes():
    background = {
        "positions": torch.zeros(2, 3),
        "density": torch.zeros(2, 1),
    }
    checkpoint = {"model": {"gaussians_nodes": {"background": background}}}
    result = module().filter_checkpoint(checkpoint, np.array([True, False]), {})
    assert result["model"]["gaussians_nodes"]["background"]["density"].shape == (2, 1)
    assert result["model"]["gaussians_nodes"]["background"]["density"][:, 0].tolist() == [-100, 0]
    assert result["mcro_bg_projection_filter"]["n_alive_candidates"] == 1
