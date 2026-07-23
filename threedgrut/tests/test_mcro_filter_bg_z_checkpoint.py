import pytest
import torch

from scripts.drivers.mcro_filter_bg_z_checkpoint import filter_checkpoint


def _checkpoint() -> dict:
    return {
        "model": {
            "gaussians_nodes": {
                "background": {
                    "positions": torch.tensor(
                        [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
                    ),
                    "density": torch.nn.Parameter(torch.tensor([[1.0], [2.0], [3.0]])),
                }
            }
        }
    }


def test_filter_checkpoint_hides_negative_z_without_changing_shapes():
    checkpoint = _checkpoint()
    positions_shape = checkpoint["model"]["gaussians_nodes"]["background"][
        "positions"
    ].shape

    result = filter_checkpoint(checkpoint, drop_side="lt", threshold=0.0)
    background = result["model"]["gaussians_nodes"]["background"]

    assert background["positions"].shape == positions_shape
    assert background["density"].squeeze().tolist() == [-100.0, 2.0, 3.0]
    assert result["mcro_bg_z_filter"]["n_dropped"] == 1


def test_filter_checkpoint_hides_positive_z_only():
    result = filter_checkpoint(_checkpoint(), drop_side="gt", threshold=0.0)
    density = result["model"]["gaussians_nodes"]["background"]["density"]

    assert density.squeeze().tolist() == [1.0, 2.0, -100.0]
    assert result["mcro_bg_z_filter"]["fraction_dropped"] == pytest.approx(1 / 3)
