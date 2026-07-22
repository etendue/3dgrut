from types import SimpleNamespace
from pathlib import Path

import torch
from omegaconf import OmegaConf

from threedgrut.model.road_ownership import apply_bg_road_exclusion
from threedgrut.model.road_region import build_road_height_field


def test_b2_config_default_is_off():
    config_path = Path(__file__).resolve().parents[2] / "configs/base_gs.yaml"
    assert OmegaConf.load(config_path).layers.bg_road_exclusion.enabled is False


class FakeBackground:
    def __init__(self, positions, scales, density):
        self.positions = torch.nn.Parameter(positions.clone())
        self.scale = torch.nn.Parameter(scales.clone())
        self.density = torch.nn.Parameter(density.clone())

    def get_scale(self):
        return self.scale.exp()

    def get_density(self):
        return self.density.sigmoid()


def _batch(road_mask):
    return SimpleNamespace(
        T_to_world=torch.eye(4).unsqueeze(0),
        image_infos={"road_mask": road_mask.unsqueeze(0)},
        intrinsics_OpenCVPinholeCameraModelParameters={
            "resolution": [9, 9],
            "focal_length": [4.0, 4.0],
            "principal_point": [4.0, 4.0],
            "radial_coeffs": [0.0] * 6,
            "tangential_coeffs": [0.0] * 2,
            "thin_prism_coeffs": [0.0] * 4,
            "max_valid_r2": 100.0,
        },
    )


def _cfg(chunk_size=2):
    return {
        "z_band": 0.2,
        "projection_max_height": 1.0,
        "chunk_size": chunk_size,
        "opacity_threshold": 0.005,
        "dead_density_raw": -50.0,
        "footprint_sigma": 2.0,
        "max_footprint_px": 4.0,
        "footprint_shrink_factor": 0.5,
        "min_footprint_scale": 1e-4,
    }


def test_b2_center_slab_projection_height_guard_and_road_outside():
    road_positions = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    height_field = build_road_height_field(road_positions, cell_size=1.0)
    positions = torch.tensor([
        [0.0, 0.0, 0.05],  # slab -> recycle
        [0.0, 0.0, 0.50],  # floating but projects to road -> recycle
        [0.0, 0.0, 1.50],  # object above guard -> untouched
        [3.0, 0.0, 1.00],  # outside road support -> untouched
    ])
    bg = FakeBackground(positions, torch.full((4, 3), -2.0), torch.full((4, 1), 2.0))
    road_mask = torch.zeros(9, 9)
    road_mask[4, 4] = 1
    stats = apply_bg_road_exclusion(bg, height_field, _batch(road_mask), _cfg())
    assert torch.equal(bg.density.detach().reshape(-1) == -50.0, torch.tensor([True, True, False, False]))
    assert stats["n_center_slab_hit"] == 1
    assert stats["n_center_proj_hit"] == 2
    assert stats["n_recycled"] == 2


def test_b2_footprint_only_shrinks_without_recycling():
    road_positions = torch.tensor([[0.0, 0.0, 2.0]])
    height_field = build_road_height_field(road_positions, cell_size=1.0)
    # Projects to u=5 while road is at u=4; large scale footprint reaches it.
    bg = FakeBackground(
        torch.tensor([[0.5, 0.0, 2.5]]),
        torch.tensor([[0.0, 0.0, -2.0]]),
        torch.tensor([[2.0]]),
    )
    road_mask = torch.zeros(9, 9)
    road_mask[4, 4] = 1
    before = bg.scale.detach().clone()
    stats = apply_bg_road_exclusion(bg, height_field, _batch(road_mask), _cfg(chunk_size=1))
    assert stats["n_recycled"] == 0
    assert stats["n_footprint_shrunk"] == 1
    assert torch.all(bg.scale[0, :2] < before[0, :2])
    assert bg.density.item() == 2.0


def test_b2_repeated_footprint_shrink_saturates_above_zero():
    """Pin the R2 step-2707 failure: repeated hits must not reach log(0)."""
    road_positions = torch.tensor([[0.0, 0.0, 2.0]])
    height_field = build_road_height_field(road_positions, cell_size=1.0)
    bg = FakeBackground(
        torch.tensor([[0.5, 0.0, 2.5]]),
        torch.tensor([[0.0, 0.0, -2.0]]),
        torch.tensor([[2.0]]),
    )
    road_mask = torch.zeros(9, 9)
    road_mask[4, 4] = 1

    for _ in range(300):
        apply_bg_road_exclusion(bg, height_field, _batch(road_mask), _cfg(chunk_size=1))

    assert torch.isfinite(bg.scale).all()
    assert torch.all(bg.get_scale()[0, :2] > 0.0)
    assert torch.allclose(bg.get_scale()[0, :2], torch.full((2,), 1e-4), rtol=1e-5, atol=0.0)


def test_b2_chunking_is_result_equivalent():
    road_positions = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    height_field = build_road_height_field(road_positions, cell_size=1.0)
    positions = torch.tensor([[0.0, 0.0, 0.05], [0.0, 0.0, 0.5], [0.0, 0.0, 1.5]])
    road_mask = torch.zeros(9, 9)
    road_mask[4, 4] = 1
    a = FakeBackground(positions, torch.full((3, 3), -2.0), torch.full((3, 1), 2.0))
    b = FakeBackground(positions, torch.full((3, 3), -2.0), torch.full((3, 1), 2.0))
    stats_a = apply_bg_road_exclusion(a, height_field, _batch(road_mask), _cfg(chunk_size=1))
    stats_b = apply_bg_road_exclusion(b, height_field, _batch(road_mask), _cfg(chunk_size=99))
    assert stats_a == stats_b
    assert torch.equal(a.density, b.density)
    assert torch.equal(a.scale, b.scale)
