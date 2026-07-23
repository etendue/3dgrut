import torch

from threedgrut.model.road_projection_candidates import (
    accumulate_projection_counts,
    make_projection_counts,
    road_and_protection_masks,
)


def intrinsics():
    return {
        "focal_length": torch.tensor([4.0, 4.0]),
        "principal_point": torch.tensor([5.0, 5.0]),
        "resolution": torch.tensor([11.0, 11.0]),
        "radial_coeffs": torch.zeros(6),
        "tangential_coeffs": torch.zeros(2),
        "thin_prism_coeffs": torch.zeros(4),
    }


def test_road_and_protection_masks_leave_safety_margin():
    road = torch.zeros(11, 11, dtype=torch.bool)
    road[3:8, 3:8] = True
    interior, protection = road_and_protection_masks(
        road, erosion_px=1, protection_margin_px=1
    )
    assert interior.sum().item() == 9
    assert not protection[2:9, 2:9].any()
    assert protection[0, 0]


def test_projection_counts_separate_road_and_protected_centers():
    positions = torch.tensor([[0.0, 0.0, 1.0], [0.75, 0.0, 1.0]])
    scales = torch.full((2, 3), 0.01)
    road = torch.zeros(11, 11, dtype=torch.bool)
    road[5, 5] = True
    counts = make_projection_counts(2, positions.device)
    accumulate_projection_counts(
        counts,
        positions_world=positions,
        scales_linear=scales,
        T_camera_to_world=torch.eye(4),
        intrinsics=intrinsics(),
        road_mask=road,
        mog_visibility=torch.ones(2, dtype=torch.bool),
        erosion_px=0,
        protection_margin_px=1,
        footprint_sigma=1.0,
        max_footprint_px=2.0,
        chunk_size=1,
    )
    assert counts["road_center_hits"].tolist() == [1, 0]
    assert counts["protected_center_hits"].tolist() == [0, 1]
    assert counts["visible_hits"].tolist() == [1, 1]
