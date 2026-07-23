# SPDX-License-Identifier: Apache-2.0
"""Multi-view projection evidence for background-on-road candidates."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from threedgrut.model.road_ownership import project_pinhole_road_overlap


def make_projection_counts(n_gaussians: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros(n_gaussians, dtype=torch.int32, device=device)
        for name in (
            "visible_hits",
            "road_center_hits",
            "road_footprint_hits",
            "protected_center_hits",
            "protected_footprint_hits",
        )
    }


def road_and_protection_masks(
    road_mask: torch.Tensor,
    *,
    erosion_px: int,
    protection_margin_px: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return eroded road interior and non-road pixels beyond a margin."""
    mask = road_mask
    while mask.ndim > 2:
        if mask.shape[0] == 1:
            mask = mask.squeeze(0)
        elif mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        else:
            raise ValueError(f"road_mask must reduce to [H,W], got {tuple(road_mask.shape)}")
    mask = mask.bool()
    image = mask[None, None].float()
    if erosion_px:
        road = (
            F.avg_pool2d(
                image,
                2 * erosion_px + 1,
                stride=1,
                padding=erosion_px,
                count_include_pad=True,
            )
            == 1
        )[0, 0]
    else:
        road = mask
    if protection_margin_px:
        dilated = F.max_pool2d(
            image,
            2 * protection_margin_px + 1,
            stride=1,
            padding=protection_margin_px,
        )[0, 0].bool()
    else:
        dilated = mask
    return road, ~dilated


@torch.no_grad()
def accumulate_projection_counts(
    counts: dict[str, torch.Tensor],
    *,
    positions_world: torch.Tensor,
    scales_linear: torch.Tensor,
    T_camera_to_world: torch.Tensor,
    intrinsics: dict,
    road_mask: torch.Tensor,
    mog_visibility: torch.Tensor | None,
    erosion_px: int = 8,
    protection_margin_px: int = 16,
    footprint_sigma: float = 2.0,
    max_footprint_px: float = 48.0,
    chunk_size: int = 100_000,
) -> None:
    """Accumulate road/protection projection hits for one camera frame."""
    road, protection = road_and_protection_masks(
        road_mask,
        erosion_px=erosion_px,
        protection_margin_px=protection_margin_px,
    )
    visibility = None
    if mog_visibility is not None:
        visibility = mog_visibility.reshape(-1).bool()
        if visibility.numel() != positions_world.shape[0]:
            raise ValueError("mog_visibility must match background Gaussian count")
    for start in range(0, positions_world.shape[0], chunk_size):
        stop = min(start + chunk_size, positions_world.shape[0])
        visible, road_center, road_footprint = project_pinhole_road_overlap(
            positions_world[start:stop],
            scales_linear[start:stop],
            T_camera_to_world,
            intrinsics,
            road,
            footprint_sigma=footprint_sigma,
            max_footprint_px=max_footprint_px,
        )
        _, protected_center, protected_footprint = project_pinhole_road_overlap(
            positions_world[start:stop],
            scales_linear[start:stop],
            T_camera_to_world,
            intrinsics,
            protection,
            footprint_sigma=footprint_sigma,
            max_footprint_px=max_footprint_px,
        )
        if visibility is not None:
            visible &= visibility[start:stop]
        counts["visible_hits"][start:stop] += visible.int()
        counts["road_center_hits"][start:stop] += (visible & road_center).int()
        counts["road_footprint_hits"][start:stop] += (visible & road_footprint).int()
        counts["protected_center_hits"][start:stop] += (visible & protected_center).int()
        counts["protected_footprint_hits"][start:stop] += (visible & protected_footprint).int()
