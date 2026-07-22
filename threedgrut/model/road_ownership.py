# SPDX-License-Identifier: Apache-2.0
"""GPU-only hard ownership actions for background Gaussians on road pixels."""

from __future__ import annotations

import math
from typing import Any

import torch

from threedgrut.model.road_region import query_ground_z


def _cfg(cfg: Any, name: str, default):
    return cfg.get(name, default) if hasattr(cfg, "get") else getattr(cfg, name, default)


def _tensor_vector(value, size: int, *, device, dtype) -> torch.Tensor:
    if value is None:
        return torch.zeros(size, device=device, dtype=dtype)
    tensor = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if tensor.numel() > size:
        raise ValueError(f"camera coefficient vector has {tensor.numel()} values, expected <= {size}")
    if tensor.numel() < size:
        tensor = torch.nn.functional.pad(tensor, (0, size - tensor.numel()))
    return tensor


def project_pinhole_road_overlap(
    positions_world: torch.Tensor,
    scales_linear: torch.Tensor,
    T_camera_to_world: torch.Tensor,
    intrinsics: dict,
    road_mask: torch.Tensor,
    *,
    footprint_sigma: float = 2.0,
    max_footprint_px: float = 48.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return visible, center-on-road and sampled-footprint-on-road masks.

    Projection mirrors the OpenCV rational/tangential/thin-prism model used
    by the renderer.  All tensors stay on ``positions_world.device``.
    """
    device, dtype = positions_world.device, positions_world.dtype
    n_points = positions_world.shape[0]
    if n_points == 0:
        empty = torch.zeros(0, dtype=torch.bool, device=device)
        return empty, empty, empty

    pose = T_camera_to_world[0] if T_camera_to_world.ndim == 3 else T_camera_to_world
    pose = pose.to(device=device, dtype=dtype)
    points_camera = (positions_world - pose[:3, 3]) @ pose[:3, :3]
    x, y, z = points_camera.unbind(dim=-1)
    safe_z = torch.where(z.abs() < 1e-9, torch.ones_like(z), z)
    xn, yn = x / safe_z, y / safe_z
    r2 = xn.square() + yn.square()
    r4, r6 = r2.square(), r2.square() * r2

    radial = _tensor_vector(intrinsics.get("radial_coeffs"), 6, device=device, dtype=dtype)
    tangential = _tensor_vector(intrinsics.get("tangential_coeffs"), 2, device=device, dtype=dtype)
    thin_prism = _tensor_vector(intrinsics.get("thin_prism_coeffs"), 4, device=device, dtype=dtype)
    numerator = 1 + radial[0] * r2 + radial[1] * r4 + radial[2] * r6
    denominator = 1 + radial[3] * r2 + radial[4] * r4 + radial[5] * r6
    denominator_ok = denominator.abs() >= 1e-12
    distortion = numerator / torch.where(denominator_ok, denominator, torch.ones_like(denominator))
    a1 = 2 * xn * yn
    a2 = r2 + 2 * xn.square()
    a3 = r2 + 2 * yn.square()
    xd = xn * distortion + tangential[0] * a1 + tangential[1] * a2
    yd = yn * distortion + tangential[0] * a3 + tangential[1] * a1
    xd = xd + r2 * (thin_prism[0] + r2 * thin_prism[1])
    yd = yd + r2 * (thin_prism[2] + r2 * thin_prism[3])

    focal = _tensor_vector(intrinsics.get("focal_length"), 2, device=device, dtype=dtype)
    principal = _tensor_vector(intrinsics.get("principal_point"), 2, device=device, dtype=dtype)
    resolution = _tensor_vector(intrinsics.get("resolution"), 2, device=device, dtype=dtype)
    u = focal[0] * xd + principal[0]
    v = focal[1] * yd + principal[1]

    mask = road_mask
    while mask.ndim > 2:
        if mask.shape[0] == 1:
            mask = mask.squeeze(0)
        elif mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        else:
            raise ValueError(f"road_mask must reduce to [H,W], got {tuple(road_mask.shape)}")
    mask = mask.to(device=device)
    height, width = mask.shape
    u = u * (float(width) / resolution[0].clamp_min(1))
    v = v * (float(height) / resolution[1].clamp_min(1))

    finite = torch.isfinite(u) & torch.isfinite(v) & torch.isfinite(distortion)
    max_valid_r2 = intrinsics.get("max_valid_r2")
    if max_valid_r2 is None:
        radial_valid = denominator_ok & (distortion > 0.8) & (distortion < 1.2)
    else:
        max_r2 = torch.as_tensor(max_valid_r2, device=device, dtype=dtype).reshape(-1)[0]
        radial_valid = denominator_ok & (r2 <= max_r2)
    visible = finite & radial_valid & (z > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)

    physical_radius = scales_linear.max(dim=-1).values
    radius_px = footprint_sigma * physical_radius * focal.abs().max() / safe_z.abs().clamp_min(1e-6)
    radius_px = radius_px.clamp(min=1.0, max=max_footprint_px)
    directions = [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0),
                  (-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
    offsets = torch.tensor(
        [[0.0, 0.0]]
        + [[fraction * dx, fraction * dy] for fraction in (0.25, 0.5, 0.75, 1.0) for dx, dy in directions],
        device=device,
        dtype=dtype,
    )
    sample_u = torch.round(u[:, None] + radius_px[:, None] * offsets[None, :, 0]).long()
    sample_v = torch.round(v[:, None] + radius_px[:, None] * offsets[None, :, 1]).long()
    in_bounds = (sample_u >= 0) & (sample_u < width) & (sample_v >= 0) & (sample_v < height)
    sampled = torch.zeros_like(in_bounds)
    if in_bounds.any():
        sampled[in_bounds] = mask[sample_v[in_bounds], sample_u[in_bounds]] > 0.5
    sampled &= visible[:, None]
    return visible, sampled[:, 0], sampled.any(dim=1)


@torch.no_grad()
def apply_bg_road_exclusion(bg_layer, height_field: dict, batch, cfg) -> dict[str, int]:
    """Clamp center hits into the dead pool and shrink footprint-only hits."""
    positions = bg_layer.positions.detach()
    n_points = positions.shape[0]
    stats = {
        "n_visible_alive": 0,
        "n_center_slab_hit": 0,
        "n_center_proj_hit": 0,
        "n_footprint_shrunk": 0,
        "n_recycled": 0,
    }
    if n_points == 0 or batch is None:
        return stats
    image_infos = getattr(batch, "image_infos", None)
    road_mask = image_infos.get("road_mask") if isinstance(image_infos, dict) else None
    intrinsics = getattr(batch, "intrinsics_OpenCVPinholeCameraModelParameters", None)
    pose = getattr(batch, "T_to_world", None)
    if road_mask is None or intrinsics is None or pose is None:
        return stats

    z_band = float(_cfg(cfg, "z_band", 0.4))
    projection_max_height = float(_cfg(cfg, "projection_max_height", 1.0))
    chunk_size = int(_cfg(cfg, "chunk_size", 200_000))
    opacity_threshold = float(_cfg(cfg, "opacity_threshold", 0.005))
    hard_mask = torch.zeros(n_points, dtype=torch.bool, device=positions.device)
    shrink_mask = torch.zeros_like(hard_mask)
    scales_linear = bg_layer.get_scale().detach()
    density_linear = bg_layer.get_density().detach().reshape(-1)

    for start in range(0, n_points, chunk_size):
        stop = min(start + chunk_size, n_points)
        pos = positions[start:stop]
        ground_z, supported = query_ground_z(pos[:, :2], height_field)
        height_above = pos[:, 2] - ground_z
        slab = supported & (height_above.abs() < z_band)
        visible, center_road, footprint_road = project_pinhole_road_overlap(
            pos,
            scales_linear[start:stop],
            pose,
            intrinsics,
            road_mask,
            footprint_sigma=float(_cfg(cfg, "footprint_sigma", 2.0)),
            max_footprint_px=float(_cfg(cfg, "max_footprint_px", 48.0)),
        )
        alive = density_linear[start:stop] > opacity_threshold
        height_safe = supported & (height_above >= -z_band) & (height_above <= projection_max_height)
        center_projection = center_road & height_safe
        hard = alive & (slab | center_projection)
        footprint_only = alive & footprint_road & ~center_road & height_safe & ~slab
        hard_mask[start:stop] = hard
        shrink_mask[start:stop] = footprint_only
        stats["n_visible_alive"] += int((visible & alive).sum().item())
        stats["n_center_slab_hit"] += int((alive & slab).sum().item())
        stats["n_center_proj_hit"] += int((alive & center_projection).sum().item())

    shrink_factor = float(_cfg(cfg, "footprint_shrink_factor", 0.5))
    if not 0.0 < shrink_factor < 1.0:
        raise ValueError("footprint_shrink_factor must be in (0,1)")
    if hard_mask.any():
        bg_layer.density[hard_mask] = float(_cfg(cfg, "dead_density_raw", -50.0))
    if shrink_mask.any():
        bg_layer.scale[shrink_mask, :2] += math.log(shrink_factor)
    stats["n_footprint_shrunk"] = int(shrink_mask.sum().item())
    stats["n_recycled"] = int(hard_mask.sum().item())
    return stats
