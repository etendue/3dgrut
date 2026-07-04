# SPDX-License-Identifier: Apache-2.0
"""E1.1 plane-induced warp — pseudo-GT for novel-view lane metrics.

At a novel (extrapolated) pose there is no pixel GT. But lane paint lies on
the road surface: for road-plane content the homography induced by that
surface is exact. So we cast each novel pixel's ray onto the road height
field, project the hit point back into the ORIGINAL camera, and sample the
original GT image / lane mask there. ``compute_lane_metrics`` then runs on
(novel render, warped GT, warped lane mask) restricted to warp-valid pixels.

Known approximations (documented in v4_plan E1.1):
- shutter-start pose only (rolling shutter ignored at warp level);
- bilinear resampling slightly smooths the warped GT → warped lane metrics
  are comparable ACROSS models under the same warp version, but not against
  interpolated-view ``mean_lane_*`` absolute values.

FTheta forward projection reuses the exact polynomial math of
``threedgrut/layers/dynamic_mask.py`` (NOT the viser projector — BUG-1
isolation); ray directions come from the dataset's per-pixel ``rays_dir``
cache (FTheta-correct by construction), so no inverse-FTheta is needed here.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from threedgrut.layers.dynamic_mask import (
    _horner_ascending_torch,
    _normalize_ftheta_params,
)
from threedgrut.model.road_region import query_ground_z


def ftheta_project_points(
    points_cam: torch.Tensor,
    ftheta_params: Dict[str, object],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project camera-frame points through the FTheta polynomial.

    Args:
        points_cam: ``[N, 3]`` points in OpenCV camera frame (x right,
            y down, z forward).
        ftheta_params: dict with ``angle_to_pixeldist_poly`` (ascending) and
            ``principal_point`` (cx, cy); numpy or torch accepted.

    Returns:
        uv:    ``[N, 2]`` float pixel coordinates.
        valid: ``[N]`` bool — True iff the point is in front of the camera
               (z > 0). Same formula as ``_corners_to_pixels_ftheta`` in
               dynamic_mask.py, point-level instead of cuboid-corner-level.
    """
    params = _normalize_ftheta_params(ftheta_params, points_cam.device)
    x, y, z = points_cam[..., 0], points_cam[..., 1], points_cam[..., 2]
    r_xy = torch.sqrt(x * x + y * y)
    angle = torch.atan2(r_xy, z)
    r_pix = _horner_ascending_torch(params["angle_to_pixeldist_poly"], angle)
    safe_r = torch.where(r_xy < 1e-9, torch.ones_like(r_xy), r_xy)
    cx = params["principal_point"][0]
    cy = params["principal_point"][1]
    u = cx + r_pix * x / safe_r
    v = cy + r_pix * y / safe_r
    return torch.stack([u, v], dim=-1), z > 0


def build_plane_warp(
    rays_dir_cam: torch.Tensor,
    c2w_novel: torch.Tensor,
    c2w_orig: torch.Tensor,
    ftheta_params: Dict[str, object],
    height_field: Optional[Dict] = None,
    z0_fallback: Optional[float] = None,
    n_iters: int = 2,
    t_max: float = 120.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a grid_sample warp: novel pixel → road plane → original pixel.

    Args:
        rays_dir_cam: ``[H, W, 3]`` per-pixel ray directions in CAMERA frame
            (dataset ``rays_dir`` cache; same intrinsics for both poses).
        c2w_novel / c2w_orig: ``[4, 4]`` camera-to-world (threedgrut
            convention: col0 right, col1 down, col2 forward).
        ftheta_params: forward-projection intrinsics for the ORIGINAL camera.
        height_field: ``road_region.build_road_height_field`` output. When
            None, ``z0_fallback`` must be given (constant plane z = z0).
        z0_fallback: constant ground height when no height field is usable.
        n_iters: fixed-point iterations on the height field (plane is locally
            flat; 2 suffices at ~1 m cells).
        t_max: max ray length in meters (drops near-horizon mega-ranges).

    Returns:
        grid:  ``[1, H, W, 2]`` normalized to [-1, 1] for
               ``F.grid_sample(..., align_corners=True)``.
        valid: ``[H, W]`` bool — ray hits the (occupied) ground AND the hit
               projects in front of the original camera AND inside the image.
    """
    if height_field is None and z0_fallback is None:
        raise ValueError("build_plane_warp: need height_field or z0_fallback")

    H, W = int(rays_dir_cam.shape[0]), int(rays_dir_cam.shape[1])
    device = rays_dir_cam.device
    dtype = torch.float32

    rays = rays_dir_cam.to(dtype)
    R_novel = c2w_novel[:3, :3].to(device=device, dtype=dtype)
    o_w = c2w_novel[:3, 3].to(device=device, dtype=dtype)

    # novel pixel rays → world
    d_w = torch.einsum("ij,hwj->hwi", R_novel, rays)  # [H, W, 3]
    d_flat = d_w.reshape(-1, 3)  # [N, 3]
    N = d_flat.shape[0]

    down = d_flat[:, 2] < -1e-6

    # --- ray ↔ ground intersection -------------------------------------
    if height_field is not None:
        # init ground guess at the camera's own footprint
        z0_cam, z0_valid = query_ground_z(o_w[:2].reshape(1, 2), height_field)
        if bool(z0_valid[0]):
            z_g = torch.full((N,), float(z0_cam[0]), dtype=dtype, device=device)
        elif z0_fallback is not None:
            z_g = torch.full((N,), float(z0_fallback), dtype=dtype, device=device)
        else:
            # fall back to the field's median occupied height
            occ = height_field["occupied"]
            if occ.numel() and bool(occ.any()):
                z_g = torch.full(
                    (N,),
                    float(height_field["grid_z"][occ].median()),
                    dtype=dtype,
                    device=device,
                )
            else:
                grid = torch.zeros(1, H, W, 2, dtype=dtype, device=device)
                return grid, torch.zeros(H, W, dtype=torch.bool, device=device)

        hit_valid = torch.zeros(N, dtype=torch.bool, device=device)
        t = torch.zeros(N, dtype=dtype, device=device)
        denom = torch.where(down, d_flat[:, 2], torch.full_like(d_flat[:, 2], -1.0))
        for _ in range(max(1, n_iters)):
            t = (z_g - o_w[2]) / denom
            xy = o_w[:2] + t.unsqueeze(-1) * d_flat[:, :2]  # [N, 2]
            z_q, q_valid = query_ground_z(xy, height_field)
            hit_valid = q_valid
            z_g = torch.where(q_valid, z_q.to(dtype), z_g)
        t = (z_g - o_w[2]) / denom
        valid = down & hit_valid & (t > 0) & (t < t_max)
    else:
        denom = torch.where(down, d_flat[:, 2], torch.full_like(d_flat[:, 2], -1.0))
        t = (float(z0_fallback) - o_w[2]) / denom
        valid = down & (t > 0) & (t < t_max)

    p_w = o_w + t.unsqueeze(-1) * d_flat  # [N, 3] world hits

    # --- world → original camera → pixels --------------------------------
    w2c_orig = torch.linalg.inv(c2w_orig.to(device=device, dtype=dtype))
    p_h = torch.cat([p_w, torch.ones(N, 1, dtype=dtype, device=device)], dim=-1)
    p_cam = (w2c_orig @ p_h.T).T[:, :3]  # [N, 3]
    uv, front = ftheta_project_points(p_cam, ftheta_params)
    valid = valid & front
    u, v = uv[:, 0], uv[:, 1]
    # Pixel-center sampling semantics: anything within half a pixel of the
    # border is a legitimate sample (guards float jitter at exactly H-1/W-1);
    # clamp to the border for grid_sample.
    valid = valid & (u >= -0.5) & (u <= W - 0.5) & (v >= -0.5) & (v <= H - 0.5)
    u = u.clamp(0, W - 1)
    v = v.clamp(0, H - 1)

    gx = 2.0 * u / max(W - 1, 1) - 1.0
    gy = 2.0 * v / max(H - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1).reshape(1, H, W, 2)
    # park invalid samples at a harmless in-range location; callers must
    # gate on `valid` anyway (warp_image zeroes them).
    grid = torch.where(valid.reshape(1, H, W, 1), grid, torch.zeros_like(grid))
    return grid, valid.reshape(H, W)


def warp_image(
    img_hwc: torch.Tensor,
    grid: torch.Tensor,
    valid: torch.Tensor,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Sample ``img_hwc`` (``[H, W, C]``) at ``grid``; zero invalid pixels.

    Use ``mode="nearest"`` for label maps (lane sseg) to avoid label mixing.
    """
    if img_hwc.dim() != 3:
        raise ValueError(f"warp_image expects [H, W, C]; got {tuple(img_hwc.shape)}")
    src = img_hwc.permute(2, 0, 1).unsqueeze(0).float()  # [1, C, H, W]
    out = F.grid_sample(
        src,
        grid.to(src.device, src.dtype),
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )
    out = out.squeeze(0).permute(1, 2, 0)  # [H, W, C]
    out = out * valid.to(out.dtype).unsqueeze(-1)
    return out.to(img_hwc.dtype) if img_hwc.dtype != out.dtype else out
