# SPDX-License-Identifier: Apache-2.0
"""Road layer initialization: BEV-grid + LiDAR-Z KNN.

Stage 3 T3.3.b — produce a flat thin-disc Gaussian layer that hugs the road
surface inferred from semantically-filtered LiDAR points. Used by trainer
init_model when 'road' is in layers.enabled; flows through
LayeredGaussians.init_layer_from_points("road", ...).

Algorithm (mirrors Reconstruction-Studio surface init thinking but avoids
PyTorch3D dependency by using torch.cdist):

  1. Compute BEV bounding box from ego trajectory ± cut_range (typically 30m).
  2. Tile a 2D grid at `resolution` (default 5 cm).
  3. For each grid cell, look up the nearest road LiDAR point in XY (cdist
     argmin) and lift Z to that point's Z. This handles ramps/banked roads.
  4. Subsample to max_n if the grid is too dense.
  5. Default rotations to identity quat; scales to log((0.1, 0.1, 0.001)) so
     the thin disc is enforced from step 0 (paired with road perturb_mask
     D1 to prevent Z-noise during MCMC).
"""
from __future__ import annotations

from typing import Tuple

import torch


def init_road_layer(
    road_points: torch.Tensor,
    ego_trajectory: torch.Tensor,
    cut_range: float = 30.0,
    resolution: float = 0.05,
    max_n: int = 200_000,
    knn_k: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate per-particle init tensors for the road layer.

    Args:
        road_points:    [M, 3] in world frame, semantically filtered as road
                        (T3.2.b get_road_lidar_points output).
        ego_trajectory: [F, 3] world-frame ego positions; used to bound BEV.
        cut_range:      meters padded around ego XY extent for BEV bbox.
        resolution:     BEV grid cell size in meters.
        max_n:          cap on returned particle count.
        knn_k:          # of nearest road points whose Z is medianed per grid
                        cell. knn_k=1 (default) = legacy nearest-single-point
                        (byte-identical to pre-E3.2.5). knn_k=5 (E3.2.5① on)
                        takes a local median to reject LiDAR outlier spikes,
                        approaching recon-studio's 8mm dense-disc init.

    Returns:
        positions:  [N, 3]  world frame, Z snapped to nearest road LiDAR Z
        rotations:  [N, 4]  identity quat (wxyz)
        scales:     [N, 3]  log-space, ~log(0.1, 0.1, 0.001)
        densities:  [N, 1]  log-space, 0.0
        colors:     [N, 3]  neutral gray in [0, 1] (TODO: project from image
                            in T3.3.c if PSNR demands; default flat works for
                            v1+road smoke).
    """
    dtype = torch.float32
    device = road_points.device if road_points.numel() > 0 else ego_trajectory.device

    if road_points.numel() == 0:
        # empty fallback — keep shapes consistent with non-empty path
        return (
            torch.zeros(0, 3, dtype=dtype, device=device),
            torch.zeros(0, 4, dtype=dtype, device=device),
            torch.zeros(0, 3, dtype=dtype, device=device),
            torch.zeros(0, 1, dtype=dtype, device=device),
            torch.zeros(0, 3, dtype=dtype, device=device),
        )

    # 1. BEV bounding box
    xy_min = ego_trajectory[:, :2].min(0).values - cut_range
    xy_max = ego_trajectory[:, :2].max(0).values + cut_range

    # 2. Tile BEV grid
    xs = torch.arange(xy_min[0].item(), xy_max[0].item(), resolution,
                      dtype=dtype, device=device)
    ys = torch.arange(xy_min[1].item(), xy_max[1].item(), resolution,
                      dtype=dtype, device=device)
    if xs.numel() == 0 or ys.numel() == 0:
        # ego trajectory too short / cut_range too small → empty grid
        return (
            torch.zeros(0, 3, dtype=dtype, device=device),
            torch.zeros(0, 4, dtype=dtype, device=device),
            torch.zeros(0, 3, dtype=dtype, device=device),
            torch.zeros(0, 1, dtype=dtype, device=device),
            torch.zeros(0, 3, dtype=dtype, device=device),
        )
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    grid_xy = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # [M', 2]

    # 4. Subsample BEFORE cdist to keep memory bounded for huge grids.
    if grid_xy.shape[0] > max_n:
        idx = torch.randperm(grid_xy.shape[0], device=device)[:max_n]
        grid_xy = grid_xy[idx]

    # 3. KNN-Z. Prefer scipy cKDTree (O(N log N), no NxM matrix); fall back to
    #    torch.cdist for small inputs / environments without scipy (Mac unit tests).
    #    Earlier torch.cdist(grid[200K], road_pts[629K]) was 500 GB host RAM →
    #    OOM kill on A800 (exit 137); cKDTree is the production path.
    #    E3.2.5①: knn_k>1 medians Z over the k nearest road points to reject
    #    LiDAR outlier spikes (recon-studio dense-disc init); knn_k=1 is the
    #    legacy nearest-single-point path (byte-identical to pre-E3.2.5).
    k = max(1, min(int(knn_k), road_points.shape[0]))
    try:
        from scipy.spatial import cKDTree
        rp_cpu = road_points[:, :2].detach().cpu().numpy()
        tree = cKDTree(rp_cpu)
        grid_cpu = grid_xy.detach().cpu().numpy()
        _, nn_idx = tree.query(grid_cpu, k=k)  # [M] if k==1 else [M, k]
        nn_idx_t = torch.from_numpy(nn_idx).to(device=road_points.device, dtype=torch.long)
    except ImportError:
        dists = torch.cdist(grid_xy.unsqueeze(0), road_points[:, :2].unsqueeze(0))[0]
        _, nn_idx_t = torch.topk(dists, k, largest=False, dim=1)  # [M, k]
    if k == 1:
        grid_z = road_points[nn_idx_t.reshape(-1), 2]  # legacy nearest single
    else:
        nn_idx_t = nn_idx_t.reshape(grid_xy.shape[0], k)
        grid_z = torch.median(road_points[nn_idx_t, 2], dim=1).values
    positions = torch.cat([grid_xy, grid_z.unsqueeze(-1)], dim=-1)  # [N, 3]

    N = positions.shape[0]

    # 5. Defaults
    rotations = torch.zeros(N, 4, dtype=dtype, device=device)
    rotations[:, 0] = 1.0  # identity quat wxyz
    scale_prior = torch.tensor([0.1, 0.1, 0.001], dtype=dtype, device=device)
    scales = torch.log(scale_prior).expand(N, 3).contiguous()
    densities = torch.zeros(N, 1, dtype=dtype, device=device)  # log-space ≈ 0
    colors = torch.full((N, 3), 0.5, dtype=dtype, device=device)  # neutral gray

    return positions, rotations, scales, densities, colors
