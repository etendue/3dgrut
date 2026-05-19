# SPDX-License-Identifier: Apache-2.0
"""Dynamic mask projection: cuboid → 2D AABB pixel mask (T4.4, U2 D5).

Why not sseg-based dyn mask: sseg covers vehicles outside our tracked set
(traffic cones, untracked pedestrians), so the dynamic_rigids layer would be
asked to explain pixels it cannot own. Cuboid projection ties dynamic loss
exactly to tracked actors.

Why AABB (not exact convex hull): cuboid projects to a 2D convex octagon;
AABB overestimates ≈10–15% in area but lands in 5 lines of vectorised
PyTorch (no scanline edge traversal needed). D5: upgrade to convex hull
only if Stage 7 KPI delta < +0.3 dB attributable to mask slack.

Why pure PyTorch (no OpenCV / nvdiffrast): keeps the dependency surface
minimal, runs on GPU without host copy, easy to differentiate later.
"""
from __future__ import annotations

import torch

# 8 cuboid corner sign template: ±1 along each of (x, y, z). Generated once
# at module load.
_CORNER_SIGNS = torch.tensor(
    [[(i & 1), ((i >> 1) & 1), ((i >> 2) & 1)] for i in range(8)],
    dtype=torch.float32,
) * 2.0 - 1.0  # [8, 3]; values in {-1, +1}


def project_cuboids_to_mask(
    tracks_poses: torch.Tensor,    # [T, 4, 4] active tracks at this frame
    tracks_size: torch.Tensor,     # [T, 3] full extent (NOT half)
    K: torch.Tensor,               # [3, 3] camera intrinsics
    T_world2cam: torch.Tensor,     # [4, 4]
    H: int,
    W: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Project each cuboid → 8 image-space corners → 2D AABB → fill mask.

    Args:
        tracks_poses: ``[T, 4, 4]`` object→world for T active tracks at
            this frame. T may be 0 (no active tracks → empty mask).
        tracks_size:  ``[T, 3]`` cuboid full extent in meters.
        K:            ``[3, 3]`` pinhole intrinsics (fx, fy, cx, cy).
        T_world2cam:  ``[4, 4]`` world → camera SE(3).
        H, W:         output mask size in pixels.
        device:       output device.

    Returns:
        bool ``[H, W]`` mask; True wherever any cuboid AABB covers.
    """
    T = int(tracks_poses.shape[0])
    mask = torch.zeros(H, W, dtype=torch.bool, device=device)
    if T == 0:
        return mask

    poses = tracks_poses.to(device=device, dtype=torch.float32)
    sizes = tracks_size.to(device=device, dtype=torch.float32)
    K = K.to(device=device, dtype=torch.float32)
    T_w2c = T_world2cam.to(device=device, dtype=torch.float32)
    signs = _CORNER_SIGNS.to(device=device)                        # [8, 3]

    # 1. 8 cuboid corners in object-local frame: half-extent × ±sign
    corners_local = signs.unsqueeze(0) * (sizes.unsqueeze(1) * 0.5)  # [T, 8, 3]
    ones = torch.ones(T, 8, 1, dtype=torch.float32, device=device)
    corners_h = torch.cat([corners_local, ones], dim=-1)             # [T, 8, 4]

    # 2. local → world (per-track pose) → camera (single T_w2c)
    world = torch.einsum("tij,tkj->tki", poses, corners_h)           # [T, 8, 4]
    cam = torch.einsum("ij,tkj->tki", T_w2c, world)                  # [T, 8, 4]

    # 3. perspective project; clamp z to avoid div-by-zero on points behind camera
    z = cam[..., 2].clamp(min=0.1)                                    # [T, 8]
    u = K[0, 0] * cam[..., 0] / z + K[0, 2]                           # [T, 8]
    v = K[1, 1] * cam[..., 1] / z + K[1, 2]                           # [T, 8]

    # 4. 2D AABB per track, clipped to image bounds
    u_min = u.min(dim=-1).values.clamp(0, W - 1).long()               # [T]
    u_max = u.max(dim=-1).values.clamp(0, W - 1).long()
    v_min = v.min(dim=-1).values.clamp(0, H - 1).long()
    v_max = v.max(dim=-1).values.clamp(0, H - 1).long()

    # 5. Fill mask. Loop over T (typically < 30 active actors per frame; the
    # python overhead is negligible vs the einsum cost above). Slice
    # assignment is in-place; Boolean OR semantics emerge naturally because
    # writes set True over existing False.
    for i in range(T):
        # +1 on max to include boundary pixel; clamp guarantees v_max ≥ v_min.
        mask[v_min[i]:v_max[i] + 1, u_min[i]:u_max[i] + 1] = True
    return mask
