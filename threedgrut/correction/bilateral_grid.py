# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""T9.1 / V3-P1.a: Per-camera 3D bilateral grid for color correction.

Replaces the old affine ``exp(a)*img + b`` ExposureModel (`exposure.py`) used
by trainer.py — both share the same ``forward(camera_idx, image) -> image``
signature so the trainer wiring is unchanged. The bilateral grid is more
expressive: 12 parameters per camera (full 3x4 color-affine matrix supporting
cross-channel mixing), vs ExposureModel's 2 scalars per camera (per-channel
gain + bias applied identically to R, G, B).

NuRec parsed_config uses 1x1x1 grid (no spatial / guidance variation) — the
fast-path branch below avoids ``F.grid_sample`` entirely and just indexes the
single voxel per camera. Higher resolutions (e.g. 8x8x4) are supported via the
fallback ``grid_sample`` path; identical math, slower.

Ported (with simplifications) from nerfstudio's ``lib_bilagrid.py`` (Apache 2.0),
which is itself derived from Wang et al.'s "Bilateral Guided Radiance Field
Processing" (https://bilarfpro.github.io/). We drop the 4-D CP-decomposed
variant (BilateralGridCP4D) and its tensorly dependency since 3dgrut only needs
the per-camera 3-D form.

Identity initialization: each voxel = [[1,0,0,0],[0,1,0,0],[0,0,1,0]] (3x4
identity affine) → output == input until the optimizer moves the grids. Same
boundary as the old ExposureModel zero-init.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def color_affine_transform(affine_mats: Tensor, rgb: Tensor) -> Tensor:
    """Apply (3,4) color-affine ``out = A @ rgb + b``.

    Args:
        affine_mats: ``(..., 3, 4)`` affine matrices.
        rgb: ``(..., 3)`` RGB values matching the leading shape.

    Returns:
        Transformed ``(..., 3)`` colors.
    """
    return (
        torch.matmul(affine_mats[..., :3], rgb.unsqueeze(-1)).squeeze(-1)
        + affine_mats[..., 3]
    )


def _identity_affine_3x4(device=None, dtype=torch.float32) -> Tensor:
    """[[1,0,0,0],[0,1,0,0],[0,0,1,0]] — identity in color-affine space."""
    return torch.tensor(
        [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]],
        dtype=dtype, device=device,
    )


def total_variation_loss(x: Tensor) -> Tensor:
    """TV regularization on multi-dim tensors (T9.2 will use this as L2 reg).

    Args:
        x: ``(B, C, ...)`` where B is batch (= num cameras) and C is channel.
    """
    batch_size = x.shape[0]
    tv = torch.zeros((), device=x.device, dtype=x.dtype)
    for i in range(2, len(x.shape)):
        n_res = x.shape[i]
        if n_res < 2:
            continue  # 1x1x1 grid has no neighbors to diff
        idx1 = torch.arange(1, n_res, device=x.device)
        idx2 = torch.arange(0, n_res - 1, device=x.device)
        x1 = x.index_select(i, idx1)
        x2 = x.index_select(i, idx2)
        count = max(
            torch.prod(torch.tensor(x1.size()[1:]).float()).item(), 1.0,
        )
        tv = tv + torch.pow(x1 - x2, 2).sum() / count
    return tv / batch_size


class BilateralGrid(nn.Module):
    """Per-camera 3-D bilateral grid (T9.1 / V3-P1.a replacement for ExposureModel).

    Holds ``num_camera`` grids of shape ``(grid_W, grid_H, grid_L)`` where each
    voxel is a 12-vector (= 3x4 color-affine matrix).

    Args:
        num_camera: number of distinct cameras (one grid per camera).
        grid_X: spatial width L_x. Default 1 (NuRec 1x1x1).
        grid_Y: spatial height L_y. Default 1.
        grid_W: guidance dimension L_z (grayscale axis). Default 1.

    Parameters:
        grids: ``(num_camera, 12, grid_W, grid_Y, grid_X)`` — identity-init.

    The compat ``forward(idx: int, image: Tensor) -> Tensor`` mirrors the old
    ExposureModel signature: takes a per-camera scalar index and a single-image
    tensor, returns the corrected image clamped to [0, 1].
    """

    # BT601 RGB-to-gray weights (used for grid guidance axis Z).
    _RGB2GRAY_BT601 = (0.299, 0.587, 0.114)

    def __init__(
        self,
        num_camera: int,
        grid_X: int = 1,
        grid_Y: int = 1,
        grid_W: int = 1,
    ) -> None:
        super().__init__()
        if num_camera < 1:
            raise ValueError(f"num_camera must be >= 1, got {num_camera}")
        if grid_X < 1 or grid_Y < 1 or grid_W < 1:
            raise ValueError(
                f"grid dims must all be >= 1, got "
                f"({grid_X}, {grid_Y}, {grid_W})"
            )

        self.num_camera = int(num_camera)
        self.grid_X = int(grid_X)
        self.grid_Y = int(grid_Y)
        self.grid_W = int(grid_W)

        # Identity affine init, tiled across (W, Y, X) and stacked across cameras.
        identity_3x4 = _identity_affine_3x4().reshape(12)  # (12,)
        grid = identity_3x4.view(12, 1, 1, 1).expand(12, grid_W, grid_Y, grid_X)
        grid = grid.unsqueeze(0).expand(num_camera, -1, -1, -1, -1).contiguous()
        self.grids = nn.Parameter(grid.clone())  # (N, 12, L_z, L_y, L_x)

        # Buffer for grayscale guidance conversion (non-learnable).
        self.register_buffer(
            "_rgb2gray_w",
            torch.tensor(list(self._RGB2GRAY_BT601), dtype=torch.float32)
            .reshape(1, 3),
        )

    # --- Compat API (matches old ExposureModel) -------------------------------

    def forward(self, idx: int, image: Tensor) -> Tensor:
        """Apply this camera's color affine to ``image``.

        Args:
            idx: 0-based camera index in ``[0, num_camera)``.
            image: RGB tensor of any shape with last channel = 3.

        Returns:
            Same-shape tensor clamped to ``[0, 1]`` (matches ExposureModel).
        """
        if not (0 <= idx < self.num_camera):
            raise IndexError(
                f"camera idx {idx} out of range [0, {self.num_camera})"
            )

        # Fast path for 1x1x1 grid (NuRec default): grid output is spatially
        # uniform + guidance-independent → just take the single voxel and
        # apply as a plain per-camera affine. Math identical to grid_sample
        # at any (xy, gray) query.
        if self.grid_X == 1 and self.grid_Y == 1 and self.grid_W == 1:
            affine = self.grids[idx, :, 0, 0, 0].reshape(3, 4)
            out = color_affine_transform(affine, image)
            return out.clamp(0.0, 1.0)

        # General path: sample per-pixel grid via F.grid_sample at the
        # pixel's (x, y, gray) coordinates. Image is assumed to be either
        # ``(H, W, 3)`` or ``(B, H, W, 3)``; we materialise xy meshgrid in
        # [-1, 1] and gray = BT601(rgb) * 2 - 1 in [-1, 1].
        if image.dim() == 3:
            image_b = image.unsqueeze(0)
        elif image.dim() == 4:
            image_b = image
        else:
            raise ValueError(
                f"image must be (H, W, 3) or (B, H, W, 3); got {image.shape}"
            )

        B, H, W, C = image_b.shape
        if C != 3:
            raise ValueError(f"image last dim must be 3 (RGB), got {C}")

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=image_b.device, dtype=image_b.dtype),
            torch.linspace(-1.0, 1.0, W, device=image_b.device, dtype=image_b.dtype),
            indexing="ij",
        )
        # Guidance axis: BT601 grayscale, rescaled to [-1, 1].
        gray = (image_b @ self._rgb2gray_w.T.to(image_b.dtype)) * 2.0 - 1.0  # (B, H, W, 1)
        gray = gray.squeeze(-1)  # (B, H, W)

        grid_xyz = torch.stack(
            [xx.unsqueeze(0).expand(B, H, W), yy.unsqueeze(0).expand(B, H, W), gray],
            dim=-1,
        )  # (B, H, W, 3)
        # F.grid_sample expects 5-D input + (B, D, H, W, 3) sample coords.
        grid_xyz = grid_xyz.unsqueeze(1)  # (B, 1, H, W, 3)

        grids_idx = self.grids[idx : idx + 1]  # (1, 12, L_z, L_y, L_x)
        grids_idx = grids_idx.expand(B, -1, -1, -1, -1)

        sampled = F.grid_sample(
            grids_idx, grid_xyz,
            mode="bilinear", align_corners=True, padding_mode="border",
        )  # (B, 12, 1, H, W)
        affine_mats = sampled.squeeze(2).permute(0, 2, 3, 1)  # (B, H, W, 12)
        affine_mats = affine_mats.reshape(B, H, W, 3, 4)

        out = color_affine_transform(affine_mats, image_b)
        if image.dim() == 3:
            out = out.squeeze(0)
        return out.clamp(0.0, 1.0)

    def tv_loss(self) -> Tensor:
        """Spatial + guidance smoothness TV. Returns 0 for 1x1x1 grids."""
        return total_variation_loss(self.grids)
