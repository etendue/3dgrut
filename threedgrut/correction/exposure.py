# SPDX-License-Identifier: Apache-2.0
"""Per-camera exposure model for v2 trainer (Stage 6, T6.1).

Affine ``out = exp(a) * img + b`` per camera. Each camera owns one ``(a, b)``
pair stored as nn.Parameter tensors of shape ``(num_camera, 1)``; zero init
makes the module identity on construction. The trainer applies it after sky
blending and before the loss.

Adapted (almost verbatim) from Reconstruction-Studio's
``models/luxury/exposure.py``. We drop the original ``num_camera == 1`` early
return because (a) NCore multi-camera training always has ≥ 2 cameras and (b)
even for single-camera fallback the zero-init forward is identical to ``img``
already.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class ExposureModel(nn.Module):
    """Learnable per-camera affine correction ``exp(a) * img + b``.

    Args:
        num_camera: number of distinct cameras in the dataset. Must be ≥ 1.
            Cameras are addressed by 0-based integer index (``Batch.camera_idx``).

    Parameters:
        exposure_a: ``[num_camera, 1]``, log-gain. Zero init → unit gain.
        exposure_b: ``[num_camera, 1]``, bias. Zero init → no offset.
    """

    def __init__(self, num_camera: int) -> None:
        super().__init__()
        if num_camera < 1:
            raise ValueError(f"num_camera must be >= 1, got {num_camera}")
        self.num_camera = int(num_camera)
        self.exposure_a = nn.Parameter(torch.zeros(num_camera, 1, dtype=torch.float32))
        self.exposure_b = nn.Parameter(torch.zeros(num_camera, 1, dtype=torch.float32))

    def forward(self, idx: int, image: Tensor) -> Tensor:
        """Apply this camera's affine to ``image``.

        Args:
            idx: 0-based camera index. Must be in ``[0, num_camera)``.
            image: RGB tensor of any shape with last channel = 3.

        Returns:
            Same-shape tensor clamped to [0, 1].
        """
        if not (0 <= idx < self.num_camera):
            raise IndexError(
                f"camera idx {idx} out of range [0, {self.num_camera})"
            )
        a = self.exposure_a[idx]   # [1]
        b = self.exposure_b[idx]   # [1]
        return (torch.exp(a) * image + b).clamp(0.0, 1.0)
