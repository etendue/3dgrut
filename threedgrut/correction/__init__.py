# SPDX-License-Identifier: Apache-2.0
"""v2/v3 correction modules — Stage 5 (sky envmap) + Stage 6/9 (color exposure).

Public exports:
    SkyEnvmapBase    — abstract base; forward(viewdirs) -> rgb_sky.
    SkyEnvmapMLP     — fallback MLP backend (no external deps).
    SkyEnvmapCubemap — default cubemap backend (requires nvdiffrast.torch).
    ExposureModel    — v2 baseline per-camera affine exp(a)*x + b (Stage 6,
                        T6.1). Kept for legacy ckpt resume; new training uses
                        BilateralGrid by default (T9.1 / V3-P1.a).
    BilateralGrid    — v3 per-camera 3-D bilateral grid color correction
                        (Stage 9 / V3-P1.a, T9.1). Default 1x1x1 = 12-param
                        per-camera color affine; identity-init = identity
                        transform.

Imports are deliberately lazy so a missing nvdiffrast (cubemap backend) does
not block importing the package or the MLP backend.
"""

from threedgrut.correction.bilateral_grid import BilateralGrid
from threedgrut.correction.exposure import ExposureModel
from threedgrut.correction.sky_envmap import (
    SkyEnvmapBase,
    SkyEnvmapCubemap,
    SkyEnvmapMLP,
)

__all__ = [
    "SkyEnvmapBase",
    "SkyEnvmapMLP",
    "SkyEnvmapCubemap",
    "ExposureModel",
    "BilateralGrid",
]
