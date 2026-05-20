# SPDX-License-Identifier: Apache-2.0
"""v2 correction modules — Stage 5 (sky envmap) + Stage 6 (per-camera exposure).

Public exports:
    SkyEnvmapBase    — abstract base; forward(viewdirs) -> rgb_sky.
    SkyEnvmapMLP     — fallback MLP backend (no external deps).
    SkyEnvmapCubemap — default cubemap backend (requires nvdiffrast.torch).
    ExposureModel    — per-camera affine exp(a)*x + b (Stage 6, T6.1).

Imports are deliberately lazy so a missing nvdiffrast (cubemap backend) does
not block importing the package or the MLP backend.
"""
from threedgrut.correction.sky_envmap import (
    SkyEnvmapBase,
    SkyEnvmapMLP,
    SkyEnvmapCubemap,
)
from threedgrut.correction.exposure import ExposureModel

__all__ = [
    "SkyEnvmapBase",
    "SkyEnvmapMLP",
    "SkyEnvmapCubemap",
    "ExposureModel",
]
