# SPDX-License-Identifier: Apache-2.0
"""Sky envmap backends for v2 LayeredGaussians sky layer (Stage 5).

Two backends share the :class:`SkyEnvmapBase` interface ``forward(viewdirs) -> rgb_sky``:

    SkyEnvmapCubemap  default; learnable 6-face cubemap, sampled via
                      ``nvdiffrast.torch.texture(..., boundary_mode="cube")``.
                      Direct port of drivestudio EnvLight
                      (drivestudio/models/modules.py:174-208).

    SkyEnvmapMLP      fallback (no external deps); sinusoidally-encoded
                      direction → 3-layer MLP → sigmoid. Loosely adapted from
                      drivestudio SkyModel but with the appearance-embedding
                      branch removed (Stage 6 ExposureModel handles per-camera
                      tone).

Trainer reads ``conf.trainer.sky_backend`` ∈ {"cubemap", "mlp"} to pick.

viewdirs are world-frame normalized ray directions (per pixel). The cubemap
backend applies a fixed (camera → OpenGL) rotation so its +Z face corresponds
to the world +Y direction (same convention as drivestudio).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    import nvdiffrast.torch as dr  # type: ignore[import]
except ImportError:  # pragma: no cover — exercised in CI without nvdiffrast
    dr = None  # SkyEnvmapCubemap.forward will raise with a clear message


# ----------------------------------------------------------------------------
# Base interface
# ----------------------------------------------------------------------------
class SkyEnvmapBase(nn.Module):
    """Abstract base for sky backends.

    Subclasses must implement :meth:`forward`: take a tensor of normalized
    world-frame ray directions and return per-direction RGB in [0, 1].
    """

    def forward(self, viewdirs: Tensor) -> Tensor:  # pragma: no cover
        raise NotImplementedError


# ----------------------------------------------------------------------------
# MLP fallback (no external deps)
# ----------------------------------------------------------------------------
class _SinusoidalEncoder(nn.Module):
    """``[..., D] -> [..., D + 2 D (max_deg - min_deg)]``.

    Concatenates raw input with ``sin(2^k * pi * x) / cos(2^k * pi * x)`` for
    ``k`` in ``[min_deg, max_deg)``. Used to feed the MLP sky head higher-
    frequency direction features without external CUDA kernels.
    """

    def __init__(self, n_input_dims: int = 3, min_deg: int = 0, max_deg: int = 6) -> None:
        super().__init__()
        self.n_input_dims = n_input_dims
        self.min_deg = min_deg
        self.max_deg = max_deg
        freqs = 2.0 ** torch.arange(min_deg, max_deg, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def n_output_dims(self) -> int:
        return self.n_input_dims * (1 + 2 * (self.max_deg - self.min_deg))

    def forward(self, x: Tensor) -> Tensor:
        # x: [..., D]. Out: [..., D * (1 + 2 * num_freqs)]
        xb = x.unsqueeze(-1) * self.freqs.to(x.device)  # [..., D, num_freqs]
        flat = xb.reshape(*x.shape[:-1], -1)  # [..., D * num_freqs]
        return torch.cat([x, flat.sin(), flat.cos()], dim=-1)


class SkyEnvmapMLP(SkyEnvmapBase):
    """Fallback sky model: sinusoidal encoding + 3-layer MLP + sigmoid.

    No external dependencies. Use when ``nvdiffrast.torch`` is unavailable or
    when ``trainer.sky_backend == "mlp"`` is explicitly requested.
    """

    def __init__(self, hidden_dim: int = 64, min_deg: int = 0, max_deg: int = 6) -> None:
        super().__init__()
        self.encoder = _SinusoidalEncoder(n_input_dims=3, min_deg=min_deg, max_deg=max_deg)
        in_dim = self.encoder.n_output_dims
        # 3 layers (input → hidden → hidden → 3) with one skip connection from
        # the encoded input to the second hidden layer. Matches the topology of
        # drivestudio's SkyModel head minus the appearance-embedding branch.
        self.layer0 = nn.Linear(in_dim, hidden_dim)
        self.layer1 = nn.Linear(hidden_dim + in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 3)

    def forward(self, viewdirs: Tensor) -> Tensor:
        prefix = viewdirs.shape[:-1]
        # Normalize once at module entry so callers don't have to.
        v = F.normalize(viewdirs.reshape(-1, 3), dim=-1)
        enc = self.encoder(v)
        h = F.relu(self.layer0(enc))
        h = F.relu(self.layer1(torch.cat([h, enc], dim=-1)))
        rgb = torch.sigmoid(self.layer2(h))
        return rgb.reshape(*prefix, 3)


# ----------------------------------------------------------------------------
# Cubemap default (drivestudio EnvLight port)
# ----------------------------------------------------------------------------
class SkyEnvmapCubemap(SkyEnvmapBase):
    """Learnable 6-face cubemap sky, sampled by ``nvdiffrast.torch.texture``.

    Direct port of ``drivestudio/models/modules.py::EnvLight`` (174-208) with
    parameter name ``base`` preserved. The ``to_opengl`` matrix rotates world-
    frame view directions into the cubemap's local frame so +Z face points up.

    Construction does NOT require nvdiffrast (so unit tests can verify the
    parameter shape on CPU); only :meth:`forward` does.
    """

    # 3×3 rotation: world (X right, Y down, Z forward) → OpenGL (X right,
    # Y up, Z back). Same matrix as drivestudio.
    _TO_OPENGL = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=torch.float32,
    )

    def __init__(self, resolution: int = 128) -> None:
        super().__init__()
        self.resolution = int(resolution)
        # Persistent buffer so .to(device) / .cuda() / state_dict() carry it.
        self.register_buffer("to_opengl", self._TO_OPENGL.clone(), persistent=False)
        # Initial 0.5 grey, matching drivestudio. 6 cubemap faces × R × R × 3.
        self.base = nn.Parameter(0.5 * torch.ones(6, self.resolution, self.resolution, 3))

    def forward(self, viewdirs: Tensor) -> Tensor:
        if dr is None:
            raise ImportError(
                "SkyEnvmapCubemap.forward requires nvdiffrast.torch but the "
                "module could not be imported. Install it with "
                "`pip install nvdiffrast`, or set `trainer.sky_backend: mlp`."
            )
        prefix = viewdirs.shape[:-1]
        # nvdiffrast.torch.texture expects [N, H, W, 3] cube lookup directions
        # with a leading batch dim on the texture (we add it via base[None]).
        v = viewdirs.reshape(-1, 3)
        v = F.normalize(v, dim=-1)
        # Rotate into cubemap's local frame.
        v = (v @ self.to_opengl.t().to(v.device, dtype=v.dtype)).contiguous()
        # dr.texture wants the lookup tensor to have 4 dims [B, H, W, 3]; we
        # repack a flat list of directions as [1, 1, N, 3] and unpack later.
        v4 = v.reshape(1, 1, -1, 3).contiguous()
        light = dr.texture(self.base[None], v4, filter_mode="linear", boundary_mode="cube")
        return light.reshape(*prefix, 3)
