# SPDX-License-Identifier: Apache-2.0
"""Pure-CPU FTheta intrinsics helpers for the viser viewer + engine.

T8.13: keeps numpy → torch conversion logic out of ``engine.py`` so it
can be unit-tested on a Mac (engine.py imports ``kaolin`` at module
level and is uninstallable on CPU-only dev machines).

The 3dgut UT rasterizer at ``threedgut_tracer/tracer.py:471`` consumes
``Batch.intrinsics_FThetaCameraModelParameters`` as a plain dict — the
8 keys (resolution / shutter_type / principal_point / reference_poly /
pixeldist_to_angle_poly / angle_to_pixeldist_poly / max_angle /
linear_cde) are forwarded verbatim to
``_3dgut_plugin.fromFThetaCameraModelParameters`` (bindings.cpp:79).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def ftheta_dict_to_tensors(d: Optional[dict],
                           device: torch.device | str = "cpu") -> Optional[dict]:
    """Convert a numpy-stored FTheta intrinsics dict → torch tensors on ``device``.

    Pass-through for str / float / int scalars (shutter_type / reference_poly
    name strings, max_angle scalar). numpy int arrays → int64; numpy float
    arrays → float32; existing torch tensors are moved to ``device``.

    Returns ``None`` if input is ``None`` (FTheta path disabled).
    """
    if d is None:
        return None
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            if v.dtype.kind in ("i", "u"):
                out[k] = torch.from_numpy(v).to(device=device, dtype=torch.int64)
            else:
                out[k] = torch.from_numpy(v).to(device=device, dtype=torch.float32)
        elif torch.is_tensor(v):
            out[k] = v.to(device=device)
        else:
            out[k] = v  # str / float / int — pass through unchanged
    return out
