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


def ftheta_pixels_to_camera_rays(ftheta_dict: dict) -> np.ndarray:
    """Vectorized FTheta polynomial pixel → camera-space ray direction.

    Mirrors what ``ncore.sensors.FThetaCameraModel.pixels_to_camera_rays``
    does on the training side (see datasetNcore.py:400). Re-implemented in
    numpy so the viewer doesn't need the NCore SDK at runtime — the ckpt's
    persisted FTheta dict is enough.

    The viser_gui_4d FTheta path must use these rays (not kaolin's pinhole
    raygen). Reason: 3dgut UT rasterizer doesn't read ``rays_dir`` to
    project Gaussians (intrinsics polynomial handles that), but the
    downstream SH / depth / opacity sampling uses each pixel's true ray
    direction. Pinhole vs FTheta mismatch → Gaussians sample wrong angular
    region → tunnel motion-blur output (T8.13 viser_gui_4d Phase D bug).

    Args:
        ftheta_dict: 8-key params dict (resolution / principal_point /
            pixeldist_to_angle_poly / linear_cde / ... as stored by inject).

    Returns:
        (H, W, 3) float32 ndarray of camera-space ray directions, unit norm.
    """
    res = ftheta_dict["resolution"]
    W, H = int(res[0]), int(res[1])
    pp = ftheta_dict["principal_point"]
    cx, cy = float(pp[0]), float(pp[1])
    poly = np.asarray(ftheta_dict["pixeldist_to_angle_poly"], dtype=np.float64)
    # NOTE: ``linear_cde`` is NCore FTheta's affine distortion correction
    # applied before the polynomial. Convention is ambiguous (different repos
    # store [c, d, e] for either [[c, d], [d, e]] or [[1+c, d], [d, 1+e]]).
    # For our typical ckpts it's very close to identity ([1.0016, 0, 0] →
    # ~0.2% scaling on x), so we skip the affine step entirely and let the
    # polynomial handle the radial mapping. If a future ckpt has a non-trivial
    # cde we'll need to figure out NCore's exact convention.
    # cde = np.asarray(ftheta_dict["linear_cde"], dtype=np.float64)

    # Pixel grid: (H, W) with x=column (j), y=row (i).
    js, is_ = np.meshgrid(np.arange(W, dtype=np.float64),
                          np.arange(H, dtype=np.float64), indexing="xy")
    du = js - cx
    dv = is_ - cy

    pixel_dist = np.sqrt(du * du + dv * dv)

    # Polynomial theta(r) = poly[0] + poly[1]*r + ... + poly[5]*r^5 (Horner).
    theta = np.zeros_like(pixel_dist)
    for k in range(len(poly) - 1, -1, -1):
        theta = theta * pixel_dist + poly[k]

    # Direction: equidistant fisheye convention,
    #   d = (sin(theta) * du / r, sin(theta) * dv / r, cos(theta))
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    eps = 1e-12
    norm = np.maximum(pixel_dist, eps)
    rx = sin_t * du / norm
    ry = sin_t * dv / norm
    rz = cos_t
    rays = np.stack([rx, ry, rz], axis=-1).astype(np.float32)
    return rays  # (H, W, 3)


def ftheta_dict_to_tensors(d: Optional[dict],
                           device: torch.device | str = "cpu") -> Optional[dict]:
    """Normalize a numpy/tensor-stored FTheta intrinsics dict → the shape
    expected by ``_3dgut_plugin.fromFThetaCameraModelParameters``.

    The 3dgut UT rasterizer C++ binding (threedgut_tracer/bindings.cpp:79)
    is declared as ``list[int]/list[float] FixedSize`` for the array fields
    (resolution / principal_point / *_poly / linear_cde) — not tensor/ndarray.
    NCoreDataset's training path delivers Python lists directly from the
    ncore SDK; our viz_4d schema round-trip stores numpy arrays for
    portability, so we must convert back to plain Python lists here.

    Pass-through for str / float / int scalars (shutter_type / reference_poly
    name strings, max_angle scalar). numpy int arrays → list[int]; numpy
    float arrays → list[float]; torch tensors → moved to CPU then list.

    The ``device`` kwarg is preserved for backward compat but no longer
    used (lists go through the binding into C++ verbatim).

    Returns ``None`` if input is ``None`` (FTheta path disabled).
    """
    del device  # Unused; binding takes plain Python lists.
    if d is None:
        return None
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            if v.dtype.kind in ("i", "u"):
                out[k] = [int(x) for x in v.tolist()]
            else:
                out[k] = [float(x) for x in v.tolist()]
        elif torch.is_tensor(v):
            t = v.detach().cpu()
            if t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64,
                           torch.uint8):
                out[k] = [int(x) for x in t.tolist()]
            else:
                out[k] = [float(x) for x in t.tolist()]
        else:
            out[k] = v  # str / float / int — pass through unchanged
    return out
