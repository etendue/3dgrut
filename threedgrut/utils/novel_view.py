# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""T8.5.3 / V3-E3 novel-view pose perturbations for v3 main-KPI evaluation.

Each training-frame c2w is taken as an anchor; we perturb it along 4 distinct
camera-extrapolation modes and re-render to assess view-extrapolation
robustness. PSNR vs the anchor GT is dominated by parallax shift at these
magnitudes (LPIPS is more meaningful), so the eval reports LPIPS in addition
to PSNR — see render.py render_all() for the integration.

Conventions (matches threedgrut camera convention, see viser_gui_util.py:191):
- c2w[:3, 0] = camera right axis (lateral direction for ego)
- c2w[:3, 1] = camera down (so world-up = -c2w[:3, 1])
- c2w[:3, 2] = camera forward
- c2w[:3, 3] = camera position in world

Rolling-shutter note: ``T_to_world`` and ``T_to_world_end`` (the shutter-start
and shutter-end poses) must both be perturbed by the SAME delta to keep the
shutter window consistent; otherwise the renderer interprets it as a wildly
distorted intra-frame motion. Use ``perturb_shutter_pair`` for that pair.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

# Legacy 4-mode set; v3_plan.md § 2.0 T8.5.3 spec.
# mean_novel_lpips_avg aggregates ONLY these 4 modes, forever — the historical
# anchor (B3 0.5962) depends on this field meaning exactly this average.
# E1.1 adds extrapolation modes below; they go into mean_novel_lpips_avg6.
LEGACY_NOVEL_AVG_MODES: Tuple[str, ...] = (
    "lateral_1m",   # +1 m along camera right axis
    "lateral_2m",   # +2 m along camera right axis
    "yaw_5deg",     # +5° rotation around camera up axis (world-up under AV convention)
    "yaw_10deg",    # +10° rotation around camera up axis
)

# Order matters: render.py writes metrics under mean_novel_lpips_<mode_name>.
NOVEL_VIEW_MODES: Tuple[str, ...] = LEGACY_NOVEL_AVG_MODES + (
    "lateral_3m",   # +3 m along camera right axis (E1.1 extrapolation gate)
    "lateral_6m",   # +6 m along camera right axis (E1.1 extrapolation gate)
)


def _so3_around_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix around unit axis by angle_rad."""
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)


def _to_numpy_44(c2w) -> np.ndarray:
    """Accept np / torch / (1,4,4) batched; return float64 (4, 4)."""
    if isinstance(c2w, torch.Tensor):
        a = c2w.detach().cpu().numpy()
    else:
        a = np.asarray(c2w)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.shape != (4, 4):
        raise ValueError(f"c2w must be (4,4) or (1,4,4); got {a.shape}")
    return a.astype(np.float64, copy=True)


def perturb_c2w(c2w, mode: str) -> np.ndarray:
    """Apply one of NOVEL_VIEW_MODES to a c2w; return new (4,4) float64.

    Lateral modes shift position along the camera's right axis (c2w[:3,0]).
    Yaw modes rotate the camera frame around its local up axis (-c2w[:3,1])
    while keeping the camera position fixed — the camera looks in a new
    direction from the same point.
    """
    if mode not in NOVEL_VIEW_MODES:
        raise ValueError(
            f"mode '{mode}' not in NOVEL_VIEW_MODES {NOVEL_VIEW_MODES}"
        )
    m = _to_numpy_44(c2w)
    out = m.copy()
    if mode == "lateral_1m":
        out[:3, 3] = m[:3, 3] + 1.0 * m[:3, 0]
    elif mode == "lateral_2m":
        out[:3, 3] = m[:3, 3] + 2.0 * m[:3, 0]
    elif mode == "lateral_3m":
        out[:3, 3] = m[:3, 3] + 3.0 * m[:3, 0]
    elif mode == "lateral_6m":
        out[:3, 3] = m[:3, 3] + 6.0 * m[:3, 0]
    elif mode == "yaw_5deg":
        # Camera up is -y in c2w convention; positive yaw_deg rotates camera
        # CCW when viewed from above (world-up).
        up = -m[:3, 1]
        R = _so3_around_axis(up, np.deg2rad(5.0))
        out[:3, :3] = R @ m[:3, :3]
    elif mode == "yaw_10deg":
        up = -m[:3, 1]
        R = _so3_around_axis(up, np.deg2rad(10.0))
        out[:3, :3] = R @ m[:3, :3]
    return out


def perturb_shutter_pair(
    c2w_start, c2w_end, mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the same perturbation delta to both shutter-start and shutter-end
    poses (rolling-shutter integrity). Returns (start_new, end_new) float64.

    Implementation detail: for lateral modes the delta in world coordinates
    depends on the camera frame (right axis). We use the **start** frame's
    right axis for both translations to preserve the same world-frame shift
    across the shutter window. Yaw is applied as a left-multiplication of R,
    matching what perturb_c2w does.
    """
    start = _to_numpy_44(c2w_start)
    end = _to_numpy_44(c2w_end)
    new_start = perturb_c2w(start, mode)
    if mode.startswith("lateral"):
        # Use the SAME world-frame translation delta for end pose to keep
        # shutter trajectory rigid.
        delta = new_start[:3, 3] - start[:3, 3]
        new_end = end.copy()
        new_end[:3, 3] = end[:3, 3] + delta
    else:  # yaw modes
        # Same yaw rotation around START frame's up axis applied to END.
        up = -start[:3, 1]
        if mode == "yaw_5deg":
            R = _so3_around_axis(up, np.deg2rad(5.0))
        else:  # yaw_10deg
            R = _so3_around_axis(up, np.deg2rad(10.0))
        new_end = end.copy()
        # Rotate both rotation and translation-from-start of end pose around
        # the start position so the shutter trajectory rotates rigidly.
        pivot = start[:3, 3]
        new_end[:3, :3] = R @ end[:3, :3]
        new_end[:3, 3] = R @ (end[:3, 3] - pivot) + pivot
    return new_start, new_end


def perturb_batch_shutter_pair_torch(
    T_to_world: torch.Tensor,
    T_to_world_end: torch.Tensor,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch-side wrapper for the renderer batch path.

    Inputs are (1, 4, 4) tensors as produced by NCoreDataset; outputs the
    same shape, same dtype, same device.
    """
    if T_to_world.shape != (1, 4, 4):
        raise ValueError(
            f"T_to_world must be (1,4,4); got {tuple(T_to_world.shape)}"
        )
    s, e = perturb_shutter_pair(T_to_world, T_to_world_end, mode)
    new_start = torch.from_numpy(s).to(T_to_world.device, dtype=T_to_world.dtype).unsqueeze(0)
    new_end = torch.from_numpy(e).to(T_to_world_end.device, dtype=T_to_world_end.dtype).unsqueeze(0)
    return new_start, new_end


# ---------------------------------------------------------------------------
# E2.1 Harmonizer frame-alignment helpers
# ---------------------------------------------------------------------------

def novel_frame_key(camera_id: str, timestamp_us) -> str:
    """E2.1 frame-alignment key, must match eval_frames_dir.resolve_pred_path's
    ``ts:<camera_id>:<timestamp_us>`` join key (NCore batches carry no frame_idx)."""
    return f"ts:{camera_id}:{int(timestamp_us)}"


def novel_frame_relpath(camera_id: str, save_idx: int) -> str:
    """Per-camera subdir, 6-digit zero-padded — matches eval_frames_dir fallback layout."""
    return f"{camera_id}/{int(save_idx):06d}.png"
