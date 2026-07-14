# SPDX-License-Identifier: Apache-2.0
"""Pure camera projection/pose state for mixed-camera viser rendering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .viser_math import mat_to_wxyz, slerp_wxyz, wxyz_to_mat


class CameraModelKind(str, Enum):
    FTHETA = "FTheta"
    OPENCV_PINHOLE = "OpenCVPinhole"
    IDEAL_PINHOLE = "IdealPinhole"


@dataclass(frozen=True)
class PoseSample:
    c2w: np.ndarray
    left_idx: int
    right_idx: int
    alpha: float
    nearest_dt_us: int
    source_gap_us: int
    interpolated: bool


@dataclass(frozen=True)
class CameraRenderState:
    camera_id: str
    model_kind: CameraModelKind
    pose_sample: PoseSample
    resolution: Optional[tuple[int, int]]
    fov_y_rad: float
    ftheta_dict: Optional[dict]
    opencv_pinhole_dict: Optional[dict]
    opencv_pinhole_rays: Optional[np.ndarray]


def _validate_pose_inputs(poses: np.ndarray, timestamps_us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    poses = np.asarray(poses, dtype=np.float64)
    timestamps_us = np.asarray(timestamps_us, dtype=np.int64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must have shape (N, 4, 4), got {poses.shape}")
    if timestamps_us.ndim != 1 or timestamps_us.shape[0] != poses.shape[0]:
        raise ValueError(
            f"timestamps_us must have shape ({poses.shape[0]},), got {timestamps_us.shape}"
        )
    if poses.shape[0] == 0:
        raise ValueError("poses/timestamps_us must be non-empty")
    if np.any(np.diff(timestamps_us) < 0):
        raise ValueError("timestamps_us must be sorted ascending")
    return poses, timestamps_us


def interpolate_c2w(poses: np.ndarray, timestamps_us: np.ndarray, t_us: int) -> PoseSample:
    """Interpolate timestamped c2w poses using translation lerp + rotation SLERP."""
    poses, timestamps_us = _validate_pose_inputs(poses, timestamps_us)
    t_us = int(t_us)
    n = timestamps_us.size
    if n == 1 or t_us <= int(timestamps_us[0]):
        dt = abs(t_us - int(timestamps_us[0]))
        return PoseSample(poses[0].copy(), 0, 0, 0.0, dt, 0, False)
    if t_us >= int(timestamps_us[-1]):
        idx = n - 1
        dt = abs(t_us - int(timestamps_us[idx]))
        return PoseSample(poses[idx].copy(), idx, idx, 0.0, dt, 0, False)

    right = int(np.searchsorted(timestamps_us, t_us, side="right"))
    left = right - 1
    t0 = int(timestamps_us[left])
    t1 = int(timestamps_us[right])
    gap = t1 - t0
    alpha = 0.0 if gap <= 0 else float((t_us - t0) / gap)
    nearest_dt = min(abs(t_us - t0), abs(t1 - t_us))

    trans = (1.0 - alpha) * poses[left, :3, 3] + alpha * poses[right, :3, 3]
    q0 = mat_to_wxyz(poses[left])
    q1 = mat_to_wxyz(poses[right])
    out = wxyz_to_mat(slerp_wxyz(q0, q1, alpha))
    out[:3, 3] = trans
    return PoseSample(out, left, right, alpha, nearest_dt, gap, True)


def resolve_camera_render_state(camera_id: str, entry: dict, t_us: int) -> CameraRenderState:
    """Resolve one camera entry into a complete, mutually-exclusive render state."""
    ftheta = entry.get("ftheta_dict")
    opencv = entry.get("opencv_pinhole_dict")
    opencv_rays = entry.get("opencv_pinhole_rays")
    if ftheta is not None and (opencv is not None or opencv_rays is not None):
        raise ValueError("FTheta and OpenCVPinhole fields are mutually exclusive")
    if (opencv is None) != (opencv_rays is None):
        raise ValueError("OpenCVPinhole intrinsics and rays must be provided together")

    if ftheta is not None:
        kind = CameraModelKind.FTHETA
    elif opencv is not None:
        kind = CameraModelKind.OPENCV_PINHOLE
    else:
        kind = CameraModelKind.IDEAL_PINHOLE

    resolution = entry.get("resolution")
    if resolution is not None:
        resolution = (int(resolution[0]), int(resolution[1]))
    sample = interpolate_c2w(entry["c2w"], entry["timestamps_us"], t_us)
    return CameraRenderState(
        camera_id=str(camera_id),
        model_kind=kind,
        pose_sample=sample,
        resolution=resolution,
        fov_y_rad=float(entry.get("fov_y_rad", 1.5708)),
        ftheta_dict=ftheta if kind is CameraModelKind.FTHETA else None,
        opencv_pinhole_dict=opencv if kind is CameraModelKind.OPENCV_PINHOLE else None,
        opencv_pinhole_rays=opencv_rays if kind is CameraModelKind.OPENCV_PINHOLE else None,
    )
