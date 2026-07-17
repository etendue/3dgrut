# SPDX-License-Identifier: Apache-2.0
"""Pure camera projection/pose state for mixed-camera viser rendering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .viser_math import mat_to_wxyz, slerp_wxyz, wxyz_to_mat


FTHETA_REQUIRED_KEYS = frozenset(
    {
        "resolution",
        "shutter_type",
        "principal_point",
        "reference_poly",
        "pixeldist_to_angle_poly",
        "angle_to_pixeldist_poly",
        "max_angle",
        "linear_cde",
    }
)


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


def _resolution_tuple(value, *, context: str) -> tuple[int, int]:
    raw = np.asarray(value, dtype=np.int64).reshape(-1)
    if raw.size != 2:
        raise ValueError(f"{context} resolution must contain W,H, got {raw.shape}")
    result = (int(raw[0]), int(raw[1]))
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError(f"{context} resolution must be positive, got {result}")
    return result


def _validate_ftheta_dict(ftheta: object, *, camera_id: str) -> dict:
    if not isinstance(ftheta, dict):
        raise ValueError(f"FTheta camera '{camera_id}' intrinsics must be a dictionary")
    missing = FTHETA_REQUIRED_KEYS.difference(ftheta)
    if missing:
        raise ValueError(
            f"FTheta camera '{camera_id}' missing required keys: {sorted(missing)}"
        )
    return ftheta


def merge_checkpoint_camera_models(
    pose_entries: dict[str, dict], camera_models: dict[str, dict]
) -> dict[str, dict]:
    """Bind checkpoint projection contracts to manifest-derived camera poses.

    The checkpoint owns the active camera set and the model used for training;
    the manifest loader only supplies timestamped poses (and, for legacy
    pinhole checkpoints, its calibrated pinhole payload). A schema-v3 FTheta
    contract therefore replaces both pinhole fields atomically and fixes the
    render resolution to that camera's stored native resolution.
    """
    if not camera_models:
        return dict(pose_entries)

    out: dict[str, dict] = {}
    for camera_id, contract in camera_models.items():
        if camera_id not in pose_entries:
            raise KeyError(
                f"active checkpoint camera '{camera_id}' has no manifest pose entry"
            )
        entry = dict(pose_entries[camera_id])
        model_type = contract.get("model_type")
        resolution = _resolution_tuple(
            contract.get("native_resolution"), context=f"camera '{camera_id}'"
        )

        if model_type == CameraModelKind.FTHETA.value:
            ftheta = _validate_ftheta_dict(
                contract.get("intrinsics_FTheta"), camera_id=camera_id
            )
            ftheta_resolution = _resolution_tuple(
                ftheta["resolution"], context=f"camera '{camera_id}' FTheta"
            )
            if ftheta_resolution != resolution:
                raise ValueError(
                    f"camera '{camera_id}' resolution mismatch: "
                    f"native={resolution}, FTheta={ftheta_resolution}"
                )
            entry["ftheta_dict"] = ftheta
            entry["opencv_pinhole_dict"] = None
            entry["opencv_pinhole_rays"] = None
        elif model_type == CameraModelKind.OPENCV_PINHOLE.value:
            entry["ftheta_dict"] = None
            if entry.get("opencv_pinhole_dict") is None or entry.get(
                "opencv_pinhole_rays"
            ) is None:
                raise ValueError(
                    f"OpenCVPinhole camera '{camera_id}' is missing calibrated "
                    "intrinsics/rays from the manifest"
                )
        elif model_type == CameraModelKind.IDEAL_PINHOLE.value:
            entry["ftheta_dict"] = None
            entry["opencv_pinhole_dict"] = None
            entry["opencv_pinhole_rays"] = None
        else:
            raise ValueError(
                f"camera '{camera_id}' has unsupported model_type={model_type!r}"
            )

        entry["resolution"] = resolution
        out[camera_id] = entry
    return out


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
        _validate_ftheta_dict(ftheta, camera_id=str(camera_id))
        kind = CameraModelKind.FTHETA
    elif opencv is not None:
        kind = CameraModelKind.OPENCV_PINHOLE
    else:
        kind = CameraModelKind.IDEAL_PINHOLE

    resolution = entry.get("resolution")
    if resolution is not None:
        resolution = _resolution_tuple(
            resolution, context=f"camera '{camera_id}' render"
        )
    if kind is CameraModelKind.FTHETA:
        if resolution is None:
            raise ValueError(f"FTheta camera '{camera_id}' is missing render resolution")
        ftheta_resolution = _resolution_tuple(
            ftheta["resolution"], context=f"camera '{camera_id}' FTheta"
        )
        if ftheta_resolution != resolution:
            raise ValueError(
                f"camera '{camera_id}' resolution mismatch: "
                f"render={resolution}, FTheta={ftheta_resolution}"
            )
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
