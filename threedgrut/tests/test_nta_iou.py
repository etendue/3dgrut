# SPDX-License-Identifier: Apache-2.0
"""E1.2 unit tests for NTA-IoU (Novel Trajectory Agent IoU).

Mac CPU pytest with synthetic boxes + injected fake detector; YOLO is only
touched in the GPU smoke (vehicle_detector.py stays the single ultralytics
coupling point). Projection goes through dynamic_mask.project_cuboids_to_mask
— NOT the viser FthetaForwardProjector (BUG-1 isolation).
"""

from __future__ import annotations

import torch

from threedgrut.model.nta_iou import compute_frame_nta_iou, project_track_to_2d_box


def _pinhole_K(fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    return torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)


def test_project_track_front_center_box():
    # cuboid dead ahead at 10 m, 2x2x2 m, T_w2c = I (world == cam)
    pose = torch.eye(4)
    pose[2, 3] = 10.0
    size = torch.tensor([2.0, 2.0, 2.0])
    box = project_track_to_2d_box(
        pose,
        size,
        K=_pinhole_K(),
        ftheta_params=None,
        T_w2c=torch.eye(4),
        H=480,
        W=640,
    )
    assert box is not None
    x1, y1, x2, y2 = box
    assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480
    # center should sit near the principal point
    assert abs((x1 + x2) / 2 - 320) < 30 and abs((y1 + y2) / 2 - 240) < 30


def test_project_track_behind_returns_none():
    pose = torch.eye(4)
    pose[2, 3] = -10.0  # behind the camera
    size = torch.tensor([2.0, 2.0, 2.0])
    box = project_track_to_2d_box(
        pose,
        size,
        K=_pinhole_K(),
        ftheta_params=None,
        T_w2c=torch.eye(4),
        H=480,
        W=640,
    )
    assert box is None


# ---------------------------------------------------------------- Task 2


class _FakeDetector:
    def __init__(self, boxes):
        self._b = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)

    def detect_vehicles(self, rgb_hw3_01):
        return self._b  # [M, 4] xyxy


def _front_vehicle_track():
    pose = torch.eye(4)
    pose[2, 3] = 10.0
    return {
        "id": 1,
        "class": "automobile",
        "pose": pose,
        "size": torch.tensor([2.0, 2.0, 2.0]),
    }


def test_nta_iou_perfect_match():
    track = _front_vehicle_track()
    gt = project_track_to_2d_box(
        track["pose"],
        track["size"],
        _pinhole_K(),
        None,
        torch.eye(4),
        480,
        640,
    )
    det = _FakeDetector([list(gt)])  # detection box == GT box
    out = compute_frame_nta_iou(
        torch.zeros(480, 640, 3),
        [track],
        det,
        K=_pinhole_K(),
        ftheta_params=None,
        T_w2c=torch.eye(4),
        H=480,
        W=640,
    )
    assert out is not None and out["n_gt"] == 1
    assert out["mean_nta_iou"] > 0.99


def test_nta_iou_no_detection_scores_zero():
    track = _front_vehicle_track()
    out = compute_frame_nta_iou(
        torch.zeros(480, 640, 3),
        [track],
        _FakeDetector([]),
        K=_pinhole_K(),
        ftheta_params=None,
        T_w2c=torch.eye(4),
        H=480,
        W=640,
    )
    assert out["n_gt"] == 1 and out["n_det"] == 0 and out["mean_nta_iou"] == 0.0


def test_nta_iou_no_gt_vehicle_returns_none():
    ped = {
        "id": 9,
        "class": "pedestrian",
        "pose": torch.eye(4),
        "size": torch.tensor([1.0, 1.0, 2.0]),
    }
    out = compute_frame_nta_iou(
        torch.zeros(480, 640, 3),
        [ped],
        _FakeDetector([[0, 0, 10, 10]]),
        K=_pinhole_K(),
        ftheta_params=None,
        T_w2c=torch.eye(4),
        H=480,
        W=640,
    )
    assert out is None


# ------------------------------------------------ E1.2 increment: novel pose


def test_nta_iou_novel_pose_shifts_gt_box():
    """GT cuboid stays in world frame; camera shifted lateral_3m (right) →
    the projected box must move LEFT by ~fx*shift/depth, and the matching
    logic must keep working against a detector that sees the shifted box."""
    from threedgrut.utils.novel_view import perturb_c2w

    track = _front_vehicle_track()
    c2w = torch.eye(4)
    box_orig = project_track_to_2d_box(
        track["pose"],
        track["size"],
        _pinhole_K(),
        None,
        torch.linalg.inv(c2w),
        480,
        640,
    )
    c2w_novel = torch.from_numpy(perturb_c2w(c2w, "lateral_3m")).float()
    T_w2c_novel = torch.linalg.inv(c2w_novel)
    box_novel = project_track_to_2d_box(
        track["pose"],
        track["size"],
        _pinhole_K(),
        None,
        T_w2c_novel,
        480,
        640,
    )
    assert box_novel is not None
    cx_orig = (box_orig[0] + box_orig[2]) / 2
    cx_novel = (box_novel[0] + box_novel[2]) / 2
    # fx * 3 / 10 = 150 px expected shift; demand at least 100 px leftward
    assert cx_novel < cx_orig - 100

    det = _FakeDetector([list(box_novel)])
    out = compute_frame_nta_iou(
        torch.zeros(480, 640, 3),
        [track],
        det,
        K=_pinhole_K(),
        ftheta_params=None,
        T_w2c=T_w2c_novel,
        H=480,
        W=640,
    )
    assert out["mean_nta_iou"] > 0.99
