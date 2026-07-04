# SPDX-License-Identifier: Apache-2.0
"""E1.2 NTA-IoU (Novel Trajectory Agent IoU).

Vehicle-pose/shape sensitivity metric: run a 2D vehicle detector on the
RENDERED image, project the GT cuboids to 2D AABBs, and score each GT with
its best-match detection IoU. A blurred / scattered / displaced vehicle
loses IoU long before PSNR notices (ReconDreamer-family protocol).

Projection reuses ``dynamic_mask.project_cuboids_to_mask`` (the eval-side
FTheta path) — never the viser ``FthetaForwardProjector`` (BUG-1 isolation,
see v2_architecture § 7). The detector is duck-typed (``detect_vehicles``)
so the matching logic is testable on Mac with a fake; the real YOLO wrapper
lives in ``vehicle_detector.py``.
"""

from __future__ import annotations

import torch
from torchvision.ops import box_iou as _tv_box_iou

from threedgrut.layers.dynamic_mask import _CORNER_SIGNS, project_cuboids_to_mask

# E1.2 Task 0 (2026-06-12, ckpt viz_4d census on clip 9ae151dc): 70 tracks =
# {automobile: 68, heavy_truck: 1, bus: 1}. Extra aliases are harmless — the
# set is only used as a filter.
VEHICLE_TRACK_CLASSES = {
    "automobile",
    "heavy_truck",
    "bus",
    "car",
    "truck",
    "vehicle",
}
# COCO class ids kept by the detector: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_COCO_IDS = (2, 3, 5, 7)


def project_track_to_2d_box(pose, size, K, ftheta_params, T_w2c, H, W):
    """Project one GT cuboid → axis-aligned 2D box ``(x1, y1, x2, y2)`` px.

    Returns None when the cuboid is invisible (behind camera / off-image),
    matching project_cuboids_to_mask's visibility semantics.
    """
    device = pose.device if isinstance(pose, torch.Tensor) else "cpu"
    # Visibility precheck independent of the intrinsics model: all 8 corners
    # behind the camera → invisible. (The pinhole branch of
    # project_cuboids_to_mask clamps z to 0.1 and would happily project a
    # behind-camera cuboid; the FTheta branch handles this itself.)
    corners = _CORNER_SIGNS.to(device=device, dtype=torch.float32) * (
        size.to(device=device, dtype=torch.float32) * 0.5
    )  # [8, 3]
    ones = torch.ones(8, 1, dtype=torch.float32, device=device)
    world = (pose.to(device=device, dtype=torch.float32) @ torch.cat([corners, ones], dim=-1).T).T  # [8, 4]
    cam_z = (T_w2c.to(device=device, dtype=torch.float32) @ world.T).T[:, 2]
    if not bool((cam_z > 0).any()):
        return None
    mask = project_cuboids_to_mask(
        pose.unsqueeze(0),
        size.unsqueeze(0),
        K,
        T_w2c,
        H,
        W,
        device=device,
        ftheta_params=ftheta_params,
    )  # [H, W] bool
    ys, xs = torch.where(mask)
    if ys.numel() == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def compute_frame_nta_iou(
    pred_rgb_hw3,
    active_tracks,
    detector,
    K,
    ftheta_params,
    T_w2c,
    H,
    W,
):
    """One frame of NTA-IoU. Returns ``{"mean_nta_iou", "n_gt", "n_det"}``;
    None when the frame has no visible GT vehicle (frame not counted).

    Each GT vehicle box takes its best-match IoU over all detections; no
    detections → 0.0 for every GT (a missing vehicle is a real failure).
    """
    gt_boxes = []
    for t in active_tracks:
        if str(t.get("class", "")).lower() not in VEHICLE_TRACK_CLASSES:
            continue
        b = project_track_to_2d_box(
            t["pose"].to(torch.float32),
            t["size"].to(torch.float32),
            K,
            ftheta_params,
            T_w2c,
            H,
            W,
        )
        if b is not None:
            gt_boxes.append(b)
    if not gt_boxes:
        return None
    det = detector.detect_vehicles(pred_rgb_hw3)
    if det is None or det.numel() == 0:
        return {"mean_nta_iou": 0.0, "n_gt": len(gt_boxes), "n_det": 0}
    gt = torch.tensor(gt_boxes, dtype=torch.float32)
    iou = _tv_box_iou(gt, det.to(torch.float32).cpu())  # [G, M]
    best = iou.max(dim=1).values  # best match per GT
    return {
        "mean_nta_iou": float(best.mean()),
        "n_gt": len(gt_boxes),
        "n_det": int(det.shape[0]),
    }
