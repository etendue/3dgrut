# SPDX-License-Identifier: Apache-2.0
"""E1.2 YOLO vehicle-detector wrapper for NTA-IoU.

The ONLY file that imports ultralytics; lazy-loaded and reused as a
singleton across the eval loop. Everything upstream consumes the duck-typed
``detect_vehicles`` protocol so Mac unit tests inject fakes instead.
"""
from __future__ import annotations

import torch

from threedgrut.model.nta_iou import VEHICLE_COCO_IDS

_SINGLETON = None


class VehicleDetector:
    def __init__(self, weights: str = "yolov8m.pt", conf: float = 0.3,
                 device: str = "cuda"):
        from ultralytics import YOLO  # local import: missing dep must not
        self.model = YOLO(weights)    # break the rest of the eval
        self.conf = conf
        self.device = device

    @torch.no_grad()
    def detect_vehicles(self, rgb_hw3_01):
        """rgb [H, W, 3] in [0, 1] → vehicle-class 2D boxes [M, 4] xyxy px."""
        arr = (rgb_hw3_01.detach().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        res = self.model.predict(
            arr, conf=self.conf, device=self.device, verbose=False,
        )[0]
        b = res.boxes
        if b is None or b.shape[0] == 0:
            return torch.zeros((0, 4), dtype=torch.float32)
        cls = b.cls.to(torch.int64)
        keep = torch.zeros_like(cls, dtype=torch.bool)
        for cid in VEHICLE_COCO_IDS:
            keep |= cls == cid
        return b.xyxy[keep].float().cpu()


def get_vehicle_detector(weights: str = "yolov8m.pt", conf: float = 0.3,
                         device: str = "cuda"):
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = VehicleDetector(weights, conf, device)
    return _SINGLETON
