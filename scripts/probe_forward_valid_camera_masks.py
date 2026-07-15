#!/usr/bin/env python3
"""Probe production forward-valid mask behavior on real NCore cameras."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import ncore.data.v4 as ncore_v4  # noqa: E402
import ncore.sensors as ncore_sensors  # noqa: E402

from threedgrut.datasets.utils import (  # noqa: E402
    compute_forward_valid_pixel_mask,
    maybe_apply_forward_valid_mask,
    repair_nonfinite_rays,
)


def open_loader(manifest: str):
    reader = ncore_v4.SequenceComponentGroupsReader([manifest])
    return ncore_v4.SequenceLoaderV4(reader)


def probe_camera(loader, camera_id: str) -> dict:
    sensor = loader.get_camera_sensor(camera_id)
    model = ncore_sensors.CameraModel.from_parameters(
        sensor.model_parameters, device="cpu", dtype=torch.float32
    )
    width, height = (int(model.resolution[0]), int(model.resolution[1]))
    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.int16),
        np.arange(height, dtype=np.int16),
    )
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=1)
    rays = model.pixels_to_camera_rays(pixels).reshape(height, width, 3).numpy()
    application_mask = np.ones((height, width), dtype=bool)
    nonfinite = repair_nonfinite_rays(rays, application_mask)
    before = application_mask.copy()
    applied = maybe_apply_forward_valid_mask(
        model, rays, application_mask, camera_id, enabled=True
    )

    if isinstance(model, ncore_sensors.OpenCVPinholeCameraModel):
        forward = compute_forward_valid_pixel_mask(model, rays)
        model_valid = int(forward.sum())
    else:
        # Production application path must be a strict no-op for these models.
        model_valid = width * height

    total = width * height
    return {
        "camera_id": camera_id,
        "model": type(model).__name__,
        "total": total,
        "nonfinite": nonfinite,
        "applied": applied,
        "unchanged": bool(np.array_equal(before, application_mask)),
        "model_valid": model_valid,
        "application_kept": int(application_mask.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--camera-ids", nargs="+", required=True)
    args = parser.parse_args()

    loader = open_loader(str(Path(args.manifest).expanduser().resolve()))
    print(f"manifest={args.manifest}")
    failed = False
    for camera_id in args.camera_ids:
        stats = probe_camera(loader, camera_id)
        total = stats["total"]
        model_pct = 100.0 * stats["model_valid"] / total
        app_pct = 100.0 * stats["application_kept"] / total
        print(
            f"camera={camera_id} model={stats['model']} total={total} "
            f"nonfinite={stats['nonfinite']} applied={stats['applied']} "
            f"unchanged={stats['unchanged']} model_forward_valid={stats['model_valid']} "
            f"model_forward_valid_pct={model_pct:.4f} application_kept={stats['application_kept']} "
            f"application_kept_pct={app_pct:.4f}"
        )
        if stats["model"] == "OpenCVPinholeCameraModel":
            failed |= not stats["applied"]
        else:
            failed |= stats["applied"] or not stats["unchanged"]
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
