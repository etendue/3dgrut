# SPDX-License-Identifier: Apache-2.0
"""Offline DepthAnythingV2 metric-depth → image-plane prior dump (Stage 11 T11.D1).

Iterates every (clip, camera_id, frame), runs the DepthAnythingV2 metric-outdoor
model on the camera RGB image, and writes a dense ``[H, W]`` metric-depth map
under ``aux/depth_anything_v2/<camera_id>/<timestamp_us>.npz``. The loader
counterpart is ``threedgrut/datasets/aux_readers.py::DepthV2AuxReader`` (Task D2),
a subclass of ``LidarDepthAuxReader`` with identical npz conventions:

    - Path:   <out_root>/<camera_id>/<ts_end_us>.npz
    - Key:    "depth"  (single key), [H, W] float32, metric meters
    - ts_end: camera_sensor.frames_timestamps_us[idx, FrameTimepoint.END]
              (SAME END-timestamp convention as the LiDAR dump T11.C1, so both
               maps align 1:1 with NCoreDataset.__getitem__'s aux lookup).
    - Native: dumped at the camera's NATIVE full resolution; __getitem__ resizes
              to render resolution after reading (same as LiDAR maps).

z-depth vs ray-depth approximation (deliberate, first-pass):
    DepthAnythingV2 metric-outdoor outputs *z-depth* (perpendicular distance to
    the image plane), whereas the LiDAR maps are *ray-depth* (‖cam_pts‖). The
    trainer's depth_prior loss uses inverse-depth L2 which is scale-tolerant, and
    in the image center z≈ray so the discrepancy is <5%. We therefore save the
    model's metric output directly as ``depth`` with NO ray-depth conversion (no
    per-pixel ray geometry needed for an inverse-depth prior).

Heavy imports (torch / transformers model, ncore SDK) are inside ``dump_clip`` so
this file imports on Mac (no SDK / no model needed for a syntax check).

CLI:
    python scripts/dump_depth_priors.py \
        --manifest /path/to/pai_<clip>.json \
        --camera-ids camera_front_wide_120fov ... \
        --weights models/depth_anything_v2 \
        --out-root /path/to/clip/aux/depth_anything_v2 \
        --device cuda \
        --max-frames 1            # sanity run first
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _load_model(weights: str, device: str):
    """Load the DepthAnythingV2 metric model + image processor from a local snapshot.

    Returns ``(model, processor)``. Imported lazily so the module imports on a
    machine without torch / transformers.
    """
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(weights)
    model = AutoModelForDepthEstimation.from_pretrained(weights)
    model = model.to(device).eval()
    logger.info(
        "loaded DepthAnythingV2 model from %s on %s (%s)",
        weights,
        device,
        type(model).__name__,
    )
    return model, processor


def _infer_metric_depth(model, processor, rgb_hwc_uint8: np.ndarray, device: str) -> np.ndarray:
    """Run metric-depth inference on one RGB image → [H, W] float32 (meters).

    The model output is at the processor's working resolution; we bilinearly
    interpolate it back to the input image's native (H, W) before returning.
    The metric-outdoor variant emits metric z-depth directly (no scaling).
    """
    import torch
    import torch.nn.functional as F

    H, W = int(rgb_hwc_uint8.shape[0]), int(rgb_hwc_uint8.shape[1])
    inputs = processor(images=rgb_hwc_uint8, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
        # predicted_depth: [B, h, w] metric meters (metric-outdoor head).
        pred = out.predicted_depth
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)  # [B, 1, h, w]
        pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)
        depth = pred[0, 0].float().cpu().numpy()
    return depth.astype(np.float32)


def dump_clip(
    manifest_path: Path,
    camera_ids: list[str],
    out_root: Path,
    weights: str,
    device: str = "cuda",
    max_frames: int | None = None,
) -> None:
    """Iterate every frame × every camera; write one DepthV2 npz per (camera, frame).

    Per (camera, camera-frame): decode the camera RGB image at native resolution,
    run DepthAnythingV2 metric inference, interpolate the model output back to
    native (H, W), and write ``<out_root>/<camera_id>/<ts_end_us>.npz`` with a
    single ``"depth"`` key (the DepthV2AuxReader / LidarDepthAuxReader contract).

    The frame key is the **END** timestamp (``FrameTimepoint.END``), matching the
    LiDAR dump (T11.C1) and the timestamp ``NCoreDataset.__getitem__`` uses to
    read aux maps.

    Imports the NCore SDK + the torch model lazily so this module imports on Mac.
    """
    try:
        import ncore.data  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "scripts/dump_depth_priors.py dump_clip needs the ncore SDK. "
            "Run from A800 (conda env 3dgrut) or a Mac venv with ncore installed."
        ) from e

    from threedgrut.datasets.datasetNcore import NCoreDataset

    dataset = NCoreDataset(
        datapath=str(manifest_path),
        device="cpu",
        split="train",
        camera_ids=list(camera_ids),
        downsample=1.0,
        load_aux_masks=False,
    )
    sid = dataset.sequence_id
    out_root = Path(out_root)

    model, processor = _load_model(weights, device)

    logger.info(
        "dump_clip(DepthV2): seq=%s cameras=%s weights=%s device=%s out=%s",
        sid,
        list(camera_ids),
        weights,
        device,
        out_root,
    )

    summary: dict[str, tuple[int, float]] = {}  # camera_id -> (n_frames, mean_median_depth)

    for camera_id in camera_ids:
        camera_sensor = dataset.sequence_camera_sensors[sid][camera_id]
        camera_model = dataset.sequence_camera_models[sid][camera_id]
        W = int(camera_model.resolution[0].item())
        H = int(camera_model.resolution[1].item())

        cam_out_dir = out_root / camera_id
        cam_out_dir.mkdir(parents=True, exist_ok=True)

        n_frames = int(np.asarray(camera_sensor.frames_timestamps_us).shape[0])
        if max_frames is not None:
            n_frames = min(n_frames, int(max_frames))

        median_total = 0.0
        for frame_idx in range(n_frames):
            ts_end_us = int(camera_sensor.frames_timestamps_us[frame_idx, ncore.data.FrameTimepoint.END])
            # Native full-resolution RGB (uint8, HWC). get_frame_image_array is the
            # full-res decode path the dataset's PIL fallback uses (downsample=1.0
            # → no resize); matches the (H, W) we save at.
            rgb = np.asarray(camera_sensor.get_frame_image_array(frame_idx))
            if rgb.ndim == 2:  # grayscale → 3ch
                rgb = np.repeat(rgb[..., None], 3, axis=-1)
            if rgb.shape[-1] == 4:  # RGBA → RGB
                rgb = rgb[..., :3]

            depth = _infer_metric_depth(model, processor, rgb.astype(np.uint8), device)
            # Guard: model output must match native (H, W) the reader expects.
            if depth.shape[0] != H or depth.shape[1] != W:
                import cv2

                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

            med = float(np.median(depth))
            median_total += med
            np.savez_compressed(cam_out_dir / f"{ts_end_us}.npz", depth=depth.astype(np.float32))

            if frame_idx % 50 == 0 or frame_idx == n_frames - 1:
                logger.info(
                    "  [%s] frame %d/%d ts=%d shape=%s min=%.2f med=%.2f max=%.2f",
                    camera_id,
                    frame_idx + 1,
                    n_frames,
                    ts_end_us,
                    depth.shape,
                    float(depth.min()),
                    med,
                    float(depth.max()),
                )

        mean_med = median_total / max(n_frames, 1)
        summary[camera_id] = (n_frames, mean_med)
        logger.info(
            "dump_clip(DepthV2): camera %s done — %d frames, mean median depth=%.2fm",
            camera_id,
            n_frames,
            mean_med,
        )

    logger.info("=== dump_clip(DepthV2) summary (seq=%s) ===", sid)
    for cam, (nf, mm) in summary.items():
        logger.info("  %s: %d frames, mean median depth=%.2fm", cam, nf, mm)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--camera-ids", nargs="+", required=True)
    p.add_argument(
        "--weights",
        type=str,
        default="models/depth_anything_v2",
        help="Local DepthAnythingV2 snapshot dir (from " "scripts/download_depth_anything_v2.sh) or a HF repo id.",
    )
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap frames per camera (smoke / sanity-check). Default: all.",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    dump_clip(
        args.manifest,
        args.camera_ids,
        args.out_root,
        weights=args.weights,
        device=args.device,
        max_frames=args.max_frames,
    )
