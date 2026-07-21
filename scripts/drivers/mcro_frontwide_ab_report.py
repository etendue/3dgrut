#!/usr/bin/env python3
"""Create a front-wide crop, radial, and sharpness A/B render report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_DRIVER_DIR = Path(__file__).resolve().parent
if str(_DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(_DRIVER_DIR))

from pin_ab_radial_analysis import (  # noqa: E402
    _compute_aggregate_gradient_ratio,
    _compute_corner_normalized_radius_map,
    _compute_gradient_magnitudes,
    _load_images_from_root,
    _psnr,
    _resolve_run_root,
)


DEFAULT_CAMERA_ID = "camera_front_wide_120fov"
_RADIAL_BINS = (("r<0.5", 0.0, 0.5), ("r0.5-0.7", 0.5, 0.7), ("r0.7-0.9", 0.7, 0.9), ("r>=0.9", 0.9, float("inf")))


def _load_frontwide_images(eval_dir: str, camera_id: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    run_root, metrics_path = _resolve_run_root(eval_dir)
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    per_camera = metrics.get("per_camera", {})
    if camera_id not in per_camera:
        raise ValueError(f"metrics.json in {run_root} has no {camera_id} entry")
    camera_ids = list(per_camera)
    offset = sum(int(per_camera[camera]["n_frames"]) for camera in camera_ids[: camera_ids.index(camera_id)])
    n_frames = int(per_camera[camera_id]["n_frames"])
    renders, gts = _load_images_from_root(run_root)
    if len(renders) != sum(int(data["n_frames"]) for data in per_camera.values()):
        raise ValueError(f"Image count in {run_root} does not match metrics.json per_camera frame counts")
    return renders[offset : offset + n_frames], gts[offset : offset + n_frames]


def _load_crops(crops_json: str) -> dict[str, Any]:
    payload = json.loads(Path(crops_json).read_text(encoding="utf-8"))
    if not isinstance(payload.get("crops"), list):
        raise ValueError(f"Crop config {crops_json} must contain a crops list")
    return payload


def _mse_sum_and_count(render: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, int]:
    if render.shape != gt.shape:
        raise ValueError(f"Render/GT shape mismatch: {render.shape} vs {gt.shape}")
    sq_error = ((render.astype(np.float64) - gt.astype(np.float64)) ** 2).mean(axis=2)
    if mask is not None:
        sq_error = sq_error[mask]
    return float(sq_error.sum()), int(sq_error.size)


def _lpips_scorer(enabled: bool):
    if not enabled:
        return None, "disabled by --no-lpips"
    try:
        import lpips
        import torch

        model = lpips.LPIPS(net="alex").eval()

        def score(render: np.ndarray, gt: np.ndarray) -> float:
            render_tensor = torch.from_numpy(render.transpose(2, 0, 1)).unsqueeze(0).float() * 2.0 - 1.0
            gt_tensor = torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0).float() * 2.0 - 1.0
            with torch.no_grad():
                return float(model(render_tensor, gt_tensor).item())

        return score, "enabled"
    except Exception as error:  # A Mac often lacks LPIPS weights; retain the rest of the report.
        return None, f"unavailable: {type(error).__name__}: {error}"


def _metric(render: np.ndarray, gt: np.ndarray, lpips_score) -> dict[str, float | None]:
    mse_sum, n_pixels = _mse_sum_and_count(render, gt)
    _, _, gradient_ratio = _compute_aggregate_gradient_ratio(render, gt, mask=None)
    return {
        "psnr": _psnr(mse_sum / n_pixels),
        "lpips": None if lpips_score is None else lpips_score(render, gt),
        "gradient_ratio": gradient_ratio,
    }


def _delta_b_minus_a(a: dict[str, float | None], b: dict[str, float | None]) -> dict[str, float | None]:
    return {key: None if a[key] is None or b[key] is None else float(b[key] - a[key]) for key in a}


def _crop_image(image: np.ndarray, crop: dict[str, Any]) -> np.ndarray:
    name = str(crop["name"])
    if not isinstance(crop.get("box"), list) or len(crop["box"]) != 4:
        raise ValueError(f"Crop {name} must provide box=[x0,y0,x1,y1]")
    x0, y0, x1, y1 = (int(value) for value in crop["box"])
    h, w = image.shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(f"Crop {name} out of bounds for {w}x{h}: {[x0, y0, x1, y1]}")
    return image[y0:y1, x0:x1]


def _write_triptych(path: Path, crop_a: np.ndarray, crop_b: np.ndarray, crop_gt: np.ndarray) -> None:
    images = [Image.fromarray(np.round(image * 255.0).clip(0, 255).astype(np.uint8)) for image in (crop_a, crop_b, crop_gt)]
    width, height = images[0].size
    montage = Image.new("RGB", (width * 3, height))
    for index, image in enumerate(images):
        montage.paste(image, (index * width, 0))
    montage.save(path)


def _radial_report(renders_a: list[np.ndarray], renders_b: list[np.ndarray], gts: list[np.ndarray]) -> list[dict[str, Any]]:
    accumulators = [{"mse_a": 0.0, "mse_b": 0.0, "n_pixels": 0} for _ in _RADIAL_BINS]
    for render_a, render_b, gt in zip(renders_a, renders_b, gts, strict=True):
        h, w = gt.shape[:2]
        radius = _compute_corner_normalized_radius_map(h, w, w / 2.0, h / 2.0)
        for index, (_, lower, upper) in enumerate(_RADIAL_BINS):
            mask = radius >= lower if np.isinf(upper) else (radius >= lower) & (radius < upper)
            mse_a, count = _mse_sum_and_count(render_a, gt, mask)
            mse_b, _ = _mse_sum_and_count(render_b, gt, mask)
            accumulators[index]["mse_a"] += mse_a
            accumulators[index]["mse_b"] += mse_b
            accumulators[index]["n_pixels"] += count
    rows = []
    for (name, _, _), values in zip(_RADIAL_BINS, accumulators, strict=True):
        a_psnr = _psnr(values["mse_a"] / values["n_pixels"])
        b_psnr = _psnr(values["mse_b"] / values["n_pixels"])
        rows.append({"bin": name, "a_psnr": a_psnr, "b_psnr": b_psnr, "delta_b_minus_a_psnr": b_psnr - a_psnr})
    return rows


def _split_metrics(
    renders_a: list[np.ndarray], renders_b: list[np.ndarray], gts: list[np.ndarray], frame_splits: dict[str, str]
) -> dict[str, dict[str, Any]]:
    sums: dict[str, dict[str, float]] = {}
    for index, (render_a, render_b, gt) in enumerate(zip(renders_a, renders_b, gts, strict=True)):
        split = frame_splits.get(str(index), "held_out")
        values = sums.setdefault(split, {"mse_a": 0.0, "mse_b": 0.0, "n_pixels": 0.0, "n_frames": 0.0})
        mse_a, count = _mse_sum_and_count(render_a, gt)
        mse_b, _ = _mse_sum_and_count(render_b, gt)
        values["mse_a"] += mse_a
        values["mse_b"] += mse_b
        values["n_pixels"] += count
        values["n_frames"] += 1
    return {
        split: {
            "n_frames": int(values["n_frames"]),
            "a_psnr": _psnr(values["mse_a"] / values["n_pixels"]),
            "b_psnr": _psnr(values["mse_b"] / values["n_pixels"]),
            "delta_b_minus_a_psnr": _psnr(values["mse_b"] / values["n_pixels"])
            - _psnr(values["mse_a"] / values["n_pixels"]),
        }
        for split, values in sums.items()
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MCRO front-wide A/B report",
        "",
        f"LPIPS: {report['lpips_status']}",
        "",
        "## Fixed crops",
        "",
        "| crop | frame | A PSNR | B PSNR | Δ PSNR（B−A） | A LPIPS | B LPIPS | A/B gradient ratio |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for name, crop in report["crops"].items():
        a, b, delta = crop["a"], crop["b"], crop["delta_b_minus_a"]
        lines.append(
            f"| {name} | {crop['frame_index']} | {a['psnr']:.4f} | {b['psnr']:.4f} | {delta['psnr']:.4f} | "
            f"{a['lpips']} | {b['lpips']} | {a['gradient_ratio']:.4f} / {b['gradient_ratio']:.4f} |"
        )
    lines.extend(["", "## Radial PSNR", "", "| bin | A | B | Δ（B−A） |", "| --- | ---: | ---: | ---: |"])
    for row in report["radial_bins"]:
        lines.append(f"| {row['bin']} | {row['a_psnr']:.4f} | {row['b_psnr']:.4f} | {row['delta_b_minus_a_psnr']:.4f} |")
    lines.extend(["", "## Train-view versus held-out", "", "| split | frames | A PSNR | B PSNR | Δ（B−A） |", "| --- | ---: | ---: | ---: | ---: |"])
    for split, row in report["split_metrics"].items():
        lines.append(f"| {split} | {row['n_frames']} | {row['a_psnr']:.4f} | {row['b_psnr']:.4f} | {row['delta_b_minus_a_psnr']:.4f} |")
    return "\n".join(lines) + "\n"


def analyze_frontwide_pair(
    eval_dir_a: str, eval_dir_b: str, crops_json: str, out_dir: str, *, use_lpips: bool = True
) -> dict[str, Any]:
    """Compare front-wide renders in two eval directories and write a report."""
    crop_config = _load_crops(crops_json)
    camera_id = str(crop_config.get("camera_id", DEFAULT_CAMERA_ID))
    renders_a, gts_a = _load_frontwide_images(eval_dir_a, camera_id)
    renders_b, gts_b = _load_frontwide_images(eval_dir_b, camera_id)
    if len(renders_a) != len(renders_b) or len(gts_a) != len(gts_b):
        raise ValueError("A/B eval directories have different front-wide frame counts")
    for index, (gt_a, gt_b) in enumerate(zip(gts_a, gts_b, strict=True)):
        if gt_a.shape != gt_b.shape or not np.allclose(gt_a, gt_b, atol=1.0 / 255.0):
            raise ValueError(f"A/B GT mismatch at front-wide frame {index}")

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    lpips_score, lpips_status = _lpips_scorer(use_lpips)
    report: dict[str, Any] = {
        "camera_id": camera_id,
        "lpips_status": lpips_status,
        "crops": {},
        "radial_bins": _radial_report(renders_a, renders_b, gts_a),
        "split_metrics": _split_metrics(renders_a, renders_b, gts_a, crop_config.get("frame_splits", {})),
    }
    for crop in crop_config["crops"]:
        name = str(crop["name"])
        frame_index = int(crop["frame_index"])
        if not 0 <= frame_index < len(renders_a):
            raise ValueError(f"Crop {name} references unavailable front-wide frame {frame_index}")
        crop_a = _crop_image(renders_a[frame_index], crop)
        crop_b = _crop_image(renders_b[frame_index], crop)
        crop_gt = _crop_image(gts_a[frame_index], crop)
        a_metrics = _metric(crop_a, crop_gt, lpips_score)
        b_metrics = _metric(crop_b, crop_gt, lpips_score)
        report["crops"][name] = {
            "frame_index": frame_index,
            "split": crop.get("split", crop_config.get("frame_splits", {}).get(str(frame_index), "held_out")),
            "box": crop["box"],
            "a": a_metrics,
            "b": b_metrics,
            "delta_b_minus_a": _delta_b_minus_a(a_metrics, b_metrics),
        }
        _write_triptych(output / f"frontwide_crop_{name}.png", crop_a, crop_b, crop_gt)

    (output / "frontwide_ab_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "frontwide_ab_report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-a", required=True)
    parser.add_argument("--eval-b", required=True)
    parser.add_argument("--crops", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-lpips", action="store_true", help="Skip LPIPS when weights are unavailable")
    args = parser.parse_args()
    analyze_frontwide_pair(args.eval_a, args.eval_b, args.crops, args.out, use_lpips=not args.no_lpips)


if __name__ == "__main__":
    main()
