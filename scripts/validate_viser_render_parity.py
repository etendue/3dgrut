#!/usr/bin/env python3
"""Compare native eval images with the viser camera/render contract.

The CPU helpers are standalone and unit-tested. GPU execution loads exact
native-camera renders and viewer-contract renders from directories produced by
the same checkpoint/camera/timestamp protocol, then reports radial parity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def radial_region_masks(
    height: int,
    width: int,
    *,
    center_radius: float = 0.35,
    peripheral_inner: float = 0.65,
    peripheral_outer: float = 0.95,
) -> dict[str, np.ndarray]:
    """Return normalized-radius center/peripheral masks around image center."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if not (0.0 < center_radius < peripheral_inner < peripheral_outer <= 1.0):
        raise ValueError("require 0 < center < peripheral_inner < peripheral_outer <= 1")
    yy, xx = np.mgrid[:height, :width]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    half_diag = float(np.hypot(max(cx, 0.5), max(cy, 0.5)))
    radius = np.hypot(xx - cx, yy - cy) / half_diag
    return {
        "full": np.ones((height, width), dtype=bool),
        "center": radius <= center_radius,
        "peripheral": (radius >= peripheral_inner) & (radius <= peripheral_outer),
    }


def _region_mae(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    diff = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    return float(diff[mask].mean())


def _psnr_from_mae_arrays(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    diff = candidate.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.square(diff[mask]).mean())
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((255.0**2) / mse))


def compute_region_metrics(
    reference_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    masks: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Compute full/center/peripheral MAE and PSNR for uint8 RGB images."""
    ref = np.asarray(reference_rgb)
    cand = np.asarray(candidate_rgb)
    if ref.shape != cand.shape or ref.ndim != 3 or ref.shape[2] != 3:
        raise ValueError(f"images must have the same HxWx3 shape, got {ref.shape}/{cand.shape}")
    if masks is None:
        masks = radial_region_masks(ref.shape[0], ref.shape[1])
    out: dict[str, float] = {}
    for name in ("full", "center", "peripheral"):
        mask = np.asarray(masks[name], dtype=bool)
        if mask.shape != ref.shape[:2] or not mask.any():
            raise ValueError(f"mask '{name}' must be non-empty with shape {ref.shape[:2]}")
        out[f"{name}_mae"] = _region_mae(ref, cand, mask)
        out[f"{name}_psnr"] = _psnr_from_mae_arrays(ref, cand, mask)
    return out


def _json_metric(value: float) -> float:
    return 99.0 if not np.isfinite(value) else float(value)


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def compare_image_pairs(native_dir: Path, viewer_dir: Path, output_dir: Path) -> dict:
    """Compare matching ``<camera>/<frame>.png`` trees and write heatmaps."""
    from PIL import Image

    metrics: dict[str, dict] = {}
    for native_path in sorted(native_dir.glob("*/*.png")):
        camera_id = native_path.parent.name
        frame_key = native_path.stem
        viewer_path = viewer_dir / camera_id / native_path.name
        if not viewer_path.exists():
            raise FileNotFoundError(f"missing viewer image for {native_path}: {viewer_path}")
        ref = _load_rgb(native_path)
        cand = _load_rgb(viewer_path)
        values = compute_region_metrics(ref, cand)
        metrics.setdefault(camera_id, {})[frame_key] = {
            key: _json_metric(value) for key, value in values.items()
        }
        diff = np.clip(np.abs(cand.astype(np.int16) - ref.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
        pair_dir = output_dir / camera_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(diff).save(pair_dir / f"{frame_key}_absdiff_x4.png")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare native eval and viser render-contract images")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint used to produce both render trees")
    parser.add_argument("--dataset_path", type=Path, help="NCore manifest path")
    parser.add_argument("--config_name", default="apps/ncore_3dgut_mcmc_multilayer_inceptio")
    parser.add_argument("--camera_id", action="append", default=[], help="camera to validate; repeatable")
    parser.add_argument("--frame_index", action="append", type=int, default=[], help="frame index; repeatable")
    parser.add_argument("--renderer", choices=["3dgrt", "3dgut"], default="3dgrt")
    parser.add_argument("--native_dir", type=Path, help="tree: <camera>/<frame>.png from native eval")
    parser.add_argument("--viewer_dir", type=Path, help="tree: <camera>/<frame>.png from viewer contract")
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.native_dir is None or args.viewer_dir is None:
        raise SystemExit(
            "GPU render collection is explicit: provide --native_dir and --viewer_dir "
            "generated for the same checkpoint/camera/timestamps."
        )
    metrics = compare_image_pairs(args.native_dir, args.viewer_dir, args.output_dir)
    out = args.output_dir / "parity_metrics.json"
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
