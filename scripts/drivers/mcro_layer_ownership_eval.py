#!/usr/bin/env python3
"""Layer ownership metrics and CLI for MCRO read-only checkpoint analysis."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def compute_ownership_metrics(
    bg_alpha,
    road_alpha,
    road_rgb,
    sky_contrib,
    road_mask,
    sky_mask=None,
    erosion_px=1,
    band_px=1,
    bg_depth=None,
    road_depth=None,
    foreground_margin_m=0.1,
):
    mask = road_mask.bool()
    if mask.dim() == 2: mask = mask[None, ..., None]
    if mask.dim() == 3: mask = mask.unsqueeze(-1)
    interior = mask
    if erosion_px:
        x = mask.permute(0,3,1,2).float()
        interior = (F.avg_pool2d(x, 2*erosion_px+1, stride=1, padding=erosion_px, count_include_pad=True)==1).permute(0,2,3,1)
    def stat(x, q=None):
        values=x[interior.expand_as(x)]
        if not values.numel(): return float('nan')
        return float(values.mean() if q is None else torch.quantile(values, q))
    outside = ~mask
    result = {"n_valid_px": int(interior.sum()), "bg_on_road_alpha_mean":stat(bg_alpha), "bg_on_road_alpha_p50":stat(bg_alpha,.5), "bg_on_road_alpha_p90":stat(bg_alpha,.9), "road_coverage_p10":stat(road_alpha,.1), "road_coverage_p50":stat(road_alpha,.5), "road_coverage_mean":stat(road_alpha), "road_outside_alpha_mean": float(road_alpha[outside.expand_as(road_alpha)].mean()) if outside.any() else float('nan'), "sky_on_road_energy": stat(sky_contrib.abs().mean(dim=-1,keepdim=True))}
    if bg_depth is not None and road_depth is not None:
        valid_depth = (
            torch.isfinite(bg_depth)
            & torch.isfinite(road_depth)
            & (bg_depth > 0)
            & (road_depth > 0)
            & (bg_alpha > 0.005)
            & (road_alpha > 0.005)
        )
        depth_domain = interior & valid_depth
        foreground = depth_domain & (bg_depth + float(foreground_margin_m) < road_depth)
        values = bg_alpha[depth_domain.expand_as(bg_alpha)]
        foreground_values = (bg_alpha * foreground)[depth_domain.expand_as(bg_alpha)]
        result.update(
            n_depth_valid_px=int(depth_domain.sum()),
            bg_in_front_of_road_fraction=(
                float(foreground[depth_domain].float().mean())
                if depth_domain.any()
                else float("nan")
            ),
            bg_in_front_of_road_alpha_mean=(
                float(foreground_values.mean()) if foreground_values.numel() else float("nan")
            ),
            bg_depth_minus_road_depth_p50=(
                float(torch.quantile((bg_depth - road_depth)[depth_domain], 0.5))
                if depth_domain.any()
                else float("nan")
            ),
            bg_depth_valid_alpha_mean=(
                float(values.mean()) if values.numel() else float("nan")
            ),
        )
    return result


def _load_png(path: Path, channels: int) -> torch.Tensor:
    """Load a debug PNG into [1,H,W,C] float32 in [0,1]."""
    image = Image.open(path).convert("RGB" if channels == 3 else "L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[..., None]
    return torch.from_numpy(array)[None]


def _load_depth(path: Path) -> torch.Tensor:
    """Load a lossless ownership depth dump into [1,H,W,1]."""
    array = np.load(path).astype(np.float32, copy=False)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3 or array.shape[-1] != 1:
        raise ValueError(f"Expected depth [H,W,1], got {array.shape} from {path}")
    return torch.from_numpy(array)[None]


def summarize_ownership_dirs(
    bg_ownership: Path,
    road_ownership: Path,
    sky_ownership: Path,
    erosion_px: int = 1,
) -> dict:
    """Pair ownership-dump images by frame and average their scalar metrics."""
    frame_stems = sorted(
        path.name.removesuffix("_alpha.png")
        for path in bg_ownership.glob("*_alpha.png")
        if (road_ownership / path.name).is_file()
        and (sky_ownership / f"{path.name.removesuffix('_alpha.png')}_sky.png").is_file()
        and (bg_ownership / f"{path.name.removesuffix('_alpha.png')}_roadmask.png").is_file()
    )
    if not frame_stems:
        raise ValueError("No matching ownership PNGs found across bg/road/sky directories")
    records = []
    for stem in frame_stems:
        bg_depth_path = bg_ownership / f"{stem}_depth.npy"
        road_depth_path = road_ownership / f"{stem}_depth.npy"
        have_depth = bg_depth_path.is_file() and road_depth_path.is_file()
        records.append(compute_ownership_metrics(
            _load_png(bg_ownership / f"{stem}_alpha.png", 1),
            _load_png(road_ownership / f"{stem}_alpha.png", 1),
            torch.zeros_like(_load_png(road_ownership / f"{stem}_alpha.png", 1)).repeat(1, 1, 1, 3),
            _load_png(sky_ownership / f"{stem}_sky.png", 3),
            _load_png(bg_ownership / f"{stem}_roadmask.png", 1) > 0.5,
            erosion_px=erosion_px,
            bg_depth=_load_depth(bg_depth_path) if have_depth else None,
            road_depth=_load_depth(road_depth_path) if have_depth else None,
        ))
    scalar_keys = [key for key in records[0] if key != "n_valid_px"]
    def nanmean(values: list[float]) -> float:
        finite = [value for value in values if not np.isnan(value)]
        return float(np.mean(finite)) if finite else float("nan")

    summary = {key: nanmean([record[key] for record in records]) for key in scalar_keys}
    summary.update(n_frames=len(records), n_valid_px_total=sum(record["n_valid_px"] for record in records), erosion_px=erosion_px)
    return {"summary": summary, "per_frame": records}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate MCRO ownership debug dumps")
    parser.add_argument("--bg-ownership", type=Path, required=True)
    parser.add_argument("--road-ownership", type=Path, required=True)
    parser.add_argument("--sky-ownership", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--erosion-px", type=int, default=1)
    args = parser.parse_args()
    report = summarize_ownership_dirs(
        args.bg_ownership, args.road_ownership, args.sky_ownership, args.erosion_px
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(json.dumps(report["summary"], indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
