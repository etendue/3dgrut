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


def _as_image4(value: torch.Tensor, channels: int | None = None) -> torch.Tensor:
    """Normalize an image-like tensor to [B,H,W,C]."""
    if value.dim() == 2:
        value = value[None, ..., None]
    elif value.dim() == 3:
        value = value.unsqueeze(0)
    if value.dim() != 4:
        raise ValueError(f"Expected image tensor with 2-4 dims, got {tuple(value.shape)}")
    if channels is not None and value.shape[-1] != channels:
        raise ValueError(f"Expected {channels} channels, got {tuple(value.shape)}")
    return value


def _eroded_road_mask(road_mask: torch.Tensor, erosion_px: int) -> torch.Tensor:
    mask = _as_image4(road_mask).bool()
    if erosion_px < 0:
        raise ValueError("erosion_px must be non-negative")
    if not erosion_px:
        return mask
    x = mask.permute(0, 3, 1, 2).float()
    return (
        F.avg_pool2d(
            x,
            2 * erosion_px + 1,
            stride=1,
            padding=erosion_px,
            count_include_pad=True,
        )
        == 1
    ).permute(0, 2, 3, 1)


def _duplicate_samples(
    bg_alpha: torch.Tensor,
    road_alpha: torch.Tensor,
    bg_rgb: torch.Tensor,
    road_rgb: torch.Tensor,
    gt_rgb: torch.Tensor | None,
    interior: torch.Tensor,
    *,
    rgb_temperature: float,
    alpha_min: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return continuous duplicate scores and their appearance errors.

    The score deliberately has no front/behind depth predicate.  It is high
    only when both layers cover an interior road pixel and background-only
    RGB resembles both the road-only render and GT.  This captures coplanar
    or slightly-behind duplicate road while down-weighting ordinary distant
    scene geometry exposed after the road layer is disabled.
    """
    if not rgb_temperature > 0:
        raise ValueError("rgb_temperature must be positive")
    bg_alpha = _as_image4(bg_alpha, 1)
    road_alpha = _as_image4(road_alpha, 1)
    bg_rgb = _as_image4(bg_rgb, 3)
    road_rgb = _as_image4(road_rgb, 3)
    gt_rgb = _as_image4(gt_rgb, 3) if gt_rgb is not None else None
    finite = (
        torch.isfinite(bg_alpha)
        & torch.isfinite(road_alpha)
        & torch.isfinite(bg_rgb).all(dim=-1, keepdim=True)
        & torch.isfinite(road_rgb).all(dim=-1, keepdim=True)
    )
    bg_to_road = (bg_rgb - road_rgb).abs().mean(dim=-1, keepdim=True)
    appearance_error = bg_to_road
    if gt_rgb is not None:
        finite &= torch.isfinite(gt_rgb).all(dim=-1, keepdim=True)
        bg_to_gt = (bg_rgb - gt_rgb).abs().mean(dim=-1, keepdim=True)
        # Requiring agreement with both references avoids treating a distant
        # grey facade that merely resembles one road render as duplication.
        appearance_error = torch.maximum(bg_to_road, bg_to_gt)
    valid = (
        interior
        & finite
        & (road_alpha > float(alpha_min))
    )
    appearance_weight = torch.exp(-appearance_error / float(rgb_temperature))
    score = bg_alpha * road_alpha * appearance_weight
    return score[valid], appearance_error[valid]


def _histogram_quantile(histogram: torch.Tensor, q: float, low: float, high: float) -> float:
    """Approximate a global quantile from a pixel-weighted histogram."""
    total = float(histogram.sum())
    if total == 0:
        return float("nan")
    target = q * total
    index = int(torch.searchsorted(histogram.cumsum(0), torch.tensor(target)).item())
    index = min(index, histogram.numel() - 1)
    return low + (index + 0.5) * (high - low) / histogram.numel()

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
    bg_rgb=None,
    gt_rgb=None,
    duplicate_rgb_temperature=0.1,
    duplicate_score_threshold=0.05,
    duplicate_alpha_min=0.005,
):
    mask = _as_image4(road_mask).bool()
    interior = _eroded_road_mask(mask, erosion_px)
    def stat(x, q=None):
        values=x[interior.expand_as(x)]
        if not values.numel(): return float('nan')
        return float(values.mean() if q is None else torch.quantile(values, q))
    outside = ~mask
    result = {"n_valid_px": int(interior.sum()), "bg_on_road_alpha_mean":stat(bg_alpha), "bg_on_road_alpha_p50":stat(bg_alpha,.5), "bg_on_road_alpha_p90":stat(bg_alpha,.9), "road_coverage_p10":stat(road_alpha,.1), "road_coverage_p50":stat(road_alpha,.5), "road_coverage_mean":stat(road_alpha), "road_outside_alpha_mean": float(road_alpha[outside.expand_as(road_alpha)].mean()) if outside.any() else float('nan'), "sky_on_road_energy": stat(sky_contrib.abs().mean(dim=-1,keepdim=True))}
    if bg_rgb is not None:
        duplicate_values, appearance_errors = _duplicate_samples(
            bg_alpha,
            road_alpha,
            bg_rgb,
            road_rgb,
            gt_rgb,
            interior,
            rgb_temperature=float(duplicate_rgb_temperature),
            alpha_min=float(duplicate_alpha_min),
        )
        result.update(
            n_duplicate_valid_px=int(duplicate_values.numel()),
            bg_road_duplicate_alpha_mean=(
                float(duplicate_values.mean()) if duplicate_values.numel() else float("nan")
            ),
            bg_road_duplicate_alpha_p90=(
                float(torch.quantile(duplicate_values, 0.9))
                if duplicate_values.numel()
                else float("nan")
            ),
            bg_road_duplicate_pixel_fraction=(
                float((duplicate_values >= float(duplicate_score_threshold)).float().mean())
                if duplicate_values.numel()
                else float("nan")
            ),
            bg_road_duplicate_rgb_mae_p50=(
                float(torch.quantile(appearance_errors, 0.5))
                if appearance_errors.numel()
                else float("nan")
            ),
        )
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


def _load_rgb(path: Path) -> torch.Tensor:
    """Load a lossless ownership RGB dump into [1,H,W,3]."""
    array = np.load(path).astype(np.float32, copy=False)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected RGB [H,W,3], got {array.shape} from {path}")
    return torch.from_numpy(array)[None]


def summarize_ownership_dirs(
    bg_ownership: Path,
    road_ownership: Path,
    sky_ownership: Path,
    erosion_px: int = 1,
    duplicate_rgb_temperature: float = 0.1,
    duplicate_score_threshold: float = 0.05,
) -> dict:
    """Pair ownership dumps and aggregate duplicate metrics by pixel."""
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
    histogram_bins = 4096
    duplicate_histogram = torch.zeros(histogram_bins, dtype=torch.float64)
    duplicate_error_histogram = torch.zeros(histogram_bins, dtype=torch.float64)
    duplicate_sum = 0.0
    duplicate_count = 0
    duplicate_above_threshold = 0
    for stem in frame_stems:
        bg_depth_path = bg_ownership / f"{stem}_depth.npy"
        road_depth_path = road_ownership / f"{stem}_depth.npy"
        have_depth = bg_depth_path.is_file() and road_depth_path.is_file()
        bg_alpha = _load_png(bg_ownership / f"{stem}_alpha.png", 1)
        road_alpha = _load_png(road_ownership / f"{stem}_alpha.png", 1)
        bg_rgb_path = bg_ownership / f"{stem}_rgb.npy"
        road_rgb_path = road_ownership / f"{stem}_rgb.npy"
        gt_rgb_path = bg_ownership / f"{stem}_gt.npy"
        have_rgb = bg_rgb_path.is_file() and road_rgb_path.is_file()
        bg_rgb = _load_rgb(bg_rgb_path) if have_rgb else None
        road_rgb = (
            _load_rgb(road_rgb_path)
            if have_rgb
            else torch.zeros_like(road_alpha).repeat(1, 1, 1, 3)
        )
        gt_rgb = _load_rgb(gt_rgb_path) if have_rgb and gt_rgb_path.is_file() else None
        road_mask = _load_png(bg_ownership / f"{stem}_roadmask.png", 1) > 0.5
        records.append(compute_ownership_metrics(
            bg_alpha,
            road_alpha,
            road_rgb,
            _load_png(sky_ownership / f"{stem}_sky.png", 3),
            road_mask,
            erosion_px=erosion_px,
            bg_depth=_load_depth(bg_depth_path) if have_depth else None,
            road_depth=_load_depth(road_depth_path) if have_depth else None,
            bg_rgb=bg_rgb,
            gt_rgb=gt_rgb,
            duplicate_rgb_temperature=duplicate_rgb_temperature,
            duplicate_score_threshold=duplicate_score_threshold,
        ))
        if have_rgb:
            duplicate_values, duplicate_errors = _duplicate_samples(
                bg_alpha,
                road_alpha,
                bg_rgb,
                road_rgb,
                gt_rgb,
                _eroded_road_mask(road_mask, erosion_px),
                rgb_temperature=duplicate_rgb_temperature,
                alpha_min=0.005,
            )
            duplicate_sum += float(duplicate_values.double().sum())
            duplicate_count += int(duplicate_values.numel())
            duplicate_above_threshold += int(
                (duplicate_values >= float(duplicate_score_threshold)).sum()
            )
            duplicate_histogram += torch.histc(
                duplicate_values.float(), bins=histogram_bins, min=0.0, max=1.0
            ).double()
            duplicate_error_histogram += torch.histc(
                duplicate_errors.float().clamp(0.0, 1.0),
                bins=histogram_bins,
                min=0.0,
                max=1.0,
            ).double()
    count_keys = {"n_valid_px", "n_depth_valid_px", "n_duplicate_valid_px"}
    scalar_keys = [key for key in records[0] if key not in count_keys]
    def nanmean(values: list[float]) -> float:
        finite = [value for value in values if not np.isnan(value)]
        return float(np.mean(finite)) if finite else float("nan")

    summary = {key: nanmean([record[key] for record in records]) for key in scalar_keys}
    summary.update(
        n_frames=len(records),
        n_valid_px_total=sum(record["n_valid_px"] for record in records),
        n_depth_valid_px_total=sum(record.get("n_depth_valid_px", 0) for record in records),
        erosion_px=erosion_px,
    )
    if duplicate_count:
        summary.update(
            n_duplicate_valid_px_total=duplicate_count,
            bg_road_duplicate_alpha_mean=duplicate_sum / duplicate_count,
            bg_road_duplicate_alpha_p90=_histogram_quantile(
                duplicate_histogram, 0.9, 0.0, 1.0
            ),
            bg_road_duplicate_pixel_fraction=duplicate_above_threshold / duplicate_count,
            bg_road_duplicate_rgb_mae_p50=_histogram_quantile(
                duplicate_error_histogram, 0.5, 0.0, 1.0
            ),
            duplicate_rgb_temperature=float(duplicate_rgb_temperature),
            duplicate_score_threshold=float(duplicate_score_threshold),
        )
    return {"summary": summary, "per_frame": records}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate MCRO ownership debug dumps")
    parser.add_argument("--bg-ownership", type=Path, required=True)
    parser.add_argument("--road-ownership", type=Path, required=True)
    parser.add_argument("--sky-ownership", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--erosion-px", type=int, default=1)
    parser.add_argument("--duplicate-rgb-temperature", type=float, default=0.1)
    parser.add_argument("--duplicate-score-threshold", type=float, default=0.05)
    args = parser.parse_args()
    report = summarize_ownership_dirs(
        args.bg_ownership,
        args.road_ownership,
        args.sky_ownership,
        args.erosion_px,
        args.duplicate_rgb_temperature,
        args.duplicate_score_threshold,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(json.dumps(report["summary"], indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
