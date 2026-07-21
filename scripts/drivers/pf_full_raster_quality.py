#!/usr/bin/env python3
"""Attach full-raster gradient correlation and edge sharpness to a P/F report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _run_root(path: Path) -> Path:
    found = sorted(path.rglob("metrics.json"))
    if len(found) != 1:
        raise RuntimeError(f"expected one metrics.json below {path}, got {found}")
    return found[0].parent


def _camera_frames(metrics: dict) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for cid, values in metrics["per_camera"].items():
        out.extend([(cid, i) for i in range(int(values["n_frames"]))])
    return out


def _rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _gradient_magnitude(rgb: np.ndarray) -> np.ndarray:
    luma = rgb @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    dy, dx = np.gradient(luma)
    return np.hypot(dx, dy)


def _empty() -> dict[str, float]:
    return {"n": 0.0, "sum_p": 0.0, "sum_g": 0.0, "sum_p2": 0.0, "sum_g2": 0.0, "sum_pg": 0.0}


def _add(stats: dict[str, float], pred: np.ndarray, gt: np.ndarray) -> None:
    stats["n"] += float(pred.size)
    stats["sum_p"] += float(pred.sum())
    stats["sum_g"] += float(gt.sum())
    stats["sum_p2"] += float(np.square(pred).sum())
    stats["sum_g2"] += float(np.square(gt).sum())
    stats["sum_pg"] += float((pred * gt).sum())


def _finish(stats: dict[str, float]) -> dict[str, float | None]:
    n = stats["n"]
    cov = stats["sum_pg"] - stats["sum_p"] * stats["sum_g"] / n
    vp = stats["sum_p2"] - stats["sum_p"] ** 2 / n
    vg = stats["sum_g2"] - stats["sum_g"] ** 2 / n
    corr = cov / np.sqrt(vp * vg) if vp > 0.0 and vg > 0.0 else None
    edge = stats["sum_p"] / n
    edge_gt = stats["sum_g"] / n
    return {
        "gradient_correlation": None if corr is None else float(corr),
        "edge_sharpness": float(edge),
        "edge_sharpness_gt": float(edge_gt),
        "edge_sharpness_ratio": float(edge / edge_gt) if edge_gt > 0.0 else None,
    }


def _mean(rows: list[dict], key: str, *, frame_weighted: bool) -> float | None:
    vals = [r[key] for r in rows if r[key] is not None]
    if not vals:
        return None
    if not frame_weighted:
        return float(np.mean(vals))
    weights = np.asarray([r["n_frames"] for r in rows if r[key] is not None], dtype=np.float64)
    return float(np.average(vals, weights=weights))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-dir", type=Path, required=True)
    parser.add_argument("--f-dir", type=Path, required=True)
    parser.add_argument("--radial-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    p_root, f_root = _run_root(args.p_dir), _run_root(args.f_dir)
    p_metrics = json.loads((p_root / "metrics.json").read_text())
    f_metrics = json.loads((f_root / "metrics.json").read_text())
    radial = json.loads(args.radial_report.read_text())
    frame_map = _camera_frames(p_metrics)
    if frame_map != _camera_frames(f_metrics):
        raise RuntimeError("P/F camera frame mapping differs")

    stats = {"P": {}, "F": {}}
    for cid, _ in frame_map:
        stats["P"].setdefault(cid, _empty())
        stats["F"].setdefault(cid, _empty())
    for index, (cid, _) in enumerate(frame_map):
        name = f"{index:05d}.png"
        gt = _rgb(p_root / "ours_30000" / "gt" / name)
        if not np.array_equal(gt, _rgb(f_root / "ours_30000" / "gt" / name)):
            raise RuntimeError(f"P/F GT differs at {name}")
        gt_grad = _gradient_magnitude(gt)
        _add(stats["P"][cid], _gradient_magnitude(_rgb(p_root / "ours_30000" / "renders" / name)), gt_grad)
        _add(stats["F"][cid], _gradient_magnitude(_rgb(f_root / "ours_30000" / "renders" / name)), gt_grad)

    rows: list[dict] = []
    for cid, p_values in p_metrics["per_camera"].items():
        r = radial["arms"]["P"]["per_camera"][cid]
        f_r = radial["arms"]["F"]["per_camera"][cid]
        p_grad, f_grad = _finish(stats["P"][cid]), _finish(stats["F"][cid])
        rows.append({
            "camera_id": cid,
            "n_frames": int(p_values["n_frames"]),
            "pixel_coverage_full_raster": 1.0,
            "psnr_full": {"P": r["full_frame"]["psnr"], "F": f_r["full_frame"]["psnr"]},
            "psnr_center": {"P": r["r<0.5"]["psnr"], "F": f_r["r<0.5"]["psnr"]},
            "psnr_periphery": {"P": r["r>=0.9"]["psnr"], "F": f_r["r>=0.9"]["psnr"]},
            "center_periphery_gap": {"P": r["r<0.5"]["psnr"] - r["r>=0.9"]["psnr"], "F": f_r["r<0.5"]["psnr"] - f_r["r>=0.9"]["psnr"]},
            "ssim": {"P": p_values["mean_ssim"], "F": f_metrics["per_camera"][cid]["mean_ssim"]},
            "lpips": {"P": p_values["mean_lpips"], "F": f_metrics["per_camera"][cid]["mean_lpips"]},
            "gradient_correlation": {"P": p_grad["gradient_correlation"], "F": f_grad["gradient_correlation"]},
            "edge_sharpness": {"P": p_grad["edge_sharpness"], "F": f_grad["edge_sharpness"]},
            "edge_sharpness_ratio": {"P": p_grad["edge_sharpness_ratio"], "F": f_grad["edge_sharpness_ratio"]},
        })

    aggregate: dict[str, dict[str, dict[str, float | None]]] = {}
    for weighting in ("macro", "frame_weighted"):
        aggregate[weighting] = {}
        for metric in ("psnr_full", "psnr_center", "psnr_periphery", "ssim", "lpips", "gradient_correlation", "edge_sharpness", "edge_sharpness_ratio"):
            aggregate[weighting][metric] = {
                arm: _mean([{"n_frames": row["n_frames"], metric: row[metric][arm]} for row in rows], metric, frame_weighted=weighting == "frame_weighted")
                for arm in ("P", "F")
            }
            p, f = aggregate[weighting][metric]["P"], aggregate[weighting][metric]["F"]
            aggregate[weighting][metric]["F_minus_P"] = None if p is None or f is None else f - p

    report = {"definition": {"raster": "full 1920x1080 raster", "center": "r < 0.5", "periphery": "r >= 0.9", "gradient": "luma finite-difference magnitude"}, "per_camera": rows, "aggregate": aggregate}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
