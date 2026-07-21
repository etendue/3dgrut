#!/usr/bin/env python3
"""Make deterministic center/periphery GT/P/F crop sheets for matched evals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def _run_root(root: Path) -> Path:
    metrics = sorted(root.rglob("metrics.json"))
    if len(metrics) != 1:
        raise RuntimeError(f"expected exactly one metrics.json below {root}, got {metrics}")
    return metrics[0].parent


def _camera_starts(metrics: dict) -> list[tuple[str, int]]:
    start = 0
    out = []
    for camera_id, values in metrics["per_camera"].items():
        out.append((camera_id, start))
        start += int(values["n_frames"])
    return out


def _crop_boxes(width: int, height: int, side: int) -> dict[str, tuple[int, int, int, int]]:
    if width < side or height < side:
        raise ValueError(f"image {width}x{height} is smaller than crop side {side}")
    return {
        "center": ((width - side) // 2, (height - side) // 2,
                   (width + side) // 2, (height + side) // 2),
        "periphery": (width - side, height - side, width, height),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-dir", type=Path, required=True)
    parser.add_argument("--f-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--crop-side", type=int, default=512)
    args = parser.parse_args()

    p_root = _run_root(args.p_dir)
    f_root = _run_root(args.f_dir)
    p_metrics = json.loads((p_root / "metrics.json").read_text())
    starts = _camera_starts(p_metrics)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for camera_id, frame_idx in starts:
        name = f"{frame_idx:05d}.png"
        gt = Image.open(p_root / "ours_30000" / "gt" / name).convert("RGB")
        p = Image.open(p_root / "ours_30000" / "renders" / name).convert("RGB")
        f = Image.open(f_root / "ours_30000" / "renders" / name).convert("RGB")
        boxes = _crop_boxes(*gt.size, args.crop_side)
        canvas = Image.new("RGB", (args.crop_side * 3, args.crop_side * 2 + 56), "white")
        draw = ImageDraw.Draw(canvas)
        for col, label in enumerate(("GT", "Pinhole", "FTheta")):
            draw.text((col * args.crop_side + 8, 4), label, fill="black")
        for row, (region, box) in enumerate(boxes.items()):
            draw.text((4, 28 + row * args.crop_side), region, fill="black")
            for col, image in enumerate((gt, p, f)):
                canvas.paste(image.crop(box), (col * args.crop_side, 28 + row * args.crop_side))
        canvas.save(args.out_dir / f"{camera_id}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
