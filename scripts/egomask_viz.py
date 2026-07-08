# SPDX-License-Identifier: Apache-2.0
"""Annotation-loop visualization for P0.3 visual-polygon static ego masks.

Two renderers:
  * ``render_grid_reference`` — raw first frame + 100px coordinate grid, so
    Claude can read off polygon vertex pixel coordinates.
  * ``render_resolved_overlay`` — overlays the *resolve*-dilated ego mask (red =
    pixels that will be masked out of supervision) on the real frame, for
    visual acceptance (same dilation semantics as ``resolve_ego_valid_mask``).

Runs on inceptio (raw ncore4 itars live there). Reads the raw camera frame from
``<clip_dir>/*.ncore4-<camera_id>.zarr.itar`` at ``cameras/<cam>/frames/<ts>/image``.

Import is dual-path so this can run either inside the repo (``from
threedgrut.datasets.aux_readers``) or standalone in a scratch dir alongside a
copied ``aux_readers.py`` (``from aux_readers``) — the latter avoids the
``threedgrut.datasets.__init__`` cascade on the GPU host.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# the 10 cameras that get a static mask (front_tele / front_standard skipped —
# ego vehicle not in frame)
CAMERAS_TO_MASK = [
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_left_wide_90fov",
    "camera_right_wide_90fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
    "camera_front_wide_120fov",
    "camera_back_rear_wide_90fov",
    "camera_front_fisheye",
    "camera_back_rear_fisheye",
]


def _get_open_itar_zarr():
    # Prefer a scratch-dir copy (e.g. /tmp/egoviz/aux_readers.py) so the GPU
    # host can run this without triggering the threedgrut.datasets.__init__
    # cascade; fall back to the in-repo package when run inside the repo
    # (scripts/ has no aux_readers.py, so the first import fails cleanly there).
    try:
        from aux_readers import _open_itar_zarr
    except ImportError:
        from threedgrut.datasets.aux_readers import _open_itar_zarr
    return _open_itar_zarr


def _load_font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def read_first_frame_rgb(clip_dir, camera_id: str) -> Image.Image:
    """Decode the first frame of ``<clip_dir>/*.ncore4-<camera_id>.zarr.itar``."""
    open_itar_zarr = _get_open_itar_zarr()
    matches = sorted(Path(clip_dir).glob(f"*.ncore4-{camera_id}.zarr.itar"))
    if not matches:
        raise FileNotFoundError(f"no raw itar for '{camera_id}' in {clip_dir}")
    g = open_itar_zarr(matches[0])
    frames = g[f"cameras/{camera_id}/frames"]
    ts = sorted(frames.group_keys(), key=lambda x: int(x))[0]
    b = bytes(frames[ts]["image"][()])
    return Image.open(io.BytesIO(b)).convert("RGB")


def render_grid_reference(clip_dir, camera_id: str, out_png, grid: int = 100) -> None:
    """Full-resolution raw frame + coordinate grid every ``grid`` px (x labels
    along the top, y labels down the left). No downscale — pixel coords are real.
    """
    img = read_first_frame_rgb(clip_dir, camera_id)
    W, H = img.size
    d = ImageDraw.Draw(img)
    font = _load_font(18)
    line = (255, 255, 0)
    for x in range(0, W, grid):
        d.line([(x, 0), (x, H)], fill=line, width=1)
        d.text((x + 2, 2), str(x), fill=line, font=font)
    for y in range(0, H, grid):
        d.line([(0, y), (W, y)], fill=line, width=1)
        d.text((2, y + 2), str(y), fill=line, font=font)
    img.save(out_png)


def render_resolved_overlay(clip_dir, camera_id: str, mask: np.ndarray, out_png, dilation_iters: int = 30) -> None:
    """Overlay the resolve-dilated ego mask (red) on the real frame.

    ``mask`` is the raw ego mask (True = ego). We dilate it exactly like
    ``resolve_ego_valid_mask`` (scipy treats iterations < 1 as dilate-to-
    convergence, so iterations == 0 means no dilation), then paint the masked
    region red. valid = logical_not(dilated).
    """
    from scipy import ndimage

    img = read_first_frame_rgb(clip_dir, camera_id)
    W, H = img.size
    m = np.asarray(mask, dtype=bool)
    if m.shape != (H, W):
        m = np.asarray(Image.fromarray(m.astype("uint8") * 255).resize((W, H), Image.NEAREST)) > 127
    if dilation_iters and dilation_iters >= 1:
        m = ndimage.binary_dilation(m, iterations=dilation_iters)
    arr = np.asarray(img).astype(float)
    arr[m] = arr[m] * 0.30 + np.array([255.0, 30.0, 30.0]) * 0.70
    Image.fromarray(arr.astype("uint8")).save(out_png)


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--camera-id", default=None, help="single camera; default = all 10 masked cameras")
    ap.add_argument("--mode", choices=["grid"], default="grid", help="overlay is driven programmatically in T5")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--grid", type=int, default=100)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cams = [args.camera_id] if args.camera_id else CAMERAS_TO_MASK
    for cam in cams:
        out = out_dir / f"grid_{cam}.png"
        render_grid_reference(args.clip_dir, cam, out, grid=args.grid)
        print(f"saved {out}")


if __name__ == "__main__":
    _main()
