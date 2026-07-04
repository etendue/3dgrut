#!/usr/bin/env python3
"""B2 alignment proof — annotate cuboid wireframes with class+tid labels.

Loads ckpt + uses the production FthetaForwardProjector to project every
active cuboid at frame_idx=0 onto a backdrop image, then draws:
  - 12 cuboid edges as a polyline (instance color)
  - class + tid text label at the projected bbox center
  - a thin circle marker at the projected cuboid pose translation

Output is a single annotated PNG suitable for visual verification of
B2's FTheta cuboid overlay vs the Gaussian backdrop.

Usage (ThinkPad):
    python scripts/annotate_b2_alignment.py \\
        --ckpt /home/yusun/work/ckpts/bug4_v2_full_30k/ckpt_with_ftheta_v2.pt \\
        --backdrop /tmp/b2dump/b2_backdrop.png \\
        --out /tmp/b2dump/b2_annotated.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from threedgrut_playground.utils.cuboid import cuboid_world_edges, instance_color
from threedgrut_playground.utils.ftheta_projector import FthetaForwardProjector
from threedgrut_playground.utils.viz4d_metadata import FourDMetadata


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--backdrop", required=True, type=Path, help="PNG of the Gaussian backdrop (from B2_DUMP_DIR)")
    ap.add_argument(
        "--c2w",
        type=Path,
        default=None,
        help="npy of the c2w used to render the backdrop. If " "absent, falls back to meta.ego_poses_c2w[frame_idx].",
    )
    ap.add_argument(
        "--t-us",
        type=Path,
        default=None,
        help="txt file with the t_us at which the backdrop was "
        "rendered. Used to look up the right frame_idx for "
        "active tracks.",
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--frame-idx", type=int, default=0)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    meta = FourDMetadata.from_ckpt(ckpt)
    if meta is None or not meta.has_ftheta():
        print("[annotate] ERROR: ckpt missing viz_4d/FTheta")
        return 1

    ftheta = meta.ego_primary_intrinsics_ftheta
    proj = FthetaForwardProjector(ftheta)

    # Backdrop image
    bg = Image.open(args.backdrop).convert("RGBA")
    W, H = bg.size
    overlay = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    # Prefer the c2w that was actually used to render the backdrop;
    # fall back to meta.ego_poses_c2w[frame_idx] for offline-only runs.
    if args.c2w is not None and args.c2w.exists():
        c2w_viser = np.load(args.c2w).astype(np.float64)
        print(f"[annotate] using c2w from {args.c2w}  trans={c2w_viser[:3,3].tolist()}")
    else:
        c2w_viser = meta.ego_poses_c2w[args.frame_idx].astype(np.float64)
        print(f"[annotate] using meta.ego_poses_c2w[{args.frame_idx}]")

    # Look up frame_idx from dumped t_us so cuboid poses match backdrop time.
    if args.t_us is not None and args.t_us.exists():
        t_us = int(args.t_us.read_text().strip())
        frame_idx = meta.lookup_frame_idx(t_us)
        print(f"[annotate] using t_us={t_us} -> frame_idx={frame_idx}")
    else:
        frame_idx = args.frame_idx

    active = meta.active_tracks_at(frame_idx)
    print(f"[annotate] {len(active)} active tracks at frame {frame_idx}")

    n_visible_boxes = 0
    for tid in active:
        t = meta.tracks[tid]
        pose = t["poses"][frame_idx]
        size = t["size"]
        klass = t["class"]
        if size is None or pose is None:
            continue

        # 12 edges in world coords -> project via FthetaForwardProjector
        edges = cuboid_world_edges(pose, size)  # (12, 2, 3)
        verts = edges.reshape(-1, 3).astype(np.float64)  # (24, 3)
        uv, vis = proj.project_points(verts, c2w_viser)
        if vis.sum() < 4:
            continue
        n_visible_boxes += 1

        # Per-cuboid color (RGBA, 90% opaque)
        r, g, b = instance_color(tid)
        col = (int(r * 255), int(g * 255), int(b * 255), 230)
        white = (255, 255, 255, 255)
        black = (0, 0, 0, 255)

        # Draw each edge segment if both endpoints visible
        for ei in range(12):
            i0, i1 = 2 * ei, 2 * ei + 1
            if vis[i0] and vis[i1]:
                draw.line([(uv[i0, 0], uv[i0, 1]), (uv[i1, 0], uv[i1, 1])], fill=col, width=2)

        # Label at bbox center (mean of visible vertices)
        visible_uv = uv[vis]
        cx = float(visible_uv[:, 0].mean())
        cy = float(visible_uv[:, 1].mean())

        # Compute cam-frame z to filter very distant boxes from labels (clutter)
        c2w_cv = c2w_viser @ np.diag([1.0, 1.0, -1.0, 1.0])
        w2c = np.linalg.inv(c2w_cv)
        center_h = np.concatenate([pose[:3, 3].astype(np.float64), [1.0]])
        z_cam = float((w2c @ center_h)[2])

        label = f"{klass}#{tid}  z={z_cam:.0f}m"
        # Draw label with black outline + white fill for readability
        tx, ty = cx + 4, cy - 18
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                draw.text((tx + dx, ty + dy), label, fill=black, font=font)
        draw.text((tx, ty), label, fill=white, font=font)

        # Tiny crosshair at the cuboid center
        draw.line([(cx - 4, cy), (cx + 4, cy)], fill=white, width=1)
        draw.line([(cx, cy - 4), (cx, cy + 4)], fill=white, width=1)

    out = Image.alpha_composite(bg, overlay).convert("RGB")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out, "PNG")
    print(f"[annotate] wrote {args.out}  ({n_visible_boxes} boxes labeled)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
