#!/usr/bin/env python3
"""BUG-1 acceptance — headless replay of viser_gui_4d.update()'s render body.

Replays exactly what the viewer does per frame for a connected client:

    c2w   = meta.ego_pose_at(t_us)            # == get_c2w after Follow-Ego snap
    img   = viewer.fast_render(kaolin_cam)    # FTheta Gaussian backdrop
    specs = viewer._collect_overlay_layer_specs(t_us)   # BUG-1 fixed path
    out   = viewer._overlay_compositor.composite(img, specs, c2w)

ckpt viz_4d ego poses are stored in the viser convention (+Y down, +Z
backward) — the same convention get_c2w() returns and the B2-calibrated
FLIP_VISER_TO_OPENCV expects (see scripts/annotate_b2_alignment.py, which
uses meta.ego_poses_c2w as the documented fallback for the dumped c2w).
Feeding the SAME matrix to fast_render and composite reproduces the
browser-session alignment contract bit-for-bit.

Frame selection: every sampled frame has >=1 active cuboid; samples spread
uniformly over the active range and force-include the frame where the ego
heading changes fastest (turning segment) — the acceptance spec in
v3_plan_revised.md § 2.5 asks for straight + turning coverage.

Outputs ``frame<idx>_backdrop.png`` + ``frame<idx>_blended.png`` per frame.
Acceptance = in each blended PNG the wireframe hugs its Gaussian vehicle
(center / heading / size), curved edges at the fisheye periphery.

Usage (inceptio)::

    python scripts/verify_bug1_cuboid_overlay.py \
        --ckpt  ~/work/output/<run>/ours_30000/ckpt_30000.pt \
        --out_dir /tmp/bug1_verify --n_frames 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from PIL import Image


def _ego_yaw(meta, f: int) -> float:
    R = meta.ego_poses_c2w[min(f, meta.ego_poses_c2w.shape[0] - 1), :3, :3]
    fwd = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return float(np.arctan2(fwd[1], fwd[0]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset_path", default=None)
    ap.add_argument("--out_dir", default="/tmp/bug1_verify")
    ap.add_argument("--n_frames", type=int, default=5)
    ap.add_argument("--port", type=int, default=18099,
                    help="throwaway viser port (no client ever connects)")
    args = ap.parse_args()

    from kaolin.render.camera import Camera

    from threedgrut_playground.engine import Engine3DGRUT
    from threedgrut_playground.viser_gui_4d import Viser4DViewer, _load_metadata

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = str(_REPO_ROOT / "threedgrut_playground" / "assets")
    engine = Engine3DGRUT(
        gs_object=args.ckpt,
        mesh_assets_folder=assets,
        envmap_assets_folder=assets,
        default_config="apps/colmap_3dgrt.yaml",
    )
    ckpt = torch.load(args.ckpt, weights_only=False)
    meta = _load_metadata(ckpt, args.dataset_path, "apps/colmap_3dgrt.yaml")
    if meta is None or not meta.has_ftheta():
        print("[bug1-verify] ERROR: ckpt has no FTheta viz_4d block")
        return 1

    viewer = Viser4DViewer(port=args.port, engine=engine, metadata=meta)
    assert viewer._overlay_compositor is not None, "FTheta compositor missing"

    ts_arr = meta.tracks_camera_timestamps_us
    n = int(ts_arr.size)
    active_counts = [len(meta.active_tracks_at(f)) for f in range(n)]
    active_frames = [f for f in range(n) if active_counts[f] > 0]
    if not active_frames:
        print("[bug1-verify] ERROR: no frames with active cuboids")
        return 1

    # Uniform sample + force the max-heading-change (turning) frame.
    k = max(1, args.n_frames - 1)
    sel = {active_frames[int(i * (len(active_frames) - 1) / max(1, k - 1))]
           for i in range(k)} if k > 1 else {active_frames[0]}
    yaw = np.array([_ego_yaw(meta, f) for f in active_frames])
    dyaw = np.abs(np.diff(np.unwrap(yaw)))
    if dyaw.size:
        sel.add(active_frames[int(np.argmax(dyaw))])
    sel_frames = sorted(sel)
    print(f"[bug1-verify] {len(active_frames)} active frames; "
          f"sampling {sel_frames}")

    W, H = viewer.ftheta_render_wh
    for f in sel_frames:
        t_us = int(ts_arr[f])
        # Real dispatch: updates _t_us_current AND exercises the fixed
        # _update_active_cuboids skip path (must not raise headlessly).
        viewer._on_time_change(t_us, source="bug1-verify")
        c2w = meta.ego_pose_at(t_us).astype(np.float32)
        cam = Camera.from_args(
            view_matrix=torch.tensor(c2w),
            fov=float(meta.ego_primary_fov_y_rad),
            width=int(W), height=int(H),
            near=0.1, far=1000.0,
            dtype=torch.float32,
            device=engine.device,
        )
        img = viewer.fast_render(cam)
        specs = viewer._collect_overlay_layer_specs(t_us)
        n_layers = sum(1 for s in specs if "active_cuboids" in s.name)
        blended = viewer._overlay_compositor.composite(
            img, specs, c2w.astype(np.float64))
        overlay_px = int((blended != img).any(axis=-1).sum())
        Image.fromarray(img).save(out_dir / f"frame{f:04d}_backdrop.png")
        Image.fromarray(blended).save(out_dir / f"frame{f:04d}_blended.png")
        print(f"[bug1-verify] frame {f:4d} t_us={t_us}: "
              f"active={active_counts[f]} cuboid_layers={n_layers} "
              f"overlay_px={overlay_px}")
        if n_layers == 0:
            print("[bug1-verify] WARNING: no cuboid overlay layers — "
                  "fix not active?")
        if overlay_px == 0:
            print("[bug1-verify] WARNING: overlay drew zero pixels — "
                  "projection off-screen?")

    print(f"[bug1-verify] DONE → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
