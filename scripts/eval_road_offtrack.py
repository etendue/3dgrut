# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""From-checkpoint ROAD off-track KPI eval launcher (/loop road task).

Runs one unified eval on a trained checkpoint so baseline and every SOTA
ablation are scored through the exact same path (A/B comparable):

  * road-only render  — enabled_layer_names={"road"}: bg / dynamic-rigid /
    sky_envmap switched off so road_crop / lane / novel metrics measure the
    ROAD layer's own reconstruction, not bg over-rendering filling the road.
  * novel-view off-track sweep — lateral 3m/6m + yaw 10/30/60deg (the task's
    required translation + rotation gates).
  * writes metrics.json under <out-dir>/ours_<step>/.

Usage (inceptio):
  python scripts/eval_road_offtrack.py \
      --ckpt  <out>/.../ckpt_last.pt \
      --path  ~/work/data/9ae151dc/pai_9ae151dc-...json \
      --out-dir <out>/<name>_roadoff_eval
"""
from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(description="Road off-track KPI eval from ckpt")
    ap.add_argument("--ckpt", required=True, help="path to ckpt_*.pt")
    ap.add_argument("--path", default="", help="manifest json (override ckpt path)")
    ap.add_argument("--out-dir", required=True, help="output dir for renders + metrics")
    ap.add_argument("--novel-fid", action="store_true", help="also compute FID/KID (slow)")
    ap.add_argument(
        "--eval-cameras", nargs="*", default=None,
        help="restrict eval to camera_id subset (default: full test split)",
    )
    args = ap.parse_args()

    # Import after argparse so --help is fast and doesn't need CUDA.
    from threedgrut.render import Renderer

    os.makedirs(args.out_dir, exist_ok=True)
    renderer = Renderer.from_checkpoint(
        checkpoint_path=args.ckpt,
        out_dir=args.out_dir,
        path=args.path,
        novel_view=True,        # off-track sweep incl. yaw_30/60deg
        road_only=True,         # road layer in isolation (bg/dyn-rigid/sky off)
        load_lane_masks=True,   # enables lane_grad_corr / lane_band_psnr
        novel_fid=bool(args.novel_fid),
        novel_save_n=3,         # a few visual samples per mode; metrics unaffected
        eval_cameras=args.eval_cameras,
    )
    renderer.render_all()
    print(f"[eval_road_offtrack] done → metrics under {args.out_dir}/ours_*/")


if __name__ == "__main__":
    main()
