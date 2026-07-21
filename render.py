# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from threedgrut.render import Renderer

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str, help="path to the pretrained checkpoint")
    parser.add_argument(
        "--path", type=str, default="", help="Path to the training data, if not provided taken from ckpt"
    )
    parser.add_argument("--out-dir", required=True, type=str, help="Output path")
    parser.add_argument(
        "--save-gt", action="store_false", help="If set, the GT images will not be saved [True by default]"
    )
    parser.add_argument(
        "--compute-extra-metrics",
        action="store_false",
        help="If set, extra image metrics will not be computed [True by default]",
    )
    parser.add_argument(
        "--eval-cameras",
        type=str,
        default="",
        help=(
            "T8.5.7 / V3-E4: comma-separated list of camera_id strings to "
            "restrict eval to a subset (e.g. 'camera_front_wide_120fov,"
            "camera_rear_tele_30fov'). Empty (default) = no filter, eval "
            "iterates the full test split."
        ),
    )
    parser.add_argument("--enabled-layers", type=str, default="", help="Comma-separated layer names for inference-only render filtering")
    parser.add_argument(
        "--ownership-dump",
        action="store_true",
        help="Write alpha, road-mask, and sky-contribution debug images during evaluation",
    )
    parser.add_argument(
        "--novel-view",
        action="store_true",
        help=(
            "T8.5.3 / V3-E3: also render 4 novel-view perturbations of each "
            "anchor frame (lateral_1m / lateral_2m / yaw_5deg / yaw_10deg) "
            "and record per-mode LPIPS vs the anchor GT in metrics.json. "
            "5x render cost; off by default."
        ),
    )
    parser.add_argument(
        "--use-difix",
        action="store_true",
        help=(
            "V3-T15.2: enable DiFix (HF nvidia/Fixer) post-processing during "
            "eval. Adds mean_psnr_difix / mean_ssim_difix / mean_lpips_difix "
            "to metrics.json. Requires cosmos_predict2 + DiFix weights — see "
            "third_party/Fixer/INSTALL.md."
        ),
    )
    parser.add_argument(
        "--load-lane-masks",
        action="store_true",
        help=(
            "Phase 3 lane GT: load *.aux.lane.zarr.itar (Mapillary lane sseg) "
            "and emit mean_lane_* metrics. Injects conf.dataset.load_lane_masks"
            "=True so a pre-trained ckpt (whose embedded config predates lane) "
            "still loads the lane product at eval."
        ),
    )
    parser.add_argument(
        "--lane-band-px",
        type=int,
        default=None,
        help="Phase 3 lane dilated-band half-width (px). Default = DEFAULT_LANE_BAND_PX (8).",
    )
    parser.add_argument(
        "--dataset-cameras",
        type=str,
        default="",
        help=(
            "E1.3 held-out protocol: comma-separated camera_id list that "
            "REPLACES the ckpt-embedded dataset.camera_ids before the eval "
            "dataset is built (e.g. eval a 4-cam ckpt on the excluded cross "
            "camera). Unlike --eval-cameras (a batch filter over loaded "
            "cameras), this changes which cameras the dataset loads. "
            "Side effect: BilateralGrid exposure is disabled (train-time "
            "camera_idx mapping invalid) — use cc_* metrics."
        ),
    )
    parser.add_argument(
        "--novel-fid",
        action="store_true",
        help=(
            "E1.4: compute FID/KID distribution metrics — interpolated "
            "renders vs GT always; per novel mode when --novel-view is also "
            "set. KID is the primary small-sample metric (subset size "
            "auto-adapted); FID reported alongside for E0.2-anchor "
            "comparability. Off by default (byte-identical metrics.json)."
        ),
    )
    parser.add_argument(
        "--novel-save-n",
        type=int,
        default=5,
        help=(
            "E2.1: # novel frames to save per mode (-1=all). Default 5 = "
            "historical visual-sample behaviour. Pass -1 to persist ALL frames "
            "with per-camera subdir naming + frames_map.json for offline "
            "Harmonizer fix."
        ),
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help=(
            "E2.1 fast frame dump: force-off GT supervision loads "
            "(aux/lane/lidar_depth/depth_prior) + NTA + extra metrics. "
            "Use with --novel-view --novel-only to dump lateral_3m/6m frames "
            "at maximum speed without any eval overhead."
        ),
    )
    parser.add_argument(
        "--novel-only",
        action="store_true",
        help=(
            "E2.1: with --novel-view, render ONLY lateral_3m + lateral_6m "
            "(skip lateral_1m/2m + yaw_5/10deg). Cuts 6-mode cost to 2-mode."
        ),
    )
    args = parser.parse_args()

    eval_cameras_list = [c.strip() for c in args.eval_cameras.split(",") if c.strip()] or None
    dataset_cameras_list = [c.strip() for c in args.dataset_cameras.split(",") if c.strip()] or None

    renderer = Renderer.from_checkpoint(
        checkpoint_path=args.checkpoint,
        path=args.path,
        out_dir=args.out_dir,
        save_gt=args.save_gt,
        computes_extra_metrics=args.compute_extra_metrics,
        eval_cameras=eval_cameras_list,
        novel_view=args.novel_view,
        use_difix=args.use_difix,
        load_lane_masks=args.load_lane_masks,
        lane_band_px=args.lane_band_px,
        dataset_cameras=dataset_cameras_list,
        novel_fid=args.novel_fid,
        novel_save_n=args.novel_save_n,
        render_only=args.render_only,
        novel_only=args.novel_only,
    )
    if args.enabled_layers:
        renderer.conf.render.enabled_layers = [name.strip() for name in args.enabled_layers.split(",") if name.strip()]
    if args.ownership_dump:
        renderer.conf.model.debug_sky_contrib = True
        renderer.conf.render.ownership_dump = True

    renderer.render_all()
