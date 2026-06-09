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
    args = parser.parse_args()

    eval_cameras_list = [c.strip() for c in args.eval_cameras.split(",") if c.strip()] or None

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
    )

    renderer.render_all()
