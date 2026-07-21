#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create an Inceptio NCore V4 dataset with native FTheta intrinsics."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from threedgrut.datasets.ftheta_derivation import (
    derive_native_ftheta_ncore_v4,
    prepare_ftheta_conversion_parameters,
    prepare_ftheta_conversion_parameters_from_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a native-FTheta NCore V4 dataset without changing the "
            "source dataset or the training-time NCore camera path."
        )
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parameter_source = parser.add_mutually_exclusive_group(required=True)
    parameter_source.add_argument(
        "--camera-id",
        action="append",
        dest="camera_ids",
        help=(
            "camera to fit from this source manifest's OpenCV calibration; "
            "repeat in the exact training-camera order"
        ),
    )
    parameter_source.add_argument(
        "--ftheta-artifact",
        type=Path,
        help="existing conversion-only FTheta parameter mapping",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-angle-margin-deg",
        type=float,
        default=0.1,
        help="strict full-raster max-angle margin (default: 0.1 degree)",
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="how unchanged raw/aux stores are materialized (default: hardlink)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate and report parameter coverage without writing NCore data",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.validate_only:
        if args.ftheta_artifact is not None:
            _, coverage, _ = prepare_ftheta_conversion_parameters(
                args.ftheta_artifact,
                margin_deg=args.max_angle_margin_deg,
            )
        else:
            _, coverage, _, _ = prepare_ftheta_conversion_parameters_from_manifest(
                args.source_manifest,
                args.camera_ids,
                margin_deg=args.max_angle_margin_deg,
            )
        print(json.dumps({key: asdict(value) for key, value in coverage.items()}, indent=2))
        return 0

    manifest_path = derive_native_ftheta_ncore_v4(
        source_manifest=args.source_manifest,
        ftheta_artifact=args.ftheta_artifact,
        camera_ids=args.camera_ids,
        output_dir=args.output_dir,
        margin_deg=args.max_angle_margin_deg,
        link_mode=args.link_mode,
    )
    print(f"native FTheta NCore manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
