#!/usr/bin/env python3
"""Validate PinholeForwardProjector against real NCore SDK OpenCVPinholeCameraModel.

Compares the NumPy projector's output (after the rational radial + tangential +
thin-prism fix) against the reference NCore SDK forward projection for every
camera in a manifest.

For each OpenCVPinhole camera:
  1. Samples pixels on a stride grid plus center, edge midpoints, corners.
  2. Calls ``model.pixels_to_camera_rays(pixels)`` to get camera-space rays.
  3. Calls ``model.camera_rays_to_pixels(rays)`` to get the SDK's own projected
     pixel and valid_flag — these are the reference ground truth.
  4. Feeds the rays as world rays (identity c2w) to
     ``PinholeForwardProjector.project_points()``.
  5. Compares ``visible`` vs SDK ``valid_flag`` and pixel coordinates.

Usage (inceptio)::

    python scripts/validate_pinhole_projector_ncore_parity.py \\
        --manifest /home/inceptio/work/data/inc_b6a9ed61_20s/...json \\
        --camera-ids \\
            camera_front_standard_55fov \\
            camera_front_tele_30fov \\
            camera_front_wide_120fov \\
            camera_cross_left_120fov \\
            camera_left_wide_90fov
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import ncore.data as _nd  # noqa: E402

from threedgrut_playground.utils.pinhole_projector import PinholeForwardProjector  # noqa: E402


def _discover_camera_ids(manifest_path: str) -> list[str]:
    """Open the manifest and discover all ``OpenCVPinholeCameraModel`` camera ids."""
    seq = _nd.CameraSequence(manifest_path)
    ids: list[str] = []
    for cam_id in seq.camera_ids:
        model = seq.camera_model(cam_id)
        if isinstance(model, _nd.OpenCVPinholeCameraModel):
            ids.append(cam_id)
    return sorted(ids)


def _sample_pixels(width: int, height: int, stride: int) -> np.ndarray:
    """Build a representative set of pixel coordinates.

    Returns (N, 2) int64 array of:
      - stride-grid sample
      - image centre
      - four edge midpoints (top, bottom, left, right)
      - four corners
    """
    xs, ys = np.meshgrid(
        np.arange(0, width, stride, dtype=np.int64),
        np.arange(0, height, stride, dtype=np.int64),
    )
    grid = np.stack([xs.ravel(), ys.ravel()], axis=1)

    specials = np.array(
        [
            [width // 2, height // 2],  # centre
            [width // 2, 0],  # top edge mid
            [width // 2, height - 1],  # bottom edge mid
            [0, height // 2],  # left edge mid
            [width - 1, height // 2],  # right edge mid
            [0, 0],  # top-left corner
            [width - 1, 0],  # top-right corner
            [0, height - 1],  # bottom-left corner
            [width - 1, height - 1],  # bottom-right corner
        ],
        dtype=np.int64,
    )
    return np.unique(np.vstack([grid, specials]), axis=0)


def _build_intrinsics_dict(model_params) -> dict:
    """Build the pinhole_dict expected by PinholeForwardProjector."""
    return {
        "resolution": model_params.resolution,
        "principal_point": np.asarray(model_params.principal_point, dtype=np.float64),
        "focal_length": np.asarray(model_params.focal_length, dtype=np.float64),
        "radial_coeffs": np.asarray(model_params.radial_coeffs, dtype=np.float64),
        "tangential_coeffs": np.asarray(model_params.tangential_coeffs, dtype=np.float64),
        "thin_prism_coeffs": np.asarray(model_params.thin_prism_coeffs, dtype=np.float64),
    }


def validate_camera(
    camera_id: str,
    model: _nd.OpenCVPinholeCameraModel,
    stride: int,
    valid_mae_threshold: float,
) -> dict:
    """Compare PinholeForwardProjector vs NCore SDK for one camera.

    Returns a stats dict.
    """
    params = model.get_parameters()
    width = int(params.resolution[0])
    height = int(params.resolution[1])

    print(f"  camera={camera_id}  resolution={width}x{height}", flush=True)

    # 1. Sample pixels
    pixels = _sample_pixels(width, height, stride)
    n_pixels = pixels.shape[0]
    print(f"    sampled {n_pixels} pixels (stride={stride})", flush=True)

    # 2. SDK: pixels → rays → pixels round-trip
    rays = model.pixels_to_camera_rays(pixels)
    sdk_pixels, sdk_valid = model.camera_rays_to_pixels(rays)

    sdk_pixels_arr = np.asarray(sdk_pixels, dtype=np.float64)
    sdk_valid_arr = np.asarray(sdk_valid, dtype=bool)
    n_sdk_valid = int(sdk_valid_arr.sum())

    # 3. NumPy projector: feed rays as world points with identity c2w
    rays_np = np.asarray(rays, dtype=np.float64)  # (N, 3) camera-space rays
    intrinsics = _build_intrinsics_dict(params)
    proj = PinholeForwardProjector(intrinsics, world_to_camera_flip=np.eye(4))
    uv_proj, visible_proj = proj.project_points(rays_np, np.eye(4))

    # 4. Compare visibility flags
    # SDK valid_flag = True means the forward projection converged
    # within the rational model's domain.
    # PinholeForwardProjector visible includes image-bound check.
    # For comparison we need to apply the same image-bound check to SDK.
    sdk_in_bound = (
        (sdk_pixels_arr[:, 0] >= 0)
        & (sdk_pixels_arr[:, 0] < width)
        & (sdk_pixels_arr[:, 1] >= 0)
        & (sdk_pixels_arr[:, 1] < height)
    )
    sdk_visible = sdk_valid_arr & sdk_in_bound
    n_sdk_visible = int(sdk_visible.sum())

    agreement = visible_proj == sdk_visible
    n_agree = int(agreement.sum())
    agreement_pct = 100.0 * n_agree / n_pixels if n_pixels > 0 else 100.0

    # 5. Compare pixel coordinates for samples where both agree visible
    both_visible = visible_proj & sdk_visible
    n_both = int(both_visible.sum())

    if n_both > 0:
        diffs = np.abs(uv_proj[both_visible] - sdk_pixels_arr[both_visible])
        mae = float(diffs.mean())
        max_err = float(diffs.max())
    else:
        mae = float("nan")
        max_err = float("nan")

    print(
        f"    SDK valid={n_sdk_valid}/{n_pixels}  SDK visible={n_sdk_visible}/{n_pixels}  "
        f"projector visible={int(visible_proj.sum())}/{n_pixels}",
        flush=True,
    )
    print(
        f"    agreement={n_agree}/{n_pixels} ({agreement_pct:.2f}%)  "
        f"both-visible={n_both}  MAE={mae:.6f}px  max_err={max_err:.6f}px",
        flush=True,
    )

    return {
        "camera_id": camera_id,
        "width": width,
        "height": height,
        "n_pixels": n_pixels,
        "n_sdk_valid": n_sdk_valid,
        "n_sdk_visible": n_sdk_visible,
        "n_projector_visible": int(visible_proj.sum()),
        "agreement": n_agree,
        "agreement_pct": agreement_pct,
        "n_both_visible": n_both,
        "mae_px": mae,
        "max_err_px": max_err,
        "passed": agreement_pct >= 99.5 and mae < valid_mae_threshold,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--manifest", required=True, help="Path to NCore manifest JSON")
    p.add_argument(
        "--camera-ids",
        nargs="*",
        default=None,
        help="Camera IDs to validate (default: all OpenCVPinhole cameras)",
    )
    p.add_argument("--stride", type=int, default=64, help="Pixel sampling stride (default 64)")
    p.add_argument(
        "--valid-mae-threshold",
        type=float,
        default=0.05,
        help="Max acceptable MAE in pixels (default 0.05)",
    )
    args = p.parse_args(argv)

    manifest_path = str(Path(args.manifest).expanduser().resolve())

    # 1. Discover cameras
    if args.camera_ids:
        camera_ids = args.camera_ids
    else:
        camera_ids = _discover_camera_ids(manifest_path)

    if not camera_ids:
        print("No OpenCVPinhole cameras found in manifest.")
        return 1

    print(f"Manifest: {manifest_path}")
    print(f"Cameras ({len(camera_ids)}): {', '.join(camera_ids)}")
    print()

    # 2. Open the sequence once and validate each camera
    seq = _nd.CameraSequence(manifest_path)
    results: list[dict] = []
    for cam_id in camera_ids:
        model = seq.camera_model(cam_id)
        if not isinstance(model, _nd.OpenCVPinholeCameraModel):
            print(f"  SKIP {cam_id}: not an OpenCVPinholeCameraModel (got {type(model).__name__})", flush=True)
            continue
        stats = validate_camera(cam_id, model, args.stride, args.valid_mae_threshold)
        results.append(stats)

    # 3. Summary
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    n_passed = sum(1 for r in results if r["passed"])
    n_total = len(results)
    all_agreement = [r["agreement_pct"] for r in results]
    all_mae = [r["mae_px"] for r in results if not np.isnan(r["mae_px"])]

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  [{status}] {r['camera_id']:>40s}: "
            f"agreement={r['agreement_pct']:.2f}%  "
            f"MAE={r['mae_px']:.6f}px  "
            f"max_err={r['max_err_px']:.6f}px  "
            f"both_visible={r['n_both_visible']}/{r['n_pixels']}"
        )

    print()
    print(f"  Passed: {n_passed}/{n_total}")
    if all_agreement:
        print(f"  Agreement range: {min(all_agreement):.2f}% – {max(all_agreement):.2f}%")
    if all_mae:
        print(f"  MAE range: {min(all_mae):.6f} – {max(all_mae):.6f} px")
    print()

    if n_passed == n_total:
        print("All cameras PASS — PinholeForwardProjector matches NCore SDK.")
        return 0
    else:
        print(f"{n_total - n_passed} camera(s) FAILED — see per-camera details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
