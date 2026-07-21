#!/usr/bin/env python3
"""
PIN-AB-1: Radial analysis — per-camera radial-binned PSNR + gradient ratio.

Reads Arm A (mask=false) and Arm B (mask=true) eval outputs, maps eval images
to camera blocks via per_camera n_frames from metrics.json, computes fixed-bin
radial metrics (r<0.5, 0.5-0.7, 0.7-0.9, r>=0.9) using **image-normalized
half-diagonal radius** (image center=0, corners≈1), forward-valid-domain
common-mask metrics using NCore production helpers, and common full-frame
metrics.  Self-masked metrics from metrics.json are reported as diagnostic only.

Usage:
    python scripts/drivers/pin_ab_radial_analysis.py \
        --arm-a-dir ~/work/output/pin_ab_nomask_5s_5k_eval \
        --arm-b-dir ~/work/output/pin_ab_mask_5s_5k_eval \
        --arm-a-name pin_ab_nomask_5s_5k \
        --arm-b-name pin_ab_mask_5s_5k \
        --out ~/work/output/pin_ab_radial_report \
        --manifest <path-to-manifest.json>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

# Force project imports to resolve from this script's worktree rather than an
# editable install pointing at another checkout (e.g. inceptio main repo).
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

_RADIAL_BINS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, float("inf"))]
_BIN_LABELS = ["r<0.5", "r0.5-0.7", "r0.7-0.9", "r>=0.9"]


# --------------------------------------------------------------------------- #
# Helpers                                                                    #
# --------------------------------------------------------------------------- #

def _load_camera_ids_from_metrics(metrics_path: str) -> list[str]:
    """Extract camera IDs in order from metrics.json per_camera section."""
    with open(metrics_path) as f:
        m = json.load(f)
    pc = m.get("per_camera", {})
    if not pc:
        raise RuntimeError(f"metrics.json missing per_camera section: {metrics_path}")
    return list(pc.keys())


def _get_camera_n_frames(metrics_path: str) -> dict[str, int]:
    """Get per-camera n_frames from metrics.json."""
    with open(metrics_path) as f:
        m = json.load(f)
    pc = m.get("per_camera", {})
    return {cid: int(info["n_frames"]) for cid, info in pc.items()}


def _resolve_run_root(eval_dir: str) -> tuple[str, str]:
    """Resolve the run root directory and selected metrics.json path.

    render.py writes metrics.json at the top level of the eval output directory,
    but the driver may wrap it in a subdirectory.  This function finds all
    metrics.json files under *eval_dir*, selects one deterministically, and
    returns (run_root, metrics_path) where run_root is the parent directory of
    the selected metrics.json (the directory that contains ours_*/renders + gt).

    Deterministic tie-breaking:
      1. Prefer metrics.json whose parent also contains ours_* dirs.
      2. Among those, take the one with the deepest path (most nested).
      3. Among equally deep, take alphabetically last (lexicographic).

    Raises RuntimeError if zero or genuinely ambiguous multiple candidates remain.
    """
    candidates = sorted(Path(eval_dir).rglob("metrics.json"))

    if not candidates:
        raise RuntimeError(
            f"No metrics.json found under {eval_dir}"
        )

    if len(candidates) == 1:
        root = str(candidates[0].parent.resolve())
        return root, str(candidates[0])

    # Multiple: prefer candidates whose parent has ours_* dirs
    with_ours = [c for c in candidates
                 if list(c.parent.glob("ours_*"))]
    if not with_ours:
        with_ours = candidates  # fall back to all candidates

    # Deterministic: deepest path wins; tie → alphabetically last
    def _depth(p: Path) -> int:
        return len(p.parent.parts)

    sel = max(with_ours, key=lambda p: (_depth(p), str(p.resolve())))
    root = str(sel.parent.resolve())
    print(f"  [nested-resolve] selected metrics.json: {sel} "
          f"(root={root})", flush=True)
    return root, str(sel)


def _load_images_from_root(run_root: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load render and GT images from the run root directory.

    Looks for ours_*/renders + ours_*/gt under *run_root*.  The *run_root*
    is the parent of the metrics.json (i.e. the actual eval output directory),
    not a potentially nested wrapper.
    """
    root = Path(run_root)
    ours_dirs = sorted(root.glob("ours_*"))
    if not ours_dirs:
        raise RuntimeError(f"No ours_* dirs found in run root {run_root}")

    # Take the last (highest-step) ours_* dir
    step_dir = ours_dirs[-1]
    render_dir = step_dir / "renders"
    gt_dir = step_dir / "gt"

    if not render_dir.exists():
        raise RuntimeError(f"renders dir not found: {render_dir}")
    if not gt_dir.exists():
        raise RuntimeError(f"gt dir not found: {gt_dir}")

    render_files = sorted(render_dir.glob("*.png"))
    gt_files = sorted(gt_dir.glob("*.png"))

    if len(render_files) != len(gt_files):
        raise RuntimeError(
            f"Render/GT count mismatch: {len(render_files)} vs {len(gt_files)}"
        )
    if len(render_files) == 0:
        raise RuntimeError(f"No render images in {render_dir}")

    renders: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    for rf, gf in zip(render_files, gt_files):
        renders.append(np.asarray(Image.open(rf).convert("RGB"), dtype=np.float32) / 255.0)
        gts.append(np.asarray(Image.open(gf).convert("RGB"), dtype=np.float32) / 255.0)

    return renders, gts


def _get_camera_intrinsics(manifest_path: str, camera_ids: list[str],
                           downsample: float = 1.0) -> dict[str, dict[str, float | int]]:
    """Load camera intrinsics from manifest using NCore model parameter APIs.

    Uses CameraModel.from_parameters and accesses attributes directly
    (not .get() on parameter objects).
    """
    try:
        import ncore.data.v4 as ncore_v4
        import ncore.sensors as ncore_sensors
    except ImportError as e:
        raise RuntimeError(
            f"ncore not importable ({e}). This script must run on a machine with ncore."
        ) from e

    reader = ncore_v4.SequenceComponentGroupsReader(
        [str(Path(manifest_path).expanduser().resolve())]
    )
    loader = ncore_v4.SequenceLoaderV4(reader)

    intrinsics: dict[str, dict[str, float | int]] = {}
    for cid in camera_ids:
        try:
            sensor = loader.get_camera_sensor(cid)
        except (ValueError, RuntimeError) as e:
            print(f"  [warn] Camera '{cid}' not available in manifest, skipping: {e}",
                  flush=True)
            continue
        mp = sensor.model_parameters
        model = ncore_sensors.CameraModel.from_parameters(
            mp, device="cpu", dtype=torch.float32
        )
        w = int(model.resolution[0].item())
        h = int(model.resolution[1].item())

        # Use attribute access — NOT .get() on parameter objects.
        # FTheta cameras use angle_to_pixeldist_poly instead of focal_length.
        if hasattr(mp, "focal_length"):
            fl = mp.focal_length
            if hasattr(fl, "__len__"):
                fx = float(fl[0])
                fy = float(fl[1]) if len(fl) > 1 else fx
            else:
                fx = fy = float(fl)
        else:
            # FTheta: approximate focal length from polynomial derivative at centre
            poly = mp.angle_to_pixeldist_poly
            fx = fy = float(poly[1]) if hasattr(poly, "__getitem__") and len(poly) > 1 else 1.0
        pp = mp.principal_point
        cx, cy = float(pp[0]), float(pp[1])

        # Apply downsample
        if downsample < 1.0:
            fx *= downsample
            fy *= downsample
            cx *= downsample
            cy *= downsample
            w = int(round(w * downsample))
            h = int(round(h * downsample))

        intrinsics[cid] = {
            "fx": fx, "fy": fy,
            "cx": cx, "cy": cy,
            "width": w, "height": h,
        }
    return intrinsics


def _compute_corner_normalized_radius_map(
    h: int, w: int, cx: float, cy: float
) -> np.ndarray:
    """Compute image-normalized radius: sqrt((x-cx)^2 + (y-cy)^2) / half_diag.

    Half-diagonal = sqrt((w/2)^2 + (h/2)^2).  Image center = 0, corners ≈ 1.
    This is *not* focal-normalized camera coordinates — only image-space
    normalized by half-diagonal, matching the prior report convention.
    """
    half_diag = np.sqrt((w / 2.0) ** 2 + (h / 2.0) ** 2)
    if half_diag < 1e-12:
        return np.zeros((h, w), dtype=np.float64)
    ys, xs = np.meshgrid(
        np.arange(h, dtype=np.float64),
        np.arange(w, dtype=np.float64),
        indexing="ij",
    )
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    return r / half_diag


def _psnr(mse: float) -> float:
    if mse < 1e-12:
        return 100.0
    return float(-10.0 * np.log10(mse))


def _compute_gradient_magnitudes(
    img: np.ndarray,
) -> np.ndarray:
    """Compute per-pixel gradient magnitude (mean over RGB), [H, W] float64."""
    from scipy.ndimage import sobel

    img64 = img.astype(np.float64)
    gx = sobel(img64, axis=1, mode="constant")
    gy = sobel(img64, axis=0, mode="constant")
    mag = np.sqrt(gx ** 2 + gy ** 2)  # [H, W, 3]
    return mag.mean(axis=2)  # [H, W], mean over RGB


def _compute_aggregate_gradient_ratio(
    render_img: np.ndarray, gt_img: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[float, float, float]:
    """Compute gradient ratio as sum(mag_render) / sum(mag_gt) for the region.

    Returns (sum_render_mag, sum_gt_mag, ratio).
    This is the *aggregate* ratio, NOT mean of per-pixel ratios.
    """
    mag_r = _compute_gradient_magnitudes(render_img)
    mag_g = _compute_gradient_magnitudes(gt_img)

    if mask is not None:
        mag_r = mag_r[mask]
        mag_g = mag_g[mask]

    sum_r = float(mag_r.sum())
    sum_g = float(mag_g.sum())

    if sum_g < 1e-12:
        ratio = float("nan")
    else:
        ratio = sum_r / sum_g

    return sum_r, sum_g, ratio


def _compute_pixel_mse(render_img: np.ndarray, gt_img: np.ndarray) -> np.ndarray:
    """Compute per-pixel MSE (mean over RGB channels)."""
    diff = (render_img.astype(np.float64) - gt_img.astype(np.float64)) ** 2
    return diff.mean(axis=2)


def _masked_frame_sums(
    render_img: np.ndarray,
    gt_img: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, int, float, float]:
    """Return MSE sum, pixel count, render-grad sum, and GT-grad sum."""
    if mask.shape != render_img.shape[:2] or gt_img.shape != render_img.shape:
        raise ValueError("mask/image shapes do not match")
    n_pixels = int(mask.sum())
    if n_pixels == 0:
        return 0.0, 0, 0.0, 0.0
    mse_map = _compute_pixel_mse(render_img, gt_img)
    mag_render = _compute_gradient_magnitudes(render_img)
    mag_gt = _compute_gradient_magnitudes(gt_img)
    return (
        float(mse_map[mask].sum()),
        n_pixels,
        float(mag_render[mask].sum()),
        float(mag_gt[mask].sum()),
    )


def _compute_forward_valid_mask(
    manifest_path: str, camera_id: str,
    h: int, w: int,
) -> np.ndarray:
    """Compute the forward-valid pixel mask for a single camera using NCore.

    Full production pipeline:
      1. Create integer pixel grid.
      2. pixels_to_camera_rays → get camera-space ray directions.
      3. repair_nonfinite_rays into an all-true mask (flags NaN rays invalid).
      4. maybe_apply_forward_valid_mask (enabled=True) → AND forward validity.

    Returns [H, W] bool mask (True = valid in common domain).
    """
    import ncore.data.v4 as ncore_v4
    import ncore.sensors as ncore_sensors
    from threedgrut.datasets.utils import (
        compute_forward_valid_pixel_mask,
        maybe_apply_forward_valid_mask,
        repair_nonfinite_rays,
    )

    reader = ncore_v4.SequenceComponentGroupsReader(
        [str(Path(manifest_path).expanduser().resolve())]
    )
    loader = ncore_v4.SequenceLoaderV4(reader)
    sensor = loader.get_camera_sensor(camera_id)
    model = ncore_sensors.CameraModel.from_parameters(
        sensor.model_parameters, device="cpu", dtype=torch.float32
    )

    # Integer pixel grid
    xs, ys = np.meshgrid(
        np.arange(w, dtype=np.int16),
        np.arange(h, dtype=np.int16),
    )
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=1)

    # pixels_to_camera_rays
    rays = model.pixels_to_camera_rays(pixels).reshape(h, w, 3).numpy()

    # Repair non-finite rays into all-true ego mask
    ego_mask = np.ones((h, w), dtype=bool)
    repair_nonfinite_rays(rays, ego_mask)

    # Forward-valid mask enabled
    maybe_apply_forward_valid_mask(model, rays, ego_mask, camera_id, enabled=True)

    return ego_mask


# --------------------------------------------------------------------------- #
# Per-arm analysis                                                           #
# --------------------------------------------------------------------------- #

def analyze_arm(
    eval_dir: str,
    arm_name: str,
    manifest_path: str,
    camera_order: list[str] | None = None,
    common_masks: dict[str, np.ndarray] | None = None,
) -> dict:
    """Run radial analysis on one arm.

    Args:
        eval_dir: Top-level eval output directory (may contain nested run).
        arm_name: Display name for this arm.
        manifest_path: Path to NCore manifest JSON.
        camera_order: Ordered list of camera IDs.  If None, derived from
                      Arm A's metrics.json.
        common_masks: Precomputed forward-valid pixel masks shared between
                      arms.  Keyed by camera_id, [H, W] bool.

    Returns a nested dict with per-camera and global results.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"  Analyzing {arm_name} ({eval_dir})", flush=True)
    print(f"{'='*60}", flush=True)

    # Resolve nested run root and metrics.json
    run_root, metrics_path = _resolve_run_root(eval_dir)
    print(f"  run_root: {run_root}", flush=True)
    print(f"  metrics.json: {metrics_path}", flush=True)

    # Get camera order and n_frames
    cam_n_frames = _get_camera_n_frames(metrics_path)
    cam_ids_list = camera_order or list(cam_n_frames.keys())
    print(f"  Cameras ({len(cam_ids_list)}): {cam_ids_list}", flush=True)
    for cid in cam_ids_list:
        print(f"    {cid}: {cam_n_frames[cid]} frames", flush=True)

    # Validate that cam_ids_list order matches metrics.json per_camera order
    metrics_cam_ids = _load_camera_ids_from_metrics(metrics_path)
    if tuple(cam_ids_list) != tuple(metrics_cam_ids):
        print(
            f"  NOTE: camera order differs from metrics.json per_camera: "
            f"{metrics_cam_ids}",
            flush=True,
        )

    # Load images from run root
    renders, gts = _load_images_from_root(run_root)

    # Load camera intrinsics
    print("  Loading camera intrinsics from manifest...", flush=True)
    intrinsics = _get_camera_intrinsics(manifest_path, cam_ids_list)

    # Filter to cameras with intrinsics (skip cameras not in manifest)
    if len(intrinsics) < len(cam_ids_list):
        skipped = set(cam_ids_list) - set(intrinsics.keys())
        print(f"  [warn] Skipping {len(skipped)} camera(s) not in manifest: {sorted(skipped)}",
              flush=True)
        cam_ids_list = [c for c in cam_ids_list if c in intrinsics]
        cam_n_frames = {c: cam_n_frames[c] for c in cam_ids_list}

    # Build per-frame camera mapping
    frame_to_camera: list[tuple[str, int]] = []
    for cid in cam_ids_list:
        nf = cam_n_frames[cid]
        for fi in range(nf):
            frame_to_camera.append((cid, fi))

    # Per-camera accumulators for radial bins
    # Structure: camera_id -> bin_idx -> {"mse_sum": ..., "n_pixels": ...,
    #             "grad_mag_render_sum": ..., "grad_mag_gt_sum": ...}
    per_cam_radial: dict[str, list[dict[str, float]]] = {}
    for cid in cam_ids_list:
        per_cam_radial[cid] = [
            {"mse_sum": 0.0, "n_pixels": 0,
             "grad_mag_render_sum": 0.0, "grad_mag_gt_sum": 0.0}
            for _ in _RADIAL_BINS
        ]

    # The common-valid diagnostics use the same per-pixel MSE and gradient
    # maps as the full-raster result.  Accumulate them in the primary pass so
    # they remain strictly diagnostic without replaying every 1080p frame once
    # per camera/bin (the old implementation did 31 extra gradient passes).
    per_cam_common: dict[str, dict[str, float]] = {}
    per_cam_common_bins: dict[str, list[dict[str, float]]] = {}
    if common_masks:
        for cid in cam_ids_list:
            if cid not in common_masks:
                continue
            per_cam_common[cid] = {
                "mse_sum": 0.0, "n_pixels": 0,
                "grad_mag_render_sum": 0.0, "grad_mag_gt_sum": 0.0,
            }
            per_cam_common_bins[cid] = [
                {"mse_sum": 0.0, "n_pixels": 0,
                 "grad_mag_render_sum": 0.0, "grad_mag_gt_sum": 0.0}
                for _ in _RADIAL_BINS
            ]

    # Precompute radius maps per camera
    print("  Precomputing corner-normalized radius maps...", flush=True)
    radius_maps: dict[str, np.ndarray] = {}
    for cid in cam_ids_list:
        ip = intrinsics[cid]
        radius_maps[cid] = _compute_corner_normalized_radius_map(
            int(ip["height"]), int(ip["width"]),
            float(ip["cx"]), float(ip["cy"]),
        )

    # Process each frame
    print(f"  Processing {len(renders)} frames...", flush=True)
    for frame_idx, (render_img, gt_img) in enumerate(zip(renders, gts)):
        if frame_idx >= len(frame_to_camera):
            print(f"  WARNING: frame {frame_idx} beyond camera map, skipping",
                  flush=True)
            continue
        cid, _ = frame_to_camera[frame_idx]

        ip = intrinsics[cid]
        h, w = render_img.shape[:2]
        ih, iw = int(ip["height"]), int(ip["width"])

        # Check resolution matches intrinsics
        if h != ih or w != iw:
            print(
                f"  NOTE: frame {frame_idx} cam {cid} size ({w}x{h}) != "
                f"intrinsics ({iw}x{ih}), recomputing radius map",
                flush=True,
            )
            rmap = _compute_corner_normalized_radius_map(
                h, w,
                float(ip["cx"]) * w / iw,
                float(ip["cy"]) * h / iw,
            )
        else:
            rmap = radius_maps[cid]

        # Per-pixel MSE
        mse_map = _compute_pixel_mse(render_img, gt_img)

        # Gradient magnitudes (aggregate, not per-pixel ratio)
        mag_r = _compute_gradient_magnitudes(render_img)
        mag_g = _compute_gradient_magnitudes(gt_img)

        # Accumulate per bin
        for bin_idx, (r_min, r_max) in enumerate(_RADIAL_BINS):
            if r_max == float("inf"):
                bin_mask = rmap >= r_min
            else:
                bin_mask = (rmap >= r_min) & (rmap < r_max)

            n_bin = int(bin_mask.sum())
            if n_bin == 0:
                continue

            per_cam_radial[cid][bin_idx]["mse_sum"] += float(
                mse_map[bin_mask].sum()
            )
            per_cam_radial[cid][bin_idx]["n_pixels"] += n_bin

            # Aggregate gradient magnitudes (sum, not per-pixel ratio)
            per_cam_radial[cid][bin_idx]["grad_mag_render_sum"] += float(
                mag_r[bin_mask].sum()
            )
            per_cam_radial[cid][bin_idx]["grad_mag_gt_sum"] += float(
                mag_g[bin_mask].sum()
            )

        # P∩F common-valid is diagnostic only.  Reuse the just-computed image
        # maps rather than recomputing their gradients in separate loops.
        if cid in per_cam_common:
            cm_f = common_masks[cid]
            if cm_f.shape != (h, w):
                from skimage.transform import resize
                cm_f = resize(cm_f.astype(np.float64), (h, w), order=0,
                              preserve_range=True).astype(bool)
            common_masks_for_bins = []
            for bin_idx, (r_min, r_max) in enumerate(_RADIAL_BINS):
                if r_max == float("inf"):
                    bin_mask = rmap >= r_min
                else:
                    bin_mask = (rmap >= r_min) & (rmap < r_max)
                common_masks_for_bins.append(bin_mask & cm_f)

            def _add_common(dst: dict[str, float], mask: np.ndarray) -> None:
                npx = int(mask.sum())
                if npx == 0:
                    return
                dst["mse_sum"] += float(mse_map[mask].sum())
                dst["n_pixels"] += npx
                dst["grad_mag_render_sum"] += float(mag_r[mask].sum())
                dst["grad_mag_gt_sum"] += float(mag_g[mask].sum())

            _add_common(per_cam_common[cid], cm_f)
            for bin_idx, mask in enumerate(common_masks_for_bins):
                _add_common(per_cam_common_bins[cid][bin_idx], mask)

        if (frame_idx + 1) % 50 == 0 or frame_idx == 0:
            print(f"    processed frame {frame_idx + 1}/{len(renders)}",
                  flush=True)

    # ------------------------------------------------------------------ #
    # Compile results                                                    #
    # ------------------------------------------------------------------ #
    results: dict[str, Any] = {
        "arm": arm_name,
        "per_camera": {},
        "global": {},
    }

    for cid in cam_ids_list:
        cam_res: dict[str, Any] = {}

        # Radial bin metrics
        for bin_idx, label in enumerate(_BIN_LABELS):
            d = per_cam_radial[cid][bin_idx]
            npx = int(d["n_pixels"])
            if npx > 0:
                mse = d["mse_sum"] / npx
                psnr_val = _psnr(mse)
                # Aggregate gradient ratio: sum_render / sum_gt
                sum_r = d["grad_mag_render_sum"]
                sum_g = d["grad_mag_gt_sum"]
                if sum_g > 1e-12:
                    grad_ratio = sum_r / sum_g
                else:
                    grad_ratio = float("nan")
            else:
                psnr_val = float("nan")
                grad_ratio = float("nan")
            cam_res[label] = {
                "n_pixels": npx,
                "psnr": round(psnr_val, 4),
                "gradient_ratio": round(grad_ratio, 6)
                if not np.isnan(grad_ratio) else None,
            }

        # Full-frame metrics (aggregate over all bins)
        all_mse = sum(per_cam_radial[cid][bi]["mse_sum"]
                      for bi in range(len(_RADIAL_BINS)))
        all_npx = sum(per_cam_radial[cid][bi]["n_pixels"]
                      for bi in range(len(_RADIAL_BINS)))
        all_sum_r = sum(per_cam_radial[cid][bi]["grad_mag_render_sum"]
                        for bi in range(len(_RADIAL_BINS)))
        all_sum_g = sum(per_cam_radial[cid][bi]["grad_mag_gt_sum"]
                        for bi in range(len(_RADIAL_BINS)))

        cam_res["full_frame"] = {
            "n_pixels": all_npx,
            "psnr": round(_psnr(all_mse / max(all_npx, 1)), 4),
        }
        if all_sum_g > 1e-12:
            cam_res["full_frame"]["gradient_ratio"] = round(
                all_sum_r / all_sum_g, 6
            )
        else:
            cam_res["full_frame"]["gradient_ratio"] = None

        # Common-domain metrics (shared forward-valid mask; diagnostic only).
        if cid in per_cam_common:
            common = per_cam_common[cid]
            if int(common["n_pixels"]) > 0:
                cm_mse = common["mse_sum"] / common["n_pixels"]
                cam_res["common_domain"] = {
                    "n_pixels": int(common["n_pixels"]),
                    "psnr": round(_psnr(cm_mse), 4),
                }
                if common["grad_mag_gt_sum"] > 1e-12:
                    cam_res["common_domain"]["gradient_ratio"] = round(
                        common["grad_mag_render_sum"] / common["grad_mag_gt_sum"], 6
                    )
                else:
                    cam_res["common_domain"]["gradient_ratio"] = None

                # Optional: bin ∩ common metrics
                for bin_idx, label in enumerate(_BIN_LABELS):
                    common_bin = per_cam_common_bins[cid][bin_idx]
                    if int(common_bin["n_pixels"]) > 0:
                        bc_mse = common_bin["mse_sum"] / common_bin["n_pixels"]
                        cam_res[f"common_{label}"] = {
                            "n_pixels": int(common_bin["n_pixels"]),
                            "psnr": round(_psnr(bc_mse), 4),
                        }
                        if common_bin["grad_mag_gt_sum"] > 1e-12:
                            cam_res[f"common_{label}"]["gradient_ratio"] = round(
                                common_bin["grad_mag_render_sum"] / common_bin["grad_mag_gt_sum"], 6
                            )

        # Diagnostic: self-masked metrics from metrics.json (NOT common domain)
        results["per_camera"][cid] = cam_res

    # Global results
    all_mse = sum(
        per_cam_radial[cid][bi]["mse_sum"]
        for cid in cam_ids_list
        for bi in range(len(_RADIAL_BINS))
    )
    all_npx = sum(
        per_cam_radial[cid][bi]["n_pixels"]
        for cid in cam_ids_list
        for bi in range(len(_RADIAL_BINS))
    )
    all_grad_r = sum(
        per_cam_radial[cid][bi]["grad_mag_render_sum"]
        for cid in cam_ids_list
        for bi in range(len(_RADIAL_BINS))
    )
    all_grad_g = sum(
        per_cam_radial[cid][bi]["grad_mag_gt_sum"]
        for cid in cam_ids_list
        for bi in range(len(_RADIAL_BINS))
    )

    results["global"]["full_frame_psnr"] = round(
        _psnr(all_mse / max(all_npx, 1)), 4
    )
    if all_grad_g > 1e-12:
        results["global"]["full_frame_gradient_ratio"] = round(
            all_grad_r / all_grad_g, 6
        )
    else:
        results["global"]["full_frame_gradient_ratio"] = None

    # Global common-domain diagnostic, aggregated from the primary pass.
    if per_cam_common:
        gc_mse_sum = sum(d["mse_sum"] for d in per_cam_common.values())
        gc_npx = sum(d["n_pixels"] for d in per_cam_common.values())
        gc_grad_r = sum(d["grad_mag_render_sum"] for d in per_cam_common.values())
        gc_grad_g = sum(d["grad_mag_gt_sum"] for d in per_cam_common.values())
        if gc_npx > 0:
            gc_mse = gc_mse_sum / gc_npx
            results["global"]["common_domain_psnr"] = round(_psnr(gc_mse), 4)
            if gc_grad_g > 1e-12:
                results["global"]["common_domain_gradient_ratio"] = round(
                    gc_grad_r / gc_grad_g, 6
                )
            else:
                results["global"]["common_domain_gradient_ratio"] = None

    # Diagnostic: read self-masked metrics from metrics.json (NOT common domain)
    print("  [diagnostic] Reading self-masked metrics from metrics.json...",
          flush=True)
    with open(metrics_path) as f:
        m = json.load(f)

    diag_keys = ("mean_psnr_masked", "mean_cc_psnr_masked",
                 "mean_ssim_masked", "mean_lpips_masked",
                 "mean_psnr", "mean_cc_psnr",
                 "mean_ssim", "mean_lpips")
    for key in diag_keys:
        if key in m:
            results["global"][f"{key}_diagnostic"] = m[key]

    pc = m.get("per_camera", {})
    for cid in cam_ids_list:
        if cid in pc:
            for k, v in pc[cid].items():
                if isinstance(v, (int, float)):
                    results["per_camera"][cid][f"{k}_diagnostic"] = v

    print(f"  {arm_name} done: "
          f"global full-frame PSNR = "
          f"{results['global'].get('full_frame_psnr', 'N/A')}",
          flush=True)
    return results


# --------------------------------------------------------------------------- #
# Main                                                                       #
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="PIN-AB-1 radial analysis: per-camera radial PSNR + gradient ratio"
    )
    parser.add_argument("--arm-a-dir", required=True, help="Arm A eval output dir")
    parser.add_argument("--arm-b-dir", required=True, help="Arm B eval output dir")
    parser.add_argument("--arm-a-name", default="pin_ab_nomask_5s_5k")
    parser.add_argument("--arm-b-name", default="pin_ab_mask_5s_5k")
    parser.add_argument("--out", required=True, help="Output directory for report json")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON")
    args = parser.parse_args()

    # Get camera order from Arm A's metrics.json (via nested resolution)
    a_run_root, a_metrics_path = _resolve_run_root(args.arm_a_dir)
    camera_order = _get_camera_n_frames(a_metrics_path)
    cam_ids_list = list(camera_order.keys())

    # Precompute common forward-valid masks shared by both arms
    print(f"\n{'='*60}", flush=True)
    print("  Precomputing common forward-valid pixel masks...", flush=True)
    print(f"{'='*60}", flush=True)
    common_masks: dict[str, np.ndarray] = {}
    for cid in cam_ids_list:
        ip_a = _get_camera_intrinsics(args.manifest, [cid])
        if cid not in ip_a:
            print(f"  [warn] Camera '{cid}' not in manifest, skipping mask", flush=True)
            continue
        h = int(ip_a[cid]["height"])
        w = int(ip_a[cid]["width"])
        print(f"  Computing forward-valid mask for {cid} ({w}x{h})...",
              flush=True)
        mask = _compute_forward_valid_mask(args.manifest, cid, h, w)
        common_masks[cid] = mask
        n_valid = int(mask.sum())
        n_total = h * w
        print(f"    {cid}: {n_valid}/{n_total} pixels valid "
              f"({100.0 * n_valid / max(n_total, 1):.2f}%)",
              flush=True)

    # Analyze both arms
    results_a = analyze_arm(
        args.arm_a_dir, args.arm_a_name, args.manifest,
        cam_ids_list, common_masks=common_masks,
    )
    results_b = analyze_arm(
        args.arm_b_dir, args.arm_b_name, args.manifest,
        cam_ids_list, common_masks=common_masks,
    )

    # ------------------------------------------------------------------ #
    # Build comparison report                                            #
    # ------------------------------------------------------------------ #
    report: dict[str, Any] = {
        "arms": {
            args.arm_a_name: results_a,
            args.arm_b_name: results_b,
        },
        "comparison": {},
    }

    # Per-camera comparison
    for cid in cam_ids_list:
        cmp: dict[str, Any] = {}
        a_data = results_a["per_camera"].get(cid, {})
        b_data = results_b["per_camera"].get(cid, {})

        # Radial bin comparison
        for label in _BIN_LABELS:
            a_bin = a_data.get(label, {})
            b_bin = b_data.get(label, {})
            for metric in ("psnr", "gradient_ratio"):
                a_v = a_bin.get(metric, float("nan"))
                b_v = b_bin.get(metric, float("nan"))
                a_ok = not (isinstance(a_v, float) and np.isnan(a_v))
                b_ok = not (isinstance(b_v, float) and np.isnan(b_v))
                if a_ok and b_ok:
                    cmp[f"{label}_{metric}_diff"] = round(
                        float(b_v) - float(a_v),
                        6 if metric == "gradient_ratio" else 4
                    )
                cmp[f"{label}_{metric}_A"] = a_v
                cmp[f"{label}_{metric}_B"] = b_v

        # Full frame
        for metric in ("psnr", "gradient_ratio"):
            a_v = a_data.get("full_frame", {}).get(metric, float("nan"))
            b_v = b_data.get("full_frame", {}).get(metric, float("nan"))
            a_ok = not (isinstance(a_v, float) and np.isnan(a_v))
            b_ok = not (isinstance(b_v, float) and np.isnan(b_v))
            if a_ok and b_ok:
                cmp[f"full_frame_{metric}_diff"] = round(
                    float(b_v) - float(a_v),
                    6 if metric == "gradient_ratio" else 4
                )
            cmp[f"full_frame_{metric}_A"] = a_v
            cmp[f"full_frame_{metric}_B"] = b_v

        # Common-domain comparison
        for label_base in ("common_domain",):
            a_cd = a_data.get(label_base, {})
            b_cd = b_data.get(label_base, {})
            for metric in ("psnr", "gradient_ratio"):
                a_v = a_cd.get(metric, float("nan"))
                b_v = b_cd.get(metric, float("nan"))
                a_ok = not (isinstance(a_v, float) and np.isnan(a_v))
                b_ok = not (isinstance(b_v, float) and np.isnan(b_v))
                if a_ok and b_ok:
                    cmp[f"{label_base}_{metric}_diff"] = round(
                        float(b_v) - float(a_v),
                        6 if metric == "gradient_ratio" else 4
                    )
                cmp[f"{label_base}_{metric}_A"] = a_v
                cmp[f"{label_base}_{metric}_B"] = b_v

        # Common ∩ bin (optional, if present)
        for label in _BIN_LABELS:
            common_key = f"common_{label}"
            a_cb = a_data.get(common_key, {})
            b_cb = b_data.get(common_key, {})
            for metric in ("psnr", "gradient_ratio"):
                a_v = a_cb.get(metric, float("nan"))
                b_v = b_cb.get(metric, float("nan"))
                a_ok = not (isinstance(a_v, float) and np.isnan(a_v))
                b_ok = not (isinstance(b_v, float) and np.isnan(b_v))
                if a_ok and b_ok:
                    cmp[f"{common_key}_{metric}_diff"] = round(
                        float(b_v) - float(a_v),
                        6 if metric == "gradient_ratio" else 4
                    )
                cmp[f"{common_key}_{metric}_A"] = a_v
                cmp[f"{common_key}_{metric}_B"] = b_v

        # Diagnostic: self-masked metrics (labeled as diagnostic, NOT common)
        for key in ("mean_psnr_masked", "mean_cc_psnr_masked", "n_frames"):
            diag_key = f"{key}_diagnostic"
            a_val = a_data.get(diag_key)
            b_val = b_data.get(diag_key)
            if a_val is not None and b_val is not None:
                cmp[f"{key}_diff_diagnostic"] = round(
                    float(b_val) - float(a_val), 4
                )
                cmp[f"{key}_A_diagnostic"] = float(a_val)
                cmp[f"{key}_B_diagnostic"] = float(b_val)
            elif a_val is not None:
                cmp[f"{key}_A_diagnostic"] = float(a_val)
            elif b_val is not None:
                cmp[f"{key}_B_diagnostic"] = float(b_val)

        report["comparison"][cid] = cmp

    # Global comparison
    gcmp: dict[str, Any] = {}
    for key in ("full_frame_psnr", "full_frame_gradient_ratio",
                "common_domain_psnr", "common_domain_gradient_ratio"):
        a_val = results_a["global"].get(key)
        b_val = results_b["global"].get(key)
        a_ok = a_val is not None and not (isinstance(a_val, float) and np.isnan(a_val))
        b_ok = b_val is not None and not (isinstance(b_val, float) and np.isnan(b_val))
        if a_ok and b_ok:
            gcmp[f"{key}_diff"] = round(float(b_val) - float(a_val),
                                        6 if "gradient" in key else 4)
            gcmp[f"{key}_A"] = float(a_val)
            gcmp[f"{key}_B"] = float(b_val)
        elif a_ok:
            gcmp[f"{key}_A"] = float(a_val)
        elif b_ok:
            gcmp[f"{key}_B"] = float(b_val)

    # Diagnostic global metrics
    for key in ("mean_psnr_masked_diagnostic", "mean_cc_psnr_masked_diagnostic",
                "mean_ssim_masked_diagnostic", "mean_lpips_masked_diagnostic",
                "mean_psnr_diagnostic", "mean_cc_psnr_diagnostic",
                "mean_ssim_diagnostic", "mean_lpips_diagnostic"):
        a_val = results_a["global"].get(key)
        b_val = results_b["global"].get(key)
        if a_val is not None and b_val is not None:
            gcmp[f"{key}_diff"] = round(float(b_val) - float(a_val), 4)
            gcmp[f"{key}_A"] = float(a_val)
            gcmp[f"{key}_B"] = float(b_val)
        elif a_val is not None:
            gcmp[f"{key}_A"] = float(a_val)
        elif b_val is not None:
            gcmp[f"{key}_B"] = float(b_val)

    report["comparison"]["global"] = gcmp

    # Write report
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "radial_analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport written to {report_path}", flush=True)

    # Summary table
    print(f"\n{'='*120}", flush=True)
    print("  PIN-AB-1 RADIAL ANALYSIS SUMMARY", flush=True)
    print(f"{'='*120}", flush=True)

    # Radial bins per-camera
    print(f"\n  --- Radial Bin Metrics ---", flush=True)
    print(f"{'Camera':<35} {'Bin':<12} {'PSNR_A':<10} {'PSNR_B':<10} "
          f"{'ΔPSNR':<10} {'GradRatio_A':<12} {'GradRatio_B':<12}",
          flush=True)
    print("-" * 101, flush=True)
    for cid in cam_ids_list:
        cmp = report["comparison"].get(cid, {})
        first_row = True
        for label in _BIN_LABELS:
            a_p = cmp.get(f"{label}_psnr_A", float("nan"))
            b_p = cmp.get(f"{label}_psnr_B", float("nan"))
            d_p = cmp.get(f"{label}_psnr_diff", None)
            a_g = cmp.get(f"{label}_gradient_ratio_A", None)
            b_g = cmp.get(f"{label}_gradient_ratio_B", None)
            cam_label = cid if first_row else ""
            a_p_str = f"{a_p:.4f}" if not (isinstance(a_p, float) and np.isnan(a_p)) else "N/A"
            b_p_str = f"{b_p:.4f}" if not (isinstance(b_p, float) and np.isnan(b_p)) else "N/A"
            d_p_str = f"{d_p:+.4f}" if d_p is not None else " N/A"
            a_g_str = f"{a_g:.6f}" if a_g is not None else "N/A"
            b_g_str = f"{b_g:.6f}" if b_g is not None else "N/A"
            print(
                f"{cam_label:<35} {label:<12} "
                f"{a_p_str:<10} {b_p_str:<10} {d_p_str:<10} "
                f"{a_g_str:<12} {b_g_str:<12}",
                flush=True,
            )
            first_row = False
        # Full frame
        ff_a_p = cmp.get("full_frame_psnr_A", float("nan"))
        ff_b_p = cmp.get("full_frame_psnr_B", float("nan"))
        ff_d_p = cmp.get("full_frame_psnr_diff", None)
        ff_a_g = cmp.get("full_frame_gradient_ratio_A", None)
        ff_b_g = cmp.get("full_frame_gradient_ratio_B", None)
        ff_a_p_str = f"{ff_a_p:.4f}" if not (isinstance(ff_a_p, float) and np.isnan(ff_a_p)) else "N/A"
        ff_b_p_str = f"{ff_b_p:.4f}" if not (isinstance(ff_b_p, float) and np.isnan(ff_b_p)) else "N/A"
        ff_d_p_str = f"{ff_d_p:+.4f}" if ff_d_p is not None else " N/A"
        ff_a_g_str = f"{ff_a_g:.6f}" if ff_a_g is not None else "N/A"
        ff_b_g_str = f"{ff_b_g:.6f}" if ff_b_g is not None else "N/A"
        print(
            f"{'':<35} {'full_frame':<12} "
            f"{ff_a_p_str:<10} {ff_b_p_str:<10} {ff_d_p_str:<10} "
            f"{ff_a_g_str:<12} {ff_b_g_str:<12}",
            flush=True,
        )
        print("-" * 101, flush=True)

    # Common-domain summary
    print(f"\n  --- Common Forward-Valid Domain Metrics ---", flush=True)
    print(f"{'Camera':<35} {'Metric':<20} {'A':<12} {'B':<12} {'Δ':<12}",
          flush=True)
    print("-" * 91, flush=True)
    for cid in cam_ids_list:
        cmp = report["comparison"].get(cid, {})
        for metric_base in ("common_domain",):
            for what in ("psnr", "gradient_ratio"):
                key = f"{metric_base}_{what}"
                a_v = cmp.get(f"{key}_A")
                b_v = cmp.get(f"{key}_B")
                diff = cmp.get(f"{key}_diff")
                cam_label = cid
                metric_label = what
                a_str = f"{float(a_v):.4f}" if a_v is not None else "N/A"
                b_str = f"{float(b_v):.4f}" if b_v is not None else "N/A"
                d_str = f"{float(diff):+.4f}" if diff is not None else " N/A"
                print(
                    f"{cam_label:<35} {metric_label:<20} "
                    f"{a_str:<12} {b_str:<12} {d_str:<12}",
                    flush=True,
                )
                cam_label = ""

    # Global summary
    print(f"\n  --- Global Metrics ---", flush=True)
    gcmp = report["comparison"].get("global", {})
    for key in ("full_frame_psnr", "full_frame_gradient_ratio",
                "common_domain_psnr", "common_domain_gradient_ratio"):
        a_val = gcmp.get(f"{key}_A", None)
        b_val = gcmp.get(f"{key}_B", None)
        diff = gcmp.get(f"{key}_diff", None)
        a_str = f"{a_val}" if a_val is not None else "N/A"
        b_str = f"{b_val}" if b_val is not None else "N/A"
        d_str = f"{diff}" if diff is not None else ""
        if a_val is not None:
            print(f"    {key:<40} A={a_str:<12} B={b_str:<12} Δ={d_str:<12}",
                  flush=True)
        else:
            print(f"    {key:<40} A={a_str:<12} B={b_str:<12}", flush=True)

    print(f"\n  Full report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
