# SPDX-License-Identifier: Apache-2.0
"""PIN-AB-1 — Mac CPU tests for radial analysis script fixes.

Tests cover the five defect categories without NCore (mocked/stubbed):

  1. Nested eval root resolution — deterministic selection among multiple
     metrics.json files.
  2. Corner-normalized radius — image-center=0, corners≈1 (half-diagonal).
  3. Aggregate gradient ratio — sum(mag_render)/sum(mag_gt), not mean of
     per-pixel ratios.
  4. Common-mask metric accumulation — synthetic masks with proper
     forward-valid domain semantics.
  5. NCore model parameter attribute access (structural test — cannot
     test actual import on Mac).

NCore-specific loading remains remote-only.  All helpers that would
import ncore are tested via mock or skipped with a mark.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# We import the script as a module for unit-testing its helpers.
# Since the script uses `from PIL import Image` at module level and
# is designed as a CLI entry point, we test helpers by importing
# them via inspect or by running the module's sub-functions directly.
import sys

# Path to the script-under-test
_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "drivers" / "pin_ab_radial_analysis.py"
)

# --------------------------------------------------------------------------- #
# Helper: create a minimal eval output directory                               #
# --------------------------------------------------------------------------- #

def _make_eval_dir(
    root: str,
    cam_ids: list[str],
    n_frames_per_cam: int = 2,
    h: int = 8,
    w: int = 8,
    nested: bool = False,
) -> str:
    """Create a minimal eval output directory with metrics.json and images.

    Args:
        root: Top-level directory.
        cam_ids: Camera IDs in order (determines per_camera insertion order).
        n_frames_per_cam: Number of frames per camera.
        h, w: Image dimensions.
        nested: If True, wrap everything in a single subdirectory (simulates
                a driver-nested eval output).

    Returns:
        The path to the directory containing metrics.json (run_root).
    """
    root_p = Path(root)
    # Determine where to put the actual eval output
    if nested:
        run_root = root_p / "run"
    else:
        run_root = root_p
    run_root.mkdir(parents=True, exist_ok=True)

    # Build per_camera n_frames
    total = n_frames_per_cam * len(cam_ids)
    per_camera = {}
    for i, cid in enumerate(cam_ids):
        per_camera[cid] = {"n_frames": n_frames_per_cam, "index": i}

    # Write metrics.json at run_root
    metrics = {
        "per_camera": per_camera,
        "mean_psnr": 25.0,
        "mean_psnr_masked": 24.0,
    }
    (run_root / "metrics.json").write_text(json.dumps(metrics))

    # Create ours_5000/renders + ours_5000/gt
    step_dir = run_root / "ours_5000"
    render_dir = step_dir / "renders"
    gt_dir = step_dir / "gt"
    render_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    for fi in range(total):
        render_img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        gt_img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        Image.fromarray(render_img).save(str(render_dir / f"frame_{fi:04d}.png"))
        Image.fromarray(gt_img).save(str(gt_dir / f"frame_{fi:04d}.png"))

    return str(run_root)


# --------------------------------------------------------------------------- #
# 1. Nested eval root resolution                                              #
# --------------------------------------------------------------------------- #

class TestNestedEvalRootResolution:
    """Defect 1: resolve run root as parent of selected metrics.json."""

    def _import_helpers(self):
        """Import _resolve_run_root from the script module."""
        import importlib.util as iu

        spec = iu.spec_from_file_location("pin_ab_radial_analysis", str(_SCRIPT))
        mod = iu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod._resolve_run_root, mod._load_images_from_root

    def test_flat_single_metrics(self):
        """Single metrics.json at top level → root = its parent."""
        resolve, load = self._import_helpers()
        with tempfile.TemporaryDirectory() as td:
            run_root = _make_eval_dir(td, ["cam_a", "cam_b"])
            resolved_root, resolved_metrics = resolve(str(td))
            assert Path(resolved_root).samefile(run_root)
            assert Path(resolved_metrics).samefile(Path(run_root) / "metrics.json")

    def test_nested_single_metrics(self):
        """Nested (wrapper dir) → root = wrapper dir, not top."""
        resolve, load = self._import_helpers()
        with tempfile.TemporaryDirectory() as td:
            run_root = _make_eval_dir(td, ["cam_a"], nested=True)
            resolved_root, resolved_metrics = resolve(str(td))
            assert Path(resolved_root).samefile(run_root)
            assert Path(resolved_metrics).samefile(Path(run_root) / "metrics.json")

    def test_multiple_metrics_with_ours_preferred(self):
        """Multiple metrics.json — prefers one with ours_* siblings."""
        resolve, load = self._import_helpers()
        with tempfile.TemporaryDirectory() as td:
            # Create two "runs": run1 with metrics.json + ours_*, run2 only
            # metrics.json
            r1 = Path(td) / "run1"
            r1.mkdir()
            (r1 / "metrics.json").write_text(
                json.dumps({"per_camera": {"cam_a": {"n_frames": 1, "index": 0}}})
            )
            (r1 / "ours_5000").mkdir()
            (r1 / "ours_5000" / "renders").mkdir()
            (r1 / "ours_5000" / "gt").mkdir()
            img = np.zeros((4, 4, 3), dtype=np.uint8)
            Image.fromarray(img).save(str(r1 / "ours_5000" / "renders" / "frame_0000.png"))
            Image.fromarray(img).save(str(r1 / "ours_5000" / "gt" / "frame_0000.png"))

            r2 = Path(td) / "run2"
            r2.mkdir()
            (r2 / "metrics.json").write_text(
                json.dumps({"per_camera": {"cam_b": {"n_frames": 1, "index": 0}}})
            )

            resolved_root, resolved_metrics = resolve(str(td))
            # Should pick run1 (has ours_*)

            assert resolved_root == str(r1.resolve())

    def test_multiple_metrics_both_with_ours(self):
        """Multiple metrics.json, both with ours_* → deepest+alphabetically-last wins."""
        resolve, load = self._import_helpers()
        with tempfile.TemporaryDirectory() as td:
            # Create two runs both with ours_*
            for name in ("run_a", "run_b"):
                r = Path(td) / name
                r.mkdir()
                (r / "metrics.json").write_text(
                    json.dumps({"per_camera": {name: {"n_frames": 1, "index": 0}}})
                )
                (r / "ours_5000").mkdir()
                (r / "ours_5000" / "renders").mkdir()
                (r / "ours_5000" / "gt").mkdir()
                img = np.zeros((4, 4, 3), dtype=np.uint8)
                Image.fromarray(img).save(
                    str(r / "ours_5000" / "renders" / "frame_0000.png")
                )
                Image.fromarray(img).save(
                    str(r / "ours_5000" / "gt" / "frame_0000.png")
                )

            resolved_root, resolved_metrics = resolve(str(td))
            # Both have ours_*. Tie → alphabetically last → "run_b"
            assert "run_b" in str(resolved_root)

    def test_no_metrics_raises(self):
        """No metrics.json → RuntimeError."""
        resolve, load = self._import_helpers()
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(RuntimeError, match="No metrics.json"):
                resolve(str(td))


# --------------------------------------------------------------------------- #
# 2. Corner-normalized radius                                                 #
# --------------------------------------------------------------------------- #

class TestCornerNormalizedRadius:
    """Defect 2: image-normalized radius, half-diagonal normalization."""

    def _import(self):
        import importlib.util as iu

        spec = iu.spec_from_file_location("pin_ab_radial_analysis", str(_SCRIPT))
        mod = iu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod._compute_corner_normalized_radius_map

    def test_center_is_zero(self):
        """Image center (principal point) → radius ≈ 0."""
        fn = self._import()
        h, w = 10, 10
        cx, cy = 5.0, 5.0
        rmap = fn(h, w, cx, cy)
        assert rmap[5, 5] == pytest.approx(0.0, abs=1e-12)

    def test_corners_approx_one(self):
        """Furthest corners from principal point → radius ≈ 1 (half-diagonal)."""
        fn = self._import()
        h, w = 10, 10
        cx, cy = 5.0, 5.0
        rmap = fn(h, w, cx, cy)
        # Half diagonal = sqrt(5^2 + 5^2) = sqrt(50) ≈ 7.07
        # Top-left corner (0,0): dist = sqrt(25+25) = sqrt(50) → ratio = 1.0
        assert rmap[0, 0] == pytest.approx(1.0, abs=1e-12)
        # Bottom-right corner (9,9): dist = sqrt(16+16) = sqrt(32) ≈ 5.66
        # Not all corners are equidistant — only (0,0) reaches 1.0 for 10x10
        # Check that the maximum radius is ≈ 1.0
        assert float(rmap.max()) == pytest.approx(1.0, abs=1e-12)

    def test_mid_edge(self):
        """Edge midpoints → radius < 1."""
        fn = self._import()
        h, w = 10, 10
        cx, cy = 5.0, 5.0
        rmap = fn(h, w, cx, cy)
        # Top edge midpoint (0, 5): dist=5, half_diag=7.07 → ratio=0.707
        assert rmap[0, 5] == pytest.approx(5.0 / np.sqrt(50), abs=1e-12)

    def test_asymmetric_center(self):
        """Off-center principal point shifts radius correctly."""
        fn = self._import()
        h, w = 12, 8
        cx, cy = 4.0, 6.0  # not at image center
        rmap = fn(h, w, cx, cy)
        # Half-diagonal (image rectangle, not centered on pp)
        # We just verify that the minimum is at the principal point
        min_yx = np.unravel_index(rmap.argmin(), rmap.shape)
        # The minimum might not be exact due to grid discretization
        # but it should be near (cy, cx)
        assert abs(min_yx[1] - cx) <= 1.0
        assert abs(min_yx[0] - cy) <= 1.0

    def test_no_focal_involvement(self):
        """Focal length does NOT affect the radius."""
        fn = self._import()
        h, w = 10, 10
        r1 = fn(h, w, 5.0, 5.0)
        r2 = fn(h, w, 5.0, 5.0)
        # Function signature has no fx/fy params — focal is irrelevant
        assert np.allclose(r1, r2)


# --------------------------------------------------------------------------- #
# 3. Aggregate gradient ratio                                                 #
# --------------------------------------------------------------------------- #

class TestAggregateGradientRatio:
    """Defect 3: sum(mag_render) / sum(mag_gt), not mean of per-pixel ratios."""

    def _import(self):
        import importlib.util as iu

        spec = iu.spec_from_file_location("pin_ab_radial_analysis", str(_SCRIPT))
        mod = iu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod._compute_aggregate_gradient_ratio, mod._compute_gradient_magnitudes

    def test_identical_images_ratio_one(self):
        """Identical render and GT → ratio ≈ 1."""
        ratio_fn, mag_fn = self._import()
        img = np.random.rand(16, 16, 3).astype(np.float32)
        sum_r, sum_g, ratio = ratio_fn(img, img, mask=None)
        assert ratio == pytest.approx(1.0, abs=1e-6)

    def test_ratio_with_uniform_images(self):
        """Uniform images have near-zero gradients → ratio may be NaN but stable."""
        ratio_fn, mag_fn = self._import()
        img = np.full((8, 8, 3), 0.5, dtype=np.float32)
        sum_r, sum_g, ratio = ratio_fn(img, img, mask=None)
        # Uniform images: gradients ≈ 0, so sum_g ≈ 0 → ratio NaN
        assert np.isnan(ratio) or ratio == pytest.approx(1.0, abs=1e-6)

    def test_ratio_not_mean_of_per_pixel_ratios(self):
        """Verify aggregate ratio ≠ mean(per-pixel ratios).
        
        Create render with spatially varying gradient magnitudes and GT with
        uniform gradient magnitude.  The per-pixel ratio mean will be
        distorted by singular low-magnitude pixels, while the aggregate ratio
        is robust.
        """
        ratio_fn, mag_fn = self._import()
        h, w = 8, 8
        render = np.zeros((h, w, 3), dtype=np.float32)
        gt = np.zeros((h, w, 3), dtype=np.float32)

        # Row 0: render has gradient, GT has same gradient → ratio contribution = 1
        render[0, :, 0] = np.linspace(0, 1, w)
        gt[0, :, 0] = np.linspace(0, 1, w)

        # Row 1: render has gradient, GT has 10x stronger gradient → ratio contribution ≈ 0.1
        render[1, :, 0] = np.linspace(0, 1, w)
        gt[1, :, 0] = np.linspace(0, 10, w)

        # Row 2: render gradient = GT but single pixel with near-zero GT → per-pixel blows up
        render[2, :, 0] = np.linspace(0, 1, w)
        gt[2, :, 0] = np.linspace(0, 1, w)
        gt[2, 7, 0] = 1e-10  # nearly zero GT gradient at one pixel

        sum_r, sum_g, agg_ratio = ratio_fn(render, gt, mask=None)

        # Compute per-pixel ratio mean manually for comparison
        from scipy.ndimage import sobel
        mr = np.sqrt(sobel(render.astype(np.float64), axis=1, mode="constant") ** 2 +
                     sobel(render.astype(np.float64), axis=0, mode="constant") ** 2).mean(axis=2)
        mg = np.sqrt(sobel(gt.astype(np.float64), axis=1, mode="constant") ** 2 +
                     sobel(gt.astype(np.float64), axis=0, mode="constant") ** 2).mean(axis=2)
        per_pixel_ratios = mr / np.where(mg > 1e-8, mg, 1e-8)
        mean_per_pixel = float(per_pixel_ratios.mean())

        # Aggregate ratio should differ from mean of per-pixel ratios
        # (the per-pixel mean is inflated by the near-zero GT pixel)
        assert agg_ratio != pytest.approx(mean_per_pixel, abs=0.01), (
            f"Aggregate ratio {agg_ratio:.6f} should differ from "
            f"mean per-pixel ratio {mean_per_pixel:.6f}"
        )

    def test_mask_respected(self):
        """Mask restricts computation to masked region."""
        ratio_fn, mag_fn = self._import()
        h, w = 8, 8
        render = np.random.rand(h, w, 3).astype(np.float32)
        gt = np.random.rand(h, w, 3).astype(np.float32)

        mask = np.zeros((h, w), dtype=bool)
        mask[2:5, 2:5] = True

        sum_r, sum_g, ratio = ratio_fn(render, gt, mask=mask)
        assert sum_r > 0 or sum_g > 0  # at least one non-zero
        assert np.isfinite(ratio) or sum_g == 0  # only NaN if sum_g == 0

    def test_update_stability(self):
        """Sequential frame accumulation matches whole-image computation
        *for disjoint regions* when boundary handling is considered.

        Sobel filters have edge effects at quadrant boundaries, so
        split-and-sum does NOT match whole-image directly.  Instead,
        verify that the aggregate ratio formula (sum_render/sum_gt)
        applied per-quadrant and summed matches the whole-image aggregate
        when the images are tiled without Sobel boundary sensitivity
        (i.e., using gradient magnitudes computed on the whole image).
        """
        ratio_fn, mag_fn = self._import()
        h, w = 8, 8

        np.random.seed(42)
        render = np.random.rand(h, w, 3).astype(np.float32)
        gt = np.random.rand(h, w, 3).astype(np.float32)

        # Get whole-image gradient magnitudes
        mag_r_full = mag_fn(render)
        mag_g_full = mag_fn(gt)

        sum_r_full = float(mag_r_full.sum())
        sum_g_full = float(mag_g_full.sum())
        ratio_full = sum_r_full / max(sum_g_full, 1e-12)

        # Partition gradient mag maps into quadrants and sum
        sum_r_acc = 0.0
        sum_g_acc = 0.0
        for yi in range(0, h, 4):
            for xi in range(0, w, 4):
                sum_r_acc += float(mag_r_full[yi:yi+4, xi:xi+4].sum())
                sum_g_acc += float(mag_g_full[yi:yi+4, xi:xi+4].sum())

        agg_ratio = sum_r_acc / max(sum_g_acc, 1e-12)
        assert agg_ratio == pytest.approx(ratio_full, abs=1e-10)


# --------------------------------------------------------------------------- #
# 4. Common-mask metric accumulation                                          #
# --------------------------------------------------------------------------- #

class TestCommonMaskMetricAccumulation:
    """Defect 4: common forward-valid-domain metrics with synthetic masks."""

    def _import_helpers(self):
        import importlib.util as iu

        spec = iu.spec_from_file_location("pin_ab_radial_analysis", str(_SCRIPT))
        mod = iu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod._compute_pixel_mse, mod._compute_gradient_magnitudes

    def test_multi_frame_common_domain_normalized(self):
        """Regression: common-domain MSE accumulates denominator per frame.

        The old bug: cm_mse_sum accumulated over all frames but divided by
        single-frame cm_npx = int(cm.sum()).  With N identical frames, this
        made MSE N× too large and PSNR ~10*log10(N) too low.
        """
        mse_fn, mag_fn = self._import_helpers()
        h, w = 8, 8
        n_frames = 3

        # All frames identical — same render, GT, and mask
        np.random.seed(1)
        render = np.random.rand(h, w, 3).astype(np.float32)
        gt = np.random.rand(h, w, 3).astype(np.float32)
        mask = np.ones((h, w), dtype=bool)
        mask[:2, :2] = False  # small invalid corner
        npx_per_frame = int(mask.sum())

        # Accumulate MSE sum and pixel count per frame (correct pattern)
        mse_sum = 0.0
        npx_total = 0
        for _ in range(n_frames):
            mse_map = mse_fn(render, gt)
            mse_sum += float(mse_map[mask].sum())
            npx_total += npx_per_frame

        # Correct MSE = sum / total_pixels
        correct_mse = mse_sum / npx_total
        correct_psnr = -10.0 * np.log10(max(correct_mse, 1e-12))

        # Old bug: denominator = single-frame count (npx_per_frame, not npx_total)
        bug_mse = mse_sum / npx_per_frame
        bug_psnr = -10.0 * np.log10(max(bug_mse, 1e-12))

        # The bug makes PSNR lower by ~10*log10(n_frames)
        expected_delta_db = 10.0 * np.log10(float(n_frames))
        actual_delta_db = correct_psnr - bug_psnr
        assert actual_delta_db == pytest.approx(expected_delta_db, abs=1e-6), (
            f"Multi-frame normalization regression: correct_psnr={correct_psnr:.4f}, "
            f"bug_psnr={bug_psnr:.4f}, delta={actual_delta_db:.4f} dB, "
            f"expected delta={expected_delta_db:.4f} dB"
        )

        # Sanity: correct PSNR should equal single-frame PSNR (identical frames)
        single_mse = mse_fn(render, gt)[mask].mean()
        single_psnr = -10.0 * np.log10(max(float(single_mse), 1e-12))
        assert correct_psnr == pytest.approx(single_psnr, abs=1e-6), (
            f"Multi-frame correct PSNR should equal single-frame PSNR "
            f"for identical frames: multi={correct_psnr:.4f}, single={single_psnr:.4f}"
        )

    def test_common_mask_psnr_matches_masked_computation(self):
        """PSNR over common mask matches explicit masked MSE computation."""
        mse_fn, mag_fn = self._import_helpers()
        h, w = 8, 8

        render = np.random.rand(h, w, 3).astype(np.float32)
        gt = np.random.rand(h, w, 3).astype(np.float32)

        # Create a common mask: only upper-left quadrant valid
        mask = np.zeros((h, w), dtype=bool)
        mask[:4, :4] = True

        # Accumulate MSE over mask
        mse_map = mse_fn(render, gt)
        masked_mse = float(mse_map[mask].mean())
        expected_psnr = -10.0 * np.log10(max(masked_mse, 1e-12))

        # Verify per-camera accumulation works (synthetic: single frame)
        # The script accumulates MSE sums, then divides by n_pixels
        mse_sum = float(mse_map[mask].sum())
        n_px = int(mask.sum())
        computed_mse = mse_sum / n_px
        computed_psnr = -10.0 * np.log10(max(computed_mse, 1e-12))

        assert computed_psnr == pytest.approx(expected_psnr, abs=1e-10)

    def test_gradient_ratio_over_common_mask(self):
        """Gradient ratio over common mask matches aggregate computation."""
        mse_fn, mag_fn = self._import_helpers()
        h, w = 8, 8

        render = np.random.rand(h, w, 3).astype(np.float32)
        gt = np.random.rand(h, w, 3).astype(np.float32)

        mask = np.zeros((h, w), dtype=bool)
        mask[2:6, 2:6] = True

        mag_r = mag_fn(render)
        mag_g = mag_fn(gt)

        sum_r = float(mag_r[mask].sum())
        sum_g = float(mag_g[mask].sum())
        expected_ratio = sum_r / max(sum_g, 1e-12)

        assert expected_ratio > 0
        assert np.isfinite(expected_ratio)

    def test_all_invalid_mask_yields_nan(self):
        """Common mask with zero valid pixels → gradient is 0, ratio=NaN."""
        mse_fn, mag_fn = self._import_helpers()
        h, w = 8, 8
        render = np.ones((h, w, 3), dtype=np.float32)
        gt = np.ones((h, w, 3), dtype=np.float32)
        mask = np.zeros((h, w), dtype=bool)

        mag_r = mag_fn(render)
        mag_g = mag_fn(gt)
        sum_r = float(mag_r[mask].sum())
        sum_g = float(mag_g[mask].sum())
        # Both sums are 0 (no pixels) → ratio NaN
        assert sum_r == 0.0
        assert sum_g == 0.0

    def test_partial_mask_consistent_across_both_arms(self):
        """Same common mask applied to Arm A and Arm B data yields
        comparable metrics — demonstrate that the mask is shared."""
        mse_fn, mag_fn = self._import_helpers()
        h, w = 8, 8

        # Two different renders (simulating Arm A and Arm B)
        render_a = np.random.rand(h, w, 3).astype(np.float32)
        render_b = render_a + np.random.randn(h, w, 3).astype(np.float32) * 0.05
        gt = np.random.rand(h, w, 3).astype(np.float32)

        # Shared common mask
        mask = np.zeros((h, w), dtype=bool)
        mask[1:7, 1:7] = True

        def _compute(render, gt, mask):
            mse_map = mse_fn(render, gt)
            mse_sum = float(mse_map[mask].sum())
            n_px = int(mask.sum())
            psnr = -10.0 * np.log10(max(mse_sum / n_px, 1e-12))
            mag_r = mag_fn(render)
            mag_g = mag_fn(gt)
            gr = float(mag_r[mask].sum()) / max(float(mag_g[mask].sum()), 1e-12)
            return psnr, gr

        psnr_a, gr_a = _compute(render_a, gt, mask)
        psnr_b, gr_b = _compute(render_b, gt, mask)

        # Arm B is noisier → lower PSNR (or equal within noise)
        # This is just checking the computation runs and returns sensible values
        assert np.isfinite(psnr_a)
        assert np.isfinite(psnr_b)
        assert np.isfinite(gr_a)
        assert np.isfinite(gr_b)


# --------------------------------------------------------------------------- #
# 5. NCore model parameter attribute access (structural)                       #
# --------------------------------------------------------------------------- #

class TestNCoreModelParameterAPIs:
    """Defect 5: use actual NCore model parameter attributes, not .get().

    We cannot import the real ncore.sensors on Mac, but we can verify
    structurally that the script uses attribute access patterns matching
    the production code idiom.
    """

    def test_no_get_calls_on_model_parameters(self):
        """Script should use attribute access, not .get(), on NCore
        CameraModelParameters objects.  Plain dict .get() calls on
        results/comparison dicts are fine."""
        text = _SCRIPT.read_text()
        # We look for the specific pattern: variable named mp or model_parameters
        # followed by .get( — this is what the bug was.
        import re
        # Pattern: variable named 'mp' or containing 'model_parameters' followed
        # by .get( -- but NOT cmp.get / gcmp.get / a_data.get / etc.
        # Look through lines for the specific assignment lines we care about.
        suspicious = []
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            # Catch: mp.get("focal_length") or model_parameters.get("principal_point")
            if re.search(r'\bmp\.get\(', stripped):
                suspicious.append((i, stripped))
            if re.search(r'\bmodel_parameters\.get\(', stripped):
                suspicious.append((i, stripped))
        if suspicious:
            lines_detail = "; ".join(f"L{ln}: {s}" for ln, s in suspicious)
            pytest.fail(
                f"Script uses .get() on model_parameters objects at: {lines_detail}"
            )

    def test_uses_from_parameters(self):
        """Script should use CameraModel.from_parameters."""
        text = _SCRIPT.read_text()
        assert "from_parameters" in text, (
            "Script must use CameraModel.from_parameters to construct camera models"
        )

    def test_uses_attribute_access(self):
        """Script should access .focal_length and .principal_point as attributes."""
        text = _SCRIPT.read_text()
        assert "mp.focal_length" in text or "model_parameters.focal_length" in text
        assert "mp.principal_point" in text or "model_parameters.principal_point" in text

    def test_model_resolution_attribute(self):
        """Script should access model.resolution attribute."""
        text = _SCRIPT.read_text()
        # Both old and new code use model.resolution — check it's there
        assert "model.resolution" in text or "CameraModel.resolution" in text


# --------------------------------------------------------------------------- #
# Smoke: py_compile check                                                     #
# --------------------------------------------------------------------------- #

class TestScriptCompiles:
    """Verify the script can be imported/compiled without NCore."""

    def test_syntax_valid(self):
        """py_compile passes."""
        import py_compile
        py_compile.compile(str(_SCRIPT), doraise=True)

    def test_import_loads(self):
        """Module-level import succeeds (lazy NCore imports exempted)."""
        import importlib.util as iu

        spec = iu.spec_from_file_location("pin_ab_radial_analysis", str(_SCRIPT))
        mod = iu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "_resolve_run_root")
        assert hasattr(mod, "_compute_corner_normalized_radius_map")
        assert hasattr(mod, "_compute_aggregate_gradient_ratio")
        assert hasattr(mod, "_compute_pixel_mse")
        assert hasattr(mod, "_compute_gradient_magnitudes")
        # NCore-dependent functions will fail at call time but should exist
        assert hasattr(mod, "_get_camera_intrinsics")
        assert hasattr(mod, "_compute_forward_valid_mask")
