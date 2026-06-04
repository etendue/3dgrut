# SPDX-License-Identifier: Apache-2.0
"""P0.2 / P0.3 — per-class (sseg-based) PSNR / LPIPS evaluator tests.

Pure-tensor functions; mockable on Mac without renderer, NCore, cv2 or
torchmetrics (LPIPS is dependency-injected so a fake stands in). Mirrors the
test style of test_class_psnr.py.
"""
from __future__ import annotations

import math

import torch

from threedgrut.model.per_class_eval import (
    DEFAULT_ACTOR_CLASS_SPECS,
    ROAD_CLASS_IDS,
    class_mask_from_sseg,
    compute_lpips_in_mask,
    compute_per_class_metrics,
)


# -----------------------------------------------------------------------------
# class_mask_from_sseg
# -----------------------------------------------------------------------------

def test_class_mask_selects_only_given_id():
    sseg = torch.tensor([[11, 12, 0], [18, 2, 11]])
    m = class_mask_from_sseg(sseg, (11,))
    assert m.dtype == torch.bool
    assert m.tolist() == [[True, False, False], [False, False, True]]


def test_class_mask_multiple_ids_union():
    sseg = torch.tensor([[0, 1, 2], [10, 11, 18]])
    m = class_mask_from_sseg(sseg, ROAD_CLASS_IDS)  # {0, 1}
    assert m.tolist() == [[True, True, False], [False, False, False]]


def test_class_mask_accepts_float_sseg():
    sseg = torch.tensor([[11.0, 0.0], [18.0, 11.0]])
    m = class_mask_from_sseg(sseg, (11,))
    assert m.tolist() == [[True, False], [False, True]]


# -----------------------------------------------------------------------------
# compute_per_class_metrics — PSNR
# -----------------------------------------------------------------------------

def test_per_class_psnr_known_value_for_present_class():
    """diff² = 0.01 inside person mask → PSNR = -10·log10(0.01) = 20 dB."""
    H = W = 64
    gt = torch.zeros(H, W, 3)
    pred = torch.full((H, W, 3), 0.1)
    sseg = torch.zeros(H, W, dtype=torch.long)
    sseg[:32] = 11  # top half = person
    out = compute_per_class_metrics(pred, gt, sseg, {"person": (11,)})
    assert out["person"]["n_pixels"] == 32 * W
    assert math.isclose(out["person"]["psnr"], 20.0, abs_tol=1e-3)
    assert out["person"]["lpips"] is None  # no lpips_fn provided


def test_per_class_absent_class_reported_with_none_psnr():
    """Class with 0 pixels is still reported (n_pixels=0, psnr=None) so
    'measured, not present' is distinguishable from 'not measured'."""
    H = W = 64
    gt = torch.zeros(H, W, 3)
    pred = torch.full((H, W, 3), 0.1)
    sseg = torch.zeros(H, W, dtype=torch.long)  # no person anywhere
    out = compute_per_class_metrics(pred, gt, sseg, {"person": (11,)})
    assert out["person"]["n_pixels"] == 0
    assert out["person"]["psnr"] is None


def test_per_class_only_mask_region_counted():
    """Pixels outside the person mask are wrong, but per-class PSNR reflects
    only the masked region."""
    H = W = 64
    gt = torch.zeros(H, W, 3)
    pred = torch.zeros(H, W, 3)
    pred[:32] = 0.1   # person region: diff² = 0.01 → 20 dB
    pred[32:] = 1.0   # would dominate the global average if not masked
    sseg = torch.zeros(H, W, dtype=torch.long)
    sseg[:32] = 11
    out = compute_per_class_metrics(pred, gt, sseg, {"person": (11,)})
    assert math.isclose(out["person"]["psnr"], 20.0, abs_tol=1e-3)


# -----------------------------------------------------------------------------
# compute_lpips_in_mask — GT-fill (fake lpips injected)
# -----------------------------------------------------------------------------

def _fake_lpips(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Stand-in matching torchmetrics' call signature: a, b are [1,3,H,W];
    returns a scalar tensor. Mean abs diff makes GT-fill behaviour checkable."""
    assert a.shape == b.shape and a.dim() == 4 and a.shape[1] == 3
    return (a - b).abs().mean()


def test_lpips_full_mask_equals_full_image():
    H = W = 64
    gt = torch.zeros(H, W, 3)
    pred = torch.full((H, W, 3), 0.2)
    mask = torch.ones(H, W)
    val = compute_lpips_in_mask(pred, gt, mask, _fake_lpips)
    # full mask → pred_filled == pred.clip(0,1) → fake = mean|0.2-0| = 0.2
    assert math.isclose(val, 0.2, abs_tol=1e-5)


def test_lpips_empty_mask_returns_none():
    H = W = 64
    gt = torch.zeros(H, W, 3)
    pred = torch.full((H, W, 3), 0.2)
    mask = torch.zeros(H, W)
    assert compute_lpips_in_mask(pred, gt, mask, _fake_lpips) is None


def test_lpips_gtfill_zeroes_outside_mask_contribution():
    """Outside-mask pixels are filled with GT → contribute 0 to the metric."""
    H = W = 64
    gt = torch.zeros(H, W, 3)
    pred = torch.full((H, W, 3), 0.4)
    mask = torch.zeros(H, W)
    mask[:32] = 1.0  # half inside
    val = compute_lpips_in_mask(pred, gt, mask, _fake_lpips)
    # inside half: |0.4-0| = 0.4 ; outside half: GT-filled → 0
    # mean over all pixels = 0.4 * 0.5 = 0.2
    assert math.isclose(val, 0.2, abs_tol=1e-5)


# -----------------------------------------------------------------------------
# compute_per_class_metrics — full spec set (P0.2 actors + P0.3 road)
# -----------------------------------------------------------------------------

def test_per_class_metrics_all_specs_with_lpips():
    H = W = 64
    gt = torch.zeros(H, W, 3)
    pred = torch.full((H, W, 3), 0.1)
    sseg = torch.zeros(H, W, dtype=torch.long)
    sseg[:20] = 11    # person
    sseg[20:40] = 12  # rider
    sseg[40:] = 0     # road
    specs = {**DEFAULT_ACTOR_CLASS_SPECS, "road_crop": ROAD_CLASS_IDS}
    out = compute_per_class_metrics(pred, gt, sseg, specs, lpips_fn=_fake_lpips)
    assert set(out.keys()) == {"person", "rider", "bicycle", "road_crop"}
    # bicycle (18) absent → reported but None
    assert out["bicycle"]["n_pixels"] == 0
    assert out["bicycle"]["psnr"] is None
    assert out["bicycle"]["lpips"] is None
    # present classes get both metrics
    assert out["person"]["n_pixels"] == 20 * W
    assert out["person"]["lpips"] is not None
    assert out["road_crop"]["n_pixels"] == 24 * W
    assert math.isclose(out["road_crop"]["psnr"], 20.0, abs_tol=1e-3)


def test_default_actor_specs_match_ncore_semantic_table():
    """Guards the cross-referenced IDs (ncore_semantic.py table) from drift."""
    assert DEFAULT_ACTOR_CLASS_SPECS["person"] == (11,)
    assert DEFAULT_ACTOR_CLASS_SPECS["rider"] == (12,)
    assert DEFAULT_ACTOR_CLASS_SPECS["bicycle"] == (18,)
    assert tuple(sorted(ROAD_CLASS_IDS)) == (0, 1)
