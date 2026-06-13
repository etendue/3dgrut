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


# -----------------------------------------------------------------------------
# dilate_mask — torch-pure 膨胀（细 lane mask → dilated band）
# -----------------------------------------------------------------------------
from threedgrut.model.per_class_eval import dilate_mask  # noqa: E402


def test_dilate_mask_grows_by_radius():
    m = torch.zeros(11, 11, dtype=torch.bool)
    m[5, 5] = True
    d = dilate_mask(m, 2)  # 5x5 方形结构元
    assert d.dtype == torch.bool
    assert d.shape == (11, 11)
    assert int(d.sum().item()) == 25


def test_dilate_mask_radius_zero_is_identity():
    m = torch.zeros(8, 8, dtype=torch.bool)
    m[3, 4] = True
    d = dilate_mask(m, 0)
    assert d.dtype == torch.bool
    assert torch.equal(d, m)


def test_dilate_mask_clamps_at_border():
    m = torch.zeros(6, 6, dtype=torch.bool)
    m[0, 0] = True  # 角点，radius=2 只有界内 3x3=9 存活
    d = dilate_mask(m, 2)
    assert int(d.sum().item()) == 9


# -----------------------------------------------------------------------------
# _grad_mag_corr_in_mask — 梯度锐度标量
# -----------------------------------------------------------------------------
from threedgrut.model.per_class_eval import _grad_mag_corr_in_mask  # noqa: E402


def test_grad_corr_identical_is_one():
    H = W = 32
    gt = torch.rand(H, W, 3)
    mask = torch.ones(H, W, dtype=torch.bool)
    c = _grad_mag_corr_in_mask(gt.clone(), gt, mask)
    assert c is not None
    assert math.isclose(c, 1.0, abs_tol=1e-4)


def test_grad_corr_flat_pred_returns_none():
    """pred 无边缘（常数）→ 梯度方差 0 → 相关无定义 → None。"""
    H = W = 32
    gt = torch.rand(H, W, 3)
    pred = torch.full((H, W, 3), 0.5)
    mask = torch.ones(H, W, dtype=torch.bool)
    assert _grad_mag_corr_in_mask(pred, gt, mask) is None


def test_grad_corr_too_few_pixels_returns_none():
    H = W = 32
    gt = torch.rand(H, W, 3)
    mask = torch.zeros(H, W, dtype=torch.bool)
    mask[0, :3] = True  # 3 < min_pixels(50)
    assert _grad_mag_corr_in_mask(gt.clone(), gt, mask) is None


def test_grad_corr_orthogonal_edges_below_one():
    """正交边缘（gt 竖边 vs pred 横边）→ 梯度幅值不同位 → 相关明显 < 1
    （非退化，证明不是恒返回 1.0 / None）。"""
    H = W = 32
    gt = torch.zeros(H, W, 3)
    gt[:, 16:] = 1.0   # 竖直边缘
    pred = torch.zeros(H, W, 3)
    pred[16:, :] = 1.0  # 水平边缘
    mask = torch.ones(H, W, dtype=torch.bool)
    c = _grad_mag_corr_in_mask(pred, gt, mask)
    assert c is not None
    assert c < 0.95


# -----------------------------------------------------------------------------
# compute_lane_metrics — dilated-band LPIPS + lane-PSNR + 梯度锐度
# -----------------------------------------------------------------------------
from threedgrut.model.per_class_eval import (  # noqa: E402
    LANE_CLASS_IDS,
    compute_lane_metrics,
)

_LANE_KEYS = {
    "lane_band_lpips", "lane_band_psnr", "lane_raw_psnr",
    "lane_grad_corr", "lane_n_pixels", "lane_band_n_pixels",
}


def test_lane_metrics_dict_keys_exact():
    H = W = 64
    gt = torch.zeros(H, W, 3)
    pred = torch.full((H, W, 3), 0.1)
    lane = torch.zeros(H, W, dtype=torch.long)
    lane[30, :] = LANE_CLASS_IDS[0]  # 一条 1px 横线
    out = compute_lane_metrics(pred, gt, lane, LANE_CLASS_IDS,
                               band_px=8, lpips_fn=_fake_lpips)
    assert set(out.keys()) == _LANE_KEYS


def test_lane_thin_mask_band_is_meaningful():
    """1px lane 线本身像素 < min_pixels（raw LPIPS 会 None），但膨胀后 band 够大
    → band LPIPS 有值。编码 P0 教训：细 mask 必须膨胀才有 LPIPS 信号。"""
    H = W = 64
    gt = torch.zeros(H, W, 3)
    pred = torch.full((H, W, 3), 0.3)
    lane = torch.zeros(H, W, dtype=torch.long)
    lane[30, :] = LANE_CLASS_IDS[0]  # 64 px raw
    out = compute_lane_metrics(pred, gt, lane, LANE_CLASS_IDS,
                               band_px=8, lpips_fn=_fake_lpips)
    assert out["lane_n_pixels"] == 64
    assert out["lane_band_n_pixels"] > 64 * 5  # 膨胀显著放大
    assert out["lane_band_lpips"] is not None
    assert out["lane_band_psnr"] is not None


def test_lane_absent_returns_none_metrics():
    H = W = 64
    gt = torch.zeros(H, W, 3)
    pred = torch.full((H, W, 3), 0.1)
    lane = torch.zeros(H, W, dtype=torch.long)  # 无 lane 像素
    out = compute_lane_metrics(pred, gt, lane, LANE_CLASS_IDS,
                               band_px=8, lpips_fn=_fake_lpips)
    assert out["lane_n_pixels"] == 0
    assert out["lane_band_n_pixels"] == 0
    assert out["lane_band_lpips"] is None
    assert out["lane_band_psnr"] is None
    assert out["lane_raw_psnr"] is None
    assert out["lane_grad_corr"] is None


def test_lane_restrict_mask_limits_region():
    """restrict_mask（如中心 crop / 前视）只保留左半 → raw 像素减半。"""
    H = W = 64
    gt = torch.zeros(H, W, 3)
    pred = torch.full((H, W, 3), 0.1)
    lane = torch.zeros(H, W, dtype=torch.long)
    lane[30, :] = LANE_CLASS_IDS[0]  # 满宽 64 px
    restrict = torch.zeros(H, W, dtype=torch.bool)
    restrict[:, :32] = True  # 左半
    out = compute_lane_metrics(pred, gt, lane, LANE_CLASS_IDS, band_px=0,
                               restrict_mask=restrict, lpips_fn=_fake_lpips)
    assert out["lane_n_pixels"] == 32


def test_lane_class_ids_guard():
    """钉死 lane 类 id：Mapillary Vistas v1.2 的 23=Lane Marking-Crosswalk /
    24=Lane Marking-General（2026-06-09 inceptio 对账 id2label 实测）。"""
    assert LANE_CLASS_IDS == (23, 24)
