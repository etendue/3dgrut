# SPDX-License-Identifier: Apache-2.0
"""P0.2 / P0.3 — per-class (sseg-based) PSNR / LPIPS evaluator.

``mean_psnr`` / ``cc_psnr_masked`` aggregate over the whole frame (minus the
ego mask), so they cannot say how well the foreground *actors* — pedestrians,
riders, cyclists — or the *lane markings* are reconstructed. ``class_psnr.py``
answers that for tracked vehicles via cuboid projection, but pedestrians have
no cuboids and lane markings have no class. This module fills both gaps using
the semantic-segmentation mask that the eval dataset already loads:

* **P0.2** per-class PSNR/LPIPS over ``person`` (11), ``rider`` (12),
  ``bicycle`` (18) sseg pixels — turns "completely unmodelled" into a number.
* **P0.3** road-crop LPIPS over ``road`` (0) + ``sidewalk`` (1) pixels — PSNR
  is drowned by flat asphalt, but LPIPS is perceptually sensitive to the
  0.1–1 m lane-stripe scale (road particles are capped at 0.3 m XY ≈ 2 stripe
  widths, see layers/layer_spec.py:66), so we report both and let LPIPS carry
  the lane-sharpness signal.

Why a separate pure-tensor module (mirrors class_psnr.py): both render.py's
eval loop and the Mac unit suite need it; keeping it free of model / trainer
state and of cv2 / NCore lets the test suite exercise it without a renderer.
LPIPS is dependency-injected (``lpips_fn``) for the same reason — the module
never imports torchmetrics, and tests pass a fake.

Class IDs mirror the Cityscapes-19 table documented in
``threedgrut/datasets/ncore_semantic.py``. They are *redeclared* here rather
than imported because ``import threedgrut.datasets.ncore_semantic`` executes
``threedgrut/datasets/__init__.py`` → ``datasetNcore`` → cv2 / NCore, which is
unavailable on a dev laptop and would break the Mac test purity this module is
designed to preserve. ``test_per_class_eval.py`` pins the two tables together
so drift is caught.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Tuple

import torch

from threedgrut.model.class_psnr import compute_psnr_in_mask

# Actor classes for P0.2 (id tuple per name; mirror ncore_semantic.py table).
DEFAULT_ACTOR_CLASS_SPECS: Dict[str, Tuple[int, ...]] = {
    "person": (11,),
    "rider": (12,),
    "bicycle": (18,),
}

# Road + sidewalk — the road-crop region for P0.3 (mirror ROAD_CLASS_IDS).
ROAD_CLASS_IDS: Tuple[int, ...] = (0, 1)

# (a, b) are [1, 3, H, W] in [0, 1]; returns a scalar (tensor or float).
LpipsFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def class_mask_from_sseg(sseg: torch.Tensor, ids: Iterable[int]) -> torch.Tensor:
    """``[H, W]`` semantic-id map → ``[H, W]`` bool mask of pixels in ``ids``."""
    sseg_long = sseg.to(torch.long)
    mask = torch.zeros_like(sseg_long, dtype=torch.bool)
    for i in ids:
        mask |= sseg_long == int(i)
    return mask


def compute_lpips_in_mask(
    rgb_pred: torch.Tensor,   # [H, W, 3] in [0, 1]
    rgb_gt: torch.Tensor,     # [H, W, 3]
    mask: torch.Tensor,       # [H, W] float or bool — 1 inside the region
    lpips_fn: LpipsFn,
    min_pixels: int = 50,
) -> Optional[float]:
    """Masked LPIPS via GT-fill.

    torchmetrics LPIPS has no pixel-mask support, so we fill non-mask pixels
    with GT (their perceptual contribution → 0) and run LPIPS on the full
    frame — the exact pattern used for ``lpips_masked`` at
    threedgrut/render.py:608-622. Returns ``None`` when the mask covers fewer
    than ``min_pixels`` (too few pixels for a stable perceptual score).
    """
    mask_f = mask.to(rgb_pred.dtype)
    n_pix = float(mask_f.sum().item())
    if n_pix < min_pixels:
        return None
    m = mask_f.unsqueeze(-1)  # [H, W, 1]
    pred_filled = rgb_pred.clip(0, 1) * m + rgb_gt * (1.0 - m)
    # → [1, 3, H, W] to match the torchmetrics LPIPS call convention.
    pred_in = pred_filled.permute(2, 0, 1).unsqueeze(0)
    gt_in = rgb_gt.permute(2, 0, 1).unsqueeze(0)
    val = lpips_fn(pred_in, gt_in)
    return float(val.item() if torch.is_tensor(val) else val)


def compute_per_class_metrics(
    rgb_pred: torch.Tensor,   # [H, W, 3] in [0, 1]
    rgb_gt: torch.Tensor,     # [H, W, 3]
    sseg: torch.Tensor,       # [H, W] semantic class ids
    class_specs: Dict[str, Iterable[int]],
    *,
    lpips_fn: Optional[LpipsFn] = None,
    min_pixels: int = 50,
) -> Dict[str, Dict[str, object]]:
    """Per-class PSNR (+ optional LPIPS) over sseg-derived masks.

    Args:
        rgb_pred, rgb_gt: one rendered + GT image, ``[H, W, 3]`` in ``[0, 1]``.
        sseg: ``[H, W]`` per-pixel semantic class ids (same resolution).
        class_specs: ``{name: ids}`` — e.g. ``DEFAULT_ACTOR_CLASS_SPECS`` plus
            ``{"road_crop": ROAD_CLASS_IDS}``.
        lpips_fn: optional injected LPIPS callable; ``None`` → lpips skipped.
        min_pixels: per-class stability floor (below → metrics ``None``).

    Returns:
        ``{name: {"psnr": float | None, "lpips": float | None,
                  "n_pixels": int}}``. Classes absent from the frame are still
        reported (``n_pixels`` recorded, metrics ``None``) so "measured, not
        present" is distinguishable from "not measured" — critical for the
        pedestrian floor case where most frames have zero person pixels.
    """
    out: Dict[str, Dict[str, object]] = {}
    for name, ids in class_specs.items():
        mask = class_mask_from_sseg(sseg, ids)
        n_pix = int(mask.sum().item())
        psnr = compute_psnr_in_mask(rgb_pred, rgb_gt, mask, min_pixels=min_pixels)
        lpips_v = (
            compute_lpips_in_mask(rgb_pred, rgb_gt, mask, lpips_fn, min_pixels=min_pixels)
            if lpips_fn is not None
            else None
        )
        out[name] = {"psnr": psnr, "lpips": lpips_v, "n_pixels": n_pix}
    return out
