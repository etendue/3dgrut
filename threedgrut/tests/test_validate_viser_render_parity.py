from __future__ import annotations

import numpy as np
import pytest

from scripts.validate_viser_render_parity import compute_region_metrics, radial_region_masks


def test_radial_masks_are_disjoint_and_nonempty():
    masks = radial_region_masks(100, 200)
    assert masks["center"].any()
    assert masks["peripheral"].any()
    assert not np.any(masks["center"] & masks["peripheral"])


def test_center_mask_contains_principal_point():
    masks = radial_region_masks(101, 201)
    assert masks["center"][50, 100]


def test_region_metrics_detect_peripheral_only_error():
    ref = np.zeros((100, 200, 3), dtype=np.uint8)
    cand = ref.copy()
    masks = radial_region_masks(100, 200)
    cand[masks["peripheral"]] = 32
    metrics = compute_region_metrics(ref, cand, masks)
    assert metrics["center_mae"] == 0.0
    assert metrics["peripheral_mae"] == pytest.approx(32.0)
    assert metrics["peripheral_psnr"] < metrics["center_psnr"]


def test_region_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError, match="same HxWx3 shape"):
        compute_region_metrics(
            np.zeros((10, 10, 3), dtype=np.uint8),
            np.zeros((8, 10, 3), dtype=np.uint8),
            radial_region_masks(10, 10),
        )
