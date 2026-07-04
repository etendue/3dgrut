# SPDX-License-Identifier: Apache-2.0
"""A1 — repair_nonfinite_rays: rational-distortion pole guard at ray precompute.

NCore ``pixels_to_camera_rays`` can emit non-finite directions where the
rational-model undistortion diverges (inc_b6a9 camera_left_wide_90fov: the
denominator 1+k4r²+k5r⁴+k6r⁶ has a pole inside the image corner → exactly one
NaN ray at px (u=1917, v=1042) on every frame). The guard repairs the cached
ray (nearest finite same-row neighbour) and flags the pixel invalid so it
never supervises training — upstream of the trainer's drop-batch fuse.
"""

from __future__ import annotations

import numpy as np

from threedgrut.datasets.utils import repair_nonfinite_rays


def _rays(h=4, w=6):
    r = np.random.rand(h, w, 3).astype(np.float32) + 0.1
    return r / np.linalg.norm(r, axis=-1, keepdims=True)


def test_clean_rays_untouched_and_zero_count():
    rays = _rays()
    before = rays.copy()
    mask = np.ones(rays.shape[:2], dtype=bool)
    n = repair_nonfinite_rays(rays, mask)
    assert n == 0
    assert np.array_equal(rays, before)
    assert mask.all()


def test_single_nan_ray_repaired_from_row_neighbour_and_masked():
    rays = _rays()
    rays[2, 4] = np.nan
    mask = np.ones(rays.shape[:2], dtype=bool)
    n = repair_nonfinite_rays(rays, mask)
    assert n == 1
    assert np.isfinite(rays).all()
    # nearest finite same-row neighbour (u=3 or u=5)
    assert np.array_equal(rays[2, 4], rays[2, 3]) or np.array_equal(rays[2, 4], rays[2, 5])
    assert not mask[2, 4]
    assert mask.sum() == mask.size - 1  # only the bad pixel flipped


def test_inf_ray_also_repaired():
    rays = _rays()
    rays[0, 0, 2] = np.inf
    mask = np.ones(rays.shape[:2], dtype=bool)
    n = repair_nonfinite_rays(rays, mask)
    assert n == 1
    assert np.isfinite(rays).all()
    assert not mask[0, 0]


def test_whole_row_bad_falls_back_to_unit_z():
    rays = _rays(h=2, w=3)
    rays[1, :, :] = np.nan
    mask = np.ones(rays.shape[:2], dtype=bool)
    n = repair_nonfinite_rays(rays, mask)
    assert n == 3
    assert np.isfinite(rays).all()
    assert np.allclose(rays[1], np.array([0.0, 0.0, 1.0]))
    assert not mask[1].any()


def test_flat_n3_rays_without_mask():
    rays = _rays(h=1, w=8).reshape(8, 3)
    rays[5] = np.nan
    n = repair_nonfinite_rays(rays, None)
    assert n == 1
    assert np.isfinite(rays).all()
