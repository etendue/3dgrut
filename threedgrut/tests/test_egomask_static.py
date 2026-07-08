# SPDX-License-Identifier: Apache-2.0
"""P0.3 unit tests for egomask_static: polygon / fisheye-outer rasterization +
union-reinforce composition (Phase C, visual-polygon static ego mask).

Pure Mac-CPU tests (numpy + PIL, no ncore SDK). Coordinates: polygon vertices
are (x, y) = (col, row); the returned mask is indexed ``mask[row, col]``.
Fisheye center is (cx, cy) = (col, row). Tolerance: binary, exact np.array_equal.
"""

from __future__ import annotations

import numpy as np

from threedgrut.datasets.egomask_static import (
    build_camera_mask,
    rasterize_fisheye_outer,
    rasterize_polygons,
)

H, W = 20, 30


# --------------------------------------------------------------------------- #
# rasterize_polygons
# --------------------------------------------------------------------------- #
def test_rasterize_single_rectangle_interior_true_exterior_false():
    poly = [(5, 4), (14, 4), (14, 15), (5, 15)]  # x in [5,14], y in [4,15]
    m = rasterize_polygons([poly], (H, W))
    assert m.shape == (H, W)
    assert m.dtype == bool
    assert m[10, 10]  # interior (row10 in [4,15], col10 in [5,14])
    assert m[5, 6]  # interior
    assert m[6:14, 7:13].all()  # conservative interior block
    assert not m[0, 0]  # exterior
    assert not m[19, 29]
    assert not m[2, 10]  # above the rectangle
    assert not m[17:20, 25:30].any()  # far exterior block


def test_rasterize_two_disjoint_polygons_union():
    p1 = [(1, 1), (4, 1), (4, 4), (1, 4)]
    p2 = [(20, 15), (28, 15), (28, 18), (20, 18)]
    m = rasterize_polygons([p1, p2], (H, W))
    assert m[2, 2]  # inside p1
    assert m[16, 24]  # inside p2
    assert not m[10, 15]  # between the two blocks


def test_rasterize_empty_polygons_all_false():
    m = rasterize_polygons([], (H, W))
    assert m.shape == (H, W)
    assert m.dtype == bool
    assert not m.any()


# --------------------------------------------------------------------------- #
# rasterize_fisheye_outer
# --------------------------------------------------------------------------- #
def test_fisheye_outer_matches_distance_field():
    cx, cy, r = 15.0, 10.0, 6.0
    m = rasterize_fisheye_outer((cx, cy), r, (H, W))
    assert m.shape == (H, W)
    assert m.dtype == bool
    assert not m[10, 15]  # image center is inside the imaging circle
    assert m[0, 0]  # far corner outside
    assert m[19, 29]
    yy, xx = np.ogrid[:H, :W]
    expected = (xx - cx) ** 2 + (yy - cy) ** 2 > r**2
    assert np.array_equal(m, expected)


# --------------------------------------------------------------------------- #
# build_camera_mask
# --------------------------------------------------------------------------- #
def test_build_camera_mask_union_of_all_three():
    base = np.zeros((H, W), dtype=bool)
    base[0:3, 0:3] = True
    poly = [(20, 15), (28, 15), (28, 18), (20, 18)]
    fc = (15.0, 10.0, 6.0)
    m = build_camera_mask((H, W), polygons=[poly], fisheye_circle=fc, base_mask=base)
    expected = base | rasterize_polygons([poly], (H, W)) | rasterize_fisheye_outer((15.0, 10.0), 6.0, (H, W))
    assert np.array_equal(m, expected)


def test_build_camera_mask_all_none_all_false():
    m = build_camera_mask((H, W))
    assert m.shape == (H, W)
    assert m.dtype == bool
    assert not m.any()


def test_build_camera_mask_polygons_only():
    poly = [(1, 1), (4, 1), (4, 4), (1, 4)]
    m = build_camera_mask((H, W), polygons=[poly])
    assert m[2, 2]
    assert not m[19, 29]
