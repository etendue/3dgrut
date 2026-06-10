# SPDX-License-Identifier: Apache-2.0
"""B2: Viser4DOverlayCompositor end-to-end integration test (no viser
required). Verifies the project→render→blend pipeline on a synthetic
flat-gray backdrop + a single cuboid.
"""
from __future__ import annotations

import numpy as np
import pytest

from threedgrut_playground.utils.cuboid import cuboid_world_edges
from threedgrut_playground.utils.viser_overlay_compositor import (
    PolylineLayerSpec,
    Viser4DOverlayCompositor,
)


def _ftheta_dict():
    return {
        "resolution":              np.array([1920, 1080], dtype=np.int64),
        "shutter_type":            "ROLLING_TOP_TO_BOTTOM",
        "principal_point":         np.array([960.0, 540.0], dtype=np.float32),
        "reference_poly":          "ANGLE_TO_PIXELDIST",
        "angle_to_pixeldist_poly": np.array(
            [0.0, 800.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "pixeldist_to_angle_poly": np.array(
            [0.0, 1.0 / 800.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "max_angle":               np.pi / 2 - 0.01,
        "linear_cde":              np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }


def test_compositor_empty_layers_returns_backdrop():
    """No polylines to draw → fast path returns backdrop unchanged."""
    cmp = Viser4DOverlayCompositor(_ftheta_dict(), height=1080, width=1920)
    backdrop = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    out = cmp.composite(backdrop, [], np.eye(4, dtype=np.float64))
    np.testing.assert_array_equal(out, backdrop)


def test_compositor_draws_cuboid_on_gray_backdrop():
    """Place a small cuboid 10 m ahead in OpenCV-cam +Z (= world -Z under
    identity viser c2w + D-combo flip). Expect green pixels somewhere in
    the rendered overlay where line_segments fall.
    """
    cmp = Viser4DOverlayCompositor(_ftheta_dict(), height=1080, width=1920,
                                   subdivide_n=5)
    backdrop = np.full((1080, 1920, 3), 128, dtype=np.uint8)

    # Cuboid 2x2x2 centered at world (0, 0, -10) → cam-frame (0, 0, +10).
    pose = np.eye(4, dtype=np.float32)
    pose[2, 3] = -10.0
    size = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    edges = cuboid_world_edges(pose, size)  # (12, 2, 3)
    polylines = [edges[i] for i in range(12)]  # list of (2, 3)

    layer = PolylineLayerSpec(
        name="cuboid_edges",
        polylines_world=polylines,
        color=(0, 255, 0, 255),
        width=2,
    )
    out = cmp.composite(backdrop, [layer], np.eye(4, dtype=np.float64))

    assert out.shape == (1080, 1920, 3)
    assert out.dtype == np.uint8

    # The backdrop was 128 gray; cuboid edges should write some pure-green
    # pixels (R≈0, G=255, B≈0) where the line lies.
    green_mask = (out[..., 0] < 50) & (out[..., 1] > 200) & (out[..., 2] < 50)
    assert green_mask.sum() > 50, (
        f"expected at least 50 green pixels from cuboid wireframe; "
        f"got {int(green_mask.sum())}"
    )


def test_compositor_shape_mismatch_raises():
    """backdrop with wrong shape vs compositor resolution → ValueError."""
    cmp = Viser4DOverlayCompositor(_ftheta_dict(), height=1080, width=1920)
    backdrop = np.zeros((720, 1280, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="doesn't match"):
        cmp.composite(backdrop, [], np.eye(4))


def test_compositor_layer_ordering_preserved():
    """Two layers with overlapping edges → registration order applied (last on top)."""
    cmp = Viser4DOverlayCompositor(_ftheta_dict(), height=1080, width=1920,
                                   subdivide_n=3)
    backdrop = np.full((1080, 1920, 3), 128, dtype=np.uint8)

    # Same cuboid drawn twice: first red, then blue. Final pixels should be blue.
    pose = np.eye(4, dtype=np.float32)
    pose[2, 3] = -10.0
    edges = cuboid_world_edges(pose, np.array([2., 2., 2.], dtype=np.float32))
    polylines = [edges[i] for i in range(12)]

    layers = [
        PolylineLayerSpec(name="L_red",  polylines_world=polylines,
                          color=(255, 0, 0, 255), width=3),
        PolylineLayerSpec(name="L_blue", polylines_world=polylines,
                          color=(0, 0, 255, 255), width=3),
    ]
    out = cmp.composite(backdrop, layers, np.eye(4, dtype=np.float64))

    # Expect blue pixels (R<50, B>200), few or no pure-red survivors.
    blue_mask = (out[..., 0] < 50) & (out[..., 2] > 200)
    red_mask  = (out[..., 0] > 200) & (out[..., 2] < 50)
    assert blue_mask.sum() > red_mask.sum(), (
        f"blue (top) should dominate red (bottom); "
        f"blue_px={int(blue_mask.sum())} red_px={int(red_mask.sum())}"
    )


# ===================================================== BUG-1b: text labels
def test_compositor_draws_label_at_projected_anchor():
    """labels_world anchors must ride the SAME FTheta projection as the
    wireframe (viewer config: flip=identity, +Z forward)."""
    cmp = Viser4DOverlayCompositor(_ftheta_dict(), height=1080, width=1920,
                                   world_to_camera_flip=np.eye(4))
    backdrop = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    anchor = np.array([0.0, 0.0, 10.0])     # on-axis, 10 m ahead (+Z)
    layer = PolylineLayerSpec(
        name="active_cuboids_t7",
        labels_world=[(anchor, "t7 | automobile")],
        color=(0, 255, 0, 255),
    )
    out = cmp.composite(backdrop, [layer], np.eye(4, dtype=np.float64))
    # On-axis anchor projects to the principal point (960, 540); text is
    # drawn adjacent to it. Some pixels in that neighborhood must differ
    # from the gray backdrop.
    win = out[540 - 40:540 + 10, 960 - 6:960 + 200]
    assert (win != 128).any(), "expected label pixels near principal point"


def test_compositor_skips_behind_camera_label():
    """Anchors behind the camera (-Z with identity flip) must be dropped by
    the compositor's visibility filter — output identical to no-label run."""
    cmp = Viser4DOverlayCompositor(_ftheta_dict(), height=1080, width=1920,
                                   world_to_camera_flip=np.eye(4))
    backdrop = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    behind = np.array([0.0, 0.0, -10.0])
    layer = PolylineLayerSpec(
        name="labels",
        labels_world=[(behind, "should_not_render")],
    )
    out = cmp.composite(backdrop, [layer], np.eye(4, dtype=np.float64))
    np.testing.assert_array_equal(out, backdrop)
