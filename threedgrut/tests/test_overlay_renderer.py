# SPDX-License-Identifier: Apache-2.0
"""B2: OverlayRenderer + alpha_blend unit tests."""
from __future__ import annotations

import numpy as np
import pytest

from threedgrut_playground.utils.overlay_renderer import (
    OverlayLayer,
    OverlayRenderer,
    alpha_blend,
)


def test_render_empty_layer_returns_transparent():
    """Renderer with no polylines → fully transparent (alpha=0 everywhere)."""
    r = OverlayRenderer(height=64, width=64)
    out = r.render([])
    assert out.shape == (64, 64, 4)
    assert out.dtype == np.uint8
    assert out[..., 3].max() == 0


def test_render_single_horizontal_segment():
    """Draw 1 segment from (10, 32) to (50, 32) with green color → those
    pixels should have nonzero green alpha.
    """
    r = OverlayRenderer(height=64, width=64)
    uv = np.array([[10.0, 32.0], [50.0, 32.0]])
    visible = np.array([True, True])
    layer = OverlayLayer(
        name="test", polylines=[(uv, visible)],
        color=(0, 255, 0, 255), width=1,
    )
    out = r.render([layer])
    # The line passes through y=32; some x in [10, 50] should have green.
    row = out[32, 10:51]
    assert (row[:, 1] > 0).sum() > 30, "expected green pixels along the segment"
    # Outside the segment row → still transparent
    assert out[0, 0, 3] == 0


def test_render_invisible_endpoint_skipped():
    """Polyline with visible=[True, False, True] → only segment 0→1 attempted;
    1→2 segment skipped because endpoint 1 is invisible.
    """
    r = OverlayRenderer(height=32, width=64)
    uv = np.array([[5.0, 16.0], [30.0, 16.0], [55.0, 16.0]])
    visible = np.array([True, False, True])
    layer = OverlayLayer(
        name="test", polylines=[(uv, visible)],
        color=(255, 0, 0, 255), width=1,
    )
    out = r.render([layer])
    # No segments should be drawn at all (endpoint pairs are (T,F), (F,T)).
    assert out[..., 3].max() == 0


def test_alpha_blend_identity_on_transparent_overlay():
    """Overlay with alpha=0 everywhere → blended == backdrop (no-op fast path)."""
    backdrop = np.full((16, 16, 3), 100, dtype=np.uint8)
    overlay = np.zeros((16, 16, 4), dtype=np.uint8)
    blended = alpha_blend(backdrop, overlay)
    np.testing.assert_array_equal(blended, backdrop)


def test_alpha_blend_opaque_overlay_replaces_backdrop():
    """Overlay with alpha=255 → blended == overlay RGB (full replacement)."""
    backdrop = np.full((16, 16, 3), 100, dtype=np.uint8)
    overlay = np.zeros((16, 16, 4), dtype=np.uint8)
    overlay[..., 0] = 200  # red
    overlay[..., 3] = 255  # opaque
    blended = alpha_blend(backdrop, overlay)
    expected = np.zeros((16, 16, 3), dtype=np.uint8)
    expected[..., 0] = 200
    np.testing.assert_array_equal(blended, expected)


def test_alpha_blend_half_alpha_averages():
    """Overlay alpha=128 (≈50%) → blended ≈ 0.5 * overlay + 0.5 * backdrop."""
    backdrop = np.full((4, 4, 3), 200, dtype=np.uint8)
    overlay = np.zeros((4, 4, 4), dtype=np.uint8)
    overlay[..., 1] = 100  # green RGB
    overlay[..., 3] = 128  # ~50% alpha
    blended = alpha_blend(backdrop, overlay)
    # ch0 (R): 0 * 0.502 + 200 * 0.498 = 99.6 ≈ 99 or 100
    # ch1 (G): 100 * 0.502 + 200 * 0.498 = 149.8 ≈ 149 or 150
    # ch2 (B): 0 * 0.502 + 200 * 0.498 = 99.6
    assert 95 <= blended[0, 0, 0] <= 105
    assert 145 <= blended[0, 0, 1] <= 155
    assert 95 <= blended[0, 0, 2] <= 105


def test_alpha_blend_shape_mismatch_raises():
    backdrop = np.zeros((10, 10, 3), dtype=np.uint8)
    overlay = np.zeros((20, 10, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape mismatch"):
        alpha_blend(backdrop, overlay)


def test_alpha_blend_dtype_mismatch_raises():
    backdrop = np.zeros((10, 10, 3), dtype=np.float32)
    overlay = np.zeros((10, 10, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="dtype must be uint8"):
        alpha_blend(backdrop, overlay)


def test_multilayer_ordering_top_layer_wins():
    """Last layer registered overrides earlier layers at overlapping pixels."""
    r = OverlayRenderer(height=32, width=32)
    uv = np.array([[5.0, 16.0], [25.0, 16.0]])
    visible = np.array([True, True])
    bottom = OverlayLayer(name="bot",  polylines=[(uv, visible)],
                          color=(255, 0, 0, 255), width=2)
    top    = OverlayLayer(name="top",  polylines=[(uv, visible)],
                          color=(0, 0, 255, 255), width=2)
    out = r.render([bottom, top])
    # At pixel (15, 16) both layers drew; top (blue) should win.
    px = out[16, 15]
    assert px[2] > 200 and px[0] < 50, f"top blue should dominate; got {px.tolist()}"


# ===================================================== BUG-1b: text labels
def test_render_text_draws_pixels_near_anchor():
    """OverlayLayer.texts → readable label pixels near the anchor point,
    sharing the layer color (the wireframe's instance color)."""
    r = OverlayRenderer(height=128, width=256)
    layer = OverlayLayer(
        name="labels", color=(0, 255, 0, 255),
        texts=[(60.0, 80.0, "t7 | bus")],
    )
    out = r.render([layer])
    # Some non-transparent pixels must appear in a window above/right of the
    # anchor (text is drawn adjacent to the anchor, not centered on it).
    win = out[80 - 40:80 + 8, 60 - 4:60 + 120]
    assert (win[..., 3] > 0).sum() > 20, "expected text pixels near anchor"


def test_render_text_out_of_bounds_anchor_no_crash():
    """Anchors far outside the canvas must not crash PIL (clipping is fine)."""
    r = OverlayRenderer(height=64, width=64)
    layer = OverlayLayer(
        name="labels", color=(255, 0, 0, 255),
        texts=[(-500.0, -500.0, "offscreen"), (10_000.0, 10_000.0, "far")],
    )
    out = r.render([layer])
    assert out.shape == (64, 64, 4)


def test_render_text_empty_list_is_noop():
    """texts=[] (default) keeps legacy polyline-only behavior byte-identical."""
    r = OverlayRenderer(height=32, width=32)
    layer = OverlayLayer(name="empty", color=(0, 255, 0, 255))
    out = r.render([layer])
    assert out[..., 3].max() == 0
