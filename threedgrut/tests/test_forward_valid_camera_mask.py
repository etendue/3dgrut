# SPDX-License-Identifier: Apache-2.0
"""PIN-MASK-1 — forward-valid pixel mask from camera_rays_to_pixels().valid_flag.

The helper ``compute_forward_valid_pixel_mask`` projects camera-space rays back
through the camera model and returns the ``valid_flag`` as a bool mask.  Only
OpenCVPinholeCameraModel produces invalid pixels (rational polynomial admits
rays that came from valid integer pixels but project back outside the finite
trust domain).  FThetaCameraModel and OpenCVFisheyeCameraModel are no-ops
(all-valid).

Test strategy (Mac CPU): mock a camera model whose ``camera_rays_to_pixels``
returns ``PixelsReturn(pixels=..., valid_flag=...)`` — we cannot import the
real ncore.sensors on Mac.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from threedgrut.datasets.utils import compute_forward_valid_pixel_mask


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _mock_camera_model(*, valid_flag: np.ndarray) -> Any:
    """Return a MagicMock camera model whose camera_rays_to_pixels returns a
    ``PixelsReturn``-like object with the given 1-D bool ``valid_flag``.

    The real NCore ``CameraModel.PixelsReturn`` is a dataclass:
        PixelsReturn(pixels: Tensor, valid_flag: Tensor)  # both (N, ...)
    We flatten rays, call the mock, and reshape the flag.
    """
    model = MagicMock()
    mock_return = MagicMock()
    mock_return.valid_flag = torch.from_numpy(valid_flag) if hasattr(torch, 'from_numpy') else valid_flag  # noqa: F821
    model.camera_rays_to_pixels.return_value = mock_return
    return model


# Use torch for the mock return type to match the real NCore SDK.
import torch  # noqa: E402


def _mock_model(valid_flag_1d: np.ndarray) -> Any:
    """Build a mock camera model returning the given 1-D valid_flag."""
    model = MagicMock()
    model.camera_rays_to_pixels.return_value = type(
        "PixelsReturn", (), {"pixels": None, "valid_flag": torch.from_numpy(valid_flag_1d)}
    )()
    return model


# --------------------------------------------------------------------------- #
# RED tests — helper does not exist yet                                       #
# --------------------------------------------------------------------------- #

class TestComputeForwardValidPixelMask:

    # --- Shape preservation ---

    def test_returns_bool_mask_with_hw_shape(self):
        """2-D grid (4,6) of rays → (4,6) bool mask."""
        h, w = 4, 6
        rays = np.random.rand(h, w, 3).astype(np.float32)
        model = _mock_model(np.ones(h * w, dtype=bool))
        mask = compute_forward_valid_pixel_mask(model, rays)
        assert mask.shape == (h, w)
        assert mask.dtype == bool

    def test_preserves_hw_values_from_valid_flag(self):
        """Exact valid_flag values are preserved after reshape."""
        h, w = 3, 5
        valid = np.zeros(h * w, dtype=bool)
        valid[[1, 7, 14]] = True  # three valid pixels
        rays = np.random.rand(h, w, 3).astype(np.float32)
        model = _mock_model(valid)
        mask = compute_forward_valid_pixel_mask(model, rays)
        assert mask.shape == (h, w)
        assert mask.sum() == 3
        assert mask.ravel()[1] == True  # noqa: E712
        assert mask.ravel()[7] == True  # noqa: E712
        assert mask.ravel()[14] == True  # noqa: E712

    def test_all_invalid_returns_all_false(self):
        """All flags False → mask is all False."""
        h, w = 4, 4
        valid = np.zeros(h * w, dtype=bool)
        rays = np.random.rand(h, w, 3).astype(np.float32)
        model = _mock_model(valid)
        mask = compute_forward_valid_pixel_mask(model, rays)
        assert mask.shape == (h, w)
        assert not mask.any()

    def test_all_valid_returns_all_true(self):
        """All flags True → mask is all True."""
        h, w = 4, 4
        valid = np.ones(h * w, dtype=bool)
        rays = np.random.rand(h, w, 3).astype(np.float32)
        model = _mock_model(valid)
        mask = compute_forward_valid_pixel_mask(model, rays)
        assert mask.shape == (h, w)
        assert mask.all()

    # --- Flat N×3 rays (subsampled / val) ---

    def test_flat_n3_rays_produces_1d_mask(self):
        """Flat [N, 3] rays → 1-D mask of length N."""
        n = 20
        valid = np.zeros(n, dtype=bool)
        valid[[3, 7, 11]] = True
        rays = np.random.rand(n, 3).astype(np.float32)
        model = _mock_model(valid)
        mask = compute_forward_valid_pixel_mask(model, rays)
        assert mask.shape == (n,)
        assert mask.dtype == bool
        assert mask.sum() == 3

    # --- Invalid / edge cases ---

    def test_raises_on_element_count_mismatch(self):
        """valid_flag length != rays.shape[:-1] product → ValueError."""
        h, w = 4, 6
        rays = np.random.rand(h, w, 3).astype(np.float32)
        # Deliberately wrong length
        model = _mock_model(np.ones(h * w - 1, dtype=bool))
        with pytest.raises(ValueError, match="element|camera_rays_to_pixels|valid_flag|length"):
            compute_forward_valid_pixel_mask(model, rays)

    def test_raises_on_empty_rays(self):
        """Empty ray array → ValueError."""
        rays = np.empty((0, 3), dtype=np.float32)
        model = _mock_model(np.ones(0, dtype=bool))
        with pytest.raises(ValueError, match="empty|no rays|no pixels"):
            compute_forward_valid_pixel_mask(model, rays)
