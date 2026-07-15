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

from threedgrut.datasets.utils import compute_forward_valid_pixel_mask, maybe_apply_forward_valid_mask, repair_nonfinite_rays


# Use torch for the mock return type to match the real NCore SDK.
import torch  # noqa: E402


# Known camera model type constants used by the guard
# (these mirror the real ncore.sensors class hierarchy)
PINHOLE_KIND = "OpenCVPinholeCameraModel"
FTHETA_KIND = "FThetaCameraModel"
FISHEYE_KIND = "OpenCVFisheyeCameraModel"



def _mock_model(valid_flag_1d: np.ndarray, kind: str = PINHOLE_KIND) -> Any:
    """Build a mock camera model returning the given 1-D valid_flag."""
    model = MagicMock()
    # Override __class__ for model-type guard checks
    model.__class__ = type(kind, (object,), {"__name__": kind})
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


# --------------------------------------------------------------------------- #
# maybe_apply_forward_valid_mask — dataset wiring                             #
# --------------------------------------------------------------------------- #

class TestMaybeApplyForwardValidMask:
    """Tests for the dataset wiring helper.

    ``maybe_apply_forward_valid_mask`` is called after ray generation/repair.
    It checks the ``enabled`` flag and camera model type, then ANDs the
    forward-valid mask into the ego mask if applicable.
    """

    # --- disabled flag (default) ---

    def test_disabled_does_not_modify_mask_and_never_calls_forward(self):
        """enabled=False → mask unchanged, camera_rays_to_pixels not called."""
        h, w = 4, 6
        rays = np.random.rand(h, w, 3).astype(np.float32)
        ego = np.ones((h, w), dtype=bool)
        orig = ego.copy()
        model = _mock_model(np.ones(h * w, dtype=bool))
        model.camera_rays_to_pixels = MagicMock(side_effect=RuntimeError("should not be called"))

        _ = maybe_apply_forward_valid_mask(model, rays, ego, "test_cam", enabled=False)
        assert np.array_equal(ego, orig)

    # --- enabled OpenCVPinhole ---

    def test_enabled_pinhole_and_with_ego_mask(self):
        """enabled=True, pinhole → forward mask ANDs with ego mask."""
        h, w = 4, 6
        n = h * w
        rays = np.random.rand(h, w, 3).astype(np.float32)

        # Ego mask: middle two rows True, outer rows False
        ego = np.zeros((h, w), dtype=bool)
        ego[1:3, :] = True
        n_ego = int(ego.sum())

        # Forward valid: right half False, left half True
        fwd_1d = np.ones(n, dtype=bool)
        fwd_1d.reshape(h, w)[:, w // 2:] = False
        model = _mock_model(fwd_1d)

        modified = maybe_apply_forward_valid_mask(model, rays, ego, "test_cam", enabled=True)
        assert modified, "forward-valid mask should be applied"

        # Expected: intersection of ego (rows 1-2) and forward (left half)
        expected = np.zeros((h, w), dtype=bool)
        expected[1:3, :w // 2] = True
        assert np.array_equal(ego, expected)
        assert int(ego.sum()) == n_ego // 2  # half of ego rows survived

    def test_enabled_pinhole_preserves_existing_invalid(self):
        """Existing False ego pixels stay False after AND with forward mask."""
        h, w = 4, 6
        n = h * w
        rays = np.random.rand(h, w, 3).astype(np.float32)

        # Ego mask: all True except a single False at (0,0)
        ego = np.ones((h, w), dtype=bool)
        ego[0, 0] = False

        # Forward valid: all True (no invalid)
        model = _mock_model(np.ones(n, dtype=bool))
        _ = maybe_apply_forward_valid_mask(model, rays, ego, "test_cam", enabled=True)
        assert not ego[0, 0], "pixel that was already ego-invalid must stay False"
        assert ego.sum() == h * w - 1  # all except (0,0)

    def test_enabled_pinhole_logs_coverage(self):
        """enabled=True logs kept/removed counts; verify function returns True."""
        h, w = 4, 6
        n = h * w
        rays = np.random.rand(h, w, 3).astype(np.float32)
        ego = np.ones((h, w), dtype=bool)
        half_invalid = np.ones(n, dtype=bool)
        half_invalid[n // 2:] = False
        model = _mock_model(half_invalid)

        modified = maybe_apply_forward_valid_mask(model, rays, ego, "test_cam", enabled=True)
        assert modified
        assert int(ego.sum()) == n // 2  # half survived

    # --- FTheta and OpenCVFisheye no-op ---

    def test_ftheta_no_op(self):
        """FTheta camera with enabled=True → no mask change."""
        h, w = 4, 6
        rays = np.random.rand(h, w, 3).astype(np.float32)
        ego = np.ones((h, w), dtype=bool)
        orig = ego.copy()
        # FTheta model: camera_rays_to_pixels should NOT be called
        model = _mock_model(np.ones(h * w, dtype=bool), kind=FTHETA_KIND)
        model.camera_rays_to_pixels = MagicMock(side_effect=RuntimeError("should not be called for FTheta"))

        modified = maybe_apply_forward_valid_mask(model, rays, ego, "test_cam", enabled=True)
        assert not modified, "FTheta must be no-op"
        assert np.array_equal(ego, orig)

    def test_fisheye_no_op(self):
        """OpenCVFisheye camera with enabled=True → no mask change."""
        h, w = 4, 6
        rays = np.random.rand(h, w, 3).astype(np.float32)
        ego = np.ones((h, w), dtype=bool)
        orig = ego.copy()
        model = _mock_model(np.ones(h * w, dtype=bool), kind=FISHEYE_KIND)
        model.camera_rays_to_pixels = MagicMock(side_effect=RuntimeError("should not be called for Fisheye"))

        modified = maybe_apply_forward_valid_mask(model, rays, ego, "test_cam", enabled=True)
        assert not modified, "OpenCVFisheye must be no-op"
        assert np.array_equal(ego, orig)

    # --- non-finite repaired pixels stay invalid ---

    def test_nonfinite_repaired_pixels_stay_invalid(self):
        """Actual NaN rays: repair marks ego mask False, then forward-valid AND keeps them False."""
        h, w = 4, 6
        n = h * w
        rays = np.random.rand(h, w, 3).astype(np.float32)

        # Inject a NaN ray at (0, 0) — simulates the real rational-distortion pole
        rays[0, 0, 0] = np.nan

        ego = np.ones((h, w), dtype=bool)
        n_repaired = repair_nonfinite_rays(rays, ego)
        assert n_repaired == 1, "repair must flag the NaN pixel"
        assert np.isfinite(rays).all(), "repair must fix the NaN"
        assert not ego[0, 0], "repaired pixel must be False in ego mask"

        # Forward-valid mask: all valid (no additional invalidation)
        fwd_1d = np.ones(n, dtype=bool)
        model = _mock_model(fwd_1d)

        modified = maybe_apply_forward_valid_mask(model, rays, ego, "test_cam", enabled=True)
        assert modified
        # Pixel that was already ego-invalid from repair must stay False
        assert not ego[0, 0], "repaired pixel must still be False after forward-valid AND"
        # All other pixels remain True (forward-valid didn't remove any)
        assert ego.sum() == n - 1  # only the repaired pixel is False
