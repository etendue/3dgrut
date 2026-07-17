# SPDX-License-Identifier: Apache-2.0
"""RED tests: UT valid-only mean/covariance for gutProjector.

The current gutProjector unscentedParticleProjection:
1. Increments numValidPoints for each successful projectPoint
2. BUT always includes ALL projectedSigmaPoints (including invalid fallback)
   in the weighted mean and covariance sums.

Fix: track validity[7], compute mean and covariance using ONLY valid points.

These are CPU reference tests — pure NumPy, no CUDA needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

# UT configuration matching the CUDA code:
#   Alpha=1, Beta=2, Kappa=0, D=3
#   -> Lambda = 1^2 * (3 + 0) - 3 = 0
#   -> weight0_center = Lambda / (D + Lambda) = 0
#   -> weightI = 1/(2 * (D + Lambda)) = 1/6
#   -> total (with Beta): weight0 = Lambda/(D+Lambda) + (1-Alpha^2+Beta) = 0+2 = 2
# This means the mean is computed from sigma points only (center contributes 0 to mean),
# and the center contributes 2x to covariance.

D = 3
ALPHA = 1.0
BETA = 2.0
KAPPA = 0.0
LAMBDA = ALPHA * ALPHA * (D + KAPPA) - D  # = 0
WEIGHT0_CENTER_MEAN = LAMBDA / (D + LAMBDA)  # = 0
WEIGHT0_COV = LAMBDA / (D + LAMBDA) + (1.0 - ALPHA * ALPHA + BETA)  # = 2.0
WEIGHT_I = 1.0 / (2.0 * (D + LAMBDA))  # = 1/6


def legacy_ut_mean_cov(sigma_points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Reference: current broken implementation (all points included)."""
    N = len(sigma_points)  # 7
    # Center point
    center = sigma_points[0]
    mean = center * WEIGHT0_CENTER_MEAN
    for i in range(D):
        mean += WEIGHT_I * sigma_points[1 + i]
        mean += WEIGHT_I * sigma_points[1 + D + i]

    # Covariance (center with weight0_cov, rest with weightI)
    centered0 = sigma_points[0] - mean
    cov = WEIGHT0_COV * np.outer(centered0, centered0)
    for i in range(1, N):
        centered = sigma_points[i] - mean
        cov += WEIGHT_I * np.outer(centered, centered)

    return mean, cov, np.sum(valid) > 0


def fixed_ut_mean_cov(sigma_points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Reference: valid-only UT. Renormalize weights for valid points only."""
    N = len(sigma_points)
    num_valid = int(np.sum(valid))
    if num_valid == 0:
        return np.zeros(2), np.zeros((2, 2)), False

    # Separate valid and invalid indices
    valid_idx = np.where(valid)[0]
    invalid_idx = np.where(~valid)[0]

    # Build base weights (same structure as UT)
    weights = np.zeros(N)
    weights[0] = WEIGHT0_CENTER_MEAN
    for i in range(D):
        weights[1 + i] = WEIGHT_I
        weights[1 + D + i] = WEIGHT_I

    # Compute valid-only sum of weights
    valid_weight_sum = float(np.sum(weights[valid_idx]))

    # If valid-weight sum is near zero, fall back to equal valid-point weights.
    if abs(valid_weight_sum) < 1e-12:
        mean = np.mean(sigma_points[valid_idx], axis=0)
        cov = np.zeros((2, 2))
        for i in valid_idx:
            centered = sigma_points[i] - mean
            cov += np.outer(centered, centered) / len(valid_idx)
        return mean, cov, True

    # Renormalize: compute mean using only valid points with renormalized weights
    mean = np.zeros(2)
    for i in valid_idx:
        mean += (weights[i] / valid_weight_sum) * sigma_points[i]

    # Covariance: use renormalized weights for valid points
    # Use the same weight ratio for covariance weights
    cov = np.zeros((2, 2))
    for i in valid_idx:
        w = WEIGHT0_COV if i == 0 else WEIGHT_I
        w_renorm = w / valid_weight_sum
        centered = sigma_points[i] - mean
        cov += w_renorm * np.outer(centered, centered)

    return mean, cov, True


# ------------------------------------------------------------------ Fixtures

@pytest.fixture
def all_valid_sigma_points() -> tuple[np.ndarray, np.ndarray]:
    """7 sigma points, all valid — should match legacy exactly."""
    rng = np.random.RandomState(42)
    pts = rng.randn(7, 2).astype(np.float64)
    valid = np.ones(7, dtype=bool)
    return pts, valid


@pytest.fixture
def one_invalid_sigma_points() -> tuple[np.ndarray, np.ndarray]:
    """7 sigma points, one valid failed far-fallback."""
    rng = np.random.RandomState(42)
    pts = rng.randn(7, 2).astype(np.float64)
    valid = np.ones(7, dtype=bool)
    # Last point: invalid, far-fallback at radius ~= hypot(resolution)
    pts[-1] = np.array([2000.0, 1500.0])  # far fallback
    valid[-1] = False
    return pts, valid


@pytest.fixture
def all_invalid_sigma_points() -> tuple[np.ndarray, np.ndarray]:
    pts = np.zeros((7, 2), dtype=np.float64)
    valid = np.zeros(7, dtype=bool)
    return pts, valid


@pytest.fixture
def mixed_valid_sigma_points() -> tuple[np.ndarray, np.ndarray]:
    """Mix of valid and invalid — should produce finite non-zero covariance."""
    rng = np.random.RandomState(123)
    pts = rng.randn(7, 2).astype(np.float64)
    valid = np.ones(7, dtype=bool)
    valid[3] = False  # middle sigma point invalid
    pts[3] = np.array([3000.0, 2000.0])  # far fallback
    return pts, valid


# ===================================================================== TESTS


class TestAllValidMatchLegacy:
    """When all sigma points valid, fixed version matches legacy exactly."""

    def test_all_valid_mean_identical(self, all_valid_sigma_points):
        pts, valid = all_valid_sigma_points
        mean_legacy, _, _ = legacy_ut_mean_cov(pts, valid)
        mean_fixed, _, ok = fixed_ut_mean_cov(pts, valid)
        assert ok
        np.testing.assert_array_almost_equal(mean_fixed, mean_legacy, decimal=12)

    def test_all_valid_cov_identical(self, all_valid_sigma_points):
        pts, valid = all_valid_sigma_points
        _, cov_legacy, _ = legacy_ut_mean_cov(pts, valid)
        _, cov_fixed, ok = fixed_ut_mean_cov(pts, valid)
        assert ok
        np.testing.assert_array_almost_equal(cov_fixed, cov_legacy, decimal=12)


class TestOneInvalidNoLongerPollutes:
    """With one invalid far-fallback, fixed UT no longer shifts mean / inflates cov."""

    def test_one_invalid_mean_different(self, one_invalid_sigma_points):
        """Legacy mean is shifted by far fallback; fixed mean is not."""
        pts, valid = one_invalid_sigma_points
        mean_legacy, _, _ = legacy_ut_mean_cov(pts, valid)
        mean_fixed, _, ok = fixed_ut_mean_cov(pts, valid)
        assert ok
        # Legacy mean should be noticeably different from valid-only mean
        diff = float(np.linalg.norm(mean_legacy - mean_fixed))
        assert diff > 1.0, (
            f"Legacy mean should be shifted by invalid point: "
            f"legacy={mean_legacy}, fixed={mean_fixed}, diff={diff}"
        )

    def test_one_invalid_cov_no_longer_inflated(self, one_invalid_sigma_points):
        """Fixed covariance should be much smaller than legacy (which included far fallback)."""
        pts, valid = one_invalid_sigma_points
        _, cov_legacy, _ = legacy_ut_mean_cov(pts, valid)
        _, cov_fixed, ok = fixed_ut_mean_cov(pts, valid)
        assert ok
        legacy_trace = float(np.trace(cov_legacy))
        fixed_trace = float(np.trace(cov_fixed))
        assert fixed_trace < legacy_trace * 0.5, (
            f"Fixed cov trace {fixed_trace} should be much smaller than "
            f"legacy trace {legacy_trace}"
        )

    def test_one_invalid_matches_subset_computation(self, one_invalid_sigma_points):
        """Fixed UT with one invalid should match manually computed valid-subset UT."""
        pts, valid = one_invalid_sigma_points
        mean_fixed, cov_fixed, ok = fixed_ut_mean_cov(pts, valid)
        assert ok

        # Manual: exclude the invalid point
        valid_pts = pts[valid]
        # With only valid points, weights must be renormalized.
        # valid points are indices [0..5] (center index 0 is invalid, has WEIGHT0_CENTER_MEAN=0).
        # Non-center weights are all WEIGHT_I=1/6 -> 5 pts * 1/6 = 5/6 total.
        # Center weight 0 means it does not contribute to mean.
        valid_weights = np.array([WEIGHT0_CENTER_MEAN, WEIGHT_I, WEIGHT_I, WEIGHT_I, WEIGHT_I, WEIGHT_I])
        sum_w = float(np.sum(valid_weights))
        # With WEIGHT0_CENTER_MEAN=0 and 5 × WEIGHT_I=1/6, sum_w = 5/6
        assert abs(sum_w - 5.0 / 6.0) < 1e-12, f"Unexpected valid weight sum = {sum_w}"
        manual_mean = np.sum(valid_pts.T * (valid_weights / sum_w), axis=1)
        np.testing.assert_array_almost_equal(mean_fixed, manual_mean, decimal=12)

        manual_cov = np.zeros((2, 2))
        for i in range(len(valid_pts)):
            w = WEIGHT0_COV if i == 0 else WEIGHT_I
            centered = valid_pts[i] - manual_mean
            manual_cov += (w / sum_w) * np.outer(centered, centered)
        np.testing.assert_array_almost_equal(cov_fixed, manual_cov, decimal=12)


class TestAllInvalidEdgeCase:
    """All invalid -> returns false (no valid UT)."""

    def test_all_invalid_returns_false(self, all_invalid_sigma_points):
        pts, valid = all_invalid_sigma_points
        _, _, ok = fixed_ut_mean_cov(pts, valid)
        assert not ok, "All invalid should return false"

    def test_all_invalid_mean_cov_zero(self, all_invalid_sigma_points):
        pts, valid = all_invalid_sigma_points
        mean, cov, ok = fixed_ut_mean_cov(pts, valid)
        assert not ok
        np.testing.assert_array_equal(mean, np.zeros(2))
        np.testing.assert_array_equal(cov, np.zeros((2, 2)))


def test_center_only_valid_uses_finite_equal_weight_fallback():
    """Alpha=1 gives the center zero mean-weight; it must still remain usable."""
    pts = np.array([[12.0, 34.0]] + [[2000.0, 1500.0]] * 6)
    valid = np.array([True, False, False, False, False, False, False])
    mean, cov, ok = fixed_ut_mean_cov(pts, valid)
    assert ok
    np.testing.assert_allclose(mean, pts[0])
    np.testing.assert_allclose(cov, np.zeros((2, 2)))


class TestMixedValid:
    """Mixed valid/invalid: result is finite and uses only valid points."""

    def test_mixed_returns_true(self, mixed_valid_sigma_points):
        pts, valid = mixed_valid_sigma_points
        _, _, ok = fixed_ut_mean_cov(pts, valid)
        assert ok, "Mixed valid/invalid should return true"

    def test_mixed_mean_finite(self, mixed_valid_sigma_points):
        pts, valid = mixed_valid_sigma_points
        mean, _, ok = fixed_ut_mean_cov(pts, valid)
        assert ok
        assert np.all(np.isfinite(mean)), f"Mean should be finite: {mean}"

    def test_mixed_cov_finite(self, mixed_valid_sigma_points):
        pts, valid = mixed_valid_sigma_points
        _, cov, ok = fixed_ut_mean_cov(pts, valid)
        assert ok
        assert np.all(np.isfinite(cov)), f"Cov should be finite: {cov}"

    def test_mixed_does_not_include_invalid_point(self, mixed_valid_sigma_points):
        """Invalid point's far-fallback coords must NOT appear in mean or cov."""
        pts, valid = mixed_valid_sigma_points
        mean_fixed, cov_fixed, ok = fixed_ut_mean_cov(pts, valid)
        assert ok
        # The invalid point is at ~(3000, 2000); valid-only mean should be nowhere near that
        invalid_coord = pts[~valid][0]
        assert np.linalg.norm(mean_fixed - invalid_coord) > 1000, (
            f"Invalid point {invalid_coord} should not shift mean {mean_fixed}"
        )


def test_cuda_ut_tracks_validity_and_excludes_invalid_sigma_points():
    """The CUDA implementation must not accumulate invalid fallback positions."""
    src = (ROOT / "threedgut_tracer/include/3dgut/kernels/cuda/renderers/gutProjector.cuh").read_text()
    assert "validSigmaPoints" in src
    assert "validMeanWeight" in src
    assert "if (validSigmaPoints[i + 1])" in src
    assert "if (validSigmaPoints[0])" in src
    assert "equalWeight" in src


def test_valid_only_ut_has_explicit_ab_switch():
    config = (ROOT / "configs/render/3dgut.yaml").read_text()
    setup = (ROOT / "threedgut_tracer/setup_3dgut.py").read_text()
    cuda = (ROOT / "threedgut_tracer/include/3dgut/kernels/cuda/renderers/gutProjector.cuh").read_text()
    assert "ut_valid_only: true" in config
    assert "GAUSSIAN_UT_VALID_ONLY" in setup
    assert "!GAUSSIAN_UT_VALID_ONLY || validSigmaPoints" in cuda
