# SPDX-License-Identifier: Apache-2.0
"""TDD tests for OpenCVPinhole inverse-ray convergence fix.

The SDK's ``__iterative_undistort`` has ``max_iterations=10`` hard-coded,
causing under-convergence at wide-FOV edges (>5 px forward residual on
inc_b6a9 front-wide 55°+ region).  The replacement pure-NumPy helper
``compute_opencv_pinhole_rays`` defaults to 30 iters (<0.02 px max
residual).

Test plan
---------
1.  RED — 10 iterations with b6a9-like corner → forward residual >5 px.
2.  30 iterations with same params → max residual <0.02 px.
3.  Zero-distortion camera → exact pinhole result (any iteration count).
4.  Standard centre-FOV camera → no regression from iteration increase.
5.  Convergence diagnostics return correct iteration count.
6.  Factory config passes through train/val/test identically (structural).
"""

import ast
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

# -------------------------------------------------------------------- fixtures

_w, _h = 1920, 1080
_cx, _cy = 960.8599853515625, 540.1849975585938
_fx, _fy = 952.8250122070312, 952.9000244140625

# Real inc_b6a9 camera_front_wide_120fov calibration.
_B6A9_RADIAL = np.array(
    [3.7687599658966064, 1.61149001121521, 0.0664215013384819,
     4.13346004486084, 2.880429983139038, 0.36570900678634644],
    dtype=np.float32,
)
_B6A9_TANGENTIAL = np.array(
    [4.691869980888441e-05, 8.77050024428172e-06], dtype=np.float32
)
_B6A9_THIN_PRISM = np.zeros(4, dtype=np.float32)

_ZERO_RADIAL = np.zeros(6, dtype=np.float32)
_ZERO_TANGENTIAL = np.zeros(2, dtype=np.float32)
_ZERO_THIN_PRISM = np.zeros(4, dtype=np.float32)

_PRINCIPAL = np.array([_cx, _cy], dtype=np.float32)
_FOCAL = np.array([_fx, _fy], dtype=np.float32)

# Far-corner pixel (worst-case convergence)
_CORNER_PX = np.array([[_w - 0.5, _h - 0.5]], dtype=np.float32)
# Centre pixel
_CENTRE_PX = np.array([[_cx + 0.5, _cy + 0.5]], dtype=np.float32)


def _forward_project(rays, fl, pp, rc, tc, tpc):
    """Forward-project ray directions back to distorted pixel coords."""
    from threedgrut.datasets.utils import _compute_distortion_np
    xy_norm = rays[:, :2] / (rays[:, 2:3] + 1e-30)
    icD, dx, dy, _ = _compute_distortion_np(xy_norm, rc, tc, tpc)
    uvND = xy_norm * icD[:, None] + np.column_stack([dx, dy])
    return uvND * fl + pp


# ===================================================================== TESTS


class TestConvergenceRED:
    """Real b6a9 tests — 10 iterations under-converge at the image corners."""

    def test_10_iterations_corner_residual_gt_5px(self):
        """Real b6a9 corner pixel: 10 iterations → forward residual >5 px."""
        from threedgrut.datasets.utils import compute_opencv_pinhole_rays

        rays_10 = compute_opencv_pinhole_rays(
            _CORNER_PX, _PRINCIPAL, _FOCAL, _B6A9_RADIAL,
            _B6A9_TANGENTIAL, _B6A9_THIN_PRISM, max_iterations=10,
        )
        proj_10 = _forward_project(
            rays_10, _FOCAL, _PRINCIPAL, _B6A9_RADIAL,
            _B6A9_TANGENTIAL, _B6A9_THIN_PRISM,
        )
        residual = float(np.linalg.norm(proj_10 - _CORNER_PX, axis=1)[0])
        assert residual > 5.0, (
            f"10-iteration forward residual {residual:.4f} px at corner — "
            f"expected >5 px (SDK under-convergence)"
        )

    def test_default_iterations_reach_high_precision(self):
        """Omitting max_iterations must use the high-precision default."""
        from threedgrut.datasets.utils import compute_opencv_pinhole_rays

        rays_default = compute_opencv_pinhole_rays(
            _CORNER_PX, _PRINCIPAL, _FOCAL, _B6A9_RADIAL,
            _B6A9_TANGENTIAL, _B6A9_THIN_PRISM,
        )
        proj_default = _forward_project(
            rays_default, _FOCAL, _PRINCIPAL, _B6A9_RADIAL,
            _B6A9_TANGENTIAL, _B6A9_THIN_PRISM,
        )
        residual = float(np.linalg.norm(proj_default - _CORNER_PX, axis=1)[0])
        assert residual < 0.02, f"default inverse residual {residual:.6f} px"

    def test_30_iterations_corner_residual_lt_0_02px(self):
        """Real b6a9 corner pixel: 30 iterations → forward residual <0.02 px."""
        from threedgrut.datasets.utils import compute_opencv_pinhole_rays

        rays_30 = compute_opencv_pinhole_rays(
            _CORNER_PX, _PRINCIPAL, _FOCAL, _B6A9_RADIAL,
            _B6A9_TANGENTIAL, _B6A9_THIN_PRISM, max_iterations=30,
        )
        proj_30 = _forward_project(
            rays_30, _FOCAL, _PRINCIPAL, _B6A9_RADIAL,
            _B6A9_TANGENTIAL, _B6A9_THIN_PRISM,
        )
        residual = float(np.linalg.norm(proj_30 - _CORNER_PX, axis=1)[0])
        assert residual < 0.02, (
            f"30-iteration forward residual {residual:.6f} px at corner — "
            f"expected <0.02 px (full convergence)"
        )

    def test_max_residual_across_four_corners(self):
        """All four corners: 30 iterations max residual <0.02 px,
        10 iterations corner residual >5 px."""
        from threedgrut.datasets.utils import compute_opencv_pinhole_rays

        corners = np.array([
            [_w - 0.5, _h - 0.5],  # bottom-right
            [_w - 0.5, 0.5],  # top-right
            [0.5, _h - 0.5],  # bottom-left
            [0.5, 0.5],  # top-left
        ], dtype=np.float32)

        rays_10 = compute_opencv_pinhole_rays(
            corners, _PRINCIPAL, _FOCAL, _B6A9_RADIAL,
            _B6A9_TANGENTIAL, _B6A9_THIN_PRISM, max_iterations=10,
        )
        proj_10 = _forward_project(
            rays_10, _FOCAL, _PRINCIPAL, _B6A9_RADIAL,
            _B6A9_TANGENTIAL, _B6A9_THIN_PRISM,
        )
        residuals_10 = np.linalg.norm(proj_10 - corners, axis=1)
        assert float(residuals_10.min()) > 5.0, (
            f"10-iteration min corner residual {residuals_10.min():.4f} px — "
            f"all corners should show under-convergence"
        )

        rays_30 = compute_opencv_pinhole_rays(
            corners, _PRINCIPAL, _FOCAL, _B6A9_RADIAL,
            _B6A9_TANGENTIAL, _B6A9_THIN_PRISM, max_iterations=30,
        )
        proj_30 = _forward_project(
            rays_30, _FOCAL, _PRINCIPAL, _B6A9_RADIAL,
            _B6A9_TANGENTIAL, _B6A9_THIN_PRISM,
        )
        residuals_30 = np.linalg.norm(proj_30 - corners, axis=1)
        assert float(residuals_30.max()) < 0.02, (
            f"30-iteration max corner residual {residuals_30.max():.6f} px — "
            f"expected <0.02 px"
        )


class TestZeroDistortion:
    """Zero-distortion camera: exact pinhole regardless of iteration count."""

    @pytest.fixture(params=[1, 10, 30])
    def n_iter(self, request):
        return request.param

    def test_center_ray(self, n_iter):
        """Centre pixel with zero distortion → ray ≈ +Z within numeric precision."""
        from threedgrut.datasets.utils import compute_opencv_pinhole_rays

        rays = compute_opencv_pinhole_rays(
            _CENTRE_PX, _PRINCIPAL, _FOCAL,
            _ZERO_RADIAL, _ZERO_TANGENTIAL, _ZERO_THIN_PRISM,
            max_iterations=n_iter,
        )
        # Centre pixel (+0.5 offset from principal point) → near-+Z
        # The exact expected direction is (dx/fx, dy/fy, 1) normalized
        dx = _CENTRE_PX[0, 0] - _PRINCIPAL[0]  # 0.5
        dy = _CENTRE_PX[0, 1] - _PRINCIPAL[1]  # 0.5
        expected_unnorm = np.array([dx / _FOCAL[0], dy / _FOCAL[1], 1.0], dtype=np.float64)
        expected = (expected_unnorm / np.linalg.norm(expected_unnorm)).astype(np.float32)

        assert_allclose(rays[0], expected, atol=1e-5, err_msg=f"n_iter={n_iter}")

    def test_forward_reprojection_exact(self, n_iter):
        """Zero distortion: forward reprojection of computed rays matches
        input pixel coords exactly."""
        from threedgrut.datasets.utils import compute_opencv_pinhole_rays

        # Sample across the image
        xs = np.arange(100, _w - 100, 200, dtype=np.float32) + 0.5
        ys = np.arange(100, _h - 100, 200, dtype=np.float32) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        pixels = np.column_stack([gx.ravel(), gy.ravel()])

        rays = compute_opencv_pinhole_rays(
            pixels, _PRINCIPAL, _FOCAL,
            _ZERO_RADIAL, _ZERO_TANGENTIAL, _ZERO_THIN_PRISM,
            max_iterations=n_iter,
        )
        reproj = _forward_project(
            rays, _FOCAL, _PRINCIPAL,
            _ZERO_RADIAL, _ZERO_TANGENTIAL, _ZERO_THIN_PRISM,
        )
        errors = np.linalg.norm(reproj - pixels, axis=1)
        assert float(errors.max()) < 1e-4, (
            f"Zero distortion max reprojection error {errors.max():.6f} px "
            f"at n_iter={n_iter}"
        )


class TestStandardCamera:
    """Standard centre-FOV camera: iteration increase doesn't regress."""

    def test_center_rays_consistent(self):
        """Moderate distortion at centre: 10 and 30 iterations agree well."""
        from threedgrut.datasets.utils import compute_opencv_pinhole_rays

        # Centre region pixels (small FOV, fast convergence)
        xs = np.arange(_cx - 100, _cx + 100, 10, dtype=np.float32) + 0.5
        ys = np.arange(_cy - 100, _cy + 100, 10, dtype=np.float32) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        pixels = np.column_stack([gx.ravel(), gy.ravel()])

        rays_10 = compute_opencv_pinhole_rays(
            pixels, _PRINCIPAL, _FOCAL, _B6A9_RADIAL,
            _B6A9_TANGENTIAL, _B6A9_THIN_PRISM, max_iterations=10,
        )
        rays_30 = compute_opencv_pinhole_rays(
            pixels, _PRINCIPAL, _FOCAL, _B6A9_RADIAL,
            _B6A9_TANGENTIAL, _B6A9_THIN_PRISM, max_iterations=30,
        )

        diff = np.linalg.norm(rays_10 - rays_30, axis=1)
        assert float(diff.max()) < 1e-5, (
            f"Centre-ray max diff between 10 and 30 iterations: "
            f"{diff.max():.2e}"
        )

    def test_output_shape_and_dtype(self):
        """Output shape (N, 3) and dtype float32 preserved."""
        from threedgrut.datasets.utils import compute_opencv_pinhole_rays

        pixels = np.array([[100.5, 100.5], [500.5, 400.5]], dtype=np.float32)
        rays = compute_opencv_pinhole_rays(
            pixels, _PRINCIPAL, _FOCAL, _ZERO_RADIAL,
            _ZERO_TANGENTIAL, _ZERO_THIN_PRISM,
        )
        assert rays.shape == (2, 3), f"Expected (2, 3), got {rays.shape}"
        assert rays.dtype == np.float32, f"Expected float32, got {rays.dtype}"
        assert np.all(np.isfinite(rays)), "Non-finite rays detected"

    def test_single_pixel(self):
        """Single-pixel (1, 2) input → (1, 3) output."""
        from threedgrut.datasets.utils import compute_opencv_pinhole_rays

        rays = compute_opencv_pinhole_rays(
            _CENTRE_PX, _PRINCIPAL, _FOCAL, _ZERO_RADIAL,
            _ZERO_TANGENTIAL, _ZERO_THIN_PRISM,
        )
        assert rays.shape == (1, 3)
        assert np.all(np.isfinite(rays))


class TestConvergenceDiagnostics:
    """_convergence_diagnostics flag returns iteration count and MSE."""

    def test_iteration_count_1(self):
        from threedgrut.datasets.utils import compute_opencv_pinhole_rays

        _, n, mse = compute_opencv_pinhole_rays(
            _CENTRE_PX, _PRINCIPAL, _FOCAL, _ZERO_RADIAL,
            _ZERO_TANGENTIAL, _ZERO_THIN_PRISM,
            max_iterations=1, _convergence_diagnostics=True,
        )
        assert n == 1, f"Expected 1 iteration, got {n}"
        assert isinstance(mse, float)

    def test_iteration_count_30(self):
        from threedgrut.datasets.utils import compute_opencv_pinhole_rays

        _, n, mse = compute_opencv_pinhole_rays(
            _CORNER_PX, _PRINCIPAL, _FOCAL, _B6A9_RADIAL,
            _B6A9_TANGENTIAL, _B6A9_THIN_PRISM,
            max_iterations=30, _convergence_diagnostics=True,
        )
        assert 1 <= n <= 30, f"Expected 1-30 iterations, got {n}"
        assert mse < 1e-6, f"MSE not converged: {mse:.2e}"

    def test_zero_distortion_early_exit(self):
        """Zero distortion should converge in 1-2 iterations."""
        from threedgrut.datasets.utils import compute_opencv_pinhole_rays

        _, n, mse = compute_opencv_pinhole_rays(
            _CENTRE_PX, _PRINCIPAL, _FOCAL, _ZERO_RADIAL,
            _ZERO_TANGENTIAL, _ZERO_THIN_PRISM,
            max_iterations=30, _convergence_diagnostics=True,
        )
        assert n <= 2, (
            f"Zero distortion should converge in ≤2 iterations, "
            f"got {n} (mse={mse:.2e})"
        )


class TestFactoryParity:
    """Config key ``opencv_pinhole_inverse_iterations`` passes through
    train/val/test factories identically (structural checks).

    These are lightweight structural tests — the full factory integration
    (NCoreDataset.__init__) is verified in the DatasetNcore integration
    points below.
    """

    def test_default_is_30(self):
        """Constructor and YAML both default to 30."""
        root = Path(__file__).resolve().parents[2]
        dataset_src = (root / "threedgrut/datasets/datasetNcore.py").read_text()
        yaml_src = (root / "configs/dataset/ncore.yaml").read_text()
        assert "opencv_pinhole_inverse_iterations: int = 30" in dataset_src
        assert "opencv_pinhole_inverse_iterations: 30" in yaml_src

    def test_train_val_test_factories_pass_identical_key(self):
        """All three NCoreDataset factories pass the same config key/default."""
        root = Path(__file__).resolve().parents[2]
        tree = ast.parse((root / "threedgrut/datasets/__init__.py").read_text())
        values = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "id", None) != "NCoreDataset":
                continue
            for kw in node.keywords:
                if kw.arg == "opencv_pinhole_inverse_iterations":
                    values.append(ast.unparse(kw.value))
        assert len(values) == 3
        assert len(set(values)) == 1
        assert values[0] == "config.dataset.get('opencv_pinhole_inverse_iterations', 30)"

    def test_value_10_reproduces_sdk(self):
        """Setting opencv_pinhole_inverse_iterations=10 reproduces old
        SDK behaviour (larger edge residuals)."""
        from threedgrut.datasets.utils import compute_opencv_pinhole_rays

        rays_10 = compute_opencv_pinhole_rays(
            _CORNER_PX, _PRINCIPAL, _FOCAL, _B6A9_RADIAL,
            _B6A9_TANGENTIAL, _B6A9_THIN_PRISM, max_iterations=10,
        )
        proj_10 = _forward_project(
            rays_10, _FOCAL, _PRINCIPAL, _B6A9_RADIAL,
            _B6A9_TANGENTIAL, _B6A9_THIN_PRISM,
        )
        residual = float(np.linalg.norm(proj_10 - _CORNER_PX, axis=1)[0])
        assert residual > 5.0, (
            f"At 10 iterations (SDK parity), corner residual {residual:.4f} px"
        )
