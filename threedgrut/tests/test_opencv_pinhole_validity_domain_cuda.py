# SPDX-License-Identifier: Apache-2.0
"""RED tests: source-structure checks for PIN-CAM-1c CUDA validity domain.

Tests that the C++ struct, bindings, tracer, and dataset config all wire
through the new hasValidityDomain/maxValidR2 fields.  These are structural /
static-analysis tests that can run on Mac (no CUDA compile).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


# =====================================================================
# CUDA struct: cameraModels.h
# =====================================================================


class TestCameraModelsStruct:
    """OpenCVPinholeProjectionParameters must carry hasValidityDomain + maxValidR2."""

    def test_hasValidityDomain_field_defined(self):
        src = (ROOT / "threedgut_tracer/include/3dgut/sensors/cameraModels.h").read_text()
        assert "hasValidityDomain" in src, (
            "OpenCVPinholeProjectionParameters missing hasValidityDomain field"
        )

    def test_maxValidR2_field_defined(self):
        src = (ROOT / "threedgut_tracer/include/3dgut/sensors/cameraModels.h").read_text()
        assert "maxValidR2" in src, (
            "OpenCVPinholeProjectionParameters missing maxValidR2 field"
        )

    def test_no_min_derivative_field(self):
        """Per parent decision: do NOT pass min derivative as certificate field."""
        src = (ROOT / "threedgut_tracer/include/3dgut/sensors/cameraModels.h").read_text()
        # These are the old design fields that should NOT be added
        for bad_field in ("minRadialDerivative", "denominatorEpsilon", "radialDerivativeEpsilon"):
            assert bad_field not in src, (
                f"Struct should NOT contain {bad_field} — only hasValidityDomain+maxValidR2"
            )


# =====================================================================
# CUDA projection: cameraProjections.cuh
# =====================================================================


class TestCameraProjectionsLogic:
    """projectPoint(OpenCVPinhole...) must have certified and legacy branches."""

    def test_has_both_validity_branches(self):
        src = (ROOT / "threedgut_tracer/include/3dgut/kernels/cuda/sensors/cameraProjections.cuh").read_text()
        # Legacy branch: 0.8 < icD < 1.2 must be preserved
        assert "kMinRadialDist" in src, "Legacy 0.8 constant missing"
        assert "kMaxRadialDist" in src, "Legacy 1.2 constant missing"
        # Certified branch: hasValidityDomain check
        assert "hasValidityDomain" in src, "Certified branch missing"
        assert "maxValidR2" in src, "maxValidR2 check missing"
        # Derivative checks
        assert "radialDerivative" in src or "dq/dr" in src or "dq_dr" in src, (
            "Radial derivative check missing"
        )

    def test_legacy_defaults_maintained(self):
        """Legacy (no certificate) must still use 0.8<icD<1.2 unchanged."""
        src = (ROOT / "threedgut_tracer/include/3dgut/kernels/cuda/sensors/cameraProjections.cuh").read_text()
        # The default/legacy path must contain both constants
        assert "0.8f" in src or "kMinRadialDist = 0.8f" in src
        assert "1.2f" in src or "kMaxRadialDist = 1.2f" in src

    def test_denominator_check_present(self):
        """Certified branch must check denominator > 1e-6."""
        src = (ROOT / "threedgut_tracer/include/3dgut/kernels/cuda/sensors/cameraProjections.cuh").read_text()
        assert "1e-6" in src or "1e-6f" in src, "Denominator epsilon check missing"


# =====================================================================
# Bindings: bindings.cpp
# =====================================================================


class TestBindingsParams:
    """fromOpenCVPinholeCameraModelParameters must accept new fields."""

    def test_has_validity_domain_arg(self):
        src = (ROOT / "threedgut_tracer/bindings.cpp").read_text()
        assert "has_validity_domain" in src, "has_validity_domain argument missing"

    def test_max_valid_r2_arg(self):
        src = (ROOT / "threedgut_tracer/bindings.cpp").read_text()
        assert "max_valid_r2" in src, "max_valid_r2 argument missing"


# =====================================================================
# Tracer: tracer.py
# =====================================================================


class TestTracerWiring:
    """__create_camera_parameters must pass validity domain from Batch."""

    def test_intrinsics_dict_gets_validity_keys(self):
        """When Batch has max_valid_r2, it's passed to fromOpenCVPinholeCameraModelParameters."""
        src = (ROOT / "threedgut_tracer/tracer.py").read_text()
        assert "max_valid_r2" in src, (
            "tracer.py must pass max_valid_r2 to the bindings function"
        )
        assert "has_validity_domain" in src, (
            "tracer.py must pass has_validity_domain to the bindings function"
        )

    def test_tracer_reads_validity_from_opencv_intrinsics_dict(self):
        src = (ROOT / "threedgut_tracer/tracer.py").read_text()
        assert 'K.get("max_valid_r2"' in src


# =====================================================================
# Dataset: config + compute
# =====================================================================


class TestDatasetConfig:
    """Config key opencv_pinhole_validity_margin passes through factories."""

    def test_yaml_has_validity_margin(self):
        """ncore.yaml declares default opencv_pinhole_validity_margin: 0.1."""
        yaml_src = (ROOT / "configs/dataset/ncore.yaml").read_text()
        assert "opencv_pinhole_validity_margin" in yaml_src, (
            "ncore.yaml missing opencv_pinhole_validity_margin"
        )
        assert "0.1" in yaml_src or "0.10" in yaml_src, (
            "Default value should be ~0.1"
        )

    def test_train_val_test_factories_pass_identical_key(self):
        """All three NCoreDataset factories pass opencv_pinhole_validity_margin."""
        src = (ROOT / "threedgrut/datasets/__init__.py").read_text()
        tree = ast.parse(src)
        values = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "id", None) != "NCoreDataset":
                continue
            for kw in node.keywords:
                if kw.arg == "opencv_pinhole_validity_margin":
                    values.append(ast.unparse(kw.value))
        assert len(values) == 3, (
            f"Expected 3 factories (train/val/test) passing "
            f"opencv_pinhole_validity_margin, got {len(values)}"
        )
        assert len(set(values)) == 1, (
            f"Not all factories pass the same value: {values}"
        )

    def test_dataset_init_has_field(self):
        """NCoreDataset.__init__ signature includes opencv_pinhole_validity_margin."""
        src = (ROOT / "threedgrut/datasets/datasetNcore.py").read_text()
        assert "opencv_pinhole_validity_margin: float" in src, (
            "NCoreDataset missing opencv_pinhole_validity_margin parameter"
        )

    def test_utils_compute_max_valid_r2_exists(self):
        """datasets/utils.py exports compute_max_valid_r2 for pure-function use."""
        src = (ROOT / "threedgrut/datasets/utils.py").read_text()
        assert "def compute_max_valid_r2" in src, (
            "utils.py missing compute_max_valid_r2 function"
        )

    def test_real_b6a9_margin_domain_uses_distorted_inverse(self):
        """The certified radius must come from rational inverse, not ideal corners."""
        import numpy as np
        from threedgrut.datasets.utils import compute_max_valid_r2

        max_r2 = compute_max_valid_r2(
            principal_point=np.array([960.8599853515625, 540.1849975585938]),
            focal_length=np.array([952.8250122070312, 952.9000244140625]),
            radial_coeffs=np.array([
                3.7687599658966064, 1.61149001121521, 0.0664215013384819,
                4.13346004486084, 2.880429983139038, 0.36570900678634644,
            ]),
            tangential_coeffs=np.array([4.691869980888441e-05, 8.77050024428172e-06]),
            thin_prism_coeffs=np.zeros(4),
            image_size=(1920, 1080),
            margin=0.1,
            boundary_samples_per_edge=20000,
        )
        assert max_r2 == pytest.approx(23.75404, rel=2e-4)

    def test_zero_distortion_domain_matches_analytic_corner(self):
        import numpy as np
        from threedgrut.datasets.utils import compute_max_valid_r2

        result = compute_max_valid_r2(
            principal_point=np.array([50.0, 40.0]),
            focal_length=np.array([100.0, 80.0]),
            radial_coeffs=np.zeros(6),
            tangential_coeffs=np.zeros(2),
            thin_prism_coeffs=np.zeros(4),
            image_size=(100, 80),
            margin=0.0,
            boundary_samples_per_edge=64,
        )
        expected = max(
            ((x - 50.0) / 100.0) ** 2 + ((y - 40.0) / 80.0) ** 2
            for x in (0.0, 99.0) for y in (0.0, 79.0)
        )
        assert result == pytest.approx(expected)

    def test_certificate_rejects_denominator_pole_inside_domain(self):
        import numpy as np
        from threedgrut.datasets.utils import _validate_opencv_radial_domain

        with pytest.raises(ValueError, match="pole|denominator"):
            _validate_opencv_radial_domain(
                np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0]),
                max_r2=2.0,
            )

    def test_certificate_rejects_radial_fold_inside_domain(self):
        import numpy as np
        from threedgrut.datasets.utils import _validate_opencv_radial_domain

        with pytest.raises(ValueError, match="monotonic|fold"):
            _validate_opencv_radial_domain(
                np.array([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
                max_r2=2.0,
            )


# =====================================================================
# Dataset: _get_camera_model_parameters_for_resolution
# =====================================================================


class TestCameraModelParamsDict:
    """The params_dict for OpenCVPinhole must include max_valid_r2."""

    def test_params_dict_includes_max_valid_r2(self):
        src = (ROOT / "threedgrut/datasets/datasetNcore.py").read_text()
        # Search for the OpenCVPinhole branch in _get_camera_model_parameters_for_resolution
        # and verify it adds max_valid_r2
        assert "max_valid_r2" in src, (
            "datasetNcore.py params_dict should include max_valid_r2"
        )


class TestMaxValidR2Certificate:
    """The certificate is over original pixels plus the configured UT margin."""

    def test_expanded_pixel_domain_uses_its_farthest_corner(self):
        from threedgrut.datasets.utils import compute_max_valid_r2

        # W=100/H=50, cx=cy=0, unit focal: the continuous domain is [0, W)
        # × [0, H), so its 10%-expanded farthest corner is (110, 55).
        assert compute_max_valid_r2(
            principal_point=np.array([0.0, 0.0]),
            focal_length=np.array([1.0, 1.0]),
            radial_coeffs=np.zeros(6),
            tangential_coeffs=np.zeros(2),
            thin_prism_coeffs=np.zeros(4),
            image_size=(100, 50),
            margin=0.1,
        ) == pytest.approx(110.0**2 + 55.0**2, rel=1e-6)

    @pytest.mark.parametrize("margin", [-0.1, np.nan])
    def test_certificate_rejects_invalid_margin(self, margin):
        from threedgrut.datasets.utils import compute_max_valid_r2

        with pytest.raises(ValueError, match="margin"):
            compute_max_valid_r2(
                principal_point=np.array([0.0, 0.0]),
                focal_length=np.array([1.0, 1.0]),
                radial_coeffs=np.zeros(6),
                tangential_coeffs=np.zeros(2),
                thin_prism_coeffs=np.zeros(4),
                image_size=(100, 50),
                margin=margin,
            )


def test_legacy_forward_mask_and_calibrated_domain_are_mutually_exclusive():
    """The SDK mask would silently crop the periphery restored by the certificate."""
    from threedgrut.datasets.utils import validate_opencv_pinhole_domain_options

    with pytest.raises(ValueError, match="cannot be enabled together"):
        validate_opencv_pinhole_domain_options(True, True)

    validate_opencv_pinhole_domain_options(False, True)
    validate_opencv_pinhole_domain_options(True, False)
    validate_opencv_pinhole_domain_options(False, False)
