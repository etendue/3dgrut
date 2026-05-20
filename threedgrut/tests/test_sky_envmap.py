# SPDX-License-Identifier: Apache-2.0
"""Stage 5 unit tests: SkyEnvmapBase / SkyEnvmapMLP / SkyEnvmapCubemap.

The cubemap forward path needs nvdiffrast.torch; we conditionally skip it on
Mac dev boxes where it is unavailable. Parameter-shape and import-error
behaviour are still verified on CPU.
"""
from __future__ import annotations

import importlib

import pytest
import torch


# ---------------------------------------------------------------------------
# Base + module import
# ---------------------------------------------------------------------------
def test_correction_package_exports():
    from threedgrut.correction import (
        SkyEnvmapBase,
        SkyEnvmapMLP,
        SkyEnvmapCubemap,
        ExposureModel,
    )
    assert issubclass(SkyEnvmapMLP, SkyEnvmapBase)
    assert issubclass(SkyEnvmapCubemap, SkyEnvmapBase)
    # ExposureModel is unrelated to SkyEnvmapBase but must be exported.
    assert ExposureModel is not None


def test_base_forward_is_abstract():
    from threedgrut.correction import SkyEnvmapBase

    base = SkyEnvmapBase()
    with pytest.raises(NotImplementedError):
        base.forward(torch.zeros(4, 3))


# ---------------------------------------------------------------------------
# SkyEnvmapMLP
# ---------------------------------------------------------------------------
def test_mlp_forward_shape_flat():
    from threedgrut.correction import SkyEnvmapMLP

    m = SkyEnvmapMLP(hidden_dim=8)
    v = torch.randn(7, 3)
    out = m(v)
    assert out.shape == (7, 3)


def test_mlp_forward_shape_image():
    from threedgrut.correction import SkyEnvmapMLP

    m = SkyEnvmapMLP(hidden_dim=8)
    v = torch.randn(2, 5, 4, 3)
    out = m(v)
    assert out.shape == (2, 5, 4, 3)


def test_mlp_output_in_unit_range():
    from threedgrut.correction import SkyEnvmapMLP

    m = SkyEnvmapMLP(hidden_dim=8)
    v = torch.randn(64, 3) * 10.0  # exaggerate to push sigmoid into saturation
    out = m(v)
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_mlp_grad_flows_to_params():
    from threedgrut.correction import SkyEnvmapMLP

    m = SkyEnvmapMLP(hidden_dim=8)
    v = torch.randn(16, 3, requires_grad=False)
    loss = m(v).sum()
    loss.backward()
    grad_present = [
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in m.parameters()
    ]
    assert all(grad_present), "all SkyEnvmapMLP params should receive gradient"


# ---------------------------------------------------------------------------
# SkyEnvmapCubemap (construction always works; forward requires nvdiffrast)
# ---------------------------------------------------------------------------
def test_cubemap_params_shape_default():
    from threedgrut.correction import SkyEnvmapCubemap

    m = SkyEnvmapCubemap()
    assert m.base.shape == (6, 128, 128, 3)
    assert m.resolution == 128


def test_cubemap_params_shape_custom_resolution():
    from threedgrut.correction import SkyEnvmapCubemap

    m = SkyEnvmapCubemap(resolution=64)
    assert m.base.shape == (6, 64, 64, 3)


def test_cubemap_to_opengl_is_orthonormal():
    """``to_opengl`` is a coordinate frame change; det ±1 and row-orthonormal."""
    from threedgrut.correction import SkyEnvmapCubemap

    m = SkyEnvmapCubemap(resolution=16)
    R = m.to_opengl
    eye = R @ R.t()
    assert torch.allclose(eye, torch.eye(3), atol=1e-6)
    assert abs(float(torch.det(R).abs()) - 1.0) < 1e-6


def test_cubemap_forward_when_nvdiffrast_available():
    """End-to-end forward only runs when nvdiffrast.torch imports cleanly.

    Skipped on Mac dev boxes / CPU CI without nvdiffrast.
    """
    pytest.importorskip("nvdiffrast.torch")
    if not torch.cuda.is_available():
        pytest.skip("nvdiffrast cubemap sampling requires CUDA")
    from threedgrut.correction import SkyEnvmapCubemap

    m = SkyEnvmapCubemap(resolution=16).cuda()
    v = torch.randn(32, 3, device="cuda")
    out = m(v)
    assert out.shape == (32, 3)


def test_cubemap_raises_clearly_without_nvdiffrast(monkeypatch):
    """When nvdiffrast is unavailable, forward must raise ImportError + hint."""
    from threedgrut.correction import sky_envmap as _sky

    monkeypatch.setattr(_sky, "dr", None)
    m = _sky.SkyEnvmapCubemap(resolution=16)
    v = torch.randn(4, 3)
    with pytest.raises(ImportError) as exc:
        m(v)
    msg = str(exc.value)
    assert "nvdiffrast" in msg
    assert "sky_backend: mlp" in msg or "backend" in msg
