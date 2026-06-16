# SPDX-License-Identifier: Apache-2.0
"""Mac-runnable unit tests for the pure helpers of
``threedgrut_playground.utils.nre_usdz_loader`` (NRE usdz → 3dgrut2 .pt).

Covers the translation layer that doesn't need cuda/hydra/a real usdz:
tolerant unpickler mechanism, key rename, Fourier-albedo collapse/eval (and
agreement with the canonical P1.3b basis), per-layer tensor pull, scene extent.
The GPU build (``build_native_ckpt`` / ``convert_usdz_to_pt``) is exercised on
inceptio against the real E0.3 usdz.
"""
from __future__ import annotations

import io
import pickle

import pytest
import torch

from threedgrut_playground.utils.nre_usdz_loader import (
    _CapturingStub,
    _TolerantUnpickler,
    eval_fourier_albedo,
    estimate_scene_extent,
    fourier_cos_basis,
    nre_layer_tensors,
    tolerant_torch_load,
)


# --------------------------------------------------------------------------- #
# Tolerant unpickler
# --------------------------------------------------------------------------- #
def test_tolerant_load_roundtrips_plain_ckpt():
    blob = {"state_dict": {"a": torch.arange(6).float().reshape(2, 3)},
            "global_step": 42}
    buf = io.BytesIO()
    torch.save(blob, buf)
    out = tolerant_torch_load(buf.getvalue())
    assert out["global_step"] == 42
    assert torch.equal(out["state_dict"]["a"], blob["state_dict"]["a"])


def test_tolerant_unpickler_stubs_unknown_class():
    up = _TolerantUnpickler(io.BytesIO(b""))
    # A module that certainly cannot be imported → must fall back to the stub,
    # never raise (this is exactly the ``No module named 'nre'`` case).
    assert up.find_class("definitely_not_a_real_module_xyz", "Whatever") is _CapturingStub
    # A real class still resolves normally.
    assert up.find_class("builtins", "dict") is dict


def test_capturing_stub_swallows_state_without_losing_dict_items():
    s = _CapturingStub()
    s["k"] = 7
    s.append(1)
    s.__setstate__({"x": 1})
    assert s["k"] == 7


# --------------------------------------------------------------------------- #
# Fourier basis + albedo collapse
# --------------------------------------------------------------------------- #
def test_fourier_cos_basis_matches_canonical_p13b():
    """Local basis must agree with the canonical P1.3b impl (plan invariant)."""
    from threedgrut.model.track_albedo_fourier import (
        fourier_cos_basis as canonical,
    )
    for k in (1, 3, 5, 20):
        for n_frames in (1, 10, 480):
            for t in (0, 3, n_frames - 1):
                got = fourier_cos_basis(t, n_frames, k)
                exp = canonical(t, n_frames, k).cpu()
                assert torch.allclose(got, exp, atol=1e-6), (k, n_frames, t)


def test_fourier_cos_basis_frame0_is_all_ones():
    # cos(i·π·0) = 1 for every harmonic.
    assert torch.allclose(fourier_cos_basis(0, 480, 5), torch.ones(5))


def test_eval_albedo_passthrough_2d():
    a = torch.rand(10, 3)
    assert torch.equal(eval_fourier_albedo(a), a)


def test_eval_albedo_dc_takes_coeff0():
    a = torch.rand(7, 5, 3)
    out = eval_fourier_albedo(a, mode="dc")
    assert out.shape == (7, 3)
    assert torch.equal(out, a[:, 0, :])


def test_eval_albedo_k1_squeezes():
    a = torch.rand(4, 1, 3)
    out = eval_fourier_albedo(a, mode="eval", frame_id=3, n_frames=10)
    assert out.shape == (4, 3)
    assert torch.equal(out, a[:, 0, :])


def test_eval_albedo_eval_frame0_sums_coeffs():
    # frame 0 → basis all-ones → DC(t=0) = Σ_i coeff_i.
    a = torch.rand(6, 5, 3)
    out = eval_fourier_albedo(a, mode="eval", frame_id=0, n_frames=480)
    assert torch.allclose(out, a.sum(dim=1), atol=1e-5)


def test_eval_albedo_rejects_bad_shape():
    with pytest.raises(ValueError):
        eval_fourier_albedo(torch.rand(3, 5, 4))


# --------------------------------------------------------------------------- #
# Per-layer tensor pull + rename
# --------------------------------------------------------------------------- #
def _fake_nre_state(layer: str, n: int, fourier_k: int, with_cuboid: bool):
    p = f"model.gaussians_nodes.{layer}."
    sd = {
        p + "positions": torch.rand(n, 3),
        p + "rotations": torch.rand(n, 4),
        p + "scales": torch.rand(n, 3),
        p + "densities": torch.rand(n, 1),
        p + "features_albedo": torch.rand(n, fourier_k, 3),
        p + "features_specular": torch.rand(n, 45),
        p + "n_active_features": torch.tensor(3, dtype=torch.int64),
        p + "camera_extra_signal": torch.rand(n, 20),  # semantic logits, dropped (not RGB)
    }
    if with_cuboid:
        sd[p + "gaussian_cuboid_ids"] = torch.randint(0, 5, (n,))
    return sd


def test_nre_layer_tensors_renames_and_collapses():
    sd = _fake_nre_state("background", 8, fourier_k=5, with_cuboid=False)
    out = nre_layer_tensors(sd, "background")
    assert set(out) == {
        "positions", "rotation", "scale", "density",
        "features_albedo", "features_specular", "n_active_features",
    }
    assert out["rotation"].shape == (8, 4)
    assert out["scale"].shape == (8, 3)
    assert out["density"].shape == (8, 1)
    assert out["features_albedo"].shape == (8, 3)        # collapsed
    assert out["features_specular"].shape == (8, 45)
    assert out["n_active_features"] == 3
    assert "camera_extra_signal" not in out             # dropped
    assert "extra_signal" not in out


def test_nre_layer_tensors_rides_cuboid_ids():
    sd = _fake_nre_state("dynamic_rigids", 12, fourier_k=20, with_cuboid=True)
    out = nre_layer_tensors(sd, "dynamic_rigids")
    assert out["cuboid_ids"].shape == (12,)
    assert out["cuboid_ids"].dtype == torch.int64
    assert out["features_albedo"].shape == (12, 3)


def test_nre_layer_tensors_missing_layer_returns_empty():
    sd = _fake_nre_state("background", 4, 5, False)
    assert nre_layer_tensors(sd, "road") == {}


# --------------------------------------------------------------------------- #
# scene extent
# --------------------------------------------------------------------------- #
def test_scene_extent_robust_to_far_outlier():
    pts = torch.randn(1000, 3)
    pts_out = torch.cat([pts, torch.tensor([[1e6, 1e6, 1e6]])], dim=0)
    # 95th percentile radius barely moves despite the 1e6 outlier.
    assert estimate_scene_extent(pts_out) < 10.0
