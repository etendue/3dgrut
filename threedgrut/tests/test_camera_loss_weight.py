# SPDX-License-Identifier: Apache-2.0
"""Task 7 / C1 — per-camera photometric loss weight tests.

Trainer3DGRUT._camera_loss_weight(camera_id) reads
``self.conf.loss.get("camera_loss_weights", {})``, returning the hit weight
or 1.0. get_losses multiplies loss_l1 and loss_ssim by that weight before
the weighted sum; other regularization terms are untouched.

We can't import the full ``threedgrut.trainer.Trainer3DGRUT`` stack on every
Mac dev box (fused_ssim is a CUDA extension). We install a minimal stub for
fused_ssim at collection time so the bound-method tests can still run on Mac.
When broader deps (addict, torchmetrics.image.lpip, ...) are unavailable, the
integration tests importorskip out — the unit tests only need omegaconf and
the class object.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

# fused_ssim: CUDA extension referenced at import by threedgrut/model/losses.py.
# Replace with a callable that returns a deterministic tensor so
# ``1.0 - ssim(pred, gt)`` in get_losses stays well-typed on Mac.
if "fused_ssim" not in sys.modules:
    _fs_mod = types.ModuleType("fused_ssim")

    def _fake_fused_ssim(img1, img2, padding="valid"):
        return torch.tensor(0.5, dtype=img1.dtype, device=img1.device)

    _fs_mod.fused_ssim = _fake_fused_ssim
    sys.modules["fused_ssim"] = _fs_mod

# NVTX profiling hooks: get_losses is decorated with @torch.cuda.nvtx.range("get_losses")
# and uses several ``with torch.cuda.nvtx.range("loss-*"):`` blocks. On CPU-only
# Mac torch builds, range_push/range_pop raise RuntimeError("NVTX functions not
# installed"). Rebind the module-level names to no-ops so both the decorator
# and the context-manager uses stay silent.
torch.cuda.nvtx.range_push = lambda *_a, **_kw: None
torch.cuda.nvtx.range_pop = lambda *_a, **_kw: None


def _maybe_trainer_cls():
    return pytest.importorskip(
        "threedgrut.trainer",
        reason="Trainer full stack unavailable on this box; math covered by "
        "unit-level _camera_loss_weight tests below.",
    ).Trainer3DGRUT


# ─── Unit: _camera_loss_weight lookup ────────────────────────────────────────


def _stub(conf):
    return SimpleNamespace(conf=conf)


def test_default_missing_key_returns_one():
    """No ``camera_loss_weights`` key at all → 1.0."""
    Trainer = _maybe_trainer_cls()
    stub = _stub(OmegaConf.create({"loss": {}}))
    assert Trainer._camera_loss_weight(stub, "camera_front_wide_120fov") == 1.0


def test_default_empty_dict_returns_one():
    """``camera_loss_weights: {}`` (base_gs.yaml default) → 1.0."""
    Trainer = _maybe_trainer_cls()
    stub = _stub(OmegaConf.create({"loss": {"camera_loss_weights": {}}}))
    assert Trainer._camera_loss_weight(stub, "camera_front_wide_120fov") == 1.0


def test_hit_returns_configured_weight():
    Trainer = _maybe_trainer_cls()
    stub = _stub(
        OmegaConf.create(
            {"loss": {"camera_loss_weights": {"camera_front_tele_30fov": 4.0}}}
        )
    )
    assert Trainer._camera_loss_weight(stub, "camera_front_tele_30fov") == 4.0


def test_miss_returns_one():
    Trainer = _maybe_trainer_cls()
    stub = _stub(
        OmegaConf.create(
            {"loss": {"camera_loss_weights": {"camera_front_tele_30fov": 4.0}}}
        )
    )
    assert Trainer._camera_loss_weight(stub, "camera_cross_left_120fov") == 1.0


def test_none_camera_id_returns_one():
    """gpu_batch may not carry camera_id (getattr returns None)."""
    Trainer = _maybe_trainer_cls()
    stub = _stub(
        OmegaConf.create(
            {"loss": {"camera_loss_weights": {"camera_front_tele_30fov": 4.0}}}
        )
    )
    assert Trainer._camera_loss_weight(stub, None) == 1.0


def test_zero_weight_is_valid():
    """0.0 must NOT be swallowed as falsy; it silences that camera photometric."""
    Trainer = _maybe_trainer_cls()
    stub = _stub(
        OmegaConf.create(
            {"loss": {"camera_loss_weights": {"camera_front_tele_30fov": 0.0}}}
        )
    )
    assert Trainer._camera_loss_weight(stub, "camera_front_tele_30fov") == 0.0


# ─── Integration: get_losses applies the weight to l1+ssim only ──────────────


def _minimal_trainer_stub(extra_loss_conf):
    """SimpleNamespace with the minimum surface get_losses needs.

    We deliberately keep image_infos absent so all image_infos-gated branches
    (layered loss, sky, lidar depth, depth prior) return zero — the test
    focuses on the L1 + SSIM + opacity + scale terms.
    """
    conf = OmegaConf.create(
        {
            "loss": {
                "use_l1": True,
                "use_l2": False,
                "lambda_l1": 0.8,
                "lambda_l2": 0.0,
                "use_ssim": True,
                "lambda_ssim": 0.2,
                "use_opacity": True,
                "lambda_opacity": 0.1,
                "exempt_layers_opacity_reg": [],
                "use_scale": True,
                "lambda_scale": 0.1,
                **extra_loss_conf,
            },
            "trainer": {},
        }
    )
    N = 4
    model = SimpleNamespace(
        get_density=lambda: torch.zeros(N, 1),
        get_scale=lambda: torch.ones(N, 3),
    )
    stub = SimpleNamespace(
        conf=conf,
        model=model,
        device=torch.device("cpu"),
        use_lidar_depth=False,
        use_depth_prior=False,
        lambda_bg_lidar=0.0,
        lambda_depth_prior=0.0,
        depth_max=100.0,
    )

    def _zero_1(_conf):
        return torch.zeros(1, device=stub.device)

    def _zero_2(_batch, _conf):
        return torch.zeros(1, device=stub.device)

    stub._maybe_fill_cuboid_mask = lambda gpu_batch, tc: None
    stub._compute_bg_cuboid_penalty_term = _zero_2
    stub._compute_bg_road_penalty_term = _zero_1
    stub._compute_pose_smoothness_term = _zero_1
    stub._compute_pose_boundary_term = _zero_1
    stub._compute_pose_prior_term = _zero_1
    # Bind the real _camera_loss_weight so get_losses can look it up on stub.
    Trainer = _maybe_trainer_cls()
    stub._camera_loss_weight = Trainer._camera_loss_weight.__get__(stub)
    return stub


def _batch_and_outputs(camera_id):
    torch.manual_seed(7)
    H, W = 4, 6
    rgb_gt = torch.rand(1, H, W, 3)
    rgb_pred = torch.rand(1, H, W, 3)
    batch = SimpleNamespace(rgb_gt=rgb_gt, mask=None)
    if camera_id is not None:
        batch.camera_id = camera_id
    outputs = {"pred_rgb": rgb_pred}
    return batch, outputs


# regularization keys that must never be touched by camera loss weights
_REG_KEYS = (
    "opacity_loss",
    "scale_loss",
    "sky_loss",
    "bg_cuboid_loss",
    "bg_road_loss",
    "pose_smooth_loss",
    "pose_boundary_loss",
    "pose_prior_loss",
    "lidar_depth_loss",
    "bg_lidar_loss",
    "depth_prior_loss",
    "road_eff_rank_loss",
)


def test_get_losses_default_empty_dict_byte_identical():
    """Default ``camera_loss_weights: {}`` → dict values identical to no-key path."""
    Trainer = _maybe_trainer_cls()
    stub_no_key = _minimal_trainer_stub({})
    stub_empty = _minimal_trainer_stub({"camera_loss_weights": {}})
    batch, outputs = _batch_and_outputs("camera_front_wide_120fov")
    a = Trainer.get_losses(stub_no_key, batch, outputs)
    b = Trainer.get_losses(stub_empty, batch, outputs)
    for k in a:
        assert torch.allclose(a[k], b[k], rtol=1e-6, atol=1e-8), k


def test_get_losses_weight_2x_doubles_l1_and_ssim_only():
    """{camX: 2.0} + batch.camera_id=camX → l1_loss/ssim_loss doubled,
    every other loss term unchanged (rtol 1e-6)."""
    Trainer = _maybe_trainer_cls()
    stub_base = _minimal_trainer_stub({})
    stub_2x = _minimal_trainer_stub({"camera_loss_weights": {"camX": 2.0}})
    batch, outputs = _batch_and_outputs("camX")
    baseline = Trainer.get_losses(stub_base, batch, outputs)
    weighted = Trainer.get_losses(stub_2x, batch, outputs)
    assert torch.allclose(weighted["l1_loss"], 2.0 * baseline["l1_loss"], rtol=1e-6)
    assert torch.allclose(weighted["ssim_loss"], 2.0 * baseline["ssim_loss"], rtol=1e-6)
    for key in _REG_KEYS:
        assert torch.allclose(weighted[key], baseline[key], rtol=1e-6, atol=1e-8), key


def test_get_losses_camera_id_absent_no_weighting():
    """No camera_id on batch → w defaults to 1.0 regardless of dict."""
    Trainer = _maybe_trainer_cls()
    stub_base = _minimal_trainer_stub({})
    stub_2x = _minimal_trainer_stub({"camera_loss_weights": {"camX": 2.0}})
    batch, outputs = _batch_and_outputs(None)
    baseline = Trainer.get_losses(stub_base, batch, outputs)
    weighted = Trainer.get_losses(stub_2x, batch, outputs)
    for k in ("l1_loss", "ssim_loss", *_REG_KEYS):
        assert torch.allclose(weighted[k], baseline[k], rtol=1e-6, atol=1e-8), k


def test_get_losses_camera_id_not_in_dict_no_weighting():
    """batch.camera_id present but not in dict → w=1.0."""
    Trainer = _maybe_trainer_cls()
    stub_base = _minimal_trainer_stub({})
    stub_2x = _minimal_trainer_stub({"camera_loss_weights": {"camY": 2.0}})
    batch, outputs = _batch_and_outputs("camX")
    baseline = Trainer.get_losses(stub_base, batch, outputs)
    weighted = Trainer.get_losses(stub_2x, batch, outputs)
    for k in ("l1_loss", "ssim_loss", *_REG_KEYS):
        assert torch.allclose(weighted[k], baseline[k], rtol=1e-6, atol=1e-8), k


def test_get_losses_weight_zero_silences_photometric():
    """weight=0.0 zeroes l1_loss + ssim_loss; regularizers untouched."""
    Trainer = _maybe_trainer_cls()
    stub_base = _minimal_trainer_stub({})
    stub_0 = _minimal_trainer_stub({"camera_loss_weights": {"camX": 0.0}})
    batch, outputs = _batch_and_outputs("camX")
    baseline = Trainer.get_losses(stub_base, batch, outputs)
    weighted = Trainer.get_losses(stub_0, batch, outputs)
    assert weighted["l1_loss"].abs().max().item() < 1e-12
    assert weighted["ssim_loss"].abs().max().item() < 1e-12
    for key in _REG_KEYS:
        assert torch.allclose(weighted[key], baseline[key], rtol=1e-6, atol=1e-8), key
