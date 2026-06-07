# SPDX-License-Identifier: Apache-2.0
"""V3 Stage B — temporal smoothness regularizer unit tests.

Tests the `compute_pose_smoothness_loss` helper in isolation (no Trainer,
no CUDA) using a minimal model stub that exposes the same surface as
`LayeredGaussians`:

  - ``tracks_active``: dict[str, BoolTensor[F]]
  - ``_track_quat_<tid>``: nn.Parameter[F, 4]   (wxyz unit quat)
  - ``_track_trans_<tid>``: nn.Parameter[F, 3]
  - ``_track_active_<tid>``: BoolTensor[F]      (mirrors tracks_active)

The helper is the unit under test; the trainer-method
``_compute_pose_smoothness_term`` is just gating + delegation, asserted by
the active-mask + freeze-window cases below.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from threedgrut.model.pose_smoothness import compute_pose_smoothness_loss


# ─── helpers ────────────────────────────────────────────────────────────────


class _ModelStub(nn.Module):
    """Tiny stand-in matching the surface compute_pose_smoothness_loss uses.

    Inherits from nn.Module so register_parameter / register_buffer work; the
    helper itself only does ``getattr`` so a plain object would also do, but
    nn.Module makes the test feel like the real LayeredGaussians.
    """

    def __init__(self, tracks: dict[str, dict]):
        super().__init__()
        self._tracks_active: dict[str, torch.Tensor] = {}
        for tid, d in tracks.items():
            t = d["trans"].clone()  # [F, 3]
            q = d["quat"].clone()   # [F, 4]
            a = d["active"].clone().to(torch.bool)
            self.register_parameter(f"_track_trans_{tid}", nn.Parameter(t))
            self.register_parameter(f"_track_quat_{tid}", nn.Parameter(q))
            self.register_buffer(f"_track_active_{tid}", a)
            self._tracks_active[tid] = a

    @property
    def tracks_active(self):
        return self._tracks_active


def _linear_trans(F: int, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """t[f] = a + f·b  →  Δ²t ≡ 0."""
    fs = torch.arange(F, dtype=torch.float32).unsqueeze(-1)   # [F, 1]
    return a.unsqueeze(0) + fs * b.unsqueeze(0)


def _identity_quat(F: int) -> torch.Tensor:
    """All-frames identity wxyz quat."""
    q = torch.zeros(F, 4)
    q[:, 0] = 1.0
    return q


def _build_tracks(
    F: int,
    n_tracks: int = 1,
    trans_init: str = "linear",
    quat_init: str = "identity",
    active: torch.Tensor | None = None,
) -> dict:
    """Build a tracks dict with controlled trans / quat / active patterns."""
    tracks = {}
    if active is None:
        active = torch.ones(F, dtype=torch.bool)
    for i in range(n_tracks):
        tid = f"t{i}"
        if trans_init == "linear":
            t = _linear_trans(F, torch.zeros(3), torch.tensor([1.0, 0.0, 0.0]))
        elif trans_init == "zero":
            t = torch.zeros(F, 3)
        elif trans_init == "jitter":
            torch.manual_seed(123 + i)
            t = torch.randn(F, 3)
        else:
            raise ValueError(trans_init)
        if quat_init == "identity":
            q = _identity_quat(F)
        elif quat_init == "alternating_sign":
            # Same rotation, alternating sign — chord distance should still be 0
            # under sign-aligned smoothing.
            q = _identity_quat(F)
            q[1::2] = -q[1::2]
        elif quat_init == "linear_yaw":
            # Constant angular velocity — Δ²(quat) is small but nonzero
            # because slerp != linear in 4-space; for the "linear" test we
            # don't use this one, only as a sanity input.
            q = torch.zeros(F, 4)
            for f in range(F):
                yaw = 0.05 * f
                q[f, 0] = math.cos(yaw / 2)
                q[f, 3] = math.sin(yaw / 2)
        else:
            raise ValueError(quat_init)
        tracks[tid] = {"trans": t, "quat": q, "active": active.clone()}
    return tracks


# ─── 1. linear trajectory → trans smoothness = 0 ────────────────────────────


def test_smoothness_zero_for_linear_trans():
    """t[f] = a + f·b → Δ²t = 0 → loss == 0."""
    model = _ModelStub(_build_tracks(F=10, trans_init="linear",
                                     quat_init="identity"))
    loss = compute_pose_smoothness_loss(model, lambda_trans=1.0, lambda_rot=0.0)
    assert loss.shape == (1,)
    assert loss.abs().item() < 1e-12, f"linear trans should give 0 loss, got {loss.item()}"


def test_smoothness_zero_for_constant_quat():
    """All-frames identity quat → Δ²q = 0 → rot loss == 0."""
    model = _ModelStub(_build_tracks(F=10, quat_init="identity",
                                     trans_init="linear"))
    loss = compute_pose_smoothness_loss(model, lambda_trans=0.0, lambda_rot=1.0)
    assert loss.abs().item() < 1e-12


# ─── 2. jitter → loss > 0, monotone in noise scale ──────────────────────────


def test_smoothness_monotone_in_noise():
    """Trans noise σ ↑ ⇒ loss ↑."""
    F = 16
    torch.manual_seed(42)
    base = torch.randn(F, 3)
    losses = []
    for sigma in [0.0, 0.1, 1.0]:
        torch.manual_seed(7)
        t = base + sigma * torch.randn(F, 3)
        tracks = {
            "t0": {
                "trans": t,
                "quat": _identity_quat(F),
                "active": torch.ones(F, dtype=torch.bool),
            }
        }
        model = _ModelStub(tracks)
        loss = compute_pose_smoothness_loss(model, lambda_trans=1.0, lambda_rot=0.0)
        losses.append(loss.item())
    assert losses[0] <= losses[1] <= losses[2], f"non-monotone: {losses}"
    assert losses[2] > losses[0] + 1e-3, "noise should produce nontrivial loss"


# ─── 3. active mask excludes boundary triples ───────────────────────────────


def test_active_mask_excludes_boundary():
    """active = [0,1,1,1,0]: only f=2 is a triple-active interior frame."""
    F = 5
    torch.manual_seed(0)
    t = torch.randn(F, 3)
    q = _identity_quat(F)
    active_partial = torch.tensor([0, 1, 1, 1, 0], dtype=torch.bool)
    active_full = torch.ones(F, dtype=torch.bool)

    m_partial = _ModelStub({"t0": {"trans": t, "quat": q, "active": active_partial}})
    m_full = _ModelStub({"t0": {"trans": t, "quat": q, "active": active_full}})

    loss_partial = compute_pose_smoothness_loss(
        m_partial, lambda_trans=1.0, lambda_rot=0.0
    ).item()
    loss_full = compute_pose_smoothness_loss(
        m_full, lambda_trans=1.0, lambda_rot=0.0
    ).item()
    # Both > 0 (random noise) but partial uses only 1 of 3 valid triples.
    assert loss_partial > 0.0
    assert loss_full > 0.0
    # The two losses are *averages* over n_valid triples, so they aren't
    # ordered by inclusion. The point of the test is that the mask gate
    # does NOT raise and only nonzero-mask frames contribute. Verify by
    # explicitly computing the expected partial value.
    d2 = t[2:] - 2.0 * t[1:-1] + t[:-2]      # [F-2, 3]
    sq = (d2 * d2).sum(dim=-1)               # [F-2]
    mask = active_partial[:-2] & active_partial[1:-1] & active_partial[2:]
    expected = (sq * mask.to(sq.dtype)).sum() / float(mask.sum().item())
    assert abs(loss_partial - expected.item()) < 1e-6


def test_no_triple_active_returns_zero():
    """active = [1,0,1,0,1]: no f has a[f-1]=a[f]=a[f+1]=1 → loss = 0."""
    F = 5
    torch.manual_seed(0)
    tracks = {
        "t0": {
            "trans": torch.randn(F, 3),
            "quat": _identity_quat(F),
            "active": torch.tensor([1, 0, 1, 0, 1], dtype=torch.bool),
        }
    }
    model = _ModelStub(tracks)
    loss = compute_pose_smoothness_loss(model, lambda_trans=1.0, lambda_rot=1.0)
    assert loss.abs().item() < 1e-12


# ─── 4. quat sign alignment ─────────────────────────────────────────────────


def test_quat_sign_alignment_zero_chord():
    """q[f+1] = -q[f] same rotation → after sign-align chord distance = 0."""
    F = 5
    q = torch.zeros(F, 4)
    q[:, 0] = 1.0
    q[1::2] = -q[1::2]   # [+,-,+,-,+]
    tracks = {
        "t0": {
            "trans": torch.zeros(F, 3),
            "quat": q,
            "active": torch.ones(F, dtype=torch.bool),
        }
    }
    model = _ModelStub(tracks)
    loss = compute_pose_smoothness_loss(model, lambda_trans=0.0, lambda_rot=1.0)
    assert loss.abs().item() < 1e-10, (
        f"sign-aligned identical-rotation quats should give 0 loss, got {loss.item()}"
    )


# ─── 5. lambda=0 path returns zero (byte-identical-loss invariant) ──────────


def test_lambda_zero_returns_zero():
    """λ_t = λ_r = 0 → return torch.zeros(1) without touching the model."""
    F = 8
    torch.manual_seed(99)
    tracks = {
        "t0": {
            "trans": torch.randn(F, 3),    # nonzero Δ²
            "quat": _identity_quat(F),
            "active": torch.ones(F, dtype=torch.bool),
        }
    }
    model = _ModelStub(tracks)
    loss = compute_pose_smoothness_loss(model, lambda_trans=0.0, lambda_rot=0.0)
    assert loss.shape == (1,)
    assert loss.item() == 0.0
    assert loss.requires_grad is False  # short-circuit, no graph


# ─── 6. gradient flows to quat + trans Parameters ───────────────────────────


def test_gradient_flows_to_trans_and_quat():
    """Loss.backward() populates grad on _track_trans_* and _track_quat_*."""
    F = 6
    torch.manual_seed(1)
    tracks = {
        "t0": {
            "trans": torch.randn(F, 3),
            "quat": _identity_quat(F)
                + 0.01 * torch.randn(F, 4),  # break exact constancy
            "active": torch.ones(F, dtype=torch.bool),
        }
    }
    model = _ModelStub(tracks)
    loss = compute_pose_smoothness_loss(model, lambda_trans=1.0, lambda_rot=1.0)
    loss.sum().backward()
    assert model._track_trans_t0.grad is not None
    assert model._track_trans_t0.grad.abs().max().item() > 0.0
    assert model._track_quat_t0.grad is not None
    assert model._track_quat_t0.grad.abs().max().item() > 0.0


# ─── 7. F < 3: no-op, no error ──────────────────────────────────────────────


@pytest.mark.parametrize("F", [1, 2])
def test_short_clip_no_error(F):
    """F=1 / F=2 can't define second-order diff → return 0, don't crash."""
    tracks = {
        "t0": {
            "trans": torch.randn(F, 3),
            "quat": _identity_quat(F),
            "active": torch.ones(F, dtype=torch.bool),
        }
    }
    model = _ModelStub(tracks)
    loss = compute_pose_smoothness_loss(model, lambda_trans=1.0, lambda_rot=1.0)
    assert loss.abs().item() < 1e-12


# ─── 8. empty / no-tracks model returns zero ────────────────────────────────


def test_no_tracks_returns_zero():
    """Module with no tracks (legacy buffer-mode pre-populate, etc.) → 0."""
    model = _ModelStub({})
    loss = compute_pose_smoothness_loss(model, lambda_trans=1.0, lambda_rot=1.0)
    assert loss.shape == (1,)
    assert loss.item() == 0.0


# ─── 9. trainer-level gating: freeze window suppresses reg ──────────────────
# These tests exercise the bound method ``Trainer._compute_pose_smoothness_term``
# which lives in ``threedgrut.trainer``. That import pulls in the full training
# stack (addict, render, CUDA-only deps) and is unavailable on Mac dev boxes.
# Per-test skip via importorskip so the helper-level tests above always run.


def _maybe_trainer():
    return pytest.importorskip(
        "threedgrut.trainer",
        reason="Trainer-method tests need full training stack; helper-level "
               "tests above already cover the math.",
    ).Trainer3DGRUT


def test_trainer_method_freeze_window_returns_zero():
    """Stage A invariant: while global_step < freeze_until_iter pose Adam
    is gated off; reg should also be off so it doesn't accumulate
    spurious gradients on Parameters that aren't being stepped.

    We don't construct a real Trainer (CUDA-heavy). Instead we exercise
    the bound method via a hand-built SimpleNamespace.
    """
    Trainer = _maybe_trainer()
    F = 8
    torch.manual_seed(2)
    tracks = {
        "t0": {
            "trans": torch.randn(F, 3),
            "quat": _identity_quat(F),
            "active": torch.ones(F, dtype=torch.bool),
        }
    }
    model = _ModelStub(tracks)
    fake = SimpleNamespace(
        device=torch.device("cpu"),
        pose_optimizer=object(),         # truthy = enabled
        global_step=10,
        pose_freeze_until_iter=200,      # 10 < 200 → freeze
        model=model,
    )
    trainer_conf = SimpleNamespace(
        learnable_pose=SimpleNamespace(
            lambda_temporal_smooth_trans=1.0,
            lambda_temporal_smooth_rot=1.0,
        )
    )
    out = Trainer._compute_pose_smoothness_term(fake, trainer_conf)
    assert out.shape == (1,)
    assert out.item() == 0.0


def test_trainer_method_disabled_returns_zero():
    """pose_optimizer is None (learnable_pose disabled) → return 0."""
    Trainer = _maybe_trainer()
    fake = SimpleNamespace(
        device=torch.device("cpu"),
        pose_optimizer=None,
        global_step=10000,
        pose_freeze_until_iter=200,
        model=_ModelStub({}),
    )
    trainer_conf = SimpleNamespace(
        learnable_pose=SimpleNamespace(
            lambda_temporal_smooth_trans=1.0,
            lambda_temporal_smooth_rot=1.0,
        )
    )
    out = Trainer._compute_pose_smoothness_term(fake, trainer_conf)
    assert out.item() == 0.0


def test_trainer_method_active_path():
    """All gates open + λ>0 + jittered trans → returns nonzero."""
    Trainer = _maybe_trainer()
    F = 8
    torch.manual_seed(3)
    tracks = {
        "t0": {
            "trans": torch.randn(F, 3),
            "quat": _identity_quat(F),
            "active": torch.ones(F, dtype=torch.bool),
        }
    }
    model = _ModelStub(tracks)
    fake = SimpleNamespace(
        device=torch.device("cpu"),
        pose_optimizer=object(),
        global_step=500,
        pose_freeze_until_iter=200,
        model=model,
    )
    trainer_conf = SimpleNamespace(
        learnable_pose=SimpleNamespace(
            lambda_temporal_smooth_trans=1.0,
            lambda_temporal_smooth_rot=0.0,
        )
    )
    out = Trainer._compute_pose_smoothness_term(fake, trainer_conf)
    assert out.item() > 0.0
