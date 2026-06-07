# SPDX-License-Identifier: Apache-2.0
"""P1.2 Stage B/C — pose boundary anchor + pose prior regularizers.

Both helpers pull the learnable per-track pose (``_track_quat_<tid>`` /
``_track_trans_<tid>``) toward the frozen GT cuboid pose
(``_track_pose_gt_<tid>`` ``[F, 4, 4]``) registered by
``LayeredGaussians._populate_tracks_impl``'s learnable branch:

  - ``compute_pose_boundary_loss``: anchors ONLY the first + last *active*
    frame of each track to GT — kills global drift of the learned trajectory
    while leaving interior frames free to refine.
  - ``compute_pose_prior_loss``: a soft L2 over *all* active frames toward
    GT — keeps the whole trajectory near GT (use a small λ relative to the
    boundary term).

Rotation is compared in rotation-matrix space (Frobenius²) so the quat
double-cover (q ≡ −q) never inflates the loss. Tested in isolation against a
minimal model stub (no Trainer, no CUDA), mirroring
``test_learnable_pose_smoothness.py``.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from threedgrut.model.pose_anchor import (
    compute_pose_boundary_loss,
    compute_pose_prior_loss,
)


# ─── helpers ────────────────────────────────────────────────────────────────


def _quat_wxyz_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Local replica of the layered_model helper (wxyz → [...,3,3])."""
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    row0 = torch.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
    row1 = torch.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)], dim=-1)
    row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _se3_from_quat_trans(q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """[F,4] wxyz + [F,3] → [F,4,4] SE(3)."""
    F = q.shape[0]
    R = _quat_wxyz_to_rotmat(q)
    T = torch.zeros(F, 4, 4, dtype=q.dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1.0
    return T


def _identity_quat(F: int) -> torch.Tensor:
    q = torch.zeros(F, 4)
    q[:, 0] = 1.0
    return q


def _yaw_quat(F: int, omega: float = 0.05) -> torch.Tensor:
    q = torch.zeros(F, 4)
    for f in range(F):
        a = omega * f
        q[f, 0] = math.cos(a / 2)
        q[f, 3] = math.sin(a / 2)
    return q


class _AnchorModelStub(nn.Module):
    """Minimal stand-in exposing the surface the anchor helpers use.

    tracks[tid] keys:
      trans [F,3] / quat [F,4]    — the *learnable* params
      active [F] bool
      gt_trans / gt_quat          — optional; default to trans/quat (→ 0 loss)
      no_gt: bool                 — skip registering _track_pose_gt_ (skip path)
    """

    def __init__(self, tracks: dict[str, dict]):
        super().__init__()
        self._tracks_active: dict[str, torch.Tensor] = {}
        for tid, d in tracks.items():
            self.register_parameter(f"_track_trans_{tid}", nn.Parameter(d["trans"].clone()))
            self.register_parameter(f"_track_quat_{tid}", nn.Parameter(d["quat"].clone()))
            a = d["active"].clone().to(torch.bool)
            self.register_buffer(f"_track_active_{tid}", a)
            self._tracks_active[tid] = a
            if not d.get("no_gt", False):
                gt = _se3_from_quat_trans(
                    d.get("gt_quat", d["quat"]).clone(),
                    d.get("gt_trans", d["trans"]).clone(),
                )
                self.register_buffer(f"_track_pose_gt_{tid}", gt)

    @property
    def tracks_active(self):
        return self._tracks_active


def _one_track(F=8, **over) -> _AnchorModelStub:
    base = {
        "trans": torch.zeros(F, 3),
        "quat": _identity_quat(F),
        "active": torch.ones(F, dtype=torch.bool),
    }
    base.update(over)
    return _AnchorModelStub({"t0": base})


# ════════════════════════════════════════════════════════════════════════════
# Boundary anchor
# ════════════════════════════════════════════════════════════════════════════


def test_boundary_zero_when_learned_equals_gt():
    """learned pose == GT → boundary loss exactly 0."""
    F = 8
    t = torch.randn(F, 3)
    q = _yaw_quat(F)
    model = _AnchorModelStub({"t0": {"trans": t, "quat": q,
                                     "active": torch.ones(F, dtype=torch.bool)}})
    loss = compute_pose_boundary_loss(model, lambda_trans=1.0, lambda_rot=1.0)
    assert loss.shape == (1,)
    assert loss.abs().item() < 1e-10


def test_boundary_penalizes_first_active_frame_translation():
    """Offset learned trans at the first frame only → boundary trans > 0."""
    F = 6
    gt_t = torch.zeros(F, 3)
    learned_t = gt_t.clone()
    learned_t[0] = torch.tensor([0.5, 0.0, 0.0])  # endpoint offset
    model = _AnchorModelStub({"t0": {
        "trans": learned_t, "quat": _identity_quat(F),
        "active": torch.ones(F, dtype=torch.bool),
        "gt_trans": gt_t, "gt_quat": _identity_quat(F),
    }})
    loss = compute_pose_boundary_loss(model, lambda_trans=1.0, lambda_rot=0.0)
    # endpoints = {0, 5}; only frame 0 offset by 0.5 → sum_sq=0.25 over n=2.
    assert abs(loss.item() - (0.25 / 2.0)) < 1e-6


def test_boundary_ignores_interior_frame():
    """Offset only an interior frame (not first/last) → boundary loss 0."""
    F = 6
    gt_t = torch.zeros(F, 3)
    learned_t = gt_t.clone()
    learned_t[3] = torch.tensor([1.0, 1.0, 1.0])  # interior, not endpoint
    model = _AnchorModelStub({"t0": {
        "trans": learned_t, "quat": _identity_quat(F),
        "active": torch.ones(F, dtype=torch.bool),
        "gt_trans": gt_t,
    }})
    loss = compute_pose_boundary_loss(model, lambda_trans=1.0, lambda_rot=0.0)
    assert loss.abs().item() < 1e-12


def test_boundary_endpoints_follow_active_mask():
    """active=[0,1,1,1,0] → endpoints are frames 1 and 3, not 0/4."""
    F = 5
    active = torch.tensor([0, 1, 1, 1, 0], dtype=torch.bool)
    gt_t = torch.zeros(F, 3)
    # offset frame 2 (interior of active span) → should NOT contribute.
    learned_interior = gt_t.clone()
    learned_interior[2] = torch.tensor([2.0, 0.0, 0.0])
    m_int = _AnchorModelStub({"t0": {"trans": learned_interior,
                                     "quat": _identity_quat(F), "active": active,
                                     "gt_trans": gt_t}})
    assert compute_pose_boundary_loss(m_int, 1.0, 0.0).abs().item() < 1e-12

    # offset frame 1 (first active) → should contribute.
    learned_edge = gt_t.clone()
    learned_edge[1] = torch.tensor([0.3, 0.0, 0.0])
    m_edge = _AnchorModelStub({"t0": {"trans": learned_edge,
                                      "quat": _identity_quat(F), "active": active,
                                      "gt_trans": gt_t}})
    assert compute_pose_boundary_loss(m_edge, 1.0, 0.0).item() > 0.0


def test_boundary_rotation_sign_robust():
    """GT stored as −q (double cover): matrix-space compare → still 0."""
    F = 6
    q = _yaw_quat(F)
    gt_q = q.clone()
    gt_q[0] = -gt_q[0]  # same rotation, flipped sign at endpoint
    model = _AnchorModelStub({"t0": {"trans": torch.zeros(F, 3), "quat": q,
                                     "active": torch.ones(F, dtype=torch.bool),
                                     "gt_quat": gt_q, "gt_trans": torch.zeros(F, 3)}})
    loss = compute_pose_boundary_loss(model, lambda_trans=0.0, lambda_rot=1.0)
    assert loss.abs().item() < 1e-10


def test_boundary_lambda_zero_returns_zero():
    F = 6
    model = _one_track(F, trans=torch.randn(F, 3), gt_trans=torch.zeros(F, 3))
    loss = compute_pose_boundary_loss(model, lambda_trans=0.0, lambda_rot=0.0)
    assert loss.shape == (1,)
    assert loss.item() == 0.0
    assert loss.requires_grad is False


def test_boundary_gradient_flows():
    F = 6
    gt_t = torch.zeros(F, 3)
    learned_t = torch.randn(F, 3)
    model = _AnchorModelStub({"t0": {"trans": learned_t,
                                     "quat": _identity_quat(F) + 0.01 * torch.randn(F, 4),
                                     "active": torch.ones(F, dtype=torch.bool),
                                     "gt_trans": gt_t, "gt_quat": _identity_quat(F)}})
    loss = compute_pose_boundary_loss(model, lambda_trans=1.0, lambda_rot=1.0)
    loss.sum().backward()
    assert model._track_trans_t0.grad.abs().max().item() > 0.0
    assert model._track_quat_t0.grad.abs().max().item() > 0.0


def test_boundary_no_tracks_returns_zero():
    model = _AnchorModelStub({})
    loss = compute_pose_boundary_loss(model, lambda_trans=1.0, lambda_rot=1.0)
    assert loss.shape == (1,)
    assert loss.item() == 0.0


def test_boundary_skips_track_without_gt():
    """Track lacking _track_pose_gt_ (legacy/buffer mode) is skipped silently."""
    F = 6
    model = _AnchorModelStub({"t0": {"trans": torch.randn(F, 3),
                                     "quat": _identity_quat(F),
                                     "active": torch.ones(F, dtype=torch.bool),
                                     "no_gt": True}})
    loss = compute_pose_boundary_loss(model, lambda_trans=1.0, lambda_rot=1.0)
    assert loss.item() == 0.0


# ════════════════════════════════════════════════════════════════════════════
# Pose prior (all active frames)
# ════════════════════════════════════════════════════════════════════════════


def test_prior_zero_when_learned_equals_gt():
    F = 8
    t = torch.randn(F, 3)
    q = _yaw_quat(F)
    model = _AnchorModelStub({"t0": {"trans": t, "quat": q,
                                     "active": torch.ones(F, dtype=torch.bool)}})
    loss = compute_pose_prior_loss(model, lambda_trans=1.0, lambda_rot=1.0)
    assert loss.abs().item() < 1e-10


def test_prior_monotone_in_offset():
    F = 10
    gt_t = torch.zeros(F, 3)
    losses = []
    for sigma in [0.0, 0.2, 1.0]:
        torch.manual_seed(5)
        learned_t = gt_t + sigma * torch.randn(F, 3)
        model = _AnchorModelStub({"t0": {"trans": learned_t, "quat": _identity_quat(F),
                                         "active": torch.ones(F, dtype=torch.bool),
                                         "gt_trans": gt_t}})
        losses.append(compute_pose_prior_loss(model, 1.0, 0.0).item())
    assert losses[0] <= losses[1] <= losses[2]
    assert losses[2] > losses[0] + 1e-3


def test_prior_ignores_inactive_frames():
    """Offsetting only inactive frames must not change the prior loss."""
    F = 4
    active = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
    gt_t = torch.zeros(F, 3)
    learned_t = gt_t.clone()
    learned_t[2:] = torch.tensor([[9.0, 9.0, 9.0], [9.0, 9.0, 9.0]])  # inactive only
    model = _AnchorModelStub({"t0": {"trans": learned_t, "quat": _identity_quat(F),
                                     "active": active, "gt_trans": gt_t}})
    assert compute_pose_prior_loss(model, 1.0, 0.0).abs().item() < 1e-12


def test_prior_lambda_zero_returns_zero():
    F = 6
    model = _one_track(F, trans=torch.randn(F, 3), gt_trans=torch.zeros(F, 3))
    loss = compute_pose_prior_loss(model, lambda_trans=0.0, lambda_rot=0.0)
    assert loss.item() == 0.0
    assert loss.requires_grad is False


def test_prior_gradient_flows():
    F = 6
    model = _AnchorModelStub({"t0": {"trans": torch.randn(F, 3),
                                     "quat": _identity_quat(F) + 0.01 * torch.randn(F, 4),
                                     "active": torch.ones(F, dtype=torch.bool),
                                     "gt_trans": torch.zeros(F, 3),
                                     "gt_quat": _identity_quat(F)}})
    loss = compute_pose_prior_loss(model, lambda_trans=1.0, lambda_rot=1.0)
    loss.sum().backward()
    assert model._track_trans_t0.grad.abs().max().item() > 0.0
    assert model._track_quat_t0.grad.abs().max().item() > 0.0


# ════════════════════════════════════════════════════════════════════════════
# Trainer-level gating (mirrors test_learnable_pose_smoothness §9). These pull
# in the full training stack and so skip on Mac via importorskip; they run on
# A800 where Trainer imports cleanly.
# ════════════════════════════════════════════════════════════════════════════

from types import SimpleNamespace  # noqa: E402


def _maybe_trainer():
    return pytest.importorskip(
        "threedgrut.trainer",
        reason="Trainer-method tests need full training stack; helper-level "
               "tests above already cover the math.",
    ).Trainer3DGRUT


def _offset_model(F=8):
    """One track whose learned trans is drifted off GT (zeros)."""
    torch.manual_seed(2)
    return _AnchorModelStub({"t0": {"trans": torch.randn(F, 3),
                                    "quat": _identity_quat(F),
                                    "active": torch.ones(F, dtype=torch.bool),
                                    "gt_trans": torch.zeros(F, 3)}})


def test_trainer_boundary_freeze_window_returns_zero():
    """global_step < freeze_until_iter → boundary term suppressed."""
    Trainer = _maybe_trainer()
    fake = SimpleNamespace(
        device=torch.device("cpu"), pose_optimizer=object(),
        global_step=10, pose_freeze_until_iter=200, model=_offset_model(),
    )
    conf = SimpleNamespace(learnable_pose=SimpleNamespace(
        fix_first_last=True,
        lambda_pose_boundary_trans=1.0, lambda_pose_boundary_rot=1.0,
    ))
    out = Trainer._compute_pose_boundary_term(fake, conf)
    assert out.shape == (1,)
    assert out.item() == 0.0


def test_trainer_boundary_disabled_returns_zero():
    """pose_optimizer None (poseopt off) → zero."""
    Trainer = _maybe_trainer()
    fake = SimpleNamespace(
        device=torch.device("cpu"), pose_optimizer=None,
        global_step=10000, pose_freeze_until_iter=200, model=_offset_model(),
    )
    conf = SimpleNamespace(learnable_pose=SimpleNamespace(
        fix_first_last=True,
        lambda_pose_boundary_trans=1.0, lambda_pose_boundary_rot=1.0,
    ))
    assert Trainer._compute_pose_boundary_term(fake, conf).item() == 0.0


def test_trainer_boundary_fix_flag_off_returns_zero():
    """fix_first_last=False master toggle → zero even with λ>0."""
    Trainer = _maybe_trainer()
    fake = SimpleNamespace(
        device=torch.device("cpu"), pose_optimizer=object(),
        global_step=5000, pose_freeze_until_iter=200, model=_offset_model(),
    )
    conf = SimpleNamespace(learnable_pose=SimpleNamespace(
        fix_first_last=False,
        lambda_pose_boundary_trans=1.0, lambda_pose_boundary_rot=1.0,
    ))
    assert Trainer._compute_pose_boundary_term(fake, conf).item() == 0.0


def test_trainer_boundary_active_path():
    """All gates open + λ>0 + drifted trans → nonzero."""
    Trainer = _maybe_trainer()
    fake = SimpleNamespace(
        device=torch.device("cpu"), pose_optimizer=object(),
        global_step=5000, pose_freeze_until_iter=200, model=_offset_model(),
    )
    conf = SimpleNamespace(learnable_pose=SimpleNamespace(
        fix_first_last=True,
        lambda_pose_boundary_trans=1.0, lambda_pose_boundary_rot=0.0,
    ))
    assert Trainer._compute_pose_boundary_term(fake, conf).item() > 0.0


def test_trainer_prior_freeze_window_returns_zero():
    Trainer = _maybe_trainer()
    fake = SimpleNamespace(
        device=torch.device("cpu"), pose_optimizer=object(),
        global_step=10, pose_freeze_until_iter=200, model=_offset_model(),
    )
    conf = SimpleNamespace(learnable_pose=SimpleNamespace(
        lambda_pose_prior_trans=1.0, lambda_pose_prior_rot=1.0,
    ))
    assert Trainer._compute_pose_prior_term(fake, conf).item() == 0.0


def test_trainer_prior_lambda_zero_returns_zero():
    """Default placeholder λ=0 → byte-identical (prior off)."""
    Trainer = _maybe_trainer()
    fake = SimpleNamespace(
        device=torch.device("cpu"), pose_optimizer=object(),
        global_step=5000, pose_freeze_until_iter=200, model=_offset_model(),
    )
    conf = SimpleNamespace(learnable_pose=SimpleNamespace(
        lambda_pose_prior_trans=0.0, lambda_pose_prior_rot=0.0,
    ))
    assert Trainer._compute_pose_prior_term(fake, conf).item() == 0.0


def test_trainer_prior_active_path():
    Trainer = _maybe_trainer()
    fake = SimpleNamespace(
        device=torch.device("cpu"), pose_optimizer=object(),
        global_step=5000, pose_freeze_until_iter=200, model=_offset_model(),
    )
    conf = SimpleNamespace(learnable_pose=SimpleNamespace(
        lambda_pose_prior_trans=1.0, lambda_pose_prior_rot=0.0,
    ))
    assert Trainer._compute_pose_prior_term(fake, conf).item() > 0.0
