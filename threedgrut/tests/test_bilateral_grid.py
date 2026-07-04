# SPDX-License-Identifier: Apache-2.0
"""T9.1 unit tests for BilateralGrid color correction module.

Replaces test_exposure.py once trainer wiring migrates; keeps key invariants:
- Identity init = exact identity transform
- per-camera grad isolation
- clamp(0, 1) output
- invalid idx → IndexError
- num_camera < 1 → ValueError
- state_dict roundtrip
- tv_loss == 0 at 1x1x1 identity init
- 1x1x1 fast path == general grid_sample path (math equivalence)
"""

from __future__ import annotations

import pytest
import torch

from threedgrut.correction.bilateral_grid import (
    BilateralGrid,
    color_affine_transform,
    total_variation_loss,
)


def test_zero_init_is_identity_1x1x1():
    """1x1x1 grid, identity init: output ≡ input for any RGB."""
    bg = BilateralGrid(num_camera=3, grid_X=1, grid_Y=1, grid_W=1)
    img = torch.rand(8, 8, 3)
    out = bg(0, img)
    assert torch.allclose(out, img.clamp(0, 1), atol=1e-6)
    out = bg(2, img)
    assert torch.allclose(out, img.clamp(0, 1), atol=1e-6)


def test_zero_init_is_identity_larger_grid():
    """Larger grid (4x4x2), identity init: output ≡ input via grid_sample path."""
    bg = BilateralGrid(num_camera=2, grid_X=4, grid_Y=4, grid_W=2)
    img = torch.rand(8, 8, 3)
    out = bg(0, img)
    assert torch.allclose(out, img.clamp(0, 1), atol=1e-5)


def test_per_camera_grad_isolation():
    """Forwarding camera 0 only flows grad into grids[0], not grids[1]."""
    bg = BilateralGrid(num_camera=4, grid_X=1, grid_Y=1, grid_W=1)
    img = torch.rand(4, 4, 3, requires_grad=False)
    out = bg(1, img)
    loss = out.sum()
    loss.backward()
    assert bg.grids.grad is not None
    grad = bg.grids.grad
    assert grad[1].abs().sum() > 0, "camera 1 grid should receive grad"
    for i in (0, 2, 3):
        assert grad[i].abs().sum() == 0, f"camera {i} grid must not receive grad"


def test_clamp_to_unit_range():
    """If learned affine pushes outside [0, 1], output is clamped."""
    bg = BilateralGrid(num_camera=1)
    # Push gain to 5x → output far above 1
    with torch.no_grad():
        bg.grids.zero_()
        bg.grids[0, 0, 0, 0, 0] = 5.0  # R = 5*R + 0
        bg.grids[0, 5, 0, 0, 0] = 5.0  # G = 5*G + 0
        bg.grids[0, 10, 0, 0, 0] = 5.0  # B = 5*B + 0
    img = torch.full((4, 4, 3), 0.5)
    out = bg(0, img)
    assert (out >= 0).all() and (out <= 1).all()


def test_invalid_camera_idx_raises():
    bg = BilateralGrid(num_camera=3)
    img = torch.rand(2, 2, 3)
    with pytest.raises(IndexError):
        bg(3, img)
    with pytest.raises(IndexError):
        bg(-1, img)


def test_constructor_rejects_zero_cameras():
    with pytest.raises(ValueError):
        BilateralGrid(num_camera=0)


def test_constructor_rejects_zero_grid_dims():
    with pytest.raises(ValueError):
        BilateralGrid(num_camera=2, grid_X=0)
    with pytest.raises(ValueError):
        BilateralGrid(num_camera=2, grid_Y=0)
    with pytest.raises(ValueError):
        BilateralGrid(num_camera=2, grid_W=0)


def test_state_dict_roundtrip():
    """Save state_dict → load into fresh module → identical forward output."""
    bg1 = BilateralGrid(num_camera=3, grid_X=2, grid_Y=2, grid_W=2)
    # Perturb grids
    with torch.no_grad():
        bg1.grids.add_(0.1 * torch.randn_like(bg1.grids))

    bg2 = BilateralGrid(num_camera=3, grid_X=2, grid_Y=2, grid_W=2)
    bg2.load_state_dict(bg1.state_dict())

    img = torch.rand(6, 6, 3)
    for idx in range(3):
        assert torch.allclose(bg1(idx, img), bg2(idx, img), atol=1e-6)


def test_tv_loss_zero_at_1x1x1_identity():
    """TV is 0 for 1x1x1 grid (no spatial neighbors to differ)."""
    bg = BilateralGrid(num_camera=5)
    assert bg.tv_loss().item() == 0.0


def test_tv_loss_zero_at_larger_identity():
    """TV is 0 when all voxels are identity (no spatial variation)."""
    bg = BilateralGrid(num_camera=2, grid_X=4, grid_Y=4, grid_W=2)
    assert bg.tv_loss().item() == 0.0


def test_tv_loss_positive_when_perturbed():
    """TV > 0 once grids deviate spatially."""
    bg = BilateralGrid(num_camera=2, grid_X=4, grid_Y=4, grid_W=2)
    with torch.no_grad():
        bg.grids.add_(0.5 * torch.randn_like(bg.grids))
    assert bg.tv_loss().item() > 0


def test_fast_path_matches_grid_sample_for_constant_grid():
    """At 1x1x1 the fast path is exercised; verify same affine result if we
    feed the same (constant) grid into the general path via a 2x2x2 build."""
    bg_fast = BilateralGrid(num_camera=1, grid_X=1, grid_Y=1, grid_W=1)
    bg_general = BilateralGrid(num_camera=1, grid_X=2, grid_Y=2, grid_W=2)
    # Make the fast-path voxel non-identity, and replicate across general grid.
    perturbed_affine = torch.tensor([[1.2, 0.1, -0.05, 0.02], [0.0, 0.9, 0.05, -0.01], [-0.03, 0.04, 1.1, 0.03]])
    with torch.no_grad():
        bg_fast.grids[0, :, 0, 0, 0] = perturbed_affine.reshape(12)
        # Replicate the same affine across every voxel of the 2x2x2 grid.
        bg_general.grids[0] = perturbed_affine.reshape(12, 1, 1, 1).expand(12, 2, 2, 2).contiguous()

    img = torch.rand(6, 6, 3)
    out_fast = bg_fast(0, img)
    out_general = bg_general(0, img)
    # Both should produce the same affine-transformed-then-clamped output.
    assert torch.allclose(out_fast, out_general, atol=1e-5)


def test_color_affine_transform_identity():
    """color_affine_transform with identity affine returns rgb unchanged."""
    identity = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]])
    rgb = torch.tensor([0.3, 0.7, 0.2])
    out = color_affine_transform(identity, rgb)
    assert torch.allclose(out, rgb, atol=1e-7)


def test_total_variation_loss_zero_uniform():
    """TV of a uniform tensor is 0."""
    x = torch.ones(2, 12, 4, 4, 4)
    assert total_variation_loss(x).item() == 0.0


# T9.2: optimizer + scheduler + 2-stage freeze tests --------------------------


def test_t9_2_adamw_with_weight_decay_constructs():
    """AdamW accepts weight_decay=1e-4; loop runs without crash."""
    bg = BilateralGrid(num_camera=3, grid_X=1, grid_Y=1, grid_W=1)
    optim = torch.optim.AdamW(bg.parameters(), lr=1e-3, weight_decay=1e-4)
    img = torch.rand(4, 4, 3)
    for _ in range(5):
        out = bg(0, img)
        loss = out.sum()
        loss.backward()
        optim.step()
        optim.zero_grad()
    assert bg.grids.requires_grad


def test_t9_2_adamw_decay_negligible_at_identity_init_zero_grad():
    """AdamW (decoupled wd): per-step pull is ~lr*wd*θ ≈ 1e-7 for θ=1
    regardless of gradient magnitude. 100 zero-grad steps → drift ~1e-5
    (well below 0.001 threshold).

    This is why trainer.py uses AdamW not Adam for BilateralGrid. Adam's
    L2-coupled weight_decay flows through the m/v accumulators and gets
    AMPLIFIED to ~lr-magnitude steps when the photometric gradient is small
    (e.g. very early in training before geometry stabilises). Observed: with
    plain Adam, the same 100-step zero-grad test produced 0.098 drift on
    identity-init diagonal voxels — would compromise identity behaviour.
    AdamW's decoupled wd fixes this.
    """
    bg = BilateralGrid(num_camera=2, grid_X=1, grid_Y=1, grid_W=1)
    optim = torch.optim.AdamW(bg.parameters(), lr=1e-3, weight_decay=1e-4)
    initial = bg.grids.detach().clone()
    for _ in range(100):
        loss = (bg.grids * 0).sum()  # zero grad source
        loss.backward()
        optim.step()
        optim.zero_grad()
    drift = (bg.grids - initial).abs().max().item()
    assert drift < 0.001, f"AdamW decay drift {drift} too large; " f"expected ~1e-5 (= 100 * lr * wd) at identity init"


def _drive_optim_step(optim):
    """Helper: do one fake optim.step() so PyTorch is happy about
    sched.step() ordering (the 1.1+ convention is optim.step() then
    sched.step()). Avoids the UserWarning + first-value-skip behavior."""
    for group in optim.param_groups:
        for p in group["params"]:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
    optim.step()
    optim.zero_grad()


def test_t9_2_cosine_annealing_lr_decreases():
    """CosineAnnealingLR(T_max=N) anneals lr from initial → 0 over N steps."""
    bg = BilateralGrid(num_camera=2)
    optim = torch.optim.AdamW(bg.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100)
    _drive_optim_step(optim)  # avoid sched-before-optim warning
    lr0 = optim.param_groups[0]["lr"]
    # Step 50 → midway, cos(pi*50/100) = 0 → lr ~ 0.5 * lr0
    for _ in range(50):
        sched.step()
    lr_mid = optim.param_groups[0]["lr"]
    assert abs(lr_mid - 0.5 * lr0) < 1e-5, f"midway lr {lr_mid} != 0.5 * {lr0}"
    # Step to end → lr ~ 0
    for _ in range(50):
        sched.step()
    lr_end = optim.param_groups[0]["lr"]
    assert lr_end < 1e-6, f"lr at T_max {lr_end} should be ~0"


def test_t9_2_freeze_skips_optimizer_step():
    """If we skip optim.step() during freeze, params don't change despite
    nonzero gradients accumulating + being zeroed."""
    bg = BilateralGrid(num_camera=2)
    optim = torch.optim.AdamW(bg.parameters(), lr=1e-3)
    img = torch.rand(4, 4, 3)
    initial = bg.grids.detach().clone()
    # 5 "frozen" iterations: backward + zero_grad without step().
    for _ in range(5):
        out = bg(0, img)
        loss = out.sum()
        loss.backward()
        # NO optim.step() here — simulate freeze.
        optim.zero_grad(set_to_none=True)
    assert torch.equal(bg.grids, initial), "frozen optimizer must not change parameters"
    # Now unfreeze: step once with new gradient → params change.
    out = bg(0, img)
    loss = out.sum()
    loss.backward()
    optim.step()
    optim.zero_grad()
    assert not torch.equal(bg.grids, initial), "after unfreeze + step(), parameters must update"


def test_t9_2_scheduler_state_dict_roundtrip():
    """CosineAnnealingLR state_dict roundtrips internal state (last_epoch,
    _last_lr, _step_count, base_lrs, T_max) for resume.

    NOTE on lr propagation: PyTorch's `load_state_dict` restores sched's
    internal `_last_lr` and `last_epoch`, but does NOT push the lr into
    `optim.param_groups[i]['lr']`. The optim's lr only updates on the
    next sched.step(), which checks `optim._step_count == sched._step_count`
    to detect ordering issues. If you load a sched state that has
    sched._step_count=251 onto a fresh optim that's been stepped only 1
    time, sched.step() warns + skips the lr write. The proper resume
    pattern in our trainer is: load full ckpt → restore optim + sched +
    model state altogether so step counts align before next sched.step().
    """
    bg = BilateralGrid(num_camera=2)
    optim1 = torch.optim.AdamW(bg.parameters(), lr=1e-3, weight_decay=1e-4)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(optim1, T_max=1000)
    _drive_optim_step(optim1)
    for _ in range(250):
        sched1.step()
    state = sched1.state_dict()

    bg2 = BilateralGrid(num_camera=2)
    optim2 = torch.optim.AdamW(bg2.parameters(), lr=1e-3, weight_decay=1e-4)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(optim2, T_max=1000)
    sched2.load_state_dict(state)

    # Verify scheduler internal state matches post-load.
    for key in ("last_epoch", "_step_count", "T_max", "eta_min", "base_lrs"):
        assert sched1.state_dict()[key] == sched2.state_dict()[key], (
            f"sched state key {key!r} did not roundtrip: " f"{sched1.state_dict()[key]} vs {sched2.state_dict()[key]}"
        )
    # _last_lr is a list of floats; compare elementwise.
    for a, b in zip(sched1.state_dict()["_last_lr"], sched2.state_dict()["_last_lr"]):
        assert abs(a - b) < 1e-10, f"_last_lr mismatch: {a} vs {b}"
    # get_last_lr() must agree (it reads the same internal _last_lr).
    assert sched1.get_last_lr() == sched2.get_last_lr()
