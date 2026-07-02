"""Viser exposure fix (2026-07-02).

Root cause: the playground engine (threedgrut_playground/engine.py) applied
only tonemap + gamma when rendering, but NOT the per-camera BilateralGrid
exposure the model was trained with (use_exposure=true). So viser showed raw
Gaussian radiance → overexposed / washed-out white, while eval (render.py,
which applies exposure to pred_rgb) looked correct.

These tests pin the BilateralGrid load-from-ckpt + apply behaviour the engine
now relies on: a full state_dict (grids + _rgb2gray_w buffer, exactly what
ckpt['exposure_state']['module'] holds) loads strict=True, and a <1 gain
darkens over-bright radiance. (Engine wiring itself is GPU + viser-visual
verified on inceptio; this is the CPU-testable core.)
"""
import torch

from threedgrut.correction import BilateralGrid


def _make_state(n: int, cam0_gain: float | None = None) -> dict:
    """Full BilateralGrid.state_dict() (grids + _rgb2gray_w) — mirrors what
    trainer saves in ckpt['exposure_state']['module']. Identity affine by
    default; optional per-channel gain on camera 0."""
    bg = BilateralGrid(num_camera=n, grid_X=1, grid_Y=1, grid_W=1)  # identity init
    if cam0_gain is not None:
        with torch.no_grad():
            bg.grids[0, 0, 0, 0, 0] = cam0_gain   # R
            bg.grids[0, 5, 0, 0, 0] = cam0_gain   # G
            bg.grids[0, 10, 0, 0, 0] = cam0_gain  # B
    return bg.state_dict()


def test_bilateral_grid_loads_from_exposure_state():
    """Engine loads BilateralGrid from ckpt['exposure_state']['module'] with
    strict=True — the full state_dict (incl. _rgb2gray_w) must round-trip."""
    state = _make_state(3)  # inc_b6a9 ckpt is 3-cam 1x1x1
    grids = state["grids"]
    N, twelve, Lz, Ly, Lx = grids.shape
    assert twelve == 12
    bg = BilateralGrid(num_camera=N, grid_X=Lx, grid_Y=Ly, grid_W=Lz)
    bg.load_state_dict(state, strict=True)  # must NOT raise (matches engine)
    bg.eval()
    assert bg.num_camera == 3


def test_exposure_darkens_overexposed_white():
    """A <1 per-channel gain must pull over-bright (raw) radiance down —
    exactly what fixes the viser washout."""
    bg = BilateralGrid(num_camera=3, grid_X=1, grid_Y=1, grid_W=1)
    bg.load_state_dict(_make_state(3, cam0_gain=0.5), strict=True)
    bg.eval()
    raw_white = torch.ones(4, 4, 3)  # overexposed
    out = bg(0, raw_white)
    assert out.shape == raw_white.shape
    assert torch.allclose(out, torch.full_like(out, 0.5), atol=1e-4), \
        "0.5 per-channel gain on white should yield 0.5 (darkened)"


def test_exposure_preserves_batched_shape_identity():
    """render_pass passes (spp, H, W, 3); identity grid must be a no-op."""
    bg = BilateralGrid(num_camera=3, grid_X=1, grid_Y=1, grid_W=1)
    bg.load_state_dict(_make_state(3), strict=True)
    bg.eval()
    x = torch.rand(4, 8, 8, 3)  # (spp, H, W, 3)
    out = bg(1, x)
    assert out.shape == x.shape
    assert torch.allclose(out, x.clamp(0.0, 1.0), atol=1e-5)
