# SPDX-License-Identifier: Apache-2.0
"""E1.1-D unit tests for plane-induced warp (lane novel-view metric core).

Pure math; CPU only; synthetic data. The warp maps a novel-view pixel ray to
the road plane and back into the original camera so the original GT / lane
mask can be sampled as pseudo-GT at the novel pose.

Geometry used throughout: world is z-up; c2w follows the threedgrut camera
convention (col0=right, col1=down, col2=forward). Equidistant fisheye FTheta
poly r_pix = f * angle is its own exact inverse pair for ray generation vs
projection, so warp consistency can be tested end-to-end without circularity.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from threedgrut.layers.dynamic_mask import _corners_to_pixels_ftheta
from threedgrut.model.plane_warp import (
    build_plane_warp,
    ftheta_project_points,
    warp_image,
)
from threedgrut.model.road_region import build_road_height_field

F_PIX = 500.0  # equidistant fisheye focal: r_pix = F_PIX * angle
H, W = 480, 640


def _pp(h: int, w: int) -> tuple[float, float]:
    """Principal point at the image center (cx, cy)."""
    return (w / 2.0, h / 2.0)


def _ftheta_params(h: int = H, w: int = W, f: float = F_PIX):
    return {
        "angle_to_pixeldist_poly": torch.tensor(
            [0.0, f, 0.0, 0.0, 0.0], dtype=torch.float32
        ),
        "principal_point": torch.tensor(_pp(h, w), dtype=torch.float32),
    }


def _rays_dir_cam(h: int = H, w: int = W, f: float = F_PIX) -> torch.Tensor:
    """Per-pixel camera-frame rays for the equidistant model (exact inverse
    of the focal-f poly): angle = r / f; dir = (sin(a)*x/r, sin(a)*y/r, cos(a))."""
    v, u = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    cx, cy = _pp(h, w)
    du = u - cx
    dv = v - cy
    r = torch.sqrt(du * du + dv * dv).clamp(min=1e-9)
    angle = r / f
    sin_a = torch.sin(angle)
    d = torch.stack([sin_a * du / r, sin_a * dv / r, torch.cos(angle)], dim=-1)
    return d  # [h, w, 3] unit rays, cam frame (right, down, forward)


def _c2w_looking_plus_x(pos_xyz) -> torch.Tensor:
    """Camera looking along world +X, world up = +Z.
    OpenCV cols: right=(0,-1,0), down=(0,0,-1), forward=(1,0,0)."""
    m = torch.eye(4, dtype=torch.float32)
    m[:3, 0] = torch.tensor([0.0, -1.0, 0.0])
    m[:3, 1] = torch.tensor([0.0, 0.0, -1.0])
    m[:3, 2] = torch.tensor([1.0, 0.0, 0.0])
    m[:3, 3] = torch.tensor(pos_xyz, dtype=torch.float32)
    return m


def _plane_texture(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Smooth scalar texture painted on the z=0 plane (smooth → bilinear
    resampling error stays small away from validity borders)."""
    return 0.5 + 0.25 * torch.sin(0.7 * x) + 0.25 * torch.cos(0.9 * y)


def test_ftheta_project_matches_dynamic_mask():
    """Point-level projection must equal the cuboid-corner FTheta math
    (pins the shared formula; BUG-1 isolation: NOT the viser projector)."""
    torch.manual_seed(0)
    pts = torch.randn(8, 3)
    pts[:, 2] = pts[:, 2].abs() + 0.5  # keep z > 0
    params = _ftheta_params()

    uv, valid = ftheta_project_points(pts, params)
    assert valid.all()

    corners = torch.cat([pts, torch.ones(8, 1)], dim=-1).unsqueeze(0)  # [1,8,4]
    u_ref, v_ref = _corners_to_pixels_ftheta(corners, params)
    assert torch.allclose(uv[:, 0], u_ref[0], atol=1e-4)
    assert torch.allclose(uv[:, 1], v_ref[0], atol=1e-4)


def test_ftheta_project_behind_camera_invalid():
    pts = torch.tensor([[0.0, 0.0, -1.0], [1.0, 1.0, 2.0]])
    _, valid = ftheta_project_points(pts, _ftheta_params())
    assert not bool(valid[0])
    assert bool(valid[1])


def test_flat_plane_warp_consistency():
    """End-to-end: paint a smooth texture on the z=0 plane, render it for the
    original camera (via its own ray-plane hits), warp that image to the novel
    camera — must match the texture evaluated at the novel camera's own
    ray-plane hits (non-circular consistency)."""
    rays = _rays_dir_cam()
    c2w_orig = _c2w_looking_plus_x([0.0, 0.0, 2.0])
    c2w_novel = c2w_orig.clone()
    c2w_novel[:3, 3] += 3.0 * c2w_orig[:3, 0]  # lateral_3m along camera right

    params = _ftheta_params()

    def plane_hits(c2w):
        d_w = torch.einsum("ij,hwj->hwi", c2w[:3, :3], rays)
        o_w = c2w[:3, 3]
        t = (0.0 - o_w[2]) / d_w[..., 2]
        hit = (d_w[..., 2] < -1e-6) & (t > 0) & (t < 120.0)
        p = o_w + t.unsqueeze(-1) * d_w
        return p, hit

    p_orig, hit_orig = plane_hits(c2w_orig)
    p_novel, hit_novel = plane_hits(c2w_novel)

    img_orig = _plane_texture(p_orig[..., 0], p_orig[..., 1]).unsqueeze(-1)
    img_orig = img_orig * hit_orig.unsqueeze(-1)  # zero where no plane hit
    expected_novel = _plane_texture(p_novel[..., 0], p_novel[..., 1])

    grid, valid = build_plane_warp(
        rays, c2w_novel, c2w_orig, params,
        height_field=None, z0_fallback=0.0,
    )
    warped = warp_image(img_orig, grid, valid)[..., 0]

    # Compare only where the warp says valid AND the source pixel itself was
    # a clean plane hit well inside the original image (sampling near the
    # validity border mixes in zeroed pixels). Restrict to the near field
    # (< 25 m): at grazing far range one pixel spans many meters of ground,
    # so the 9 m-period texture aliases — that is a sampling-density limit
    # of the test texture, not a warp error.
    eroded = valid & hit_novel
    o_novel = c2w_novel[:3, 3]
    t_novel = (p_novel - o_novel).norm(dim=-1)
    near_field = t_novel < 25.0
    # erode by ignoring a 12px border of the valid region via grid bounds
    gx, gy = grid[0, ..., 0], grid[0, ..., 1]
    margin_x = 2.0 * 12.0 / (W - 1)
    margin_y = 2.0 * 12.0 / (H - 1)
    inner = (gx > -1 + margin_x) & (gx < 1 - margin_x) \
        & (gy > -1 + margin_y) & (gy < 1 - margin_y)
    sel = eroded & inner & near_field
    assert int(sel.sum()) > 5000, "test geometry must keep a usable region"
    err = (warped[sel] - expected_novel[sel]).abs()
    assert float(err.max()) < 3e-2, f"max warp error {float(err.max()):.4f}"


def test_identity_warp_grid_normalization():
    """novel == orig camera → uv equals the pixel's own coordinates →
    align_corners=True normalization maps (0,0)→(-1,-1), (W-1,H-1)→(1,1)."""
    rays = _rays_dir_cam(h=64, w=96, f=40.0)
    c2w = _c2w_looking_plus_x([0.0, 0.0, 2.0])
    grid, valid = build_plane_warp(
        rays, c2w, c2w, _ftheta_params(64, 96, f=40.0), height_field=None, z0_fallback=0.0,
    )
    gx, gy = grid[0, ..., 0], grid[0, ..., 1]
    # pick a pixel guaranteed to hit the plane: bottom-center looks downward
    vv, uu = 60, 48
    assert bool(valid[vv, uu])
    exp_gx = 2.0 * uu / (96 - 1) - 1.0
    exp_gy = 2.0 * vv / (64 - 1) - 1.0
    assert abs(float(gx[vv, uu]) - exp_gx) < 1e-3
    assert abs(float(gy[vv, uu]) - exp_gy) < 1e-3


def test_sky_rays_invalid():
    """Rays with non-negative world d_z (at/above horizon) must be invalid."""
    rays = _rays_dir_cam(h=64, w=96, f=40.0)
    c2w = _c2w_looking_plus_x([0.0, 0.0, 2.0])
    _, valid = build_plane_warp(
        rays, c2w, c2w, _ftheta_params(64, 96, f=40.0), height_field=None, z0_fallback=0.0,
    )
    d_w = torch.einsum("ij,hwj->hwi", c2w[:3, :3], rays)
    sky = d_w[..., 2] >= 0
    assert sky.any(), "test FOV must include some at/above-horizon rays"
    assert not valid[sky].any()


def test_height_field_gates_valid_and_adjusts_t():
    """With a real height field: hits outside the occupied BEV region are
    invalid; inside, intersection uses the cell ground height."""
    rays = _rays_dir_cam(h=64, w=96, f=40.0)
    c2w = _c2w_looking_plus_x([0.0, 0.0, 2.0])
    # occupied road patch only for x ∈ [0, 25] (y ∈ [-15, 15]), at z = 0
    xs = torch.arange(0.0, 25.0, 0.5)
    ys = torch.arange(-15.0, 15.0, 0.5)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    road_pts = torch.stack(
        [gx.reshape(-1), gy.reshape(-1), torch.zeros(gx.numel())], dim=-1,
    )
    hf = build_road_height_field(road_pts, cell_size=1.0)

    grid, valid = build_plane_warp(
        rays, c2w, c2w, _ftheta_params(64, 96, f=40.0), height_field=hf,
    )
    d_w = torch.einsum("ij,hwj->hwi", c2w[:3, :3], rays)
    o_w = c2w[:3, 3]
    t_flat = (0.0 - o_w[2]) / d_w[..., 2]
    x_hit = o_w[0] + t_flat * d_w[..., 0]
    down = d_w[..., 2] < -1e-6
    far = down & (t_flat > 0) & (x_hit > 30.0) & (t_flat < 120.0)
    near = down & (t_flat > 0) & (x_hit > 2.0) & (x_hit < 20.0) \
        & (o_w[1] + t_flat * d_w[..., 1] > -12.0) \
        & (o_w[1] + t_flat * d_w[..., 1] < 12.0)
    assert near.any() and far.any()
    assert valid[near].all(), "hits inside occupied region must be valid"
    assert not valid[far].any(), "hits beyond occupied region must be invalid"


def test_warp_image_nearest_preserves_labels():
    """Label maps must be sampled with nearest (no label mixing)."""
    img = torch.zeros(8, 8, 1)
    img[:, 4:] = 7.0
    # identity grid
    v, u = torch.meshgrid(
        torch.arange(8, dtype=torch.float32),
        torch.arange(8, dtype=torch.float32),
        indexing="ij",
    )
    gx = 2.0 * u / 7.0 - 1.0
    gy = 2.0 * v / 7.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    valid = torch.ones(8, 8, dtype=torch.bool)
    out = warp_image(img, grid, valid, mode="nearest")
    assert set(out.unique().tolist()) <= {0.0, 7.0}
    assert torch.equal(out, img)


def test_warp_image_zeroes_invalid():
    img = torch.ones(8, 8, 3)
    grid = torch.zeros(1, 8, 8, 2)  # all sample center
    valid = torch.zeros(8, 8, dtype=torch.bool)
    valid[2, 3] = True
    out = warp_image(img, grid, valid)
    assert float(out[2, 3].sum()) == 3.0
    out_masked = out.clone()
    out_masked[2, 3] = 0
    assert float(out_masked.abs().sum()) == 0.0
