# SPDX-License-Identifier: Apache-2.0
"""A1 road-slab background exclusion: geometry + strategy clamp (CPU-only)."""
import torch


def test_road_slab_mask_geometry():
    from threedgrut.model.road_reg import build_road_bev_height, bg_in_road_slab_mask

    # Flat road patch at z=0 over an 11x11 grid in [0,1]^2.
    xs = torch.linspace(0.0, 1.0, 11)
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    road = torch.stack([gx.flatten(), gy.flatten(), torch.zeros(121)], dim=-1)

    bev = build_road_bev_height(road, cell=0.2)
    assert bool(bev.mask.any())

    bg = torch.tensor([
        [0.5, 0.5, 0.05],   # on the road, within +/-0.15 band -> True
        [0.5, 0.5, 0.50],   # car-height above road -> False
        [5.0, 5.0, 0.00],   # outside road footprint -> False
        [0.5, 0.5, -0.10],  # slightly below surface, within band -> True
    ])
    m = bg_in_road_slab_mask(bg, bev, band_z=0.15)
    assert m.tolist() == [True, False, False, True]


def _layer(pos, dens):
    L = type("L", (), {})()
    L.positions = torch.nn.Parameter(pos)
    L.density = torch.nn.Parameter(dens)
    return L


def test_bg_road_slab_exclude_clamps_in_slab_only():
    from omegaconf import OmegaConf
    from threedgrut.strategy.layered_mcmc import LayeredMCMCStrategy

    xs = torch.linspace(0.0, 1.0, 6)
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    road = _layer(torch.stack([gx.flatten(), gy.flatten(), torch.zeros(36)], -1),
                  torch.zeros(36, 1))
    bg_pos = torch.tensor([[0.5, 0.5, 0.05], [0.5, 0.5, 0.60], [9.0, 9.0, 0.0]])
    bg = _layer(bg_pos.clone(), torch.zeros(3, 1))
    model = type("M", (), {"layers": {"background": bg, "road": road}})()

    strat = LayeredMCMCStrategy.__new__(LayeredMCMCStrategy)
    strat.model = model
    strat._road_bev = None
    strat.conf = OmegaConf.create({"strategy": {"bg_road_slab_exclude": {
        "enabled": True, "band_z": 0.15, "cell": 0.2,
        "xy_dilate": 0.0, "mode": "clamp", "clamp_value": -50.0}}})

    strat._maybe_exclude_bg_from_road_slab()
    assert bg.density[0].item() == -50.0   # in slab -> clamped
    assert bg.density[1].item() == 0.0     # above road -> untouched
    assert bg.density[2].item() == 0.0     # off footprint -> untouched

    # disabled -> strict no-op
    bg2 = _layer(bg_pos.clone(), torch.zeros(3, 1))
    model2 = type("M", (), {"layers": {"background": bg2, "road": road}})()
    strat2 = LayeredMCMCStrategy.__new__(LayeredMCMCStrategy)
    strat2.model = model2
    strat2._road_bev = None
    strat2.conf = OmegaConf.create({"strategy": {"bg_road_slab_exclude": {"enabled": False}}})
    strat2._maybe_exclude_bg_from_road_slab()
    assert bool(torch.all(bg2.density == 0.0))


def test_project_bg_road_hits_pinhole():
    """A2 core (CPU/numpy): project bg centers into a pinhole training cam, hit
    iff in-front + on-image + lands on a road-mask pixel. Pins the projection
    convention + behind-camera + off-image rejection + mask sampling."""
    import numpy as np
    from threedgrut.model.road_reg import project_bg_road_hits

    intr = {  # 640x480, fx=fy=500, principal (320,240), no distortion
        "resolution":        np.array([640, 480], dtype=np.int64),
        "principal_point":   np.array([320.0, 240.0], dtype=np.float64),
        "focal_length":      np.array([500.0, 500.0], dtype=np.float64),
        "radial_coeffs":     np.array([], dtype=np.float64),
        "tangential_coeffs": np.array([], dtype=np.float64),
    }
    c2w = np.eye(4, dtype=np.float64)  # OpenCV: cam at origin, +Z forward, +Y down
    rm = np.zeros((480, 640), dtype=np.float32)
    rm[240:, :] = 1.0  # road = lower half (v >= 240)

    pts = np.array([
        [0.0,  1.0, 10.0],   # front, v=240+50=290 -> road -> HIT
        [0.0, -1.0, 10.0],   # front, v=240-50=190 -> not road -> miss
        [0.0,  1.0, -10.0],  # behind camera -> not visible -> miss
        [100.0, 0.0, 1.0],   # front but u far off-image -> miss
    ], dtype=np.float64)

    hits = project_bg_road_hits(pts, c2w, intr, "pinhole", rm)
    assert hits.tolist() == [True, False, False, False]

    # empty input -> empty mask, no crash
    assert project_bg_road_hits(np.zeros((0, 3)), c2w, intr, "pinhole", rm).shape == (0,)
