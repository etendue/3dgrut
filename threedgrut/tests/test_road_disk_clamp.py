# SPDX-License-Identifier: Apache-2.0
"""E3.2.5② — road z-scale 1mm floor characterization tests.

E3.2.5 hard-degrades the road into a ~zero-thickness horizontal disc by
dropping the per-step MCMC scale clamp upper bound ``scale_z_max`` from
0.05 (5cm, V3-R1.2) to 0.001 (1mm). That is a pure config change carried by
the roaddisk preset — ``clamp_layer_scales`` already enforces ``scale_z_max``.

These tests pin the clamp's behaviour at the new 1mm value, and in
particular DECIDE whether the preset must disable ``anisotropy_ratio_max``:
the anisotropy step raises the smallest axis to ``s_max/ratio`` (= 0.3/8 =
37.5mm for a road disc), which would re-thicken the disc — UNLESS the final
re-apply of the hard caps (road_reg.py:82) clamps z back to 1mm. If
``test_clamp_anisotropy_vs_1mm_zfloor`` passes, registry anisotropy=8 is safe
to keep; if it fails, the preset must set ``anisotropy_ratio_max=null``.

Pure CPU (clamp_layer_scales is torch-only, no CUDA / Trainer).
"""

from __future__ import annotations

import math

import torch

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.model.road_reg import clamp_layer_scales


def test_clamp_z_floor_1mm():
    """z-scale above 1mm is clamped down to the 1mm floor."""
    spec = LayerSpec(name="road", layer_id=1, max_n_particles=10, scale_z_max=0.001)
    scale_log = torch.full((20, 3), math.log(0.05))  # 5cm on every axis
    out = clamp_layer_scales(scale_log, spec)
    assert out[:, 2].exp().max().item() <= 0.001 + 1e-9, f"z not clamped to 1mm: max={out[:, 2].exp().max().item()}"


def test_clamp_1mm_keeps_xy():
    """1mm z floor does not shrink the in-plane (XY) extent."""
    spec = LayerSpec(name="road", layer_id=1, max_n_particles=10, scale_xy_max=0.3, scale_z_max=0.001)
    scale_log = torch.tensor([[math.log(0.2), math.log(0.2), math.log(0.05)]] * 10)
    out = clamp_layer_scales(scale_log, spec)
    assert out[:, 2].exp().max().item() <= 0.001 + 1e-9  # z → 1mm
    # XY 0.2m < 0.3m cap → untouched
    assert torch.allclose(out[:, 0].exp(), torch.full((10,), 0.2), atol=1e-6)
    assert torch.allclose(out[:, 1].exp(), torch.full((10,), 0.2), atol=1e-6)


def test_clamp_anisotropy_vs_1mm_zfloor():
    """E3.2.5② DECISION: z=1mm + xy=0.3 + anisotropy=8 → z still hard-capped 1mm.

    The anisotropy step lifts the min axis to xy_max/8 = 37.5mm, but the final
    re-apply of hard caps (road_reg.py:82) clamps z back to 1mm (caps win over
    ratio). Proves keeping registry ``anisotropy_ratio_max=8`` does NOT
    re-thicken the disc → preset need not disable anisotropy.
    """
    spec = LayerSpec(
        name="road",
        layer_id=1,
        max_n_particles=10,
        scale_xy_max=0.3,
        scale_z_max=0.001,
        anisotropy_ratio_max=8.0,
    )
    scale_log = torch.tensor([[math.log(0.3), math.log(0.3), math.log(0.05)]] * 10)
    out = clamp_layer_scales(scale_log, spec)
    z_mm = out[:, 2].exp().max().item() * 1000.0
    assert z_mm <= 1.0 + 1e-6, (
        f"1mm z floor breached by anisotropy re-thickening: {z_mm}mm " f"(preset must set anisotropy_ratio_max=null)"
    )
