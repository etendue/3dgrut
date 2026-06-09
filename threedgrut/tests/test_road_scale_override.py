# SPDX-License-Identifier: Apache-2.0
"""P3.1 T4：layers.overrides.road.anisotropy_ratio_max 真传到 LayerSpec → clamp 放宽。

``clamp_layer_scales`` / ``specs_from_config`` 已存在（V3-R1.2）；这些是**特征
测试**（锚定 P3.1 preset 的假设成立，非新功能 TDD）：放宽 ``anisotropy_ratio_max``
会让 clamp 抬高最小轴的力度变小，允许更细长（更大 max/min）的 road 高斯拟合
车道线条纹——这正是 P3.1-A 第二变量「放宽几何」依赖的行为。
"""
from __future__ import annotations

import torch
from omegaconf import OmegaConf

from threedgrut.layers.registry import specs_from_config
from threedgrut.model.road_reg import clamp_layer_scales


def _road_spec(aniso):
    """road LayerSpec with anisotropy_ratio_max overridden via Hydra-style conf."""
    conf = OmegaConf.create({
        "layers": {
            "enabled": ["road"],
            "overrides": {"road": {"anisotropy_ratio_max": aniso}},
        }
    })
    return [s for s in specs_from_config(conf) if s.name == "road"][0]


def test_override_reaches_spec():
    """conf override 真改到 LayerSpec 字段（preset 放宽生效的前提）。"""
    spec = _road_spec(30.0)
    assert spec.anisotropy_ratio_max == 30.0
    # 默认 road 几何上限仍在（override 只动 anisotropy）
    assert spec.scale_xy_max == 0.3
    assert spec.scale_z_max == 0.05


def test_relaxed_anisotropy_allows_thinner_min_axis():
    """放宽 ratio_max → clamp 抬最小轴力度变小 → 允许更细长（更大 max/min）road 高斯。"""
    needle = torch.log(torch.tensor([[0.30, 0.02, 0.001]]))  # raw 各向异性 300
    tight = torch.exp(clamp_layer_scales(needle.clone(), _road_spec(8.0)))[0]
    loose = torch.exp(clamp_layer_scales(needle.clone(), _road_spec(30.0)))[0]
    tight_ratio = (tight.max() / tight.min()).item()
    loose_ratio = (loose.max() / loose.min()).item()
    assert loose_ratio > tight_ratio                # 放宽 → 允许更大各向异性
    assert loose.min().item() < tight.min().item()  # 最小轴更细（更细长的条）


def test_default_road_clamps_to_ratio_8():
    """默认 road（ratio 8）把 needle clamp 到各向异性 ≈ 8（V3-R1.2 现状锚点）。"""
    needle = torch.log(torch.tensor([[0.30, 0.02, 0.001]]))
    out = torch.exp(clamp_layer_scales(needle.clone(), _road_spec(8.0)))[0]
    ratio = (out.max() / out.min()).item()
    assert abs(ratio - 8.0) < 1e-4
