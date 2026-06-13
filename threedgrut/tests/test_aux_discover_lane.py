# SPDX-License-Identifier: Apache-2.0
"""discover_aux_path 对 lane 产物的 glob 行为（纯 pathlib，Mac 可测）。"""
from __future__ import annotations

import pytest

from threedgrut.datasets.aux_readers import discover_aux_path


def test_discover_lane_finds_single(tmp_path):
    (tmp_path / "clip.aux.lane.zarr.itar").touch()
    (tmp_path / "clip.aux.sseg.zarr.itar").touch()  # 不应被 lane 命中
    p = discover_aux_path(tmp_path, "lane")
    assert p is not None
    assert p.name == "clip.aux.lane.zarr.itar"


def test_discover_lane_absent_returns_none(tmp_path):
    (tmp_path / "clip.aux.sseg.zarr.itar").touch()
    assert discover_aux_path(tmp_path, "lane") is None


def test_discover_lane_ambiguous_raises(tmp_path):
    (tmp_path / "a.aux.lane.zarr.itar").touch()
    (tmp_path / "b.aux.lane.zarr.itar").touch()
    with pytest.raises(ValueError):
        discover_aux_path(tmp_path, "lane")
