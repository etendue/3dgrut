# SPDX-License-Identifier: Apache-2.0
"""Phase 2A unit tests for opacity-reg layer exemption selection logic.

Tests the pure ``particle_layer_names_excluding`` helper that decides which
layers feed the opacity L1 reg. Loads ``layer_spec.py`` standalone via importlib
so it runs without the torch-heavy ``threedgrut`` package __init__ (Mac venv).
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_LS = pathlib.Path(__file__).resolve().parents[2] / "threedgrut" / "layers" / "layer_spec.py"
_spec = importlib.util.spec_from_file_location("_layer_spec_standalone", _LS)
_m = importlib.util.module_from_spec(_spec)
# Register before exec so @dataclass (with `from __future__ import annotations`)
# can resolve field types via sys.modules[cls.__module__] on Python 3.12+.
sys.modules["_layer_spec_standalone"] = _m
_spec.loader.exec_module(_m)
LayerSpec = _m.LayerSpec
particle_layer_names_excluding = _m.particle_layer_names_excluding


def _specs():
    # background/road/dynamic_rigids are particle layers; sky_envmap is not.
    return [
        LayerSpec("background", 0, 1000),
        LayerSpec("road", 1, 200_000),
        LayerSpec("sky_envmap", 9, 0, is_particle_layer=False),
        LayerSpec("dynamic_rigids", 2, 300_000),
    ]


def test_exclude_empty_returns_all_particle_layers_in_spec_order():
    assert particle_layer_names_excluding(_specs(), []) == ["background", "road", "dynamic_rigids"]


def test_exclude_road_drops_only_road():
    assert particle_layer_names_excluding(_specs(), ["road"]) == ["background", "dynamic_rigids"]


def test_exclude_none_treated_as_empty():
    assert particle_layer_names_excluding(_specs(), None) == ["background", "road", "dynamic_rigids"]


def test_non_particle_layer_always_excluded():
    assert "sky_envmap" not in particle_layer_names_excluding(_specs(), [])


def test_unknown_exclude_name_is_noop():
    assert particle_layer_names_excluding(_specs(), ["does_not_exist"]) == ["background", "road", "dynamic_rigids"]


def test_exclude_multiple_layers():
    assert particle_layer_names_excluding(_specs(), ["road", "background"]) == ["dynamic_rigids"]
