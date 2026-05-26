# SPDX-License-Identifier: Apache-2.0
"""Unit tests for scripts/diagnose_dyn_per_cuboid.py pure helpers.

Mirrors the test approach for the bg-side diagnostic
(test_diagnose_bg_in_cuboid.py): load the script as a module without invoking
``main()``, then exercise the pure tensor functions with hand-built inputs.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def diag_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "diagnose_dyn_per_cuboid.py"
    assert script_path.exists(), f"missing {script_path}"
    spec = importlib.util.spec_from_file_location("_diag_dyn", script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# -----------------------------------------------------------------------------
# per_cuboid_counts_owner_aware
# -----------------------------------------------------------------------------

def test_owner_aware_two_tracks_basic_count(diag_module):
    """4 particles, 2 owned by t0 and 2 by t1, all inside their cuboid."""
    positions_local = torch.tensor([
        [0.1, 0.0, 0.0],   # owner t0
        [-0.2, 0.0, 0.0],  # owner t0
        [0.5, 0.0, 0.0],   # owner t1
        [-0.5, 0.0, 0.0],  # owner t1
    ])
    densities_raw = torch.zeros(4)  # sigmoid(0) = 0.5 → all alive
    track_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    sizes = {"t0": torch.tensor([2.0, 2.0, 2.0]),
             "t1": torch.tensor([2.0, 2.0, 2.0])}
    out = diag_module.per_cuboid_counts_owner_aware(
        positions_local, densities_raw, track_ids,
        ["t0", "t1"], sizes,
    )
    by_tid = {r["track_id"]: r for r in out}
    assert by_tid["t0"]["n_particles"] == 2
    assert by_tid["t0"]["alive"] == 2
    assert by_tid["t0"]["dead"] == 0
    assert by_tid["t0"]["out_of_cuboid"] == 0
    assert by_tid["t1"]["n_particles"] == 2


def test_owner_aware_detects_dead_particles(diag_module):
    """1 alive (density=0), 1 dead (density=-10 → sigmoid ≈ 4.5e-5 < 5e-3)."""
    positions_local = torch.tensor([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    densities_raw = torch.tensor([0.0, -10.0])
    track_ids = torch.tensor([0, 0], dtype=torch.int64)
    sizes = {"t0": torch.tensor([2.0, 2.0, 2.0])}
    out = diag_module.per_cuboid_counts_owner_aware(
        positions_local, densities_raw, track_ids,
        ["t0"], sizes, opacity_threshold=0.005,
    )
    assert out[0]["n_particles"] == 2
    assert out[0]["alive"] == 1
    assert out[0]["dead"] == 1
    assert out[0]["alive_pct"] == 50.0


def test_owner_aware_detects_out_of_cuboid_particles(diag_module):
    """3 particles owned by t0 with half=1.0: 2 inside, 1 outside x by 0.5."""
    positions_local = torch.tensor([
        [0.5, 0.0, 0.0],   # inside
        [-0.5, 0.0, 0.0],  # inside
        [1.5, 0.0, 0.0],   # outside x by 0.5
    ])
    densities_raw = torch.zeros(3)
    track_ids = torch.zeros(3, dtype=torch.int64)
    sizes = {"t0": torch.tensor([2.0, 2.0, 2.0])}  # half=1.0
    out = diag_module.per_cuboid_counts_owner_aware(
        positions_local, densities_raw, track_ids, ["t0"], sizes,
    )
    assert out[0]["out_of_cuboid"] == 1
    assert math.isclose(out[0]["outlier_max_dist"], 0.5, abs_tol=1e-5)


def test_owner_aware_track_with_zero_particles_reported(diag_module):
    """Track t1 has no particles assigned — report 0 across the board."""
    positions_local = torch.tensor([[0.1, 0.0, 0.0]])
    densities_raw = torch.zeros(1)
    track_ids = torch.tensor([0], dtype=torch.int64)
    sizes = {
        "t0": torch.tensor([2.0, 2.0, 2.0]),
        "t1": torch.tensor([3.0, 3.0, 3.0]),
    }
    out = diag_module.per_cuboid_counts_owner_aware(
        positions_local, densities_raw, track_ids,
        ["t0", "t1"], sizes,
    )
    by_tid = {r["track_id"]: r for r in out}
    assert by_tid["t1"]["n_particles"] == 0
    assert by_tid["t1"]["alive"] == 0
    assert by_tid["t1"]["dead"] == 0
    assert by_tid["t1"]["alive_pct"] == 0.0
    assert by_tid["t1"]["outlier_max_dist"] is None


def test_owner_aware_density_2d_shape_handled(diag_module):
    """MoG.density has shape [N, 1]; helper should view(-1) transparently."""
    positions_local = torch.tensor([[0.1, 0.0, 0.0]])
    densities_raw = torch.zeros(1, 1)
    track_ids = torch.tensor([0], dtype=torch.int64)
    sizes = {"t0": torch.tensor([2.0, 2.0, 2.0])}
    out = diag_module.per_cuboid_counts_owner_aware(
        positions_local, densities_raw, track_ids, ["t0"], sizes,
    )
    assert out[0]["alive"] == 1


def test_owner_aware_empty_positions(diag_module):
    out = diag_module.per_cuboid_counts_owner_aware(
        torch.zeros(0, 3), torch.zeros(0),
        torch.zeros(0, dtype=torch.int64),
        ["t0"], {"t0": torch.tensor([2.0, 2.0, 2.0])},
    )
    assert out == []


# -----------------------------------------------------------------------------
# per_cuboid_counts_world_fallback
# -----------------------------------------------------------------------------

def test_world_fallback_reports_global_alive_count(diag_module):
    """Path (b) is degenerate per-owner but reports global alive across tracks."""
    positions_local = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    densities_raw = torch.tensor([0.0, -10.0])  # 1 alive, 1 dead
    tracks = {
        "t0": {
            "poses": torch.eye(4).unsqueeze(0),
            "active": torch.tensor([True]),
            "size": torch.tensor([2.0, 2.0, 2.0]),
        }
    }
    out = diag_module.per_cuboid_counts_world_fallback(
        positions_local, densities_raw, tracks, frame_idx=0,
    )
    assert len(out) == 1
    assert out[0]["track_id"] == "t0"
    assert out[0]["n_particles"] == -1  # owner unknown
    assert out[0]["alive_global"] == 1


def test_world_fallback_empty(diag_module):
    out = diag_module.per_cuboid_counts_world_fallback(
        torch.zeros(0, 3), torch.zeros(0), {}, frame_idx=0,
    )
    assert out == []


# -----------------------------------------------------------------------------
# Sanity: script-level imports work
# -----------------------------------------------------------------------------

def test_script_module_imports_diagnose_function(diag_module):
    """The script's main entry point ``diagnose`` should be importable as a
    callable so other test harnesses can wire it up against fake ckpts."""
    assert callable(diag_module.diagnose)
    assert callable(diag_module.main)
