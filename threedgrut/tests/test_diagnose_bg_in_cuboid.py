# SPDX-License-Identifier: Apache-2.0
"""Unit tests for scripts/diagnose_bg_in_cuboid.py — pure containment helpers.

The script's ckpt-loading path needs a real v2 ckpt + CUDA bookkeeping, but
the core algorithm (per-track cuboid containment in world frame, plus the
local-frame "outside own cuboid" leak detector) is a pair of pure tensor
functions we can exercise on CPU with hand-built inputs.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def diag_module():
    """Load scripts/diagnose_bg_in_cuboid.py as a module without invoking main()."""
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "diagnose_bg_in_cuboid.py"
    assert script_path.exists(), f"missing {script_path}"
    spec = importlib.util.spec_from_file_location("_diag_bg_in_cuboid", script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _identity_pose() -> torch.Tensor:
    return torch.eye(4, dtype=torch.float32)


def _translation_pose(tx: float, ty: float, tz: float) -> torch.Tensor:
    pose = torch.eye(4, dtype=torch.float32)
    pose[0, 3] = tx
    pose[1, 3] = ty
    pose[2, 3] = tz
    return pose


# -----------------------------------------------------------------------------
# _select_sample_frames
# -----------------------------------------------------------------------------

def test_select_sample_frames_empty(diag_module):
    active = torch.zeros(10, dtype=torch.bool)
    out = diag_module._select_sample_frames(active, max_frames=5)
    assert out.numel() == 0
    assert out.dtype == torch.int64


def test_select_sample_frames_fewer_than_cap(diag_module):
    active = torch.zeros(20, dtype=torch.bool)
    active[[2, 5, 11]] = True
    out = diag_module._select_sample_frames(active, max_frames=10)
    assert out.tolist() == [2, 5, 11]


def test_select_sample_frames_evenly_spaced(diag_module):
    active = torch.zeros(100, dtype=torch.bool)
    active[10:30] = True  # 20 active frames at indices 10..29
    out = diag_module._select_sample_frames(active, max_frames=5)
    # picks first, last, plus 3 interior, evenly spaced over [0, 19]
    assert out.numel() == 5
    assert int(out[0].item()) == 10  # first active
    assert int(out[-1].item()) == 29  # last active
    assert out.dtype == torch.int64


# -----------------------------------------------------------------------------
# count_world_positions_in_any_active_cuboid
# -----------------------------------------------------------------------------

def test_count_inside_identity_pose_unit_cuboid(diag_module):
    # Cuboid at world origin, size [2, 2, 2] → half-extent 1
    tracks = {
        "t0": {
            "poses": _identity_pose().unsqueeze(0),       # [1, 4, 4]
            "active": torch.tensor([True]),
            "size": torch.tensor([2.0, 2.0, 2.0]),
        }
    }
    positions = torch.tensor([
        [0.0, 0.0, 0.0],     # inside (center)
        [0.5, 0.5, 0.5],     # inside
        [-0.99, 0.99, 0.0],  # inside (edge)
        [1.01, 0.0, 0.0],    # outside (just past x boundary)
        [5.0, 0.0, 0.0],     # outside (far)
    ])
    mask, records = diag_module.count_world_positions_in_any_active_cuboid(
        positions, tracks, max_frames_per_track=5,
    )
    assert mask.tolist() == [True, True, True, False, False]
    assert len(records) == 1
    rec = records[0]
    assert rec["track_id"] == "t0"
    assert rec["n_inside_any_sampled_frame"] == 3
    assert rec["n_active_frames"] == 1
    assert rec["n_frames_sampled"] == 1


def test_count_inside_translated_cuboid(diag_module):
    # Cuboid centered at (10, 0, 0), size [2, 2, 2]
    tracks = {
        "t0": {
            "poses": _translation_pose(10.0, 0.0, 0.0).unsqueeze(0),
            "active": torch.tensor([True]),
            "size": torch.tensor([2.0, 2.0, 2.0]),
        }
    }
    positions = torch.tensor([
        [0.0, 0.0, 0.0],   # outside (origin is far from cuboid)
        [10.0, 0.5, 0.0],  # inside
        [9.5, 0.0, 0.0],   # inside (within half-extent)
        [11.5, 0.0, 0.0],  # outside (past x boundary at 11)
    ])
    mask, records = diag_module.count_world_positions_in_any_active_cuboid(
        positions, tracks, max_frames_per_track=5,
    )
    assert mask.tolist() == [False, True, True, False]
    assert records[0]["n_inside_any_sampled_frame"] == 2


def test_count_inside_two_tracks_or_aggregation(diag_module):
    # Two tracks at different locations; any_inside is the OR across tracks.
    tracks = {
        "t0": {
            "poses": _translation_pose(0.0, 0.0, 0.0).unsqueeze(0),
            "active": torch.tensor([True]),
            "size": torch.tensor([2.0, 2.0, 2.0]),
        },
        "t1": {
            "poses": _translation_pose(10.0, 0.0, 0.0).unsqueeze(0),
            "active": torch.tensor([True]),
            "size": torch.tensor([2.0, 2.0, 2.0]),
        },
    }
    positions = torch.tensor([
        [0.5, 0.0, 0.0],  # inside t0
        [10.5, 0.0, 0.0],  # inside t1
        [5.0, 0.0, 0.0],  # outside both
    ])
    mask, records = diag_module.count_world_positions_in_any_active_cuboid(
        positions, tracks, max_frames_per_track=5,
    )
    assert mask.tolist() == [True, True, False]
    by_tid = {r["track_id"]: r for r in records}
    assert by_tid["t0"]["n_inside_any_sampled_frame"] == 1
    assert by_tid["t1"]["n_inside_any_sampled_frame"] == 1


def test_count_inside_inactive_track_contributes_nothing(diag_module):
    tracks = {
        "t0": {
            "poses": _identity_pose().unsqueeze(0),
            "active": torch.tensor([False]),  # no active frames
            "size": torch.tensor([2.0, 2.0, 2.0]),
        }
    }
    positions = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    mask, records = diag_module.count_world_positions_in_any_active_cuboid(
        positions, tracks, max_frames_per_track=5,
    )
    assert mask.tolist() == [False, False]
    assert records[0]["n_inside_any_sampled_frame"] == 0
    assert records[0]["n_frames_sampled"] == 0


def test_count_inside_moving_cuboid_across_frames(diag_module):
    # Track has 3 active frames; cuboid moves x = 0 → 5 → 10.
    poses = torch.stack([
        _translation_pose(0.0, 0.0, 0.0),
        _translation_pose(5.0, 0.0, 0.0),
        _translation_pose(10.0, 0.0, 0.0),
    ])
    tracks = {
        "t0": {
            "poses": poses,
            "active": torch.tensor([True, True, True]),
            "size": torch.tensor([2.0, 2.0, 2.0]),
        }
    }
    positions = torch.tensor([
        [0.5, 0.0, 0.0],   # inside at frame 0
        [5.5, 0.0, 0.0],   # inside at frame 1
        [9.5, 0.0, 0.0],   # inside at frame 2
        [3.0, 0.0, 0.0],   # outside all three (between cuboids 0 and 1)
    ])
    mask, records = diag_module.count_world_positions_in_any_active_cuboid(
        positions, tracks, max_frames_per_track=5,
    )
    assert mask.tolist() == [True, True, True, False]
    assert records[0]["n_inside_any_sampled_frame"] == 3
    assert records[0]["n_frames_sampled"] == 3


def test_count_inside_empty_positions(diag_module):
    tracks = {
        "t0": {
            "poses": _identity_pose().unsqueeze(0),
            "active": torch.tensor([True]),
            "size": torch.tensor([2.0, 2.0, 2.0]),
        }
    }
    empty = torch.zeros(0, 3)
    mask, records = diag_module.count_world_positions_in_any_active_cuboid(
        empty, tracks, max_frames_per_track=5,
    )
    assert mask.shape == (0,)
    assert mask.dtype == torch.bool
    # records still populated for each track, but n_inside=0
    assert records[0]["n_inside_any_sampled_frame"] == 0


# -----------------------------------------------------------------------------
# count_local_positions_outside_own_cuboid
# -----------------------------------------------------------------------------

def test_outside_own_cuboid_all_inside(diag_module):
    # 4 particles in local frame, all within size/2; no leaks.
    tracks = {
        "t0": {
            "poses": _identity_pose().unsqueeze(0),
            "active": torch.tensor([True]),
            "size": torch.tensor([2.0, 2.0, 2.0]),
        }
    }
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.99, 0.0, 0.0],
        [-0.5, 0.5, -0.5],
        [0.0, -0.99, 0.99],
    ])
    track_ids = torch.zeros(4, dtype=torch.long)
    n_out = diag_module.count_local_positions_outside_own_cuboid(
        positions, track_ids, ["t0"], tracks,
    )
    assert n_out == 0


def test_outside_own_cuboid_some_leaks(diag_module):
    tracks = {
        "t0": {
            "poses": _identity_pose().unsqueeze(0),
            "active": torch.tensor([True]),
            "size": torch.tensor([2.0, 2.0, 2.0]),
        }
    }
    positions = torch.tensor([
        [0.0, 0.0, 0.0],     # inside
        [1.01, 0.0, 0.0],    # outside x
        [0.0, -1.5, 0.0],    # outside y
        [0.5, 0.5, 0.5],     # inside
    ])
    track_ids = torch.zeros(4, dtype=torch.long)
    n_out = diag_module.count_local_positions_outside_own_cuboid(
        positions, track_ids, ["t0"], tracks,
    )
    assert n_out == 2


def test_outside_own_cuboid_per_track_routing(diag_module):
    # Two tracks of different sizes; particles owned by their own track.
    tracks = {
        "t0": {
            "poses": _identity_pose().unsqueeze(0),
            "active": torch.tensor([True]),
            "size": torch.tensor([2.0, 2.0, 2.0]),   # half = 1.0
        },
        "t1": {
            "poses": _identity_pose().unsqueeze(0),
            "active": torch.tensor([True]),
            "size": torch.tensor([10.0, 10.0, 10.0]),  # half = 5.0
        },
    }
    keys_sorted = ["t0", "t1"]  # name_to_id: t0→0, t1→1
    positions = torch.tensor([
        [0.5, 0.0, 0.0],   # owner t0, inside
        [3.0, 0.0, 0.0],   # owner t0, outside (would be inside t1's bigger box)
        [0.5, 0.0, 0.0],   # owner t1, inside
        [4.99, 0.0, 0.0],  # owner t1, inside
    ])
    track_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    n_out = diag_module.count_local_positions_outside_own_cuboid(
        positions, track_ids, keys_sorted, tracks,
    )
    # Only one leak: index 1 has owner t0 but at x=3.0 (outside t0's half=1.0).
    assert n_out == 1


def test_outside_own_cuboid_empty(diag_module):
    tracks = {
        "t0": {
            "poses": _identity_pose().unsqueeze(0),
            "active": torch.tensor([True]),
            "size": torch.tensor([2.0, 2.0, 2.0]),
        }
    }
    n_out = diag_module.count_local_positions_outside_own_cuboid(
        torch.zeros(0, 3), torch.zeros(0, dtype=torch.long), ["t0"], tracks,
    )
    assert n_out == 0
