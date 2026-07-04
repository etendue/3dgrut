# SPDX-License-Identifier: Apache-2.0
"""Calibration + regression guards for the Stage 11 LiDAR-domain depth PSNR.

Complements ``test_eval_metrics.py`` (which pins the *formula* and boundary
cases). This file pins the *calibration口径* — the normalization constant and
the depth-convention invariants — that ``mean_lidar_psnr`` depends on but that
no unit test previously locked down.

Background (C1 audit, 2026-06-01):
  * ``compute_lidar_psnr`` normalizes by ``max_depth ** 2``. Both eval paths
    — ``threedgrut/render.py`` (offline) and ``threedgrut/trainer.py`` (val
    loop) — call it WITHOUT passing ``max_depth``, so they rely on the
    signature DEFAULT (100.0, the NuRec reference normalization, v3_plan.md:426).
    The LiDAR depth *dump* / *loss*, by contrast, clip at 80m (``depth_max=80``).
    The 80-vs-100 split is deliberate (eval matches NuRec's reported number;
    no GT exists in (80,100] because the dump clips at 80) but fragile: if a
    future change passes ``max_depth=depth_max`` in ONE eval path and not the
    other, the two paths silently diverge (the CLAUDE.md §B two-path hazard).
  * Depth CONVENTION: ``pred_dist`` (tracer) is RAY-depth (‖cam ray‖, the
    tracer's ``canonicalRayDistance``). The LiDAR dump produces RAY-depth too
    (``ray_depth_from_cam_pts`` = ``‖cam_pts‖``) → the lidar_depth loss + this
    PSNR compare like-for-like. DepthAnythingV2 produces Z-depth (perpendicular
    to image plane); its depth_prior loss feeds a DELIBERATELY mismatched
    target (tolerated via inverse-depth + center-image z≈ray, but the ≈ breaks
    down at the clip's 120° wide-FOV edges).

These guards parse source statically (ast / substring) where possible, so they
run on Mac CPU with no torch/CUDA. The numeric tests use pure-torch CPU tensors.
"""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import pytest
import torch

from threedgrut.utils import eval_metrics
from threedgrut.utils.eval_metrics import compute_lidar_psnr

_REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# 1. Normalization constant (the口径)                                          #
# --------------------------------------------------------------------------- #
def test_default_max_depth_is_100():
    """Default normalization is 100m — the NuRec reference both eval paths use.

    Both render.py and trainer.py call compute_lidar_psnr WITHOUT max_depth, so
    this default IS the reported口径. Changing it silently re-scales every
    published mean_lidar_psnr and breaks comparability with the NuRec ref (25).
    """
    default = inspect.signature(compute_lidar_psnr).parameters["max_depth"].default
    assert default == pytest.approx(100.0), (
        f"compute_lidar_psnr default max_depth changed to {default}; both eval "
        "paths rely on it (they pass no max_depth). If this is intentional, "
        "re-baseline mean_lidar_psnr and update v3_plan.md:426 (NuRec ref=100m)."
    )


def test_default_matches_explicit_100():
    """Calling without max_depth == calling with 100.0 (pins the default value)."""
    pred = torch.full((1, 2, 2, 1), 10.0)
    gt = torch.full((1, 2, 2), 20.0)
    hit = torch.ones(1, 2, 2)
    assert compute_lidar_psnr(pred, gt, hit) == pytest.approx(
        compute_lidar_psnr(pred, gt, hit, max_depth=100.0), abs=1e-6
    )


def test_norm_80_vs_100_offset():
    """Documents the 80-vs-100 normalization gap: ~1.938 dB for identical MSE.

    dump/loss use 80m, eval uses 100m → for the same depth error the eval
    number reads ~1.94 dB HIGHER than an 80m-normalized one would. Pins the
    awareness so anyone unifying the constant knows the expected re-baseline.
    """
    pred = torch.full((1, 2, 2, 1), 10.0)
    gt = torch.full((1, 2, 2), 20.0)  # MSE = 100 over all valid
    hit = torch.ones(1, 2, 2)
    p100 = compute_lidar_psnr(pred, gt, hit, max_depth=100.0)
    p80 = compute_lidar_psnr(pred, gt, hit, max_depth=80.0)
    # -10log10(mse/d^2): smaller d → larger ratio → LOWER psnr.
    expected_gap = 20.0 * math.log10(100.0 / 80.0)  # ≈ 1.9382
    assert (p100 - p80) == pytest.approx(expected_gap, abs=1e-2)


# --------------------------------------------------------------------------- #
# 2. Two-path口径 consistency (CLAUDE.md §B) — source-level guard             #
# --------------------------------------------------------------------------- #
def _compute_lidar_psnr_calls(py_path: Path) -> list[ast.Call]:
    """All ast.Call nodes invoking ``compute_lidar_psnr`` in a source file."""
    tree = ast.parse(py_path.read_text())
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
            if name == "compute_lidar_psnr":
                calls.append(node)
    return calls


@pytest.mark.parametrize("rel_path", ["threedgrut/render.py", "threedgrut/trainer.py"])
def test_eval_call_sites_omit_max_depth(rel_path: str):
    """Both eval paths must call compute_lidar_psnr WITHOUT a max_depth arg.

    They share the default (100m) → the two eval paths stay口径-identical. A
    future edit that passes max_depth=depth_max (80) in one path but not the
    other would make render.py's metrics.json and the trainer val-loop TB
    scalar disagree — exactly the silent two-path divergence CLAUDE.md §B warns
    about. If you intend to switch the口径, change it in BOTH paths (or change
    the shared default) and update test_default_max_depth_is_100.
    """
    path = _REPO_ROOT / rel_path
    calls = _compute_lidar_psnr_calls(path)
    assert calls, f"no compute_lidar_psnr call found in {rel_path} (moved?)"
    for call in calls:
        kw_names = {kw.arg for kw in call.keywords}
        assert "max_depth" not in kw_names, (
            f"{rel_path}:{call.lineno} passes max_depth= to compute_lidar_psnr. "
            "Both eval paths must rely on the shared default to stay口径-"
            "consistent; change the default instead, in both paths."
        )
        # pred, gt, hit only → a 4th positional would be max_depth.
        assert len(call.args) <= 3, (
            f"{rel_path}:{call.lineno} passes a positional max_depth "
            f"({len(call.args)} positional args); keep it at the shared default."
        )


# --------------------------------------------------------------------------- #
# 3. Depth-convention invariants (C1) — source-level guard                    #
# --------------------------------------------------------------------------- #
def test_lidar_dump_stays_ray_depth():
    """LiDAR GT must remain ray-depth (‖cam_pts‖) to match pred_dist (ray-depth).

    If a refactor converts the LiDAR dump to z-depth, the lidar_depth loss +
    mean_lidar_psnr would compare ray-vs-z (mismatched) — silently degrading
    the one depth channel that is currently geometrically correct.
    """
    src = (_REPO_ROOT / "scripts" / "dump_lidar_depth_map.py").read_text()
    assert "np.linalg.norm" in src, (
        "dump_lidar_depth_map no longer uses np.linalg.norm — ray-depth "
        "(‖cam_pts‖) is the invariant that matches tracer pred_dist."
    )
    assert (
        "ray-depth" in src and "not z-depth" in src
    ), "ray-depth-vs-z-depth contract comment dropped from dump_lidar_depth_map."


def test_depthv2_dump_documents_z_vs_ray_mismatch():
    """DepthV2 dump must keep documenting its z-depth vs ray-depth mismatch.

    Pins the C1 finding in code: DepthAnythingV2 emits z-depth while pred_dist
    is ray-depth. The mismatch is deliberate (inverse-depth + center z≈ray) but
    must stay visible — at the clip's 120° FOV edges ray=z/cos(θ) diverges and
    inverse-depth scale-tolerance does NOT absorb the per-pixel cos(θ) bias.
    If a future change adds a proper z→ray conversion, update this guard.
    """
    src = (_REPO_ROOT / "scripts" / "dump_depth_priors.py").read_text()
    assert "z-depth" in src and "ray-depth" in src, (
        "dump_depth_priors dropped the z-depth/ray-depth approximation note — "
        "the deliberate convention mismatch must remain documented (C1)."
    )
