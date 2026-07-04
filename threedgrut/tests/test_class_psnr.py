# SPDX-License-Identifier: Apache-2.0
"""T8/B3 Phase E.5 — class_psnr (per-cuboid PSNR) tool tests.

Pure-tensor function; mockable on Mac without renderer or NCore.
"""

from __future__ import annotations

import math

import pytest
import torch

from threedgrut.model.class_psnr import (
    collect_active_tracks_for_frame,
    compute_class_psnr,
    compute_psnr_in_mask,
)

# -----------------------------------------------------------------------------
# compute_psnr_in_mask
# -----------------------------------------------------------------------------


def test_psnr_in_mask_perfect_prediction_is_infinity():
    H, W = 64, 64
    rgb = torch.rand(H, W, 3)
    mask = torch.ones(H, W)
    out = compute_psnr_in_mask(rgb, rgb, mask)
    assert out == float("inf")


def test_psnr_in_mask_known_mse_matches_formula():
    """MSE = 0.01 → PSNR = -10·log10(0.01) = 20 dB."""
    H, W = 64, 64
    rgb_gt = torch.zeros(H, W, 3)
    rgb_pred = torch.full((H, W, 3), 0.1)  # diff² = 0.01 everywhere
    mask = torch.ones(H, W)
    out = compute_psnr_in_mask(rgb_pred, rgb_gt, mask)
    assert math.isclose(out, 20.0, abs_tol=1e-3)


def test_psnr_in_mask_only_inside_mask_counted():
    """Mask covers only top half; only those pixels contribute to MSE."""
    H, W = 64, 64
    rgb_gt = torch.zeros(H, W, 3)
    rgb_pred = torch.zeros(H, W, 3)
    rgb_pred[:32] = 0.1  # diff² = 0.01 in top half
    rgb_pred[32:] = 1.0  # would dominate if NOT masked
    mask = torch.zeros(H, W)
    mask[:32] = 1.0
    out = compute_psnr_in_mask(rgb_pred, rgb_gt, mask)
    # Only top half MSE counts: 0.01 → 20 dB
    assert math.isclose(out, 20.0, abs_tol=1e-3)


def test_psnr_in_mask_too_few_pixels_returns_none():
    H, W = 64, 64
    rgb = torch.rand(H, W, 3)
    mask = torch.zeros(H, W)
    mask[0, 0] = 1.0  # 1 pixel only
    out = compute_psnr_in_mask(rgb, rgb, mask, min_pixels=50)
    assert out is None


def test_psnr_in_mask_bool_mask_accepted():
    H, W = 64, 64
    rgb_gt = torch.zeros(H, W, 3)
    rgb_pred = torch.full((H, W, 3), 0.1)
    mask = torch.ones(H, W, dtype=torch.bool)
    out = compute_psnr_in_mask(rgb_pred, rgb_gt, mask)
    assert math.isclose(out, 20.0, abs_tol=1e-3)


# -----------------------------------------------------------------------------
# compute_class_psnr
# -----------------------------------------------------------------------------


def _pinhole_K(fx=300.0, fy=300.0, cx=128.0, cy=128.0):
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _identity_pose():
    return torch.eye(4)


def test_compute_class_psnr_empty_active_tracks_returns_no_psnr():
    rgb_pred = torch.zeros(1, 64, 64, 3)
    rgb_gt = torch.zeros(1, 64, 64, 3)
    out = compute_class_psnr(
        rgb_pred,
        rgb_gt,
        valid_mask=None,
        active_tracks=[],
        T_world2cam=_identity_pose(),
        H=64,
        W=64,
        K=_pinhole_K(),
    )
    assert out["per_track"] == []
    assert out["mean"] is None
    assert out["by_class"] == {}
    assert out["n_tracks"] == 0


def test_compute_class_psnr_one_active_track_known_psnr():
    """1m cuboid at z=5m with cam at origin → AABB ~ 60×60 px (300*0.5/5=30 half).
    Set pred = 0.1 everywhere in image, gt = 0 → mse = 0.01 → PSNR = 20 dB
    inside the cuboid mask too."""
    H, W = 256, 256
    rgb_gt = torch.zeros(1, H, W, 3)
    rgb_pred = torch.full((1, H, W, 3), 0.1)
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([0.0, 0.0, 5.0])
    size = torch.tensor([1.0, 1.0, 1.0])

    out = compute_class_psnr(
        rgb_pred,
        rgb_gt,
        valid_mask=None,
        active_tracks=[{"id": "t0", "class": "automobile", "pose": pose, "size": size}],
        T_world2cam=_identity_pose(),
        H=H,
        W=W,
        K=_pinhole_K(),
    )
    assert len(out["per_track"]) == 1
    r = out["per_track"][0]
    assert r["track_id"] == "t0"
    assert r["class"] == "automobile"
    assert r["n_pixels"] > 100  # at least some cuboid area
    assert math.isclose(r["psnr"], 20.0, abs_tol=1e-2)
    assert out["mean"] == r["psnr"]
    assert "automobile" in out["by_class"]
    assert out["n_tracks_with_psnr"] == 1


def test_compute_class_psnr_by_class_aggregation():
    """Two cars + one truck, all at different positions; each renders correctly
    in their region. Verify per-class aggregation."""
    H, W = 256, 256
    rgb_gt = torch.zeros(1, H, W, 3)
    rgb_pred = torch.full((1, H, W, 3), 0.1)  # uniform 0.1 → 20 dB for any region
    tracks = []
    for i, (cls, x_shift) in enumerate(
        [
            ("automobile", -0.5),
            ("automobile", 0.5),
            ("heavy_truck", 1.5),
        ]
    ):
        pose = torch.eye(4)
        pose[:3, 3] = torch.tensor([x_shift, 0.0, 5.0])
        tracks.append({"id": f"t{i}", "class": cls, "pose": pose, "size": torch.tensor([1.0, 1.0, 1.0])})

    out = compute_class_psnr(
        rgb_pred,
        rgb_gt,
        valid_mask=None,
        active_tracks=tracks,
        T_world2cam=_identity_pose(),
        H=H,
        W=W,
        K=_pinhole_K(),
    )
    assert "automobile" in out["by_class"]
    assert "heavy_truck" in out["by_class"]
    assert out["by_class"]["automobile"]["n_tracks"] == 2
    assert out["by_class"]["heavy_truck"]["n_tracks"] == 1
    assert math.isclose(out["by_class"]["automobile"]["mean_psnr"], 20.0, abs_tol=1e-2)


def test_compute_class_psnr_track_outside_image_yields_zero_pixels():
    """Cuboid behind camera projects to empty AABB → psnr None."""
    H, W = 128, 128
    rgb_pred = torch.zeros(1, H, W, 3)
    rgb_gt = torch.zeros(1, H, W, 3)
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([0.0, 0.0, -5.0])  # behind cam
    out = compute_class_psnr(
        rgb_pred,
        rgb_gt,
        valid_mask=None,
        active_tracks=[{"id": "t0", "class": "automobile", "pose": pose, "size": torch.tensor([1.0, 1.0, 1.0])}],
        T_world2cam=_identity_pose(),
        H=H,
        W=W,
        ftheta_params={"angle_to_pixeldist_poly": [0.0, 200.0], "principal_point": [64.0, 64.0]},
    )
    # FTheta skips behind-camera tracks → empty mask → psnr None
    r = out["per_track"][0]
    assert r["n_pixels"] == 0
    assert r["psnr"] is None
    assert out["mean"] is None


def test_compute_class_psnr_valid_mask_excludes_invalid_pixels():
    """A track region intersected with valid_mask=0 → fewer pixels → may flip
    below min_pixels threshold."""
    H, W = 256, 256
    rgb_pred = torch.full((1, H, W, 3), 0.1)
    rgb_gt = torch.zeros(1, H, W, 3)
    valid = torch.zeros(1, H, W, 1)
    # Only 5 pixels valid — below default min_pixels=50
    valid[0, 0:1, 0:5, 0] = 1.0
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([0.0, 0.0, 5.0])
    out = compute_class_psnr(
        rgb_pred,
        rgb_gt,
        valid_mask=valid,
        active_tracks=[{"id": "t0", "class": "automobile", "pose": pose, "size": torch.tensor([1.0, 1.0, 1.0])}],
        T_world2cam=_identity_pose(),
        H=H,
        W=W,
        K=_pinhole_K(),
    )
    # Cuboid mask intersected with valid → ≤ 5 px → below threshold → None
    assert out["per_track"][0]["psnr"] is None
    assert out["n_tracks_with_psnr"] == 0


def test_compute_class_psnr_with_ftheta_intrinsics():
    """FTheta path produces same magnitude (small-angle linear poly ≈ pinhole)."""
    H, W = 256, 256
    rgb_gt = torch.zeros(1, H, W, 3)
    rgb_pred = torch.full((1, H, W, 3), 0.1)
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([0.0, 0.0, 5.0])
    out = compute_class_psnr(
        rgb_pred,
        rgb_gt,
        valid_mask=None,
        active_tracks=[{"id": "t0", "class": "automobile", "pose": pose, "size": torch.tensor([1.0, 1.0, 1.0])}],
        T_world2cam=_identity_pose(),
        H=H,
        W=W,
        ftheta_params={"angle_to_pixeldist_poly": [0.0, 300.0], "principal_point": [128.0, 128.0]},
    )
    r = out["per_track"][0]
    # Same uniform diff, so any non-empty mask gives 20 dB
    assert math.isclose(r["psnr"], 20.0, abs_tol=0.5)


# -----------------------------------------------------------------------------
# collect_active_tracks_for_frame
# -----------------------------------------------------------------------------


def test_collect_active_tracks_filters_inactive_and_no_size():
    poses = {
        "alice": torch.eye(4).unsqueeze(0).expand(3, 4, 4).clone(),
        "bob": torch.eye(4).unsqueeze(0).expand(3, 4, 4).clone(),
        "carol": torch.eye(4).unsqueeze(0).expand(3, 4, 4).clone(),
    }
    poses["alice"][1, :3, 3] = torch.tensor([5.0, 0.0, 0.0])
    active = {
        "alice": torch.tensor([False, True, False]),  # active at frame 1
        "bob": torch.tensor([True, True, True]),  # always active
        "carol": torch.tensor([True, True, True]),  # always active, but no size
    }
    metadata = {
        "alice": {"class": "automobile", "size": torch.tensor([4.0, 2.0, 1.5])},
        "bob": {"class": "heavy_truck", "size": torch.tensor([10.0, 3.0, 3.5])},
        # carol has no metadata entry → no size → skipped
    }
    out = collect_active_tracks_for_frame(poses, active, metadata, frame_idx=1)
    ids = [t["id"] for t in out]
    assert ids == ["alice", "bob"]
    # alice's pose at frame 1 should have the translation we set
    alice = next(t for t in out if t["id"] == "alice")
    assert torch.allclose(alice["pose"][:3, 3], torch.tensor([5.0, 0.0, 0.0]))
    assert alice["class"] == "automobile"
    assert alice["size"].tolist() == [4.0, 2.0, 1.5]


def test_collect_active_tracks_empty_when_no_frame_active():
    poses = {"a": torch.eye(4).unsqueeze(0).expand(3, 4, 4).clone()}
    active = {"a": torch.tensor([False, False, False])}
    metadata = {"a": {"class": "automobile", "size": torch.ones(3)}}
    out = collect_active_tracks_for_frame(poses, active, metadata, frame_idx=1)
    assert out == []
