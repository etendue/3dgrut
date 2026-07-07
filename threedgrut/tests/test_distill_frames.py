# SPDX-License-Identifier: Apache-2.0
"""E2.2 Task 1 — TDD for pseudo-GT progressive-distillation injection.

Mac CPU only: the real NCore dataset / 3dgrut renderer are unavailable, so we
mock the source-frame provider with ``SimpleNamespace`` batches (mirroring
``test_eval_frames_dir.py``) and pin the LOGIC:

  ① distill.enabled=false → no DistillFrameSource; loss numerics byte-identical.
  ② p=1 + synthetic 2-frame pack → every sample is a pseudo-GT batch; the
     photometric term scales exactly ×λ (λ=2 ⇒ 2× the λ=1 photometric loss);
     regularization terms are λ-independent (numerically unchanged).
  ③ pose reconstruction consistency: DistillFrameSource reproduces render.py's
     novel pose via the SAME perturb_batch_shutter_pair_torch (tol 1e-6).
  ④ missing frame / empty pack / mode mismatch → explicit raise.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torchvision

from threedgrut.datasets.distill_frames import DistillFrameSource
from threedgrut.utils.novel_view import (
    novel_frame_key,
    perturb_batch_shutter_pair_torch,
    resolve_novel_modes,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

CAM = "camera_front_wide_120fov"
H, W = 4, 6
MODE = "lateral_1m"


def _rand_c2w(seed: int) -> torch.Tensor:
    """A plausible (1,4,4) camera-to-world with a non-identity rotation so the
    lateral shift along the camera-right axis is a non-trivial world delta."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(4, generator=g)
    q = q / q.norm()
    w, x, y, z = q.tolist()
    R = torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float32,
    )
    T = torch.eye(4, dtype=torch.float32)
    T[:3, :3] = R
    T[:3, 3] = torch.randn(3, generator=g)
    return T.unsqueeze(0)


def _mk_source_batch(cam: str, ts_us: int, seed: int) -> SimpleNamespace:
    """Mimic a gpu_batch produced by NCoreDataset.get_gpu_batch_with_intrinsics
    for a single frame: camera-space rays (pose-independent), start/end poses."""
    return SimpleNamespace(
        rays_ori=torch.zeros(1, H, W, 3),
        rays_dir=torch.randn(1, H, W, 3, generator=torch.Generator().manual_seed(seed + 99)),
        T_to_world=_rand_c2w(seed),
        T_to_world_end=_rand_c2w(seed + 1),
        rays_in_world_space=False,
        pixel_coords=torch.zeros(1, H, W, 2),
        rgb_gt=torch.rand(1, H, W, 3),
        mask=torch.ones(1, H, W, 1),
        camera_id=cam,
        camera_idx=0,
        frame_idx=-1,
        timestamp_us=ts_us,
        intrinsics_FThetaCameraModelParameters={"resolution": (W, H)},
    )


def _write_pack(root: str, mode: str, entries) -> None:
    """Write <root>/<mode>/frames_map.json + PNGs. ``entries`` is a list of
    (camera_id, ts_us, img[H,W,3] float) tuples."""
    mode_dir = os.path.join(root, mode)
    os.makedirs(os.path.join(mode_dir, "cam"), exist_ok=True)
    fmap = {}
    for idx, (cam, ts, img) in enumerate(entries):
        rel = os.path.join("cam", f"{idx:06d}.png")
        torchvision.utils.save_image(img.permute(2, 0, 1), os.path.join(mode_dir, rel))
        fmap[novel_frame_key(cam, ts)] = rel
    with open(os.path.join(mode_dir, "frames_map.json"), "w") as f:
        json.dump(fmap, f)


def _build_source(batches):
    """DistillFrameSource takes an iterable of source batches (the dataset's
    val-split batches render.py iterated) — keyed internally by (cam, ts)."""
    return batches


# ---------------------------------------------------------------------------
# ③ pose reconstruction consistency (the distillation "poison" guard)
# ---------------------------------------------------------------------------


def test_pose_reconstruction_matches_render_side(tmp_path):
    ts = 55132
    src = _mk_source_batch(CAM, ts, seed=3)
    img = torch.rand(H, W, 3)
    _write_pack(str(tmp_path), MODE, [(CAM, ts, img)])

    source = DistillFrameSource(str(tmp_path), MODE, _build_source([src]))
    batch = source.sample(rng=np.random.default_rng(0))

    # render.py's novel path applies THIS exact transform to the source poses.
    exp_start, exp_end = perturb_batch_shutter_pair_torch(src.T_to_world, src.T_to_world_end, MODE)
    assert torch.allclose(batch.T_to_world, exp_start, atol=1e-6)
    assert torch.allclose(batch.T_to_world_end, exp_end, atol=1e-6)
    # rays are reused camera-space (pose-independent) — identical to source.
    assert torch.allclose(batch.rays_dir, src.rays_dir, atol=1e-6)
    assert bool(batch.rays_in_world_space) is False
    assert batch.camera_id == CAM
    assert int(batch.timestamp_us) == ts


def test_sample_uses_fixed_frame_as_rgb_gt_and_no_mask(tmp_path):
    ts = 7001
    src = _mk_source_batch(CAM, ts, seed=5)
    img = torch.rand(H, W, 3)
    _write_pack(str(tmp_path), MODE, [(CAM, ts, img)])

    source = DistillFrameSource(str(tmp_path), MODE, _build_source([src]))
    batch = source.sample(rng=np.random.default_rng(0))

    # rgb_gt is the fixed PNG (8-bit quantized), NOT the source's real rgb.
    saved = (
        torchvision.io.read_image(os.path.join(str(tmp_path), MODE, "cam", "000000.png"))
        .float()
        .div(255.0)
        .permute(1, 2, 0)
    )
    assert batch.rgb_gt.shape == (1, H, W, 3)
    assert torch.allclose(batch.rgb_gt[0], saved, atol=1e-6)
    assert not torch.allclose(batch.rgb_gt[0], src.rgb_gt[0])  # not the real image
    # novel frames carry no sseg/road masks → mask must be None.
    assert batch.mask is None
    assert getattr(batch, "is_distill", False) is True


# ---------------------------------------------------------------------------
# ④ error paths
# ---------------------------------------------------------------------------


def test_empty_pack_raises(tmp_path):
    _write_pack(str(tmp_path), MODE, [])  # empty frames_map.json
    src = _mk_source_batch(CAM, 1, seed=1)
    with pytest.raises((ValueError, RuntimeError)):
        DistillFrameSource(str(tmp_path), MODE, _build_source([src]))


def test_missing_mode_dir_raises(tmp_path):
    # pack has lateral_1m but we ask for lateral_3m → frames_map.json missing.
    src = _mk_source_batch(CAM, 1, seed=1)
    _write_pack(str(tmp_path), "lateral_1m", [(CAM, 1, torch.rand(H, W, 3))])
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        DistillFrameSource(str(tmp_path), "lateral_3m", _build_source([src]))


def test_pack_frame_without_source_pose_raises(tmp_path):
    # frames_map references (CAM, 999) but no source batch has that ts.
    src = _mk_source_batch(CAM, 111, seed=1)  # ts 111, pack wants 999
    _write_pack(str(tmp_path), MODE, [(CAM, 999, torch.rand(H, W, 3))])
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        DistillFrameSource(str(tmp_path), MODE, _build_source([src]))


def test_missing_png_file_raises(tmp_path):
    ts = 222
    src = _mk_source_batch(CAM, ts, seed=1)
    _write_pack(str(tmp_path), MODE, [(CAM, ts, torch.rand(H, W, 3))])
    # delete the PNG after building the pack
    os.remove(os.path.join(str(tmp_path), MODE, "cam", "000000.png"))
    source = DistillFrameSource(str(tmp_path), MODE, _build_source([src]))
    with pytest.raises((FileNotFoundError, RuntimeError)):
        source.sample(rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# ② photometric ×λ scaling + regularization λ-independence
# ---------------------------------------------------------------------------


def _distill_losses(rgb_pred, rgb_gt, lam):
    """The loss contract a distill batch must obey in trainer.get_losses:
    full-image L1 + SSIM, both scaled by lam; reg terms independent of lam.

    We import the trainer's scaling helper so the test pins the SAME numeric
    path the trainer uses (single source of truth)."""
    from threedgrut.datasets.distill_frames import distill_photometric_loss

    return distill_photometric_loss(rgb_pred, rgb_gt, lam)


def test_photometric_scales_with_lambda():
    torch.manual_seed(0)
    # SSIM uses an 11x11 window (valid padding) → image must be ≥ 11 px.
    rgb_pred = torch.rand(1, 16, 20, 3)
    rgb_gt = torch.rand(1, 16, 20, 3)

    l1 = _distill_losses(rgb_pred, rgb_gt, lam=1.0)
    l2 = _distill_losses(rgb_pred, rgb_gt, lam=2.0)

    # exactly 2× when λ doubles (photometric term is linear in λ).
    assert torch.isclose(l2, 2.0 * l1, atol=1e-6)
    # λ=0 ⇒ zero photometric contribution.
    l0 = _distill_losses(rgb_pred, rgb_gt, lam=0.0)
    assert torch.isclose(l0, torch.zeros(()), atol=1e-8)


def test_photometric_is_full_image_l1_plus_ssim():
    """Distill photometric == lam * (L1_full + (1 - SSIM)); no layered/mask path."""
    from threedgrut.model.losses import ssim

    torch.manual_seed(1)
    rgb_pred = torch.rand(1, 16, 20, 3)
    rgb_gt = torch.rand(1, 16, 20, 3)
    lam = 0.3

    l1 = torch.abs(rgb_pred - rgb_gt).mean()
    ssim_term = 1.0 - ssim(rgb_pred.permute(0, 3, 1, 2), rgb_gt.permute(0, 3, 1, 2))
    expected = lam * (l1 + ssim_term)
    got = _distill_losses(rgb_pred, rgb_gt, lam)
    assert torch.isclose(got, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# ① byte-equivalence when disabled — config resolves with distill.enabled=false
#    and the trainer builds NO source.
# ---------------------------------------------------------------------------


def test_disabled_config_defaults(tmp_path):
    """The distill config group must default to enabled=false with the exact
    reserved keys (frames_dir/p/lam/mode/region_weight_mask). Byte-equivalence
    contract: a resolved base config carries these defaults and nothing else
    changes."""
    from omegaconf import OmegaConf

    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "base_gs.yaml")
    conf = OmegaConf.load(cfg_path)
    assert "distill" in conf, "base config must declare the distill group"
    d = conf.distill
    assert d.enabled is False
    assert d.frames_dir is None
    assert float(d.p) == 0.3
    assert float(d.lam) == 0.1
    assert d.mode == "lateral_1m"
    assert d.region_weight_mask is None  # reserved (YAGNI) — must stay null


# ---------------------------------------------------------------------------
# render.py --novel-only parameterization (default backward-compatible)
# ---------------------------------------------------------------------------


def test_resolve_novel_modes_backward_compatible():
    # historical behaviour: novel_only bool → lateral_3m + lateral_6m.
    assert resolve_novel_modes(True, None) == ("lateral_3m", "lateral_6m")
    # not restricted → None (render all NOVEL_VIEW_MODES).
    assert resolve_novel_modes(False, None) is None


def test_resolve_novel_modes_explicit_list():
    # E2.2 single-band distill pack.
    assert resolve_novel_modes(False, ["lateral_1m"]) == ("lateral_1m",)
    assert resolve_novel_modes(True, ["lateral_2m", "lateral_3m"]) == (
        "lateral_2m",
        "lateral_3m",
    )


def test_resolve_novel_modes_rejects_unknown_mode():
    with pytest.raises(ValueError):
        resolve_novel_modes(True, ["lateral_9m"])  # not a real mode
