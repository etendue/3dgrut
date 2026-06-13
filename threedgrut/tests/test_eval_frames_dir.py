# SPDX-License-Identifier: Apache-2.0
"""E0.4-O2 unit tests for the offline frames-dir evaluator.

Mac CPU: synthetic gpu_batch-like objects + temp PNG prediction files. The
heavy integrations (NCore dataset, YOLO, Inception, plane warp on a real
height field) are exercised in the inceptio runtime validation; here we pin
the alignment logic (pred file ↔ batch) and the metric plumbing on the
identity case (pred == GT → PSNR huge, SSIM ≈ 1).
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torchvision

from scripts.eval_frames_dir import evaluate_frames, resolve_pred_path


def _mk_batch(cam: str, frame_idx: int, img: torch.Tensor):
    return SimpleNamespace(
        rgb_gt=img.unsqueeze(0),            # [1, H, W, 3] float [0,1]
        camera_id=cam,
        frame_idx=frame_idx,
        timestamp_us=1000 + frame_idx,
        T_to_world=torch.eye(4).unsqueeze(0),
        T_to_world_end=torch.eye(4).unsqueeze(0),
        image_infos={},
        intrinsics_FThetaCameraModelParameters=None,
        rays_dir=torch.zeros(1, 4, 6, 3),
        rays_in_world_space=False,
    )


def test_resolve_pred_path_template_and_map(tmp_path):
    # template fallback: <frames_dir>/<camera_id>/<frame_idx:06d>.png
    p = resolve_pred_path(str(tmp_path), "cam_a", 7, frames_map=None)
    assert p.endswith("cam_a/000007.png")
    # explicit map wins
    m = {"cam_a:7": "weird/name 0007.png"}
    p2 = resolve_pred_path(str(tmp_path), "cam_a", 7, frames_map=m)
    assert p2.endswith("weird/name 0007.png")


def test_resolve_pred_path_timestamp_key_takes_precedence(tmp_path):
    """NCore batches carry no frame_idx (-1) — sensor timestamp is the only
    honest join key against nre's timestamps.json."""
    m = {
        "ts:cam_a:55132": "cam_a/cam_a/000000.png",
        "cam_a:-1": "wrong.png",
    }
    p = resolve_pred_path(str(tmp_path), "cam_a", -1, frames_map=m,
                          timestamp_us=55132)
    assert p.endswith("cam_a/cam_a/000000.png")
    # no ts entry → falls back to frame_idx key
    p2 = resolve_pred_path(str(tmp_path), "cam_a", -1, frames_map=m,
                           timestamp_us=99999)
    assert p2.endswith("wrong.png")


def test_evaluate_identity_frames(tmp_path):
    torch.manual_seed(0)
    H, W = 16, 24
    batches = []
    for cam, fi in (("cam_a", 0), ("cam_a", 8), ("cam_b", 0)):
        img = torch.rand(H, W, 3)
        (tmp_path / cam).mkdir(exist_ok=True)
        torchvision.utils.save_image(
            img.permute(2, 0, 1), str(tmp_path / cam / f"{fi:06d}.png"),
        )
        # GT must equal the SAVED png (8-bit quantized) for a clean identity
        saved = torchvision.io.read_image(
            str(tmp_path / cam / f"{fi:06d}.png")
        ).float().div(255.0).permute(1, 2, 0)
        batches.append(_mk_batch(cam, fi, saved))

    out = evaluate_frames(
        batches, frames_dir=str(tmp_path), frames_map=None,
        mode="interpolated", lpips_fn=None, detector=None,
        height_field=None, ground_z=None, fid_kid=False,
    )
    assert out["n_frames"] == 3
    assert out["mean_psnr"] > 60.0       # identity up to PNG quantization
    assert out["mean_ssim"] > 0.99


def test_cameras_filter_skips_other_cameras(tmp_path):
    """E0.4-O3: rig-offset lateral passes only match our per-camera lateral
    definition for the FRONT camera — eval must be restrictable to it. A
    filtered-out camera must be skipped BEFORE prediction loading (no
    FileNotFoundError for frames we never rendered)."""
    img = torch.rand(8, 8, 3)
    (tmp_path / "cam_front").mkdir()
    torchvision.utils.save_image(
        img.permute(2, 0, 1), str(tmp_path / "cam_front" / "000000.png"),
    )
    saved = torchvision.io.read_image(
        str(tmp_path / "cam_front" / "000000.png")
    ).float().div(255.0).permute(1, 2, 0)
    batches = [
        _mk_batch("cam_front", 0, saved),
        _mk_batch("cam_other", 0, torch.rand(8, 8, 3)),  # no frames on disk
    ]
    out = evaluate_frames(
        batches, frames_dir=str(tmp_path), frames_map=None,
        mode="interpolated", lpips_fn=None, detector=None,
        height_field=None, ground_z=None, fid_kid=False,
        cameras=("cam_front",),
    )
    assert out["n_frames"] == 1


def test_evaluate_missing_pred_raises(tmp_path):
    img = torch.rand(8, 8, 3)
    batches = [_mk_batch("cam_a", 3, img)]
    try:
        evaluate_frames(
            batches, frames_dir=str(tmp_path), frames_map=None,
            mode="interpolated", lpips_fn=None, detector=None,
            height_field=None, ground_z=None, fid_kid=False,
        )
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised, "missing prediction file must be a hard error (alignment!)"
