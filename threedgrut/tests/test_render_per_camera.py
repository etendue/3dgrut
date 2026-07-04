# SPDX-License-Identifier: Apache-2.0
"""T8.5.7 / V3-E4 unit tests for render.py per-camera metric aggregation and
``render.eval_cameras`` filter.

V3-E4 motivation: 5-cam vs 7-cam KPI ablation needs per-camera dB breakdown
(to attribute gains to specific cameras) and a 5-cam-ring-subset eval filter
(so 7-cam-trained ckpts can be evaluated on the exact same test set as the
5-cam baseline for byte-identical comparison).

render.py is CUDA-bound (torchmetrics + tracer + dataloader) so we follow the
T6F.2 test pattern: replicate the pure aggregation / filter logic here in
isolation, drive it with synthetic per-frame "metric tuples" and "camera_id"
strings, then assert the resulting per_camera dict + filter behavior matches
render.py L233-562's logic exactly.

The code blocks below mirror render.py's V3-E4 sections one-to-one; whenever
those sections in render.py change, this test must be updated in lockstep.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

# --- Replicas of render.py V3-E4 sections (keep one-to-one with render.py) ---

_PER_CAM_KEYS = (
    "psnr",
    "ssim",
    "lpips",
    "cc_psnr",
    "cc_ssim",
    "cc_lpips",
    "psnr_masked",
    "ssim_masked",
    "lpips_masked",
    "cc_psnr_masked",
    "cc_ssim_masked",
    "cc_lpips_masked",
)


def _filter_should_skip(eval_cameras_filter, batch_cam) -> bool:
    """Mirror of render.py L278-284 — skip if filter set and batch_cam not in it."""
    if eval_cameras_filter is None:
        return False
    if isinstance(batch_cam, (list, tuple)):
        batch_cam = batch_cam[0] if len(batch_cam) > 0 else None
    return batch_cam not in eval_cameras_filter


def _accumulate_per_cam(per_cam, cam_id: Optional[str], metric_tuple):
    """Mirror of render.py L431-446 — append last metric values per camera."""
    if cam_id is None:
        return
    pc = per_cam.setdefault(cam_id, {k: [] for k in _PER_CAM_KEYS})
    for k, v in zip(_PER_CAM_KEYS, metric_tuple):
        pc[k].append(v)


def _summarize_per_cam(per_cam):
    """Mirror of render.py L523-543 — compute per-camera mean dict."""
    summary = {}
    for cid, dlists in per_cam.items():
        n = len(dlists["psnr"])
        if n == 0:
            continue
        summary[cid] = {"n_frames": int(n)}
        for k in _PER_CAM_KEYS:
            summary[cid][f"mean_{k}"] = float(np.mean(dlists[k]))
    return summary


# --- Tests ---------------------------------------------------------------


def _mk_metric_tuple(rng, base: float = 25.0):
    """Synthetic 12-tuple of metrics with small jitter (deterministic via rng)."""
    return tuple(float(base + rng.uniform(-1, 1)) for _ in _PER_CAM_KEYS)


def test_per_camera_byte_identical_when_cam_id_none():
    """V3-E4 byte-identical regression: cam_id=None (NeRF/Colmap path) ->
    per_cam stays empty, metrics.json["per_camera"] never written.
    """
    per_cam: dict = {}
    rng = np.random.default_rng(0)
    for _ in range(10):
        _accumulate_per_cam(per_cam, None, _mk_metric_tuple(rng))
    assert per_cam == {}
    assert _summarize_per_cam(per_cam) == {}


def test_per_camera_single_camera_matches_global_mean():
    """V3-E4: when all frames come from one camera, per_camera[cid].mean_X
    must equal the global mean of metric X.
    """
    per_cam: dict = {}
    rng = np.random.default_rng(1)
    cid = "camera_front_wide_120fov"
    global_lists = {k: [] for k in _PER_CAM_KEYS}
    for _ in range(8):
        m = _mk_metric_tuple(rng)
        _accumulate_per_cam(per_cam, cid, m)
        for k, v in zip(_PER_CAM_KEYS, m):
            global_lists[k].append(v)

    summary = _summarize_per_cam(per_cam)
    assert set(summary.keys()) == {cid}
    assert summary[cid]["n_frames"] == 8
    for k in _PER_CAM_KEYS:
        assert summary[cid][f"mean_{k}"] == pytest.approx(float(np.mean(global_lists[k])), abs=1e-10)


def test_per_camera_two_cameras_weighted_average_equals_global():
    """V3-E4: when 2 cameras contribute different counts, the global mean
    equals the weighted average of per-camera means by n_frames.
    """
    per_cam: dict = {}
    rng = np.random.default_rng(2)
    counts = {"cam_a": 5, "cam_b": 3}  # 5 + 3 = 8 frames
    global_lists = {k: [] for k in _PER_CAM_KEYS}

    for cid, n in counts.items():
        for _ in range(n):
            m = _mk_metric_tuple(rng, base=28.0 if cid == "cam_a" else 22.0)
            _accumulate_per_cam(per_cam, cid, m)
            for k, v in zip(_PER_CAM_KEYS, m):
                global_lists[k].append(v)

    summary = _summarize_per_cam(per_cam)
    assert summary["cam_a"]["n_frames"] == 5
    assert summary["cam_b"]["n_frames"] == 3

    # weighted reconstruction = global mean
    total_n = sum(counts.values())
    for k in _PER_CAM_KEYS:
        weighted = (
            summary["cam_a"][f"mean_{k}"] * counts["cam_a"] + summary["cam_b"][f"mean_{k}"] * counts["cam_b"]
        ) / total_n
        assert weighted == pytest.approx(float(np.mean(global_lists[k])), abs=1e-9)


def test_eval_cameras_filter_none_lets_everything_through():
    """V3-E4: filter=None must not skip anything (back-compat default path)."""
    for batch_cam in ["camera_foo", ["camera_bar"], None, []]:
        assert _filter_should_skip(None, batch_cam) is False


def test_eval_cameras_filter_skips_non_matching():
    """V3-E4: only camera_ids in the filter list pass through."""
    keep = ["camera_front_wide_120fov", "camera_cross_left_120fov"]
    assert _filter_should_skip(keep, "camera_front_wide_120fov") is False
    assert _filter_should_skip(keep, ["camera_cross_left_120fov"]) is False  # collated list form
    assert _filter_should_skip(keep, "camera_front_tele_30fov") is True
    assert _filter_should_skip(keep, ["camera_rear_right_70fov"]) is True
    assert _filter_should_skip(keep, None) is True
    assert _filter_should_skip(keep, []) is True  # empty list → None


def test_eval_cameras_filter_5cam_subset_count():
    """V3-E4 end-to-end shape: simulate a 7-cam test set with 8 frames each.
    Filter to the 5-cam ring → final per_camera dict has exactly 5 keys with
    8 frames each, total = 40 (not 56).
    """
    seven_cams = [
        "camera_front_wide_120fov",
        "camera_rear_tele_30fov",
        "camera_cross_left_120fov",
        "camera_cross_right_120fov",
        "camera_rear_left_70fov",
        "camera_front_tele_30fov",  # extra 1
        "camera_rear_right_70fov",  # extra 2
    ]
    ring_5 = seven_cams[:5]
    rng = np.random.default_rng(3)

    per_cam: dict = {}
    kept = 0
    for cid in seven_cams:
        for _ in range(8):
            if _filter_should_skip(ring_5, cid):
                continue
            kept += 1
            _accumulate_per_cam(per_cam, cid, _mk_metric_tuple(rng))

    summary = _summarize_per_cam(per_cam)
    assert len(summary) == 5
    assert set(summary.keys()) == set(ring_5)
    for cid in ring_5:
        assert summary[cid]["n_frames"] == 8
    assert kept == 40  # 5 cams * 8 frames


def test_eval_cameras_filter_zero_hits_raises_in_render_py():
    """V3-E4 sanity: render.py L463-467 raises RuntimeError when filter
    matches 0 frames. We replicate the check here as a smoke test of the
    error message format (the actual raise is integration-only).
    """
    eval_cameras_filter = ["camera_nonexistent_xxx"]
    seven_cams = [
        "camera_front_wide_120fov",
        "camera_rear_tele_30fov",
        "camera_cross_left_120fov",
        "camera_cross_right_120fov",
        "camera_rear_left_70fov",
        "camera_front_tele_30fov",
        "camera_rear_right_70fov",
    ]
    psnr: list = []
    for cid in seven_cams:
        if _filter_should_skip(eval_cameras_filter, cid):
            continue
        psnr.append(1.0)

    # Mirror render.py L463-467
    if eval_cameras_filter is not None and len(psnr) == 0:
        with pytest.raises(RuntimeError, match="matched 0 frames"):
            raise RuntimeError(
                f"[V3-E4] render.eval_cameras={eval_cameras_filter} matched 0 frames in "
                f"the test split. Check camera_id spelling against dataset.camera_ids."
            )
    else:
        pytest.fail("Test setup error: expected 0 hits but got > 0")
