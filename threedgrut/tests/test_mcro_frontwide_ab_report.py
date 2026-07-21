"""CPU tests for the MCRO front-wide render A/B report driver."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


_DRIVER_PATH = Path(__file__).resolve().parents[2] / "scripts/drivers/mcro_frontwide_ab_report.py"
_FRONT_WIDE = "camera_front_wide_120fov"


def _load_driver():
    spec = importlib.util.spec_from_file_location("mcro_frontwide_ab_report", _DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_eval_dir(root: Path, noise: int) -> Path:
    run_root = root / "run"
    renders = run_root / "ours_5000" / "renders"
    gts = run_root / "ours_5000" / "gt"
    renders.mkdir(parents=True)
    gts.mkdir(parents=True)
    (run_root / "metrics.json").write_text(
        json.dumps({"per_camera": {_FRONT_WIDE: {"n_frames": 2}}}), encoding="utf-8"
    )
    yy, xx = np.mgrid[:8, :8]
    gt = np.stack((xx * 20, yy * 20, (xx + yy) * 10), axis=-1).astype(np.uint8)
    render = np.clip(gt.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for index in range(2):
        Image.fromarray(gt).save(gts / f"frame_{index:04d}.png")
        Image.fromarray(render).save(renders / f"frame_{index:04d}.png")
    return root


def _write_crops(path: Path, box: list[int]) -> None:
    path.write_text(
        json.dumps(
            {
                "camera_id": _FRONT_WIDE,
                "frame_splits": {"0": "train", "1": "held_out"},
                "crops": [{"name": "detail", "frame_index": 1, "box": box, "split": "held_out"}],
            }
        ),
        encoding="utf-8",
    )


def test_report_prefers_less_noisy_arm_and_writes_artifacts(tmp_path: Path) -> None:
    driver = _load_driver()
    eval_a = _make_eval_dir(tmp_path / "eval_a", noise=20)
    eval_b = _make_eval_dir(tmp_path / "eval_b", noise=2)
    crops = tmp_path / "crops.json"
    _write_crops(crops, [1, 1, 7, 7])

    report = driver.analyze_frontwide_pair(str(eval_a), str(eval_b), str(crops), str(tmp_path / "report"), use_lpips=False)

    crop = report["crops"]["detail"]
    expected_a_psnr = -20.0 * np.log10(20.0 / 255.0)
    assert crop["a"]["psnr"] == pytest.approx(expected_a_psnr, abs=1e-4)
    assert crop["delta_b_minus_a"]["psnr"] > 0.0
    assert len(report["radial_bins"]) == 4
    assert report["split_metrics"]["train"]["n_frames"] == 1
    assert report["split_metrics"]["held_out"]["n_frames"] == 1
    assert (tmp_path / "report" / "frontwide_ab_report.md").is_file()
    assert (tmp_path / "report" / "frontwide_crop_detail.png").is_file()


def test_crop_out_of_bounds_raises_actionable_error(tmp_path: Path) -> None:
    driver = _load_driver()
    eval_a = _make_eval_dir(tmp_path / "eval_a", noise=10)
    eval_b = _make_eval_dir(tmp_path / "eval_b", noise=0)
    crops = tmp_path / "crops.json"
    _write_crops(crops, [0, 0, 99, 4])

    with pytest.raises(ValueError, match="detail.*out of bounds"):
        driver.analyze_frontwide_pair(str(eval_a), str(eval_b), str(crops), str(tmp_path / "report"), use_lpips=False)
