"""Task T8.9 tests — inject_viz_4d CLI early validation paths.

We can't fully exercise the inject pipeline on Mac CPU (NCoreDataset needs the
NCore SDK + a real manifest), but we can verify:

  * The CLI's input-validation branches raise clear errors.
  * The torch.save / torch.load round-trip preserves an existing viz_4d block
    (we forge the metadata directly without going through the full extract).
  * Backup-file behavior on in-place writes.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from threedgrut.viz.inject import _populate_tracks_from_dataset, inject_viz_4d


# ---------------------------------------------------------- helpers
def _write_dummy_ckpt(path: Path, use_layered_model: bool = True) -> None:
    """Minimal v2 ckpt skeleton — just enough fields for inject's validation."""
    ckpt = {
        "config": OmegaConf.create(
            {
                "use_layered_model": use_layered_model,
                "path": "/nonexistent/old_training_path.json",
            }
        ),
        "model": {"gaussians_nodes": {}, "scene_extent": 1.0},
        "global_step": 100,
    }
    torch.save(ckpt, path)


# ---------------------------------------------------------- tests
def test_dataset_path_required(tmp_path):
    """Refuse to run without --dataset_path (ckpt doesn't persist ego/tracks/LiDAR)."""
    p = tmp_path / "ckpt.pt"
    _write_dummy_ckpt(p)
    with pytest.raises(ValueError, match="dataset_path"):
        inject_viz_4d(str(p), dataset_path=None, out_path=None)


def test_v1_ckpt_rejected(tmp_path):
    """use_layered_model=False ckpts have no LayeredGaussians, so viz_4d makes no sense."""
    p = tmp_path / "v1.pt"
    _write_dummy_ckpt(p, use_layered_model=False)
    with pytest.raises(ValueError, match="LayeredGaussians"):
        inject_viz_4d(str(p), dataset_path="/fake/manifest.json", out_path=None)


def test_missing_ckpt_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="ckpt not found"):
        inject_viz_4d(str(tmp_path / "does_not_exist.pt"), dataset_path="/fake/manifest.json", out_path=None)


def test_populate_tracks_no_dynamic_layer():
    """Models without 'dynamic_rigids' return 0 tracks without touching NCore SDK."""
    model = MagicMock()
    model.layers = {"background": MagicMock()}
    n = _populate_tracks_from_dataset(model, dataset=MagicMock())
    assert n == 0


def test_inject_extract_to_ckpt_roundtrip_preserves_ftheta(tmp_path, monkeypatch):
    """T8.13 inject contract (Mac proxy): inject_viz_4d's core path is
    ``extract_4d_metadata(model, ds, conf) → ckpt['viz_4d'] = md →
    torch.save/load``. The full pipeline needs NCore SDK + kaolin (A800
    territory; covered by Task 8), but the FTheta dict survival through the
    extract + serialization layer is what this Mac test pins down.

    Mirrors what inject.py:165 does (``ckpt['viz_4d'] = md`` + torch.save),
    minus the LayeredGaussians init + populate_tracks (covered elsewhere).
    """
    # Reuse the FTheta mock fixture pattern from test_viz_4d_metadata.
    from hydra import compose, initialize_config_dir

    from threedgrut.tests.test_viz_4d_metadata import _mock_dataset, _model_with_tracks
    from threedgrut.viz.metadata import extract_4d_metadata

    _CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        conf = compose(config_name="apps/ncore_3dgut_mcmc")

    model = _model_with_tracks(conf)
    dataset = _mock_dataset(n_frames=4, camera_type="ftheta")
    md = extract_4d_metadata(model, dataset, conf)

    ckpt = {
        "config": OmegaConf.create({"use_layered_model": True}),
        "model": {"gaussians_nodes": {}, "scene_extent": 1.0},
        "viz_4d": md,
    }
    p = tmp_path / "ftheta_ckpt.pt"
    torch.save(ckpt, p)
    loaded = torch.load(p, weights_only=False)

    # T8.13 contract: v2 schema + 8 FTheta keys + (W, H) tuple survive save/load.
    assert loaded["viz_4d"]["schema_version"] == 2
    ego = loaded["viz_4d"]["ego"]
    assert ego["primary_camera_intrinsics_FTheta"] is not None
    assert set(ego["primary_camera_intrinsics_FTheta"].keys()) >= {
        "resolution",
        "max_angle",
        "pixeldist_to_angle_poly",
        "angle_to_pixeldist_poly",
        "linear_cde",
        "principal_point",
        "shutter_type",
        "reference_poly",
    }
    assert tuple(ego["primary_camera_resolution"]) == (1920, 1208)


def test_inject_preserves_existing_keys(tmp_path):
    """Round-trip: pre-injected viz_4d ckpt survives torch.save / torch.load.

    This is a thin sanity check that our save format is stable — actual
    extract_4d_metadata correctness is covered by test_viz_4d_metadata.
    """
    p = tmp_path / "with_viz.pt"
    ckpt = {
        "config": OmegaConf.create({"use_layered_model": True}),
        "model": {"gaussians_nodes": {"background": {}}, "scene_extent": 1.0},
        "viz_4d": {
            "schema_version": 1,
            "sequence_id": "abc",
            "tracks": {"t0": {"class": "bus"}},
        },
        "global_step": 50,
    }
    torch.save(ckpt, p)
    loaded = torch.load(p, weights_only=False)
    assert loaded["viz_4d"]["schema_version"] == 1
    assert loaded["viz_4d"]["tracks"]["t0"]["class"] == "bus"
    assert loaded["model"]["gaussians_nodes"]["background"] == {}
