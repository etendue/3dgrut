# SPDX-License-Identifier: Apache-2.0
"""E1.3-H1 unit tests for the --dataset-cameras held-out eval override.

The held-out protocol trains a 4-cam ckpt and evaluates on the excluded
camera. The ckpt-embedded conf has only the training cameras, and
``make_test`` builds the dataset from ``conf.dataset.camera_ids`` — so the
existing ``--eval-cameras`` batch filter cannot reach a camera the dataset
never loaded. ``apply_dataset_cameras_override`` rewrites the embedded conf.

Exposure trap (R-v4.8 / plan E1.3-H1): BilateralGrid indexes by train-time
camera_idx; overriding the camera set scrambles that mapping, so callers
must drop the exposure model whenever the override is active — held-out
numbers are read from the cc_* (per-frame affine) metrics instead.
"""
from __future__ import annotations

from omegaconf import OmegaConf

from threedgrut.render import apply_dataset_cameras_override


def _locked_conf(with_camera_ids: bool = True):
    d = {"dataset": {"camera_ids": ["cam_a", "cam_b"]} if with_camera_ids else {}}
    conf = OmegaConf.create(d)
    OmegaConf.set_struct(conf, True)
    return conf


def test_override_replaces_existing_camera_ids():
    conf = _locked_conf()
    applied = apply_dataset_cameras_override(conf, ["cam_heldout"])
    assert applied is True
    assert list(conf.dataset.camera_ids) == ["cam_heldout"]


def test_override_adds_key_on_struct_locked_conf_without_camera_ids():
    """Old ckpts may predate camera_ids: struct lock must not break the add."""
    conf = _locked_conf(with_camera_ids=False)
    applied = apply_dataset_cameras_override(conf, ["cam_x", "cam_y"])
    assert applied is True
    assert list(conf.dataset.camera_ids) == ["cam_x", "cam_y"]


def test_override_noop_on_none_or_empty():
    conf = _locked_conf()
    assert apply_dataset_cameras_override(conf, None) is False
    assert apply_dataset_cameras_override(conf, []) is False
    assert list(conf.dataset.camera_ids) == ["cam_a", "cam_b"]


def test_override_accepts_plain_dict_conf():
    """from_checkpoint's conf may behave dict-like; tolerate plain dicts."""
    conf = {"dataset": {"camera_ids": ["cam_a"]}}
    applied = apply_dataset_cameras_override(conf, ("cam_b",))
    assert applied is True
    assert conf["dataset"]["camera_ids"] == ["cam_b"]
