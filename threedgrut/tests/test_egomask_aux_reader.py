# SPDX-License-Identifier: Apache-2.0
"""P0.2 unit tests for EgomaskAuxReader + resolve_ego_valid_mask (Phase C Task 1).

Pure Mac-CPU tests: a fake zarr root (nested _FakeGroup/_FakeArray) is fed to the
reader by monkeypatching ``aux_readers._open_itar_zarr``, so no ncore SDK / real
itar is needed (conftest.py already stubs ncore). Frame storage is exercised in
both forms nre-tools emits: a plain ``(H, W) uint8 {0, 255}`` array and 0-D PNG
bytes.

Contract under test (spec §3 P0.2):
  * ``read_static_mask(cam)`` = per-camera **union** over all frames (any frame
    marking a pixel ego -> True), as ``(H, W)`` bool.
  * ``resolve_ego_valid_mask`` three branches:
      1. SDK mask present AND non-zero -> convert("L") -> dilate -> logical_not.
      2. else clip_dir has an egomask itar whose reader has_camera -> itar union
         -> dilate -> logical_not.
      3. else all-True with shape == resolution_hw.
  * ``dilation_iters == 0`` means "no dilation" (scipy would otherwise dilate to
    convergence for iterations < 1); binary masks -> exact equality asserted.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from threedgrut.datasets import aux_readers
from threedgrut.datasets.aux_readers import (
    EgomaskAuxReader,
    resolve_ego_valid_mask,
    resolve_ego_valid_mask_with_source,
)

H, W = 4, 4


# --------------------------------------------------------------------------- #
# fake zarr primitives (mimic the minimal zarr.Group / zarr.Array surface the
# reader touches: group_keys / array_keys / __getitem__ / .shape / arr[()] /
# arr[...])
# --------------------------------------------------------------------------- #
class _FakeArray:
    def __init__(self, data, png_bytes: bool = False) -> None:
        arr = np.asarray(data, dtype=np.uint8)
        if png_bytes:
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            self._bytes = buf.getvalue()
            self.shape = ()
        else:
            self._arr = arr
            self.shape = arr.shape
            self._bytes = None

    def __getitem__(self, idx):
        return self._bytes if self._bytes is not None else self._arr


class _FakeGroup:
    def __init__(self, groups=None, arrays=None) -> None:
        self._groups = dict(groups or {})
        self._arrays = dict(arrays or {})

    def group_keys(self):
        return list(self._groups.keys())

    def array_keys(self):
        return list(self._arrays.keys())

    def __contains__(self, k):
        return k in self._groups or k in self._arrays

    def __getitem__(self, k):
        return self._groups[k] if k in self._groups else self._arrays[k]


def _frame(coords):
    m = np.zeros((H, W), dtype=np.uint8)
    for r, c in coords:
        m[r, c] = 255
    return m


def _build_root(png_bytes: bool = False):
    """camA: two frames marking DISJOINT regions (union = both); camB: all zero.

    Layout mirrors nre-tools aux: aux/ego_mask/<camera_id>/<timestamp>.
    """
    camA = _FakeGroup(
        arrays={
            "1000": _FakeArray(_frame([(0, 0), (0, 1)]), png_bytes=png_bytes),
            "2000": _FakeArray(_frame([(3, 3)]), png_bytes=png_bytes),
        }
    )
    camB = _FakeGroup(
        arrays={
            "1000": _FakeArray(_frame([]), png_bytes=png_bytes),
            "2000": _FakeArray(_frame([]), png_bytes=png_bytes),
        }
    )
    ego = _FakeGroup(groups={"camA": camA, "camB": camB})
    aux = _FakeGroup(groups={"ego_mask": ego})
    return _FakeGroup(groups={"aux": aux})


# expected union for camA: (0,0), (0,1), (3,3)
CAMA_UNION = np.zeros((H, W), dtype=bool)
CAMA_UNION[0, 0] = CAMA_UNION[0, 1] = CAMA_UNION[3, 3] = True


@pytest.fixture
def patched_reader(monkeypatch):
    root = _build_root()
    monkeypatch.setattr(aux_readers, "_open_itar_zarr", lambda p: root)
    return EgomaskAuxReader("dummy.egomask.zarr.itar")


# --------------------------------------------------------------------------- #
# EgomaskAuxReader
# --------------------------------------------------------------------------- #
def test_read_static_mask_is_exact_frame_union(patched_reader):
    m = patched_reader.read_static_mask("camA")
    assert m.dtype == bool
    assert m.shape == (H, W)
    assert np.array_equal(m, CAMA_UNION)


def test_all_zero_camera_is_all_false(patched_reader):
    m = patched_reader.read_static_mask("camB")
    assert m.shape == (H, W)
    assert not m.any()


def test_camera_ids_sorted(patched_reader):
    assert patched_reader.camera_ids() == ["camA", "camB"]


def test_missing_camera_has_camera_false_and_raises(patched_reader):
    assert patched_reader.has_camera("camA")
    assert not patched_reader.has_camera("camX")
    with pytest.raises(KeyError):
        patched_reader.read_static_mask("camX")


def test_png_bytes_storage_decodes_same_union(monkeypatch):
    root = _build_root(png_bytes=True)
    monkeypatch.setattr(aux_readers, "_open_itar_zarr", lambda p: root)
    reader = EgomaskAuxReader("dummy.egomask.zarr.itar")
    assert np.array_equal(reader.read_static_mask("camA"), CAMA_UNION)
    assert not reader.read_static_mask("camB").any()


# --------------------------------------------------------------------------- #
# resolve_ego_valid_mask — branch 1 (SDK mask present & non-zero)
# --------------------------------------------------------------------------- #
def test_resolve_sdk_nonzero_does_not_touch_itar(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("itar fallback must not be reached when SDK mask is non-zero")

    monkeypatch.setattr(aux_readers, "discover_aux_path", _boom)
    sdk = np.zeros((H, W), dtype=np.uint8)
    sdk[1, 1] = 255
    valid = resolve_ego_valid_mask(
        Image.fromarray(sdk), clip_dir="explode-if-used", camera_id="camA", resolution_hw=(H, W), dilation_iters=0
    )
    expected = np.ones((H, W), dtype=bool)
    expected[1, 1] = False  # dilation_iters=0 -> no dilation
    assert np.array_equal(valid, expected)


# --------------------------------------------------------------------------- #
# resolve_ego_valid_mask — branch 2 (fallback to egomask itar)
# --------------------------------------------------------------------------- #
def test_resolve_sdk_none_uses_itar_no_dilation(monkeypatch, tmp_path):
    (tmp_path / "clip.aux.egomask.zarr.itar").touch()
    root = _build_root()
    monkeypatch.setattr(aux_readers, "_open_itar_zarr", lambda p: root)
    valid = resolve_ego_valid_mask(None, clip_dir=tmp_path, camera_id="camA", resolution_hw=(H, W), dilation_iters=0)
    assert np.array_equal(valid, np.logical_not(CAMA_UNION))


def test_resolve_sdk_all_zero_falls_back_to_itar(monkeypatch, tmp_path):
    (tmp_path / "clip.aux.egomask.zarr.itar").touch()
    root = _build_root()
    monkeypatch.setattr(aux_readers, "_open_itar_zarr", lambda p: root)
    sdk_zero = Image.fromarray(np.zeros((H, W), dtype=np.uint8))
    valid = resolve_ego_valid_mask(
        sdk_zero, clip_dir=tmp_path, camera_id="camA", resolution_hw=(H, W), dilation_iters=0
    )
    assert np.array_equal(valid, np.logical_not(CAMA_UNION))


def test_resolve_itar_applies_dilation(monkeypatch, tmp_path):
    (tmp_path / "clip.aux.egomask.zarr.itar").touch()
    root = _build_root()
    monkeypatch.setattr(aux_readers, "_open_itar_zarr", lambda p: root)
    valid = resolve_ego_valid_mask(None, clip_dir=tmp_path, camera_id="camA", resolution_hw=(H, W), dilation_iters=1)
    expected = np.logical_not(ndimage.binary_dilation(CAMA_UNION, iterations=1))
    assert np.array_equal(valid, expected)
    # dilation genuinely shrinks the valid region vs the un-dilated case
    assert valid.sum() < np.logical_not(CAMA_UNION).sum()


def test_resolve_itar_present_but_camera_absent_all_true(monkeypatch, tmp_path):
    (tmp_path / "clip.aux.egomask.zarr.itar").touch()
    root = _build_root()
    monkeypatch.setattr(aux_readers, "_open_itar_zarr", lambda p: root)
    valid = resolve_ego_valid_mask(None, clip_dir=tmp_path, camera_id="camNONE", resolution_hw=(H, W), dilation_iters=30)
    assert valid.shape == (H, W)
    assert valid.all()


# --------------------------------------------------------------------------- #
# resolve_ego_valid_mask — branch 3 (nothing -> all valid)
# --------------------------------------------------------------------------- #
def test_resolve_no_sdk_no_itar_all_true(tmp_path):
    valid = resolve_ego_valid_mask(None, clip_dir=tmp_path, camera_id="camA", resolution_hw=(H, W), dilation_iters=30)
    assert valid.shape == (H, W)
    assert valid.all()


def test_resolve_clip_dir_none_all_true():
    valid = resolve_ego_valid_mask(None, clip_dir=None, camera_id="camA", resolution_hw=(5, 7), dilation_iters=30)
    assert valid.shape == (5, 7)
    assert valid.dtype == bool
    assert valid.all()


# --------------------------------------------------------------------------- #
# resolve_ego_valid_mask_with_source — wire semantics for datasetNcore
# integration (P0.2 Task 2). The caller (datasetNcore L429-440) needs to know
# which branch produced the mask so it can log ``[P0.2] ego mask via aux itar
# fallback: <cam> coverage=<pct>%`` only when the itar fallback actually fired.
# --------------------------------------------------------------------------- #
def _build_root_multi_cam(H_: int, W_: int):
    """Fake root with 4 cameras mirroring the b6a9 fallback shape (varying mask
    coverage across cameras; one all-zero camera to cover the trivial branch).
    """

    def _f(coords):
        m = np.zeros((H_, W_), dtype=np.uint8)
        for r, c in coords:
            m[r, c] = 255
        return m

    cams = {
        "camera_cross_left_120fov": _FakeGroup(
            arrays={
                "1000": _FakeArray(_f([(0, 0), (0, 1), (1, 0)])),
                "2000": _FakeArray(_f([(0, 1), (1, 1)])),
            }
        ),
        "camera_cross_right_120fov": _FakeGroup(
            arrays={"1000": _FakeArray(_f([(H_ - 1, W_ - 1), (H_ - 1, W_ - 2)]))}
        ),
        "camera_right_wide_120fov": _FakeGroup(
            arrays={"1000": _FakeArray(_f([(H_ // 2, W_ // 2)]))}
        ),
        "camera_back_rear_wide_120fov": _FakeGroup(arrays={"1000": _FakeArray(_f([]))}),
    }
    ego = _FakeGroup(groups=dict(cams))
    aux = _FakeGroup(groups={"ego_mask": ego})
    return _FakeGroup(groups={"aux": aux}), list(cams.keys())


def test_wire_fallback_multi_camera_pixel_count(monkeypatch, tmp_path):
    """P0.2 wire: SDK all-zero + itar covering 4 cameras → each camera's valid
    pixel count = H*W − dilate(union).sum() (mirrors datasetNcore L429-440 with
    the dataset-default 30 dilation iters). source=='itar' unlocks the log."""
    H_, W_ = 8, 8
    (tmp_path / "clip.aux.egomask.zarr.itar").touch()
    root, camera_ids = _build_root_multi_cam(H_, W_)
    monkeypatch.setattr(aux_readers, "_open_itar_zarr", lambda p: root)
    dilation = 30  # datasetNcore default (n_camera_mask_dilation_iterations)

    sdk_zero = Image.fromarray(np.zeros((H_, W_), dtype=np.uint8))
    for cam_id in camera_ids:
        valid, source = resolve_ego_valid_mask_with_source(
            sdk_zero,
            clip_dir=tmp_path,
            camera_id=cam_id,
            resolution_hw=(H_, W_),
            dilation_iters=dilation,
        )
        raw_union = EgomaskAuxReader(str(tmp_path / "clip.aux.egomask.zarr.itar")).read_static_mask(cam_id)
        expected_ego = ndimage.binary_dilation(raw_union, iterations=dilation) if raw_union.any() else raw_union
        assert valid.shape == (H_, W_)
        assert valid.dtype == bool
        assert valid.sum() == (H_ * W_) - int(expected_ego.sum())
        assert source == "itar"


def test_wire_pai_line_sdk_none_no_itar_all_valid(tmp_path):
    """P0.2 regression: PAI clip (SDK ``get_mask_images().get("ego")`` returns
    None and no aux egomask itar exists) → all-True mask, byte-identical to the
    pre-P0.2 ``np.ones((h, w), dtype=bool)`` datasetNcore path. source=='none'
    means no log line is emitted."""
    H_, W_ = 6, 10
    valid, source = resolve_ego_valid_mask_with_source(
        None,
        clip_dir=tmp_path,
        camera_id="camera_front_wide_120fov",
        resolution_hw=(H_, W_),
        dilation_iters=30,
    )
    assert valid.shape == (H_, W_)
    assert valid.dtype == bool
    assert source == "none"
    assert np.array_equal(valid, np.ones((H_, W_), dtype=bool))


def test_wire_source_is_sdk_when_sdk_nonzero(monkeypatch):
    """SDK non-zero path reports source='sdk' (no [P0.2] fallback log)."""

    def _boom(*a, **k):
        raise AssertionError("itar fallback must not be reached when SDK mask is non-zero")

    monkeypatch.setattr(aux_readers, "discover_aux_path", _boom)
    sdk = np.zeros((H, W), dtype=np.uint8)
    sdk[2, 2] = 255
    _, source = resolve_ego_valid_mask_with_source(
        Image.fromarray(sdk),
        clip_dir="explode-if-used",
        camera_id="camA",
        resolution_hw=(H, W),
        dilation_iters=0,
    )
    assert source == "sdk"


def test_resolve_ego_valid_mask_still_returns_ndarray(monkeypatch, tmp_path):
    """Task 1 API preserved: legacy ``resolve_ego_valid_mask`` still returns a
    single ndarray (thin wrapper over the with_source variant)."""
    (tmp_path / "clip.aux.egomask.zarr.itar").touch()
    root = _build_root()
    monkeypatch.setattr(aux_readers, "_open_itar_zarr", lambda p: root)
    valid = resolve_ego_valid_mask(
        None, clip_dir=tmp_path, camera_id="camA", resolution_hw=(H, W), dilation_iters=0
    )
    assert isinstance(valid, np.ndarray)
    assert np.array_equal(valid, np.logical_not(CAMA_UNION))
