"""Regression tests for threedgrut.utils.misc.to_torch read-only handling.

to_torch converts DataLoader-decoded numpy arrays into torch tensors. Aux masks
decoded via ``np.asarray(Image.open(...))`` are non-writable; ``torch.from_numpy``
on them emits a per-worker UserWarning ("NumPy array is not writable...") that
floods render.py multi-camera held-out eval. And because ``.to("cpu")`` is a
no-op that keeps sharing the buffer, any downstream in-place write is undefined
behavior that can silently corrupt the dataset's cached sseg/mask arrays.

to_torch must copy read-only inputs (kills the warning + severs the shared
buffer) while keeping the zero-copy path for writable inputs (no added copy).
"""

import warnings

import numpy as np
import torch

from threedgrut.utils.misc import to_torch


def _readonly(arr: np.ndarray) -> np.ndarray:
    """Mark a numpy array non-writable, mirroring np.asarray(Image.open(...))."""
    arr.flags.writeable = False
    return arr


def test_readonly_input_emits_no_not_writable_warning():
    ro = _readonly(np.zeros((4, 4), dtype=np.uint8))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        to_torch(ro, device="cpu")
    not_writable = [str(w.message) for w in caught if "not writable" in str(w.message)]
    assert not_writable == [], f"unexpected non-writable warnings: {not_writable}"


def test_readonly_input_tensor_does_not_alias_source():
    ro = _readonly(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    original = np.array(ro, copy=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # isolate this test from the warning assertion above
        tensor = to_torch(ro, device="cpu")
    tensor.add_(100.0)  # in-place write must NOT reach the read-only source buffer
    assert np.array_equal(ro, original), "in-place write leaked into the read-only source array"


def test_writable_input_stays_zero_copy():
    wr = np.zeros((4, 4), dtype=np.float32)
    tensor = to_torch(wr, device="cpu")
    wr[0, 0] = 99.0  # mutate numpy; a shared (zero-copy) tensor must observe it
    assert tensor[0, 0].item() == 99.0, "writable input was copied — zero-copy path regressed"
