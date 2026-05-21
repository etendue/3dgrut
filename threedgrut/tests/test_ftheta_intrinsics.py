# SPDX-License-Identifier: Apache-2.0
"""T8.13: ftheta_dict_to_tensors numpy → torch conversion contract.

Lives outside engine.py because engine.py imports kaolin at top level and
won't load on a Mac. This pure-CPU helper is what
``Engine3DGRUT._trace_scene_mog`` calls to materialize the FTheta dict
for ``Batch.intrinsics_FThetaCameraModelParameters``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from threedgrut_playground.utils.ftheta_intrinsics import ftheta_dict_to_tensors


def _fake_ftheta_dict_numpy():
    return {
        "resolution":              np.array([1920, 1208], dtype=np.int64),
        "shutter_type":            "ROLLING_TOP_TO_BOTTOM",
        "principal_point":         np.array([960.0, 604.0], dtype=np.float32),
        "reference_poly":          "PIXELDIST_TO_ANGLE",
        "pixeldist_to_angle_poly": np.zeros(5, dtype=np.float32),
        "angle_to_pixeldist_poly": np.zeros(5, dtype=np.float32),
        "max_angle":               1.047,
        "linear_cde":              np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }


def test_none_passthrough():
    # FTheta disabled → None in, None out.
    assert ftheta_dict_to_tensors(None) is None


def test_numpy_int_array_to_int64():
    d = ftheta_dict_to_tensors(_fake_ftheta_dict_numpy())
    assert torch.is_tensor(d["resolution"])
    assert d["resolution"].dtype == torch.int64
    assert d["resolution"].tolist() == [1920, 1208]


def test_numpy_float_arrays_to_float32():
    d = ftheta_dict_to_tensors(_fake_ftheta_dict_numpy())
    for k in ("principal_point", "pixeldist_to_angle_poly",
              "angle_to_pixeldist_poly", "linear_cde"):
        assert torch.is_tensor(d[k]), k
        assert d[k].dtype == torch.float32, k


def test_scalar_pass_through():
    d = ftheta_dict_to_tensors(_fake_ftheta_dict_numpy())
    # str / float scalars stay as-is — tracer.py:471 reads K["shutter_type"]
    # and K["max_angle"] expecting str / float, not torch.Tensor.
    assert d["shutter_type"] == "ROLLING_TOP_TO_BOTTOM"
    assert d["reference_poly"] == "PIXELDIST_TO_ANGLE"
    assert d["max_angle"] == pytest.approx(1.047)


def test_existing_torch_tensor_moved_to_device():
    src = {
        "resolution": torch.tensor([1920, 1208], dtype=torch.int64),
        "principal_point": torch.tensor([960.0, 604.0]),
        "max_angle": 1.0,
        "shutter_type": "GLOBAL",
        "reference_poly": "PIXELDIST_TO_ANGLE",
        "pixeldist_to_angle_poly": torch.zeros(5),
        "angle_to_pixeldist_poly": torch.zeros(5),
        "linear_cde": torch.tensor([1.0, 0.0, 0.0]),
    }
    out = ftheta_dict_to_tensors(src, device="cpu")
    assert out["resolution"].device.type == "cpu"
    assert torch.equal(out["resolution"], torch.tensor([1920, 1208], dtype=torch.int64))


def test_all_8_required_keys_preserved():
    out = ftheta_dict_to_tensors(_fake_ftheta_dict_numpy())
    REQUIRED = {
        "resolution", "shutter_type", "principal_point", "reference_poly",
        "pixeldist_to_angle_poly", "angle_to_pixeldist_poly", "max_angle",
        "linear_cde",
    }
    assert set(out.keys()) == REQUIRED
