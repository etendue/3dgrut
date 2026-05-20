# SPDX-License-Identifier: Apache-2.0
"""Stage 6 unit tests: per-camera ExposureModel."""
from __future__ import annotations

import pytest
import torch


def test_zero_init_is_identity():
    from threedgrut.correction import ExposureModel

    m = ExposureModel(num_camera=5)
    img = torch.rand(1, 4, 5, 3)
    # All cameras → identity at zero init (output clamped to [0, 1]; img
    # already in [0, 1] so clamp is a no-op).
    for i in range(5):
        out = m(i, img)
        assert torch.allclose(out, img, atol=1e-6)


def test_per_camera_grad_isolation():
    """Backprop through forward(cam=0, ...) must NOT touch params at cam>0."""
    from threedgrut.correction import ExposureModel

    m = ExposureModel(num_camera=4)
    img = torch.rand(1, 3, 3, 3)
    out = m(0, img)
    out.sum().backward()
    # exposure_a / exposure_b come back as zero gradient (or None) for other
    # cameras, since index-0 forward doesn't see them.
    for i in range(1, 4):
        g_a = m.exposure_a.grad[i]
        g_b = m.exposure_b.grad[i]
        assert torch.allclose(g_a, torch.zeros_like(g_a))
        assert torch.allclose(g_b, torch.zeros_like(g_b))


def test_clamp_to_unit_range():
    """Large positive ``a`` / large ``b`` outputs must still clamp into [0,1]."""
    from threedgrut.correction import ExposureModel

    m = ExposureModel(num_camera=2)
    with torch.no_grad():
        m.exposure_a.fill_(5.0)   # exp(5) ≈ 148× gain
        m.exposure_b.fill_(2.0)
    img = torch.rand(1, 4, 4, 3)
    out = m(0, img)
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_invalid_camera_idx_raises():
    from threedgrut.correction import ExposureModel

    m = ExposureModel(num_camera=2)
    img = torch.rand(1, 2, 2, 3)
    with pytest.raises(IndexError):
        m(2, img)
    with pytest.raises(IndexError):
        m(-1, img)


def test_constructor_rejects_zero_cameras():
    from threedgrut.correction import ExposureModel

    with pytest.raises(ValueError):
        ExposureModel(num_camera=0)


def test_state_dict_roundtrip():
    """ckpt save/load must preserve learned a/b bit-for-bit."""
    from threedgrut.correction import ExposureModel

    m1 = ExposureModel(num_camera=3)
    with torch.no_grad():
        m1.exposure_a.copy_(torch.tensor([[0.1], [0.2], [-0.3]]))
        m1.exposure_b.copy_(torch.tensor([[0.01], [-0.02], [0.05]]))
    state = m1.state_dict()

    m2 = ExposureModel(num_camera=3)
    m2.load_state_dict(state)
    assert torch.equal(m1.exposure_a, m2.exposure_a)
    assert torch.equal(m1.exposure_b, m2.exposure_b)
