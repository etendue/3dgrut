# SPDX-License-Identifier: Apache-2.0
"""Mac-CPU smoke tests for ``threedgrut.correction.difix.DifixPostProcessor``.

These verify the *wrapper* contract without touching the real DiFix model or
its heavy NVIDIA stack (cosmos_predict2 / imaginaire / transformer_engine).
Real-forward smoke is intentionally left to GPU hosts — see
``third_party/Fixer/INSTALL.md``.

Run:
    pytest threedgrut/tests/test_difix_smoke.py -v
"""
from __future__ import annotations

import sys

import pytest
import torch

from threedgrut.correction.difix import DifixPostProcessor


def test_import_does_not_load_cosmos_predict2():
    """Importing the wrapper must NOT pull cosmos_predict2 into sys.modules.

    This guards the lazy-import contract: render.py imports
    ``threedgrut.correction.difix`` at startup, and Mac dev machines have no
    cosmos_predict2 installed. If anyone adds a top-level ``from third_party.
    Fixer.*`` import the test fails immediately.
    """
    assert "cosmos_predict2" not in sys.modules
    assert "transformer_engine" not in sys.modules


def test_disabled_forward_is_identity():
    """enabled=False short-circuits before any lazy init / dep import."""
    m = DifixPostProcessor(enabled=False)
    x = torch.rand(2, 32, 48, 3)
    y = m(x)
    assert torch.equal(y, x), "disabled forward must return the input as-is"
    # Confirm no heavy deps got imported as a side effect.
    assert "cosmos_predict2" not in sys.modules


def test_enabled_missing_ckpt_raises_runtime_error(tmp_path):
    """enabled=True + non-existent ckpt must fail loudly on first forward,
    not silently no-op (CLAUDE.md guardrail #5: never let metric mismatches
    pass through unnoticed)."""
    missing = tmp_path / "no_such.pkl"
    m = DifixPostProcessor(enabled=True, ckpt_path=str(missing))
    x = torch.rand(1, 16, 24, 3)
    with pytest.raises(RuntimeError, match="DiFix enabled but checkpoint missing"):
        m(x)


def test_forward_rejects_bad_shape():
    """Sanity check on input shape — must be (B,H,W,3) or (H,W,3).

    Uses enabled=False path so we don't try to lazy-init the model. Even
    disabled, we still want to reject obviously wrong inputs early."""
    m = DifixPostProcessor(enabled=True)
    # Wrong channel count — only valid if we actually enter forward's shape
    # check, which requires bypassing the lazy_init RuntimeError. The shape
    # validation runs *before* lazy_init in forward(), so this should hit
    # ValueError first.
    bad = torch.rand(1, 10, 10, 5)   # 5 channels, not 3
    with pytest.raises(ValueError, match=r"expects \(B,H,W,3\)"):
        m(bad)

    bad2 = torch.rand(10, 10)         # 2D
    with pytest.raises(ValueError, match=r"expects \(B,H,W,3\)"):
        m(bad2)


def test_ckpt_path_falls_back_to_hf_home(monkeypatch, tmp_path):
    """When ckpt_path is None, resolver uses HF_HOME (or default cache).

    Real layout from ``hf download nvidia/Fixer`` is
    ``<HF_HOME>/nvidia-Fixer/pretrained/pretrained_fixer.pkl`` (HF mirrors
    the repo's ``pretrained/`` subdirectory)."""
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    m = DifixPostProcessor(enabled=True, ckpt_path=None)
    expected = tmp_path / "nvidia-Fixer" / "pretrained" / "pretrained_fixer.pkl"
    assert m._resolve_ckpt_path() == expected
