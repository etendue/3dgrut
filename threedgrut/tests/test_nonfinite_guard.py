# SPDX-License-Identifier: Apache-2.0
"""A1 — replace_nonfinite_pixels guard (renderer NaN-pixel containment)."""

from __future__ import annotations

import torch

from threedgrut.utils.misc import replace_nonfinite_pixels


def test_clean_input_passthrough_same_object():
    pred = torch.rand(1, 4, 4, 3)
    gt = torch.rand(1, 4, 4, 3)
    out, n = replace_nonfinite_pixels(pred, gt)
    assert n == 0
    assert out is pred  # no copy on the hot path


def test_single_nan_pixel_replaced_with_gt():
    pred = torch.rand(1, 4, 4, 3)
    gt = torch.rand(1, 4, 4, 3)
    pred[0, 2, 1, 0] = float("nan")
    out, n = replace_nonfinite_pixels(pred, gt)
    assert n == 1
    # the poisoned CHANNEL is replaced by gt; untouched channels keep pred
    assert out[0, 2, 1, 0] == gt[0, 2, 1, 0]
    assert torch.isfinite(out).all()


def test_inf_counts_as_nonfinite():
    pred = torch.rand(1, 2, 2, 3)
    gt = torch.zeros(1, 2, 2, 3)
    pred[0, 0, 0, :] = float("inf")
    pred[0, 1, 1, :] = float("-inf")
    out, n = replace_nonfinite_pixels(pred, gt)
    assert n == 2
    assert torch.isfinite(out).all()
    assert torch.equal(out[0, 0, 0], gt[0, 0, 0])


def test_where_guard_cannot_stop_grad_poison_hence_train_drops_batch():
    """Pins the hazard that motivates the TRAIN-side batch drop (trainer
    run_train_iter): when the upstream op that produced the NaN has a NaN
    local jacobian, autograd computes 0·NaN=NaN through torch.where's
    zero-gradient branch — i.e. loss-side substitution protects the loss
    VALUE but NOT the gradients. replace_nonfinite_pixels is therefore only
    used on no-backward eval paths; training drops the whole batch instead.
    If this assertion ever flips, torch changed where()/mul backward
    semantics and the batch drop could be revisited.
    """
    pred = torch.rand(1, 2, 2, 3, requires_grad=True)
    mask = torch.ones_like(pred)
    mask[0, 0, 0, :] = float("nan")
    bad = pred * mask  # NaN-producing upstream op with NaN local jacobian
    gt = torch.rand(1, 2, 2, 3)
    out, n = replace_nonfinite_pixels(bad, gt)
    assert n == 1
    assert torch.isfinite(out).all()  # loss VALUE is protected...
    (out - gt).abs().mean().backward()
    assert torch.isnan(pred.grad[0, 0, 0]).all(), "0·NaN=NaN leak expected — if gone, torch changed where() semantics"
