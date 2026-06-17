# SPDX-License-Identifier: Apache-2.0
"""E2.8 Task 4 — QA sanity gate (coverage + anti-smoke opacity floor + skips)."""
import torch
from threedgrut.layers.e28_replace import qa_sanity, AssignRow


def _node(opacity_logit, n_per_tid):
    tids = torch.cat([torch.full((n,), t) for t, n in n_per_tid.items()])
    N = tids.shape[0]
    return {"track_ids": tids,
            "density": torch.full((N, 1), float(opacity_logit)),
            "positions": torch.randn(N, 3)}


def test_full_coverage_passes():
    report = [AssignRow("0", "car", "a", 0, False),
              AssignRow("1", "bus", "b", 0, False)]
    after = _node(2.0, {0: 50, 1: 60})   # sigmoid(2.0)=0.88 正常 opacity
    qa = qa_sanity(after, report)
    assert qa["coverage"] == 1.0
    assert qa["n_skipped"] == 0
    assert qa["opacity_median"] > 0.3
    assert qa["passed"] is True


def test_smoke_opacity_fails():
    report = [AssignRow("0", "car", "a", 0, False)]
    after = _node(-2.1, {0: 50})         # sigmoid(-2.1)=0.109 ≈ E2.7 烟雾区
    qa = qa_sanity(after, report)
    assert qa["opacity_median"] < 0.15
    assert qa["passed"] is False         # 烟雾回归被闸住


def test_skips_lower_coverage():
    report = [AssignRow("0", "car", "a", 0, False),
              AssignRow("1", "truck", None, None, True)]   # skip
    after = _node(2.0, {0: 50})
    qa = qa_sanity(after, report)
    assert qa["coverage"] == 0.5
    assert qa["n_skipped"] == 1
