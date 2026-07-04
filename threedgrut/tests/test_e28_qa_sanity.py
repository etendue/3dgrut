# SPDX-License-Identifier: Apache-2.0
"""E2.8 Task 4 — QA sanity gate (coverage + anti-smoke opacity floor + skips)."""

import torch

from threedgrut.layers.e28_replace import AssignRow, qa_sanity


def _node(opacity_logit, n_per_tid):
    tids = torch.cat([torch.full((n,), t) for t, n in n_per_tid.items()])
    N = tids.shape[0]
    return {"track_ids": tids, "density": torch.full((N, 1), float(opacity_logit)), "positions": torch.randn(N, 3)}


def test_full_coverage_passes():
    report = [AssignRow("0", "car", "a", 0, False), AssignRow("1", "bus", "b", 0, False)]
    after = _node(2.0, {0: 50, 1: 60})  # sigmoid(2.0)=0.88 正常 opacity
    qa = qa_sanity(after, report)
    assert qa["coverage"] == 1.0
    assert qa["n_skipped"] == 0
    assert qa["opacity_median"] > 0.3
    assert qa["passed"] is True


def test_degenerate_opacity_fails():
    # E2.8 inceptio 实测：正常 dynamic opacity 中位数 ~0.08-0.10，floor 校准为
    # 0.02 只挡 near-zero 退化注入（convention bug → opacity≈0）。
    report = [AssignRow("0", "car", "a", 0, False)]
    after = _node(-4.5, {0: 50})  # sigmoid(-4.5)=0.011 — near-zero 退化
    qa = qa_sanity(after, report)
    assert qa["opacity_median"] < 0.02
    assert qa["passed"] is False  # 退化注入被闸住


def test_normal_low_opacity_passes():
    # 0.10（NRE 重建正常 dynamic 水平）不再误判 fail（修正 plan 的 0.15 误标定）
    report = [AssignRow("0", "car", "a", 0, False)]
    after = _node(-2.1, {0: 50})  # sigmoid(-2.1)=0.109 — 正常水平
    qa = qa_sanity(after, report)
    assert 0.10 < qa["opacity_median"] < 0.12
    assert qa["passed"] is True


def test_opacity_scope_replaced_only():
    # replaced_slots 给定时只测替换粒子，不被未替换 recon(行人) 低 opacity 污染。
    tids = torch.tensor([0, 0, 0, 9, 9, 9, 9, 9])  # track 0 替换, track 9 recon
    dens = torch.tensor(
        [[2.0], [2.0], [2.0], [-6.0], [-6.0], [-6.0], [-6.0], [-6.0]]  # 替换粒子 sigmoid(2)=0.88
    )  # recon 近 0
    node = {"track_ids": tids, "density": dens, "positions": torch.randn(8, 3)}
    report = [AssignRow("0", "car", "a", 0, False)]
    qa_whole = qa_sanity(node, report)  # 整节点中位数被 recon 拉低
    qa_repl = qa_sanity(node, report, replaced_slots={0})
    assert qa_whole["opacity_median"] < 0.02  # whole-node 被污染
    assert qa_repl["opacity_median"] > 0.5  # replaced-only = 0.88
    assert qa_repl["passed"] is True
    assert qa_repl["opacity_scope"] == "replaced_only"


def test_skips_lower_coverage():
    report = [AssignRow("0", "car", "a", 0, False), AssignRow("1", "truck", None, None, True)]  # skip
    after = _node(2.0, {0: 50})
    qa = qa_sanity(after, report)
    assert qa["coverage"] == 0.5
    assert qa["n_skipped"] == 1
