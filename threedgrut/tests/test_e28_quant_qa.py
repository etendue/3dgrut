# SPDX-License-Identifier: Apache-2.0
"""E2.8 Task 6 — qa_report.json aggregation (pure CPU).

The GPU-heavy parts (render.py dump, eval_frames_dir NTA/FID, harmonizer batch)
are subprocess calls verified on inceptio. Here we pin the only new *logic*:
how per-mode eval jsons + qa_sanity + replace_report fold into one qa_report,
and how the cross-mode summary + harmonizer before/after delta are derived.
Keys mirror eval_frames_dir.evaluate_frames output (``mean_novel_<metric>_<mode>``).
"""

from scripts.e28_quant_qa import build_qa_report, summarize_replace_report


def _raw(nta_3m=0.6, fid_3m=10.0, nta_6m=0.8, fid_6m=20.0):
    return {
        "lateral_3m": {
            "mode": "lateral_3m",
            "mean_novel_nta_iou_lateral_3m": nta_3m,
            "mean_novel_fid_lateral_3m": fid_3m,
            "mean_novel_kid_lateral_3m": 0.05,
        },
        "lateral_6m": {
            "mode": "lateral_6m",
            "mean_novel_nta_iou_lateral_6m": nta_6m,
            "mean_novel_fid_lateral_6m": fid_6m,
            "mean_novel_kid_lateral_6m": 0.07,
        },
    }


MODES = ["lateral_3m", "lateral_6m"]


def test_summarize_replace_report_counts_by_class_and_fallback():
    report = [
        {"track": "24", "label_class": "automobile", "chosen_asset": "a", "fallback_level": 1, "skipped": False},
        {"track": "54", "label_class": "automobile", "chosen_asset": "a", "fallback_level": 1, "skipped": False},
        {"track": "405", "label_class": "bus", "chosen_asset": None, "fallback_level": 0, "skipped": True},
    ]
    s = summarize_replace_report(report)
    assert s["n_tracks"] == 3
    assert s["n_skipped"] == 1
    assert s["n_replaced"] == 2
    assert s["by_class"] == {"automobile": 2, "bus": 1}
    assert s["by_fallback_level"] == {0: 1, 1: 2}


def test_summary_means_nta_and_fid_over_modes():
    rep = build_qa_report(_raw(), MODES)
    assert rep["summary"]["mean_novel_nta_iou"] == 0.7  # (0.6 + 0.8) / 2
    assert rep["summary"]["mean_novel_fid"] == 15.0  # (10 + 20) / 2
    # per-mode raw metrics are preserved verbatim for the audit trail
    assert rep["raw"]["lateral_3m"]["mean_novel_fid_lateral_3m"] == 10.0
    assert rep["modes"] == MODES


def test_missing_metric_key_yields_none_not_crash():
    raw = _raw()
    del raw["lateral_3m"]["mean_novel_fid_lateral_3m"]
    del raw["lateral_6m"]["mean_novel_fid_lateral_6m"]
    rep = build_qa_report(raw, MODES)
    # FID absent in both modes → summary None (graceful), NTA still computed
    assert rep["summary"]["mean_novel_fid"] is None
    assert rep["summary"]["mean_novel_nta_iou"] == 0.7


def test_harmonizer_fid_improvement_flag():
    raw = _raw(fid_3m=20.0, fid_6m=30.0)  # raw mean FID = 25
    fixed = _raw(fid_3m=12.0, fid_6m=18.0)  # fixed mean FID = 15 (lower = better)
    rep = build_qa_report(raw, MODES, fixed_metrics=fixed)
    harm = rep["summary"]["harmonizer"]
    assert harm["raw_fid"] == 25.0
    assert harm["fixed_fid"] == 15.0
    assert harm["fid_delta"] == -10.0
    assert harm["improved"] is True


def test_no_harmonizer_section_when_fixed_absent():
    rep = build_qa_report(_raw(), MODES)
    assert rep["summary"]["harmonizer"] is None
    assert rep["fixed"] is None


def test_qa_sanity_and_replace_report_folded_through():
    qa = {"coverage": 1.0, "passed": True, "opacity_median": 0.101}
    report = [{"track": "24", "label_class": "automobile", "chosen_asset": "a", "fallback_level": 1, "skipped": False}]
    rep = build_qa_report(_raw(), MODES, qa_sanity=qa, replace_report=report)
    assert rep["qa_sanity"]["passed"] is True
    assert rep["replace_summary"]["n_replaced"] == 1
