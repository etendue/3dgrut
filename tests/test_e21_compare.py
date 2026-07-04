from scripts.e21_compare_metrics import build_table_rows, compare_metric


def test_compare_metric_delta_and_direction():
    row = compare_metric("FID", "lateral_3m", before=168.0, after=120.0, higher_is_better=False)
    assert row["delta"] == -48.0
    assert row["improved"] is True  # lower FID is better


def test_compare_metric_grad_corr_higher_is_better():
    row = compare_metric("lane_grad_corr", "lateral_6m", before=0.303, after=0.300, higher_is_better=True)
    assert round(row["delta"], 3) == -0.003
    assert row["improved"] is False


def test_build_table_rows_handles_missing_after():
    rows = build_table_rows(
        before={"mean_novel_fid_lateral_3m": 168.0},
        after={},  # fixed eval failed for this key
        modes=["lateral_3m"],
    )
    fid = [r for r in rows if r["metric"] == "FID" and r["mode"] == "lateral_3m"][0]
    assert fid["after"] is None
