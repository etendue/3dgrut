"""E2.1: before/after comparison table from raw/fixed eval metrics jsons."""
import argparse, json

# (display, key_suffix, higher_is_better) — key = f"{suffix}_{mode}", matches
# eval_frames_dir.evaluate_frames output keys.
METRICS = [
    ("lane_grad_corr", "mean_novel_lane_grad_corr", True),
    ("lane_band_psnr", "mean_novel_lane_band_psnr", True),
    ("NTA_IoU",        "mean_novel_nta_iou",        True),
    ("FID",            "mean_novel_fid",            False),
    ("KID",            "mean_novel_kid",            False),
]


def compare_metric(metric, mode, before, after, higher_is_better):
    delta = None if (before is None or after is None) else round(after - before, 4)
    improved = None
    if delta is not None:
        improved = (delta > 0) if higher_is_better else (delta < 0)
    return {"metric": metric, "mode": mode, "before": before, "after": after,
            "delta": delta, "improved": improved, "higher_is_better": higher_is_better}


def build_table_rows(before, after, modes):
    rows = []
    for disp, suf, hib in METRICS:
        for mode in modes:
            k = f"{suf}_{mode}"
            rows.append(compare_metric(disp, mode, before.get(k), after.get(k), hib))
    return rows


def _markdown(rows):
    out = ["| 指标 | 档 | 修复前 | 修复后 | Δ | 方向 |", "|---|---|---|---|---|---|"]
    for r in rows:
        arrow = "—" if r["improved"] is None else ("✅↑" if r["improved"] else "⚠️↓")
        b = "—" if r["before"] is None else f'{r["before"]:.3f}'
        a = "—" if r["after"] is None else f'{r["after"]:.3f}'
        d = "—" if r["delta"] is None else f'{r["delta"]:+.3f}'
        out.append(f'| {r["metric"]} | {r["mode"]} | {b} | {a} | {d} | {arrow} |')
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", nargs="+", required=True, help="raw/anchor metrics json(s)")
    ap.add_argument("--after", nargs="+", required=True, help="fixed metrics json(s)")
    ap.add_argument("--modes", nargs="+", default=["lateral_3m", "lateral_6m"])
    a = ap.parse_args()
    before = {}; after = {}
    for f in a.before:
        before.update(json.load(open(f)))
    for f in a.after:
        after.update(json.load(open(f)))
    rows = build_table_rows(before, after, a.modes)
    print(_markdown(rows))


if __name__ == "__main__":
    main()
