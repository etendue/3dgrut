#!/usr/bin/env python3
"""Aggregate per-variant render timing into the comparison table.

Phase 2 analysis step. Parses each variant's timing (from serve-grpc
--metrics-output-dir json, or a render-grpc results.json / *.jsonl), drops
frame 0 (warmup spike), and reports median + p95 ms/frame, then the delta vs a
named baseline.

    python3 analyze.py \
      --baseline full \
      --variant full=/wk/profile/metrics_full \
      --variant no_rigid=/wk/profile/metrics_no_rigid \
      --variant no_person=/wk/profile/metrics_no_person \
      --variant drop_background=/wk/profile/metrics_drop_background \
      --counts /wk/profile/node_counts.json \
      --out /wk/profile/report.md

The per-frame timing field name is discovered (any numeric key containing
'ms'/'time'/'render'/'latency'). VALIDATE the picked field on inceptio against
one real metrics file (printed in --debug) — schemas vary by NRE version.
"""
import argparse
import glob
import json
import os
import statistics

TIME_HINTS = ("ms", "time", "render", "latency", "elapsed", "duration")


def _load_records(path):
    """Yield dict records from a dir of *.json / a .jsonl / a single .json."""
    files = []
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.json")) +
                       glob.glob(os.path.join(path, "*.jsonl")))
    elif os.path.exists(path):
        files = [path]
    for f in files:
        txt = open(f).read().strip()
        if f.endswith(".jsonl"):
            for line in txt.splitlines():
                if line.strip():
                    yield json.loads(line)
        else:
            d = json.loads(txt)
            if isinstance(d, list):
                yield from (x for x in d if isinstance(x, dict))
            elif isinstance(d, dict):
                # a results.json may carry a list under some key
                listy = [v for v in d.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
                if listy:
                    for lst in listy:
                        yield from lst
                else:
                    yield d


def _pick_field(records):
    """Choose the per-frame timing key: numeric, name hints at time, varies."""
    cand = {}
    for r in records:
        for k, v in r.items():
            if isinstance(v, (int, float)) and any(h in k.lower() for h in TIME_HINTS):
                cand.setdefault(k, []).append(float(v))
    # prefer a key that actually varies and looks like a per-frame value
    best, best_key = None, None
    for k, vals in cand.items():
        if len(vals) < 2:
            continue
        spread = (max(vals) - min(vals))
        score = (spread, len(vals))
        if best is None or score > best:
            best, best_key = score, k
    return best_key, cand


def _to_ms(vals):
    """Heuristic: if values look like seconds (<5), convert to ms."""
    if vals and statistics.median(vals) < 5:
        return [v * 1000.0 for v in vals]
    return vals


def summarize(path, debug=False):
    records = list(_load_records(path))
    if not records:
        return None
    key, cand = _pick_field(records)
    if debug:
        print(f"[debug] {path}: {len(records)} records; timing candidates: "
              + ", ".join(f"{k}(n={len(v)})" for k, v in cand.items()))
        print(f"[debug] sample record: {json.dumps(records[0])[:300]}")
    if not key:
        return None
    vals = _to_ms([float(r[key]) for r in records if key in r])
    vals = vals[1:] if len(vals) > 1 else vals  # drop frame 0 warmup
    if not vals:
        return None
    vals.sort()
    p95 = vals[min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))]
    return {"field": key, "n": len(vals), "median": statistics.median(vals),
            "p95": p95, "mean": statistics.fmean(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", default=[], metavar="label=path")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--counts", help="json {node: N_gaussians}")
    ap.add_argument("--out", help="write markdown table here")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    counts = json.load(open(args.counts)) if args.counts else {}
    rows = {}
    for spec in args.variant:
        label, _, path = spec.partition("=")
        rows[label] = summarize(path, debug=args.debug)

    base_med = None
    if args.baseline and rows.get(args.baseline):
        base_med = rows[args.baseline]["median"]

    lines = ["| 变体 | n_gaussians | median ms/帧 | p95 | mean | Δ vs base | 占比% | n帧 | field |",
             "|---|---|---|---|---|---|---|---|---|"]
    for label, s in rows.items():
        if not s:
            lines.append(f"| {label} | {counts.get(label,'?')} | (no timing parsed) | | | | | | |")
            continue
        delta = pct = ""
        if base_med is not None and label != args.baseline:
            d = base_med - s["median"]
            delta = f"{d:+.2f}"
            pct = f"{100*d/base_med:+.1f}" if base_med else ""
        lines.append(f"| {label} | {counts.get(label,'?')} | {s['median']:.2f} | {s['p95']:.2f} | "
                     f"{s['mean']:.2f} | {delta} | {pct} | {s['n']} | {s['field']} |")

    table = "\n".join(lines)
    print(table)
    if args.out:
        open(args.out, "w").write("# NuRec USDZ render-cost ablation\n\n" + table + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
