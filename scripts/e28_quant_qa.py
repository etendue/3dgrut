#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E2.8 Task 6 — offline quantitative QA for a packed (edited) ckpt.

The E2.8 driver (``e28_systematic_replace_pipeline.py``) stops at the QA
*sanity* gate (coverage + anti-smoke opacity). This adds the *quantitative*
QA: render the edited scene's novel views, measure NTA-IoU (does each
replacement vehicle still land inside its cuboid and read as a vehicle to a
detector) + FID/KID (does the edited render look like real captures), and —
optionally — run the DiffusionHarmonizer over the frames and report the
before/after delta. One ``qa_report.json`` per packed ckpt.

This is an **orchestrator**: it shells out to the already-verified
``render.py`` (frame dump), ``scripts/eval_frames_dir.py`` (NTA + FID/KID, same
口径 as the E1.2/E1.4 anchors), ``scripts/e21_harmonizer_batch_fix.py`` and
``scripts/e21_compare_metrics.py``. The only new *logic* — folding the per-mode
eval jsons into one report and deriving the cross-mode summary + harmonizer
delta — lives in :func:`build_qa_report` / :func:`summarize_replace_report` and
is unit-tested in ``threedgrut/tests/test_e28_quant_qa.py`` (pure CPU). The
subprocess wiring is integration-verified on inceptio GPU.

Usage (inceptio worktree)::

    python scripts/e28_quant_qa.py \
        --ckpt   ~/work/output/e28_run/packed_ckpt.pt \
        --path   ~/work/data/9ae151dc/pai_9ae151dc-...json \
        --out_dir ~/work/output/e28_run/quant \
        [--harmonizer_host 127.0.0.1 --harmonizer_port 59490]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from glob import glob
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------
# Pure aggregation logic (unit-tested, no torch / no GPU).
# --------------------------------------------------------------------------
def summarize_replace_report(report: list) -> dict:
    """Counts from ``replace_report.json`` (list of per-track assign rows)."""
    n_tracks = len(report)
    n_skipped = sum(1 for r in report if r.get("skipped"))
    by_class: dict = {}
    by_fallback_level: dict = {}
    for r in report:
        by_class[r["label_class"]] = by_class.get(r["label_class"], 0) + 1
        lvl = r.get("fallback_level")
        if lvl is not None:
            by_fallback_level[lvl] = by_fallback_level.get(lvl, 0) + 1
    return {
        "n_tracks": n_tracks,
        "n_skipped": n_skipped,
        "n_replaced": n_tracks - n_skipped,
        "by_class": by_class,
        "by_fallback_level": by_fallback_level,
    }


def _mean_over_modes(metrics_by_mode: dict, stem: str, modes: list) -> Optional[float]:
    """Mean of ``mean_novel_<stem>_<mode>`` across modes, ignoring absent/None.

    Returns None when no mode carries the metric (graceful: FID/KID can be
    dropped when a split is degenerate; NTA when no tracks are visible).
    """
    vals = []
    for m in modes:
        v = (metrics_by_mode.get(m) or {}).get(f"mean_novel_{stem}_{m}")
        if v is not None:
            vals.append(float(v))
    return statistics.mean(vals) if vals else None


def build_qa_report(
    raw_metrics: dict,
    modes: list,
    fixed_metrics: Optional[dict] = None,
    qa_sanity: Optional[dict] = None,
    replace_report: Optional[list] = None,
    ckpt_path: Optional[str] = None,
) -> dict:
    """Fold per-mode eval jsons + sanity + replace report into one report.

    ``raw_metrics`` / ``fixed_metrics`` map ``mode -> evaluate_frames(...)``
    output dict (keys ``mean_novel_<metric>_<mode>``). ``fixed_metrics`` is the
    harmonizer pass; when present the summary carries a before/after FID delta.
    """
    summary: dict = {
        "mean_novel_nta_iou": _mean_over_modes(raw_metrics, "nta_iou", modes),
        "mean_novel_fid": _mean_over_modes(raw_metrics, "fid", modes),
        "mean_novel_kid": _mean_over_modes(raw_metrics, "kid", modes),
        "harmonizer": None,
    }
    if fixed_metrics is not None:
        raw_fid = _mean_over_modes(raw_metrics, "fid", modes)
        fixed_fid = _mean_over_modes(fixed_metrics, "fid", modes)
        delta = None if (raw_fid is None or fixed_fid is None) else fixed_fid - raw_fid
        summary["harmonizer"] = {
            "raw_fid": raw_fid,
            "fixed_fid": fixed_fid,
            "fid_delta": delta,
            "improved": None if delta is None else delta < 0,
        }
    return {
        "ckpt": ckpt_path,
        "modes": modes,
        "raw": raw_metrics,
        "fixed": fixed_metrics,
        "qa_sanity": qa_sanity,
        "replace_summary": summarize_replace_report(replace_report) if replace_report else None,
        "summary": summary,
    }


# --------------------------------------------------------------------------
# Orchestration (integration-verified on inceptio GPU; not unit-tested).
# --------------------------------------------------------------------------
def _run(cmd: list, cwd: str) -> None:
    print(f"[e28-quant] $ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _eval_mode(py, repo_root, ckpt, path, frames_dir, mode, cameras, out_json) -> dict:
    cmd = [py, "scripts/eval_frames_dir.py",
           "--checkpoint", ckpt, "--path", path,
           "--frames-dir", str(frames_dir),
           "--frames-map", str(Path(frames_dir) / "frames_map.json"),
           "--mode", mode, "--nta-iou", "--kid",
           "--output", str(out_json)]
    if cameras:
        cmd += ["--cameras", cameras]
    _run(cmd, cwd=repo_root)
    with open(out_json) as f:
        return json.load(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="packed (edited) ckpt")
    ap.add_argument("--path", required=True, help="dataset manifest json (GT distribution)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--modes", nargs="+", default=["lateral_3m", "lateral_6m"])
    ap.add_argument("--cameras", default="camera_front_wide_120fov",
                    help="restrict eval cameras (lateral rig-offset = front cam only)")
    ap.add_argument("--harmonizer_host", default=None,
                    help="enable before/after harmonizer pass when host+port given")
    ap.add_argument("--harmonizer_port", type=int, default=None)
    ap.add_argument("--replace_report", default=None,
                    help="defaults to <ckpt dir>/replace_report.json if present")
    ap.add_argument("--qa_sanity", default=None,
                    help="defaults to <ckpt dir>/qa_sanity.json if present")
    ap.add_argument("--skip_render", action="store_true",
                    help="reuse existing novel_view frames under <out_dir>/render "
                         "(re-run eval/aggregate without re-rendering)")
    a = ap.parse_args(argv)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    py = sys.executable
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(a.ckpt).parent

    # 1) render the edited scene's novel views (frames only, no inline metrics).
    # render.py nests under <out_dir>/<exp>/<run_name>/ours_<step>/novel_view, so
    # glob recursively rather than assuming a fixed depth.
    render_dir = out / "render"

    def _find_novel_root():
        roots = sorted(glob(str(render_dir / "**" / "ours_*" / "novel_view"), recursive=True))
        return Path(roots[-1]) if roots else None

    if not (a.skip_render and _find_novel_root() is not None):
        render_cmd = [py, "render.py", "--checkpoint", a.ckpt, "--path", a.path,
                      "--out-dir", str(render_dir), "--render-only",
                      "--novel-view", "--novel-only", "--novel-save-n", "-1"]
        # lateral rig-offset novel poses only equal our per-camera lateral for
        # the front camera, and eval restricts to it — so render only those
        # cameras (≈5× fewer frames than the full ring).
        if a.cameras:
            render_cmd += ["--dataset-cameras", a.cameras]
        _run(render_cmd, cwd=repo_root)
    novel_root = _find_novel_root()
    if novel_root is None:
        raise SystemExit(f"[e28-quant] no novel_view frames under {render_dir} — render failed")
    print(f"[e28-quant] novel frames: {novel_root}", flush=True)

    # 2) eval raw frames: NTA-IoU + FID/KID per mode.
    raw_metrics = {
        m: _eval_mode(py, repo_root, a.ckpt, a.path, novel_root / m, m,
                      a.cameras, out / f"metrics_raw_{m}.json")
        for m in a.modes
    }

    # 3) optional harmonizer before/after.
    fixed_metrics = None
    if a.harmonizer_host and a.harmonizer_port:
        fixed_root = out / "fixed"
        _run([py, "scripts/e21_harmonizer_batch_fix.py",
              "--raw-dir", str(novel_root), "--fixed-dir", str(fixed_root),
              "--modes", *a.modes,
              "--host", a.harmonizer_host, "--port", str(a.harmonizer_port)], cwd=repo_root)
        fixed_metrics = {
            m: _eval_mode(py, repo_root, a.ckpt, a.path, fixed_root / m, m,
                          a.cameras, out / f"metrics_fixed_{m}.json")
            for m in a.modes
        }
        try:
            cmp_cmd = [py, "scripts/e21_compare_metrics.py",
                       "--before", *[str(out / f"metrics_raw_{m}.json") for m in a.modes],
                       "--after", *[str(out / f"metrics_fixed_{m}.json") for m in a.modes],
                       "--modes", *a.modes]
            print("[e28-quant] before/after table:", flush=True)
            subprocess.run(cmp_cmd, cwd=repo_root, check=False)
        except Exception as e:  # comparison table is advisory, not a gate
            print(f"[e28-quant] compare table skipped: {e}")

    # 4) fold in sanity + replace report, write qa_report.json.
    def _load(p, default):
        p = Path(p) if p else None
        return json.load(open(p)) if p and p.exists() else default

    qa_sanity = _load(a.qa_sanity or ckpt_dir / "qa_sanity.json", None)
    replace_report = _load(a.replace_report or ckpt_dir / "replace_report.json", None)

    report = build_qa_report(raw_metrics, a.modes, fixed_metrics=fixed_metrics,
                             qa_sanity=qa_sanity, replace_report=replace_report,
                             ckpt_path=a.ckpt)
    report_path = out / "qa_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    s = report["summary"]
    print(f"[e28-quant] ✓ qa_report → {report_path}\n"
          f"  NTA-IoU={s['mean_novel_nta_iou']}  FID={s['mean_novel_fid']}  "
          f"KID={s['mean_novel_kid']}  harmonizer={s['harmonizer']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
