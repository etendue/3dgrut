#!/usr/bin/env python3
"""Settle frozen MCRO ownership and front-wide quality guards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _finite_number(mapping: dict, key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"required finite metric missing: {key}")
    return float(value)


def evaluate_guards(
    ownership: dict,
    metrics: dict,
    guard_config: dict,
    quality_baseline: dict | None = None,
) -> dict:
    own = ownership.get("summary", ownership)
    thresholds = guard_config["comparison"]
    reference = guard_config["r0_reference"]["six_cam"]
    if quality_baseline is None:
        quality_reference = {
            "front_cc_psnr_masked": _finite_number(reference, "front_cc_psnr_masked"),
            "front_road_crop_psnr": _finite_number(reference, "front_road_crop_psnr"),
            "front_road_crop_lpips": _finite_number(reference, "front_road_crop_lpips"),
        }
    else:
        quality_reference = {
            "front_cc_psnr_masked": _finite_number(quality_baseline, "mean_cc_psnr_masked"),
            "front_road_crop_psnr": _finite_number(quality_baseline, "mean_road_crop_psnr"),
            "front_road_crop_lpips": _finite_number(quality_baseline, "mean_road_crop_lpips"),
        }

    checks = []

    def add(name: str, actual: float, limit: float, op: str) -> None:
        passed = actual <= limit if op == "<=" else actual >= limit
        checks.append(
            {"name": name, "actual": actual, "operator": op, "limit": limit, "passed": passed}
        )

    bg_limit = _finite_number(reference, "bg_on_road_alpha_mean") * (
        1.0 - _finite_number(thresholds, "background_on_road_alpha_reduction_fraction_min")
    )
    add("background_on_road_alpha", _finite_number(own, "bg_on_road_alpha_mean"), bg_limit, "<=")
    add(
        "road_interior_alpha_p10",
        _finite_number(own, "road_coverage_p10"),
        _finite_number(thresholds, "road_interior_alpha_p10_min"),
        ">=",
    )
    add(
        "sky_on_road_energy",
        _finite_number(own, "sky_on_road_energy"),
        _finite_number(thresholds, "sky_on_road_energy_max"),
        "<=",
    )
    add(
        "front_cc_psnr_masked",
        _finite_number(metrics, "mean_cc_psnr_masked"),
        quality_reference["front_cc_psnr_masked"]
        - _finite_number(thresholds, "full_cc_psnr_masked_drop_db_max"),
        ">=",
    )
    add(
        "front_road_crop_psnr",
        _finite_number(metrics, "mean_road_crop_psnr"),
        quality_reference["front_road_crop_psnr"]
        - _finite_number(thresholds, "road_crop_psnr_drop_db_max"),
        ">=",
    )
    add(
        "front_road_crop_lpips",
        _finite_number(metrics, "mean_road_crop_lpips"),
        quality_reference["front_road_crop_lpips"]
        + _finite_number(thresholds, "road_crop_lpips_increase_max"),
        "<=",
    )
    return {
        "passed": all(check["passed"] for check in checks),
        "n_passed": sum(check["passed"] for check in checks),
        "n_checks": len(checks),
        "checks": checks,
    }


def _markdown(report: dict) -> str:
    lines = [
        f"# MCRO guard result: {'PASS' if report['passed'] else 'FAIL'}",
        "",
        "| guard | actual | requirement | result |",
        "|---|---:|---:|:---:|",
    ]
    for check in report["checks"]:
        lines.append(
            f"| {check['name']} | {check['actual']:.6f} | "
            f"{check['operator']} {check['limit']:.6f} | "
            f"{'PASS' if check['passed'] else 'FAIL'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ownership", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--guards", type=Path, required=True)
    parser.add_argument(
        "--quality-baseline",
        type=Path,
        help="Same-budget baseline metrics (required for 5s arms; omit for B6 frozen reference)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_guards(
        json.loads(args.ownership.read_text()),
        json.loads(args.metrics.read_text()),
        json.loads(args.guards.read_text()),
        json.loads(args.quality_baseline.read_text()) if args.quality_baseline else None,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "guard_result.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "guard_result.md").write_text(_markdown(report))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 3)


if __name__ == "__main__":
    main()
