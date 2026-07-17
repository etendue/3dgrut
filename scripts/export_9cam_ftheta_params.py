#!/usr/bin/env python3
"""Export the frozen b6a9 nine-camera FTheta parameter artifact as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.pin_ftheta_camera_survey import survey_bundle  # noqa: E402


def build_artifact(calibration_bundle: dict) -> dict:
    survey = survey_bundle(calibration_bundle)
    return {
        "schema_version": 1,
        "provenance": survey["provenance"],
        "fitter_version": survey["fitter_version"],
        "fit_gate_thresholds": survey["fit_gate_thresholds"],
        "camera_order": survey["camera_order"],
        "all_cameras_passed": survey["all_cameras_passed"],
        "failed_cameras": survey["failed_cameras"],
        "cameras": {
            camera_id: {
                "source_model_type": survey["cameras"][camera_id]["source_model_type"],
                "source_parameters_type": survey["cameras"][camera_id]["source_parameters_type"],
                "source_calibration_sha256": survey["cameras"][camera_id][
                    "source_calibration_sha256"
                ],
                "fitter_version": survey["cameras"][camera_id]["fitter_version"],
                "ftheta_parameters": survey["cameras"][camera_id]["ftheta_parameters"],
                "fit_metrics": survey["cameras"][camera_id]["fit_metrics"],
                "gate": survey["cameras"][camera_id]["gate"],
            }
            for camera_id in survey["camera_order"]
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibrations",
        type=Path,
        default=_REPO_ROOT / "scripts" / "pin_ftheta_b6a9_calibs.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    with args.calibrations.open() as handle:
        calibrations = json.load(handle)
    artifact = build_artifact(calibrations)
    rendered = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(f"wrote {args.output}", file=sys.stderr)
    return 0 if artifact["all_cameras_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
