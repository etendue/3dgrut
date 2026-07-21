#!/usr/bin/env python3
"""Best-effort parser for historical training telemetry.

Older runs have no per-camera sidecar; this tool extracts scalar TensorBoard
loss curves when available and labels per-camera telemetry as unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_run(run_dir: str) -> dict:
    root = Path(run_dir)
    result = {"run_dir": str(root), "per_camera_telemetry": "unavailable_in_historical_run", "scalar_tags": {}}
    event_files = sorted(root.glob("events.out.tfevents.*"))
    if not event_files:
        return result
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        accumulator = EventAccumulator(str(event_files[-1]), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            values = accumulator.Scalars(tag)
            if values:
                result["scalar_tags"][tag] = {
                    "n_points": len(values),
                    "first": {"step": values[0].step, "value": values[0].value},
                    "last": {"step": values[-1].step, "value": values[-1].value},
                }
    except Exception as error:
        result["parse_error"] = f"{type(error).__name__}: {error}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    Path(args.out).write_text(json.dumps(parse_run(args.run_dir), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
