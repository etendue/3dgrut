"""Default-off per-camera training telemetry."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping


class PerCameraTelemetry:
    """Accumulate scalar loss, gradient, and MCMC relocation summaries by camera."""

    def __init__(self) -> None:
        self._camera: dict[str, dict[str, object]] = {}
        self._relocations: dict[str, int] = defaultdict(int)

    def record_step(self, camera_id: str, losses: Mapping[str, float], grad_norm: float) -> None:
        entry = self._camera.setdefault(
            str(camera_id), {"n_steps": 0, "loss_sums": defaultdict(float), "grad_norm_sum": 0.0, "grad_norm_bins": defaultdict(int)},
        )
        entry["n_steps"] = int(entry["n_steps"]) + 1
        loss_sums = entry["loss_sums"]
        assert isinstance(loss_sums, defaultdict)
        for name, value in losses.items():
            loss_sums[str(name)] += float(value)
        entry["grad_norm_sum"] = float(entry["grad_norm_sum"]) + float(grad_norm)
        bins = entry["grad_norm_bins"]
        assert isinstance(bins, defaultdict)
        bins[self._grad_bin(float(grad_norm))] += 1

    def record_relocation(self, layer: str, n_relocated: int) -> None:
        self._relocations[str(layer)] += int(n_relocated)

    @staticmethod
    def _grad_bin(grad_norm: float) -> str:
        if grad_norm < 1e-6:
            return "[0,1e-6)"
        if grad_norm < 1e-3:
            return "[1e-6,1e-3)"
        if grad_norm < 1.0:
            return "[1e-3,1)"
        return "[1,inf)"

    def as_dict(self) -> dict:
        cameras = {}
        for camera_id, entry in sorted(self._camera.items()):
            n_steps = int(entry["n_steps"])
            loss_sums = entry["loss_sums"]
            assert isinstance(loss_sums, defaultdict)
            cameras[camera_id] = {
                "n_steps": n_steps,
                "mean_losses": {name: value / n_steps for name, value in sorted(loss_sums.items())},
                "mean_grad_norm": float(entry["grad_norm_sum"]) / n_steps,
                "grad_norm_bins": dict(sorted(entry["grad_norm_bins"].items())),
            }
        return {"cameras": cameras, "relocations_by_layer": dict(sorted(self._relocations.items()))}

    def dump(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
