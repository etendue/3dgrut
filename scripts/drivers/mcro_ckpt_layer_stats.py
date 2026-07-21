#!/usr/bin/env python3
"""Offline per-layer particle statistics for layered 3DGUT checkpoints.

The driver intentionally loads checkpoints on CPU only.  It reports activated
scale (``exp(raw_scale)``) and opacity (``sigmoid(raw_density)``), matching the
model accessors used by the renderer and MCMC strategy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_ALIVE_THRESHOLD = 0.005
_REQUIRED_NODE_TENSORS = ("positions", "scale", "density")
_PERCENTILES = (10, 50, 90)


def _load_checkpoint(path: str) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6 has no weights_only argument.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {path} is not a dictionary payload")
    return checkpoint


def _threshold_from_parsed_yaml(ckpt_path: Path) -> float:
    """Read strategy.opacity_threshold when a sibling parsed.yaml provides it."""
    parsed_yaml = ckpt_path.parent / "parsed.yaml"
    if not parsed_yaml.is_file():
        return DEFAULT_ALIVE_THRESHOLD
    try:
        import yaml

        payload = yaml.safe_load(parsed_yaml.read_text(encoding="utf-8")) or {}
        threshold = payload.get("strategy", {}).get("opacity_threshold")
        if threshold is not None:
            return float(threshold)
    except (ImportError, OSError, TypeError, ValueError):
        pass
    return DEFAULT_ALIVE_THRESHOLD


def _as_float_array(value: Any, *, layer_name: str, tensor_name: str) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Layer '{layer_name}' tensor '{tensor_name}' is not a torch.Tensor")
    return value.detach().to(device="cpu", dtype=torch.float64).numpy()


def _percentiles(values: np.ndarray) -> tuple[float, float, float] | None:
    if values.size == 0:
        return None
    result = np.percentile(values, _PERCENTILES, axis=0)
    if result.ndim == 1:
        return tuple(float(x) for x in result)
    return tuple(result[index].astype(float).tolist() for index in range(len(_PERCENTILES)))  # type: ignore[return-value]


def _layer_stats(node: dict[str, Any], layer_name: str, alive_threshold: float) -> dict[str, Any]:
    missing = [name for name in _REQUIRED_NODE_TENSORS if name not in node]
    if missing:
        raise ValueError(f"Layer '{layer_name}' missing required tensor(s): {', '.join(missing)}")

    positions = _as_float_array(node["positions"], layer_name=layer_name, tensor_name="positions")
    raw_scale = _as_float_array(node["scale"], layer_name=layer_name, tensor_name="scale")
    raw_density = _as_float_array(node["density"], layer_name=layer_name, tensor_name="density")
    n_particles = int(positions.shape[0])
    if raw_scale.shape[0] != n_particles or raw_density.shape[0] != n_particles:
        raise ValueError(f"Layer '{layer_name}' has inconsistent particle tensor lengths")
    if raw_scale.ndim != 2 or raw_scale.shape[1] != 3:
        raise ValueError(f"Layer '{layer_name}' scale must have shape (N, 3), got {tuple(raw_scale.shape)}")

    scale = np.exp(raw_scale)
    opacity = 1.0 / (1.0 + np.exp(-raw_density.reshape(-1)))
    return {
        "n_particles": n_particles,
        "alive_ratio": float(np.mean(opacity > alive_threshold)) if n_particles else 0.0,
        "scale_p10": _percentiles(scale)[0] if n_particles else None,
        "scale_p50": _percentiles(scale)[1] if n_particles else None,
        "scale_p90": _percentiles(scale)[2] if n_particles else None,
        "opacity_p10": _percentiles(opacity)[0] if n_particles else None,
        "opacity_p50": _percentiles(opacity)[1] if n_particles else None,
        "opacity_p90": _percentiles(opacity)[2] if n_particles else None,
    }


def compute_layer_stats(ckpt_path: str, alive_threshold: float | None = None) -> dict[str, Any]:
    """Return per-particle-layer statistics from a checkpoint stored on disk."""
    path = Path(ckpt_path)
    threshold = _threshold_from_parsed_yaml(path) if alive_threshold is None else float(alive_threshold)
    checkpoint = _load_checkpoint(str(path))
    try:
        nodes = checkpoint["model"]["gaussians_nodes"]
    except KeyError as error:
        raise ValueError(f"Checkpoint {path} missing model.gaussians_nodes") from error
    if not isinstance(nodes, dict) or not nodes:
        raise ValueError(f"Checkpoint {path} has no particle-layer gaussians_nodes")

    return {
        "checkpoint": str(path),
        "alive_threshold": threshold,
        "layers": {
            str(layer_name): _layer_stats(node, str(layer_name), threshold)
            for layer_name, node in sorted(nodes.items())
        },
    }


def _delta(b_value: Any, a_value: Any) -> Any:
    if b_value is None or a_value is None:
        return None
    if isinstance(a_value, list) and isinstance(b_value, list):
        return [float(b - a) for a, b in zip(a_value, b_value, strict=True)]
    return float(b_value - a_value)


def compare_checkpoints(
    ckpt_a: str, ckpt_b: str, alive_threshold: float | None = None
) -> dict[str, Any]:
    """Compare checkpoint B against A; all delta values use ``B - A``."""
    stats_a = compute_layer_stats(ckpt_a, alive_threshold)
    stats_b = compute_layer_stats(ckpt_b, alive_threshold)
    layers_a = set(stats_a["layers"])
    layers_b = set(stats_b["layers"])
    if layers_a != layers_b:
        missing_from_a = sorted(layers_b - layers_a)
        missing_from_b = sorted(layers_a - layers_b)
        detail = []
        if missing_from_a:
            detail.append(f"missing from A: {', '.join(missing_from_a)}")
        if missing_from_b:
            detail.append(f"missing from B: {', '.join(missing_from_b)}")
        raise ValueError("Checkpoint layer mismatch (" + "; ".join(detail) + ")")

    layers: dict[str, dict[str, Any]] = {}
    for layer_name in sorted(layers_a):
        a_layer = stats_a["layers"][layer_name]
        b_layer = stats_b["layers"][layer_name]
        layers[layer_name] = {
            "a": a_layer,
            "b": b_layer,
            "delta": {key: _delta(b_layer[key], a_layer[key]) for key in a_layer},
        }
    return {
        "checkpoint_a": stats_a["checkpoint"],
        "checkpoint_b": stats_b["checkpoint"],
        "alive_threshold_a": stats_a["alive_threshold"],
        "alive_threshold_b": stats_b["alive_threshold"],
        "delta_convention": "b_minus_a",
        "layers": layers,
    }


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, list):
        return "[" + ", ".join(f"{item:.6g}" for item in value) + "]"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _render_markdown(result: dict[str, Any]) -> str:
    if "checkpoint_b" not in result:
        lines = [
            "# MCRO checkpoint layer statistics",
            "",
            f"Checkpoint: `{result['checkpoint']}`",
            f"Alive threshold: `{result['alive_threshold']:.6g}`",
            "",
            "| layer | particles | alive ratio | scale p10 / p50 / p90 | opacity p10 / p50 / p90 |",
            "| --- | ---: | ---: | --- | --- |",
        ]
        for layer, stats in result["layers"].items():
            scales = " / ".join(_format_value(stats[f"scale_p{p}"]) for p in _PERCENTILES)
            opacity = " / ".join(_format_value(stats[f"opacity_p{p}"]) for p in _PERCENTILES)
            lines.append(
                f"| {layer} | {stats['n_particles']} | {_format_value(stats['alive_ratio'])} | {scales} | {opacity} |"
            )
        return "\n".join(lines) + "\n"

    lines = [
        "# MCRO checkpoint layer comparison",
        "",
        f"A: `{result['checkpoint_a']}`",
        f"B: `{result['checkpoint_b']}`",
        "",
        "All deltas are **B − A**.",
        "",
        "| layer | A particles | B particles | Δ particles | A alive | B alive | Δ alive |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for layer, values in result["layers"].items():
        a = values["a"]
        b = values["b"]
        delta = values["delta"]
        lines.append(
            f"| {layer} | {a['n_particles']} | {b['n_particles']} | {delta['n_particles']} | "
            f"{_format_value(a['alive_ratio'])} | {_format_value(b['alive_ratio'])} | "
            f"{_format_value(delta['alive_ratio'])} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(result: dict[str, Any], out_dir: str) -> tuple[Path, Path]:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "layer_stats.json"
    markdown_path = output / "layer_stats.md"
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_markdown(result), encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-a", required=True, help="Checkpoint A path")
    parser.add_argument("--ckpt-b", help="Optional checkpoint B path for B - A deltas")
    parser.add_argument("--out", required=True, help="Directory for layer_stats.json and layer_stats.md")
    parser.add_argument("--alive-threshold", type=float, help="Override MCMC opacity alive threshold")
    args = parser.parse_args()

    result = (
        compare_checkpoints(args.ckpt_a, args.ckpt_b, args.alive_threshold)
        if args.ckpt_b
        else compute_layer_stats(args.ckpt_a, args.alive_threshold)
    )
    json_path, markdown_path = write_outputs(result, args.out)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
