"""Create a render-only checkpoint arm with one side of background Z hidden.

The tensor shapes stay unchanged so the checkpoint remains load-compatible.  Selected
background particles are made invisible by setting their density logits to a very
negative value; all geometry and optimizer state are otherwise preserved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def filter_checkpoint(
    checkpoint: dict,
    *,
    drop_side: str,
    threshold: float,
    density_logit: float = -100.0,
    alive_threshold: float = 0.005,
) -> dict:
    background = checkpoint["model"]["gaussians_nodes"]["background"]
    positions = background["positions"]
    density = background["density"]
    z = positions[:, 2]
    if drop_side == "lt":
        drop_mask = z < threshold
    elif drop_side == "gt":
        drop_mask = z > threshold
    else:
        raise ValueError(f"drop_side must be 'lt' or 'gt', got {drop_side!r}")

    opacity_before = torch.sigmoid(density.detach().squeeze(-1))
    alive_before = opacity_before > alive_threshold
    with torch.no_grad():
        density[drop_mask] = density_logit

    checkpoint["mcro_bg_z_filter"] = {
        "drop_side": drop_side,
        "threshold": float(threshold),
        "density_logit": float(density_logit),
        "n_background": int(z.numel()),
        "n_dropped": int(drop_mask.sum().item()),
        "fraction_dropped": float(drop_mask.float().mean().item()),
        "n_alive_dropped": int((drop_mask & alive_before).sum().item()),
    }
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--drop-side", choices=("lt", "gt"), required=True)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--density-logit", type=float, default=-100.0)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    filter_checkpoint(
        checkpoint,
        drop_side=args.drop_side,
        threshold=args.threshold,
        density_logit=args.density_logit,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out)
    print(json.dumps(checkpoint["mcro_bg_z_filter"], indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
