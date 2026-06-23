#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""E3.3 G1 gate: verify the BEV grid was actually TRAINED (not frozen at init).

The grid is initialized at the road colour mean → every cell identical → std≈0.
After real training the cells must diverge (std > 0). A near-zero std means the
optimizer wiring is broken and the grid never trained — so the whole spike would
be measuring an untrained texture. Fail loud here (cheap) before the 6k A/B.

Usage: python scripts/e33_verify_grid.py <ckpt_last.pt>
"""
import sys

import torch


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: e33_verify_grid.py <ckpt_last.pt>")
        return 2
    ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    model = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    state = model.get("road_bev_state") if isinstance(model, dict) else None
    if state is None:
        print("FAIL: no 'road_bev_state' in ckpt — grid was not saved "
              "(get_model_parameters round-trip not wired or BEV was OFF).")
        return 1
    grid = state["grid"]
    std = float(grid.std())
    rng = float(grid.max() - grid.min())
    print(f"grid shape={tuple(grid.shape)} std={std:.6f} range={rng:.6f} "
          f"mean={float(grid.mean()):.4f}")
    if std < 1e-5:
        print(f"FAIL: grid std {std:.2e} ≈ 0 → still at init (all cells equal). "
              f"Optimizer not stepping the grid?")
        return 1
    print("PASS: BEV grid diverged from its uniform init → it IS being trained.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
