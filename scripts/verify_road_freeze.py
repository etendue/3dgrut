#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""E3.2.5 freeze/clamp end-to-end verify on a trained ckpt.

Walks the ckpt for road-layer rotation/scale tensors and checks:
  - rotation tilt-from-identity ~0  → freeze_rotation_grad locked normal vertical
  - z-scale max <= 1mm              → scale_z_max=0.001 clamp held
  - road N constant (vs init 200000) → exclude_layer_ids=[road] (no densify)

Usage: python verify_road_freeze.py <ckpt.pt>
"""

import sys

import torch


def walk(d, prefix="", hits=None):
    if hits is None:
        hits = {}
    if isinstance(d, dict):
        for k, v in d.items():
            kp = f"{prefix}.{k}" if prefix else str(k)
            if torch.is_tensor(v):
                if "road" in kp.lower():
                    hits[kp] = v
            elif isinstance(v, (dict, list)):
                walk(v, kp, hits)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            walk(v, f"{prefix}[{i}]", hits)
    return hits


def main():
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    hits = walk(ck)
    print(f"=== road tensors found: {len(hits)} ===")
    for k, v in hits.items():
        print(f"  {k}: {tuple(v.shape)} dtype={v.dtype}")

    for k, v in hits.items():
        kl = k.lower()
        if "rotation" in kl and v.ndim == 2 and v.shape[1] == 4:
            q = v.float()
            q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
            w = q[:, 0].abs().clamp(max=1.0)
            tilt = torch.rad2deg(2 * torch.acos(w))
            print(f"\n[ROTATION {k}] N={v.shape[0]}")
            print(
                f"  tilt-from-identity deg: mean={tilt.mean():.5f} "
                f"p95={tilt.quantile(0.95):.5f} max={tilt.max():.5f}"
            )
            print(
                f"  → freeze_rotation_grad {'OK (normal vertical locked)' if tilt.quantile(0.95) < 0.5 else 'LEAKED (>0.5deg)'}"
            )
        if "scale" in kl and v.ndim == 2 and v.shape[1] == 3:
            zmm = v.float()[:, 2].exp() * 1000.0
            print(f"\n[SCALE {k}] N={v.shape[0]}")
            print(f"  z-scale mm: max={zmm.max():.5f} p95={zmm.quantile(0.95):.5f} " f"median={zmm.median():.5f}")
            print(
                f"  → 1mm clamp {'OK' if zmm.max() <= 1.0 + 1e-3 else 'BREACHED'}; "
                f"road N={v.shape[0]} (init was 200000 → exclude_layer_ids {'OK' if v.shape[0]==200000 else 'changed'})"
            )

    if not hits:
        print("(no road tensors — top-level keys:)", list(ck.keys())[:25])


if __name__ == "__main__":
    main()
