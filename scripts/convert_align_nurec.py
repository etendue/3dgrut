# SPDX-License-Identifier: Apache-2.0
"""Convert a NuRec USDZ → 3dgrut2 .pt AND align it to NCore world frame.

The bare `convert_usdz_to_pt` keeps gaussian positions in the NRE training
frame (origin shifted ~ -38m x for clip 9ae151dc to keep float32 small). To
evaluate a NuRec ckpt with NCore-frame cameras (the 9ae151dc manifest), we must
undo that: translate every static-layer position by `-world_to_nre[:3,3]`. This
mirrors what viser_gui_4d does inline (E2.7 P1 fix, viser_gui_4d.py:2106-2192),
extracted here so eval_road_offtrack can score NuRec on the SAME footing as our
own ckpts (3dgrut2 render + 9ae151dc cameras + road-only).

Usage:
  python scripts/convert_align_nurec.py \
      --usdz <last.usdz> --out /tmp/nurec_baseline_aligned.pt --layers road
"""

from __future__ import annotations

import argparse
import json
import zipfile

import numpy as np
import torch

from threedgrut_playground.utils.nre_usdz_loader import convert_usdz_to_pt


def main() -> None:
    ap = argparse.ArgumentParser(description="NuRec USDZ → aligned 3dgrut2 .pt")
    ap.add_argument("--usdz", required=True)
    ap.add_argument("--out", required=True, help="output ALIGNED .pt")
    ap.add_argument("--layers", default="road", help="comma list; road-only eval needs just 'road'")
    args = ap.parse_args()
    layers = tuple(s.strip() for s in args.layers.split(",") if s.strip())

    raw = args.out[:-3] + "_raw.pt" if args.out.endswith(".pt") else args.out + "_raw.pt"

    # 1) USDZ → native .pt (NRE frame, unaligned)
    print(f"[convert] {args.usdz} → {raw} layers={layers}", flush=True)
    convert_usdz_to_pt(args.usdz, raw, layers=layers)

    # 2) read world_to_nre from the USDZ container
    with zipfile.ZipFile(args.usdz) as z:
        rt = json.load(z.open("rig_trajectories.json"))
    w2nre = rt.get("world_to_nre") or {}
    mat = np.asarray(
        w2nre.get("matrix") if isinstance(w2nre, dict) else w2nre,
        dtype=np.float64,
    ).reshape(4, 4)
    translate = (-mat[:3, 3]).astype(np.float32)
    R = mat[:3, :3]
    print(f"[align] world_to_nre.translation={mat[:3,3].tolist()}", flush=True)
    print(f"[align] NRE→world translate={translate.tolist()}", flush=True)
    if not np.allclose(R, np.eye(3), atol=1e-4):
        raise SystemExit(
            f"[align] world_to_nre rotation NOT identity:\n{R}\n"
            "translate-only align insufficient — needs full-matrix align."
        )

    # 3) apply +translate to every static layer's positions (skip dynamic_rigids
    #    which are object-local and get their world pose from track poses).
    ckpt = torch.load(raw, weights_only=False)
    tt = torch.as_tensor(translate, dtype=torch.float32)
    for layer, node in ckpt["model"]["gaussians_nodes"].items():
        if layer == "dynamic_rigids":
            print(f"[align]   {layer}: kept object-local (skipped)", flush=True)
            continue
        p = node["positions"]
        with torch.no_grad():
            pn = p.detach() + tt.to(p.device, p.dtype)
        node["positions"] = torch.nn.Parameter(pn.contiguous(), requires_grad=False)
        print(f"[align]   {layer}: shifted {p.shape[0]} gaussians", flush=True)

    torch.save(ckpt, args.out)
    print(f"[align] wrote aligned ckpt → {args.out}", flush=True)


if __name__ == "__main__":
    main()
