#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E2.5 CLI — frozen-inject asset-harvester cars to replace recon cars.

Offline ckpt surgery (no training): loads a baseline LayeredGaussians ckpt,
maps N AH car assets onto N recon car tracks by cuboid size, aligns each AH
PLY into the target track's object-local frame (filling its live cuboid),
and swaps the recon particles for the AH particles while every other track /
layer stays byte-identical. The edited ckpt loads in viser_gui_4d.py and the
AH cars replay along the recon trajectories.

Pure logic is unit-tested in threedgrut/tests/test_e25_inject_ah_replace.py;
this file is the IO orchestration. Run --dry_run first to eyeball the mapping
and per-axis size deltas before writing the ckpt.

Usage (inceptio)::

    python scripts/e25_inject_ah_replace.py \
        --baseline_ckpt ~/work/output/v3_base_scratch30k_lam01/ckpt_30000.pt \
        --ah_bundle     ~/work/nurec_e0/assets/bundle \
        --dataset_path  ~/work/data/9ae151dc/pai_9ae151dc-...json \
        --out_ckpt      ~/work/output/e25_ah_frozen/ckpt_e25_frozen.pt \
        --ensure_viz_4d --dry_run        # then drop --dry_run to commit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from threedgrut.layers.e25_inject import (  # noqa: E402
    build_name_to_int_id,
    match_assets_by_size,
    replace_tracks_in_dyn_node,
)
from threedgrut.layers.warmstart_metadata import (  # noqa: E402
    load_bundle_metadata,
    resolve_ply_path,
)
from threedgrut.layers.warmstart_ply import (  # noqa: E402
    apply_alignment,
    asset_extent,
    compute_axis_alignment,
    load_warmstart_ply,
    subsample_asset,
)

# NCore cuboid autolabel classes that count as a "car" for E2.5 (pedestrians
# are skipped per the spike decision — AH is the wrong tool for them).
_VEHICLE_CLASS_TOKENS = (
    "automobile", "bus", "truck", "consumer_vehicles", "car", "vehicle",
)
_AH_CAR_CLASS_TOKENS = ("consumer_vehicles", "automobile")


def _to_tuple3(size) -> tuple[float, float, float]:
    t = torch.as_tensor(size, dtype=torch.float32).flatten()
    return (float(t[0]), float(t[1]), float(t[2]))


def _dims_delta(a, b) -> tuple[float, float, float]:
    return tuple(round(float(x - y), 3) for x, y in zip(a, b))


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline_ckpt", required=True)
    ap.add_argument("--ah_bundle", required=True, help="dir containing metadata.yaml")
    ap.add_argument("--dataset_path", default=None, help="NCore pai_*.json (for --ensure_viz_4d)")
    ap.add_argument("--out_ckpt", required=True)
    ap.add_argument("--mapping", default=None, help="JSON {track_name: asset_hash} to override auto-match")
    ap.add_argument("--max_pts_per_track", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ensure_viz_4d", action="store_true",
                    help="if ckpt lacks viz_4d, run inject_viz_4d first (needs --dataset_path + NCore SDK)")
    ap.add_argument("--no_class_filter", action="store_true",
                    help="consider every present track (not just vehicle classes)")
    ap.add_argument("--dry_run", action="store_true",
                    help="print probe + mapping + size deltas, write nothing")
    return ap.parse_args(argv)


def _load_ckpt_with_viz_4d(args) -> dict:
    ckpt = torch.load(args.baseline_ckpt, weights_only=False, map_location="cpu")
    if ckpt.get("viz_4d") is not None:
        return ckpt
    if not args.ensure_viz_4d:
        raise SystemExit(
            "baseline ckpt has no viz_4d block (track names/size/poses). "
            "Re-run with --ensure_viz_4d --dataset_path <NCore pai_*.json>."
        )
    from threedgrut.viz.inject import inject_viz_4d  # lazy: needs NCore SDK
    tmp = Path(tempfile.mkdtemp()) / "baseline_with_viz4d.pt"
    print(f"[e25] ckpt lacks viz_4d → inject_viz_4d → {tmp}")
    inject_viz_4d(args.baseline_ckpt, args.dataset_path, str(tmp))
    return torch.load(tmp, weights_only=False, map_location="cpu")


def main(argv=None) -> int:
    args = _parse_args(argv)
    ckpt = _load_ckpt_with_viz_4d(args)
    tracks = ckpt["viz_4d"]["tracks"]
    name_to_id = build_name_to_int_id(tracks)

    if "dynamic_rigids" not in ckpt["model"]["gaussians_nodes"]:
        raise SystemExit("ckpt has no dynamic_rigids layer — nothing to replace.")
    dyn = ckpt["model"]["gaussians_nodes"]["dynamic_rigids"]
    if "track_ids" not in dyn:
        raise SystemExit("dynamic_rigids node has no track_ids buffer — cannot map per-track.")
    present_ids = {int(i) for i in dyn["track_ids"].unique().tolist()}

    # recon vehicle tracks actually present in the geometry buffer
    recon_sizes: dict[str, tuple] = {}
    for name, meta in tracks.items():
        tid = name_to_id[name]
        if tid not in present_ids:
            continue
        cls = str(meta.get("class", "unknown")).lower()
        if not args.no_class_filter and not any(tok in cls for tok in _VEHICLE_CLASS_TOKENS):
            continue
        recon_sizes[name] = _to_tuple3(meta["size"])

    # AH car assets only (skip pedestrians)
    bundle = load_bundle_metadata(Path(args.ah_bundle) / "metadata.yaml")
    ah_dims = {
        h: spec.cuboids_dims
        for h, spec in bundle.items()
        if any(tok in spec.label_class for tok in _AH_CAR_CLASS_TOKENS)
    }

    # mapping {track_name: asset_hash}
    if args.mapping:
        with open(args.mapping) as f:
            mapping = {str(k): str(v) for k, v in json.load(f).items()}
    else:
        mapping = match_assets_by_size(ah_dims, recon_sizes)

    # ---- report ----
    print(f"=== recon vehicle tracks present ({len(recon_sizes)}) ===")
    for name, sz in sorted(recon_sizes.items(), key=lambda kv: name_to_id[kv[0]]):
        n = int((dyn["track_ids"] == name_to_id[name]).sum())
        print(f"  id={name_to_id[name]:>3d}  n={n:>6d}  size={sz}  {name}")
    print(f"=== AH car assets ({len(ah_dims)}) ===")
    for h, d in ah_dims.items():
        print(f"  {h}  dims={tuple(round(x,3) for x in d)}")
    print(f"=== mapping recon_track ← AH ({len(mapping)}) ===")
    for name, h in mapping.items():
        print(f"  id={name_to_id[name]:>3d} {recon_sizes[name]} ← {h} {tuple(round(x,3) for x in ah_dims[h])}"
              f"  Δ(L,W,H)={_dims_delta(recon_sizes[name], ah_dims[h])}  {name}")

    if not mapping:
        raise SystemExit("empty mapping — no recon vehicle track matched any AH car. Check --dataset_path / classes.")

    if args.dry_run:
        print("[e25] --dry_run: nothing written.")
        return 0

    # ---- align each AH to its target track's LIVE cuboid size, then surgery ----
    gen = torch.Generator().manual_seed(args.seed)
    aligned_by_id: dict[int, object] = {}
    for name, h in mapping.items():
        spec = bundle[h]
        ply = resolve_ply_path(args.ah_bundle, spec)
        asset = load_warmstart_ply(ply)
        half, center = asset_extent(asset)
        dims = recon_sizes[name]  # fill the recon car's cuboid, not AH metadata dims
        xf = compute_axis_alignment(spec.label_class, dims, half, center)
        aligned = apply_alignment(asset, xf)
        aligned = subsample_asset(aligned, args.max_pts_per_track, generator=gen)
        aligned_by_id[name_to_id[name]] = aligned
        print(f"[e25] aligned {h} → id={name_to_id[name]} ({aligned.positions.shape[0]} pts)")

    n_before = dyn["track_ids"].shape[0]
    ckpt["model"]["gaussians_nodes"]["dynamic_rigids"] = replace_tracks_in_dyn_node(dyn, aligned_by_id)
    n_after = ckpt["model"]["gaussians_nodes"]["dynamic_rigids"]["track_ids"].shape[0]
    print(f"[e25] dynamic_rigids particles: {n_before} → {n_after}")

    out = Path(args.out_ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    print(f"[e25] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
