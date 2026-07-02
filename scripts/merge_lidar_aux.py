#!/usr/bin/env python3
"""Merge N per-segment lidar aux itars (from parallel `sensor-frames` runs) into
one complete itar. Each segment holds a disjoint slice of lidar frames; the
merged itar contains all frames — byte-identical layout to a single full run,
so datasetNcore reads it unchanged.

Companion to the multi-container parallel lidar-seg workflow (see CLAUDE.md
"⚡ lidar-seg 多 container 并行"): run A generates full-frame sseg+egomask, then
N docker containers each run `ncore-aux-data ... sensor-frames --start-frame
--stop-frame` over a disjoint lidar-frame range (reusing the full sseg), and
this script stitches their per-segment lidar-sseg / lidar-camvis itars back into
one. Turns a 5.8h single-core lidar-seg into ~1h (N-way parallel + merge).

itar layouts handled (auto — shape is read from the source, not hard-coded):
    /aux/lidar_semantic_segmentation/<lidar_id>/<ts_us>   -> 0-D  |S<n>  PNG bytes
    /aux/lidar_camera_visibility/<lidar_id>/<ts_us>       -> (N_pts, 1) uint8 array
    /aux/<component>/<sensor_id>/.zattrs                   -> group-level attrs

Usage:
    python merge_lidar_aux.py <out.itar> <seg0.itar> <seg1.itar> ...
"""
import sys

import zarr
from ncore.impl.data import stores


def merge(out_path: str, seg_paths: list[str]) -> None:
    roots = [zarr.open(store=stores.IndexedTarStore(p, mode="r"), mode="r") for p in seg_paths]

    out_store = stores.IndexedTarStore(out_path, mode="w")
    out_root = zarr.open(store=out_store, mode="w")

    # discover /aux/<component>/<sensor> groups as the UNION across all inputs —
    # seg0-only discovery drops cameras when merging sseg itars generated for
    # different camera sets (A1: original 3-cam + side 3-cam → 6-cam).
    comp_sensors: dict[str, list[str]] = {}
    for root in roots:
        aux = root["aux"]
        for comp in list(aux.group_keys()):
            known = comp_sensors.setdefault(comp, [])
            for sensor in list(aux[comp].group_keys()):
                if sensor not in known:
                    known.append(sensor)
    total = 0
    for comp, sensors in comp_sensors.items():
        for sensor in sensors:
            out_grp = out_root.create_group(f"aux/{comp}/{sensor}")
            # group .zattrs are identical across segments — write once from the
            # first input that has this group
            for root in roots:
                try:
                    out_grp.attrs.put(dict(root["aux"][comp][sensor].attrs))
                    break
                except KeyError:
                    continue
            seen: set[str] = set()
            for root in roots:
                try:
                    grp = root["aux"][comp][sensor]
                except KeyError:
                    continue
                for ts in list(grp.array_keys()):
                    if ts in seen:
                        continue  # dedup (segments should be disjoint, but be safe)
                    seen.add(ts)
                    src = grp[ts]
                    data = src[...]  # scalar (|S<n> PNG bytes) OR N-D array (camvis)
                    # use src.shape — do NOT hard-code shape=(); camvis is a
                    # (N_pts, 1) array and shape=() raises "setting an array
                    # element with a sequence".
                    ds = out_grp.create_dataset(
                        ts, shape=src.shape, dtype=src.dtype, compressor=None)
                    ds[...] = data
                    ds.attrs.put(dict(src.attrs))
                    total += 1
            print(f"  {comp}/{sensor}: {len(seen)} frames", flush=True)

    if hasattr(out_store, "close"):
        out_store.close()
    print(f"merged {len(seg_paths)} segs -> {out_path} ({total} frame-datasets)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: merge_lidar_aux.py <out.itar> <seg0.itar> [seg1.itar ...]", file=sys.stderr)
        sys.exit(2)
    merge(sys.argv[1], sys.argv[2:])
