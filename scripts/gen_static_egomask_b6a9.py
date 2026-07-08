# SPDX-License-Identifier: Apache-2.0
"""Generate + write the b6a9 visual-polygon static ego-mask itar (P0.3).

Writes one static ego mask per camera into an ``aux.egomask.zarr.itar`` whose
internal layout matches nre-tools (``aux/egomask/<camera_id>/<ts>`` = 0-D ``|S<n>``
PNG bytes), so the already-merged ``EgomaskAuxReader`` / ``resolve_ego_valid_mask``
read it unchanged.

Pipeline: read the HTML annotator's polygon JSON -> compose_egomask_set (union-
reinforce reads the CURRENT itar for the 4 reinforce cameras) -> either render
resolve-overlays for visual acceptance (``--mode overlay``) or write the new itar
and replace the old one write-once (``--mode write``). ``--selfcheck`` runs a
write->read round-trip.

itar write API mirrors ``scripts/merge_lidar_aux.py``. Imports are dual-path so
this runs standalone in a scratch dir (copied aux_readers.py / egomask_static.py
/ egomask_viz.py) or inside the repo.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def _get_egomask_reader():
    try:
        from aux_readers import EgomaskAuxReader
    except ImportError:
        from threedgrut.datasets.aux_readers import EgomaskAuxReader
    return EgomaskAuxReader


def _get_discover():
    try:
        from aux_readers import discover_aux_path
    except ImportError:
        from threedgrut.datasets.aux_readers import discover_aux_path
    return discover_aux_path


def _get_compose():
    try:
        from egomask_static import compose_egomask_set
    except ImportError:
        from threedgrut.datasets.egomask_static import compose_egomask_set
    return compose_egomask_set


def _get_overlay():
    from egomask_viz import render_resolved_overlay  # scratch-dir copy on the GPU host

    return render_resolved_overlay


def _encode_png(mask: np.ndarray) -> bytes:
    """Encode a ``(H, W)`` bool ego mask as PNG bytes ({0,255} grayscale)."""
    arr = (np.asarray(mask, dtype=bool).astype("uint8")) * 255
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def write_egomask_itar(masks: dict, out_path) -> None:
    """Write ``{camera_id: (H, W) bool}`` into a new egomask itar at ``out_path``.

    Each camera gets one static frame ``aux/egomask/<camera_id>/"0"`` = 0-D
    ``|S<n>`` PNG bytes. write-once: the tar index header is finalized on close;
    never interrupt mid-write.
    """
    import zarr
    from ncore.impl.data import stores

    store = stores.IndexedTarStore(str(out_path), mode="w")
    root = zarr.open(store=store, mode="w")
    for cam, mask in masks.items():
        png = _encode_png(mask)
        dt = f"|S{len(png)}"
        grp = root.create_group(f"aux/egomask/{cam}")
        ds = grp.create_dataset("0", shape=(), dtype=dt, compressor=None)
        ds[...] = np.array(png, dtype=dt)
    if hasattr(store, "close"):
        store.close()


def build_masks(clip_dir, polygons_json) -> dict:
    """Load the annotator JSON and compose the per-camera static mask set.

    JSON schema (from the HTML annotator):
        {"resolution_hw": [H, W],
         "reinforce": [camera_id, ...],
         "cameras": {camera_id: {"polygons": [[[x,y],...], ...],
                                 "fisheye_circle": [cx,cy,r] | null}}}
    The 4 reinforce cameras union with the CURRENT egomask itar (pixel-accurate
    body kept + hand-added omissions); the rest are pure-visual.
    """
    spec = json.loads(Path(polygons_json).read_text())
    hw = tuple(int(x) for x in spec["resolution_hw"])
    reinforce = set(spec.get("reinforce", []))
    cams = spec["cameras"]
    visual_specs = {
        cam: {"polygons": v.get("polygons", []), "fisheye_circle": v.get("fisheye_circle")}
        for cam, v in cams.items()
    }

    discover_aux_path = _get_discover()
    itar = discover_aux_path(clip_dir, "egomask")
    reader = _get_egomask_reader()(itar) if itar is not None else None

    compose_egomask_set = _get_compose()
    return compose_egomask_set(visual_specs, reader, hw, reinforce_cams=reinforce, skip_cams=set())


def cmd_overlay(clip_dir, polygons_json, out_dir, dilation_iters) -> None:
    masks = build_masks(clip_dir, polygons_json)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    render_resolved_overlay = _get_overlay()
    for cam, m in masks.items():
        p = out / f"overlay_{cam}.png"
        render_resolved_overlay(clip_dir, cam, m, p, dilation_iters=dilation_iters)
        print(f"overlay {cam}: ego_px={int(m.sum())} frac={float(m.mean()):.4f} -> {p}")


def cmd_write(clip_dir, polygons_json) -> None:
    """Compose (reading old itar for reinforce), then write-once replace."""
    masks = build_masks(clip_dir, polygons_json)  # reads old itar BEFORE replacing
    discover_aux_path = _get_discover()
    old = discover_aux_path(clip_dir, "egomask")
    if old is None:
        raise SystemExit(f"no existing egomask itar in {clip_dir}; expected one to replace")
    old = Path(old)
    tmp = old.with_name(old.name + ".new")
    write_egomask_itar(masks, tmp)  # write to temp name first

    backup = Path(clip_dir) / "aux_backup"
    backup.mkdir(exist_ok=True)
    old.rename(backup / old.name)  # move old out of the way
    tmp.rename(old)  # promote temp to the canonical name
    print(f"replaced {old.name} ({len(masks)} cams); old -> aux_backup/")
    for cam, m in masks.items():
        print(f"  {cam}: ego_px={int(m.sum())} frac={float(m.mean()):.4f}")


def _selfcheck() -> None:
    """Round-trip: write 2 known masks -> EgomaskAuxReader reads back exactly."""
    H, W = 40, 60
    mA = np.zeros((H, W), dtype=bool)
    mA[5:15, 5:20] = True
    mB = np.zeros((H, W), dtype=bool)
    mB[20:35, 30:55] = True
    d = tempfile.mkdtemp()
    p = os.path.join(d, "selfcheck.aux.egomask.zarr.itar")
    write_egomask_itar({"camA": mA, "camB": mB}, p)
    reader = _get_egomask_reader()(p)
    assert sorted(reader.camera_ids()) == ["camA", "camB"], reader.camera_ids()
    assert np.array_equal(reader.read_static_mask("camA"), mA), "camA mismatch"
    assert np.array_equal(reader.read_static_mask("camB"), mB), "camB mismatch"
    print("ROUNDTRIP OK")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--clip-dir")
    ap.add_argument("--polygons", help="annotator JSON")
    ap.add_argument("--mode", choices=["overlay", "write"])
    ap.add_argument("--out-dir", help="overlay output dir")
    ap.add_argument("--dilation-iters", type=int, default=30)
    args = ap.parse_args()

    if args.selfcheck:
        _selfcheck()
        return
    if not (args.clip_dir and args.polygons and args.mode):
        raise SystemExit("need --clip-dir --polygons --mode {overlay|write} (or --selfcheck)")
    if args.mode == "overlay":
        if not args.out_dir:
            raise SystemExit("--mode overlay needs --out-dir")
        cmd_overlay(args.clip_dir, args.polygons, args.out_dir, args.dilation_iters)
    else:
        cmd_write(args.clip_dir, args.polygons)


if __name__ == "__main__":
    _main()
