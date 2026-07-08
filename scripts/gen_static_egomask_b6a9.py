# SPDX-License-Identifier: Apache-2.0
"""Generate + write the b6a9 visual-polygon static ego-mask itar (P0.3).

Writes one static ego mask per camera into an ``aux.egomask.zarr.itar`` whose
internal layout matches nre-tools (``aux/egomask/<camera_id>/<ts>`` = 0-D ``|S<n>``
PNG bytes), so the already-merged ``EgomaskAuxReader`` / ``resolve_ego_valid_mask``
read it unchanged.

T4 (this commit) provides ``write_egomask_itar`` + a ``--selfcheck`` round-trip
(write -> EgomaskAuxReader read-back, exact). The full driver (read polygon JSON
-> compose_egomask_set -> overlays -> write-once replace) is added in T5.

itar write API mirrors ``scripts/merge_lidar_aux.py``. Import is dual-path so it
runs standalone in a scratch dir (copied ``aux_readers.py``) or inside the repo.
"""

from __future__ import annotations

import argparse
import io
import os
import tempfile

import numpy as np
from PIL import Image


def _get_egomask_reader():
    try:
        from aux_readers import EgomaskAuxReader
    except ImportError:
        from threedgrut.datasets.aux_readers import EgomaskAuxReader
    return EgomaskAuxReader


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

    EgomaskAuxReader = _get_egomask_reader()
    r = EgomaskAuxReader(p)
    assert sorted(r.camera_ids()) == ["camA", "camB"], r.camera_ids()
    assert np.array_equal(r.read_static_mask("camA"), mA), "camA mismatch"
    assert np.array_equal(r.read_static_mask("camB"), mB), "camB mismatch"
    print("ROUNDTRIP OK")


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true", help="run write->read round-trip and exit")
    args = ap.parse_args()
    if args.selfcheck:
        _selfcheck()
        return
    raise SystemExit("driver mode is added in T5; run with --selfcheck for now")


if __name__ == "__main__":
    _main()
