"""P0.1 diagnosis: why is b6a9 ego mask all-black?

Checks, per aux dir:
  1. aux-meta.json ego_mask fields
  2. egomask itar: layout tree + per-camera nonzero stats
  3. sseg itar: egocar(19) pixel counts on sampled frames
     -> if sseg has no egocar pixels, regenerating --ego-mask cannot work
"""

import glob
import io
import json
import sys

import numpy as np
import zarr
from ncore.impl.data import stores
from PIL import Image

CLIP = "/home/inceptio/work/data/inc_b6a9ed61_20s/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9"
ALT_DIRS = [
    "/home/inceptio/work/data/inc_b6a9ed61_20s/aux_run_full2",
    "/home/inceptio/work/data/inc_b6a9ed61_20s/aux_runA_side3",
    "/home/inceptio/work/data/inc_b6a9ed61_20s/aux_occtest",
]


def open_itar(path):
    return zarr.open(store=stores.IndexedTarStore(path, mode="r"), mode="r")


def leaf_arrays(grp, prefix=""):
    out = []
    for k in grp.group_keys():
        out += leaf_arrays(grp[k], prefix + k + "/")
    for k in grp.array_keys():
        out.append((prefix + k, grp[k]))
    return out


def decode(arr):
    """0-D |S PNG bytes or plain ndarray -> ndarray."""
    if arr.shape == ():
        return np.asarray(Image.open(io.BytesIO(bytes(arr[()]))))
    return arr[...]


def check_dir(d):
    print(f"\n######## {d}")
    metas = glob.glob(d + "/*.aux-meta.json")
    if metas:
        meta = json.load(open(metas[0]))
        ego_keys = {k: v for k, v in meta.items() if "ego" in str(k).lower()}
        print("META ego fields:", json.dumps(ego_keys))
        print("META top keys:", sorted(meta.keys())[:20])
    egos = glob.glob(d + "/*.aux.egomask.zarr.itar")
    if not egos:
        print("no egomask itar")
        return
    try:
        g = open_itar(egos[0])
    except Exception as e:
        print("egomask OPEN FAIL:", repr(e))
        return
    leaves = leaf_arrays(g)
    print(f"egomask leaves: {len(leaves)}")
    by_cam = {}
    for path, arr in leaves:
        cam = path.split("/")[-2] if "/" in path else path
        by_cam.setdefault(cam, []).append((path, arr))
    for cam, items in sorted(by_cam.items()):
        path, arr = items[0]
        try:
            img = decode(arr)
            nz = int((img != 0).sum())
            print(f"  {cam}: n_frames={len(items)} shape={img.shape} dtype={img.dtype} " f"nonzero_first={nz}/{img.size} uniq={np.unique(img)[:6]}")
        except Exception as e:
            print(f"  {cam}: decode fail {path}: {repr(e)}")


def check_sseg_egocar(d, n_sample=3):
    ssegs = glob.glob(d + "/*.aux.sseg.zarr.itar")
    if not ssegs:
        return
    print(f"\n==== sseg egocar(19) scan: {d}")
    g = open_itar(ssegs[0])
    root = g["aux/semantic_segmentation"] if "aux" in g else g
    for cam in root.group_keys():
        cg = root[cam]
        keys = sorted(cg.array_keys())
        counts = []
        for k in [keys[0], keys[len(keys) // 2], keys[-1]][:n_sample]:
            img = decode(cg[k])
            counts.append(int((img == 19).sum()))
        classes = cg.attrs.get("stuff_classes", "?")
        cls19 = classes[19] if isinstance(classes, list) and len(classes) > 19 else "?"
        print(f"  {cam}: egocar_px(first/mid/last)={counts} total_px={img.size} class19='{cls19}'")


check_dir(CLIP)
check_sseg_egocar(CLIP)
for d in ALT_DIRS:
    check_dir(d)
print("\nDONE")
