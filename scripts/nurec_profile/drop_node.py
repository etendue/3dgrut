#!/usr/bin/env python3
"""Physically remove a gaussians-node's splats from an NRE artifact.

Phase 1 + Phase 2 of the NuRec USDZ render-profiling plan
(.claude/plans/nurec-usdz-4090-nurec-atomic-dewdrop.md).

Why "physically remove rows" and not "zero opacity": a gaussian with density
pushed to -inf still gets uploaded + traced every frame, so it costs the same
render time. To measure a node's true trace cost we must drop its rows so the
fused buffer is genuinely smaller.

INPUT/OUTPUT can be a raw .ckpt OR a .usdz:
- .usdz input  -> extracts the embedded `checkpoint.ckpt`, edits it.
- .usdz output -> repackages the source usdz with the edited checkpoint.ckpt
  swapped in (all other members copied verbatim; default.usda kept first;
  ZIP_STORED). This is the cheap ablation lever: render reads the usdz's
  embedded ckpt directly, so an edited usdz renders the ablated scene with no
  export-usdz-artifact / dataset / gRPC needed.

NRE's `run-script SCRIPT_PATH` cannot forward CLI args, so params are read from
env vars when no --in is given:
  DROP_IN / DROP_OUT / DROP_NODE / DROP_KEEP / DROP_LIST

    docker run --rm --gpus all --shm-size 64g -v ~/work/nurec_e0:/wk \
      -e DROP_LIST=1 -e DROP_IN=/wk/.../artifacts/last.usdz \
      nvcr.io/nvidia/nre/nre-ga:latest run-script /wk/profile/drop_node.py

    docker run --rm --gpus all --shm-size 64g -v ~/work/nurec_e0:/wk \
      -e DROP_NODE=background -e DROP_KEEP=1 \
      -e DROP_IN=/wk/.../artifacts/last.usdz \
      -e DROP_OUT=/wk/profile/usdz_drop_background/last.usdz \
      nvcr.io/nvidia/nre/nre-ga:latest run-script /wk/profile/drop_node.py
"""
import argparse
import collections
import io
import os
import sys
import zipfile

CKPT_MEMBER = "checkpoint.ckpt"


def _torch_load(data_or_path):
    import torch
    obj = io.BytesIO(data_or_path) if isinstance(data_or_path, (bytes, bytearray)) else data_or_path
    try:
        return torch.load(obj, map_location="cpu", weights_only=False)
    except TypeError:
        if hasattr(obj, "seek"):
            obj.seek(0)
        return torch.load(obj, map_location="cpu")


def _load(path):
    if path.endswith(".usdz"):
        with zipfile.ZipFile(path) as z:
            if CKPT_MEMBER not in z.namelist():
                raise SystemExit(f"{path} has no embedded {CKPT_MEMBER}")
            return _torch_load(z.read(CKPT_MEMBER)), "usdz"
    return _torch_load(path), "ckpt"


def _save(ckpt, out, src_path, src_kind):
    import torch
    if out.endswith(".usdz"):
        if src_kind != "usdz":
            raise SystemExit("usdz output requires usdz input (need the other members to copy)")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        bio = io.BytesIO()
        torch.save(ckpt, bio)
        new_ckpt = bio.getvalue()
        with zipfile.ZipFile(src_path) as zin:
            names = zin.namelist()
            order = sorted(names, key=lambda n: (not n.endswith("default.usda")))  # default.usda first
            with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zout:
                for n in order:
                    zout.writestr(n, new_ckpt if n == CKPT_MEMBER else zin.read(n))
        print(f"wrote repackaged usdz {out}  ({CKPT_MEMBER} swapped, {len(names)} members)")
    else:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        torch.save(ckpt, out)
        print(f"wrote {out}")


def _find_state_dict(ckpt):
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise SystemExit(f"unexpected checkpoint type: {type(ckpt)}")


def _node_prefixes(sd):
    marker = "gaussians_nodes."
    out = {}
    for k in sd:
        i = k.find(marker)
        if i == -1:
            continue
        node = k[i + len(marker):].split(".", 1)[0]
        out.setdefault(node, k[: i + len(marker)] + node + ".")
    return out


def _get_args():
    env = os.environ
    if env.get("DROP_IN") and not any(a.startswith("--in") for a in sys.argv[1:]):
        class A:
            inp = env["DROP_IN"]
            out = env.get("DROP_OUT")
            node = env.get("DROP_NODE")
            keep = int(env.get("DROP_KEEP", "0"))
            list = env.get("DROP_LIST", "").lower() in ("1", "true", "yes")
        return A()
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out")
    ap.add_argument("--node")
    ap.add_argument("--keep", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    return ap.parse_args()


def main():
    args = _get_args()
    ckpt, src_kind = _load(args.inp)
    sd = _find_state_dict(ckpt)
    prefixes = _node_prefixes(sd)
    if not prefixes:
        raise SystemExit("no 'gaussians_nodes.<node>.' keys found.")

    print("=== gaussians_nodes in checkpoint ===")
    node_counts = {}
    for node, pfx in sorted(prefixes.items()):
        keys = [k for k in sd if k.startswith(pfx)]
        n = None
        for k in keys:
            t = sd[k]
            if hasattr(t, "shape") and getattr(t, "ndim", 0) >= 1 and (k.endswith("positions") or k.endswith(".means")):
                n = int(t.shape[0])
        if n is None:
            dims = collections.Counter(int(sd[k].shape[0]) for k in keys
                                       if hasattr(sd[k], "shape") and getattr(sd[k], "ndim", 0) >= 1)
            n = dims.most_common(1)[0][0] if dims else 0
        node_counts[node] = n
        print(f"  {node:24s} N={n:>9}  ({len(keys)} tensors)")
        if getattr(args, "list", False):
            for k in sorted(keys):
                t = sd[k]
                print(f"      {k}  {tuple(t.shape) if hasattr(t,'shape') else type(t).__name__}")

    if getattr(args, "list", False):
        return
    if not args.node:
        raise SystemExit("DROP_NODE/--node required. Available: " + ", ".join(sorted(prefixes)))
    if args.node not in prefixes:
        raise SystemExit(f"node '{args.node}' not found. Available: " + ", ".join(sorted(prefixes)))
    if not args.out:
        raise SystemExit("DROP_OUT/--out required when dropping")

    n = node_counts[args.node]
    pfx = prefixes[args.node]
    keep = max(0, min(args.keep, n))
    print(f"\n=== dropping node '{args.node}': N={n} -> keep={keep} ===")
    changed = 0
    for k in [k for k in sd if k.startswith(pfx)]:
        t = sd[k]
        if hasattr(t, "shape") and getattr(t, "ndim", 0) >= 1 and int(t.shape[0]) == n:
            sd[k] = t[:keep].clone()
            changed += 1
    print(f"sliced {changed} per-gaussian tensors of node '{args.node}'")
    _save(ckpt, args.out, args.inp, src_kind)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--"]:
        sys.argv.pop(1)
    main()
