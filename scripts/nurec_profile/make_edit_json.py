#!/usr/bin/env python3
"""Build edit-assets.json remove-lists by actor class for the ablation.

Phase 2 / Tier 1 of the NuRec USDZ render-profiling plan. Reads the artifact's
sequence_tracks.json (CuboidTracks.to_dict() layout) and emits two edit jsons:

  edit_rigid.json  -> remove all rigid actors (car/truck/bus/vehicle/...)
  edit_person.json -> remove all deformable actors (person/pedestrian/...)

Each file has the four top-level keys render-grpc expects (metadata/replace/
remove/insert), even when replace/insert are empty.

Get sequence_tracks.json out of the USDZ first:
    unzip -p last.usdz sequence_tracks.json > /wk/profile/sequence_tracks.json

Then (host python is fine — pure json):
    python3 make_edit_json.py --tracks /wk/profile/sequence_tracks.json --out-dir /wk/profile

VALIDATE: run with --list first to see the exact label strings present in the
clip; the class buckets below are best-effort and may need a tweak per clip.
"""
import argparse
import json
import os

RIGID = {"car", "truck", "bus", "vehicle", "trailer", "motorcycle", "bicycle", "van", "automobile"}
DEFORM = {"person", "pedestrian", "people", "human", "cyclist", "rider"}


def _extract_tracks(doc):
    """Return list of (track_id:str, label_class:str). Defensive across shapes."""
    td = doc.get("tracks_data", doc) if isinstance(doc, dict) else {}
    ids = td.get("tracks_id") or td.get("track_ids") or doc.get("tracks_id")
    cls = td.get("tracks_label_class") or td.get("label_class") or doc.get("tracks_label_class")
    if ids and cls and len(ids) == len(cls):
        return [(str(i), str(c)) for i, c in zip(ids, cls)]
    # fallback: list-of-dicts
    items = doc.get("tracks") if isinstance(doc, dict) else None
    if isinstance(items, list):
        out = []
        for t in items:
            tid = t.get("id") or t.get("track_id")
            lc = t.get("label_class") or t.get("semantic_class") or t.get("class")
            if tid is not None:
                out.append((str(tid), str(lc)))
        if out:
            return out
    raise SystemExit("could not parse tracks; inspect the json and adjust _extract_tracks()")


def _edit(remove_ids):
    return {"metadata": {"external_assets_metadata": []},
            "replace": [], "remove": list(remove_ids),
            "insert": {"asset_ids": [], "data": {}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    doc = json.load(open(args.tracks))
    tracks = _extract_tracks(doc)

    from collections import Counter
    counts = Counter(c.lower() for _, c in tracks)
    print(f"=== {len(tracks)} tracks; class histogram ===")
    for c, n in counts.most_common():
        bucket = "rigid" if c in RIGID else ("deform" if c in DEFORM else "OTHER?")
        print(f"  {c:18s} {n:>4}  [{bucket}]")
    if args.list:
        return

    rigid_ids = [tid for tid, c in tracks if c.lower() in RIGID]
    person_ids = [tid for tid, c in tracks if c.lower() in DEFORM]
    other = sorted({c.lower() for _, c in tracks} - RIGID - DEFORM)
    if other:
        print(f"WARNING: unbucketed classes {other} — edit RIGID/DEFORM sets if these matter")

    os.makedirs(args.out_dir, exist_ok=True)
    for name, ids in (("edit_rigid.json", rigid_ids), ("edit_person.json", person_ids)):
        p = os.path.join(args.out_dir, name)
        json.dump(_edit(ids), open(p, "w"), indent=2)
        print(f"wrote {p}  (remove {len(ids)} tracks)")


if __name__ == "__main__":
    main()
