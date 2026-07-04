# SPDX-License-Identifier: Apache-2.0
"""P1.4 asset-harvester warm-start — bundle metadata parsing + PLY path resolve.

The asset-harvester ``metadata.yaml`` maps each harvested asset hash to its
label class, metric cuboid dims ``[L, W, H]``, and a (nested) PLY path. The demo
bundle stores PLYs flat (``<class>__<hash>.ply``) while production bundles use
the nested ``<class>/<hash>/gaussians.ply`` layout; ``resolve_ply_path`` accepts
either.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AssetSpec:
    """One harvested asset's metadata.

    Attributes:
        asset_hash: NCore track hash (bundle key).
        ply_file:   nested rel path from metadata (``class/hash/gaussians.ply``).
        label_class: e.g. ``consumer_vehicles`` / ``VRU_pedestrians``.
        cuboids_dims: metric ``[L, W, H]`` (object-local X/Y/Z order).
    """

    asset_hash: str
    ply_file: str
    label_class: str
    cuboids_dims: tuple[float, float, float]


def load_bundle_metadata(metadata_yaml: str | Path) -> dict[str, AssetSpec]:
    """Parse an asset-harvester ``metadata.yaml`` into ``{hash: AssetSpec}``."""
    with open(metadata_yaml) as f:
        doc = yaml.safe_load(f) or {}
    assets = doc.get("assets", {}) or {}
    out: dict[str, AssetSpec] = {}
    for asset_hash, info in assets.items():
        dims = tuple(float(d) for d in info["cuboids_dims"])
        if len(dims) != 3:
            raise ValueError(f"asset {asset_hash!r} cuboids_dims must be [L,W,H], got {dims}")
        out[str(asset_hash)] = AssetSpec(
            asset_hash=str(asset_hash),
            ply_file=str(info["ply_file"]),
            label_class=str(info["label_class"]),
            cuboids_dims=dims,  # type: ignore[arg-type]
        )
    return out


def resolve_ply_path(bundle_root: str | Path, spec: AssetSpec) -> Path:
    """Resolve a spec's PLY file under ``bundle_root``.

    Tries the nested ``ply_file`` layout first, then the demo flat layout
    ``<label_class>__<asset_hash>.ply``. Raises ``FileNotFoundError`` if neither
    exists (never returns a non-existent path silently).
    """
    root = Path(bundle_root)
    nested = root / spec.ply_file
    if nested.is_file():
        return nested
    flat = root / f"{spec.label_class}__{spec.asset_hash}.ply"
    if flat.is_file():
        return flat
    raise FileNotFoundError(
        f"no PLY for asset {spec.asset_hash!r} under {root} " f"(tried nested {spec.ply_file!r} and flat {flat.name!r})"
    )


def map_assets_to_tracks(
    bundle: dict[str, AssetSpec],
    tracks: dict,
    mapping,
) -> dict[str, AssetSpec]:
    """Resolve which warm-start asset feeds which NCore track.

    ``mapping`` is an explicit ``{track_id: asset_hash}`` dict (or a path to a
    JSON file of one). Demo assets have no clip correspondence, so the mapping
    must be supplied explicitly — a ``None`` mapping is an error rather than a
    silent no-op. Raises ``KeyError`` if a mapped track or asset is unknown.

    Returns ``{track_id: AssetSpec}`` for the mapped tracks only.
    """
    if mapping is None:
        raise ValueError(
            "warm-start needs an explicit warmstart_ply_mapping "
            "(track_id -> asset_hash); demo assets have no clip correspondence"
        )
    if isinstance(mapping, (str, Path)):
        with open(mapping) as f:
            mapping = json.load(f)
    # 3dgrut keeps the raw '<id>@scene:...' NCore track id; harvested assets use
    # the cleaned '<id>'. Build a cleaned→raw lookup so a mapping keyed by either
    # form resolves to the raw track key (needed for name_to_id indexing).
    clean_to_raw: dict[str, str] = {}
    for raw_key in tracks:
        clean_to_raw.setdefault(str(raw_key).split("@", 1)[0], raw_key)
    out: dict[str, AssetSpec] = {}
    for track_id, asset_hash in dict(mapping).items():
        if track_id in tracks:
            resolved = track_id
        elif str(track_id).split("@", 1)[0] in clean_to_raw:
            resolved = clean_to_raw[str(track_id).split("@", 1)[0]]
        else:
            raise KeyError(
                f"warm-start mapping track {track_id!r} not among "
                f"{len(tracks)} loaded tracks (cleaned ids: {sorted(clean_to_raw)})"
            )
        if asset_hash not in bundle:
            raise KeyError(f"warm-start mapping asset {asset_hash!r} not in bundle " f"(have {sorted(bundle)})")
        out[resolved] = bundle[asset_hash]
    return out
