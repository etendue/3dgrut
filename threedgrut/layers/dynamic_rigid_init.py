# SPDX-License-Identifier: Apache-2.0
"""DynamicRigid layer initialization: cuboid-inside LiDAR → object-local frame.

Stage 4 T4.2.b — produce per-track Gaussian particle positions in OBJECT-LOCAL
frame (not world). LayeredGaussians.fused_view(frame_id) then applies the
per-frame world transform at render time (T4.3).

Algorithm (parallel to drivestudio get_init_objects, schema only; rebuilt
without OmniRe pixel_source coupling):

  1. For each active frame of each track:
       - Compute pose_inv = (object→world)⁻¹
       - Transform every dyn-LiDAR point into the track's local frame
       - Keep points inside the cuboid: |local_i| ≤ size_i/2 ∀ i ∈ {x,y,z}
  2. Concatenate per-frame local hits per track (gives a denser sample than
     single-frame, helps short-lived tracks).
  3. Random subsample to max_pts_per_track if over.
  4. Concatenate across tracks and emit:
       - positions  [Σ, 3]   in object-local frame
       - track_ids  [Σ]      int64, sorted(track_keys) → 0..K-1
     Plus mutate ``instance_pts_dict[tid]["pts"]`` in place so callers can
     inspect per-track contributions.

Also exports the int↔name mapping (``track_names``) used by T4.3 to build the
per-particle pose lookup.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


# V3-L5: axis-name → object-local frame coordinate index. Cuboid object-local
# frame convention (matches load_tracks_from_ncore_cuboids euler_xyz_to_rotation
# decoding): X=forward, Y=left, Z=up. Vehicles are predominantly Y-symmetric.
_SYMMETRIC_AXIS_INDEX: dict[str, int] = {"X": 0, "Y": 1, "Z": 2}


def init_dynamic_rigid_layer(
    instance_pts_dict: Dict[str, dict],
    dynamic_lidar_pts: torch.Tensor,
    max_pts_per_track: int = 5_000,
    symmetric_axis: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Filter dyn LiDAR → object-local per track; concat across tracks.

    Args:
        instance_pts_dict: output of ``load_tracks_from_manifest``; each value
            must have ``poses[F, 4, 4]``, ``size[3]``, ``frame_info[F]``.
            Mutated in place: ``["pts"]`` filled with this track's local-frame
            points.
        dynamic_lidar_pts: ``[M, 3]`` world frame, semantically filtered to
            dynamic classes (NCoreDataset.get_dynamic_lidar_points).
        max_pts_per_track: per-track subsample cap.
        symmetric_axis: V3-L5 (NuRec ``symmetric_axis``). When set to ``'X'``,
            ``'Y'``, or ``'Z'``, every per-track local-frame point is mirrored
            across the named axis (i.e. coordinate ``i`` negated) and the
            mirrored copy concatenated **before** the ``max_pts_per_track``
            subsample. For vehicles (predominantly left-right symmetric) the
            NuRec default is ``'Y'``. ``None`` disables the augmentation
            (baseline behaviour, byte-identical to pre-V3-L5).

    Returns:
        positions:   ``[Σ, 3]`` object-local frame
        track_ids:   ``[Σ]`` int64, maps each particle to its track's int id
        track_names: ``[K]`` list of track names in the order matching
                     track_ids values 0..K-1 (sorted by key for determinism)
    """
    axis_idx: Optional[int] = None
    if symmetric_axis is not None:
        if symmetric_axis not in _SYMMETRIC_AXIS_INDEX:
            raise ValueError(
                f"symmetric_axis must be one of {sorted(_SYMMETRIC_AXIS_INDEX)} "
                f"or None, got {symmetric_axis!r}"
            )
        axis_idx = _SYMMETRIC_AXIS_INDEX[symmetric_axis]

    track_keys = sorted(instance_pts_dict.keys())
    name_to_id = {k: i for i, k in enumerate(track_keys)}
    dtype = torch.float32

    if dynamic_lidar_pts.numel() == 0 or len(track_keys) == 0:
        device = (dynamic_lidar_pts.device if dynamic_lidar_pts.numel()
                  else torch.device("cpu"))
        return (
            torch.zeros(0, 3, dtype=dtype, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
            track_keys,
        )

    M = dynamic_lidar_pts.shape[0]
    device = dynamic_lidar_pts.device
    ones = torch.ones(M, 1, dtype=dtype, device=device)
    pts_h = torch.cat([dynamic_lidar_pts.to(dtype), ones], dim=-1)  # [M, 4]

    all_pts: List[torch.Tensor] = []
    all_ids: List[torch.Tensor] = []
    for tid in track_keys:
        info = instance_pts_dict[tid]
        active_idx = info["frame_info"].nonzero(as_tuple=False).squeeze(-1)
        if active_idx.numel() == 0:
            info["pts"] = torch.zeros(0, 3, dtype=dtype, device=device)
            continue

        size_half = info["size"].to(dtype).to(device) / 2.0
        collected_local: List[torch.Tensor] = []
        for fi in active_idx.tolist():
            pose = info["poses"][fi].to(dtype).to(device)
            pose_inv = torch.linalg.inv(pose)
            # (4,4) @ (4,M) → (4,M); slice xyz
            local = (pose_inv @ pts_h.T).T[:, :3]                       # [M, 3]
            inside = (local.abs() <= size_half).all(dim=-1)
            collected_local.append(local[inside])

        track_pts = (torch.cat(collected_local, dim=0) if collected_local
                     else torch.zeros(0, 3, dtype=dtype, device=device))

        # V3-L5: NuRec ``symmetric_axis`` augmentation. Concatenate the
        # axis-mirrored copy BEFORE max_pts_per_track subsample so that the
        # cap acts as a single combined budget — the mirror does not increase
        # the per-track particle count when the original already saturates
        # the cap, but for sparse tracks (rear of clip, oblique cuboids) the
        # mirror doubles density and supplies the missing far-side LiDAR
        # returns we never observed.
        if axis_idx is not None and track_pts.shape[0] > 0:
            mirrored = track_pts.clone()
            mirrored[:, axis_idx] = -mirrored[:, axis_idx]
            track_pts = torch.cat([track_pts, mirrored], dim=0)

        if track_pts.shape[0] > max_pts_per_track:
            sel = torch.randperm(track_pts.shape[0], device=device)[:max_pts_per_track]
            track_pts = track_pts[sel]

        info["pts"] = track_pts
        all_pts.append(track_pts)
        all_ids.append(torch.full(
            (track_pts.shape[0],),
            name_to_id[tid],
            dtype=torch.long,
            device=device,
        ))

    positions = (torch.cat(all_pts, dim=0) if all_pts
                 else torch.zeros(0, 3, dtype=dtype, device=device))
    track_ids = (torch.cat(all_ids, dim=0) if all_ids
                 else torch.zeros(0, dtype=torch.long, device=device))
    return positions, track_ids, track_keys


# ---------------------------------------------------------------- P1.4 warm-start
_MERGE_KEYS = ("positions", "rotations", "scales", "densities", "colors", "track_ids")


def merge_warmstart_with_lidar(
    lidar_positions: torch.Tensor,
    lidar_track_ids: torch.Tensor,
    warm: Dict[str, torch.Tensor],
    *,
    max_pts_per_track: int,
    scale_prior,
    density_init: float,
    mode: str = "replace",
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """P1.4: combine LiDAR-init particles with warm-start (asset-harvester) ones.

    Both are object-local. LiDAR particles carry only positions+track_ids; their
    appearance/shape defaults (identity rot, ``log(scale_prior)``, ``density_init``,
    neutral 0.5 color) are materialized here so the merged set is a single
    ``init_layer_from_points`` call.

    Args:
        lidar_positions: ``[M, 3]`` object-local LiDAR particles.
        lidar_track_ids: ``[M]`` int track id per LiDAR particle.
        warm: ``assets_to_layer_inputs`` output (full per-particle attrs).
        max_pts_per_track: per-track combined budget (randperm cap).
        scale_prior: dynamic_rigids ``LayerSpec.scale_prior`` (3-tuple, metric).
        density_init: dynamic_rigids ``LayerSpec.density_init`` (pre-sigmoid).
        mode: ``"replace"`` (warm asset replaces a track's LiDAR) or ``"augment"``
            (concat LiDAR + warm under one budget). Tracks without a warm asset
            always keep their LiDAR particles.
        generator: optional seeded RNG for the subsample.

    Returns:
        kwargs dict for ``init_layer_from_points`` (``positions`` + colors /
        rotations / scales / densities / track_ids).
    """
    if mode not in ("replace", "augment"):
        raise ValueError(f"mode must be 'replace' or 'augment', got {mode!r}")

    device = lidar_positions.device
    dtype = torch.float32
    scale_default = torch.log(
        torch.tensor(list(scale_prior), dtype=dtype, device=device)
    )

    def _lidar_part(tid: int) -> Optional[Dict[str, torch.Tensor]]:
        mask = lidar_track_ids == tid
        k = int(mask.sum().item())
        if k == 0:
            return None
        rot = torch.zeros(k, 4, dtype=dtype, device=device)
        rot[:, 0] = 1.0
        return {
            "positions": lidar_positions[mask].to(dtype),
            "rotations": rot,
            "scales": scale_default.expand(k, 3).clone(),
            "densities": torch.full((k, 1), float(density_init), dtype=dtype, device=device),
            "colors": torch.full((k, 3), 0.5, dtype=dtype, device=device),
            "track_ids": torch.full((k,), tid, dtype=torch.int64, device=device),
        }

    def _warm_part(tid: int) -> Dict[str, torch.Tensor]:
        mask = warm["track_ids"] == tid
        return {
            "positions": warm["positions"][mask].to(dtype),
            "rotations": warm["rotations"][mask].to(dtype),
            "scales": warm["scales"][mask].to(dtype),
            "densities": warm["densities"][mask].to(dtype),
            "colors": warm["colors"][mask].to(dtype),
            "track_ids": warm["track_ids"][mask].to(torch.int64),
        }

    warm_ids = {int(t) for t in warm["track_ids"].unique().tolist()}
    lidar_ids = ({int(t) for t in lidar_track_ids.unique().tolist()}
                 if lidar_track_ids.numel() else set())

    out: Dict[str, List[torch.Tensor]] = {k: [] for k in _MERGE_KEYS}
    for tid in sorted(warm_ids | lidar_ids):
        has_warm = tid in warm_ids
        parts: List[Dict[str, torch.Tensor]] = []
        # Tracks without an asset keep LiDAR; augment keeps LiDAR even when warm.
        if (not has_warm) or mode == "augment":
            lp = _lidar_part(tid)
            if lp is not None:
                parts.append(lp)
        if has_warm:
            parts.append(_warm_part(tid))
        if not parts:
            continue

        merged = {k: torch.cat([p[k] for p in parts], dim=0) for k in _MERGE_KEYS}
        n_track = merged["positions"].shape[0]
        if n_track > max_pts_per_track:
            sel = torch.randperm(n_track, generator=generator, device=device)[:max_pts_per_track]
            merged = {k: v[sel] for k, v in merged.items()}
        for k in _MERGE_KEYS:
            out[k].append(merged[k])

    def _empty(key: str) -> torch.Tensor:
        if key == "track_ids":
            return torch.zeros(0, dtype=torch.int64, device=device)
        cols = {"positions": 3, "rotations": 4, "scales": 3, "densities": 1, "colors": 3}[key]
        return torch.zeros(0, cols, dtype=dtype, device=device)

    return {k: (torch.cat(v, dim=0) if v else _empty(k)) for k, v in out.items()}
