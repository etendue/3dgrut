# SPDX-License-Identifier: Apache-2.0
"""P1.4 asset-harvester warm-start — PLY load + canonical→object-local alignment.

Pure CPU engine (no CUDA, no trainer coupling) so the AH-0 gate runs on Mac.
Wraps the existing :class:`PLYImporter` (do not reimplement binary PLY parsing)
and turns an Objaverse-Y-up-canonical, per-axis-normalized asset into
object-local (X=forward, Y=left, Z=up), metric particles ready for
``LayeredGaussians.init_layer_from_points``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from threedgrut.export.importers.ply import PLYImporter
from threedgrut.layers.layered_model import (
    _SH_C0,
    _quat_multiply_wxyz,
    _rotmat_to_quat_wxyz,
)


@dataclass
class WarmStartAsset:
    """Raw asset loaded from PLY (Objaverse-canonical, normalized, pre-activation).

    positions [N,3] canonical · rotations [N,4] wxyz · scales_log [N,3] log ·
    density_logit [N,1] (pre-sigmoid) · albedo [N,3] (=f_dc DC-SH term).
    """

    positions: torch.Tensor
    rotations: torch.Tensor
    scales_log: torch.Tensor
    density_logit: torch.Tensor
    albedo: torch.Tensor


@dataclass(frozen=True)
class AxisMap:
    """Objaverse-canonical → object-local signed axis permutation.

    ``perm[i]`` = PLY axis feeding object-local axis ``i`` (X=fwd, Y=left, Z=up);
    ``sign[i]`` its sign. Chosen so the resulting matrix is a proper rotation
    (det +1).
    """

    perm: tuple[int, int, int]
    sign: tuple[float, float, float]


# Objaverse Y-up → NCore object-local (X=fwd, Y=left, Z=up). Empirically every
# demo asset (3 cars + 3 peds) rank-matches half-spans↔cuboids_dims to this same
# perm (0,2,1): local-X←ply-x (length), local-Z←ply-y (up), local-Y←ply-z (width,
# sign-flipped to keep det(R)=+1). Both classes share the canonical orientation.
_VEHICLE_AXIS_MAP = AxisMap(perm=(0, 2, 1), sign=(1.0, -1.0, 1.0))
_CANONICAL_AXIS_MAP: dict[str, AxisMap] = {
    # NuRec demo benchmark class names
    "consumer_vehicles": _VEHICLE_AXIS_MAP,
    "VRU_pedestrians": _VEHICLE_AXIS_MAP,
    # NCore ncore_parser class names (same Objaverse Y-up canonical)
    "automobile": _VEHICLE_AXIS_MAP,
    "bus": _VEHICLE_AXIS_MAP,
    "heavy_truck": _VEHICLE_AXIS_MAP,
    "person": _VEHICLE_AXIS_MAP,
}


@dataclass
class AlignmentTransform:
    """Resolved canonical→object-local transform (see :func:`compute_axis_alignment`)."""

    R: torch.Tensor           # [3,3] proper rotation (signed permutation)
    q_R: torch.Tensor         # [4] wxyz form of R
    scale_local: torch.Tensor  # [3] per-object-local-axis metric factor
    center: torch.Tensor      # [3] PLY-frame center to subtract first
    perm: tuple[int, int, int]


@dataclass
class AlignedAsset:
    """Object-local, metric particles ready for ``init_layer_from_points``.

    colors [N,3] in [0,1] (init re-derives albedo); density_logit passthrough.
    """

    positions: torch.Tensor
    rotations: torch.Tensor
    scales_log: torch.Tensor
    density_logit: torch.Tensor
    colors: torch.Tensor


def albedo_to_colors(albedo: torch.Tensor) -> torch.Tensor:
    """DC-SH albedo (=f_dc) → RGB in (approx) [0,1].

    Exact inverse of ``init_layer_from_points``'s ``(colors-0.5)/_SH_C0`` albedo
    recovery, so warm-start colors survive the round-trip through the existing
    injection entrypoint losslessly.
    """
    return albedo * _SH_C0 + 0.5


def load_warmstart_ply(path: str | Path, *, max_sh_degree: int = 0) -> WarmStartAsset:
    """Load an asset-harvester PLY into a :class:`WarmStartAsset`.

    Wraps :class:`PLYImporter` (pre-activation semantics: opacity=logit,
    scale=log, f_dc=albedo, rot=wxyz). Higher-order SH is dropped — warm-start
    re-learns view-dependence in-scene, and ``init_layer_from_points`` zero-fills
    ``features_specular``. Raises on empty / non-finite / malformed assets.
    """
    attrs, _caps = PLYImporter(max_sh_degree=max_sh_degree).load(Path(path))
    pos = torch.from_numpy(attrs.positions).float()
    rot = torch.from_numpy(attrs.rotations).float()
    scl = torch.from_numpy(attrs.scales).float()
    den = torch.from_numpy(attrs.densities).float()
    alb = torch.from_numpy(attrs.albedo).float()

    n = pos.shape[0]
    if n == 0:
        raise ValueError(f"warm-start PLY {path!s} has zero vertices")
    if rot.shape != (n, 4):
        raise ValueError(f"warm-start PLY {path!s} rotations {tuple(rot.shape)} != ({n}, 4)")
    if scl.shape != (n, 3):
        raise ValueError(f"warm-start PLY {path!s} scales {tuple(scl.shape)} != ({n}, 3)")
    for name, t in (("positions", pos), ("rotations", rot), ("scales", scl),
                    ("densities", den), ("albedo", alb)):
        if not torch.isfinite(t).all():
            raise ValueError(f"warm-start PLY {path!s} has non-finite {name}")
    return WarmStartAsset(
        positions=pos, rotations=rot, scales_log=scl,
        density_logit=den, albedo=alb,
    )


def asset_extent(asset: WarmStartAsset) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-axis half-span and center of an asset's point cloud → ``(half[3], center[3])``."""
    lo = asset.positions.amin(dim=0)
    hi = asset.positions.amax(dim=0)
    return (hi - lo) * 0.5, (hi + lo) * 0.5


def _rank_match_axismap(cuboids_dims: torch.Tensor, ply_halfspan: torch.Tensor) -> AxisMap:
    """Fallback for unknown classes: match PLY half-spans to cuboid dims by rank
    (largest↔largest), signs default +1 with one flip to force det(R)=+1."""
    rank_dim = sorted(range(3), key=lambda i: float(cuboids_dims[i]))
    rank_half = sorted(range(3), key=lambda j: float(ply_halfspan[j]))
    perm = [0, 0, 0]
    for r in range(3):
        perm[rank_dim[r]] = rank_half[r]
    sign = [1.0, 1.0, 1.0]
    R = torch.zeros(3, 3)
    for i in range(3):
        R[i, perm[i]] = sign[i]
    if torch.det(R).item() < 0:
        sign[1] = -1.0  # flip local-Y (left/right) → proper rotation
    return AxisMap(perm=(perm[0], perm[1], perm[2]),
                   sign=(sign[0], sign[1], sign[2]))


def compute_axis_alignment(
    label_class: str,
    cuboids_dims,
    ply_halfspan: torch.Tensor,
    ply_center: torch.Tensor,
) -> AlignmentTransform:
    """Resolve the canonical→object-local transform for one asset.

    Uses the per-class :data:`_CANONICAL_AXIS_MAP` (Objaverse Y-up) when known,
    else rank-matches half-spans↔dims. Per-axis metric scale makes the aligned
    cloud fill the cuboid exactly along each axis (containment + fill by
    construction). Raises if the resolved matrix is not a proper rotation.
    """
    dims = torch.as_tensor(cuboids_dims, dtype=torch.float32)
    half = torch.as_tensor(ply_halfspan, dtype=torch.float32)
    center = torch.as_tensor(ply_center, dtype=torch.float32)
    amap = _CANONICAL_AXIS_MAP.get(label_class) or _rank_match_axismap(dims, half)

    R = torch.zeros(3, 3)
    for i in range(3):
        R[i, amap.perm[i]] = amap.sign[i]
    det = torch.det(R).item()
    if abs(det - 1.0) > 1e-5:
        raise ValueError(
            f"axis map for {label_class!r} yields det(R)={det:.3f}, not +1 "
            f"(perm={amap.perm}, sign={amap.sign})"
        )

    half_perm = half[list(amap.perm)]
    scale_local = (dims * 0.5) / half_perm.clamp_min(1e-9)
    q_R = _rotmat_to_quat_wxyz(R)
    return AlignmentTransform(
        R=R, q_R=q_R, scale_local=scale_local, center=center, perm=amap.perm,
    )


def apply_alignment(asset: WarmStartAsset, xf: AlignmentTransform) -> AlignedAsset:
    """Apply a resolved transform: center → rotate → per-axis metric scale.

    positions exact; quaternions rotated by R and renormalized; log-scales get
    the axis permutation + an additive ``log(scale_local)`` shift (exact when the
    metric stretch is isotropic, a close approximation under mild anisotropy that
    training refines); density passthrough; colors = albedo→[0,1].
    """
    p = asset.positions - xf.center                       # [N,3] centered
    pos_local = (xf.R @ p.T).T * xf.scale_local           # rotate then scale
    n = asset.positions.shape[0]
    q_local = _quat_multiply_wxyz(xf.q_R.unsqueeze(0).expand(n, 4), asset.rotations)
    q_local = q_local / q_local.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scales_log_local = asset.scales_log[:, list(xf.perm)] + torch.log(xf.scale_local)
    colors = albedo_to_colors(asset.albedo)
    return AlignedAsset(
        positions=pos_local,
        rotations=q_local,
        scales_log=scales_log_local,
        density_logit=asset.density_logit,
        colors=colors,
    )


def subsample_asset(
    aligned: AlignedAsset, max_pts: int, *, generator: torch.Generator | None = None
) -> AlignedAsset:
    """Uniform random subsample to ``max_pts`` (no-op if already under budget).

    Mirrors the LiDAR-init randperm cap; pass a seeded ``generator`` for
    reproducible picks in tests / deterministic runs.
    """
    n = aligned.positions.shape[0]
    if n <= max_pts:
        return aligned
    sel = torch.randperm(n, generator=generator)[:max_pts]
    return AlignedAsset(
        positions=aligned.positions[sel],
        rotations=aligned.rotations[sel],
        scales_log=aligned.scales_log[sel],
        density_logit=aligned.density_logit[sel],
        colors=aligned.colors[sel],
    )


def assets_to_layer_inputs(aligned_list: list[tuple[int, AlignedAsset]]) -> dict:
    """Concat per-track aligned assets → ``init_layer_from_points`` kwargs.

    Returns ``{positions, rotations, scales, densities, colors, track_ids}`` with
    ``track_ids[Σ]`` int64 tagging each particle with its integer track id.
    """
    positions, rotations, scales, densities, colors, track_ids = [], [], [], [], [], []
    for tid, a in aligned_list:
        k = a.positions.shape[0]
        positions.append(a.positions)
        rotations.append(a.rotations)
        scales.append(a.scales_log)
        densities.append(a.density_logit)
        colors.append(a.colors)
        track_ids.append(torch.full((k,), int(tid), dtype=torch.int64))
    return {
        "positions": torch.cat(positions, dim=0),
        "rotations": torch.cat(rotations, dim=0),
        "scales": torch.cat(scales, dim=0),
        "densities": torch.cat(densities, dim=0),
        "colors": torch.cat(colors, dim=0),
        "track_ids": torch.cat(track_ids, dim=0),
    }
