# SPDX-License-Identifier: Apache-2.0
"""Phase 2A road-layer BEV hole quantifier (pure numpy, CPU/Mac-safe).

Splits "the road looks holey from a top-down (BEV) angle" into two measurable
failure modes over an ego-corridor grid:

  * **B-type — geometry hole**: a corridor cell that contains *no* road particle
    at all. Pure absence. (Init places particles across the whole bbox, so any
    B-type holes are produced by MCMC relocate concentrating particles +
    sub-sampling gaps, not by missing init coverage — see road_init.py.)

  * **A-type — transparency hole**: a corridor cell that *does* contain road
    particle(s), but the strongest one is too transparent to render an opaque
    surface from straight above (max opacity < floor). Particle is there, but
    invisible top-down — the signature of weak grazing-angle supervision.

The "corridor" is the drivable region we actually care about: cells whose
center is within ``corridor_half_width`` metres of the ego trajectory polyline.
bbox padding outside the corridor is excluded so we don't count never-road
padding as holes.

Pure function ``compute_bev_hole_stats`` takes plain numpy arrays so it unit
tests on Mac with synthetic data — no ckpt / torch / GPU needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


def _nearest_ego_distance(cell_centers_xy: np.ndarray, ego_xy: np.ndarray) -> np.ndarray:
    """Distance from each cell center to the nearest ego sample.

    Uses scipy cKDTree when available (fast); falls back to chunked brute force
    so the function has no hard scipy dependency for small inputs / unit tests.
    Approximates distance-to-polyline by distance-to-nearest-vertex, which is
    accurate when ego samples are dense (NCore ego poses are ~1-2 m apart).
    """
    if cell_centers_xy.size == 0:
        return np.empty((0,), dtype=np.float64)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(np.ascontiguousarray(ego_xy))
        dist, _ = tree.query(np.ascontiguousarray(cell_centers_xy), k=1)
        return np.asarray(dist, dtype=np.float64)
    except ImportError:
        out = np.empty(cell_centers_xy.shape[0], dtype=np.float64)
        chunk = 4096
        for i in range(0, cell_centers_xy.shape[0], chunk):
            c = cell_centers_xy[i : i + chunk]  # [k, 2]
            d2 = ((c[:, None, :] - ego_xy[None, :, :]) ** 2).sum(-1)  # [k, F]
            out[i : i + chunk] = np.sqrt(d2.min(axis=1))
        return out


@dataclass
class BevHoleStats:
    cell_size: float
    corridor_half_width: float
    opacity_floors: tuple[float, ...]

    # grid geometry
    x0: float
    y0: float
    nx: int
    ny: int

    # particle-level
    n_particles: int
    opacity_percentiles: dict[str, float]  # p10/p25/p50/p75/p90/mean

    # corridor-level counts
    n_corridor_cells: int
    n_corridor_occupied: int

    # headline rates
    b_geometry_hole_rate: float  # empty corridor cells / corridor cells
    # per-floor: A-type rate over *occupied* corridor cells, and opaque coverage
    a_transparency_hole_rate: dict[str, float] = field(default_factory=dict)
    opaque_coverage: dict[str, float] = field(default_factory=dict)

    # grids (for heatmaps); excluded from to_dict's JSON-light form
    count_grid: np.ndarray | None = None
    maxop_grid: np.ndarray | None = None
    corridor_mask_grid: np.ndarray | None = None

    def to_dict(self) -> dict:
        """JSON-serialisable summary (drops the heavy grids)."""
        return {
            "cell_size": self.cell_size,
            "corridor_half_width": self.corridor_half_width,
            "opacity_floors": list(self.opacity_floors),
            "grid": {"x0": self.x0, "y0": self.y0, "nx": self.nx, "ny": self.ny},
            "n_particles": self.n_particles,
            "opacity_percentiles": self.opacity_percentiles,
            "n_corridor_cells": self.n_corridor_cells,
            "n_corridor_occupied": self.n_corridor_occupied,
            "b_geometry_hole_rate": self.b_geometry_hole_rate,
            "a_transparency_hole_rate": self.a_transparency_hole_rate,
            "opaque_coverage": self.opaque_coverage,
        }


def compute_bev_hole_stats(
    particle_xy: np.ndarray,       # [N, 2] world XY of road particles
    particle_opacity: np.ndarray,  # [N]    in [0, 1] (post-sigmoid)
    ego_xy: np.ndarray,            # [F, 2] world XY of ego trajectory
    *,
    cell_size: float = 0.5,
    corridor_half_width: float = 12.0,
    opacity_floors: Sequence[float] = (0.05, 0.1, 0.3),
) -> BevHoleStats:
    """Quantify A-type (transparency) vs B-type (geometry) BEV road holes.

    Returns a :class:`BevHoleStats`. See module docstring for definitions.
    """
    particle_xy = np.asarray(particle_xy, dtype=np.float64).reshape(-1, 2)
    particle_opacity = np.asarray(particle_opacity, dtype=np.float64).reshape(-1)
    ego_xy = np.asarray(ego_xy, dtype=np.float64).reshape(-1, 2)
    if particle_xy.shape[0] != particle_opacity.shape[0]:
        raise ValueError(
            f"particle_xy ({particle_xy.shape[0]}) and opacity "
            f"({particle_opacity.shape[0]}) length mismatch"
        )
    if particle_xy.shape[0] == 0 or ego_xy.shape[0] == 0:
        raise ValueError("need at least one particle and one ego sample")
    if cell_size <= 0:
        raise ValueError("cell_size must be > 0")

    floors = tuple(float(f) for f in opacity_floors)

    # --- particle opacity percentiles (validation handle: B3_30k median ~0.014) ---
    pcts = {
        "p10": float(np.percentile(particle_opacity, 10)),
        "p25": float(np.percentile(particle_opacity, 25)),
        "p50": float(np.percentile(particle_opacity, 50)),
        "p75": float(np.percentile(particle_opacity, 75)),
        "p90": float(np.percentile(particle_opacity, 90)),
        "mean": float(particle_opacity.mean()),
    }

    # --- grid bounds: the ego corridor only (ego bbox + pad), NOT the particle
    # bbox. bg/road particles sprawl hundreds of metres (sky-ish / far field);
    # the corridor is all we score, and ego_bbox + corridor_half_width fully
    # contains every point within corridor_half_width of the ego polyline. This
    # keeps the grid bounded and drops far-field particles instead of clipping
    # them onto boundary cells (which would pollute corridor-edge stats). ---
    pad = corridor_half_width + cell_size
    x0 = float(ego_xy[:, 0].min() - pad)
    y0 = float(ego_xy[:, 1].min() - pad)
    x1 = float(ego_xy[:, 0].max() + pad)
    y1 = float(ego_xy[:, 1].max() + pad)
    nx = max(1, int(np.ceil((x1 - x0) / cell_size)))
    ny = max(1, int(np.ceil((y1 - y0) / cell_size)))
    ncells = nx * ny

    # --- bin only particles inside the grid bbox (outside = beyond corridor) ---
    in_box = (
        (particle_xy[:, 0] >= x0) & (particle_xy[:, 0] < x1)
        & (particle_xy[:, 1] >= y0) & (particle_xy[:, 1] < y1)
    )
    bxy = particle_xy[in_box]
    bop = particle_opacity[in_box]
    ix = np.clip(((bxy[:, 0] - x0) / cell_size).astype(np.int64), 0, nx - 1)
    iy = np.clip(((bxy[:, 1] - y0) / cell_size).astype(np.int64), 0, ny - 1)
    flat = ix * ny + iy  # row-major [nx, ny]

    count_flat = np.bincount(flat, minlength=ncells).astype(np.int64)
    maxop_flat = np.zeros(ncells, dtype=np.float64)
    if flat.size:
        np.maximum.at(maxop_flat, flat, bop)

    # --- corridor mask: cell center within corridor_half_width of ego polyline ---
    cell_ix = np.arange(ncells) // ny
    cell_iy = np.arange(ncells) % ny
    centers = np.stack(
        [x0 + (cell_ix + 0.5) * cell_size, y0 + (cell_iy + 0.5) * cell_size], axis=-1
    )
    ego_dist = _nearest_ego_distance(centers, ego_xy)
    corridor = ego_dist <= corridor_half_width  # [ncells] bool

    n_corr = int(corridor.sum())
    occupied = count_flat > 0
    corr_occ = corridor & occupied
    n_corr_occ = int(corr_occ.sum())

    # --- B-type: empty corridor cells ---
    b_rate = float((corridor & ~occupied).sum() / n_corr) if n_corr else 0.0

    # --- A-type per floor: occupied-but-too-transparent / occupied; opaque coverage ---
    a_rate: dict[str, float] = {}
    cov: dict[str, float] = {}
    for f in floors:
        key = f"{f:g}"
        transp = corr_occ & (maxop_flat < f)
        a_rate[key] = float(transp.sum() / n_corr_occ) if n_corr_occ else 0.0
        opaque = corridor & (maxop_flat >= f)
        cov[key] = float(opaque.sum() / n_corr) if n_corr else 0.0

    return BevHoleStats(
        cell_size=cell_size,
        corridor_half_width=corridor_half_width,
        opacity_floors=floors,
        x0=x0,
        y0=y0,
        nx=nx,
        ny=ny,
        n_particles=int(particle_xy.shape[0]),
        opacity_percentiles=pcts,
        n_corridor_cells=n_corr,
        n_corridor_occupied=n_corr_occ,
        b_geometry_hole_rate=b_rate,
        a_transparency_hole_rate=a_rate,
        opaque_coverage=cov,
        count_grid=count_flat.reshape(nx, ny),
        maxop_grid=maxop_flat.reshape(nx, ny),
        corridor_mask_grid=corridor.reshape(nx, ny),
    )
