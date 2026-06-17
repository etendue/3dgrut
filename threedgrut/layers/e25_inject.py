# SPDX-License-Identifier: Apache-2.0
"""E2.5 — frozen AH-asset drop-in replacing recon cars (offline ckpt surgery).

Pure-CPU core (no CUDA, no trainer) so it runs on Mac and is unit-testable.
Reuses the PR #18 alignment engine (``warmstart_ply`` / ``warmstart_metadata``)
to turn asset-harvester PLYs into object-local, metric, cuboid-filling particles,
then performs *per-track subset replacement* directly on a checkpoint dict:
the mapped recon tracks' particles are swapped for the AH particles while every
untouched recon track keeps its learned tensors byte-identical.

Frozen: no optimizer is built, no training happens. The edited dict is consumed
by ``LayeredGaussians.init_from_checkpoint(..., setup_optimizer=False)``.
"""
from __future__ import annotations

import torch

from threedgrut.layers.warmstart_ply import AlignedAsset, AlignmentTransform
from threedgrut.layers.layered_model import _SH_C0, _rotmat_to_quat_wxyz

# MoG dynamic_rigids node tensors that scale with particle count.
_PARTICLE_KEYS = (
    "positions", "rotation", "scale", "density",
    "features_albedo", "features_specular", "track_ids",
)


def flip_forward_180(xf: AlignmentTransform) -> AlignmentTransform:
    """Compose a 180° yaw (about object-local up/Z) onto an alignment transform.

    PR #18's ``_VEHICLE_AXIS_MAP`` is calibrated to the NuRec demo USDZ canonical
    orientation, but E2.5 places cars onto **NCore cuboid** trajectories whose
    forward(+X) convention is the opposite — so without this the injected car
    drives backwards (head/tail swapped). A 180° yaw negates the forward(X) and
    left(Y) axes while keeping up(Z), which is a proper rotation (det +1), not a
    mirror. scale_local / center / perm are unchanged.
    """
    Rz = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=xf.R.dtype,
    )
    R_new = Rz @ xf.R
    return AlignmentTransform(
        R=R_new,
        q_R=_rotmat_to_quat_wxyz(R_new),
        scale_local=xf.scale_local,
        center=xf.center,
        perm=xf.perm,
    )


def build_name_to_int_id(tracks: dict) -> dict:
    """``{track_name: integer_id}`` via ``enumerate(sorted(names))``.

    This MUST mirror ``_transform_means_and_active``'s
    ``sorted(self.tracks_poses.keys())`` indexing, so the integer ids we reuse
    for injected particles point at the same cuboid trajectory the recon car
    occupied. Diverge here and AH cars render on the wrong track.
    """
    return {name: i for i, name in enumerate(sorted(tracks.keys()))}


def match_assets_by_size(ah_dims, recon_sizes):
    """Greedy size match → ``{recon_track_key: ah_hash}``.

    ``ah_dims``: ``{asset_hash: (L,W,H)}``; ``recon_sizes``: ``{track_key:
    (L,W,H)}`` (same order/units). Pairs are taken in ascending per-axis L2
    distance, each asset/track used once; at most ``min(len)`` tracks mapped.
    """
    pairs = []
    for tk, rsize in recon_sizes.items():
        r = torch.as_tensor(rsize, dtype=torch.float32)
        for ah, asize in ah_dims.items():
            a = torch.as_tensor(asize, dtype=torch.float32)
            dist = float(torch.linalg.norm(a - r))
            pairs.append((dist, str(tk), str(ah)))
    pairs.sort()  # ascending dist, then track_key, then hash → deterministic
    used_tracks: set = set()
    used_ah: set = set()
    out: dict = {}
    for _dist, tk, ah in pairs:
        if tk in used_tracks or ah in used_ah:
            continue
        out[tk] = ah
        used_tracks.add(tk)
        used_ah.add(ah)
    return out


def aligned_to_node_tensors(aligned: AlignedAsset, spec_dim: int) -> dict:
    """One AlignedAsset → the 6 MoG node tensors (specular zero-filled).

    ``features_albedo`` is the exact inverse of init_layer_from_points'
    ``(colors-0.5)/_SH_C0`` recovery, so colours survive losslessly. Geometry
    (positions/rotation/scale/density) is passed through verbatim.
    """
    n = aligned.positions.shape[0]
    return {
        "positions": aligned.positions,
        "rotation": aligned.rotations,
        "scale": aligned.scales_log,
        "density": aligned.density_logit,
        "features_albedo": (aligned.colors - 0.5) / _SH_C0,
        "features_specular": torch.zeros(
            (n, spec_dim), dtype=aligned.positions.dtype
        ),
    }


def replace_tracks_in_dyn_node(dyn_node: dict, aligned_by_id: dict) -> dict:
    """Per-track subset replacement on a dynamic_rigids node dict.

    ``aligned_by_id``: ``{int_track_id: AlignedAsset}``. Returns a new node dict
    with those tracks' particles replaced by AH particles, untouched tracks kept
    byte-identical, non-tensor metadata carried over.
    """
    track_ids = dyn_node["track_ids"]
    spec_dim = dyn_node["features_specular"].shape[1]
    target_ids = sorted(int(t) for t in aligned_by_id.keys())
    keep = ~torch.isin(track_ids, torch.tensor(target_ids, dtype=track_ids.dtype))

    # Pre-build AH node tensors per target (deterministic order = sorted ids).
    ah_nodes = {
        tid: aligned_to_node_tensors(aligned_by_id[tid], spec_dim)
        for tid in target_ids
    }

    new: dict = {}
    for key in (
        "positions", "rotation", "scale", "density",
        "features_albedo", "features_specular",
    ):
        kept = dyn_node[key][keep]
        ah_parts = [ah_nodes[tid][key] for tid in target_ids]
        # MoG.init_from_checkpoint assigns these onto registered nn.Parameter
        # fields, so the edited dict must carry Parameters (frozen: no grad).
        new[key] = torch.nn.Parameter(
            torch.cat([kept, *ah_parts], dim=0), requires_grad=False
        )

    ah_ids = [
        torch.full(
            (aligned_by_id[tid].positions.shape[0],), tid, dtype=track_ids.dtype
        )
        for tid in target_ids
    ]
    new["track_ids"] = torch.cat([track_ids[keep], *ah_ids], dim=0)

    # Carry non-particle metadata (n_active_features / scene_extent / optimizer /
    # background / config / ...) verbatim so the dict stays reload-compatible.
    for k, v in dyn_node.items():
        if k not in _PARTICLE_KEYS:
            new[k] = v
    return new
