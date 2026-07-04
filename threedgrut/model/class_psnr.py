# SPDX-License-Identifier: Apache-2.0
"""T8/B3 Phase E.5 — per-cuboid (per-class / per-object) PSNR.

``mean_psnr`` / ``psnr_masked`` aggregate across the entire image so they cannot
tell whether the ``dynamic_rigids`` layer actually explained vehicle pixels.
Phase C's smoke run had decent mean_psnr but cc_psnr fell 3 dB because dyn
particles rendered to the wrong positions (Bug E1/E2/E3) — bg ended up
explaining vehicle pixels and the per-image average hid the structural
failure.

``compute_class_psnr`` projects each active cuboid to a 2D AABB mask via
``project_cuboids_to_mask`` (FTheta-aware after Phase B), computes per-track
PSNR over that mask, and aggregates per class. This gives a direct yes/no
answer to "did dyn explain track X's pixels?".

Why a separate module: trainer.compute_metrics and render.py both need to
call this on the eval path (CLAUDE.md A/B "T6F.2 教训" — metric must be
plumbed in both places); keeping the function pure (no model / no trainer
state) lets both consumers reuse the same code and lets the Mac test suite
exercise it without a real renderer.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional

import torch

from threedgrut.layers.dynamic_mask import project_cuboids_to_mask


def compute_psnr_in_mask(
    rgb_pred: torch.Tensor,  # [H, W, 3] in [0, 1]
    rgb_gt: torch.Tensor,  # [H, W, 3]
    mask: torch.Tensor,  # [H, W] float or bool — 1 inside the region
    min_pixels: int = 50,
) -> Optional[float]:
    """Compute PSNR over the masked region only.

    Returns ``None`` when ``mask`` covers fewer than ``min_pixels`` (numerical
    instability — the masked-mse is too sensitive to a handful of pixels).
    Mirrors trainer.compute_metrics' PSNR_masked formula at
    threedgrut/trainer.py:691-695, restricted to the cuboid AABB.
    """
    mask_f = mask.to(rgb_pred.dtype)
    n_pix = float(mask_f.sum().item())
    if n_pix < min_pixels:
        return None
    # Broadcast mask [H, W] → [H, W, 1] over channels
    diff_sq = (rgb_pred - rgb_gt).pow(2) * mask_f.unsqueeze(-1)
    denom = n_pix * 3.0
    mse = diff_sq.sum().item() / denom
    if mse <= 0:
        return float("inf")
    psnr = -10.0 * math.log10(mse)
    return float(psnr)


def compute_class_psnr(
    rgb_pred: torch.Tensor,  # [B, H, W, 3] in [0, 1]
    rgb_gt: torch.Tensor,  # [B, H, W, 3]
    valid_mask: Optional[torch.Tensor],  # [B, H, W, 1] or None
    active_tracks: List[Dict[str, object]],  # see schema in fn body
    T_world2cam: torch.Tensor,  # [4, 4]
    H: int,
    W: int,
    *,
    K: Optional[torch.Tensor] = None,
    ftheta_params: Optional[Dict[str, object]] = None,
    min_pixels: int = 50,
) -> Dict[str, object]:
    """Per-cuboid PSNR aggregated by class.

    Args:
        rgb_pred, rgb_gt: ``[B, H, W, 3]`` rendered + GT images, values in
            ``[0, 1]``. Only batch[0] is used (one image per metric call,
            matching trainer.compute_metrics convention).
        valid_mask: optional ``[B, H, W, 1]`` per-pixel validity (ego mask
            removed already). When provided, per-track cuboid mask is AND-ed
            with it.
        active_tracks: list of ``{"id": str, "class": str, "pose": [4,4]
            tensor, "size": [3] tensor}`` for tracks active at this frame.
            Build via ``bg_cuboid_loss.collect_active_cuboids_for_frame`` +
            tracks_metadata.
        T_world2cam: ``[4, 4]`` world→camera SE(3) (OpenCV convention).
        H, W: image dimensions in pixels.
        K, ftheta_params: pinhole vs FTheta intrinsics (exclusive — see
            ``project_cuboids_to_mask``).
        min_pixels: minimum cuboid-mask pixel count for a stable PSNR; tracks
            below this threshold are tagged ``psnr=None``.

    Returns:
        ``{"per_track": [...], "mean": float | None, "by_class": {...},
           "n_tracks": int, "n_tracks_with_psnr": int}``

        Where ``per_track`` is ``[{"track_id", "class", "psnr": float|None,
        "n_pixels": int}]`` and ``by_class`` is ``{class_name: {"mean_psnr":
        float | None, "n_tracks": int}}``.
    """
    device = rgb_pred.device
    rgb_pred_one = rgb_pred[0]  # [H, W, 3]
    rgb_gt_one = rgb_gt[0]  # [H, W, 3]
    valid_mask_one = valid_mask[0, ..., 0].to(rgb_pred.dtype) if valid_mask is not None else None

    per_track: List[Dict[str, object]] = []
    n_with_psnr = 0
    by_class: Dict[str, List[float]] = defaultdict(list)

    for trk in active_tracks:
        tid = str(trk["id"])
        cls = str(trk.get("class", "unknown"))
        pose = trk["pose"].to(device=device, dtype=rgb_pred.dtype).unsqueeze(0)  # [1, 4, 4]
        size = trk["size"].to(device=device, dtype=rgb_pred.dtype).unsqueeze(0)  # [1, 3]

        cuboid_mask = project_cuboids_to_mask(
            pose,
            size,
            K,
            T_world2cam,
            H,
            W,
            device=device,
            ftheta_params=ftheta_params,
        )  # [H, W] bool
        if valid_mask_one is not None:
            mask_f = cuboid_mask.to(rgb_pred.dtype) * valid_mask_one
        else:
            mask_f = cuboid_mask.to(rgb_pred.dtype)
        n_pix = int(mask_f.sum().item())
        psnr = compute_psnr_in_mask(
            rgb_pred_one,
            rgb_gt_one,
            mask_f,
            min_pixels=min_pixels,
        )
        per_track.append(
            {
                "track_id": tid,
                "class": cls,
                "psnr": psnr,
                "n_pixels": n_pix,
            }
        )
        if psnr is not None:
            n_with_psnr += 1
            by_class[cls].append(psnr)

    # Aggregate
    psnr_values = [r["psnr"] for r in per_track if r["psnr"] is not None]
    mean_psnr = (sum(psnr_values) / len(psnr_values)) if psnr_values else None
    by_class_agg: Dict[str, Dict[str, object]] = {}
    for cls, vals in by_class.items():
        by_class_agg[cls] = {
            "mean_psnr": sum(vals) / len(vals) if vals else None,
            "n_tracks": len(vals),
        }
    return {
        "per_track": per_track,
        "mean": mean_psnr,
        "by_class": by_class_agg,
        "n_tracks": len(active_tracks),
        "n_tracks_with_psnr": n_with_psnr,
    }


def collect_active_tracks_for_frame(
    tracks_poses: Dict[str, torch.Tensor],
    tracks_active: Dict[str, torch.Tensor],
    tracks_metadata: Dict[str, Dict[str, object]],
    frame_idx: int,
) -> List[Dict[str, object]]:
    """Build the ``active_tracks`` list expected by ``compute_class_psnr``.

    Mirrors ``bg_cuboid_loss.collect_active_cuboids_for_frame`` but also
    threads through the per-track ``class`` and ``id`` strings via
    ``tracks_metadata``. Tracks without a size in metadata are skipped (we
    can't project them).
    """
    out: List[Dict[str, object]] = []
    for tid in sorted(tracks_poses.keys()):
        poses = tracks_poses[tid]
        active = tracks_active.get(tid)
        if active is None or frame_idx < 0 or frame_idx >= int(active.shape[0]):
            continue
        if not bool(active[frame_idx]):
            continue
        meta = tracks_metadata.get(tid, {}) or {}
        size = meta.get("size")
        if size is None:
            continue
        out.append(
            {
                "id": str(tid),
                "class": str(meta.get("class", "unknown")),
                "pose": poses[frame_idx],
                "size": size,
            }
        )
    return out
