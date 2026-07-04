# SPDX-License-Identifier: Apache-2.0
"""Minimal CPU-safe LayeredGaussians ckpt accessor for diagnostic scripts (V3-VIZ).

Avoids instantiating ``LayeredGaussians`` (whose import chain pulls in
``threedgrut.model.model`` → ``threedgrt_tracer`` → ``ncore``) so diagnostics
run on Mac / ThinkPad without the NCore SDK installed. We only need per-layer
``positions`` and dyn_rigids ``track_ids``, both of which are stored verbatim
in ``ckpt["model"]["gaussians_nodes"][layer_name]`` by ``MoG.get_model_parameters``
and ``LayeredGaussians.get_model_parameters`` (T8/B3 Phase E.4).

For cuboid poses + sizes + activity, callers use ``FourDMetadata.from_ckpt``
directly (already pure-CPU).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch


def _to_np(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_ckpt_cpu(ckpt_path: Path) -> dict:
    """Load a v2 LayeredGaussians ckpt on CPU. Validates v2 shape."""
    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"ckpt is not a dict (got {type(ckpt).__name__}).")
    conf = ckpt.get("config")
    if conf is None and isinstance(ckpt.get("model"), dict):
        nodes = ckpt["model"].get("gaussians_nodes") or {}
        for _ln, payload in nodes.items():
            if isinstance(payload, dict) and "config" in payload:
                conf = payload["config"]
                break
    if conf is None:
        raise RuntimeError(
            "ckpt has no 'config' nor per-layer 'config' fallback — " "diagnostic requires a v2 LayeredGaussians ckpt."
        )
    use_layered = bool(conf.get("use_layered_model", False))
    if not use_layered:
        raise RuntimeError("ckpt is not a v2 LayeredGaussians ckpt (use_layered_model=false).")
    return ckpt


def extract_layer_positions(ckpt: dict) -> dict[str, np.ndarray]:
    """Return ``{layer_name: (N, 3) float32 np.ndarray}`` from ckpt state.

    Reads directly from ``ckpt["model"]["gaussians_nodes"][name]["positions"]``
    (v2 NRE-wrapped) or ``ckpt["gaussians_nodes"][name]["positions"]``
    (unwrapped fallback). Sky and other non-particle layers are absent.

    Positions are world-frame for static layers (background, road) and
    object-local for dynamic_rigids. Caller is responsible for transforming
    dynamic_rigids to world for a target frame via per-track SE(3) poses.
    """
    model_block = ckpt.get("model") if isinstance(ckpt.get("model"), dict) else None
    nodes = None
    if model_block is not None and isinstance(model_block.get("gaussians_nodes"), dict):
        nodes = model_block["gaussians_nodes"]
    elif isinstance(ckpt.get("gaussians_nodes"), dict):
        nodes = ckpt["gaussians_nodes"]
    if nodes is None:
        raise RuntimeError("ckpt has no 'gaussians_nodes' block.")

    out: dict[str, np.ndarray] = {}
    for name, payload in nodes.items():
        if not isinstance(payload, dict):
            continue
        pos = payload.get("positions")
        if pos is None:
            continue
        arr = _to_np(pos)
        if arr is None or arr.size == 0:
            out[name] = np.empty((0, 3), dtype=np.float32)
            continue
        out[name] = arr.astype(np.float32).reshape(-1, 3)
    return out


def extract_dyn_track_ids(ckpt: dict) -> Optional[np.ndarray]:
    """Return ``(N,)`` int64 track-id buffer for dynamic_rigids layer, or None.

    Each entry maps to ``sorted(viz_4d['tracks'].keys())[track_ids[i]]`` (the
    same convention LayeredGaussians uses at populate time).
    """
    model_block = ckpt.get("model") if isinstance(ckpt.get("model"), dict) else None
    if model_block is not None and isinstance(model_block.get("gaussians_nodes"), dict):
        nodes = model_block["gaussians_nodes"]
    elif isinstance(ckpt.get("gaussians_nodes"), dict):
        nodes = ckpt["gaussians_nodes"]
    else:
        return None
    payload = nodes.get("dynamic_rigids")
    if not isinstance(payload, dict):
        return None
    tids = payload.get("track_ids")
    if tids is None:
        return None
    arr = _to_np(tids)
    if arr is None:
        return None
    return arr.astype(np.int64).reshape(-1)


def dyn_local_to_world_at_frame(
    local_positions: np.ndarray,
    track_ids: np.ndarray,
    sorted_track_names: list[str],
    tracks: Mapping[str, dict],
    frame_idx: int,
) -> np.ndarray:
    """Transform dynamic_rigids object-local positions to world at ``frame_idx``.

    Args:
        local_positions: ``(N, 3)`` object-local positions of dyn_rigids particles.
        track_ids: ``(N,)`` int64 — index into ``sorted_track_names``.
        sorted_track_names: ordered track keys matching the ckpt's track_id map.
        tracks: FourDMetadata-style ``{tid: {"poses": (F,4,4), "frame_info": (F,) bool, ...}}``.
        frame_idx: which frame in the per-track timeline.

    Returns:
        ``(M, 3)`` float32 world positions for particles whose owning track is
        active at ``frame_idx``. Particles on inactive tracks are dropped.
    """
    if local_positions.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    chunks: list[np.ndarray] = []
    for ti, name in enumerate(sorted_track_names):
        mask = track_ids == ti
        if not mask.any():
            continue
        track = tracks.get(name)
        if track is None:
            continue
        active = track.get("frame_info")
        if active is None or frame_idx >= active.shape[0] or not bool(active[frame_idx]):
            continue
        poses = track.get("poses")
        if poses is None or frame_idx >= poses.shape[0]:
            continue
        pose = poses[frame_idx]
        local = local_positions[mask]
        world = local @ pose[:3, :3].T + pose[:3, 3]
        chunks.append(world.astype(np.float32))
    if not chunks:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(chunks, axis=0)
