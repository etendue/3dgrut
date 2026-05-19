# SPDX-License-Identifier: Apache-2.0
"""scene_manifest tracks → instance_pts_dict loader (T4.1.b).

Lives in its own module (not in datasetNcore.py) so unit tests can import it
without triggering the NCore SDK / cv2 / kornia chain that datasets/__init__.py
pulls in on Mac.

datasetNcore.py re-exports this at module-level (T4.5) so trainer.init_model
can call `from threedgrut.datasets.datasetNcore import load_tracks_from_manifest`
in line with v2_plan.md's path table.

Output schema mirrors drivestudio's get_init_objects (driving_dataset.py:263-396)
but is rebuilt from scratch — no drivestudio dep, no OmniRe pixel_source coupling:

    {track_id: {
        "pts":        None,                 # filled by T4.2.b dynamic_rigid_init
        "colors":     None,                 # T4.2.b
        "poses":      Tensor[F, 4, 4],      # object → world SE(3) per frame
        "size":       Tensor[3],            # cuboid full extent (not half)
        "frame_info": BoolTensor[F],        # active flag per frame
        "class":      str,                  # "vehicle", "pedestrian", etc.
    }}

NCore manifest shape (T3a.2 verified empty for current clip — tracks field
needs separate generation). When tracks field is missing → returns empty dict
(not a crash; trainer.init_model logs and skips dynamic_rigids layer).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

import torch


def load_tracks_from_manifest(manifest_path: Union[str, Path]) -> Dict[str, dict]:
    """Parse scene_manifest.tracks → instance_pts_dict.

    Args:
        manifest_path: path to ``pai_<clip>.json`` (or any JSON with a
            top-level ``"tracks"`` array of track dicts).

    Returns:
        Dict keyed by track id; empty if the manifest has no ``tracks`` key
        or the array is empty.

    Raises:
        FileNotFoundError: manifest_path does not exist.
        json.JSONDecodeError: manifest is not valid JSON.
        ValueError: a track dict is missing required fields
            (``id`` / ``poses`` / ``extent`` / ``active_frames``).
    """
    path = Path(manifest_path)
    m = json.loads(path.read_text())

    raw_tracks = m.get("tracks", [])
    out: Dict[str, dict] = {}
    for trk in raw_tracks:
        tid = trk.get("id")
        if tid is None:
            raise ValueError(f"track missing 'id' field: keys={list(trk.keys())}")
        for required in ("poses", "extent", "active_frames"):
            if required not in trk:
                raise ValueError(
                    f"track '{tid}' missing required field '{required}'; "
                    f"keys={list(trk.keys())}"
                )
        poses = torch.tensor(trk["poses"], dtype=torch.float32)
        if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
            raise ValueError(
                f"track '{tid}' poses shape invalid: {tuple(poses.shape)}, "
                f"expected [F, 4, 4]"
            )
        size = torch.tensor(trk["extent"], dtype=torch.float32)
        if size.shape != (3,):
            raise ValueError(
                f"track '{tid}' extent shape invalid: {tuple(size.shape)}, "
                f"expected [3]"
            )
        frame_info = torch.tensor(trk["active_frames"], dtype=torch.bool)
        if frame_info.shape[0] != poses.shape[0]:
            raise ValueError(
                f"track '{tid}' active_frames len {frame_info.shape[0]} "
                f"!= poses F {poses.shape[0]}"
            )
        out[str(tid)] = {
            "pts": None,
            "colors": None,
            "poses": poses,
            "size": size,
            "frame_info": frame_info,
            "class": str(trk.get("class", "vehicle")),
        }
    return out
