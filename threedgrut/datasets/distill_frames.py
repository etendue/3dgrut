# SPDX-License-Identifier: Apache-2.0
"""E2.2 — pseudo-GT progressive-distillation frame source.

The distillation loop (render → Harmonizer-fix → distill back) feeds
Harmonizer-fixed novel-view frames into training as INDEPENDENT pseudo-GT
samples. This module turns an E2.1 frame pack (``<frames_dir>/<mode>/
frames_map.json`` + per-camera PNGs) plus the source dataset's per-frame
camera batches into a stream of renderable pseudo-GT ``Batch`` objects.

Design (spec §2.2):
  * Each pack frame is keyed ``ts:<camera_id>:<timestamp_us>`` (E2.1 alignment,
    see ``novel_view.novel_frame_key``). We recover the SOURCE frame's shutter
    poses from a provided iterable of source batches, apply the SAME
    ``perturb_batch_shutter_pair_torch`` render.py used, and swap in the fixed
    PNG as ``rgb_gt``. Rays stay camera-space (pose-independent) so the novel
    pose is reproduced exactly (the distillation "poison" guard, test ③).
  * The fixed frame is an independent training sample — its pixels are NEVER
    blended with a real image (different poses can't be mixed). ``mask=None``
    (novel frames carry no sseg/road masks) and an ``is_distill`` marker routes
    the loss to the full-image photometric path.

The source-batch iterable is the seam that keeps this CPU-testable: the trainer
passes its val-split batches (the same split render.py dumped), while tests pass
``SimpleNamespace`` stand-ins. We only ever read pose/ray/intrinsic fields off
them — never call the renderer.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Optional

import numpy as np
import torch

from threedgrut.datasets.protocols import Batch
from threedgrut.utils.novel_view import (
    NOVEL_VIEW_MODES,
    novel_frame_key,
    perturb_batch_shutter_pair_torch,
)

# Fields copied verbatim from the source frame's batch onto the pseudo-GT batch
# (camera-space, pose-independent — reusable at the perturbed pose).
_CARRY_FIELDS = (
    "rays_ori",
    "rays_dir",
    "rays_in_world_space",
    "pixel_coords",
    "camera_idx",
    "intrinsics",
    "intrinsics_OpenCVPinholeCameraModelParameters",
    "intrinsics_OpenCVFisheyeCameraModelParameters",
    "intrinsics_FThetaCameraModelParameters",
)


def distill_photometric_loss(rgb_pred: torch.Tensor, rgb_gt: torch.Tensor, lam: float) -> torch.Tensor:
    """Full-image L1 + (1 - SSIM), scaled by ``lam``.

    The single numeric contract for a distill batch's photometric term (spec
    §2.2): no layered_l1, no region masks — novel frames have no sseg/road
    masks. Both terms carry the SAME λ so the whole photometric contribution is
    linear in λ (test ② asserts λ=2 ⇒ 2× λ=1).

    ``rgb_pred`` / ``rgb_gt`` are ``[B, H, W, 3]`` in ``[0, 1]``. Uses the SAME
    ``model.losses.ssim`` the trainer uses (imported lazily so this module stays
    importable on CPU-only hosts where ``fused_ssim`` is absent).
    """
    from threedgrut.model.losses import ssim

    l1 = torch.abs(rgb_pred - rgb_gt).mean()
    ssim_term = 1.0 - ssim(rgb_pred.permute(0, 3, 1, 2), rgb_gt.permute(0, 3, 1, 2))
    return lam * (l1 + ssim_term)


def apply_distill_warmstart(conf) -> str:
    """E2.2 warm-start: route ``init_checkpoint`` → the existing
    ``initialization.method='checkpoint'`` branch (loads model params only →
    fresh optimizer → global_step=0), setting the anchor path.

    Uses ``open_dict`` because the ``initialization`` config group (e.g.
    ``configs/initialization/checkpoint.yaml``) is struct-locked and has NO
    ``path`` key by default — a plain ``conf.initialization.path = ...`` raises
    ``ConfigAttributeError: Key 'path' is not in struct`` (the Task 4 GPU
    crash this regressed). No-op when ``init_checkpoint`` is unset or ``resume``
    is set (explicit resume wins). Returns the resolved checkpoint path (``""``
    when inactive).
    """
    from omegaconf import open_dict

    init_ckpt = conf.get("init_checkpoint", "") if hasattr(conf, "get") else getattr(conf, "init_checkpoint", "")
    if init_ckpt and not getattr(conf, "resume", ""):
        with open_dict(conf.initialization):
            conf.initialization.method = "checkpoint"
            conf.initialization.path = init_ckpt
        return init_ckpt
    return ""


class DistillFrameSource:
    """Samples Harmonizer-fixed novel frames as pseudo-GT ``Batch`` objects."""

    def __init__(
        self,
        frames_dir: str,
        mode: str,
        source_batches: Iterable,
    ) -> None:
        """
        Args:
            frames_dir: pack root; the mode subdir ``<frames_dir>/<mode>/`` holds
                ``frames_map.json`` + PNGs.
            mode: novel-view mode (e.g. ``lateral_1m``) — must be in
                ``NOVEL_VIEW_MODES`` and must match render.py's rendered mode.
            source_batches: iterable of source-frame batches (objects exposing
                ``camera_id``, ``timestamp_us``, ``T_to_world``,
                ``T_to_world_end`` and the ray/intrinsic fields). Consumed once
                to build a ``ts:<cam>:<ts>`` → source-batch index.
        """
        if mode not in NOVEL_VIEW_MODES:
            raise ValueError(f"distill mode '{mode}' not in NOVEL_VIEW_MODES {NOVEL_VIEW_MODES}")
        self.frames_dir = frames_dir
        self.mode = mode
        self._mode_dir = os.path.join(frames_dir, mode)

        map_path = os.path.join(self._mode_dir, "frames_map.json")
        if not os.path.isfile(map_path):
            raise FileNotFoundError(
                f"distill frame pack missing frames_map.json: {map_path} " f"(mode '{mode}' not rendered here?)"
            )
        with open(map_path) as f:
            self.frames_map: dict[str, str] = json.load(f)
        if not self.frames_map:
            raise ValueError(f"distill frame pack is empty: {map_path}")

        # Index source frames by their alignment key.
        self._source_by_key: dict[str, Batch] = {}
        for b in source_batches:
            cam = getattr(b, "camera_id", None)
            ts = getattr(b, "timestamp_us", None)
            if cam is None or ts is None or int(ts) < 0:
                continue
            self._source_by_key[novel_frame_key(cam, int(ts))] = b

        # Every pack frame MUST have a source pose — a missing one means the
        # pack and the dataset split disagree (frame-pose misalignment is
        # distillation poison). Fail loudly at construction, not mid-training.
        missing = [k for k in self.frames_map if k not in self._source_by_key]
        if missing:
            raise KeyError(
                f"{len(missing)} distill pack frame(s) have no matching source "
                f"pose (e.g. {missing[0]!r}); pack/dataset split mismatch"
            )

        self._keys = list(self.frames_map.keys())
        self._fire_count = 0

    def __len__(self) -> int:
        return len(self._keys)

    def sample(self, rng: Optional[np.random.Generator] = None) -> Batch:
        """Draw a random fixed frame → pseudo-GT ``Batch`` at the novel pose."""
        if rng is None:
            rng = np.random.default_rng()
        key = self._keys[int(rng.integers(0, len(self._keys)))]
        src = self._source_by_key[key]

        rgb_gt = self._load_fixed_frame(self.frames_map[key], src)

        # SAME transform render.py applied → exact novel-pose reconstruction.
        new_start, new_end = perturb_batch_shutter_pair_torch(src.T_to_world, src.T_to_world_end, self.mode)

        fields = {
            "rays_ori": src.rays_ori,
            "rays_dir": src.rays_dir,
            "T_to_world": new_start,
            "T_to_world_end": new_end,
            "rays_in_world_space": bool(getattr(src, "rays_in_world_space", False)),
            "rgb_gt": rgb_gt,
            "mask": None,  # novel frames have no sseg/road masks
            "camera_id": getattr(src, "camera_id", None),
            "camera_idx": int(getattr(src, "camera_idx", -1)),
            "timestamp_us": int(getattr(src, "timestamp_us", -1)),
        }
        for f in _CARRY_FIELDS:
            if f in fields:
                continue
            val = getattr(src, f, None)
            if val is not None:
                fields[f] = val

        batch = Batch(**fields)
        # Route the loss to the full-image photometric path (spec §2.2).
        batch.is_distill = True
        self._fire_count += 1
        return batch

    @property
    def fire_count(self) -> int:
        """# pseudo-GT batches sampled so far (Task 4 smoke sanity log)."""
        return self._fire_count

    def _load_fixed_frame(self, relpath: str, src) -> torch.Tensor:
        """Read the fixed PNG → ``[1, H, W, 3]`` float on the source's device."""
        import torchvision

        path = os.path.join(self._mode_dir, relpath)
        if not os.path.exists(path):
            raise FileNotFoundError(f"distill fixed frame missing: {path} — pack integrity broken")
        img = torchvision.io.read_image(path).float().div(255.0)  # [C, H, W]
        img = img[:3].permute(1, 2, 0)  # [H, W, 3]
        device = src.rgb_gt.device if getattr(src, "rgb_gt", None) is not None else src.rays_dir.device
        return img.unsqueeze(0).to(device)
