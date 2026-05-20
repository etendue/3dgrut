"""Inject ``viz_4d`` metadata into an existing v2 LayeredGaussians ckpt (T8.9).

Use case: you trained a v2 ckpt *before* T8.2 (so ``viz_4d`` isn't there) but
still have the NCore dataset. Run this once to write the block back into the
ckpt — then ``viser_gui_4d.py`` can replay it without needing ``--dataset_path``
on every launch.

The script mirrors the minimal slice of ``Trainer.setup_training`` that
populates ``LayeredGaussians.tracks_metadata`` from NCore cuboid autolabels.
LiDAR points are still pulled from the dataset for the viewer overlay.

Usage::

    python -m threedgrut.viz.inject \\
        --ckpt /path/to/old_v2_ckpt.pt \\
        --dataset_path /path/to/pai_xxx.json \\
        --out /path/to/new_ckpt_with_viz_4d.pt          # or omit for in-place

The output ckpt is byte-identical with the input except for the new top-level
``viz_4d`` key. All other ckpt blocks (model / strategy / post_processing /
exposure_state / sky_envmap_state) pass through untouched.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from threedgrut.utils.logger import logger


def _populate_tracks_from_dataset(model, dataset) -> int:
    """Replicate ``Trainer.setup_training`` tracks loading (lines 380-447).

    Returns number of tracks populated. Caller logs / raises as needed.
    """
    if "dynamic_rigids" not in getattr(model, "layers", {}):
        return 0
    try:
        import ncore.data as _nd
    except ImportError as e:
        raise RuntimeError(f"NCore SDK required for tracks loading: {e}")
    from threedgrut.datasets.tracks_loader import load_tracks_from_ncore_cuboids

    loader = dataset.sequence_loaders[dataset.sequence_id]
    ref_cam = dataset.camera_ids[0]
    ref_sensor = dataset.sequence_camera_sensors[dataset.sequence_id][ref_cam]
    cam_ts = ref_sensor.frames_timestamps_us[:, _nd.FrameTimepoint.END]
    time_range = dataset.time_range_us
    in_window = np.array([int(t) in time_range for t in cam_ts])
    cam_ts_active = np.asarray(cam_ts)[in_window]
    tracks = load_tracks_from_ncore_cuboids(loader, cam_ts_active)
    if tracks:
        model.populate_tracks(tracks)
    return len(tracks)


def inject_viz_4d(ckpt_path: str, dataset_path: str | None,
                  out_path: str | None) -> dict:
    """Inject a ``viz_4d`` block into an existing v2 LayeredGaussians ckpt.

    Args:
        ckpt_path:    Path to the source ckpt (``.pt``).
        dataset_path: NCore manifest ``.json`` to source ego / tracks /
                      LiDAR from. Required because the ckpt itself doesn't
                      persist these.
        out_path:     Destination ckpt. ``None`` ⇒ overwrite ``ckpt_path``
                      in place (a ``.bak`` is left next to it).

    Returns:
        The injected ``viz_4d`` block (for caller introspection).
    """
    if dataset_path is None:
        raise ValueError(
            "inject_viz_4d requires --dataset_path: the ckpt itself does not "
            "persist ego trajectories / track class+size / LiDAR clouds; we "
            "need to re-pull them from the NCore manifest."
        )

    src = Path(ckpt_path)
    if not src.is_file():
        raise FileNotFoundError(f"ckpt not found: {src}")

    logger.info(f"[inject] loading ckpt: {src}")
    ckpt = torch.load(src, weights_only=False)
    conf = ckpt.get("config")
    if conf is None:
        raise ValueError("ckpt missing 'config' key — not a 3dgrut ckpt?")
    if not bool(conf.get("use_layered_model", False)):
        raise ValueError(
            "ckpt is not a v2 LayeredGaussians ckpt (use_layered_model=false). "
            "viz_4d only applies to layered models."
        )

    # Override dataset path so NCoreDataset resolves to the user-given manifest
    # (the original training path may not exist on this machine).
    conf = OmegaConf.merge(conf, OmegaConf.create({"path": dataset_path}))

    # Lazy import so machines without NCore SDK / kaolin don't crash on `import`.
    from threedgrut.datasets.datasetNcore import NCoreDataset
    from threedgrut.layers.layered_model import LayeredGaussians
    from threedgrut.layers.registry import specs_from_config
    from threedgrut.viz.metadata import extract_4d_metadata

    # Build LayeredGaussians on CPU (no CUDA needed for metadata extraction).
    specs = specs_from_config(conf)
    scene_extent = float(ckpt.get("model", {}).get("scene_extent", 1.0))
    model = LayeredGaussians(conf, specs=specs, scene_extent=scene_extent)
    model.init_from_checkpoint(ckpt, setup_optimizer=False)
    logger.info(
        f"[inject] LayeredGaussians built: layers="
        f"{[s.name for s in specs]}"
    )

    # Build dataset (train split — only need its metadata accessors).
    logger.info(f"[inject] loading NCore dataset: {dataset_path}")
    train_ds = NCoreDataset(conf, split="train")
    train_ds._init_worker()

    # Repopulate dynamic-rigid tracks from the NCore cuboid autolabels so
    # tracks_metadata (class, size) is filled — populate_tracks attaches it
    # to model.tracks_metadata, which extract_4d_metadata reads.
    n_tracks = _populate_tracks_from_dataset(model, train_ds)
    logger.info(f"[inject] populated {n_tracks} dynamic_rigid tracks")

    # Build the viz_4d block.
    md = extract_4d_metadata(model, train_ds, conf)
    ckpt["viz_4d"] = md

    # Write out.
    if out_path is None:
        backup = src.with_suffix(src.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(src, backup)
            logger.info(f"[inject] backup written: {backup}")
        else:
            logger.info(f"[inject] backup already exists, not overwriting: {backup}")
        dst = src
    else:
        dst = Path(out_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"[inject] writing ckpt with viz_4d → {dst}")
    torch.save(ckpt, dst)
    logger.info(
        f"[inject] ✅ done. viz_4d schema_v{md['schema_version']} "
        f"({len(md['tracks'])} tracks, ego_N={md['ego']['poses_c2w'].shape[0]}, "
        f"road_pts={md['lidar'].get('road_subsample')})"
    )
    return md


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inject viz_4d metadata into an existing v2 ckpt.",
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Source ckpt (.pt) — must be a v2 LayeredGaussians ckpt.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="NCore manifest .json (re-pulled for ego/tracks/LiDAR).")
    parser.add_argument("--out", type=str, default=None,
                        help="Output ckpt path. Omit to overwrite --ckpt in place "
                             "(a .bak file is left next to it).")
    args = parser.parse_args()
    try:
        inject_viz_4d(args.ckpt, args.dataset_path, args.out)
    except Exception as e:  # noqa: BLE001
        logger.error(f"[inject] FAILED: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
