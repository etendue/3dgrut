"""Dynamic-track slot reconstruction required before layered checkpoint load."""

from __future__ import annotations

from typing import Any

from threedgrut.utils.logger import logger


def populate_dynamic_tracks_for_checkpoint_resume(
    model: Any,
    train_dataset: Any,
    conf: Any,
) -> None:
    """Recreate track slots without touching checkpoint Gaussian parameters.

    Fresh LiDAR initialization calls ``populate_tracks`` while creating the
    dynamic layer. A resume skips initialization, yet PyTorch can only load
    saved ``_track_pose_*``/``_track_active_*`` tensors into existing slots.
    Rebuild the same NCore timeline immediately before checkpoint loading.
    """
    from threedgrut.layers.layered_model import LayeredGaussians

    if not isinstance(model, LayeredGaussians) or "dynamic_rigids" not in model.layers:
        return
    if not hasattr(train_dataset, "sequence_loaders") or not hasattr(
        train_dataset, "sequence_id"
    ):
        logger.warning(
            "[ckpt] dynamic_rigids resume cannot repopulate tracks for "
            f"dataset type {type(train_dataset).__name__}"
        )
        return

    from threedgrut.datasets.tracks_loader import (
        CUBOID_TS_MODES,
        build_cuboid_frame_timeline_us,
        load_tracks_from_ncore_cuboids,
    )

    loader = train_dataset.sequence_loaders[train_dataset.sequence_id]
    cuboid_ts_mode = str(getattr(conf.dataset, "cuboid_ts_mode", "ref_nearest"))
    cam_ts_active = build_cuboid_frame_timeline_us(train_dataset, cuboid_ts_mode)
    tracks = load_tracks_from_ncore_cuboids(
        loader,
        cam_ts_active,
        pose_time_mode=CUBOID_TS_MODES[cuboid_ts_mode],
    )
    logger.info(
        f"[ckpt] repopulating {len(tracks)} dynamic_rigid tracks before resume "
        f"(frames={cam_ts_active.shape[0]}, cuboid_ts_mode={cuboid_ts_mode})"
    )
    if tracks:
        model.populate_tracks(tracks)
