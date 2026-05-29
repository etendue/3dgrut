# SPDX-License-Identifier: Apache-2.0
"""Direct readers for NRE nre-tools aux stores (T3.1.b / T3.2.b).

Why bypass NCore SDK SequenceLoaderV4: nre-tools (`ncore-aux-data`) produces
aux.<type>.zarr.itar archives whose root `.zattrs` lacks the ``version`` key
that ``ncore.data.v4.SequenceComponentGroupsReader.__init__`` requires
(verified A800 2026-05-19: KeyError: 'version'). Rather than monkey-patching
the SDK or asking NRE to re-emit aux with version metadata, we read the
zarr.itar stores directly with ``IndexedTarStore + zarr.open``.

Internal layout (NRE 2026-05-19 release):

    aux.sseg.zarr.itar
      /aux/semantic_segmentation/<camera_id>/<timestamp_us>
        - shape: (), dtype: |S<n>  (PNG bytes, 0-D)
        - attrs: {format: "png"}
      /aux/semantic_segmentation/<camera_id>/.zattrs
        - attrs.stuff_classes: [road, sidewalk, ..., sky(10), person(11),
                                ..., bicycle(18), egocar(19)]
        - attrs.resolution: [W, H]

    aux.lidar-sseg.zarr.itar
      /aux/lidar_semantic_segmentation/<lidar_id>/<timestamp_us>
        - shape: (), dtype: |S<n>  (PNG bytes encoding 1-D label array)
        - attrs: {format: "png"}
      /aux/lidar_semantic_segmentation/<lidar_id>/.zattrs
        - attrs.stuff_classes: [..., egocar(19), ignore(20)]
        - attrs.ignore_label: 255

Timestamp keys match the START-of-frame timestamp from
``camera_sensor.frames_timestamps_us[frame_index, FrameTimepoint.START]``
(verified by manually matching 599 sseg keys against camera_front_wide_120fov
frame START timestamps).
"""
from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image


def _open_itar_zarr(itar_path: Union[str, Path]):
    """Open an aux.<type>.zarr.itar archive as a zarr group (read-only).

    Imports ncore.impl.data.stores lazily so this module can be imported in
    environments without ncore SDK (unit tests on Mac do this via stub).
    """
    import zarr
    from ncore.impl.data import stores

    itar = stores.IndexedTarStore(str(itar_path), mode="r")
    return zarr.open(store=itar, mode="r")


class SsegAuxReader:
    """Per-camera per-frame semantic segmentation label reader.

    Lazily opens the zarr group on first ``read()`` and caches per-camera
    subgroup handles. Each ``read(camera_id, timestamp_us)`` returns the
    decoded ``[H, W] uint8`` class-id image.
    """

    def __init__(self, itar_path: Union[str, Path]) -> None:
        self.itar_path = Path(itar_path)
        self._root = None  # lazy zarr group root
        self._cam_groups: dict = {}
        self._class_palette: Optional[list] = None

    def _ensure_open(self) -> None:
        if self._root is None:
            self._root = _open_itar_zarr(self.itar_path)

    @property
    def class_palette(self) -> list:
        """Stuff classes list from sseg group attrs (first camera's group)."""
        self._ensure_open()
        if self._class_palette is None:
            sseg_grp = self._root["aux/semantic_segmentation"]
            for cam_id in sseg_grp.group_keys():
                attrs = dict(sseg_grp[cam_id].attrs)
                self._class_palette = list(attrs.get("stuff_classes", []))
                break
        return self._class_palette or []

    def _cam_group(self, camera_id: str):
        if camera_id not in self._cam_groups:
            self._ensure_open()
            self._cam_groups[camera_id] = self._root["aux/semantic_segmentation"][camera_id]
        return self._cam_groups[camera_id]

    def read(self, camera_id: str, timestamp_us: int) -> np.ndarray:
        """Read one sseg frame.

        Args:
            camera_id: e.g. ``"camera_front_wide_120fov"``.
            timestamp_us: frame START timestamp in microseconds (matches
                ``camera_sensor.frames_timestamps_us[idx, FrameTimepoint.START]``).

        Returns:
            ``[H, W] uint8`` numpy array of class ids (Cityscapes-like palette
            with 20 = ignore; see ``class_palette``).

        Raises:
            KeyError: timestamp not found in this camera's sseg group.
        """
        grp = self._cam_group(camera_id)
        key = str(int(timestamp_us))
        if key not in grp:
            raise KeyError(
                f"SsegAuxReader: timestamp_us={timestamp_us} not in sseg "
                f"group for camera '{camera_id}' (first 5 keys: "
                f"{list(grp.array_keys())[:5]})"
            )
        png_bytes = bytes(grp[key][()])
        return np.asarray(Image.open(io.BytesIO(png_bytes)))


class LidarSsegAuxReader:
    """Per-frame per-point lidar semantic segmentation label reader.

    Same lazy-open pattern as SsegAuxReader. Each ``read(lidar_id, timestamp_us)``
    returns the decoded ``[N_points] uint8`` per-point class-id array
    (PNG-encoded as a 1D image internally to save space).
    """

    IGNORE_LABEL = 255  # nre-tools fills unlabeled points with this

    def __init__(self, itar_path: Union[str, Path]) -> None:
        self.itar_path = Path(itar_path)
        self._root = None
        self._lidar_groups: dict = {}

    def _ensure_open(self) -> None:
        if self._root is None:
            self._root = _open_itar_zarr(self.itar_path)

    def _lidar_group(self, lidar_id: str):
        if lidar_id not in self._lidar_groups:
            self._ensure_open()
            self._lidar_groups[lidar_id] = (
                self._root["aux/lidar_semantic_segmentation"][lidar_id]
            )
        return self._lidar_groups[lidar_id]

    def has_frame(self, lidar_id: str, timestamp_us: int) -> bool:
        """Cheap presence check; aux generation may have skipped some frames."""
        grp = self._lidar_group(lidar_id)
        return str(int(timestamp_us)) in grp

    def read(self, lidar_id: str, timestamp_us: int) -> np.ndarray:
        """Read one frame's per-point class labels as a flat ``[N_points] uint8``."""
        grp = self._lidar_group(lidar_id)
        key = str(int(timestamp_us))
        if key not in grp:
            raise KeyError(
                f"LidarSsegAuxReader: timestamp_us={timestamp_us} not in "
                f"lidar-sseg for '{lidar_id}' (first 5: "
                f"{list(grp.array_keys())[:5]})"
            )
        png_bytes = bytes(grp[key][()])
        img = np.asarray(Image.open(io.BytesIO(png_bytes)))  # PNG decoded shape may be (1, N) or (N,)
        return img.ravel()


def discover_aux_path(clip_dir: Union[str, Path], aux_type: str) -> Optional[Path]:
    """Find the aux.<aux_type>.zarr.itar file in a clip directory.

    Args:
        clip_dir: clip directory (containing ``pai_<clip>.json``).
        aux_type: one of ``"sseg" / "lidar-sseg" / "lidar-camvis" / "egomask" / "depth"``.

    Returns:
        Path object if exactly one match; None if no match. Raises ValueError
        if multiple matches (ambiguous).
    """
    matches = sorted(Path(clip_dir).glob(f"*.aux.{aux_type}.zarr.itar"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            f"discover_aux_path: multiple aux.{aux_type}.zarr.itar files in "
            f"{clip_dir}: {matches}"
        )
    return matches[0]


# ---------------------------------------------------------------------------
# Stage 11 T11.B2 — per-frame npz depth-map readers (NOT zarr.itar pattern)
# ---------------------------------------------------------------------------


class LidarDepthAuxReader:
    """Reads pre-dumped LiDAR → image-plane depth maps (Stage 11 T11.B2).

    Layout: ``<root>/<camera_id>/<timestamp_us>.npz`` with a single key
    ``depth`` of shape ``[H, W]`` float32 (ray-depth, 0 = no LiDAR hit).

    This is a deliberately simple per-frame npz reader — NOT the zarr.itar
    pattern of SsegAuxReader / LidarSsegAuxReader above. It loads depth maps
    that scripts/dump_lidar_depth_map.py wrote, decoupled from NCore's packed
    aux archives.

    Args:
        root: directory containing ``<camera_id>/<timestamp_us>.npz`` files.
        default_shape: when set, a missing frame returns an all-zeros
            ``[H, W]`` map (no LiDAR coverage is a legal situation → hit_mask
            naturally zeros out the loss). When None, a missing frame raises
            FileNotFoundError (caller must explicitly opt into the zeros
            fallback).
    """

    def __init__(
        self,
        root: Union[str, Path],
        default_shape: Optional[Tuple[int, int]] = None,
        cache_maxsize: int = 256,
    ) -> None:
        self.root = Path(root)
        self.default_shape = default_shape
        # Bounded FIFO cache: unlike the unbounded growth that would reach
        # ~14GB on a full-res 30k run, keep at most cache_maxsize decoded
        # maps. The existing zarr readers in this file don't cache decoded
        # arrays at all; this is a middle ground (recent-frame reuse without
        # unbounded RAM). cache_maxsize <= 0 disables caching entirely.
        self.cache_maxsize = cache_maxsize
        self._cache: dict[Tuple[str, int], np.ndarray] = {}
        self._cache_order: list[Tuple[str, int]] = []

    def _path(self, camera_id: str, timestamp_us: int) -> Path:
        return self.root / camera_id / f"{int(timestamp_us)}.npz"

    def has_frame(self, camera_id: str, timestamp_us: int) -> bool:
        return self._path(camera_id, timestamp_us).exists()

    def read(self, camera_id: str, timestamp_us: int) -> np.ndarray:
        """Return the ``[H, W]`` float32 depth map for one (camera, frame).

        Cached (bounded FIFO, cache_maxsize) after first read. Missing frame →
        zeros (if default_shape set) or FileNotFoundError.
        """
        key = (camera_id, int(timestamp_us))
        if key in self._cache:
            return self._cache[key]
        path = self._path(camera_id, timestamp_us)
        if not path.exists():
            if self.default_shape is None:
                raise FileNotFoundError(f"{type(self).__name__}: missing {path}")
            depth = np.zeros(self.default_shape, dtype=np.float32)
        else:
            with np.load(path) as f:
                if "depth" not in f:
                    raise KeyError(
                        f"{type(self).__name__}: npz at {path} has no 'depth' "
                        f"key (found: {list(f.keys())})"
                    )
                depth = f["depth"].astype(np.float32)
        self._cache_put(key, depth)
        return depth

    def _cache_put(self, key: Tuple[str, int], depth: np.ndarray) -> None:
        if self.cache_maxsize <= 0:
            return
        self._cache[key] = depth
        self._cache_order.append(key)
        while len(self._cache_order) > self.cache_maxsize:
            evicted = self._cache_order.pop(0)
            self._cache.pop(evicted, None)


class DepthV2AuxReader(LidarDepthAuxReader):
    """Reads pre-dumped DepthAnythingV2 metric depth maps (Stage 11 T11.D2).

    Identical npz layout to LidarDepthAuxReader, just rooted at the DepthV2
    dump directory (aux/depth_anything_v2/<camera_id>/<timestamp_us>.npz).
    Kept as a distinct class so the dataset can hold two readers with clear
    names and so future DepthV2-specific handling has a home.
    """
    pass
