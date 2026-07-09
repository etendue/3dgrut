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


def _decode_egomask_frame(arr) -> np.ndarray:
    """Decode one egomask frame to a ``(H, W)`` bool array (True = ego pixel).

    nre-tools stores egomask frames in either of two forms; both are handled:
      * a plain ``(H, W)`` uint8 array with values in ``{0, 255}``, or
      * 0-D ``|S<n>`` PNG bytes (same 0-D pattern as SsegAuxReader).
    A pixel is "ego" iff it is non-zero.
    """
    if getattr(arr, "shape", None) == ():
        img = np.asarray(Image.open(io.BytesIO(bytes(arr[()]))))
    else:
        img = np.asarray(arr[...])
    mask = img != 0
    if mask.ndim == 3:  # e.g. an RGB(A) PNG — any channel set means ego
        mask = mask.any(axis=-1)
    return mask


class EgomaskAuxReader:
    """Per-camera static ego-mask reader over an ``aux.egomask.zarr.itar`` store.

    Same lazy-open pattern as :class:`SsegAuxReader`. The internal group that
    holds the per-camera frame arrays is discovered generically (any group with
    array children is treated as a camera group, keyed by its own name) rather
    than hard-coding ``aux/ego_mask`` — the nre-tools internal group name is not
    version-pinned here, and the diagnostic that inspected b6a9 walked the tree
    generically (grouping frames by their parent group name).

    ``read_static_mask(camera_id)`` returns the **union** over all of that
    camera's frames: any frame marking a pixel ego makes it ego in the static
    mask (conservative — the ego structure is static w.r.t. the camera, so a
    hit in any frame is real).
    """

    def __init__(self, itar_path: Union[str, Path]) -> None:
        self.itar_path = Path(itar_path)
        self._root = None  # lazy zarr group root
        self._cam_groups: Optional[dict] = None  # camera_id -> frame group

    def _ensure_open(self) -> None:
        if self._root is None:
            self._root = _open_itar_zarr(self.itar_path)
        if self._cam_groups is None:
            self._cam_groups = self._discover_camera_groups(self._root)

    @staticmethod
    def _discover_camera_groups(root) -> dict:
        """Walk the tree: a group holding frame arrays is a per-camera group."""
        cams: dict = {}

        def _walk(grp, name: str) -> None:
            if list(grp.array_keys()):
                cams[name] = grp
                return
            for k in grp.group_keys():
                _walk(grp[k], k)

        _walk(root, "")
        cams.pop("", None)  # never register the root itself as a camera
        return cams

    def camera_ids(self) -> list:
        self._ensure_open()
        return sorted(self._cam_groups.keys())

    def has_camera(self, camera_id: str) -> bool:
        self._ensure_open()
        return camera_id in self._cam_groups

    def read_static_mask(self, camera_id: str) -> np.ndarray:
        """Return the ``(H, W)`` bool union of this camera's ego-mask frames.

        Raises:
            KeyError: camera_id not present in this egomask itar (or the camera
                group is empty).
        """
        self._ensure_open()
        if camera_id not in self._cam_groups:
            raise KeyError(
                f"EgomaskAuxReader: camera '{camera_id}' not in egomask itar "
                f"(available: {sorted(self._cam_groups)})"
            )
        grp = self._cam_groups[camera_id]
        union: Optional[np.ndarray] = None
        for k in grp.array_keys():
            frame = _decode_egomask_frame(grp[k])
            union = frame if union is None else np.logical_or(union, frame)
        if union is None:
            raise KeyError(f"EgomaskAuxReader: camera '{camera_id}' has no frames")
        return union


def _dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Binary-dilate ``mask`` ``iterations`` times, treating ``0`` as identity.

    ``scipy.ndimage.binary_dilation`` interprets ``iterations < 1`` as "dilate
    until the result stops changing" (which fills a connected region entirely).
    We instead want ``iterations == 0`` to mean "no dilation". datasetNcore's
    default is 30 (>= 1), so for every real value this is byte-for-byte identical
    to ``ndimage.binary_dilation(mask, iterations=iterations)``.
    """
    if iterations and iterations >= 1:
        from scipy import ndimage  # lazy: keep module import light for non-mask consumers

        return ndimage.binary_dilation(mask, iterations=iterations)
    return mask


def resolve_ego_valid_mask_with_source(
    sdk_mask_image,
    clip_dir: Optional[Union[str, Path]],
    camera_id: str,
    resolution_hw: Tuple[int, int],
    dilation_iters: int,
) -> Tuple[np.ndarray, str]:
    """Resolve the ``(H, W)`` bool valid-pixel ego mask and the branch used.

    Three branches, in priority order (source labels in parentheses):
      1. (``"sdk"``) ``sdk_mask_image`` present AND non-zero → the NCore-SDK
         sequence embeds a real ego mask: ``convert("L") != 0`` → dilate →
         ``logical_not`` (byte-identical to datasetNcore's pre-P0.2 path).
      2. (``"itar"``) else if ``clip_dir`` is not None and an
         ``aux.egomask.zarr.itar`` exists there whose :class:`EgomaskAuxReader`
         ``has_camera(camera_id)`` → itar union → dilate → ``logical_not``.
      3. (``"none"``) else → all-True with shape ``resolution_hw`` (nothing
         masks anything). Byte-equivalent to ``np.ones((h, w), dtype=bool)``.

    The source label lets callers (datasetNcore) emit a ``[P0.2] ego mask via
    aux itar fallback`` log line only when branch 2 fires.
    """
    if sdk_mask_image is not None:
        sdk_mask = np.asarray(sdk_mask_image.convert("L")) != 0
        if sdk_mask.any():
            return np.logical_not(_dilate_mask(sdk_mask, dilation_iters)), "sdk"

    if clip_dir is not None:
        itar_path = discover_aux_path(clip_dir, "egomask")
        if itar_path is not None:
            reader = EgomaskAuxReader(itar_path)
            if reader.has_camera(camera_id):
                mask = reader.read_static_mask(camera_id)
                return np.logical_not(_dilate_mask(mask, dilation_iters)), "itar"

    return np.ones(tuple(int(x) for x in resolution_hw), dtype=bool), "none"


def resolve_ego_valid_mask(
    sdk_mask_image,
    clip_dir: Optional[Union[str, Path]],
    camera_id: str,
    resolution_hw: Tuple[int, int],
    dilation_iters: int,
) -> np.ndarray:
    """Thin wrapper around :func:`resolve_ego_valid_mask_with_source` that
    returns just the ``(H, W)`` bool valid-pixel mask (Task 1 API — kept for
    callers that don't need the source label)."""
    mask, _ = resolve_ego_valid_mask_with_source(
        sdk_mask_image, clip_dir, camera_id, resolution_hw, dilation_iters
    )
    return mask


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
            self._lidar_groups[lidar_id] = self._root["aux/lidar_semantic_segmentation"][lidar_id]
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
        aux_type: one of ``"sseg" / "lane" / "lidar-sseg" / "lidar-camvis" / "egomask" / "depth"``.

    Returns:
        Path object if exactly one match; None if no match. Raises ValueError
        if multiple matches (ambiguous).
    """
    matches = sorted(Path(clip_dir).glob(f"*.aux.{aux_type}.zarr.itar"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"discover_aux_path: multiple aux.{aux_type}.zarr.itar files in " f"{clip_dir}: {matches}")
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
                        f"{type(self).__name__}: npz at {path} has no 'depth' " f"key (found: {list(f.keys())})"
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
