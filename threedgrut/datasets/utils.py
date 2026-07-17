# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import collections
import math
import platform
import struct
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

DEFAULT_DEVICE = torch.device("cuda")

from threedgrut.utils.logger import logger


def fov2focal(fov_radians: float, pixels: int):
    return pixels / (2 * math.tan(fov_radians / 2))


def focal2fov(focal: float, pixels: int):
    return 2 * math.atan(pixels / (2 * focal))


def create_pixel_coords(width: int, height: int, device: torch.device = None) -> torch.Tensor:
    """Generate pixel coordinates with +0.5 center offset for post-processing.

    Creates a grid of pixel coordinates where each coordinate represents the center
    of the pixel (hence the +0.5 offset).

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        device: Target device for the tensor. Defaults to None (CPU).

    Returns:
        Pixel coordinates tensor of shape [1, H, W, 2] containing (x, y) coordinates.
    """
    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device) + 0.5,
        torch.arange(width, dtype=torch.float32, device=device) + 0.5,
        indexing="ij",
    )
    return torch.stack([pixel_x, pixel_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]


def pinhole_camera_rays(x, y, f_x, f_y, w, h, ray_jitter=None, cx=None, cy=None):
    """
    return:
        ray_origin (sz_y, sz_x, 3)
        normalized ray_direction (sz_y, sz_x, 3)
    """

    if ray_jitter is not None:
        jitter = ray_jitter(x.shape).numpy()
        jitter_xs = jitter[:, 0]
        jitter_ys = jitter[:, 1]
    else:
        jitter_xs = jitter_ys = 0.5

    if cx is None:
        cx = 0.5 * w
    if cy is None:
        cy = 0.5 * h

    xs = ((x + jitter_xs) - cx) / f_x
    ys = ((y + jitter_ys) - cy) / f_y

    ray_lookat = np.stack((xs, ys, np.ones_like(xs)), axis=-1)
    ray_origin = np.zeros_like(ray_lookat)

    return ray_origin, ray_lookat / np.linalg.norm(ray_lookat, axis=-1, keepdims=True)


def camera_to_world_rays(ray_o, ray_d, poses):
    """
    input:
        ray_o_cam [n, 3] - ray origins in the camera coordinate system
        ray_d_cam [n, 3] - ray origins in the camera coordinate system
        poses [n, 4,4] - camera to world transformation matrices

    return:
        ray_o [n, 3] - ray origins in the world coordinate system
        ray_d [n, 3] - ray directions in the world coordinate system
    """
    if isinstance(poses, torch.Tensor):
        ray_o = torch.einsum("ijk,ik->ij", poses[:, :3, :3], ray_o) + poses[:, :3, 3]
        ray_d = torch.einsum("ijk,ik->ij", poses[:, :3, :3], ray_d)
    else:
        ray_o = np.einsum("ijk,ik->ij", poses[:, :3, :3], ray_o) + poses[:, :3, 3]
        ray_d = np.einsum("ijk,ik->ij", poses[:, :3, :3], ray_d)

    return ray_o, ray_d


@dataclass(slots=True, kw_only=True)
class PointCloud:
    """Represents a 3d point cloud consisting of corresponding start and end points"""

    xyz_start: torch.Tensor  # [N,3]
    xyz_end: torch.Tensor  # [N,3]
    device: str
    dtype = torch.float32
    color: torch.Tensor | None = None  # uint8 RGB colors in [0, 255], shape [N,3]

    def __post_init__(self) -> None:
        assert len(self.xyz_start) == len(self.xyz_end)
        assert self.xyz_start.shape[1] == self.xyz_end.shape[1] == 3

        self.xyz_start.to(self.device, dtype=self.dtype)
        self.xyz_end.to(self.device, dtype=self.dtype)

        if self.color is not None:
            assert self.color.shape[1] == 3
            assert len(self.color) == len(self.xyz_end)

            self.color.to(self.device, dtype=self.dtype)

    @staticmethod
    def from_sequence(point_clouds: Sequence[PointCloud], device: str) -> PointCloud:
        point_clouds_list = list(point_clouds)

        return PointCloud(
            xyz_start=torch.cat([pc.xyz_start for pc in point_clouds_list]),
            xyz_end=torch.cat([pc.xyz_end for pc in point_clouds_list]),
            color=(
                torch.cat([pc.color for pc in point_clouds_list]) if point_clouds_list[0].color is not None else None
            ),
            device=device,
        )

    def selected_idxs(self, idxs):
        return PointCloud(
            xyz_start=self.xyz_start[idxs],
            xyz_end=self.xyz_end[idxs],
            color=self.color[idxs] if self.color is not None else None,
            device=self.device,
        )


def get_center_and_diag(cam_centers):
    avg_cam_center = np.median(cam_centers, axis=0, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=1, keepdims=True)
    diagonal = np.median(dist)
    return center.flatten(), diagonal


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def compute_max_distance_to_border(image_size_component: float, principal_point_component: float) -> float:
    """Given an image size component (x or y) and corresponding principal point component (x or y),
    returns the maximum distance (in image domain units) from the principal point to either image boundary.
    """
    center = 0.5 * image_size_component
    if principal_point_component > center:
        return principal_point_component
    else:
        return image_size_component - principal_point_component


def compute_max_radius(image_size: np.ndarray, principal_point: np.ndarray) -> float:
    """Compute the maximum radius from the principal point to the image boundaries."""
    max_diag = np.array(
        [
            compute_max_distance_to_border(image_size[0], principal_point[0]),
            compute_max_distance_to_border(image_size[1], principal_point[1]),
        ]
    )
    return np.linalg.norm(max_diag).item()


def create_camera_visualization(cam_list):
    """
    Given a list-of-dicts of camera & image info, register them in polyscope
    to create a visualization
    """

    import polyscope as ps

    for i_cam, cam in enumerate(cam_list):

        ps_cam_param = ps.CameraParameters(
            ps.CameraIntrinsics(
                fov_vertical_deg=np.degrees(cam["fov_h"]),
                fov_horizontal_deg=np.degrees(cam["fov_w"]),
            ),
            ps.CameraExtrinsics(mat=cam["ext_mat"]),
        )

        cam_color = (1.0, 1.0, 1.0)
        if cam["split"] == "train":
            cam_color = (1.0, 0.7, 0.7)
        elif cam["split"] == "val":
            cam_color = (0.7, 0.1, 0.7)

        ps_cam = ps.register_camera_view(f"{cam['split']}_view_{i_cam:03d}", ps_cam_param, widget_color=cam_color)

        ps_cam.add_color_image_quantity("target image", cam["rgb_img"][:, :, :3], enabled=True)


CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_colmap_points3D_text(path):
    """
    Read points3D.txt file from COLMAP output.
    Returns numpy arrays of xyz coordinates, RGB values, and reprojection errors.
    """
    # Pre-allocate lists for data
    xyzs = []
    rgbs = []
    errors = []

    # Single file read
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                # Convert directly to numpy arrays while appending
                xyzs.append([float(x) for x in elems[1:4]])
                rgbs.append([int(x) for x in elems[4:7]])
                errors.append(float(elems[7]))

    # Convert lists to numpy arrays all at once
    return (
        np.array(xyzs, dtype=np.float64),
        np.array(rgbs, dtype=np.int32),
        np.array(errors, dtype=np.float64).reshape(-1, 1),
    )


def read_colmap_points3D_binary(path_to_model_file):
    """
    Read points3D.bin file from COLMAP output.
    Returns numpy arrays of xyz coordinates, RGB values, and reprojection errors.
    """
    # Pre-allocate lists for data
    xyzs = []
    rgbs = []
    errors = []

    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_points):
            # Read the point data
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            # Append coordinates, colors, and error
            xyzs.append(binary_point_line_properties[1:4])
            rgbs.append(binary_point_line_properties[4:7])
            errors.append(binary_point_line_properties[7])

            # Skip track length and elements as they're not used
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            fid.seek(8 * track_length, 1)

    # Convert lists to numpy arrays all at once
    return (
        np.array(xyzs, dtype=np.float64),
        np.array(rgbs, dtype=np.int32),
        np.array(errors, dtype=np.float64).reshape(-1, 1),
    )


def read_colmap_intrinsics_text(path):
    """
    Read camera intrinsics from a COLMAP text file.
    Args:
        path: Path to the cameras.txt file
    Returns:
        Dict of Camera objects indexed by camera ID
    """
    cameras = {}
    with open(path, "r") as fid:
        # Skip comment lines at the start
        lines = (line.strip() for line in fid)
        lines = (line for line in lines if line and not line.startswith("#"))
        for line in lines:
            # Unpack elements directly using split with maxsplit
            camera_id, model, width, height, *params = line.split()
            camera_id = int(camera_id)
            width, height = int(width), int(height)
            assert camera_id not in cameras, f"Camera ID {camera_id} already exists"
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model,
                width=width,
                height=height,
                params=np.array([float(p) for p in params]),
            )
    return cameras


def read_colmap_intrinsics_binary(path_to_model_file):
    """
    Read camera intrinsics from a COLMAP binary file.
    Args:
        path_to_model_file: Path to the cameras.bin file
    Returns:
        Dict of Camera objects indexed by camera ID
    Raises:
        ValueError: If the number of cameras read doesn't match the expected count
        KeyError: If an invalid camera model ID is encountered
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        # Read number of cameras
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            # Read fixed-size camera properties
            camera_id, model_id, width, height = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            # Get camera model information
            try:
                camera_model = CAMERA_MODEL_IDS[model_id]
            except KeyError:
                raise KeyError(f"Invalid camera model ID: {model_id}")
            # Read camera parameters
            params = read_next_bytes(
                fid,
                num_bytes=8 * camera_model.num_params,
                format_char_sequence="d" * camera_model.num_params,
            )
            # Create camera object
            assert camera_id not in cameras, f"Camera ID {camera_id} already exists"
            cameras[camera_id] = Camera(
                id=camera_id,
                model=camera_model.model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
    # Verify camera count
    assert len(cameras) == num_cameras, f"Expected {num_cameras} cameras, but read {len(cameras)}"
    return cameras


def qvec_to_so3(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


class Image(BaseImage):
    def qvec_to_so3(self):
        return qvec_to_so3(self.qvec)


def read_colmap_extrinsics_binary(path_to_model_file):
    """
    Read camera extrinsics from a COLMAP binary file.
    Args:
        path_to_model_file: Path to the images.bin file
    Returns:
        List of Image objects sorted by image name
    Raises:
        ValueError: If string parsing or data reading fails
    """
    images = []
    with open(path_to_model_file, "rb") as fid:
        # Read number of registered images
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            # Read image properties (id, rotation, translation, camera_id)
            props = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id, *qvec_tvec, camera_id = props
            qvec = np.array(qvec_tvec[:4])
            tvec = np.array(qvec_tvec[4:7])
            # Read image name (null-terminated string)
            image_name = ""
            while True:
                current_char = read_next_bytes(fid, 1, "c")[0]
                if current_char == b"\x00":
                    break
                try:
                    image_name += current_char.decode("utf-8")
                except UnicodeDecodeError:
                    raise ValueError(f"Invalid character in image name at position {len(image_name)}")
            # Read 2D points
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            point_data = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            # Parse point data into coordinates and IDs
            xys = np.array([(point_data[i], point_data[i + 1]) for i in range(0, len(point_data), 3)])
            point3D_ids = np.array([int(point_data[i + 2]) for i in range(0, len(point_data), 3)])
            # Create image object
            images.append(
                Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
            )
    return sorted(images, key=lambda x: x.name)


def read_colmap_extrinsics_text(path):
    """
    Read camera extrinsics from a COLMAP text file.
    Args:
        path: Path to the images.txt file
    Returns:
        List of Image objects sorted by image name
    Raises:
        ValueError: If file format is invalid or data parsing fails
    """
    images = []
    with open(path, "r") as fid:
        # Skip comment lines and get valid lines
        lines = (line.strip() for line in fid)
        # This update handles cases in images.txt where no 2D points are associated with an image.
        # In such cases, some entries contain only:
        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        # and do not include the usual POINTS2D[] line (which normally lists (X, Y, POINT3D_ID)).
        # The following logic checks for missing points line and handles it properly:
        lines = (line for line in lines if line == "" or not line.startswith("#"))
        # Process lines in pairs (image info + points info)
        try:
            while True:
                # Read image info line
                image_line = next(lines, None)
                if image_line is None:
                    break
                # Parse image properties
                elems = image_line.split()
                if len(elems) < 10:  # Minimum required elements
                    raise ValueError(f"Invalid image line format: {image_line}")
                image_id = int(elems[0])
                qvec = np.array([float(x) for x in elems[1:5]])
                tvec = np.array([float(x) for x in elems[5:8]])
                camera_id = int(elems[8])
                image_name = elems[9]
                # Read points line
                points_line = next(lines, None)
                if points_line is None:
                    raise ValueError(f"Missing points data for image {image_name}")
                # Parse 2D points and 3D point IDs
                point_elems = points_line.split()
                if len(point_elems) % 3 != 0:
                    raise ValueError(f"Invalid points format for image {image_name}")
                xys = np.array(
                    [(float(point_elems[i]), float(point_elems[i + 1])) for i in range(0, len(point_elems), 3)]
                )
                point3D_ids = np.array([int(point_elems[i + 2]) for i in range(0, len(point_elems), 3)])
                # Create image object
                images.append(
                    Image(
                        id=image_id,
                        qvec=qvec,
                        tvec=tvec,
                        camera_id=camera_id,
                        name=image_name,
                        xys=xys,
                        point3D_ids=point3D_ids,
                    )
                )
        except (ValueError, IndexError) as e:
            raise ValueError(f"Error parsing extrinsics file: {e}")
    return sorted(images, key=lambda x: x.name)


def worker_init_fn(worker_id):
    """
    Worker initialization function for DataLoader multiprocessing.

    This function ensures that each worker process has a proper CUDA context
    and random number generator state, which is especially important on Windows.
    """
    import random

    import numpy as np
    import torch

    # Set random seeds for reproducibility
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    # Initialize CUDA context in worker process
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.cuda.current_device())
        # Force CUDA context creation by doing a small operation
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def configure_dataloader_for_platform(dataloader_kwargs: dict) -> dict:
    """
    Configure DataLoader kwargs for the current platform.

    Args:
        dataloader_kwargs: Dictionary of DataLoader arguments
        force_windows_multiprocessing: If True, allow multiprocessing on Windows despite potential issues

    Returns:
        Updated DataLoader kwargs
    """
    kwargs = dataloader_kwargs.copy()

    if "num_workers" in kwargs:
        original_num_workers = kwargs["num_workers"]
        kwargs["num_workers"] = original_num_workers

        # Adjust persistent_workers based on actual num_workers
        if "persistent_workers" in kwargs:
            kwargs["persistent_workers"] = kwargs["num_workers"] > 0

        # On Windows with multiprocessing, add worker initialization function
        if platform.system() == "Windows" and kwargs["num_workers"] > 0:
            kwargs["worker_init_fn"] = worker_init_fn

    return kwargs


def get_worker_id():
    """Get current worker ID for thread-local caching."""
    import threading

    # Get worker ID from current process/thread
    try:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            return f"worker_{worker_info.id}"
        else:
            return "main_process"
    except:
        return f"thread_{threading.get_ident()}"


def repair_nonfinite_rays(rays: np.ndarray, valid_mask: np.ndarray | None) -> int:
    """A1 — repair non-finite precomputed camera ray directions in place.

    NCore ``pixels_to_camera_rays`` can emit non-finite directions where the
    rational-model undistortion diverges (the denominator polynomial
    1+k4·r²+k5·r⁴+k6·r⁶ can have a pole inside the image; first observed
    2026-07-02 on inc_b6a9 camera_left_wide_90fov at px (1917, 1042)). A NaN
    ray direction poisons the rendered pixel and, through the tracer's
    backward, the whole model within one training step.

    Repair: replace each bad direction with the nearest finite same-row
    neighbour (geometrically closest available ray; the exact value is
    irrelevant because the pixel is also flagged invalid), falling back to
    +Z when the entire row is bad. When ``valid_mask`` ([H, W] bool) is
    given, bad pixels are flagged False so they never supervise training —
    same semantics as ego-masked pixels.

    Args:
        rays: ``[H, W, 3]`` or ``[N, 3]`` float array, modified in place.
        valid_mask: optional ``[H, W]`` bool array, modified in place.

    Returns:
        Number of repaired rays.
    """
    flat_input = rays.ndim == 2
    rays_hw = rays[None] if flat_input else rays
    bad = ~np.isfinite(rays_hw).all(axis=-1)  # [H, W]
    n_bad = int(bad.sum())
    if n_bad == 0:
        return 0
    fallback = np.array([0.0, 0.0, 1.0], dtype=rays_hw.dtype)
    w = rays_hw.shape[1]
    for v, u in zip(*np.nonzero(bad)):
        repaired = None
        for du in range(1, w):
            for cand in (u - du, u + du):
                if 0 <= cand < w and not bad[v, cand]:
                    repaired = rays_hw[v, cand]
                    break
            if repaired is not None:
                break
        rays_hw[v, u] = fallback if repaired is None else repaired
    if valid_mask is not None and not flat_input:
        valid_mask[bad] = False
    return n_bad


def compute_forward_valid_pixel_mask(camera_model, rays: np.ndarray) -> np.ndarray:
    """Return a bool mask from ``camera_rays_to_pixels().valid_flag``.

    Projects ``rays`` (``[H, W, 3]`` or ``[N, 3]``) back through the camera
    model's forward projection and returns a bool mask with shape ``rays.shape[:-1]``
    indicating which rays project to valid pixel coordinates.

    For ``OpenCVPinholeCameraModel`` the rational polynomial distortion polynomial's
    denominator can diverge, producing ``valid_flag=False`` even for rays that came
    from integer pixel locations.  ``FThetaCameraModel`` and
    ``OpenCVFisheyeCameraModel`` are always all-valid (no-op).

    Args:
        camera_model: An NCore camera model with ``camera_rays_to_pixels`` method.
        rays: ``[H, W, 3]`` or ``[N, 3]`` float array of camera-space ray directions.

    Returns:
        Bool mask shaped ``rays.shape[:-1]``.  ``True`` = ray projects to a valid
        pixel (within the finite trust domain).

    Raises:
        ValueError: If the number of valid-flag elements does not match the number
            of input rays, or if the input is empty.
    """
    flat_input = rays.ndim == 2
    rays_flat = rays.reshape(-1, 3) if not flat_input else rays

    n_rays = rays_flat.shape[0]
    if n_rays == 0:
        raise ValueError("compute_forward_valid_pixel_mask: received empty rays (no pixels).")

    result = camera_model.camera_rays_to_pixels(rays_flat)
    # result.valid_flag is a torch.Tensor of shape (n_rays,), dtype bool
    valid_flag = result.valid_flag
    if isinstance(valid_flag, torch.Tensor):
        valid_flag = valid_flag.cpu().numpy()

    if valid_flag.shape[0] != n_rays:
        raise ValueError(
            f"compute_forward_valid_pixel_mask: camera_rays_to_pixels returned "
            f"{valid_flag.shape[0]} valid flags for {n_rays} rays — "
            f"element count mismatch."
        )

    mask = valid_flag.astype(bool)
    if not flat_input:
        h, w = rays.shape[0], rays.shape[1]
        mask = mask.reshape(h, w)
    return mask


def maybe_apply_forward_valid_mask(
    camera_model,
    rays: np.ndarray,
    ego_mask: np.ndarray,
    camera_id: str,
    enabled: bool,
) -> bool:
    """Optionally AND a forward-valid pixel mask into ``ego_mask``.

    When ``enabled=True`` and the camera model is an OpenCVPinholeCameraModel
    (rational polynomial distortion), projects the camera-space ``rays`` back
    through ``camera_rays_to_pixels()`` and masks out pixels where the forward
    projection is invalid.  For FThetaCameraModel and OpenCVFisheyeCameraModel
    the function is a strict no-op (no forward projection is called).

    The operation is **in-place** on ``ego_mask`` (ANDed in).  Kept/removed pixel
    counts are logged via ``threedgrut.utils.logger.logger``.

    Args:
        camera_model: An NCore camera model.
        rays: ``[H, W, 3]`` full-resolution camera-space ray directions.
        ego_mask: ``[H, W]`` bool ego mask, modified in place.
        camera_id: Camera identifier for logging.
        enabled: Master switch — when False the function returns immediately.

    Returns:
        True if the mask was modified (forward validity was applied), False if
        no-op (disabled, FTheta, or OpenCVFisheye).
    """
    if not enabled:
        return False

    # Guard: only OpenCVPinholeCameraModel has the rational-polynomial
    # forward-valid mismatch that we need to mask.  FTheta and OpenCVFisheye
    # share the same polynomial pair for forward/inverse, so their valid_flag
    # is always all-True.
    # Use isinstance when ncore.sensors is available (production);
    # fall back to class-name check for Mac test environments or when
    # ncore.sensors is stubbed (pytest conftest stub is a MagicMock).
    _is_pinhole = False
    try:
        from ncore.sensors import OpenCVPinholeCameraModel
        if isinstance(OpenCVPinholeCameraModel, type) and isinstance(camera_model, OpenCVPinholeCameraModel):
            _is_pinhole = True
    except (ImportError, TypeError):
        pass
    if not _is_pinhole and camera_model.__class__.__name__ == "OpenCVPinholeCameraModel":
        _is_pinhole = True
    if not _is_pinhole:
        return False

    forward_valid = compute_forward_valid_pixel_mask(camera_model, rays)

    n_before = int(ego_mask.sum())
    ego_mask &= forward_valid
    n_after = int(ego_mask.sum())
    n_removed = n_before - n_after

    if n_removed > 0:
        kept_pct = 100.0 * n_after / max(n_before, 1)
        logger.info(
            f"[PIN-MASK-1] {camera_id}: forward-valid mask removed "
            f"{n_removed}/{n_before} pixels ({kept_pct:.2f}% kept)"
        )

    return True


def _compute_distortion_np(xy: np.ndarray, radial_coeffs: np.ndarray,
                           tangential_coeffs: np.ndarray,
                           thin_prism_coeffs: np.ndarray):
    """Pure-NumPy version of ``OpenCVPinholeCameraModel.__compute_distortion``.

    Args:
        xy: ``(N, 2)`` normalized image coordinates (ideal pinhole).
        radial_coeffs: ``(6,)`` rational radial distortion coefficients
            ``[k1, k2, k3, k4, k5, k6]``.
        tangential_coeffs: ``(2,)`` tangential distortion ``[p1, p2]``.
        thin_prism_coeffs: ``(4,)`` thin-prism coefficients.

    Returns:
        Tuple ``(icD, delta_x, delta_y, r_2)`` matching the NCore
        ``__compute_distortion`` signature.
    """
    x = xy[:, 0]
    y = xy[:, 1]
    x2 = x * x
    y2 = y * y
    r2 = x2 + y2
    xy_prod = x * y
    a1 = 2.0 * xy_prod
    a2 = r2 + 2.0 * x2
    a3 = r2 + 2.0 * y2

    # Rational radial distortion
    r4 = r2 * r2
    r6 = r4 * r2
    icD_num = 1.0 + r2 * (radial_coeffs[0] + r2 * (radial_coeffs[1] + r2 * radial_coeffs[2]))
    icD_den = 1.0 + r2 * (radial_coeffs[3] + r2 * (radial_coeffs[4] + r2 * radial_coeffs[5]))
    icD = icD_num / icD_den

    # Tangential + thin-prism distortion
    delta_x = (tangential_coeffs[0] * a1
               + tangential_coeffs[1] * a2
               + r2 * (thin_prism_coeffs[0] + r2 * thin_prism_coeffs[1]))
    delta_y = (tangential_coeffs[0] * a3
               + tangential_coeffs[1] * a1
               + r2 * (thin_prism_coeffs[2] + r2 * thin_prism_coeffs[3]))

    return icD, delta_x, delta_y, r2


def compute_opencv_pinhole_rays(
    pixel_coords: np.ndarray,
    principal_point: np.ndarray,
    focal_length: np.ndarray,
    radial_coeffs: np.ndarray,
    tangential_coeffs: np.ndarray,
    thin_prism_coeffs: np.ndarray,
    max_iterations: int = 30,
    stop_mse_px2: float = 1e-12,
    _convergence_diagnostics: bool = False,
) -> np.ndarray:
    """High-precision ``pixels_to_camera_rays`` for OpenCVPinholeCameraModel.

    Replicates the exact NCore iterative undistortion formula
    (``__iterative_undistort`` + ``__compute_distortion``) but with a
    configurable ``max_iterations`` defaulting to 30 instead of the SDK's
    hard-coded 10.  Pure-NumPy, no ``ncore`` or ``torch`` dependency.

    The SDK's 10-iteration cap causes under-convergence at wide FOV edges
    (max forward residual >5 px on b6a9 front-wide at 10 iters).  30 iterations
    reduces the max to <0.02 px.

    Args:
        pixel_coords: ``(N, 2)`` float32 pixel coordinates (with +0.5 center
            offset applied, i.e. the same as what ``pixels_to_image_points``
            produces internally).
        principal_point: ``(2,)`` ``(cx, cy)``.
        focal_length: ``(2,)`` ``(fx, fy)``.
        radial_coeffs: ``(6,)`` rational radial coeffs ``[k1..k6]``.
        tangential_coeffs: ``(2,)`` tangential coeffs ``[p1, p2]``.
        thin_prism_coeffs: ``(4,)`` thin-prism coeffs.
        max_iterations: Max Newton-like iterations (default 30).
        stop_mse_px2: Early-stop MSE threshold (same default as SDK).
        _convergence_diagnostics: If True, also return per-point convergence
            info (see returns).

    Returns:
        Normalized ray directions ``(N, 3)`` float32, same as
        ``camera_model.pixels_to_camera_rays(pixel_coords)``.

        When ``_convergence_diagnostics=True``, returns a tuple
        ``(rays, n_iters, final_mse_px2)`` where ``n_iters`` is the actual
        iteration count and ``final_mse_px2`` is the mean squared error.
    """
    pixel_coords = np.asarray(pixel_coords)
    principal_point = np.asarray(principal_point)
    focal_length = np.asarray(focal_length)
    radial_coeffs = np.asarray(radial_coeffs)
    tangential_coeffs = np.asarray(tangential_coeffs)
    thin_prism_coeffs = np.asarray(thin_prism_coeffs)
    if pixel_coords.ndim != 2 or pixel_coords.shape[1] != 2:
        raise ValueError(f"pixel_coords must have shape (N, 2), got {pixel_coords.shape}")
    expected_shapes = {
        "principal_point": (principal_point, (2,)),
        "focal_length": (focal_length, (2,)),
        "radial_coeffs": (radial_coeffs, (6,)),
        "tangential_coeffs": (tangential_coeffs, (2,)),
        "thin_prism_coeffs": (thin_prism_coeffs, (4,)),
    }
    for name, (value, expected) in expected_shapes.items():
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
    if np.any(focal_length == 0):
        raise ValueError("focal_length entries must be non-zero")

    # Step 1: ideal pinhole unprojection (same as SDK)
    cam_rays_0 = (pixel_coords.astype(np.float64) - principal_point.astype(np.float64)) \
        / focal_length.astype(np.float64)

    cam_rays = cam_rays_0.copy()
    n_iters = 0
    mse = float("inf")

    for _ in range(max_iterations):
        n_iters += 1
        icD, delta_x, delta_y, _ = _compute_distortion_np(
            cam_rays, radial_coeffs, tangential_coeffs, thin_prism_coeffs
        )

        # Build delta array: (N, 2)
        delta = np.column_stack([delta_x, delta_y])
        icD_col = icD[:, None]

        # Update: cam_rays_{t+1} = (cam_rays_0 - delta) / icD
        cam_rays_new = (cam_rays_0 - delta) / icD_col

        # Residual for convergence check (same as SDK: cam_rays - cam_rays_new)
        residual = cam_rays.astype(np.float64) - cam_rays_new.astype(np.float64)
        mse = float(np.mean(residual * residual))

        cam_rays = cam_rays_new

        if mse <= stop_mse_px2:
            break

    # Step 2: form 3D rays [x, y, 1.0] and normalize (same as SDK)
    ones = np.ones((cam_rays.shape[0], 1), dtype=np.float64)
    rays_3d = np.concatenate([cam_rays, ones], axis=1)
    norm = np.linalg.norm(rays_3d, axis=1, keepdims=True)
    rays = (rays_3d / norm).astype(np.float32)

    if _convergence_diagnostics:
        return rays, n_iters, mse

    return rays


def _validate_opencv_radial_domain(
    radial_coeffs: np.ndarray,
    max_r2: float,
    *,
    samples: int = 16384,
    epsilon: float = 1e-6,
) -> None:
    """Prove the rational radial map is pole-free and monotonic on a domain."""
    radial = np.asarray(radial_coeffs, dtype=np.float64)
    if radial.shape != (6,) or not np.isfinite(radial).all():
        raise ValueError("radial_coeffs must be a finite (6,) array")
    if not np.isfinite(max_r2) or max_r2 < 0.0 or samples < 2:
        raise ValueError("max_r2 must be finite/non-negative and samples >= 2")
    s = np.linspace(0.0, max_r2, samples, dtype=np.float64)
    numerator = 1.0 + s * (radial[0] + s * (radial[1] + s * radial[2]))
    denominator = 1.0 + s * (radial[3] + s * (radial[4] + s * radial[5]))
    numerator_prime = radial[0] + 2.0 * radial[1] * s + 3.0 * radial[2] * s * s
    denominator_prime = radial[3] + 2.0 * radial[4] * s + 3.0 * radial[5] * s * s
    if not np.isfinite(numerator).all() or not np.isfinite(denominator).all() or np.any(denominator <= epsilon):
        raise ValueError("unsafe OpenCV rational domain: denominator pole or non-positive denominator")
    scale = numerator / denominator
    derivative = scale + 2.0 * s * (
        numerator_prime * denominator - numerator * denominator_prime
    ) / (denominator * denominator)
    if not np.isfinite(derivative).all() or np.any(derivative <= epsilon):
        raise ValueError("unsafe OpenCV rational domain: radial mapping is non-monotonic or folded")


def validate_opencv_pinhole_domain_options(
    mask_forward_invalid_pixels: bool,
    use_validity_domain: bool,
) -> None:
    """Reject the legacy SDK mask combined with the calibrated CUDA domain."""
    if mask_forward_invalid_pixels and use_validity_domain:
        raise ValueError(
            "mask_forward_invalid_pixels uses the legacy SDK validity gate and cannot be enabled "
            "together with opencv_pinhole_use_validity_domain"
        )


def compute_max_valid_r2(
    principal_point: np.ndarray,
    focal_length: np.ndarray,
    radial_coeffs: np.ndarray,
    tangential_coeffs: np.ndarray,
    thin_prism_coeffs: np.ndarray,
    image_size: tuple[int, int],
    margin: float = 0.1,
    boundary_samples_per_edge: int = 4096,
    inverse_iterations: int = 80,
) -> float:
    """Certify the largest ideal normalized radius in an expanded image domain.

    The renderer's validity gate operates on the *undistorted* normalized ray
    radius, so dividing distorted image corners by focal length is incorrect for
    a rational wide-angle calibration.  This function densely samples the
    continuous boundary of the image plus its UT margin, applies the same
    high-precision OpenCV rational inverse used for dataset rays, then returns
    ``max((ray_x/ray_z)^2 + (ray_y/ray_z)^2)``.

    RGB is never read or remapped; this is calibration-only geometry.
    """
    pp = np.asarray(principal_point, dtype=np.float64)
    fl = np.asarray(focal_length, dtype=np.float64)
    radial = np.asarray(radial_coeffs, dtype=np.float64)
    tangential = np.asarray(tangential_coeffs, dtype=np.float64)
    thin_prism = np.asarray(thin_prism_coeffs, dtype=np.float64)
    w, h = int(image_size[0]), int(image_size[1])
    if pp.shape != (2,) or fl.shape != (2,) or w <= 0 or h <= 0:
        raise ValueError("principal_point/focal_length must be (2,) and image_size must be positive")
    if radial.shape != (6,) or tangential.shape != (2,) or thin_prism.shape != (4,):
        raise ValueError("distortion arrays must have shapes radial=(6,), tangential=(2,), thin_prism=(4,)")
    if not np.isfinite(pp).all() or not np.isfinite(fl).all() or np.any(fl <= 0.0):
        raise ValueError("principal_point must be finite and focal_length must be finite and positive")
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("margin must be finite and non-negative")
    if boundary_samples_per_edge < 2:
        raise ValueError("boundary_samples_per_edge must be >= 2")

    margin_x, margin_y = w * margin, h * margin
    x_min, x_max = -margin_x, w + margin_x
    y_min, y_max = -margin_y, h + margin_y
    xs = np.linspace(x_min, x_max, boundary_samples_per_edge, dtype=np.float64)
    ys = np.linspace(y_min, y_max, boundary_samples_per_edge, dtype=np.float64)
    boundary_pixels = np.concatenate(
        [
            np.column_stack([xs, np.full_like(xs, y_min)]),
            np.column_stack([xs, np.full_like(xs, y_max)]),
            np.column_stack([np.full_like(ys, x_min), ys]),
            np.column_stack([np.full_like(ys, x_max), ys]),
        ],
        axis=0,
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rays = compute_opencv_pinhole_rays(
            boundary_pixels,
            pp,
            fl,
            radial,
            tangential,
            thin_prism,
            max_iterations=inverse_iterations,
            stop_mse_px2=0.0,
        ).astype(np.float64)
    if not np.isfinite(rays).all() or np.any(np.abs(rays[:, 2]) < 1e-12):
        raise ValueError("unsafe OpenCV rational domain: inverse diverged, indicating a pole or fold")
    xy_ideal = rays[:, :2] / rays[:, 2:3]
    max_r2 = float(np.max(np.sum(xy_ideal * xy_ideal, axis=1)))
    _validate_opencv_radial_domain(radial, max_r2)
    return max_r2
