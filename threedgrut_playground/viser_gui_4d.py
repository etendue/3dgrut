# SPDX-License-Identifier: Apache-2.0
"""4D scene visualization GUI for v2 LayeredGaussians ckpts (Stage 8 / T8.3-T8.6).

Reads the ``viz_4d`` block written by ``Trainer.save_checkpoint`` (see
``threedgrut/viz/metadata.py``) and renders the full 4D driving scene in a
browser via viser 1.0:

  * Gaussian background — rendered every frame, dynamic-rigid layer poses
    follow ``timestamp_us`` via ``engine.render_pass(.., timestamp_us=t_us)``.
  * Ego polyline + per-frame camera frustum (green).
  * Dynamic track polylines colored by class, current-frame cuboid wireframes
    colored per-instance.
  * Road LiDAR (on by default) + dynamic LiDAR (off by default).

Three fallback modes (matches Task G):
  (a) v2 ckpt with viz_4d block      → full 4D viewer.
  (b) v2 ckpt without viz_4d block
        + ``--dataset_path`` given    → lazy-import NCoreDataset + extract on
                                        the fly.
  (c) v1 ckpt or fallback off         → static-3D mode (equivalent to the
                                        original viser_gui.py).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import deque
from typing import Optional

import numpy as np
import torch

try:
    import viser
    import viser.transforms as tf
except ImportError:
    print('viser not installed, please run "pip install viser"')
    sys.exit(1)

import kaolin
from kaolin.render.camera import Camera

from threedgrut.layers.layered_model import LayeredGaussians
from threedgrut.utils.misc import quaternion_to_so3
from threedgrut_playground.engine import Engine3DGRUT
from threedgrut_playground.utils.cuboid import (
    class_color,
    cuboid_world_edges,
    instance_color,
)
from threedgrut_playground.utils.viser_math import mat_to_wxyz
from threedgrut_playground.utils.viz4d_metadata import FourDMetadata


# =========================================================================== #
#                          Viser4DViewer                                       #
# =========================================================================== #
def get_c2w(camera: "viser.CameraHandle") -> np.ndarray:
    c2w = np.eye(4, dtype=np.float32)
    q = np.asarray(camera.wxyz)[None, :]
    q_torch = torch.from_numpy(q).float()
    R = quaternion_to_so3(q_torch)[0].cpu().numpy()
    c2w[:3, :3] = R
    c2w[:3, 3] = camera.position
    return c2w


class Viser4DViewer:
    """4D viewer for v2 LayeredGaussians ckpts.

    When ``engine`` is ``None`` we run in "no-gaussian-render" mode: no
    OptiX-backed Gaussian background is drawn (the viewer only shows scene
    primitives + timeline). This is the only viable mode on GPUs without RT
    cores — specifically the Ampere datacenter SKUs **A100 / A800** (RT
    cores were intentionally fused off; Hopper-era H100/H800/H200 keep
    third-gen RT cores and run OptiX fine, as does any RTX consumer card).
    On RT-less GPUs ``lib3dgrt_cc.so`` dlopen segfaults during Engine3DGRUT
    init, hence the bypass.
    """

    def __init__(self, *, port: int, engine: "Engine3DGRUT | None",
                 metadata: Optional[FourDMetadata],
                 target_fps: float = 20.0,
                 initial_fov_rad: Optional[float] = None):
        self.engine = engine
        self.meta = metadata
        self.port = port
        self.target_fps = target_fps
        # T8.13: when FourDMetadata carries FTheta polynomial intrinsics
        # (viz_4d schema_v2), the viewer pipes them straight into
        # engine.render_pass and locks the rendered W×H to the trained
        # resolution. FTheta principal_point is in pixel coords, so changing
        # W/H without re-scaling polynomial params would desync projection.
        # Pinhole / non-FTheta ckpts keep the user-resizable behavior.
        self.ftheta_intrinsics: Optional[dict] = (
            metadata.ego_primary_intrinsics_ftheta
            if metadata is not None and metadata.has_ftheta()
            else None
        )
        self.ftheta_render_wh: Optional[tuple] = (
            metadata.ego_primary_resolution
            if metadata is not None and metadata.has_ftheta()
            else None
        )
        # T8.12-FIX: explicit viser client camera fov on connect / Reset View.
        # Reference repo tools/viser_multilayer_nurec.py:280 hard-sets
        # client.camera.fov = math.radians(90); we never did and used viser's
        # default (~80°) which interacts poorly with FTheta-trained Gaussians.
        self.initial_fov_rad = initial_fov_rad
        self.render_times: deque[float] = deque(maxlen=3)

        # Timeline state
        self._t_us_current: int = metadata.t_us_first if metadata else 0
        self._is_playing: bool = False
        self._is_loop: bool = True
        self._speed: float = 1.0
        self._last_tick_wallclock: float = time.time()
        # Bug 1 fix: Play 模式下让 viser client camera 跟随 ego_pose_at(t_us)
        # 飘 (默认 OFF, free-orbit 保持原行为). 见 plan
        # /Users/etendue/.claude/plans/v2-t8-13-t8-14-bug-happy-starfish.md Phase B
        self._follow_ego_enabled: bool = False

        # Slider-mutation guard so programmatic updates don't re-fire the
        # on_update callback into an infinite loop.
        self._suppress_slider_cb: bool = False

        # Scene handles populated by _populate_static_scene / _update_active_cuboids.
        self.h_ego_traj = None
        self.h_ego_frustum = None
        self.h_road = None
        self.h_dyn_pts = None
        self.h_cuboid_lines = None
        self.h_track_trajectories = None
        self.h_world_axes = None

        # Render dirtiness — set by camera move, slider drag, visibility toggle.
        self.need_update = True

        # --- viser server + GUI ---
        self.server = viser.ViserServer(port=self.port)
        self._clear_engine_meshes()
        self._build_static_gui()
        if self.meta is not None:
            self._build_timeline_gui()
            self._build_visibility_gui()
            self._populate_static_scene()
            # Force initial cuboid + frustum at t_us_first.
            self._on_time_change(self._t_us_current, source="init")

        @self.server.on_client_connect
        def _on_connect(client: viser.ClientHandle):
            # Apply viewer_defaults to the new client's free camera.
            if self.meta is not None:
                client.camera.wxyz = mat_to_wxyz(self.meta.initial_c2w)
                client.camera.position = self.meta.initial_c2w[:3, 3]
            if self.initial_fov_rad is not None:
                client.camera.fov = float(self.initial_fov_rad)
            @client.camera.on_update
            def _(_):
                self.need_update = True

    # ---------------------------------------------------------------- engine
    def _clear_engine_meshes(self) -> None:
        """Drop the default Engine3DGRUT mesh primitives so we render the
        scene cleanly (mirrors viser_gui.py:set_initial_mesh).

        No-op when no engine (no-gaussian-render mode).
        """
        if self.engine is None:
            return
        for mesh_name in list(self.engine.primitives.objects.keys()):
            self.engine.primitives.remove_primitive(mesh_name)

    # ---------------------------------------------------------------- GUI
    def _build_static_gui(self) -> None:
        """Resolution / near / far / FPS — only when Gaussian rendering is on.

        In no-gaussian-render mode (engine=None) we only expose Reset View,
        since resolution/near/far/fps are render-side controls with no effect
        on scene primitives.
        """
        folder = self.server.gui.add_folder("Render Controls")
        with folder:
            self.reset_view_button = self.server.gui.add_button("Reset View")
            if self.engine is not None:
                self.resolution_slider = self.server.gui.add_slider(
                    "Resolution", min=384, max=4096, step=2, initial_value=1024)
                self.near_plane_slider = self.server.gui.add_slider(
                    "Near", min=0.1, max=30, step=0.5, initial_value=0.1)
                self.far_plane_slider = self.server.gui.add_slider(
                    "Far", min=30.0, max=1000.0, step=10.0, initial_value=1000.0)
                self.fps = self.server.gui.add_text("FPS", initial_value="-1",
                                                    disabled=True)
            else:
                self.server.gui.add_text(
                    "Mode",
                    initial_value="scene primitives only (no Gaussian render)",
                    disabled=True,
                )
                self.resolution_slider = None
                self.near_plane_slider = None
                self.far_plane_slider = None
                self.fps = None

        if self.engine is not None:
            for slider in (self.resolution_slider, self.near_plane_slider,
                           self.far_plane_slider):
                @slider.on_update
                def _(_, _self=self):
                    _self.need_update = True
            # T8.13: FTheta mode locks render W×H to trained resolution
            # (principal_point is in pixels); hide the slider + show why.
            if self.ftheta_render_wh is not None:
                self.resolution_slider.visible = False
                with folder:
                    w, h = self.ftheta_render_wh
                    self.server.gui.add_markdown(
                        f"⚠️ **FTheta 模式**: render W×H 锁定到 "
                        f"`{w}×{h}` (训练分辨率)，不可调节。"
                    )

        # Per-layer Gaussian render toggles. Nested as a sub-folder under
        # Render Controls so users find them next to Near/Far/Resolution.
        # Skipped for v1 ckpts (scene_mog is MixtureOfGaussians, no .specs)
        # and no-gaussian-render mode (engine=None) where nothing renders.
        self.layer_checkboxes: dict = {}
        if self.engine is not None and isinstance(
            getattr(self.engine, "scene_mog", None), LayeredGaussians
        ):
            scene_mog = self.engine.scene_mog
            with folder:
                gaussian_layers_folder = self.server.gui.add_folder(
                    "Gaussian Layers"
                )
            with gaussian_layers_folder:
                for spec in scene_mog.specs:
                    # dynamic_deformables: registry stub, no module in
                    # self.layers; skip silently.
                    if not spec.is_particle_layer and spec.name != "sky_envmap":
                        continue
                    if spec.name not in scene_mog.layers:
                        continue
                    cb = self.server.gui.add_checkbox(
                        spec.name, initial_value=True
                    )
                    self.layer_checkboxes[spec.name] = cb

                    @cb.on_update
                    def _(_, _self=self, _name=spec.name, _cb=cb):
                        mog = _self.engine.scene_mog
                        new_set = set(mog.enabled_layer_names)
                        if bool(_cb.value):
                            new_set.add(_name)
                        else:
                            new_set.discard(_name)
                        # Wholesale replace (not in-place mutate) so a render
                        # iterating self.specs sees an atomic flip under GIL.
                        object.__setattr__(
                            mog, "enabled_layer_names", new_set
                        )
                        _self.need_update = True

        @self.reset_view_button.on_click
        def _(_):
            self.need_update = True
            for client in self.server.get_clients().values():
                # T8.12 fix: snap camera back to ckpt's initial_c2w so user
                # sees the dashcam viewpoint matching the training cameras
                # (Gaussian Splatting fails to generalize far from training
                # cameras → garbage if user drifts to aerial view).
                if self.meta is not None:
                    client.camera.position = self.meta.initial_c2w[:3, 3]
                    client.camera.wxyz = mat_to_wxyz(self.meta.initial_c2w)
                    # Set look_at along camera forward; viser uses orbit
                    # controls so look_at anchors the rotation pivot.
                    R = self.meta.initial_c2w[:3, :3]
                    forward_cam = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                    forward_world = R @ forward_cam
                    client.camera.look_at = (
                        self.meta.initial_c2w[:3, 3] + 10.0 * forward_world
                    )
                # Re-apply up vector after wxyz update so orbit controls
                # don't snap to default world-up (which is wrong for NCore
                # convention where camera +Y = world -Z = gravity down).
                client.camera.up_direction = (
                    tf.SO3(client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])
                )
                if self.initial_fov_rad is not None:
                    client.camera.fov = float(self.initial_fov_rad)

    def _build_timeline_gui(self) -> None:
        """Time slider + Play/Pause/Loop/Speed (only when metadata present)."""
        assert self.meta is not None
        folder = self.server.gui.add_folder("Timeline")
        with folder:
            self.time_slider = self.server.gui.add_slider(
                "Time (us)",
                min=int(self.meta.t_us_first),
                max=max(int(self.meta.t_us_last), int(self.meta.t_us_first) + 1),
                step=max(1, (self.meta.t_us_last - self.meta.t_us_first) // 1000 or 1),
                initial_value=int(self.meta.t_us_first),
            )
            self.frame_number = self.server.gui.add_number(
                "Frame", min=0, max=max(self.meta.n_frames() - 1, 0),
                step=1, initial_value=0,
            )
            self.play_button = self.server.gui.add_button("▶ Play")
            self.loop_cb = self.server.gui.add_checkbox("Loop", True)
            self.speed_slider = self.server.gui.add_slider(
                "Speed", min=0.1, max=4.0, step=0.1, initial_value=1.0)

        @self.time_slider.on_update
        def _(_):
            if self._suppress_slider_cb:
                return
            self._on_time_change(int(self.time_slider.value), source="slider")

        @self.frame_number.on_update
        def _(_):
            if self._suppress_slider_cb:
                return
            idx = int(self.frame_number.value)
            idx = max(0, min(idx, self.meta.n_frames() - 1))
            ts = self.meta.tracks_camera_timestamps_us
            if ts.size > 0:
                self._on_time_change(int(ts[idx]), source="frame")

        @self.play_button.on_click
        def _(_):
            self._is_playing = not self._is_playing
            self.play_button.label = "⏸ Pause" if self._is_playing else "▶ Play"
            self._last_tick_wallclock = time.time()

        @self.loop_cb.on_update
        def _(_):
            self._is_loop = bool(self.loop_cb.value)

        @self.speed_slider.on_update
        def _(_):
            self._speed = float(self.speed_slider.value)

    def _build_visibility_gui(self) -> None:
        """Show/hide checkboxes for every scene group."""
        assert self.meta is not None
        folder = self.server.gui.add_folder("Visibility")
        with folder:
            self.show_ego_traj  = self.server.gui.add_checkbox("Ego trajectory", True)
            self.show_ego_frust = self.server.gui.add_checkbox("Ego frustum", True)
            self.show_tracks    = self.server.gui.add_checkbox("Track trajectories", True)
            self.show_cuboids   = self.server.gui.add_checkbox("Active cuboids", True)
            self.show_road      = self.server.gui.add_checkbox(
                "Road LiDAR", self.meta.road_xyz is not None)
            self.show_dyn_pts   = self.server.gui.add_checkbox("Dynamic LiDAR", False)
            self.show_axes      = self.server.gui.add_checkbox("World axes", False)
            # Bug 1 fix: Follow Ego — Play 时把 viser client camera 自动 snap
            # 到 ego_pose_at(t_us). 默认 OFF 保留 free-orbit, 勾选立刻同步一次
            # (避免要等下一次 slider/play tick).
            self.show_follow_ego = self.server.gui.add_checkbox(
                "Follow Ego", False
            )

        @self.show_ego_traj.on_update
        def _(_):
            if self.h_ego_traj is not None:
                self.h_ego_traj.visible = bool(self.show_ego_traj.value)

        @self.show_ego_frust.on_update
        def _(_):
            if self.h_ego_frustum is not None:
                self.h_ego_frustum.visible = bool(self.show_ego_frust.value)

        @self.show_tracks.on_update
        def _(_):
            if self.h_track_trajectories is not None:
                self.h_track_trajectories.visible = bool(self.show_tracks.value)

        @self.show_cuboids.on_update
        def _(_):
            if self.h_cuboid_lines is not None:
                self.h_cuboid_lines.visible = bool(self.show_cuboids.value)

        @self.show_road.on_update
        def _(_):
            if self.h_road is not None:
                self.h_road.visible = bool(self.show_road.value)

        @self.show_dyn_pts.on_update
        def _(_):
            if self.h_dyn_pts is not None:
                self.h_dyn_pts.visible = bool(self.show_dyn_pts.value)

        @self.show_axes.on_update
        def _(_):
            if self.h_world_axes is not None:
                self.h_world_axes.visible = bool(self.show_axes.value)

        @self.show_follow_ego.on_update
        def _(_):
            self._follow_ego_enabled = bool(self.show_follow_ego.value)
            if self._follow_ego_enabled:
                self._snap_clients_to_ego(self._t_us_current)
                self.need_update = True

    # ---------------------------------------------------------------- scene
    def _populate_static_scene(self) -> None:
        """One-shot scene primitives: world axes, ego polyline + frustum,
        LiDAR clouds, all-tracks polylines."""
        assert self.meta is not None
        self._add_world_axes()
        self._add_ego_trajectory()
        self._add_lidar_clouds()
        self._add_track_trajectories()

    def _add_world_axes(self) -> None:
        self.h_world_axes = self.server.scene.add_frame(
            "/world_axes", show_axes=True, axes_length=2.0, axes_radius=0.05)
        self.h_world_axes.visible = False  # default off, toggle via Visibility

    def _add_ego_trajectory(self) -> None:
        assert self.meta is not None
        if self.meta.ego_poses_c2w.size == 0:
            return
        pts = self.meta.ego_poses_c2w[:, :3, 3].astype(np.float32)
        # add_spline_catmull_rom needs at least 4 control points; fall back to
        # add_point_cloud-style line if the trajectory is too short.
        if pts.shape[0] >= 4:
            self.h_ego_traj = self.server.scene.add_spline_catmull_rom(
                "/ego/trajectory",
                positions=pts,
                color=(0.2, 1.0, 0.2),
                line_width=2.0,
            )
        else:
            # Degenerate trajectory → render as line_segments between consecutive points.
            if pts.shape[0] >= 2:
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                self.h_ego_traj = self.server.scene.add_line_segments(
                    "/ego/trajectory",
                    points=segs,
                    colors=np.full_like(segs, fill_value=0.2),
                    line_width=2.0,
                )
        # Ego frustum at frame 0.
        pose0 = self.meta.ego_poses_c2w[0]
        self.h_ego_frustum = self.server.scene.add_camera_frustum(
            "/ego/cur_frustum",
            fov=self.meta.ego_primary_fov_y_rad,
            aspect=self.meta.ego_primary_aspect,
            scale=0.6,
            color=(0.2, 1.0, 0.2),
            wxyz=mat_to_wxyz(pose0),
            position=pose0[:3, 3],
        )

    def _add_lidar_clouds(self) -> None:
        assert self.meta is not None
        if self.meta.road_xyz is not None and self.meta.road_xyz.size > 0:
            colors = (self.meta.road_rgb
                      if self.meta.road_rgb is not None
                      else np.full_like(self.meta.road_xyz, 0.5))
            self.h_road = self.server.scene.add_point_cloud(
                "/lidar/road",
                points=self.meta.road_xyz.astype(np.float32),
                colors=colors.astype(np.float32),
                point_size=0.04,
            )
        # Dynamic LiDAR: T8.11 prefers per-track object-local points so the
        # cloud follows the cuboid at each frame (added per-frame in
        # _update_dynamic_lidar). Fall back to static world-frame union only
        # when the per-track block is absent (legacy ckpts).
        if self.meta.has_per_track_dyn_lidar():
            # Precompute lookup: track_name → indices of dyn_local_xyz rows
            # belonging to it. Lets _build_dyn_lidar_world avoid scanning the
            # full track_ids array every frame.
            names = self.meta.dyn_track_names or []
            track_ids = self.meta.dyn_track_ids
            self._dyn_idx_by_track: dict[str, np.ndarray] = {
                tid: np.where(track_ids == i)[0] for i, tid in enumerate(names)
            }
        elif self.meta.dyn_xyz is not None and self.meta.dyn_xyz.size > 0:
            # Legacy static fallback.
            colors = (self.meta.dyn_rgb
                      if self.meta.dyn_rgb is not None
                      else np.broadcast_to(
                          np.array([1.0, 0.5, 0.0], dtype=np.float32),
                          self.meta.dyn_xyz.shape,
                      ).copy())
            self.h_dyn_pts = self.server.scene.add_point_cloud(
                "/lidar/dynamic",
                points=self.meta.dyn_xyz.astype(np.float32),
                colors=colors.astype(np.float32),
                point_size=0.06,
            )
            self.h_dyn_pts.visible = False

    def _build_dyn_lidar_world(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Transform per-track object-local dyn LiDAR points to world frame
        using each track's pose at ``frame_idx``. Only includes tracks that
        are active at this frame.

        Returns:
            (pts[N, 3], colors[N, 3]) — empty arrays when no active tracks
            or per-track block is absent.
        """
        assert self.meta is not None
        if (not self.meta.has_per_track_dyn_lidar()
                or not hasattr(self, "_dyn_idx_by_track")):
            return (np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0, 3), dtype=np.float32))
        active = self.meta.active_tracks_at(frame_idx)
        if not active:
            return (np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0, 3), dtype=np.float32))
        local_xyz = self.meta.dyn_local_xyz
        pts_list: list[np.ndarray] = []
        col_list: list[np.ndarray] = []
        for tid in active:
            idxs = self._dyn_idx_by_track.get(tid)
            if idxs is None or idxs.size == 0:
                continue
            pose = self.meta.tracks[tid]["poses"][frame_idx]   # (4, 4)
            R = pose[:3, :3]
            t = pose[:3, 3]
            local = local_xyz[idxs]                            # (M, 3)
            world = (local @ R.T) + t                          # (M, 3)
            pts_list.append(world.astype(np.float32))
            color = np.array(instance_color(tid), dtype=np.float32)
            col_list.append(np.broadcast_to(color, world.shape).astype(np.float32).copy())
        if not pts_list:
            return (np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0, 3), dtype=np.float32))
        return np.concatenate(pts_list, axis=0), np.concatenate(col_list, axis=0)

    def _update_dynamic_lidar(self, frame_idx: int) -> None:
        """Per-frame refresh of /lidar/dynamic_active by remove+add.

        Skipped entirely when per-track block is absent (legacy ckpts keep
        their static world-frame snapshot from _add_lidar_clouds).
        """
        if self.meta is None or not self.meta.has_per_track_dyn_lidar():
            return
        pts, cols = self._build_dyn_lidar_world(frame_idx)
        # Carry forward user's visibility toggle when re-adding.
        prev_visible = (self.h_dyn_pts.visible
                        if self.h_dyn_pts is not None
                        else True)
        if self.h_dyn_pts is not None:
            self.h_dyn_pts.remove()
            self.h_dyn_pts = None
        if pts.shape[0] == 0:
            return
        self.h_dyn_pts = self.server.scene.add_point_cloud(
            "/lidar/dynamic_active",
            points=pts,
            colors=cols,
            point_size=0.06,
        )
        self.h_dyn_pts.visible = prev_visible

    def _add_track_trajectories(self) -> None:
        """All tracks' active-frame polylines as one batched line_segments."""
        assert self.meta is not None
        seg_list: list[np.ndarray] = []
        col_list: list[np.ndarray] = []
        for tid, t in self.meta.tracks.items():
            mask = t["frame_info"]
            poses = t["poses"]
            if mask is None or poses is None or mask.sum() < 2:
                continue
            centers = poses[mask, :3, 3]  # (Mi, 3) world-frame centers
            if centers.shape[0] < 2:
                continue
            segs = np.stack([centers[:-1], centers[1:]], axis=1).astype(np.float32)
            color = np.array(class_color(t["class"]), dtype=np.float32)
            col = np.broadcast_to(color, (segs.shape[0], 2, 3)).astype(np.float32).copy()
            seg_list.append(segs)
            col_list.append(col)
        if not seg_list:
            return
        all_segs = np.concatenate(seg_list, axis=0)
        all_cols = np.concatenate(col_list, axis=0)
        self.h_track_trajectories = self.server.scene.add_line_segments(
            "/tracks/all_trajectories",
            points=all_segs,
            colors=all_cols,
            line_width=1.5,
        )

    # ---------------------------------------------------------------- per-frame
    def _on_time_change(self, t_us: int, *, source: str = "") -> None:
        """Central time-update dispatch."""
        if self.meta is None:
            return
        # Clamp to [first, last] range and store.
        t_us = max(self.meta.t_us_first, min(int(t_us), self.meta.t_us_last))
        self._t_us_current = t_us
        # Update ego frustum + cuboid + UI mirrors.
        self._update_ego_frustum(t_us)
        frame_idx = self.meta.lookup_frame_idx(t_us)
        self._update_active_cuboids(frame_idx)
        self._update_dynamic_lidar(frame_idx)
        self._mirror_ui(t_us, frame_idx)
        # Bug 1 fix: if Follow Ego is on, snap viewer cameras to the new
        # ego pose so Play visibly tracks the trajectory.
        if self._follow_ego_enabled:
            self._snap_clients_to_ego(t_us)
        self.need_update = True

    def _mirror_ui(self, t_us: int, frame_idx: int) -> None:
        """Programmatically update slider + frame number without re-firing CBs."""
        self._suppress_slider_cb = True
        try:
            if hasattr(self, "time_slider"):
                self.time_slider.value = int(t_us)
            if hasattr(self, "frame_number"):
                self.frame_number.value = int(frame_idx)
        finally:
            self._suppress_slider_cb = False

    def _update_ego_frustum(self, t_us: int) -> None:
        assert self.meta is not None
        if self.h_ego_frustum is None:
            return
        pose = self.meta.ego_pose_at(t_us)
        self.h_ego_frustum.wxyz = mat_to_wxyz(pose)
        self.h_ego_frustum.position = pose[:3, 3]

    def _snap_clients_to_ego(self, t_us: int) -> None:
        """Snap every connected viser client's free camera onto the ego pose
        at ``t_us``. Mirrors Reset View's wxyz/position/look_at/up_direction
        writes (see _build_static_gui Reset View handler) but uses the
        per-frame ego pose instead of meta.initial_c2w.

        No-op when metadata absent (v1 ckpt static-3D mode) or no clients.
        """
        if self.meta is None:
            return
        pose = self.meta.ego_pose_at(t_us)
        R = pose[:3, :3]
        # NCore camera convention: +Z_cam = forward, +Y_cam = down. Match the
        # Reset View handler (_build_static_gui line 273-285) so Follow Ego
        # behaves like an automated Reset View at each frame.
        forward_world = R @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        up_world = R @ np.array([0.0, -1.0, 0.0], dtype=np.float32)
        for client in self.server.get_clients().values():
            client.camera.wxyz = mat_to_wxyz(pose)
            client.camera.position = pose[:3, 3]
            client.camera.look_at = pose[:3, 3] + 10.0 * forward_world
            client.camera.up_direction = up_world

    def _build_cuboid_edges(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        assert self.meta is not None
        active = self.meta.active_tracks_at(frame_idx)
        if not active:
            return (np.zeros((0, 2, 3), dtype=np.float32),
                    np.zeros((0, 2, 3), dtype=np.float32))
        pts_list, col_list = [], []
        for tid in active:
            t = self.meta.tracks[tid]
            poses = t["poses"]
            if poses is None or frame_idx >= poses.shape[0]:
                continue
            pose = poses[frame_idx]
            size = t["size"] if t["size"] is not None else np.array(
                [1.0, 1.0, 1.0], dtype=np.float32)
            world = cuboid_world_edges(pose, size)
            color = np.array(instance_color(tid), dtype=np.float32)
            col = np.broadcast_to(color, (12, 2, 3)).astype(np.float32).copy()
            pts_list.append(world)
            col_list.append(col)
        return (np.concatenate(pts_list, axis=0),
                np.concatenate(col_list, axis=0))

    def _update_active_cuboids(self, frame_idx: int) -> None:
        """Remove + re-add line_segments handle for active cuboids.

        viser 1.0 line-segment handle exposes ``.visible/.position/.wxyz`` but
        not in-place vertex updates → we replace the node each frame.
        """
        pts, cols = self._build_cuboid_edges(frame_idx)
        if self.h_cuboid_lines is not None:
            self.h_cuboid_lines.remove()
            self.h_cuboid_lines = None
        if pts.shape[0] == 0:
            return
        self.h_cuboid_lines = self.server.scene.add_line_segments(
            "/tracks/active_cuboids",
            points=pts,
            colors=cols,
            line_width=2.5,
        )

    # ---------------------------------------------------------------- render
    def _play_tick(self) -> None:
        """Advance ``_t_us_current`` by wallclock_dt * speed * 1e6 when playing."""
        if not self._is_playing or self.meta is None:
            self._last_tick_wallclock = time.time()
            return
        now = time.time()
        dt_us = int((now - self._last_tick_wallclock) * 1e6 * self._speed)
        self._last_tick_wallclock = now
        if dt_us == 0:
            return
        new_t = self._t_us_current + dt_us
        if new_t > self.meta.t_us_last:
            new_t = (self.meta.t_us_first
                     if self._is_loop
                     else self.meta.t_us_last)
            if not self._is_loop:
                self._is_playing = False
                self.play_button.label = "▶ Play"
        self._on_time_change(new_t, source="play")

    def fast_render(self, kaolin_camera: Camera) -> np.ndarray:
        """Run engine for one Gaussian pass; return RGB uint8 frame."""
        out = self.engine.render_pass(
            kaolin_camera, is_first_pass=True, timestamp_us=self._t_us_current,
            fisheye_intrinsics=self.ftheta_intrinsics,  # T8.13
        )
        rgba = torch.cat([out["rgb"], out["opacity"]], dim=-1)
        rgba = torch.clamp(rgba, 0.0, 1.0)
        img = (rgba[0, :, :, :3] * 255).to(torch.uint8)
        return img.cpu().numpy()

    @torch.no_grad()
    def update(self) -> None:
        """Outer render loop: advance timeline, then re-render if dirty.

        In no-gaussian-render mode (engine is None) we still tick the timeline
        — scene primitives + frustum stay live — but skip the Gaussian
        background pass entirely (no Engine3DGRUT means no OptiX, the whole
        point of this mode).
        """
        self._play_tick()
        if self.engine is None:
            # Scene primitives auto-update via _on_time_change; nothing to
            # render every frame here.
            self.need_update = False
            return
        if not self.need_update:
            return
        interval = 0.0
        for client in self.server.get_clients().values():
            try:
                # T8.13 FTheta path locks W×H to trained resolution; pinhole
                # path keeps the user-controllable slider (T8.12 behavior).
                if self.ftheta_render_wh is not None:
                    W, H = self.ftheta_render_wh
                else:
                    W = self.resolution_slider.value
                    H = int(self.resolution_slider.value / client.camera.aspect)
                view_matrix = get_c2w(client.camera)
                fov_y = client.camera.fov
                near = self.near_plane_slider.value
                far  = self.far_plane_slider.value
                kaolin_camera = Camera.from_args(
                    view_matrix=view_matrix,
                    fov=fov_y,
                    width=W,
                    height=H,
                    near=near,
                    far=far,
                    dtype=torch.float32,
                    device=self.engine.device,
                )
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                img = self.fast_render(kaolin_camera)
                end.record()
                torch.cuda.synchronize()
                interval = start.elapsed_time(end) / 1000.0
            except RuntimeError as e:
                print(e)
                interval = 1
                continue
            client.scene.set_background_image(img, format="jpeg")
        self.render_times.append(interval)
        if self.render_times and self.fps is not None:
            self.fps.value = f"{1.0 / max(np.mean(self.render_times), 1e-6):.3g}"
        self.need_update = False


# =========================================================================== #
#                          Loader + main                                       #
# =========================================================================== #
def _print_startup_diagnostics(viewer: "Viser4DViewer", ckpt: dict) -> None:
    """Emit [T8.13-DIAG] block to terminal once on startup to anchor the
    4-bug investigation tracked in
    /Users/etendue/.claude/plans/v2-t8-13-t8-14-bug-happy-starfish.md
    (Phase A). Output is read-only — no GUI / render side effects.

    Four sections, each guarded so absence of feature doesn't crash:
      A.1  layer registry + enabled_layer_names + checkbox bindings (Bug 3)
      A.2  viz_4d time range + train.duration_sec + per-track frame_info (Bug 4)
      A.3  ego_pose_at(t_us_first) vs meta.initial_c2w sanity      (Bug 1)
      A.4  FTheta vs viser-pinhole projection asymmetry warning    (Bug 2)
    """
    print("[T8.13-DIAG] ============================================================")

    # ---- A.1 Layer registry --------------------------------------------------
    print("[T8.13-DIAG] A.1 Layer registry (Bug 3 evidence):")
    engine = viewer.engine
    if engine is None or not hasattr(engine, "scene_mog"):
        print("[T8.13-DIAG]   - engine=None (no_gaussian_render); skipping")
    else:
        mog = engine.scene_mog
        if isinstance(mog, LayeredGaussians):
            specs = list(mog.specs)
            enabled = set(getattr(mog, "enabled_layer_names", set()))
            print(f"[T8.13-DIAG]   - scene_mog type: LayeredGaussians "
                  f"({len(specs)} specs, {len(mog.layers)} modules)")
            for spec in specs:
                in_layers = spec.name in mog.layers
                n_particles = 0
                if in_layers and hasattr(mog.layers[spec.name], "positions"):
                    try:
                        n_particles = int(mog.layers[spec.name].positions.shape[0])
                    except Exception:
                        n_particles = -1
                print(f"[T8.13-DIAG]     - {spec.name:24s} "
                      f"is_particle={spec.is_particle_layer!s:5s}  "
                      f"in_layers={in_layers!s:5s}  "
                      f"n_particles={n_particles}  "
                      f"in_enabled={spec.name in enabled!s}")
            print(f"[T8.13-DIAG]   - enabled_layer_names: {sorted(enabled)}")
            # Bug 3 prior observation 321: tracks_poses is a python dict that
            # _populate_tracks_impl writes only in __init__; init_from_checkpoint
            # does NOT repopulate it. If empty, dynamic_rigid Gaussians render
            # at object-local frame regardless of toggle → user perceives "no
            # effect". Surface size here to confirm/refute.
            tracks_poses = getattr(mog, "tracks_poses", None)
            if isinstance(tracks_poses, dict):
                print(f"[T8.13-DIAG]   - tracks_poses dict: "
                      f"{len(tracks_poses)} tracks "
                      f"(empty after ckpt load → known Bug 3 root cause hint)")
            else:
                print(f"[T8.13-DIAG]   - tracks_poses: {type(tracks_poses).__name__}")
        else:
            print(f"[T8.13-DIAG]   - scene_mog type: {type(mog).__name__} "
                  f"(v1 MixtureOfGaussians, no layers)")
    cbs = getattr(viewer, "layer_checkboxes", {}) or {}
    print(f"[T8.13-DIAG]   - viser checkboxes registered: "
          f"{[(n, bool(cb.value)) for n, cb in cbs.items()]}")

    # ---- A.2 viz_4d time range ----------------------------------------------
    print("[T8.13-DIAG] A.2 viz_4d time range (Bug 4 evidence):")
    md = viewer.meta
    if md is None:
        print("[T8.13-DIAG]   - metadata=None (v1 ckpt static mode); skipping")
    else:
        dur_us = md.t_us_last - md.t_us_first
        print(f"[T8.13-DIAG]   - schema_version:      {md.schema_version}")
        print(f"[T8.13-DIAG]   - t_us_first:          {md.t_us_first} us "
              f"({md.t_us_first / 1e6:.3f} s)")
        print(f"[T8.13-DIAG]   - t_us_last:           {md.t_us_last} us "
              f"({md.t_us_last / 1e6:.3f} s)")
        print(f"[T8.13-DIAG]   - duration:            {dur_us / 1e6:.3f} s")
        print(f"[T8.13-DIAG]   - n_ego_frames:        "
              f"{md.ego_frame_timestamps_us.shape[0]}")
        print(f"[T8.13-DIAG]   - n_track_frames:      {md.n_frames()}")
        # Best-effort: read training duration_sec / seek_offset_sec from
        # ckpt['config'] (OmegaConf DictConfig in v2 ckpts).
        cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
        if cfg is not None:
            try:
                from omegaconf import OmegaConf  # local import: optional
                dur_sec = OmegaConf.select(cfg, "dataset.train.duration_sec",
                                            default="<unset>")
                seek_sec = OmegaConf.select(cfg, "dataset.train.seek_offset_sec",
                                             default="<unset>")
                iters = OmegaConf.select(cfg, "n_iterations", default="<unset>")
                print(f"[T8.13-DIAG]   - cfg.dataset.train.duration_sec:    "
                      f"{dur_sec}")
                print(f"[T8.13-DIAG]   - cfg.dataset.train.seek_offset_sec: "
                      f"{seek_sec}")
                print(f"[T8.13-DIAG]   - cfg.n_iterations:                  "
                      f"{iters}")
            except Exception as e:
                print(f"[T8.13-DIAG]   - cfg parse failed: {e!r}")
        else:
            print("[T8.13-DIAG]   - cfg: <ckpt has no 'config' key>")
        # Per-track frame_info coverage (first 5 tracks).
        print(f"[T8.13-DIAG]   - per-track frame_info coverage "
              f"(first 5 of {md.n_tracks()} tracks):")
        for i, (tid, t) in enumerate(md.tracks.items()):
            if i >= 5:
                break
            fi = t.get("frame_info")
            if fi is None or fi.size == 0:
                print(f"[T8.13-DIAG]     - {tid}: <empty frame_info>")
                continue
            n_active = int(fi.sum())
            n_total = int(fi.size)
            active_idx = np.where(fi)[0]
            rng = (f"[{int(active_idx[0])}, {int(active_idx[-1])}]"
                   if active_idx.size > 0 else "[]")
            print(f"[T8.13-DIAG]     - {tid}: active "
                  f"{n_active}/{n_total} ({100.0 * n_active / max(n_total, 1):.1f}%) "
                  f"range={rng}")

    # ---- A.3 ego_pose vs initial_c2w ----------------------------------------
    print("[T8.13-DIAG] A.3 Camera vs ego pose @ t_us_first (Bug 1 evidence):")
    if md is None:
        print("[T8.13-DIAG]   - metadata=None; skipping")
    else:
        ep = md.ego_pose_at(md.t_us_first)
        ic = md.initial_c2w
        dpos = float(np.linalg.norm(ep[:3, 3] - ic[:3, 3]))
        print(f"[T8.13-DIAG]   - ego_pose_at(t_us_first)[:3,3]: "
              f"{ep[:3, 3].tolist()}")
        print(f"[T8.13-DIAG]   - meta.initial_c2w[:3,3]:        "
              f"{ic[:3, 3].tolist()}")
        print(f"[T8.13-DIAG]   - delta:                         {dpos:.3f} m "
              f"({'OK' if dpos < 1.0 else 'WARN >1m suggests metadata stale'})")

    # ---- A.4 FTheta vs pinhole asymmetry ------------------------------------
    print("[T8.13-DIAG] A.4 Projection model (Bug 2 evidence):")
    if md is None:
        print("[T8.13-DIAG]   - metadata=None; viser draws pinhole only")
    elif md.has_ftheta():
        print("[T8.13-DIAG]   - engine path (Gaussian):  FTheta polynomial "
              "(8-key intrinsics)")
        print("[T8.13-DIAG]   - viser scene primitives:  pinhole "
              "(kaolin Camera.fov)")
        print("[T8.13-DIAG]   - WARN: cuboid/frustum/lidar drawn by viser "
              "frontend may not align with FTheta Gaussian backdrop, "
              "especially near image periphery. See plan Phase D for fix.")
    else:
        print("[T8.13-DIAG]   - engine path + viser:     pinhole (consistent, "
              "alignment should hold)")
    print("[T8.13-DIAG] ============================================================")


def _load_metadata(ckpt: dict, dataset_path: Optional[str],
                   default_config: str) -> Optional[FourDMetadata]:
    """Try ckpt['viz_4d'] first; fall back to dataset on-the-fly extract."""
    md = FourDMetadata.from_ckpt(ckpt)
    if md is not None:
        print(f"[viz_4d] loaded schema_v{md.schema_version} "
              f"({md.n_tracks()} tracks, ego_N={md.ego_poses_c2w.shape[0]})")
        return md
    if dataset_path is None:
        print("[viz_4d] ckpt has no viz_4d block; running static 3D mode "
              "(pass --dataset_path to extract 4D on the fly)")
        return None
    # Lazy import — only triggered when fallback is requested, so machines
    # without NCore SDK don't crash on plain v2 ckpts.
    try:
        from omegaconf import OmegaConf

        from threedgrut.datasets.datasetNcore import NCoreDataset
        from threedgrut.layers.layered_model import LayeredGaussians
        from threedgrut.layers.registry import specs_from_config
        from threedgrut.viz.metadata import extract_4d_metadata
    except ImportError as e:
        raise RuntimeError(
            "--dataset_path requires NCore SDK + LayeredGaussians stack; "
            f"falling back failed: {e}"
        )
    conf = ckpt["config"]
    conf = OmegaConf.merge(conf, OmegaConf.create({"path": dataset_path}))
    print(f"[viz_4d] no viz_4d block; extracting on-the-fly from {dataset_path}")
    train_ds = NCoreDataset(conf, split="train")
    specs = specs_from_config(conf)
    model = LayeredGaussians(conf, specs=specs, scene_extent=1.0)
    model.init_from_checkpoint(ckpt, setup_optimizer=False)
    md_dict = extract_4d_metadata(model, train_ds, conf)
    return FourDMetadata.from_ckpt({"viz_4d": md_dict})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs_object", type=str, required=True,
                        help="Path of pretrained 3dgrt checkpoint (.pt).")
    parser.add_argument("--mesh_assets", type=str,
                        default=os.path.join(os.path.dirname(__file__), "assets"))
    parser.add_argument("--default_gs_config", type=str,
                        default="apps/colmap_3dgrt.yaml")
    parser.add_argument("--envmap_assets", type=str,
                        default=os.path.join(os.path.dirname(__file__), "assets"))
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Optional NCore dataset path for 4D fallback when ckpt has no viz_4d.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--target_fps", type=float, default=20.0)
    parser.add_argument("--no_gaussian_render", action="store_true",
                        help="Skip Engine3DGRUT init + Gaussian background "
                             "rendering. Required on Ampere datacenter SKUs "
                             "WITHOUT RT cores (A100 / A800), where the OptiX "
                             "extension dlopen segfaults. Hopper datacenter "
                             "(H100/H800/H200) and all RTX cards have RT "
                             "cores and don't need this flag. Scene "
                             "primitives (ego/cuboid/LiDAR) + timeline still "
                             "work in this mode.")
    # T8.12-FIX (Phase A.2 + A.5): explicit viser fov + optional fisheye
    # raygen switch. Reference repo (tools/viser_multilayer_nurec.py) uses
    # --fov_deg=90 hard-set for fisheye-trained ckpts and renders cleanly via
    # pinhole approximation. Our engine already has _raygen_fisheye gated by
    # engine.camera_type; we just never wire it through viser_gui_4d. These
    # flags let A800/vast.ai operator A/B-test pinhole-90 vs fisheye-120
    # without code edits.
    parser.add_argument("--initial_fov_deg", type=float, default=90.0,
                        help="Initial viser client camera vertical fov (deg). "
                             "Default 90 matches reference repo "
                             "tools/viser_multilayer_nurec.py. Set explicitly "
                             "to test e.g. 60 / 75 / 120; viser UI still lets "
                             "the user override at runtime.")
    parser.add_argument("--camera_type", type=str, default="Pinhole",
                        choices=["Pinhole", "Fisheye"],
                        help="Engine raygen mode. 'Pinhole' (default) matches "
                             "T8.12 + reference repo behavior. 'Fisheye' "
                             "routes through engine._raygen_fisheye + "
                             "generate_fisheye_rays; use with --camera_fov_deg "
                             "matching training (e.g. 120 for NCore "
                             "camera_front_wide_120fov).")
    parser.add_argument("--camera_fov_deg", type=float, default=None,
                        help="Engine fisheye fov (deg). Only used when "
                             "--camera_type=Fisheye. Defaults to "
                             "--initial_fov_deg when omitted.")
    args = parser.parse_args()

    if args.no_gaussian_render:
        engine = None
        print("[viz_4d] --no_gaussian_render: skipping Engine3DGRUT "
              "(scene primitives + timeline only)")
    else:
        engine = Engine3DGRUT(
            gs_object=args.gs_object,
            mesh_assets_folder=args.mesh_assets,
            envmap_assets_folder=args.envmap_assets,
            default_config=args.default_gs_config,
        )
        # T8.12-FIX: opt-in fisheye raygen for FTheta-trained ckpts. Reference
        # repo + our T8.12 stuck with pinhole; this hook lets operators flip
        # without rebuilding the engine.
        if args.camera_type == "Fisheye":
            fisheye_fov_deg = (args.camera_fov_deg
                               if args.camera_fov_deg is not None
                               else args.initial_fov_deg)
            engine.camera_type = "Fisheye"
            engine.camera_fov = float(fisheye_fov_deg)
            print(f"[viz_4d] engine.camera_type=Fisheye, "
                  f"engine.camera_fov={fisheye_fov_deg}°")
    # Need ckpt dict for metadata; re-load explicitly (engine loaded model only).
    if args.gs_object.endswith(".pt"):
        ckpt = torch.load(args.gs_object, weights_only=False)
    else:
        ckpt = {}
    metadata = _load_metadata(ckpt, args.dataset_path, args.default_gs_config)
    # T8.13: announce projection path so vast.ai / A800 operator sees
    # at a glance whether FTheta or pinhole approximation is in effect.
    if metadata is not None and metadata.has_ftheta():
        ft = metadata.ego_primary_intrinsics_ftheta
        print(f"[T8.13] FTheta intrinsics 已加载 "
              f"(resolution={metadata.ego_primary_resolution}, "
              f"max_angle={ft['max_angle']:.3f}rad). "
              f"GUI resolution slider 已锁定到训练分辨率。")
    else:
        print("[T8.13] 无 FTheta intrinsics, 走 pinhole approximation 路径 "
              "(T8.12 行为).")
    viewer = Viser4DViewer(
        port=args.port, engine=engine, metadata=metadata,
        target_fps=args.target_fps,
        initial_fov_rad=math.radians(args.initial_fov_deg),
    )
    # Phase A diagnostic block — one-shot startup print, no GUI/render side
    # effects. See plan v2-t8-13-t8-14-bug-happy-starfish.md for the 4-bug
    # anchor points each section maps to.
    try:
        _print_startup_diagnostics(viewer, ckpt)
    except Exception as e:
        print(f"[T8.13-DIAG] diagnostics failed (non-fatal): {e!r}")
    while True:
        start = time.time()
        viewer.update()
        elapsed = time.time() - start
        time.sleep(max(0, (1.0 / args.target_fps) - elapsed))


if __name__ == "__main__":
    main()
