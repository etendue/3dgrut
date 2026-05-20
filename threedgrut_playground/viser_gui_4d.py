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
                 target_fps: float = 20.0):
        self.engine = engine
        self.meta = metadata
        self.port = port
        self.target_fps = target_fps
        self.render_times: deque[float] = deque(maxlen=3)

        # Timeline state
        self._t_us_current: int = metadata.t_us_first if metadata else 0
        self._is_playing: bool = False
        self._is_loop: bool = True
        self._speed: float = 1.0
        self._last_tick_wallclock: float = time.time()

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

        @self.reset_view_button.on_click
        def _(_):
            self.need_update = True
            for client in self.server.get_clients().values():
                client.camera.up_direction = (
                    tf.SO3(client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])
                )

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
            kaolin_camera, is_first_pass=True, timestamp_us=self._t_us_current
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
    # Need ckpt dict for metadata; re-load explicitly (engine loaded model only).
    if args.gs_object.endswith(".pt"):
        ckpt = torch.load(args.gs_object, weights_only=False)
    else:
        ckpt = {}
    metadata = _load_metadata(ckpt, args.dataset_path, args.default_gs_config)
    viewer = Viser4DViewer(
        port=args.port, engine=engine, metadata=metadata,
        target_fps=args.target_fps,
    )
    while True:
        start = time.time()
        viewer.update()
        elapsed = time.time() - start
        time.sleep(max(0, (1.0 / args.target_fps) - elapsed))


if __name__ == "__main__":
    main()
