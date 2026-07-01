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
import hashlib
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
from threedgrut_playground.utils.difix_client import DifixClient
from threedgrut_playground.utils.harmonizer_client import (
    HarmonizerTemporalClient,
)
from threedgrut_playground.utils.viser_math import mat_to_wxyz
from threedgrut_playground.utils.viser_overlay_compositor import (
    PolylineLayerSpec,
    Viser4DOverlayCompositor,
)
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


# --- world-referenced yaw/pitch/roll <-> camera rotation --------------------
# Camera local convention (see get_c2w usage above): +Z forward, -Y up, +X
# right. World +Z up. yaw = heading in the xy-plane, pitch = elevation
# (+ looks up, - looks down), roll = rotation about the view axis. Round-trip
# verified to 1e-13 over 2000 random rotations.
_WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
_FWD_L = np.array([0.0, 0.0, 1.0], dtype=np.float64)
_UP_L = np.array([0.0, -1.0, 0.0], dtype=np.float64)


def cam_rot_to_ypr(R) -> tuple[float, float, float]:
    R = np.asarray(R, np.float64)
    fwd = R @ _FWD_L
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    up = R @ _UP_L
    yaw = math.degrees(math.atan2(fwd[1], fwd[0]))
    pitch = math.degrees(math.atan2(fwd[2], math.hypot(fwd[0], fwd[1])))
    proj = _WORLD_UP - fwd * float(np.dot(_WORLD_UP, fwd))
    n = np.linalg.norm(proj)
    if n < 1e-6:  # gimbal (straight up/down) → yaw-based zero-roll reference
        zero_up = np.array([-math.sin(math.radians(yaw)), math.cos(math.radians(yaw)), 0.0])
    else:
        zero_up = proj / n
    roll = math.degrees(math.atan2(float(np.dot(np.cross(zero_up, up), fwd)),
                                   float(np.dot(zero_up, up))))
    return yaw, pitch, roll


def ypr_to_cam_rot(yaw: float, pitch: float, roll: float) -> np.ndarray:
    y, p, r = map(math.radians, (yaw, pitch, roll))
    fwd = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    proj = _WORLD_UP - fwd * float(np.dot(_WORLD_UP, fwd))
    n = np.linalg.norm(proj)
    zero_up = (np.array([-math.sin(y), math.cos(y), 0.0]) if n < 1e-6 else proj / n)
    w = np.cross(fwd, zero_up)
    cam_up = zero_up * math.cos(r) + w * math.sin(r)
    down = -cam_up
    right = np.cross(down, fwd)
    return np.column_stack([right, down, fwd])


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
                 initial_fov_rad: Optional[float] = None,
                 multi_cam_poses: Optional[dict] = None,
                 difix_server: Optional[str] = None,
                 harmonizer_temporal_server: Optional[str] = None,
                 harmonizer_temporal_K: int = 4,
                 initial_cam_id: Optional[str] = None):
        self.engine = engine
        self.meta = metadata
        self.port = port
        self.target_fps = target_fps
        # V3-VIZ.3: per-camera per-frame c2w lookup, used by Camera dropdown +
        # Follow Camera. Shape: {cam_id: {"c2w": (F, 4, 4) float32,
        # "timestamps_us": (F,) int64}}. None when launched without
        # --dataset_path → dropdown contains only the primary camera.
        self._multi_cam_poses: dict = multi_cam_poses or {}
        # E2.7 (H1 fix): lock initial viser client camera to this NCore camera
        # id when present in multi_cam_poses. Without this, viser's default
        # camera lands in far-field background gaussian noise on outdoor
        # driving scenes, producing the "unrecognizable artifacts" symptom that
        # killed the amazing-lalande session's visual validation.
        self._initial_cam_id: Optional[str] = initial_cam_id
        self._follow_camera_enabled: bool = False
        self._cam_dropdown = None
        self._show_follow_cam = None
        self._current_dropdown_cam: Optional[str] = None
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
        # E2.7-OPCV: OpenCVPinhole rational distortion path — same T8.13 fix
        # as FTheta. Active when ckpt camera is OpenCVPinhole and
        # --initial_cam_id resolves to a cam with opencv_pinhole_dict in
        # multi_cam_poses. Without this, kaolin's ideal-pinhole raygen
        # samples a different angular cone than what the Gaussians learned
        # under rational distortion → 渲染采样到空气 → 大片黑。
        self.opencv_pinhole_intrinsics: Optional[dict] = None
        self.opencv_pinhole_rays: Optional[np.ndarray] = None  # (H, W, 3)
        self.opencv_pinhole_render_wh: Optional[tuple] = None
        if (self.ftheta_intrinsics is None
                and initial_cam_id is not None
                and initial_cam_id in self._multi_cam_poses):
            entry = self._multi_cam_poses[initial_cam_id]
            opcv = entry.get("opencv_pinhole_dict")
            opcv_rays = entry.get("opencv_pinhole_rays")
            if opcv is not None and opcv_rays is not None:
                self.opencv_pinhole_intrinsics = opcv
                self.opencv_pinhole_rays = opcv_rays
                self.opencv_pinhole_render_wh = entry.get("resolution")
                print(f"[viz_4d] OpenCVPinhole rational ray path active for "
                      f"cam '{initial_cam_id}', W×H={self.opencv_pinhole_render_wh}",
                      flush=True)
        # B2: FTheta cuboid overlay path — projects cuboid/track/ego_traj
        # polylines through the same FTheta polynomial used by the backdrop
        # and alpha-blends them into the rendered image before
        # set_background_image. None for pinhole ckpts. Calibration constants
        # pinned by docs/T8_artifacts/B2_calibration_probe_log.md.
        self._overlay_compositor: Optional[Viser4DOverlayCompositor] = None
        # B2 perf: ego trajectory + per-track trajectories are static (depend
        # only on meta, not on t_us or c2w), so we build their world-space
        # polyline lists exactly once and reuse them across every frame.
        # Active cuboids still need per-frame rebuild (active set + poses
        # depend on t_us). BUG-1c: track entries carry the class name so the
        # overlay can keep the per-class colors of the 3D primitive path.
        self._overlay_static_ego_polylines: list[np.ndarray] = []
        self._overlay_static_track_polylines: list[tuple[str, np.ndarray]] = []
        if (self.ftheta_intrinsics is not None
                and self.ftheta_render_wh is not None):
            W_ft, H_ft = self.ftheta_render_wh
            # BUG-1 (2026-06-10): flip=identity, NOT the legacy
            # FLIP_VISER_TO_OPENCV. The c2w fed to composite() is the same
            # matrix the backdrop is rendered with, and the backdrop's
            # viewing direction is that matrix's +Z column (FTheta rays have
            # +Z forward). The legacy Z-flip aimed the overlay 180° away —
            # wireframes were mirror-images of the tracks BEHIND the ego,
            # which only looked plausible because streets are fore-aft
            # symmetric. See Viser4DOverlayCompositor.__init__ docstring.
            self._overlay_compositor = Viser4DOverlayCompositor(
                ftheta_dict=self.ftheta_intrinsics,
                height=H_ft, width=W_ft, subdivide_n=20,
                world_to_camera_flip=np.eye(4),
            )
            if metadata is not None:
                if metadata.ego_poses_c2w.size > 0:
                    ego_pts = metadata.ego_poses_c2w[:, :3, 3].astype(np.float64)
                    if ego_pts.shape[0] >= 2:
                        self._overlay_static_ego_polylines = [ego_pts]
                for tid, t in metadata.tracks.items():
                    mask = t["frame_info"]
                    poses = t["poses"]
                    if mask is None or poses is None or mask.sum() < 2:
                        continue
                    centers = poses[mask, :3, 3].astype(np.float64)
                    if centers.shape[0] >= 2:
                        self._overlay_static_track_polylines.append(
                            (str(t.get("class", "unknown")), centers))
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
        # V3-VIZ.2: per-cuboid 3D billboard label, keyed by track_id.
        self._cuboid_label_handles: dict[str, object] = {}

        # Render dirtiness — set by camera move, slider drag, visibility toggle.
        self.need_update = True

        # DiFix novel-view post-process (optional). When --difix_server is set,
        # rendered frames are shipped to an out-of-process DiFix server (cosmos
        # Docker env) and the corrected frame is sent to the browser instead.
        # The toggle defaults OFF so DiFix costs nothing until the user opts in.
        # See difix_server.py / utils/difix_client.py. The connection is lazy —
        # nothing is contacted until the first frame is actually fixed.
        self.difix_client: Optional[DifixClient] = (
            DifixClient.from_addr(difix_server) if difix_server else None
        )
        self.difix_enabled: bool = False
        self.difix_rtt = None  # viser text handle, populated in _build_static_gui

        # E2.6: temporal DiffusionHarmonizer post-process (optional). Sister to
        # DiFix above but carries Harmonizer's temporal mode — the client holds
        # a K-frame self-reference deque so the model de-flickers a continuous
        # play sequence. Mutually exclusive with --difix_server (enforced in
        # main()). The history deque lives on the client; a non-play timeline
        # change (seek/scrub/loop-wrap) sets _postproc_reset_flag so the next
        # frame is sent cold (V=1), preventing stale history leaking across a
        # discontinuous jump.
        self.harmonizer_temporal_client: Optional[HarmonizerTemporalClient] = (
            HarmonizerTemporalClient.from_addr(
                harmonizer_temporal_server, K=harmonizer_temporal_K
            ) if harmonizer_temporal_server else None
        )
        self._postproc_reset_flag: bool = True  # cold-start the first frame
        # Track the last-rendered resolution so a slider change forces a temporal
        # history reset (history frames must match curr's HxW).
        self._postproc_last_wh: Optional[tuple] = None

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
            # E2.7 (H1): if --initial_cam_id given and present in
            # multi_cam_poses, snap to that camera's c2w at the current
            # timeline timestamp (default = t_us_first). Falls back to
            # meta.initial_c2w (legacy behavior) when no cam_id given or not
            # in the cam ring.
            snapped = False
            if (self._initial_cam_id
                    and self._initial_cam_id in self._multi_cam_poses):
                entry = self._multi_cam_poses[self._initial_cam_id]
                c2w_arr = entry["c2w"]
                ts_arr = entry["timestamps_us"]
                if c2w_arr.shape[0] > 0:
                    if ts_arr.size > 0:
                        idx = int(np.searchsorted(ts_arr, self._t_us_current))
                        idx = max(0, min(idx, ts_arr.size - 1))
                    else:
                        idx = 0
                    c2w0 = c2w_arr[idx]
                    R = c2w0[:3, :3]
                    forward = R @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
                    up = R @ np.array([0.0, -1.0, 0.0], dtype=np.float32)
                    client.camera.wxyz = mat_to_wxyz(c2w0)
                    client.camera.position = c2w0[:3, 3]
                    client.camera.look_at = c2w0[:3, 3] + 10.0 * forward
                    client.camera.up_direction = up
                    snapped = True
                    print(f"[E2.7] client snapped to initial_cam_id="
                          f"'{self._initial_cam_id}' (frame_idx={idx})",
                          flush=True)
            if not snapped and self.meta is not None:
                client.camera.wxyz = mat_to_wxyz(self.meta.initial_c2w)
                client.camera.position = self.meta.initial_c2w[:3, 3]
            if self.initial_fov_rad is not None:
                client.camera.fov = float(self.initial_fov_rad)
            @client.camera.on_update
            def _(_, _client=client):
                self.need_update = True
                self._sync_pose_gui_from_camera(_client.camera)
            # Initialise the pose controls to this client's starting camera.
            self._sync_pose_gui_from_camera(client.camera)

        # E2.7 (H1): swap engine FTheta + render WH to the initial cam's
        # intrinsics once at startup, so the very first rendered backdrop
        # uses the selected camera's projection. _snap_clients_to_camera
        # also iterates connected clients (none yet at this point — safe
        # no-op for client part; the FTheta swap is what we want here).
        if (self._initial_cam_id
                and self._initial_cam_id in self._multi_cam_poses):
            self._snap_clients_to_camera(
                self._initial_cam_id, self._t_us_current)
            self._current_dropdown_cam = self._initial_cam_id
            print(f"[E2.7] engine FTheta + dropdown locked to "
                  f"'{self._initial_cam_id}'", flush=True)
        elif self._initial_cam_id:
            print(f"[E2.7-WARN] initial_cam_id='{self._initial_cam_id}' "
                  f"not in multi_cam_poses "
                  f"(have {list(self._multi_cam_poses.keys())[:5]}...); "
                  f"viser will use meta.initial_c2w default — far-field "
                  f"camera may produce 'unrecognizable artifacts'", flush=True)

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
    def _sync_pose_gui_from_camera(self, camera) -> None:
        """Push the current camera pose into the yaw/pitch/roll + X/Y/Z controls
        (camera → GUI). Guarded by ``_pose_syncing`` so the value-set echoes
        don't drive the camera back. No-op before the controls exist."""
        if not hasattr(self, "yaw_slider"):
            return
        c2w = get_c2w(camera)
        yaw, pitch, roll = cam_rot_to_ypr(c2w[:3, :3])
        px, py, pz = (float(v) for v in c2w[:3, 3])
        self._pose_syncing = True
        try:
            self.yaw_slider.value = float(yaw)
            self.pitch_slider.value = float(pitch)
            self.roll_slider.value = float(roll)
            self.cam_x.value = px
            self.cam_y.value = py
            self.cam_z.value = pz
        finally:
            self._pose_syncing = False

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
                # ---- Camera pose: world-referenced yaw/pitch/roll + position ----
                # Absolute readout AND control, two-way synced with mouse orbit.
                # yaw = heading about world up (+z), pitch = elevation
                # (+ looks up / − looks down), roll about the view axis; X/Y/Z is
                # the camera position in world. Drag a control → the camera jumps
                # there; orbit with the mouse → these values follow. Lets you read
                # off e.g. "cam z = −2 m, pitch −30°" to place the view precisely
                # above / below the road.
                self._pose_syncing = False
                self.yaw_slider = self.server.gui.add_slider(
                    "Yaw °", min=-180.0, max=180.0, step=0.5, initial_value=0.0)
                self.pitch_slider = self.server.gui.add_slider(
                    "Pitch °", min=-90.0, max=90.0, step=0.5, initial_value=0.0)
                self.roll_slider = self.server.gui.add_slider(
                    "Roll °", min=-180.0, max=180.0, step=0.5, initial_value=0.0)
                self.cam_x = self.server.gui.add_number("Cam X", initial_value=0.0, step=0.1)
                self.cam_y = self.server.gui.add_number("Cam Y", initial_value=0.0, step=0.1)
                self.cam_z = self.server.gui.add_number("Cam Z", initial_value=0.0, step=0.1)

                def _apply_pose_from_gui(_self=self):
                    if _self._pose_syncing:      # ignore echoes from camera→GUI sync
                        return
                    R = ypr_to_cam_rot(_self.yaw_slider.value,
                                       _self.pitch_slider.value,
                                       _self.roll_slider.value)
                    pos = np.array([_self.cam_x.value, _self.cam_y.value,
                                    _self.cam_z.value], dtype=np.float32)
                    c2w = np.eye(4, dtype=np.float32)
                    c2w[:3, :3] = R.astype(np.float32)
                    c2w[:3, 3] = pos
                    for client in _self.server.get_clients().values():
                        client.camera.wxyz = mat_to_wxyz(c2w)
                        client.camera.position = pos
                    _self.need_update = True

                for _w in (self.yaw_slider, self.pitch_slider, self.roll_slider,
                           self.cam_x, self.cam_y, self.cam_z):
                    @_w.on_update
                    def _(_, _self=self):
                        _apply_pose_from_gui(_self)
                # DiFix novel-view fix toggle — only when --difix_server was
                # given. Enabling routes each rendered frame through the
                # out-of-process DiFix server before display (~9–11 FPS).
                if self.difix_client is not None:
                    self.difix_checkbox = self.server.gui.add_checkbox(
                        "DiFix (novel-view fix)", initial_value=False)
                    self.difix_rtt = self.server.gui.add_text(
                        "DiFix RTT", initial_value="-", disabled=True)
                    self.server.gui.add_markdown(
                        "_DiFix 启用后约 9–11 FPS，适合静止精修；"
                        "连续拖动建议关闭。_")

                    @self.difix_checkbox.on_update
                    def _(_, _self=self):
                        _self.difix_enabled = bool(_self.difix_checkbox.value)
                        _self.need_update = True
                        print(f"[DiFix] enabled={_self.difix_enabled}",
                              flush=True)
                # E2.6: temporal Harmonizer toggle — only when
                # --harmonizer_temporal_server was given. Mutually exclusive
                # with DiFix (main() rejects both flags). On enable, reset the
                # temporal history so the model starts from a clean cold frame.
                if self.harmonizer_temporal_client is not None:
                    self.harmonizer_checkbox = self.server.gui.add_checkbox(
                        "Harmonizer (temporal, de-flicker)",
                        initial_value=False)
                    self.difix_rtt = self.server.gui.add_text(
                        "Harmonizer RTT", initial_value="-", disabled=True)
                    self.server.gui.add_markdown(
                        "_Harmonizer temporal 模式：回读前 K 帧已修复输出做"
                        "时序参考，连续 Play 去闪烁。Seek/拖动会自动重置历史。_")

                    @self.harmonizer_checkbox.on_update
                    def _(_, _self=self):
                        _self.difix_enabled = bool(
                            _self.harmonizer_checkbox.value)
                        if _self.difix_enabled:
                            _self._postproc_reset_flag = True
                        _self.need_update = True
                        print(f"[Harmonizer-temporal] enabled="
                              f"{_self.difix_enabled}", flush=True)
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
                        # Bug 3 runtime instrumentation: print the toggle
                        # event + post-write enabled_layer_names so the log
                        # gives ground-truth evidence about whether the
                        # callback actually fired and whether the wholesale
                        # set replacement reached the server-side mog.
                        # Confirms or refutes "GUI state desync" hypothesis.
                        print(f"[BUG3-DIAG] toggle layer='{_name}' "
                              f"value={bool(_cb.value)} → enabled_layer_names="
                              f"{sorted(new_set)}", flush=True)

        if self.engine is not None:
            with folder:
                initial_renderer = getattr(self.engine, "_requested_renderer", "3dgrt")
                self._gui_renderer = self.server.gui.add_dropdown(
                    "Renderer",
                    options=["3dgrt", "3dgut"],
                    initial_value=initial_renderer,
                )

                @self._gui_renderer.on_update
                def _(_ev, _self=self):
                    if _self.engine is not None:
                        try:
                            _self.engine.set_renderer(_self._gui_renderer.value)
                            print(f"[viz_4d] renderer → {_self._gui_renderer.value}", flush=True)
                        except Exception as exc:
                            print(f"[viz_4d] renderer switch failed: {exc}", flush=True)
                    _self.need_update = True

        @self.reset_view_button.on_click
        def _(_):
            self.need_update = True
            # E2.7 P3 fix: if --initial_cam_id was given and resolves into
            # multi_cam_poses, Reset View snaps back to THAT camera at the
            # current timeline timestamp — matching the connect-time snap
            # behavior. Without this, Reset would jump to meta.initial_c2w
            # (typically alphabetical first camera, e.g. camera_cross_left)
            # which surprised the user. Falls back to legacy meta.initial_c2w
            # behavior when no initial_cam_id given or not in cam ring.
            snapped = False
            if (self._initial_cam_id
                    and self._initial_cam_id in self._multi_cam_poses):
                entry = self._multi_cam_poses[self._initial_cam_id]
                c2w_arr = entry["c2w"]
                ts_arr = entry["timestamps_us"]
                if c2w_arr.shape[0] > 0:
                    if ts_arr.size > 0:
                        idx = int(np.searchsorted(ts_arr, self._t_us_current))
                        idx = max(0, min(idx, ts_arr.size - 1))
                    else:
                        idx = 0
                    c2w0 = c2w_arr[idx]
                    R = c2w0[:3, :3]
                    forward_world = R @ np.array([0.0, 0.0, 1.0],
                                                 dtype=np.float32)
                    up_world = R @ np.array([0.0, -1.0, 0.0],
                                            dtype=np.float32)
                    for client in self.server.get_clients().values():
                        client.camera.wxyz = mat_to_wxyz(c2w0)
                        client.camera.position = c2w0[:3, 3]
                        client.camera.look_at = c2w0[:3, 3] + 10.0 * forward_world
                        client.camera.up_direction = up_world
                        if self.initial_fov_rad is not None:
                            client.camera.fov = float(self.initial_fov_rad)
                    snapped = True
            if snapped:
                return
            # Legacy fallback: snap to meta.initial_c2w (the alphabetical
            # first camera in 5/7-cam ring, kept for backwards compat).
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
            # V3-VIZ.3: multi-camera dropdown + Follow Camera. Populated from
            # NCore dataset via --dataset_path; degrades to {primary} when not
            # available. Mutually exclusive with Follow Ego (only one snap
            # source active at a time).
            cam_options = (sorted(self._multi_cam_poses.keys())
                           if self._multi_cam_poses else [
                               self.meta.ego_primary_camera_id])
            initial_cam = (self.meta.ego_primary_camera_id
                           if self.meta.ego_primary_camera_id in cam_options
                           else cam_options[0])
            self._current_dropdown_cam = initial_cam
            self._cam_dropdown = self.server.gui.add_dropdown(
                "Camera", options=tuple(cam_options), initial_value=initial_cam,
            )
            self._show_follow_cam = self.server.gui.add_checkbox(
                "Follow Camera", False,
            )
            # BUG-1c (2026-06-10): the "FTheta overlay (debug)" toggle is
            # GONE. In FTheta mode the overlay is the ONLY correct projection
            # path (BUG-1/1b/1c verified); the legacy pinhole line_segments
            # path it used to re-enable is misaligned by construction, so
            # exposing it only confused users (user feedback 2026-06-10).
            # The content checkboxes above (Ego trajectory / Track
            # trajectories / Active cuboids) gate the overlay layers instead.

        @self.show_ego_traj.on_update
        def _(_):
            if self.h_ego_traj is not None:
                self.h_ego_traj.visible = bool(self.show_ego_traj.value)
            # BUG-1c: overlay-path trajectory is baked into the backdrop.
            self.need_update = True

        @self.show_ego_frust.on_update
        def _(_):
            if self.h_ego_frustum is not None:
                self.h_ego_frustum.visible = bool(self.show_ego_frust.value)

        @self.show_tracks.on_update
        def _(_):
            if self.h_track_trajectories is not None:
                self.h_track_trajectories.visible = bool(self.show_tracks.value)
            # BUG-1c: overlay-path trajectories are baked into the backdrop.
            self.need_update = True

        @self.show_cuboids.on_update
        def _(_):
            v = bool(self.show_cuboids.value)
            if self.h_cuboid_lines is not None:
                self.h_cuboid_lines.visible = v
            # V3-VIZ.2: labels follow the same Active-cuboids toggle.
            for h in self._cuboid_label_handles.values():
                try:
                    h.visible = v
                except Exception:
                    pass
            # BUG-1: overlay-path cuboids are baked into the backdrop image,
            # so a visibility change only shows up after a re-render.
            self.need_update = True

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
                # Mutual exclusion with Follow Camera.
                if self._show_follow_cam is not None and self._show_follow_cam.value:
                    self._show_follow_cam.value = False
                    self._follow_camera_enabled = False
                self._snap_clients_to_ego(self._t_us_current)
                self.need_update = True

        # V3-VIZ.3: dropdown snap (one-shot) on selection change.
        @self._cam_dropdown.on_update
        def _(_):
            self._current_dropdown_cam = str(self._cam_dropdown.value)
            self._snap_clients_to_camera(
                self._current_dropdown_cam, self._t_us_current)
            self.need_update = True

        # V3-VIZ.3: Follow Camera checkbox — snap every timeline tick to the
        # selected camera's current-frame c2w. Mutually exclusive with Follow Ego.
        @self._show_follow_cam.on_update
        def _(_):
            self._follow_camera_enabled = bool(self._show_follow_cam.value)
            if self._follow_camera_enabled:
                if self.show_follow_ego.value:
                    self.show_follow_ego.value = False
                    self._follow_ego_enabled = False
                self._snap_clients_to_camera(
                    self._current_dropdown_cam, self._t_us_current)
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
        # BUG-1c (2026-06-10): in FTheta mode the trajectory polyline rides
        # the overlay path (_collect_overlay_layer_specs) so it shares the
        # backdrop's fisheye projection — the 3D line_segments primitive was
        # pinhole-projected and drifted off the backdrop like the pre-fix
        # cuboids. The ego frustum below is still a 3D primitive: it's a
        # position widget in space, not a backdrop annotation.
        # Pinhole ckpts keep the straight-chord line_segments (V3-VIZ.5
        # replaced spline_catmull_rom whose control-point spacing overshot at
        # the trajectory's bimodal 33/66 ms dt cadence).
        if pts.shape[0] >= 2 and self._overlay_compositor is None:
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            cols = np.broadcast_to(
                np.array([0.2, 1.0, 0.2], dtype=np.float32),
                (segs.shape[0], 2, 3),
            ).copy()
            self.h_ego_traj = self.server.scene.add_line_segments(
                "/ego/trajectory",
                points=segs,
                colors=cols,
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
        # Carry forward user's visibility toggle when re-adding. Bug B8 fix:
        # the previous default of True for the first call (h_dyn_pts is None,
        # which fires during __init__'s _on_time_change(t_us_first)) added the
        # cloud as visible regardless of the checkbox initial state (False).
        # User saw LiDAR points on startup even though "Dynamic LiDAR" was
        # unchecked. Read the checkbox value on first call so initial state
        # matches the GUI.
        prev_visible = (self.h_dyn_pts.visible
                        if self.h_dyn_pts is not None
                        else bool(getattr(self, "show_dyn_pts", None)
                                  and self.show_dyn_pts.value))
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
        """All tracks' active-frame polylines as one batched line_segments.

        BUG-1c (2026-06-10): FTheta mode skips the 3D primitive entirely —
        trajectories are drawn by the overlay path with the same per-class
        colors (see _collect_overlay_layer_specs). The "scattered pixels
        across the image" that V3-VIZ.5 blamed on the overlay's behind-camera
        clipping was actually the legacy 180° FLIP projecting behind-ego
        segments forward; fixed by flip=identity.
        """
        assert self.meta is not None
        if self._overlay_compositor is not None:
            return
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
        # E2.6: any non-play timeline change (seek / scrub / loop-wrap / init)
        # is a discontinuous jump — flag the temporal post-proc to reset its
        # history on the next frame so stale frames don't leak across the jump.
        # "play" is the only continuous-sequence source (advances by dt_us).
        if source != "play":
            self._postproc_reset_flag = True
        # Update ego frustum + cuboid + UI mirrors.
        self._update_ego_frustum(t_us)
        frame_idx = self.meta.lookup_frame_idx(t_us)
        self._update_active_cuboids(frame_idx)
        self._update_dynamic_lidar(frame_idx)
        self._mirror_ui(t_us, frame_idx)
        # V3-VIZ.3: if Follow Camera is on, snap viewer to selected camera's
        # c2w at the new time. Checked before Follow Ego so a Camera-mode user
        # doesn't get bumped to ego by the ego branch below.
        if self._follow_camera_enabled and self._current_dropdown_cam:
            self._snap_clients_to_camera(self._current_dropdown_cam, t_us)
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
        # E2.7 P4 fix: when --initial_cam_id is given and present in
        # multi_cam_poses, use THAT camera's c2w as the ego-frustum pose.
        # Default NCore meta.ego_pose_at returns the primary camera which is
        # the alphabetical first sensor (camera_cross_left_120fov) — its
        # mount points down-left (A-pillar blindspot coverage), making the
        # frustum visually "point at the ground" instead of "point forward".
        # The user expects a dashcam-style forward frustum.
        pose = None
        if (self._initial_cam_id
                and self._initial_cam_id in self._multi_cam_poses):
            entry = self._multi_cam_poses[self._initial_cam_id]
            c2w_arr = entry["c2w"]
            ts_arr = entry["timestamps_us"]
            if c2w_arr.shape[0] > 0 and ts_arr.size > 0:
                idx = int(np.searchsorted(ts_arr, int(t_us)))
                idx = max(0, min(idx, ts_arr.size - 1))
                pose = c2w_arr[idx]
        if pose is None:
            pose = self.meta.ego_pose_at(t_us)
        self.h_ego_frustum.wxyz = mat_to_wxyz(pose)
        self.h_ego_frustum.position = pose[:3, 3]

    def _snap_clients_to_camera(self, cam_id: Optional[str], t_us: int) -> None:
        """V3-VIZ.3: snap viewer to ``cam_id``'s c2w at the nearest frame ≤ t_us.

        Also swaps the engine's FTheta intrinsics + render WH so the Gaussian
        backdrop uses the SELECTED camera's polynomial — otherwise the engine
        keeps using primary camera's intrinsics and front_wide vs front_tele
        renders identically (since browser pinhole approx ignores FTheta).
        """
        if cam_id is None or cam_id not in self._multi_cam_poses:
            return
        entry = self._multi_cam_poses[cam_id]
        c2w_arr = entry["c2w"]
        ts_arr = entry["timestamps_us"]
        if ts_arr.size == 0:
            return
        idx = int(np.searchsorted(ts_arr, int(t_us)))
        idx = max(0, min(idx, ts_arr.size - 1))
        if idx > 0 and abs(int(ts_arr[idx - 1]) - t_us) < abs(int(ts_arr[idx]) - t_us):
            idx -= 1
        pose = c2w_arr[idx]
        R = pose[:3, :3]
        forward_world = R @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        up_world = R @ np.array([0.0, -1.0, 0.0], dtype=np.float32)
        # Swap engine intrinsics so backdrop matches this camera's projection.
        new_ftheta = entry.get("ftheta_dict")
        new_res = entry.get("resolution")
        new_fov = float(entry.get("fov_y_rad", 1.5708))
        if new_ftheta is not None and new_res is not None:
            self.ftheta_intrinsics = new_ftheta
            self.ftheta_render_wh = new_res
            # Rebuild compositor at the new render resolution so any overlay
            # path that still uses it (ego/track trajectories) stays in sync.
            if self._overlay_compositor is not None:
                from threedgrut_playground.utils.viser_overlay_compositor import (
                    Viser4DOverlayCompositor,
                )
                # BUG-1: flip=identity to stay co-aligned with the backdrop
                # camera (+Z forward) — see __init__ compositor comment.
                self._overlay_compositor = Viser4DOverlayCompositor(
                    ftheta_dict=new_ftheta,
                    height=new_res[1], width=new_res[0], subdivide_n=20,
                    world_to_camera_flip=np.eye(4),
                )
        for client in self.server.get_clients().values():
            client.camera.wxyz = mat_to_wxyz(pose)
            client.camera.position = pose[:3, 3]
            client.camera.look_at = pose[:3, 3] + 10.0 * forward_world
            client.camera.up_direction = up_world
            client.camera.fov = new_fov
        self.need_update = True

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

    @staticmethod
    def _cuboid_label_anchor(pose: np.ndarray, size) -> np.ndarray:
        """World-space label anchor: cuboid top-back-right corner (vertex 7 =
        (+sx/2, +sy/2, +sz/2)). Single source of truth shared by the 3D-label
        path (_update_cuboid_labels) and the overlay text path
        (_collect_overlay_layer_specs) — BUG-1b keeps both anchored alike.
        """
        sx = float(size[0]) if size is not None and size.size >= 1 else 1.0
        sy = float(size[1]) if size is not None and size.size >= 2 else 1.0
        sz = float(size[2]) if size is not None and size.size >= 3 else 1.0
        local = np.array([sx * 0.5, sy * 0.5, sz * 0.5], dtype=np.float32)
        return (pose[:3, :3] @ local + pose[:3, 3]).astype(np.float64)

    # E2.8 (大g 2026-06-17): active-cuboid display 只留 vehicle 类——person /
    # animal / protruding_object 等 cuboid 是杂物（无 rigid gaussian，纯遮挡
    # 视觉）。子串匹配覆盖 heavy_truck 等复合类。改这一处 + _update_cuboid_labels
    # 即覆盖 line_segments + FTheta overlay + 两条 label 路径（共用 _iter）。
    _VEHICLE_CUBOID_TOKENS = (
        "automobile", "bus", "truck", "consumer_vehicles", "car", "vehicle",
    )

    def _is_vehicle_cuboid(self, tid) -> bool:
        if self.meta is None:
            return True
        cls = str(self.meta.tracks.get(tid, {}).get("class", "")).lower()
        return any(tok in cls for tok in self._VEHICLE_CUBOID_TOKENS)

    def _iter_active_cuboid_edges(self, frame_idx: int):
        """Yield ``(track_id, (12, 2, 3) world edges)`` per active track.

        Single source of truth for the active-set / pose-bounds / size
        fallback logic shared by the line_segments path
        (_build_cuboid_edges) and the FTheta overlay path
        (_collect_overlay_layer_specs) — BUG-1 fix keeps both paths from
        drifting apart again. E2.8: non-vehicle cuboids filtered out here.
        """
        assert self.meta is not None
        for tid in self.meta.active_tracks_at(frame_idx):
            if not self._is_vehicle_cuboid(tid):
                continue
            t = self.meta.tracks[tid]
            poses = t["poses"]
            if poses is None or frame_idx >= poses.shape[0]:
                continue
            size = (t["size"] if t["size"] is not None
                    else np.array([1.0, 1.0, 1.0], dtype=np.float32))
            yield tid, cuboid_world_edges(poses[frame_idx], size)

    def _build_cuboid_edges(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        assert self.meta is not None
        pts_list, col_list = [], []
        for tid, world in self._iter_active_cuboid_edges(frame_idx):
            color = np.array(instance_color(tid), dtype=np.float32)
            col = np.broadcast_to(color, (12, 2, 3)).astype(np.float32).copy()
            pts_list.append(world)
            col_list.append(col)
        if not pts_list:
            return (np.zeros((0, 2, 3), dtype=np.float32),
                    np.zeros((0, 2, 3), dtype=np.float32))
        return (np.concatenate(pts_list, axis=0),
                np.concatenate(col_list, axis=0))

    def _update_active_cuboids(self, frame_idx: int) -> None:
        """Remove + re-add line_segments handle for active cuboids.

        viser 1.0 line-segment handle exposes ``.visible/.position/.wxyz`` but
        not in-place vertex updates → we replace the node each frame.

        Bug B7 fix: each Play tick this method removes + re-adds the node,
        which defaults to ``visible=True``. Previously the user could uncheck
        "Active cuboids" in Visibility, then on the very next Play tick the
        cuboids reappeared. We preserve the prior handle's visibility (or
        fall back to the checkbox value on the first frame) and re-apply it.
        Mirrors _update_dynamic_lidar's preserve-prev-visible pattern.
        """
        # BUG-1 fix (2026-06-10, reverts the V3-VIZ.2 trade-off): when the
        # FTheta overlay is live, cuboid edges are alpha-blended into the
        # backdrop through the SAME fisheye polynomial as the Gaussians
        # (_collect_overlay_layer_specs → Viser4DOverlayCompositor), so we
        # must NOT also add a browser-side line_segments node — its pinhole
        # projection visibly detaches from the FTheta backdrop away from the
        # image center (NCore wide-FoV: tens of pixels, not "a few"), which
        # was exactly the user-reported misalignment. Labels stay as 3D text
        # (annotations, not alignment targets; viser has no image-space text
        # path) and may drift near the periphery — acceptable.
        # With the overlay checkbox OFF the legacy line_segments path below
        # remains available for A/B comparison.
        # BUG-1c: the overlay is unconditionally live in FTheta mode (the
        # legacy "FTheta overlay (debug)" escape hatch back to the misaligned
        # pinhole path was removed).
        overlay_live = self._overlay_compositor is not None
        if overlay_live:
            if self.h_cuboid_lines is not None:
                self.h_cuboid_lines.remove()
                self.h_cuboid_lines = None
            # BUG-1b: labels are baked into the backdrop by the overlay text
            # path (_collect_overlay_layer_specs labels_world), pixel-aligned
            # with the wireframe. Remove the browser-side 3D labels so a
            # pinhole-projected duplicate doesn't drift off the aligned box.
            for tid in list(self._cuboid_label_handles.keys()):
                try:
                    self._cuboid_label_handles[tid].remove()
                except Exception:
                    pass
                self._cuboid_label_handles.pop(tid, None)
            return
        prev_visible = (self.h_cuboid_lines.visible
                        if self.h_cuboid_lines is not None
                        else bool(getattr(self, "show_cuboids", None)
                                  and self.show_cuboids.value))
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
        self.h_cuboid_lines.visible = prev_visible
        # V3-VIZ.2: per-cuboid "tID | class" billboard label, top-center.
        self._update_cuboid_labels(frame_idx, visible=prev_visible)

    def _update_cuboid_labels(self, frame_idx: int, *, visible: bool) -> None:
        """V3-VIZ.2: add/remove ``/labels/cuboid_<tid>`` 3D text labels.

        Each label shows ``"t<tid> | <class>"`` positioned ~0.3 m above the
        cuboid's top face. Labels follow the same Active-cuboids visibility
        toggle as the line wireframe. In FTheta mode line_segments are
        skipped (overlay path), so this method is called with visible=False
        from that early-return branch.
        """
        if self.meta is None:
            return
        # E2.8: vehicle-only display (matches _iter_active_cuboid_edges filter).
        active_ids = {tid for tid in self.meta.active_tracks_at(frame_idx)
                      if self._is_vehicle_cuboid(tid)}
        # Remove labels for tracks no longer active.
        for tid in list(self._cuboid_label_handles.keys()):
            if tid not in active_ids:
                try:
                    self._cuboid_label_handles[tid].remove()
                except Exception:
                    pass
                self._cuboid_label_handles.pop(tid, None)
        if not active_ids:
            return
        for tid in active_ids:
            t = self.meta.tracks[tid]
            poses = t.get("poses")
            size = t.get("size")
            if poses is None or frame_idx >= poses.shape[0]:
                continue
            pose = poses[frame_idx]
            # V3-VIZ.2 follow-up: anchor label on the cuboid top-back-right
            # corner (vertex 7) so the text touches the wireframe instead of
            # floating above (user feedback 2026-05-26). Anchor math shared
            # with the overlay text path via _cuboid_label_anchor (BUG-1b).
            top = self._cuboid_label_anchor(pose, size).astype(np.float32)
            text = f"t{tid} | {t.get('class', 'unknown')}"
            old = self._cuboid_label_handles.get(tid)
            if old is not None:
                try:
                    old.remove()
                except Exception:
                    pass
            try:
                handle = self.server.scene.add_label(
                    f"/labels/cuboid_{tid}",
                    text=text,
                    position=(float(top[0]), float(top[1]), float(top[2])),
                    wxyz=(1.0, 0.0, 0.0, 0.0),
                )
                handle.visible = bool(visible)
                self._cuboid_label_handles[tid] = handle
            except Exception:
                # Older viser without add_label: silently skip; cuboid edges
                # still render correctly.
                pass

    # ---------------------------------------------------------------- B2 overlay
    def _collect_overlay_layer_specs(self, t_us: int) -> list[PolylineLayerSpec]:
        """Build world-space polyline specs for the FTheta overlay path.

        Layers are ordered bottom→top: ego_trajectory → tracks → cuboids
        (cuboids on top because they are the P1 visual verification target).
        Each layer respects the corresponding ``show_*`` Visibility checkbox,
        so the user can hide individual primitives even with overlay on.

        Returns ``[]`` when metadata absent or all layers are hidden / empty.
        """
        if self.meta is None:
            return []
        specs: list[PolylineLayerSpec] = []

        # BUG-1c (2026-06-10, reverts V3-VIZ.5): ego + track trajectories are
        # BACK on the overlay path — as 3D primitives they were pinhole-
        # projected and drifted off the FTheta backdrop exactly like the
        # pre-fix cuboids. V3-VIZ.5's removal rationale ("behind-camera
        # segment-clipping bug scattered pixels") was a symptom of the legacy
        # 180° FLIP: real-forward segments got flipped behind the camera and
        # vice versa; with flip=identity the projector's z>0 / max_angle
        # culling clips them correctly. Static world-space polylines are
        # cached in __init__ (B2 perf); low subdivide_n=3 because the
        # polylines are already dense (per-frame vertices).
        if (getattr(self, "show_ego_traj", None) is not None
                and bool(self.show_ego_traj.value)
                and self._overlay_static_ego_polylines):
            specs.append(PolylineLayerSpec(
                name="ego_trajectory",
                polylines_world=self._overlay_static_ego_polylines,
                color=(51, 255, 51, 220),
                width=2,
                subdivide_n=3,
            ))

        if (getattr(self, "show_tracks", None) is not None
                and bool(self.show_tracks.value)
                and self._overlay_static_track_polylines):
            # One layer per class so the overlay keeps the 3D path's
            # class_color coding (automobile blue / heavy_truck orange / ...).
            by_class: dict[str, list[np.ndarray]] = {}
            for cls_name, centers in self._overlay_static_track_polylines:
                by_class.setdefault(cls_name, []).append(centers)
            for cls_name, polylines in by_class.items():
                rgb = tuple(int(round(c * 255)) for c in class_color(cls_name))
                specs.append(PolylineLayerSpec(
                    name=f"track_trajectories_{cls_name}",
                    polylines_world=polylines,
                    color=(rgb[0], rgb[1], rgb[2], 180),
                    width=1,
                    subdivide_n=3,
                ))

        # BUG-1 fix (2026-06-10): active cuboids are BACK on the overlay path
        # (reverting that part of V3-VIZ.2). The wireframe's whole job is to
        # hug the dynamic Gaussian actor rendered through the FTheta
        # polynomial; browser-side pinhole line_segments visibly detach from
        # it away from the image center. One layer per track keeps the
        # per-instance color the 3D path used. subdivide_n=20 because a
        # straight 3D edge projects to a curve under fisheye.
        if (getattr(self, "show_cuboids", None) is not None
                and bool(self.show_cuboids.value)):
            frame_idx = self.meta.lookup_frame_idx(t_us)
            for tid, edges in self._iter_active_cuboid_edges(frame_idx):
                rgb = tuple(int(round(c * 255)) for c in instance_color(tid))
                # BUG-1b: the "t<tid> | <class>" label rides the overlay too
                # (same anchor as the legacy 3D label = cuboid top corner) so
                # text and wireframe share the FTheta projection and can
                # never separate — the browser-side pinhole 3D label drifted
                # off the now-aligned box exactly like the old line_segments.
                t = self.meta.tracks[tid]
                anchor = self._cuboid_label_anchor(
                    t["poses"][frame_idx], t["size"])
                label = f"t{tid} | {t.get('class', 'unknown')}"
                specs.append(PolylineLayerSpec(
                    name=f"active_cuboids_t{tid}",
                    polylines_world=[edges[i].astype(np.float64)
                                     for i in range(12)],
                    color=(rgb[0], rgb[1], rgb[2], 255),
                    width=2,
                    subdivide_n=20,
                    labels_world=[(anchor, label)],
                ))

        return specs

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
            opencv_pinhole_intrinsics=self.opencv_pinhole_intrinsics,  # E2.7-OPCV
            opencv_pinhole_rays=self.opencv_pinhole_rays,  # E2.7-OPCV pre-computed
        )
        rgba = torch.cat([out["rgb"], out["opacity"]], dim=-1)
        rgba = torch.clamp(rgba, 0.0, 1.0)
        img = (rgba[0, :, :, :3] * 255).to(torch.uint8)
        return img.cpu().numpy()

    def _maybe_difix(self, img: np.ndarray) -> np.ndarray:
        """Route a rendered ``(H,W,3)`` uint8 frame through the active post-proc.

        Dispatches to whichever out-of-process server was configured:
          * Harmonizer temporal client (E2.6) — when present, sends ``1 + K``
            frames (curr + history) for de-flickering. Consumes and clears
            ``_postproc_reset_flag`` (set by non-play timeline changes) so a
            seek/scrub starts cold. Also forces a reset when the rendered
            resolution changed since the last frame (history frames must match
            curr's HxW).
          * DiFix client (original single-frame path) — unchanged.

        No-op (returns ``img`` unchanged) when the toggle is off or no server
        was configured. Both clients degrade gracefully — on any socket /
        protocol error they return the raw frame, so this never blocks or
        crashes the interactive loop.
        """
        if not self.difix_enabled:
            return img
        if self.harmonizer_temporal_client is not None:
            hc = self.harmonizer_temporal_client
            # Resolution lock: if HxW changed since the last frame, the history
            # deque holds stale-sized frames → force a cold reset.
            wh = (img.shape[1], img.shape[0])
            if self._postproc_last_wh is not None and \
                    self._postproc_last_wh != wh:
                self._postproc_reset_flag = True
            self._postproc_last_wh = wh
            reset = self._postproc_reset_flag
            self._postproc_reset_flag = False  # one-shot consume
            out = hc.fix(img, reset=reset)
            if self.difix_rtt is not None:
                self.difix_rtt.value = (
                    f"{hc.last_rtt_ms:.0f} ms"
                    if hc.healthy else "unavailable"
                )
            return out
        if self.difix_client is None:
            return img
        out = self.difix_client.fix(img)
        if self.difix_rtt is not None:
            self.difix_rtt.value = (
                f"{self.difix_client.last_rtt_ms:.0f} ms"
                if self.difix_client.healthy else "unavailable"
            )
        return out

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
            # DiFix novel-view post-process (optional, out-of-process). Applied
            # to the Gaussian backdrop *before* any overlay so cuboid wireframes
            # drawn on top aren't "corrected" away. No-op / graceful fallback
            # when disabled or the server is unreachable.
            img = self._maybe_difix(img)
            # B2: FTheta cuboid overlay — project + alpha-blend wireframes
            # into the backdrop so they share the fisheye projection
            # (cuboid wireframes drawn via add_line_segments use pinhole
            # projection in the browser and drift toward image edges).
            img_pre_overlay = None
            if self._overlay_compositor is not None:
                try:
                    layer_specs = self._collect_overlay_layer_specs(
                        self._t_us_current)
                    img_pre_overlay = img
                    img = self._overlay_compositor.composite(
                        img, layer_specs, view_matrix.astype(np.float64))
                except Exception as e:
                    print(f"[B2-OVERLAY] composite failed (continuing without "
                          f"overlay): {e}")
            # B2 debug: dump pre-overlay backdrop + post-overlay blended to
            # disk so the result can be inspected outside the browser when
            # WebGL canvas toDataURL is not viable. Activated by env var
            # B2_DUMP_DIR=/path; writes <dir>/b2_backdrop.png and
            # <dir>/b2_blended.png on each frame (last frame wins).
            _dump_dir = os.environ.get("B2_DUMP_DIR")
            if _dump_dir:
                try:
                    from PIL import Image as _PILImage
                    if img_pre_overlay is not None:
                        _PILImage.fromarray(img_pre_overlay).save(
                            os.path.join(_dump_dir, "b2_backdrop.png"))
                    _PILImage.fromarray(img).save(
                        os.path.join(_dump_dir, "b2_blended.png"))
                    # Also dump the c2w used for THIS frame so offline
                    # annotation tools can reproject the same cuboids
                    # with the matching camera pose.
                    np.save(os.path.join(_dump_dir, "b2_c2w.npy"),
                            view_matrix.astype(np.float64))
                    with open(os.path.join(_dump_dir, "b2_t_us.txt"), "w") as _f:
                        _f.write(str(self._t_us_current))
                except Exception as e:
                    print(f"[B2-DUMP] save failed: {e}")
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


def _load_multi_cam_poses(dataset_path: Optional[str],
                          default_config: str) -> dict:
    """V3-VIZ.3: extract per-camera per-frame c2w + timestamps for the
    Camera dropdown + Follow Camera modes.

    Returns ``{cam_id: {"c2w": (F, 4, 4) float32, "timestamps_us": (F,) int64}}``
    in world_global frame (T_world_to_world_global applied so the poses match
    ckpt ego frame). Empty dict when ``dataset_path`` is None or NCore SDK
    is unavailable.
    """
    if dataset_path is None:
        return {}
    try:
        import ncore  # noqa: F401 — runtime-only SDK
        from threedgrut.datasets.datasetNcore import NCoreDataset
    except ImportError as e:
        print(f"[viz_4d] multi-camera load skipped — NCore SDK missing: {e}",
              flush=True)
        return {}
    try:
        # NCoreDataset auto-fails when manifest has multiple sensors and
        # camera_ids=None. Open with the (camera-id wildcard) auto-discovery
        # by catching the "Multiple camera sensors" error message — the
        # exception payload lists every available sensor.
        try:
            NCoreDataset(
                datapath=str(dataset_path), split="train", device="cpu",
                sample_full_image=True, camera_ids=None, load_aux_masks=False,
                n_val_image_subsample=1,
            )
            all_cam_ids = ["camera_front_wide_120fov"]  # single-sensor clip
        except ValueError as err:
            msg = str(err)
            if "Multiple camera sensors" not in msg or "[" not in msg:
                raise
            # Extract the bracketed list literal from the error message.
            import ast
            lit = msg[msg.index("["):msg.rindex("]") + 1]
            all_cam_ids = sorted(ast.literal_eval(lit))
        ds = NCoreDataset(
            datapath=str(dataset_path), split="train", device="cpu",
            sample_full_image=True,
            camera_ids=all_cam_ids, load_aux_masks=False,
            n_val_image_subsample=1,
        )
        out: dict[str, dict] = {}
        end_col = int(ncore.data.FrameTimepoint.END)
        start_tp = ncore.data.FrameTimepoint.START
        T_w2wg = np.asarray(ds.T_world_to_world_global, dtype=np.float64)
        seq_id = ds.sequence_id
        for cam_id in ds.camera_ids:
            sensor = ds.sequence_camera_sensors[seq_id][cam_id]
            frame_indices = ds.camera_train_frame_indices.get(cam_id, [])
            if frame_indices is None or len(frame_indices) == 0:
                continue
            c2w_native = sensor.get_frames_T_source_target(
                source_node=sensor.sensor_id, target_node="world",
                frame_indices=frame_indices, frame_timepoint=start_tp,
            )
            c2w_native = np.asarray(c2w_native, dtype=np.float64).reshape(-1, 4, 4)
            c2w_wg = np.einsum("ij,njk->nik", T_w2wg, c2w_native).astype(np.float32)
            ts = np.asarray(sensor.frames_timestamps_us)[
                np.asarray(frame_indices), end_col].astype(np.int64)
            # Per-camera FTheta intrinsics (so dropdown switch actually changes
            # what the engine renders, not just the viewer pose). Falls back
            # to None for non-FTheta cameras (pinhole/distorted) — engine then
            # uses pinhole approx.
            cam_model = ds.sequence_camera_models[seq_id][cam_id]
            ftheta_dict = None
            opencv_pinhole_dict = None
            opencv_pinhole_rays = None  # (H, W, 3) camera-space rays for raygen
            fov_y_rad = 1.5708  # ~90°, generic default
            resolution = None
            try:
                W = int(cam_model.resolution[0].item())
                H = int(cam_model.resolution[1].item())
                resolution = (W, H)
                max_angle = getattr(cam_model, "max_angle", None)
                if isinstance(cam_model, ncore.sensors.FThetaCameraModel):
                    p = cam_model.get_parameters()
                    ftheta_dict = {
                        "resolution":              np.asarray(p.resolution, dtype=np.int64),
                        "shutter_type":            p.shutter_type.name,
                        "principal_point":         np.asarray(p.principal_point, dtype=np.float32),
                        "reference_poly":          p.reference_poly.name,
                        "pixeldist_to_angle_poly": np.asarray(p.pixeldist_to_angle_poly, dtype=np.float32),
                        "angle_to_pixeldist_poly": np.asarray(p.angle_to_pixeldist_poly, dtype=np.float32),
                        "max_angle":               float(p.max_angle),
                        "linear_cde":              np.asarray(p.linear_cde, dtype=np.float32),
                    }
                    fov_y_rad = 2.0 * float(max_angle)
                elif isinstance(cam_model, ncore.sensors.OpenCVPinholeCameraModel):
                    # E2.7-OPCV: same T8.13 fix as FTheta — bypass kaolin's
                    # pinhole raygen; use NCore SDK's rational distortion
                    # inverse to pre-compute camera-space rays that match the
                    # training contract (datasetNcore.py:443/449).
                    p = cam_model.get_parameters()
                    opencv_pinhole_dict = {
                        "resolution":         np.asarray(p.resolution, dtype=np.int64),
                        "shutter_type":       p.shutter_type.name,
                        "principal_point":    np.asarray(p.principal_point, dtype=np.float32),
                        "focal_length":       np.asarray(p.focal_length, dtype=np.float32),
                        "radial_coeffs":      np.asarray(p.radial_coeffs, dtype=np.float32),
                        "tangential_coeffs":  np.asarray(p.tangential_coeffs, dtype=np.float32),
                        "thin_prism_coeffs":  np.asarray(p.thin_prism_coeffs, dtype=np.float32),
                    }
                    xs = np.arange(W, dtype=np.int64)
                    ys = np.arange(H, dtype=np.int64)
                    px, py = np.meshgrid(xs, ys, indexing='xy')
                    pixels = np.stack([px, py], axis=-1).reshape(-1, 2)
                    rays = cam_model.pixels_to_camera_rays(pixels)
                    rays = rays.detach().cpu().numpy() if hasattr(rays, "detach") else np.asarray(rays)
                    opencv_pinhole_rays = rays.reshape(H, W, 3).astype(np.float32)
                    fy = float(cam_model.focal_length[1])
                    if fy > 0:
                        fov_y_rad = 2.0 * float(np.arctan(0.5 * H / fy))
                else:
                    fy = float(cam_model.focal_length[1])
                    if fy > 0:
                        fov_y_rad = 2.0 * float(np.arctan(0.5 * H / fy))
            except Exception as e:
                print(f"[viz_4d] cam {cam_id}: intrinsics extract failed ({e})",
                      flush=True)
            out[cam_id] = {
                "c2w": c2w_wg, "timestamps_us": ts,
                "ftheta_dict": ftheta_dict, "resolution": resolution,
                "fov_y_rad": fov_y_rad,
                "opencv_pinhole_dict": opencv_pinhole_dict,
                "opencv_pinhole_rays": opencv_pinhole_rays,
            }
            print(f"[viz_4d] cam {cam_id}: {c2w_wg.shape[0]} frames, "
                  f"FOV_y={np.rad2deg(fov_y_rad):.1f}°, "
                  f"ftheta={'YES' if ftheta_dict else 'no'}",
                  flush=True)
        return out
    except Exception as e:
        print(f"[viz_4d] multi-camera load failed: {e}", flush=True)
        return {}


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
    # E2.7 fix: NCoreDataset.__init__ takes ``datapath: str`` as first positional
    # arg, NOT a conf object. The previous ``NCoreDataset(conf, split="train")``
    # call was a dormant bug since T8.6 (2026-05-20) because every v2 ckpt
    # carried a viz_4d block and the fallback branch never ran. USDZ-converted
    # ckpts have no viz_4d block, so this is the first real exercise of the
    # fallback. Mirror _load_multi_cam_poses (L1604) signature — proven on the
    # same NCore clip 9ae151dc — including the "Multiple camera sensors"
    # ValueError fallback for multi-cam rings (camera_ids=None first probe
    # surfaces the sensor list, second open passes the full list).
    try:
        train_ds = NCoreDataset(
            datapath=str(dataset_path), split="train", device="cpu",
            sample_full_image=True, camera_ids=None, load_aux_masks=False,
            n_val_image_subsample=1,
        )
    except ValueError as _err:
        _msg = str(_err)
        if "Multiple camera sensors" not in _msg or "[" not in _msg:
            raise
        import ast as _ast
        _lit = _msg[_msg.index("["):_msg.rindex("]") + 1]
        _all_cam_ids = sorted(_ast.literal_eval(_lit))
        train_ds = NCoreDataset(
            datapath=str(dataset_path), split="train", device="cpu",
            sample_full_image=True, camera_ids=_all_cam_ids,
            load_aux_masks=False, n_val_image_subsample=1,
        )
    specs = specs_from_config(conf)
    model = LayeredGaussians(conf, specs=specs, scene_extent=1.0)
    model.init_from_checkpoint(ckpt, setup_optimizer=False)
    md_dict = extract_4d_metadata(model, train_ds, conf)
    return FourDMetadata.from_ckpt({"viz_4d": md_dict})


def _cleanup_stale_jit_baton_locks(min_age_s: float = 60.0) -> None:
    """E2.7: remove stale PyTorch JIT FileBaton lock files left by SIGKILL'd
    extension-loading processes.

    Root cause: ``torch.utils.file_baton.FileBaton.wait()`` is a presence-only
    poll (``while os.path.exists(lock_path): sleep``). The lock file is created
    on compile start and deleted on compile end — but if the holding process is
    ``pkill -9``'d mid-compile (no atexit hook runs), the lock file persists
    forever and ALL future processes block in an infinite loop trying to load
    the same extension. nvidia-smi shows clean GPU; ``flock -n`` succeeds (it's
    not an fcntl lock); restart doesn't fix it (lock is on disk).

    Symptom: Engine3DGRUT init hangs silently in ``Tracer → load_3dgut_plugin
    → jit.load → _jit_compile → file_baton.wait`` with WCHAN=hrtimer_nanosleep
    and 0% CPU forever. faulthandler dump traces straight to file_baton.py:51.

    Safe-removal heuristic: only delete locks older than ``min_age_s`` seconds.
    A lock younger than that may belong to a peer process currently compiling.

    Related upstream issues:
      https://github.com/pytorch/pytorch/issues/9711
      https://github.com/pytorch/pytorch/issues/41511
    """
    import glob as _glob
    import time as _time
    base = os.path.expanduser("~/.cache/torch_extensions")
    if not os.path.isdir(base):
        return
    locks = _glob.glob(os.path.join(base, "py*/*/lock"))
    now = _time.time()
    cleared = []
    for lk in locks:
        try:
            st = os.stat(lk)
        except FileNotFoundError:
            continue
        age = now - st.st_mtime
        if age < min_age_s:
            continue
        try:
            os.remove(lk)
            cleared.append((lk, age))
        except OSError as e:
            print(f"[viz_4d] stale-lock cleanup: {lk} rm failed ({e})",
                  flush=True)
    if cleared:
        for lk, age in cleared:
            print(f"[viz_4d] removed stale JIT FileBaton lock "
                  f"{lk} (age {age:.0f}s) — prevents file_baton.wait() "
                  f"infinite loop. See https://github.com/pytorch/pytorch/"
                  f"issues/9711", flush=True)


def main() -> None:
    # E2.7: guard against zombie JIT locks from previously SIGKILL'd processes.
    # Must run BEFORE the first import that triggers torch.utils.cpp_extension
    # (Engine3DGRUT → Tracer → load_3dgut_plugin → jit.load). Cheap (~1ms).
    _cleanup_stale_jit_baton_locks()

    parser = argparse.ArgumentParser()
    parser.add_argument("--gs_object", type=str, default=None,
                        help="Path of pretrained 3dgrt checkpoint (.pt). "
                             "Mutually exclusive with --usdz; one of the two "
                             "is required.")
    # E2.7: NVIDIA NRE/NuRec USDZ entry — transparently downgrades to
    # --gs_object path after on-the-fly conversion. Lets viser_gui_4d.py
    # load NVIDIA-trained USDZ checkpoints for visual A/B comparison against
    # 3dgrut2-native .pt outputs in a second viser instance.
    parser.add_argument("--usdz", type=str, default=None,
                        help="NVIDIA NRE/NuRec training-checkpoint USDZ path. "
                             "Mutually exclusive with --gs_object. When given, "
                             "main() calls convert_usdz_to_pt() to write a "
                             "3dgrut2-native .pt to --usdz_cache_dir, then "
                             "transparently downgrades to --gs_object path. "
                             "Requires --dataset_path (USDZ-converted .pt has "
                             "no viz_4d block; metadata comes from NCore "
                             "fallback).")
    parser.add_argument("--usdz_cache_dir", type=str,
                        default=os.path.expanduser(
                            "~/.cache/3dgrut2/nre_usdz_pt"),
                        help="Where converted .pt files are cached "
                             "(named <usdz_basename>_<mtime_hash>.pt for "
                             "idempotent reload across viser restarts).")
    parser.add_argument("--usdz_layers", type=str,
                        default="background,road,dynamic_rigids",
                        help="Comma-separated NRE Gaussian layers to load. "
                             "Default includes dynamic_rigids (E2.7-B: "
                             "vehicles move with timeline via populate_tracks "
                             "auto-hook on viz_4d.tracks). Skip "
                             "dynamic_rigids if --dataset_path is absent or "
                             "USDZ has no sequence_tracks.json. Don't add "
                             "sky_envmap (needs nvdiffrast, not installed "
                             "on inceptio/A800).")
    parser.add_argument("--initial_cam_id", type=str, default=None,
                        help="E2.7 H1 fix: lock viser initial camera to this "
                             "NCore camera id (must be present in "
                             "_load_multi_cam_poses ring). Recommended "
                             "'camera_front_120' for outdoor driving scenes "
                             "— random/default viser cameras can land in "
                             "far-field gaussian noise and produce "
                             "'unrecognizable artifacts'. When omitted, "
                             "viser uses meta.initial_c2w (legacy behavior).")
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
    parser.add_argument("--difix_server", type=str, default=None,
                        help="host:port of an out-of-process DiFix server "
                             "(e.g. 127.0.0.1:8765). When set, a 'DiFix' toggle "
                             "appears under Render Controls; enabling it routes "
                             "each rendered frame through DiFix before display. "
                             "Start the server with "
                             "threedgrut_playground/difix_server.py inside the "
                             "cosmos Docker env. Default: off.")
    parser.add_argument("--harmonizer_temporal_server", type=str, default=None,
                        help="E2.6: host:port of a temporal DiffusionHarmonizer "
                             "server (e.g. 127.0.0.1:59490). When set, a "
                             "'Harmonizer (temporal)' toggle appears; enabling "
                             "it routes each rendered frame through Harmonizer's "
                             "temporal mode (curr + last K corrected outputs) "
                             "for de-flickering continuous play sequences. "
                             "Seek/scrub auto-resets the history. Mutually "
                             "exclusive with --difix_server. Start the server "
                             "with threedgrut_playground/harmonizer_temporal_"
                             "server.py inside the harmonizer-cosmos-env Docker "
                             "env. Default: off.")
    parser.add_argument("--harmonizer_temporal_K", type=int, default=4,
                        help="E2.6: history depth K for temporal Harmonizer "
                             "(default 4, the paper default). The client holds "
                             "up to K prior corrected outputs; each request "
                             "carries 1 + min(history, K) frames.")
    parser.add_argument("--no_gaussian_render", action="store_true",
                        help="Skip Engine3DGRUT init + Gaussian background "
                             "rendering. Required on Ampere datacenter SKUs "
                             "WITHOUT RT cores (A100 / A800), where the OptiX "
                             "extension dlopen segfaults. Hopper datacenter "
                             "(H100/H800/H200) and all RTX cards have RT "
                             "cores and don't need this flag. Scene "
                             "primitives (ego/cuboid/LiDAR) + timeline still "
                             "work in this mode.")
    parser.add_argument("--renderer", type=str, default="3dgrt",
                        choices=["3dgrt", "3dgut"],
                        help="Rendering backend. '3dgrt' (default) uses OptiX ray tracing "
                             "(requires RT cores). '3dgut' uses tile-based rasterization "
                             "and works on A100/A800 without RT cores.")
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

    # E2.7: USDZ entry — convert USDZ → 3dgrut2-native .pt, then
    # transparently set args.gs_object so downstream Engine3DGRUT /
    # torch.load / _load_metadata paths are unchanged.
    if args.usdz is not None:
        if args.gs_object is not None:
            parser.error(
                "--usdz and --gs_object are mutually exclusive; pick one.")
        if not args.dataset_path:
            parser.error(
                "--usdz mode REQUIRES --dataset_path to the matching NCore "
                "clip json — USDZ-converted .pt has no viz_4d block, so "
                "metadata (timeline, ego pose, cam ring) must come from "
                "NCore fallback. Without --dataset_path the viser panel "
                "(timeline/frustum/cam dropdown) will be empty.")
        from threedgrut_playground.utils.nre_usdz_loader import (
            convert_usdz_to_pt,
        )
        os.makedirs(args.usdz_cache_dir, exist_ok=True)
        usdz_abs = os.path.abspath(args.usdz)
        mtime_hash = hashlib.md5(
            f"{usdz_abs}:{os.path.getmtime(args.usdz)}".encode()
        ).hexdigest()[:10]
        cache_pt = os.path.join(
            args.usdz_cache_dir,
            f"{os.path.basename(args.usdz)}_{mtime_hash}.pt",
        )
        if not os.path.exists(cache_pt):
            print(f"[E2.7-usdz] converting {args.usdz} → {cache_pt} "
                  f"(layers={args.usdz_layers}, albedo_mode=dc)")
            convert_usdz_to_pt(
                args.usdz,
                cache_pt,
                config_name=(args.default_gs_config
                             or "apps/ncore_3dgut_mcmc_multilayer"),
                layers=tuple(l.strip() for l in args.usdz_layers.split(",")
                             if l.strip()),
                albedo_mode="dc",
            )
        else:
            print(f"[E2.7-usdz] reusing cache {cache_pt} "
                  f"(mtime_hash={mtime_hash})")
        args.gs_object = cache_pt

        # E2.7 P1/P4 fix: align NRE gaussians to NCore world frame.
        # NRE/NuRec training puts its sequence origin at ego[0] (sequence-
        # local frame); NCore SDK ego trajectory / cam poses are in
        # world_global frame where ego[0] sits at a non-zero offset
        # (~few meters). Without this translation, viser renders gaussians
        # correctly but ego frustum / trajectory polyline / initial camera
        # land a few meters off the actual road position (the "P1 + P4"
        # symptoms in plan H4/H5).
        # Per-dataset offset is cached alongside the NRE-frame ckpt so the
        # ~5s NCore SDK init only happens once per (usdz, dataset_path) pair.
        # E2.7-B FINAL FRAME-OF-REFERENCE RULES (2026-06-15, 大g insight):
        # **Two different data sources live in two different frames** in the
        # E0.3 NRE Lightning USDZ container, and cross-source diff proved it:
        #
        #   (1) NRE state_dict gaussians (background / road / dynamic_rigids
        #       per-particle positions, ego frame poses inside ckpt):
        #       **NRE local frame** = NCore world − (38.00, -2.155, -0.278).
        #       MUST apply +translate to bring into NCore world frame.
        #
        #   (2) sequence_tracks.json cuboid track poses + sequence_tracks.usda
        #       cuboid declaration:
        #       **NCore world frame** directly (NRE training dumped them
        #       verbatim from NCore SDK annotations). DO NOT apply translate.
        #
        # Evidence on clip 9ae151dc tid='18' parked car:
        #   3dgrut2-own viz_4d.tracks["18"][0].translation = (-51.30, 1.07, 1.42)
        #   NRE sequence_tracks  ["18"] raw 7-vec[0:3]    = (-51.30, 1.12, 1.47)
        # ↑ both in NCore world frame, agree to 0.05m — cuboid pose: no translate.
        #
        # NRE raw background median = (3.28, -4.16, 7.83); NCore ego_pose[0] =
        # (2.17, 0.03, 1.44). NRE bg median near (0,0,0) is the
        # "ego-centric" NRE local origin — bg must be translated by +38m to
        # cover the trajectory in NCore world frame.
        #
        # Net rule below: align block translates background+road, skips
        # dynamic_rigids (object-local). Dyn track-pose build below uses
        # world_translate=None.
        aligned_pt = cache_pt[:-3] + "_aligned.pt"
        if not os.path.exists(aligned_pt):
            print(f"[E2.7-align] reading USDZ rig_trajectories.world_to_nre "
                  f"for NRE→NCore world translate...", flush=True)
            try:
                # E2.7 P1 ROOT-CAUSE FIX: USDZ container carries
                # ``rig_trajectories.json`` with a top-level ``world_to_nre``
                # 4x4 matrix — the EXACT transform NRE applied to map NCore
                # world → NRE training frame (typically pure translate by
                # ego trajectory midpoint, e.g. -38m x for clip 9ae151dc, to
                # keep float32 positions small near origin). Inverse =
                # NRE→world translate = -world_to_nre.matrix[:3, 3]. This
                # replaces the previous wrong heuristic of using NCore SDK
                # ego_pose[0] (~2m, wrong magnitude by 18×, wrong direction).
                # Detection by user (大g): "3dgrut2 ckpt 和 USDZ 同源，c2w
                # 数值应该一样" — diff revealed world_to_nre as the missing
                # transform.
                import zipfile as _zf, json as _json
                with _zf.ZipFile(args.usdz) as _zip:
                    with _zip.open("rig_trajectories.json") as _fp:
                        _rt = _json.load(_fp)
                _w2nre = _rt.get("world_to_nre") or {}
                _w2nre_mat = np.asarray(
                    _w2nre.get("matrix") if isinstance(_w2nre, dict)
                    else _w2nre,
                    dtype=np.float64,
                ).reshape(4, 4)
                # NRE → NCore world translate = -world_to_nre.translation
                _translate = (-_w2nre_mat[:3, 3]).astype(np.float32)
                print(f"[E2.7-align] world_to_nre.translation="
                      f"{_w2nre_mat[:3,3].tolist()}", flush=True)
                print(f"[E2.7-align] NRE→world translate "
                      f"={_translate.tolist()} (= -world_to_nre.translation)",
                      flush=True)
                # Sanity: rotation block should be identity (NRE typically
                # only translates origin, never rotates). If non-identity,
                # we'd need full matrix multiply on positions + per-axis
                # rotation on rotation quaternions — warn but proceed with
                # translate-only.
                _R = _w2nre_mat[:3, :3]
                if not np.allclose(_R, np.eye(3), atol=1e-4):
                    print(f"[E2.7-align] WARN: world_to_nre rotation NOT "
                          f"identity (R=\n{_R}\n); translate-only align is "
                          f"insufficient. Visual artifact possible. Full "
                          f"matrix align TODO if user sees rotation skew.",
                          flush=True)
                # Load NRE ckpt, apply +translate to every layer's positions,
                # save to aligned cache.
                _ckpt = torch.load(cache_pt, weights_only=False)
                _trans_t = torch.as_tensor(_translate, dtype=torch.float32)
                for _layer, _node in _ckpt["model"]["gaussians_nodes"].items():
                    # E2.7-B fix: dynamic_rigids gaussians live in
                    # **object-local frame** (each gaussian position is
                    # relative to its track's box center 0,0,0). At render
                    # time LayeredGaussians' ``_transform_means_and_active``
                    # transforms object-local → world using
                    # ``tracks_poses[tid][frame_idx]`` (which IS in world
                    # frame, post-translate per build_dynamic_tracks_for_viz4d).
                    # Applying world_translate here too would DOUBLE-translate
                    # the per-vehicle world position by 38m: all dyn gaussians
                    # would pile up around (+38, -2.16, -0.28) instead of
                    # following their respective track poses scattered along
                    # the trajectory. fervent-knuth 873L: ``node_offset =
                    # offset if spec.name != 'dynamic_rigids' else None``.
                    if _layer == "dynamic_rigids":
                        print(f"[E2.7-align]   layer={_layer}: "
                              f"{_node['positions'].shape[0]} gaussians kept "
                              f"in object-local frame (render_pass applies "
                              f"track_pose per timestamp)", flush=True)
                        continue
                    _p = _node["positions"]
                    _orig_param = isinstance(_p, torch.nn.Parameter)
                    _p_dev = _p.device
                    _p_dtype = _p.dtype
                    with torch.no_grad():
                        _p_new = _p.detach() + _trans_t.to(
                            device=_p_dev, dtype=_p_dtype)
                    if _orig_param:
                        _node["positions"] = torch.nn.Parameter(
                            _p_new.contiguous(), requires_grad=False)
                    else:
                        _node["positions"] = _p_new.contiguous()
                    print(f"[E2.7-align]   layer={_layer}: shifted "
                          f"{_p.shape[0]} gaussians by "
                          f"({_translate[0]:.2f},{_translate[1]:.2f},"
                          f"{_translate[2]:.2f})", flush=True)
                torch.save(_ckpt, aligned_pt)
                print(f"[E2.7-align] wrote aligned ckpt → {aligned_pt}",
                      flush=True)
            except Exception as _ae:
                print(f"[E2.7-align] WARN: world-align skipped ({_ae!r}); "
                      f"P1/P4 frustum/trajectory may show ~few-meter offset",
                      flush=True)
                aligned_pt = None
        else:
            print(f"[E2.7-align] reusing aligned cache {aligned_pt}",
                  flush=True)
        if aligned_pt is not None and os.path.exists(aligned_pt):
            args.gs_object = aligned_pt
    elif args.gs_object is None:
        parser.error("one of --gs_object or --usdz is required.")

    print(f"[E2.7-init] about to init Engine3DGRUT(gs_object={args.gs_object}, "
          f"default_config={args.default_gs_config}, renderer={args.renderer})",
          flush=True)
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
            renderer=args.renderer,
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

    # E2.7 sanity-check (USDZ mode only): print per-layer position range +
    # scene_extent so any coordinate-system / clip_floater issue surfaces
    # immediately at startup instead of staring at a smeared viser. Cheap and
    # diagnostic for H3 (far-field floaters) + H5 (coord-system mismatch).
    if args.usdz and isinstance(ckpt, dict) and "model" in ckpt:
        m = ckpt["model"]
        nodes = m.get("gaussians_nodes", {}) if isinstance(m, dict) else {}
        for layer, node in nodes.items():
            try:
                p = node["positions"]
            except (KeyError, TypeError):
                continue
            try:
                p_cpu = p.detach().cpu() if hasattr(p, "detach") else p
                med = p_cpu.median(0).values.tolist()
                print(
                    f"[E2.7-sanity] layer={layer} N={p_cpu.shape[0]} "
                    f"median=({med[0]:.2f},{med[1]:.2f},{med[2]:.2f}) "
                    f"x=[{p_cpu[:,0].min().item():.1f},"
                    f"{p_cpu[:,0].max().item():.1f}] "
                    f"y=[{p_cpu[:,1].min().item():.1f},"
                    f"{p_cpu[:,1].max().item():.1f}] "
                    f"z=[{p_cpu[:,2].min().item():.1f},"
                    f"{p_cpu[:,2].max().item():.1f}]",
                    flush=True,
                )
            except Exception as _e:
                print(f"[E2.7-sanity] layer={layer} stats failed: {_e!r}",
                      flush=True)
        ext = m.get("scene_extent", None) if isinstance(m, dict) else None
        try:
            ext_v = float(ext) if ext is not None else float("nan")
        except (TypeError, ValueError):
            ext_v = float("nan")
        print(
            f"[E2.7-sanity] scene_extent={ext_v:.2f}m "
            f"(driving clip expect ~100-1000m; >5000m means "
            f"clip_floater_gaussians 阈值还不够紧 — see plan H3)",
            flush=True,
        )

    metadata = _load_metadata(ckpt, args.dataset_path, args.default_gs_config)
    # E2.7 H2 fix: explicit metadata source + USDZ-mode必填 enforcement.
    if args.usdz and metadata is None:
        raise RuntimeError(
            f"--usdz mode failed to load metadata from "
            f"--dataset_path={args.dataset_path}. USDZ-converted .pt has no "
            f"viz_4d block, so NCore fallback is the only metadata source. "
            f"Verify the dataset_path is the right pai_*.json for the same "
            f"clip as {args.usdz}."
        )
    if metadata is not None:
        _n_frames = (metadata.n_frames()
                     if callable(getattr(metadata, "n_frames", None))
                     else metadata.n_frames)
        print(
            f"[E2.7] metadata source: "
            f"{'USDZ→.pt (NCore fallback)' if args.usdz else 'ckpt viz_4d or NCore fallback'}, "
            f"frames={_n_frames}, "
            f"t_us=[{metadata.t_us_first},{metadata.t_us_last}]",
            flush=True,
        )

    # E2.7-B: wire dynamic_rigids tracks. amazing-lalande's simplified loader
    # only ride-along's the per-gaussian ``_nre_cuboid_ids`` from NRE state_dict;
    # to make vehicles MOVE with the timeline we must parse volume.usda +
    # sequence_tracks.json out of the USDZ container, resample each track's
    # raw 7-vec poses onto the NCore shared camera timeline, apply the same
    # world_to_nre.inv translate the static layers got, then both:
    #   (a) register layer.track_ids buffer on dynamic_rigids (gaussian → tid slot)
    #   (b) populate_tracks(...) so render_pass(timestamp_us=t_us) finds
    #       the per-frame pose to transform object-local gaussians
    #   (c) inject metadata.tracks + metadata.tracks_camera_timestamps_us
    #       so viser cuboid wireframes also follow the timeline (the
    #       NCore SDK fallback path used by --dataset_path does NOT
    #       provide tracks; only viz_4d-ckpt path does, see _load_metadata)
    if args.usdz and metadata is not None and engine is not None:
        try:
            dyn_node = ckpt.get("model", {}).get("gaussians_nodes", {}).get(
                "dynamic_rigids")
            if dyn_node is None:
                print("[E2.7-B] dynamic_rigids not in ckpt (loader skipped it "
                      "or --usdz_layers excluded it); no track wiring.",
                      flush=True)
            elif "_nre_cuboid_ids" not in dyn_node:
                print("[E2.7-B] dynamic_rigids loaded but no _nre_cuboid_ids "
                      "ride-along — NRE state_dict missing gaussian_cuboid_ids; "
                      "vehicles will render at static base pose, no track motion.",
                      flush=True)
            else:
                from threedgrut_playground.utils.nre_usdz_loader import (
                    build_dynamic_tracks_for_viz4d,
                )
                cuboid_ids_t = dyn_node["_nre_cuboid_ids"]
                cuboid_ids_np = (
                    cuboid_ids_t.detach().cpu().numpy()
                    if hasattr(cuboid_ids_t, "detach")
                    else np.asarray(cuboid_ids_t)
                )
                # E2.7-B ROOT-CAUSE-CORRECTION: NRE world frame == NCore
                # world frame (verified by tid='18' cross-source diff between
                # 3dgrut2-own viz_4d.tracks and NRE sequence_tracks). Do NOT
                # apply any world_translate to track poses — they're already
                # in NCore world frame directly. The static-layer align block
                # above also skips translate now.
                world_translate = None

                # Shared timeline = NCore primary cam frame timestamps the
                # FourDMetadata already loaded. Tracks are resampled onto
                # this so frame_info masks the in-window frames.
                timeline_us = np.asarray(
                    metadata.ego_frame_timestamps_us, dtype=np.int64
                )
                print(f"[E2.7-B] resampling dynamic_rigids tracks onto "
                      f"{timeline_us.size}-frame NCore timeline "
                      f"(world_translate=None; NRE pose == NCore world)...",
                      flush=True)
                track_ids_np, tracks_dict = build_dynamic_tracks_for_viz4d(
                    args.usdz, cuboid_ids_np, timeline_us,
                    world_translate=world_translate,
                )
                print(f"[E2.7-B] built {len(tracks_dict)} tracks for "
                      f"{cuboid_ids_np.size} gaussians; track_ids range "
                      f"[{track_ids_np.min()},{track_ids_np.max()}].",
                      flush=True)
                # (a) register track_ids buffer on dynamic_rigids layer.
                # layered_model.py:639 also restores this from
                # ckpt["model"]["gaussians_nodes"][name]["track_ids"], but
                # amazing-lalande loader doesn't write it that way — do the
                # register here instead (idempotent: delattr first).
                _dyn_layer = engine.scene_mog.layers["dynamic_rigids"]
                _dev = _dyn_layer.positions.device
                if hasattr(_dyn_layer, "track_ids"):
                    delattr(_dyn_layer, "track_ids")
                _dyn_layer.register_buffer(
                    "track_ids",
                    torch.as_tensor(track_ids_np, dtype=torch.long, device=_dev),
                    persistent=True,
                )
                # (b) populate_tracks. First-tid carries shared timeline.
                _first_tid = next(iter(tracks_dict))
                _torch_tracks = {}
                _n_active_per_track = []
                for _tid, _t in tracks_dict.items():
                    _fi = torch.as_tensor(
                        _t["frame_info"], dtype=torch.bool, device=_dev)
                    _n_active_per_track.append(int(_fi.sum().item()))
                    _torch_tracks[_tid] = {
                        "poses": torch.as_tensor(
                            _t["poses"], dtype=torch.float32, device=_dev),
                        "size": torch.as_tensor(
                            _t["size"], dtype=torch.float32, device=_dev),
                        "frame_info": _fi,
                        "class": _t["class"],
                    }
                _act = np.asarray(_n_active_per_track)
                print(f"[E2.7-B] active-frame distribution per track: "
                      f"min={_act.min()} max={_act.max()} "
                      f"median={int(np.median(_act))} of {timeline_us.size} frames, "
                      f"all-zero tracks={(_act==0).sum()}/{len(_act)}",
                      flush=True)
                _torch_tracks[_first_tid]["cam_timestamps_us"] = torch.as_tensor(
                    timeline_us, dtype=torch.int64, device=_dev)
                engine.scene_mog.populate_tracks(_torch_tracks)
                print(f"[E2.7-B] populate_tracks done: "
                      f"{len(_torch_tracks)} tracks on dynamic_rigids "
                      f"({_dyn_layer.positions.shape[0]} gaussians).",
                      flush=True)
                # E2.7-B diagnostic: verify the 3 conditions LayeredGaussians
                # checks in _transform_means_and_active path
                # (layered_model.py:999-1003). If any is False, dyn gaussians
                # stay in object-local frame (invisible far from camera).
                _tp = engine.scene_mog.tracks_poses
                _has_tid = hasattr(_dyn_layer, "track_ids")
                _n_track_pose_bufs = sum(
                    1 for name, _ in engine.scene_mog.named_buffers()
                    if name.startswith("_track_pose_")
                )
                _n_track_active_bufs = sum(
                    1 for name, _ in engine.scene_mog.named_buffers()
                    if name.startswith("_track_active_")
                )
                _has_shared_ts = hasattr(
                    engine.scene_mog, "tracks_camera_timestamps_us"
                )
                _shared_ts_len = (
                    int(engine.scene_mog.tracks_camera_timestamps_us.shape[0])
                    if _has_shared_ts else 0
                )
                print(
                    f"[E2.7-B-diag] dynamic transform pre-flight:\n"
                    f"  hasattr(dyn_layer, track_ids)        = {_has_tid}\n"
                    f"  len(scene_mog.tracks_poses)          = "
                    f"{len(_tp) if _tp is not None else 'None'}\n"
                    f"  _track_pose_* buffer count           = {_n_track_pose_bufs}\n"
                    f"  _track_active_* buffer count         = {_n_track_active_bufs}\n"
                    f"  tracks_camera_timestamps_us shape    = "
                    f"({_shared_ts_len},)\n"
                    f"  layer.positions median (object-local)= "
                    f"{_dyn_layer.positions.median(0).values.tolist()}\n"
                    f"  layer.track_ids range                = "
                    f"[{_dyn_layer.track_ids.min().item()},"
                    f"{_dyn_layer.track_ids.max().item()}]",
                    flush=True,
                )
                # Sample one track's pose to confirm world-frame translate is in
                if _n_track_pose_bufs > 0:
                    _first_tid = sorted(_torch_tracks.keys())[0]
                    _sample_buf = getattr(
                        engine.scene_mog, f"_track_pose_{_first_tid}", None
                    )
                    if _sample_buf is not None:
                        print(
                            f"  sample _track_pose_{_first_tid}[0] pos="
                            f"{_sample_buf[0, :3, 3].tolist()}",
                            flush=True,
                        )
                # (c) inject metadata.tracks + tracks_camera_timestamps_us
                # so viser cuboid wireframes follow the timeline too.
                metadata.tracks = {
                    _tid: {
                        "poses": _t["poses"],
                        "size": _t["size"],
                        "frame_info": _t["frame_info"],
                        "class": _t["class"],
                    }
                    for _tid, _t in tracks_dict.items()
                }
                metadata.tracks_camera_timestamps_us = timeline_us
                print(f"[E2.7-B] injected metadata.tracks ({len(metadata.tracks)} "
                      f"tids) — viser cuboid wireframes now timeline-aware.",
                      flush=True)
        except Exception as _e:
            print(f"[E2.7-B] WARN: dynamic_rigids track wiring failed ({_e!r}); "
                  f"vehicles will render at static base pose, no track motion.",
                  flush=True)
            import traceback as _tb
            _tb.print_exc()

    # E2.7 H1 alias fallback: NCore cam ids vary between clip generations
    # (camera_front_120 vs camera_front_wide_120fov vs front_120). If user
    # passed a non-existent id, try a few common forward-camera aliases so
    # the visual-comparison workflow works without re-launching.
    if args.initial_cam_id:
        # Just an early hint; the actual snap happens in Viser4DViewer based
        # on this exact arg value. _load_multi_cam_poses is called later
        # (next block), so we don't have the cam ring here yet — the
        # Viser4DViewer init prints a WARN with the available list when
        # the id misses. This early alias note is purely informational.
        _common_front_aliases = (
            "camera_front_120", "front_120",
            "camera_front_wide_120fov", "camera_front_wide", "front",
        )
        if args.initial_cam_id not in _common_front_aliases:
            print(f"[E2.7] initial_cam_id='{args.initial_cam_id}' (custom; "
                  f"if missing from cam ring, viser will WARN with the "
                  f"available list — try one of {_common_front_aliases})",
                  flush=True)
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
    # V3-VIZ.3: per-camera per-frame c2w lookup (multi-camera dropdown +
    # Follow Camera). Empty when --dataset_path not provided → dropdown
    # falls back to {primary} only.
    multi_cam_poses = _load_multi_cam_poses(
        args.dataset_path, args.default_gs_config)
    if multi_cam_poses:
        print(f"[viz_4d] V3-VIZ.3: {len(multi_cam_poses)} cameras available "
              f"for dropdown / Follow Camera")

    # E2.7 P1/P4 diagnostic: dump NCore metadata frame origin vs USDZ gaussian
    # center side-by-side so the operator can immediately tell if the two are
    # in the same coordinate frame. If ego_poses[0] is at e.g. (200, 100, 0)
    # while background median is at (3, -4, 8), the NCore SDK trajectory is
    # in world_global (accumulated) frame but NRE USDZ gaussians are in
    # ego/rig local frame — viser will then render gaussians correctly but
    # the ego frustum / trajectory polyline / initial camera land in a
    # different coordinate system (looks like "frustum position not at car",
    # "initial camera x offset", "ego trajectory not on the road").
    if args.usdz and metadata is not None:
        try:
            ego_first = metadata.ego_pose_at(metadata.t_us_first)
            ep = ego_first[:3, 3]
            print(f"[E2.7-coord] NCore ego_pose_at(t_us_first) "
                  f"position=({ep[0]:.2f},{ep[1]:.2f},{ep[2]:.2f})", flush=True)
            if hasattr(metadata, "ego_poses_c2w") and metadata.ego_poses_c2w is not None and metadata.ego_poses_c2w.size > 0:
                e0 = metadata.ego_poses_c2w[0, :3, 3]
                eN = metadata.ego_poses_c2w[-1, :3, 3]
                print(f"[E2.7-coord] NCore ego_poses_c2w[0]={tuple(round(float(v),2) for v in e0)} "
                      f"[-1]={tuple(round(float(v),2) for v in eN)} "
                      f"(span={(eN-e0).round(1).tolist()})", flush=True)
            if multi_cam_poses and args.initial_cam_id in multi_cam_poses:
                c0 = multi_cam_poses[args.initial_cam_id]["c2w"][0, :3, 3]
                print(f"[E2.7-coord] '{args.initial_cam_id}' c2w[0] "
                      f"position=({c0[0]:.2f},{c0[1]:.2f},{c0[2]:.2f})", flush=True)
            # Background gaussian median already printed by sanity-check
            # above (E2.7-sanity layer=background median=...). Compare visually
            # against these ego-frame coordinates: if magnitudes differ >100x
            # or signs flip, gaussians and metadata are in different frames.
            print("[E2.7-coord] compare with [E2.7-sanity] background median "
                  "above. Same-frame: numbers should be in the same ballpark "
                  "(driving scene ego trajectory ≤ a few km accumulated; "
                  "NRE local frame: a few hundred m centered on ~0).",
                  flush=True)
        except Exception as _e:
            print(f"[E2.7-coord] diagnostic failed: {_e!r}", flush=True)

    # E2.6: DiFix (single-frame) and Harmonizer (temporal) post-proc backends
    # are mutually exclusive — both wire the same toggle/RTT slot.
    if args.difix_server and args.harmonizer_temporal_server:
        raise SystemExit(
            "[viz_4d] --difix_server and --harmonizer_temporal_server are "
            "mutually exclusive (both wire the single post-proc toggle). "
            "Pick one."
        )
    viewer = Viser4DViewer(
        port=args.port, engine=engine, metadata=metadata,
        target_fps=args.target_fps,
        initial_fov_rad=math.radians(args.initial_fov_deg),
        multi_cam_poses=multi_cam_poses,
        difix_server=args.difix_server,
        harmonizer_temporal_server=args.harmonizer_temporal_server,
        harmonizer_temporal_K=args.harmonizer_temporal_K,
        initial_cam_id=args.initial_cam_id,
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
