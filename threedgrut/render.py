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

import json
import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import threedgrut.datasets as datasets
from threedgrut.model.class_psnr import (
    collect_active_tracks_for_frame,
    compute_class_psnr,
)
from threedgrut.model.per_class_eval import (
    DEFAULT_ACTOR_CLASS_SPECS,
    ROAD_CLASS_IDS,
    compute_per_class_metrics,
    compute_lane_metrics,
    LANE_CLASS_IDS,
    DEFAULT_LANE_BAND_PX,
)
from threedgrut.correction.difix import DifixPostProcessor
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.color_correct import color_correct_affine
from threedgrut.utils.eval_metrics import (  # T11.F1 / E1.4
    compute_lidar_psnr,
    kid_subset_size,
    rgb01_to_uint8_chw,
)
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import create_summary_writer
from threedgrut.utils.render import apply_post_processing


def apply_dataset_cameras_override(conf, dataset_cameras) -> bool:
    """E1.3 held-out protocol: replace the ckpt-embedded ``dataset.camera_ids``.

    ``make_test`` builds the eval dataset from ``conf.dataset.camera_ids``
    (datasets/__init__.py), so a camera excluded at train time is invisible
    to the existing ``--eval-cameras`` batch filter — the dataset never loads
    it. This override swaps the camera set before dataset construction.

    Struct note: ckpt confs are struct-locked OmegaConf; adding/replacing a
    dataset key requires ``set_struct(conf, False)`` first (same pattern as
    the load_lane_masks injection in ``from_checkpoint``).

    Returns True when an override was applied, False on None/empty.
    """
    if not dataset_cameras:
        return False
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(conf):
            OmegaConf.set_struct(conf, False)
    except ImportError:  # plain-dict confs (tests)
        pass
    conf["dataset"]["camera_ids"] = list(dataset_cameras)
    return True


class Renderer:
    def __init__(
        self,
        model,
        conf,
        global_step,
        out_dir,
        path="",
        save_gt=True,
        writer=None,
        compute_extra_metrics=True,
        post_processing=None,
        novel_view=False,
        exposure_model=None,
        novel_fid=False,
        novel_save_n: int = 5,
    ) -> None:

        if path:  # Replace the path to the test data
            conf.path = path

        self.model = model
        self.out_dir = out_dir
        self.save_gt = save_gt
        self.path = path
        self.conf = conf
        self.global_step = global_step
        self.dataset, self.dataloader = self.create_test_dataloader(conf)
        self.writer = writer
        self.compute_extra_metrics = compute_extra_metrics
        self.post_processing = post_processing
        # T8.5.3 / V3-E3: 5x render cost when enabled — 4 extra renders per
        # anchor frame (lateral_1m / lateral_2m / yaw_5deg / yaw_10deg).
        self.novel_view = bool(novel_view)
        # E1.4: FID/KID distribution metrics (interpolated always; per
        # novel mode when novel_view is also on). Off by default so
        # metrics.json stays byte-identical.
        self.novel_fid = bool(novel_fid)
        # E2.1: how many novel-view frames to persist (-1 = all). Default 5
        # preserves the historical visual-sample behaviour byte-for-byte.
        self.novel_save_n = int(novel_save_n)
        # T9.3 / V3-P1.c: BilateralGrid (or legacy ExposureModel) applied
        # AFTER model forward + post_processing, BEFORE metrics. Aligns the
        # eval-time pred_rgb with the train-time loss target (which goes
        # through trainer.py:1641-1643 exposure_model). None = no-op (legacy
        # behavior; eval skips correction). Set by from_preloaded_model (live
        # trainer pass-through) or from_checkpoint (reconstructs from
        # ckpt["exposure_state"]).
        self.exposure_model = exposure_model

        if conf.model.background.color == "black":
            self.bg_color = torch.zeros((3,), dtype=torch.float32, device="cuda")
        elif conf.model.background.color == "white":
            self.bg_color = torch.ones((3,), dtype=torch.float32, device="cuda")
        else:
            assert False, f"{conf.model.background.color} is not a supported background color."

        # V3-T15.2: optional DiFix post-process. Module-level import above is
        # safe on dev machines without cosmos_predict2 because DifixPostProcessor
        # uses lazy import — the heavy stack is only loaded if ``enabled=True``
        # and ``forward`` is actually invoked.
        self.difix = DifixPostProcessor(
            enabled=bool(conf.render.get("use_difix", False)),
            ckpt_path=conf.render.get("difix_ckpt_path", None),
            timestep=int(conf.render.get("difix_timestep", 250)),
        )

    def create_test_dataloader(self, conf):
        """Create the test dataloader for the given configuration."""
        from threedgrut.datasets.utils import configure_dataloader_for_platform

        dataset = datasets.make_test(name=conf.dataset.type, config=conf)

        # Configure DataLoader arguments for the current platform
        dataloader_kwargs = configure_dataloader_for_platform(
            {
                "num_workers": 8,
                "batch_size": 1,
                "shuffle": False,
                "collate_fn": None,
            }
        )

        dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
        return dataset, dataloader

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path,
        out_dir,
        path="",
        save_gt=True,
        writer=None,
        model=None,
        computes_extra_metrics=True,
        eval_cameras=None,
        novel_view=False,
        use_difix=False,
        load_lane_masks=False,
        lane_band_px=None,
        dataset_cameras=None,
        novel_fid=False,
        novel_save_n: int = 5,
    ):
        """Loads checkpoint for test path.
        If path is stated, it will override the test path in checkpoint.
        If model is None, it will be loaded base on the

        Args:
            eval_cameras: T8.5.7 / V3-E4 — optional list[str] of camera_id
                strings to restrict eval to a subset (e.g. NCore 5-cam ring).
                None / empty → no filter. Injected into ``conf.render.eval_cameras``
                so the same Hydra-style key works for both ckpt eval and
                training-end eval.
            use_difix: V3-T15.2 Stage A.4 — when True, inject
                ``conf.render.use_difix = True`` so Renderer.__init__ builds
                an enabled DifixPostProcessor and render_all() computes the
                ``mean_*_difix`` metric trio. Pre-T15.2 ckpts have no
                ``use_difix`` key in their embedded conf, so we must inject
                here rather than relying on conf default.
        """

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        global_step = checkpoint["global_step"]

        conf = checkpoint["config"]
        # overrides
        if conf["render"]["method"] == "3dgrt":
            conf["render"]["particle_kernel_density_clamping"] = True
            conf["render"]["min_transmittance"] = 0.03
        conf["render"]["enable_kernel_timings"] = True
        # T8.5.7 / V3-E4: inject eval_cameras subset filter into the ckpt-embedded
        # config. Old ckpts (pre-V3-E4) lack this key — assignment via dict-style
        # works on the OmegaConf used here (see L107-110 patterns above).
        if eval_cameras:
            conf["render"]["eval_cameras"] = list(eval_cameras)
        # V3-T15.2 Stage A.4: CLI --use-difix injection (same dict-style pattern).
        if use_difix:
            conf["render"]["use_difix"] = True
        # Phase 3 lane: inject dataset.load_lane_masks so a pre-trained baseline
        # ckpt (whose embedded conf predates lane) loads *.aux.lane.zarr.itar at
        # eval → render_all() emits mean_lane_*. Unlike conf.render (where
        # eval_cameras already exists), conf.dataset is struct-locked, so adding
        # a brand-new key raises ConfigKeyError — open the struct first.
        if load_lane_masks or lane_band_px is not None:
            from omegaconf import OmegaConf
            OmegaConf.set_struct(conf, False)
        if load_lane_masks:
            conf["dataset"]["load_lane_masks"] = True
        if lane_band_px is not None:
            conf["render"]["lane_band_px"] = int(lane_band_px)
        # E1.3 held-out protocol: swap the dataset camera set (e.g. eval a
        # 4-cam ckpt on the excluded cross camera). Must happen before
        # make_test builds the dataset below.
        dataset_cameras_active = apply_dataset_cameras_override(conf, dataset_cameras)

        object_name = Path(conf.path).stem
        experiment_name = conf["experiment_name"]
        writer, out_dir, run_name = create_summary_writer(conf, object_name, out_dir, experiment_name, use_wandb=False)

        if model is None:
            # T8.5.7 fix: respect conf.use_layered_model — multilayer ckpts
            # store params under nested ``gaussians_nodes`` and MoG's
            # init_from_checkpoint() looks for top-level ``positions`` and
            # raises KeyError. Mirror trainer.init_model dispatch so
            # standalone ``python render.py --checkpoint ...`` works on
            # both flat MoG and LayeredGaussians ckpts.
            if conf.get("use_layered_model", False):
                from threedgrut.layers.layered_model import LayeredGaussians
                from threedgrut.layers.registry import specs_from_config

                specs = specs_from_config(conf)
                # V3-E4.1 fix: pass the saved scene_extent (live trainer does
                # the same) instead of None — mirrors engine.py:1340-1344.
                scene_extent = float(
                    checkpoint.get("model", {}).get("scene_extent", 1.0)
                )
                model = LayeredGaussians(
                    conf, specs=specs, scene_extent=scene_extent,
                )
                # V3 Stage A/B/D.2 bugfix: ``populate_tracks`` MUST run BEFORE
                # ``init_from_checkpoint`` for learnable_pose ckpts. Reason:
                # ``LayeredGaussians.init_from_checkpoint`` (layered_model.py
                # L632-672) calls ``load_state_dict(layered_track_state,
                # strict=False)`` to restore _track_quat_/_track_trans_/
                # _track_pose_gt_/_track_active_ entries — but
                # ``load_state_dict`` only writes into pre-existing slots, and
                # those slots are created by ``populate_tracks``. If we call
                # them in the wrong order, the learned Parameter values are
                # silently dropped ("unexpected keys" warning) and the model
                # ends up with GT-init values from tracks_dict instead of the
                # ckpt's learned poses (yesterday's D.2 triptych diff ≈ 0
                # was caused by exactly this). The trainer's order is correct
                # (trainer.init_model L386-449); render.py + engine.py were
                # both inverted since V3-E4.1 — fixed simultaneously.
                viz_4d = checkpoint.get("viz_4d")
                if viz_4d is not None and isinstance(viz_4d, dict):
                    tracks_dict = viz_4d.get("tracks")
                    shared_ts = viz_4d.get("tracks_camera_timestamps_us")
                    if tracks_dict and shared_ts is not None:
                        # Inject shared timestamps into the first track so
                        # _populate_tracks_impl picks them up via its
                        # first-track scan (single shared buffer across all
                        # tracks; same NCore camera schedule).
                        first_tid = next(iter(tracks_dict))
                        tracks_dict[first_tid]["cam_timestamps_us"] = shared_ts
                        model.populate_tracks(tracks_dict)
                model.init_from_checkpoint(checkpoint, setup_optimizer=False)
            else:
                model = MixtureOfGaussians(conf)
                model.init_from_checkpoint(checkpoint, setup_optimizer=False)
        model.build_acc()

        # Load post-processing if present in checkpoint
        post_processing = None
        method = conf.post_processing.method
        if "post_processing" in checkpoint and method == "ppisp":
            from ppisp import PPISP, PPISPConfig

            # Derive config from training settings to match trainer.py
            use_controller = conf.post_processing.get("use_controller", True)
            n_distillation_steps = conf.post_processing.get("n_distillation_steps", 5000)
            if use_controller and n_distillation_steps > 0:
                main_training_steps = conf.n_iterations - n_distillation_steps
                controller_activation_ratio = main_training_steps / conf.n_iterations
                controller_distillation = True
            elif use_controller:
                controller_activation_ratio = 0.8
                controller_distillation = False
            else:
                controller_activation_ratio = 0.0
                controller_distillation = False

            ppisp_config = PPISPConfig(
                use_controller=use_controller,
                controller_distillation=controller_distillation,
                controller_activation_ratio=controller_activation_ratio,
            )

            post_processing = PPISP.from_state_dict(checkpoint["post_processing"]["module"], config=ppisp_config)
            post_processing = post_processing.to("cuda")
            num_cameras = post_processing.crf_params.shape[0]
            num_frames = post_processing.exposure_params.shape[0]
            logger.info(f"📷 {method.upper()} loaded from checkpoint: {num_cameras} cameras, {num_frames} frames")

        # T9.3 / V3-P1.c: rebuild BilateralGrid from ckpt["exposure_state"]
        # so standalone reload eval applies the same color correction the
        # trainer applied during training. Without this, raw psnr_masked
        # measures (model_output vs GT) — which can be arbitrarily far off
        # if BilateralGrid absorbed cross-channel tone (the v2 ExposureModel
        # 退化 mechanism). With this, raw psnr_masked measures
        # (bilateral_grid(model_output) vs GT), matching the train-time loss.
        #
        # Legacy v2 ExposureModel ckpts contain {exposure_a, exposure_b}
        # instead of {grids, _rgb2gray_w} — skip with a warning so v2 ckpts
        # still load (eval falls back to no exposure applied, matching the
        # pre-T9.3 behavior for those ckpts).
        # E1.3: with a dataset-camera override the BilateralGrid camera_idx
        # mapping no longer matches training (grid i belongs to train-camera
        # i) — applying it would color-correct the held-out camera with some
        # other camera's grid. Force exposure off; read held-out numbers from
        # the cc_* (per-frame affine) metrics instead. Protocol doc: R-v4.8.
        exposure_model = None
        if dataset_cameras_active and "exposure_state" in checkpoint:
            logger.warning(
                "📷 --dataset-cameras active → BilateralGrid exposure model "
                "DISABLED (train-time camera_idx mapping invalid for an "
                "overridden camera set); use cc_* metrics for held-out eval."
            )
        elif "exposure_state" in checkpoint:
            from threedgrut.correction import BilateralGrid

            module_state = checkpoint["exposure_state"]["module"]
            if "grids" in module_state:
                grids = module_state["grids"]
                N, twelve, L_z, L_y, L_x = grids.shape
                assert twelve == 12, f"unexpected grids shape {grids.shape}"
                exposure_model = BilateralGrid(
                    num_camera=N, grid_X=L_x, grid_Y=L_y, grid_W=L_z,
                ).to("cuda")
                exposure_model.load_state_dict(module_state, strict=True)
                exposure_model.eval()
                logger.info(
                    f"📷 BilateralGrid loaded from checkpoint: "
                    f"{N} cameras, grid={L_x}x{L_y}x{L_z}"
                )
            else:
                legacy_keys = set(module_state.keys()) & {
                    "exposure_a", "exposure_b",
                }
                if legacy_keys:
                    logger.warning(
                        f"📷 v2 ckpt exposure_state has legacy keys "
                        f"{sorted(legacy_keys)} (old ExposureModel); "
                        f"eval will run without exposure correction "
                        f"(matches pre-T9.3 behaviour for v2 ckpts)."
                    )

        return Renderer(
            model=model,
            conf=conf,
            global_step=global_step,
            out_dir=out_dir,
            path=path,
            save_gt=save_gt,
            writer=writer,
            compute_extra_metrics=computes_extra_metrics,
            post_processing=post_processing,
            novel_view=novel_view,
            exposure_model=exposure_model,
            novel_fid=novel_fid,
            novel_save_n=novel_save_n,
        )

    @classmethod
    def from_preloaded_model(
        cls,
        model,
        out_dir,
        path="",
        save_gt=True,
        writer=None,
        global_step=None,
        compute_extra_metrics=False,
        post_processing=None,
        novel_view=False,
        exposure_model=None,
        novel_fid=False,
        novel_save_n: int = 5,
    ):
        """Loads checkpoint for test path.

        T9.3 / V3-P1.c: accepts ``exposure_model`` so the train-end eval
        path (trainer.py:1267-1277) can pass ``trainer.exposure_model``
        directly — keeps eval-time pred_rgb aligned with the train-time
        loss target. None = no-op (matches pre-T9.3 behavior).
        """

        conf = model.conf
        if global_step is None:
            global_step = ""
        model.build_acc()
        return Renderer(
            model=model,
            conf=conf,
            global_step=global_step,
            out_dir=out_dir,
            path=path,
            save_gt=save_gt,
            writer=writer,
            compute_extra_metrics=compute_extra_metrics,
            post_processing=post_processing,
            novel_view=novel_view,
            exposure_model=exposure_model,
            novel_fid=novel_fid,
            novel_save_n=novel_save_n,
        )

    @torch.no_grad()
    def render_all(self):
        """Render all the images in the test dataset and log the metrics."""

        # Criterions that we log during training
        criterions = {"psnr": PeakSignalNoiseRatio(data_range=1).to("cuda")}

        if self.compute_extra_metrics:
            criterions |= {
                "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda"),
                "lpips": LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to("cuda"),
            }

        # T8.5.7: optional camera subset filter (Hydra: render.eval_cameras=[...]).
        # None / empty → no filter, eval iterates the full test split unchanged.
        eval_cameras_filter = self.conf.render.get("eval_cameras", None)
        if eval_cameras_filter:
            eval_cameras_filter = list(eval_cameras_filter)
            logger.info(
                f"[V3-E4] render.eval_cameras filter active "
                f"({len(eval_cameras_filter)} cameras): {eval_cameras_filter}"
            )
        else:
            eval_cameras_filter = None

        output_path_renders = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "renders")
        os.makedirs(output_path_renders, exist_ok=True)

        if self.save_gt:
            output_path_gt = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "gt")
            os.makedirs(output_path_gt, exist_ok=True)

        psnr = []
        ssim = []
        lpips = []
        cc_psnr = []
        cc_ssim = []
        cc_lpips = []
        # V3-T15.2: DiFix post-process metrics. Populated only when
        # self.difix.enabled is True; otherwise the lists stay empty and
        # metrics.json gets no difix_* keys (byte-identical pre-T15.2).
        psnr_difix = []
        ssim_difix = []
        lpips_difix = []
        # T6F.2: masked 双指标（Stage 6-fix）—— Batch.mask 由 NCoreDataset
        # 注入 ego mask 后非 None；NeRF/Colmap 等 dataset mask=None 时直接
        # 复制全图值保证 byte-identical 回归. masked-cc 同理.
        psnr_masked = []
        ssim_masked = []
        lpips_masked = []
        cc_psnr_masked = []
        cc_ssim_masked = []
        cc_lpips_masked = []
        # T11.F1: LiDAR-domain depth PSNR accumulator. NaN frames (no LiDAR
        # coverage) are skipped so cameras without LiDAR don't poison the mean.
        lidar_psnrs: list = []
        inference_time = []
        # T8/B3 Phase E.6 — per-cuboid (per-class) PSNR. Records one entry per
        # active track per frame; only computed when ckpt is v2 LayeredGaussians
        # with populated tracks_poses AND the batch carries FTheta intrinsics.
        class_psnr_records: list = []

        # P0.2 / P0.3 — sseg-based per-class PSNR/LPIPS accumulators. Keys are
        # class names (person/rider/bicycle + road_crop); each maps to a list of
        # per-frame values. Populated only when the eval batch carries
        # semantic_sseg (NCore load_aux_masks=true) → absent on NeRF/Colmap so
        # metrics.json stays byte-identical for those datasets.
        per_class_eval_specs = {**DEFAULT_ACTOR_CLASS_SPECS, "road_crop": ROAD_CLASS_IDS}
        per_class_psnr: dict[str, list] = {}
        per_class_lpips: dict[str, list] = {}
        per_class_npix: dict[str, list] = {}

        # Phase 3 lane — 独立 lane 产物（semantic_lane_sseg）的候选指标累加器。
        # 限前视相机（risk L2：lane 最清处）。无 lane 产物 → 累加器空 →
        # metrics.json 字节等价（与 per_class 同样的"缺即不写"语义）。
        # 前视相机白名单（risk L2，lane 最清处）。可经 conf.render.lane_eval_cameras
        # override；默认仅前视宽角。
        lane_eval_cameras = list(self.conf.render.get("lane_eval_cameras", ["camera_front_wide_120fov"]))
        lane_band_px = int(self.conf.render.get("lane_band_px", DEFAULT_LANE_BAND_PX))
        lane_metric_keys = ("lane_band_lpips", "lane_band_psnr", "lane_raw_psnr", "lane_grad_corr")
        lane_acc: dict[str, list] = {}
        lane_npix_acc: list = []
        lane_band_npix_acc: list = []

        # T8.5.3 / V3-E3 — novel-view perturbation accumulator. Per-mode
        # LPIPS list (vs anchor GT). PSNR at these magnitudes is dominated
        # by parallax shift so we report LPIPS only (perceptual robust to
        # small content shifts). Empty when self.novel_view=False so
        # metrics.json stays byte-identical for the default eval path.
        from threedgrut.utils.novel_view import (
            LEGACY_NOVEL_AVG_MODES,
            NOVEL_VIEW_MODES,
            perturb_batch_shutter_pair_torch,
        )
        from threedgrut.model.plane_warp import build_plane_warp, warp_image
        novel_lpips: dict[str, list] = (
            {m: [] for m in NOVEL_VIEW_MODES} if self.novel_view else {}
        )
        # E2.1: save up to novel_save_n frames per mode (-1 = all frames).
        # Default novel_save_n=5 keeps historical visual-sample count.
        # Per-camera subdir naming + frames_map.json are written regardless
        # of n — they don't affect metrics.json (LPIPS etc.) at all.
        novel_save_first_n = (
            (self.novel_save_n if self.novel_save_n >= 0 else 10**9)
            if self.novel_view else 0
        )
        # E2.1: per-mode {ts:<cam>:<ts> -> relpath} maps, written after the loop.
        novel_frames_map: dict[str, dict] = {m: {} for m in NOVEL_VIEW_MODES} if self.novel_view else {}
        novel_save_counter: dict[str, int] = {m: 0 for m in NOVEL_VIEW_MODES} if self.novel_view else {}

        # E1.2 — NTA-IoU accumulators + lazy YOLO detector. Soft-fail: a
        # missing/undownloadable detector logs once and skips (no new keys →
        # metrics.json byte-identical, downstream comparators safe).
        nta_iou_records: list[dict] = []
        novel_nta_records: dict[str, list] = {}
        _nta_detector = None
        if bool(self.conf.render.get("enable_nta_iou", True)):
            try:
                from threedgrut.model.vehicle_detector import get_vehicle_detector
                _nta_detector = get_vehicle_detector(device="cuda")
            except Exception as e:  # ultralytics missing / weights offline
                logger.warning(f"[E1.2 NTA-IoU] detector unavailable, skipping: {e}")
                _nta_detector = None

        # E1.4 — FID/KID accumulators (accumulate-then-finalize): one
        # fid/kid pair for the interpolated render distribution and one per
        # novel mode (novel pairs only when novel_view is on). Soft-fail on
        # missing torch-fidelity backend → no keys, byte-identical.
        _fid_pairs = None
        _fid_counts: dict[str, int] = {}
        if self.novel_fid:
            try:
                from torchmetrics.image.fid import FrechetInceptionDistance
                from torchmetrics.image.kid import KernelInceptionDistance

                def _mk_fid_pair():
                    return {
                        "fid": FrechetInceptionDistance(feature=2048).to("cuda"),
                        # subset_size finalized via kid_subset_size(n) before
                        # compute() — constructor value is a placeholder.
                        "kid": KernelInceptionDistance(subset_size=2).to("cuda"),
                    }

                _fid_pairs = {"render": _mk_fid_pair()}
                if self.novel_view:
                    for _m in NOVEL_VIEW_MODES:
                        _fid_pairs[_m] = _mk_fid_pair()
                _fid_counts = {k: 0 for k in _fid_pairs}
                _fid_counts["real"] = 0
            except Exception as e:
                logger.warning(f"[E1.4 FID/KID] unavailable, skipping: {e}")
                _fid_pairs = None
        if self.novel_view:
            for m in NOVEL_VIEW_MODES:
                os.makedirs(
                    os.path.join(self.out_dir, f"ours_{int(self.global_step)}",
                                 "novel_view", m),
                    exist_ok=True,
                )
            logger.info(
                f"[T8.5.3 / V3-E3] novel-view mode ON — {len(NOVEL_VIEW_MODES)}"
                f" extra renders per anchor: {NOVEL_VIEW_MODES}"
            )

        # E1.1 — novel-view lane metrics via plane-induced warp. Lane paint
        # lies on the road surface, so the surface-induced warp is exact for
        # lane content: novel pixel ray → road height field hit → reproject
        # into the original camera → sample GT / lane mask as pseudo-GT.
        # Needs the road layer for the height field; absent → metric skipped
        # (soft-fail, same "missing → no keys" semantics as lane_acc above).
        # NOTE: warped metrics are comparable ACROSS models under the same
        # warp version only — not against interpolated mean_lane_* absolutes
        # (bilinear resampling smooths the warped GT slightly).
        novel_lane_acc: dict[str, dict[str, list]] = {}
        novel_lane_frames: dict[str, int] = {}
        novel_lane_valid_ratio: dict[str, list] = {}
        plane_warp_height_field = None
        if self.novel_view:
            _layers = getattr(self.model, "layers", None)
            if (_layers is not None and "road" in _layers
                    and _layers["road"].positions.numel() > 0):
                from threedgrut.model.road_region import build_road_height_field
                plane_warp_height_field = build_road_height_field(
                    _layers["road"].positions.detach(), cell_size=1.0,
                )
                logger.info(
                    f"[E1.1] plane-warp height field: "
                    f"{int(plane_warp_height_field['occupied'].sum())} occupied cells @ 1.0m"
                )
            else:
                logger.warning(
                    "[E1.1] no road layer on model — novel-view lane metrics skipped"
                )

        # T8.5.7 / V3-E4 — per-camera metric aggregation. keys = camera_id
        # strings (set by NCoreDataset's __getitem__); each value mirrors the
        # 12 global lists above so per-camera mean is straightforward. Empty
        # when dataset doesn't set Batch.camera_id (NeRF/Colmap path) →
        # metrics.json byte-identical to pre-V3-E4 for those datasets.
        per_cam: dict[str, dict[str, list]] = {}
        _per_cam_keys = (
            "psnr", "ssim", "lpips",
            "cc_psnr", "cc_ssim", "cc_lpips",
            "psnr_masked", "ssim_masked", "lpips_masked",
            "cc_psnr_masked", "cc_ssim_masked", "cc_lpips_masked",
        )

        best_psnr = -1.0
        worst_psnr = 2**16 * 1.0

        best_psnr_img = None
        best_psnr_img_gt = None

        worst_psnr_img = None
        worst_psnr_img_gt = None

        logger.start_progress(task_name="Rendering", total_steps=len(self.dataloader), color="orange1")

        for iteration, batch in enumerate(self.dataloader):

            # T8.5.7: skip frames not in the requested camera subset before
            # any GPU work. DataLoader collates the string camera_id into a
            # length-1 list under default collation; handle both forms.
            if eval_cameras_filter is not None:
                _bcid = batch.get("camera_id", None)
                if isinstance(_bcid, (list, tuple)):
                    _bcid = _bcid[0] if len(_bcid) > 0 else None
                if _bcid not in eval_cameras_filter:
                    continue

            # Get the GPU-cached batch
            gpu_batch = self.dataset.get_gpu_batch_with_intrinsics(batch)
            # T8.5.7: per-camera id (None on NeRF/Colmap, unused there)
            _cam_id = getattr(gpu_batch, "camera_id", None)

            # Compute the outputs of a single batch
            outputs = self.model(gpu_batch)

            # Apply post-processing
            if self.post_processing is not None:
                outputs = apply_post_processing(self.post_processing, outputs, gpu_batch, training=False)

            # T9.3 / V3-P1.c: apply BilateralGrid (or legacy ExposureModel)
            # color correction so eval-time pred_rgb matches the train-time
            # loss target (trainer.py:1641-1643 applies the same module
            # before computing photometric loss). Without this hook, raw
            # psnr_masked measured (raw_model_output vs GT) while training
            # optimized (bilateral_grid(model_output) vs GT) — the v2
            # ExposureModel退化 mode produced a +10.75 dB raw/cc gap at 30k.
            if self.exposure_model is not None:
                _cidx = getattr(gpu_batch, "camera_idx", None)
                if _cidx is not None:
                    outputs["pred_rgb"] = self.exposure_model(
                        int(_cidx), outputs["pred_rgb"],
                    )

            pred_rgb_full = outputs["pred_rgb"]
            rgb_gt_full = gpu_batch.rgb_gt

            # The values are already alpha composited with the background
            torchvision.utils.save_image(
                pred_rgb_full.squeeze(0).permute(2, 0, 1),
                os.path.join(output_path_renders, "{0:05d}".format(iteration) + ".png"),
            )
            pred_img_to_write = pred_rgb_full[-1].clip(0, 1.0)
            gt_img_to_write = rgb_gt_full[-1].clip(0, 1.0)

            if self.save_gt:
                torchvision.utils.save_image(
                    rgb_gt_full.squeeze(0).permute(2, 0, 1),
                    os.path.join(output_path_gt, "{0:05d}".format(iteration) + ".png"),
                )

            # Compute the loss
            psnr_single_img = criterions["psnr"](outputs["pred_rgb"], gpu_batch.rgb_gt).item()
            psnr.append(psnr_single_img)  # evaluation on valid rays only
            logger.info(f"Frame {iteration}, PSNR: {psnr[-1]}")

            if psnr_single_img > best_psnr:
                best_psnr = psnr_single_img
                best_psnr_img = pred_img_to_write
                best_psnr_img_gt = gt_img_to_write

            if psnr_single_img < worst_psnr:
                worst_psnr = psnr_single_img
                worst_psnr_img = pred_img_to_write
                worst_psnr_img_gt = gt_img_to_write

            # evaluate on full image
            ssim.append(
                criterions["ssim"](
                    pred_rgb_full.permute(0, 3, 1, 2),
                    rgb_gt_full.permute(0, 3, 1, 2),
                ).item()
            )
            lpips.append(
                criterions["lpips"](
                    pred_rgb_full.clip(0, 1).permute(0, 3, 1, 2),
                    rgb_gt_full.permute(0, 3, 1, 2),
                ).item()
            )

            # V3-T15.2: optional DiFix post-process metrics. Runs against the
            # raw pred_rgb (not color-corrected) so the comparison is the pure
            # "diffusion fixer Δ" rather than mixed with affine cc. Skipped
            # entirely when use_difix=false → no GPU work, byte-identical with
            # pre-T15.2 metrics.json.
            if self.difix.enabled:
                pred_rgb_difix = self.difix(pred_rgb_full)
                psnr_difix.append(
                    criterions["psnr"](pred_rgb_difix, rgb_gt_full).item()
                )
                ssim_difix.append(
                    criterions["ssim"](
                        pred_rgb_difix.permute(0, 3, 1, 2),
                        rgb_gt_full.permute(0, 3, 1, 2),
                    ).item()
                )
                lpips_difix.append(
                    criterions["lpips"](
                        pred_rgb_difix.clip(0, 1).permute(0, 3, 1, 2),
                        rgb_gt_full.permute(0, 3, 1, 2),
                    ).item()
                )

            # Color-corrected metrics
            pred_rgb_cc = color_correct_affine(pred_rgb_full, rgb_gt_full)
            cc_psnr.append(criterions["psnr"](pred_rgb_cc, rgb_gt_full).item())
            cc_ssim.append(
                criterions["ssim"](
                    pred_rgb_cc.permute(0, 3, 1, 2),
                    rgb_gt_full.permute(0, 3, 1, 2),
                ).item()
            )
            cc_lpips.append(
                criterions["lpips"](
                    pred_rgb_cc.clip(0, 1).permute(0, 3, 1, 2),
                    rgb_gt_full.permute(0, 3, 1, 2),
                ).item()
            )

            # T6F.2: masked PSNR / SSIM / LPIPS (raw + cc)
            # mask=None (NeRF/Colmap 等): masked 指标 ≡ 全图指标 (byte-identical 回归)
            # mask 非 None (NCore ego mask): PSNR_masked 解析公式 + SSIM/LPIPS GT-fill 近似
            mask = gpu_batch.mask  # [B, H, W, 1] 或 None
            if mask is not None:
                mask = mask.to(pred_rgb_full.dtype)
                # PSNR_masked (raw)
                diff_sq = (pred_rgb_full - rgb_gt_full).pow(2) * mask
                denom = mask.sum().clamp(min=1.0) * 3
                mse_masked = diff_sq.sum() / denom
                psnr_masked.append(
                    (-10.0 * torch.log10(mse_masked.clamp(min=1e-10))).item()
                )
                # PSNR_masked (cc)
                diff_sq_cc = (pred_rgb_cc - rgb_gt_full).pow(2) * mask
                mse_masked_cc = diff_sq_cc.sum() / denom
                cc_psnr_masked.append(
                    (-10.0 * torch.log10(mse_masked_cc.clamp(min=1e-10))).item()
                )
                # SSIM / LPIPS via GT-fill
                m4d = mask.permute(0, 3, 1, 2)  # [B, 1, H, W]
                rgb_gt_perm = rgb_gt_full.permute(0, 3, 1, 2)
                pred_perm = pred_rgb_full.permute(0, 3, 1, 2)
                pred_perm_clipped = pred_rgb_full.clip(0, 1).permute(0, 3, 1, 2)
                pred_cc_perm = pred_rgb_cc.permute(0, 3, 1, 2)
                pred_cc_perm_clipped = pred_rgb_cc.clip(0, 1).permute(0, 3, 1, 2)
                pred_filled = pred_perm * m4d + rgb_gt_perm * (1.0 - m4d)
                pred_filled_clipped = pred_perm_clipped * m4d + rgb_gt_perm * (1.0 - m4d)
                pred_cc_filled = pred_cc_perm * m4d + rgb_gt_perm * (1.0 - m4d)
                pred_cc_filled_clipped = pred_cc_perm_clipped * m4d + rgb_gt_perm * (1.0 - m4d)
                ssim_masked.append(criterions["ssim"](pred_filled, rgb_gt_perm).item())
                lpips_masked.append(criterions["lpips"](pred_filled_clipped, rgb_gt_perm).item())
                cc_ssim_masked.append(criterions["ssim"](pred_cc_filled, rgb_gt_perm).item())
                cc_lpips_masked.append(criterions["lpips"](pred_cc_filled_clipped, rgb_gt_perm).item())
            else:
                # byte-identical 回归：直接复制全图值
                psnr_masked.append(psnr[-1])
                ssim_masked.append(ssim[-1])
                lpips_masked.append(lpips[-1])
                cc_psnr_masked.append(cc_psnr[-1])
                cc_ssim_masked.append(cc_ssim[-1])
                cc_lpips_masked.append(cc_lpips[-1])

            # T11.F1: LiDAR-domain depth PSNR per frame. Requires pred_dist in
            # outputs AND lidar_depth_map in gpu_batch.image_infos (NCore only;
            # NeRF/Colmap batches have neither → lidar_psnrs stays empty →
            # mean_lidar_psnr absent from metrics.json, byte-identical for those
            # datasets). NaN returned when a frame has 0 valid hit pixels →
            # skip so it doesn't pull the mean down to -inf.
            _image_infos = getattr(gpu_batch, "image_infos", None) or {}
            _pred_dist = outputs.get("pred_dist") if isinstance(outputs, dict) else None
            if (
                _pred_dist is not None
                and isinstance(_image_infos, dict)
                and "lidar_depth_map" in _image_infos
            ):
                _lidar_gt = _image_infos["lidar_depth_map"]  # [B, H, W]
                _hit = (_lidar_gt > 0).float()
                _lp = compute_lidar_psnr(_pred_dist, _lidar_gt, _hit)
                if _lp == _lp:  # not NaN
                    lidar_psnrs.append(_lp)

            # T8.5.3 / V3-E3 — novel-view perturbation renders. For each
            # mode, mutate gpu_batch.T_to_world + T_to_world_end with the
            # SAME world-frame delta (rolling-shutter rigid trajectory) and
            # re-render. LPIPS vs anchor GT only; PSNR omitted intentionally
            # — at ±1-2m / ±5-10° magnitudes the image content shifts enough
            # that PSNR is dominated by parallax noise floor, not view-
            # extrapolation quality. Skipped entirely when self.novel_view
            # is False (metrics.json stays byte-identical with pre-T8.5.3).
            # E1.4 — feed the frame into every FID/KID pair: GT is the
            # shared real distribution; the interpolated render is the
            # "render" fake; novel fakes are fed inside the loop below.
            if _fid_pairs is not None:
                _real_u8 = rgb01_to_uint8_chw(rgb_gt_full)
                _fake_u8 = rgb01_to_uint8_chw(pred_rgb_full)
                for _pk, _pair in _fid_pairs.items():
                    _pair["fid"].update(_real_u8, real=True)
                    _pair["kid"].update(_real_u8, real=True)
                _fid_pairs["render"]["fid"].update(_fake_u8, real=False)
                _fid_pairs["render"]["kid"].update(_fake_u8, real=False)
                _fid_counts["real"] += 1
                _fid_counts["render"] += 1

            # E1.2 — resolve GT tracks once per batch; consumed by BOTH the
            # novel-view NTA-IoU (inside the loop below) and the interpolated
            # class_psnr / NTA-IoU block further down. Cuboids stay at their
            # world-frame GT pose for novel views (timestamp unchanged).
            tp = getattr(self.model, "tracks_poses", None)
            ftheta_params = getattr(
                gpu_batch, "intrinsics_FThetaCameraModelParameters", None,
            )
            active = []
            if tp and ftheta_params is not None and hasattr(self.model, "_resolve_pose_idx"):
                _pose_idx = self.model._resolve_pose_idx(
                    int(getattr(gpu_batch, "timestamp_us", -1)),
                    int(getattr(gpu_batch, "frame_idx", -1)) if int(getattr(gpu_batch, "frame_idx", -1)) >= 0 else None,
                )
                active = collect_active_tracks_for_frame(
                    self.model.tracks_poses,
                    self.model.tracks_active,
                    getattr(self.model, "tracks_metadata", {}),
                    _pose_idx,
                )

            if self.novel_view:
                orig_T = gpu_batch.T_to_world.detach().clone()
                orig_T_end = gpu_batch.T_to_world_end.detach().clone()
                rgb_gt_perm = rgb_gt_full.permute(0, 3, 1, 2)
                for mode in NOVEL_VIEW_MODES:
                    nT, nTe = perturb_batch_shutter_pair_torch(
                        orig_T, orig_T_end, mode,
                    )
                    gpu_batch.T_to_world = nT
                    gpu_batch.T_to_world_end = nTe
                    out_novel = self.model(gpu_batch)
                    pred_novel = out_novel["pred_rgb"]
                    novel_lpips[mode].append(
                        criterions["lpips"](
                            pred_novel.clip(0, 1).permute(0, 3, 1, 2),
                            rgb_gt_perm,
                        ).item()
                    )
                    if iteration < novel_save_first_n:
                        from threedgrut.utils.novel_view import (
                            novel_frame_key, novel_frame_relpath,
                        )
                        _cam = str(getattr(gpu_batch, "camera_id", "cam0"))
                        _ts = int(getattr(gpu_batch, "timestamp_us", -1))
                        _sidx = novel_save_counter[mode]
                        novel_save_counter[mode] += 1
                        _rel = novel_frame_relpath(_cam, _sidx)
                        _dst = os.path.join(
                            self.out_dir, f"ours_{int(self.global_step)}",
                            "novel_view", mode, _rel,
                        )
                        os.makedirs(os.path.dirname(_dst), exist_ok=True)
                        torchvision.utils.save_image(
                            pred_novel.squeeze(0).permute(2, 0, 1), _dst,
                        )
                        if _ts >= 0:
                            novel_frames_map[mode][novel_frame_key(_cam, _ts)] = _rel

                    # E1.1 — lane metrics at the novel pose via plane-induced
                    # warp (all 6 modes; yaw warps are equally exact for
                    # road-plane content). Same camera whitelist as the
                    # interpolated lane block; rays must be camera-frame
                    # (NCore default) for build_plane_warp.
                    _lane_nv = (getattr(gpu_batch, "image_infos", None) or {}).get(
                        "semantic_lane_sseg"
                    )
                    if (
                        plane_warp_height_field is not None
                        and _lane_nv is not None
                        and _cam_id in lane_eval_cameras
                        and ftheta_params is not None
                        and not getattr(gpu_batch, "rays_in_world_space", False)
                    ):
                        lane_one_nv = _lane_nv[0] if _lane_nv.dim() == 3 else _lane_nv
                        wgrid, wvalid = build_plane_warp(
                            gpu_batch.rays_dir[0], nT[0], orig_T[0], ftheta_params,
                            height_field=plane_warp_height_field,
                        )
                        gt_warp = warp_image(rgb_gt_full[0].float(), wgrid, wvalid)
                        lane_warp = warp_image(
                            lane_one_nv.unsqueeze(-1).float(), wgrid, wvalid,
                            mode="nearest",
                        )[..., 0].long()
                        lm_nv = compute_lane_metrics(
                            pred_novel[0], gt_warp, lane_warp, LANE_CLASS_IDS,
                            band_px=lane_band_px, restrict_mask=wvalid,
                            lpips_fn=criterions.get("lpips"),
                        )
                        _acc = novel_lane_acc.setdefault(mode, {})
                        for _k in lane_metric_keys:
                            if lm_nv[_k] is not None:
                                _acc.setdefault(_k, []).append(lm_nv[_k])
                        novel_lane_frames[mode] = novel_lane_frames.get(mode, 0) + 1
                        novel_lane_valid_ratio.setdefault(mode, []).append(
                            float(wvalid.float().mean())
                        )

                    # E1.2 — NTA-IoU at the novel pose: GT cuboids stay at
                    # their world-frame GT pose (timestamp unchanged), only
                    # the camera moves → T_w2c = inv(perturbed c2w).
                    if _nta_detector is not None and active:
                        from threedgrut.model.nta_iou import compute_frame_nta_iou
                        H_nv = int(pred_novel.shape[1])
                        W_nv = int(pred_novel.shape[2])
                        nta_nv = compute_frame_nta_iou(
                            pred_novel[0], active, _nta_detector,
                            K=None, ftheta_params=ftheta_params,
                            T_w2c=torch.linalg.inv(nT[0]), H=H_nv, W=W_nv,
                        )
                        if nta_nv is not None:
                            novel_nta_records.setdefault(mode, []).append(nta_nv)

                    # E1.4 — novel-mode render joins its mode's fake set.
                    if _fid_pairs is not None and mode in _fid_pairs:
                        _nv_u8 = rgb01_to_uint8_chw(pred_novel)
                        _fid_pairs[mode]["fid"].update(_nv_u8, real=False)
                        _fid_pairs[mode]["kid"].update(_nv_u8, real=False)
                        _fid_counts[mode] += 1
                # Restore originals so downstream class_psnr / per_cam see
                # the unperturbed batch.
                gpu_batch.T_to_world = orig_T
                gpu_batch.T_to_world_end = orig_T_end

            # T8/B3 Phase E.6 — per-cuboid class PSNR. Skipped when the model
            # has no dyn tracks loaded (single-bg / road-only multi-layer) OR
            # the batch lacks FTheta intrinsics (current dyn projection only
            # supports FTheta to match training-side cuboid mask path).
            # (tracks resolved above, before the novel-view block — E1.2)
            if active:
                T_w2c = torch.linalg.inv(gpu_batch.T_to_world[0])
                H_, W_ = int(pred_rgb_full.shape[1]), int(pred_rgb_full.shape[2])
                cp = compute_class_psnr(
                    pred_rgb_full, rgb_gt_full, mask,
                    active, T_world2cam=T_w2c, H=H_, W=W_,
                    ftheta_params=ftheta_params,
                )
                for r in cp["per_track"]:
                    r["frame"] = int(iteration)
                    class_psnr_records.append(r)

                # E1.2 — interpolated-view NTA-IoU (same active/T_w2c as
                # class_psnr; FTheta clip → K=None).
                if _nta_detector is not None:
                    from threedgrut.model.nta_iou import compute_frame_nta_iou
                    nta = compute_frame_nta_iou(
                        pred_rgb_full[0], active, _nta_detector,
                        K=None, ftheta_params=ftheta_params,
                        T_w2c=T_w2c, H=H_, W=W_,
                    )
                    if nta is not None:
                        nta_iou_records.append(nta)

            # P0.2 / P0.3 — sseg-based per-class PSNR/LPIPS. Reads the raw sseg
            # id map passed through image_infos (load_aux_masks=true) and derives
            # person/rider/bicycle (P0.2) + road_crop (P0.3) masks per frame.
            # Uses raw pred (matching class_psnr above); LPIPS is GT-filled
            # inside the helper. Skipped when semantic_sseg absent → no keys →
            # byte-identical for NeRF/Colmap.
            _sseg = (getattr(gpu_batch, "image_infos", None) or {}).get("semantic_sseg")
            if _sseg is not None:
                sseg_one = _sseg[0] if _sseg.dim() == 3 else _sseg  # [H, W]
                pcm = compute_per_class_metrics(
                    pred_rgb_full[0], rgb_gt_full[0], sseg_one,
                    per_class_eval_specs, lpips_fn=criterions.get("lpips"),
                )
                for _name, _d in pcm.items():
                    if _d["psnr"] is not None:
                        per_class_psnr.setdefault(_name, []).append(_d["psnr"])
                    if _d["lpips"] is not None:
                        per_class_lpips.setdefault(_name, []).append(_d["lpips"])
                    per_class_npix.setdefault(_name, []).append(_d["n_pixels"])

            # Phase 3 lane：独立 lane 产物 + 膨胀 band 指标。限前视相机。
            # 缺 semantic_lane_sseg（未开 load_lane_masks / 非前视）→ 跳过 → 无新字段。
            _lane = (getattr(gpu_batch, "image_infos", None) or {}).get("semantic_lane_sseg")
            if _lane is not None and _cam_id in lane_eval_cameras:
                lane_one = _lane[0] if _lane.dim() == 3 else _lane  # [H, W]
                lm = compute_lane_metrics(
                    pred_rgb_full[0], rgb_gt_full[0], lane_one, LANE_CLASS_IDS,
                    band_px=lane_band_px, lpips_fn=criterions.get("lpips"),
                )
                for _k in lane_metric_keys:
                    if lm[_k] is not None:
                        lane_acc.setdefault(_k, []).append(lm[_k])
                lane_npix_acc.append(lm["lane_n_pixels"])
                lane_band_npix_acc.append(lm["lane_band_n_pixels"])

            # T8.5.7 / V3-E4: mirror the 12 metric lists into the per-camera
            # dict using the last appended value of each. Skipped when the
            # dataset doesn't set Batch.camera_id (NeRF/Colmap) so old
            # metrics.json stays byte-identical.
            if _cam_id is not None:
                pc = per_cam.setdefault(_cam_id, {k: [] for k in _per_cam_keys})
                pc["psnr"].append(psnr[-1])
                pc["ssim"].append(ssim[-1])
                pc["lpips"].append(lpips[-1])
                pc["cc_psnr"].append(cc_psnr[-1])
                pc["cc_ssim"].append(cc_ssim[-1])
                pc["cc_lpips"].append(cc_lpips[-1])
                pc["psnr_masked"].append(psnr_masked[-1])
                pc["ssim_masked"].append(ssim_masked[-1])
                pc["lpips_masked"].append(lpips_masked[-1])
                pc["cc_psnr_masked"].append(cc_psnr_masked[-1])
                pc["cc_ssim_masked"].append(cc_ssim_masked[-1])
                pc["cc_lpips_masked"].append(cc_lpips_masked[-1])

            # Record the time
            inference_time.append(outputs["frame_time_ms"])

            logger.log_progress(task_name="Rendering", advance=1, iteration=f"{str(iteration)}", psnr=psnr[-1])

        logger.end_progress(task_name="Rendering")

        # T8.5.7: sanity-check the eval_cameras filter — if the user passes a
        # subset and 0 frames matched, fail loudly instead of writing an
        # empty metrics.json (which would silently corrupt the comparison).
        if eval_cameras_filter is not None and len(psnr) == 0:
            raise RuntimeError(
                f"[V3-E4] render.eval_cameras={eval_cameras_filter} matched 0 frames in "
                f"the test split. Check camera_id spelling against dataset.camera_ids."
            )

        mean_psnr = np.mean(psnr)
        mean_ssim = np.mean(ssim)
        mean_lpips = np.mean(lpips)
        mean_cc_psnr = np.mean(cc_psnr)
        mean_cc_ssim = np.mean(cc_ssim)
        mean_cc_lpips = np.mean(cc_lpips)
        std_psnr = np.std(psnr)
        # T6F.2: masked aggregates
        mean_psnr_masked = np.mean(psnr_masked)
        mean_ssim_masked = np.mean(ssim_masked)
        mean_lpips_masked = np.mean(lpips_masked)
        mean_cc_psnr_masked = np.mean(cc_psnr_masked)
        mean_cc_ssim_masked = np.mean(cc_ssim_masked)
        mean_cc_lpips_masked = np.mean(cc_lpips_masked)
        mean_inference_time = np.mean(inference_time)

        table = dict(
            mean_psnr=mean_psnr,
            mean_ssim=mean_ssim,
            mean_lpips=mean_lpips,
            mean_cc_psnr=mean_cc_psnr,
            mean_cc_ssim=mean_cc_ssim,
            mean_cc_lpips=mean_cc_lpips,
            std_psnr=std_psnr,
            # T6F.2 双指标：与全图列并排
            mean_psnr_masked=mean_psnr_masked,
            mean_cc_psnr_masked=mean_cc_psnr_masked,
        )

        if self.conf.render.enable_kernel_timings:
            table["mean_inference_time"] = f"{'{:.2f}'.format(mean_inference_time)}" + " ms/frame"

        # Save metrics to JSON file
        metrics_json = dict(
            mean_psnr=float(mean_psnr),
            mean_ssim=float(mean_ssim),
            mean_lpips=float(mean_lpips),
            mean_cc_psnr=float(mean_cc_psnr),
            mean_cc_ssim=float(mean_cc_ssim),
            mean_cc_lpips=float(mean_cc_lpips),
            # T6F.2 双指标全量进 metrics.json (table 列受宽度限制只显部分)
            mean_psnr_masked=float(mean_psnr_masked),
            mean_ssim_masked=float(mean_ssim_masked),
            mean_lpips_masked=float(mean_lpips_masked),
            mean_cc_psnr_masked=float(mean_cc_psnr_masked),
            mean_cc_ssim_masked=float(mean_cc_ssim_masked),
            mean_cc_lpips_masked=float(mean_cc_lpips_masked),
        )
        # E2.1 — write per-mode frames_map.json so eval_frames_dir can join
        # rendered novel frames by timestamp. Only written when frames were
        # saved (novel_view=True and novel_save_n != 0); no effect on
        # metrics.json content (pure side-output).
        if self.novel_view:
            import json as _json
            for _m, _fm in novel_frames_map.items():
                if _fm:
                    _mp = os.path.join(self.out_dir, f"ours_{int(self.global_step)}",
                                       "novel_view", _m, "frames_map.json")
                    with open(_mp, "w") as _f:
                        _json.dump(_fm, _f)

        # T8.5.3 / V3-E3 — per-mode novel-view LPIPS averaged across all
        # eval frames. Only populated when self.novel_view=True; absent
        # otherwise so metrics.json stays byte-identical for the default
        # path. Also report the 4-mode mean as the v3 baseline single
        # number.
        if novel_lpips and any(novel_lpips.values()):
            nvjson: dict[str, float] = {}
            means = []
            legacy_means = []
            for mode in NOVEL_VIEW_MODES:
                lst = novel_lpips[mode]
                if lst:
                    m = float(np.mean(lst))
                    nvjson[f"mean_novel_lpips_{mode}"] = m
                    means.append(m)
                    if mode in LEGACY_NOVEL_AVG_MODES:
                        legacy_means.append(m)
            # E1.1: mean_novel_lpips_avg aggregates ONLY the legacy 4 modes —
            # the v3 anchor (B3 0.5962) depends on this exact field meaning.
            # The 6-mode aggregate (incl. lateral_3m/6m) goes to _avg6.
            if legacy_means:
                nvjson["mean_novel_lpips_avg"] = float(np.mean(legacy_means))
            if means:
                nvjson["mean_novel_lpips_avg6"] = float(np.mean(means))
            metrics_json.update(nvjson)
            logger.info(
                f"[T8.5.3 / V3-E3] novel-view LPIPS — "
                + " | ".join(
                    f"{m}={novel_lpips and np.mean(novel_lpips[m]):.4f}"
                    for m in NOVEL_VIEW_MODES if novel_lpips[m]
                )
            )

        # E1.1 — novel-view lane metric aggregation (plane-induced warp
        # pseudo-GT). Absent when no lane product / no road layer / novel off
        # → metrics.json byte-identical (same soft-fail semantics as lane_*).
        if novel_lane_frames:
            for mode in NOVEL_VIEW_MODES:
                if mode not in novel_lane_frames:
                    continue
                _acc = novel_lane_acc.get(mode, {})
                for _k in lane_metric_keys:
                    _v = _acc.get(_k, [])
                    metrics_json[f"mean_novel_{_k}_{mode}"] = (
                        float(np.mean(_v)) if _v else None
                    )
                metrics_json[f"novel_lane_n_records_{mode}"] = int(
                    novel_lane_frames[mode]
                )
                metrics_json[f"novel_lane_warp_valid_ratio_{mode}"] = float(
                    np.mean(novel_lane_valid_ratio[mode])
                )
            _gc = {
                m: metrics_json.get(f"mean_novel_lane_grad_corr_{m}")
                for m in NOVEL_VIEW_MODES if m in novel_lane_frames
            }
            logger.info(
                "[E1.1] novel-view lane grad_corr — "
                + " | ".join(
                    f"{m}={v:.4f}" for m, v in _gc.items() if v is not None
                )
            )

        # V3-T15.2: append DiFix aggregates only when the lists were populated
        # (i.e. self.difix.enabled was true on this run). Otherwise no key is
        # written, keeping metrics.json byte-identical with pre-T15.2.
        if psnr_difix:
            metrics_json["mean_psnr_difix"] = float(np.mean(psnr_difix))
            metrics_json["mean_ssim_difix"] = float(np.mean(ssim_difix))
            metrics_json["mean_lpips_difix"] = float(np.mean(lpips_difix))
            table["mean_psnr_difix"] = float(np.mean(psnr_difix))

        # T8.5.7 / V3-E4 — per-camera aggregated metrics. ``per_cam`` is
        # empty for NeRF/Colmap (Batch.camera_id is None there), so
        # metrics.json stays byte-identical on those paths. NCore eval
        # always populates it (both train and val branches set camera_id
        # in __getitem__).
        if per_cam:
            per_camera_summary: dict[str, dict] = {}
            for cid, dlists in per_cam.items():
                n = len(dlists["psnr"])
                if n == 0:
                    continue
                per_camera_summary[cid] = {
                    "n_frames": int(n),
                    "mean_psnr": float(np.mean(dlists["psnr"])),
                    "mean_ssim": float(np.mean(dlists["ssim"])),
                    "mean_lpips": float(np.mean(dlists["lpips"])),
                    "mean_cc_psnr": float(np.mean(dlists["cc_psnr"])),
                    "mean_cc_ssim": float(np.mean(dlists["cc_ssim"])),
                    "mean_cc_lpips": float(np.mean(dlists["cc_lpips"])),
                    "mean_psnr_masked": float(np.mean(dlists["psnr_masked"])),
                    "mean_ssim_masked": float(np.mean(dlists["ssim_masked"])),
                    "mean_lpips_masked": float(np.mean(dlists["lpips_masked"])),
                    "mean_cc_psnr_masked": float(np.mean(dlists["cc_psnr_masked"])),
                    "mean_cc_ssim_masked": float(np.mean(dlists["cc_ssim_masked"])),
                    "mean_cc_lpips_masked": float(np.mean(dlists["cc_lpips_masked"])),
                }
            if per_camera_summary:
                metrics_json["per_camera"] = per_camera_summary
                # Compact per-camera table for the console — show the masked
                # variants which are the v2 multilayer KPI proxies.
                pc_table = {
                    cid: (
                        f"n={m['n_frames']} "
                        f"psnr_m={m['mean_psnr_masked']:.2f} "
                        f"cc_psnr_m={m['mean_cc_psnr_masked']:.2f}"
                    )
                    for cid, m in per_camera_summary.items()
                }
                logger.log_table(
                    f"📷 Per-Camera Metrics - Step {self.global_step}", record=pc_table
                )

        # T8/B3 Phase E.6 — append per-cuboid PSNR aggregates when available.
        # ``class_psnr_records`` stays empty when the ckpt has no dyn tracks
        # or the dataset doesn't carry FTheta intrinsics (NeRF / Colmap eval
        # paths) — keep metrics.json byte-identical with pre-E.6 in that case.
        cp_values = [r["psnr"] for r in class_psnr_records if r["psnr"] is not None]
        if cp_values:
            by_class: dict[str, list] = {}
            for r in class_psnr_records:
                if r["psnr"] is not None:
                    by_class.setdefault(r["class"], []).append(r["psnr"])
            metrics_json["mean_class_psnr"] = float(np.mean(cp_values))
            metrics_json["class_psnr_by_class"] = {
                cls: float(np.mean(vals)) for cls, vals in by_class.items()
            }
            metrics_json["class_psnr_n_records"] = int(len(cp_values))
            metrics_json["class_psnr_n_low_15db"] = int(
                sum(1 for v in cp_values if v < 15.0)
            )
            table["mean_class_psnr"] = float(np.mean(cp_values))

        # P0.2 / P0.3 — per-class (sseg-based) PSNR/LPIPS aggregates. Every
        # spec'd class is written even when absent from all frames
        # (n_records=0, mean=None) so "measured, not present" (the pedestrian
        # floor) is distinguishable from "not measured". Whole block absent on
        # NeRF/Colmap (no semantic_sseg) → metrics.json byte-identical there.
        if per_class_npix:
            for _name in per_class_eval_specs:
                _pv = per_class_psnr.get(_name, [])
                _lv = per_class_lpips.get(_name, [])
                _nv = per_class_npix.get(_name, [])
                metrics_json[f"mean_{_name}_psnr"] = float(np.mean(_pv)) if _pv else None
                metrics_json[f"mean_{_name}_lpips"] = float(np.mean(_lv)) if _lv else None
                metrics_json[f"{_name}_n_records"] = int(len(_pv))
                metrics_json[f"{_name}_total_pixels"] = int(np.sum(_nv)) if _nv else 0

        # E1.4 — FID/KID finalize. KID subset_size adapted to the actual
        # sample count (legal: subset_size <= n). compute() wrapped so a
        # degenerate split degrades to missing keys, not a crashed eval.
        if _fid_pairs is not None and _fid_counts.get("real", 0) >= 2:
            for _pk, _pair in _fid_pairs.items():
                _n_fake = _fid_counts.get(_pk, 0)
                if _n_fake < 2:
                    continue
                # Key naming per plan §6: mean_render_fid / mean_novel_fid_<mode>
                # (metric name BEFORE mode, consistent with mean_novel_lpips_<mode>).
                if _pk == "render":
                    _k_fid, _k_kid, _k_kid_std = (
                        "mean_render_fid", "mean_render_kid", "mean_render_kid_std",
                    )
                else:
                    _k_fid, _k_kid, _k_kid_std = (
                        f"mean_novel_fid_{_pk}", f"mean_novel_kid_{_pk}",
                        f"mean_novel_kid_std_{_pk}",
                    )
                try:
                    metrics_json[_k_fid] = float(_pair["fid"].compute())
                except Exception as e:
                    logger.warning(f"[E1.4] FID compute failed for {_pk}: {e}")
                try:
                    _pair["kid"].subset_size = kid_subset_size(
                        min(_fid_counts["real"], _n_fake)
                    )
                    _kid_mean, _kid_std = _pair["kid"].compute()
                    metrics_json[_k_kid] = float(_kid_mean)
                    metrics_json[_k_kid_std] = float(_kid_std)
                except Exception as e:
                    logger.warning(f"[E1.4] KID compute failed for {_pk}: {e}")
                metrics_json[f"fid_n_fake_{_pk}"] = int(_n_fake)
            metrics_json["fid_n_real"] = int(_fid_counts["real"])
            _fid_log = {k: v for k, v in metrics_json.items()
                        if k.startswith("mean_") and ("_fid" in k or "_kid" in k)}
            logger.info(f"[E1.4] FID/KID — {_fid_log}")

        # E1.2 — NTA-IoU aggregation (interpolated + per novel mode). Keys
        # absent when the detector was unavailable or no vehicle frames →
        # metrics.json byte-identical for those runs.
        if nta_iou_records:
            _vals = [r["mean_nta_iou"] for r in nta_iou_records]
            metrics_json["mean_nta_iou"] = float(np.mean(_vals))
            metrics_json["nta_iou_n_frames"] = int(len(_vals))
        if novel_nta_records:
            for mode in NOVEL_VIEW_MODES:
                _recs = novel_nta_records.get(mode, [])
                if _recs:
                    metrics_json[f"mean_novel_nta_iou_{mode}"] = float(
                        np.mean([r["mean_nta_iou"] for r in _recs])
                    )
                    metrics_json[f"novel_nta_iou_n_frames_{mode}"] = int(len(_recs))
            logger.info(
                "[E1.2] NTA-IoU — interp="
                + (f"{metrics_json.get('mean_nta_iou'):.4f}"
                   if nta_iou_records else "n/a")
                + " | "
                + " | ".join(
                    f"{m}={metrics_json[f'mean_novel_nta_iou_{m}']:.4f}"
                    for m in NOVEL_VIEW_MODES
                    if f"mean_novel_nta_iou_{m}" in metrics_json
                )
            )

        # Phase 3 lane 指标聚合。仅当 lane 产物被加载（lane_npix_acc 非空）→
        # 否则整块缺省，metrics.json 字节等价（守护线零回归）。
        if lane_npix_acc:
            for _k in lane_metric_keys:
                _v = lane_acc.get(_k, [])
                metrics_json[f"mean_{_k}"] = float(np.mean(_v)) if _v else None
            # 注：lane_n_records = 评测的前视帧数（含 lane 像素=0 的帧）；与各
            # mean_lane_* 的分母（仅该指标非 None 的帧）可能不同——前者是"测了多少
            # 前视帧"，后者是"其中多少帧该指标有效"。下游 A/B 脚本勿混淆二者。
            metrics_json["lane_n_records"] = int(len(lane_npix_acc))
            metrics_json["lane_total_pixels"] = int(np.sum(lane_npix_acc))
            metrics_json["lane_band_total_pixels"] = int(np.sum(lane_band_npix_acc))

        # T11.F1: mean LiDAR-domain depth PSNR. Only written when at least one
        # frame had valid LiDAR coverage (lidar_psnrs non-empty); absent on
        # NeRF/Colmap paths → metrics.json byte-identical for those datasets.
        if lidar_psnrs:
            metrics_json["mean_lidar_psnr"] = float(np.mean(lidar_psnrs))
            table["mean_lidar_psnr"] = float(np.mean(lidar_psnrs))
            logger.info(
                f"[T11.F1] mean_lidar_psnr={metrics_json['mean_lidar_psnr']:.3f} dB"
                f" over {len(lidar_psnrs)} frames"
            )

        # V3-L5/L8/L9 — diagnostic fields. Written even when the toggles are
        # OFF (as ``null``) so downstream A/B-comparison scripts can rely on
        # the keys being present in every metrics.json.
        albedo_t = getattr(self.model, "_track_albedo_table", None)
        log_scale_t = getattr(self.model, "_track_log_scale_table", None)
        # symmetric_axis lives on the dynamic_rigids LayerSpec.extra.
        sym_axis_val = None
        specs = getattr(self.model, "specs", None)
        if specs is not None:
            dyn = next((s for s in specs if s.name == "dynamic_rigids"), None)
            if dyn is not None:
                sym_axis_val = (getattr(dyn, "extra", {}) or {}).get("symmetric_axis")
        metrics_json["symmetric_axis"] = sym_axis_val  # 'Y' / 'X' / 'Z' / null
        metrics_json["track_albedo_l2_mean"] = (
            float(albedo_t.detach().norm(dim=-1).mean().cpu())
            if albedo_t is not None else None
        )
        metrics_json["track_log_scale_mean"] = (
            float(log_scale_t.detach().mean().cpu())
            if log_scale_t is not None else None
        )
        metrics_json["track_log_scale_std"] = (
            float(log_scale_t.detach().std().cpu())
            if log_scale_t is not None and log_scale_t.numel() > 1 else None
        )

        metrics_path = os.path.join(self.out_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_json, f, indent=2)
        logger.info(f"📄 Metrics saved to: {metrics_path}")

        logger.log_table(f"⭐ Test Metrics - Step {self.global_step}", record=table)

        if self.writer is not None:
            self.writer.add_scalar("psnr/test", mean_psnr, self.global_step)
            self.writer.add_scalar("ssim/test", mean_ssim, self.global_step)
            self.writer.add_scalar("lpips/test", mean_lpips, self.global_step)
            self.writer.add_scalar("cc_psnr/test", mean_cc_psnr, self.global_step)
            self.writer.add_scalar("cc_ssim/test", mean_cc_ssim, self.global_step)
            self.writer.add_scalar("cc_lpips/test", mean_cc_lpips, self.global_step)
            # T6F.2: masked aggregates to TB
            self.writer.add_scalar("psnr_masked/test", mean_psnr_masked, self.global_step)
            self.writer.add_scalar("ssim_masked/test", mean_ssim_masked, self.global_step)
            self.writer.add_scalar("lpips_masked/test", mean_lpips_masked, self.global_step)
            self.writer.add_scalar("cc_psnr_masked/test", mean_cc_psnr_masked, self.global_step)
            self.writer.add_scalar("cc_ssim_masked/test", mean_cc_ssim_masked, self.global_step)
            self.writer.add_scalar("cc_lpips_masked/test", mean_cc_lpips_masked, self.global_step)
            self.writer.add_scalar("time/inference/test", mean_inference_time, self.global_step)

            if best_psnr_img is not None:
                self.writer.add_images(
                    "image/best_psnr/test",
                    torch.stack([best_psnr_img, best_psnr_img_gt]),
                    self.global_step,
                    dataformats="NHWC",
                )

            if worst_psnr_img is not None:
                self.writer.add_images(
                    "image/worst_psnr/test",
                    torch.stack([worst_psnr_img, worst_psnr_img_gt]),
                    self.global_step,
                    dataformats="NHWC",
                )

        return mean_psnr, std_psnr, mean_inference_time
