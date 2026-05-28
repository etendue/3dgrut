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
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.color_correct import color_correct_affine
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import create_summary_writer
from threedgrut.utils.render import apply_post_processing


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

        if conf.model.background.color == "black":
            self.bg_color = torch.zeros((3,), dtype=torch.float32, device="cuda")
        elif conf.model.background.color == "white":
            self.bg_color = torch.ones((3,), dtype=torch.float32, device="cuda")
        else:
            assert False, f"{conf.model.background.color} is not a supported background color."

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
    ):
        """Loads checkpoint for test path."""

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
        # T6F.2: masked 双指标（Stage 6-fix）—— Batch.mask 由 NCoreDataset
        # 注入 ego mask 后非 None；NeRF/Colmap 等 dataset mask=None 时直接
        # 复制全图值保证 byte-identical 回归. masked-cc 同理.
        psnr_masked = []
        ssim_masked = []
        lpips_masked = []
        cc_psnr_masked = []
        cc_ssim_masked = []
        cc_lpips_masked = []
        inference_time = []
        # T8/B3 Phase E.6 — per-cuboid (per-class) PSNR. Records one entry per
        # active track per frame; only computed when ckpt is v2 LayeredGaussians
        # with populated tracks_poses AND the batch carries FTheta intrinsics.
        class_psnr_records: list = []

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

            # T8/B3 Phase E.6 — per-cuboid class PSNR. Skipped when the model
            # has no dyn tracks loaded (single-bg / road-only multi-layer) OR
            # the batch lacks FTheta intrinsics (current dyn projection only
            # supports FTheta to match training-side cuboid mask path).
            tp = getattr(self.model, "tracks_poses", None)
            ftheta_params = getattr(
                gpu_batch, "intrinsics_FThetaCameraModelParameters", None,
            )
            if tp and ftheta_params is not None and hasattr(self.model, "_resolve_pose_idx"):
                idx = self.model._resolve_pose_idx(
                    int(getattr(gpu_batch, "timestamp_us", -1)),
                    int(getattr(gpu_batch, "frame_idx", -1)) if int(getattr(gpu_batch, "frame_idx", -1)) >= 0 else None,
                )
                active = collect_active_tracks_for_frame(
                    self.model.tracks_poses,
                    self.model.tracks_active,
                    getattr(self.model, "tracks_metadata", {}),
                    idx,
                )
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
