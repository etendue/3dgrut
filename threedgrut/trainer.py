# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from addict import Dict
from omegaconf import DictConfig, OmegaConf
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import threedgrut.datasets as datasets
from threedgrut.datasets.protocols import BoundedMultiViewDataset
from threedgrut.datasets.utils import DEFAULT_DEVICE, MultiEpochsDataLoader, PointCloud
from threedgrut.model.bg_cuboid_loss import (
    collect_active_cuboids_for_frame,
    compute_bg_cuboid_opacity_penalty,
    lambda_schedule,
)
from threedgrut.layers.dynamic_mask import project_cuboids_to_mask
from threedgrut.model.layered_loss import compute_layered_l1_loss, compute_sky_loss
from threedgrut.model.pose_smoothness import compute_pose_smoothness_loss
from threedgrut.model.losses import ssim
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.optimizers import SelectiveAdam
from threedgrut.render import Renderer
from threedgrut.strategy.base import BaseStrategy
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import check_step_condition, create_summary_writer, jet_map
from threedgrut.utils.render import apply_post_processing
from threedgrut.utils.timer import CudaTimer


class Trainer3DGRUT:
    """Trainer for paper: "3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes" """

    model: MixtureOfGaussians
    """ Gaussian Model """

    train_dataset: BoundedMultiViewDataset
    val_dataset: BoundedMultiViewDataset

    train_dataloader: torch.utils.data.DataLoader
    val_dataloader: torch.utils.data.DataLoader

    scene_extent: float = 1.0
    """TODO: Add docstring"""

    scene_bbox: tuple[torch.Tensor, torch.Tensor]  # Tuple of vec3 (min,max)
    """TODO: Add docstring"""

    strategy: BaseStrategy
    """ Strategy for optimizing the Gaussian model in terms of densification, pruning, etc. """

    gui = None
    """ If GUI is enabled, references the GUI interface """

    criterions: Dict
    """ Contains functors required to compute evaluation metrics, i.e. psnr, ssim, lpips """

    tracking: Dict
    """ Contains all components used to report progress of training """

    post_processing: Optional[nn.Module] = None
    """ Post-processing module """

    post_processing_optimizers: Optional[list] = None
    """ Optimizers for post-processing module """

    post_processing_schedulers: Optional[list] = None
    """ Schedulers for post-processing module optimizers """

    exposure_model: Optional[nn.Module] = None
    """ T6.1: per-camera affine exposure correction (Stage 6). None when use_exposure=false. """

    exposure_optimizer: Optional[torch.optim.Optimizer] = None
    """ T6.2: independent Adam stepped alongside the main MoG optimizer. """

    exposure_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    """ T9.2: CosineAnnealingLR over n_iterations on the BilateralGrid Adam.
    Stepped every iteration regardless of the freeze gate so the LR profile
    is independent of when the optimizer actually starts updating. """

    exposure_freeze_until_iter: int = 0
    """ T9.2 / V3-P1.b: 2-stage freeze — Adam step gated until global_step
    >= this value. 0 means immediate updates (legacy behavior). Default
    config sets 2000 so Gaussians absorb tone first; BilateralGrid then
    learns residual cross-channel tone (≤ ExposureModel退化 mode). """

    pose_optimizer: Optional[torch.optim.Optimizer] = None
    """ V3 Stage A: independent Adam over per-track wxyz quat + trans Parameters
    on LayeredGaussians.dynamic_rigids. Stepped alongside the main MoG
    optimizer; gated by ``conf.trainer.learnable_pose.enabled`` and the
    freeze-until-iter warmup. None when learnable pose is off. """

    pose_freeze_until_iter: int = 0
    """ V3 Stage A: warmup iterations during which the pose optimizer's
    accumulated gradients are zeroed before any step is taken — lets the main
    Gaussian / scale / opacity parameters converge under GT pose before the
    pose Parameters start drifting. """

    _distillation_start_step: int = -1
    """ Step at which distillation starts (-1 means disabled) """

    @staticmethod
    def create_from_checkpoint(resume: str, conf: DictConfig):
        """Create a new trainer from a checkpoint file"""

        conf.resume = resume
        conf.import_ply.enabled = False
        return Trainer3DGRUT(conf)

    @staticmethod
    def create_from_ply(ply_path: str, conf: DictConfig):
        """Create a new trainer from a PLY file"""

        conf.resume = ""
        conf.import_ply.enabled = True
        conf.import_ply.path = ply_path
        return Trainer3DGRUT(conf)

    @torch.cuda.nvtx.range("setup-trainer")
    def __init__(self, conf: DictConfig, device=None):
        """Set up a new training session, or continue an existing one based on configuration"""

        # Keep track of useful fields
        self.conf = conf
        """ Global configuration of model, scene, optimization, etc"""
        self.device = device if device is not None else DEFAULT_DEVICE
        """ Device used for training and visualizations """
        self.global_step = 0
        """ Current global iteration of the trainer """
        self.n_iterations = conf.n_iterations
        """ Total number of train iterations to take (for multiple passes over the dataset) """
        self.n_epochs = 0
        """ Total number of train epochs / passes, e.g. single pass over the dataset."""
        self.val_frequency = conf.val_frequency
        """ Validation frequency, in terms on global steps """

        # Setup the trainer and components
        logger.log_rule("Load Datasets")
        self.init_dataloaders(conf)
        self.init_scene_extents(self.train_dataset)
        logger.log_rule("Initialize Model")
        self.init_model(conf, self.scene_extent)
        self.init_densification_and_pruning_strategy(conf)
        logger.log_rule("Setup Model Weights & Training")
        self.init_metrics()
        self.setup_training(conf, self.model, self.train_dataset)
        self.init_experiments_tracking(conf)
        self.init_post_processing(conf)
        self.init_exposure_model(conf)
        self.init_depth_losses(conf)  # T11.A2
        self.init_pose_optimizer(conf)
        self.init_gui(conf, self.model, self.train_dataset, self.val_dataset, self.scene_bbox)

    def init_dataloaders(self, conf: DictConfig):
        from threedgrut.datasets.utils import configure_dataloader_for_platform

        train_dataset, val_dataset = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)
        train_dataloader_kwargs = configure_dataloader_for_platform(
            {
                "num_workers": conf.num_workers,
                "batch_size": 1,
                "shuffle": True,
                "pin_memory": True,
                "persistent_workers": True if conf.num_workers > 0 else False,
            }
        )

        val_dataloader_kwargs = configure_dataloader_for_platform(
            {
                "num_workers": conf.num_workers,
                "batch_size": 1,
                "shuffle": False,
                "pin_memory": True,
                "persistent_workers": True if conf.num_workers > 0 else False,
            }
        )

        train_dataloader = MultiEpochsDataLoader(train_dataset, **train_dataloader_kwargs)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, **val_dataloader_kwargs)

        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.val_dataset = val_dataset
        self.val_dataloader = val_dataloader

    def teardown_dataloaders(self):
        if self.train_dataloader is not None:
            del self.train_dataloader
        if self.val_dataloader is not None:
            del self.val_dataloader
        if self.train_dataset is not None:
            del self.train_dataset
        if self.val_dataset is not None:
            del self.val_dataset

    def init_scene_extents(self, train_dataset: BoundedMultiViewDataset) -> None:
        scene_bbox: tuple[torch.Tensor, torch.Tensor]  # Tuple of vec3 (min,max)
        scene_extent = train_dataset.get_scene_extent()
        scene_bbox = train_dataset.get_scene_bbox()
        self.scene_extent = scene_extent
        self.scene_bbox = scene_bbox

    def init_model(self, conf: DictConfig, scene_extent=None) -> None:
        """Initializes the gaussian model and the optix context.

        When conf.use_layered_model is True, builds a LayeredGaussians container
        with layers driven by conf.layers.enabled (defaults to ['background']).
        Standard layer specs come from threedgrut.layers.registry.
        """
        if conf.get("use_layered_model", False):
            from threedgrut.layers.layered_model import LayeredGaussians
            from threedgrut.layers.registry import specs_from_config

            specs = specs_from_config(conf)
            self.model = LayeredGaussians(conf, specs=specs, scene_extent=scene_extent)
            layer_names = [s.name for s in specs]
            logger.info(f"🔆 Using LayeredGaussians with layers={layer_names}")
        else:
            self.model = MixtureOfGaussians(conf, scene_extent=scene_extent)

    def init_densification_and_pruning_strategy(self, conf: DictConfig) -> None:
        """Set pre-train / post-train iteration logic. i.e. densification and pruning"""
        assert self.model is not None
        match self.conf.strategy.method:
            case "GSStrategy":
                from threedgrut.strategy.gs import GSStrategy

                self.strategy = GSStrategy(conf, self.model)
                logger.info("🔆 Using GS strategy")
            case "MCMCStrategy":
                from threedgrut.strategy.mcmc import MCMCStrategy

                self.strategy = MCMCStrategy(conf, self.model)
                logger.info("🔆 Using MCMC strategy")
            case "LayeredMCMCStrategy":
                from threedgrut.layers.layered_model import LayeredGaussians
                from threedgrut.strategy.layered_mcmc import LayeredMCMCStrategy

                assert isinstance(self.model, LayeredGaussians), (
                    "LayeredMCMCStrategy requires use_layered_model=true"
                )
                self.strategy = LayeredMCMCStrategy(conf, self.model, self.model.specs)
                logger.info("🔆 Using LayeredMCMC strategy")
            case _:
                raise ValueError(f"unrecognized model.strategy {conf.strategy.method}")

    def setup_training(
        self,
        conf: DictConfig,
        model: MixtureOfGaussians,
        train_dataset: BoundedMultiViewDataset,
    ):
        """
        Performs required steps to setup the optimization:
        1. Initialize the gaussian model fields: load previous weights from checkpoint, or initialize from scratch.
        2. Build BVH acceleration structure for gaussian model, if not loaded with checkpoint
        3. Set up the optimizer to optimize the gaussian model params
        4. Initialize the densification buffers in the densificaiton strategy
        """

        # Initialize
        if conf.resume:  # Load a checkpoint
            logger.info(f"🤸 Loading a pretrained checkpoint from {conf.resume}!")
            checkpoint = torch.load(conf.resume, weights_only=False)
            model.init_from_checkpoint(checkpoint)
            self.strategy.init_densification_buffer(checkpoint)
            global_step = checkpoint["global_step"]

            # Restore post-processing state
            if "post_processing" in checkpoint and self.post_processing is not None:
                self.post_processing.load_state_dict(checkpoint["post_processing"]["module"])
                for opt, opt_state in zip(
                    self.post_processing_optimizers,
                    checkpoint["post_processing"]["optimizers"],
                ):
                    opt.load_state_dict(opt_state)
                for sched, sched_state in zip(
                    self.post_processing_schedulers,
                    checkpoint["post_processing"]["schedulers"],
                ):
                    sched.load_state_dict(sched_state)
                logger.info("📷 Post-processing state restored from checkpoint")
        elif conf.import_ply.enabled:
            ply_path = (
                conf.import_ply.path
                if conf.import_ply.path
                else f"{conf.out_dir}/{conf.experiment_name}/export_last.ply"
            )
            logger.info(f"Loading a ply model from {ply_path}!")
            model.init_from_ply(ply_path)
            self.strategy.init_densification_buffer()
            model.build_acc()
            global_step = conf.import_ply.init_global_step
        else:
            logger.info(f"🤸 Initiating new 3dgrut training..")
            match conf.initialization.method:
                case "random":
                    model.init_from_random_point_cloud(
                        num_gaussians=conf.initialization.num_gaussians,
                        xyz_max=conf.initialization.xyz_max,
                        xyz_min=conf.initialization.xyz_min,
                    )
                case "colmap":
                    observer_points = torch.tensor(
                        train_dataset.get_observer_points(),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    model.init_from_colmap(conf.path, observer_points)
                case "fused_point_cloud":
                    observer_points = torch.tensor(
                        train_dataset.get_observer_points(),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    ply_path = conf.initialization.fused_point_cloud_path
                    logger.info(f"Initializing from accumulated point cloud: {ply_path}")
                    model.init_from_fused_point_cloud(ply_path, observer_points)
                case "point_cloud":
                    try:
                        ply_path = os.path.join(conf.path, "point_cloud.ply")
                        model.init_from_pretrained_point_cloud(ply_path)
                    except FileNotFoundError as e:
                        logger.error(e)
                        raise e
                case "checkpoint":
                    checkpoint = torch.load(conf.initialization.path, weights_only=False)
                    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
                case "lidar":
                    assert isinstance(
                        train_dataset, datasets.NCoreDataset
                    ), "can only initialize from lidar with NCoreDataset"
                    pc = PointCloud.from_sequence(
                        list(train_dataset.get_point_clouds(step_frame=1, non_dynamic_points_only=True)),
                        device="cpu",
                    )
                    if conf.initialization.num_points < len(pc.xyz_end):
                        # Deterministically random subsample points if there are more points than the specified number of gaussians
                        rng = torch.Generator().manual_seed(conf.seed_initialization)
                        idxs = torch.randperm(len(pc.xyz_end), generator=rng)[: conf.initialization.num_points]
                        pc = pc.selected_idxs(idxs)
                    observer_points = torch.tensor(
                        train_dataset.get_observer_points(),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    # T3.5.b: multi-layer init for LayeredGaussians road / dyn layers.
                    # Single-bg mode: model.init_from_lidar transparently routes
                    # to background via __getattr__ bridge (byte-identical with v1).
                    # Multi-layer: explicitly init each particle layer.
                    from threedgrut.layers.layered_model import LayeredGaussians
                    if isinstance(model, LayeredGaussians) and model._single_bg_layer() is None:
                        # multi-layer: per-spec init dispatcher
                        layer_names = [s.name for s in model.specs]
                        # 1. background: standard LiDAR init (non-dynamic points)
                        if "background" in model.layers:
                            model.layers["background"].init_from_lidar(pc, observer_points)
                            logger.info(
                                f"🔆 background layer initialized: "
                                f"{model.layers['background'].num_gaussians} particles"
                            )
                        # 2. road: BEV-grid + LiDAR-Z KNN from road-semantic LiDAR
                        if "road" in model.layers:
                            from threedgrut.layers.road_init import init_road_layer
                            road_pts, road_rgb = train_dataset.get_road_lidar_points()
                            traj = torch.tensor(
                                train_dataset.get_observer_points(),
                                dtype=torch.float32,
                            )
                            road_spec = next(s for s in model.specs if s.name == "road")
                            r_pos, r_rot, r_sca, r_den, r_col = init_road_layer(
                                road_pts, traj,
                                max_n=road_spec.max_n_particles,
                            )
                            device = model.layers["road"].device
                            model.init_layer_from_points(
                                "road",
                                r_pos.to(device),
                                rotations=r_rot.to(device),
                                scales=r_sca.to(device),
                                densities=r_den.to(device),
                                colors=r_col.to(device),
                                setup_optimizer=True,
                            )
                            logger.info(
                                f"🔆 road layer initialized: {r_pos.shape[0]} "
                                f"particles (from {road_pts.shape[0]} road LiDAR pts)"
                            )
                        # 3. dynamic_rigids: real-cuboids path (T4.5).
                        # Source tracks from NCore manifest cuboid autolabels
                        # (loader.get_cuboid_track_observations); no mock.
                        if "dynamic_rigids" in model.layers:
                            import ncore.data as _nd
                            from threedgrut.datasets.tracks_loader import (
                                load_tracks_from_ncore_cuboids,
                            )
                            from threedgrut.layers.dynamic_rigid_init import (
                                init_dynamic_rigid_layer,
                            )
                            loader = train_dataset.sequence_loaders[
                                train_dataset.sequence_id
                            ]
                            # Reference camera (first) for frame timestamps;
                            # cuboid_track_observations are sensor-agnostic so
                            # any camera's END timestamps work as the canonical
                            # timeline (matches sseg/lidar-sseg key convention).
                            ref_cam = train_dataset.camera_ids[0]
                            ref_sensor = train_dataset.sequence_camera_sensors[
                                train_dataset.sequence_id
                            ][ref_cam]
                            cam_ts = ref_sensor.frames_timestamps_us[
                                :, _nd.FrameTimepoint.END
                            ]
                            # Restrict to clip's active time window so we don't
                            # iterate frames outside the duration_sec slice.
                            time_range = train_dataset.time_range_us
                            in_window = np.array([
                                int(t) in time_range for t in cam_ts
                            ])
                            cam_ts_active = np.asarray(cam_ts)[in_window]
                            tracks = load_tracks_from_ncore_cuboids(
                                loader, cam_ts_active,
                            )
                            logger.info(
                                f"🔆 NCore cuboids → {len(tracks)} dynamic_rigid "
                                f"tracks (over {cam_ts_active.shape[0]} frames in window)"
                            )
                            if not tracks:
                                logger.warning(
                                    "🔆 dynamic_rigids layer enabled but no "
                                    "vehicle tracks within time window; layer "
                                    "stays empty (this is OK for clips with no "
                                    "vehicles in the chosen duration_sec slice)."
                                )
                            else:
                                model.populate_tracks(tracks)
                                # Pull dyn LiDAR + filter per-cuboid → object-local
                                dyn_pts, _ = train_dataset.get_dynamic_lidar_points()
                                # V3-L5: read NuRec ``symmetric_axis`` from the
                                # dynamic_rigids LayerSpec.extra (yaml override).
                                # None → baseline (no mirror), 'Y' → vehicle
                                # left-right symmetry init augmentation.
                                _dyn_spec = next(
                                    (s for s in model.specs
                                     if s.name == "dynamic_rigids"),
                                    None,
                                )
                                _sym_axis = (
                                    (getattr(_dyn_spec, "extra", {}) or {})
                                    .get("symmetric_axis")
                                    if _dyn_spec is not None else None
                                )
                                d_pos, d_track_ids, _track_names = (
                                    init_dynamic_rigid_layer(
                                        tracks, dyn_pts,
                                        max_pts_per_track=5_000,
                                        symmetric_axis=_sym_axis,
                                    )
                                )
                                device = model.layers["dynamic_rigids"].device
                                model.init_layer_from_points(
                                    "dynamic_rigids",
                                    d_pos.to(device),
                                    track_ids=d_track_ids.to(device),
                                    setup_optimizer=True,
                                )
                                logger.info(
                                    f"🔆 dynamic_rigids layer initialized: "
                                    f"{d_pos.shape[0]} particles "
                                    f"(from {dyn_pts.shape[0]} dyn LiDAR pts × "
                                    f"{len(tracks)} cuboids)"
                                    + (f" [V3-L5 symmetric_axis={_sym_axis!r}]"
                                       if _sym_axis else "")
                                )
                    else:
                        # single-bg or v1: original byte-identical path
                        model.init_from_lidar(pc, observer_points)
                case _:
                    raise ValueError(
                        f"unrecognized initialization.method {conf.initialization.method}, choose from [colmap, point_cloud, random, checkpoint, lidar]"
                    )

            self.strategy.init_densification_buffer()

            model.build_acc()
            model.setup_optimizer()
            global_step = 0

        self.global_step = global_step
        self.n_epochs = int((conf.n_iterations + len(train_dataset) - 1) / len(train_dataset))

    def init_gui(
        self,
        conf: DictConfig,
        model: MixtureOfGaussians,
        train_dataset: BoundedMultiViewDataset,
        val_dataset: BoundedMultiViewDataset,
        scene_bbox,
    ):
        gui = None

        if conf.with_gui:
            from threedgrut.utils.gui import GUI

            gui = GUI(conf, model, train_dataset, val_dataset, scene_bbox)

        elif conf.with_viser_gui:
            from threedgrut.utils.viser_gui_util import ViserGUI

            gui = ViserGUI(conf, model, train_dataset, val_dataset, scene_bbox)

        self.gui = gui

    def init_metrics(self):
        self.criterions = Dict(
            psnr=PeakSignalNoiseRatio(data_range=1).to(self.device),
            ssim=StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device),
            lpips=LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(self.device),
        )

    def init_experiments_tracking(self, conf: DictConfig):
        # Initialize the tensorboard writer
        object_name = Path(conf.path).stem
        writer, out_dir, run_name = create_summary_writer(
            conf, object_name, conf.out_dir, conf.experiment_name, conf.use_wandb
        )
        logger.info(f"📊 Training logs & will be saved to: {out_dir}")

        # Store parsed config for reference
        with open(os.path.join(out_dir, "parsed.yaml"), "w") as fp:
            OmegaConf.save(config=conf, f=fp)

        # Pack all components used to track progress of training
        self.tracking = Dict(
            writer=writer,
            run_name=run_name,
            object_name=object_name,
            output_dir=out_dir,
        )

    def init_post_processing(self, conf: DictConfig):
        """Initialize post-processing module based on config."""
        method = conf.post_processing.method

        if method is None:
            return

        if method == "ppisp":
            from ppisp import PPISP, PPISPConfig

            frames_per_camera = self.train_dataset.get_frames_per_camera()
            num_cameras = len(frames_per_camera)
            num_frames = sum(frames_per_camera)

            use_controller = conf.post_processing.get("use_controller", True)

            # Distillation mode: controller activates after main training
            # Total iterations = n_iterations, distillation starts at n_iterations - n_distillation_steps
            n_distillation_steps = conf.post_processing.get("n_distillation_steps", 5000)
            if use_controller and n_distillation_steps > 0:
                main_training_steps = conf.n_iterations - n_distillation_steps
                controller_activation_ratio = main_training_steps / conf.n_iterations
                controller_distillation = True
                self._distillation_start_step = main_training_steps
                logger.info(f"📷 PPISP distillation mode: controller activates at step {main_training_steps}")
            elif use_controller:
                controller_activation_ratio = 0.8
                controller_distillation = False
                self._distillation_start_step = -1
            else:
                controller_activation_ratio = 0.0
                controller_distillation = False
                self._distillation_start_step = -1

            ppisp_config = PPISPConfig(
                use_controller=use_controller,
                controller_distillation=controller_distillation,
                controller_activation_ratio=controller_activation_ratio,
            )

            self.post_processing = PPISP(
                num_cameras=num_cameras,
                num_frames=num_frames,
                config=ppisp_config,
            ).to(self.device)

            self.post_processing_optimizers = self.post_processing.create_optimizers()
            self.post_processing_schedulers = self.post_processing.create_schedulers(
                self.post_processing_optimizers,
                max_optimization_iters=conf.n_iterations,
            )

            logger.info(f"📷 {method.upper()} initialized: {num_cameras} cameras, {num_frames} frames")
        else:
            raise ValueError(f"Unknown post-processing method: {method}")

    def init_exposure_model(self, conf: DictConfig) -> None:
        """Stage 6 T6.2 / Stage 9 T9.1: per-camera color correction with
        independent Adam.

        Enabled by ``conf.trainer.use_exposure``. Camera count comes from the
        train dataset's ``get_frames_per_camera()`` so any dataset that
        implements :class:`BoundedMultiViewDataset` works (NCore + COLMAP). On
        resume, parameters and the optimizer state are restored from the
        ``"exposure_state"`` key in the checkpoint (added by
        :meth:`save_checkpoint`).

        T9.1 / V3-P1.a: defaults to :class:`BilateralGrid` (1x1x1 = 12-param
        per-camera color affine) — replaces the v2 ExposureModel
        ``exp(a)*img + b``. Loading v2 ckpts with the old ``exposure_a`` /
        ``exposure_b`` keys silently warns and keeps the BilateralGrid at
        identity init (the v2 affine cannot be losslessly mapped onto a 12-
        parameter color matrix; a fresh BilateralGrid converges to the right
        color in a few k steps under V3-P1 reg).
        """
        trainer_conf = getattr(conf, "trainer", None)
        if trainer_conf is None:
            return
        use_exposure = trainer_conf.get("use_exposure", False) if hasattr(trainer_conf, "get") \
            else getattr(trainer_conf, "use_exposure", False)
        if not use_exposure:
            return

        from threedgrut.correction import BilateralGrid

        num_camera = len(self.train_dataset.get_frames_per_camera())
        if num_camera < 1:
            logger.warning(
                "📷 BilateralGrid requested but dataset reports 0 cameras; "
                "skipping (use_exposure=true → no-op)."
            )
            return
        # T9.1: 1x1x1 grid by default (NuRec parsed_config). Future ablations
        # can override via conf.trainer.bilateral_grid_X / _Y / _W.
        grid_X = int(
            trainer_conf.get("bilateral_grid_X", 1) if hasattr(trainer_conf, "get")
            else getattr(trainer_conf, "bilateral_grid_X", 1)
        )
        grid_Y = int(
            trainer_conf.get("bilateral_grid_Y", 1) if hasattr(trainer_conf, "get")
            else getattr(trainer_conf, "bilateral_grid_Y", 1)
        )
        grid_W = int(
            trainer_conf.get("bilateral_grid_W", 1) if hasattr(trainer_conf, "get")
            else getattr(trainer_conf, "bilateral_grid_W", 1)
        )
        self.exposure_model = BilateralGrid(
            num_camera=num_camera,
            grid_X=grid_X, grid_Y=grid_Y, grid_W=grid_W,
        ).to(self.device)
        exposure_lr = float(
            trainer_conf.get("exposure_lr", 1e-3) if hasattr(trainer_conf, "get")
            else getattr(trainer_conf, "exposure_lr", 1e-3)
        )
        # T9.2 / V3-P1.b: L2 reg (weight_decay) + 2-stage freeze + cosine LR
        # decay. NOTE on weight_decay vs identity init: Adam decay pulls
        # params toward 0, but BilateralGrid identity init has 1's at diagonal
        # voxel positions. At lr=1e-3 / wd=1e-4 the decay component per step
        # is ~1e-7 — negligible against photometric gradient. Trades a tiny
        # pull-to-black baseline for a stable bound on grid magnitude growth
        # (key for preventing the 30k ExposureModel退化 mode where exposure
        # absorbed ~10 dB of tone shift; v3_plan.md § 2.1 R1).
        weight_decay = float(
            trainer_conf.get("bilateral_grid_weight_decay", 1e-4)
            if hasattr(trainer_conf, "get")
            else getattr(trainer_conf, "bilateral_grid_weight_decay", 1e-4)
        )
        freeze_until = int(
            trainer_conf.get("bilateral_grid_freeze_until_iter", 2000)
            if hasattr(trainer_conf, "get")
            else getattr(trainer_conf, "bilateral_grid_freeze_until_iter", 2000)
        )
        self.exposure_freeze_until_iter = max(0, freeze_until)
        # T9.2: AdamW (decoupled weight_decay) not Adam — Adam's L2-coupled
        # decay gets amplified by momentum/RMS when the photometric gradient
        # is small (early steps before geometry stabilises), producing
        # ~lr-magnitude pulls toward zero rather than ~lr*wd. Decoupled
        # decay (AdamW) keeps the per-step pull at lr*wd ≈ 1e-7 regardless
        # of gradient magnitude — safe for identity-init BilateralGrid.
        # See test_t9_2_weight_decay_pull_negligible_at_identity_init.
        self.exposure_optimizer = torch.optim.AdamW(
            self.exposure_model.parameters(),
            lr=exposure_lr,
            weight_decay=weight_decay,
        )
        # CosineAnnealingLR over the full training horizon. n_iterations
        # comes from conf (root level, not trainer.) — same source as the
        # main loop bound.
        T_max = int(getattr(conf, "n_iterations", 30000))
        self.exposure_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.exposure_optimizer, T_max=max(T_max, 1),
        )
        logger.info(
            f"📷 BilateralGrid initialized: {num_camera} cameras, "
            f"grid={grid_X}x{grid_Y}x{grid_W}, lr={exposure_lr}, "
            f"weight_decay={weight_decay}, "
            f"freeze_until={self.exposure_freeze_until_iter}, "
            f"cosine T_max={T_max}"
        )

        # Restore from checkpoint if resuming.
        if conf.resume:
            ckpt = torch.load(conf.resume, weights_only=False, map_location=self.device)
            if "exposure_state" in ckpt:
                module_state = ckpt["exposure_state"]["module"]
                try:
                    self.exposure_model.load_state_dict(module_state, strict=True)
                    self.exposure_optimizer.load_state_dict(
                        ckpt["exposure_state"]["optimizer"]
                    )
                    # T9.2: scheduler state is optional (older T9.1 ckpts
                    # didn't save it). Resume without it just continues
                    # cosine from current global_step; resume with it
                    # restores the exact LR profile.
                    sched_state = ckpt["exposure_state"].get("scheduler")
                    if sched_state is not None and self.exposure_scheduler is not None:
                        self.exposure_scheduler.load_state_dict(sched_state)
                    logger.info(
                        "📷 BilateralGrid state restored from checkpoint "
                        f"(scheduler={'restored' if sched_state else 'reset'})"
                    )
                except (KeyError, RuntimeError) as e:
                    # T9.1: v2 ckpts store {exposure_a, exposure_b} which
                    # don't fit BilateralGrid.grids. Keep identity init.
                    legacy_keys = set(module_state.keys()) & {
                        "exposure_a", "exposure_b",
                    }
                    if legacy_keys:
                        logger.warning(
                            f"📷 v2 ckpt exposure_state has legacy keys "
                            f"{sorted(legacy_keys)} incompatible with "
                            f"BilateralGrid; keeping identity init "
                            f"(BilateralGrid will re-learn). Original "
                            f"load_state_dict error: {e}"
                        )
                    else:
                        raise

    def init_depth_losses(self, conf: DictConfig) -> None:
        """T11.A2: register DepthLoss + bg_lidar heads + λ params.

        Wired next to init_exposure_model. Enabled by ``conf.trainer.use_lidar_depth``
        / ``conf.trainer.use_depth_prior``. Defaults: both off → byte-identical to
        pre-Stage-11 behavior.

        λ schedule for the LiDAR head uses drivestudio's exponential decay
        (lidar_w_decay > 0 → λ_eff = λ_base * exp(-step/8000 * decay_rate)).
        Set lidar_w_decay <= 0 to disable decay.
        """
        from threedgrut.correction.depth_prior import DepthLoss

        trainer_conf = conf.trainer
        def _get(name, default):
            if hasattr(trainer_conf, "get"):
                return trainer_conf.get(name, default)
            return getattr(trainer_conf, name, default)

        self.use_lidar_depth = bool(_get("use_lidar_depth", False))
        self.use_depth_prior = bool(_get("use_depth_prior", False))

        self.depth_max = float(_get("depth_max", 80.0))
        self.lambda_lidar_depth_base = float(_get("lambda_lidar_depth", 0.03))
        self.lambda_lidar_decay_rate = float(_get("lidar_w_decay", 1.0))
        self.lambda_bg_lidar = float(_get("lambda_bg_lidar", 0.005))
        self.lambda_depth_prior = float(_get("lambda_depth_prior", 0.01))

        # Heads constructed once; reused every train step.
        self.lidar_depth_loss_fn = DepthLoss(
            loss_type=str(_get("lidar_depth_loss_type", "l1")),
            normalize=True,
            use_inverse_depth=False,
            max_depth=self.depth_max,
            eps=0.01,
        )
        self.depth_prior_loss_fn = DepthLoss(
            loss_type="l2",
            normalize=False,
            use_inverse_depth=True,
            max_depth=self.depth_max,
            eps=0.01,
        )

        logger.info(
            f"init_depth_losses: use_lidar={self.use_lidar_depth} "
            f"use_depth_prior={self.use_depth_prior} "
            f"λ_lidar={self.lambda_lidar_depth_base} (decay={self.lambda_lidar_decay_rate}) "
            f"λ_bg={self.lambda_bg_lidar} λ_depth={self.lambda_depth_prior}"
        )

    def _lidar_lambda_decayed(self) -> float:
        """drivestudio L678-682: λ_lidar * exp(-step/8000 * decay_rate)."""
        import math
        if self.lambda_lidar_decay_rate <= 0:
            return self.lambda_lidar_depth_base
        decay = math.exp(-self.global_step / 8000.0 * self.lambda_lidar_decay_rate)
        return self.lambda_lidar_depth_base * decay

    def init_pose_optimizer(self, conf: DictConfig) -> None:
        """V3 Stage A: independent Adam over per-track learnable cuboid pose.

        Gated by ``conf.trainer.learnable_pose.enabled``. The Parameters are
        registered on ``LayeredGaussians`` itself by ``populate_tracks`` (see
        ``_populate_tracks_impl``'s learnable branch), so this method just
        gathers them into two param groups (one per learning rate — rotation
        is far more sensitive than translation) and creates the optimizer.
        The Parameters round-trip through the regular ``model.state_dict()``
        path; we additionally save the optimizer state under
        ``"learnable_pose_state"`` so Adam moments survive resume.
        """
        trainer_conf = getattr(conf, "trainer", None)
        if trainer_conf is None:
            return
        lp_conf = trainer_conf.get("learnable_pose", None) if hasattr(trainer_conf, "get") \
            else getattr(trainer_conf, "learnable_pose", None)
        if lp_conf is None:
            return
        enabled = lp_conf.get("enabled", False) if hasattr(lp_conf, "get") \
            else getattr(lp_conf, "enabled", False)
        if not enabled:
            return
        # Only LayeredGaussians registers per-track Parameters; vanilla MoG
        # has no pose state to optimize. Quietly no-op otherwise.
        from threedgrut.layers.layered_model import LayeredGaussians
        if not isinstance(self.model, LayeredGaussians):
            logger.warning(
                "🛞 learnable_pose enabled but model is not LayeredGaussians; skipping"
            )
            return
        # Source of truth for which tids exist: tracks_active property keys
        # (one entry per registered _track_active_<tid> buffer, mode-agnostic).
        quat_params, trans_params = [], []
        for tid in sorted(self.model.tracks_active.keys()):
            q = getattr(self.model, f"_track_quat_{tid}", None)
            t = getattr(self.model, f"_track_trans_{tid}", None)
            if q is not None:
                quat_params.append(q)
            if t is not None:
                trans_params.append(t)
        if not quat_params:
            logger.warning(
                "🛞 learnable_pose enabled but no _track_quat_<tid> Parameters "
                "found on model; pose_optimizer not created"
            )
            return
        lr_rotation = float(
            lp_conf.get("lr_rotation", 1.0e-5) if hasattr(lp_conf, "get")
            else getattr(lp_conf, "lr_rotation", 1.0e-5)
        )
        lr_translation = float(
            lp_conf.get("lr_translation", 1.0e-4) if hasattr(lp_conf, "get")
            else getattr(lp_conf, "lr_translation", 1.0e-4)
        )
        self.pose_optimizer = torch.optim.Adam([
            {"params": quat_params,  "lr": lr_rotation,    "name": "track_quat"},
            {"params": trans_params, "lr": lr_translation, "name": "track_trans"},
        ])
        self.pose_freeze_until_iter = int(
            lp_conf.get("freeze_until_iter", 5000) if hasattr(lp_conf, "get")
            else getattr(lp_conf, "freeze_until_iter", 5000)
        )
        logger.info(
            f"🛞 LearnablePose: {len(quat_params)} tracks, "
            f"lr_rot={lr_rotation}, lr_trans={lr_translation}, "
            f"freeze_until_iter={self.pose_freeze_until_iter}"
        )

        # Restore from checkpoint if resuming. Parameters themselves are
        # restored via the regular model.state_dict() path (init_from_checkpoint);
        # here we only restore the Adam moment buffers.
        if conf.resume:
            ckpt = torch.load(conf.resume, weights_only=False, map_location=self.device)
            if "learnable_pose_state" in ckpt:
                self.pose_optimizer.load_state_dict(
                    ckpt["learnable_pose_state"]["optimizer"]
                )
                logger.info("🛞 LearnablePose optimizer state restored from checkpoint")

    @torch.cuda.nvtx.range("get_metrics")
    def get_metrics(
        self,
        gpu_batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        losses: dict[str, torch.Tensor],
        profilers: dict[str, CudaTimer],
        split: str = "training",
        iteration: Optional[int] = None,
    ) -> dict[str, Union[int, float]]:
        """Computes dictionary of single batch metrics based on current batch output.
        Args:
            gpu_batch: GT data of current batch
            output: model prediction for current batch
            losses: dictionary of loss terms computed for current batch
            split: name of split metrics are computed for - 'training' or 'validation'
            iteration: optional, local iteration number within the current pass, e.g 0 <= iter < len(dataset).
        Returns:
            Dictionary of metrics
        """
        metrics = dict()
        step = self.global_step

        rgb_gt = gpu_batch.rgb_gt
        rgb_pred = outputs["pred_rgb"]

        psnr = self.criterions["psnr"]
        ssim = self.criterions["ssim"]
        lpips = self.criterions["lpips"]

        # Move losses to cpu once
        metrics["losses"] = {k: v.detach().item() for k, v in losses.items()}

        is_compute_train_hit_metrics = (split == "training") and (step % self.conf.writer.hit_stat_frequency == 0)
        is_compute_validation_metrics = split == "validation"

        if is_compute_train_hit_metrics or is_compute_validation_metrics:
            metrics["hits_mean"] = outputs["hits_count"].mean().item()
            metrics["hits_std"] = outputs["hits_count"].std().item()
            metrics["hits_min"] = outputs["hits_count"].min().item()
            metrics["hits_max"] = outputs["hits_count"].max().item()

        if is_compute_validation_metrics:
            with torch.cuda.nvtx.range(f"criterions_psnr"):
                metrics["psnr"] = psnr(rgb_pred, rgb_gt).item()

            rgb_gt_full = rgb_gt.permute(0, 3, 1, 2)
            pred_rgb_full = rgb_pred.permute(0, 3, 1, 2)
            pred_rgb_full_clipped = rgb_pred.clip(0, 1).permute(0, 3, 1, 2)

            with torch.cuda.nvtx.range(f"criterions_ssim"):
                metrics["ssim"] = ssim(pred_rgb_full, rgb_gt_full).item()
            with torch.cuda.nvtx.range(f"criterions_lpips"):
                metrics["lpips"] = lpips(pred_rgb_full_clipped, rgb_gt_full).item()

            # T6F.2: masked PSNR / SSIM / LPIPS（Stage 6-fix）
            # Stage 1-6 PSNR/SSIM/LPIPS 都在含 ego 车身像素的全图上算，导致虚高。
            # 这里在保留全图三指标（与历史 Stage 3/4/5/6 baseline 可比）同时，
            # 当 Batch.mask 不为 None 时追加 psnr_masked / ssim_masked / lpips_masked.
            #   - PSNR_masked: 解析公式 sum((p-g)^2 * m) / (sum(m) * 3) → -10·log10
            #   - SSIM / LPIPS 不支持像素级掩膜 → 用 GT-fill：mask=False 区填 GT
            #     (差=0)，再算 SSIM/LPIPS；该区 SSIM≈1 / LPIPS≈0，按面积稀释稳定.
            # mask=None（NeRF/Colmap 等无 ego mask 的 dataset）走 byte-identical
            # 回归：三指标直接复制全图值，保证不引入回归.
            mask = gpu_batch.mask  # [B, H, W, 1] 或 None
            if mask is not None:
                mask = mask.to(rgb_pred.dtype)
                # PSNR_masked
                diff_sq = (rgb_pred - rgb_gt).pow(2) * mask  # broadcast last dim 1→3
                denom = mask.sum().clamp(min=1.0) * 3
                mse_masked = diff_sq.sum() / denom
                metrics["psnr_masked"] = (
                    -10.0 * torch.log10(mse_masked.clamp(min=1e-10))
                ).item()
                # SSIM_masked / LPIPS_masked via GT-fill
                m4d = mask.permute(0, 3, 1, 2)  # [B, 1, H, W]
                rgb_pred_filled = pred_rgb_full * m4d + rgb_gt_full * (1.0 - m4d)
                rgb_pred_filled_clipped = (
                    pred_rgb_full_clipped * m4d + rgb_gt_full * (1.0 - m4d)
                )
                metrics["ssim_masked"] = ssim(rgb_pred_filled, rgb_gt_full).item()
                metrics["lpips_masked"] = lpips(
                    rgb_pred_filled_clipped, rgb_gt_full
                ).item()
            else:
                # byte-identical 回归：mask=None → masked 指标 ≡ 全图指标
                metrics["psnr_masked"] = metrics["psnr"]
                metrics["ssim_masked"] = metrics["ssim"]
                metrics["lpips_masked"] = metrics["lpips"]

            # T9.4 / V3-P1.d: cc_psnr (+ masked) for the V3-P1 acceptance
            # criterion "raw vs cc gap ≤ 2 dB" — only computed when an
            # exposure_model is in play (otherwise the gap is just normal
            # tone correction noise, not a退化 signal). Cheap: same shape
            # as raw, one per-image affine fit. Logged downstream by
            # log_validation_pass as exposure/raw_minus_cc_db_val.
            if self.exposure_model is not None:
                from threedgrut.utils.color_correct import color_correct_affine
                rgb_pred_cc = color_correct_affine(rgb_pred, rgb_gt)
                metrics["cc_psnr"] = psnr(rgb_pred_cc, rgb_gt).item()
                if mask is not None:
                    diff_sq_cc = (rgb_pred_cc - rgb_gt).pow(2) * mask
                    mse_masked_cc = diff_sq_cc.sum() / denom
                    metrics["cc_psnr_masked"] = (
                        -10.0 * torch.log10(mse_masked_cc.clamp(min=1e-10))
                    ).item()
                else:
                    metrics["cc_psnr_masked"] = metrics["cc_psnr"]

            if iteration in self.conf.writer.log_image_views:
                metrics["img_hit_counts"] = jet_map(outputs["hits_count"][-1], self.conf.writer.max_num_hits)
                metrics["img_gt"] = gpu_batch.rgb_gt[-1].clip(0, 1.0)
                metrics["img_pred"] = outputs["pred_rgb"][-1].clip(0, 1.0)
                metrics["img_pred_dist"] = jet_map(outputs["pred_dist"][-1], 100)
                metrics["img_pred_opacity"] = jet_map(outputs["pred_opacity"][-1], 1)

        if profilers:
            timings = {}
            for key, timer in profilers.items():
                if timer.enabled:
                    timings[key] = timer.timing()
            if timings:
                metrics["timings"] = timings

        return metrics

    @torch.cuda.nvtx.range("get_losses")
    def get_losses(
        self, gpu_batch: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Computes dictionary of losses for current batch.
        Args:
            gpu_batch: GT data of current batch
            outputs: model prediction for current batch
        Returns:
            losses: dictionary of loss terms computed for current batch.
        """
        rgb_gt = gpu_batch.rgb_gt
        rgb_pred = outputs["pred_rgb"]
        mask = gpu_batch.mask
        image_infos = getattr(gpu_batch, "image_infos", None)
        trainer_conf = getattr(self.conf, "trainer", {})

        # T8/B3 — optionally fill image_infos["dyn_mask_cuboid"] from active
        # cuboids + FTheta intrinsics so compute_layered_l1_loss prefers cuboid
        # over sseg (cuboid mask is exact: tied to tracked actors, no leak from
        # untracked vehicles / cones). Mutates image_infos in place.
        self._maybe_fill_cuboid_mask(gpu_batch, trainer_conf)
        image_infos = getattr(gpu_batch, "image_infos", None)

        # Mask out the invalid pixels if the mask is provided
        if mask is not None:
            rgb_gt = rgb_gt * mask
            rgb_pred = rgb_pred * mask

        # L1 loss
        # T3.4: when conf.trainer.layered_loss is enabled and the batch
        # carries per-region masks in image_infos, partition L1 across
        # {bg, road, dyn} regions and sum per-region means (sky excluded;
        # Stage 5 envmap takes over). SSIM stays full-image (D7).
        # v1 byte-identical when layered_loss=false or image_infos missing.
        loss_l1 = torch.zeros(1, device=self.device)
        lambda_l1 = 0.0
        if self.conf.loss.use_l1:
            with torch.cuda.nvtx.range(f"loss-l1"):
                use_layered = trainer_conf.get("layered_loss", False) if hasattr(trainer_conf, "get") \
                    else getattr(trainer_conf, "layered_loss", False)
                if use_layered and image_infos is not None:
                    loss_l1 = compute_layered_l1_loss(
                        rgb_pred, rgb_gt,
                        image_infos=image_infos,
                        valid_mask=mask,
                    )
                else:
                    loss_l1 = torch.abs(rgb_pred - rgb_gt).mean()
                lambda_l1 = self.conf.loss.lambda_l1

        # L2 loss
        loss_l2 = torch.zeros(1, device=self.device)
        lambda_l2 = 0.0
        if self.conf.loss.use_l2:
            with torch.cuda.nvtx.range(f"loss-l2"):
                loss_l2 = torch.nn.functional.mse_loss(outputs["pred_rgb"], rgb_gt)
                lambda_l2 = self.conf.loss.lambda_l2

        # DSSIM loss
        loss_ssim = torch.zeros(1, device=self.device)
        lambda_ssim = 0.0
        if self.conf.loss.use_ssim:
            with torch.cuda.nvtx.range(f"loss-ssim"):
                rgb_gt_full = torch.permute(rgb_gt, (0, 3, 1, 2))
                pred_rgb_full = torch.permute(rgb_pred, (0, 3, 1, 2))
                loss_ssim = 1.0 - ssim(pred_rgb_full, rgb_gt_full)
                lambda_ssim = self.conf.loss.lambda_ssim

        # Opacity regularization
        loss_opacity = torch.zeros(1, device=self.device)
        lambda_opacity = 0.0
        if self.conf.loss.use_opacity:
            with torch.cuda.nvtx.range(f"loss-opacity"):
                loss_opacity = torch.abs(self.model.get_density()).mean()
                lambda_opacity = self.conf.loss.lambda_opacity

        # Scale regularization
        loss_scale = torch.zeros(1, device=self.device)
        lambda_scale = 0.0
        if self.conf.loss.use_scale:
            with torch.cuda.nvtx.range(f"loss-scale"):
                loss_scale = torch.abs(self.model.get_scale()).mean()
                lambda_scale = self.conf.loss.lambda_scale

        # T5.5: sky envmap region L1.
        # Uses the *pre-blend* rgb_sky (set in LayeredGaussians._blend_sky) so
        # sky supervision stays decoupled from per-camera exposure (T6.2
        # transforms only the final pred_rgb). compute_sky_loss returns 0
        # without NaN when sky_mask is empty (D6 min_pixels guard).
        loss_sky = torch.zeros(1, device=self.device)
        lambda_sky = 0.0
        use_sky = trainer_conf.get("use_sky_envmap", False) if hasattr(trainer_conf, "get") \
            else getattr(trainer_conf, "use_sky_envmap", False)
        if use_sky and "rgb_sky" in outputs and image_infos is not None:
            sky_mask = image_infos.get("sky_mask") if hasattr(image_infos, "get") \
                else getattr(image_infos, "sky_mask", None)
            with torch.cuda.nvtx.range(f"loss-sky"):
                loss_sky = compute_sky_loss(outputs["rgb_sky"], gpu_batch.rgb_gt, sky_mask)
                lambda_sky = float(
                    trainer_conf.get("lambda_sky", 0.1) if hasattr(trainer_conf, "get")
                    else getattr(trainer_conf, "lambda_sky", 0.1)
                )

        # T8/B3 — Background-layer cuboid opacity penalty.
        # Pushes bg particles whose world position lies inside an active
        # dynamic-rigid cuboid to lower opacity, so MCMC relocate_gaussians
        # can move them to alive donors elsewhere. λ ramps 0 → λ_max over
        # warmup_iters; off entirely when bg_dyn_cuboid_penalty.enabled=false
        # (v2 baseline byte-identical default).
        loss_bg_cuboid = self._compute_bg_cuboid_penalty_term(gpu_batch, trainer_conf)

        # V3 Stage B — temporal smoothness reg on learnable per-track pose.
        # Second-order finite-difference penalty (DriveStudio
        # RigidNodes.temporal_smooth_reg form) on _track_quat_<tid> +
        # _track_trans_<tid> Parameters. Gated triply: pose_optimizer
        # exists (learnable_pose enabled), global_step >= freeze_until_iter
        # (Adam is actually stepping pose), and at least one of λ_trans /
        # λ_rot > 0. Disabled path returns torch.zeros(1) on device,
        # baseline byte-identical when poseopt off.
        loss_pose_smooth = self._compute_pose_smoothness_term(trainer_conf)

        # T11.A2: image-space LiDAR sparse + bg-sky + DepthV2 dense depth
        # supervision. All three heads are no-ops when the corresponding
        # use_* flags are off (default false → byte-identical to pre-stage-11).
        loss_lidar_depth = torch.zeros((), device=self.device)
        loss_bg_lidar = torch.zeros((), device=self.device)
        loss_depth_prior = torch.zeros((), device=self.device)
        lambda_lidar_eff = 0.0

        pred_dist = outputs.get("pred_dist") if isinstance(outputs, dict) else None
        if pred_dist is not None and image_infos is not None:
            sky = image_infos.get("sky_mask")
            dyn = image_infos.get("dyn_mask_cuboid", image_infos.get("dyn_mask_sseg"))
            valid_px = image_infos.get("valid_pixel_mask")

            # ---- LiDAR sparse depth ----
            if self.use_lidar_depth and "lidar_depth_map" in image_infos:
                lidar_gt = image_infos["lidar_depth_map"]
                hit = (lidar_gt > 0).float()
                if sky is not None:
                    sky2d = sky.squeeze(-1) if sky.dim() == hit.dim() + 1 else sky
                    hit = hit * (1.0 - sky2d.float())
                if dyn is not None:
                    dyn2d = dyn.squeeze(-1) if dyn.dim() == hit.dim() + 1 else dyn
                    hit = hit * (1.0 - dyn2d.float())
                if valid_px is not None:
                    vp2d = valid_px.squeeze(-1) if valid_px.dim() == hit.dim() + 1 else valid_px
                    hit = hit * vp2d.float()
                loss_lidar_depth = self.lidar_depth_loss_fn(pred_dist, lidar_gt, hit)
                lambda_lidar_eff = self._lidar_lambda_decayed()

                if sky is not None and self.lambda_bg_lidar > 0:
                    from threedgrut.correction.depth_prior import compute_bg_lidar_loss
                    sky2d = sky.squeeze(-1) if sky.dim() == pred_dist.dim() - 1 else sky
                    loss_bg_lidar = compute_bg_lidar_loss(pred_dist, sky2d.float(), self.depth_max)

            # ---- DepthAnythingV2 dense prior ----
            if self.use_depth_prior and "depth_prior" in image_infos:
                dp_gt = image_infos["depth_prior"]
                valid = torch.ones_like(dp_gt)
                if sky is not None:
                    sky2d = sky.squeeze(-1) if sky.dim() == valid.dim() + 1 else sky
                    valid = valid * (1.0 - sky2d.float())
                if dyn is not None:
                    dyn2d = dyn.squeeze(-1) if dyn.dim() == valid.dim() + 1 else dyn
                    valid = valid * (1.0 - dyn2d.float())
                if valid_px is not None:
                    vp2d = valid_px.squeeze(-1) if valid_px.dim() == valid.dim() + 1 else valid_px
                    valid = valid * vp2d.float()
                valid = valid * (dp_gt < self.depth_max).float()
                loss_depth_prior = self.depth_prior_loss_fn(pred_dist, dp_gt, valid)

        # Total loss
        loss = (
            lambda_l1 * loss_l1
            + lambda_ssim * loss_ssim
            + lambda_opacity * loss_opacity
            + lambda_scale * loss_scale
            + lambda_sky * loss_sky
            + loss_bg_cuboid
            + loss_pose_smooth
            + lambda_lidar_eff * loss_lidar_depth
            + self.lambda_bg_lidar * loss_bg_lidar
            + self.lambda_depth_prior * loss_depth_prior
        )
        return dict(
            total_loss=loss,
            l1_loss=lambda_l1 * loss_l1,
            l2_loss=lambda_l2 * loss_l2,
            ssim_loss=lambda_ssim * loss_ssim,
            opacity_loss=lambda_opacity * loss_opacity,
            scale_loss=lambda_scale * loss_scale,
            sky_loss=lambda_sky * loss_sky,
            bg_cuboid_loss=loss_bg_cuboid,
            pose_smooth_loss=loss_pose_smooth,
            lidar_depth_loss=lambda_lidar_eff * loss_lidar_depth,
            bg_lidar_loss=self.lambda_bg_lidar * loss_bg_lidar,
            depth_prior_loss=self.lambda_depth_prior * loss_depth_prior,
        )

    # ---------------------------------------------------------------- T8/B3
    def _bg_cuboid_conf(self, trainer_conf) -> dict:
        """Pull the bg_dyn_cuboid_penalty sub-dict from trainer conf.

        Returns ``{"enabled": False}`` when the section is absent so callers
        can do a single ``cfg["enabled"]`` check without optional handling.
        """
        if trainer_conf is None:
            return {"enabled": False}
        cfg = (
            trainer_conf.get("bg_dyn_cuboid_penalty", None)
            if hasattr(trainer_conf, "get")
            else getattr(trainer_conf, "bg_dyn_cuboid_penalty", None)
        )
        if cfg is None:
            return {"enabled": False}
        enabled = (
            cfg.get("enabled", False) if hasattr(cfg, "get")
            else getattr(cfg, "enabled", False)
        )
        if not enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "lambda_max": float(
                cfg.get("lambda", 0.05) if hasattr(cfg, "get")
                else getattr(cfg, "lambda", 0.05)
            ),
            "warmup_iters": int(
                cfg.get("lambda_warmup_iters", 5000) if hasattr(cfg, "get")
                else getattr(cfg, "lambda_warmup_iters", 5000)
            ),
            "use_cuboid_mask": bool(
                cfg.get("use_cuboid_mask", True) if hasattr(cfg, "get")
                else getattr(cfg, "use_cuboid_mask", True)
            ),
        }

    def _gather_active_tracks_for_batch(self, gpu_batch):
        """Return (poses [T,4,4], sizes [T,3]) for tracks active at this batch.

        Returns ``(None, None)`` when the model has no tracks, or when no
        track is active at the batch's timestamp.

        V3 Stage A: poses are detached from the autograd graph here. In
        learnable-pose mode ``model.tracks_poses`` (now an ``@property``)
        returns gradient-tracking tensors built from the per-track quat/trans
        Parameters; the caller of this method feeds those poses into two
        regularization paths — ``_maybe_fill_cuboid_mask`` (FTheta cuboid
        projection) and ``_compute_bg_cuboid_penalty_term`` (bg opacity
        penalty inside cuboids). If we let gradients flow back from those
        paths to the pose Parameters, the bg layer would "drag" cuboids
        around to escape the penalty rather than relocating particles —
        adversarial optimization. ``.detach()`` cuts that loop, leaving
        photometric loss + ``_transform_means_and_active`` as the only
        gradient sources for pose.
        """
        model = self.model
        tracks_poses = getattr(model, "tracks_poses", None)
        if not tracks_poses:
            return None, None
        # Snapshot to a plain dict of detached tensors so collect_active_cuboids_for_frame
        # and downstream consumers can't accidentally tap pose gradients. This
        # is a no-op for the buffer path (poses are already non-differentiable).
        tracks_poses = {tid: p.detach() for tid, p in tracks_poses.items()}
        tracks_active = getattr(model, "tracks_active", {})
        tracks_meta = getattr(model, "tracks_metadata", {})
        # Build {tid: size} from the metadata dict (size is the cuboid full extent).
        size_map: dict = {}
        for tid, meta in tracks_meta.items():
            sz = meta.get("size") if isinstance(meta, dict) else None
            if sz is not None:
                size_map[tid] = sz
        if not size_map:
            return None, None

        timestamp_us = int(getattr(gpu_batch, "timestamp_us", -1))
        frame_idx = int(getattr(gpu_batch, "frame_idx", -1))
        idx = model._resolve_pose_idx(timestamp_us, frame_idx if frame_idx >= 0 else None)
        poses, sizes = collect_active_cuboids_for_frame(
            tracks_poses, tracks_active, size_map, idx,
        )
        if poses.shape[0] == 0:
            return None, None
        return poses, sizes

    def _maybe_fill_cuboid_mask(self, gpu_batch, trainer_conf) -> None:
        """Project active cuboids → FTheta 2D mask → ``image_infos["dyn_mask_cuboid"]``.

        No-op when ``bg_dyn_cuboid_penalty.use_cuboid_mask=false``, when the
        model has no tracks, or when the batch lacks ``intrinsics_FThetaCameraModelParameters``.
        """
        cfg = self._bg_cuboid_conf(trainer_conf)
        if not cfg["enabled"] or not cfg.get("use_cuboid_mask", True):
            return
        image_infos = getattr(gpu_batch, "image_infos", None)
        if image_infos is None:
            return
        ftheta_params = getattr(gpu_batch, "intrinsics_FThetaCameraModelParameters", None)
        if ftheta_params is None:
            # T8/B3: only FTheta path implemented; pinhole AABB at ±90° clamps
            # to image edges and would paint whole columns as dyn (defeats the
            # mask). Quietly skip — sseg mask remains the fallback.
            return
        poses, sizes = self._gather_active_tracks_for_batch(gpu_batch)
        if poses is None:
            return

        # T_to_world is c2w (camera→world, START pose). project_cuboids_to_mask
        # expects world→cam; invert. B=1 in current trainer (one camera per iter).
        T_c2w = gpu_batch.T_to_world[0]
        T_w2c = torch.linalg.inv(T_c2w)
        # Image dimensions from rgb_gt.
        H = int(gpu_batch.rgb_gt.shape[1])
        W = int(gpu_batch.rgb_gt.shape[2])
        mask = project_cuboids_to_mask(
            poses, sizes,
            K=None, T_world2cam=T_w2c, H=H, W=W,
            device=self.device,
            ftheta_params=ftheta_params,
        )
        # compute_layered_l1_loss expects [B, H, W] float; we have [H, W] bool.
        image_infos["dyn_mask_cuboid"] = mask.to(dtype=torch.float32).unsqueeze(0)

    def _compute_bg_cuboid_penalty_term(self, gpu_batch, trainer_conf) -> torch.Tensor:
        """Compute scalar bg-cuboid opacity penalty for this batch.

        Returns ``torch.zeros(1, device=self.device)`` when disabled / no
        tracks / lambda still at 0 — matches the dtype of the other loss
        contributions in :meth:`get_losses`.
        """
        zero = torch.zeros(1, device=self.device)
        cfg = self._bg_cuboid_conf(trainer_conf)
        if not cfg["enabled"]:
            return zero
        lam = lambda_schedule(self.global_step, cfg["lambda_max"], cfg["warmup_iters"])
        if lam == 0.0:
            return zero
        # Need access to the background layer's positions + density. Only
        # meaningful in LayeredGaussians mode.
        layers = getattr(self.model, "layers", None)
        if layers is None or "background" not in layers:
            return zero
        poses, sizes = self._gather_active_tracks_for_batch(gpu_batch)
        if poses is None:
            return zero
        bg_layer = layers["background"]
        return compute_bg_cuboid_opacity_penalty(
            bg_layer.positions, bg_layer.density,
            poses.to(self.device), sizes.to(self.device),
            lambda_val=lam,
        )

    # ---------------------------------------------------------------- V3 Stage B
    def _compute_pose_smoothness_term(self, trainer_conf) -> torch.Tensor:
        """Compute the Stage B temporal smoothness reg on learnable pose.

        Returns ``torch.zeros(1, device=self.device)`` when:
          * ``pose_optimizer`` is None (learnable_pose disabled), OR
          * ``global_step < pose_freeze_until_iter`` (freeze warmup — pose
            Parameters are not being stepped, so we don't pay reg either),
          * Both ``λ_temporal_smooth_trans`` / ``_rot`` are 0.

        Delegates to :func:`compute_pose_smoothness_loss` for the actual
        finite-difference math. Wires self.model + λ values + device.
        """
        zero = torch.zeros(1, device=self.device)
        if self.pose_optimizer is None:
            return zero
        if self.global_step < self.pose_freeze_until_iter:
            return zero
        lp_conf = getattr(trainer_conf, "learnable_pose", None)
        if lp_conf is None:
            return zero
        lam_t = float(
            lp_conf.get("lambda_temporal_smooth_trans", 0.0)
            if hasattr(lp_conf, "get")
            else getattr(lp_conf, "lambda_temporal_smooth_trans", 0.0)
        )
        lam_r = float(
            lp_conf.get("lambda_temporal_smooth_rot", 0.0)
            if hasattr(lp_conf, "get")
            else getattr(lp_conf, "lambda_temporal_smooth_rot", 0.0)
        )
        if lam_t <= 0.0 and lam_r <= 0.0:
            return zero
        return compute_pose_smoothness_loss(
            self.model, lam_t, lam_r, device=self.device,
        )

    @torch.cuda.nvtx.range("log_validation_iter")
    def log_validation_iter(
        self,
        gpu_batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        batch_metrics: dict[str, Any],
        iteration: Optional[int] = None,
    ) -> None:
        """Log information after a single validation iteration.
        Args:
            gpu_batch: GT data of current batch
            outputs: model prediction for current batch
            batch_metrics: dictionary of metrics computed for current batch
            iteration: optional, local iteration number within the current pass, e.g 0 <= iter < len(dataset).
        """
        logger.log_progress(
            task_name="Validation",
            advance=1,
            iteration=f"{str(iteration)}",
            psnr=batch_metrics["psnr"],
            loss=batch_metrics["losses"]["total_loss"],
        )

    @torch.cuda.nvtx.range("log_validation_pass")
    def log_validation_pass(self, metrics: dict[str, Any]) -> None:
        """Log information after a single validation pass.
        Args:
            metrics: dictionary of aggregated metrics for all batches in current pass.
        """
        writer = self.tracking.writer
        global_step = self.global_step

        if "img_pred" in metrics:
            writer.add_images(
                "image/pred/val",
                torch.stack(metrics["img_pred"]),
                global_step,
                dataformats="NHWC",
            )
        if "img_gt" in metrics:
            writer.add_images(
                "image/gt",
                torch.stack(metrics["img_gt"]),
                global_step,
                dataformats="NHWC",
            )
        if "img_hit_counts" in metrics:
            writer.add_images(
                "image/hit_counts/val",
                torch.stack(metrics["img_hit_counts"]),
                global_step,
                dataformats="NHWC",
            )
        if "img_pred_dist" in metrics:
            writer.add_images(
                "image/dist/val",
                torch.stack(metrics["img_pred_dist"]),
                global_step,
                dataformats="NHWC",
            )
        if "img_pred_opacity" in metrics:
            writer.add_images(
                "image/opacity/val",
                torch.stack(metrics["img_pred_opacity"]),
                global_step,
                dataformats="NHWC",
            )

        mean_timings = {}
        if "timings" in metrics:
            for time_key in metrics["timings"]:
                mean_timings[time_key] = np.mean(metrics["timings"][time_key])
                writer.add_scalar("time/" + time_key + "/val", mean_timings[time_key], global_step)

        writer.add_scalar("num_particles/val", self.model.num_gaussians, self.global_step)

        mean_psnr = np.mean(metrics["psnr"])
        writer.add_scalar("psnr/val", mean_psnr, global_step)
        writer.add_scalar("ssim/val", np.mean(metrics["ssim"]), global_step)
        writer.add_scalar("lpips/val", np.mean(metrics["lpips"]), global_step)

        # T9.4 / V3-P1.d: BilateralGrid退化 health gauge — raw vs cc gap.
        # v2 ExposureModel退化 mode at 30k produced raw_psnr_masked 15.29 /
        # cc_psnr_masked 26.04 = +10.75 dB gap (cc dominates because raw
        # output drifted far from GT and the eval-time affine fit picked up
        # the slack). v3_plan §2.1 target: gap ≤ 2 dB. Only logged when
        # exposure_model is on (cc_psnr is in metrics dict iff get_metrics
        # added it under self.exposure_model branch).
        if "cc_psnr" in metrics:
            mean_cc_psnr = float(np.mean(metrics["cc_psnr"]))
            mean_cc_psnr_masked = float(np.mean(metrics["cc_psnr_masked"]))
            mean_psnr_masked = float(np.mean(metrics["psnr_masked"]))
            gap_db = mean_cc_psnr - mean_psnr  # signed: + means cc > raw
            gap_db_masked = mean_cc_psnr_masked - mean_psnr_masked
            writer.add_scalar("cc_psnr/val", mean_cc_psnr, global_step)
            writer.add_scalar("cc_psnr_masked/val", mean_cc_psnr_masked, global_step)
            writer.add_scalar(
                "exposure/raw_minus_cc_db_val", -gap_db, global_step,
            )
            writer.add_scalar(
                "exposure/raw_minus_cc_db_masked_val",
                -gap_db_masked,
                global_step,
            )
            # Warn only after BilateralGrid has had a fair chance to learn
            # past the freeze window + a small buffer. Catches退化 mode if
            # raw vs cc drifts apart > 2 dB late in training.
            warn_after = self.exposure_freeze_until_iter + 500
            if (
                global_step > warn_after
                and abs(gap_db_masked) > 2.0
            ):
                logger.warning(
                    f"📷 [T9.4 alert] exposure raw_minus_cc gap = "
                    f"{-gap_db_masked:+.2f} dB at step {global_step} "
                    f"(|gap|>2 dB target). v2 ExposureModel退化 mode "
                    f"signal — BilateralGrid may be absorbing too much "
                    f"tone or Gaussians too little. See v3_plan.md §2.1."
                )
        writer.add_scalar("hits/min/val", np.mean(metrics["hits_min"]), global_step)
        writer.add_scalar("hits/max/val", np.mean(metrics["hits_max"]), global_step)
        writer.add_scalar("hits/mean/val", np.mean(metrics["hits_mean"]), global_step)

        loss = np.mean(metrics["losses"]["total_loss"])
        writer.add_scalar("loss/total/val", loss, global_step)
        if self.conf.loss.use_l1:
            l1_loss = np.mean(metrics["losses"]["l1_loss"])
            writer.add_scalar("loss/l1/val", l1_loss, global_step)
        if self.conf.loss.use_l2:
            l2_loss = np.mean(metrics["losses"]["l2_loss"])
            writer.add_scalar("loss/l2/val", l2_loss, global_step)
        if self.conf.loss.use_ssim:
            ssim_loss = np.mean(metrics["losses"]["ssim_loss"])
            writer.add_scalar("loss/ssim/val", ssim_loss, global_step)

        table = {k: np.mean(v) for k, v in metrics.items() if k in ("psnr", "ssim", "lpips")}
        for time_key in mean_timings:
            table[time_key] = f"{'{:.2f}'.format(mean_timings[time_key])}" + " ms/it"
        logger.log_table(f"📊 Validation Metrics - Step {global_step}", record=table)

    @torch.cuda.nvtx.range(f"log_training_iter")
    def log_training_iter(
        self,
        gpu_batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        batch_metrics: dict[str, Any],
        iteration: Optional[int] = None,
    ) -> None:
        """Log information after a single training iteration.
        Args:
            gpu_batch: GT data of current batch
            outputs: model prediction for current batch
            batch_metrics: dictionary of metrics computed for current batch
            iteration: optional, local iteration number within the current pass, e.g 0 <= iter < len(dataset).
        """
        writer = self.tracking.writer
        global_step = self.global_step

        if self.conf.enable_writer and global_step > 0 and global_step % self.conf.log_frequency == 0:
            loss = np.mean(batch_metrics["losses"]["total_loss"])
            writer.add_scalar("loss/total/train", loss, global_step)
            if self.conf.loss.use_l1:
                l1_loss = np.mean(batch_metrics["losses"]["l1_loss"])
                writer.add_scalar("loss/l1/train", l1_loss, global_step)
            if self.conf.loss.use_l2:
                l2_loss = np.mean(batch_metrics["losses"]["l2_loss"])
                writer.add_scalar("loss/l2/train", l2_loss, global_step)
            if self.conf.loss.use_ssim:
                ssim_loss = np.mean(batch_metrics["losses"]["ssim_loss"])
                writer.add_scalar("loss/ssim/train", ssim_loss, global_step)
            if self.conf.loss.use_opacity:
                opacity_loss = np.mean(batch_metrics["losses"]["opacity_loss"])
                writer.add_scalar("loss/opacity/train", opacity_loss, global_step)
            if self.conf.loss.use_scale:
                scale_loss = np.mean(batch_metrics["losses"]["scale_loss"])
                writer.add_scalar("loss/scale/train", scale_loss, global_step)
            if self.post_processing is not None and "post_processing_reg_loss" in batch_metrics["losses"]:
                post_processing_reg_loss = np.mean(batch_metrics["losses"]["post_processing_reg_loss"])
                writer.add_scalar(
                    "loss/post_processing_reg/train",
                    post_processing_reg_loss,
                    global_step,
                )
            if "psnr" in batch_metrics:
                writer.add_scalar("psnr/train", batch_metrics["psnr"], self.global_step)
            if "ssim" in batch_metrics:
                writer.add_scalar("ssim/train", batch_metrics["ssim"], self.global_step)
            if "lpips" in batch_metrics:
                writer.add_scalar("lpips/train", batch_metrics["lpips"], self.global_step)
            if "hits_mean" in batch_metrics:
                writer.add_scalar("hits/mean/train", batch_metrics["hits_mean"], self.global_step)
            if "hits_std" in batch_metrics:
                writer.add_scalar("hits/std/train", batch_metrics["hits_std"], self.global_step)
            if "hits_min" in batch_metrics:
                writer.add_scalar("hits/min/train", batch_metrics["hits_min"], self.global_step)
            if "hits_max" in batch_metrics:
                writer.add_scalar("hits/max/train", batch_metrics["hits_max"], self.global_step)

            # T9.4 / V3-P1.d: BilateralGrid health monitoring. Cheap stats
            # (no eval-time renders); fires every log_frequency step.
            # Tracks:
            #   exposure/grids_std            — overall magnitude of the
            #       BilateralGrid Parameter; identity init has std ~0.4367
            #       (12-elem 3x4 with 3 ones + 9 zeros). Monotonic increase
            #       past unfreeze signals BilateralGrid absorbing tone.
            #   exposure/grids_drift_from_identity — mean |grids - identity|.
            #       The v3_plan §2.1 退化 indicator; identity-init = 0.
            #       Should stay small (~<0.05) at 30k for a healthy v3
            #       BilateralGrid; v2 ExposureModel退化 mode at 30k pushed
            #       grids equivalent (exp(a), b) far from identity.
            #   exposure/lr                   — current cosine-annealed LR.
            #   exposure/frozen               — 1.0 while step <
            #       freeze_until_iter, 0.0 once AdamW starts stepping.
            if self.exposure_model is not None and hasattr(self.exposure_model, "grids"):
                with torch.no_grad():
                    grids = self.exposure_model.grids
                    writer.add_scalar(
                        "exposure/grids_std",
                        float(grids.std().item()),
                        self.global_step,
                    )
                    # Identity tile across all voxels; compute drift mean(|·|).
                    identity_3x4 = torch.tensor(
                        [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]],
                        device=grids.device, dtype=grids.dtype,
                    ).reshape(12, 1, 1, 1)
                    drift = (grids - identity_3x4).abs().mean().item()
                    writer.add_scalar(
                        "exposure/grids_drift_from_identity",
                        float(drift),
                        self.global_step,
                    )
                if self.exposure_optimizer is not None:
                    writer.add_scalar(
                        "exposure/lr",
                        float(self.exposure_optimizer.param_groups[0]["lr"]),
                        self.global_step,
                    )
                writer.add_scalar(
                    "exposure/frozen",
                    1.0 if self.global_step < self.exposure_freeze_until_iter else 0.0,
                    self.global_step,
                )

            if "timings" in batch_metrics:
                for time_key in batch_metrics["timings"]:
                    writer.add_scalar(
                        "time/" + time_key + "/train",
                        batch_metrics["timings"][time_key],
                        self.global_step,
                    )

            writer.add_scalar("num_particles/train", self.model.num_gaussians, self.global_step)
            writer.add_scalar("train/num_GS", self.model.num_gaussians, self.global_step)

            # # NOTE: hack to easily compare with 3DGS
            # writer.add_scalar("train_loss_patches/total_loss", loss, global_step)
            # writer.add_scalar("gaussians/count", self.model.num_gaussians, self.global_step)

        logger.log_progress(
            task_name="Training",
            advance=1,
            step=f"{str(self.global_step)}",
            loss=batch_metrics["losses"]["total_loss"],
        )

    @torch.cuda.nvtx.range(f"log_training_pass")
    def log_training_pass(self, metrics):
        """Log information after a single training pass.
        Args:
            metrics: dictionary of aggregated metrics for all batches in current pass.
        """
        pass

    @torch.cuda.nvtx.range(f"on_training_end")
    def on_training_end(self):
        """Callback that prompts at the end of training."""
        conf = self.conf
        out_dir = self.tracking.output_dir

        # Export the mixture-of-3d-gaussians
        logger.log_rule("Exporting Models")

        if conf.export_ply.enabled:
            from threedgrut.export import PLYExporter

            ply_path = conf.export_ply.path if conf.export_ply.path else os.path.join(out_dir, "export_last.ply")
            exporter = PLYExporter()
            exporter.export(self.model, Path(ply_path), dataset=self.train_dataset, conf=conf)

        if conf.export_usd.enabled:
            from threedgrut.export import NuRecExporter, USDExporter

            # Determine format for filename suffix
            usdz_format = getattr(conf.export_usd, "format", "nurec")
            if usdz_format == "standard":
                format_suffix = "lightfield"
                exporter = USDExporter.from_config(conf)
            else:
                format_suffix = "nurec"
                exporter = NuRecExporter()

            # Handle path: if not set or relative, put in output directory
            if conf.export_usd.path:
                usdz_path = conf.export_usd.path
                if not os.path.isabs(usdz_path):
                    usdz_path = os.path.join(out_dir, usdz_path)
            else:
                # Default filename includes format suffix
                usdz_path = os.path.join(out_dir, f"export_last_{format_suffix}.usdz")

            exporter.export(
                self.model,
                Path(usdz_path),
                dataset=self.train_dataset,
                conf=conf,
                background=getattr(self, "background", None),
            )

        # Export post-processing report (PPISP-based)
        if self.post_processing is not None and conf.post_processing.method == "ppisp":
            from ppisp.report import export_ppisp_report

            logger.info("📊 Exporting PPISP report...")

            ppisp_report_dir = Path(out_dir) / "ppisp_report"
            frames_per_camera = self.train_dataset.get_frames_per_camera()

            # Get camera names if available
            camera_names = None
            if hasattr(self.train_dataset, "get_camera_names"):
                camera_names = self.train_dataset.get_camera_names()

            export_ppisp_report(
                self.post_processing,
                frames_per_camera=frames_per_camera,
                output_dir=ppisp_report_dir,
                camera_names=camera_names,
            )
            logger.info(f"📊 PPISP report saved to: {ppisp_report_dir}")

        # T8.2: save_checkpoint needs self.train_dataset to extract viz_4d
        # metadata, so it must run BEFORE teardown_dataloaders. v1 behavior
        # is unchanged (teardown is a memory release, order doesn't affect
        # the ckpt blob; test_last loads its own renderer dataset from
        # conf.path).
        self.save_checkpoint(last_checkpoint=True)
        self.teardown_dataloaders()

        # Evaluate on test set
        if conf.test_last:
            logger.log_rule("Evaluation on Test Set")

            # Renderer test split. T9.3: pass the live exposure_model so
            # train-end eval applies the same BilateralGrid (or legacy
            # ExposureModel) that the train loop applied before loss.
            # Aligns eval-time raw psnr with the train-time loss target;
            # previously the v2 ExposureModel退化 mode produced +10.75 dB
            # raw-vs-cc gap at 30k because eval did NOT apply exposure.
            renderer = Renderer.from_preloaded_model(
                model=self.model,
                out_dir=out_dir,
                path=conf.path,
                save_gt=False,
                writer=self.tracking.writer,
                global_step=self.global_step,
                compute_extra_metrics=conf.compute_extra_metrics,
                post_processing=self.post_processing,
                exposure_model=self.exposure_model,
            )
            renderer.render_all()

    @torch.cuda.nvtx.range(f"save_checkpoint")
    def save_checkpoint(self, last_checkpoint: bool = False):
        """Saves checkpoint to a path under {conf.out_dir}/{conf.experiment_name}.
        Args:
            last_checkpoint: If true, will update checkpoint title to 'last'.
                             Otherwise uses global step
        """
        global_step = self.global_step
        out_dir = self.tracking.output_dir
        model_params = self.model.get_model_parameters()

        # LayeredGaussians emits {"gaussians_nodes": {...}, "scene_extent": ...}
        # which we wrap under "model" to match the NRE on-disk schema:
        #     ckpt["model"]["gaussians_nodes"]["<layer>"]["positions"]
        # MixtureOfGaussians emits a flat dict (positions/rotation/.../optimizer
        # at top level) which we keep verbatim for v1 backwards-compat.
        from threedgrut.layers.layered_model import LayeredGaussians

        if isinstance(self.model, LayeredGaussians):
            # v2: config doesn't ride along in get_model_parameters() (each
            # per-layer MoG keeps its own under gaussians_nodes/<name>/config
            # but that's nested and ad-hoc). Mirror v1's flat top-level key so
            # downstream tools (engine.load_3dgrt_object, viz.inject) can read
            # ckpt["config"] uniformly across v1 / v2.
            parameters = {"model": model_params, "config": self.conf}
        else:
            parameters = model_params
        parameters |= {"global_step": self.global_step, "epoch": self.n_epochs - 1}

        strategy_parameters = self.strategy.get_strategy_parameters()
        parameters = {**parameters, **strategy_parameters}

        # Add post-processing state to checkpoint (module + optimizers + schedulers)
        if self.post_processing is not None:
            parameters["post_processing"] = {
                "module": self.post_processing.state_dict(),
                "optimizers": [opt.state_dict() for opt in self.post_processing_optimizers],
                "schedulers": [sched.state_dict() for sched in self.post_processing_schedulers],
            }

        # T6.2: per-camera ExposureModel / T9.1 BilateralGrid + its
        # independent Adam state. T9.2: also persist the CosineAnnealingLR
        # scheduler state so resume continues the cosine curve from the
        # right step (otherwise resume restarts at full lr).
        if self.exposure_model is not None:
            ex_state = {
                "module": self.exposure_model.state_dict(),
                "optimizer": self.exposure_optimizer.state_dict(),
            }
            if self.exposure_scheduler is not None:
                ex_state["scheduler"] = self.exposure_scheduler.state_dict()
            parameters["exposure_state"] = ex_state

        # V3 Stage A: independent Adam for per-track learnable cuboid pose.
        # The Parameters themselves (``_track_quat_<tid>`` /
        # ``_track_trans_<tid>``) ride along inside ``model.state_dict()`` —
        # only the Adam moment buffers need a sibling key.
        if self.pose_optimizer is not None:
            parameters["learnable_pose_state"] = {
                "optimizer": self.pose_optimizer.state_dict(),
                "freeze_until_iter": self.pose_freeze_until_iter,
            }

        # T8.2: 4D viz metadata for viser_gui_4d. Only written when explicitly
        # enabled via conf.viz_4d.enabled — keeps v1 ckpts byte-identical with
        # pre-T8.2 layouts. Failure logs a warning but never aborts the save.
        # getattr() guards against teardown_dataloaders having del'd it (the
        # final last_checkpoint path used to run teardown first — fixed by
        # reordering at line 1095, but stay defensive).
        viz_conf = self.conf.get("viz_4d", {}) if hasattr(self.conf, "get") else {}
        train_ds = getattr(self, "train_dataset", None)
        if (isinstance(self.model, LayeredGaussians)
                and viz_conf
                and bool(viz_conf.get("enabled", False) if hasattr(viz_conf, "get") else False)
                and train_ds is not None):
            try:
                from threedgrut.viz.metadata import extract_4d_metadata
                parameters["viz_4d"] = extract_4d_metadata(
                    self.model, train_ds, self.conf
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[viz_4d] extract_4d_metadata failed, skipping: {e}")

        os.makedirs(os.path.join(out_dir, f"ours_{int(global_step)}"), exist_ok=True)
        if not last_checkpoint:
            ckpt_path = os.path.join(out_dir, f"ours_{int(global_step)}", f"ckpt_{global_step}.pt")
        else:
            ckpt_path = os.path.join(out_dir, "ckpt_last.pt")
        torch.save(parameters, ckpt_path)
        logger.info(f'💾 Saved checkpoint to: "{os.path.abspath(ckpt_path)}"')

    def render_gui(self, scene_updated):
        """Render & refresh a single frame for the gui"""
        gui = self.gui
        if gui is not None:
            import polyscope as ps

            if gui.live_update:
                if scene_updated or self.model.positions.requires_grad:
                    gui.update_cloud_viz()
                gui.update_render_view_viz()

            ps.frame_tick()
            while not gui.viz_do_train:
                ps.frame_tick()

            if ps.window_requests_close():
                logger.warning(
                    "Terminating training from GUI window is not supported. Please terminate it from the terminal."
                )

    def render_gui_viser(self, scene_updated):
        gui = self.gui
        if gui is not None:
            if gui.live_update:
                # update render view
                if scene_updated or self.model.positions.requires_grad:
                    gui.update_point_cloud()
                for client in gui.server.get_clients().values():
                    gui.update_render_view(client, force=True)
                while not gui.viz_do_train:
                    time.sleep(0.0001)

    @torch.cuda.nvtx.range(f"run_train_iter")
    def run_train_iter(
        self,
        global_step: int,
        batch: dict,
        profilers: dict,
        metrics: list,
        conf: DictConfig,
    ):
        # Freeze Gaussians and suspend strategy when distillation starts
        if self._distillation_start_step >= 0 and global_step >= self._distillation_start_step:
            self.model.freeze_gaussians()
            self.strategy.suspend()

        # Access the GPU-cache batch data
        with torch.cuda.nvtx.range(f"train_iter{global_step}_get_gpu_batch"):
            gpu_batch = self.train_dataset.get_gpu_batch_with_intrinsics(batch)

        # Perform validation if required
        is_time_to_validate = (global_step > 0 or conf.validate_first) and (global_step % self.val_frequency == 0)
        if is_time_to_validate:
            self.run_validation_pass(conf)

        # Compute the outputs of a single batch
        with torch.cuda.nvtx.range(f"train_{global_step}_fwd"):
            profilers["inference"].start()
            outputs = self.model(gpu_batch, train=True, frame_id=global_step)
            profilers["inference"].end()

        # Apply post-processing to rendered output
        if self.post_processing is not None:
            with torch.cuda.nvtx.range(f"train_{global_step}_post_processing"):
                outputs = apply_post_processing(self.post_processing, outputs, gpu_batch, training=True)

        # T6.2: per-camera exposure (applied AFTER sky blend in
        # LayeredGaussians.forward, BEFORE the loss). Decouples sky_loss in
        # get_losses (which uses outputs["rgb_sky"] pre-exposure) from
        # per-camera tone, see test_sky_loss_zero_when_no_sky_pixels rationale.
        if self.exposure_model is not None:
            with torch.cuda.nvtx.range(f"train_{global_step}_exposure"):
                outputs["pred_rgb"] = self.exposure_model(
                    gpu_batch.camera_idx, outputs["pred_rgb"]
                )

        # Compute the losses of a single batch
        with torch.cuda.nvtx.range(f"train_{global_step}_loss"):
            batch_losses = self.get_losses(gpu_batch, outputs)
            # Add post-processing regularization loss
            if self.post_processing is not None:
                post_processing_reg_loss = self.post_processing.get_regularization_loss()
                batch_losses["total_loss"] = batch_losses["total_loss"] + post_processing_reg_loss
                batch_losses["post_processing_reg_loss"] = post_processing_reg_loss

        # Backward strategy step
        with torch.cuda.nvtx.range(f"train_{global_step}_pre_bwd"):
            self.strategy.pre_backward(
                step=global_step,
                scene_extent=self.scene_extent,
                train_dataset=self.train_dataset,
                batch=gpu_batch,
                writer=self.tracking.writer,
            )

        # Back-propagate the gradients and update the parameters
        with torch.cuda.nvtx.range(f"train_{global_step}_bwd"):
            profilers["backward"].start()
            batch_losses["total_loss"].backward()
            profilers["backward"].end()

        # Post backward strategy step
        with torch.cuda.nvtx.range(f"train_{global_step}_post_bwd"):
            scene_updated = self.strategy.post_backward(
                step=global_step,
                scene_extent=self.scene_extent,
                train_dataset=self.train_dataset,
                batch=gpu_batch,
                writer=self.tracking.writer,
            )

        # V3-L8/L9: per-track albedo/scale warmup gate. Flips requires_grad
        # on the per-track Parameter tables exactly once when
        # global_step >= track_warmup_steps. No-op when the tables are not
        # registered (OFF mode → baseline byte-identical).
        if hasattr(self.model, "maybe_activate_track_params"):
            if self.model.maybe_activate_track_params(global_step):
                logger.info(
                    f"⚡ V3-L8/L9: per-track albedo/scale params activated at "
                    f"global_step={global_step} (warmup complete)"
                )

        # Optimizer step
        with torch.cuda.nvtx.range(f"train_{global_step}_backprop"):
            if isinstance(self.model.optimizer, SelectiveAdam):
                assert (
                    outputs["mog_visibility"].shape == self.model.density.shape
                ), f"Visibility shape {outputs['mog_visibility'].shape} does not match density shape {self.model.density.shape}"
                self.model.optimizer.step(outputs["mog_visibility"])
            else:
                self.model.optimizer.step()
            self.model.optimizer.zero_grad()

        # Scheduler step
        with torch.cuda.nvtx.range(f"train_{global_step}_scheduler"):
            self.model.scheduler_step(global_step)

        # Post-processing optimizer/scheduler step
        if self.post_processing_optimizers is not None:
            with torch.cuda.nvtx.range(f"train_{global_step}_post_processing_opt"):
                for opt in self.post_processing_optimizers:
                    opt.step()
                    opt.zero_grad()
                for sched in self.post_processing_schedulers:
                    sched.step()

        # T6.2 / T9.2: per-camera exposure optimizer step (independent of MoG opt).
        # T9.2 / V3-P1.b: gate the Adam step on freeze_until — let Gaussians
        # absorb tone first; mirror the pose_optimizer freeze pattern below.
        # Always zero grads (backward populated them) + always step scheduler
        # so the LR profile is independent of the freeze gate.
        if self.exposure_optimizer is not None:
            with torch.cuda.nvtx.range(f"train_{global_step}_exposure_opt"):
                if global_step >= self.exposure_freeze_until_iter:
                    self.exposure_optimizer.step()
                self.exposure_optimizer.zero_grad(set_to_none=True)
                if self.exposure_scheduler is not None:
                    self.exposure_scheduler.step()

        # V3 Stage A: per-track learnable pose optimizer step. During the
        # freeze warmup we still clear gradients (backward already populated
        # them) but skip the Adam step — Parameters stay at GT init while the
        # main Gaussians converge. After freeze_until_iter the optimizer
        # starts updating quat/trans and pose drift begins.
        if self.pose_optimizer is not None:
            with torch.cuda.nvtx.range(f"train_{global_step}_pose_opt"):
                if global_step >= self.pose_freeze_until_iter:
                    self.pose_optimizer.step()
                self.pose_optimizer.zero_grad(set_to_none=True)

        # Post backward strategy step
        with torch.cuda.nvtx.range(f"train_{global_step}_post_opt_step"):
            scene_updated = self.strategy.post_optimizer_step(
                step=global_step,
                scene_extent=self.scene_extent,
                train_dataset=self.train_dataset,
                batch=gpu_batch,
                writer=self.tracking.writer,
            )

        # Update the SH if required
        if self.model.progressive_training and check_step_condition(
            global_step, 0, 1e6, self.model.feature_dim_increase_interval
        ):
            self.model.increase_num_active_features()

        # Update the BVH if required
        if scene_updated or (
            conf.model.bvh_update_frequency > 0 and global_step % conf.model.bvh_update_frequency == 0
        ):
            with torch.cuda.nvtx.range(f"train_{global_step}_bvh"):
                profilers["build_as"].start()
                self.model.build_acc(rebuild=True)
                profilers["build_as"].end()

        # Increment the global step
        global_step += 1
        self.global_step = global_step

        # Compute metrics
        batch_metrics = self.get_metrics(
            gpu_batch,
            outputs,
            batch_losses,
            profilers,
            split="training",
            iteration=iter,
        )
        if "forward_render" in self.model.renderer.timings:
            batch_metrics["timings"]["forward_render_cuda"] = self.model.renderer.timings["forward_render"]
        if "backward_render" in self.model.renderer.timings:
            batch_metrics["timings"]["backward_render_cuda"] = self.model.renderer.timings["backward_render"]
        metrics.append(batch_metrics)

        # !!! Below global step has been incremented !!!
        with torch.cuda.nvtx.range(f"train_{global_step - 1}_log_iter"):
            self.log_training_iter(gpu_batch, outputs, batch_metrics, iter)
        with torch.cuda.nvtx.range(f"train_{global_step - 1}_save_ckpt"):
            if global_step in conf.checkpoint.iterations:
                self.save_checkpoint()

        # Updating the GUI
        with torch.cuda.nvtx.range(f"train_{global_step - 1}_update_gui"):
            if self.conf.with_viser_gui:
                self.render_gui_viser(scene_updated)
            elif self.conf.with_gui:
                self.render_gui(scene_updated)

    @torch.cuda.nvtx.range(f"run_train_pass")
    def run_train_pass(self, conf: DictConfig):
        """Runs a single train epoch over the dataset."""
        metrics = []
        profilers = {
            "inference": CudaTimer(enabled=self.conf.enable_frame_timings),
            "backward": CudaTimer(enabled=self.conf.enable_frame_timings),
            "build_as": CudaTimer(enabled=self.conf.enable_frame_timings),
        }

        for iter, batch in enumerate(self.train_dataloader):
            # Check if we have reached the maximum number of iterations
            if self.global_step >= conf.n_iterations:
                return

            # Step for training iteration
            self.run_train_iter(self.global_step, batch, profilers, metrics, conf)

        self.log_training_pass(metrics)

    @torch.cuda.nvtx.range(f"run_validation_pass")
    @torch.no_grad()
    def run_validation_pass(self, conf: DictConfig) -> dict[str, Any]:
        """Runs a single validation epoch over the dataset.
        Returns:
             dictionary of metrics computed and aggregated over validation set.
        """

        profilers = {
            "inference": CudaTimer(),
        }
        metrics = []
        logger.info(f"Step {self.global_step} -- Running validation..")
        logger.start_progress(
            task_name="Validation",
            total_steps=len(self.val_dataloader),
            color="medium_purple3",
        )

        for val_iteration, batch_idx in enumerate(self.val_dataloader):
            # Access the GPU-cache batch data
            gpu_batch = self.val_dataset.get_gpu_batch_with_intrinsics(batch_idx)

            # Compute the outputs of a single batch
            with torch.cuda.nvtx.range(f"train.validation_step_{self.global_step}"):
                profilers["inference"].start()
                outputs = self.model(gpu_batch, train=False)
                # Apply post-processing for validation (novel view mode)
                if self.post_processing is not None:
                    outputs = apply_post_processing(self.post_processing, outputs, gpu_batch, training=False)
                profilers["inference"].end()

                batch_losses = self.get_losses(gpu_batch, outputs)
                batch_metrics = self.get_metrics(
                    gpu_batch,
                    outputs,
                    batch_losses,
                    profilers,
                    split="validation",
                    iteration=val_iteration,
                )

                self.log_validation_iter(gpu_batch, outputs, batch_metrics, iteration=val_iteration)
                metrics.append(batch_metrics)

        logger.end_progress(task_name="Validation")

        metrics = self._flatten_list_of_dicts(metrics)
        self.log_validation_pass(metrics)
        return metrics

    @staticmethod
    def _flatten_list_of_dicts(list_of_dicts):
        """
        Converts list of dicts -> dict of lists.
        Supports flattening of up to 2 levels of dict hierarchies
        """
        flat_dict = defaultdict(list)
        for d in list_of_dicts:
            for k, v in d.items():
                if isinstance(v, dict):
                    flat_dict[k] = defaultdict(list) if k not in flat_dict else flat_dict[k]
                    for inner_k, inner_v in v.items():
                        flat_dict[k][inner_k].append(inner_v)
                else:
                    flat_dict[k].append(v)
        return flat_dict

    def run_training(self):
        """Initiate training logic for n_epochs.
        Training and validation are controlled by the config.
        """
        assert self.model.optimizer is not None, "Optimizer needs to be initialized before the training can start!"
        conf = self.conf

        logger.log_rule(f"Training {conf.render.method.upper()}")

        # Training loop
        logger.start_progress(task_name="Training", total_steps=conf.n_iterations, color="spring_green1")

        for epoch_idx in range(self.n_epochs):
            self.run_train_pass(conf)

        logger.end_progress(task_name="Training")

        # Report training statistics
        stats = logger.finished_tasks["Training"]
        table = dict(
            n_steps=f"{self.global_step}",
            n_epochs=f"{self.n_epochs}",
            training_time=f"{stats['elapsed']:.2f} s",
            iteration_speed=f"{self.global_step / stats['elapsed']:.2f} it/s",
        )
        logger.log_table(f"🎊 Training Statistics", record=table)

        # Perform testing
        self.on_training_end()
        logger.info(f"🥳 Training Complete.")

        # Updating the GUI
        if self.gui is not None:
            self.gui.training_done = True
            logger.info(f"🎨 GUI Blocking... Terminate GUI to Stop.")
            self.gui.block_in_rendering_loop(fps=60)
