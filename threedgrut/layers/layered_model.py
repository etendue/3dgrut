# SPDX-License-Identifier: Apache-2.0
"""LayeredGaussians: container of per-layer MixtureOfGaussians.

T1.1 scope: container skeleton + ckpt interop only.
    - Each layer is a full MixtureOfGaussians instance.
    - get_model_parameters() emits NRE-aligned nested schema:
        {"gaussians_nodes": {layer_name: <MoG params>}, "scene_extent": float}
    - init_from_checkpoint() auto-detects three input shapes:
        (a) ckpt["model"]["gaussians_nodes"][name]  -- NRE / v2 with outer wrap
        (b) ckpt["gaussians_nodes"][name]           -- v2 already unwrapped
        (c) flat v1: ckpt["positions"], ...         -- legacy, routed to "background"
    - Missing layers in ckpt: warn + skip (do not raise).

T2/T3/T4 scope (NOT in T1.1):
    - fused_view(cur_frame) -> flat tensors for renderer
    - per-frame dynamic pose transforms
    - LayeredMCMCStrategy hooks
    - per-layer optimizers (T1.1 keeps single-optimizer compat via bridge)

Single-bg-layer bridge: when there is exactly one layer named "background",
LayeredGaussians transparently forwards attribute access (positions, rotation,
optimizer, renderer, num_gaussians, get_density(), etc.) to that layer. This
keeps the existing Trainer + MCMCStrategy code paths working without
modification for T1.1 smoke. T2 replaces this with explicit fused-view logic.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import sh_degree_to_specular_dim


def _build_sky_module(spec: LayerSpec, conf) -> nn.Module:
    """Instantiate the sky envmap backend for the sky_envmap layer.

    ``conf.trainer`` overrides ``spec.extra`` defaults so users can flip
    cubemap → mlp without touching the registry.
    """
    # Lazy import keeps test_layer_spec_registry.py from pulling nvdiffrast
    # / torch.nn graph stuff during spec inspection.
    from threedgrut.correction.sky_envmap import SkyEnvmapCubemap, SkyEnvmapMLP

    extra = dict(getattr(spec, "extra", {}) or {})
    trainer_conf = getattr(conf, "trainer", None)
    # Lookup order: conf.trainer.{sky_backend,sky_resolution} (explicit user
    # override, only when not None) → spec.extra defaults → hardcoded
    # ("cubemap", 128).
    conf_backend = None
    conf_resolution = None
    if trainer_conf is not None:
        conf_backend = (
            trainer_conf.get("sky_backend", None)
            if hasattr(trainer_conf, "get")
            else getattr(trainer_conf, "sky_backend", None)
        )
        conf_resolution = (
            trainer_conf.get("sky_resolution", None)
            if hasattr(trainer_conf, "get")
            else getattr(trainer_conf, "sky_resolution", None)
        )
    backend = conf_backend if conf_backend is not None else extra.get("backend", "cubemap")
    resolution = conf_resolution if conf_resolution is not None else extra.get("resolution", 128)

    if backend == "mlp":
        return SkyEnvmapMLP()
    if backend == "cubemap":
        return SkyEnvmapCubemap(resolution=int(resolution))
    raise ValueError(f"unknown sky backend '{backend}', expected 'cubemap' or 'mlp'")


# Degree-0 SH coefficient: 1 / (2 * sqrt(pi)). Used to convert RGB in [0, 1]
# to the DC term of the SH expansion in init_layer_from_points.
_SH_C0 = 0.28209479177387814


def _rotmat_to_quat_wxyz(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix ``R`` of shape ``[..., 3, 3]`` → quaternion ``[..., 4]``
    in ``(w, x, y, z)`` convention.

    Uses Shepperd's case-switching for numerical stability across the full
    rotation range — cuboid yaws can be near ±π where the naive
    ``sqrt(1 + trace)`` form is unstable.
    """
    R00, R01, R02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    R10, R11, R12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    R20, R21, R22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    trace = R00 + R11 + R22

    # Case A: trace > 0 — most numerically stable
    s_a = torch.sqrt(torch.clamp(trace + 1.0, min=1e-12)) * 2.0
    w_a = 0.25 * s_a
    x_a = (R21 - R12) / s_a
    y_a = (R02 - R20) / s_a
    z_a = (R10 - R01) / s_a

    # Case B: R00 is the largest diagonal element
    s_b = torch.sqrt(torch.clamp(1.0 + R00 - R11 - R22, min=1e-12)) * 2.0
    w_b = (R21 - R12) / s_b
    x_b = 0.25 * s_b
    y_b = (R01 + R10) / s_b
    z_b = (R02 + R20) / s_b

    # Case C: R11 is the largest diagonal element
    s_c = torch.sqrt(torch.clamp(1.0 + R11 - R00 - R22, min=1e-12)) * 2.0
    w_c = (R02 - R20) / s_c
    x_c = (R01 + R10) / s_c
    y_c = 0.25 * s_c
    z_c = (R12 + R21) / s_c

    # Case D: R22 is the largest diagonal element
    s_d = torch.sqrt(torch.clamp(1.0 + R22 - R00 - R11, min=1e-12)) * 2.0
    w_d = (R10 - R01) / s_d
    x_d = (R02 + R20) / s_d
    y_d = (R12 + R21) / s_d
    z_d = 0.25 * s_d

    cond_a = trace > 0
    cond_b = (~cond_a) & (R00 >= R11) & (R00 >= R22)
    cond_c = (~cond_a) & (~cond_b) & (R11 >= R22)

    w = torch.where(cond_a, w_a, torch.where(cond_b, w_b, torch.where(cond_c, w_c, w_d)))
    x = torch.where(cond_a, x_a, torch.where(cond_b, x_b, torch.where(cond_c, x_c, x_d)))
    y = torch.where(cond_a, y_a, torch.where(cond_b, y_b, torch.where(cond_c, y_c, y_d)))
    z = torch.where(cond_a, z_a, torch.where(cond_b, z_b, torch.where(cond_c, z_c, z_d)))
    return torch.stack([w, x, y, z], dim=-1)


def _quat_multiply_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton quaternion product ``q1 ⊗ q2`` in ``(w, x, y, z)`` convention.

    Shapes broadcast: ``q1`` and ``q2`` may be any common-broadcast shape
    ending in 4. Result has the broadcast shape.
    """
    w1 = q1[..., 0]; x1 = q1[..., 1]; y1 = q1[..., 2]; z1 = q1[..., 3]
    w2 = q2[..., 0]; x2 = q2[..., 1]; y2 = q2[..., 2]; z2 = q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def _quat_wxyz_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Quaternion ``q`` of shape ``[..., 4]`` in ``(w, x, y, z)`` convention →
    rotation matrix ``[..., 3, 3]``.

    Inverse of :func:`_rotmat_to_quat_wxyz`. Caller should normalize ``q`` to
    unit norm before calling — non-unit quaternions encode a uniform scaling
    and produce non-orthogonal R. Used by the learnable-pose path
    (``_compose_pose_for_track``) where Adam updates can violate unit-norm.
    """
    w = q[..., 0]; x = q[..., 1]; y = q[..., 2]; z = q[..., 3]
    ww = w * w; xx = x * x; yy = y * y; zz = z * z
    wx = w * x; wy = w * y; wz = w * z
    xy = x * y; xz = x * z; yz = y * z
    r00 = ww + xx - yy - zz
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = ww - xx + yy - zz
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = ww - xx - yy + zz
    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


class _LayeredOptimizerView:
    """Lightweight optimizer-like wrapper that fans calls across per-layer Adams.

    LayeredGaussians.optimizer returns one of these in multi-layer mode so the
    Trainer's main loop ``self.model.optimizer.step()`` works unchanged. In
    single-bg mode the property short-circuits to the bg layer's own optimizer
    (byte-identical with the v1 path; this wrapper is not used).
    """

    def __init__(self, layers: nn.ModuleDict) -> None:
        self._layers = layers

    def step(self) -> None:
        for layer in self._layers.values():
            opt = getattr(layer, "optimizer", None)
            if opt is not None:
                opt.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for layer in self._layers.values():
            opt = getattr(layer, "optimizer", None)
            if opt is not None:
                opt.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self) -> list[dict]:
        groups: list[dict] = []
        for layer in self._layers.values():
            opt = getattr(layer, "optimizer", None)
            if opt is not None:
                groups.extend(opt.param_groups)
        return groups


# Per-particle parameter names forwarded to the single background layer when in
# single-layer mode (so MCMCStrategy's setattr(model, name, new_param) writes
# to bg layer's Parameter rather than registering one on LayeredGaussians).
_FORWARD_PARAM_NAMES = frozenset({
    "positions", "rotation", "scale", "density",
    "features_albedo", "features_specular",
})


class _FusedView:
    """MoG-like view onto fused multi-layer tensors for renderer compatibility.

    ``Renderer.render(model, batch, ...)`` accesses these on the model:
      - ``positions`` / ``rotation`` / ``scale`` / ``density``
      - ``features_albedo`` / ``features_specular``
      - ``num_gaussians`` (shape property)
      - ``n_active_features`` (SH progressive feature dim)
      - ``get_rotation() / get_scale() / get_density() / get_features() / get_positions()``
      - ``background`` (callable module applying bg color)

    The fused tensors come from ``LayeredGaussians.fused_view(frame_id)`` (concat
    across particle layers, dynamic positions already world-transformed by T4.3);
    activation functions + background module are borrowed from a "reference" layer
    (the first particle layer in spec order). All layers share the same conf so
    activations match.

    This is a thin façade with no torch.nn.Module semantics — the renderer only
    reads attributes, never calls .train() / .parameters() / .state_dict() on the
    model itself, so a plain Python object suffices and avoids ModuleDict /
    Parameter registration overhead per frame.
    """

    def __init__(self, fused_tensors: dict, ref_layer: MixtureOfGaussians):
        self._t = fused_tensors
        self._ref = ref_layer

    # Direct fused tensor access ---------------------------------------------
    @property
    def positions(self) -> torch.Tensor: return self._t["positions"]
    @property
    def rotation(self) -> torch.Tensor: return self._t["rotation"]
    @property
    def scale(self) -> torch.Tensor: return self._t["scale"]
    @property
    def density(self) -> torch.Tensor: return self._t["density"]
    @property
    def features_albedo(self) -> torch.Tensor: return self._t["features_albedo"]
    @property
    def features_specular(self) -> torch.Tensor: return self._t["features_specular"]

    # Shape / config ---------------------------------------------------------
    @property
    def num_gaussians(self) -> int: return int(self._t["positions"].shape[0])
    @property
    def n_active_features(self) -> int: return self._ref.n_active_features
    @property
    def max_n_features(self) -> int: return self._ref.max_n_features
    @property
    def background(self): return self._ref.background
    @property
    def device(self): return self._ref.device

    # Activated accessors borrowed from ref layer ----------------------------
    def get_positions(self) -> torch.Tensor: return self._t["positions"]
    def get_rotation(self, preactivation: bool = False) -> torch.Tensor:
        return self._t["rotation"] if preactivation else self._ref.rotation_activation(self._t["rotation"])
    def get_scale(self, preactivation: bool = False) -> torch.Tensor:
        return self._t["scale"] if preactivation else self._ref.scale_activation(self._t["scale"])
    def get_density(self, preactivation: bool = False) -> torch.Tensor:
        return self._t["density"] if preactivation else self._ref.density_activation(self._t["density"])
    def get_features(self) -> torch.Tensor:
        return torch.cat((self._t["features_albedo"], self._t["features_specular"]), dim=1)
    def get_features_albedo(self) -> torch.Tensor: return self._t["features_albedo"]
    def get_features_specular(self) -> torch.Tensor: return self._t["features_specular"]


class LayeredGaussians(nn.Module):
    """Drop-in replacement for MixtureOfGaussians when use_layered_model=True."""

    def __init__(self, conf, specs: List[LayerSpec], scene_extent: float,
                 *, tracks: Optional[dict] = None):
        super().__init__()
        # Use object.__setattr__ for non-Module fields to bypass our custom
        # __setattr__ forwarding logic below.
        object.__setattr__(self, "conf", conf)
        object.__setattr__(self, "specs", list(specs))
        object.__setattr__(self, "scene_extent", scene_extent)
        # Inference-only filter for viser_gui_4d's "Gaussian Layers" toggles.
        # Default: every contributing layer enabled (particle layers + sky_envmap);
        # GUI callbacks mutate via wholesale set replacement so the render loop
        # always observes a consistent snapshot. Never persisted in state_dict.
        default_enabled = {
            s.name for s in specs
            if (s.is_particle_layer or s.name == "sky_envmap")
        }
        object.__setattr__(self, "enabled_layer_names", default_enabled)
        # nn.ModuleDict so .to(device) / state_dict() recurse into layers.
        self.layers: nn.ModuleDict = nn.ModuleDict()
        for spec in specs:
            if spec.is_particle_layer:
                self.layers[spec.name] = MixtureOfGaussians(conf, scene_extent)
            elif spec.name == "sky_envmap":
                # T5.4: sky envmap is a small parametric module (cubemap or
                # MLP); shares ModuleDict so .to(device) / state_dict() / load
                # recurse into it, but does NOT contribute particles to
                # fused_view.
                self.layers[spec.name] = _build_sky_module(spec, conf)
            # dynamic_deformables: registered as a non-particle stub in the
            # registry; no module instantiated (v2.x placeholder, T1.2).

        # T4.0 / V3 Stage A: per-clip dynamic-rigid track poses.
        # Storage mode depends on ``conf.trainer.learnable_pose.enabled``:
        #   - false (default, v2 baseline byte-identical): one buffer
        #     ``_track_pose_<tid>`` of shape ``[F, 4, 4]`` per track, plus
        #     ``_track_active_<tid>`` BoolTensor[F].
        #   - true (V3 Stage A): split into ``_track_quat_<tid>`` Parameter[F,4]
        #     (wxyz) + ``_track_trans_<tid>`` Parameter[F,3] for SE(3) refinement
        #     under photometric loss, plus a frozen ``_track_pose_gt_<tid>``
        #     buffer for resume detection / future viz diff. Active mask remains
        #     a buffer (never learnable).
        # ``tracks_poses`` / ``tracks_active`` are derived ``@property`` dicts —
        # NOT init-time empty dicts — so they always reflect the actual
        # registered buffers/parameters (fixes observation #321/#349/#851 where
        # the old Python-dict mirror went stale after ckpt load).
        # T4.5 timestamp-aligned dyn pose lookup: shared per-frame absolute
        # camera END timestamps in microseconds. Single buffer across all
        # tracks (all tracks share the same camera frame schedule). Used by
        # _transform_means to binary-search a batch's timestamp_us → pose idx.
        # None when no tracks (single-bg / road-only multi-layer).
        if tracks is not None:
            self._populate_tracks_impl(tracks)

    # ------------------------------------------------------------------ V3 Stage A: learnable pose helpers

    def _read_learnable_pose_flag(self) -> bool:
        """Read ``conf.trainer.learnable_pose.enabled`` defensively.

        Defaults to ``False`` so legacy configs / mock configs without the
        ``learnable_pose`` subtree stay on the v2 buffer path (byte-identical).
        """
        trainer_conf = getattr(self.conf, "trainer", None)
        if trainer_conf is None:
            return False
        lp = trainer_conf.get("learnable_pose", None) if hasattr(trainer_conf, "get") \
            else getattr(trainer_conf, "learnable_pose", None)
        if lp is None:
            return False
        enabled = lp.get("enabled", False) if hasattr(lp, "get") \
            else getattr(lp, "enabled", False)
        return bool(enabled)

    def _iter_track_tids(self) -> List[str]:
        """Sorted list of track ids currently registered.

        Source of truth: ``_track_active_<tid>`` buffers (registered in both
        buffer and learnable modes). Sorted to match the
        ``track_ids`` indexing convention (``sorted(tracks_poses.keys())``)
        used by ``init_layer_from_points`` (L951) — see observation #677.
        """
        prefix = "_track_active_"
        tids = [name[len(prefix):] for name in self._buffers
                if name.startswith(prefix)]
        tids.sort()
        return tids

    # ─── V3 Stage D.2 — pose_source flag for A/B render comparison ──────────
    # "learned" (default) → compose pose from _track_quat / _track_trans
    #                       Parameters; gradient-tracking; what training uses.
    # "gt"                → return frozen _track_pose_gt_<tid> buffer slice;
    #                       no gradient. For render_learned_vs_gt.py to
    #                       toggle at inference time without re-initializing
    #                       the model. Falls through to learned in buffer-
    #                       only / legacy mode (no _track_pose_gt_ exists).
    def set_pose_source(self, source: str) -> None:
        if source not in ("learned", "gt"):
            raise ValueError(f"pose_source must be 'learned' or 'gt', got {source!r}")
        self._pose_source = source

    @property
    def pose_source(self) -> str:
        return getattr(self, "_pose_source", "learned")

    def _get_track_pose_F(self, tid: str) -> int:
        """Number of frames in a track's pose schedule. Mode-agnostic."""
        # Learnable: quat[F, 4]; buffer: pose[F, 4, 4]; active[F]
        q = getattr(self, f"_track_quat_{tid}", None)
        if q is not None:
            return int(q.shape[0])
        p = getattr(self, f"_track_pose_{tid}", None)
        if p is not None:
            return int(p.shape[0])
        a = getattr(self, f"_track_active_{tid}", None)
        if a is not None:
            return int(a.shape[0])
        raise KeyError(f"No registered pose/active state for track '{tid}'")

    def _compose_pose_for_track(self, tid: str, idx: int) -> torch.Tensor:
        """Return ``[4, 4]`` SE(3) pose for ``tid`` at frame ``idx``.

        Learnable mode: composes a fresh rotation matrix from the wxyz quat
        Parameter (normalized) and trans Parameter — gradients flow back to
        both. Buffer mode: returns the stored ``_track_pose_<tid>[idx]``
        slice (no gradient). Mode is detected per call so a model whose
        ``populate_tracks`` half-restored from a buffer-mode ckpt into a
        learnable-mode session still works (see ``_populate_tracks_impl``
        cross-mode adoption path).

        V3 Stage D.2: when ``self.pose_source == "gt"`` and a frozen
        ``_track_pose_gt_<tid>`` buffer is registered (learnable mode only),
        return that slice instead — bypasses learned drift for A/B viz.
        """
        # V3 Stage D.2: GT route (only meaningful in learnable mode).
        if self.pose_source == "gt":
            pose_gt = getattr(self, f"_track_pose_gt_{tid}", None)
            if pose_gt is not None:
                return pose_gt[idx]
        # Buffer mode (or legacy ckpt restored before mode flip).
        pose_buf = getattr(self, f"_track_pose_{tid}", None)
        if pose_buf is not None:
            return pose_buf[idx]
        q_all = getattr(self, f"_track_quat_{tid}", None)
        t_all = getattr(self, f"_track_trans_{tid}", None)
        if q_all is None or t_all is None:
            raise KeyError(f"No pose state for track '{tid}'")
        q = q_all[idx]
        q = q / q.norm().clamp(min=1e-12)             # Adam breaks unit-norm — renormalize per access
        R = _quat_wxyz_to_rotmat(q)                   # [3, 3]
        T = q.new_zeros(4, 4)
        T[:3, :3] = R
        T[:3, 3]  = t_all[idx]
        T[3, 3]   = 1.0
        return T

    def _compose_pose_all_frames(self, tid: str) -> torch.Tensor:
        """Batched variant of ``_compose_pose_for_track`` over all F frames.

        Returns ``[F, 4, 4]``. Used by the ``tracks_poses`` ``@property``,
        ``get_model_parameters``-time persistence, and any caller that wants
        the full schedule for one track. Gradient-tracking in learnable mode.

        V3 Stage D.2: see ``_compose_pose_for_track`` for ``pose_source``
        flag semantics — same gt-routing applies here for full-schedule
        callers (e.g. ``tracks_poses`` property used by viz).
        """
        # V3 Stage D.2: GT route (only meaningful in learnable mode).
        if self.pose_source == "gt":
            pose_gt = getattr(self, f"_track_pose_gt_{tid}", None)
            if pose_gt is not None:
                return pose_gt
        pose_buf = getattr(self, f"_track_pose_{tid}", None)
        if pose_buf is not None:
            return pose_buf
        q_all = getattr(self, f"_track_quat_{tid}", None)
        t_all = getattr(self, f"_track_trans_{tid}", None)
        if q_all is None or t_all is None:
            raise KeyError(f"No pose state for track '{tid}'")
        q = q_all / q_all.norm(dim=-1, keepdim=True).clamp(min=1e-12)  # [F, 4]
        R = _quat_wxyz_to_rotmat(q)                                    # [F, 3, 3]
        F = q.shape[0]
        T = q.new_zeros(F, 4, 4)
        T[:, :3, :3] = R
        T[:, :3,  3] = t_all
        T[:,  3,  3] = 1.0
        return T

    @property
    def tracks_poses(self) -> dict:
        """Dynamic ``{tid: Tensor[F, 4, 4]}`` derived from the registered
        ``_track_pose_<tid>`` buffers OR ``_track_quat_<tid>`` + ``_track_trans_<tid>``
        Parameters (learnable mode). Keys are returned in ``sorted(tids)`` order
        — same convention ``init_layer_from_points`` uses to assign per-particle
        ``track_ids``. Computed on every access; in learnable mode the tensors
        are differentiable (do NOT cache the dict across train steps).
        """
        return {tid: self._compose_pose_all_frames(tid)
                for tid in self._iter_track_tids()}

    @property
    def tracks_active(self) -> dict:
        """Dynamic ``{tid: BoolTensor[F]}`` derived from the registered
        ``_track_active_<tid>`` buffers. Never learnable.
        """
        out = {}
        for tid in self._iter_track_tids():
            buf = getattr(self, f"_track_active_{tid}", None)
            if buf is not None:
                out[tid] = buf
        return out

    # ------------------------------------------------------------------ checkpoint

    def get_model_parameters(self) -> dict:
        """Return NRE-aligned sub-dict.

        Output (consumed by Trainer.save_checkpoint, which adds outer wrappers
        like 'global_step', 'epoch', strategy state, post_processing):

            {
                "gaussians_nodes": {layer_name: <MoG.get_model_parameters()>},
                "scene_extent": float,
            }

        After Trainer writes this under the 'model' key, the on-disk ckpt is:
            ckpt["model"]["gaussians_nodes"]["background"]["positions"]
        matching NRE (nvcr.io/nvidia/nre/nre-ga:latest) output format.

        T5.4: non-particle modules (sky_envmap) are stored under separate
        sibling keys via nn.Module.state_dict(), since they don't implement
        the MoG ``get_model_parameters`` contract.
        """
        out: dict = {
            "gaussians_nodes": {
                s.name: self.layers[s.name].get_model_parameters()
                for s in self.specs
                if s.is_particle_layer and s.name in self.layers
            },
            "scene_extent": self.scene_extent,
        }
        # T8/B3 Phase E.4: persist per-layer ``track_ids`` buffer (set by
        # ``init_layer_from_points`` at L713 for the dynamic_rigids layer).
        # Without this, viewer/playground loads of v2 ckpts lose the
        # per-particle owner mapping → ``_transform_means`` indexes pose_stack
        # incorrectly → all dyn particles render at track[0]'s pose → "勾掉
        # dynamic_rigids 视觉无变化". MoG.get_model_parameters omits this
        # buffer because it's a LayeredGaussians-only field; we inject here.
        for spec in self.specs:
            if not spec.is_particle_layer or spec.name not in self.layers:
                continue
            layer = self.layers[spec.name]
            track_ids = getattr(layer, "track_ids", None)
            if track_ids is not None:
                out["gaussians_nodes"][spec.name]["track_ids"] = (
                    track_ids.detach().cpu().to(torch.int64)
                )
        # T5.4: sky envmap state — saved as raw state_dict so SkyEnvmapMLP /
        # SkyEnvmapCubemap parameters (base / Linear weights) round-trip.
        if "sky_envmap" in self.layers:
            out["sky_envmap_state"] = self.layers["sky_envmap"].state_dict()
        # V3 Stage A: LayeredGaussians-level per-track state — wxyz quat /
        # trans nn.Parameter (learnable mode) or _track_pose_<tid> buffer
        # (legacy/disabled mode), _track_active_<tid> buffers, and the frozen
        # _track_pose_gt_<tid> reference buffer. ``get_model_parameters``
        # returns a structured dict (NOT a flat nn.Module state_dict), so
        # these would otherwise be silently dropped on save — fatal for
        # learnable_pose since the Adam-updated quat/trans Parameters carry
        # the only persistent copy of the refined pose. Sibling buffers
        # ``tracks_camera_timestamps_us`` and ``track_ids`` are still re-
        # derived by populate_tracks() / init_layer_from_points() each
        # session, so they don't need a slot here.
        layered_track_state = {
            k: v.detach().cpu() for k, v in self.state_dict().items()
            if k.startswith("_track_")
        }
        if layered_track_state:
            out["layered_track_state"] = layered_track_state
        return out

    def init_from_checkpoint(self, checkpoint: dict, setup_optimizer: bool = True):
        """Dispatch checkpoint into per-layer MoG. Accepts three input shapes:

        (a) NRE-wrapped v2:    checkpoint["model"]["gaussians_nodes"][name]
        (b) already-unwrapped: checkpoint["gaussians_nodes"][name]
        (c) v1 legacy flat:    checkpoint["positions"], ... (no nesting)
                               -> route entire dict into layers["background"]

        Missing layers warn + skip. Extra ckpt keys ignored silently.
        """
        nodes_dict = None
        if (
            "model" in checkpoint
            and isinstance(checkpoint["model"], dict)
            and "gaussians_nodes" in checkpoint["model"]
        ):
            nodes_dict = checkpoint["model"]["gaussians_nodes"]
        elif "gaussians_nodes" in checkpoint:
            nodes_dict = checkpoint["gaussians_nodes"]

        if nodes_dict is not None:
            # v2 path: dispatch per layer.
            for name, layer in self.layers.items():
                # T5.4: sky envmap (non-particle) restored separately below,
                # not from gaussians_nodes — skip the warn+miss path for it.
                if not hasattr(layer, "init_from_checkpoint"):
                    continue
                if name not in nodes_dict:
                    logger.warning(
                        f"[ckpt] Layer '{name}' not found in checkpoint "
                        f"['gaussians_nodes']; keeping it empty (warn+skip)."
                    )
                    continue
                layer.init_from_checkpoint(
                    nodes_dict[name], setup_optimizer=setup_optimizer
                )
                # T8/B3 Phase E.4: restore per-layer ``track_ids`` buffer if it
                # was saved by ``get_model_parameters``. MoG.init_from_checkpoint
                # reads only its 6 standard params (positions/density/etc) and
                # silently ignores extra keys, so the track_ids entry rides
                # along untouched in nodes_dict[name].
                ckpt_track_ids = nodes_dict[name].get("track_ids")
                if ckpt_track_ids is not None:
                    if not torch.is_tensor(ckpt_track_ids):
                        ckpt_track_ids = torch.as_tensor(ckpt_track_ids)
                    # Drop any prior buffer so re-load is idempotent.
                    if hasattr(layer, "track_ids"):
                        delattr(layer, "track_ids")
                    layer.register_buffer(
                        "track_ids", ckpt_track_ids.long(), persistent=True,
                    )
            # T5.4: sky envmap state restore — under either NRE-wrapped key
            # "model.sky_envmap_state" or unwrapped "sky_envmap_state".
            sky_state = None
            if (
                "model" in checkpoint
                and isinstance(checkpoint["model"], dict)
                and "sky_envmap_state" in checkpoint["model"]
            ):
                sky_state = checkpoint["model"]["sky_envmap_state"]
            elif "sky_envmap_state" in checkpoint:
                sky_state = checkpoint["sky_envmap_state"]
            if sky_state is not None and "sky_envmap" in self.layers:
                self.layers["sky_envmap"].load_state_dict(sky_state)
            # V3 Stage A: restore LayeredGaussians-level per-track state.
            # Must run AFTER the trainer has called populate_tracks() (it does
            # so before init_from_checkpoint, see trainer.init_model L386-449),
            # so the buffer/Parameter slots already exist and load_state_dict
            # only has to fill them. ``strict=False`` because we only carry
            # _track_* entries — everything else is filled by the per-layer
            # MoG init_from_checkpoint calls above.
            track_state = None
            if (
                "model" in checkpoint
                and isinstance(checkpoint["model"], dict)
                and "layered_track_state" in checkpoint["model"]
            ):
                track_state = checkpoint["model"]["layered_track_state"]
            elif "layered_track_state" in checkpoint:
                track_state = checkpoint["layered_track_state"]
            if track_state is not None and len(track_state) > 0:
                missing_keys, unexpected_keys = self.load_state_dict(
                    track_state, strict=False,
                )
                # ``missing_keys`` will contain every non-track key (per-layer
                # MoG params, sky_envmap, etc.) — those are filled by the
                # paths above, so we DON'T warn on them. ``unexpected_keys``
                # signals a true ckpt schema mismatch.
                if unexpected_keys:
                    logger.warning(
                        f"[ckpt] Unexpected layered_track_state keys (first "
                        f"5): {unexpected_keys[:5]} — pose state may be "
                        f"partially restored"
                    )
            # T8.12 fix: ckpt state_dicts hold CPU tensors and load_state_dict
            # does not migrate device. Trainer-path eval always calls
            # ``model.cuda()`` after construction so this is invisible there;
            # the playground engine path (Engine3DGRUT.load_3dgrt_object) does
            # NOT call .cuda(), so SkyEnvmapMLP layer weights stay on CPU and
            # the multi-layer _blend_sky path crashes with ``cpu vs cuda:0``
            # in addmm on first browser render. Migrate the whole ModuleDict
            # here so all entry points (trainer, eval, playground,
            # inject_viz_4d) see a consistent device state.
            if torch.cuda.is_available():
                self.cuda()
            return

        # v1 legacy path: route entire flat dict into background.
        if "background" not in self.layers:
            enabled = [s.name for s in self.specs]
            raise ValueError(
                f"v1 checkpoint detected (flat schema with 'positions' key) "
                f"but 'background' layer is not in conf.layers.enabled "
                f"(got {enabled}). Add 'background' to layers.enabled to "
                f"resume v1 checkpoints."
            )
        n_particles = checkpoint["positions"].shape[0]
        logger.info(
            f"[v1->v2] Detected v1-shape checkpoint ({n_particles} particles); "
            f"routing all into layer 'background'."
        )
        self.layers["background"].init_from_checkpoint(
            checkpoint, setup_optimizer=setup_optimizer
        )

    # ------------------------------------------------------------------ test helpers

    def setup_optimizer_for_test(self):
        """Minimal optimizer attach for unit tests; avoids touching Trainer.

        Plays the role of MoG.setup_optimizer() but skips conf-driven schedulers
        and per-name LR multipliers (the test only needs get_model_parameters()
        to pass its assert). T5.4: non-particle layers (sky_envmap) get a flat
        Adam over ``.parameters()`` instead of the MoG 6-group split.
        """
        for spec in self.specs:
            if spec.name not in self.layers:
                continue
            layer = self.layers[spec.name]
            if spec.is_particle_layer:
                layer.optimizer = torch.optim.Adam(
                    [
                        {"params": [layer.positions],         "name": "positions"},
                        {"params": [layer.rotation],          "name": "rotation"},
                        {"params": [layer.scale],             "name": "scale"},
                        {"params": [layer.density],           "name": "density"},
                        {"params": [layer.features_albedo],   "name": "features_albedo"},
                        {"params": [layer.features_specular], "name": "features_specular"},
                    ],
                    lr=1e-3,
                )
            else:
                layer.optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)

    # ------------------------------------------------------------------ single-layer bridge
    # When the container holds exactly one "background" layer (T1.1 default),
    # forward attribute reads and Parameter writes through to that layer so the
    # existing Trainer + MCMCStrategy + renderer code paths continue working.

    def forward(self, gpu_batch, train: bool = False, frame_id: int = 0):
        """Render the scene through whatever layer setup we have.

        Single-bg mode (T1.1 fast path): directly call the bg layer's
        ``forward`` → byte-identical with v1 (no allocations, no wrapping).

        Multi-layer mode (T3.5): build a fused-tensor view across all particle
        layers via ``fused_view(frame_id)`` (dynamic layer positions are
        world-transformed inline by T4.3), wrap it in a ``_FusedView`` so the
        renderer's MoG-style attribute access works unchanged, then dispatch
        to the reference (first) layer's renderer. All layers share conf so
        renderer params (camera model, sensor params, tracer settings) are
        identical regardless of which layer we pick.
        """
        bg = self._single_bg_layer()
        if bg is not None:
            # Viser layer toggle: in the v1-equivalent single-bg path, "off"
            # means "render nothing"; sky has no entry in this setup.
            if "background" not in self.enabled_layer_names:
                return self._empty_render(gpu_batch)
            return bg(gpu_batch, train=train, frame_id=frame_id)

        # T4.5: use absolute camera END timestamp_us from the batch as the
        # universal time coordinate for dyn pose lookup (binary-search in
        # the shared tracks_camera_timestamps_us buffer). frame_id arg from
        # trainer is global_step for tracer NVTX, not a dataset index, and
        # using it as a dyn pose index causes timing drift (verified Stage 4
        # 10k run: late frames lost ~3 dB due to closest_idx vs cover_range
        # absolute-position mismatch). Fall back to -1 → no transform when
        # batch lacks timestamp_us (non-NCore datasets / inference).
        ts_us = getattr(gpu_batch, "timestamp_us", -1)
        # Multi-layer: fused view + reference layer renderer.
        # ref_layer must be a particle layer (sky module has no .renderer).
        # When viser toggles disable every particle layer we skip the OptiX
        # pass entirely and let _blend_sky composite onto a blank canvas.
        ref_layer = next(
            (self.layers[s.name] for s in self.specs
             if s.is_particle_layer and s.name in self.enabled_layer_names),
            None,
        )
        if ref_layer is None:
            return self._blend_sky(self._empty_render(gpu_batch), gpu_batch)
        fused = self.fused_view(timestamp_us=ts_us)
        view = _FusedView(fused, ref_layer)
        outputs = ref_layer.renderer.render(view, gpu_batch, train, frame_id)
        # T5.4: sky envmap blend (only if sky layer is present in the spec
        # list). Composites learned sky onto the rendered gaussians using the
        # tracer's accumulated alpha.
        return self._blend_sky(outputs, gpu_batch)

    # ------------------------------------------------------------------ empty render
    def _empty_render(self, gpu_batch) -> dict:
        """Zero-RGB / zero-alpha render dict for the all-layers-disabled case.

        Returned when viser_gui_4d's per-layer toggles leave no particle layer
        enabled, so we never call ``ref_layer.renderer.render`` on an empty
        fused view (avoids an OptiX pass and undefined renderer behavior).
        Sky compositing in ``_blend_sky`` still runs on top of this, so a
        sky-only image is recoverable when only background/road/dyn are off.
        """
        B, H, W = gpu_batch.rays_dir.shape[:3]
        device = gpu_batch.rays_dir.device
        return {
            "pred_rgb":     torch.zeros(B, H, W, 3, device=device),
            "pred_opacity": torch.zeros(B, H, W, 1, device=device),
            "pred_dist":    torch.zeros(B, H, W, 1, device=device),
        }

    # ------------------------------------------------------------------ T5.4 sky blend
    def _blend_sky(self, outputs: dict, batch) -> dict:
        """Composite per-pixel sky RGB onto the Gaussian render using alpha.

        ``rgb_final = rgb_gauss + rgb_sky * (1 - alpha)`` where ``alpha`` is
        the tracer's accumulated opacity (``outputs["pred_opacity"]``).

        No-op when the sky layer is not in ``self.layers``. Always preserves
        the original Gaussian RGB under the ``rgb_gaussians`` key so
        ``get_losses`` can compute a sky-only L1 on the pre-blend signal.
        """
        if ("sky_envmap" not in self.layers
                or "sky_envmap" not in self.enabled_layer_names):
            return outputs
        sky_module = self.layers["sky_envmap"]
        # viewdirs in world frame: rays_dir is camera-space when
        # rays_in_world_space=False; apply T_to_world's rotation only.
        rays = batch.rays_dir   # [B, H, W, 3]
        if not getattr(batch, "rays_in_world_space", False):
            R = batch.T_to_world[..., :3, :3]                # [B, 3, 3]
            # Broadcast [B, 1, 1, 3, 3] @ [B, H, W, 3, 1] → [B, H, W, 3, 1]
            rays = (R[:, None, None, :, :] @ rays.unsqueeze(-1)).squeeze(-1)
        rgb_sky = sky_module(rays)                           # [B, H, W, 3]
        rgb_gauss = outputs["pred_rgb"]
        alpha = outputs["pred_opacity"]                      # [B, H, W, 1]
        rgb_final = rgb_gauss + rgb_sky * (1.0 - alpha)
        # Shallow copy so we don't mutate the renderer's returned dict.
        out = dict(outputs)
        out["rgb_gaussians"] = rgb_gauss
        out["rgb_sky"] = rgb_sky
        out["pred_rgb"] = rgb_final
        return out

    def _single_bg_layer(self):
        """Return the bg layer iff this is a single-bg-layer setup, else None."""
        modules = self.__dict__.get("_modules", {})
        layers = modules.get("layers")
        if layers is not None and len(layers) == 1 and "background" in layers:
            return layers["background"]
        return None

    # ------------------------------------------------------------------ T2.5 fused-view
    def fused_view(self, frame_id: int | None = None, *,
                   timestamp_us: int = -1) -> dict[str, torch.Tensor]:
        """Return concat of 6 per-particle parameters across all particle layers.

        Single-bg-layer mode: short-circuits to direct attribute access (returns
        the bg layer's Parameters themselves, byte-identical to the v1 path).

        Multi-layer mode: returns `torch.cat(..., dim=0)` across layers in
        `self.specs` order (particle layers only).

        T4.3/T4.5 dynamic-rigid handling: when a particle layer is named
        ``dynamic_rigids`` and ``timestamp_us > 0``, its `positions` (stored
        in object-local frame) are routed through ``_transform_means`` using
        the per-track pose at that absolute camera timestamp (binary-search
        in the shared ``tracks_camera_timestamps_us`` buffer). When neither
        timestamp_us nor frame_id is provided (e.g. inference free camera),
        the dynamic layer's positions are passed through unchanged (Stage 8
        "nearest active frame" fallback TODO, D4).

        ``frame_id`` arg kept for backward compat (T2.5 test helper, T4.3
        unit tests). New code paths should pass timestamp_us instead.
        """
        bg = self._single_bg_layer()
        if bg is not None:
            return {n: getattr(bg, n) for n in _FORWARD_PARAM_NAMES}

        pieces: dict[str, list[torch.Tensor]] = {n: [] for n in _FORWARD_PARAM_NAMES}
        for spec in self.specs:
            if not spec.is_particle_layer:
                continue
            # Viser layer toggle: skip disabled particle layers so their
            # tensors never reach OptiX (cleanest + cheapest "hide" semantics).
            if spec.name not in self.enabled_layer_names:
                continue
            layer = self.layers[spec.name]
            # T8/B3 Phase E.3: compute transformed positions + per-particle
            # active mask once for dynamic_rigids; then use active mask to
            # suppress inactive-track particles in the density field.
            transformed_positions = None
            active_mask = None
            transformed_rotations = None
            if (
                spec.name == "dynamic_rigids"
                and hasattr(layer, "track_ids")
                and len(self.tracks_poses) > 0
            ):
                # E.2.c: always transform when track buffers are populated.
                # ``_transform_means_and_active`` handles the no-time fallback
                # (each track uses its first active frame) so inference free
                # cameras don't dump dyn particles to world origin.
                transformed_positions, active_mask, transformed_rotations = \
                    self._transform_means_and_active(
                        layer.positions, layer.track_ids,
                        rotations_local=layer.rotation,
                        timestamp_us=timestamp_us, frame_id=frame_id,
                    )
            for n in _FORWARD_PARAM_NAMES:
                v = getattr(layer, n)
                if n == "positions" and transformed_positions is not None:
                    v = transformed_positions
                elif n == "rotation" and transformed_rotations is not None:
                    # Phase E.2.b: q_world = q_pose ⊗ q_local — without this
                    # composition, MCMC inflates scales to compensate for the
                    # missing orientation, producing scenes-wide smudge.
                    v = transformed_rotations
                elif (
                    n == "density"
                    and active_mask is not None
                    and not bool(active_mask.all())
                ):
                    # density is [N, 1] raw (pre-sigmoid). Push inactive
                    # particles to a large-negative value so sigmoid(density)
                    # ≈ 0 → no render contribution. No mutation of the
                    # underlying nn.Parameter (only this fused view copy).
                    inactive_value = torch.full_like(v, -50.0)
                    v = torch.where(
                        active_mask.unsqueeze(-1), v, inactive_value,
                    )
                pieces[n].append(v)
        # All particle layers disabled → return 0-row tensors with correct
        # trailing dims so callers (and _FusedView consumers) never trip on
        # torch.cat([]). forward() short-circuits before reaching this path
        # in the all-off case, but a defensive guard here lets fused_view be
        # called independently (e.g. from tests).
        first_param = next(iter(_FORWARD_PARAM_NAMES))
        if not pieces[first_param]:
            ref = next(iter(self.layers.values()))
            return {n: getattr(ref, n).new_zeros((0,) + getattr(ref, n).shape[1:])
                    for n in _FORWARD_PARAM_NAMES}
        return {n: torch.cat(pieces[n], dim=0) for n in _FORWARD_PARAM_NAMES}

    # ------------------------------------------------------------------ T4.5 transform_means (timestamp-aligned)
    def _resolve_pose_idx(self, timestamp_us: int, frame_id: int | None) -> int:
        """Convert (timestamp_us | frame_id) → pose-buffer index.

        Priority:
          1. timestamp_us > 0 + tracks_camera_timestamps_us populated →
             binary-search nearest cam timestamp index. This is the
             T4.5 correct path (universal time coordinate).
          2. frame_id is not None → use as-is, clamped to [0, F-1].
             Backward compat for T2.5 / T4.3 unit tests and inference
             callers without timestamp_us.
          3. Else → 0 (defensive fallback).
        """
        # V3 Stage A: derive F via the mode-agnostic helper instead of touching
        # ``_track_pose_<tid>`` directly (that buffer doesn't exist in learnable
        # mode — the wxyz quat Parameter does). ``tracks_active`` keys are the
        # source of truth in both modes.
        tids = self._iter_track_tids()
        if not tids:
            return 0
        any_track = tids[0]
        F = self._get_track_pose_F(any_track)

        if timestamp_us > 0 and hasattr(self, "tracks_camera_timestamps_us"):
            ts_buf = self.tracks_camera_timestamps_us
            # Binary search; clamp to [0, F-1]; check whether prev is closer.
            idx = int(torch.searchsorted(
                ts_buf, torch.tensor(int(timestamp_us), dtype=ts_buf.dtype, device=ts_buf.device)
            ).item())
            idx = max(0, min(idx, F - 1))
            if idx > 0:
                d_curr = abs(int(ts_buf[idx].item()) - int(timestamp_us))
                d_prev = abs(int(ts_buf[idx - 1].item()) - int(timestamp_us))
                if d_prev < d_curr:
                    idx -= 1
            return idx
        if frame_id is not None:
            return max(0, min(int(frame_id), F - 1))
        return 0

    def _transform_means_and_active(
        self,
        positions_local: torch.Tensor,
        track_ids: torch.Tensor,
        rotations_local: torch.Tensor | None = None,
        timestamp_us: int = -1,
        frame_id: int | None = None,
    ):
        """Like ``_transform_means`` but also returns ``(active_mask,
        rotations_world)``.

        Phase E.3: ``active_mask`` lets fused_view suppress inactive-track
        particles by overriding their density to a large-negative sentinel.

        Phase E.2.b: when ``rotations_local`` is provided, compose the
        per-track pose rotation with each particle's local rotation
        ``q_world = q_pose ⊗ q_local`` so the renderer sees the rotated
        Gaussian's covariance pointing along the cuboid's natural axes
        (not the world axes). Without this, a yaw=π/2 car's particles
        sit at the right world coords but face the wrong direction →
        MCMC inflates ``scale`` to multiple meters to compensate → fused
        view becomes a sky-wide smudge (observed in E.7 5k fix ckpt).

        Returns:
            (positions_world ``[N, 3]``,
             active_per_pt ``[N]`` bool,
             rotations_world ``[N, 4]`` or None if ``rotations_local`` is None)
            ``active_per_pt`` is all True when ``_track_active_<n>`` buffers
            are missing (defensive — pre-T4.0 legacy code paths).
        """
        track_names = sorted(self.tracks_poses.keys())
        device = positions_local.device
        track_ids = track_ids.to(device)

        # E.2.c: free-camera fallback. When the caller has no time signal
        # (inference / playground rendering a static view), pick each track's
        # FIRST ACTIVE FRAME independently so the user sees a composite "all
        # visible actors" scene instead of dyn particles snapping to world
        # origin (which is what the old ``timestamp_us > 0 or frame_id is not
        # None`` gate in fused_view used to do).
        use_per_track_first_active = (timestamp_us <= 0 and frame_id is None)

        if use_per_track_first_active:
            pose_list = []
            active_track_flags = []
            for n in track_names:
                active_buf = getattr(self, f"_track_active_{n}", None)
                if active_buf is None or active_buf.numel() == 0:
                    fallback_idx = 0
                    is_active = True
                else:
                    nonzero = active_buf.nonzero(as_tuple=False)
                    if nonzero.numel() == 0:
                        fallback_idx = 0
                        is_active = False    # track has no active frames at all
                    else:
                        fallback_idx = int(nonzero[0, 0].item())
                        is_active = True
                # V3 Stage A: route through _compose_pose_for_track so the
                # learnable-pose Parameter path produces a fresh, gradient-
                # tracking SE(3) from quat+trans. Buffer path is unchanged.
                pose_list.append(
                    self._compose_pose_for_track(n, fallback_idx).to(device)
                )
                active_track_flags.append(
                    torch.tensor(is_active, dtype=torch.bool, device=device)
                )
            pose_stack = torch.stack(pose_list)                              # [K, 4, 4]
            active_stack = torch.stack(active_track_flags)                   # [K] bool
        else:
            idx = self._resolve_pose_idx(timestamp_us, frame_id)
            # V3 Stage A: see comment above — _compose_pose_for_track is
            # mode-agnostic.
            pose_stack = torch.stack(
                [self._compose_pose_for_track(n, idx).to(device) for n in track_names]
            )                                                                # [K, 4, 4]
            # Per-track active flag at this frame; defaults to all-True when
            # the _track_active_<n> buffer is missing (pre-T4.0).
            active_list = []
            for n in track_names:
                buf = getattr(self, f"_track_active_{n}", None)
                if buf is None:
                    active_list.append(
                        torch.tensor(True, dtype=torch.bool, device=device)
                    )
                else:
                    active_list.append(buf[idx].to(device).to(torch.bool))
            active_stack = torch.stack(active_list)                          # [K] bool

        pose_per_pt = pose_stack[track_ids]                                  # [N, 4, 4]
        R = pose_per_pt[:, :3, :3]                                           # [N, 3, 3]
        t = pose_per_pt[:, :3, 3]                                            # [N, 3]
        positions_world = (R @ positions_local.to(R.dtype).unsqueeze(-1)).squeeze(-1) + t
        active_per_pt = active_stack[track_ids]                              # [N] bool

        # Phase E.2.b: compose pose rotation with per-particle local rotation.
        # Done in quaternion space (wxyz). When rotations_local is omitted
        # (legacy callers / unit tests), return None so the caller keeps the
        # original behaviour.
        rotations_world = None
        if rotations_local is not None:
            # pose_stack[..., :3, :3] is the rotation block per track; convert
            # ONCE per track (not per particle) for efficiency, then gather.
            pose_R_per_track = pose_stack[:, :3, :3].to(rotations_local.dtype)  # [K, 3, 3]
            q_pose_per_track = _rotmat_to_quat_wxyz(pose_R_per_track)            # [K, 4]
            q_pose_per_pt = q_pose_per_track[track_ids]                          # [N, 4]
            q_local = rotations_local.to(device=device, dtype=q_pose_per_pt.dtype)
            q_world = _quat_multiply_wxyz(q_pose_per_pt, q_local)                # [N, 4]
            # Renderer / MoG conventionally re-normalizes via rotation_activation
            # ("normalize" in config) but it's cheap to do so here too — bounds
            # numerical drift from float32 multiplication.
            q_world = q_world / q_world.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            rotations_world = q_world
        return positions_world, active_per_pt, rotations_world

    def _transform_means(
        self,
        positions_local: torch.Tensor,
        track_ids: torch.Tensor,
        timestamp_us: int = -1,
        frame_id: int | None = None,
    ) -> torch.Tensor:
        """Apply per-particle ``object → world`` SE(3) using the per-track
        pose at the frame nearest ``timestamp_us`` (T4.5 timestamp-aligned).

        Pose source: ``self.tracks_poses`` buffer dict (populate_tracks).
        Particle-to-track routing: per-particle ``track_ids`` int buffer
        on the layer, mapped via ``sorted(self.tracks_poses.keys())`` to
        the dict insertion order used by ``init_dynamic_rigid_layer``.

        Args:
            positions_local: ``[N, 3]`` object-local positions.
            track_ids:       ``[N]`` int64; values in ``[0, len(tracks_poses))``.
            timestamp_us:    absolute camera END timestamp (preferred).
            frame_id:        legacy index for backward-compat callers.

        Returns:
            ``[N, 3]`` world-frame positions.
        """
        track_names = sorted(self.tracks_poses.keys())
        device = positions_local.device
        idx = self._resolve_pose_idx(timestamp_us, frame_id)
        # Stack poses for this frame: [K, 4, 4] on positions device.
        # Buffers may sit on CPU even after layer Parameters move to CUDA
        # (LayeredGaussians has no explicit .to() call from trainer), so
        # sync each per-track pose tensor to the positions device.
        # V3 Stage A: _compose_pose_for_track is mode-agnostic (buffer or
        # learnable quat+trans Parameter); same call shape in both paths.
        pose_stack = torch.stack(
            [self._compose_pose_for_track(n, idx).to(device) for n in track_names]
        )
        track_ids = track_ids.to(device)
        pose_per_pt = pose_stack[track_ids]                                   # [N, 4, 4]
        R = pose_per_pt[:, :3, :3]                                            # [N, 3, 3]
        t = pose_per_pt[:, :3, 3]                                             # [N, 3]
        return (R @ positions_local.to(R.dtype).unsqueeze(-1)).squeeze(-1) + t

    # ------------------------------------------------------------------ T3.0 init
    def init_layer_from_points(
        self,
        layer_name: str,
        positions: torch.Tensor,
        *,
        colors: Optional[torch.Tensor] = None,
        rotations: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        densities: Optional[torch.Tensor] = None,
        track_ids: Optional[torch.Tensor] = None,
        observer_pts: Optional[torch.Tensor] = None,
        setup_optimizer: bool = True,
    ) -> None:
        """Initialize one named layer's MoG parameters from a point cloud.

        Spec-aware defaults: when scales / densities / colors / rotations are
        omitted, fall back to ``LayerSpec.scale_prior`` (log-applied) /
        ``LayerSpec.density_init`` / identity quat / neutral gray (0.5).

        Args:
            layer_name: must be in ``self.layers``.
            positions: ``[N, 3]``. World frame for background / road;
                object-local frame for dynamic_rigids (pair with ``track_ids``).
            colors:    ``[N, 3]`` in [0, 1]; default neutral gray.
            rotations: ``[N, 4]`` quat wxyz; default identity ``(1, 0, 0, 0)``.
            scales:    ``[N, 3]`` log-space; default ``torch.log(spec.scale_prior)``.
            densities: ``[N, 1]`` log-space; default ``spec.density_init``.
            track_ids: ``[N]`` int64; only meaningful for dynamic_rigids.
                Registered as a persistent buffer ``track_ids`` on the layer.
            observer_pts: reserved; ignored in T3.0 (full Parameter path,
                no observer-distance scale estimation).
            setup_optimizer: when True (default), wire the per-layer Adam via
                ``layer.set_optimizable_parameters()`` + ``layer.setup_optimizer()``.
                Tests that bypass CUDA conf paths pass False.
        """
        if layer_name not in self.layers:
            raise ValueError(
                f"unknown layer '{layer_name}', enabled = {list(self.layers)}"
            )
        spec = next(s for s in self.specs if s.name == layer_name)
        layer = self.layers[layer_name]
        N = positions.shape[0]
        dtype = torch.float32
        device = positions.device  # T3.5.b: all defaults follow caller device

        if rotations is None:
            rotations = torch.zeros(N, 4, dtype=dtype, device=device)
            rotations[:, 0] = 1.0
        if scales is None:
            s_phys = torch.tensor(list(spec.scale_prior), dtype=dtype, device=device)
            scales = torch.log(s_phys).expand(N, 3).clone()
        if densities is None:
            densities = torch.full((N, 1), float(spec.density_init), dtype=dtype, device=device)
        if colors is None:
            colors = torch.full((N, 3), 0.5, dtype=dtype, device=device)

        features_albedo = (colors.to(dtype=dtype, device=device) - 0.5) / _SH_C0
        num_specular_dims = sh_degree_to_specular_dim(layer.max_n_features)
        features_specular = torch.zeros((N, num_specular_dims), dtype=dtype, device=device)

        # Tensors keep their incoming device. Caller (Trainer) is responsible
        # for putting them on GPU; tests stay on CPU. Note layer.device is
        # hardcoded to "cuda" in MoG.__init__ but is only consulted by code
        # paths that allocate new tensors (not by the assignments below).
        layer.positions         = nn.Parameter(positions.to(dtype=dtype, device=device))
        layer.rotation          = nn.Parameter(rotations.to(dtype=dtype, device=device))
        layer.scale             = nn.Parameter(scales.to(dtype=dtype, device=device))
        layer.density           = nn.Parameter(densities.to(dtype=dtype, device=device))
        layer.features_albedo   = nn.Parameter(features_albedo)
        layer.features_specular = nn.Parameter(features_specular)

        if setup_optimizer:
            layer.set_optimizable_parameters()
            layer.setup_optimizer()
            layer.validate_fields()

        if track_ids is not None:
            layer.register_buffer("track_ids", track_ids.long(), persistent=True)

    # ------------------------------------------------------------------ T4.5: populate tracks post-construct
    def populate_tracks(self, tracks: dict) -> None:
        """Register per-track pose / active buffers + shared timestamp buffer.

        Used by trainer.setup_training when the dataset (loader) becomes
        available — at __init__ time we typically don't have it yet, so
        tracks is None there. Idempotent: calling with the same track_id
        replaces the existing buffer; calling with new ids adds them.

        Args:
            tracks: ``{track_id: {poses[F,4,4], active|frame_info[F bool],
                                  cam_timestamps_us[F int64], ...}}``
                — schema from load_tracks_from_ncore_cuboids. All tracks
                share the same cam_timestamps_us (NCore camera schedule
                is sensor-driven, not track-specific).
        """
        self._populate_tracks_impl(tracks)

    def _populate_tracks_impl(self, tracks: dict) -> None:
        """Shared impl for __init__ and populate_tracks paths.

        V3 Stage A: branches on ``self._read_learnable_pose_flag()``.

        * **Buffer mode (default)** — legacy v2 path: register
          ``_track_pose_<tid>`` buffer ``[F, 4, 4]`` + ``_track_active_<tid>``
          BoolTensor[F]. Byte-identical with the pre-V3 code path.

        * **Learnable mode** — split each pose into wxyz quat Parameter[F,4]
          + trans Parameter[F,3]; also keep a frozen ``_track_pose_gt_<tid>``
          buffer for resume detection and (future) viz diff. Active mask
          remains a buffer.

        **Resume guard** — if a learnable-mode tid is repopulated and its
        ``_track_quat_<tid>`` Parameter already exists, the pose Parameter is
        NOT overwritten (we assume it's been training). Only the active mask
        and shared metadata are refreshed. This is what makes 2nd-train-from-ckpt
        not clobber learned pose.

        **Cross-mode adoption** — if learnable mode is on but only a legacy
        ``_track_pose_<tid>`` buffer is present (resume from a buffer-mode
        ckpt), the buffer is converted to Parameter once, then deleted.
        """
        learnable_mode = self._read_learnable_pose_flag()

        # Shared cam timestamp buffer (single tensor across all tracks).
        # Take from the first track that supplies it; verify subsequent
        # tracks match (they should — same NCore loader, same camera).
        shared_ts = None
        for tid, info in tracks.items():
            ts = info.get("cam_timestamps_us") if isinstance(info, dict) else None
            if ts is not None:
                shared_ts = ts.to(torch.int64) if torch.is_tensor(ts) else torch.as_tensor(ts, dtype=torch.int64)
                break
        if shared_ts is not None:
            # Replace existing if any; persistent=True so save/load roundtrips.
            if hasattr(self, "tracks_camera_timestamps_us"):
                delattr(self, "tracks_camera_timestamps_us")
            self.register_buffer("tracks_camera_timestamps_us",
                                 shared_ts, persistent=True)

        # T8.2: track-level metadata (class label, cuboid size) that the viz_4d
        # ckpt block needs but the C++ tracer never consumes. Stored as a plain
        # Python dict (not register_buffer) since classes are str and sizes are
        # tiny 3-vectors — ride-along with save_checkpoint via extract_4d_metadata.
        if not hasattr(self, "tracks_metadata"):
            object.__setattr__(self, "tracks_metadata", {})
        for tid, info in tracks.items():
            poses = info["poses"] if isinstance(info, dict) else info[0]
            active = (info["active"] if isinstance(info, dict) and "active" in info
                      else info.get("frame_info") if isinstance(info, dict)
                      else info[1])
            buf_pose_name   = f"_track_pose_{tid}"
            buf_active_name = f"_track_active_{tid}"
            buf_gt_name     = f"_track_pose_gt_{tid}"
            param_quat_name = f"_track_quat_{tid}"
            param_trans_name = f"_track_trans_{tid}"

            # Active mask is identical in both modes (never learnable).
            if hasattr(self, buf_active_name):
                delattr(self, buf_active_name)
            self.register_buffer(buf_active_name, active.to(torch.bool),
                                 persistent=True)

            if learnable_mode:
                # Resume guard: keep existing Parameter (already trained).
                if (param_quat_name in self._parameters
                        and param_trans_name in self._parameters):
                    continue

                # Cross-mode adoption: drop any leftover buffers from previous
                # mode so register_parameter doesn't collide.
                for stale in (buf_pose_name, buf_gt_name, param_quat_name, param_trans_name):
                    if hasattr(self, stale):
                        delattr(self, stale)

                poses_f32 = poses.to(torch.float32)
                R_init = poses_f32[:, :3, :3]                     # [F, 3, 3]
                t_init = poses_f32[:, :3, 3].contiguous()         # [F, 3]
                q_init = _rotmat_to_quat_wxyz(R_init).contiguous() # [F, 4]
                self.register_parameter(param_quat_name,
                                        nn.Parameter(q_init.detach().clone()))
                self.register_parameter(param_trans_name,
                                        nn.Parameter(t_init.detach().clone()))
                # Frozen GT pose for resume detection + (future) viz diff.
                self.register_buffer(buf_gt_name, poses_f32.clone(), persistent=True)
            else:
                # Legacy buffer path (v2 baseline, byte-identical).
                if hasattr(self, buf_pose_name):
                    delattr(self, buf_pose_name)
                self.register_buffer(buf_pose_name, poses, persistent=True)

            # T8.2 — capture class/size when supplied by the loader. Falls back
            # silently when absent (e.g. legacy callers / partial dicts) so we
            # never block the existing pose+active contract.
            if isinstance(info, dict):
                meta = {}
                if "class" in info:
                    meta["class"] = str(info["class"])
                if "size" in info:
                    sz = info["size"]
                    meta["size"] = (sz.detach().to(dtype=torch.float32, device="cpu")
                                    if torch.is_tensor(sz)
                                    else torch.as_tensor(sz, dtype=torch.float32))
                if meta:
                    self.tracks_metadata[tid] = meta

    # ------------------------------------------------------------------ T3.5.b trainer compat
    def build_acc(self, rebuild: bool = True) -> None:
        """Multi-layer: forward build_acc to every particle layer.

        Single-bg mode goes through __getattr__ bridge (transparent v1 path).
        For 3DGUT this is a no-op; for 3DGRT it builds per-layer OptiX BVH.
        """
        bg = self._single_bg_layer()
        if bg is not None:
            bg.build_acc(rebuild=rebuild)
            return
        for layer in self.layers.values():
            # T5.4: skip sky module (no build_acc); only particle layers
            # ship BVH state to OptiX.
            if hasattr(layer, "build_acc"):
                layer.build_acc(rebuild=rebuild)

    def setup_optimizer(self, state_dict=None) -> None:
        """Multi-layer compat shim: each layer's optimizer is already
        configured by init_layer_from_points(..., setup_optimizer=True);
        this is a no-op when called again from trainer.setup_training.

        Single-bg mode passes through __getattr__ bridge so v1 behaviour
        (state_dict resume from ckpt etc.) is byte-identical.

        T5.4: also attaches an Adam to the sky_envmap module (multi-layer
        only). The sky lr defaults to 0.01 (drivestudio pvg.yaml) but is
        overrideable via ``conf.trainer.sky_lr``.
        """
        bg = self._single_bg_layer()
        if bg is not None:
            bg.setup_optimizer(state_dict=state_dict)
            return
        # Multi-layer: per-particle-layer optimizers already attached via
        # init_layer_from_points. Sky envmap needs one attached here.
        if "sky_envmap" in self.layers:
            sky = self.layers["sky_envmap"]
            # T5.4: sync sky module to the same device as the first particle
            # layer's parameters. Particle layers are moved to CUDA inside
            # init_layer_from_points (positions.to(device=positions.device)),
            # but sky was constructed in __init__ before the device was known
            # — without this sync its nn.Linear weights stay on CPU and the
            # forward fails with "Expected all tensors to be on the same
            # device". Runs after Trainer.setup_training calls .setup_optimizer().
            for spec in self.specs:
                if spec.is_particle_layer:
                    target_device = self.layers[spec.name].positions.device
                    sky.to(target_device)
                    break
            if getattr(sky, "optimizer", None) is None:
                trainer_conf = getattr(self.conf, "trainer", None)
                if trainer_conf is None:
                    sky_lr = 0.01
                else:
                    sky_lr = (
                        trainer_conf.get("sky_lr", 0.01)
                        if hasattr(trainer_conf, "get")
                        else getattr(trainer_conf, "sky_lr", 0.01)
                    )
                sky.optimizer = torch.optim.Adam(sky.parameters(), lr=float(sky_lr))
        if state_dict is not None:
            logger.warning(
                "LayeredGaussians.setup_optimizer: multi-layer mode ignores "
                "state_dict (per-layer optimizers were set up at init); "
                "ckpt optimizer state restore not yet plumbed in v2 multi-layer."
            )

    # ------------------------------------------------------------------ T3.0 optimizer view
    @property
    def optimizer(self):
        """Drop-in optimizer view across all layers.

        Single-bg mode: return the bg layer's own optimizer directly, so
        ``self.model.optimizer.step()`` from the trainer is byte-identical
        with v1 (no wrapper allocation, no fan-out cost).

        Multi-layer mode: return a ``_LayeredOptimizerView`` that fans
        ``step()`` / ``zero_grad()`` / ``param_groups`` across each layer's
        sub-optimizer.
        """
        bg = self._single_bg_layer()
        if bg is not None and getattr(bg, "optimizer", None) is not None:
            return bg.optimizer
        return _LayeredOptimizerView(self.layers)

    def get_layer_mask(self, name: str) -> torch.Tensor:
        """Return Bool mask of shape [N_total] selecting particles of layer `name`.

        Mask layout follows the same particle-layer concat order as `fused_view`
        (specs order, non-particle layers skipped). Used by per-layer loss /
        region gating downstream.
        """
        particle_layers = [s for s in self.specs if s.is_particle_layer]
        if name not in {s.name for s in particle_layers}:
            raise ValueError(
                f"unknown layer '{name}' (or it is a non-particle layer); "
                f"particle layers: {[s.name for s in particle_layers]}"
            )

        total = 0
        target_start, target_end = -1, -1
        for spec in particle_layers:
            n_local = self.layers[spec.name].num_gaussians
            if spec.name == name:
                target_start, target_end = total, total + n_local
            total += n_local

        device = self.layers[name].positions.device
        mask = torch.zeros(total, dtype=torch.bool, device=device)
        mask[target_start:target_end] = True
        return mask

    def __getattr__(self, name):
        # nn.Module.__getattr__ is only invoked when normal attribute lookup
        # fails (so layers/specs/conf/scene_extent etc. handled by base class
        # first). Forward to bg layer when in single-layer mode.
        try:
            return super().__getattr__(name)
        except AttributeError:
            bg = self._single_bg_layer()
            if bg is not None:
                return getattr(bg, name)
            # T3.5.b multi-layer fused fallback for trainer MoG-style accessors.
            modules = self.__dict__.get("_modules", {})
            layers = modules.get("layers", {})
            specs = self.__dict__.get("specs") or []
            particle_layer_names = [s.name for s in specs if s.is_particle_layer]
            # Per-particle Parameter attributes -> concat across particle layers.
            if name in _FORWARD_PARAM_NAMES:
                pieces = [getattr(layers[n], name) for n in particle_layer_names]
                return torch.cat(pieces, dim=0) if pieces else torch.empty(0)
            if name == "num_gaussians":
                return sum(layers[n].num_gaussians for n in particle_layer_names)
            # MoG accessor methods -> return callable that fuses per-layer results.
            if name in ("get_density", "get_scale", "get_rotation",
                        "get_features", "get_features_albedo",
                        "get_features_specular", "get_positions"):
                def _fused(*args, **kwargs):
                    pieces = [getattr(layers[n], name)(*args, **kwargs)
                              for n in particle_layer_names]
                    return torch.cat(pieces, dim=0) if pieces else torch.empty(0)
                return _fused
            # Per-layer broadcast methods (no return-value fusion, just dispatch).
            # scheduler_step / setup_scheduler etc. step each layer independently.
            if name in ("scheduler_step", "setup_scheduler",
                        "validate_fields", "set_optimizable_parameters"):
                def _broadcast(*args, **kwargs):
                    for n in particle_layer_names:
                        getattr(layers[n], name)(*args, **kwargs)
                return _broadcast
            # Last-resort fallback: delegate to the first particle layer.
            # All layers share the same conf so scalar / config attributes
            # like progressive_training / max_n_features / n_active_features /
            # feature_dim_increase_interval / feature_dim_increase_step /
            # scene_extent are identical across layers and need no fusion.
            if particle_layer_names:
                return getattr(layers[particle_layer_names[0]], name)
            raise

    def __setattr__(self, name, value):
        # Redirect Parameter writes (e.g. MCMCStrategy.setattr(model, "positions", new))
        # to the single bg layer so its Parameter identity stays in sync with the
        # optimizer state. Non-bridged names fall through to nn.Module.__setattr__.
        if name in _FORWARD_PARAM_NAMES:
            bg = self._single_bg_layer()
            if bg is not None:
                setattr(bg, name, value)
                return
        super().__setattr__(name, value)
