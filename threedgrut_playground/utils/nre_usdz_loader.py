# SPDX-License-Identifier: Apache-2.0
"""Parse an NVIDIA NRE / NuRec ``*.usdz`` (training-checkpoint flavour) into a
3dgrut2-native checkpoint dict that ``engine.load_3dgrt_object`` / the viser 4D
viewer consume **unchanged**.

Design (see plan ``viser-gui-4d-py-nvidia-sorted-brooks.md``):

* The viewer already handles LayeredGaussians + dynamic tracks + FTheta + rig
  replay correctly. The ONLY job here is to *translate* the NRE
  ``checkpoint.ckpt`` (embedded in the usdz) into the exact dict shape the
  trainer writes (``ckpt["model"]["gaussians_nodes"][layer]`` + ``config`` +
  optional ``viz_4d``), then ``torch.save`` it. ``viser_gui_4d.py`` only gains a
  ``--usdz`` branch that converts then sets ``args.gs_object`` to the produced
  ``.pt`` — no render/track/metadata code changes.

What makes the NRE ckpt awkward (and why prior attempts went sideways):

* Its pickle references the ``nre`` python package (``No module named 'nre'``
  on a vanilla 3dgrut env). We bypass that with a *tolerant* unpickler that
  maps any unknown class to a capturing stub — the torch tensors rebuild
  normally, only the ``nre`` wrapper objects become inert stubs.
* Key names differ (``rotations``/``scales``/``densities`` vs the 3dgrut2
  singular ``rotation``/``scale``/``density``).
* ``features_albedo`` is per-gaussian *Fourier-in-time* with shape
  ``(N, fourier_features_dim, 3)`` (background K=5, dynamic_rigids K=20),
  whereas 3dgrut2's tracer wants ``(N, 3)`` DC. We evaluate the cosine-Fourier
  series at a chosen frame (reusing :func:`fourier_cos_basis`) → ``(N, 3)``.
  ``camera_extra_signal`` (N,20) is per-gaussian *semantic logits* (20 classes),
  feature-splatted through the camera for semantic supervision — NOT appearance.
  Dropping it costs zero RGB fidelity (just loses the semantic channel). The
  real per-camera photometric balancing lives in a separate image-level ISP
  (``post_processings.0.ppisp``: exposure/vignetting/color/CRF), which we also
  don't replicate — that (plus difix) is the actual appearance gap vs NVIDIA's
  renderer.

The pure-python helpers (unpickler / key rename / albedo eval) carry no cuda /
hydra import so they unit-test on a Mac. The model-build + ``torch.save``
orchestration (:func:`convert_usdz_to_pt`) needs the 3dgrut env + GPU.
"""
from __future__ import annotations

import io
import math
import pickle
import types
import zipfile
from pathlib import Path
from typing import Any, Optional

import torch

# Layers we translate from NRE → 3dgrut2. ``dynamic_deformables`` is
# intentionally absent: viser does not support its neural deform field (大g
# decision), and the multilayer registry doesn't enable it either.
NRE_PARTICLE_LAYERS = ("background", "road", "dynamic_rigids")

# NRE → 3dgrut2 per-node tensor key renames (singular).
_KEY_RENAME = {
    "rotations": "rotation",
    "scales": "scale",
    "densities": "density",
}


# --------------------------------------------------------------------------- #
# 1. Tolerant unpickler — load checkpoint.ckpt without the ``nre`` package.
# --------------------------------------------------------------------------- #
class _CapturingStub:
    """Stand-in for any class the local env can't import (``nre.*``).

    Captures whatever pickle throws at it (constructor args, ``__setstate__``,
    dict ``__setitem__``, list ``append``/``extend``) so nested torch tensors
    are never lost, while the unknown wrapper itself stays inert.
    """

    def __init__(self, *args, **kwargs):
        object.__setattr__(self, "_args", args)
        object.__setattr__(self, "_d", {})
        object.__setattr__(self, "_l", [])
        object.__setattr__(self, "_st", None)

    def __setstate__(self, state):
        object.__setattr__(self, "_st", state)

    def __setitem__(self, k, v):
        self._d[k] = v

    def __getitem__(self, k):
        return self._d.get(k)

    def append(self, v):
        self._l.append(v)

    def extend(self, v):
        self._l.extend(list(v))

    def __setattr__(self, k, v):
        self._d[k] = v


class _TolerantUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except Exception:
            return _CapturingStub


def _tolerant_pickle_module() -> types.ModuleType:
    shim = types.ModuleType("nre_tolerant_pickle")
    shim.Unpickler = _TolerantUnpickler
    shim.load = lambda f, **kw: _TolerantUnpickler(f).load()
    shim.loads = lambda b, **kw: _TolerantUnpickler(io.BytesIO(b)).load()
    return shim


def tolerant_torch_load(data: bytes) -> dict:
    """``torch.load`` raw ckpt bytes, tolerating missing ``nre`` classes."""
    return torch.load(
        io.BytesIO(data),
        map_location="cpu",
        pickle_module=_tolerant_pickle_module(),
        weights_only=False,
    )


def extract_nre_checkpoint(usdz_path: str | Path) -> dict:
    """Read the embedded ``checkpoint.ckpt`` from an NRE usdz → ckpt dict.

    The usdz is a plain zip; the training-flavour export stores a
    PyTorch-Lightning checkpoint as ``checkpoint.ckpt`` (no ``volume.nurec``).
    """
    usdz_path = Path(usdz_path)
    with zipfile.ZipFile(usdz_path, "r") as zf:
        names = zf.namelist()
        if "checkpoint.ckpt" not in names:
            raise FileNotFoundError(
                f"{usdz_path} has no embedded 'checkpoint.ckpt' "
                f"(found: {names}). This loader only handles the NRE "
                f"training-checkpoint usdz flavour."
            )
        data = zf.read("checkpoint.ckpt")
    ckpt = tolerant_torch_load(data)
    if not (isinstance(ckpt, dict) and "state_dict" in ckpt):
        raise ValueError(
            "NRE checkpoint.ckpt missing top-level 'state_dict' "
            f"(got keys: {list(ckpt)[:8] if isinstance(ckpt, dict) else type(ckpt)})"
        )
    return ckpt


# --------------------------------------------------------------------------- #
# 2. Pure tensor translation (Mac-testable, no cuda/hydra).
# --------------------------------------------------------------------------- #
def fourier_cos_basis(frame_id: int, n_frames: int, n_terms: int) -> torch.Tensor:
    """Cosine basis ``[cos(i·π·t/N)]_{i=0}^{k-1}``, shape ``[k]``.

    Mirrors :func:`threedgrut.model.track_albedo_fourier.fourier_cos_basis`
    (StreetGaussian 4D-SH). Kept local so this module stays cuda-free for Mac
    tests; the canonical impl is unit-tested in P1.3b and they must agree.
    ``frame_id`` clamped to ``[0, N-1]``; ``N<=1`` collapses the time axis.
    """
    k = int(n_terms)
    if k < 1:
        raise ValueError(f"n_terms must be >= 1, got {k}")
    N = int(n_frames)
    t = max(0, min(int(frame_id), N - 1)) if N >= 1 else 0
    denom = float(N) if N > 1 else 1.0
    t_norm = (t / denom) if N > 1 else 0.0
    i = torch.arange(k, dtype=torch.float32)
    return torch.cos(i * math.pi * t_norm)


def eval_fourier_albedo(
    albedo: torch.Tensor,
    frame_id: int = 0,
    n_frames: int = 1,
    mode: str = "dc",
) -> torch.Tensor:
    """Collapse NRE per-gaussian Fourier albedo ``(N,K,3)`` → DC ``(N,3)``.

    * ``mode="dc"``   → coefficient 0 only (time-mean base colour; the safe,
      frame-independent choice used in Phase A geometry checks).
    * ``mode="eval"`` → full cosine-Fourier eval at ``frame_id`` over
      ``n_frames`` (per-gaussian time-varying colour; Phase D).

    Accepts already-collapsed ``(N,3)`` (e.g. dynamic_deformables / K=1 road)
    and returns it unchanged.
    """
    if albedo.dim() == 2:
        return albedo  # already (N, 3)
    if albedo.dim() != 3 or albedo.shape[-1] != 3:
        raise ValueError(f"expected albedo (N,K,3) or (N,3); got {tuple(albedo.shape)}")
    K = albedo.shape[1]
    if mode == "dc" or K == 1:
        return albedo[:, 0, :].contiguous()
    if mode == "eval":
        basis = fourier_cos_basis(frame_id, n_frames, K).to(albedo.dtype)  # [K]
        # (N,K,3) · (K) -> (N,3)
        return torch.einsum("nkc,k->nc", albedo, basis).contiguous()
    raise ValueError(f"unknown mode {mode!r}")


def nre_layer_tensors(
    state_dict: dict,
    layer: str,
    *,
    albedo_frame_id: int = 0,
    albedo_n_frames: int = 1,
    albedo_mode: str = "dc",
) -> dict:
    """Pull + translate one NRE ``gaussians_nodes.<layer>`` group.

    Returns a dict with 3dgrut2-native keys: ``positions``/``rotation``/
    ``scale``/``density``/``features_albedo``(N,3)/``features_specular``/
    ``n_active_features``(int). ``gaussian_cuboid_ids`` (if present) rides along
    under ``cuboid_ids`` for the caller (Phase C track wiring). Returns ``{}``
    if the layer is absent.
    """
    prefix = f"model.gaussians_nodes.{layer}."
    raw = {
        k[len(prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }
    if "positions" not in raw:
        return {}

    out: dict = {}
    for src in ("positions", "rotations", "scales", "densities"):
        if src in raw:
            out[_KEY_RENAME.get(src, src)] = raw[src].float().contiguous()
    out["features_albedo"] = eval_fourier_albedo(
        raw["features_albedo"].float(),
        frame_id=albedo_frame_id,
        n_frames=albedo_n_frames,
        mode=albedo_mode,
    )
    out["features_specular"] = raw["features_specular"].float().contiguous()
    n_active = raw.get("n_active_features")
    out["n_active_features"] = (
        int(n_active.item()) if torch.is_tensor(n_active) else int(n_active or 0)
    )
    if "gaussian_cuboid_ids" in raw:
        out["cuboid_ids"] = raw["gaussian_cuboid_ids"].long().contiguous()
    return out


def estimate_scene_extent(positions: torch.Tensor, q: float = 0.95) -> float:
    """Robust scene-extent estimate (percentile radius) so far-field
    background points don't blow up the value. Render-only, so the exact number
    is not critical (it mostly drives training densification)."""
    center = positions.median(dim=0).values
    dists = (positions - center).norm(dim=-1)
    return float(dists.quantile(q).item())


def clip_floater_gaussians(
    t: dict,
    *,
    clip_radius_m: float = 1500.0,
    clip_scale_m: float = 20.0,
) -> tuple[dict, int]:
    """Drop far-field / huge-scale floater gaussians from a translated layer.

    NRE's ``background`` carries a sky/floater tail — positions out to ±1e6 m
    and scales up to ~300 m — that NVIDIA's renderer culls (via opacity /
    ``invisible_steps``) but 3dgrut's UT rasterizer smears across the whole
    frame. We drop gaussians whose distance-from-median exceeds ``clip_radius_m``
    OR whose largest (exp-activated) scale exceeds ``clip_scale_m``. Clean layers
    (road: all within ~150 m, scale < 0.4 m) are untouched. Per-gaussian tensors
    (shape[0]==N) are masked; scalars (n_active_features) ride along.

    Returns ``(masked_dict, n_dropped)``. No-op (``clip_*<=0``) disables clipping.
    """
    pos = t["positions"]
    N = pos.shape[0]
    keep = torch.ones(N, dtype=torch.bool)
    if clip_radius_m and clip_radius_m > 0:
        center = pos.median(dim=0).values
        keep &= (pos - center).norm(dim=1) <= clip_radius_m
    if clip_scale_m and clip_scale_m > 0:
        scale_m = t["scale"].exp().max(dim=1).values  # scale stored log-space
        keep &= scale_m <= clip_scale_m
    if bool(keep.all()):
        return t, 0
    masked = {
        k: (v[keep] if (torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == N) else v)
        for k, v in t.items()
    }
    return masked, int((~keep).sum().item())


# --------------------------------------------------------------------------- #
# 3. Build native ckpt dict + orchestration (needs 3dgrut env + GPU).
# --------------------------------------------------------------------------- #
def build_native_ckpt(
    state_dict: dict,
    *,
    config_name: str = "apps/ncore_3dgut_mcmc_multilayer",
    layers: tuple[str, ...] = ("background", "road"),
    experiment_name: str = "nre_usdz",
    albedo_mode: str = "dc",
    clip_radius_m: float = 1500.0,
    clip_scale_m: float = 20.0,
    global_step: int = 0,
) -> dict:
    """Translate NRE ``state_dict`` → 3dgrut2-native ckpt dict.

    Mirrors what ``Trainer.save_checkpoint`` writes so
    ``engine.load_3dgrt_object`` reads it unchanged:
    ``{"model": {"gaussians_nodes": {...}, "scene_extent": float},
       "config": <DictConfig use_layered_model=True>, ...}``.

    Per-node aux fields (``background`` module state, progressive flags) are
    taken from a throwaway reference ``MixtureOfGaussians(conf)`` so we never
    hand-fake module state. GPU required (MoG lives on cuda).
    """
    from hydra.compose import compose
    from hydra.initialize import initialize

    from threedgrut.model.model import MixtureOfGaussians

    with initialize(version_base=None, config_path="../../configs"):
        conf = compose(config_name=config_name)
    conf.experiment_name = experiment_name
    # E2.7 fix: force layered-model route. ``apps/ncore_3dgut_mcmc_multilayer``
    # yaml sets ``use_layered_model: true`` at line 50, but its ``# @package
    # _global_`` directive doesn't propagate via ``compose(config_name="apps/
    # ncore_3dgut_mcmc_multilayer")`` — the resulting conf still carries the
    # ``base_gs.yaml`` default ``use_layered_model: false``. Without this
    # explicit override, engine.py:load_3dgrt_object detects
    # ``use_layered_ckpt=False`` and routes our nested-gaussians_nodes ckpt
    # through the v1 MixtureOfGaussians.init_from_checkpoint, which expects
    # flat ``checkpoint["positions"]`` and crashes with KeyError: 'positions'.
    # Force the flag so the engine takes the LayeredGaussians branch and reads
    # ``checkpoint["model"]["gaussians_nodes"][<layer>][...]``.
    conf.use_layered_model = True
    # Restrict enabled layers to exactly what we load. Crucially this drops
    # ``sky_envmap`` (multilayer default), whose cubemap backend needs
    # nvdiffrast (absent on inceptio/A800) and would crash _blend_sky on the
    # first render. We don't translate the NRE sky envmap here, so no-op sky is
    # correct: NRE background gaussians already cover the scene; uncovered
    # pixels render black (cleanest for geometry inspection). _blend_sky no-ops
    # when 'sky_envmap' is absent from enabled layers.
    conf.layers.enabled = list(layers)

    # Reference MoG → default background module state + progressive attrs.
    ref = MixtureOfGaussians(conf)
    bg_state = {k: v.detach().cpu() for k, v in ref.background.state_dict().items()}
    progressive = bool(getattr(ref, "progressive_training", False))
    max_n_features = int(ref.max_n_features)

    nodes: dict = {}
    all_positions = []
    for layer in layers:
        t = nre_layer_tensors(state_dict, layer, albedo_mode=albedo_mode)
        if not t:
            print(f"[nre-usdz] layer '{layer}' absent in NRE state_dict — skipping")
            continue
        t, n_drop = clip_floater_gaussians(
            t, clip_radius_m=clip_radius_m, clip_scale_m=clip_scale_m
        )
        if n_drop:
            print(f"[nre-usdz] {layer}: clipped {n_drop} floater gaussians "
                  f"(radius>{clip_radius_m}m or scale>{clip_scale_m}m) "
                  f"→ {t['positions'].shape[0]} kept")
        all_positions.append(t["positions"])
        nodes[layer] = t
    if not nodes:
        raise ValueError("no NRE layers translated; nothing to load")

    scene_extent = estimate_scene_extent(torch.cat(all_positions, dim=0))

    # MoG.init_from_checkpoint assigns ``self.positions = checkpoint["positions"]``
    # directly onto pre-existing nn.Parameter slots, so the ckpt values must be
    # nn.Parameter (not plain tensors) and live on cuda (no device coercion
    # happens before build_acc/BVH) — mirroring what Trainer.save_checkpoint
    # writes (cuda Parameters).
    dev = ref.positions.device

    def _param(x: torch.Tensor) -> torch.nn.Parameter:
        return torch.nn.Parameter(x.to(dev).contiguous(), requires_grad=False)

    gaussians_nodes: dict = {}
    for layer, t in nodes.items():
        cuboid_ids = t.pop("cuboid_ids", None)
        node = {
            "positions": _param(t["positions"]),
            "rotation": _param(t["rotation"]),
            "scale": _param(t["scale"]),
            "density": _param(t["density"]),
            "features_albedo": _param(t["features_albedo"]),
            "features_specular": _param(t["features_specular"]),
            "n_active_features": min(t["n_active_features"], max_n_features),
            "max_n_features": max_n_features,
            "progressive_training": progressive,
            "scene_extent": scene_extent,
            "background": bg_state,
        }
        if progressive:
            # init_from_checkpoint reads these when self.progressive_training.
            node["feature_dim_increase_interval"] = int(
                getattr(ref, "feature_dim_increase_interval", 1)
            )
            node["feature_dim_increase_step"] = int(
                getattr(ref, "feature_dim_increase_step", 1)
            )
        if cuboid_ids is not None:
            # Phase C will turn this into track_ids; for now ride along so the
            # data isn't lost (init_from_checkpoint reads node['track_ids']).
            node["_nre_cuboid_ids"] = cuboid_ids
        gaussians_nodes[layer] = node

    n_total = sum(int(n["positions"].shape[0]) for n in gaussians_nodes.values())
    print(
        f"[nre-usdz] translated {len(gaussians_nodes)} layers, {n_total} gaussians, "
        f"scene_extent≈{scene_extent:.2f}, albedo_mode={albedo_mode}"
    )
    return {
        "model": {"gaussians_nodes": gaussians_nodes, "scene_extent": scene_extent},
        "config": conf,
        "global_step": int(global_step),
        "epoch": 0,
    }


def convert_usdz_to_pt(
    usdz_path: str | Path,
    out_pt: str | Path,
    *,
    config_name: str = "apps/ncore_3dgut_mcmc_multilayer",
    layers: tuple[str, ...] = ("background", "road"),
    albedo_mode: str = "dc",
    clip_radius_m: float = 1500.0,
    clip_scale_m: float = 20.0,
) -> str:
    """usdz → 3dgrut2-native ``.pt``. Returns the written path."""
    usdz_path = Path(usdz_path)
    ckpt = extract_nre_checkpoint(usdz_path)
    state_dict = ckpt["state_dict"]
    global_step = int(ckpt.get("global_step", 0))
    native = build_native_ckpt(
        state_dict,
        config_name=config_name,
        layers=tuple(layers),
        experiment_name=usdz_path.stem,
        albedo_mode=albedo_mode,
        clip_radius_m=clip_radius_m,
        clip_scale_m=clip_scale_m,
        global_step=global_step,
    )
    out_pt = Path(out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(native, out_pt)
    print(f"[nre-usdz] wrote native ckpt → {out_pt}")
    return str(out_pt)


def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Convert NRE usdz → 3dgrut2 .pt")
    ap.add_argument("usdz", type=str)
    ap.add_argument("out_pt", type=str)
    ap.add_argument("--config-name", type=str,
                    default="apps/ncore_3dgut_mcmc_multilayer")
    ap.add_argument("--layers", type=str, nargs="+",
                    default=list(("background", "road")))
    ap.add_argument("--albedo-mode", type=str, default="dc", choices=["dc", "eval"])
    ap.add_argument("--clip-radius-m", type=float, default=1500.0,
                    help="Drop background gaussians beyond this radius (0=off).")
    ap.add_argument("--clip-scale-m", type=float, default=20.0,
                    help="Drop gaussians whose max scale exceeds this (0=off).")
    a = ap.parse_args()
    convert_usdz_to_pt(
        a.usdz, a.out_pt, config_name=a.config_name,
        layers=tuple(a.layers), albedo_mode=a.albedo_mode,
        clip_radius_m=a.clip_radius_m, clip_scale_m=a.clip_scale_m,
    )


if __name__ == "__main__":
    _cli()
