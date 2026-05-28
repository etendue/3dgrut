# SPDX-License-Identifier: Apache-2.0
"""V3-E4.1 regression tests — Renderer.from_checkpoint() must restore
dynamic-rigid tracks_poses + scene_extent from ckpt, otherwise reloading a
v2 LayeredGaussians ckpt yields ~3 dB lower psnr than the train-end eval
because 300K dynamic Gaussians render at canonical pose (not animated).

Root cause (commit fixing this): render.py L143-175 originally did
``LayeredGaussians(conf, specs=specs, scene_extent=None)`` + ``init_from_checkpoint``
only — never called ``populate_tracks`` on ``ckpt["viz_4d"]["tracks"]``. The
playground engine (``engine.py:1340-1360``) already had the correct pattern;
the fix copied it to render.py.

A800 byte-identical proof: sym5cam 30k ckpt reload now matches train-end
metrics.json to all printed digits (cc_psnr_masked 26.0436 / psnr_masked
15.2878 vs reload ditto — see v3_plan.md §5 Done Log V3-E4.1).

This test suite stays CPU-only: full ``from_checkpoint`` is CUDA-bound
(``model.build_acc`` + dataloader), so we replicate the load *sequence* on a
synthetic ckpt blob (same pattern as ``test_render_per_camera.py``) plus add
two source-level smoke checks that catch accidental code removal from
render.py.
"""
from __future__ import annotations

import inspect
import os

import pytest
import torch
from hydra import compose, initialize_config_dir

from threedgrut.layers.layer_spec import LayerSpec


_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs")
)


@pytest.fixture(scope="module")
def real_conf():
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="apps/ncore_3dgut_mcmc")


def _build_synthetic_ckpt(num_tracks: int = 3, num_frames: int = 4,
                           scene_extent: float = 1.103) -> dict:
    """Mimic a v2 LayeredGaussians ckpt with viz_4d block.

    Shape: ``{model: {gaussians_nodes, scene_extent, sky_envmap_state},
              viz_4d: {tracks, tracks_camera_timestamps_us}, ...}``
    Only the bits exercised by render.py:from_checkpoint's load sequence are
    populated; gaussians_nodes is deliberately tiny so MoG.init_from_checkpoint
    runs in milliseconds on CPU.
    """
    # Minimal Gaussian params (1 particle in background, others empty).
    one = lambda *s: torch.zeros(*s, dtype=torch.float32)
    bg_params = {
        "positions": one(1, 3),
        "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "scale": one(1, 3),
        "density": one(1, 1),
        "features_albedo": one(1, 3),
        "features_specular": one(1, 45),
        "scene_extent": scene_extent,
    }
    tracks = {}
    for i in range(num_tracks):
        tid = str(10 + i)
        poses = torch.eye(4).repeat(num_frames, 1, 1)
        poses[:, 0, 3] = torch.arange(num_frames, dtype=torch.float32) + i * 10.0
        tracks[tid] = {
            "poses":      poses,
            "size":       torch.tensor([2.0, 1.5, 4.5], dtype=torch.float32),
            "frame_info": torch.ones(num_frames, dtype=torch.bool),
            "class":      "automobile",
        }
    shared_ts = torch.tensor(
        [(i + 1) * 1000 for i in range(num_frames)], dtype=torch.int64,
    )
    return {
        "model": {
            "gaussians_nodes": {"background": bg_params},
            "scene_extent": scene_extent,
            "sky_envmap_state": {},  # left empty — sky_envmap layer absent in single-bg spec
        },
        "viz_4d": {
            "tracks": tracks,
            "tracks_camera_timestamps_us": shared_ts,
            "schema_version": 2,
        },
        "global_step": 100,
        "config": None,  # not used directly by populate_tracks
    }


def _replay_render_py_load(ckpt: dict, conf, specs):
    """Lightweight mirror of render.py L143-175 (V3-E4.1 fix): ctor with
    scene_extent from ckpt + the viz_4d → populate_tracks call. Skips
    ``model.init_from_checkpoint`` (heavy + needs real nn.Parameter Gaussian
    params; that path is independently covered by ``test_layered_gaussians``).

    Any change to render.py L143-175 must be reflected here, and tests below
    must keep passing.
    """
    from threedgrut.layers.layered_model import LayeredGaussians

    scene_extent = float(ckpt.get("model", {}).get("scene_extent", 1.0))
    model = LayeredGaussians(conf, specs=specs, scene_extent=scene_extent)
    # NOTE: skipping model.init_from_checkpoint here (would need real
    # Parameter tensors); the populate_tracks portion is what V3-E4.1 fixed.
    viz_4d = ckpt.get("viz_4d")
    if viz_4d is not None and isinstance(viz_4d, dict):
        tracks_dict = viz_4d.get("tracks")
        shared_ts = viz_4d.get("tracks_camera_timestamps_us")
        if tracks_dict and shared_ts is not None:
            first_tid = next(iter(tracks_dict))
            tracks_dict[first_tid]["cam_timestamps_us"] = shared_ts
            model.populate_tracks(tracks_dict)
    return model


def test_reload_populates_tracks_poses_from_viz_4d(real_conf):
    """Primary regression: after the load sequence runs, tracks_poses must be
    non-empty and contain the per-track pose tensors.

    Before the V3-E4.1 fix, model.tracks_poses stayed {} → fused_view
    rendered 300K dyn Gaussians at canonical pose → ~2 dB raw psnr drop.
    """
    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=10)]
    ckpt = _build_synthetic_ckpt(num_tracks=3, num_frames=4)

    model = _replay_render_py_load(ckpt, real_conf, specs)

    assert model.tracks_poses, (
        "V3-E4.1 regression: tracks_poses empty after reload. "
        "render.py:from_checkpoint must call model.populate_tracks() on "
        "ckpt['viz_4d']['tracks'] (see playground engine.py:1346-1360)."
    )
    assert set(model.tracks_poses.keys()) == {"10", "11", "12"}
    assert model.tracks_poses["10"].shape == (4, 4, 4)
    assert hasattr(model, "tracks_camera_timestamps_us")
    assert torch.equal(
        model.tracks_camera_timestamps_us,
        torch.tensor([1000, 2000, 3000, 4000], dtype=torch.int64),
    )


def test_reload_passes_scene_extent_to_ctor(real_conf):
    """scene_extent must propagate from ckpt['model']['scene_extent'] to the
    LayeredGaussians ctor. (Before V3-E4.1 fix, render.py:164 passed
    scene_extent=None instead of reading the saved value.)
    """
    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=10)]
    ckpt = _build_synthetic_ckpt(scene_extent=2.718)

    model = _replay_render_py_load(ckpt, real_conf, specs)

    # LayeredGaussians stores ctor scene_extent on itself (line 269 of
    # layered_model.py uses object.__setattr__). Per-layer MoG also receives
    # it (line 283).
    assert abs(model.scene_extent - 2.718) < 1e-6, (
        f"model.scene_extent={model.scene_extent} != 2.718; render.py:164 "
        f"may have reverted to scene_extent=None"
    )


def test_reload_no_viz_4d_does_not_crash(real_conf):
    """v1 ckpts and ckpts trained before T8.2 have no viz_4d block — load
    must succeed silently with empty tracks_poses (static dyn layer)."""
    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=10)]
    ckpt = _build_synthetic_ckpt(num_tracks=2)
    del ckpt["viz_4d"]  # simulate pre-T8.2 ckpt

    model = _replay_render_py_load(ckpt, real_conf, specs)

    assert model.tracks_poses == {}, "no viz_4d → tracks_poses must stay empty"


def test_reload_viz_4d_missing_shared_ts_skipped(real_conf):
    """If viz_4d.tracks present but tracks_camera_timestamps_us missing,
    populate_tracks is correctly skipped (avoid crash on partial schemas)."""
    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=10)]
    ckpt = _build_synthetic_ckpt(num_tracks=2)
    del ckpt["viz_4d"]["tracks_camera_timestamps_us"]

    model = _replay_render_py_load(ckpt, real_conf, specs)

    assert model.tracks_poses == {}


_RENDER_PY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "render.py")
)


def _render_py_text() -> str:
    """Read render.py source directly — avoids ``import threedgrut.render``
    which pulls torchvision/CUDA-bound deps on Mac venv."""
    with open(_RENDER_PY, "r") as f:
        return f.read()


def test_render_py_source_calls_populate_tracks():
    """Source-level guard: catches accidental removal of populate_tracks from
    render.py:from_checkpoint, which would silently regress V3-E4.1 even if
    the contract tests above still pass on synthetic ckpts."""
    src = _render_py_text()
    assert "populate_tracks" in src, (
        "V3-E4.1 regression: render.py no longer calls "
        "model.populate_tracks(). Dynamic-rigid Gaussians will render at "
        "canonical pose, causing ~2 dB raw psnr drop. See V3-E4.1 fix and "
        "playground engine.py:1340-1360 for the reference pattern."
    )
    # populate_tracks must be reachable from from_checkpoint and gated on
    # viz_4d availability (not unconditional, which would crash pre-T8.2
    # ckpts).
    assert 'viz_4d = checkpoint.get("viz_4d")' in src
    assert "tracks_camera_timestamps_us" in src


def test_render_py_source_reads_scene_extent_from_ckpt():
    """Source-level guard: catches accidental revert to scene_extent=None."""
    src = _render_py_text()
    assert "scene_extent" in src
    flat = src.replace(" ", "").replace("\n", "")
    assert "scene_extent=None" not in flat, (
        "V3-E4.1 regression: Renderer.from_checkpoint() reverted to "
        "scene_extent=None instead of reading it from ckpt['model']."
    )


# T9.3 / V3-P1.c: source-level guards for eval-time exposure_model apply ------

def test_render_py_source_renderer_init_accepts_exposure_model():
    """Renderer.__init__ must accept and store ``exposure_model``."""
    src = _render_py_text()
    assert "exposure_model=None" in src.replace(" ", ""), (
        "T9.3 regression: Renderer.__init__ no longer has exposure_model "
        "kwarg. eval would not be able to apply BilateralGrid → reverts to "
        "v2-style raw-vs-train mismatch."
    )
    assert "self.exposure_model = exposure_model" in src, (
        "T9.3 regression: Renderer.__init__ stops storing exposure_model "
        "on self. render_all can't reach it."
    )


def test_render_py_source_render_all_applies_exposure_model():
    """render_all must apply self.exposure_model after model forward +
    post_processing, before metrics — same site the trainer applies it
    pre-loss (trainer.py:1641-1643)."""
    src = _render_py_text()
    assert "self.exposure_model is not None" in src, (
        "T9.3 regression: render_all no longer gates on exposure_model. "
        "eval would skip the BilateralGrid apply → metrics measure "
        "(raw_model_output vs GT), not (bilateral_grid(output) vs GT)."
    )
    # Ensure the apply mutates outputs["pred_rgb"] so downstream
    # color_correct_affine + masked metrics see the corrected tensor.
    assert 'outputs["pred_rgb"] = self.exposure_model' in src, (
        "T9.3 regression: render_all applies exposure_model but does not "
        "write back into outputs['pred_rgb']. Metrics downstream still "
        "see the pre-exposure output."
    )


def test_render_py_source_from_checkpoint_rebuilds_bilateral_grid():
    """from_checkpoint must reconstruct BilateralGrid from
    ckpt['exposure_state']['module']['grids'] shape + load state."""
    src = _render_py_text()
    # Construct BilateralGrid (import + ctor).
    assert "from threedgrut.correction import BilateralGrid" in src, (
        "T9.3 regression: from_checkpoint no longer imports BilateralGrid."
    )
    assert "BilateralGrid(" in src
    # Read grids tensor's shape — the only thing we have at ckpt-load time.
    assert "module_state" in src and '"grids"' in src
    # Legacy v2 ckpt detection so old ExposureModel ckpts don't crash.
    assert 'exposure_a' in src and 'exposure_b' in src, (
        "T9.3 regression: legacy v2 ckpt path removed; loading a v2 "
        "exposure_state.module would raise KeyError."
    )


def _trainer_py_text() -> str:
    """Read trainer.py source directly (avoids torchvision/CUDA imports)."""
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "trainer.py")
    )
    with open(path, "r") as f:
        return f.read()


def test_trainer_py_passes_exposure_model_to_from_preloaded_model():
    """trainer.py train-end eval must pass exposure_model so the
    from_preloaded_model path applies BilateralGrid (the standalone
    from_checkpoint path is covered by the render.py guards above)."""
    src = _trainer_py_text()
    # Find the from_preloaded_model call site (only 1 in trainer.py).
    assert "Renderer.from_preloaded_model(" in src
    # The block must include exposure_model=self.exposure_model. We accept
    # whitespace variation but not the omission case.
    flat = src.replace(" ", "").replace("\n", "")
    assert "exposure_model=self.exposure_model" in flat, (
        "T9.3 regression: train-end eval no longer passes self.exposure_model "
        "to Renderer.from_preloaded_model. Train-end metrics.json would "
        "measure raw model_output (not bilateral_grid(output)), so V3-P1 "
        "validation comparisons would be invalid."
    )
