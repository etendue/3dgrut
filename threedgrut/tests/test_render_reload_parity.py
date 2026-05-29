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


# T9.4 / V3-P1.d: source-level guards for BilateralGrid health monitoring ----

def test_trainer_py_t9_4_logs_exposure_grids_std():
    """Per log_frequency train step must emit exposure/grids_std + drift +
    lr + frozen scalars when exposure_model has a 'grids' attr (BilateralGrid)."""
    src = _trainer_py_text()
    assert '"exposure/grids_std"' in src, (
        "T9.4 regression: exposure/grids_std scalar removed from per-train-step "
        "log block. health monitoring of BilateralGrid retired."
    )
    assert '"exposure/grids_drift_from_identity"' in src, (
        "T9.4 regression: drift-from-identity gauge missing. v3_plan §2.1 "
        "退化 indicator gone — won't catch BilateralGrid absorbing tone."
    )
    assert '"exposure/lr"' in src and '"exposure/frozen"' in src, (
        "T9.4 regression: cosine-LR / freeze-gate trace missing."
    )


def test_trainer_py_t9_4_logs_raw_minus_cc_gap():
    """log_validation_pass must compute and log exposure/raw_minus_cc_db_val
    when exposure_model is in play. Also wires a >2 dB warn."""
    src = _trainer_py_text()
    assert '"exposure/raw_minus_cc_db_val"' in src, (
        "T9.4 regression: raw-vs-cc gap scalar missing from val logging. "
        "Can't track V3-P1 ≤ 2 dB acceptance through training."
    )
    assert '"exposure/raw_minus_cc_db_masked_val"' in src, (
        "T9.4 regression: masked variant (ego mask applied) missing. "
        "Acceptance is on masked metrics per v3_plan §2.1."
    )
    # Warn must include a freeze-window buffer so it doesn't fire while
    # BilateralGrid is still frozen at identity.
    assert "warn_after" in src and "exposure_freeze_until_iter" in src
    assert "[T9.4 alert]" in src


def test_trainer_py_t9_4_computes_cc_psnr_in_val_metrics():
    """get_metrics(split=='validation') must compute cc_psnr (+ masked) when
    exposure_model is on, otherwise log_validation_pass has no data to
    write the gap scalars from."""
    src = _trainer_py_text()
    assert 'metrics["cc_psnr"]' in src, (
        "T9.4 regression: cc_psnr not populated in val metrics; the val "
        "gap scalar will silently no-op forever."
    )
    assert "color_correct_affine" in src, (
        "T9.4 regression: cc_psnr computation removed — gap monitoring "
        "would emit zero or stale data."
    )


# ─── V3-E4.1 follow-up — populate_tracks BEFORE init_from_checkpoint ────────
#
# Original V3-E4.1 fix added populate_tracks but placed it AFTER
# init_from_checkpoint. For buffer-mode ckpts (pre-Stage A) that's harmless,
# because the tracks_dict GT poses equal the saved buffer values. For
# learnable_pose ckpts (Stage A/B) it silently drops the learned
# _track_quat_<tid> / _track_trans_<tid> Parameter values: load_state_dict in
# LayeredGaussians.init_from_checkpoint reports them as unexpected keys
# (slots not yet created), then populate_tracks creates the slots and fills
# them with GT-init values from tracks_dict.
#
# These guards pin the corrected order in both call sites.


_ENGINE_PY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..",
                 "threedgrut_playground", "engine.py")
)


def _engine_py_text() -> str:
    with open(_ENGINE_PY, "r") as f:
        return f.read()


def _find_block(src: str, anchor: str, window: int = 4000) -> str:
    """Slice the source text starting at ``anchor`` for ``window`` chars."""
    idx = src.find(anchor)
    if idx < 0:
        raise AssertionError(
            f"Source anchor not found: {anchor!r} — file may have been "
            f"refactored away from the V3 ckpt-load idiom."
        )
    return src[idx:idx + window]


def test_render_py_populate_tracks_before_init_from_checkpoint():
    """render.py:from_checkpoint must call populate_tracks BEFORE
    init_from_checkpoint for the layered branch."""
    src = _render_py_text()
    block = _find_block(src, "if conf.get(\"use_layered_model\"")
    pop_at = block.find("model.populate_tracks(tracks_dict)")
    init_at = block.find("model.init_from_checkpoint(checkpoint")
    assert pop_at > 0, "populate_tracks call missing from layered branch"
    assert init_at > 0, "init_from_checkpoint call missing from layered branch"
    assert pop_at < init_at, (
        f"V3-E4.1 follow-up regression: in render.py, populate_tracks "
        f"appears AFTER init_from_checkpoint (pop_offset={pop_at}, "
        f"init_offset={init_at} relative to the layered branch). For "
        f"learnable_pose ckpts this drops the learned _track_quat_<tid> / "
        f"_track_trans_<tid> Parameter values — load_state_dict only "
        f"writes into pre-existing slots and those slots are created by "
        f"populate_tracks. Swap the two lines back."
    )


def test_engine_py_populate_tracks_before_init_from_checkpoint():
    """engine.py:load_3dgrt_object must call populate_tracks BEFORE
    init_from_checkpoint for the layered branch (same logic as render.py
    above; both call sites had the inversion since the original V3-E4.1 fix
    and were corrected simultaneously)."""
    src = _engine_py_text()
    block = _find_block(src, "if use_layered:")
    pop_at = block.find("model.populate_tracks(tracks_dict)")
    init_at = block.find("model.init_from_checkpoint(checkpoint")
    assert pop_at > 0, "populate_tracks call missing from engine.py layered branch"
    assert init_at > 0, "init_from_checkpoint call missing from engine.py layered branch"
    assert pop_at < init_at, (
        f"V3-E4.1 follow-up regression: in engine.py:load_3dgrt_object, "
        f"populate_tracks appears AFTER init_from_checkpoint "
        f"(pop_offset={pop_at}, init_offset={init_at} relative to the "
        f"use_layered branch). See render.py companion test for the same "
        f"reasoning — swap the two lines back."
    )


def _make_learnable_conf():
    """Compose a tiny multilayer conf with learnable_pose.enabled=true so
    LayeredGaussians.populate_tracks creates Parameters, not buffers."""
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(
            config_name="apps/ncore_3dgut_mcmc",
            overrides=["trainer.learnable_pose.enabled=true"],
        )


def test_populate_tracks_before_load_state_dict_preserves_learned_pose():
    """Behavioural test mirroring the engine.py/render.py fix end-to-end on
    the layered_track_state portion.

    Setup: build a fake layered_track_state with _track_quat_<tid> values
    that DIFFER from the GT poses in tracks_dict (simulate post-Stage-B
    drift). Run the ``correct`` order on a learnable LayeredGaussians:
      1. populate_tracks(tracks_dict)  → creates _track_quat_/_track_trans_/
         _track_pose_gt_/_track_active_ slots, init values = GT
      2. load_state_dict(track_state, strict=False)  → overwrites the
         _track_quat_/_track_trans_ Parameter values with the "learned"
         post-training values from track_state
    Then verify the Parameters hold the LEARNED values, NOT the GT.
    """
    from threedgrut.layers.layered_model import LayeredGaussians

    conf = _make_learnable_conf()
    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=10)]
    ckpt = _build_synthetic_ckpt(num_tracks=2, num_frames=4)
    tracks_dict = ckpt["viz_4d"]["tracks"]
    shared_ts = ckpt["viz_4d"]["tracks_camera_timestamps_us"]

    # Build the model + populate in learnable mode.
    model = LayeredGaussians(conf, specs=specs, scene_extent=1.0)
    first_tid = next(iter(tracks_dict))
    tracks_dict[first_tid]["cam_timestamps_us"] = shared_ts
    model.populate_tracks(tracks_dict)

    # Sanity: in learnable mode, _track_quat_/_track_trans_ slots exist and
    # the GT pose buffer is registered.
    tids = sorted(tracks_dict.keys())
    for tid in tids:
        assert f"_track_quat_{tid}" in model._parameters, \
            f"populate_tracks did not register _track_quat_{tid} Parameter"
        assert f"_track_trans_{tid}" in model._parameters, \
            f"populate_tracks did not register _track_trans_{tid} Parameter"
        assert f"_track_pose_gt_{tid}" in model._buffers, \
            f"populate_tracks did not register _track_pose_gt_{tid} buffer"

    # Build a "post-training" layered_track_state where the learned quat is
    # NOT identity (different from GT init). This is what a Stage A/B 30k
    # ckpt actually contains.
    fake_learned_state = {}
    for tid in tids:
        # Different quat: rotate ~10° around z axis, non-identity for every frame.
        q = torch.tensor([0.996, 0.0, 0.0, 0.087]).repeat(4, 1)  # ~10° z-rot
        q = q / q.norm(dim=-1, keepdim=True)
        fake_learned_state[f"_track_quat_{tid}"] = q.clone()
        # Different trans: shift by [+1m, +2m, +3m] per frame
        t = torch.tensor([[1.0, 2.0, 3.0]]).repeat(4, 1)
        fake_learned_state[f"_track_trans_{tid}"] = t.clone()
        # Keep GT pose buffer + active mask as-is (mirror Stage B ckpt).
        fake_learned_state[f"_track_pose_gt_{tid}"] = \
            model._buffers[f"_track_pose_gt_{tid}"].clone()
        fake_learned_state[f"_track_active_{tid}"] = \
            model._buffers[f"_track_active_{tid}"].clone()

    # Run the load_state_dict portion of init_from_checkpoint (mirror
    # layered_model.py L648-651).
    missing, unexpected = model.load_state_dict(fake_learned_state, strict=False)
    assert not unexpected, (
        f"slots missing for learned keys: {unexpected[:5]} — populate_tracks "
        f"may have failed to create _track_quat_/_track_trans_ in learnable "
        f"mode (regression in _populate_tracks_impl)."
    )

    # Now verify the model picked up the learned values, NOT the GT.
    for tid in tids:
        q_loaded = model._parameters[f"_track_quat_{tid}"].detach()
        t_loaded = model._parameters[f"_track_trans_{tid}"].detach()
        q_expected = fake_learned_state[f"_track_quat_{tid}"]
        t_expected = fake_learned_state[f"_track_trans_{tid}"]
        assert torch.allclose(q_loaded, q_expected, atol=1e-6), (
            f"_track_quat_{tid} did NOT get loaded from state_dict. "
            f"Got {q_loaded[0]}, expected {q_expected[0]}. This means the "
            f"V3-E4.1 follow-up fix regressed: populate_tracks must run "
            f"BEFORE init_from_checkpoint's load_state_dict call."
        )
        assert torch.allclose(t_loaded, t_expected, atol=1e-6), (
            f"_track_trans_{tid} did NOT get loaded from state_dict."
        )

        # GT pose buffer must still reflect the original tracks_dict GT
        # (load_state_dict for _track_pose_gt_ above writes the cloned
        # populate-time value back; this confirms the buffer slot exists).
        gt = model._buffers[f"_track_pose_gt_{tid}"]
        assert gt.shape == (4, 4, 4), \
            f"_track_pose_gt_{tid} shape changed: {gt.shape}"


def test_load_state_dict_before_populate_drops_learned_pose_unexpected():
    """The WRONG order (which both render.py + engine.py had pre-fix) must
    yield unexpected_keys for _track_quat_/_track_trans_. This is the
    canary for the bug behaviour: if a future refactor lands the inverted
    order, this test fails."""
    from threedgrut.layers.layered_model import LayeredGaussians

    conf = _make_learnable_conf()
    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=10)]

    # Build the model BUT do NOT call populate_tracks yet — slots absent.
    model = LayeredGaussians(conf, specs=specs, scene_extent=1.0)

    fake_learned_state = {
        "_track_quat_10": torch.tensor([0.996, 0.0, 0.0, 0.087]).repeat(4, 1),
        "_track_trans_10": torch.zeros(4, 3),
    }
    missing, unexpected = model.load_state_dict(fake_learned_state, strict=False)
    # ALL of fake_learned_state should be reported as unexpected because
    # populate_tracks hasn't created the slots.
    unexpected_set = set(unexpected)
    assert "_track_quat_10" in unexpected_set, (
        "Test premise broken: _track_quat_10 should be unexpected when "
        "populate_tracks has not yet run. If this fails, LayeredGaussians "
        "now eagerly creates pose slots in __init__ — that's a behaviour "
        "change worth investigating."
    )
    assert "_track_trans_10" in unexpected_set


# T9.2 fix: scheduler step paired with optim step (avoid PyTorch warn) -------

def test_trainer_py_t9_2_scheduler_step_paired_with_optim_step():
    """T9.2 fix: exposure_scheduler.step() must be INSIDE the
    `if global_step >= freeze_until` block, paired with optimizer.step().
    Calling sched.step() before optim.step() during the freeze window
    triggers PyTorch's UserWarning + silently skips the first lr value.
    """
    src = _trainer_py_text()
    # Find the exposure-opt block (between cuda.nvtx range marker and end).
    nvtx_marker = 'cuda.nvtx.range(f"train_{global_step}_exposure_opt")'
    assert nvtx_marker in src, "exposure_opt nvtx block missing"
    block_start = src.find(nvtx_marker)
    block_end = src.find("\n        # ", block_start + len(nvtx_marker))
    block = src[block_start:block_end]
    # Inside this block, find the `if global_step >= self.exposure_freeze_until_iter`
    # gate. exposure_scheduler.step() must appear INSIDE this gate (after
    # optimizer.step()), not after the gate (which would tick the scheduler
    # during freeze).
    gate = "if global_step >= self.exposure_freeze_until_iter:"
    assert gate in block
    gate_idx = block.find(gate)
    scheduler_idx = block.find("exposure_scheduler.step()")
    assert scheduler_idx > gate_idx, (
        "T9.2 fix regression: exposure_scheduler.step() must be after "
        "(inside) the freeze_until gate. Otherwise scheduler ticks during "
        "the freeze window with optim.step() never called → PyTorch warns "
        "and the first lr value is silently dropped."
    )


def test_trainer_py_t9_2_cosine_t_max_minus_freeze():
    """T9.2 fix: CosineAnnealingLR T_max must be (n_iterations - freeze_until_iter),
    not n_iterations. Pairing sched.step() to optim.step() means the
    BilateralGrid's actual training window is (n_iter - freeze) steps —
    cosine must span exactly that window or it plateaus mid-curve."""
    src = _trainer_py_text()
    flat = src.replace(" ", "").replace("\n", "")
    # The exact computation we expect; tolerate the simple form.
    assert "n_iters-self.exposure_freeze_until_iter" in flat, (
        "T9.2 fix regression: cosine T_max should be (n_iter - freeze_until)."
        " If T_max stays = n_iter, cosine plateaus at lr0*0.011 at end of "
        "training instead of decaying to ~0."
    )


# T9.4 fix: val loop must apply exposure_model (train/val/test parity) -------

def test_trainer_py_t9_4_val_loop_applies_exposure_model():
    """T9.4 fix: run_validation_pass must apply exposure_model after
    model.forward + post_processing, mirroring trainer.step_iter (L1712)
    and render.py:render_all (L451). Without it, val metrics measure raw
    model output WITHOUT BilateralGrid correction → val/psnr scalar shows
    退化-mode-like trajectory (T9.5 30k: val/psnr 20.44→13.72 across val
    passes) that contradicts the actual test_last raw psnr_masked=27.25
    for the same ckpt — misleading the monitoring story by ~13.5 dB.
    """
    src = _trainer_py_text()
    # Find the run_validation_pass impl.
    rvp_idx = src.find("def run_validation_pass(self")
    assert rvp_idx >= 0, "run_validation_pass missing"
    # Body extends to next def or end of file.
    body_end = src.find("\n    def ", rvp_idx + 5)
    body = src[rvp_idx:body_end if body_end > 0 else len(src)]

    assert "self.exposure_model is not None" in body, (
        "T9.4 fix regression: run_validation_pass no longer gates on "
        "exposure_model. Val metrics will measure raw pre-correction "
        "output → cc/raw gap scalar will show fake退化 mode."
    )
    assert 'outputs["pred_rgb"] = self.exposure_model(' in body, (
        "T9.4 fix regression: val loop doesn't write back through "
        "exposure_model. get_metrics will compute psnr on uncorrected "
        "rgb_pred."
    )
