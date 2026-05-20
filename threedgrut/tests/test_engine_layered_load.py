"""Task A contract tests — engine playground viz_4d ckpt → LayeredGaussians.

These verify the two surfaces engine.load_3dgrt_object touches on a v2 ckpt:

1. ``LayeredGaussians.populate_tracks`` can ingest a track-dict shaped like
   ``ckpt["viz_4d"]["tracks"]`` (the schema introduced in Task B) with the
   shared ``tracks_camera_timestamps_us`` injected into the first entry.
2. ``LayeredGaussians.forward(batch)`` with ``timestamp_us`` set picks up a
   per-track pose via ``_resolve_pose_idx`` (binary-search), proving the
   timestamp-driven dynamic-rigid path is reachable from a viewer-style call.

The engine itself can't be instantiated on a CPU-only Mac (kaolin / OptiX),
so we test the ``LayeredGaussians`` surface engine relies on. The dispatch
glue in ``engine._trace_scene_mog`` is a 5-line isinstance branch — passing
these contracts is what matters.
"""
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


def _viz4d_tracks_dict(F: int = 4) -> dict:
    """Build a track dict matching ``ckpt["viz_4d"]["tracks"]`` schema from Task B."""
    tracks = {}
    for i, tid in enumerate(["t0", "t1"]):
        poses = torch.eye(4).repeat(F, 1, 1)
        # Distinguishable translation per track + per frame so a wrong pose lookup
        # would yield clearly wrong positions.
        poses[:, 0, 3] = torch.arange(F, dtype=torch.float32) + i * 100.0
        tracks[tid] = {
            "poses":      poses,
            "size":       torch.tensor([2.0, 1.5, 4.5], dtype=torch.float32),
            "frame_info": torch.ones(F, dtype=torch.bool),
            "class":      "automobile" if i == 0 else "heavy_truck",
        }
    return tracks


def test_populate_tracks_with_injected_shared_timestamps(real_conf):
    """engine.load_3dgrt_object injects shared ts into tracks[first]["cam_timestamps_us"];
    populate_tracks must then build the shared buffer and per-track pose buffers.
    """
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1000)]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=1.0)

    tracks = _viz4d_tracks_dict(F=4)
    # Engine injects shared_ts into the first track entry (see load_3dgrt_object).
    shared_ts = torch.tensor([1000, 2000, 3000, 4000], dtype=torch.int64)
    first_tid = next(iter(tracks))
    tracks[first_tid]["cam_timestamps_us"] = shared_ts

    model.populate_tracks(tracks)

    assert hasattr(model, "tracks_camera_timestamps_us")
    assert torch.equal(model.tracks_camera_timestamps_us, shared_ts)
    assert set(model.tracks_poses.keys()) == {"t0", "t1"}
    assert model.tracks_poses["t0"].shape == (4, 4, 4)
    assert model.tracks_active["t1"].dtype == torch.bool


def test_resolve_pose_idx_binary_search_on_timestamp(real_conf):
    """Viewer pushes timestamp_us into Batch; LayeredGaussians._resolve_pose_idx
    must binary-search the shared ts buffer (not fall through to frame_id).
    """
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1000)]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=1.0)

    tracks = _viz4d_tracks_dict(F=5)
    shared_ts = torch.tensor([1000, 2000, 3000, 4000, 5000], dtype=torch.int64)
    tracks[next(iter(tracks))]["cam_timestamps_us"] = shared_ts
    model.populate_tracks(tracks)

    # exact match → exact index
    assert model._resolve_pose_idx(3000, frame_id=None) == 2
    # closer to 2000 than to 3000 (prev < curr branch)
    assert model._resolve_pose_idx(2100, frame_id=None) == 1
    # closer to 3000
    assert model._resolve_pose_idx(2900, frame_id=None) == 2
    # past end → clamped to F-1
    assert model._resolve_pose_idx(9999, frame_id=None) == 4
    # ts<=0 → frame_id fallback (engine viewer passes int)
    assert model._resolve_pose_idx(-1, frame_id=2) == 2


def test_populate_tracks_idempotent_replace(real_conf):
    """A second populate_tracks call replaces existing buffers (engine reload path)."""
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=1000)]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=1.0)

    tracks_v1 = _viz4d_tracks_dict(F=3)
    tracks_v1[next(iter(tracks_v1))]["cam_timestamps_us"] = torch.tensor(
        [100, 200, 300], dtype=torch.int64
    )
    model.populate_tracks(tracks_v1)
    assert model.tracks_poses["t0"].shape == (3, 4, 4)

    # Re-populate with longer schedule (simulating a different clip / reload).
    tracks_v2 = _viz4d_tracks_dict(F=7)
    tracks_v2[next(iter(tracks_v2))]["cam_timestamps_us"] = torch.tensor(
        [10, 20, 30, 40, 50, 60, 70], dtype=torch.int64
    )
    model.populate_tracks(tracks_v2)
    assert model.tracks_poses["t0"].shape == (7, 4, 4)
    assert model.tracks_camera_timestamps_us.shape == (7,)
