from pathlib import Path
from types import SimpleNamespace

import numpy as np

from threedgrut.layers.resume_tracks import populate_dynamic_tracks_for_checkpoint_resume


def test_layered_resume_repopulates_dynamic_track_slots(monkeypatch):
    class FakeNCoreDataset:
        sequence_id = "sequence"
        sequence_loaders = {"sequence": object()}

    class FakeLayeredGaussians:
        def __init__(self):
            self.layers = {"dynamic_rigids": object()}
            self.populated = None

        def populate_tracks(self, tracks):
            self.populated = tracks

    expected_tracks = {"4629": {"poses": object()}}
    import threedgrut.layers.layered_model as layered_model_module
    import threedgrut.datasets.tracks_loader as tracks_loader_module

    monkeypatch.setattr(layered_model_module, "LayeredGaussians", FakeLayeredGaussians)
    monkeypatch.setattr(
        tracks_loader_module,
        "build_cuboid_frame_timeline_us",
        lambda dataset, mode: np.zeros((7,), dtype=np.int64),
    )
    monkeypatch.setattr(
        tracks_loader_module,
        "load_tracks_from_ncore_cuboids",
        lambda loader, timeline, pose_time_mode: expected_tracks,
    )
    monkeypatch.setattr(tracks_loader_module, "CUBOID_TS_MODES", {"ref_nearest": "end"})

    model = FakeLayeredGaussians()
    conf = SimpleNamespace(dataset=SimpleNamespace(cuboid_ts_mode="ref_nearest"))
    populate_dynamic_tracks_for_checkpoint_resume(model, FakeNCoreDataset(), conf)

    assert model.populated is expected_tracks


def test_resume_populates_tracks_before_loading_checkpoint_state():
    # Source-only assertion: importing trainer pulls optional metrics deps.
    source = (Path(__file__).resolve().parents[1] / "trainer.py").read_text()
    resume_block = source[
        source.index("if conf.resume:") : source.index("elif conf.import_ply.enabled:")
    ]
    assert resume_block.index("populate_dynamic_tracks_for_checkpoint_resume") < resume_block.index(
        "model.init_from_checkpoint(checkpoint)"
    )
