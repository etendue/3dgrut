# SPDX-License-Identifier: Apache-2.0
"""E2.8 Task 2/3 — vehicle track enumeration + bank assignment + replace orchestration."""
from threedgrut.layers.warmstart_metadata import AssetSpec
from threedgrut.layers.e28_replace import assign_assets_to_tracks, VEHICLE_CLASSES


def _spec(h, cls, dims):
    return AssetSpec(h, f"{cls}/{h}/gaussians.ply", cls, tuple(dims))


BUNDLE = {
    "sedan1": _spec("sedan1", "consumer_vehicles", (4.5, 1.8, 1.5)),
    "bus1":   _spec("bus1",   "bus",               (12.0, 2.5, 3.2)),
}


def test_only_vehicle_tracks_assigned():
    recon = {  # track_name -> (label_class, dims)
        "car_a":  ("automobile", (4.6, 1.8, 1.5)),
        "ped_b":  ("VRU_pedestrians", (0.6, 0.6, 1.7)),  # 非 vehicle → 不分配
        "bus_c":  ("bus", (11.8, 2.5, 3.1)),
    }
    assign, report = assign_assets_to_tracks(recon, BUNDLE, on_miss="global")
    assert set(assign.keys()) == {"car_a", "bus_c"}      # ped 不在
    assert assign["car_a"] == "sedan1"
    assert assign["bus_c"] == "bus1"


def test_report_records_fallback_and_skips():
    recon = {
        "truck_x": ("truck", (11.5, 2.5, 3.0)),  # bank 无 truck → 跨 class
    }
    assign, report = assign_assets_to_tracks(recon, BUNDLE, on_miss="global")
    row = next(r for r in report if r.track == "truck_x")
    assert row.chosen_asset == "bus1"
    assert row.fallback_level == 1
    assert row.skipped is False


def test_on_miss_skip_keeps_recon():
    recon = {"truck_x": ("truck", (11.5, 2.5, 3.0))}
    empty = {}
    assign, report = assign_assets_to_tracks(recon, empty, on_miss="skip")
    assert "truck_x" not in assign                 # 不替换
    row = next(r for r in report if r.track == "truck_x")
    assert row.skipped is True
    assert row.chosen_asset is None


def test_vehicle_classes_cover_ncore_autolabels():
    for c in ("automobile", "bus", "truck", "consumer_vehicles", "car", "vehicle"):
        assert c in VEHICLE_CLASSES
