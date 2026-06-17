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


# ----------------------------------------------------------------------------
# Task 3: replace_all_vehicle_tracks orchestration (guards bg/road/non-vehicle)
# ----------------------------------------------------------------------------
import torch
from threedgrut.layers.e28_replace import replace_all_vehicle_tracks


def _toy_dyn_node():
    # 2 个 vehicle track (ids 0,1) 各 3 粒子 + 1 ped track (id 2) 2 粒子
    tids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.int64)
    n = tids.shape[0]

    def p(c):
        return torch.nn.Parameter(torch.arange(n * c, dtype=torch.float32).reshape(n, c))

    return {
        "positions": p(3), "rotation": p(4), "scale": p(3), "density": p(1),
        "features_albedo": p(3), "features_specular": p(2),
        "track_ids": tids, "n_active_features": 0,
    }


def test_non_vehicle_track_particles_unchanged(monkeypatch):
    # ped track (id 2) 的粒子在替换后逐字节不变；bg 不动
    node = _toy_dyn_node()
    ped_before = node["positions"][node["track_ids"] == 2].clone()
    ckpt = {"model": {"gaussians_nodes": {
        "background": {"positions": torch.nn.Parameter(torch.randn(5, 3))},
        "road": {"positions": torch.nn.Parameter(torch.randn(4, 3))},
        "dynamic_rigids": node,
    }}}
    bg_before = ckpt["model"]["gaussians_nodes"]["background"]["positions"].clone()

    # stub align: 让 vehicle track 各换成 2 粒子的假 AlignedAsset（避免依赖真 PLY）
    from threedgrut.layers import e28_replace as M

    class _Fake:  # 鸭子类型 AlignedAsset 的 5 字段
        positions = torch.zeros(2, 3)
        rotations = torch.zeros(2, 4)
        scales_log = torch.zeros(2, 3)
        density_logit = torch.zeros(2, 1)
        colors = torch.full((2, 3), 0.5)

    monkeypatch.setattr(M, "_align_asset", lambda *a, **k: _Fake())

    recon = {"0": ("automobile", (4, 2, 1.5)), "1": ("car", (4, 2, 1.5)),
             "2": ("VRU_pedestrians", (0.6, 0.6, 1.7))}
    name_to_id = {"0": 0, "1": 1, "2": 2}
    bundle = {"x": AssetSpec("x", "c/x/g.ply", "consumer_vehicles", (4, 2, 1.5))}

    out, report = replace_all_vehicle_tracks(
        ckpt, bundle_root="/tmp", bundle=bundle, recon=recon,
        name_to_id=name_to_id, on_miss="global",
    )
    new = out["model"]["gaussians_nodes"]["dynamic_rigids"]
    ped_after = new["positions"][new["track_ids"] == 2]
    assert torch.equal(ped_before, ped_after)                       # ped 不动
    bg_after = out["model"]["gaussians_nodes"]["background"]["positions"]
    assert torch.equal(bg_before, bg_after)                         # bg 不动
    # vehicle track 0/1 各变 2 粒子
    assert int((new["track_ids"] == 0).sum()) == 2
    assert int((new["track_ids"] == 1).sum()) == 2
    assert {r.track for r in report if not r.skipped} == {"0", "1"}
