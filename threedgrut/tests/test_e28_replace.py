# SPDX-License-Identifier: Apache-2.0
"""E2.8 Task 2/3 — vehicle track enumeration + bank assignment + replace orchestration."""
import pytest
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


def test_is_vehicle_substring_matches_compound_classes():
    # 真实 NCore autolabel: dynamic_rigids 同时含 automobile + heavy_truck + person
    # (E2.8 inceptio convert 实测)。子串匹配须命中复合 truck 类、放过行人/骑行。
    from threedgrut.layers.e28_replace import is_vehicle
    assert is_vehicle("heavy_truck") is True
    assert is_vehicle("pickup_truck") is True
    assert is_vehicle("automobile") is True
    assert is_vehicle("person") is False
    assert is_vehicle("VRU_pedestrians") is False
    assert is_vehicle("cyclist") is False


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


# ----------------------------------------------------------------------------
# E2.8 insert: select active/nearby vehicle tracks (replace ∪ insert)
# ----------------------------------------------------------------------------
from threedgrut.layers.e28_replace import select_vehicle_tracks_to_place


def _cat(cls, slot, active, dist, present):
    return {"class": cls, "dims": (4.0, 2.0, 1.5), "slot": slot,
            "active_frames": active, "min_dist_to_ego": dist, "present": present}


def test_select_present_always_kept_insert_filtered():
    catalog = {
        "p":     _cat("automobile", 1, 5, 999.0, True),    # present → kept (far+brief OK)
        "near":  _cat("automobile", 2, 100, 10.0, False),  # insert: active+near → kept
        "far":   _cat("automobile", 3, 100, 80.0, False),  # too far → drop
        "brief": _cat("automobile", 4, 3, 10.0, False),    # too brief → drop
    }
    recon, name_to_id = select_vehicle_tracks_to_place(
        catalog, min_active_frames=20, max_dist_m=40.0)
    assert set(recon.keys()) == {"p", "near"}
    assert name_to_id == {"p": 1, "near": 2}
    assert recon["near"] == ("automobile", (4.0, 2.0, 1.5))


def test_select_thresholds_tunable():
    catalog = {"x": _cat("automobile", 0, 100, 80.0, False)}  # far
    assert select_vehicle_tracks_to_place(catalog, max_dist_m=40.0)[0] == {}   # dropped
    assert "x" in select_vehicle_tracks_to_place(catalog, max_dist_m=100.0)[0]  # kept


# ----------------------------------------------------------------------------
# E2.8 cross-source recon fallback (bus/truck AH can't size-match → real recon)
# ----------------------------------------------------------------------------
from threedgrut.layers.e28_replace import (
    _sorted_dims_ratio, split_vehicle_tracks_by_ah_match,
    place_tracks_in_dyn_node, extract_recon_node_tensors, keep_only_track_slots,
)


def test_keep_only_track_slots_drops_unlisted():
    tids = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)

    def p(c):
        return torch.nn.Parameter(torch.arange(6 * c, dtype=torch.float32).reshape(6, c))

    dyn = {"positions": p(3), "rotation": p(4), "scale": p(3), "density": p(1),
           "features_albedo": p(3), "features_specular": p(2),
           "track_ids": tids, "n_active_features": 0}
    new = keep_only_track_slots(dyn, {0, 2})              # drop slot 1
    assert int((new["track_ids"] == 0).sum()) == 2
    assert int((new["track_ids"] == 1).sum()) == 0        # dropped
    assert int((new["track_ids"] == 2).sum()) == 2
    assert new["positions"].shape[0] == 4
    assert new["n_active_features"] == 0                   # metadata carried over


def test_sorted_dims_ratio_orientation_agnostic():
    assert _sorted_dims_ratio((12.5, 3.1, 3.5), (5.8, 2.25, 1.88)) > 2.0   # bus vs pickup
    assert _sorted_dims_ratio((4.5, 1.8, 1.5), (5.76, 2.25, 1.88)) < 1.5   # car vs pickup
    # dim order shuffled → same ratio (sorted internally)
    assert _sorted_dims_ratio((1.5, 4.5, 1.8), (4.5, 1.8, 1.5)) == pytest.approx(1.0)


def test_split_routes_oversized_to_recon():
    bundle = {"pk": _spec("pk", "consumer_vehicles", (5.8, 2.25, 1.88))}
    sel = {"car": ("automobile", (4.5, 1.8, 1.5)),
           "bus": ("bus", (12.5, 3.1, 3.5))}
    ah, recon_tids = split_vehicle_tracks_by_ah_match(sel, bundle, max_size_ratio=1.5)
    assert set(ah.keys()) == {"car"}      # car within ratio → AH
    assert recon_tids == ["bus"]          # bus too big → recon


def test_place_tracks_insert_keeps_existing():
    import pytest as _p
    tids = torch.tensor([0, 0, 0], dtype=torch.int64)

    def p(c):
        return torch.nn.Parameter(torch.arange(3 * c, dtype=torch.float32).reshape(3, c))

    dyn = {"positions": p(3), "rotation": p(4), "scale": p(3), "density": p(1),
           "features_albedo": p(3), "features_specular": p(2),
           "track_ids": tids, "n_active_features": 0}
    before = dyn["positions"].detach().clone()
    nt = {5: {"positions": torch.ones(2, 3), "rotation": torch.ones(2, 4),
              "scale": torch.ones(2, 3), "density": torch.ones(2, 1),
              "features_albedo": torch.ones(2, 3), "features_specular": torch.ones(2, 2)}}
    new = place_tracks_in_dyn_node(dyn, nt)
    assert int((new["track_ids"] == 0).sum()) == 3            # existing kept
    assert int((new["track_ids"] == 5).sum()) == 2            # inserted
    assert torch.equal(new["positions"][new["track_ids"] == 0], before)  # byte-identical


def test_extract_recon_node_tensors_adjusts_spec_dim():
    rtids = torch.tensor([0, 1, 1], dtype=torch.int64)

    def rp(c, v):
        return torch.full((3, c), float(v))

    rdyn = {"positions": rp(3, 9), "rotation": rp(4, 9), "scale": rp(3, 9),
            "density": rp(1, 9), "features_albedo": rp(3, 9),
            "features_specular": rp(4, 9), "track_ids": rtids}
    out = extract_recon_node_tensors(rdyn, ["a", "7"], {"7": 20}, spec_dim=2)
    assert set(out.keys()) == {20}                            # tid 7 → usdz slot 20
    assert out[20]["positions"].shape == (2, 3)              # track 7 = 2 particles
    assert out[20]["features_specular"].shape == (2, 2)      # 4 → truncated to 2
