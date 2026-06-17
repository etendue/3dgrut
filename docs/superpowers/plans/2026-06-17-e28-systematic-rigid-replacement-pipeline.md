# E2.8 系统性 dynamic rigid 全替流水线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 USDZ(NuRec) 场景拆成「静态底 + per-track dynamic rigid」，将**所有** vehicle track 成批换成 AH 资产库里按 (class, size) 匹配的干净 3DGS asset（deformable 丢弃），harmonizer 离线协调，QA 闸（sanity + NTA-IoU/FID）把关，产出 3dgrut2 viser 可编辑场景——一条可复现可批量的流水线。

**Architecture:** 纯函数核心（Mac 可单测）+ inceptio GPU 节点（渲染/harmonizer/检测）。新增 `asset_bank.py`（bank 查询 + fallback）和 `e28_replace.py`（全 track 枚举 + 编排），复用 E2.5 的 `e25_inject.replace_tracks_in_dyn_node`（已支持批替换）、`warmstart_metadata`/`warmstart_ply`（对齐引擎）、E2.7 `nre_usdz_loader`（拆）、E2.1/E2.6 harmonizer、E1.2 NTA-IoU、E1.4 FID。

**Tech Stack:** PyTorch(CPU 核)、PyYAML、pytest；inceptio RTX 4090（depth-off，git worktree 工作流）；asset-harvester skill（建库）。

---

## 设计依据
spec：[`docs/superpowers/specs/2026-06-17-e28-systematic-rigid-replacement-pipeline-design.md`](../specs/2026-06-17-e28-systematic-rigid-replacement-pipeline-design.md)

## 既有接口（执行前必读，避免签名错位）
- `threedgrut/layers/warmstart_metadata.py`：`AssetSpec(asset_hash, ply_file, label_class, cuboids_dims:tuple[L,W,H])`、`load_bundle_metadata(yaml)->{hash:AssetSpec}`、`resolve_ply_path(root, spec)->Path`。
- `threedgrut/layers/e25_inject.py`：`match_assets_by_size`（greedy **bijection**，本卡不复用它做 bank 查询）、`replace_tracks_in_dyn_node(dyn_node, aligned_by_id:{int_id:AlignedAsset})->new_node`（**已支持批替换 + 守护非目标 track 字节不变**，本卡直接复用）、`aligned_to_node_tensors`、`flip_forward_180`、`build_name_to_int_id`。
- `threedgrut/layers/warmstart_ply.py`：`AlignedAsset`、`AlignmentTransform`、PLY→AlignedAsset 对齐（见 `scripts/e25_inject_ah_replace.py` main L169–188 的 align 调用，T3 抽成可复用函数）。
- `threedgrut_playground/utils/nre_usdz_loader.py`：`NRE_PARTICLE_LAYERS=("background","road","dynamic_rigids")`（deformable 天然丢弃）。

## File Structure
- **Create** `threedgrut/layers/asset_bank.py` — bank 查询（class 过滤 + L2 最近 + fallback ladder）。
- **Create** `threedgrut/layers/e28_replace.py` — 全 vehicle 枚举 + bank 分配 + 批 align + 调 `replace_tracks_in_dyn_node` + `ReplaceReport`；QA sanity 纯函数。
- **Create** `scripts/e28_systematic_replace_pipeline.py` — 编排 driver（拆→替→渲→协调→QA）。
- **Create** tests `threedgrut/tests/test_e28_asset_bank.py`、`test_e28_replace.py`、`test_e28_qa_sanity.py`。
- **Reuse**（不改）`e25_inject.py`、`warmstart_*.py`、`nre_usdz_loader.py`、`e21_harmonizer_batch_fix.py`、`vehicle_detector.py`。

---

### Task 1: AssetBank 查询（class 过滤 + size 最近 + fallback ladder）

**Files:**
- Create: `threedgrut/layers/asset_bank.py`
- Test: `threedgrut/tests/test_e28_asset_bank.py`

- [ ] **Step 1: 写失败测试**

```python
# threedgrut/tests/test_e28_asset_bank.py
import pytest
from threedgrut.layers.warmstart_metadata import AssetSpec
from threedgrut.layers.asset_bank import query_bank, BankMiss

def _spec(h, cls, dims):
    return AssetSpec(asset_hash=h, ply_file=f"{cls}/{h}/gaussians.ply",
                     label_class=cls, cuboids_dims=tuple(dims))

BUNDLE = {
    "sedan1": _spec("sedan1", "consumer_vehicles", (4.5, 1.8, 1.5)),
    "suv1":   _spec("suv1",   "consumer_vehicles", (4.9, 2.0, 1.8)),
    "bus1":   _spec("bus1",   "bus",               (12.0, 2.5, 3.2)),
}

def test_same_class_nearest_size():
    # 4.6×1.85×1.55 最接近 sedan1
    h, level = query_bank(BUNDLE, "consumer_vehicles", (4.6, 1.85, 1.55))
    assert h == "sedan1"
    assert level == 0  # same-class exact-ish

def test_one_asset_reused_across_calls():
    # bank 查询不消耗资产：同一资产可被多次返回（区别 bijection）
    h1, _ = query_bank(BUNDLE, "consumer_vehicles", (4.5, 1.8, 1.5))
    h2, _ = query_bank(BUNDLE, "consumer_vehicles", (4.5, 1.8, 1.5))
    assert h1 == h2 == "sedan1"

def test_cross_class_fallback_warns_level1():
    # truck 类 bank 没有 → 跨 class 全局最近 + level 1
    h, level = query_bank(BUNDLE, "truck", (11.5, 2.5, 3.0))
    assert h == "bus1"        # 全局 L2 最近
    assert level == 1

def test_on_miss_skip_raises_bankmiss():
    empty = {}
    with pytest.raises(BankMiss):
        query_bank(empty, "consumer_vehicles", (4.5, 1.8, 1.5), on_miss="skip")

def test_deterministic_tie_break():
    # 两资产等距 → 按 hash 字典序定 deterministic
    b = {"a": _spec("a", "c", (1, 1, 1)), "b": _spec("b", "c", (1, 1, 1))}
    h, _ = query_bank(b, "c", (1, 1, 1))
    assert h == "a"
```

- [ ] **Step 2: 跑测试看失败**

Run: `python -m pytest threedgrut/tests/test_e28_asset_bank.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'threedgrut.layers.asset_bank'`

- [ ] **Step 3: 写实现**

```python
# threedgrut/layers/asset_bank.py
# SPDX-License-Identifier: Apache-2.0
"""E2.8 — AH 资产库查询：按 (label_class, cuboid dims) 从 bank 选最近资产。

区别 e25_inject.match_assets_by_size（greedy bijection，每资产用一次）：bank
查询**不消耗**资产——同一库资产可被多个 track 复用（编辑场景不要 identity）。
fallback ladder：① 同 class L2 最近（level 0）② 跨 class 全局最近 + WARN（level
1）③ on_miss='skip' 时抛 BankMiss 让调用方保留 recon（level 跳过，不 silent）。
"""
from __future__ import annotations

import logging

import torch

from threedgrut.layers.warmstart_metadata import AssetSpec

logger = logging.getLogger(__name__)


class BankMiss(Exception):
    """bank 无可用资产且 on_miss='skip' —— 调用方应保留该 track recon。"""


def _l2(a, b) -> float:
    ta = torch.as_tensor(a, dtype=torch.float32)
    tb = torch.as_tensor(b, dtype=torch.float32)
    return float(torch.linalg.norm(ta - tb))


def query_bank(
    bundle: dict[str, AssetSpec],
    label_class: str,
    dims,
    *,
    on_miss: str = "global",
) -> tuple[str, int]:
    """选最匹配的 ``asset_hash``。返回 ``(asset_hash, fallback_level)``。

    fallback_level: 0=同 class 最近 · 1=跨 class 全局最近（WARN）。
    ``on_miss``: 'global'（默认，空 class 退全局）或 'skip'（空 class 抛 BankMiss）。
    平手按 ``asset_hash`` 字典序（deterministic）。
    """
    same = [s for s in bundle.values() if s.label_class == label_class]
    if same:
        best = min(same, key=lambda s: (_l2(s.cuboids_dims, dims), s.asset_hash))
        return best.asset_hash, 0
    if on_miss == "skip" or not bundle:
        raise BankMiss(
            f"no asset for class {label_class!r} (on_miss={on_miss}, "
            f"bank classes={sorted({s.label_class for s in bundle.values()})})"
        )
    best = min(bundle.values(), key=lambda s: (_l2(s.cuboids_dims, dims), s.asset_hash))
    logger.warning(
        "asset_bank: class %r absent; cross-class fallback → %r (%s)",
        label_class, best.asset_hash, best.label_class,
    )
    return best.asset_hash, 1
```

- [ ] **Step 4: 跑测试看通过**

Run: `python -m pytest threedgrut/tests/test_e28_asset_bank.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: commit**

```bash
git add threedgrut/layers/asset_bank.py threedgrut/tests/test_e28_asset_bank.py
git commit -m "feat(E2.8): asset_bank query — class+size nearest + fallback ladder"
```

---

### Task 2: 全 vehicle track 枚举 + bank 分配 + ReplaceReport

**Files:**
- Create: `threedgrut/layers/e28_replace.py`
- Test: `threedgrut/tests/test_e28_replace.py`

> 注：本 task 只做**纯数据层**的「哪个 track 用哪个资产」分配 + 报告；真正的 PLY align + 粒子替换在 Task 3。先把可 Mac 单测的分配逻辑钉死。

- [ ] **Step 1: 写失败测试**

```python
# threedgrut/tests/test_e28_replace.py
from dataclasses import asdict
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
```

- [ ] **Step 2: 跑测试看失败**

Run: `python -m pytest threedgrut/tests/test_e28_replace.py -v`
Expected: FAIL — `ImportError: cannot import name 'assign_assets_to_tracks'`

- [ ] **Step 3: 写实现**

```python
# threedgrut/layers/e28_replace.py
# SPDX-License-Identifier: Apache-2.0
"""E2.8 — 系统性全替编排：全 vehicle track 枚举 + bank 分配 + 批 align + 替换。

Task 2 = 纯数据分配（Mac 可测）；Task 3 = 加 PLY align + 调
e25_inject.replace_tracks_in_dyn_node 做粒子替换。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from threedgrut.layers.warmstart_metadata import AssetSpec
from threedgrut.layers.asset_bank import query_bank, BankMiss

logger = logging.getLogger(__name__)

# 与 e25_inject_ah_replace._CAR_CLASS 对齐的 NCore autolabel vehicle 类。
VEHICLE_CLASSES = frozenset({
    "automobile", "bus", "truck", "consumer_vehicles", "car", "vehicle",
})


@dataclass(frozen=True)
class AssignRow:
    track: str
    label_class: str
    chosen_asset: str | None
    fallback_level: int | None
    skipped: bool


def is_vehicle(label_class: str) -> bool:
    return str(label_class).lower() in VEHICLE_CLASSES


def assign_assets_to_tracks(
    recon: dict[str, tuple[str, tuple]],
    bundle: dict[str, AssetSpec],
    *,
    on_miss: str = "global",
) -> tuple[dict[str, str], list[AssignRow]]:
    """``recon``: ``{track_name: (label_class, (L,W,H))}``.

    返回 ``({track_name: asset_hash}, [AssignRow])``。非 vehicle track 不分配；
    bank miss + on_miss='skip' → 该 track 留 recon（skipped=True，不 silent）。
    """
    assign: dict[str, str] = {}
    report: list[AssignRow] = []
    for track, (cls, dims) in recon.items():
        if not is_vehicle(cls):
            continue
        try:
            asset_hash, level = query_bank(bundle, str(cls), dims, on_miss=on_miss)
        except BankMiss:
            report.append(AssignRow(track, str(cls), None, None, True))
            logger.warning("e28: track %r kept recon (bank miss)", track)
            continue
        assign[track] = asset_hash
        report.append(AssignRow(track, str(cls), asset_hash, level, False))
    return assign, report
```

- [ ] **Step 4: 跑测试看通过**

Run: `python -m pytest threedgrut/tests/test_e28_replace.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: commit**

```bash
git add threedgrut/layers/e28_replace.py threedgrut/tests/test_e28_replace.py
git commit -m "feat(E2.8): vehicle track enumeration + bank assignment + ReplaceReport"
```

---

### Task 3: 批注入 orchestration（align + replace_tracks_in_dyn_node）

**Files:**
- Modify: `threedgrut/layers/e28_replace.py`（加 `replace_all_vehicle_tracks`）
- Test: `threedgrut/tests/test_e28_replace.py`（加守护测试）

> 复用 E2.5：(a) `scripts/e25_inject_ah_replace.py` main L169–188 的 PLY→AlignedAsset 对齐（填 recon live cuboid + `flip_forward_180` + convention）抽成 `_align_asset(spec, bundle_root, dims) -> AlignedAsset`；(b) `e25_inject.replace_tracks_in_dyn_node(dyn_node, aligned_by_id)` 做批替换（**已守护非目标 track 字节不变**）。

- [ ] **Step 1: 写失败守护测试**

```python
# 追加到 threedgrut/tests/test_e28_replace.py
import torch
from threedgrut.layers.e28_replace import replace_all_vehicle_tracks

def _toy_dyn_node():
    # 2 个 vehicle track (ids 0,1) 各 3 粒子 + 1 ped track (id 2) 2 粒子
    tids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.int64)
    n = tids.shape[0]
    def p(c): return torch.nn.Parameter(torch.arange(n * c, dtype=torch.float32).reshape(n, c))
    return {
        "positions": p(3), "rotation": p(4), "scale": p(3), "density": p(1),
        "features_albedo": p(3), "features_specular": p(2),
        "track_ids": tids, "n_active_features": 0,
    }

def test_non_vehicle_track_particles_unchanged(monkeypatch):
    # ped track (id 2) 的粒子在替换后逐字节不变
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
    class _Fake:  # 鸭子类型 AlignedAsset 的 6 字段
        positions = torch.zeros(2, 3); rotations = torch.zeros(2, 4)
        scales_log = torch.zeros(2, 3); density_logit = torch.zeros(2, 1)
        colors = torch.full((2, 3), 0.5)
    monkeypatch.setattr(M, "_align_asset", lambda *a, **k: _Fake())

    recon = {"0": ("automobile", (4, 2, 1.5)), "1": ("car", (4, 2, 1.5)),
             "2": ("VRU_pedestrians", (0.6, 0.6, 1.7))}
    name_to_id = {"0": 0, "1": 1, "2": 2}
    bundle = {"x": __import__("threedgrut.layers.warmstart_metadata", fromlist=["AssetSpec"]).AssetSpec("x", "c/x/g.ply", "consumer_vehicles", (4, 2, 1.5))}

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
```

- [ ] **Step 2: 跑测试看失败**

Run: `python -m pytest threedgrut/tests/test_e28_replace.py::test_non_vehicle_track_particles_unchanged -v`
Expected: FAIL — `ImportError: cannot import name 'replace_all_vehicle_tracks'`

- [ ] **Step 3: 写实现（追加到 e28_replace.py）**

```python
# 追加 import（文件顶部）
from pathlib import Path
from threedgrut.layers.warmstart_ply import AlignedAsset, AlignmentTransform
from threedgrut.layers.warmstart_metadata import resolve_ply_path
from threedgrut.layers.e25_inject import flip_forward_180, replace_tracks_in_dyn_node


def _align_asset(spec, bundle_root, dims) -> AlignedAsset:
    """PLY → 填 recon live cuboid `dims` 的 AlignedAsset（含 180° yaw-flip）。

    复用 e25_inject_ah_replace.py main L169–188 的对齐：load PLY → build
    AlignmentTransform 填 cuboid size/center → flip_forward_180 → apply。
    执行时把 main 里那段 align 调用原样搬进来（它已在 A800/inceptio 验证）。
    """
    raise NotImplementedError(
        "搬入 scripts/e25_inject_ah_replace.py main L169-188 的 align 调用"
    )


def replace_all_vehicle_tracks(
    ckpt: dict, *, bundle_root, bundle, recon, name_to_id, on_miss="global",
) -> tuple[dict, list[AssignRow]]:
    """全 vehicle track 批替换。``recon``: {track_name:(class,dims)}；
    ``name_to_id``: build_name_to_int_id(tracks)。返回 (ckpt, report)。"""
    assign, report = assign_assets_to_tracks(recon, bundle, on_miss=on_miss)
    dyn = ckpt["model"]["gaussians_nodes"]["dynamic_rigids"]
    aligned_by_id: dict[int, AlignedAsset] = {}
    for track, asset_hash in assign.items():
        spec = bundle[asset_hash]
        dims = recon[track][1]
        aligned_by_id[int(name_to_id[track])] = _align_asset(
            spec, Path(bundle_root), dims
        )
    if aligned_by_id:
        ckpt["model"]["gaussians_nodes"]["dynamic_rigids"] = \
            replace_tracks_in_dyn_node(dyn, aligned_by_id)
    return ckpt, report
```

> ⚠️ 测试用 `monkeypatch` stub 了 `_align_asset`，所以 Task 3 单测不依赖真 PLY；`_align_asset` 的真实实现在 Task 5 端到端时从 E2.5 main 搬入并 inceptio 验证（搬入后此处 `raise` 删除）。

- [ ] **Step 4: 跑测试看通过**

Run: `python -m pytest threedgrut/tests/test_e28_replace.py -v`
Expected: PASS（守护测试 + Task 2 的 4 个）

- [ ] **Step 5: commit**

```bash
git add threedgrut/layers/e28_replace.py threedgrut/tests/test_e28_replace.py
git commit -m "feat(E2.8): replace_all_vehicle_tracks orchestration (guards bg/road/non-vehicle)"
```

---

### Task 4: QA sanity 纯函数（覆盖率 / opacity 防烟雾 / 粒子数）

**Files:**
- Modify: `threedgrut/layers/e28_replace.py`（加 `qa_sanity`）
- Test: `threedgrut/tests/test_e28_qa_sanity.py`

> proj-IoU / NTA-IoU / FID 需渲染 → 留 Task 6（inceptio）；本 task 只做 Mac 可测的廉价 sanity 闸。

- [ ] **Step 1: 写失败测试**

```python
# threedgrut/tests/test_e28_qa_sanity.py
import torch
from threedgrut.layers.e28_replace import qa_sanity, AssignRow

def _node(opacity_logit, n_per_tid):
    tids = torch.cat([torch.full((n,), t) for t, n in n_per_tid.items()])
    N = tids.shape[0]
    return {"track_ids": tids,
            "density": torch.full((N, 1), float(opacity_logit)),
            "positions": torch.randn(N, 3)}

def test_full_coverage_passes():
    report = [AssignRow("0", "car", "a", 0, False),
              AssignRow("1", "bus", "b", 0, False)]
    after = _node(2.0, {0: 50, 1: 60})   # sigmoid(2.0)=0.88 正常 opacity
    qa = qa_sanity(after, report)
    assert qa["coverage"] == 1.0
    assert qa["n_skipped"] == 0
    assert qa["opacity_median"] > 0.3
    assert qa["passed"] is True

def test_smoke_opacity_fails():
    report = [AssignRow("0", "car", "a", 0, False)]
    after = _node(-2.1, {0: 50})         # sigmoid(-2.1)=0.109 ≈ E2.7 烟雾区
    qa = qa_sanity(after, report)
    assert qa["opacity_median"] < 0.15
    assert qa["passed"] is False         # 烟雾回归被闸住

def test_skips_lower_coverage():
    report = [AssignRow("0", "car", "a", 0, False),
              AssignRow("1", "truck", None, None, True)]   # skip
    after = _node(2.0, {0: 50})
    qa = qa_sanity(after, report)
    assert qa["coverage"] == 0.5
    assert qa["n_skipped"] == 1
```

- [ ] **Step 2: 跑测试看失败**

Run: `python -m pytest threedgrut/tests/test_e28_qa_sanity.py -v`
Expected: FAIL — `ImportError: cannot import name 'qa_sanity'`

- [ ] **Step 3: 写实现（追加到 e28_replace.py）**

```python
def qa_sanity(dyn_node_after: dict, report: list[AssignRow],
              *, opacity_floor: float = 0.15) -> dict:
    """廉价 sanity 闸（Mac 可跑）。覆盖率 + opacity 防烟雾 + skip 计数。

    opacity_floor 0.15：E2.7 烟雾态 dynamic gaussian opacity≈0.11，正常 AH
    资产应远高于此 → median ≤ floor 判 fail（防烟雾回归）。
    """
    total = len(report)
    skipped = [r for r in report if r.skipped]
    replaced = [r for r in report if not r.skipped]
    coverage = (len(replaced) / total) if total else 1.0
    opacity = torch.sigmoid(dyn_node_after["density"].flatten())
    opacity_median = float(opacity.median()) if opacity.numel() else 0.0
    passed = (opacity_median > opacity_floor) and (len(replaced) > 0)
    return {
        "coverage": coverage,
        "n_total": total,
        "n_replaced": len(replaced),
        "n_skipped": len(skipped),
        "skipped_tracks": [r.track for r in skipped],
        "opacity_median": opacity_median,
        "passed": passed,
    }
```

- [ ] **Step 4: 跑测试看通过**

Run: `python -m pytest threedgrut/tests/test_e28_qa_sanity.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: commit**

```bash
git add threedgrut/layers/e28_replace.py threedgrut/tests/test_e28_qa_sanity.py
git commit -m "feat(E2.8): qa_sanity gate — coverage + anti-smoke opacity floor"
```

---

### Task 5: 编排 driver + `_align_asset` 落地（inceptio 端到端）

**Files:**
- Create: `scripts/e28_systematic_replace_pipeline.py`
- Modify: `threedgrut/layers/e28_replace.py`（`_align_asset` 删 `raise`，搬入真实 align）

> inceptio git worktree 工作流（CLAUDE.md）。前置：Task 7 资产库已建（或先用现有 3 车 bundle 跑通链路）。

- [ ] **Step 1: `_align_asset` 落地** — 把 `scripts/e25_inject_ah_replace.py` main L169–188 的 align 调用搬进 `_align_asset`（load PLY via `resolve_ply_path` → build `AlignmentTransform` 填 `dims` 的 size/center → `flip_forward_180` → apply 得 `AlignedAsset`）。删掉 `NotImplementedError`。

- [ ] **Step 2: 写 driver**

```python
# scripts/e28_systematic_replace_pipeline.py
"""E2.8 编排：USDZ → 全 vehicle 替换 → render-only → harmonizer → QA。"""
import argparse, json
from pathlib import Path
import torch
from threedgrut_playground.utils.nre_usdz_loader import load_nre_usdz  # ① 拆
from threedgrut.layers.warmstart_metadata import load_bundle_metadata
from threedgrut.layers.e25_inject import build_name_to_int_id
from threedgrut.layers.e28_replace import replace_all_vehicle_tracks, qa_sanity
from dataclasses import asdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usdz", required=True)
    ap.add_argument("--asset_bank", required=True, help="dir with metadata.yaml")
    ap.add_argument("--dataset_path", required=True, help="NCore pai_*.json")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--on_miss", default="global", choices=["global", "skip"])
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    ckpt, tracks, recon = load_usdz_to_ckpt(a.usdz, a.dataset_path)  # 见 Step 3
    bundle = load_bundle_metadata(Path(a.asset_bank) / "metadata.yaml")
    name_to_id = build_name_to_int_id(tracks)
    ckpt, report = replace_all_vehicle_tracks(
        ckpt, bundle_root=a.asset_bank, bundle=bundle, recon=recon,
        name_to_id=name_to_id, on_miss=a.on_miss)
    torch.save(ckpt, out / "ckpt_replaced.pt")
    json.dump([asdict(r) for r in report], open(out / "replace_report.json", "w"), indent=2)

    dyn = ckpt["model"]["gaussians_nodes"]["dynamic_rigids"]
    qa = qa_sanity(dyn, report)
    json.dump(qa, open(out / "qa_sanity.json", "w"), indent=2)
    print(f"[E2.8] coverage={qa['coverage']} opacity_med={qa['opacity_median']:.3f} "
          f"passed={qa['passed']}")
    if not qa["passed"]:
        raise SystemExit("QA sanity FAILED — 不要进协调阶段，先查 replace_report")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 接 `load_usdz_to_ckpt`** — 用 `nre_usdz_loader` 拆出 `(ckpt, tracks_dict)`，并从 `sequence_tracks` autolabel + cuboid dims 组 `recon={name:(class,dims)}`（复用 E2.7-B 的 `parse_sequence_tracks` / cuboid 解析；recon dims 取 live cuboid，与 E2.5 main 同源）。

- [ ] **Step 4: inceptio 跑替换 + sanity**（git worktree，depth 无关——纯 CPU 手术）

```bash
git push inceptio e28:e28
ssh inceptio 'cd ~/repo/3dgrut2 && git worktree add ~/repo/3dgrut2-wt/e28 e28'
ssh inceptio 'cd ~/repo/3dgrut2; WT=~/repo/3dgrut2-wt/e28; for p in $(git config --file .gitmodules --get-regexp path | cut -d" " -f2); do rsync -a ~/repo/3dgrut2/$p/ $WT/$p/; done'
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH \
  && cd ~/repo/3dgrut2-wt/e28 \
  && python scripts/e28_systematic_replace_pipeline.py \
       --usdz <last.usdz> --asset_bank <bank_dir> \
       --dataset_path <pai_*.json> --out_dir ~/work/output/e28_run'
```
Expected: 打印 `coverage=1.0 ... passed=True`，产 `ckpt_replaced.pt` + 两 json。

- [ ] **Step 5: viser 目测**（复用 viser-gui-4d skill）加载 `ckpt_replaced.pt`，确认全部 vehicle 是干净 AH 车、无烟雾、无行人（deformable 已丢）。

- [ ] **Step 6: commit**

```bash
git add scripts/e28_systematic_replace_pipeline.py threedgrut/layers/e28_replace.py
git commit -m "feat(E2.8): end-to-end driver + _align_asset landed (inceptio verified)"
```

---

### Task 6: QA 定量（NTA-IoU + FID）+ harmonizer 协调（inceptio）

**Files:**
- Modify: `scripts/e28_systematic_replace_pipeline.py`（加 `--with-quant` 段）

- [ ] **Step 1: 协调节点** — 复用 E2.1 `scripts/e21_harmonizer_batch_fix.py`：起 `harmonizer_server`（E2.6 temporal）→ render-only 渲 `ckpt_replaced.pt` 帧 → batch-fix → `harmonized_frames/`。命令模式照搬 E2.1 Done Log（`--render-only` 关监督，~4.62s/帧）。

- [ ] **Step 2: NTA-IoU** — 用 `threedgrut/model/vehicle_detector.py`（E1.2 yolov8m conf 0.3）对替换后渲染帧检出车 + 与投影 cuboid IoU；对照 recon 原 actor（replace 后 NTA-IoU 不低于 recon 基线 0.117/原轨迹）。

- [ ] **Step 3: FID** — 复用 E1.4 `--novel-fid`：协调前 vs 协调后帧分布（期望协调后 FID 下降，对标 E2.1 离线 −33%/−28% 量级）。写 `qa_report.json`（合并 sanity + NTA-IoU + FID）。

- [ ] **Step 4: 验收** — `qa_report.json` 含 `nta_iou_{interp}`、`fid_{before,after}`、sanity 全字段；NTA-IoU ≥ recon 基线、FID after < before。Monitor 只 grep 关键节点（`⭐|FID|NTA|Traceback|FAILED|OOM`，勿 grep 逐帧 PSNR）。

- [ ] **Step 5: commit**

```bash
git add scripts/e28_systematic_replace_pipeline.py
git commit -m "feat(E2.8): quantitative QA — NTA-IoU + FID + harmonizer coordination"
```

---

### Task 7: 建 AH 资产库（可与 Task 1–4 并行先行，inceptio）

**Files:**
- Create: `<asset_bank>/metadata.yaml` + `<class>/<hash>/gaussians.ply`

- [ ] **Step 1** — 用 `asset-harvester` skill 从 NCore clip per-object 收割补资产：覆盖 sedan / SUV / van / bus / truck，每类 1–2 尺寸代表（现有 3 车为起点）。
- [ ] **Step 2** — 每收割一款，先用 viser（`--gs_object`）目测渲染干净（不烟雾、朝向对）再入库；`AxisMap` 若新类朝向异常，按 class 校准（spec § 6 风险表）。
- [ ] **Step 3** — 汇 `metadata.yaml`：`assets: {hash: {label_class, cuboids_dims:[L,W,H], ply_file}}`（`load_bundle_metadata` 格式）。
- [ ] **Step 4** — `python -c "from threedgrut.layers.warmstart_metadata import load_bundle_metadata; print(load_bundle_metadata('<bank>/metadata.yaml'))"` 验证可解析。
- [ ] **Step 5: commit**（库 manifest；ply 大文件按项目 .gitignore 约定，必要时只入 manifest + 路径）

---

### Task 8: 文档回填（plan 看板 + Done Log）

- [ ] **Step 1** — `v4_plan.md`：§1.2 加 E2.8 行（状态从实测回填）、§1.1 Mermaid Kanban 加 E2.8 卡（**全角括号铁律**）、§1.3 E2 计数 +1、§5 Done Log 追加（commit hash + NTA-IoU/FID/coverage 实测）。
- [ ] **Step 2** — `v2_architecture.md`：§6 文件清单加 `asset_bank.py` / `e28_replace.py` / driver；§7 不变量加「E2.8 全替守护 bg/road/非 vehicle 字节不变」锚。
- [ ] **Step 3** — 跑 Mermaid 全角括号自查：`awk '/```mermaid/{i=1;next} /```/&&i{i=0;next} i&&/\(/{print FILENAME":"NR": "$0}' v4_plan.md`（应零输出）。
- [ ] **Step 4: commit**

```bash
git commit -am "docs(plan): mark E2.8 done — systematic rigid replacement pipeline"
```

---

## Self-Review（spec 覆盖核对）
- ① 拆（deformable 丢弃）→ Task 5 Step 3（load_usdz_to_ckpt 走 loader 白名单）✅
- ② 配（bank + fallback）→ Task 1（query）+ Task 2（assign）+ Task 7（建库）✅
- ③ 批注入（全 track + 守护）→ Task 3（replace_all_vehicle_tracks 守护测试）✅
- ④ 协调（harmonizer 离线）→ Task 6 Step 1 ✅
- ⑤ QA（sanity + NTA-IoU/FID）→ Task 4（sanity）+ Task 6（定量）✅
- ⑥ viser 消费 → Task 5 Step 5 ✅
- 编排 driver → Task 5 ✅；文档 → Task 8 ✅
- 类型一致：`AssignRow`/`query_bank(->（hash,level))`/`replace_all_vehicle_tracks` 跨 task 签名一致 ✅
- 无占位符（除 `_align_asset` 明确标注 Task 5 搬入 E2.5 main，附 stub 测试隔离）✅

## 验收门
- Mac：Task 1–4 全 pytest 绿（12 测试）。
- inceptio：coverage=1.0（或 skip 有据）· opacity 防烟雾 passed · NTA-IoU ≥ recon 基线 · FID after<before · viser 目测干净无行人。
- gate：E2.1/E2.5/E2.6/E1.2/E1.4 ✅（均已 done）。
