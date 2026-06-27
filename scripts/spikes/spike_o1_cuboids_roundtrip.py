#!/usr/bin/env python3
"""O1 spike — 官方 SequenceComponentGroupsWriter 写的 cuboids shard 能否被
SequenceComponentGroupsReader([meta, shard]) append 读回 + 定死 reference_frame_id 约定。

探测性强：自适应找 SDK 的 sequence_id / interval / pose_graph accessor，打印 dir，
逐条断言 A0–A5 并在末行打印 O1 决策（Branch A/B + ref_frame + 真实 accessor）。

Usage (inceptio, env 3dgrut2):
    python scripts/spikes/spike_o1_cuboids_roundtrip.py \
        --meta /home/inceptio/work/data/9ae151dc_consolidated/pai_9ae151dc-e87b-41a7-8e85-71772f9603d7.json \
        --out  /home/inceptio/work/cuboid_demo_out
"""
from __future__ import annotations

import argparse
import sys
import traceback


def _introspect(obj, name):
    attrs = [a for a in dir(obj) if not a.startswith("_")]
    print(f"[introspect] {name}: {attrs}")
    return attrs


def _try(getter, *names):
    for n in names:
        try:
            v = getter(n)
            print(f"  [ok] {n} = {v!r}")
            return n, v
        except Exception as e:
            print(f"  [--] {n}: {type(e).__name__}: {e}")
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import ncore.data as nd
    from ncore.data.v4 import (
        CuboidsComponent,
        SequenceComponentGroupsReader,
        SequenceComponentGroupsWriter,
        SequenceLoaderV4,
    )
    from upath import UPath

    print("=== STEP A: 打开源 meta + introspect ===")
    src = SequenceComponentGroupsReader([args.meta], open_consolidated=True)
    _introspect(src, "reader")
    loader = SequenceLoaderV4(
        src,
        poses_component_group_name="default",
        intrinsics_component_group_name="default",
        masks_component_group_name="default",
    )
    _introspect(loader, "loader")

    _, seq_id = _try(lambda n: getattr(loader, n), "sequence_id")
    if seq_id is None:
        _, seq_id = _try(lambda n: getattr(src, n), "sequence_id")
    _, interval = _try(
        lambda n: getattr(loader, n),
        "sequence_timestamp_interval_us", "timestamp_interval_us", "time_range",
    )
    if interval is None:
        _, interval = _try(
            lambda n: getattr(src, n), "sequence_timestamp_interval_us", "time_range")
    _, pose_graph = _try(lambda n: getattr(loader, n), "pose_graph")
    print(f"\nseq_id={seq_id!r}  interval={interval!r}  pose_graph_present={pose_graph is not None}")
    if interval is not None:
        _introspect(interval, "interval")

    existing = []
    try:
        existing = list(loader.get_cuboid_track_observations())
        print(f"[existing] get_cuboid_track_observations -> {len(existing)} 条")
        if existing:
            o0 = existing[0]
            _introspect(o0, "existing_obs[0]")
            print(f"  obs0: track_id={getattr(o0,'track_id',None)} "
                  f"class={getattr(o0,'class_id',None)} ts={getattr(o0,'timestamp_us',None)} "
                  f"ref={getattr(o0,'reference_frame_id',None)}")
            _introspect(o0.bbox3, "existing_bbox3")
    except Exception as e:
        print(f"[existing] FAIL: {e}")

    # 取两个 ts：interval start 附近 → 退路 existing obs ts → 硬编码
    T0 = None
    for cand in ("start", "begin", "lower", "min"):
        if interval is not None and hasattr(interval, cand):
            try:
                T0 = int(getattr(interval, cand))
                break
            except Exception:
                pass
    if T0 is None and existing:
        T0 = int(existing[0].timestamp_us)
    if T0 is None:
        T0 = 1_000_000_000
    T1 = T0 + 100_000
    print(f"T0={T0} T1={T1}")

    print("\n=== STEP B: 写 2 条 obs（reference_frame_id='world'）===")
    out = UPath(args.out)
    gmd = getattr(src, "generic_meta_data", {}) or {}
    try:
        _keys = list(gmd) if hasattr(gmd, "__iter__") else gmd
        print(f"[gmd] source generic_meta_data = {_keys!r}")
    except Exception:
        print(f"[gmd] source generic_meta_data = {gmd!r}")
    try:
        w = SequenceComponentGroupsWriter(
            output_dir_path=out, store_base_name="autocuboids",
            sequence_id=seq_id, sequence_timestamp_interval_us=interval,
            generic_meta_data=gmd, store_type="itar")
        cw = w.register_component_writer(CuboidsComponent.Writer, "auto_v0", group_name="auto_cuboids")
        obs = []
        for ts, cx in ((T0, 10.0), (T1, 11.0)):
            obs.append(nd.CuboidTrackObservation(
                track_id="auto_0", class_id="automobile", timestamp_us=ts,
                reference_frame_id="world", reference_frame_timestamp_us=ts,
                bbox3=nd.BBox3(centroid=(cx, 0.0, 0.85), dim=(4.5, 2.0, 1.7), rot=(0.0, 0.0, 0.3)),
                source=nd.LabelSource.AUTOLABEL, source_version="lidar-cluster-v1"))
        cw.store_observations(obs).finalize()
        shard_paths = w.finalize()
        print(f"[A0] write OK, shard_paths = {shard_paths}")
        assert shard_paths, "A0 FAIL: finalize() 空"
    except Exception:
        print("[A0] WRITE FAILED:")
        traceback.print_exc()
        return 1

    print("\n=== STEP C: A1 append 读（决策 fork）===")
    branch = "B"
    rd = None
    try:
        rd = SequenceComponentGroupsReader([args.meta, *map(str, shard_paths)], open_consolidated=True)
        print("[A1] PASS: reader append 构造成功 → Branch A")
        branch = "A"
    except Exception:
        print("[A1] FAIL: reader append 拒绝 → Branch B")
        traceback.print_exc()

    loader2 = None
    cub_group = "default"
    if rd is not None:
        try:
            cub_keys = list(rd.open_component_readers(CuboidsComponent.Reader))
            print(f"[groups] cuboids reader keys = {cub_keys}")
            mine = [k for k in cub_keys if k != "default"]
            cub_group = mine[0] if mine else "default"
            print(f"[groups] 用 cuboids group = {cub_group!r} 读回我的 obs")
            loader2 = SequenceLoaderV4(
                rd, poses_component_group_name="default",
                intrinsics_component_group_name="default", masks_component_group_name="default",
                cuboids_component_group_name=cub_group)
        except Exception:
            print("[groups] loader2 构造失败:")
            traceback.print_exc()

    print("\n=== STEP D: A5 standalone open ===")
    loader_s = None
    try:
        rds = SequenceComponentGroupsReader([*map(str, shard_paths)], open_consolidated=True)
        loader_s = SequenceLoaderV4(
            rds, poses_component_group_name="default",
            intrinsics_component_group_name="default", masks_component_group_name="default")
        sp = getattr(loader_s, "pose_graph", None)
        print(f"[A5] standalone open OK; pose_graph_present={sp is not None}")
    except Exception:
        print("[A5] standalone open FAIL:")
        traceback.print_exc()

    print("\n=== STEP E: A2/A3 obs 可见 + 字段保真 ===")
    use = loader2 if branch == "A" else loader_s
    if use is not None:
        try:
            got = [o for o in use.get_cuboid_track_observations() if o.track_id == "auto_0"]
            print(f"[A2] 读回 auto_0 obs: {len(got)} 条")
            if got:
                o = sorted(got, key=lambda x: x.timestamp_us)[0]
                pg = getattr(use, "pose_graph", None) or pose_graph
                wb = o.transform("world", int(o.timestamp_us), pg).bbox3
                print(f"[A3] world centroid={tuple(round(float(v),3) for v in wb.centroid)} "
                      f"dim={tuple(round(float(v),3) for v in wb.dim)} "
                      f"rot={tuple(round(float(v),3) for v in wb.rot)}")
        except Exception:
            print("[A2/A3] FAIL:")
            traceback.print_exc()

    print(f"\n=== O1 决策：Branch {branch} | reference_frame_id='world' | seq_id={seq_id!r} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
