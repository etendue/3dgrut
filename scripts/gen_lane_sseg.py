#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Phase 3 lane GT (方案 A, 分支 B) — 自跑 mask2former + Mapillary Vistas 逐帧
生成车道线分割产物 ``*.aux.lane.zarr.itar``。

为什么自跑：NCore 现成 sseg（``mask2former_dinov2_nv_private12k``，Cityscapes-20
类）**无 lane-marking 类**（road 是粗类）。Mapillary Vistas 标签集含
``Lane Marking - General`` 等专门类，正是 Phase 3 测量门所需。本脚本不依赖
nre-tools，直接用 HF transformers 的 Mapillary 预训练 mask2former 在 NCore 前视
RGB 上逐帧推理，按 **与现 sseg 字节同构** 的布局写出独立 itar，让现成
``SsegAuxReader`` / ``per_class_eval.compute_lane_metrics`` 原样消费。

产物契约（与 aux.sseg.zarr.itar 同构，复用 SsegAuxReader 零改）：
    <clip>/<stem>.aux.lane.zarr.itar
      /aux/semantic_segmentation/<camera_id>/<END_ts_us>  -> 0-D |S<n> PNG bytes
        attrs.format = "png"
      /aux/semantic_segmentation/<camera_id>/.zattrs
        stuff_classes = [<Mapillary 65 类标签>], resolution = [W, H],
        method / dataset_name / pretrained_checkpoint

用法（inceptio，env 3dgrut2，HF 经 hf-mirror）：
    HF_ENDPOINT=https://hf-mirror.com python scripts/gen_lane_sseg.py \
        --clip ~/work/data/9ae151dc --camera camera_front_wide_120fov

IndexedTarStore mode='w' 是 **write-once**：每个节点的 .zattrs 只能写一次
（``attrs.put({...})`` 一次性写全），不可逐 key 追加（已实测 ValueError）。
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def find_manifest(clip_dir: Path) -> Path:
    cands = sorted(clip_dir.glob("pai_*.json"))
    if not cands:
        raise FileNotFoundError(f"no pai_*.json manifest in {clip_dir}")
    # prefer the one without an aux suffix (the sequence manifest)
    for c in cands:
        if ".aux." not in c.name:
            return c
    return cands[0]


def out_itar_path(clip_dir: Path, manifest: Path) -> Path:
    # mirror the sseg naming: <stem>.aux.lane.zarr.itar where <stem> matches
    # the existing aux files (e.g. pai_<clip>).
    sseg = sorted(clip_dir.glob("*.aux.sseg.zarr.itar"))
    if sseg:
        stem = sseg[0].name.replace(".aux.sseg.zarr.itar", "")
    else:
        stem = manifest.stem
    return clip_dir / f"{stem}.aux.lane.zarr.itar"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, type=Path, help="clip dir holding pai_*.json + aux itars")
    ap.add_argument("--camera", default="camera_front_wide_120fov")
    ap.add_argument("--model", default="facebook/mask2former-swin-large-mapillary-vistas-semantic")
    ap.add_argument("--out", default=None, type=Path, help="override output itar path")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N frames (0=all)")
    ap.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fp16 推理（默认开；--no-fp16 关，用于 fp16/fp32 精度对比）",
    )
    args = ap.parse_args()

    clip_dir = args.clip.expanduser()
    manifest = find_manifest(clip_dir)
    out_path = args.out or out_itar_path(clip_dir, manifest)
    print(f"[gen_lane] clip={clip_dir}")
    print(f"[gen_lane] manifest={manifest.name}")
    print(f"[gen_lane] camera={args.camera}")
    print(f"[gen_lane] out={out_path}")
    print(f"[gen_lane] HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '(default)')}")

    # ---- model ----
    import torch
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[gen_lane] loading model {args.model} on {device} ...")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model).to(device).eval()
    if args.fp16 and device == "cuda":
        model = model.half()
    id2label = model.config.id2label
    lane_ids = sorted(int(i) for i, lbl in id2label.items() if "lane marking" in str(lbl).lower())
    print(f"[gen_lane] num classes={len(id2label)}")
    print(f"[gen_lane] LANE-MARKING ids (label contains 'lane marking'):")
    for i in lane_ids:
        print(f"            {i}: {id2label[i]}")
    if not lane_ids:
        print("[gen_lane][WARN] no 'lane marking' label found; dumping full id2label for manual reconcile:")
        for i in sorted(id2label, key=int):
            print(f"            {i}: {id2label[i]}")
    stuff_classes = [str(id2label[i]) for i in sorted(id2label, key=int)]

    # ---- ncore clip reader ----
    import ncore.data
    import ncore.data.v4

    loader = ncore.data.v4.SequenceLoaderV4(
        ncore.data.v4.SequenceComponentGroupsReader([manifest]),
    )
    cam = loader.get_camera_sensor(args.camera)
    n_frames = cam.frames_timestamps_us.shape[0]
    if args.limit:
        n_frames = min(n_frames, args.limit)
    print(f"[gen_lane] frames to process: {n_frames}")

    # ---- itar writer (write-once; single .zattrs per node) ----
    import zarr
    from ncore.impl.data import stores

    if out_path.exists():
        print(f"[gen_lane] removing existing {out_path}")
        out_path.unlink()
    st = stores.IndexedTarStore(str(out_path), mode="w")
    root = zarr.open(store=st, mode="w")
    cam_grp = root.create_group(f"aux/semantic_segmentation/{args.camera}")

    t0 = time.time()
    H_ref = W_ref = None
    written = 0
    try:
        for fi in range(n_frames):
            ts_end = int(cam.frames_timestamps_us[fi, ncore.data.FrameTimepoint.END])
            rgb = cam.get_frame_image_array(fi)  # [H,W,3] uint8 (raw/distorted, matches GT)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                print(f"[gen_lane][WARN] frame {fi} unexpected rgb shape {rgb.shape}; skip")
                continue
            H, W = rgb.shape[:2]
            if H_ref is None:
                H_ref, W_ref = H, W
                # write camera-group attrs ONCE, after first frame gives resolution
                cam_grp.attrs.put(
                    {
                        "stuff_classes": stuff_classes,
                        "resolution": [int(W), int(H)],
                        "method": "mask2former-mapillary-vistas (self-run, gen_lane_sseg.py)",
                        "pretrained_checkpoint": args.model,
                        "dataset_name": "mapillary-vistas-v1.2",
                        "lane_marking_ids": lane_ids,
                    }
                )
            inputs = processor(images=Image.fromarray(rgb), return_tensors="pt")
            pix = inputs["pixel_values"].to(device)
            if args.fp16 and device == "cuda":
                pix = pix.half()
            with torch.no_grad():
                out = model(pixel_values=pix)
            seg = processor.post_process_semantic_segmentation(out, target_sizes=[(H, W)])[0]  # [H,W] long
            seg_np = seg.to(torch.uint8).cpu().numpy()
            buf = io.BytesIO()
            Image.fromarray(seg_np).save(buf, format="png")
            pb = buf.getvalue()
            ds = cam_grp.create_dataset(str(ts_end), shape=(), dtype=f"|S{len(pb)}", compressor=None)
            ds[()] = np.bytes_(pb)
            ds.attrs.put({"format": "png"})
            written += 1
            if written % 50 == 0 or fi == n_frames - 1:
                dt = time.time() - t0
                print(f"[gen_lane] {written}/{n_frames}  ({dt:.1f}s, {dt/max(written,1):.2f}s/frame)")
    finally:
        # IndexedTarStore(mode="w") write-once：异常中途也要 flush 已写帧
        # （下次 run 会 unlink 重来，但避免半写 itar 留坑）。
        if hasattr(st, "close"):
            st.close()
    print(f"[gen_lane] DONE: wrote {written} frames -> {out_path} " f"({out_path.stat().st_size/1e6:.1f} MB)")
    print(
        f"[gen_lane] RECONCILE: set LANE_CLASS_IDS = {tuple(lane_ids)} "
        f"in threedgrut/model/per_class_eval.py + test guard"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
