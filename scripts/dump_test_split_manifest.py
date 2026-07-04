# SPDX-License-Identifier: Apache-2.0
"""E0.4-O1 — dump the project test split as a pose manifest (JSON).

Enumerates the SAME test dataloader render.py evaluates on (identical
make_test + DataLoader construction) and records, per batch: iteration
index, camera_id, frame_idx, timestamp, shutter start/end c2w, image size.
The NuRec side (nre render) uses this to produce frames at exactly the
project's eval poses, and scripts/eval_frames_dir.py uses it to align
prediction files with GT batches — the 口径-unification backbone of the
E0.4 bidirectional comparison.

Run on a machine with the NCore clip (inceptio; dataset is CUDA-backed):
    python scripts/dump_test_split_manifest.py \
        --checkpoint <ckpt.pt> --path <pai_*.json> --output manifest.json
"""

from __future__ import annotations

import os as _os
import sys as _sys

# scripts/ 下直跑时 sys.path[0] 是 scripts/ 而非仓库根 —— import threedgrut 会
# 落到 conda env 的 editable 安装（可能是另一份 checkout/旧分支）。强制本仓库优先。
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

import argparse
import json

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--path", default="", help="override ckpt dataset path")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import threedgrut.datasets as datasets
    from threedgrut.datasets.utils import configure_dataloader_for_platform

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    conf = ckpt["config"]
    if args.path:
        conf.path = args.path

    # Mirror Renderer.create_test_dataloader exactly (same split semantics).
    dataset = datasets.make_test(name=conf.dataset.type, config=conf)
    dataloader_kwargs = configure_dataloader_for_platform(
        {"num_workers": 0, "batch_size": 1, "shuffle": False, "collate_fn": None}
    )
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    entries = []
    for it, batch in enumerate(dataloader):
        gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
        T = gpu_batch.T_to_world
        Te = getattr(gpu_batch, "T_to_world_end", T)
        entries.append(
            {
                "iteration": it,
                "camera_id": getattr(gpu_batch, "camera_id", None),
                "frame_idx": int(getattr(gpu_batch, "frame_idx", -1)),
                "timestamp_us": int(getattr(gpu_batch, "timestamp_us", -1)),
                "c2w_start": T[0].detach().cpu().numpy().tolist(),
                "c2w_end": Te[0].detach().cpu().numpy().tolist(),
                "H": int(gpu_batch.rgb_gt.shape[1]),
                "W": int(gpu_batch.rgb_gt.shape[2]),
            }
        )

    header = {
        "checkpoint": args.checkpoint,
        "n_entries": len(entries),
        "camera_ids": list(getattr(dataset, "camera_ids", []) or []),
        "val_frame_interval": int(conf.dataset.get("val_frame_interval", 8)),
    }
    with open(args.output, "w") as f:
        json.dump({"header": header, "entries": entries}, f)
    print(f"wrote {len(entries)} entries → {args.output}")
    by_cam: dict = {}
    for e in entries:
        by_cam[e["camera_id"]] = by_cam.get(e["camera_id"], 0) + 1
    print("per-camera:", by_cam)


if __name__ == "__main__":
    main()
