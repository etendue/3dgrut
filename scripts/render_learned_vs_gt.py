"""V3 Stage D.2 — Render learned vs GT cuboid poses side-by-side.

For each sampled (camera, frame), produces a triptych PNG:

    [ GT pose render | Learned pose render | |learned - gt| × amplify ]

Driven by ``LayeredGaussians.set_pose_source("gt" | "learned")``: each
sample is rendered twice through the same Gaussians, with only the
``_compose_pose_*`` route flipped. Everything else (gaussian positions,
opacities, exposure, sky, view) is identical between the two renders,
so the diff column isolates the contribution of learned pose drift.

Requires a Stage A / Stage B ckpt whose model has both
``_track_quat_<tid>`` Parameters AND frozen ``_track_pose_gt_<tid>``
buffers (the standard learnable_pose.enabled=true output). On a
buffer-mode ckpt the "gt" route falls through and the diff is trivially
zero — a warning is emitted.

Usage::

    python scripts/render_learned_vs_gt.py \\
        --checkpoint /path/to/ckpt_30000.pt \\
        --out-dir    /tmp/d2_30k_stageB \\
        --num-samples 8

The output dir holds one PNG per sample, plus a printed log line with
the per-sample mean / max diff magnitude (pre-amplify).
"""

import argparse
import os
import sys

import torch
import torchvision.utils as tvu

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from threedgrut.layers.layered_model import LayeredGaussians  # noqa: E402
from threedgrut.render import Renderer  # noqa: E402


@torch.no_grad()
def _render_with_source(model, gpu_batch, source: str) -> torch.Tensor:
    """Toggle pose_source on the model, do one forward pass, return
    pred_rgb as a ``[H, W, 3]`` float tensor clamped to [0, 1]."""
    model.set_pose_source(source)
    outputs = model(gpu_batch)
    return outputs["pred_rgb"].squeeze(0).clamp(0, 1)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True, help="path to ckpt_<iter>.pt with learnable_pose state")
    ap.add_argument("--out-dir", required=True, help="output directory for triptych PNGs")
    ap.add_argument("--num-samples", type=int, default=8, help="how many (cam, frame) samples to render (default: 8)")
    ap.add_argument(
        "--stride",
        type=int,
        default=0,
        help="dataloader stride between picked samples; "
        "0 → auto = max(1, n_total // num_samples) "
        "to spread samples evenly across the clip",
    )
    ap.add_argument(
        "--diff-amplify",
        type=float,
        default=10.0,
        help="per-pixel abs-diff multiplier for visualization "
        "(default 10×; pre-amplify mean/max printed per sample)",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"=== render_learned_vs_gt ===")
    print(f"  ckpt:    {args.checkpoint}")
    print(f"  out:     {args.out_dir}")

    renderer = Renderer.from_checkpoint(
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        save_gt=False,
        computes_extra_metrics=False,
    )
    model = renderer.model
    if not isinstance(model, LayeredGaussians):
        print(
            f"ERROR: model is {type(model).__name__}, not LayeredGaussians — "
            f"this script only works on multilayer ckpts."
        )
        sys.exit(2)
    if not hasattr(model, "set_pose_source"):
        print("ERROR: LayeredGaussians has no set_pose_source — " "rebuild from a branch that contains V3 Stage D.2.")
        sys.exit(2)

    has_gt_buf = any(name.startswith("_track_pose_gt_") for name in model._buffers)
    if not has_gt_buf:
        print(
            "WARN: no _track_pose_gt_<tid> buffer registered — "
            "the 'gt' route will fall through to the buffer/learned route, "
            "yielding an all-zero diff. Continuing anyway."
        )

    n_total = len(renderer.dataset)
    stride = args.stride if args.stride > 0 else max(1, n_total // max(args.num_samples, 1))
    print(f"  dataset: {n_total} frames, stride={stride}, num_samples={args.num_samples}")
    print(f"  tracks:  {len(model._iter_track_tids())}")
    print()

    n_done = 0
    for i, batch in enumerate(renderer.dataloader):
        if i % stride != 0:
            continue
        if n_done >= args.num_samples:
            break

        # NCore path stores camera_id in the raw batch (DataLoader collates
        # the string into a length-1 list). After get_gpu_batch_with_intrinsics
        # it may be elided, so resolve here.
        _bcid = batch.get("camera_id", None) if isinstance(batch, dict) else None
        if isinstance(_bcid, (list, tuple)) and _bcid:
            _bcid = _bcid[0]
        cam_id = _bcid or "cam"

        gpu_batch = renderer.dataset.get_gpu_batch_with_intrinsics(batch)

        img_learned = _render_with_source(model, gpu_batch, "learned")  # [H, W, 3]
        img_gt = _render_with_source(model, gpu_batch, "gt")  # [H, W, 3]

        diff = (img_learned - img_gt).abs().mean(dim=-1, keepdim=True)  # [H, W, 1]
        diff_vis = (diff * args.diff_amplify).clamp(0, 1).expand(-1, -1, 3)

        triptych = torch.cat([img_gt, img_learned, diff_vis], dim=1)  # [H, 3*W, 3]

        fname = f"sample{n_done:02d}_batch{i:04d}_{cam_id}.png"
        out_path = os.path.join(args.out_dir, fname)
        tvu.save_image(triptych.permute(2, 0, 1), out_path)
        print(
            f"  [{n_done + 1:>2}/{args.num_samples}] {fname}  "
            f"diff mean={diff.mean().item():.5f}  max={diff.max().item():.5f}"
        )
        n_done += 1

    # Restore default route so the renderer object is reusable.
    model.set_pose_source("learned")
    print(f"\nDone — wrote {n_done} triptychs to {args.out_dir}")
    print("Columns (left → right): GT pose | Learned pose | |learned − gt| × amplify")


if __name__ == "__main__":
    main()
