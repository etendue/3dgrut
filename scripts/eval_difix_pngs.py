"""V3-T15.2 Stage A.4 — offline DiFix evaluation from saved PNGs.

Designed to run inside the cosmos-predict2-container on a GPU host that has
the DiFix runtime stack installed (TE / flash_attn / cosmos_predict2 / Pix2Pix_Turbo).
Reads pairs of pred/gt PNGs produced by a prior ``render.py`` run, pushes each
pred through ``DifixPostProcessor.forward()``, and writes a JSON with the
``mean_*_difix`` trio plus the matching ``mean_*`` baseline so we can compute
Δ-PSNR / Δ-SSIM / Δ-LPIPS without re-running the 3DGUT renderer.

Usage:
    python scripts/eval_difix_pngs.py \
        --renders-dir /path/to/.../ours_<step>/renders \
        --gt-dir      /path/to/.../ours_<step>/gt \
        --output      /tmp/difix_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Make the project root importable so ``threedgrut.correction.difix`` resolves.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from threedgrut.correction.difix import DifixPostProcessor


def load_png_to_tensor(p: Path) -> torch.Tensor:
    """Read a PNG to (H, W, 3) float32 ∈ [0, 1] on cuda."""
    img = Image.open(p).convert("RGB")
    t = torchvision.transforms.functional.to_tensor(img)  # (3, H, W) ∈ [0, 1]
    return t.permute(1, 2, 0).contiguous().unsqueeze(0).cuda()  # (1, H, W, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders-dir", required=True, type=Path)
    ap.add_argument(
        "--gt-dir",
        required=False,
        type=Path,
        default=None,
        help="Optional. If provided, compute baseline + DiFix PSNR/SSIM/LPIPS Δ. "
        "Omit for novel-view inputs where ground truth doesn't exist — "
        "only DiFix forward + (optional) PNG save happens.",
    )
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Override DiFix ckpt path (default $HF_HOME/nvidia-Fixer/pretrained/pretrained_fixer.pkl)",
    )
    ap.add_argument(
        "--max-frames", type=int, default=0, help="Optionally limit number of frames for a quick smoke (0 = all)"
    )
    ap.add_argument(
        "--save-difix-dir",
        type=Path,
        default=None,
        help="If set, save each DiFix-processed pred PNG into this dir (e.g. <out>/ours_<step>/difix). "
        "Filename mirrors the source render PNG (00000.png ... 00374.png).",
    )
    args = ap.parse_args()
    if args.save_difix_dir is not None:
        args.save_difix_dir.mkdir(parents=True, exist_ok=True)
        print(f"[eval] saving DiFix PNGs to {args.save_difix_dir}")

    render_paths = sorted(args.renders_dir.glob("*.png"))
    have_gt = args.gt_dir is not None
    if have_gt:
        gt_paths = sorted(args.gt_dir.glob("*.png"))
        assert len(render_paths) == len(gt_paths), f"render/gt count mismatch: {len(render_paths)} vs {len(gt_paths)}"
    else:
        gt_paths = [None] * len(render_paths)
    if args.max_frames > 0:
        render_paths = render_paths[: args.max_frames]
        gt_paths = gt_paths[: args.max_frames]
    n = len(render_paths)
    print(f"[eval] processing {n} frames " f"({args.renders_dir} ↔ {args.gt_dir or '(no gt, save-only)'})")

    # Metrics modules — only allocate when we have gt.
    if have_gt:
        psnr_m = PeakSignalNoiseRatio(data_range=1).to("cuda")
        ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
        lpips_m = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to("cuda")
    else:
        psnr_m = ssim_m = lpips_m = None

    # DiFix wrapper. Lazy init triggers on first forward.
    difix = DifixPostProcessor(enabled=True, ckpt_path=args.ckpt)

    baseline_psnr: list[float] = []
    baseline_ssim: list[float] = []
    baseline_lpips: list[float] = []
    difix_psnr: list[float] = []
    difix_ssim: list[float] = []
    difix_lpips: list[float] = []
    timings_ms: list[float] = []

    for i, (rp, gp) in enumerate(zip(render_paths, gt_paths)):
        pred = load_png_to_tensor(rp)  # (1, H, W, 3) float32
        gt = load_png_to_tensor(gp) if gp is not None else None

        # Baseline metrics (pred vs gt) — only when gt provided
        if have_gt:
            baseline_psnr.append(psnr_m(pred.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)).item())
            baseline_ssim.append(ssim_m(pred.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)).item())
            baseline_lpips.append(lpips_m(pred.clip(0, 1).permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)).item())

        # DiFix
        torch.cuda.synchronize()
        t0 = time.time()
        pred_difix = difix(pred)  # (1, H, W, 3) float32 ∈ [0,1]
        torch.cuda.synchronize()
        timings_ms.append((time.time() - t0) * 1000)

        if args.save_difix_dir is not None:
            torchvision.utils.save_image(
                pred_difix.squeeze(0).permute(2, 0, 1).clip(0, 1),
                args.save_difix_dir / rp.name,
            )

        # DiFix metrics — only when gt provided
        if have_gt:
            difix_psnr.append(psnr_m(pred_difix.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)).item())
            difix_ssim.append(ssim_m(pred_difix.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)).item())
            difix_lpips.append(lpips_m(pred_difix.clip(0, 1).permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)).item())

        if (i + 1) % 25 == 0 or i == n - 1 or not have_gt:
            if have_gt:
                print(
                    f"  [{i+1:3d}/{n}] "
                    f"psnr={baseline_psnr[-1]:5.2f}→{difix_psnr[-1]:5.2f} "
                    f"lpips={baseline_lpips[-1]:.3f}→{difix_lpips[-1]:.3f} "
                    f"({timings_ms[-1]:.0f} ms)"
                )
            else:
                print(
                    f"  [{i+1:3d}/{n}] DiFix forward {timings_ms[-1]:.0f} ms"
                    + (f"  saved → {(args.save_difix_dir/rp.name).name}" if args.save_difix_dir else "")
                )

    def avg(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    out: dict = {
        "n_frames": n,
        "have_gt": have_gt,
        "mean_difix_forward_ms": avg(timings_ms),
        "first_forward_ms": timings_ms[0] if timings_ms else 0.0,
    }
    if have_gt:
        out.update(
            {
                "mean_psnr": avg(baseline_psnr),
                "mean_ssim": avg(baseline_ssim),
                "mean_lpips": avg(baseline_lpips),
                "mean_psnr_difix": avg(difix_psnr),
                "mean_ssim_difix": avg(difix_ssim),
                "mean_lpips_difix": avg(difix_lpips),
                "delta_psnr": avg(difix_psnr) - avg(baseline_psnr),
                "delta_ssim": avg(difix_ssim) - avg(baseline_ssim),
                "delta_lpips": avg(difix_lpips) - avg(baseline_lpips),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print()
    print(f"=== written {args.output} ===")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
