# SPDX-License-Identifier: Apache-2.0
"""E0.4-O2 — offline same-protocol evaluator for externally rendered frames.

"render_all minus the model": iterates the project's OWN test split (GT,
sseg, lane masks, cuboids, FTheta intrinsics all from the project dataset)
but reads predictions from a directory of PNGs — e.g. frames produced by
``nre render`` from a NuRec USDZ. This is what makes the E0.4 NuRec ↔
multilayer comparison single-口径: all metric CODE is project-side, NuRec
only contributes pixels.

Modes:
- ``interpolated``: pred frames at the original test poses → PSNR/SSIM/
  (LPIPS)/cc_PSNR + per-class + lane + NTA-IoU + FID/KID. Keys mirror
  render.py naming.
- ``lateral_3m`` / ``lateral_6m``: pred frames rendered at the laterally
  shifted poses (same shift definition as novel_view.perturb_c2w) → ONLY
  no-pixel-GT metrics: plane-warp lane metrics, NTA-IoU at the perturbed
  pose, FID/KID vs the GT distribution.

Frame alignment: ``<frames_dir>/<camera_id>/<frame_idx:06d>.png`` by
default; ``--frames-map map.json`` ({"<camera_id>:<frame_idx>": "relpath"})
overrides. A missing prediction is a HARD error — silent misalignment would
poison the anchor.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from threedgrut.model.per_class_eval import (
    DEFAULT_ACTOR_CLASS_SPECS,
    DEFAULT_LANE_BAND_PX,
    LANE_CLASS_IDS,
    ROAD_CLASS_IDS,
    compute_lane_metrics,
    compute_per_class_metrics,
)
from threedgrut.model.plane_warp import build_plane_warp, warp_image
from threedgrut.utils.color_correct import color_correct_affine
from threedgrut.utils.novel_view import perturb_c2w

LANE_METRIC_KEYS = ("lane_band_lpips", "lane_band_psnr", "lane_raw_psnr", "lane_grad_corr")
DEFAULT_LANE_EVAL_CAMERAS = ("camera_front_wide_120fov",)


def resolve_pred_path(
    frames_dir: str, camera_id: str, frame_idx: int,
    frames_map: Optional[Dict[str, str]] = None,
) -> str:
    """Map a (camera_id, frame_idx) batch to its prediction file."""
    if frames_map:
        key = f"{camera_id}:{int(frame_idx)}"
        if key in frames_map:
            return os.path.join(frames_dir, frames_map[key])
    return os.path.join(frames_dir, str(camera_id), f"{int(frame_idx):06d}.png")


def _load_pred(path: str, device) -> torch.Tensor:
    import torchvision

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"prediction frame missing: {path} — frame alignment is broken, "
            f"refusing to continue (E0.4 anchor integrity)"
        )
    img = torchvision.io.read_image(path).float().div(255.0)  # [C,H,W]
    return img[:3].permute(1, 2, 0).to(device)                # [H,W,3]


def _psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = float(((pred - gt) ** 2).mean())
    return float(-10.0 * np.log10(max(mse, 1e-12)))


def evaluate_frames(
    batches,
    frames_dir: str,
    frames_map: Optional[Dict[str, str]],
    mode: str,
    lpips_fn=None,
    detector=None,
    tracks_provider: Optional[Callable] = None,
    height_field: Optional[dict] = None,
    ground_z: Optional[float] = None,
    fid_kid: bool = False,
    lane_band_px: int = DEFAULT_LANE_BAND_PX,
    lane_eval_cameras=DEFAULT_LANE_EVAL_CAMERAS,
) -> dict:
    """Run the offline eval over an iterable of gpu_batch-like objects."""
    is_novel = mode != "interpolated"
    if is_novel and mode not in ("lateral_3m", "lateral_6m", "lateral_1m", "lateral_2m"):
        raise ValueError(f"unsupported mode {mode}")

    ssim_fn = None
    psnr_l: List[float] = []
    ssim_l: List[float] = []
    lpips_l: List[float] = []
    cc_psnr_l: List[float] = []
    per_class_psnr: Dict[str, list] = {}
    per_class_lpips: Dict[str, list] = {}
    lane_acc: Dict[str, list] = {}
    lane_frames = 0
    warp_valid_ratio: List[float] = []
    nta_records: List[dict] = []
    fid_pair = None
    n_real = n_fake = 0

    specs = {**DEFAULT_ACTOR_CLASS_SPECS, "road_crop": ROAD_CLASS_IDS}

    for batch in batches:
        gt = batch.rgb_gt[0]                                   # [H,W,3]
        device = gt.device
        cam = getattr(batch, "camera_id", None)
        fi = int(getattr(batch, "frame_idx", -1))
        pred = _load_pred(
            resolve_pred_path(frames_dir, cam, fi, frames_map), device,
        )
        if pred.shape != gt.shape:
            raise ValueError(
                f"pred/gt shape mismatch at {cam}:{fi}: "
                f"{tuple(pred.shape)} vs {tuple(gt.shape)}"
            )
        ftheta = getattr(batch, "intrinsics_FThetaCameraModelParameters", None)
        infos = getattr(batch, "image_infos", None) or {}
        H, W = int(gt.shape[0]), int(gt.shape[1])

        if fid_kid and fid_pair is None:
            from torchmetrics.image.fid import FrechetInceptionDistance
            from torchmetrics.image.kid import KernelInceptionDistance
            fid_pair = {
                "fid": FrechetInceptionDistance(feature=2048).to(device),
                "kid": KernelInceptionDistance(subset_size=2).to(device),
            }
        if fid_pair is not None:
            from threedgrut.utils.eval_metrics import rgb01_to_uint8_chw
            fid_pair["fid"].update(rgb01_to_uint8_chw(gt.unsqueeze(0)), real=True)
            fid_pair["kid"].update(rgb01_to_uint8_chw(gt.unsqueeze(0)), real=True)
            fid_pair["fid"].update(rgb01_to_uint8_chw(pred.unsqueeze(0)), real=False)
            fid_pair["kid"].update(rgb01_to_uint8_chw(pred.unsqueeze(0)), real=False)
            n_real += 1
            n_fake += 1

        if not is_novel:
            psnr_l.append(_psnr(pred, gt))
            if ssim_fn is None:
                from torchmetrics.image import StructuralSimilarityIndexMeasure
                ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            p4 = pred.permute(2, 0, 1).unsqueeze(0)
            g4 = gt.permute(2, 0, 1).unsqueeze(0)
            ssim_l.append(float(ssim_fn(p4, g4)))
            if lpips_fn is not None:
                lpips_l.append(float(lpips_fn(p4.clamp(0, 1), g4)))
            pred_cc = color_correct_affine(pred.unsqueeze(0), gt.unsqueeze(0))[0]
            cc_psnr_l.append(_psnr(pred_cc, gt))

            sseg = infos.get("semantic_sseg")
            if sseg is not None:
                sseg_one = sseg[0] if sseg.dim() == 3 else sseg
                pcm = compute_per_class_metrics(
                    pred, gt, sseg_one, specs, lpips_fn=lpips_fn,
                )
                for name, d in pcm.items():
                    if d["psnr"] is not None:
                        per_class_psnr.setdefault(name, []).append(d["psnr"])
                    if d["lpips"] is not None:
                        per_class_lpips.setdefault(name, []).append(d["lpips"])

            lane = infos.get("semantic_lane_sseg")
            if lane is not None and cam in lane_eval_cameras:
                lane_one = lane[0] if lane.dim() == 3 else lane
                lm = compute_lane_metrics(
                    pred, gt, lane_one, LANE_CLASS_IDS,
                    band_px=lane_band_px, lpips_fn=lpips_fn,
                )
                for k in LANE_METRIC_KEYS:
                    if lm[k] is not None:
                        lane_acc.setdefault(k, []).append(lm[k])
                lane_frames += 1

            if detector is not None and tracks_provider is not None:
                active = tracks_provider(batch)
                if active:
                    from threedgrut.model.nta_iou import compute_frame_nta_iou
                    T_w2c = torch.linalg.inv(batch.T_to_world[0])
                    nta = compute_frame_nta_iou(
                        pred, active, detector, K=None, ftheta_params=ftheta,
                        T_w2c=T_w2c, H=H, W=W,
                    )
                    if nta is not None:
                        nta_records.append(nta)
        else:
            # novel mode: pred was rendered at the PERTURBED pose; pixel-GT
            # metrics are undefined. Plane-warp lane + NTA at perturbed pose.
            c2w_novel = torch.from_numpy(
                perturb_c2w(batch.T_to_world[0].cpu(), mode)
            ).to(device=device, dtype=torch.float32)

            lane = infos.get("semantic_lane_sseg")
            if (
                lane is not None and cam in lane_eval_cameras
                and ftheta is not None
                and (height_field is not None or ground_z is not None)
                and not getattr(batch, "rays_in_world_space", False)
            ):
                lane_one = lane[0] if lane.dim() == 3 else lane
                grid, valid = build_plane_warp(
                    batch.rays_dir[0], c2w_novel,
                    batch.T_to_world[0].to(torch.float32), ftheta,
                    height_field=height_field, z0_fallback=ground_z,
                )
                gt_warp = warp_image(gt.float(), grid, valid)
                lane_warp = warp_image(
                    lane_one.unsqueeze(-1).float(), grid, valid, mode="nearest",
                )[..., 0].long()
                lm = compute_lane_metrics(
                    pred, gt_warp, lane_warp, LANE_CLASS_IDS,
                    band_px=lane_band_px, restrict_mask=valid, lpips_fn=lpips_fn,
                )
                for k in LANE_METRIC_KEYS:
                    if lm[k] is not None:
                        lane_acc.setdefault(k, []).append(lm[k])
                lane_frames += 1
                warp_valid_ratio.append(float(valid.float().mean()))

            if detector is not None and tracks_provider is not None:
                active = tracks_provider(batch)
                if active:
                    from threedgrut.model.nta_iou import compute_frame_nta_iou
                    nta = compute_frame_nta_iou(
                        pred, active, detector, K=None, ftheta_params=ftheta,
                        T_w2c=torch.linalg.inv(c2w_novel), H=H, W=W,
                    )
                    if nta is not None:
                        nta_records.append(nta)

    # ---------------------------------------------------------- aggregate
    out: dict = {"mode": mode, "n_frames": len(psnr_l) if not is_novel else lane_frames}
    if not is_novel:
        out["n_frames"] = len(psnr_l)
        if psnr_l:
            out["mean_psnr"] = float(np.mean(psnr_l))
            out["mean_ssim"] = float(np.mean(ssim_l))
            out["mean_cc_psnr"] = float(np.mean(cc_psnr_l))
        if lpips_l:
            out["mean_lpips"] = float(np.mean(lpips_l))
        for name in per_class_psnr:
            out[f"mean_{name}_psnr"] = float(np.mean(per_class_psnr[name]))
            out[f"{name}_n_records"] = len(per_class_psnr[name])
        for name in per_class_lpips:
            out[f"mean_{name}_lpips"] = float(np.mean(per_class_lpips[name]))
        if lane_frames:
            for k in LANE_METRIC_KEYS:
                v = lane_acc.get(k, [])
                out[f"mean_{k}"] = float(np.mean(v)) if v else None
            out["lane_n_records"] = lane_frames
        if nta_records:
            out["mean_nta_iou"] = float(
                np.mean([r["mean_nta_iou"] for r in nta_records])
            )
            out["nta_iou_n_frames"] = len(nta_records)
    else:
        if lane_frames:
            for k in LANE_METRIC_KEYS:
                v = lane_acc.get(k, [])
                out[f"mean_novel_{k}_{mode}"] = float(np.mean(v)) if v else None
            out[f"novel_lane_n_records_{mode}"] = lane_frames
            out[f"novel_lane_warp_valid_ratio_{mode}"] = float(
                np.mean(warp_valid_ratio)
            )
        if nta_records:
            out[f"mean_novel_nta_iou_{mode}"] = float(
                np.mean([r["mean_nta_iou"] for r in nta_records])
            )
            out[f"novel_nta_iou_n_frames_{mode}"] = len(nta_records)

    if fid_pair is not None and n_real >= 2:
        from threedgrut.utils.eval_metrics import kid_subset_size
        if not is_novel:
            k_fid, k_kid, k_kid_std = (
                "mean_render_fid", "mean_render_kid", "mean_render_kid_std",
            )
        else:  # plan §6 naming: metric before mode
            k_fid, k_kid, k_kid_std = (
                f"mean_novel_fid_{mode}", f"mean_novel_kid_{mode}",
                f"mean_novel_kid_std_{mode}",
            )
        try:
            out[k_fid] = float(fid_pair["fid"].compute())
        except Exception as e:  # degenerate split → keys absent
            print(f"[eval_frames_dir] FID compute failed: {e}")
        try:
            fid_pair["kid"].subset_size = kid_subset_size(min(n_real, n_fake))
            km, ks = fid_pair["kid"].compute()
            out[k_kid] = float(km)
            out[k_kid_std] = float(ks)
        except Exception as e:
            print(f"[eval_frames_dir] KID compute failed: {e}")
        out["fid_n_real"] = n_real
        out[f"fid_n_fake_{'render' if not is_novel else mode}"] = n_fake
    return out


def _road_positions_from_ckpt(ckpt) -> Optional[torch.Tensor]:
    """Best-effort extraction of road-layer positions from a layered ckpt
    (no model build). Returns None when the layout is unknown."""
    model_sd = ckpt.get("model", {})
    nodes = model_sd.get("gaussians_nodes") or model_sd.get("layers")
    if isinstance(nodes, dict) and "road" in nodes:
        road = nodes["road"]
        for key in ("positions", "means", "xyz"):
            if isinstance(road, dict) and key in road:
                return torch.as_tensor(road[key])
    return None


def _build_tracks_provider(ckpt):
    """Timestamp-matched GT-track lookup from ckpt viz_4d (no model build).

    Returns callable(batch) -> active list [{"id","class","pose","size"}],
    or None when viz_4d tracks are unavailable.
    """
    viz = ckpt.get("viz_4d") or {}
    tracks = viz.get("tracks") or {}
    shared_ts = viz.get("tracks_camera_timestamps_us")
    if not tracks or shared_ts is None:
        return None
    ts = torch.as_tensor(shared_ts).flatten().long()

    def provider(batch):
        t = int(getattr(batch, "timestamp_us", -1))
        if t < 0:
            return []
        idx = int(torch.argmin((ts - t).abs()))
        active = []
        for tid, tr in tracks.items():
            poses = torch.as_tensor(tr.get("poses"))
            # per-frame validity flag: ckpt viz_4d uses "frame_info"
            # (bool [T]); older dumps may use "active".
            act = tr.get("frame_info", tr.get("active"))
            if poses is None or poses.dim() != 3 or idx >= poses.shape[0]:
                continue
            if act is not None and not bool(torch.as_tensor(act)[idx]):
                continue
            size = torch.as_tensor(
                tr.get("size") or tr.get("dims") or tr.get("cuboid_dims")
            )
            active.append({
                "id": tid,
                "class": str(tr.get("class", "unknown")),
                "pose": poses[idx].float(),
                "size": size.flatten()[:3].float(),
            })
        return active

    return provider


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="project ckpt — conf/test split/tracks/road source")
    ap.add_argument("--path", default="")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--frames-map", default="")
    ap.add_argument("--mode", default="interpolated",
                    choices=["interpolated", "lateral_1m", "lateral_2m",
                             "lateral_3m", "lateral_6m"])
    ap.add_argument("--lane", action="store_true", help="load lane masks")
    ap.add_argument("--nta-iou", action="store_true")
    ap.add_argument("--kid", action="store_true", help="FID/KID vs GT dist")
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--ground-z", type=float, default=None,
                    help="constant ground plane fallback (no road layer)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import threedgrut.datasets as datasets
    from omegaconf import OmegaConf
    from threedgrut.datasets.utils import configure_dataloader_for_platform

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    conf = ckpt["config"]
    if args.path:
        conf.path = args.path
    if args.lane:
        OmegaConf.set_struct(conf, False)
        conf["dataset"]["load_lane_masks"] = True

    dataset = datasets.make_test(name=conf.dataset.type, config=conf)
    dataloader = torch.utils.data.DataLoader(
        dataset, **configure_dataloader_for_platform(
            {"num_workers": 0, "batch_size": 1, "shuffle": False,
             "collate_fn": None}
        ),
    )

    def batches():
        for b in dataloader:
            yield dataset.get_gpu_batch_with_intrinsics(b)

    lpips_fn = None
    if args.lpips:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        lpips_fn = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True,
        ).to("cuda")

    detector = tracks_provider = None
    if args.nta_iou:
        from threedgrut.model.vehicle_detector import get_vehicle_detector
        detector = get_vehicle_detector(device="cuda")
        tracks_provider = _build_tracks_provider(ckpt)
        if tracks_provider is None:
            print("[eval_frames_dir] no viz_4d tracks in ckpt → NTA skipped")
            detector = None

    height_field = None
    if args.mode != "interpolated":
        road_pos = _road_positions_from_ckpt(ckpt)
        if road_pos is not None and road_pos.numel() > 0:
            from threedgrut.model.road_region import build_road_height_field
            height_field = build_road_height_field(
                road_pos.float().to("cuda"), cell_size=1.0,
            )
            print(f"[eval_frames_dir] height field from ckpt road layer: "
                  f"{int(height_field['occupied'].sum())} cells")
        elif args.ground_z is None:
            raise SystemExit(
                "novel mode needs a road height field (layered ckpt) or "
                "--ground-z"
            )

    frames_map = None
    if args.frames_map:
        with open(args.frames_map) as f:
            frames_map = json.load(f)

    out = evaluate_frames(
        batches(), frames_dir=args.frames_dir, frames_map=frames_map,
        mode=args.mode, lpips_fn=lpips_fn, detector=detector,
        tracks_provider=tracks_provider, height_field=height_field,
        ground_z=args.ground_z, fid_kid=args.kid,
    )
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
