"""V3 Stage A — per-track pose drift analysis: learned quat/trans vs frozen GT.

Reads ckpt["model"]["layered_track_state"] and compares:
  - learned trans: ckpt._track_trans_<tid>   shape [F, 3]
  - learned quat:  ckpt._track_quat_<tid>    shape [F, 4]  (wxyz, may be non-unit)
  - GT pose:       ckpt._track_pose_gt_<tid> shape [F, 4, 4]  (SE(3) reference)
  - active mask:   ckpt._track_active_<tid>  shape [F]

For each active frame of each track:
  - translation_delta_norm = ||trans_learned - trans_gt||_2 (meters)
  - rotation_delta_deg     = 2 * arccos(|q_learned · q_gt|) * 180/π
"""
import sys, math, re, json
from collections import defaultdict
import torch

# wxyz quat conjugate-multiply for geodesic distance:
def quat_geodesic_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """q1, q2: [..., 4] wxyz unit quaternions → angle in degrees [0, 180]."""
    q1n = q1 / q1.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    q2n = q2 / q2.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    dot = (q1n * q2n).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot) * 180.0 / math.pi

def rotmat_to_quat_wxyz(R: torch.Tensor) -> torch.Tensor:
    """Stable rotmat → wxyz quat via Shepperd. Inline copy from layered_model."""
    R00, R01, R02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    R10, R11, R12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    R20, R21, R22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    trace = R00 + R11 + R22
    s_a = torch.sqrt(torch.clamp(trace + 1.0, min=1e-12)) * 2.0
    w_a = 0.25 * s_a; x_a = (R21 - R12) / s_a; y_a = (R02 - R20) / s_a; z_a = (R10 - R01) / s_a
    s_b = torch.sqrt(torch.clamp(1.0 + R00 - R11 - R22, min=1e-12)) * 2.0
    w_b = (R21 - R12) / s_b; x_b = 0.25 * s_b; y_b = (R01 + R10) / s_b; z_b = (R02 + R20) / s_b
    s_c = torch.sqrt(torch.clamp(1.0 + R11 - R00 - R22, min=1e-12)) * 2.0
    w_c = (R02 - R20) / s_c; x_c = (R01 + R10) / s_c; y_c = 0.25 * s_c; z_c = (R12 + R21) / s_c
    s_d = torch.sqrt(torch.clamp(1.0 + R22 - R00 - R11, min=1e-12)) * 2.0
    w_d = (R10 - R01) / s_d; x_d = (R02 + R20) / s_d; y_d = (R12 + R21) / s_d; z_d = 0.25 * s_d
    cond_a = trace > 0
    cond_b = (~cond_a) & (R00 >= R11) & (R00 >= R22)
    cond_c = (~cond_a) & (~cond_b) & (R11 >= R22)
    w = torch.where(cond_a, w_a, torch.where(cond_b, w_b, torch.where(cond_c, w_c, w_d)))
    x = torch.where(cond_a, x_a, torch.where(cond_b, x_b, torch.where(cond_c, x_c, x_d)))
    y = torch.where(cond_a, y_a, torch.where(cond_b, y_b, torch.where(cond_c, y_c, y_d)))
    z = torch.where(cond_a, z_a, torch.where(cond_b, z_b, torch.where(cond_c, z_c, z_d)))
    return torch.stack([w, x, y, z], dim=-1)


def percentile(t: torch.Tensor, q: float) -> float:
    if t.numel() == 0: return float('nan')
    return float(torch.quantile(t, q).item())


def main(ckpt_path: str, plot: bool = False, out_dir: str | None = None) -> None:
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    state = ckpt["model"].get("layered_track_state") or ckpt.get("layered_track_state")
    if state is None:
        print(f"ERROR: ckpt has no layered_track_state — Stage A ckpt persistence may be broken")
        sys.exit(2)

    tid_pat = re.compile(r"^_track_quat_(.+)$")
    tids = sorted([m.group(1) for k in state if (m := tid_pat.match(k))])
    if not tids:
        print(f"ERROR: no _track_quat_<tid> keys — this ckpt is from buffer mode, not learnable_pose")
        sys.exit(2)

    print(f"=== Pose drift analysis ({len(tids)} tracks) ===")
    print(f"ckpt: {ckpt_path}")
    if "learnable_pose_state" in ckpt:
        lps = ckpt["learnable_pose_state"]
        print(f"freeze_until_iter: {lps['freeze_until_iter']}")
        print(f"optimizer.state populated: {len(lps['optimizer']['state'])} (>0 means freeze ended & step ran)")
    print()

    all_trans_delta = []
    all_rot_delta_deg = []
    all_gt_xy = []         # [N, 2] BEV positions of active samples for --plot scatter
    per_track_summary = []

    for tid in tids:
        q_learned = state[f"_track_quat_{tid}"]                # [F, 4] wxyz
        t_learned = state[f"_track_trans_{tid}"]               # [F, 3]
        pose_gt   = state[f"_track_pose_gt_{tid}"]             # [F, 4, 4]
        active    = state[f"_track_active_{tid}"]              # [F] bool

        F = q_learned.shape[0]
        assert t_learned.shape == (F, 3) and pose_gt.shape == (F, 4, 4) and active.shape == (F,)
        active_mask = active.bool()
        n_active = int(active_mask.sum().item())
        if n_active == 0:
            continue

        t_gt = pose_gt[:, :3, 3]                                # [F, 3]
        R_gt = pose_gt[:, :3, :3]                               # [F, 3, 3]
        q_gt = rotmat_to_quat_wxyz(R_gt)                        # [F, 4]

        trans_delta = (t_learned - t_gt).norm(dim=-1)           # [F]
        rot_delta_deg = quat_geodesic_deg(q_learned, q_gt)      # [F]

        # restrict to active frames
        trans_delta_act = trans_delta[active_mask]
        rot_delta_act = rot_delta_deg[active_mask]

        per_track_summary.append({
            "tid": tid,
            "F_active": n_active,
            "trans_delta_max":    float(trans_delta_act.max().item()),
            "trans_delta_median": float(trans_delta_act.median().item()),
            "rot_delta_max_deg":  float(rot_delta_act.max().item()),
            "rot_delta_median_deg": float(rot_delta_act.median().item()),
        })
        all_trans_delta.append(trans_delta_act)
        all_rot_delta_deg.append(rot_delta_act)
        all_gt_xy.append(t_gt[active_mask, :2])

    flat_trans = torch.cat(all_trans_delta)
    flat_rot = torch.cat(all_rot_delta_deg)

    print(f"=== Aggregate over {flat_trans.numel()} active (track, frame) pairs ===")
    print(f"Translation delta (m):")
    print(f"  min/median/p90/p99/max = "
          f"{flat_trans.min().item():.4f} / {percentile(flat_trans, 0.5):.4f} / "
          f"{percentile(flat_trans, 0.9):.4f} / {percentile(flat_trans, 0.99):.4f} / "
          f"{flat_trans.max().item():.4f}")
    print(f"  mean = {flat_trans.mean().item():.4f}")
    print(f"Rotation delta (deg):")
    print(f"  min/median/p90/p99/max = "
          f"{flat_rot.min().item():.4f} / {percentile(flat_rot, 0.5):.4f} / "
          f"{percentile(flat_rot, 0.9):.4f} / {percentile(flat_rot, 0.99):.4f} / "
          f"{flat_rot.max().item():.4f}")
    print(f"  mean = {flat_rot.mean().item():.4f}")
    print()
    # Top 5 most-drifted tracks
    by_trans = sorted(per_track_summary, key=lambda d: -d["trans_delta_max"])[:5]
    print(f"=== Top 5 tracks by max trans drift ===")
    print(f"  {'tid':>10}  {'F_act':>6}  {'tΔmax':>8}  {'tΔmed':>8}  {'rΔmax°':>8}  {'rΔmed°':>8}")
    for d in by_trans:
        print(f"  {d['tid']:>10}  {d['F_active']:>6}  {d['trans_delta_max']:>8.3f}  "
              f"{d['trans_delta_median']:>8.3f}  {d['rot_delta_max_deg']:>8.3f}  "
              f"{d['rot_delta_median_deg']:>8.3f}")

    if plot:
        _emit_plots(
            ckpt_path=ckpt_path,
            out_dir=out_dir,
            flat_trans=flat_trans,
            flat_rot=flat_rot,
            flat_xy=torch.cat(all_gt_xy),
            n_tracks=len(tids),
        )


def _emit_plots(ckpt_path, out_dir, flat_trans, flat_rot, flat_xy, n_tracks):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = out_dir or os.path.dirname(os.path.abspath(ckpt_path))
    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.basename(os.path.dirname(os.path.abspath(ckpt_path)))  # e.g. ours_30000

    t_np = flat_trans.numpy()
    r_np = flat_rot.numpy()
    xy = flat_xy.numpy()

    # 1) Translation drift histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(t_np, bins=80, color="#3a86ff", alpha=0.85, edgecolor="black", linewidth=0.3)
    med, p99 = float(np.median(t_np)), float(np.quantile(t_np, 0.99))
    ax.axvline(med, color="orange", linestyle="--", label=f"median={med:.3f} m")
    ax.axvline(p99, color="red", linestyle="--", label=f"p99={p99:.3f} m")
    ax.set_xlabel("‖trans_learned - trans_gt‖₂ (m)")
    ax.set_ylabel("count (track, frame) pairs")
    ax.set_title(f"Per-frame translation drift — {tag}  (N={t_np.size}, {n_tracks} tracks)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "pose_drift_trans_hist.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  wrote {p}")

    # 2) Rotation drift histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(r_np, bins=80, color="#06d6a0", alpha=0.85, edgecolor="black", linewidth=0.3)
    med, p99 = float(np.median(r_np)), float(np.quantile(r_np, 0.99))
    ax.axvline(med, color="orange", linestyle="--", label=f"median={med:.3f}°")
    ax.axvline(p99, color="red", linestyle="--", label=f"p99={p99:.3f}°")
    ax.set_xlabel("geodesic(q_learned, q_gt) (deg)")
    ax.set_ylabel("count (track, frame) pairs")
    ax.set_title(f"Per-frame rotation drift — {tag}  (N={r_np.size}, {n_tracks} tracks)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "pose_drift_rot_hist.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  wrote {p}")

    # 3) BEV scatter: cuboid GT position colored by trans drift (clipped to p99)
    fig, ax = plt.subplots(figsize=(7, 6))
    vmax = max(float(np.quantile(t_np, 0.99)), 1e-6)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=t_np, s=4, alpha=0.7, cmap="viridis",
                    vmin=0.0, vmax=vmax)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title(f"BEV cuboid GT positions colored by trans drift — {tag}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    cb = fig.colorbar(sc, ax=ax, label=f"trans drift (m, vmax=p99={vmax:.3f})")
    fig.tight_layout()
    p = os.path.join(out_dir, "pose_drift_bev.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  wrote {p}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", help="path to ckpt_<iter>.pt with layered_track_state")
    ap.add_argument("--plot", action="store_true", help="emit PNG histograms + BEV scatter next to ckpt")
    ap.add_argument("--out-dir", default=None, help="output dir for PNGs (default: ckpt dir)")
    args = ap.parse_args()
    main(args.ckpt, plot=args.plot, out_dir=args.out_dir)
