"""Inspect V3 Stage A ckpt: confirm learnable_pose_state + _track_quat_<tid>/_track_trans_<tid>/_track_pose_gt_<tid> rideshare."""

import re
import sys

import torch

ckpt_path = sys.argv[1]
ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")

print("=== Top-level keys ===")
for k in sorted(ckpt.keys()):
    v = ckpt[k]
    if torch.is_tensor(v):
        print(f"  {k}: tensor{tuple(v.shape)} dtype={v.dtype}")
    elif isinstance(v, dict):
        print(f"  {k}: dict, sub-keys = {sorted(v.keys())}")
    else:
        print(f"  {k}: {type(v).__name__}")

print()
print("=== learnable_pose_state ===")
if "learnable_pose_state" in ckpt:
    lps = ckpt["learnable_pose_state"]
    print("  freeze_until_iter:", lps["freeze_until_iter"])
    opt_state = lps["optimizer"]
    print("  optimizer.param_groups:")
    for g in opt_state["param_groups"]:
        print(f"    name={g.get('name','?')} lr={g['lr']} num_params={len(g['params'])}")
    print("  optimizer.state entries:", len(opt_state["state"]))
else:
    print("  MISSING")

print()
print("=== Track params in model state_dict ===")


def find_track(obj, prefix=""):
    """Walk and emit (key, tensor) for keys containing 'track'."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from find_track(v, f"{prefix}.{k}" if prefix else k)
    elif torch.is_tensor(obj):
        if re.search(r"track", prefix, re.IGNORECASE):
            yield prefix, obj


count_quat = count_trans = count_pose_gt = count_active = 0
for key, t in find_track(ckpt):
    if "_track_quat_" in key:
        count_quat += 1
    elif "_track_trans_" in key:
        count_trans += 1
    elif "_track_pose_gt_" in key:
        count_pose_gt += 1
    elif "_track_active_" in key:
        count_active += 1
    if count_quat + count_trans + count_pose_gt + count_active <= 12:
        print(f"  {key}: {tuple(t.shape)} dtype={t.dtype} requires_grad={getattr(t,'requires_grad',False)}")

print()
print(f"=== Totals ===")
print(f"  _track_quat_*:    {count_quat}")
print(f"  _track_trans_*:   {count_trans}")
print(f"  _track_pose_gt_*: {count_pose_gt}")
print(f"  _track_active_*:  {count_active}")
