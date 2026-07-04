"""E2.1: batch-fix baseline novel frames through the Harmonizer IPC server.

Reads <raw_dir>/<mode>/frames_map.json + frames, sends each (H,W,3) RGB to the
harmonizer_server (nontemporal, socket length-prefixed protocol), applies
Reinhard color_transfer(fixed -> raw) to match nre DifixModel behaviour, writes
fixed frames to <fixed_dir>/<mode>/ with an identical frames_map.json.
"""

import argparse
import io
import json
import os
import shutil
import socket
import struct

import torch
import torchvision

try:
    import kornia
except ImportError:
    kornia = None


def _recvall(s, n):
    b = b""
    while len(b) < n:
        d = s.recv(n - len(b))
        if not d:
            raise EOFError("harmonizer server closed")
        b += d
    return b


def harmonizer_fix_frame(img_hw3: torch.Tensor, host: str, port: int) -> torch.Tensor:
    """img_hw3: (H,W,3) float[0,1] CPU -> repaired (H,W,3). Protocol mirrors
    harmonizer_server.py: {input:(h*w,3), img_size:[h,w]} length-prefixed."""
    H, W, _ = img_hw3.shape
    inp = img_hw3.reshape(H * W, 3).contiguous()
    bio = io.BytesIO()
    torch.save({"input": inp.cpu(), "img_size": (int(H), int(W))}, bio)
    p = bio.getvalue()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.sendall(struct.pack(">Q", len(p)) + p)
    n = struct.unpack(">Q", _recvall(s, 8))[0]
    out = torch.load(io.BytesIO(_recvall(s, n)), weights_only=False)
    s.close()
    return out.reshape(H, W, 3).float()


def color_transfer(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Reinhard et al. — move source(fixed) color stats onto target(raw). (H,W,3)."""
    if kornia is None:
        return source
    src = kornia.color.rgb_to_lab(source.permute(2, 0, 1)).permute(1, 2, 0)
    tgt = kornia.color.rgb_to_lab(target.permute(2, 0, 1)).permute(1, 2, 0)
    sm = src.reshape(-1, 1, 3).mean(0, keepdim=True)
    ss = src.reshape(-1, 1, 3).std(0, keepdim=True)
    tm = tgt.reshape(-1, 1, 3).mean(0, keepdim=True)
    ts = tgt.reshape(-1, 1, 3).std(0, keepdim=True)
    lab = (src - sm) * (ts / (ss + 1e-8)) + tm
    lab = lab.clamp(-128, 127)
    return kornia.color.lab_to_rgb(lab.permute(2, 0, 1)).permute(1, 2, 0).clamp(0, 1)


def _load_img(path):
    return torchvision.io.read_image(path).float().div(255.0)[:3].permute(1, 2, 0)


def fix_mode(raw_dir, fixed_dir, mode, host, port, do_ct=True):
    src_root = os.path.join(raw_dir, mode)
    dst_root = os.path.join(fixed_dir, mode)
    with open(os.path.join(src_root, "frames_map.json")) as f:
        fmap = json.load(f)
    os.makedirs(dst_root, exist_ok=True)
    for i, (key, rel) in enumerate(sorted(fmap.items())):
        raw = _load_img(os.path.join(src_root, rel))
        fixed = harmonizer_fix_frame(raw, host, port)
        if do_ct:
            fixed = color_transfer(fixed, raw)
        dst = os.path.join(dst_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        torchvision.utils.save_image(fixed.permute(2, 0, 1).clamp(0, 1), dst)
        if (i + 1) % 50 == 0:
            print(f"[{mode}] {i + 1}/{len(fmap)}", flush=True)
    shutil.copy(os.path.join(src_root, "frames_map.json"), os.path.join(dst_root, "frames_map.json"))
    print(f"[{mode}] done {len(fmap)} frames -> {dst_root}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="<.../ours_N/novel_view>")
    ap.add_argument("--fixed-dir", required=True)
    ap.add_argument("--modes", nargs="+", default=["lateral_3m", "lateral_6m"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=59489)
    ap.add_argument("--no-color-transfer", action="store_true")
    a = ap.parse_args()
    for m in a.modes:
        fix_mode(a.raw_dir, a.fixed_dir, m, a.host, a.port, do_ct=not a.no_color_transfer)


if __name__ == "__main__":
    main()
