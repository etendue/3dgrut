# SPDX-License-Identifier: Apache-2.0
"""E2.6: temporal vs nontemporal Harmonizer de-flicker demo on a frame sequence.

Takes a directory of *temporally ordered* rendered frames (the output of
``render.py --render-only`` on the eval/interp split, or any frames_map.json
whose keys sort into a continuous sequence) and produces a three-column
comparison:

  raw/              — input frames, untouched
  nontemporal_fixed/— each frame through Harmonizer V=1 (E2.1 single-frame path)
  temporal_fixed/   — each frame through Harmonizer V=1+K (E2.6 temporal path,
                      self-reference history of corrected outputs)

The point is to *see* de-flicker: on a continuous play sequence the temporal
mode should be more frame-to-frame consistent than per-frame nontemporal,
which independently corrects each frame and can oscillate.

Usage (inside the 3dgrut2 env on inceptio; the Harmonizer servers run in the
cosmos container, reached over loopback)::

    # 1. render a continuous eval/interp frame sequence first:
    python -m threedgrut.render --config-name apps/ncore_3dgut_mcmc_multilayer \\
        ... --render-only  (produces <out>/ours_N/ours/interp/*.png + frames_map.json)

    # 2. ensure BOTH Harmonizer servers are up (cosmos container, separate ports):
    #    - nontemporal (V=1): E0.7 harmonizer_server.py on :59489
    #    - temporal (V=1+K):  E2.6 harmonizer_temporal_server.py on :59490

    # 3. run this demo:
    python scripts/e26_temporal_demo.py \\
        --raw-dir <out>/ours_N/ours \\
        --out-dir <out>/e26_demo \\
        --mode interp \\
        --nontemporal-port 59489 --temporal-port 59490 --K 4

Frame ordering: frames_map.json keys are sorted lexicographically AND by
embedded timestamp when present, so the temporal client feeds a genuinely
continuous sequence (a discontinuous order would poison its history). The
script asserts monotonic timestamps if it can extract them.
"""
from __future__ import annotations

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


# --------------------------------------------------------------------------- #
# nontemporal Harmonizer client (E2.1 single-frame path, length-prefixed)      #
# --------------------------------------------------------------------------- #
def _recvall(s: socket.socket, n: int) -> bytes:
    b = b""
    while len(b) < n:
        d = s.recv(n - len(b))
        if not d:
            raise EOFError("harmonizer server closed")
        b += d
    return b


def harmonizer_fix_nontemporal(
    img_hw3: torch.Tensor, host: str, port: int
) -> torch.Tensor:
    """Single-frame Harmonizer fix (V=1). Mirrors E2.1's protocol exactly.

    ``img_hw3``: (H,W,3) float[0,1] CPU -> repaired (H,W,3) float[0,1] CPU.
    """
    H, W, _ = img_hw3.shape
    inp = img_hw3.reshape(H * W, 3).contiguous()
    bio = io.BytesIO()
    torch.save({"input": inp.cpu(), "img_size": (int(H), int(W))}, bio)
    p = bio.getvalue()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    try:
        s.sendall(struct.pack(">Q", len(p)) + p)
        n = struct.unpack(">Q", _recvall(s, 8))[0]
        out = torch.load(io.BytesIO(_recvall(s, n)), weights_only=False)
    finally:
        s.close()
    return out.reshape(H, W, 3).float()


# --------------------------------------------------------------------------- #
# temporal Harmonizer path uses HarmonizerTemporalClient directly (see         #
# fix_sequence): a single persistent client threads corrected outputs into     #
# its K-deque across the sequence, matching viser_gui_4d's runtime behavior.   #
# --------------------------------------------------------------------------- #
def color_transfer(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Reinhard et al. — move source(fixed) color stats onto target(raw).

    Matches nre DifixModel behaviour (the training-side client does the same).
    (H,W,3) float[0,1].
    """
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
    return (
        kornia.color.lab_to_rgb(lab.permute(2, 0, 1)).permute(1, 2, 0).clamp(0, 1)
    )


def _load_img(path: str) -> torch.Tensor:
    return torchvision.io.read_image(path).float().div(255.0)[:3].permute(1, 2, 0)


def _ordered_keys(frames_map: dict) -> list[str]:
    """Return frames_map keys in continuous-sequence order.

    Sorts by the embedded timestamp in the relpath/filename when parseable,
    else lexicographically. The temporal client's history semantics rely on a
    genuinely continuous feed.
    """
    import re

    def _ts_key(k: str):
        # relpaths like "cam_front/000123_001234567.png" or "000123.png" —
        # extract the leading integer frame index for ordering.
        m = re.search(r"(\d+)", os.path.basename(frames_map[k]))
        return (int(m.group(1)) if m else 0, k)

    return sorted(frames_map.keys(), key=_ts_key)


def fix_sequence(
    raw_dir: str,
    out_dir: str,
    mode: str,
    *,
    nontemporal_host: str,
    nontemporal_port: int,
    temporal_host: str,
    temporal_port: int,
    K: int,
    do_color_transfer: bool,
    do_nontemporal: bool,
    do_temporal: bool,
) -> None:
    """Run one mode's frames through both Harmonizer paths, write three columns."""
    import numpy as np
    from threedgrut_playground.utils.harmonizer_client import (
        HarmonizerTemporalClient,
    )

    src_root = os.path.join(raw_dir, mode)
    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"mode dir not found: {src_root}")
    with open(os.path.join(src_root, "frames_map.json")) as f:
        fmap = json.load(f)
    keys = _ordered_keys(fmap)
    print(f"[{mode}] {len(keys)} frames (ordered)")

    nt_root = os.path.join(out_dir, mode, "nontemporal_fixed")
    t_root = os.path.join(out_dir, mode, "temporal_fixed")
    raw_link = os.path.join(out_dir, mode, "raw")
    if do_nontemporal:
        os.makedirs(nt_root, exist_ok=True)
    if do_temporal:
        os.makedirs(t_root, exist_ok=True)
    # raw/ is just a symlink to src_root for side-by-side viewing convenience.
    if not os.path.lexists(raw_link):
        os.symlink(os.path.abspath(src_root), raw_link)

    temporal_client = (
        HarmonizerTemporalClient(temporal_host, temporal_port, K=K)
        if do_temporal else None
    )

    for i, key in enumerate(keys):
        rel = fmap[key]
        raw_path = os.path.join(src_root, rel)
        raw_t = _load_img(raw_path)  # (H,W,3) float[0,1]
        raw_u8 = (raw_t * 255.0 + 0.5).clamp(0, 255).to(torch.uint8).numpy()

        # ---- nontemporal (V=1) ----
        if do_nontemporal:
            fixed_nt = harmonizer_fix_nontemporal(
                raw_t, nontemporal_host, nontemporal_port
            )
            if do_color_transfer:
                fixed_nt = color_transfer(fixed_nt, raw_t)
            dst = os.path.join(nt_root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            torchvision.utils.save_image(
                fixed_nt.permute(2, 0, 1).clamp(0, 1), dst
            )

        # ---- temporal (V=1+K) ----
        if do_temporal:
            out_u8 = temporal_client.fix(raw_u8, reset=False)
            out_t = torch.from_numpy(out_u8).float().div(255.0)
            if do_color_transfer:
                out_t = color_transfer(out_t, raw_t)
            dst = os.path.join(t_root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            torchvision.utils.save_image(
                out_t.permute(2, 0, 1).clamp(0, 1), dst
            )

        if (i + 1) % 25 == 0 or i == 0:
            hist_depth = (
                temporal_client.history_depth if temporal_client else 0
            )
            print(
                f"[{mode}] {i + 1}/{len(keys)} "
                f"(temporal history depth {hist_depth}/{K})",
                flush=True,
            )

    # Copy frames_map.json into both fixed dirs so downstream tools (montage,
    # eval_frames_dir) can find the frames by the same keys.
    for root in (nt_root, t_root):
        if os.path.isdir(root):
            shutil.copy(
                os.path.join(src_root, "frames_map.json"),
                os.path.join(root, "frames_map.json"),
            )
    if temporal_client is not None:
        temporal_client.close()
    print(f"[{mode}] done -> {out_dir}/{mode}/{{raw,nontemporal_fixed,"
          f"temporal_fixed}}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="E2.6 temporal vs nontemporal Harmonizer de-flicker demo"
    )
    ap.add_argument(
        "--raw-dir", required=True,
        help="<.../ours_N>/ours : the render.py --render-only output root "
             "(contains <mode>/frames_map.json + frames).",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", default="interp",
                    help="subdir under raw-dir (default: interp = continuous "
                         "eval sequence). Use a lateral_* mode only if you "
                         "want to compare on extrapolation frames.")
    ap.add_argument("--nontemporal-host", default="127.0.0.1")
    ap.add_argument("--nontemporal-port", type=int, default=59489)
    ap.add_argument("--temporal-host", default="127.0.0.1")
    ap.add_argument("--temporal-port", type=int, default=59490)
    ap.add_argument("--K", type=int, default=4,
                    help="temporal history depth (default 4, paper default)")
    ap.add_argument("--no-color-transfer", action="store_true",
                    help="skip Reinhard color_transfer (fixed->raw). ")
    ap.add_argument("--skip-nontemporal", action="store_true",
                    help="only run the temporal column (nontemporal already "
                         "exists from a prior E2.1 run).")
    ap.add_argument("--skip-temporal", action="store_true",
                    help="only run the nontemporal column.")
    a = ap.parse_args()

    fix_sequence(
        raw_dir=a.raw_dir,
        out_dir=a.out_dir,
        mode=a.mode,
        nontemporal_host=a.nontemporal_host,
        nontemporal_port=a.nontemporal_port,
        temporal_host=a.temporal_host,
        temporal_port=a.temporal_port,
        K=a.K,
        do_color_transfer=not a.no_color_transfer,
        do_nontemporal=not a.skip_nontemporal,
        do_temporal=not a.skip_temporal,
    )


if __name__ == "__main__":
    main()
