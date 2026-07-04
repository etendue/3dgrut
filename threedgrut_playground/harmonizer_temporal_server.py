# SPDX-License-Identifier: Apache-2.0
"""Out-of-process temporal DiffusionHarmonizer inference server (E2.6).

Sister to ``difix_server.py`` but serves Harmonizer's **temporal mode**: each
request carries ``1 + K_in`` frames (the current frame plus ``K_in`` previously
corrected outputs), the model runs one 5D ``(1, C, V=1+K_in, H, W)`` forward
pass for de-flickering, and the server returns a single corrected frame for the
current timestep.

**History lives on the client** (see ``utils/harmonizer_protocol.py``): this
server is stateless — it forwards exactly the frames it receives. No
per-connection ring buffer, no seq bookkeeping. This is what makes the protocol
trivially testable on a Mac (an echo stand-in needs no model state).

Runs inside the ``harmonizer-cosmos-env`` Docker container (its dependency stack
— ``cosmos_predict2`` / ``transformer_engine`` / ``flash-attn`` / the
``pix2pix_turbo_harmonizer`` module — cannot be imported from the ``3dgrut``
env where the viewer lives).

Launch (inside the cosmos container; repo mounted at ``/work``, Harmonizer src
+ models + HF cache mounted so the 5 GB weights are visible)::

    docker run -d --name harmonizer_temporal_server --gpus all --net=host \\
        -e HARMONIZER_PORT=59490 \\
        -v <repo>:/work \\
        -v ~/repo/harmonizer/src:/work/harm_src:ro \\
        -v ~/repo/harmonizer/models:/work/harm_models:ro \\
        -v ~/.cache/huggingface:/root/.cache/huggingface \\
        harmonizer-cosmos-env:latest \\
        python /work/threedgrut_playground/harmonizer_temporal_server.py

The two operational gotchas from E0.7 (documented in
``docs/superpowers/specs/2026-06-12-e07-harmonizer-as-fixer-handoff.md`` §3) are
preserved:
  * ``os.chdir("/work/harm_src")`` before importing the model — Cosmos base
    weights resolve relative to cwd (``checkpoints/nvidia/...``).
  * Port configurable via ``HARMONIZER_PORT`` env; default ``59490`` (distinct
    from E0.7/E2.1's 59487/59489 to prevent cross-wiring).
"""

from __future__ import annotations

import argparse
import os
import socket
import threading
import time
from typing import Callable, Optional

import numpy as np

from threedgrut_playground.utils.difix_protocol import pack_frame
from threedgrut_playground.utils.harmonizer_protocol import read_temporal

# A temporal transform maps (curr, history) uint8 frames to one uint8 frame.
# history is oldest-first with len in 0..K_max.
TemporalTransform = Callable[[np.ndarray, "list[np.ndarray]"], np.ndarray]


def serve(
    host: str = "0.0.0.0",
    port: int = 59490,
    transform: Optional[TemporalTransform] = None,
    *,
    stop_event=None,
    on_listening: Optional[Callable[[int], None]] = None,
    poll: float = 0.2,
    conn_timeout: float = 300.0,
) -> None:
    """Accept connections and serve temporal requests until ``stop_event`` set.

    Mirrors ``difix_server.serve()``: each connection runs in its own daemon
    thread (a stalled client never blocks others), many requests are served in a
    loop per connection (persistent, matching ``HarmonizerTemporalClient``), a
    per-request socket timeout lets wedged connections self-clear, and a single
    ``gpu_lock`` serializes the GPU forward so concurrent connections do not run
    the model simultaneously (Pix2Pix_Turbo + CUDA context are not thread-safe).
    """
    if transform is None:
        raise ValueError("serve() requires a transform")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(16)
    srv.settimeout(poll)
    actual_port = srv.getsockname()[1]
    if on_listening is not None:
        on_listening(actual_port)
    print(f"[harm-temporal] listening on {host}:{actual_port}", flush=True)

    gpu_lock = threading.Lock()

    def _handle(conn: socket.socket) -> None:
        with conn:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(conn_timeout)
            try:
                while stop_event is None or not stop_event.is_set():
                    t0 = time.perf_counter()
                    curr, hist = read_temporal(conn)
                    if gpu_lock is not None:
                        with gpu_lock:
                            out = transform(curr, hist)
                    else:
                        out = transform(curr, hist)
                    infer_ms = (time.perf_counter() - t0) * 1000.0
                    print(
                        f"[harm-temporal] V={1 + len(hist)} {infer_ms:.0f} ms",
                        flush=True,
                    )
                    conn.sendall(pack_frame(out))
            except (ConnectionError, OSError):
                pass  # client disconnected / timed out — drop this connection

    try:
        while stop_event is None or not stop_event.is_set():
            try:
                conn, _addr = srv.accept()
            except (TimeoutError, socket.timeout):
                continue
            threading.Thread(target=_handle, args=(conn,), daemon=True).start()
    finally:
        srv.close()


def make_harmonizer_temporal_transform(
    model,
    device: str = "cuda",
    model_res: "tuple[int, int]" = (576, 1024),
) -> TemporalTransform:
    """Wrap a Harmonizer Pix2Pix_Turbo callable into a uint8 temporal transform.

    ``model`` is a ``Pix2Pix_Turbo`` whose ``forward`` expects 5D
    ``(B, C, V, H, W)`` and returns ``(B, C, V, H, W)`` (per-frame output). This
    helper owns: numpy<->torch, uint8<->[0,1] / [-1,1], resize to ``model_res``
    for the forward, resize back to the source frame size, V-stacking of
    curr+history, and selecting the V=0 (current-timestep) output channel.

    Mirrors the pre/post-process math of the E0.7 nontemporal server
    (``~/work/nurec_e0/e07/ipc/harmonizer_server.py``), generalized to V>1.
    Uses plain torch ops (no einops) so the module imports cleanly on a Mac for
    unit-testing the shape contract without the cosmos stack.
    """
    import torch
    import torchvision.transforms as T

    pre = T.Compose(
        [
            T.Resize(model_res, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.Lambda(lambda x: x * 2 - 1),
        ]
    )

    def transform(curr: np.ndarray, history: "list[np.ndarray]") -> np.ndarray:
        # Source frame size — identical for curr and all history (validated by
        # the protocol). Resize happens *after* V-stacking so one call covers
        # all frames. Return is resized back to this size.
        h, w, _ = curr.shape
        # Stack curr + history -> (V, H, W, 3) uint8, then to (V, 3, H, W) float.
        frames = [curr] + list(history)
        stack = np.stack(frames, axis=0)  # (V, H, W, 3) u8
        t = torch.from_numpy(np.ascontiguousarray(stack)).to(device)
        t = t.permute(0, 3, 1, 2).contiguous().float().div_(255.0)  # (V,3,H,W)
        # Resize + normalize to [-1,1] across all V at once.
        t = pre(t)  # (V,3,h',w') [-1,1]
        # Harmonizer forward wants (B, C, V, H, W); B=1. Pure-permute rearrange
        # of "b v c h w -> b c v h w": insert B, move V after C.
        x5 = t.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()  # (1,C,V,h',w')
        x5 = x5.to(torch.bfloat16)
        with torch.autocast(device, dtype=torch.bfloat16):
            y5 = model(x5)  # (1,C,V,h',w')
        # Take the current-timestep output (V index 0) — the other V outputs
        # are Harmonizer's predictions for the history frames, which we discard
        # (the client only feeds history as reference, not to re-correct them).
        # Rearrange "b c v h w -> b v c h w" then take v=0: permute C/V axes.
        y = y5.permute(0, 2, 1, 3, 4)[:, 0]  # (1,C,h',w')
        y = (y.float() + 1) / 2
        post = T.Compose(
            [
                T.Resize((h, w), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            ]
        )
        y = post(y)  # (1,C,H,W) [0,1]
        y = (y.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
        return y[0].permute(1, 2, 0).cpu().numpy()  # (H,W,3) u8

    return transform


def _warmup(transform: TemporalTransform, h: int, w: int, K: int = 4) -> None:
    """Run one dummy V=1+K request to trigger lazy-init + CUDA kernel compile."""
    t0 = time.perf_counter()
    curr = np.zeros((h, w, 3), dtype=np.uint8)
    hist = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(K)]
    transform(curr, hist)
    print(
        f"[harm-temporal] warmup done V={1 + K} " f"({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal DiffusionHarmonizer inference TCP server (E2.6)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("HARMONIZER_PORT", "59490")),
    )
    parser.add_argument(
        "--ckpt_path",
        default=os.environ.get("HARMONIZER_CKPT", "/work/harm_models/diffusion_harmonizer.pkl"),
    )
    parser.add_argument("--timestep", type=int, default=250)
    parser.add_argument("--warmup_h", type=int, default=576)
    parser.add_argument("--warmup_w", type=int, default=1024)
    parser.add_argument("--warmup_K", type=int, default=4)
    args = parser.parse_args()

    # Operational gotcha 1: Cosmos base weights resolve relative to cwd
    # (checkpoints/nvidia/...). chdir BEFORE importing the model module.
    harm_src = os.environ.get("HARMONIZER_SRC", "/work/harm_src")
    if os.path.isdir(harm_src):
        sys_path_added = harm_src
        import sys

        sys.path.insert(0, harm_src)
        os.chdir(harm_src)
    else:
        sys_path_added = None

    import torch
    from pix2pix_turbo_harmonizer import Pix2Pix_Turbo

    print("building Harmonizer Pix2Pix_Turbo...", flush=True)
    model = Pix2Pix_Turbo(
        pretrained_path=args.ckpt_path,
        timestep=args.timestep,
        train_full_unet=True,
        freeze_vae=False,
        vae_skip_connection=False,
        use_sched=True,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    model.set_eval()
    model = model.to(device="cuda", dtype=torch.bfloat16)
    print("model built", flush=True)

    transform = make_harmonizer_temporal_transform(model, device="cuda", model_res=(576, 1024))
    _warmup(transform, args.warmup_h, args.warmup_w, K=args.warmup_K)

    print(f"[harm-temporal] READY (sys.path added: {sys_path_added})", flush=True)
    serve(args.host, args.port, transform)


if __name__ == "__main__":
    main()
