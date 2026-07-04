# SPDX-License-Identifier: Apache-2.0
"""Out-of-process DiFix inference server (runs inside the cosmos Docker env).

The viser viewer lives in the ``3dgrut`` conda env and cannot import DiFix's
stack (``cosmos_predict2`` / ``transformer_engine`` / ``flash-attn``). So we run
DiFix here, inside the NVIDIA cosmos container, and expose it over a tiny
loopback TCP protocol (see ``utils/difix_protocol.py``). The viewer's
``DifixClient`` ships rendered frames in and gets corrected frames back.

Launch (inside the cosmos container; repo mounted at ``/work``, HF cache
mounted so the 5.2 GB weights are visible)::

    docker run --gpus all -p 8765:8765 \\
        -v <repo>:/work -v ~/.cache/huggingface:/root/.cache/huggingface \\
        nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2 \\
        python /work/threedgrut_playground/difix_server.py --port 8765

On startup it loads the 3.8 GB Fixer weights and runs one warm-up forward
(~11.5 s on a 4090) so the first real request from the viewer is already hot
(~80 ms). Then it serves frames until killed.
"""

from __future__ import annotations

import argparse
import socket
import threading
import time
from typing import Callable, Optional

import numpy as np

from threedgrut_playground.utils.difix_protocol import pack_frame, read_frame

# A transform maps an (H,W,3) uint8 frame to an (H,W,3) uint8 frame.
Transform = Callable[[np.ndarray], np.ndarray]


def make_difix_transform(difix, device: str = "cuda") -> Transform:
    """Wrap a tensor->tensor DiFix callable into a uint8-frame transform.

    ``difix`` takes and returns an ``(H,W,3)`` float tensor in ``[0,1]`` (the
    ``DifixPostProcessor.forward`` contract). This helper owns the
    numpy<->torch and uint8<->float[0,1] conversion on ``device`` so the wire
    protocol can stay raw-uint8.
    """
    import torch

    def transform(img: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.ascontiguousarray(img)).to(device)
        t = t.float().div_(255.0)  # (H,W,3) [0,1]
        y = difix(t)  # (H,W,3) [0,1]
        # +0.5 then truncate = round-half-up back to [0,255] uint8.
        y = (y.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
        return y.cpu().numpy()

    return transform


def _serve_one(
    conn: socket.socket,
    transform: Transform,
    lock: "threading.Lock | None" = None,
) -> None:
    """Read one frame from ``conn``, apply ``transform``, write it back.

    ``lock`` (when given) serializes the ``transform`` call so concurrent
    connections never run GPU forward simultaneously (Pix2Pix_Turbo + the CUDA
    context are not thread-safe). Socket I/O stays outside the lock.
    """
    img = read_frame(conn)
    if lock is not None:
        with lock:
            out = transform(img)
    else:
        out = transform(img)
    conn.sendall(pack_frame(out))


def serve(
    host: str = "0.0.0.0",
    port: int = 8765,
    transform: Optional[Transform] = None,
    *,
    stop_event=None,
    on_listening: Optional[Callable[[int], None]] = None,
    poll: float = 0.2,
    conn_timeout: float = 300.0,
) -> None:
    """Accept connections and serve frames until ``stop_event`` is set.

    Each connection is handled in its **own daemon thread** so a stalled or
    half-open client can never block other clients (e.g. the viewer
    reconnecting after a network blip). Within a connection many frames are
    served in a loop (persistent, matching ``DifixClient``). A per-frame socket
    timeout (``conn_timeout``) lets a wedged connection self-clear instead of
    pinning a thread forever. A single ``gpu_lock`` serializes the actual
    ``transform`` so concurrent connections don't run GPU forward at once.

    ``on_listening(actual_port)`` fires once the socket is bound — lets callers
    using ``port=0`` discover the chosen port (used by tests). ``poll`` bounds
    how often the accept loop checks ``stop_event``.
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
    print(f"[difix-server] listening on {host}:{actual_port}", flush=True)

    gpu_lock = threading.Lock()

    def _handle(conn: socket.socket) -> None:
        with conn:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # Bound each frame read so a half-open / stalled client self-clears
            # instead of pinning this thread forever.
            conn.settimeout(conn_timeout)
            try:
                while stop_event is None or not stop_event.is_set():
                    t0 = time.perf_counter()
                    _serve_one(conn, transform, lock=gpu_lock)
                    infer_ms = (time.perf_counter() - t0) * 1000.0
                    print(f"[difix-server] frame {infer_ms:.0f} ms", flush=True)
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


def _warmup(transform: Transform, h: int, w: int) -> None:
    """Run one dummy frame to trigger DiFix lazy-init + CUDA kernel compile."""
    t0 = time.perf_counter()
    transform(np.zeros((h, w, 3), dtype=np.uint8))
    print(f"[difix-server] warmup done ({time.perf_counter() - t0:.1f}s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="DiFix inference TCP server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timestep", type=int, default=250)
    parser.add_argument("--ckpt_path", default=None, help="pretrained_fixer.pkl path; default = HF cache.")
    parser.add_argument("--warmup_h", type=int, default=640)
    parser.add_argument("--warmup_w", type=int, default=1024)
    args = parser.parse_args()

    from threedgrut.correction.difix import DifixPostProcessor

    difix = DifixPostProcessor(
        enabled=True,
        ckpt_path=args.ckpt_path,
        timestep=args.timestep,
    )
    transform = make_difix_transform(difix, device="cuda")
    _warmup(transform, args.warmup_h, args.warmup_w)
    serve(args.host, args.port, transform)


if __name__ == "__main__":
    main()
