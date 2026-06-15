# SPDX-License-Identifier: Apache-2.0
"""Temporal wire protocol for the DiffusionHarmonizer out-of-process server.

E2.6 extends the single-frame DiFix post-process path (``difix_protocol.py``,
magic ``DFX1``) to Harmonizer's **temporal mode**: the model consumes a stack of
``V = 1 + K`` frames — the current frame plus the ``K`` previously *corrected*
outputs — and returns a single time-consistent frame. This is what gives
Harmonizer its de-flicker advantage over per-frame Fixer on continuous play.

Design (E2.6 plan, client-side deque):
  * **History lives on the client.** The server is stateless — it sees exactly
    the ``1 + K_in`` frames the client sends in one request and forwards them
    through the model as a single 5D ``(1, C, V, H, W)`` pass. No per-connection
    ring buffer, no seq-number bookkeeping on the server side.
  * **Cold start is implicit.** The client sends whatever history it has
    (``0..K`` frames), and ``K`` in the header is the *actual* count carried in
    the body — so the first frame of a sequence is ``K_in=0`` (V=1, nontemporal
    equivalent) and history grows frame by frame up to the deque's ``K``.
  * **Reset is just ``history.clear()``** on the client — no protocol signal.

Frame format on the wire::

    | 4s magic | uint32 H | uint32 W | uint32 C | uint32 K |
    |<-------------- 20-byte header -------------->|
    |        (1 + K) * H * W * C uint8 pixels     |
    |<---- curr frame, then history[0..K-1] ----->|

The server replies with exactly one frame (the corrected current frame) using
``difix_protocol``'s ``DFX1`` single-frame reply format — so the return path is
shared with the existing DiFix client plumbing.
"""
from __future__ import annotations

import socket
import struct

import numpy as np

# Magic + (H, W, C, K) as big-endian uint32s. 4 + 4*4 = 20 bytes.
# "HMN1" = HarMoNizer temporal v1. Distinct from DFX1 to prevent cross-wiring.
MAGIC: bytes = b"HMN1"
_HEADER = struct.Struct(">4sIIII")
HEADER_SIZE: int = _HEADER.size  # 20

# Reuse the single-frame DFX1 reply codec for the server's return path.
from .difix_protocol import (  # noqa: E402 — re-export for callers
    pack_frame,
    read_frame,
)


def pack_temporal(
    curr: np.ndarray, history: "list[np.ndarray]", K: int
) -> bytes:
    """Serialize a temporal request: ``curr`` + the last ``history`` frames.

    Args:
        curr: ``(H, W, 3)`` uint8 current frame.
        history: most-recent-last list of previously corrected frames. Only the
            last ``min(len(history), K)`` are sent; older entries are dropped.
        K: maximum history depth the client is configured for. The actual count
            ``K_in`` written to the header is ``min(len(history), K)`` so a cold
            start naturally carries fewer frames.

    Returns:
        ``HEADER_SIZE`` header bytes followed by ``(1 + K_in) * H * W * C``
        pixel bytes: curr first, then history[−K_in:].

    Raises:
        ValueError: if any frame is not ``(H, W, 3)`` uint8, or if history
            frames' shape/dtype mismatch ``curr``.
    """
    if curr.dtype != np.uint8:
        raise ValueError(f"pack_temporal expects uint8 curr, got {curr.dtype}")
    if curr.ndim != 3 or curr.shape[2] != 3:
        raise ValueError(f"pack_temporal expects (H,W,3), got {curr.shape}")

    k_in = max(0, min(len(history), K))
    tail = list(history[-k_in:]) if k_in else []
    for i, h in enumerate(tail):
        if h.shape != curr.shape or h.dtype != np.uint8:
            raise ValueError(
                f"history[-{i+1}] shape/dtype {h.shape}/{h.dtype} != curr "
                f"{curr.shape}/uint8"
            )

    h, w, c = curr.shape
    parts = [_HEADER.pack(MAGIC, h, w, c, k_in)]
    parts.append(np.ascontiguousarray(curr).tobytes())
    for hframe in tail:
        parts.append(np.ascontiguousarray(hframe).tobytes())
    return b"".join(parts)


def recvall(conn: socket.socket, n: int) -> bytes:
    """Receive exactly ``n`` bytes from ``conn``, looping over TCP fragments.

    Raises:
        ConnectionError: if the peer closes before ``n`` bytes arrive.
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(
                f"socket closed mid-request: got {len(buf)}/{n} bytes"
            )
        buf.extend(chunk)
    return bytes(buf)


def read_temporal(conn: socket.socket) -> "tuple[np.ndarray, list[np.ndarray]]":
    """Read one temporal request: header + (1+K_in) frames.

    Returns:
        ``(curr, history)`` where ``history`` is oldest-first and ``len ==
        K_in``. ``curr`` is always frame index 0 in the body.

    Raises:
        ValueError: if the header magic does not match (desync / corruption).
        ConnectionError: if the peer closes mid-request.
    """
    head = recvall(conn, HEADER_SIZE)
    magic, h, w, c, k_in = _HEADER.unpack(head)
    if magic != MAGIC:
        raise ValueError(f"bad temporal magic {magic!r} (expected {MAGIC!r})")
    body = recvall(conn, (1 + k_in) * h * w * c)
    frames = [
        np.frombuffer(
            body[i * h * w * c:(i + 1) * h * w * c], dtype=np.uint8
        ).reshape(h, w, c)
        for i in range(1 + k_in)
    ]
    return frames[0], frames[1:]
