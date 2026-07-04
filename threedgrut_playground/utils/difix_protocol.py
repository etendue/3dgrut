# SPDX-License-Identifier: Apache-2.0
"""Tiny localhost wire protocol for shipping frames to an out-of-process DiFix.

The viser viewer (``3dgrut`` conda env) and the DiFix inference server (NVIDIA
cosmos Docker env) cannot share a Python process — their dependency stacks are
incompatible (see ``third_party/Fixer/INSTALL.md``). So the viewer renders a
frame, sends it over a TCP socket to the server, and gets the corrected frame
back. This module is the shared, GPU-free, dependency-light codec used by both
ends (``difix_client.py`` and ``difix_server.py``).

Frame format on the wire::

    | 4s magic | uint32 H | uint32 W | uint32 C |  H*W*C uint8 pixels |
    |<-------------- 16-byte header -------------->|<--- raw body --->|

Raw uint8 (not JPEG) is deliberate: it avoids a second lossy compression on top
of the Gaussian render *and* the encode/decode latency, and a ~2 MB frame over
loopback costs only a few ms. All integers are big-endian via ``struct``.
"""

from __future__ import annotations

import socket
import struct

import numpy as np

# Magic + (H, W, C) as big-endian uint32s. 4 + 4*3 = 16 bytes.
MAGIC: bytes = b"DFX1"
_HEADER = struct.Struct(">4sIII")
HEADER_SIZE: int = _HEADER.size  # 16


def pack_frame(img: np.ndarray) -> bytes:
    """Serialize an ``(H, W, 3)`` uint8 image to header + raw bytes.

    Args:
        img: ``(H, W, 3)`` ``uint8`` numpy array.

    Returns:
        ``HEADER_SIZE`` header bytes followed by ``H*W*3`` pixel bytes.

    Raises:
        ValueError: if ``img`` is not ``(H, W, 3)`` uint8.
    """
    if img.dtype != np.uint8:
        raise ValueError(f"pack_frame expects uint8, got {img.dtype}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"pack_frame expects (H,W,3), got {img.shape}")
    h, w, c = img.shape
    body = np.ascontiguousarray(img).tobytes()
    return _HEADER.pack(MAGIC, h, w, c) + body


def recvall(conn: socket.socket, n: int) -> bytes:
    """Receive exactly ``n`` bytes from ``conn``, looping over TCP fragments.

    Raises:
        ConnectionError: if the peer closes before ``n`` bytes arrive.
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f"socket closed mid-frame: got {len(buf)}/{n} bytes")
        buf.extend(chunk)
    return bytes(buf)


def read_frame(conn: socket.socket) -> np.ndarray:
    """Read one frame (header + body) from ``conn`` into an ``(H, W, 3)`` array.

    Raises:
        ValueError: if the header magic does not match (desync / corruption).
        ConnectionError: if the peer closes mid-frame.
    """
    head = recvall(conn, HEADER_SIZE)
    magic, h, w, c = _HEADER.unpack(head)
    if magic != MAGIC:
        raise ValueError(f"bad frame magic {magic!r} (expected {MAGIC!r})")
    body = recvall(conn, h * w * c)
    return np.frombuffer(body, dtype=np.uint8).reshape(h, w, c)
