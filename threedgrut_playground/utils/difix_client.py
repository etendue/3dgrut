# SPDX-License-Identifier: Apache-2.0
"""Viewer-side client for the out-of-process DiFix server.

Lives in the ``3dgrut`` conda env (alongside ``viser_gui_4d``). Sends a rendered
``(H, W, 3)`` uint8 frame to the DiFix server over loopback TCP and returns the
corrected frame. Pure standard library + numpy — no torch, no DiFix deps.

Design rules:
  * **Never crash the viewer.** Any socket / protocol error makes ``fix()`` log
    once and return the *input* frame unchanged, so the interactive session
    degrades to raw rendering instead of dying.
  * **Persistent connection.** The socket is reused across frames (the server
    speaks many frames per connection); a broken pipe transparently reconnects
    on the next ``fix()``.
  * **Latency visibility.** ``last_rtt_ms`` exposes the round-trip time so the
    GUI can surface it next to the FPS counter.
"""

from __future__ import annotations

import socket
import time

import numpy as np

from .difix_protocol import pack_frame, read_frame


class DifixClient:
    """Loopback client that ships frames to a DiFix server and back.

    Args:
        host: server host (typically ``127.0.0.1``).
        port: server TCP port.
        timeout: socket connect/IO timeout in seconds.
    """

    def __init__(self, host: str, port: int, timeout: float = 5.0) -> None:
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.last_rtt_ms: float = 0.0
        self.healthy: bool = True
        self._sock: socket.socket | None = None
        self._warned: bool = False

    @classmethod
    def from_addr(cls, addr: str, timeout: float = 5.0) -> "DifixClient":
        """Build from a ``host:port`` string (e.g. ``"127.0.0.1:8765"``)."""
        host, port = addr.rsplit(":", 1)
        return cls(host, int(port), timeout=timeout)

    def _connect(self) -> socket.socket:
        if self._sock is None:
            s = socket.create_connection((self.host, self.port), timeout=self.timeout)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock = s
        return self._sock

    def _close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def close(self) -> None:
        """Close the underlying socket (idempotent)."""
        self._close()

    def fix(self, img: np.ndarray) -> np.ndarray:
        """Send ``img`` to the server and return the corrected frame.

        On any failure, logs once and returns ``img`` unchanged so the viewer
        keeps running (degraded to raw rendering).

        Args:
            img: ``(H, W, 3)`` uint8 frame.

        Returns:
            ``(H, W, 3)`` uint8 — corrected frame, or ``img`` on failure.
        """
        try:
            t0 = time.perf_counter()
            s = self._connect()
            s.sendall(pack_frame(img))
            out = read_frame(s)
            self.last_rtt_ms = (time.perf_counter() - t0) * 1000.0
            self.healthy = True
            self._warned = False
            return out
        except Exception as exc:  # noqa: BLE001 — never let the viewer die
            self._close()
            self.healthy = False
            if not self._warned:
                print(
                    f"[DiFix] client error → falling back to raw frame " f"({self.host}:{self.port}): {exc}",
                    flush=True,
                )
                self._warned = True
            return img
