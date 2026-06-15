# SPDX-License-Identifier: Apache-2.0
"""Viewer-side client for the temporal DiffusionHarmonizer server (E2.6).

Sister to ``difix_client.DifixClient`` but carries Harmonizer's temporal mode:
each ``fix()`` sends the current frame **plus the last ``K`` corrected outputs**
so the model can de-flicker a continuous play sequence. The history deque lives
on this client (see ``harmonizer_protocol`` for why); the server stays
stateless.

Design rules (mirrors ``DifixClient``):
  * **Never crash the viewer.** Any socket / protocol error makes ``fix()`` log
    once and return the *input* frame unchanged, so the interactive session
    degrades to raw rendering instead of dying.
  * **Persistent connection** with transparent reconnect.
  * **Latency visibility** via ``last_rtt_ms`` / ``healthy`` (GUI reuses these).

Temporal semantics:
  * ``fix(img, reset=False)`` — on ``reset=True`` the history deque is cleared
    *before* this frame is sent, so the model sees ``V=1`` (cold start). The
    viewer sets ``reset=True`` on any non-play timeline change (seek / scrub /
    loop-wrap) so a discontinuous jump does not feed stale history into the
    model.
  * After a successful round-trip, the *returned* (corrected) frame is appended
    to the deque — this is the Harmonizer convention (self-reference, not raw).
"""
from __future__ import annotations

import socket
import time
from collections import deque

import numpy as np

from .harmonizer_protocol import pack_temporal, read_frame


class HarmonizerTemporalClient:
    """Loopback client that ships ``1 + K`` frames to a Harmonizer server.

    Args:
        host: server host (typically ``127.0.0.1``).
        port: server TCP port.
        K: maximum history depth. The deque holds up to ``K`` prior corrected
            outputs; each request carries ``1 + min(len(deque), K)`` frames.
        timeout: socket connect/IO timeout in seconds.
    """

    def __init__(
        self, host: str, port: int, K: int = 4, timeout: float = 5.0
    ) -> None:
        self.host = host
        self.port = int(port)
        self.K = int(K)
        self.timeout = float(timeout)
        self.last_rtt_ms: float = 0.0
        self.healthy: bool = True
        self._sock: socket.socket | None = None
        self._warned: bool = False
        # Most-recent-last deque of corrected outputs (self-reference history).
        self._history: "deque[np.ndarray]" = deque(maxlen=self.K)

    @classmethod
    def from_addr(
        cls, addr: str, K: int = 4, timeout: float = 5.0
    ) -> "HarmonizerTemporalClient":
        """Build from a ``host:port`` string (e.g. ``"127.0.0.1:59490"``)."""
        host, port = addr.rsplit(":", 1)
        return cls(host, int(port), K=K, timeout=timeout)

    @property
    def history_depth(self) -> int:
        """Current number of history frames queued (0..K)."""
        return len(self._history)

    def reset(self) -> None:
        """Drop all history (next ``fix()`` sends V=1, cold start)."""
        self._history.clear()

    def _connect(self) -> socket.socket:
        if self._sock is None:
            s = socket.create_connection(
                (self.host, self.port), timeout=self.timeout
            )
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
        """Close the underlying socket and clear history (idempotent)."""
        self._close()
        self._history.clear()

    def fix(self, img: np.ndarray, reset: bool = False) -> np.ndarray:
        """Send ``img`` (+ history) to the server, return the corrected frame.

        On any failure, logs once and returns ``img`` unchanged so the viewer
        keeps running (degraded to raw rendering). The deque is left untouched
        on failure — a dropped frame does not poison history; the next call
        retries with the same (still-valid) history.

        Args:
            img: ``(H, W, 3)`` uint8 current frame.
            reset: if True, clear history before sending (cold start). Use on
                any discontinuous timeline change (seek / scrub / loop-wrap).

        Returns:
            ``(H, W, 3)`` uint8 — corrected frame, or ``img`` on failure.
        """
        if reset:
            self._history.clear()
        try:
            t0 = time.perf_counter()
            s = self._connect()
            s.sendall(pack_temporal(img, list(self._history), self.K))
            out = read_frame(s)
            self.last_rtt_ms = (time.perf_counter() - t0) * 1000.0
            self.healthy = True
            self._warned = False
            # Self-reference: the corrected output feeds the next request's
            # history (Harmonizer convention).
            self._history.append(out)
            return out
        except Exception as exc:  # noqa: BLE001 — never let the viewer die
            self._close()
            self.healthy = False
            if not self._warned:
                print(
                    f"[Harmonizer] client error → falling back to raw frame "
                    f"({self.host}:{self.port}): {exc}",
                    flush=True,
                )
                self._warned = True
            return img
