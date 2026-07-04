# SPDX-License-Identifier: Apache-2.0
"""IPC tests for the DiFix viewer integration (protocol / client / server).

These exercise the localhost wire protocol that lets ``viser_gui_4d`` (3dgrut
env) ship rendered frames to an out-of-process DiFix server (cosmos Docker env)
and get the corrected frame back. Everything here is GPU-free: the protocol is
raw uint8 over a socket, and the DiFix model itself is replaced by identity /
echo stand-ins so the IPC plumbing can be verified on a Mac.

Run:
    pytest threedgrut/tests/test_difix_ipc.py -v
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time

import numpy as np
import pytest

from threedgrut_playground.difix_server import _serve_one, make_difix_transform, serve
from threedgrut_playground.utils.difix_client import DifixClient
from threedgrut_playground.utils.difix_protocol import pack_frame, read_frame, recvall


class _IdentityDifix:
    """Stand-in for DifixPostProcessor: returns its torch input unchanged."""

    def __call__(self, t):
        return t


def _rand_img(h: int = 7, w: int = 11) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


@contextlib.contextmanager
def _running_server(transform):
    """Spin up a localhost TCP server that applies ``transform`` per frame.

    Mirrors the persistent-connection protocol the real DiFix server speaks:
    one connection carries many frames in a loop. Yields the bound port.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    srv.settimeout(0.2)
    port = srv.getsockname()[1]
    stop = threading.Event()

    def _loop():
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except (TimeoutError, socket.timeout):
                continue
            with conn:
                try:
                    while True:
                        img = read_frame(conn)
                        conn.sendall(pack_frame(transform(img)))
                except (ConnectionError, OSError):
                    pass

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    try:
        yield port
    finally:
        stop.set()
        t.join(timeout=1.0)
        srv.close()


# --------------------------------------------------------------------------- #
#                              protocol layer                                  #
# --------------------------------------------------------------------------- #
def test_pack_read_roundtrip():
    """A frame packed and sent over a socket reads back byte-identical."""
    img = _rand_img()
    a, b = socket.socketpair()
    try:
        a.sendall(pack_frame(img))
        got = read_frame(b)
    finally:
        a.close()
        b.close()
    assert got.shape == img.shape
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, img)


def test_recvall_reassembles_fragments():
    """recvall must loop until all n bytes arrive (TCP may fragment)."""
    a, b = socket.socketpair()
    payload = bytes(range(256)) * 40  # 10240 bytes
    try:

        def _send():
            for i in range(0, len(payload), 100):
                a.sendall(payload[i : i + 100])

        t = threading.Thread(target=_send)
        t.start()
        got = recvall(b, len(payload))
        t.join()
    finally:
        a.close()
        b.close()
    assert got == payload


def test_read_frame_rejects_bad_magic():
    """A corrupt header (wrong magic) raises rather than mis-parsing."""
    a, b = socket.socketpair()
    try:
        a.sendall(b"XXXX" + b"\x00" * 12)  # bad magic + junk header
        with pytest.raises(ValueError, match="magic"):
            read_frame(b)
    finally:
        a.close()
        b.close()


def test_recvall_raises_on_closed_socket():
    """If the peer closes mid-stream, recvall raises ConnectionError."""
    a, b = socket.socketpair()
    a.sendall(b"123")
    a.close()
    with pytest.raises(ConnectionError):
        recvall(b, 10)  # only 3 bytes will ever arrive
    b.close()


def test_pack_frame_rejects_non_uint8_hwc():
    """pack_frame contract is (H,W,3) uint8 — guard wrong dtype / rank early."""
    with pytest.raises((ValueError, AssertionError)):
        pack_frame(np.zeros((4, 4, 3), dtype=np.float32))
    with pytest.raises((ValueError, AssertionError)):
        pack_frame(np.zeros((4, 4), dtype=np.uint8))


# --------------------------------------------------------------------------- #
#                              client layer                                    #
# --------------------------------------------------------------------------- #
def test_client_roundtrip_identity():
    """fix() over an identity server returns the same frame + records RTT."""
    img = _rand_img()
    with _running_server(lambda x: x) as port:
        c = DifixClient("127.0.0.1", port)
        try:
            out = c.fix(img)
            np.testing.assert_array_equal(out, img)
            assert c.healthy is True
            assert c.last_rtt_ms > 0.0
        finally:
            c.close()


def test_client_applies_server_transform():
    """The frame really crosses the wire: a non-identity server is observed."""
    img = _rand_img()
    with _running_server(lambda x: 255 - x) as port:  # color invert
        c = DifixClient("127.0.0.1", port)
        try:
            out = c.fix(img)
            np.testing.assert_array_equal(out, 255 - img)
        finally:
            c.close()


def test_client_reuses_connection_across_frames():
    """A persistent connection serves multiple fix() calls correctly."""
    img1, img2 = _rand_img(5, 9), _rand_img(8, 4)
    with _running_server(lambda x: x) as port:
        c = DifixClient("127.0.0.1", port)
        try:
            np.testing.assert_array_equal(c.fix(img1), img1)
            np.testing.assert_array_equal(c.fix(img2), img2)
        finally:
            c.close()


def test_client_falls_back_to_raw_on_refused_connection():
    """No server → fix() must return the input unchanged, never raise."""
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))  # bind but do NOT listen → connect refused
    port = holder.getsockname()[1]
    try:
        img = _rand_img()
        c = DifixClient("127.0.0.1", port, timeout=0.3)
        out = c.fix(img)
        np.testing.assert_array_equal(out, img)  # graceful raw fallback
        assert c.healthy is False
        c.close()
    finally:
        holder.close()


def test_client_recovers_after_server_comes_back():
    """After a failed call, a later call against a live server succeeds."""
    img = _rand_img()
    # First call: refused.
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    dead_port = holder.getsockname()[1]
    holder.close()  # free the port; nobody listens now
    c = DifixClient("127.0.0.1", dead_port, timeout=0.3)
    assert np.array_equal(c.fix(img), img)
    assert c.healthy is False
    # Now point the same client at a live server and retry.
    with _running_server(lambda x: x) as port:
        c.host, c.port = "127.0.0.1", port
        out = c.fix(img)
        np.testing.assert_array_equal(out, img)
        assert c.healthy is True
    c.close()


def test_client_from_addr_parses_host_port():
    c = DifixClient.from_addr("127.0.0.1:8765")
    assert c.host == "127.0.0.1"
    assert c.port == 8765


# --------------------------------------------------------------------------- #
#                              server layer                                    #
# --------------------------------------------------------------------------- #
def test_server_serve_one_applies_transform():
    """_serve_one reads one frame, applies transform, writes it back."""
    a, b = socket.socketpair()
    img = _rand_img()
    try:
        a.sendall(pack_frame(img))
        _serve_one(b, lambda x: 255 - x)  # invert
        got = read_frame(a)
        np.testing.assert_array_equal(got, 255 - img)
    finally:
        a.close()
        b.close()


def test_make_difix_transform_uint8_roundtrip_cpu():
    """Identity DiFix on CPU: uint8 -> float[0,1] -> uint8 must be lossless.

    Guards the conversion that wraps the real GPU model: the dtype/shape and
    [0,1]<->[0,255] math, without needing CUDA or the cosmos stack.
    """
    img = _rand_img()
    transform = make_difix_transform(_IdentityDifix(), device="cpu")
    out = transform(img)
    assert out.dtype == np.uint8
    assert out.shape == img.shape
    np.testing.assert_array_equal(out, img)


def test_server_end_to_end_with_real_client():
    """Real serve() + real DifixClient: frames cross the wire and transform."""
    holder: dict = {}
    stop = threading.Event()
    th = threading.Thread(
        target=serve,
        kwargs=dict(
            host="127.0.0.1",
            port=0,
            transform=lambda x: 255 - x,
            stop_event=stop,
            on_listening=lambda p: holder.__setitem__("port", p),
        ),
        daemon=True,
    )
    th.start()
    try:
        for _ in range(200):  # wait up to ~4s for the listen socket
            if "port" in holder:
                break
            time.sleep(0.02)
        assert "port" in holder, "server never reported a listening port"
        c = DifixClient("127.0.0.1", holder["port"])
        img = _rand_img()
        np.testing.assert_array_equal(c.fix(img), 255 - img)
        np.testing.assert_array_equal(c.fix(img), 255 - img)  # second frame
        c.close()
    finally:
        stop.set()
        th.join(timeout=2.0)


def test_server_serves_second_client_while_first_is_stalled():
    """A stalled/half-open client must NOT block other clients.

    Regression for the single-connection-serial bug: the original serve()
    accept()ed one connection and looped on it forever, so a client that
    connected but never sent a full frame wedged read_frame() and starved every
    later connection (the viewer would then see each fix() time out → very
    laggy). serve() must handle connections concurrently.
    """
    holder: dict = {}
    stop = threading.Event()
    th = threading.Thread(
        target=serve,
        kwargs=dict(
            host="127.0.0.1",
            port=0,
            transform=lambda x: 255 - x,
            stop_event=stop,
            on_listening=lambda p: holder.__setitem__("port", p),
        ),
        daemon=True,
    )
    th.start()
    stalled = None
    try:
        for _ in range(200):
            if "port" in holder:
                break
            time.sleep(0.02)
        port = holder["port"]
        # Client A: connect, send a partial header, then never finish → the
        # server's read_frame() blocks in recvall() waiting for the rest.
        stalled = socket.create_connection(("127.0.0.1", port), timeout=2.0)
        stalled.sendall(b"DFX1")  # 4 of 16 header bytes; rest never arrives
        time.sleep(0.4)  # let the server accept + block on A
        # Client B must still be served promptly despite A being wedged.
        b = DifixClient("127.0.0.1", port, timeout=3.0)
        img = _rand_img()
        out = b.fix(img)
        np.testing.assert_array_equal(out, 255 - img)
        assert b.healthy is True, "second client was blocked by the stalled one"
        b.close()
    finally:
        if stalled is not None:
            stalled.close()
        stop.set()
        th.join(timeout=2.0)
