# SPDX-License-Identifier: Apache-2.0
"""IPC tests for the temporal Harmonizer viewer integration (E2.6).

Exercises the localhost wire protocol (``harmonizer_protocol``, magic ``HMN1``)
and the history-carrying client (``HarmonizerTemporalClient``) that lets
``viser_gui_4d`` ship ``1 + K`` frames to an out-of-process Harmonizer server
for de-flickering continuous play sequences.

Everything here is GPU-free: the Harmonizer model is replaced by an echo stand-in
(the server returns the current frame unchanged) so the temporal IPC plumbing —
history growth, reset semantics, cold start, graceful fallback — can be verified
on a Mac.

Run:
    pytest threedgrut/tests/test_harmonizer_temporal_ipc.py -v
"""
from __future__ import annotations

import contextlib
import socket
import threading
import time

import numpy as np
import pytest

from threedgrut_playground.utils.difix_protocol import (
    pack_frame,
    read_frame,
)
from threedgrut_playground.utils.harmonizer_client import (
    HarmonizerTemporalClient,
)
from threedgrut_playground.utils.harmonizer_protocol import (
    HEADER_SIZE,
    MAGIC,
    pack_temporal,
    read_temporal,
    recvall,
)


def _rand_img(h: int = 7, w: int = 11, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


@contextlib.contextmanager
def _echo_temporal_server():
    """Spin up a localhost TCP server that echoes the current frame.

    Mirrors the temporal protocol the real Harmonizer server speaks: one request
    carries ``(1 + K_in)`` frames, the reply is a single ``DFX1`` frame. The
    "model" is echo (return curr unchanged). Yields the bound port.

    Also records the ``K_in`` seen per request into ``observed_kin`` so tests
    can assert history growth / reset semantics.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    srv.settimeout(0.2)
    port = srv.getsockname()[1]
    stop = threading.Event()
    observed_kin: list[int] = []

    def _loop():
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except (TimeoutError, socket.timeout):
                continue
            with conn:
                try:
                    while True:
                        curr, hist = read_temporal(conn)
                        observed_kin.append(len(hist))
                        conn.sendall(pack_frame(curr))  # echo curr as DFX1 reply
                except (ConnectionError, OSError):
                    pass

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    try:
        yield port, observed_kin
    finally:
        stop.set()
        t.join(timeout=1.0)
        srv.close()


# --------------------------------------------------------------------------- #
#                              protocol layer                                  #
# --------------------------------------------------------------------------- #
def test_pack_read_roundtrip_single_frame():
    """A cold-start request (empty history) round-trips curr over a socket."""
    img = _rand_img()
    a, b = socket.socketpair()
    try:
        a.sendall(pack_temporal(img, [], K=4))
        curr, hist = read_temporal(b)
    finally:
        a.close()
        b.close()
    np.testing.assert_array_equal(curr, img)
    assert hist == []


def test_pack_read_roundtrip_with_history():
    """A request with K history frames round-trips all frames in order."""
    curr = _rand_img(seed=1)
    hist = [_rand_img(seed=2), _rand_img(seed=3), _rand_img(seed=4)]
    a, b = socket.socketpair()
    try:
        a.sendall(pack_temporal(curr, hist, K=4))
        got_curr, got_hist = read_temporal(b)
    finally:
        a.close()
        b.close()
    np.testing.assert_array_equal(got_curr, curr)
    assert len(got_hist) == len(hist)
    for g, h in zip(got_hist, hist):
        np.testing.assert_array_equal(g, h)


def test_pack_temporal_truncates_to_K():
    """If history has more than K frames, only the last K are packed."""
    curr = _rand_img(seed=1)
    hist = [_rand_img(seed=i) for i in range(10)]
    a, b = socket.socketpair()
    try:
        a.sendall(pack_temporal(curr, hist, K=3))
        _, got_hist = read_temporal(b)
    finally:
        a.close()
        b.close()
    assert len(got_hist) == 3
    # Last K of history (most-recent-last): seeds 7, 8, 9.
    for g, expected_seed in zip(got_hist, [7, 8, 9]):
        np.testing.assert_array_equal(g, _rand_img(seed=expected_seed))


def test_pack_temporal_cold_start_carries_fewer_frames():
    """Before history is full, K_in < K (server sees V=1+k_in)."""
    curr = _rand_img(seed=1)
    hist = [_rand_img(seed=2)]  # only 1 of K=4
    a, b = socket.socketpair()
    try:
        a.sendall(pack_temporal(curr, hist, K=4))
        _, got_hist = read_temporal(b)
    finally:
        a.close()
        b.close()
    assert len(got_hist) == 1


def test_read_temporal_rejects_bad_magic():
    """A corrupt header (wrong magic) raises rather than mis-parsing."""
    a, b = socket.socketpair()
    try:
        a.sendall(b"XXXX" + b"\x00" * 16)  # bad magic + junk 20-byte header
        with pytest.raises(ValueError, match="magic"):
            read_temporal(b)
    finally:
        a.close()
        b.close()


def test_pack_temporal_rejects_non_uint8_hwc():
    """pack_temporal contract is (H,W,3) uint8 — guard wrong dtype / rank."""
    with pytest.raises(ValueError):
        pack_temporal(np.zeros((4, 4, 3), dtype=np.float32), [], K=4)
    with pytest.raises(ValueError):
        pack_temporal(np.zeros((4, 4), dtype=np.uint8), [], K=4)


def test_pack_temporal_rejects_history_shape_mismatch():
    """A history frame whose shape != curr raises (guards a size change)."""
    curr = _rand_img(7, 11)
    bad_hist = [_rand_img(8, 4)]  # different HxW
    with pytest.raises(ValueError):
        pack_temporal(curr, bad_hist, K=4)


def test_header_size_is_20():
    """Sanity: the temporal header is 20 bytes (4 magic + 4 uint32)."""
    assert HEADER_SIZE == 20
    assert MAGIC == b"HMN1"


# --------------------------------------------------------------------------- #
#                              client layer                                    #
# --------------------------------------------------------------------------- #
def test_client_roundtrip_echo():
    """fix() over an echo server returns the same frame + records RTT."""
    img = _rand_img()
    with _echo_temporal_server() as (port, kin):
        c = HarmonizerTemporalClient("127.0.0.1", port, K=4)
        try:
            out = c.fix(img)
            np.testing.assert_array_equal(out, img)
            assert c.healthy is True
            assert c.last_rtt_ms > 0.0
        finally:
            c.close()


def test_client_history_grows_then_caps_at_K():
    """History deque fills to K, but V stays 1 until full (Conv3d constraint).

    Harmonizer's temporal Conv3d (kernel=3 on V axis) only accepts V=1 or V>=3;
    V=2 crashes forward. So the client mirrors the official inference script:
    send V=1 (curr alone) while history < K, then V=1+K once full. The deque
    still fills frame-by-frame (self-reference), but the *wire* V jumps 1→1+K
    only when full. observed_kin confirms: 0,0,0,0,4,4,4 for K=4.
    """
    with _echo_temporal_server() as (port, kin):
        c = HarmonizerTemporalClient("127.0.0.1", port, K=4)
        try:
            for i in range(7):
                c.fix(_rand_img(seed=i))
            # deque caps at K=4
            assert c.history_depth == 4
        finally:
            c.close()
    # Server saw V=1 (K_in=0) for the first 4 frames (warmup), then V=5 (K_in=4).
    assert kin == [0, 0, 0, 0, 4, 4, 4]


def test_client_reset_clears_history():
    """fix(img, reset=True) drops history — next request is V=1 again.

    With the Conv3d warmup rule, V is already 1 while history < K, so a reset
    during warmup is a no-op on the wire (still V=1). The meaningful test is
    reset *after* history is full: it drops back to V=1.
    """
    with _echo_temporal_server() as (port, kin):
        c = HarmonizerTemporalClient("127.0.0.1", port, K=2)
        try:
            c.fix(_rand_img(seed=1))  # V=1 (warmup, history 0<2)
            c.fix(_rand_img(seed=2))  # V=1 (warmup, history 1<2)
            assert c.history_depth == 2
            c.fix(_rand_img(seed=3))  # V=3 (history full: 1+K=3)
            # Reset: history cleared → V=1 again.
            c.fix(_rand_img(seed=4), reset=True)
            assert c.history_depth == 1
            c.fix(_rand_img(seed=5))  # V=1 (warmup again, 1<2)
            c.fix(_rand_img(seed=6))  # V=1 (warmup, 2 not yet, history 1<2... actually 2 now)
        finally:
            c.close()
    # K=2: frame1 V=1(0), frame2 V=1(0, 1<2), frame3 V=3(2 full),
    #       frame4 reset V=1(0), frame5 V=1(0, 1<2), frame6 V=3(2 full)
    assert kin == [0, 0, 2, 0, 0, 2]


def test_client_reset_method_clears_history():
    """The explicit reset() method also clears the deque."""
    with _echo_temporal_server() as (port, kin):
        c = HarmonizerTemporalClient("127.0.0.1", port, K=2)
        try:
            c.fix(_rand_img(seed=1))
            c.fix(_rand_img(seed=2))
            c.fix(_rand_img(seed=3))  # history full → V=3
            assert c.history_depth == 2
            c.reset()
            assert c.history_depth == 0
            c.fix(_rand_img(seed=4))  # V=1 after reset
        finally:
            c.close()
    assert kin == [0, 0, 2, 0]


def test_client_cold_start_is_V1_until_full():
    """While history < K, every request is V=1 (Conv3d warmup constraint).

    The client never sends V=2..K — those crash Harmonizer's temporal Conv3d
    (kernel size 3 > V). Only V=1 (warmup) and V=1+K (full) are valid.
    """
    with _echo_temporal_server() as (port, kin):
        c = HarmonizerTemporalClient("127.0.0.1", port, K=3)
        try:
            for i in range(5):
                c.fix(_rand_img(seed=i))
        finally:
            c.close()
    # V=1,1,1 (warmup, history 0,1,2 < 3), then V=4,4 (history full at 3+)
    assert kin == [0, 0, 0, 3, 3]


def test_client_falls_back_to_raw_on_refused_connection():
    """No server → fix() must return the input unchanged, never raise."""
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))  # bind but do NOT listen → connect refused
    port = holder.getsockname()[1]
    try:
        img = _rand_img()
        c = HarmonizerTemporalClient("127.0.0.1", port, K=4, timeout=0.3)
        out = c.fix(img)
        np.testing.assert_array_equal(out, img)  # graceful raw fallback
        assert c.healthy is False
        c.close()
    finally:
        holder.close()


def test_client_failure_does_not_poison_history():
    """A failed fix() leaves the deque intact (no partial/poison entry).

    Important for the viewer: a transient server blip must not corrupt the
    temporal state, or subsequent frames would feed wrong history.
    """
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    dead_port = holder.getsockname()[1]
    holder.close()
    c = HarmonizerTemporalClient("127.0.0.1", dead_port, K=4, timeout=0.3)
    try:
        # Prime history against a live server first.
        with _echo_temporal_server() as (port, _):
            c.host, c.port = "127.0.0.1", port
            c.fix(_rand_img(seed=1))
            c.fix(_rand_img(seed=2))
            assert c.history_depth == 2
        # Simulate server crash: drop the persistent connection (a real server
        # kill surfaces as a broken pipe on the next send, which the client
        # _close()s). Then point at a dead port so reconnect cannot succeed.
        c._close()
        c.host, c.port = "127.0.0.1", dead_port
        out = c.fix(_rand_img(seed=3))
        assert np.array_equal(out, _rand_img(seed=3))  # raw fallback
        assert c.healthy is False
        assert c.history_depth == 2  # NOT 3 — failure did not append
    finally:
        c.close()


def test_client_from_addr_parses_host_port():
    c = HarmonizerTemporalClient.from_addr("127.0.0.1:59490", K=3)
    assert c.host == "127.0.0.1"
    assert c.port == 59490
    assert c.K == 3


def test_client_persistent_connection_across_frames():
    """A persistent connection serves multiple fix() calls (no reconnect spam)."""
    imgs = [_rand_img(seed=i) for i in range(3)]
    with _echo_temporal_server() as (port, _):
        c = HarmonizerTemporalClient("127.0.0.1", port, K=2)
        first_sock = None
        try:
            for img in imgs:
                out = c.fix(img)
                np.testing.assert_array_equal(out, img)
                if first_sock is None:
                    first_sock = c._sock
            # Same socket object reused across all three calls.
            assert c._sock is first_sock
        finally:
            c.close()


# --------------------------------------------------------------------------- #
#                              server layer                                    #
# --------------------------------------------------------------------------- #
def test_server_end_to_end_with_real_client():
    """Real serve() + real HarmonizerTemporalClient: temporal frames cross the wire.

    Uses an echo transform (returns curr unchanged) as the stand-in model. The
    client feeds a 3-frame sequence so history grows 0→1→2 and the server sees
    V=1, 2, 3 respectively. This verifies the full serve()/read_temporal/
    pack_frame loop end-to-end, without needing the Harmonizer model or GPU.
    """
    from threedgrut_playground.harmonizer_temporal_server import serve

    def _echo(curr, history):  # TemporalTransform stand-in
        return curr

    holder: dict = {}
    stop = threading.Event()
    th = threading.Thread(
        target=serve,
        kwargs=dict(
            host="127.0.0.1",
            port=0,
            transform=_echo,
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
        c = HarmonizerTemporalClient("127.0.0.1", holder["port"], K=2)
        for seed in (1, 2, 3):
            img = _rand_img(seed=seed)
            out = c.fix(img)
            np.testing.assert_array_equal(out, img)
        assert c.history_depth == 2  # capped at K=2 after 3 frames
        c.close()
    finally:
        stop.set()
        th.join(timeout=2.0)


def test_server_serves_second_client_while_first_is_stalled():
    """A stalled/half-open client must NOT block other clients (concurrency).

    Regression mirror of the analogous DiFix test: serve() must handle
    connections concurrently so a client that connects but never sends a full
    temporal request cannot starve later connections.
    """
    from threedgrut_playground.harmonizer_temporal_server import serve

    def _echo(curr, history):
        return curr

    holder: dict = {}
    stop = threading.Event()
    th = threading.Thread(
        target=serve,
        kwargs=dict(
            host="127.0.0.1",
            port=0,
            transform=_echo,
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
        # Client A: connect, send the HMN1 magic (4 of 20 header bytes), never
        # finish → the server's read_temporal() blocks in recvall() for the
        # remaining 16 bytes.
        stalled = socket.create_connection(("127.0.0.1", port), timeout=2.0)
        stalled.sendall(b"HMN1")
        time.sleep(0.4)  # let the server accept + block on A
        # Client B must still be served promptly despite A being wedged.
        b = HarmonizerTemporalClient("127.0.0.1", port, K=2, timeout=3.0)
        img = _rand_img()
        out = b.fix(img)
        np.testing.assert_array_equal(out, img)
        assert b.healthy is True, "second client was blocked by the stalled one"
        b.close()
    finally:
        if stalled is not None:
            stalled.close()
        stop.set()
        th.join(timeout=2.0)


def test_server_make_transform_shape_contract():
    """make_harmonizer_temporal_transform shape contract (V-stacking math).

    Verifies the V-stacking / V=0 selection / resize-back math without needing
    the Harmonizer model: swap in a fake model whose forward just returns its
    input (identity over 5D), and confirm the transform maps (curr, history) to
    a uint8 (H,W,3) frame of the *curr* shape.
    """
    from threedgrut_playground.harmonizer_temporal_server import (
        make_harmonizer_temporal_transform,
    )

    class _IdentityModel:
        def __call__(self, x5):  # (1,C,V,h,w) -> same
            return x5

        def to(self, *a, **k):
            return self

    transform = make_harmonizer_temporal_transform(
        _IdentityModel(), device="cpu", model_res=(8, 12)
    )
    curr = _rand_img(7, 11, seed=1)
    hist = [_rand_img(7, 11, seed=2), _rand_img(7, 11, seed=3)]
    out = transform(curr, hist)
    assert out.shape == curr.shape
    assert out.dtype == np.uint8
    # Identity model + resize round-trip is lossy at small sizes, so we only
    # assert shape/dtype/range here (the real model's correctness is a GPU
    # integration test, not a Mac unit test).
    assert out.min() >= 0 and out.max() <= 255
