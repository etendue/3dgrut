import io
import socket
import struct
import threading

import torch

from scripts.e21_harmonizer_batch_fix import harmonizer_fix_frame


def _recvall(s, n):
    b = b""
    while len(b) < n:
        d = s.recv(n - len(b))
        b += d
    return b


def _echo_server(port, ready):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    ready.set()
    c, _ = srv.accept()
    n = struct.unpack(">Q", _recvall(c, 8))[0]
    d = torch.load(io.BytesIO(_recvall(c, n)), weights_only=False)
    out = d["input"]  # echo input unchanged (h*w, 3)
    bio = io.BytesIO()
    torch.save(out, bio)
    p = bio.getvalue()
    c.sendall(struct.pack(">Q", len(p)) + p)
    c.close()
    srv.close()


def test_fix_frame_roundtrip_shape_and_values():
    port = 59600
    ready = threading.Event()
    threading.Thread(target=_echo_server, args=(port, ready), daemon=True).start()
    ready.wait(timeout=5)
    img = torch.rand(8, 12, 3)  # (H, W, 3) float [0,1]
    out = harmonizer_fix_frame(img, host="127.0.0.1", port=port)
    assert out.shape == (8, 12, 3)
    assert torch.allclose(out, img, atol=1e-5)  # echo server returns input unchanged
