# SPDX-License-Identifier: Apache-2.0
"""pytest conftest.py for threedgrut/tests — session-wide sys.modules stubs.

**I-1 fix history** (T2.4):
Previously, `_install_stubs()` and the `MCMCStrategy.__init__` no-CUDA patch
lived at module-level in `test_layered_mcmc.py`. This caused test_layered_gaussians.py
to fail when run standalone (without test_layered_mcmc.py being collected first),
because `threedgrt_tracer/__init__.py` → `threedgrut.datasets` → `ncore` would raise
ModuleNotFoundError.

pytest auto-loads conftest.py before any test module in the directory. Moving the
stubs here ensures they are installed regardless of collection order.

**Important**: The `MCMCStrategy.__init__` monkey-patch is a permanent
process-level mutation (not scoped to any fixture). It affects every test in this
directory that uses MCMCStrategy. This is intentional: the patch replaces the
CUDA-dependent __init__ with a CPU-safe version so structural invariant tests
can run on Mac without OptiX/CUDA.

Stub inventory and rationale:
  1.  ncore / ncore.sensors / ncore.data
          NVIDIA-internal SDK; unavailable on Mac.
  2.  threedgrt_tracer / threedgut_tracer
          CUDA extensions; unavailable without OptiX / CUDA toolkit.
  3.  tqdm
          Not installed in the minimal CPU venv.
  4.  sklearn / sklearn.neighbors
          scikit-learn; used by threedgrut/model/geometry.py for KD-tree;
          not installed in CPU venv.
  5.  torch.utils.tensorboard / tensorboard
          Not installed in CPU pip venv; pulled in by utils/misc.py.
  6.  threedgrut.datasets  (package-level stub with __path__)
          datasets/__init__.py imports ALL dataset loaders, which cascade into
          cv2, imageio, einops, kornia, simplejpeg, ncore.data.v4, etc.
          We provide an empty package body so submodules can still be loaded
          individually on demand.
  7.  threedgrut.datasets.utils  (real module, DEFAULT_DEVICE → cpu)
          model/background.py imports DEFAULT_DEVICE from here and uses it to
          create tensors in __init__. The real module is loaded directly
          (bypassing datasets/__init__.py), then DEFAULT_DEVICE is overridden
          to torch.device("cpu") so background model construction works on Mac.
"""

import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock


def _install_stubs() -> None:
    """Install sys.modules stubs for packages that require CUDA or ncore SDK."""
    import contextlib

    import torch  # noqa: E402 — torch is available

    # 0. torch.cuda.nvtx.range: on a CPU-only torch build this raises
    #    "NVTX functions not installed" the moment it is used — even as a bare
    #    @decorator (e.g. model.losses.ssim). It is used BOTH as a decorator and
    #    as a context manager across threedgrut, so replace it with a no-op that
    #    is valid in both forms. GPU builds are unaffected (this only runs when
    #    the tests import). Lets CPU tests call NVTX-decorated code (E2.2 ssim).
    class _NoopNvtxRange(contextlib.ContextDecorator):
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    if not torch.cuda.is_available():
        torch.cuda.nvtx.range = _NoopNvtxRange  # type: ignore[assignment]

    # 1. ncore: NVIDIA-internal SDK
    for name in ("ncore", "ncore.sensors", "ncore.data"):
        if name not in sys.modules:
            sys.modules[name] = MagicMock()

    # 2. CUDA tracers
    for name in ("threedgrt_tracer", "threedgut_tracer"):
        if name not in sys.modules:
            sys.modules[name] = MagicMock()

    # 3. tqdm: not installed in CPU venv.
    #    Must be a types.ModuleType with a valid __spec__ so torch._dynamo does not
    #    raise "tqdm.__spec__ is not set".  Also expose tqdm.tqdm as a no-op callable
    #    so `from tqdm import tqdm` succeeds in threedgrut/export/scripts/.
    if "tqdm" not in sys.modules:
        _tqdm_mod = types.ModuleType("tqdm")
        _tqdm_mod.__path__ = []
        _tqdm_mod.__spec__ = importlib.util.spec_from_loader("tqdm", loader=None)
        _tqdm_mod.tqdm = MagicMock()  # covers `from tqdm import tqdm`
        sys.modules["tqdm"] = _tqdm_mod

    # 4. sklearn: not installed in CPU venv; used by model/geometry.py.
    #    Must use a types.ModuleType with a valid __spec__ (not MagicMock and not
    #    __spec__=None) so that torch._dynamo.trace_rules.find_spec("sklearn")
    #    succeeds when torch.optim.Adam triggers dynamo initialisation.
    if "sklearn" not in sys.modules:
        _sklearn_mod = types.ModuleType("sklearn")
        _sklearn_mod.__path__ = []  # mark as package
        _sklearn_mod.__spec__ = importlib.util.spec_from_loader("sklearn", loader=None)
        sys.modules["sklearn"] = _sklearn_mod
    if "sklearn.neighbors" not in sys.modules:
        _sklearn_nbrs = types.ModuleType("sklearn.neighbors")
        _sklearn_nbrs.__spec__ = importlib.util.spec_from_loader("sklearn.neighbors", loader=None)
        sys.modules["sklearn.neighbors"] = _sklearn_nbrs

    # 5. torch.utils.tensorboard: not installed in CPU pip venv
    import torch.utils  # noqa: E402

    if not hasattr(torch.utils, "tensorboard") or not hasattr(getattr(torch.utils, "tensorboard", None), "writer"):
        tb_stub = MagicMock()
        tb_stub.writer.SummaryWriter = MagicMock()
        sys.modules.setdefault("tensorboard", MagicMock())
        sys.modules.setdefault("torch.utils.tensorboard", tb_stub)
        sys.modules.setdefault("torch.utils.tensorboard.writer", tb_stub.writer)

    # 5b. fused_ssim: CUDA extension (threedgrut.model.losses.ssim wraps it).
    #     Provide a pure-torch SSIM so CPU tests can exercise the real
    #     photometric loss path (E2.2 distill). Matches fused_ssim's public
    #     signature ``fused_ssim(img1, img2, padding="valid") -> scalar``:
    #     11x11 Gaussian window, "valid" padding (no border), mean over the map.
    if "fused_ssim" not in sys.modules:
        import math

        import torch.nn.functional as F  # noqa: N812

        def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype):
            coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2.0
            g = torch.exp(-(coords**2) / (2.0 * sigma**2))
            g = (g / g.sum()).unsqueeze(1)
            w2d = (g @ g.t()).unsqueeze(0).unsqueeze(0)  # [1,1,ws,ws]
            return w2d.expand(channels, 1, window_size, window_size).contiguous()

        def _fused_ssim(img1, img2, padding="valid", train=True):
            ws, sigma = 11, 1.5
            c = img1.shape[1]
            pad = 0 if padding == "valid" else ws // 2
            win = _gaussian_window(ws, sigma, c, img1.device, img1.dtype)
            mu1 = F.conv2d(img1, win, padding=pad, groups=c)
            mu2 = F.conv2d(img2, win, padding=pad, groups=c)
            mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
            sigma1_sq = F.conv2d(img1 * img1, win, padding=pad, groups=c) - mu1_sq
            sigma2_sq = F.conv2d(img2 * img2, win, padding=pad, groups=c) - mu2_sq
            sigma12 = F.conv2d(img1 * img2, win, padding=pad, groups=c) - mu1_mu2
            C1, C2 = 0.01**2, 0.03**2
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
            )
            return ssim_map.mean()

        _fs_mod = types.ModuleType("fused_ssim")
        _fs_mod.__spec__ = importlib.util.spec_from_loader("fused_ssim", loader=None)
        _fs_mod.fused_ssim = _fused_ssim
        _ = math  # silence unused when linters fold the import
        sys.modules["fused_ssim"] = _fs_mod

    # 6. threedgrut.datasets: package-level stub with __path__ so submodule
    #    imports (e.g. threedgrut.datasets.utils) succeed without executing
    #    datasets/__init__.py (which would pull cv2, imageio, kornia, etc.).
    if "threedgrut.datasets" not in sys.modules:
        _datasets_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
        ds_mod = types.ModuleType("threedgrut.datasets")
        ds_mod.__path__ = [_datasets_path]
        ds_mod.__package__ = "threedgrut.datasets"
        sys.modules["threedgrut.datasets"] = ds_mod

    # 7. threedgrut.datasets.utils: load the real module file directly (so
    #    model.py gets read_colmap_points3D_text etc.), then override
    #    DEFAULT_DEVICE to cpu so background.py can create tensors on Mac.
    if "threedgrut.datasets.utils" not in sys.modules:
        _utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "utils.py"))
        _spec = importlib.util.spec_from_file_location("threedgrut.datasets.utils", _utils_path)
        _utils_mod = importlib.util.module_from_spec(_spec)
        _utils_mod.__package__ = "threedgrut.datasets"
        sys.modules["threedgrut.datasets.utils"] = _utils_mod
        _spec.loader.exec_module(_utils_mod)
        # Override DEFAULT_DEVICE: the real value is torch.device("cuda") which
        # causes background.py to fail on Mac (no CUDA).
        _utils_mod.DEFAULT_DEVICE = torch.device("cpu")


_install_stubs()

# After stubs are installed, patch MCMCStrategy.__init__ to skip the two
# CUDA-only operations: load_mcmc_plugin() (JIT-compiles CUDA kernels) and
# `self.binoms = torch.tensor(..., device="cuda")`.  The patched init still
# calls BaseStrategy.__init__ so self.conf / self.model / self._suspended are
# set correctly.  This patch runs at conftest import time and is permanent for
# the test session (process-level mutation).
import threedgrut.strategy.mcmc as _mcmc_mod  # noqa: E402

_mcmc_mod.load_mcmc_plugin = lambda: None  # no-op: skip CUDA JIT

# Capture the real __init__ BEFORE overwriting it so a future test that needs
# the real CUDA-initializing MCMCStrategy.__init__ can restore it via
# monkeypatch.  Import _original_init from this module to use it:
#
#     from threedgrut.tests.conftest import _original_init
#
# WARNING — how to restore the real CUDA __init__ in a future test:
#
#     If a future test needs the real CUDA-initializing MCMCStrategy.__init__,
#     restore it inside the test via monkeypatch:
#
#         from threedgrut.tests.conftest import _original_init
#         from threedgrut.strategy.mcmc import MCMCStrategy
#
#         def test_real_init(monkeypatch):
#             monkeypatch.setattr(MCMCStrategy, "__init__", _original_init)
#             ...
#
#     (_original_init is captured below, before the module-level patch fires,
#     so it always refers to the unpatched version from the real source file.)
_original_init = _mcmc_mod.MCMCStrategy.__init__


def _mcmc_init_no_cuda(self, config, model):
    """Drop-in __init__ that skips load_mcmc_plugin() and CUDA binoms tensor."""
    from threedgrut.strategy.base import BaseStrategy

    BaseStrategy.__init__(self, config=config, model=model)
    self.binoms = None  # not used in structural / cap tests


_mcmc_mod.MCMCStrategy.__init__ = _mcmc_init_no_cuda
