# SPDX-License-Identifier: Apache-2.0
"""Contract tests for MCMCStrategy._get_add_cap() (T2.1) and LayeredMCMCStrategy (T2.2).

Uses __new__ to bypass __init__ (which JIT-compiles CUDA, unavailable on Mac).
Module-level sys.modules stubs prevent import errors from CUDA/ncore deps on Mac.
"""

import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock

import pytest
from hydra import compose, initialize_config_dir


# ----------------------------------------------------------------------- conf
_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs")
)


@pytest.fixture(scope="module")
def real_conf():
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="apps/ncore_3dgut_mcmc")


# --------------------------------------------------------------------------
# Stub out modules that are unavailable on Mac CPU.
#
# T2.1 tests use MCMCStrategy.__new__ to bypass __init__ entirely, so they
# only need to be able to *import* the class.
#
# T2.2 tests instantiate LayeredMCMCStrategy, which calls MCMCStrategy.__init__
# on each sub-strategy and requires real MixtureOfGaussians (nn.Module).  This
# requires a more complete stub set plus a run-time patch of the CUDA-specific
# portions of MCMCStrategy.__init__.
#
# Stub inventory and rationale:
#   1.  ncore / ncore.sensors / ncore.data
#           NVIDIA-internal SDK; unavailable on Mac.
#   2.  threedgrt_tracer / threedgut_tracer
#           CUDA extensions; unavailable without OptiX / CUDA toolkit.
#   3.  tqdm
#           Not installed in the minimal CPU venv.
#   4.  sklearn / sklearn.neighbors
#           scikit-learn; used by threedgrut/model/geometry.py for KD-tree;
#           not installed in CPU venv.
#   5.  torch.utils.tensorboard / tensorboard
#           Not installed in CPU pip venv; pulled in by utils/misc.py.
#   6.  threedgrut.datasets  (package-level stub with __path__)
#           datasets/__init__.py imports ALL dataset loaders, which cascade into
#           cv2, imageio, einops, kornia, simplejpeg, ncore.data.v4, etc.
#           We provide an empty package body so submodules can still be loaded
#           individually on demand.
#   7.  threedgrut.datasets.utils  (real module, DEFAULT_DEVICE → cpu)
#           model/background.py imports DEFAULT_DEVICE from here and uses it to
#           create tensors in __init__.  The real module is loaded directly
#           (bypassing datasets/__init__.py), then DEFAULT_DEVICE is overridden
#           to torch.device("cpu") so background model construction works on Mac.
# --------------------------------------------------------------------------
def _install_stubs() -> None:
    """Install sys.modules stubs for packages that require CUDA or ncore SDK."""
    import torch  # noqa: E402 — torch is available

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
    if not hasattr(torch.utils, "tensorboard") or not hasattr(
        getattr(torch.utils, "tensorboard", None), "writer"
    ):
        tb_stub = MagicMock()
        tb_stub.writer.SummaryWriter = MagicMock()
        sys.modules.setdefault("tensorboard", MagicMock())
        sys.modules.setdefault("torch.utils.tensorboard", tb_stub)
        sys.modules.setdefault("torch.utils.tensorboard.writer", tb_stub.writer)

    # 6. threedgrut.datasets: package-level stub with __path__ so submodule
    #    imports (e.g. threedgrut.datasets.utils) succeed without executing
    #    datasets/__init__.py (which would pull cv2, imageio, kornia, etc.).
    if "threedgrut.datasets" not in sys.modules:
        _datasets_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "datasets")
        )
        ds_mod = types.ModuleType("threedgrut.datasets")
        ds_mod.__path__ = [_datasets_path]
        ds_mod.__package__ = "threedgrut.datasets"
        sys.modules["threedgrut.datasets"] = ds_mod

    # 7. threedgrut.datasets.utils: load the real module file directly (so
    #    model.py gets read_colmap_points3D_text etc.), then override
    #    DEFAULT_DEVICE to cpu so background.py can create tensors on Mac.
    if "threedgrut.datasets.utils" not in sys.modules:
        _utils_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "datasets", "utils.py")
        )
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
# set correctly.  This patch runs at module import time and is permanent for
# the test session.
import threedgrut.strategy.mcmc as _mcmc_mod  # noqa: E402

_mcmc_mod.load_mcmc_plugin = lambda: None  # no-op: skip CUDA JIT


def _mcmc_init_no_cuda(self, config, model):
    """Drop-in __init__ that skips load_mcmc_plugin() and CUDA binoms tensor."""
    from threedgrut.strategy.base import BaseStrategy

    BaseStrategy.__init__(self, config=config, model=model)
    self.binoms = None  # not used in structural / cap tests


_mcmc_mod.MCMCStrategy.__init__ = _mcmc_init_no_cuda


def test_mcmc_get_add_cap_defaults_to_conf(real_conf):
    """T2.1: MCMCStrategy._get_add_cap() returns conf.strategy.add.max_n_gaussians by default.

    Uses __new__ to bypass __init__ (which JIT-compiles CUDA, unavailable on Mac).
    """
    from threedgrut.strategy.mcmc import MCMCStrategy

    strat = MCMCStrategy.__new__(MCMCStrategy)
    strat.conf = real_conf
    assert strat._get_add_cap() == real_conf.strategy.add.max_n_gaussians


def test_layered_mcmc_holds_sub_strategy_per_particle_layer(real_conf):
    """T2.2: each is_particle_layer=True layer gets one MCMCStrategy; non-particle layers skipped."""
    from threedgrut.layers.layer_spec import LayerSpec
    from threedgrut.layers.layered_model import LayeredGaussians
    from threedgrut.strategy.layered_mcmc import LayeredMCMCStrategy

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(
            name="road",
            layer_id=1,
            max_n_particles=200_000,
            scale_prior=(0.1, 0.1, 0.001),
            scale_lr_mult=0.2,
            mask_field="road_mask",
        ),
        LayerSpec(name="sky_envmap", layer_id=-1, max_n_particles=0, is_particle_layer=False),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    strat = LayeredMCMCStrategy(real_conf, model, specs)
    assert set(strat.sub_strategies.keys()) == {"background", "road"}
    assert "sky_envmap" not in strat.sub_strategies


def test_layered_mcmc_sub_uses_per_layer_cap(real_conf):
    """T2.2: each sub-strategy's _get_add_cap() returns spec.max_n_particles."""
    from threedgrut.layers.layer_spec import LayerSpec
    from threedgrut.layers.layered_model import LayeredGaussians
    from threedgrut.strategy.layered_mcmc import LayeredMCMCStrategy

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=600_000),
        LayerSpec(name="road", layer_id=1, max_n_particles=200_000),
    ]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    strat = LayeredMCMCStrategy(real_conf, model, specs)
    assert strat.sub_strategies["background"]._get_add_cap() == 600_000
    assert strat.sub_strategies["road"]._get_add_cap() == 200_000


def test_layered_mcmc_single_bg_equivalent_to_v1(real_conf):
    """T2.2: single-bg mode produces exactly one sub-strategy that is an MCMCStrategy instance.

    Invariant: sub.model is model.layers['background'] (the layer MoG, not the wrapper).
    """
    from threedgrut.layers.layer_spec import LayerSpec
    from threedgrut.layers.layered_model import LayeredGaussians
    from threedgrut.strategy.layered_mcmc import LayeredMCMCStrategy
    from threedgrut.strategy.mcmc import MCMCStrategy

    specs = [LayerSpec(name="background", layer_id=0, max_n_particles=600_000)]
    model = LayeredGaussians(real_conf, specs=specs, scene_extent=10.0)
    strat = LayeredMCMCStrategy(real_conf, model, specs)
    assert len(strat.sub_strategies) == 1
    assert isinstance(strat.sub_strategies["background"], MCMCStrategy)
    # Key invariant: sub.model references the layer MoG, not the LayeredGaussians wrapper.
    assert strat.sub_strategies["background"].model is model.layers["background"]
