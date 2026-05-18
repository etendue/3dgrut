# SPDX-License-Identifier: Apache-2.0
"""T2.1 contract tests for MCMCStrategy._get_add_cap() hook.

Uses __new__ to bypass __init__ (which JIT-compiles CUDA, unavailable on Mac).
Module-level sys.modules stubs prevent import errors from CUDA/ncore deps on Mac.
"""

import os
import sys
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
# Only the minimum set needed to let mcmc.py be imported:
#   1. threedgrut.model.model  — pulls in threedgrt_tracer → ncore (NVIDIA-internal)
#   2. torch.utils.tensorboard — not installed in CPU venv; needed by utils/misc.py
# We do NOT stub threedgrut.strategy.base or threedgrut.utils.misc so that
# MCMCStrategy remains a real Python class (not a MagicMock).
# --------------------------------------------------------------------------
def _install_stubs() -> None:
    """Install sys.modules stubs for packages that require CUDA or ncore SDK."""
    # ncore and CUDA tracers
    for name in ("ncore", "ncore.sensors", "threedgrt_tracer", "threedgut_tracer"):
        if name not in sys.modules:
            sys.modules[name] = MagicMock()

    # threedgrut.model.model (and its sub-imports that hit ncore)
    for name in ("threedgrut.model.model",):
        if name not in sys.modules:
            sys.modules[name] = MagicMock()

    # torch.utils.tensorboard (not installed in CPU pip venv)
    import torch.utils  # noqa: E402 — torch is available
    if not hasattr(torch.utils, "tensorboard") or not hasattr(
        getattr(torch.utils, "tensorboard", None), "writer"
    ):
        tb_stub = MagicMock()
        tb_stub.writer.SummaryWriter = MagicMock()
        sys.modules.setdefault("tensorboard", MagicMock())
        sys.modules.setdefault("torch.utils.tensorboard", tb_stub)
        sys.modules.setdefault("torch.utils.tensorboard.writer", tb_stub.writer)


_install_stubs()


def test_mcmc_get_add_cap_defaults_to_conf(real_conf):
    """T2.1: MCMCStrategy._get_add_cap() 默认返回 conf.strategy.add.max_n_gaussians.

    Uses __new__ to bypass __init__ (which JIT-compiles CUDA, unavailable on Mac).
    """
    from threedgrut.strategy.mcmc import MCMCStrategy

    strat = MCMCStrategy.__new__(MCMCStrategy)
    strat.conf = real_conf
    assert strat._get_add_cap() == real_conf.strategy.add.max_n_gaussians
