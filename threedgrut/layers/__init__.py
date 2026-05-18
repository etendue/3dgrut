# SPDX-License-Identifier: Apache-2.0
"""threedgrut.layers package: dataclass-level utilities are always importable;
LayeredGaussians (which imports torch + MoG) is exported lazily so the layer
spec / registry can be inspected on a dev laptop without CUDA."""
from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.layers.registry import STANDARD_LAYERS, specs_from_config

__all__ = ["LayerSpec", "STANDARD_LAYERS", "specs_from_config"]

try:
    from threedgrut.layers.layered_model import LayeredGaussians  # noqa: F401

    __all__.append("LayeredGaussians")
except ImportError:
    # torch unavailable in this environment (e.g. CI lint, dev laptop). The
    # explicit submodule path `from threedgrut.layers.layered_model import
    # LayeredGaussians` still works when torch is installed.
    pass
