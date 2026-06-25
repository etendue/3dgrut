# SPDX-License-Identifier: Apache-2.0
"""Regression: ``Renderer.render_all`` must NOT shadow the module-level
``json`` import with a function-local ``import json``.

Bug history: a conditional ``if self.novel_view: import json`` lived inside
``render_all``. A function-local ``import`` binds ``json`` as a LOCAL for the
WHOLE function scope (CPython compiles every later ``json.xxx`` to LOAD_FAST),
so when ``novel_view`` was False the import never ran and the final
``json.dump(metrics_json, f)`` at the metrics.json write raised
``UnboundLocalError: cannot access local variable 'json'``. This crashed the
eval stage of every 3dgrut training that didn't take the novel_view path,
leaving ``metrics.json`` at 0 bytes (per-frame PSNR had to be grepped from the
stdout log) — observed 4+ times on inceptio.

This test pins the *cause* without needing a GPU/model: if ``json`` is a local
of ``render_all`` it appears in ``co_varnames``; a module-global reference does
not. Cheap, deterministic, GPU-free.
"""
import inspect
import json as _json

from threedgrut.render import Renderer


def test_render_all_does_not_localize_json():
    """A function-local ``import json`` (or ``json = ...``) would put 'json'
    in co_varnames and re-introduce the UnboundLocalError shadowing bug.

    NOTE: ``render_all`` is wrapped by ``@torch.no_grad()`` — inspecting
    ``render_all.__code__`` directly would see torch's ``decorate_context``
    wrapper (no 'json'), masking the bug. ``inspect.unwrap`` follows the
    ``__wrapped__`` chain to the real function body.
    """
    real = inspect.unwrap(Renderer.render_all)
    local_names = real.__code__.co_varnames
    assert "json" not in local_names, (
        "render_all has a function-local `json` (likely a stray `import json` "
        "inside the function) — this shadows the module-level import and makes "
        "the metrics.json write crash with UnboundLocalError when the branch "
        f"that binds it isn't taken. co_varnames={local_names!r}"
    )


def test_render_module_has_module_level_json():
    """The module-level ``import json`` must exist (render_all relies on it)."""
    import threedgrut.render as render_mod

    assert getattr(render_mod, "json", None) is _json
