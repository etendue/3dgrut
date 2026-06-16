# SPDX-License-Identifier: Apache-2.0
"""Regression: render_all UnboundLocalError on `json`.

E2.1 (530a27c) added a local `import json` inside render_all's
`if self.novel_view:` branch, shadowing the module-level json (render.py L16).
With novel_view=False that branch is skipped, so the later json.dump() at the
metrics.json write raised `UnboundLocalError: cannot access local variable 'json'`
— crashing E3.6 B1 baseline at on_training_end eval (ckpt written, metrics lost).

Static AST check (no import → no CUDA extension build on Mac): render_all must
rely on the module-level json import only.
"""
from __future__ import annotations

import ast
from pathlib import Path


def test_render_all_does_not_shadow_module_json():
    render_py = Path(__file__).resolve().parents[1] / "render.py"
    tree = ast.parse(render_py.read_text())
    render_all = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "render_all"),
        None,
    )
    assert render_all is not None, "render_all not found in render.py"
    local_json = [
        n for n in ast.walk(render_all)
        if isinstance(n, ast.Import) and any(a.name == "json" for a in n.names)
    ]
    assert not local_json, (
        "render_all must not locally `import json` (shadows module-level json → "
        "UnboundLocalError on the novel_view=False path). Use module-level import."
    )
    # and the module-level import must still be present
    mod_json = [
        n for n in tree.body
        if isinstance(n, ast.Import) and any(a.name == "json" for a in n.names)
    ]
    assert mod_json, "render.py must keep its module-level `import json`"
