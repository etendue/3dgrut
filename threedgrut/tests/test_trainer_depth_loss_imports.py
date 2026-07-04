# SPDX-License-Identifier: Apache-2.0
"""Source-level regression guard for the Stage 11 depth-loss import scope.

trainer.py cannot be imported on Mac (it pulls in the CUDA 3dgrt/3dgut
tracers), so the spec/quality reviews and unit tests could only ast.parse it —
which does NOT catch a NameError from a function-local import being used in a
different method.

That exact bug shipped in f512fb4 (T11.A2 review fix I-2): it moved
``compute_bg_lidar_loss`` import from ``get_losses`` (where it is used as a
bare name) into ``init_depth_losses`` (a different method's local scope). The
name was then out of scope in ``get_losses`` → ``NameError`` that only
surfaced at the A800 1k smoke (T11.C2), wasting a launch.

This guard pins the fix: any depth_prior symbol used as a bare name in
``get_losses`` MUST be importable at module scope (or be a ``self.`` attribute).
It parses the source statically, so it runs on Mac CPU with no torch/CUDA.
"""

from __future__ import annotations

import ast
from pathlib import Path

TRAINER = Path(__file__).resolve().parents[1] / "trainer.py"


def _module_level_imported_names(tree: ast.Module) -> set[str]:
    """Names bound by ``import`` / ``from ... import`` at MODULE top level only."""
    names: set[str] = set()
    for node in tree.body:  # module body only — not nested in functions/classes
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
    return names


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in trainer.py")


def test_compute_bg_lidar_loss_module_level_imported():
    """compute_bg_lidar_loss must be a module-level import (used bare in get_losses)."""
    tree = ast.parse(TRAINER.read_text())
    mod_names = _module_level_imported_names(tree)
    assert "compute_bg_lidar_loss" in mod_names, (
        "compute_bg_lidar_loss must be imported at module scope in trainer.py — "
        "get_losses uses it as a bare name. A function-local import (e.g. inside "
        "init_depth_losses) is NOT visible in get_losses → NameError at runtime "
        "(regression f512fb4, caught at A800 smoke T11.C2)."
    )


def test_get_losses_bare_depth_symbols_are_module_imported():
    """Every depth_prior symbol used bare in get_losses resolves at module scope.

    Catches the general class: a depth-loss helper used as a bare Name in
    get_losses but only imported in some other function's local scope.
    """
    tree = ast.parse(TRAINER.read_text())
    mod_names = _module_level_imported_names(tree)
    get_losses = _find_function(tree, "get_losses")

    # Bare names referenced (loaded) anywhere in get_losses.
    used = {n.id for n in ast.walk(get_losses) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    # Names bound locally inside get_losses (assignments / local imports) are fine.
    local_bound: set[str] = set()
    for node in ast.walk(get_losses):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                local_bound.add(alias.asname or alias.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            local_bound.add(node.id)

    # The depth_prior public helpers get_losses may call bare.
    depth_helpers = {"compute_bg_lidar_loss"}
    for sym in depth_helpers:
        if sym in used and sym not in local_bound:
            assert sym in mod_names, (
                f"{sym} is used bare in get_losses but is neither a module-level "
                f"import nor bound locally — NameError at runtime."
            )
