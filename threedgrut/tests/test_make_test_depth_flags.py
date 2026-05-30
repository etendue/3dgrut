# SPDX-License-Identifier: Apache-2.0
"""Source-level regression guard: make_test must forward the depth load flags.

datasets/__init__.py has THREE dataset factory paths — make_train / make_val /
make_test. C1/D1 wired load_lidar_depth_map / load_depth_prior / load_aux_masks
into make_train + make_val, but MISSED make_test. render.py's offline eval
(render_all → datasets.make_test) therefore built its NCoreDataset WITHOUT
load_lidar_depth_map, so image_infos never carried lidar_depth_map and
mean_lidar_psnr came out ABSENT from the G1 30k metrics.json — a silent gap
that only surfaced after a 107-minute run (the CLAUDE.md §B L51-54 trap: a
metric's INPUT missing from one of the parallel eval paths).

This guard parses datasets/__init__.py statically (it can't be imported on Mac
— pulls in the ncore SDK) and asserts make_test forwards the same depth flags
as make_train.
"""
from __future__ import annotations

import ast
from pathlib import Path

INIT = Path(__file__).resolve().parents[1] / "datasets" / "__init__.py"

REQUIRED_FLAGS = {"load_aux_masks", "load_lidar_depth_map", "load_depth_prior"}


def _kwargs_in_function(tree: ast.Module, func_name: str) -> set[str]:
    """All keyword-argument names passed in any call inside the named function."""
    fn = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func_name),
        None,
    )
    assert fn is not None, f"{func_name} not found in datasets/__init__.py"
    kwargs: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg:
                    kwargs.add(kw.arg)
    return kwargs


def test_make_test_forwards_depth_flags():
    """make_test must pass load_lidar_depth_map/load_depth_prior/load_aux_masks
    to NCoreDataset — else render.py eval can't compute mean_lidar_psnr."""
    tree = ast.parse(INIT.read_text())
    test_kwargs = _kwargs_in_function(tree, "make_test")
    missing = REQUIRED_FLAGS - test_kwargs
    assert not missing, (
        f"make_test does not forward {sorted(missing)} to its dataset — "
        f"render.py eval will silently drop these inputs (mean_lidar_psnr absent "
        f"from metrics.json). Mirror make_train/make_val."
    )


def test_make_train_and_make_test_agree_on_depth_flags():
    """The three factory paths must stay in sync on the depth flags."""
    tree = ast.parse(INIT.read_text())
    train_kwargs = _kwargs_in_function(tree, "make")  # make_train alias / main factory
    test_kwargs = _kwargs_in_function(tree, "make_test")
    for flag in REQUIRED_FLAGS:
        if flag in train_kwargs:
            assert flag in test_kwargs, (
                f"{flag} is forwarded in make() but NOT make_test() — the two "
                f"factory paths have diverged on depth-supervision inputs."
            )
