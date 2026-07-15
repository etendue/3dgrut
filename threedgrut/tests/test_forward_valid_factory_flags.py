# SPDX-License-Identifier: Apache-2.0
"""PIN-MASK-1: train, val, and test factories must forward the same flag."""

from __future__ import annotations

import ast
from pathlib import Path

INIT = Path(__file__).resolve().parents[1] / "datasets" / "__init__.py"
FLAG = "mask_forward_invalid_pixels"


def _keyword_occurrences(tree: ast.Module, function_name: str, keyword: str) -> list[ast.keyword]:
    fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    return [
        kw
        for call in ast.walk(fn)
        if isinstance(call, ast.Call)
        for kw in call.keywords
        if kw.arg == keyword
    ]


def test_train_val_and_test_forward_mask_flag():
    tree = ast.parse(INIT.read_text())
    # make() constructs both train and val NCoreDataset objects.
    assert len(_keyword_occurrences(tree, "make", FLAG)) == 2
    assert len(_keyword_occurrences(tree, "make_test", FLAG)) == 1


def test_all_factory_paths_read_same_config_key_with_false_default():
    tree = ast.parse(INIT.read_text())
    occurrences = _keyword_occurrences(tree, "make", FLAG) + _keyword_occurrences(tree, "make_test", FLAG)
    assert len(occurrences) == 3
    for kw in occurrences:
        call = kw.value
        assert isinstance(call, ast.Call)
        assert isinstance(call.func, ast.Attribute) and call.func.attr == "get"
        assert len(call.args) == 2
        assert isinstance(call.args[0], ast.Constant) and call.args[0].value == FLAG
        assert isinstance(call.args[1], ast.Constant) and call.args[1].value is False
