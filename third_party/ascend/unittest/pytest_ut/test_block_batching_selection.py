# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Automatic requests compile the baseline once while LLVM scoring is retired."""
import importlib.util
from pathlib import Path
from unittest.mock import Mock
from block_batching_test_utils import assign_npu  # noqa: F401 - CPU-only fixture


def test_retired_selector_compiles_only_f1():
    path = Path(__file__).resolve().parents[2] / "backend/block_auto.py"
    spec = importlib.util.spec_from_file_location("retired_block_auto", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compile_variant = Mock(return_value=object())
    options = dict(block_batching_factor="auto", block_batching_axis=2, block_batching_grid=(8, 4, 2),
                   block_auto_contract="{}", num_warps=4)
    result = module.compile_auto_bundle(object(), object(), options, compile_variant)
    assert result is compile_variant.return_value
    compile_variant.assert_called_once_with(dict(block_batching_factor=1, block_batching_axis=-1, num_warps=4))
    assert options["block_batching_factor"] == "auto"
