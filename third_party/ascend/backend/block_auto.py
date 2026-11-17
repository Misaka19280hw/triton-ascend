# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Keep automatic requests on the baseline while the LLVM selector is retired."""


def compile_auto_bundle(backend, src, options, compile_variant):
    values = {key: value for key, value in options.items() if key not in ("block_auto_contract", "block_batching_grid")}
    return compile_variant(dict(values, block_batching_factor=1, block_batching_axis=-1))
