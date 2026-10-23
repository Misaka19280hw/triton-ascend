# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _bulk_snapshot(A, B, Snapshot, C, stride_bk, stride_bn, MODE: tl.constexpr):
    m = tl.arange(0, 16)
    n = tl.arange(0, 64)
    k = tl.arange(0, 64)
    bp = B + k[:, None] * stride_bk + n[None, :] * stride_bn
    b = tl.load(bp, volatile=MODE == "volatile")
    a = tl.load(A + m[:, None] * 64 + k[None, :])
    if MODE == "alias_store":
        tl.store(bp, tl.full((64, 64), 0, tl.float16))
    if MODE == "shared":
        tl.store(Snapshot + k[:, None] * 64 + n[None, :], b)
    c = tl.dot_scaled(a, None, "fp16", b, None, "fp16")
    # Keep memory lowering in the unblocked path used by native MX kernels.
    tl.store(C + m[:, None] * 64 + n[None, :], c, cache_modifier=".wt")


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "strided"])
@pytest.mark.parametrize("mode", ["plain", "alias_store", "shared", "volatile"])
def test_dot_scaled_bulk_snapshot(layout, mode):
    a = ((torch.arange(16 * 64).reshape(16, 64) % 11 - 5) / 8).half()
    b = ((torch.arange(64 * 64).reshape(64, 64) % 13 - 6) / 4).half()
    expected = a.float() @ b.float()
    strides = {"contiguous": (64, 1), "transpose": (1, 64), "strided": (160, 2)}[layout]
    bn = torch.empty_strided((64, 64), strides, dtype=torch.float16, device="npu")
    bn.copy_(b)
    snapshot = torch.empty((64, 64), dtype=torch.float16, device="npu")
    actual = torch.empty((16, 64), dtype=torch.float32, device="npu")
    _bulk_snapshot[(1, )](a.to("npu"), bn, snapshot, actual, *strides, mode)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    if mode == "alias_store":
        torch.testing.assert_close(bn.cpu(), torch.zeros_like(b), atol=0, rtol=0)
    if mode == "shared":
        torch.testing.assert_close(snapshot.cpu(), b, atol=0, rtol=0)
