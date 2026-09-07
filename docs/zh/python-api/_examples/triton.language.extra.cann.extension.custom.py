from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.runtime import driver

# CustomOp connects a device function to a Triton kernel. This example adds
# two int32 vectors using the matching precompiled device function in custom_op/.
# Keep this file beside the custom_op/ directory from the repository. Its C++
# source includes rebuild commands; running the example does not rebuild it.
# This also supports CANN 8.5, whose CCE compiler cannot export this bitcode.
# Before pytest, configure a compatible Triton-Ascend/CANN environment.
# From the repository root:
# python3 -m pytest --import-mode=importlib -q \
#     docs/zh/python-api/_examples/triton.language.extra.cann.extension.custom.py


@triton.jit
def custom_example_kernel(x_ptr, y_ptr, out_ptr):
    # 64 int32 elements fill one 256-byte vector on A5.
    offsets = tl.arange(0, 64)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    result = al.custom("custom_example_op", x, y, out=tl.full((64, ), 0, tl.int32))
    tl.store(out_ptr + offsets, result)


def test_custom():
    # Load bitcode for the same device architecture as the Triton kernel.
    target = driver.active.get_current_target().arch
    if target.startswith(("Ascend950", "Ascend910_95")):
        bitcode_name = "custom_add_a5.bc"
    elif target.startswith(("Ascend910B", "Ascend910_93")):
        bitcode_name = "custom_add_a3.bc"
    else:
        raise RuntimeError(f"This example supports A2/A3 and A5, not {target}")

    bitcode_path = Path(__file__).resolve().parent / "custom_op" / bitcode_name

    @al.register_custom_op
    class custom_example_op:
        core = al.CORE.VECTOR
        pipe = al.PIPE.PIPE_V
        mode = al.MODE.SIMD
        symbol = "custom_add_int32"
        bitcode = str(bitcode_path)

    x = torch.arange(64, dtype=torch.int32) - 32
    y = torch.arange(64, dtype=torch.int32) * 3 - 7
    output = torch.full((64, ), -123, dtype=torch.int32, device="npu")
    custom_example_kernel[(1, )](x.to("npu"), y.to("npu"), output)
    torch.testing.assert_close(output.cpu(), x + y, rtol=0, atol=0)


if __name__ == "__main__":
    test_custom()
