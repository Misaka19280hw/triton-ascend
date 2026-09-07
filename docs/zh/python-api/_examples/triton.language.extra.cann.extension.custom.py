import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

# CustomOp connects a device function to a Triton kernel. This example adds
# two int32 vectors using a device function compiled from the source below.
# It targets dav-c220-vec (A2/A3), not A5. Before running pytest, configure
# a compatible Triton-Ascend/CANN environment with ccec available on PATH.
# From the repository root:
# python3 -m pytest --import-mode=importlib -q \
#     docs/zh/python-api/_examples/triton.language.extra.cann.extension.custom.py

# The memref descriptor carries each tensor's address, offset, shape and stride.
# The device function handles the fixed 32-element vectors used by this test.
DEVICE_SOURCE = r"""
#define __aiv__ [aicore]

template <typename T, size_t Dim>
struct memref_t {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[Dim];
  int64_t strides[Dim];
};

extern "C" __aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_add_int32(memref_t<__ubuf__ int32_t, 1> *src0,
                             memref_t<__ubuf__ int32_t, 1> *src1,
                             memref_t<__ubuf__ int32_t, 1> *dst) {
  set_mask_count();
  set_vector_mask(0, dst->sizes[0]);
  vadd(dst->aligned + dst->offset, src0->aligned + src0->offset,
       src1->aligned + src1->offset, 1, 1, 1, 1, 8, 8, 8);
  set_mask_norm();
}
"""


@triton.jit
def custom_example_kernel(x_ptr, y_ptr, out_ptr):
    offsets = tl.arange(0, 32)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    result = al.custom("custom_example_op", x, y, out=tl.full((32, ), 0, tl.int32))
    tl.store(out_ptr + offsets, result)


def test_custom():
    # Build real device bitcode in a temporary directory, not a placeholder.
    with TemporaryDirectory() as directory:
        source_path = Path(directory) / "custom_add.cpp"
        bitcode_path = Path(directory) / "custom_add.bc"
        source_path.write_text(DEVICE_SOURCE, encoding="utf-8")
        subprocess.run([
            "ccec", "-x", "cce", "--cce-aicore-arch=dav-c220-vec", "--cce-aicore-only", "-c", "-emit-llvm",
            str(source_path), "-o",
            str(bitcode_path)
        ], check=True)

        @al.register_custom_op
        class custom_example_op:
            core = al.CORE.VECTOR
            pipe = al.PIPE.PIPE_V
            mode = al.MODE.SIMD
            symbol = "custom_add_int32"
            bitcode = str(bitcode_path)

        x = torch.arange(32, dtype=torch.int32) - 16
        y = torch.arange(32, dtype=torch.int32) * 3 - 7
        output = torch.full((32, ), -123, dtype=torch.int32, device="npu")
        custom_example_kernel[(1, )](x.to("npu"), y.to("npu"), output)
        torch.testing.assert_close(output.cpu(), x + y, rtol=0, atol=0)


if __name__ == "__main__":
    test_custom()
