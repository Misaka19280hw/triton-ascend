// Device implementation for the CustomOp documentation example (64 int32s).
// The memref descriptor carries an address, offset, shape and stride.
//
// The accompanying bitcode was built with CANN 9.1.0 (clang 15.0.5,
// 2026-07-30T20:53:21+08:00). To rebuild, run from this directory:
// custom_resource_dir=$(bisheng -print-resource-dir)
// bisheng -x cce --cce-aicore-arch=dav-c220-vec --cce-aicore-only -O2 -c
//   -emit-llvm -g0 -fdebug-compilation-dir=.
//   -fdebug-prefix-map="$custom_resource_dir"=bisheng-resource custom_add.cpp
//   -o custom_add_a3.bc
// bisheng -x cce --cce-aicore-arch=dav-c310-vec --cce-aicore-only -O2 -c
//   -emit-llvm -g0 -fdebug-compilation-dir=.
//   -fdebug-prefix-map="$custom_resource_dir"=bisheng-resource custom_add.cpp
//   -o custom_add_a5.bc
// Join the continued comment lines of each compile command before running.
// The debug prefix mapping keeps local toolchain paths out of the bitcode.
// The A2/A3 and A5 implementations need separate, architecture-matched bitcode.
// CANN 8.5 can consume the A2/A3 bitcode without compiling this CCE source.

#define __aiv__ [aicore]

template <typename T, size_t Dim> struct memref_t {
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
  auto *a = src0->aligned + src0->offset;
  auto *b = src1->aligned + src1->offset;
  auto *out = dst->aligned + dst->offset;
#if defined(__DAV_C310__)
  __VEC_SCOPE__ {
    vector_s32 va, vb, result;
    uint32_t count = dst->sizes[0];
    vector_bool mask = plt_b32(count, POST_UPDATE);
    vlds(va, a, 0, NORM);
    vlds(vb, b, 0, NORM);
    vadd(result, va, vb, mask, MODE_ZEROING);
    vsts(result, out, 0, NORM_B32, mask);
  }
#else
  set_mask_count();
  set_vector_mask(0, dst->sizes[0]);
  vadd(out, a, b, 1, 1, 1, 1, 8, 8, 8);
  set_mask_norm();
#endif
}
