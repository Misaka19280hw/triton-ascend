// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL: func.func @all_sources_tileable
// CHECK-NOT: memref.alloc() : memref<64x128xf8E4M3FN>
// CHECK: scf.for
// CHECK-COUNT-2: memref.alloc() : memref<32x128xf8E4M3FN>
// CHECK: linalg.transpose
// CHECK-SAME: tensor<32x128xf8E4M3FN>
// CHECK: hfusion.matmul_mx
func.func @all_sources_tileable(%base0: memref<?xf8E4M3FN>, %base1: memref<?xf8E4M3FN>, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %s0 = memref.reinterpret_cast %base0 to offset: [0], sizes: [64, 128], strides: [128, 1] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN>
  %s1 = memref.reinterpret_cast %base1 to offset: [0], sizes: [64, 128], strides: [128, 1] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN>
  %buf0 = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %s0, %buf0 {isVolatile = false} : memref<64x128xf8E4M3FN> to memref<64x128xf8E4M3FN>
  %t0 = bufferization.to_tensor %buf0 restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %buf1 = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %s1, %buf1 {isVolatile = false} : memref<64x128xf8E4M3FN> to memref<64x128xf8E4M3FN>
  %t1 = bufferization.to_tensor %buf1 restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %sum = arith.addf %t0, %t1 : tensor<64x128xf8E4M3FN>
  %empty = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = linalg.transpose ins(%sum : tensor<64x128xf8E4M3FN>) outs(%empty : tensor<128x64xf8E4M3FN>) permutation = [1, 0]
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @mixed_snapshot_source
// CHECK-COUNT-2: memref.alloc() : memref<64x128xf8E4M3FN>
// CHECK: linalg.transpose
// CHECK-SAME: tensor<64x128xf8E4M3FN>
// CHECK: scf.for
// CHECK-NOT: memref.alloc() : memref<32x128xf8E4M3FN>
// CHECK: tensor.extract_slice
// CHECK: hfusion.matmul_mx
func.func @mixed_snapshot_source(%base0: memref<?xf8E4M3FN>, %base1: memref<?xf8E4M3FN>, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %s0 = memref.reinterpret_cast %base0 to offset: [0], sizes: [64, 128], strides: [128, 1] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN>
  %s1 = memref.reinterpret_cast %base1 to offset: [0], sizes: [64, 128], strides: [128, 1] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN>
  %buf0 = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %s0, %buf0 {isVolatile = false} : memref<64x128xf8E4M3FN> to memref<64x128xf8E4M3FN>
  %t0 = bufferization.to_tensor %buf0 restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %buf1 = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %s1, %buf1 {isVolatile = true} : memref<64x128xf8E4M3FN> to memref<64x128xf8E4M3FN>
  %t1 = bufferization.to_tensor %buf1 restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %sum = arith.addf %t0, %t1 : tensor<64x128xf8E4M3FN>
  %empty = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = linalg.transpose ins(%sum : tensor<64x128xf8E4M3FN>) outs(%empty : tensor<128x64xf8E4M3FN>) permutation = [1, 0]
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}
