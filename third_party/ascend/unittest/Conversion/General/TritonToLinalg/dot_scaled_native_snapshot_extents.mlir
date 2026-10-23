// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// Keep address arithmetic in its original integer width, including before
// signed extension. Mask and other follow exactly the same bounded slice.
// The complete result buffer is populated before the whole native product.
// CHECK-LABEL: func.func @native_masked_384_columns
// CHECK-DAG: %[[ZERO:[^ ]+]] = arith.constant 0 : index
// CHECK-DAG: %[[TWO:[^ ]+]] = arith.constant 2 : index
// CHECK-DAG: %[[ROWS:[^ ]+]] = arith.constant 16 : index
// CHECK-DAG: %[[COLS:[^ ]+]] = arith.constant 384 : index
// CHECK-DAG: %[[WIDTH:[^ ]+]] = arith.constant 128 : index
// CHECK-NOT: arith.muli {{.*}} : tensor<16x384xi32>
// CHECK: %[[BUFFER:[^ ]+]] = memref.alloc() : memref<16x384xf8E4M3FN>
// CHECK: scf.for %[[ROW:[^ ]+]] = %[[ZERO]] to %[[ROWS]] step %[[TWO]]
// CHECK: scf.for %[[COL:[^ ]+]] = %[[ZERO]] to %[[COLS]] step %[[WIDTH]]
// CHECK: %[[PRODUCT:[^ ]+]] = arith.muli {{.*}} : tensor<2x128xi32>
// CHECK: %[[EXT:[^ ]+]] = arith.extsi %[[PRODUCT]] : tensor<2x128xi32> to tensor<2x128xi64>
// CHECK: %[[MASK:[^ ]+]] = tensor.extract_slice {{.*}}[%[[ROW]], %[[COL]]] [2, 128] [1, 1] : tensor<16x384xi1> to tensor<2x128xi1>
// CHECK: %[[OTHER:[^ ]+]] = tensor.extract_slice {{.*}}[%[[ROW]], %[[COL]]] [2, 128] [1, 1] : tensor<16x384xf8E4M3FN> to tensor<2x128xf8E4M3FN>
// CHECK: call @triton_indirect_load{{.*}}(%{{[^,]+}}, %[[EXT]], %[[MASK]], %[[OTHER]]) {{.*}} (memref<?xf8E4M3FN>, tensor<2x128xi64>, tensor<2x128xi1>, tensor<2x128xf8E4M3FN>) -> tensor<2x128xf8E4M3FN>
// CHECK: memref.subview %[[BUFFER]][%[[ROW]], %[[COL]]] [2, 128] [1, 1]
// CHECK: bufferization.materialize_in_destination
// CHECK-NOT: hivm.parallel_loop
// CHECK: bufferization.to_tensor %[[BUFFER]]
// CHECK-NOT: scf.for
// CHECK-NOT: tensor.insert_slice
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<16x384xf8E4M3FN>, tensor<384x32xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<16x32xf32>) -> tensor<16x32xf32>
// CHECK: return
func.func private @triton_indirect_load_masked(memref<?xf8E4M3FN>, tensor<16x384xi64>, tensor<16x384xi1>, tensor<16x384xf8E4M3FN>) -> tensor<16x384xf8E4M3FN> attributes {isVolatile = false}
func.func @native_masked_384_columns(%base: memref<?xf8E4M3FN>, %indices: tensor<16x384xi32>, %stride: tensor<16x384xi32>, %mask: tensor<16x384xi1>, %other: tensor<16x384xf8E4M3FN>, %b: tensor<384x32xf8E4M3FN>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %product = arith.muli %indices, %stride : tensor<16x384xi32>
  %offsets = arith.extsi %product : tensor<16x384xi32> to tensor<16x384xi64>
  %a = func.call @triton_indirect_load_masked(%base, %offsets, %mask, %other) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<16x384xi64>, tensor<16x384xi1>, tensor<16x384xf8E4M3FN>) -> tensor<16x384xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x384xf8E4M3FN> * tensor<384x32xf8E4M3FN> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}


// -----

// Preserve a legal unsliced read when an exact two-row partition is unavailable.
// CHECK-LABEL: func.func @native_snapshot_rows_1
// CHECK-NOT: scf.for
// CHECK-NOT: memref.alloc
// CHECK: call @triton_indirect_load{{.*}} -> tensor<1x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<1x128xf8E4M3FN>, tensor<128x32xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<1x32xf32>) -> tensor<1x32xf32>
// CHECK-NOT: scf.for
// CHECK-NOT: memref.alloc
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<1x128xi64>) -> tensor<1x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_snapshot_rows_1(%base: memref<?xf8E4M3FN>, %offsets: tensor<1x128xi64>, %b: tensor<128x32xf8E4M3FN>, %c: tensor<1x32xf32>) -> tensor<1x32xf32> {
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<1x128xi64>) -> tensor<1x128xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<1x128xf8E4M3FN> * tensor<128x32xf8E4M3FN> -> tensor<1x32xf32>
  return %d : tensor<1x32xf32>
}

// -----

// Preserve a legal unsliced read when an exact two-row partition is unavailable.
// CHECK-LABEL: func.func @native_snapshot_rows_3
// CHECK-NOT: scf.for
// CHECK-NOT: memref.alloc
// CHECK: call @triton_indirect_load{{.*}} -> tensor<3x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<3x128xf8E4M3FN>, tensor<128x32xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<3x32xf32>) -> tensor<3x32xf32>
// CHECK-NOT: scf.for
// CHECK-NOT: memref.alloc
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<3x128xi64>) -> tensor<3x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_snapshot_rows_3(%base: memref<?xf8E4M3FN>, %offsets: tensor<3x128xi64>, %b: tensor<128x32xf8E4M3FN>, %c: tensor<3x32xf32>) -> tensor<3x32xf32> {
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<3x128xi64>) -> tensor<3x128xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<3x128xf8E4M3FN> * tensor<128x32xf8E4M3FN> -> tensor<3x32xf32>
  return %d : tensor<3x32xf32>
}
