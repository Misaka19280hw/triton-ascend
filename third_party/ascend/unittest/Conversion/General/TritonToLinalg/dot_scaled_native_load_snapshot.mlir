// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// Keep address arithmetic in its original integer width, including before
// signed extension. Mask and other follow exactly the same bounded slice.
// The complete result buffer is populated before the whole native product.
// CHECK-LABEL: func.func @native_masked_bounded_snapshot
// CHECK-NOT: arith.muli {{.*}} : tensor<16x512xi32>
// CHECK: %[[BUFFER:[^ ]+]] = memref.alloc() : memref<16x512xf8E4M3FN>
// CHECK: scf.for
// CHECK: scf.for
// CHECK: %[[PRODUCT:[^ ]+]] = arith.muli {{.*}} : tensor<1x256xi32>
// CHECK: %[[EXT:[^ ]+]] = arith.extsi %[[PRODUCT]] : tensor<1x256xi32> to tensor<1x256xi64>
// CHECK: call @triton_indirect_load{{.*}}(%{{[^,]+}}, %[[EXT]], {{.*}}) {{.*}} (memref<?xf8E4M3FN>, tensor<1x256xi64>, tensor<1x256xi1>, tensor<1x256xf8E4M3FN>) -> tensor<1x256xf8E4M3FN>
// CHECK: memref.subview %[[BUFFER]]{{.*}} [1, 256] [1, 1]
// CHECK: bufferization.materialize_in_destination
// CHECK-NOT: hivm.parallel_loop
// CHECK: bufferization.to_tensor %[[BUFFER]]
// CHECK-NOT: scf.for
// CHECK-NOT: tensor.insert_slice
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<16x512xf8E4M3FN>, tensor<512x32xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<16x32xf32>) -> tensor<16x32xf32>
// CHECK: return
func.func private @triton_indirect_load_masked(memref<?xf8E4M3FN>, tensor<16x512xi64>, tensor<16x512xi1>, tensor<16x512xf8E4M3FN>) -> tensor<16x512xf8E4M3FN> attributes {isVolatile = false}
func.func @native_masked_bounded_snapshot(%base: memref<?xf8E4M3FN>, %indices: tensor<16x512xi32>, %stride: tensor<16x512xi32>, %mask: tensor<16x512xi1>, %other: tensor<16x512xf8E4M3FN>, %b: tensor<512x32xf8E4M3FN>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %product = arith.muli %indices, %stride : tensor<16x512xi32>
  %offsets = arith.extsi %product : tensor<16x512xi32> to tensor<16x512xi64>
  %a = func.call @triton_indirect_load_masked(%base, %offsets, %mask, %other) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<16x512xi64>, tensor<16x512xi1>, tensor<16x512xf8E4M3FN>) -> tensor<16x512xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x512xf8E4M3FN> * tensor<512x32xf8E4M3FN> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// A shared address calculation must remain available to its other consumer.
// Cloning its bounded computation for the read must not erase the original.
// CHECK-LABEL: func.func @native_shared_address
// CHECK: %[[ORIGINAL:[^ ]+]] = arith.muli {{.*}} : tensor<16x512xi32>
// CHECK: %[[BUFFER:[^ ]+]] = memref.alloc() : memref<16x512xf8E4M3FN>
// CHECK: scf.for
// CHECK: scf.for
// CHECK: arith.muli {{.*}} : tensor<1x256xi32>
// CHECK: call @triton_indirect_load{{.*}} -> tensor<1x256xf8E4M3FN>
// CHECK: bufferization.materialize_in_destination
// CHECK: bufferization.to_tensor %[[BUFFER]]
// CHECK-NOT: tensor.insert_slice
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<16x512xf8E4M3FN>, tensor<512x32xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<16x32xf32>) -> tensor<16x32xf32>
// CHECK: return {{.*}}, %[[ORIGINAL]] : tensor<16x32xf32>, tensor<16x512xi32>
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<16x512xi64>) -> tensor<16x512xf8E4M3FN> attributes {isVolatile = false}
func.func @native_shared_address(%base: memref<?xf8E4M3FN>, %indices: tensor<16x512xi32>, %stride: tensor<16x512xi32>, %b: tensor<512x32xf8E4M3FN>, %c: tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x512xi32>) {
  %product = arith.muli %indices, %stride : tensor<16x512xi32>
  %offsets = arith.extsi %product : tensor<16x512xi32> to tensor<16x512xi64>
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<16x512xi64>) -> tensor<16x512xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x512xf8E4M3FN> * tensor<512x32xf8E4M3FN> -> tensor<16x32xf32>
  return %d, %product : tensor<16x32xf32>, tensor<16x512xi32>
}
