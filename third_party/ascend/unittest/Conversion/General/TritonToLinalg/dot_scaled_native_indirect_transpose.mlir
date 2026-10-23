// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s --implicit-check-not=hivm.parallel_loop --implicit-check-not=tensor.insert_slice

// An indirect snapshot below a transpose has one logical user even while
// dialect conversion retains both the original tt.trans and its replacement.
// True shared users must retain the complete original snapshot.

// -----

// CHECK-LABEL: func.func @native_indirect_original_transpose
// CHECK-SAME: (%[[BASE:[^:]+]]: memref<?xf8E4M3FN>, %[[OFFSETS:[^:]+]]: tensor<64x128xi64>, %[[LHS:[^:]+]]: tensor<32x128xf8E4M3FN>, %[[C:[^:]+]]: tensor<32x64xf32>)
// CHECK-DAG: %[[C64:[^ ]+]] = arith.constant 64 : index
// CHECK-DAG: %[[C2:[^ ]+]] = arith.constant 2 : index
// CHECK-DAG: %[[C0:[^ ]+]] = arith.constant 0 : index
// CHECK: %[[SNAP:[^ ]+]] = memref.alloc() : memref<64x128xf8E4M3FN>
// CHECK: scf.for %[[ROW:[^ ]+]] = %[[C0]] to %[[C64]] step %[[C2]] {
// CHECK: %[[OFF_TILE:[^ ]+]] = tensor.extract_slice %[[OFFSETS]][%[[ROW]], 0] [2, 128] [1, 1] : tensor<64x128xi64> to tensor<2x128xi64>
// CHECK: %[[READ:[^ ]+]] = {{(func\.)?}}call @triton_indirect_load{{[^ (]*}}(%[[BASE]], %[[OFF_TILE]]) {isVolatile = false}
// CHECK-SAME: -> tensor<2x128xf8E4M3FN>
// CHECK: %[[VIEW:[^ ]+]] = memref.subview %[[SNAP]][%[[ROW]], 0] [2, 128] [1, 1]
// CHECK: bufferization.materialize_in_destination %[[READ]] in writable %[[VIEW]]
// CHECK: } {ExtractedLoadOrStore}
// CHECK: %[[WHOLE:[^ ]+]] = bufferization.to_tensor %[[SNAP]] restrict writable : memref<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: %[[TRANS:[^ ]+]] = linalg.transpose ins(%[[WHOLE]] : tensor<64x128xf8E4M3FN>)
// CHECK-SAME: tensor<128x64xf8E4M3FN>) permutation = [1, 0]
// CHECK-NOT: scf.for
// CHECK: %[[RESULT:[^ ]+]] = hfusion.matmul_mx
// CHECK-SAME: ins(%[[LHS]], %[[TRANS]],
// CHECK-SAME: outs(%[[C]] : tensor<32x64xf32>) -> tensor<32x64xf32>
// CHECK-NOT: hfusion.matmul_mx
// CHECK-NOT: scf.for
// CHECK: return %[[RESULT]] : tensor<32x64xf32>
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_indirect_original_transpose(%base: memref<?xf8E4M3FN>, %offsets: tensor<64x128xi64>, %a: tensor<32x128xf8E4M3FN>, %c: tensor<32x64xf32>) -> tensor<32x64xf32> {
  %loaded = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  %b = tt.trans %loaded {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<32x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<32x64xf32>
  return %d : tensor<32x64xf32>
}

// -----

// CHECK-LABEL: func.func @native_indirect_converted_transpose
// CHECK-SAME: (%[[BASE:[^:]+]]: memref<?xf8E4M3FN>, %[[OFFSETS:[^:]+]]: tensor<64x128xi64>, %[[LHS:[^:]+]]: tensor<32x128xf8E4M3FN>, %[[C:[^:]+]]: tensor<32x64xf32>)
// CHECK-DAG: %[[C64:[^ ]+]] = arith.constant 64 : index
// CHECK-DAG: %[[C2:[^ ]+]] = arith.constant 2 : index
// CHECK-DAG: %[[C0:[^ ]+]] = arith.constant 0 : index
// CHECK: %[[SNAP:[^ ]+]] = memref.alloc() : memref<64x128xf8E4M3FN>
// CHECK: scf.for %[[ROW:[^ ]+]] = %[[C0]] to %[[C64]] step %[[C2]] {
// CHECK: %[[OFF_TILE:[^ ]+]] = tensor.extract_slice %[[OFFSETS]][%[[ROW]], 0] [2, 128] [1, 1] : tensor<64x128xi64> to tensor<2x128xi64>
// CHECK: %[[READ:[^ ]+]] = {{(func\.)?}}call @triton_indirect_load{{[^ (]*}}(%[[BASE]], %[[OFF_TILE]]) {isVolatile = false}
// CHECK-SAME: -> tensor<2x128xf8E4M3FN>
// CHECK: %[[VIEW:[^ ]+]] = memref.subview %[[SNAP]][%[[ROW]], 0] [2, 128] [1, 1]
// CHECK: bufferization.materialize_in_destination %[[READ]] in writable %[[VIEW]]
// CHECK: } {ExtractedLoadOrStore}
// CHECK: %[[WHOLE:[^ ]+]] = bufferization.to_tensor %[[SNAP]] restrict writable : memref<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: %[[TRANS:[^ ]+]] = linalg.transpose ins(%[[WHOLE]] : tensor<64x128xf8E4M3FN>)
// CHECK-SAME: tensor<128x64xf8E4M3FN>) permutation = [1, 0]
// CHECK-NOT: scf.for
// CHECK: %[[RESULT:[^ ]+]] = hfusion.matmul_mx
// CHECK-SAME: ins(%[[LHS]], %[[TRANS]],
// CHECK-SAME: outs(%[[C]] : tensor<32x64xf32>) -> tensor<32x64xf32>
// CHECK-NOT: hfusion.matmul_mx
// CHECK-NOT: scf.for
// CHECK: return %[[RESULT]] : tensor<32x64xf32>
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_indirect_converted_transpose(%base: memref<?xf8E4M3FN>, %offsets: tensor<64x128xi64>, %a: tensor<32x128xf8E4M3FN>, %c: tensor<32x64xf32>) -> tensor<32x64xf32> {
  %loaded = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = linalg.transpose ins(%loaded : tensor<64x128xf8E4M3FN>) outs(%init : tensor<128x64xf8E4M3FN>) permutation = [1, 0]
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<32x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<32x64xf32>
  return %d : tensor<32x64xf32>
}

// -----

// CHECK-LABEL: func.func @native_indirect_transpose_two_dots
// CHECK-SAME: (%[[BASE:[^:]+]]: memref<?xf8E4M3FN>, %[[OFFSETS:[^:]+]]: tensor<64x128xi64>, %[[LHS:[^:]+]]: tensor<32x128xf8E4M3FN>, %[[C:[^:]+]]: tensor<32x64xf32>)
// CHECK-NOT: memref.alloc
// CHECK-NOT: scf.for
// CHECK: %[[WHOLE:[^ ]+]] = {{(func\.)?}}call @triton_indirect_load(%[[BASE]], %[[OFFSETS]]) {isVolatile = false}
// CHECK-SAME: -> tensor<64x128xf8E4M3FN>
// CHECK-NOT: memref.alloc
// CHECK-NOT: scf.for
// CHECK: %[[TRANS:[^ ]+]] = linalg.transpose ins(%[[WHOLE]] : tensor<64x128xf8E4M3FN>)
// CHECK-SAME: tensor<128x64xf8E4M3FN>) permutation = [1, 0]
// CHECK-NOT: memref.alloc
// CHECK-NOT: scf.for
// CHECK: %[[FIRST:[^ ]+]] = hfusion.matmul_mx
// CHECK-SAME: ins(%[[LHS]], %[[TRANS]],
// CHECK-SAME: outs(%[[C]] : tensor<32x64xf32>) -> tensor<32x64xf32>
// CHECK-NOT: memref.alloc
// CHECK-NOT: scf.for
// CHECK: %[[SECOND:[^ ]+]] = hfusion.matmul_mx
// CHECK-SAME: ins(%[[LHS]], %[[TRANS]],
// CHECK-SAME: outs(%[[FIRST]] : tensor<32x64xf32>) -> tensor<32x64xf32>
// CHECK-NOT: memref.alloc
// CHECK-NOT: scf.for
// CHECK: return %[[FIRST]], %[[SECOND]] : tensor<32x64xf32>, tensor<32x64xf32>
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_indirect_transpose_two_dots(%base: memref<?xf8E4M3FN>, %offsets: tensor<64x128xi64>, %a: tensor<32x128xf8E4M3FN>, %c: tensor<32x64xf32>) -> (tensor<32x64xf32>, tensor<32x64xf32>) {
  %loaded = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  %b = tt.trans %loaded {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<32x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<32x64xf32>
  %e = tt.dot_scaled %a, %b, %d lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<32x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<32x64xf32>
  return %d, %e : tensor<32x64xf32>, tensor<32x64xf32>
}

// -----

// CHECK-LABEL: func.func @native_indirect_transpose_shared_source
// CHECK-SAME: (%[[BASE:[^:]+]]: memref<?xf8E4M3FN>, %[[OFFSETS:[^:]+]]: tensor<64x128xi64>, %[[LHS:[^:]+]]: tensor<32x128xf8E4M3FN>, %[[C:[^:]+]]: tensor<32x64xf32>)
// CHECK-NOT: memref.alloc
// CHECK-NOT: scf.for
// CHECK: %[[WHOLE:[^ ]+]] = {{(func\.)?}}call @triton_indirect_load(%[[BASE]], %[[OFFSETS]]) {isVolatile = false}
// CHECK-SAME: -> tensor<64x128xf8E4M3FN>
// CHECK-NOT: memref.alloc
// CHECK-NOT: scf.for
// CHECK: %[[TRANS:[^ ]+]] = linalg.transpose ins(%[[WHOLE]] : tensor<64x128xf8E4M3FN>)
// CHECK-SAME: tensor<128x64xf8E4M3FN>) permutation = [1, 0]
// CHECK-NOT: memref.alloc
// CHECK-NOT: scf.for
// CHECK: %[[FIRST:[^ ]+]] = hfusion.matmul_mx
// CHECK-SAME: ins(%[[LHS]], %[[TRANS]],
// CHECK-SAME: outs(%[[C]] : tensor<32x64xf32>) -> tensor<32x64xf32>
// CHECK-NOT: memref.alloc
// CHECK-NOT: scf.for
// CHECK: return %[[FIRST]], %[[WHOLE]] : tensor<32x64xf32>, tensor<64x128xf8E4M3FN>
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_indirect_transpose_shared_source(%base: memref<?xf8E4M3FN>, %offsets: tensor<64x128xi64>, %a: tensor<32x128xf8E4M3FN>, %c: tensor<32x64xf32>) -> (tensor<32x64xf32>, tensor<64x128xf8E4M3FN>) {
  %loaded = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  %b = tt.trans %loaded {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<32x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<32x64xf32>
  return %d, %loaded : tensor<32x64xf32>, tensor<64x128xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @native_indirect_transpose_two_transposes
// CHECK-SAME: (%[[BASE:[^:]+]]: memref<?xf8E4M3FN>, %[[OFFSETS:[^:]+]]: tensor<64x128xi64>, %[[LHS:[^:]+]]: tensor<32x128xf8E4M3FN>, %[[C:[^:]+]]: tensor<32x64xf32>)
// CHECK-NOT: memref.alloc
// CHECK-NOT: scf.for
// CHECK: %[[WHOLE:[^ ]+]] = {{(func\.)?}}call @triton_indirect_load(%[[BASE]], %[[OFFSETS]]) {isVolatile = false}
// CHECK-SAME: -> tensor<64x128xf8E4M3FN>
// CHECK-NOT: memref.alloc
// CHECK-NOT: scf.for
// CHECK: %[[TRANS:[^ ]+]] = linalg.transpose ins(%[[WHOLE]] : tensor<64x128xf8E4M3FN>)
// CHECK-SAME: tensor<128x64xf8E4M3FN>) permutation = [1, 0]
// CHECK-NOT: memref.alloc
// CHECK-NOT: scf.for
// CHECK: %[[FIRST:[^ ]+]] = hfusion.matmul_mx
// CHECK-SAME: ins(%[[LHS]], %[[TRANS]],
// CHECK-SAME: outs(%[[C]] : tensor<32x64xf32>) -> tensor<32x64xf32>
// CHECK-NOT: memref.alloc
// CHECK-NOT: scf.for
// CHECK: %[[SECOND:[^ ]+]] = hfusion.matmul_mx
// CHECK-SAME: ins(%[[LHS]], %[[TRANS]],
// CHECK-SAME: outs(%[[FIRST]] : tensor<32x64xf32>) -> tensor<32x64xf32>
// CHECK-NOT: memref.alloc
// CHECK-NOT: scf.for
// CHECK: return %[[FIRST]], %[[SECOND]] : tensor<32x64xf32>, tensor<32x64xf32>
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_indirect_transpose_two_transposes(%base: memref<?xf8E4M3FN>, %offsets: tensor<64x128xi64>, %a: tensor<32x128xf8E4M3FN>, %c: tensor<32x64xf32>) -> (tensor<32x64xf32>, tensor<32x64xf32>) {
  %loaded = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  %b = tt.trans %loaded {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
  %b2 = tt.trans %loaded {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<32x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<32x64xf32>
  %e = tt.dot_scaled %a, %b2, %d lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<32x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<32x64xf32>
  return %d, %e : tensor<32x64xf32>, tensor<32x64xf32>
}
