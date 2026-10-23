// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL: func.func @bulk_plain
// CHECK-SAME: %[[BASE:.*]]: memref<?xf8E4M3FN>, %[[STRIDE:.*]]: index
// CHECK-DAG: %[[ORIGIN:.*]] = arith.constant 5 : index
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2 : index
// CHECK-NOT: memref.alloc() : memref<128x64xf8E4M3FN>
// CHECK: scf.for %[[TILE:[^ ]+]] =
// CHECK: %[[DISP:.*]] = arith.muli %[[TILE]], %[[TWO]] : index
// CHECK: %[[OFFSET:.*]] = arith.addi %[[DISP]], %[[ORIGIN]] : index
// CHECK: memref.reinterpret_cast %[[BASE]] to offset: [%[[OFFSET]]], sizes: [128, 32], strides: [%[[STRIDE]], 2]
// CHECK: memref.alloc() : memref<128x32xf8E4M3FN>
// CHECK: memref.copy
// CHECK: hfusion.matmul_mx
// CHECK: tensor.insert_slice
// CHECK: return {{.*}} : tensor<16x64xf32>
func.func @bulk_plain(%base: memref<?xf8E4M3FN>, %row_stride: index, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %base_offset = arith.constant 5 : index
  %src = memref.reinterpret_cast %base to offset: [%base_offset], sizes: [128, 64], strides: [%row_stride, 2] : memref<?xf8E4M3FN> to memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>>
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %buf = memref.alloc() : memref<128x64xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>> to memref<128x64xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<128x64xf8E4M3FN> to tensor<128x64xf8E4M3FN>
  %d = tt.dot_scaled %a, %loaded, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @bulk_alias_store
// CHECK: memref.alloc() : memref<128x64xf8E4M3FN>
// CHECK: memref.copy
// CHECK: tensor.extract_slice {{.*}} : tensor<128x64xf8E4M3FN> to tensor<128x32xf8E4M3FN>
// CHECK-NOT: memref.alloc() : memref<128x32xf8E4M3FN>
func.func @bulk_alias_store(%base: memref<?xf8E4M3FN>, %row_stride: index, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %base_offset = arith.constant 5 : index
  %src = memref.reinterpret_cast %base to offset: [%base_offset], sizes: [128, 64], strides: [%row_stride, 2] : memref<?xf8E4M3FN> to memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>>
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %buf = memref.alloc() : memref<128x64xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>> to memref<128x64xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<128x64xf8E4M3FN> to tensor<128x64xf8E4M3FN>
  memref.store %v, %src[%zero, %zero] : memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>>
  %d = tt.dot_scaled %a, %loaded, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @bulk_shared
// CHECK: memref.alloc() : memref<128x64xf8E4M3FN>
// CHECK: memref.copy
// CHECK: tensor.extract_slice {{.*}} : tensor<128x64xf8E4M3FN> to tensor<128x32xf8E4M3FN>
// CHECK-NOT: memref.alloc() : memref<128x32xf8E4M3FN>
func.func @bulk_shared(%base: memref<?xf8E4M3FN>, %row_stride: index, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> (tensor<16x64xf32>, f8E4M3FN) {
  %base_offset = arith.constant 5 : index
  %src = memref.reinterpret_cast %base to offset: [%base_offset], sizes: [128, 64], strides: [%row_stride, 2] : memref<?xf8E4M3FN> to memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>>
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %buf = memref.alloc() : memref<128x64xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>> to memref<128x64xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<128x64xf8E4M3FN> to tensor<128x64xf8E4M3FN>
  %d = tt.dot_scaled %a, %loaded, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  %snapshot = tensor.extract %loaded[%zero, %zero] : tensor<128x64xf8E4M3FN>
  return %d, %snapshot : tensor<16x64xf32>, f8E4M3FN
}

// -----

// CHECK-LABEL: func.func @bulk_escape
// CHECK: memref.alloc() : memref<128x64xf8E4M3FN>
// CHECK: memref.copy
// CHECK: tensor.extract_slice {{.*}} : tensor<128x64xf8E4M3FN> to tensor<128x32xf8E4M3FN>
// CHECK-NOT: memref.alloc() : memref<128x32xf8E4M3FN>
func.func private @escape(memref<128x64xf8E4M3FN>)
func.func @bulk_escape(%base: memref<?xf8E4M3FN>, %row_stride: index, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %base_offset = arith.constant 5 : index
  %src = memref.reinterpret_cast %base to offset: [%base_offset], sizes: [128, 64], strides: [%row_stride, 2] : memref<?xf8E4M3FN> to memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>>
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %buf = memref.alloc() : memref<128x64xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>> to memref<128x64xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<128x64xf8E4M3FN> to tensor<128x64xf8E4M3FN>
  func.call @escape(%buf) : (memref<128x64xf8E4M3FN>) -> ()
  %d = tt.dot_scaled %a, %loaded, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @bulk_volatile
// CHECK: memref.alloc() : memref<128x64xf8E4M3FN>
// CHECK: memref.copy
// CHECK: tensor.extract_slice {{.*}} : tensor<128x64xf8E4M3FN> to tensor<128x32xf8E4M3FN>
// CHECK-NOT: memref.alloc() : memref<128x32xf8E4M3FN>
func.func @bulk_volatile(%base: memref<?xf8E4M3FN>, %row_stride: index, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %base_offset = arith.constant 5 : index
  %src = memref.reinterpret_cast %base to offset: [%base_offset], sizes: [128, 64], strides: [%row_stride, 2] : memref<?xf8E4M3FN> to memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>>
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %buf = memref.alloc() : memref<128x64xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = true} : memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>> to memref<128x64xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<128x64xf8E4M3FN> to tensor<128x64xf8E4M3FN>
  %d = tt.dot_scaled %a, %loaded, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @bulk_unmarked
// CHECK: memref.alloc() : memref<128x64xf8E4M3FN>
// CHECK: memref.copy
// CHECK: tensor.extract_slice {{.*}} : tensor<128x64xf8E4M3FN> to tensor<128x32xf8E4M3FN>
// CHECK-NOT: memref.alloc() : memref<128x32xf8E4M3FN>
func.func @bulk_unmarked(%base: memref<?xf8E4M3FN>, %row_stride: index, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %base_offset = arith.constant 5 : index
  %src = memref.reinterpret_cast %base to offset: [%base_offset], sizes: [128, 64], strides: [%row_stride, 2] : memref<?xf8E4M3FN> to memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>>
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %buf = memref.alloc() : memref<128x64xf8E4M3FN>
  memref.copy %src, %buf : memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>> to memref<128x64xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<128x64xf8E4M3FN> to tensor<128x64xf8E4M3FN>
  %d = tt.dot_scaled %a, %loaded, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @bulk_transpose
// CHECK-SAME: %[[BASE:.*]]: memref<?xf8E4M3FN>, %[[STRIDE:.*]]: index
// CHECK-DAG: %[[ORIGIN:.*]] = arith.constant 5 : index
// CHECK-NOT: memref.alloc() : memref<64x128xf8E4M3FN>
// CHECK: scf.for %[[TILE:[^ ]+]] =
// CHECK: %[[DISP:.*]] = arith.muli %[[TILE]], %[[STRIDE]] : index
// CHECK: %[[OFFSET:.*]] = arith.addi %[[DISP]], %[[ORIGIN]] : index
// CHECK: memref.reinterpret_cast %[[BASE]] to offset: [%[[OFFSET]]], sizes: [32, 128], strides: [%[[STRIDE]], 2]
// CHECK: memref.alloc() : memref<32x128xf8E4M3FN>
// CHECK: memref.copy
// CHECK: linalg.transpose {{.*}} tensor<32x128xf8E4M3FN>{{.*}}tensor<128x32xf8E4M3FN>
// CHECK: hfusion.matmul_mx
// CHECK: tensor.insert_slice
// CHECK: return {{.*}} : tensor<16x64xf32>
func.func @bulk_transpose(%base: memref<?xf8E4M3FN>, %row_stride: index, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %base_offset = arith.constant 5 : index
  %src = memref.reinterpret_cast %base to offset: [%base_offset], sizes: [64, 128], strides: [%row_stride, 2] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>>
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %buf = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>> to memref<64x128xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = linalg.transpose ins(%loaded : tensor<64x128xf8E4M3FN>) outs(%init : tensor<128x64xf8E4M3FN>) permutation = [1, 0]
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @bulk_transpose_alias
// CHECK: memref.alloc() : memref<64x128xf8E4M3FN>
// CHECK: memref.copy
// CHECK: %[[SNAPSHOT:.*]] = linalg.transpose ins({{.*}} : tensor<64x128xf8E4M3FN>)
// CHECK: scf.for
// CHECK: tensor.extract_slice %[[SNAPSHOT]]{{.*}} : tensor<128x64xf8E4M3FN> to tensor<128x32xf8E4M3FN>
// CHECK-NOT: memref.alloc() : memref<32x128xf8E4M3FN>
func.func @bulk_transpose_alias(%base: memref<?xf8E4M3FN>, %row_stride: index, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %base_offset = arith.constant 5 : index
  %src = memref.reinterpret_cast %base to offset: [%base_offset], sizes: [64, 128], strides: [%row_stride, 2] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>>
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %buf = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>> to memref<64x128xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  memref.store %v, %src[%zero, %zero] : memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>>
  %b = linalg.transpose ins(%loaded : tensor<64x128xf8E4M3FN>) outs(%init : tensor<128x64xf8E4M3FN>) permutation = [1, 0]
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @bulk_transpose_shared
// CHECK: memref.alloc() : memref<64x128xf8E4M3FN>
// CHECK: memref.copy
// CHECK: %[[SNAPSHOT:.*]] = linalg.transpose ins({{.*}} : tensor<64x128xf8E4M3FN>)
// CHECK: scf.for
// CHECK: tensor.extract_slice %[[SNAPSHOT]]{{.*}} : tensor<128x64xf8E4M3FN> to tensor<128x32xf8E4M3FN>
// CHECK-NOT: memref.alloc() : memref<32x128xf8E4M3FN>
func.func @bulk_transpose_shared(%base: memref<?xf8E4M3FN>, %row_stride: index, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> (tensor<16x64xf32>, f8E4M3FN) {
  %base_offset = arith.constant 5 : index
  %src = memref.reinterpret_cast %base to offset: [%base_offset], sizes: [64, 128], strides: [%row_stride, 2] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>>
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %buf = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>> to memref<64x128xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = linalg.transpose ins(%loaded : tensor<64x128xf8E4M3FN>) outs(%init : tensor<128x64xf8E4M3FN>) permutation = [1, 0]
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  %snapshot = tensor.extract %b[%zero, %zero] : tensor<128x64xf8E4M3FN>
  return %d, %snapshot : tensor<16x64xf32>, f8E4M3FN
}

// -----

// CHECK-LABEL: func.func @bulk_external_write
// CHECK: memref.alloc() : memref<128x64xf8E4M3FN>
// CHECK: memref.copy
// CHECK: tensor.extract_slice {{.*}} : tensor<128x64xf8E4M3FN> to tensor<128x32xf8E4M3FN>
// CHECK-NOT: memref.alloc() : memref<128x32xf8E4M3FN>
func.func private @external_write(memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>>)
func.func @bulk_external_write(%base: memref<?xf8E4M3FN>, %row_stride: index, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %base_offset = arith.constant 5 : index
  %src = memref.reinterpret_cast %base to offset: [%base_offset], sizes: [128, 64], strides: [%row_stride, 2] : memref<?xf8E4M3FN> to memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>>
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %buf = memref.alloc() : memref<128x64xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>> to memref<128x64xf8E4M3FN>
  func.call @external_write(%src) : (memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>>) -> ()
  %loaded = bufferization.to_tensor %buf restrict writable : memref<128x64xf8E4M3FN> to tensor<128x64xf8E4M3FN>
  %d = tt.dot_scaled %a, %loaded, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----


// CHECK-LABEL: func.func @bulk_unproven
// CHECK: memref.alloc() : memref<128x64xf8E4M3FN>
// CHECK: memref.copy
// CHECK: tensor.extract_slice
// CHECK-NOT: memref.alloc() : memref<128x32xf8E4M3FN>
func.func @bulk_unproven(%src: memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>>, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %buf = memref.alloc() : memref<128x64xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>> to memref<128x64xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<128x64xf8E4M3FN> to tensor<128x64xf8E4M3FN>
  %d = tt.dot_scaled %a, %loaded, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}


// -----


// CHECK-LABEL: func.func @bulk_transpose_unproven
// CHECK: memref.alloc() : memref<64x128xf8E4M3FN>
// CHECK: memref.copy
// CHECK: tensor.extract_slice
// CHECK-NOT: memref.alloc() : memref<32x128xf8E4M3FN>
func.func @bulk_transpose_unproven(%src: memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>>, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %buf = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>> to memref<64x128xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = linalg.transpose ins(%loaded : tensor<64x128xf8E4M3FN>) outs(%init : tensor<128x64xf8E4M3FN>) permutation = [1, 0]
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}


// -----

// Bulk sinking is specific to native MX. The software path keeps its full
// snapshot, including the original dynamic stride and nonzero source origin.
// CHECK-LABEL: func.func @bulk_both_axes
// CHECK: memref.reinterpret_cast {{.*}} sizes: [64, 64], strides: [{{.*}}, 2]
// CHECK: memref.alloc() : memref<64x64xf16>
// CHECK: memref.copy {{.*}} {isVolatile = false}
// CHECK: scf.for
// CHECK: tensor.extract_slice {{.*}} : tensor<64x64xf16> to tensor<32x32xf16>
// CHECK-NOT: memref.alloc() : memref<32x32xf16>
// CHECK: linalg.matmul
func.func @bulk_both_axes(%base: memref<?xf16>, %row_stride: index, %a: tensor<16x64xf16>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %base_offset = arith.constant 5 : index
  %src = memref.reinterpret_cast %base to offset: [%base_offset], sizes: [64, 64], strides: [%row_stride, 2] : memref<?xf16> to memref<64x64xf16, strided<[?, 2], offset: ?>>
  %buf = memref.alloc() : memref<64x64xf16>
  memref.copy %src, %buf {isVolatile = false} : memref<64x64xf16, strided<[?, 2], offset: ?>> to memref<64x64xf16>
  %b = bufferization.to_tensor %buf restrict writable : memref<64x64xf16> to tensor<64x64xf16>
  %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x64xf16> * tensor<64x64xf16> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}
