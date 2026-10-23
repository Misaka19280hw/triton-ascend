// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL: func.func @bulk_plain
// CHECK-NOT: memref.alloc() : memref<128x64xf8E4M3FN>
// CHECK: scf.for
// CHECK: memref.subview
// CHECK: memref.alloc() : memref<128x32xf8E4M3FN>
// CHECK: memref.copy
// CHECK: hfusion.matmul_mx
// CHECK: tensor.insert_slice
// CHECK: return {{.*}} : tensor<16x64xf32>
func.func @bulk_plain(%src: memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>>, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
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
func.func @bulk_alias_store(%src: memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>>, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
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
func.func @bulk_shared(%src: memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>>, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> (tensor<16x64xf32>, f8E4M3FN) {
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
func.func @bulk_escape(%src: memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>>, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
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
func.func @bulk_volatile(%src: memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>>, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
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
func.func @bulk_unmarked(%src: memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>>, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
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
// CHECK-NOT: memref.alloc() : memref<64x128xf8E4M3FN>
// CHECK: scf.for
// CHECK: memref.subview
// CHECK: memref.alloc() : memref<32x128xf8E4M3FN>
// CHECK: memref.copy
// CHECK: linalg.transpose {{.*}} tensor<32x128xf8E4M3FN>{{.*}}tensor<128x32xf8E4M3FN>
// CHECK: hfusion.matmul_mx
// CHECK: tensor.insert_slice
// CHECK: return {{.*}} : tensor<16x64xf32>
func.func @bulk_transpose(%src: memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>>, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
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
// CHECK: tensor.extract_slice {{.*}} : tensor<64x128xf8E4M3FN> to tensor<32x128xf8E4M3FN>
// CHECK-NOT: memref.alloc() : memref<32x128xf8E4M3FN>
func.func @bulk_transpose_alias(%src: memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>>, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
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
// CHECK: tensor.extract_slice {{.*}} : tensor<64x128xf8E4M3FN> to tensor<32x128xf8E4M3FN>
// CHECK-NOT: memref.alloc() : memref<32x128xf8E4M3FN>
func.func @bulk_transpose_shared(%src: memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>>, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> (tensor<16x64xf32>, f8E4M3FN) {
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
func.func @bulk_external_write(%src: memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>>, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %buf = memref.alloc() : memref<128x64xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>> to memref<128x64xf8E4M3FN>
  func.call @external_write(%src) : (memref<128x64xf8E4M3FN, strided<[?, 2], offset: ?>>) -> ()
  %loaded = bufferization.to_tensor %buf restrict writable : memref<128x64xf8E4M3FN> to tensor<128x64xf8E4M3FN>
  %d = tt.dot_scaled %a, %loaded, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}
