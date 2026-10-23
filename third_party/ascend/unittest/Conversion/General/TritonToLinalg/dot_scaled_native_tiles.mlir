// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// M/N tiling must retain all 256 logical K elements in each native MX dot.
// CHECK-LABEL: func.func @large_native_fp4
// CHECK: tensor.extract_slice {{.*}} : tensor<128x128xf32> to tensor<32x32xf32>
// CHECK: tensor.extract_slice {{.*}} : tensor<128x128xi8> to tensor<32x128xi8>
// CHECK: tensor.extract_slice {{.*}} : tensor<128x128xi8> to tensor<128x32xi8>
// CHECK: tensor.extract_slice {{.*}} : tensor<128x8xi8> to tensor<32x8xi8>
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<32x128xi8>, tensor<128x32xi8>, tensor<32x8xi8>, tensor<32x8xi8>
// CHECK-SAME: outs({{.*}} : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: tensor.insert_slice {{.*}} : tensor<32x32xf32> into tensor<128x128xf32>
// CHECK-NOT: tt.dot_scaled
func.func @large_native_fp4(%a: tensor<128x128xi8>, %b: tensor<128x128xi8>, %sa: tensor<128x8xi8>, %sb: tensor<128x8xi8>, %c: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8>, tensor<128x8xi8> * tensor<128x128xi8>, tensor<128x8xi8> -> tensor<128x128xf32>
  return %d : tensor<128x128xf32>
}

// -----

// Optional scales are generated per native FP8 tile and keep their full K.
// CHECK-LABEL: func.func @large_native_fp8
// CHECK: tensor.extract_slice {{.*}} : tensor<128x32xf32> to tensor<32x32xf32>
// CHECK: tensor.extract_slice {{.*}} : tensor<128x256xf8E5M2> to tensor<32x256xf8E5M2>
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<32x256xf8E5M2>, tensor<256x32xf8E4M3FN>, tensor<32x8xi8>, tensor<32x8xi8>
// CHECK-SAME: outs({{.*}} : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: tensor.insert_slice {{.*}} : tensor<32x32xf32> into tensor<128x32xf32>
func.func @large_native_fp8(%a: tensor<128x256xf8E5M2>, %b: tensor<256x32xf8E4M3FN>, %c: tensor<128x32xf32>) -> tensor<128x32xf32> {
  %d = tt.dot_scaled %a, %b, %c lhs = e5m2 rhs = e4m3 {fastMath = false} : tensor<128x256xf8E5M2> * tensor<256x32xf8E4M3FN> -> tensor<128x32xf32>
  return %d : tensor<128x32xf32>
}
