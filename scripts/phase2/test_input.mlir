// test_input.mlir — Phase 2 test input
// A minimal 4x4 matmul in linalg-on-tensors form.
// Use this with all three exercise passes:
//
//   build/bin/mlir-opt --load-pass-plugin=... --ex1-noop test_input.mlir
//   build/bin/mlir-opt --load-pass-plugin=... --ex2-counter test_input.mlir
//   build/bin/mlir-opt --load-pass-plugin=... --ex3-rewrite --verify-each test_input.mlir

func.func @forward(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // output buffer — linalg.matmul writes into a pre-existing tensor
  %init = tensor.empty() : tensor<4x4xf32>
  // linalg.matmul: %arg0 * %arg1 -> %init
  %result = linalg.matmul ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>)
                          outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}
