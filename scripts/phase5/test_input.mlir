// Phase 5 test input: hand-written MLIR using the custom myml dialect.
// These ops are NOT in any standard MLIR dialect — they are defined in
// passes/phase5/MyMLDialect.h and registered by Ex1DefineDialect.
//
// Run the full pipeline via: scripts/phase5/ex3_end_to_end.sh

func.func @test_relu(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // myml.relu: elementwise max(x, 0.0) — lowers to linalg.generic
  %0 = myml.relu %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func @test_add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // myml.add: elementwise addition — lowers to linalg.generic
  %0 = myml.add %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
