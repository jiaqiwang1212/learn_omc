//===----------------------------------------------------------------------===//
// Phase 2 — Exercise 2: Counter Pass
//
// Goal: Extend the walk from Ex1 to *count* linalg.matmul ops specifically.
//       Print the count at the end of the function.
//
// What you need to implement:
//   runOnOperation() — the body marked with TODO below
//
// Build: scripts/build_passes.sh
// Run:
//   build/bin/mlir-opt \
//     --load-pass-plugin=build/passes/phase2/libEx2CounterPass.dylib \
//     --ex2-counter \
//     scripts/phase2/test_input.mlir
//
// Expected output: "Found N linalg.matmul op(s) in function 'forward'"
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Ex2CounterPass — counts linalg.matmul ops
//===----------------------------------------------------------------------===//
struct Ex2CounterPass
    : public PassWrapper<Ex2CounterPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Ex2CounterPass)

  StringRef getName() const override { return "ex2-counter"; }
  StringRef getArgument() const override { return "ex2-counter"; }
  StringRef getDescription() const override {
    return "Phase 2 Ex2: Count linalg.matmul ops in a function";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    unsigned matmulCount = 0;

    func.walk([&](linalg::MatmulOp matmul) {
      ++matmulCount;
      (void)matmul;
    });

    llvm::errs() << "\n=== Ex2CounterPass: function '" << func.getName()
                 << "' ===\n";
    llvm::errs() << "  Found " << matmulCount << " linalg.matmul op(s)\n";

    // NOTE: matmulCount will be 0 until you implement the walk above.
    // After implementing, run test_input.mlir — you should see count = 1.
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Plugin entry point
//===----------------------------------------------------------------------===//
extern "C" ::mlir::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "Ex2CounterPass", "0.1", []() {
    static mlir::PassRegistration<Ex2CounterPass> reg;
  }};
}
