//===----------------------------------------------------------------------===//
// Phase 2 — Exercise 1: No-op Pass
//
// Goal: Walk all operations in a func.func and print their names.
//       This is your first MLIR C++ pass — no transformation, just observation.
//
// What you need to implement:
//   runOnOperation() — the body marked with TODO below
//
// Build: scripts/build_passes.sh
// Run:
//   build/bin/mlir-opt \
//     --load-pass-plugin=build/passes/phase2/libEx1NoopPass.dylib \
//     --ex1-noop \
//     scripts/phase2/test_input.mlir
//
// Expected output: prints every op name in the function to stderr.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Ex1NoopPass — walks all ops and prints their names
//===----------------------------------------------------------------------===//
struct Ex1NoopPass
    : public PassWrapper<Ex1NoopPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Ex1NoopPass)

  StringRef getName() const override { return "ex1-noop"; }
  StringRef getArgument() const override { return "ex1-noop"; }
  StringRef getDescription() const override {
    return "Phase 2 Ex1: Walk linalg ops and print their names";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    llvm::errs() << "\n=== Ex1NoopPass: function '" << func.getName()
                 << "' ===\n";

    func.walk([](Operation *op) {
      llvm::errs() << "  op: " << op->getName().getStringRef() << "\n";
    });
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Plugin entry point
// mlir-opt calls this function when --load-pass-plugin is used.
//===----------------------------------------------------------------------===//
extern "C" ::mlir::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "Ex1NoopPass", "0.1", []() {
    static mlir::PassRegistration<Ex1NoopPass> reg;
  }};
}
