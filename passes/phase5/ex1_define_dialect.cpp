//===----------------------------------------------------------------------===//
// Phase 5 — Exercise 1: Define and Register the myml Dialect
//
// This pass plugin does two things:
//   1. Defines the "myml" custom dialect (ReluOp, AddOp) by including
//      MyMLDialect.h
//   2. Registers it with mlir-opt via a trivial pass (--ex1-load-myml)
//      so the parser knows about myml.* ops
//
// What dialect registration does:
//   mlir-opt's parser uses a string → C++ class mapping. When it sees
//   "myml.relu" in IR text, it needs to know which C++ class to instantiate.
//   That mapping is registered here. Without it, the parser errors:
//   "use of undefined dialect 'myml'".
//
// Why use a pass plugin for dialect loading?
//   MLIR has two plugin types: pass plugins and dialect plugins.
//   We use a pass plugin here because it reuses the same loading mechanism
//   we already know from Phase 2 (--load-pass-plugin). The pass itself is
//   a no-op; its getDependentDialects() is what triggers registration.
//
// ML analogy:
//   Like registering a custom torch.autograd.Function — the name becomes
//   callable in a traced graph only after registration. Before registration,
//   torch.jit.script would error "unknown function myml.relu".
//
// Build: scripts/build_passes.sh (builds Ex1DefineDialect target)
// Run:
//   build/bin/mlir-opt \
//     --load-pass-plugin=build/passes/phase5/libEx1DefineDialect.so \
//     --ex1-load-myml \
//     scripts/phase5/test_input.mlir
//
// Expected: IR is parsed and printed unchanged (no transformation).
//===----------------------------------------------------------------------===//

#include "MyMLDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

namespace {

// NOTE on LLVM 23 dialect registration:
// In LLVM 23, getDependentDialects() fires AFTER the input file is parsed.
// To register a dialect BEFORE parsing (so mlir-opt can read custom ops),
// you must use a dialect plugin (mlirGetDialectPluginInfo below) and load
// it with --load-dialect-plugin=./libEx1DefineDialect.so.
//
// This is different from the pass plugin (mlirGetPassPluginInfo), which
// only registers passes for use in --pass-pipeline.
//
// Two plugin types, two entry points — both in this single .so file:
//   --load-dialect-plugin → calls mlirGetDialectPluginInfo → registers myml dialect
//   --load-pass-plugin    → calls mlirGetPassPluginInfo    → registers ex1-load-myml pass

struct Ex1LoadMyMLPass
    : public PassWrapper<Ex1LoadMyMLPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Ex1LoadMyMLPass)

  StringRef getName() const override { return "ex1-load-myml"; }
  StringRef getArgument() const override { return "ex1-load-myml"; }
  StringRef getDescription() const override {
    return "Phase 5 Ex1: No-op pass; dialect already registered by dialect plugin";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<myml::MyMLDialect>();
  }

  void runOnOperation() override {}
};

} // namespace

// Dialect plugin entry point — called by --load-dialect-plugin.
// Registers myml dialect with the context BEFORE input parsing.
extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "MyMLDialect", "0.1",
          [](mlir::DialectRegistry *registry) {
            registry->insert<myml::MyMLDialect>();
          }};
}

// Pass plugin entry point — called by --load-pass-plugin.
// Registers the ex1-load-myml pass for use in --pass-pipeline.
extern "C" ::mlir::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "Ex1DefineDialect", "0.1", []() {
    static mlir::PassRegistration<Ex1LoadMyMLPass> reg;
  }};
}
