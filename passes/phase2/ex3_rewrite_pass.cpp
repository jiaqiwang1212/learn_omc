//===----------------------------------------------------------------------===//
// Phase 2 — Exercise 3: Rewrite Pass
//
// Goal: Use PatternRewriter to match linalg.matmul and insert a func.call
//       to a stub function "matmul_hook" immediately before it.
//
// This teaches you:
//   - How to write a RewritePattern
//   - How to use rewriter.create<>() to insert ops
//   - The SSA invariant: you never edit existing values, you create new ones
//
// What you need to implement:
//   matchAndRewrite() — the body marked with TODO below
//
// Build: scripts/build_passes.sh
// Run:
//   build/bin/mlir-opt \
//     --load-pass-plugin=build/passes/phase2/libEx3RewritePass.dylib \
//     --ex3-rewrite \
//     --verify-each \
//     scripts/phase2/test_input.mlir
//
// TIP: Always use --verify-each when developing passes. It runs the MLIR
//      verifier after your pass and gives you a clear error if you produce
//      invalid IR, rather than a mysterious crash later.
//
// Expected output: IR with a func.call @matmul_hook() inserted before
//                  the linalg.matmul op.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// InsertMatmulHookPattern
//
// This RewritePattern matches every linalg.matmul op and inserts a
// func.call @matmul_hook() immediately before it.
//
// PyTorch analogy: this is like a torch.fx graph transformation that
// inserts a new node before an existing matmul node. The key difference
// is that in SSA form you never "move" the matmul — you insert a new op
// that runs first.
//===----------------------------------------------------------------------===//
struct InsertMatmulHookPattern
    : public OpRewritePattern<linalg::MatmulOp> {

  // Inherit the constructor — PatternBenefit(1) = default priority
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    // Get the location of the matmul op (used for new ops we create)
    Location loc = matmulOp.getLoc();

    // Guard: don't fire again if we already inserted the hook.
    if (auto *prevOp = matmulOp->getPrevNode())
      if (auto callOp = dyn_cast<func::CallOp>(prevOp))
        if (callOp.getCallee() == "matmul_hook")
          return failure();

    rewriter.setInsertionPoint(matmulOp);
    rewriter.create<func::CallOp>(loc, "matmul_hook", TypeRange{}, ValueRange{});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Ex3RewritePass — drives InsertMatmulHookPattern via greedy rewrite
//===----------------------------------------------------------------------===//
struct Ex3RewritePass
    : public PassWrapper<Ex3RewritePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Ex3RewritePass)

  StringRef getName() const override { return "ex3-rewrite"; }
  StringRef getArgument() const override { return "ex3-rewrite"; }
  StringRef getDescription() const override {
    return "Phase 2 Ex3: Insert func.call before linalg.matmul via RewritePattern";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    // Ensure the parent module has a declaration for @matmul_hook so the
    // verifier accepts the func.call we are about to insert.
    auto moduleOp = func->getParentOfType<ModuleOp>();
    if (moduleOp && !moduleOp.lookupSymbol<func::FuncOp>("matmul_hook")) {
      OpBuilder builder(moduleOp.getBodyRegion());
      builder.setInsertionPointToStart(moduleOp.getBody());
      auto hookType = FunctionType::get(ctx, /*inputs=*/{}, /*results=*/{});
      auto hookDecl = builder.create<func::FuncOp>(
          func.getLoc(), "matmul_hook", hookType);
      hookDecl.setPrivate();
    }

    // Register our pattern and run it with the greedy rewrite driver.
    // The greedy driver applies patterns repeatedly until no more fire.
    RewritePatternSet patterns(ctx);
    patterns.add<InsertMatmulHookPattern>(ctx);

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Plugin entry point
//===----------------------------------------------------------------------===//
extern "C" ::mlir::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "Ex3RewritePass", "0.1", []() {
    static mlir::PassRegistration<Ex3RewritePass> reg;
  }};
}
