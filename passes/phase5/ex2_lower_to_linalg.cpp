//===----------------------------------------------------------------------===//
// Phase 5 — Exercise 2: Lower myml Dialect to linalg.generic
//
// This pass plugin lowers the custom "myml" dialect ops to standard
// linalg.generic ops, which can then be lowered to LLVM IR using the
// Phase 3 pipeline.
//
// This is the compiler equivalent of torch.fx graph rewriting:
//   - We match a high-level op (myml.relu)
//   - We replace it with the lower-level ops that implement its semantics
//     (a linalg.generic loop nest with arith.maximumf)
//
// The lowering path:
//   myml.relu → linalg.generic (elementwise max(x, 0.0))
//   myml.add  → linalg.generic (elementwise x + y)
//
// After this pass, the IR contains only standard dialects and can be
// lowered to LLVM IR using the same pipeline as Phase 3.
//
// Build: scripts/build_passes.sh (builds Ex2LowerToLinalg target)
// Run:
//   build/bin/mlir-opt \
//     --load-pass-plugin=build/passes/phase5/libEx1DefineDialect.so \
//     --load-pass-plugin=build/passes/phase5/libEx2LowerToLinalg.so \
//     --ex1-load-myml \
//     --ex2-lower-myml-to-linalg \
//     scripts/phase5/test_input.mlir
//===----------------------------------------------------------------------===//

#include "MyMLDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// LowerReluToLinalg
//
// myml.relu %input : tensor<NxMxf32>
//   → linalg.generic { max(in, 0.0) } ins(%input) outs(%empty)
//
// ML analogy: like a torch.fx graph rewrite that replaces a relu node
// with the underlying loop nest that implements it.
//===----------------------------------------------------------------------===//
struct LowerReluToLinalg : public OpRewritePattern<myml::ReluOp> {
  using OpRewritePattern<myml::ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(myml::ReluOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // OneOperand trait provides getOperand() with no index
    Value input = op.getOperand();
    auto tensorType = cast<RankedTensorType>(input.getType());
    int64_t rank = tensorType.getRank();

    // Create an empty output tensor of the same shape/type
    auto emptyOp = rewriter.create<tensor::EmptyOp>(
        loc, tensorType.getShape(), tensorType.getElementType());

    // Identity affine maps: (d0, d1, ...) -> (d0, d1, ...) for all dims
    AffineMap identityMap = AffineMap::getMultiDimIdentityMap(rank, ctx);
    SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};

    // All iterator types are parallel (no reduction)
    SmallVector<utils::IteratorType> iteratorTypes(rank,
                                                    utils::IteratorType::parallel);

    // Build linalg.generic with an arith.maximumf body
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/TypeRange{tensorType},
        /*inputs=*/ValueRange{input},
        /*outputs=*/ValueRange{emptyOp.getResult()},
        indexingMaps,
        iteratorTypes,
        [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
          // args[0] = current input element, args[1] = current output element
          Value zero = b.create<arith::ConstantOp>(
              bodyLoc, b.getF32FloatAttr(0.0f));
          Value relu = b.create<arith::MaximumFOp>(bodyLoc, args[0], zero);
          b.create<linalg::YieldOp>(bodyLoc, relu);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LowerAddToLinalg
//
// myml.add %lhs, %rhs : tensor<NxMxf32>
//   → linalg.generic { lhs + rhs } ins(%lhs, %rhs) outs(%empty)
//===----------------------------------------------------------------------===//
struct LowerAddToLinalg : public OpRewritePattern<myml::AddOp> {
  using OpRewritePattern<myml::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(myml::AddOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);
    auto tensorType = cast<RankedTensorType>(lhs.getType());
    int64_t rank = tensorType.getRank();

    auto emptyOp = rewriter.create<tensor::EmptyOp>(
        loc, tensorType.getShape(), tensorType.getElementType());

    AffineMap identityMap = AffineMap::getMultiDimIdentityMap(rank, ctx);
    // 3 maps: lhs input, rhs input, output
    SmallVector<AffineMap> indexingMaps = {identityMap, identityMap,
                                           identityMap};
    SmallVector<utils::IteratorType> iteratorTypes(rank,
                                                    utils::IteratorType::parallel);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        TypeRange{tensorType},
        ValueRange{lhs, rhs},
        ValueRange{emptyOp.getResult()},
        indexingMaps,
        iteratorTypes,
        [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
          // args[0] = lhs element, args[1] = rhs element, args[2] = output
          Value sum = b.create<arith::AddFOp>(bodyLoc, args[0], args[1]);
          b.create<linalg::YieldOp>(bodyLoc, sum);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Ex2LowerMyMLToLinalgPass
//===----------------------------------------------------------------------===//
struct Ex2LowerMyMLToLinalgPass
    : public PassWrapper<Ex2LowerMyMLToLinalgPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Ex2LowerMyMLToLinalgPass)

  StringRef getName() const override { return "ex2-lower-myml-to-linalg"; }
  StringRef getArgument() const override { return "ex2-lower-myml-to-linalg"; }
  StringRef getDescription() const override {
    return "Phase 5 Ex2: Lower myml.relu and myml.add to linalg.generic";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<myml::MyMLDialect, linalg::LinalgDialect,
                    tensor::TensorDialect, arith::ArithDialect,
                    affine::AffineDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<LowerReluToLinalg, LowerAddToLinalg>(ctx);

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

extern "C" ::mlir::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "Ex2LowerToLinalg", "0.1", []() {
    static mlir::PassRegistration<Ex2LowerMyMLToLinalgPass> reg;
  }};
}
