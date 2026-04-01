//===----------------------------------------------------------------------===//
// Phase 5 — MyML Dialect Header
//
// Defines the "myml" (My ML) custom dialect with two ops:
//   myml.relu  — elementwise ReLU: max(x, 0) on tensor<...xf32>
//   myml.add   — elementwise add of two tensors of the same type
//
// Why no TableGen?
//   TableGen generates this C++ boilerplate for you in production MLIR code.
//   We write it by hand here so you can see exactly what TableGen produces,
//   making the generated code readable when you encounter it in torch-mlir.
//
// ML analogy for the dialect:
//   A dialect is like a PyTorch operator namespace ("aten::", "prims::").
//   Registering it maps "myml.relu" strings → ReluOp C++ class at parse time,
//   just like registering a custom torch.autograd.Function makes its name
//   callable in a traced graph.
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

namespace myml {

//===----------------------------------------------------------------------===//
// MyMLDialect
//===----------------------------------------------------------------------===//
class MyMLDialect : public mlir::Dialect {
public:
  explicit MyMLDialect(mlir::MLIRContext *ctx);
  static llvm::StringRef getDialectNamespace() { return "myml"; }
};

//===----------------------------------------------------------------------===//
// ReluOp — myml.relu %input : tensor<...xf32>
//
// Computes elementwise max(x, 0.0) on a ranked tensor of f32.
// Lowers to linalg.generic in ex2_lower_to_linalg.cpp.
//===----------------------------------------------------------------------===//
class ReluOp
    : public mlir::Op<ReluOp,
                      mlir::OpTrait::OneOperand,
                      mlir::OpTrait::OneResult,
                      mlir::OpTrait::SameOperandsAndResultType> {
public:
  using Op::Op;
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReluOp)

  // Required by LLVM 23: ops registered via addOperations<> must declare
  // their attribute names (empty for ops with no attributes).
  static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

  static llvm::StringRef getOperationName() { return "myml.relu"; }

  static void build(mlir::OpBuilder &b, mlir::OperationState &state,
                    mlir::Value input) {
    state.addOperands(input);
    state.addTypes(input.getType());
  }

  mlir::LogicalResult verify() {
    // Input must be a ranked tensor of f32
    // OneOperand trait provides getOperand() with no index argument
    auto tensorTy = llvm::dyn_cast<mlir::RankedTensorType>(
        getOperand().getType());
    if (!tensorTy)
      return emitOpError("operand must be a ranked tensor");
    if (!tensorTy.getElementType().isF32())
      return emitOpError("operand element type must be f32");
    return mlir::success();
  }

  // Text syntax: myml.relu %arg : tensor<4xf32>
  static mlir::ParseResult parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
    mlir::OpAsmParser::UnresolvedOperand operand;
    mlir::Type type;
    if (parser.parseOperand(operand) ||
        parser.parseColonType(type) ||
        parser.resolveOperand(operand, type, result.operands))
      return mlir::failure();
    result.addTypes(type);
    return mlir::success();
  }

  void print(mlir::OpAsmPrinter &p) {
    // OneOperand: getOperand(); use getOperation() (non-dependent base) for result type
    p << " " << getOperand() << " : " << getOperation()->getResult(0).getType();
  }
};

//===----------------------------------------------------------------------===//
// AddOp — myml.add %lhs, %rhs : tensor<...xf32>
//
// Computes elementwise lhs + rhs on ranked tensors of f32 (same type).
// Lowers to linalg.generic in ex2_lower_to_linalg.cpp.
//===----------------------------------------------------------------------===//
class AddOp
    : public mlir::Op<AddOp,
                      mlir::OpTrait::NOperands<2>::Impl,
                      mlir::OpTrait::OneResult,
                      mlir::OpTrait::SameOperandsAndResultType> {
public:
  using Op::Op;
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddOp)

  static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

  static llvm::StringRef getOperationName() { return "myml.add"; }

  static void build(mlir::OpBuilder &b, mlir::OperationState &state,
                    mlir::Value lhs, mlir::Value rhs) {
    state.addOperands({lhs, rhs});
    state.addTypes(lhs.getType());
  }

  mlir::LogicalResult verify() {
    auto tensorTy = llvm::dyn_cast<mlir::RankedTensorType>(
        getOperand(0).getType());
    if (!tensorTy)
      return emitOpError("operand must be a ranked tensor");
    if (!tensorTy.getElementType().isF32())
      return emitOpError("operand element type must be f32");
    return mlir::success();
  }

  // Text syntax: myml.add %lhs, %rhs : tensor<4xf32>
  static mlir::ParseResult parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
    mlir::Type type;
    if (parser.parseOperandList(operands, 2) ||
        parser.parseColonType(type) ||
        parser.resolveOperands(operands, type, result.operands))
      return mlir::failure();
    result.addTypes(type);
    return mlir::success();
  }

  void print(mlir::OpAsmPrinter &p) {
    // NOperands<2>: getOperand(unsigned); use getOperation() (non-dependent base) for result type
    p << " " << getOperand(0) << ", " << getOperand(1) << " : "
      << getOperation()->getResult(0).getType();
  }
};

} // namespace myml

// Dialect constructor — defined here to keep everything in one file
inline myml::MyMLDialect::MyMLDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx,
                    mlir::TypeID::get<MyMLDialect>()) {
  addOperations<ReluOp, AddOp>();
}
