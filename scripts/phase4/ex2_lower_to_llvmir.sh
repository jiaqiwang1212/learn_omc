#!/usr/bin/env bash
# Phase 4 — Exercise 2: Lower build/phase4/matmul.mlir to LLVM IR.
#
# Key difference from Phase 3: adds --llvm-request-c-wrappers BEFORE
# --convert-func-to-llvm. This generates the _mlir_ciface_forward wrapper
# function, which accepts MemRef2D* pointers instead of the exploded
# argument form, making the function callable from plain C.
#
# Usage: bash scripts/phase4/ex2_lower_to_llvmir.sh [input.mlir]
#   Defaults to build/phase4/matmul.mlir

set -euo pipefail
trap 'echo "FAILED at line $LINENO — check the error above"; exit 1' ERR

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MLIR_OPT="$REPO_ROOT/build/bin/mlir-opt"
MLIR_TRANSLATE="$REPO_ROOT/build/bin/mlir-translate"
INPUT="${1:-$REPO_ROOT/build/phase4/matmul.mlir}"
OUT_DIR="$REPO_ROOT/build/phase4"

mkdir -p "$OUT_DIR"

# --- Check tools exist ---
if [[ ! -x "$MLIR_OPT" ]]; then
  echo "ERROR: mlir-opt not found at $MLIR_OPT"
  exit 1
fi
if [[ ! -x "$MLIR_TRANSLATE" ]]; then
  echo "ERROR: mlir-translate not found at $MLIR_TRANSLATE"
  exit 1
fi
if [[ ! -f "$INPUT" ]]; then
  echo "ERROR: input not found: $INPUT"
  echo "Run first: uv run python scripts/phase4/ex1_export_matmul.py"
  exit 1
fi

echo "Input:  $INPUT"
echo "Output: $OUT_DIR/matmul.ll"
echo ""

# --- Full lowering pipeline with C-interface wrappers ---
# --llvm-request-c-wrappers: adds the llvm.emit_c_interface attribute to all
#   public func.func ops BEFORE they are lowered to LLVM dialect. This causes
#   --convert-func-to-llvm to emit a second wrapper function with the
#   _mlir_ciface_ prefix that takes memref descriptors by pointer (struct*),
#   instead of the exploded argument form (individual fields as scalars).
#
# Pass ordering:
#   bufferize → linalg→loops → lower loops → scf→cf → llvm conversions →
#   request-c-wrappers → convert-func-to-llvm → reconcile → translate

"$MLIR_OPT" "$INPUT" \
  --one-shot-bufferize="bufferize-function-boundaries" \
  --convert-linalg-to-affine-loops \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --llvm-request-c-wrappers \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  --verify-each \
| "$MLIR_TRANSLATE" --mlir-to-llvmir -o "$OUT_DIR/matmul.ll"

echo "Success! LLVM IR → $OUT_DIR/matmul.ll"
echo ""

# Show the generated C-interface wrapper
echo "Generated function names:"
grep "^define" "$OUT_DIR/matmul.ll" | sed 's/define[^@]*/  /'
