#!/usr/bin/env bash
# Phase 3.5 — Break the Pipeline: diagnostic exercises
# Runs 4 deliberately broken pipeline variants and shows the errors.
# Each break includes a diagnostic question for the learner.

set -uo pipefail
# NOTE: set -e is intentionally omitted — we expect failures.

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MLIR_OPT="$REPO_ROOT/build/bin/mlir-opt"
MLIR_TRANSLATE="$REPO_ROOT/build/bin/mlir-translate"
INPUT="${1:-$REPO_ROOT/scripts/phase2/test_input.mlir}"

# =========================================================================
# BREAK 1: Missing --reconcile-unrealized-casts
# =========================================================================
echo "=== BREAK 1: Missing --reconcile-unrealized-casts ==="
echo ""

ERROR_OUTPUT=$("$MLIR_OPT" "$INPUT" \
  --one-shot-bufferize="bufferize-function-boundaries" \
  --convert-linalg-to-affine-loops \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --convert-func-to-llvm \
  --verify-each \
| "$MLIR_TRANSLATE" --mlir-to-llvmir 2>&1 || true)
echo "$ERROR_OUTPUT" | head -20

echo ""
echo "QUESTION: What error does mlir-translate report? What does 'unrealized_conversion_cast' mean — why is this pass required?"
echo ""
echo "---"
echo ""

# =========================================================================
# BREAK 2: Wrong order — loops before bufferize
# =========================================================================
echo "=== BREAK 2: Wrong order: loops before bufferize ==="
echo ""

ERROR_OUTPUT=$("$MLIR_OPT" "$INPUT" \
  --convert-linalg-to-affine-loops \
  --one-shot-bufferize="bufferize-function-boundaries" \
  --verify-each 2>&1 || true)
echo "$ERROR_OUTPUT" | head -20

echo ""
echo "QUESTION: What does --verify-each report? The error mentions type mismatches. Why must bufferization happen BEFORE loop lowering?"
echo ""
echo "---"
echo ""

# =========================================================================
# BREAK 3: Missing --convert-func-to-llvm
# =========================================================================
echo "=== BREAK 3: Missing --convert-func-to-llvm ==="
echo ""

ERROR_OUTPUT=$("$MLIR_OPT" "$INPUT" \
  --one-shot-bufferize="bufferize-function-boundaries" \
  --convert-linalg-to-affine-loops \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --reconcile-unrealized-casts \
  --verify-each \
| "$MLIR_TRANSLATE" --mlir-to-llvmir 2>&1 || true)
echo "$ERROR_OUTPUT" | head -20

echo ""
echo "QUESTION: What error does mlir-translate report? Why must func.func be lowered to LLVM before translation?"
echo ""
echo "---"
echo ""

# =========================================================================
# BREAK 4: Bonus — try lowering relu
# =========================================================================
echo "=== BREAK 4: Bonus: lower relu instead of matmul ==="
echo ""

RELU_SCRIPT="$REPO_ROOT/scripts/phase1/ex5_relu_linalg_generic.py"
RELU_IR="$REPO_ROOT/build/phase1/relu_linalg.mlir"

if [[ ! -f "$RELU_SCRIPT" ]]; then
  echo "Skipping: relu generator script not found at $RELU_SCRIPT"
elif [[ ! -f "$RELU_IR" ]]; then
  echo "Skipping: run 'uv run python scripts/phase1/ex5_relu_linalg_generic.py' first to generate the relu IR"
else
  ERROR_OUTPUT=$("$MLIR_OPT" "$RELU_IR" \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --convert-linalg-to-affine-loops \
    --lower-affine \
    --convert-scf-to-cf \
    --convert-cf-to-llvm \
    --convert-arith-to-llvm \
    --finalize-memref-to-llvm \
    --convert-func-to-llvm \
    --reconcile-unrealized-casts \
    --verify-each \
  | "$MLIR_TRANSLATE" --mlir-to-llvmir 2>&1 || true)
  echo "$ERROR_OUTPUT" | head -20
fi

echo ""
echo "QUESTION: Does the same pipeline handle relu? If it fails, use --debug-only=dialect-conversion to find which pass doesn't handle the new ops."
echo ""
