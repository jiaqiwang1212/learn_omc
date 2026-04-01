#!/usr/bin/env bash
# Phase 3 — Full Lowering Chain
# Lowers a linalg-on-tensors MLIR program all the way to LLVM IR (.ll).
#
# Usage: bash scripts/phase3/ex3_full_chain.sh [input.mlir]
#   Defaults to scripts/phase2/test_input.mlir (4x4 matmul).

set -euo pipefail
trap 'echo "FAILED at line $LINENO — check the error above"; exit 1' ERR

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MLIR_OPT="$REPO_ROOT/build/bin/mlir-opt"
MLIR_TRANSLATE="$REPO_ROOT/build/bin/mlir-translate"
INPUT="${1:-$REPO_ROOT/scripts/phase2/test_input.mlir}"
OUT_DIR="$REPO_ROOT/build/phase3"

mkdir -p "$OUT_DIR"

# --- Check tools exist ---
if [[ ! -x "$MLIR_OPT" ]]; then
  echo "ERROR: mlir-opt not found at $MLIR_OPT"
  echo "Build MLIR first:  cmake --build build --target mlir-opt"
  exit 1
fi
if [[ ! -x "$MLIR_TRANSLATE" ]]; then
  echo "ERROR: mlir-translate not found at $MLIR_TRANSLATE"
  echo "Build MLIR first:  cmake --build build --target mlir-translate"
  exit 1
fi

# --- Full lowering pipeline ---
echo "Running: bufferize..."
echo "Running: convert-linalg-to-affine-loops..."
echo "Running: lower-affine..."
echo "Running: convert-scf-to-cf..."
echo "Running: convert-cf-to-llvm..."
echo "Running: convert-arith-to-llvm..."
echo "Running: finalize-memref-to-llvm..."
echo "Running: convert-func-to-llvm..."
echo "Running: reconcile-unrealized-casts..."
echo "Running: mlir-translate --mlir-to-llvmir..."

"$MLIR_OPT" "$INPUT" \
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
| "$MLIR_TRANSLATE" --mlir-to-llvmir -o "$OUT_DIR/matmul.ll"

echo ""
echo "Success! LLVM IR written to $OUT_DIR/matmul.ll"
echo ""
echo "Usage: bash scripts/phase3/ex3_full_chain.sh [input.mlir]"
